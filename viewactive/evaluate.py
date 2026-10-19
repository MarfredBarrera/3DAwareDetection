"""Evaluate ViewActive: field accuracy vs the mean prior, paper Fig. 4 rollouts,
speed, and Fig. 2-style VQF heatmaps.

Fig. 4 protocol: from a uniformly random start view, run ``optimize.steps``
rounds of Algorithm 1 and record, at every step, the ground-truth VQF channels
of the current view (normalized per object, as plotted in the paper, and raw)
and each ImageNet classifier's top-1 correctness. Controls use the same starts:
``gt_field`` (Algorithm 1 on the true VQF; an oracle), ``mean_prior`` (the
training-set mean relative field; it only knows which elevations tend to be
good), ``random_walk`` and ``stay``.
"""
import json
import time

import matplotlib
matplotlib.use("Agg")  # The container is headless.
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import rankdata

from study.figures import Heatmap, Images, save_panels
from viewactive.grid import combine, to_absolute, to_relative
from viewactive.labels import CHANNELS
from viewactive.optimize import field_policy, random_policy, rollout, stay

POLICIES = ("viewactive", "gt_field", "mean_prior", "random_walk", "stay")
SPLIT_NAMES = {"test": "seen", "test_unseen": "unseen"}


def predict_all(net, features, device="cuda", batch=8192):
    """Predicted relative fields ``[O, E, A, 3, E, A]`` for every cached view."""
    import torch

    shape = features.shape[:3]
    flat = np.asarray(features).reshape(-1, features.shape[-1])
    out = []
    with torch.inference_mode():
        for start in range(0, len(flat), batch):
            x = torch.from_numpy(flat[start:start + batch]).to(device).float()
            out.append(net(x).cpu().numpy())
    return np.concatenate(out).reshape(*shape, *out[0].shape[1:])


def relative_truth(fields):
    """Ground-truth relative fields ``[O, A, 3, E, A]``: one per camera azimuth column."""
    return np.stack([to_relative(fields, a) for a in range(fields.shape[-1])], axis=1)


def spearman(pred, truth):
    """Per-row Spearman correlation over cells where both are finite; NaN if < 3 cells."""
    valid = np.isfinite(pred) & np.isfinite(truth)
    p = rankdata(np.where(valid, pred, np.nan), axis=-1, nan_policy="omit")
    t = rankdata(np.where(valid, truth, np.nan), axis=-1, nan_policy="omit")
    with np.errstate(all="ignore"):
        p = p - np.nanmean(p, axis=-1, keepdims=True)
        t = t - np.nanmean(t, axis=-1, keepdims=True)
        rho = np.nansum(p * t, -1) / np.sqrt(np.nansum(p * p, -1) * np.nansum(t * t, -1))
    return np.where(valid.sum(-1) >= 3, rho, np.nan)


def field_metrics(pred, truth, weights=None):
    """Per-channel MAE and Spearman, plus how good the predicted best view truly is.

    ``pred, truth [N, 3, E, A]`` in the same (relative) frame. ``top1`` = the
    predicted argmax is a true best cell; ``top5`` = it is among the true five
    best; ``regret`` = true best combined quality - true quality at the predicted argmax.
    """
    out = {}
    flat_p, flat_t = pred.reshape(*pred.shape[:2], -1), truth.reshape(*truth.shape[:2], -1)
    for c, name in enumerate(CHANNELS):
        out[name] = {"mae": float(np.nanmean(np.abs(flat_p[:, c] - flat_t[:, c]))),
                     "spearman": float(np.nanmean(spearman(flat_p[:, c], flat_t[:, c])))}
    p = combine(pred, weights).reshape(len(pred), -1)
    t = combine(truth, weights).reshape(len(truth), -1)
    t_safe = np.where(np.isfinite(t), t, -np.inf)
    choice = np.argmax(np.where(np.isfinite(p) & np.isfinite(t), p, -np.inf), axis=1)
    chosen = t_safe[np.arange(len(t)), choice]
    best = t_safe.max(1)
    fifth = -np.sort(-t_safe, axis=1)[:, min(4, t.shape[1] - 1)]
    out["combined"] = {"mae": float(np.nanmean(np.abs(p - t))),
                       "spearman": float(np.nanmean(spearman(p, t))),
                       "top1": float(np.mean(chosen >= best - 1e-9)),
                       "top5": float(np.mean(chosen >= fifth - 1e-9)),
                       "regret": float(np.mean(best - chosen))}
    return out


def bootstrap(values, rounds=1000, seed=0):
    """Mean and 95% bootstrap interval over objects, ignoring undefined labels."""
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    means = finite_mean(values[rng.integers(len(values), size=(rounds, len(values)))], axis=1)
    center = finite_mean(values, axis=0)
    return {"mean": center.tolist(), "lo": np.nanpercentile(means, 2.5, axis=0).tolist(),
            "hi": np.nanpercentile(means, 97.5, axis=0).tolist()}


def finite_mean(values, axis):
    """Mean of finite values; NaN only if the whole reduction is undefined."""
    valid = np.isfinite(values)
    count = valid.sum(axis=axis)
    total = np.where(valid, values, 0).sum(axis=axis)
    return np.divide(total, count, out=np.full(np.shape(total), np.nan, dtype=float), where=count > 0)


def rollouts(policies, starts, steps, fields, raw, correct, weights=None, seed=0):
    """Run every policy from the same starts.

    ``policies[name](o)`` builds a step function for object ``o``; ``starts``
    ``[O, S, 2]``; ``fields``/``raw`` ``[O, 3, E, A]`` absolute; ``correct``
    ``{classifier: [O, E, A] bool}``. Returns per-policy arrays averaged over
    starts: channels ``[O, T, 3]``, raw ``[O, T, 3]``, combined ``[O, T]``,
    accuracy ``{classifier: [O, T]}``.
    """
    combined = combine(fields, weights)
    out = {}
    for name, make in policies.items():
        views = np.empty((*starts.shape[:2], steps + 1, 2), dtype=int)
        for o in range(len(starts)):
            step = make(o)
            rng = np.random.default_rng([seed, o])
            for s, start in enumerate(starts[o]):
                views[o, s] = rollout(step, start, steps, rng)
        o_index = np.arange(len(starts))[:, None, None]
        e, a = views[..., 0], views[..., 1]
        out[name] = {
            "channels": finite_mean(fields[o_index, :, e, a], axis=1),  # [O, S, T, 3]
            "raw": finite_mean(raw[o_index, :, e, a], axis=1),
            "combined": finite_mean(combined[o_index, e, a], axis=1),
            "accuracy": {k: v[o_index, e, a].mean(1) for k, v in correct.items()},
            "views": views,
        }
    return out


def summarize(results):
    summary = {}
    for name, r in results.items():
        entry = {"channels": {c: bootstrap(r["channels"][..., i]) for i, c in enumerate(CHANNELS)},
                 "raw": {c: bootstrap(r["raw"][..., i]) for i, c in enumerate(CHANNELS)},
                 "combined": bootstrap(r["combined"]),
                 "accuracy": {k: bootstrap(v) for k, v in r["accuracy"].items()}}
        if r["accuracy"]:
            entry["mean_accuracy"] = bootstrap(np.mean(list(r["accuracy"].values()), axis=0))
        entry["gain"] = {"combined": bootstrap(r["combined"][:, -1] - r["combined"][:, 0]),
                         **{c: bootstrap(r["channels"][:, -1, i] - r["channels"][:, 0, i])
                            for i, c in enumerate(CHANNELS)}}
        if r["accuracy"]:
            accuracy = np.mean(list(r["accuracy"].values()), axis=0)
            entry["gain"]["mean_accuracy"] = bootstrap(accuracy[:, -1] - accuracy[:, 0])
        summary[name] = entry
    if "viewactive" in results:
        va = results["viewactive"]
        summary["viewactive_minus"] = {}
        for name, r in results.items():
            if name == "viewactive":
                continue
            diff = {"combined": bootstrap(va["combined"][:, -1] - r["combined"][:, -1])}
            if va["accuracy"]:
                diff["mean_accuracy"] = bootstrap(np.mean(list(va["accuracy"].values()), axis=0)[:, -1]
                                                  - np.mean(list(r["accuracy"].values()), axis=0)[:, -1])
            summary["viewactive_minus"][name] = diff
    return summary


def plot_fig4(summaries, path, policy="viewactive"):
    """Paper Fig. 4 layout: metrics and classifier accuracy per step, seen and unseen."""
    splits = [s for s in ("seen", "unseen") if s in summaries]
    fig, axes = plt.subplots(len(splits), 2, figsize=(10, 3.6 * len(splits)), squeeze=False)
    labels = {"visible_ratio": "Visible ratio", "normal_entropy": "Normal entropy",
              "visual_entropy": "Visual entropy"}
    for row, split in enumerate(splits):
        entry = summaries[split]["policies"][policy]
        for ax, key, title in ((axes[row, 0], "channels", "metrics"), (axes[row, 1], "accuracy", "accuracy")):
            for name, curve in entry[key].items():
                steps = np.arange(len(curve["mean"]))
                ax.plot(steps, curve["mean"], marker="o", ms=3, label=labels.get(name, name))
                ax.fill_between(steps, curve["lo"], curve["hi"], alpha=0.15)
            ax.set(title=f"{split} classes: {title} ({summaries[split]['objects']} objects)",
                   xlabel="optimization step", ylabel="normalized value" if key == "channels" else "top-1 accuracy")
            ax.grid(alpha=0.3)
            ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_policies(summaries, path):
    """Combined GT quality and mean classifier accuracy per step for every policy."""
    splits = [s for s in ("seen", "unseen") if s in summaries]
    fig, axes = plt.subplots(len(splits), 2, figsize=(10, 3.6 * len(splits)), squeeze=False)
    for row, split in enumerate(splits):
        for ax, key in ((axes[row, 0], "combined"), (axes[row, 1], "mean_accuracy")):
            for name, entry in summaries[split]["policies"].items():
                if key not in entry:
                    continue
                curve = entry[key]
                steps = np.arange(len(curve["mean"]))
                ax.plot(steps, curve["mean"], marker="o", ms=3, label=name)
                ax.fill_between(steps, curve["lo"], curve["hi"], alpha=0.12)
            ax.set(title=f"{split}: {key.replace('_', ' ')}", xlabel="optimization step")
            ax.grid(alpha=0.3)
            ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def heatmap_figure(grid, truth, pred, images, view, title, path, weights=None):
    """Ground-truth vs predicted VQF (absolute frame) for the image at ``view`` = (e, a)."""
    e, a = view
    pred = to_absolute(pred, a)

    def values(field):
        return {(az, el): float(field[i, j]) for i, el in enumerate(grid.elevations)
                for j, az in enumerate(grid.azimuths)}

    mark = {(grid.azimuths[a], grid.elevations[e])}
    panels = [Images({(az, el): images[i][j] for i, el in enumerate(grid.elevations)
                      for j, az in enumerate(grid.azimuths)}, "renders", marks=mark)]
    names = [*CHANNELS, "combined"]
    for label, field in (("truth", truth), ("predicted", pred)):
        stack = [*field, combine(field, weights)]
        panels += [Heatmap(values(f), f"{label}: {n}", 0, 1, "viridis", n, marks=mark)
                   for n, f in zip(names, stack)]
    save_panels(panels, grid.azimuths, grid.elevations, path, title=title, ncols=5,
                note="box = input view; azimuth in the object frame")


def measure_fps(checkpoint, image, n=200, device="cuda"):
    """End-to-end predictions per second at batch 1 (encoders + decoders)."""
    import torch
    from viewactive.model import ViewActive

    model = ViewActive.load(checkpoint, device=device)
    for _ in range(10):
        model.predict([image])
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(n):
        model.predict([image])
    torch.cuda.synchronize()
    return n / (time.perf_counter() - start)


def evaluate(config, net, grid, data, out, *, classifiers, images, device="cuda", log=print):
    """Write ``fields.json``, ``fig4.json`` and plots to ``out``.

    ``data`` holds ``uids``, ``categories``, ``splits``, ``norm``/``raw`` ``[O, 3, E, A]``
    and ``features [O, E, A, D]``; ``classifiers`` maps names to ``[O, E, A]``
    correctness (or is empty); ``images[o][e][a]`` are RGB paths.
    """
    out.mkdir(parents=True, exist_ok=True)
    settings = config["optimize"]
    weights = settings["combine_weights"]
    reach = settings["reach_cells"]
    limits = settings["elevation_range"]
    splits = np.asarray(data["splits"])
    norm, raw = data["norm"], data["raw"]
    truth = relative_truth(norm)                                    # [O, A, 3, E, A]
    train = splits == "train"
    prior = np.nanmean(truth[train].reshape(-1, *truth.shape[2:]), axis=0)
    report, summaries = {"reach_cells": reach, "combine_weights": weights,
                         "elevation_range": limits}, {}
    for number, (split, label) in enumerate(SPLIT_NAMES.items()):
        members = np.flatnonzero(splits == split)
        if not len(members):
            continue
        log(f"[evaluate] {label}: {len(members)} objects")
        pred = predict_all(net, data["features"][members], device)  # [O, E, A, 3, E, A]
        rows = grid.shape[0]
        target = np.broadcast_to(truth[members][:, None], (len(members), rows, *truth.shape[1:]))
        flat_pred = pred.reshape(-1, *pred.shape[3:])
        flat_true = target.reshape(-1, *target.shape[3:])
        report[label] = {"objects": len(members), "views": len(flat_pred),
                         "viewactive": field_metrics(flat_pred, flat_true, weights),
                         "mean_prior": field_metrics(np.broadcast_to(prior, flat_true.shape),
                                                     flat_true, weights)}
        pred_combined = combine(pred, weights)                     # [O, E, A, E, A]
        prior_combined = combine(prior, weights)
        truth_combined = combine(truth[members], weights)         # [O, A, E, A]
        policies = {
            "viewactive": lambda o: field_policy(lambda e, a: pred_combined[o, e, a], grid, reach, limits),
            "gt_field": lambda o: field_policy(lambda e, a: truth_combined[o, a], grid, reach, limits),
            "mean_prior": lambda o: field_policy(lambda e, a: prior_combined, grid, reach, limits),
            "random_walk": lambda o: random_policy(grid, reach, limits),
            "stay": lambda o: stay,
        }
        rng = np.random.default_rng([settings["seed"], number])
        starts = np.stack([rng.integers(0, grid.shape[0], (len(members), settings["starts_per_object"])),
                           rng.integers(0, grid.shape[1], (len(members), settings["starts_per_object"]))], -1)
        correct = {k: v[members] for k, v in classifiers.items()}
        results = rollouts(policies, starts, settings["steps"], norm[members], raw[members], correct,
                           weights, settings["seed"])
        summaries[label] = {"objects": len(members), "policies": summarize(results)}
        if label == "seen":
            for o in members[:: max(1, len(members) // 6)][:6]:
                view = (int(np.argmin(np.abs(np.asarray(grid.elevations) - 15))), 0)
                local = int(np.flatnonzero(members == o)[0])
                heatmap_figure(grid, norm[o], pred[local, view[0], view[1]], images[o], view,
                               f"{data['categories'][o]} {data['uids'][o][:8]}",
                               out / "heatmaps" / f"{data['categories'][o]}_{data['uids'][o]}.png", weights)
    (out / "fields.json").write_text(json.dumps(report, indent=1) + "\n")
    (out / "fig4.json").write_text(json.dumps(summaries, indent=1) + "\n")
    if summaries:
        plot_fig4(summaries, out / "fig4_viewactive.png")
        plot_fig4({k: {**v, "policies": {"gt_field": v["policies"]["gt_field"]}} for k, v in summaries.items()},
                  out / "fig4_gt_field.png", policy="gt_field")
        plot_policies(summaries, out / "policies.png")
    return report, summaries
