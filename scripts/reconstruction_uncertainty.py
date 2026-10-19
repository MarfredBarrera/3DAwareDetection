"""Prototype: three uncertainty estimates for SAM3D reconstructions, per input view.

Needs a study reconstructed with ``reconstruction.save_gaussians: true``.

    # GPU, sam3d env. Laplace on the saved Gaussians -> <reconstruction>/laplace.npz, laplace/*.png
    sam3d-env sam3d python -m scripts.reconstruction_uncertainty laplace --config configs/jacobian_pilot.yaml
    # GPU, sam3d env. Stage-1 Tweedie + seed ensemble -> <reconstruction>/stage1.npz
    sam3d-env sam3d python -m scripts.reconstruction_uncertainty stage1 --config ... [--shard 0 3]
    # core env. Per-object grids and the cross-method comparison -> figures/reconstruction_uncertainty/
    sam3d-env core python -m scripts.reconstruction_uncertainty figures --config ...
    # core env, needs SAM3 detections of the candidates. Ensemble threshold sweep -> threshold.md
    sam3d-env core python -m scripts.reconstruction_uncertainty threshold --config ... [--detections DIR]

Methods (lower = more certain, all in [0, 1]):
- ``laplace`` (reconstruction/uncertainty.py): how much of each candidate view
  the input image constrains; ``unseen`` averaged over the 24 candidates.
- ``tweedie`` (reconstruction/stage1_uncertainty.py): SAM3D stage 1's own
  closed-form posterior spread given x_t*, decoded to occupancy; per t*.
- ``ensemble``: stage 1 rerun from other seeds; spread of the decoded shapes.
Both stage-1 scores are 1 - mean pairwise F-score (1-voxel tolerance) of
occupancy samples on SAM3D's 64^3 object grid.
"""
import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
from PIL import Image

from study import config as settings_io
from study import db as store

COLORMAP = "magma"
FIGURES = "figures/reconstruction_uncertainty"


def _reconstructions(db):
    rows = db.execute("SELECT input_id, directory FROM reconstructions WHERE status='ok' "
                      "ORDER BY input_id").fetchall()
    inputs = {image["image_id"]: image for image in store.images(db, "source='gt'")}
    return [(inputs[input_id], directory) for input_id, directory in rows]


def _pending(config, filename, shard, per_category=None):
    output = Path(config["output_dir"])
    db = store.connect(output)
    rows = _reconstructions(db)
    db.close()
    if per_category is not None:
        # The first uids per category, as the study's object selection orders them.
        chosen = {}
        for image, _ in rows:
            chosen.setdefault(image["category"], set()).add(image["uid"])
        chosen = {uid for uids in chosen.values() for uid in sorted(uids)[:per_category]}
        rows = [(image, directory) for image, directory in rows if image["uid"] in chosen]
    index, count = shard
    return [(image, directory) for position, (image, directory) in enumerate(rows)
            if position % count == index and not (output / directory / filename).is_file()]


def _colorize(values, alpha=None):
    import matplotlib
    colors = matplotlib.colormaps[COLORMAP](np.clip(values, 0, 1))[..., :3]
    if alpha is not None:
        colors[alpha < 0.5] = 1.0
    return (colors * 255).astype(np.uint8)


# ---------------------------------------------------------------- laplace

def laplace(config, probes, prior, shard, limit=None):
    from tools.envs import use_gpu
    use_gpu(config["gpu_id"])
    from reconstruction.uncertainty import score_reconstruction

    output = Path(config["output_dir"])
    recon = config["reconstruction"]
    candidates = [(float(a), float(e)) for e in recon["candidate_elevations"]
                  for a in recon["candidate_azimuths"]]
    pending = _pending(config, "laplace.npz", shard)[:limit]
    print(f"[laplace] {len(pending)} reconstructions to score", flush=True)
    for count, (image, directory) in enumerate(pending, 1):
        started = time.time()
        folder = output / directory
        if not (folder / "gaussians.npz").is_file():
            print(f"[laplace] {image['key']}: no gaussians.npz (save_gaussians off?)", flush=True)
            continue
        gaussian_u, rows = score_reconstruction(
            folder / "gaussians.npz", candidates, recon["candidate_distance"],
            recon["resolution"], image["elevation"], probes=probes, prior=prior)
        (folder / "laplace").mkdir(exist_ok=True)
        for row in rows:
            Image.fromarray(_colorize(row["map"], row["alpha"])).save(
                folder / "laplace" / f"azimuth_{row['azimuth']:g}_elevation_{row['elevation']:g}.png")
        np.savez_compressed(
            folder / "laplace.npz", gaussian_u=gaussian_u.astype(np.float16),
            azimuths=[r["azimuth"] for r in rows], elevations=[r["elevation"] for r in rows],
            unseen=[np.nan if r["unseen"] is None else r["unseen"] for r in rows],
            trace=[r["trace"] for r in rows], probes=probes, prior=prior)
        print(f"[laplace] {count}/{len(pending)} {image['key']} {time.time() - started:.1f}s", flush=True)


# ---------------------------------------------------------------- stage 1

def stage1(config, t_stars, probes, samples, ensemble, shard, limit=None, per_category=None):
    from tools.envs import use_gpu
    use_gpu(config["gpu_id"])
    from reconstruction.sam3d import SAM3D

    output = Path(config["output_dir"])
    pending = _pending(config, "stage1.npz", shard, per_category)[:limit]
    print(f"[stage1] {len(pending)} reconstructions to score", flush=True)
    if not pending:
        return
    model = SAM3D(checkpoint_dir=config["checkpoints"]["sam3d"])
    for count, (image, directory) in enumerate(pending, 1):
        started = time.time()
        result = model.stage1_uncertainty(
            output / image["path"], output / image["mask_path"], seed=config["reconstruction"]["seed"],
            t_stars=t_stars, probes=probes, samples=samples, ensemble=ensemble)
        arrays, scalars = {}, {"probes": probes, "samples": samples, "ensemble": ensemble}
        for name, summary in [("ensemble", result["ensemble"])] + [
                (f"tweedie_{t:g}", value) for t, value in result["tweedie"].items()]:
            arrays[f"{name}_probability"] = summary["probability"]
            arrays[f"{name}_samples"] = summary["samples"]
            scalars[name] = {key: value for key, value in summary.items()
                             if key not in ("probability", "samples")}
        np.savez_compressed(output / directory / "stage1.npz", scalars=json.dumps(scalars), **arrays)
        print(f"[stage1] {count}/{len(pending)} {image['key']} "
              f"ens={scalars['ensemble']['disagreement']:.3f} {time.time() - started:.1f}s", flush=True)


def load_stage1(path):
    data = np.load(path)
    return json.loads(str(data["scalars"])), data


def samples_of(data, name):
    return np.unpackbits(data[f"{name}_samples"], axis=-1).astype(bool)


# ---------------------------------------------------------------- figures

def projection(samples, axis):
    """(64, 64, 3) image of occupancy samples (B, 64, 64, 64) seen along ``axis``.

    Colour = mean ``voxel_disagreement`` of the occupied voxels on the ray (magma
    0 dark -> 1 bright); gray = the samples agree within one voxel; white = empty.
    """
    from reconstruction.stage1_uncertainty import voxel_disagreement

    disputed = voxel_disagreement(samples)
    present = np.isfinite(disputed).any(axis=axis)
    occupied = np.isfinite(disputed)
    mean = np.where(present, np.where(occupied, disputed, 0).sum(axis=axis)
                    / np.maximum(occupied.sum(axis=axis), 1), 0)
    image = _colorize(mean).astype(np.float32)
    image[mean < 0.05] = [190, 190, 190]
    image[~present] = 255
    return np.flipud(np.ascontiguousarray(np.moveaxis(image, 0, 1))).astype(np.uint8)


def _projection_png(samples, axis, path, size=96):
    Image.fromarray(projection(samples, axis)).resize((size, size), Image.NEAREST).save(path)
    return path


def figures(config, laplace_view, t_main, axis):
    from study.figures import Heatmap, Images, save_panels

    output = Path(config["output_dir"])
    dataset = config["dataset"]
    db = store.connect(output)
    objects = {}
    for image, directory in _reconstructions(db):
        objects.setdefault((image["category"], image["uid"]), []).append((image, directory))
    db.close()
    directory_out = output / FIGURES
    records = []
    for (category, uid), members in sorted(objects.items()):
        inputs, maps, scores = {}, {"laplace": {}, "ensemble": {}, "tweedie": {}}, {}
        for image, directory in members:
            folder = output / directory
            if not (folder / "laplace.npz").is_file() or not (folder / "stage1.npz").is_file():
                continue
            angle = (float(image["azimuth"]), float(image["elevation"]))
            lap = np.load(folder / "laplace.npz")
            scalars, data = load_stage1(folder / "stage1.npz")
            record = {"category": category, "uid": uid, "azimuth": angle[0], "elevation": angle[1],
                      "directory": directory, "laplace": float(np.nanmean(lap["unseen"])),
                      "ensemble": scalars["ensemble"]["disagreement"]}
            for name, value in scalars.items():
                if name.startswith("tweedie_"):
                    record[name] = value["disagreement"]
            records.append(record)
            inputs[angle] = output / image["path"]
            maps["laplace"][angle] = (folder / "laplace" /
                                      f"azimuth_{laplace_view[0]:g}_elevation_{laplace_view[1]:g}.png")
            (folder / "stage1").mkdir(exist_ok=True)
            maps["ensemble"][angle] = _projection_png(samples_of(data, "ensemble"), axis,
                                                      folder / "stage1" / f"ensemble_axis{axis}.png")
            maps["tweedie"][angle] = _projection_png(samples_of(data, f"tweedie_{t_main:g}"), axis,
                                                     folder / "stage1" / f"tweedie_{t_main:g}_axis{axis}.png")
            for key in ("laplace", "ensemble", f"tweedie_{t_main:g}"):
                scores.setdefault(key, {})[angle] = record[key]
        if not inputs:
            continue
        top = max(1e-6, max(v for key in scores if key != "laplace"
                            for v in scores[key].values() if v is not None))
        heat = lambda key, title: Heatmap(scores[key], title, 0, None if key == "laplace" else top,
                                          COLORMAP, "uncertainty", annotate=True)
        save_panels(
            [Images(inputs, "input views (ground truth)"), Images(inputs, "input views (ground truth)"),
             Images(maps["laplace"], f"Laplace: unconstrained-by-input map, SAM3D view {laplace_view}"),
             heat("laplace", "Laplace: mean over 24 candidates"),
             Images(maps["ensemble"], f"ensemble: occupancy uncertainty, projected on grid axis {axis}"),
             heat("ensemble", "ensemble: 1 - mean pairwise F-score"),
             Images(maps["tweedie"], f"Tweedie t*={t_main:g}: occupancy uncertainty, axis {axis}"),
             heat(f"tweedie_{t_main:g}", f"Tweedie t*={t_main:g}: 1 - mean pairwise F-score")],
            dataset["azimuths"], dataset["elevations"], directory_out / f"{category}_{uid[:8]}.png",
            title=f"{category} {uid[:8]}: input views and three uncertainty estimates", ncols=2,
            note="stage-1 maps: gray = all samples agree within 1 voxel, colour = mean share of samples "
                 "missing the voxels on each ray (magma: 0 dark -> 1 bright); "
                 "stage-1 heatmaps share one colour scale per object")
        worst = max((r for r in records if r["uid"] == uid), key=lambda r: r["ensemble"] or 0)
        explainer(output, worst, laplace_view, axis, directory_out / f"{category}_{uid[:8]}_samples.png")
    directory_out.mkdir(parents=True, exist_ok=True)
    keys = list(records[0]) if records else []
    (directory_out / "inputs.csv").write_text(
        "\n".join([",".join(keys)] + [",".join(str(r[k]) for k in keys) for r in records]) + "\n")
    compare(records, directory_out)
    print(f"[figures] -> {directory_out}", flush=True)


def explainer(output, record, laplace_view, axis, path):
    """One input: what each method actually looks at, sample by sample."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    folder = output / record["directory"]
    scalars, data = load_stage1(folder / "stage1.npz")
    tweedies = sorted((float(k.split("_")[1]), k) for k in scalars if k.startswith("tweedie_"))
    rows = [("ensemble", "ensemble"), *[(f"Tweedie t*={t:g}", key) for t, key in tweedies]]
    columns = 9
    fig, axes = plt.subplots(len(rows) + 2, columns, figsize=(columns * 1.3, (len(rows) + 2) * 1.45))
    for ax in axes.flat:
        ax.axis("off")
    db = store.connect(output)
    image = store.images(db, "source='gt' AND uid=? AND azimuth=? AND elevation=?",
                         (record["uid"], record["azimuth"], record["elevation"]))[0]
    db.close()
    axes[0, 0].imshow(Image.open(output / image["path"]))
    axes[0, 0].set_title("input", fontsize=8)
    lap_files = sorted((folder / "laplace").glob("*.png"),
                       key=lambda p: (float(p.stem.split("_")[3]), float(p.stem.split("_")[1])))
    for ax, file in zip(list(axes[0, 1:]) + list(axes[1, :]), lap_files[:columns - 1 + columns]):
        ax.imshow(Image.open(file))
        az, el = file.stem.split("_")[1], file.stem.split("_")[3]
        ax.set_title(f"Laplace {az},{el}", fontsize=7)
    for row, (label, key) in enumerate(rows, 2):
        occupancies = samples_of(data, key)
        for col in range(min(columns - 1, len(occupancies))):
            axes[row, col].imshow(_silhouette(occupancies[col], axis), cmap="gray", vmin=0, vmax=255)
            axes[row, col].set_title(f"{label.split(' ')[0]} {col}", fontsize=7)
        axes[row, columns - 1].imshow(projection(occupancies, axis))
        axes[row, columns - 1].set_title(f"{label}\nscore {scalars[key]['disagreement']:.3f}", fontsize=7)
    fig.suptitle(f"{record['category']} {record['uid'][:8]} input ({record['azimuth']:g}, "
                 f"{record['elevation']:g}): samples behind each estimate "
                 f"(occupancy seen along grid axis {axis}; last column = uncertainty)", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def _silhouette(occupancy, axis):
    """Depth-shaded projection of one boolean 64^3 sample (darker = nearer)."""
    depth = np.argmax(occupancy, axis=axis).astype(np.float32)
    present = occupancy.any(axis=axis)
    shade = np.where(present, 60 + 150 * depth / occupancy.shape[axis], 255).astype(np.uint8)
    return np.flipud(np.moveaxis(shade, 0, 1))


def _spearman(a, b):
    from scipy.stats import spearmanr
    keep = [(x, y) for x, y in zip(a, b) if x is not None and y is not None
            and math.isfinite(x) and math.isfinite(y)]
    return float(spearmanr(*zip(*keep)).statistic) if len(keep) > 2 else float("nan")


def compare(records, directory):
    """Scatter matrix over all inputs plus pooled and within-object rank correlations."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if not records:
        return
    keys = ["laplace", "ensemble"] + sorted(
        (k for k in records[0] if k.startswith("tweedie_")), key=lambda k: float(k.split("_")[1]))
    categories = sorted({r["category"] for r in records})
    colors = {c: plt.get_cmap("tab10")(i) for i, c in enumerate(categories)}
    n = len(keys)
    fig, axes = plt.subplots(n, n, figsize=(2.3 * n, 2.3 * n))
    lines = ["| | " + " | ".join(keys) + " |", "|---" * (n + 1) + "|"]
    for i, ki in enumerate(keys):
        row = []
        for j, kj in enumerate(keys):
            ax = axes[i, j]
            pooled = _spearman([r[kj] for r in records], [r[ki] for r in records])
            within = np.nanmean([_spearman([r[kj] for r in records if r["uid"] == uid],
                                           [r[ki] for r in records if r["uid"] == uid])
                                 for uid in {r["uid"] for r in records}])
            row.append(f"{pooled:+.2f} ({within:+.2f})")
            if i == j:
                ax.hist([r[ki] for r in records if r[ki] is not None], bins=20, color="#888")
            else:
                for category in categories:
                    members = [r for r in records if r["category"] == category]
                    ax.scatter([r[kj] for r in members], [r[ki] for r in members], s=6,
                               color=colors[category], label=category)
                ax.set_title(f"ρ {pooled:+.2f} / within {within:+.2f}", fontsize=7)
            if i == n - 1:
                ax.set_xlabel(kj, fontsize=8)
            if j == 0:
                ax.set_ylabel(ki, fontsize=8)
            ax.tick_params(labelsize=6)
        lines.append(f"| {ki} | " + " | ".join(row) + " |")
    axes[0, 1].legend(fontsize=6, markerscale=2)
    fig.suptitle("Uncertainty estimates per input view (Spearman ρ pooled / mean within object)")
    fig.tight_layout()
    fig.savefig(directory / "comparison_scatter.png", dpi=110)
    plt.close(fig)

    # Per-object mean and the view dependence each method sees.
    fig, ax = plt.subplots(figsize=(1.6 + 1.2 * len(keys), 3.2))
    uids = sorted({(r["category"], r["uid"]) for r in records})
    width = 0.8 / len(uids)
    for k, (category, uid) in enumerate(uids):
        values = [[r[key] for r in records if r["uid"] == uid and r[key] is not None] for key in keys]
        ax.bar(np.arange(len(keys)) + k * width, [np.mean(v) for v in values], width,
               yerr=[np.std(v) for v in values], color=colors[category], label=f"{category} {uid[:6]}")
    ax.set_xticks(np.arange(len(keys)) + 0.4 - width / 2, keys, fontsize=8)
    ax.set_ylabel("mean over 60 input views (± sd)")
    ax.legend(fontsize=6)
    ax.set_title("Per-object uncertainty; error bars = how much it varies with the input view", fontsize=9)
    fig.tight_layout()
    fig.savefig(directory / "comparison_objects.png", dpi=110)
    plt.close(fig)
    (directory / "comparison.md").write_text(
        "Spearman ρ between methods over all inputs (in parentheses: mean within-object ρ "
        "across the 60 views).\n\n" + "\n".join(lines) + "\n")


def _auroc(scores, labels):
    """P(score of a positive > score of a negative), ties count half; NaN without both classes."""
    from scipy.stats import rankdata
    scores, labels = np.asarray(scores, float), np.asarray(labels, bool)
    positives, negatives = labels.sum(), (~labels).sum()
    if not positives or not negatives:
        return float("nan")
    ranks = rankdata(scores)
    return float((ranks[labels].sum() - positives * (positives + 1) / 2) / (positives * negatives))


def _sweep(scores, labels, thresholds):
    """Per threshold: flag ``score > threshold`` as unreliable; detection of ``labels``."""
    scores, labels = np.asarray(scores, float), np.asarray(labels, bool)
    rows = []
    for threshold in thresholds:
        flagged = scores > threshold
        tp, fp = (flagged & labels).sum(), (flagged & ~labels).sum()
        fn, tn = (~flagged & labels).sum(), (~flagged & ~labels).sum()
        precision = tp / (tp + fp) if tp + fp else float("nan")
        recall = tp / (tp + fn) if tp + fn else float("nan")
        specificity = tn / (tn + fp) if tn + fp else float("nan")
        rows.append({"threshold": float(threshold), "flagged": float(flagged.mean()),
                     "precision": precision, "recall": recall,
                     "f1": 2 * tp / (2 * tp + fp + fn) if tp else 0.0,
                     "youden": recall + specificity - 1,
                     "kept_unreliable": float(labels[~flagged].mean()) if (~flagged).any() else float("nan")})
    return rows


def _best(rows):
    """Row with the highest Youden J (NaN last), lowest threshold on ties."""
    return max(rows, key=lambda row: (np.nan_to_num(row["youden"], nan=-np.inf), -row["threshold"]))


def threshold(config, detections, reliable_accuracy):
    """Sweep ensemble-disagreement thresholds for flagging unreliable reconstructions.

    A reconstruction is unreliable for detection when SAM3's top-1 is wrong on
    more than ``1 - reliable_accuracy`` of its candidate renders. The best
    threshold maximizes Youden's J; leave-one-object-out picks it on the other
    objects and scores the held-out one.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from study.labels import outcomes

    output = Path(config["output_dir"])
    db = store.connect(output)
    members = _reconstructions(db)
    db.close()
    det = store.connect(detections)
    vocabulary = [row[0] for row in det.execute("SELECT DISTINCT prompt FROM detections ORDER BY prompt")]
    before = outcomes(det, vocabulary, config["detection"], "i.source='gt'")
    after = outcomes(det, vocabulary, config["detection"], "i.source='sam3d'")
    children = {}
    for image in store.images(det, "source='sam3d'"):
        if image["image_id"] in after:
            children.setdefault(image["parent_id"], []).append(image)
    det.close()
    records = []
    for image, directory in members:
        path = output / directory / "stage1.npz"
        views = children.get(image["image_id"])
        if not path.is_file() or not views or image["image_id"] not in before:
            continue
        score = load_stage1(path)[0]["ensemble"]["disagreement"]
        if score is None:
            continue
        identity = [after[v["image_id"]] for v in views if (v["azimuth"], v["elevation"]) == (180, 0)]
        records.append({"category": image["category"], "uid": image["uid"],
                        "azimuth": image["azimuth"], "elevation": image["elevation"], "ensemble": score,
                        "input_correct": float(before[image["image_id"]]["correct"]),
                        "identity_correct": float(identity[0]["correct"]) if identity else float("nan"),
                        "candidate_accuracy": float(np.mean([after[v["image_id"]]["correct"] for v in views])),
                        "candidate_margin": float(np.mean([after[v["image_id"]]["margin"] for v in views]))})
    if not records:
        raise SystemExit("no inputs with both stage1.npz and candidate detections")
    scores = np.array([r["ensemble"] for r in records])
    labels = np.array([r["candidate_accuracy"] < reliable_accuracy for r in records])
    uids = np.array([r["uid"] for r in records])
    thresholds = np.concatenate([[0.0], np.unique(scores)])
    sweep = _sweep(scores, labels, thresholds)
    best = _best(sweep)
    held_out = np.zeros(len(records), bool)
    chosen = {}
    for uid in sorted(set(uids)):
        train = uids != uid
        rows = _sweep(scores[train], labels[train], thresholds)
        chosen[uid] = _best(rows)["threshold"]
        held_out[~train] = scores[~train] > chosen[uid]
    loo = {"precision": float(labels[held_out].mean()) if held_out.any() else float("nan"),
           "recall": float(held_out[labels].mean()) if labels.any() else float("nan"),
           "flagged": float(held_out.mean())}
    within = [_auroc(scores[uids == uid], labels[uids == uid]) for uid in sorted(set(uids))]

    directory_out = output / FIGURES
    directory_out.mkdir(parents=True, exist_ok=True)
    keys = list(records[0])
    (directory_out / "threshold_inputs.csv").write_text(
        "\n".join([",".join(keys + ["unreliable"])] +
                  [",".join([str(r[k]) for k in keys] + [str(int(label))])
                   for r, label in zip(records, labels)]) + "\n")
    categories = sorted({r["category"] for r in records})
    colors = {c: plt.get_cmap("tab10")(i) for i, c in enumerate(categories)}
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for category in categories:
        mask = np.array([r["category"] == category for r in records])
        axes[0].scatter(scores[mask], [r["candidate_accuracy"] for r, m in zip(records, mask) if m],
                        s=10, color=colors[category], label=category)
    axes[0].axhline(reliable_accuracy, color="k", lw=0.8, ls=":")
    axes[0].axvline(best["threshold"], color="r", lw=1)
    axes[0].set(xlabel="ensemble disagreement", ylabel="SAM3 top-1 accuracy over 24 candidates",
                title=f"one dot per input view; red = best threshold {best['threshold']:.3f}")
    axes[0].legend(fontsize=7)
    x = [row["threshold"] for row in sweep]
    for key in ("precision", "recall", "f1", "youden"):
        axes[1].plot(x, [row[key] for row in sweep], label=key)
    axes[1].plot(x, [row["flagged"] for row in sweep], color="#888", ls="--", label="share flagged")
    axes[1].axvline(best["threshold"], color="r", lw=1)
    axes[1].set(xlabel="threshold (flag if disagreement > threshold)",
                title=f"flagging unreliable reconstructions (accuracy < {reliable_accuracy:g})")
    axes[1].legend(fontsize=7)
    kept = [1 - row["flagged"] for row in sweep]
    axes[2].plot(kept, [1 - row["kept_unreliable"] for row in sweep], color="k")
    axes[2].scatter([1 - best["flagged"]], [1 - best["kept_unreliable"]], color="r", zorder=3)
    axes[2].set(xlabel="share of inputs kept (disagreement <= threshold)",
                ylabel="share of kept inputs that are reliable",
                title="what the threshold buys")
    fig.suptitle(f"Ensemble threshold sweep: {len(records)} inputs, {len(set(uids))} objects, "
                 f"{labels.sum()} unreliable")
    fig.tight_layout()
    fig.savefig(directory_out / "threshold_sweep.png", dpi=110)
    plt.close(fig)
    lines = [f"Unreliable = SAM3 top-1 accuracy over the 24 candidate renders < {reliable_accuracy:g}; "
             f"{labels.sum()}/{len(records)} inputs. Flag when ensemble disagreement > threshold.", "",
             f"- AUROC pooled {_auroc(scores, labels):.2f}; within object "
             + ", ".join(f"{u[:6]} {a:.2f}" for u, a in zip(sorted(set(uids)), within)),
             f"- best threshold (max Youden J) {best['threshold']:.4f}: J {best['youden']:.2f}, "
             f"precision {best['precision']:.2f}, recall {best['recall']:.2f}, flags {best['flagged']:.0%}",
             "- leave-one-object-out thresholds: "
             + ", ".join(f"{u[:6]} {t:.4f}" for u, t in chosen.items())
             + f"; held-out precision {loo['precision']:.2f}, recall {loo['recall']:.2f}, "
             f"flags {loo['flagged']:.0%}", "",
             "| threshold | flagged | precision | recall | F1 | J |", "|---|---|---|---|---|---|"]
    for row in sweep[::max(1, len(sweep) // 25)] + [best]:
        lines.append(f"| {row['threshold']:.4f} | {row['flagged']:.0%} | {row['precision']:.2f} | "
                     f"{row['recall']:.2f} | {row['f1']:.2f} | {row['youden']:.2f} |")
    (directory_out / "threshold.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines[:5]), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("stage", choices=["laplace", "stage1", "figures", "threshold"])
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--probes", type=int, default=None,
                        help="Rademacher probes per camera (laplace, default 16) or per t* (stage1, default 8)")
    parser.add_argument("--prior", type=float, default=0.01,
                        help="laplace: prior precision as a fraction of each group's mean information")
    parser.add_argument("--t-stars", type=float, nargs="*", default=[0.2, 0.5, 0.8],
                        help="stage1: Tweedie t* values; pass none (`--t-stars`) for the ensemble only")
    parser.add_argument("--samples", type=int, default=8, help="stage1: Tweedie draws per t*")
    parser.add_argument("--ensemble", type=int, default=8, help="stage1: seeds including the reconstruction's")
    parser.add_argument("--laplace-view", type=float, nargs=2, default=(0.0, 0.0),
                        help="figures: SAM3D-frame (azimuth, elevation) of the Laplace maps")
    parser.add_argument("--t-main", type=float, default=0.5, help="figures: t* shown in the grids")
    parser.add_argument("--axis", type=int, default=0, help="figures: occupancy-grid projection axis")
    parser.add_argument("--shard", type=int, nargs=2, default=(0, 1), metavar=("INDEX", "COUNT"))
    parser.add_argument("--gpu", type=int, help="override the config's gpu_id")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--objects-per-category", type=int,
                        help="stage1: only the first N objects (by uid) of each category")
    parser.add_argument("--detections", type=Path,
                        help="threshold: study directory whose DB has SAM3 detections (default: output_dir)")
    parser.add_argument("--reliable-accuracy", type=float, default=0.5,
                        help="threshold: candidate top-1 accuracy below which a reconstruction is unreliable")
    args = parser.parse_args()
    config = settings_io.load(args.config)
    if args.gpu is not None:
        config["gpu_id"] = args.gpu
    if args.stage == "laplace":
        laplace(config, args.probes or 16, args.prior, args.shard, args.limit)
    elif args.stage == "stage1":
        stage1(config, args.t_stars, args.probes or 8, args.samples, args.ensemble, args.shard, args.limit,
               args.objects_per_category)
    elif args.stage == "figures":
        figures(config, tuple(args.laplace_view), args.t_main, args.axis)
    else:
        threshold(config, args.detections or Path(config["output_dir"]), args.reliable_accuracy)


if __name__ == "__main__":
    main()
