"""Prototype: uncertainty estimates for SAM3D stage-1 reconstructions, per input view.

    # GPU, sam3d env. Stage-1 seed ensemble -> <reconstruction>/stage1.npz, for reconstructions
    # made without reconstruction.ensemble (which writes it during the reconstruct stage)
    sam3d-env sam3d python -m scripts.reconstruction_uncertainty stage1 --config ... [--shard 0 3]
    # GPU, sam3d env. Last-layer Laplace: diagonal Fisher once -> <output>/laplace_fisher.npz,
    # then per reconstruction -> <reconstruction>/laplace.npz
    sam3d-env sam3d python -m scripts.reconstruction_uncertainty fisher --config ...
    sam3d-env sam3d python -m scripts.reconstruction_uncertainty laplace --config ... [--prior P] [--data-size N]
    # GPU, sam3d env. Image-shape mutual information -> <reconstruction>/information.npz
    sam3d-env sam3d python -m scripts.reconstruction_uncertainty information --config ...
    # core env. Per-object grids and the cross-method comparison -> figures/reconstruction_uncertainty/
    sam3d-env core python -m scripts.reconstruction_uncertainty figures --config ...
    # GPU, core env, after the pick stage. SAM3 on the real render at each gt_vqf_max target
    # view (the "disagree"/"harm" criteria) -> <reconstruction>/agreement.json
    sam3d-env core python -m scripts.reconstruction_uncertainty agreement --config ...
    # core env, needs the analyze stage and stage1.npz. Before/after figures for: abstain when the
    # ensemble disagreement > lambda, else classify the --selector view -> figures/after_ensemble<lambda>/
    sam3d-env core python -m scripts.reconstruction_uncertainty after --config ... --lambdas 0.2
    # core env, needs stage1.npz and gt_error.json. Per object and lambda: input views, mesh F-score and
    # disagreement, rejected views outlined -> figures/rejection_grids/
    sam3d-env core python -m scripts.reconstruction_uncertainty rejection --config ... [--lambdas 0.1 0.2 0.3]
    # core env, needs SAM3 detections of the picks. Abstention threshold by Learn-then-Test on
    # held-out categories -> threshold_<method>_<criterion>.md, risk_coverage_<method>_<criterion>.png
    sam3d-env core python -m scripts.reconstruction_uncertainty threshold --config ... [--criterion mesh]
    # sam3d env, CPU. Shape error against the ground-truth mesh -> <reconstruction>/gt_error.json
    sam3d-env sam3d python -m scripts.reconstruction_uncertainty gt_error --config ... [--requires '']
    # GPU, sam3d env. SAM3D's shape prior (image removed) for each object's 3 explainer views -> prior.npz
    sam3d-env sam3d python -m scripts.reconstruction_uncertainty prior --config ... --uids UID ...
    # GPU, sam3d env. Reconstruction vs SAM3D's one-step mean shape -> <reconstruction>/meangap.npz;
    # core env: how well it and fewer seeds track an independent 4-seed ensemble -> figures/meangap/
    sam3d-env sam3d python -m scripts.reconstruction_uncertainty meangap --config ... --uids UID ...
    sam3d-env core python -m scripts.reconstruction_uncertainty meangap_test --config ...
    # core env. Per object: ensemble vs information over its views, and the explainer
    # -> figures/views_uncertainty/
    sam3d-env core python -m scripts.reconstruction_uncertainty views --config ... --uids UID ...
    sam3d-env core python -m scripts.reconstruction_uncertainty explain --config ... --uids UID ...
    # core env. Laplace proof of concept (rank stability over data sizes, extremes) -> figures/laplace_poc/
    sam3d-env core python -m scripts.reconstruction_uncertainty poc --config ... --names laplace.npz laplace_n10.npz

Methods (lower = more certain):
- ``ensemble``: stage 1 rerun from other seeds (fixed weights); sampling spread.
- ``laplace``: stage 1 rerun from the same seed with shape-head weights drawn
  from a last-layer Laplace posterior (Jazbec et al., arXiv:2502.20946); model
  (epistemic) spread. ``laplace_entropy`` is the paper's score: Gaussian entropy
  in the shape-latent space (nats per dimension, not bounded); with a realistic
  posterior the occupancy rarely changes, so prefer it over ``laplace``.
- ``information``: pointwise mutual information (nats) between the input and the
  stage-1 shape, from conditional vs unconditional velocities (Kong et al.,
  arXiv:2310.07972). Not an uncertainty: low = the image barely shaped the reconstruction.
The occupancy scores are 1 - mean pairwise F-score (1-voxel tolerance) of
occupancy samples on SAM3D's 64^3 object grid, in [0, 1].
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


def _pending(config, filename, shard, per_category=None, uids=None, elevations=None):
    output = Path(config["output_dir"])
    db = store.connect(output)
    rows = _reconstructions(db)
    db.close()
    if uids:
        rows = [(image, directory) for image, directory in rows if image["uid"] in uids]
    if elevations:
        rows = [(image, directory) for image, directory in rows if image["elevation"] in elevations]
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

FISHER = "laplace_fisher.npz"


def fisher(config, inputs, times, split, limit=None):
    """Diagonal Fisher of the shape head over ``inputs`` reconstructions of ``split``.

    Gaussian likelihood of the target velocity, summed over the 4096 x 8 shape
    latent; its noise sigma^2 is the mean squared residual.
    """
    from tools.envs import use_gpu
    use_gpu(config["gpu_id"])
    from reconstruction.sam3d import SAM3D

    output = Path(config["output_dir"])
    manifest = json.loads(Path(config["dataset"]["manifest"]).read_text())
    db = store.connect(output)
    rows = [(image, directory) for image, directory in _reconstructions(db)
            if manifest.get(image["uid"], {}).get("split") == split]
    db.close()
    # Spread over categories: a fixed shuffle of the split's inputs.
    rows = [rows[i] for i in np.random.default_rng(0).permutation(len(rows))][:inputs][:limit]
    if not rows:
        raise SystemExit(f"no reconstructions in split {split!r}")
    sampler = np.random.default_rng(1)
    model = SAM3D(checkpoint_dir=config["checkpoints"]["sam3d"])
    total, count = None, 0
    for position, (image, _) in enumerate(rows, 1):
        started = time.time()
        # The training time sampler: logit-normal(-1, 1), t = 1 at data.
        drawn = list(1 / (1 + np.exp(-sampler.normal(-1.0, 1.0, size=times))))
        terms = model.stage1_fisher(output / image["path"], output / image["mask_path"], drawn,
                                    seed=config["reconstruction"]["seed"])
        total = terms if total is None else {key: total[key] + terms[key] for key in terms}
        count += times
        print(f"[fisher] {position}/{len(rows)} {image['key']} {time.time() - started:.1f}s", flush=True)
    # Gaussian likelihood of the target velocity; noise = mean squared residual.
    noise = total["squared"] / total["elements"]
    weight, bias = total["weight"].numpy() / noise ** 2, total["bias"].numpy() / noise ** 2
    np.savez(output / FISHER, weight=weight, bias=bias, count=count, noise=noise,
             inputs=[image["key"] for image, _ in rows], split=split)
    print(f"[fisher] {count} terms; noise sigma^2 {noise:.3g}; mean Fisher weight "
          f"{weight.mean() / count:.3g} (median {np.median(weight) / count:.3g})", flush=True)


def laplace(config, prior, data_size, samples, noise, shard, limit=None, per_category=None,
            name="laplace.npz", uids=None, elevations=None, fisher_path=None):
    from tools.envs import use_gpu
    use_gpu(config["gpu_id"])
    import torch
    from reconstruction.sam3d import SAM3D
    from reconstruction.stage1_uncertainty import evidence_prior, posterior_std, shape_head

    output = Path(config["output_dir"])
    pending = _pending(config, name, shard, per_category, uids, elevations)[:limit]
    print(f"[laplace] {len(pending)} reconstructions to score -> {name}", flush=True)
    if not pending:
        return
    model = SAM3D(checkpoint_dir=config["checkpoints"]["sam3d"])
    data = np.load(fisher_path or output / FISHER)
    fisher_sum = {key: data[key] for key in ("weight", "bias")}
    head = shape_head(model._inference._pipeline.models["ss_generator"])
    parameters = {key: getattr(head, key).detach().float().cpu().numpy() for key in fisher_sum}
    if prior is None:
        prior = evidence_prior(fisher_sum, int(data["count"]), data_size, parameters)
    std = posterior_std({key: torch.from_numpy(value) for key, value in fisher_sum.items()},
                        int(data["count"]), data_size, prior)
    print(f"[laplace] prior precision {prior:.3g}; posterior std weight mean {std['weight'].mean():.3g} "
          f"(MAP weight rms {np.sqrt(np.mean(parameters['weight'] ** 2)):.3g})", flush=True)
    for count, (image, directory) in enumerate(pending, 1):
        started = time.time()
        result = model.stage1_laplace(output / image["path"], output / image["mask_path"], std,
                                      seed=config["reconstruction"]["seed"], samples=samples, noise=noise)
        scalars = {"laplace": {key: value for key, value in result.items()
                               if key not in ("probability", "samples")},
                   "prior": prior, "data_size": data_size, "samples": samples, "noise": noise}
        np.savez_compressed(output / directory / name, scalars=json.dumps(scalars),
                            laplace_probability=result["probability"], laplace_samples=result["samples"])
        print(f"[laplace] {count}/{len(pending)} {image['key']} "
              f"dis={result['disagreement'] if result['disagreement'] is not None else float('nan'):.3f} "
              f"H={result['entropy']:.3f} {time.time() - started:.1f}s", flush=True)


# ---------------------------------------------------------------- stage 1

def stage1(config, ensemble, shard, limit=None, per_category=None, uids=None,
           elevations=None):
    from tools.envs import use_gpu
    use_gpu(config["gpu_id"])
    from reconstruction.sam3d import SAM3D
    from reconstruction.stage1_uncertainty import save_ensemble

    output = Path(config["output_dir"])
    pending = _pending(config, "stage1.npz", shard, per_category, uids, elevations)[:limit]
    print(f"[stage1] {len(pending)} reconstructions to score", flush=True)
    if not pending:
        return
    model = SAM3D(checkpoint_dir=config["checkpoints"]["sam3d"])
    for count, (image, directory) in enumerate(pending, 1):
        started = time.time()
        summary = model.stage1_ensemble(output / image["path"], output / image["mask_path"],
                                        seed=config["reconstruction"]["seed"], ensemble=ensemble)
        save_ensemble(output / directory / "stage1.npz", summary, ensemble)
        print(f"[stage1] {count}/{len(pending)} {image['key']} "
              f"ens={summary['disagreement']:.3f} {time.time() - started:.1f}s", flush=True)


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


def _scores(folder):
    """Per-method scores and their sample arrays for one reconstruction (missing files skipped)."""
    record, sources = {}, {}
    for filename in ("stage1.npz", "laplace.npz"):
        if (folder / filename).is_file():
            scalars, data = load_stage1(folder / filename)
            for name, value in scalars.items():
                if isinstance(value, dict) and "disagreement" in value:
                    record[name] = value["disagreement"]
                    sources[name] = data
                    if "entropy" in value:
                        record[f"{name}_entropy"] = value["entropy"]
    return record, sources


def figures(config, axis):
    from study.figures import Heatmap, Images, save_panels

    output = Path(config["output_dir"])
    dataset = config["dataset"]
    db = store.connect(output)
    objects = {}
    for image, directory in _reconstructions(db):
        objects.setdefault((image["category"], image["uid"]), []).append((image, directory))
    db.close()
    directory_out = output / FIGURES
    shown = ["ensemble", "laplace"]
    records = []
    for (category, uid), members in sorted(objects.items()):
        inputs, maps, scores = {}, {}, {}
        for image, directory in members:
            folder = output / directory
            scored, sources = _scores(folder)
            if not scored:
                continue
            angle = (float(image["azimuth"]), float(image["elevation"]))
            record = {"category": category, "uid": uid, "azimuth": angle[0], "elevation": angle[1],
                      "directory": directory, **scored}
            records.append(record)
            inputs[angle] = output / image["path"]
            (folder / "stage1").mkdir(exist_ok=True)
            for name in shown:
                if name in sources:
                    maps.setdefault(name, {})[angle] = _projection_png(
                        samples_of(sources[name], name), axis, folder / "stage1" / f"{name}_axis{axis}.png")
                    scores.setdefault(name, {})[angle] = record[name]
        if not maps:
            continue
        top = max(1e-6, max(v for values in scores.values() for v in values.values() if v is not None))
        panels = [Images(inputs, "input views (ground truth)"), Images(inputs, "input views (ground truth)")]
        for name in maps:
            panels += [Images(maps[name], f"{name}: occupancy uncertainty, projected on grid axis {axis}"),
                       Heatmap(scores[name], f"{name}: 1 - mean pairwise F-score", 0, top, COLORMAP,
                               "uncertainty", annotate=True)]
        save_panels(
            panels, dataset["azimuths"], dataset["elevations"], directory_out / f"{category}_{uid[:8]}.png",
            title=f"{category} {uid[:8]}: input views and stage-1 uncertainty estimates", ncols=2,
            note="maps: gray = all samples agree within 1 voxel, colour = mean share of samples "
                 "missing the voxels on each ray (magma: 0 dark -> 1 bright); "
                 "heatmaps share one colour scale per object")
        worst = max((r for r in records if r["uid"] == uid), key=lambda r: r.get("ensemble") or 0)
        explainer(output, worst, axis, directory_out / f"{category}_{uid[:8]}_samples.png")
    directory_out.mkdir(parents=True, exist_ok=True)
    keys = sorted({k for r in records for k in r}, key=lambda k: (k not in records[0], k)) if records else []
    (directory_out / "inputs.csv").write_text(
        "\n".join([",".join(keys)] + [",".join(str(r.get(k, "")) for k in keys) for r in records]) + "\n")
    compare(records, directory_out)
    print(f"[figures] -> {directory_out}", flush=True)


def explainer(output, record, axis, path):
    """One input: the occupancy samples behind each method."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    folder = output / record["directory"]
    _, sources = _scores(folder)
    rows = sorted(sources)
    columns = 9
    fig, axes = plt.subplots(len(rows) + 1, columns, figsize=(columns * 1.3, (len(rows) + 1) * 1.45))
    for ax in axes.flat:
        ax.axis("off")
    db = store.connect(output)
    image = store.images(db, "source='gt' AND uid=? AND azimuth=? AND elevation=?",
                         (record["uid"], record["azimuth"], record["elevation"]))[0]
    db.close()
    axes[0, 0].imshow(Image.open(output / image["path"]))
    axes[0, 0].set_title("input", fontsize=8)
    for row, key in enumerate(rows, 1):
        occupancies = samples_of(sources[key], key)
        for col in range(min(columns - 1, len(occupancies))):
            axes[row, col].imshow(_silhouette(occupancies[col], axis), cmap="gray", vmin=0, vmax=255)
            axes[row, col].set_title(f"{key.split('_')[0]} {col}", fontsize=7)
        axes[row, columns - 1].imshow(projection(occupancies, axis))
        axes[row, columns - 1].set_title(f"{key}\nscore {record[key]:.3f}", fontsize=7)
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
    present = {k for r in records for k in r}
    keys = [k for k in ("ensemble", "laplace", "laplace_entropy") if k in present]
    records = [{"category": r["category"], "uid": r["uid"],
                **{k: np.nan if r.get(k) is None else r[k] for k in keys}} for r in records]
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
                ax.hist([r[ki] for r in records if math.isfinite(r[ki])], bins=20, color="#888")
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

    (directory / "comparison.md").write_text(
        "Spearman ρ between methods over all inputs (in parentheses: mean within-object ρ "
        "across its input views).\n\n" + "\n".join(lines) + "\n")
    uids = sorted({(r["category"], r["uid"]) for r in records})
    if len(uids) > 20:  # The per-object bar chart is for multi-view pilots.
        return

    # Per-object mean and the view dependence each method sees.
    fig, ax = plt.subplots(figsize=(1.6 + 1.2 * len(keys), 3.2))
    width = 0.8 / len(uids)
    for k, (category, uid) in enumerate(uids):
        values = [[r[key] for r in records if r["uid"] == uid and math.isfinite(r[key])] for key in keys]
        ax.bar(np.arange(len(keys)) + k * width, [np.mean(v) for v in values], width,
               yerr=[np.std(v) for v in values], color=colors[category], label=f"{category} {uid[:6]}")
    ax.set_xticks(np.arange(len(keys)) + 0.4 - width / 2, keys, fontsize=8)
    ax.set_ylabel("mean over input views (± sd)")
    ax.legend(fontsize=6)
    ax.set_title("Per-object uncertainty; error bars = how much it varies with the input view", fontsize=9)
    fig.tight_layout()
    fig.savefig(directory / "comparison_objects.png", dpi=110)
    plt.close(fig)


def _auroc(scores, labels):
    """P(score of a positive > score of a negative), ties count half; NaN without both classes."""
    from scipy.stats import rankdata
    scores, labels = np.asarray(scores, float), np.asarray(labels, bool)
    positives, negatives = labels.sum(), (~labels).sum()
    if not positives or not negatives:
        return float("nan")
    ranks = rankdata(scores)
    return float((ranks[labels].sum() - positives * (positives + 1) / 2) / (positives * negatives))


def agreement(config):
    """SAM3 on the real render at each input's gt_vqf_max target view, the view the
    pick approximates -> <reconstruction>/agreement.json (GPU, core env)."""
    from tools.envs import use_gpu
    use_gpu(config["gpu_id"])
    from reconstruction.segmentation import SAM3Preprocessor
    from study.labels import prompt_text

    output = Path(config["output_dir"])
    db = store.connect(output)
    members = _reconstructions(db)
    targets = {input_id: store.unpack(meta)["target"] for input_id, meta in db.execute(
        "SELECT input_id, meta FROM picks WHERE selector='gt_vqf_max'")}
    db.close()
    renders = {}
    for index in sorted((Path(config["dataset"]["source_run"]) / "renders").glob("shard-*/index.jsonl")):
        for line in index.open():
            row = json.loads(line)
            renders[(row["uid"], float(row["azimuth"]), float(row["elevation"]))] = index.parent / row["files"]["rgb"]
    pending = [(image, directory) for image, directory in members if image["image_id"] in targets
               and not (output / directory / "agreement.json").is_file()]
    print(f"[agreement] {len(pending)} target views to detect", flush=True)
    if not pending:
        return
    vocabulary = config["dataset"]["categories"]
    prompts = {category: prompt_text(category) for category in vocabulary}
    model = SAM3Preprocessor(checkpoint=config["checkpoints"]["sam3"], threshold=0.0)
    for count, (image, directory) in enumerate(pending, 1):
        azimuth, elevation = targets[image["image_id"]]
        path = renders[(image["uid"], float(azimuth), float(elevation))]
        with Image.open(path) as source:
            rgb = source.convert("RGB")
        found = model.detect_many(rgb, list(prompts.values()), spatial_queries={prompts[image["category"]]})
        scores = {}
        for category in vocabulary:
            values = found[prompts[category]]["scores"]
            values = values[np.isfinite(values)]
            scores[category] = float(values.max()) if values.size else 0.0
        (output / directory / "agreement.json").write_text(json.dumps(
            {"target": [azimuth, elevation], "path": str(path), "scores": scores}) + "\n")
        if count % 100 == 0 or count == len(pending):
            print(f"[agreement] {count}/{len(pending)}", flush=True)


def _upper_bound(bad, kept, delta):
    """Clopper-Pearson (1 - delta) upper bound on P(bad | kept); 1 when nothing is kept."""
    from scipy.stats import beta
    count, errors = kept.sum(), bad[kept].sum()
    return 1.0 if not count or errors == count else float(beta.ppf(1 - delta, errors + 1, count - errors))


def _certified(scores, bad, grid, alpha, delta):
    """Learn-then-Test: the largest threshold in the fixed ``grid`` whose risk
    P(bad | score <= threshold) is <= ``alpha`` with probability >= 1 - ``delta``.

    Bonferroni over the grid (exact binomial p-values at delta / len(grid)), so no
    ordering is needed: confidently wrong reconstructions at the lowest scores
    would stop a fixed-sequence test at its first, smallest threshold. None if
    no threshold is certified.
    """
    from scipy.stats import binom
    chosen = None
    for value in grid:
        kept = scores <= value
        if kept.any() and binom.cdf(bad[kept].sum(), kept.sum(), alpha) <= delta / len(grid):
            chosen = value
    return chosen


# Not-useful labels per input, from the gt_vqf_max pick (the view ViewActive would use).
CRITERIA = {
    "error": "SAM3 margin < 0 on the pick",
    "disagree": "SAM3 verdict (margin sign) on the pick differs from the real render at the target view",
    "harm": "SAM3 right on the real render at the target view, wrong on the pick",
    "mesh": "scale-searched F-score@0.05 vs the GT mesh below --mesh-f",
}


def _abstention_records(config, method, mesh_f):
    """Per input: the uncertainty score, SAM3 confidence on the pick, and each criterion (True = not useful)."""
    from study.labels import outcomes

    output = Path(config["output_dir"])
    db = store.connect(output)
    members = _reconstructions(db)
    vocabulary = list(config["dataset"]["categories"])
    detected = outcomes(db, vocabulary, config["detection"])
    picks = dict(db.execute("SELECT input_id, pick_id FROM picks WHERE selector='gt_vqf_max'"))
    db.close()
    records = []
    for image, directory in members:
        folder = output / directory
        score = _scores(folder)[0].get(method)
        if score is None:
            continue
        with Image.open(output / image["mask_path"]) as source:
            mask = np.asarray(source.convert("L")) > 127
        with Image.open(output / image["path"]) as source:
            pixels = np.asarray(source.convert("RGB"))[mask]
        record = {"uid": image["uid"], "category": image["category"], "score": score, "confidence": np.nan,
                  "brightness": float(pixels.mean() / 255) if pixels.size else 1.0}
        pick = detected.get(picks.get(image["image_id"]))
        if pick is not None:
            record["confidence"] = max(pick["true_score"], pick["wrong_score"])
            record["error"] = pick["margin"] < 0
        if pick is not None and (folder / "agreement.json").is_file():
            real = json.loads((folder / "agreement.json").read_text())["scores"]
            category = image["category"]
            real_wrong = real[category] - max(v for k, v in real.items() if k != category) < 0
            record["disagree"] = record["error"] != real_wrong
            record["harm"] = record["error"] and not real_wrong
        if (folder / "gt_error.json").is_file():
            mesh = json.loads((folder / "gt_error.json").read_text())
            if "fscore_0.05_scaled" in mesh:
                record["mesh"] = mesh["fscore_0.05_scaled"] < mesh_f
        records.append(record)
    return records


def threshold(config, method, criterion, alphas, delta, splits, seed, mesh_f, sweep, max_brightness=None):
    """Calibrate an abstention threshold: trust a reconstruction only when ``method`` uncertainty <= lambda.

    Risk = P(not useful | kept) under ``criterion`` (see CRITERIA); FPR = share
    of not-useful reconstructions kept. lambda is chosen by Learn-then-Test on
    calibration categories and evaluated on held-out categories: the ViewActive
    unseen categories, and ``splits`` random half/half category splits. The
    baseline keeps the same number of held-out inputs by SAM3 confidence on the
    pick (inputs whose pick is not detected yet rank last). ``max_brightness``
    drops inputs whose object pixels average brighter than it: untextured
    models rendered white on white, where SAM3D sees little but a silhouette.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    records = [r for r in _abstention_records(config, method, mesh_f) if criterion in r]
    dropped = 0
    if max_brightness is not None:
        dropped = sum(r["brightness"] > max_brightness for r in records)
        records = [r for r in records if r["brightness"] <= max_brightness]
    if not records:
        raise SystemExit(f"no inputs with a {method} score and the {criterion!r} label")
    manifest = json.loads(Path(config["dataset"]["manifest"]).read_text())
    unseen = {r["category"] for r in records if manifest[r["uid"]]["split"] == "test_unseen"}
    scores = np.array([r["score"] for r in records])
    categories = np.array([r["category"] for r in records])
    bad = np.array([r[criterion] for r in records])
    confidence = np.array([r["confidence"] for r in records])
    grid = np.round(np.arange(0.01, 0.501, 0.01), 2)

    def rates(kept, test):
        return {"coverage": kept.sum() / test.sum(),
                "risk": bad[kept].mean() if kept.any() else float("nan"),
                "fpr": kept[test & bad].mean() if (test & bad).any() else float("nan"),
                "tpr": kept[test & ~bad].mean() if (test & ~bad).any() else float("nan")}

    def evaluate(calibration, test, alpha):
        value = _certified(scores[calibration], bad[calibration], grid, alpha, delta)
        if value is None:
            return None
        kept = test & (scores <= value)
        # Baseline: keep as many held-out inputs, most SAM3-confident pick first.
        order = np.flatnonzero(test)[np.argsort(-confidence[test], kind="stable")][:kept.sum()]
        by_confidence = np.zeros_like(kept)
        by_confidence[order] = True
        return {"lambda": float(value), **rates(kept, test),
                "confidence_risk": bad[by_confidence].mean() if kept.any() else float("nan"),
                "base_risk": bad[test].mean()}

    everything = np.ones(len(records), bool)
    lines = [f"Trust the reconstruction when {method} uncertainty <= lambda. Not useful = {CRITERIA[criterion]}"
             + (f" ({mesh_f:g})" if criterion == "mesh" else "") + f". {len(records)} inputs (one per object), "
             f"{len(set(categories))} categories, {bad.mean():.1%} not useful"
             + (f"; {dropped} near-white inputs (brightness > {max_brightness:g}) dropped" if dropped else "")
             + f". Risk = not useful among kept; "
             f"FPR = share of not-useful reconstructions kept; TPR = share of useful ones kept.", "",
             "## Sweep, all inputs", "", f"| lambda | coverage | risk | risk {1 - delta:.0%} upper bound | FPR | TPR |",
             "|---|---|---|---|---|---|"]
    for value in sweep:
        row = rates(scores <= value, everything)
        lines.append(f"| {value:.2f} | {row['coverage']:.0%} | {row['risk']:.1%} | "
                     f"{_upper_bound(bad, scores <= value, delta):.1%} | {row['fpr']:.1%} | {row['tpr']:.1%} |")
    held_out = np.isin(categories, sorted(unseen))
    lines += ["", f"## Learn-then-Test (delta = {delta:g}, Bonferroni over lambda = 0.01..0.50), held-out ViewActive unseen categories "
              f"({held_out.sum()} inputs: {', '.join(sorted(unseen))})", "",
              "| alpha | lambda | coverage | risk | FPR | TPR | SAM3-conf. risk | no-filter risk |",
              "|---|---|---|---|---|---|---|---|"]
    for alpha in alphas:
        row = evaluate(~held_out, held_out, alpha)
        lines.append(f"| {alpha:g} | none certified | | | | | | {bad[held_out].mean():.1%} |" if row is None else
                     f"| {alpha:g} | {row['lambda']:.3f} | {row['coverage']:.0%} | {row['risk']:.1%} | "
                     f"{row['fpr']:.1%} | {row['tpr']:.1%} | {row['confidence_risk']:.1%} | {row['base_risk']:.1%} |")
    rng = np.random.default_rng(seed)
    names = np.unique(categories)
    tests = [np.isin(categories, rng.choice(names, len(names) // 2, replace=False)) for _ in range(splits)]
    lines += ["", f"## {splits} random half/half category splits", "",
              "| alpha | certified | median lambda | mean coverage | mean risk | P(risk > alpha) | mean FPR | "
              "mean SAM3-conf. risk |", "|---|---|---|---|---|---|---|---|"]
    for alpha in alphas:
        valid = [r for r in (evaluate(~test, test, alpha) for test in tests) if r is not None]
        if not valid:
            lines.append(f"| {alpha:g} | 0% | | | | | | |")
            continue
        mean = lambda key: np.nanmean([r[key] for r in valid])
        lines.append(f"| {alpha:g} | {len(valid) / splits:.0%} | {np.median([r['lambda'] for r in valid]):.3f} | "
                     f"{mean('coverage'):.0%} | {mean('risk'):.1%} | {np.mean([r['risk'] > alpha for r in valid]):.0%} | "
                     f"{mean('fpr'):.1%} | {mean('confidence_risk'):.1%} |")
    within = [_auroc(scores[categories == c], bad[categories == c]) for c in names]
    lines += ["", f"AUROC for not useful: {method} {_auroc(scores, bad):.2f} pooled, "
              f"{np.nanmedian(within):.2f} median within category; SAM3 confidence "
              f"{_auroc(-confidence[np.isfinite(confidence)], bad[np.isfinite(confidence)]):.2f} (picks detected so far)."]

    directory_out = Path(config["output_dir"]) / FIGURES
    directory_out.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 4))
    top = 0.0
    for name, ranking in ((method, scores), ("SAM3 confidence on the pick", -confidence)):
        order = np.argsort(ranking, kind="stable")
        kept = np.arange(1, len(order) + 1)
        risk = np.cumsum(bad[order]) / kept
        ax.plot(kept / len(order), risk, label=name)
        top = max(top, risk[kept >= 0.05 * len(order)].max())
    for alpha in alphas:
        ax.axhline(alpha, color="k", lw=0.6, ls=":")
        ax.text(1, alpha, f" alpha {alpha:g}", fontsize=6, va="center", ha="left")
    for index, value in enumerate(sweep[::max(1, len(sweep) // 5)]):
        kept = scores <= value
        ax.plot(kept.mean(), bad[kept].mean(), "o", color="C0", ms=4)
        ax.annotate(f"lambda {value:.2f}\n{kept.mean():.0%} kept, {bad[kept].mean():.1%} not useful",
                    (kept.mean(), bad[kept].mean()), textcoords="offset points", xytext=(0, 10 if index % 2 else -24),
                    fontsize=6, ha="center")
    ax.set(xlabel="coverage (share of inputs kept)", ylabel="not useful among kept",
           title=f"{criterion}: {CRITERIA[criterion]}" + (f" {mesh_f:g}" if criterion == "mesh" else "")
           + f" ({len(records)} inputs)", xlim=(0.05, 1), ylim=(0, 1.3 * max(top, max(alphas))))
    ax.title.set_fontsize(8)
    ax.legend(fontsize=7)
    fig.tight_layout()
    suffix = f"{method}_{criterion}" + ("_textured" if max_brightness is not None else "")
    fig.savefig(directory_out / f"risk_coverage_{suffix}.png", dpi=110)
    plt.close(fig)
    (directory_out / f"threshold_{suffix}.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines), flush=True)


def information(config, samples, shard, limit=None, per_category=None, uids=None, elevations=None):
    """Pointwise MI between each input and its stage-1 shape -> <reconstruction>/information.npz."""
    from tools.envs import use_gpu
    use_gpu(config["gpu_id"])
    from reconstruction.sam3d import SAM3D

    output = Path(config["output_dir"])
    pending = _pending(config, "information.npz", shard, per_category, uids, elevations)[:limit]
    print(f"[information] {len(pending)} reconstructions to score", flush=True)
    if not pending:
        return
    model = SAM3D(checkpoint_dir=config["checkpoints"]["sam3d"])
    for count, (image, directory) in enumerate(pending, 1):
        started = time.time()
        result = model.stage1_information(output / image["path"], output / image["mask_path"],
                                          seed=config["reconstruction"]["seed"], samples=samples)
        scalars = {"information": {key: result[key] for key in ("information", "stderr")}, "samples": samples}
        np.savez_compressed(output / directory / "information.npz", scalars=json.dumps(scalars),
                            per_token=result["per_token"])
        print(f"[information] {count}/{len(pending)} {image['key']} i={result['information']:.1f} "
              f"± {result['stderr']:.1f} nats {time.time() - started:.1f}s", flush=True)


def meangap(config, shard, limit=None, uids=None, elevations=None, per_category=None):
    """The reconstruction's stage-1 shape vs SAM3D's one-step mean shape -> <reconstruction>/meangap.npz."""
    from tools.envs import use_gpu
    use_gpu(config["gpu_id"])
    from reconstruction.sam3d import SAM3D
    from reconstruction.stage1_uncertainty import disagreement

    output = Path(config["output_dir"])
    pending = _pending(config, "meangap.npz", shard, per_category, uids, elevations)[:limit]
    print(f"[meangap] {len(pending)} reconstructions to score", flush=True)
    if not pending:
        return
    model = SAM3D(checkpoint_dir=config["checkpoints"]["sam3d"])
    for count, (image, directory) in enumerate(pending, 1):
        result = model.stage1_mean_gap(output / image["path"], output / image["mask_path"],
                                       seed=config["reconstruction"]["seed"])
        scalars = {key: value for key, value in result.items() if not key.startswith(("sample", "mean_"))}
        for name in ("guided", "unguided", "prior"):
            scalars[f"occupancy_gap_{name}"] = disagreement([result["sample"], result[f"mean_{name}"]])
        for name in ("guided", "unguided"):  # Share of the shape's spread the image left unexplained.
            scalars[f"ratio_latent_{name}"] = scalars[f"latent_gap_{name}"] / scalars["latent_gap_prior"]
            scalars[f"ratio_occupancy_{name}"] = ((scalars[f"occupancy_gap_{name}"] or 0)
                                                 / max(scalars["occupancy_gap_prior"] or 0, 1e-6))
        np.savez_compressed(output / directory / "meangap.npz", scalars=json.dumps(scalars),
                            **{key: np.packbits(result[key], axis=-1)
                               for key in ("sample", "mean_guided", "mean_unguided", "mean_prior")})
        print(f"[meangap] {count}/{len(pending)} {image['key']} ratio {scalars['ratio_latent_unguided']:.3f} "
              f"+{scalars['seconds_unguided'] + scalars['seconds_prior']:.2f}s", flush=True)


PRIMARY, PRIMARY_KEY = "ratio, latent, unguided", "ratio_latent_unguided"  # fixed before testing


def meangap_test(config, uids=None, reference_seeds=8):
    """How well cheap scores track an independent ensemble reference: seeds 4-7 of 8-seed runs
    (``reference_seeds`` 8), or seeds 1-3 of 4-seed runs (4; seed 0 is the reconstruction the
    mean-gap uses). The criterion for one abstention threshold is the pooled ρ."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from reconstruction.stage1_uncertainty import disagreement

    output = Path(config["output_dir"])
    db = store.connect(output)
    rows = [(image, directory) for image, directory in _reconstructions(db) if not uids or image["uid"] in uids]
    db.close()
    names = {"ratio, latent, unguided": "ratio_latent_unguided", "ratio, latent, guided": "ratio_latent_guided",
             "ratio, occupancy, unguided": "ratio_occupancy_unguided",
             "ratio, occupancy, guided": "ratio_occupancy_guided",
             "mean-gap, latent, unguided": "latent_gap_unguided", "mean-gap, occupancy, unguided":
             "occupancy_gap_unguided", "mean-gap, latent, guided": "latent_gap_guided"}
    records = []
    for image, directory in rows:
        folder = output / directory
        if not ((folder / "stage1.npz").is_file() and (folder / "meangap.npz").is_file()):
            continue
        seeds = samples_of(load_stage1(folder / "stage1.npz")[1], "ensemble")
        if len(seeds) < reference_seeds:
            continue
        gap = json.loads(str(np.load(folder / "meangap.npz")["scalars"]))
        if "ratio_latent_unguided" not in gap:
            continue
        record = {"category": image["category"], "uid": image["uid"], "seconds": gap["seconds_unguided"]
                  + gap["seconds_prior"]}
        if reference_seeds >= 8:
            record["reference"] = disagreement(seeds[4:8])
            for k in (2, 3, 4):
                record[f"ensemble, {k} seeds" + (" (split half)" if k == 4 else "")] = disagreement(seeds[:k])
        else:
            record["reference"] = disagreement(seeds[1:4])
        record.update({label: gap[key] for label, key in names.items()})
        records.append(record)
    if not records:
        raise SystemExit("no inputs with stage1.npz and a meangap.npz that has the prior mean")
    methods = [key for key in records[0] if key.startswith(("ensemble", "ratio", "mean-gap"))]
    value = lambda r, key: np.nan if r[key] is None else r[key]
    objects = sorted({(r["category"], r["uid"]) for r in records})
    multi = len(records) > len(objects)
    table = {}
    for key in methods:
        pooled = _spearman([value(r, key) for r in records], [value(r, "reference") for r in records])
        within = [_spearman([value(r, key) for r in records if r["uid"] == uid],
                            [value(r, "reference") for r in records if r["uid"] == uid])
                  for _, uid in objects] if multi else []
        table[key] = (pooled, within)
    directory_out = output / "figures/meangap"
    directory_out.mkdir(parents=True, exist_ok=True)
    reference = "seeds 4-7 of 8-seed runs" if reference_seeds >= 8 else "seeds 1-3 of 4-seed runs"
    lines = [f"# Normalized mean-gap vs seed ensemble: {len(records)} inputs, {len(objects)} objects", "",
             f"Reference: ensemble disagreement on {reference}, independent of every score below. "
             "Pooled Spearman ρ is the criterion for one abstention threshold across objects. "
             f"Primary score (fixed before testing): {PRIMARY}; the rest are exploratory.", "",
             "| score | extra network work | ρ pooled | ρ within object (mean) |", "|---|---|---|---|"]
    work = {"ensemble, 2 seeds": "1 stage-1 run", "ensemble, 3 seeds": "2 stage-1 runs",
            "ensemble, 4 seeds (split half)": "3 stage-1 runs"}
    for key in methods:
        pooled, within = table[key]
        cost = work.get(key, "2 passes + decode" if key.startswith("ratio") else "1 pass + decode")
        lines.append(f"| {'**' + key + '**' if key == PRIMARY else key} | {cost} | {pooled:+.2f} | "
                     f"{np.nanmean(within) if within else float('nan'):+.2f} |")
    lines += ["", f"Ratio extra time: median {np.median([r['seconds'] for r in records]):.2f} s per input."]
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6), gridspec_kw={"width_ratios": [1.2, 1]})
    ax = axes[0]
    for row, key in enumerate(methods):
        color = ENSEMBLE_COLOR if key.startswith("ensemble") else LAPLACE_COLOR
        ax.scatter([table[key][0]], [row], s=60, color=color, edgecolor="white", linewidth=1.5, zorder=3)
        ax.hlines(row, 0, table[key][0], color=GRID, lw=2)
    ax.axvline(0, color=MUTED, lw=0.8)
    ax.set_yticks(range(len(methods)), methods, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("pooled Spearman ρ with the reference ensemble", fontsize=8, color=INK)
    ax.set_title(f"{len(records)} inputs, {len(objects)} objects", fontsize=9, color=INK, loc="left")
    _style(ax)
    ax = axes[1]
    ax.scatter([value(r, PRIMARY) for r in records], [value(r, "reference") for r in records], s=14,
               color=LAPLACE_COLOR, edgecolor="white", linewidth=1)
    ax.set_xlabel(PRIMARY, fontsize=8, color=INK)
    ax.set_ylabel(f"reference ensemble ({reference})", fontsize=8, color=INK)
    ax.set_title(f"one dot per input; ρ {table[PRIMARY][0]:+.2f}", fontsize=9, color=INK, loc="left")
    _style(ax)
    fig.tight_layout()
    name = f"ratio_{Path(config['output_dir']).name}"
    fig.savefig(directory_out / f"{name}.png", dpi=120)
    plt.close(fig)
    lines += ["", f"![{name}]({name}.png)"]
    (directory_out / f"{name}.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines[:6 + len(methods)]), flush=True)
    print(f"[meangap_test] -> {directory_out}", flush=True)


# ---------------------------------------------------------------- ground truth

def gt_error(config, requires, shard, limit=None):
    """Shape error against the ground-truth mesh -> <reconstruction>/gt_error.json (CPU, sam3d env).

    Only reconstructions that already have ``requires`` (e.g. laplace.npz), if given.
    """
    from harness.meshio import ObjaverseStore
    from reconstruction.gt_error import shape_error

    output = Path(config["output_dir"])
    pending = [(image, directory) for image, directory in _pending(config, "gt_error.json", shard)
               if requires is None or (output / directory / requires).is_file()][:limit]
    print(f"[gt_error] {len(pending)} reconstructions", flush=True)
    meshes = ObjaverseStore(config["dataset"]["root"], config["dataset"]["canonical"])
    for count, (image, directory) in enumerate(pending, 1):
        truth = meshes.load(image["uid"], device="cpu")
        if truth is None:
            print(f"[gt_error] {image['key']}: no ground-truth mesh", flush=True)
            continue
        predicted = np.load(output / directory / "mesh.npz")
        error = shape_error((predicted["vertices"], predicted["faces"]),
                            (truth.verts.numpy(), truth.faces.numpy()), image["azimuth"], image["elevation"])
        (output / directory / "gt_error.json").write_text(json.dumps(error) + "\n")
        print(f"[gt_error] {count}/{len(pending)} {image['key']} F@0.05={error['fscore_0.05']:.3f}", flush=True)


# ---------------------------------------------------------------- proof of concept

POC = "figures/laplace_poc"


def poc(config, names, top, axis):
    """Last-layer Laplace proof of concept on the inputs that have every ``names`` file.

    Rank stability of ``laplace_entropy`` across the files (e.g. data sizes),
    agreement with the seed ensemble where ``stage1.npz`` exists, and a grid of
    the ``top`` most and least uncertain inputs (by the first file).
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output = Path(config["output_dir"])
    db = store.connect(output)
    rows = _reconstructions(db)
    db.close()
    records = []
    for image, directory in rows:
        folder = output / directory
        if not all((folder / name).is_file() for name in names):
            continue
        record = {"image": image, "folder": folder}
        for name in names:
            scalars, data = load_stage1(folder / name)
            record[name] = scalars["laplace"]
            record[f"{name}:data_size"] = scalars["data_size"]
            if name == names[0]:
                record["samples"] = samples_of(data, "laplace")
        if (folder / "stage1.npz").is_file():
            scalars, data = load_stage1(folder / "stage1.npz")
            record["ensemble"] = scalars["ensemble"]["disagreement"]
            record["ensemble_samples"] = samples_of(data, "ensemble")
        if (folder / "gt_error.json").is_file():
            record["gt"] = json.loads((folder / "gt_error.json").read_text())
        records.append(record)
    if len(records) < 3:
        raise SystemExit(f"only {len(records)} inputs have all of {names}")
    directory_out = output / POC
    directory_out.mkdir(parents=True, exist_ok=True)
    label = {name: f"N={records[0][f'{name}:data_size']:g}" for name in names}
    entropy = {name: [r[name]["entropy"] for r in records] for name in names}
    lines = [f"# Last-layer Laplace proof of concept: {len(records)} inputs, "
             f"{len({r['image']['category'] for r in records})} categories", "",
             "## Rank stability of laplace_entropy across data sizes (Spearman)", "",
             "| | " + " | ".join(label[n] for n in names) + " |", "|---" * (len(names) + 1) + "|"]
    for a in names:
        lines.append(f"| {label[a]} | " + " | ".join(f"{_spearman(entropy[a], entropy[b]):+.2f}"
                                                    for b in names) + " |")
    lines += ["", "## Score summary", "", "| data size | entropy median [min, max] | "
              "occupancy disagreement median | share with disagreement > 0 |", "|---|---|---|---|"]
    for name in names:
        dis = np.array([np.nan if r[name]["disagreement"] is None else r[name]["disagreement"]
                        for r in records])
        lines.append(f"| {label[name]} | {np.median(entropy[name]):.3f} [{min(entropy[name]):.3f}, "
                     f"{max(entropy[name]):.3f}] | {np.nanmedian(dis):.3f} | {np.mean(dis > 0):.0%} |")
    lines += ["", "## Does uncertainty track reconstruction error? (ground-truth mesh)", ""]
    known = [r for r in records if "gt" in r]
    if len(known) > 2:
        error = {"1 - F@0.05": [1 - r["gt"]["fscore_0.05"] for r in known],
                 "1 - F@0.02": [1 - r["gt"]["fscore_0.02"] for r in known],
                 "Chamfer": [r["gt"]["chamfer"] for r in known]}
        candidates = {f"laplace_entropy {label[n]}": [r[n]["entropy"] for r in known] for n in names}
        candidates[f"laplace disagreement {label[names[0]]}"] = [
            np.nan if r[names[0]]["disagreement"] is None else r[names[0]]["disagreement"] for r in known]
        if sum(r.get("ensemble") is not None for r in known) > 2:
            candidates["ensemble disagreement"] = [np.nan if r.get("ensemble") is None else r["ensemble"]
                                                   for r in known]
        lines += [f"Spearman ρ between each score and the shape error over {len(known)} inputs "
                  "(positive = higher uncertainty, worse shape). Last column: mean 1 - F@0.05 of the "
                  "most vs least uncertain quarter.", "",
                  "| score | " + " | ".join(error) + " | top / bottom quarter |",
                  "|---" * (len(error) + 2) + "|"]
        target = np.array(error["1 - F@0.05"])
        for key, values in candidates.items():
            values = np.array(values, float)
            finite = np.flatnonzero(np.isfinite(values))
            order = finite[np.argsort(values[finite])]
            quarter = max(1, len(order) // 4)
            lines.append(f"| {key} (n={len(finite)}) | "
                         + " | ".join(f"{_spearman(list(values), e):+.2f}" for e in error.values())
                         + f" | {target[order[-quarter:]].mean():.2f} / {target[order[:quarter]].mean():.2f} |")
        lines += ["", f"Median shape error 1 - F@0.05 = {np.median(target):.2f}; pose from the camera "
                  f"alone: {np.median([1 - r['gt']['fscore_0.05_camera'] for r in known]):.2f}."]
    else:
        lines.append("Not yet: run the `gt_error` stage.")
    paired = [r for r in records if r.get("ensemble") is not None]
    lines += ["", "## Agreement with the seed ensemble", ""]
    if len(paired) > 2:
        main = names[0]
        lines.append(f"{len(paired)} inputs with stage1.npz: Spearman(ensemble, laplace_entropy "
                     f"{label[main]}) = {_spearman([r['ensemble'] for r in paired], [r[main]['entropy'] for r in paired]):+.2f}; "
                     f"Spearman(ensemble, laplace disagreement) = "
                     f"{_spearman([r['ensemble'] for r in paired], [r[main]['disagreement'] for r in paired]):+.2f}.")
    else:
        lines.append(f"Not yet: {len(paired)} of these inputs have stage1.npz (rerun `poc` once they do).")

    # Most and least uncertain inputs: input, re-render from the input view, side view, Laplace draws.
    ranked = sorted(records, key=lambda r: r[names[0]]["entropy"])
    shown = ranked[::-1][:top] + ranked[:top]
    draws = len(shown[0]["samples"])
    columns = 3 + draws
    fig, axes = plt.subplots(len(shown), columns, figsize=(columns * 1.4, len(shown) * 1.5))
    for row, record in enumerate(shown):
        image, folder = record["image"], record["folder"]
        panels = [(Image.open(output / image["path"]), "input"),
                  (Image.open(folder / "views" / "azimuth_180_elevation_0.png"), "SAM3D input view"),
                  (Image.open(folder / "views" / "azimuth_90_elevation_0.png"), "SAM3D side view")]
        panels += [(_silhouette(sample, axis), f"draw {k}") for k, sample in enumerate(record["samples"])]
        for col, (picture, title) in enumerate(panels):
            ax = axes[row, col]
            ax.imshow(picture, cmap="gray" if col >= 3 else None, vmin=0, vmax=255)
            ax.set_xticks([]), ax.set_yticks([])
            if row == 0 or row == top:
                ax.set_title(title, fontsize=7)
        group = "most" if row < top else "least"
        gt = f"\nF@.05={record['gt']['fscore_0.05']:.2f}" if "gt" in record else ""
        axes[row, 0].set_ylabel(f"{group}\n{image['category'][:14]}\nH={record[names[0]]['entropy']:.2f}{gt}",
                                fontsize=7)
    fig.suptitle(f"Last-layer Laplace ({label[names[0]]}): {top} most (top) and {top} least uncertain "
                 f"inputs; draws = occupancy along grid axis {axis}", fontsize=9)
    fig.tight_layout()
    fig.savefig(directory_out / "extremes.png", dpi=110)
    plt.close(fig)
    lines += ["", "## Most and least uncertain inputs", "", "![extremes](extremes.png)", "",
              "| rank | category | uid | input (az, el) | entropy | occupancy disagreement | GT F@0.05 |",
              "|---|---|---|---|---|---|---|"]
    for rank, record in enumerate(ranked[::-1], 1):
        image = record["image"]
        dis = record[names[0]]["disagreement"]
        lines.append(f"| {rank} | {image['category']} | {image['uid'][:8]} | ({image['azimuth']:g}, "
                     f"{image['elevation']:g}) | {record[names[0]]['entropy']:.3f} | "
                     f"{'n/a' if dis is None else f'{dis:.3f}'} | "
                     + (f"{record['gt']['fscore_0.05']:.3f} |" if "gt" in record else " |"))
    if len(paired) > 2:
        comparison(paired, output, names[0], label[names[0]], axis, directory_out)
        lines += ["", "## Laplace vs seed ensemble", "",
                  "![scores](compare_scores.png)", "", "![cases](compare_cases.png)"]
    (directory_out / "poc.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines[:4 + len(names) + 2 + len(names) + 4]), flush=True)
    print(f"[poc] -> {directory_out}", flush=True)


ENSEMBLE_COLOR, LAPLACE_COLOR = "#2a78d6", "#eb6834"  # categorical slots 1 and 2
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"


def _agreement_map(samples, axis):
    """Occupancy seen along ``axis``: light gray where all draws agree within one voxel,
    a light-to-dark violet ramp for the share of draws missing it, white where empty."""
    from matplotlib.colors import LinearSegmentedColormap
    from reconstruction.stage1_uncertainty import voxel_disagreement

    disputed = voxel_disagreement(samples)
    occupied = np.isfinite(disputed)
    present = occupied.any(axis=axis)
    mean = np.where(present, np.where(occupied, disputed, 0).sum(axis=axis)
                    / np.maximum(occupied.sum(axis=axis), 1), 0)
    ramp = LinearSegmentedColormap.from_list("disagree", ["#c9c3f0", "#6250d6", "#241a6b"])
    image = ramp(np.clip(mean / 0.75, 0, 1))[..., :3]
    image[mean < 0.05] = 0.85
    image[~present] = 1.0
    return np.flipud(np.moveaxis(image, 0, 1))


def _style(ax):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.grid(color=GRID, lw=0.6)
    ax.set_axisbelow(True)


def comparison(records, output, name, label, axis, directory_out):
    """Laplace vs ensemble on the same inputs: score scatter, paired ranking, and example draws."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter

    ens = np.array([r["ensemble"] for r in records], float)
    lap = np.array([np.nan if r[name]["disagreement"] is None else r[name]["disagreement"]
                    for r in records], float)
    rho = _spearman(list(ens), list(lap))
    ens_cut, lap_cut = np.nanmedian(ens), np.nanmedian(lap)

    # Scores: scatter (left) and each input's pair of scores, sorted by the ensemble (right).
    fig, (left, right) = plt.subplots(1, 2, figsize=(13, 0.16 * len(records) + 1.6),
                                      gridspec_kw={"width_ratios": [1, 1.1]})
    root = FuncFormatter(lambda value, _: f"{value ** 2:.2g}")
    x, y = np.sqrt(ens), np.sqrt(np.nan_to_num(lap))
    left.scatter(x, y, s=40, color=MUTED, edgecolor="white", linewidth=1.5, zorder=3)
    left.axvline(np.sqrt(ens_cut), color=MUTED, lw=0.8, ls="--")
    left.axhline(np.sqrt(lap_cut), color=MUTED, lw=0.8, ls="--")
    for text, (hx, hy), align in [("Laplace only", (0.02, 0.98), "left"), ("both high", (0.98, 0.98), "right"),
                                  ("both low", (0.02, 0.02), "left"), ("ensemble only", (0.98, 0.02), "right")]:
        left.text(hx, hy, text, transform=left.transAxes, ha=align, va="top" if hy > 0.5 else "bottom",
                  color=MUTED, fontsize=9, style="italic")
    # Label the inputs at the extremes of each score.
    for i in sorted(set(np.argsort(-y)[:5]) | set(np.argsort(-x)[:3])):
        left.annotate(records[i]["image"]["category"].split("_(")[0], (x[i], y[i]), xytext=(5, 4),
                      textcoords="offset points", fontsize=8, color=INK)
    left.set(xlim=(0, x.max() * 1.12), ylim=(0, y.max() * 1.08))
    left.xaxis.set_major_formatter(root)
    left.yaxis.set_major_formatter(root)
    left.set_xlabel("seed ensemble: 1 - mean pairwise F-score  (√ scale)", color=INK, fontsize=9)
    left.set_ylabel(f"last-layer Laplace ({label}): same measure  (√ scale)", color=INK, fontsize=9)
    left.set_title(f"One dot per input (n={len(records)}); Spearman ρ = {rho:+.2f}; dashed = medians;\n"
                   f"Laplace is below 0.01 for {np.mean(np.nan_to_num(lap) < 0.01):.0%} of inputs", fontsize=10, color=INK,
                   loc="left")
    _style(left)

    order = np.argsort(ens)
    rows = np.arange(len(order))
    right.hlines(rows, np.sqrt(np.nan_to_num(lap[order])), np.sqrt(ens[order]), color=GRID, lw=2, zorder=1)
    right.scatter(np.sqrt(ens[order]), rows, s=36, color=ENSEMBLE_COLOR, edgecolor="white", linewidth=1.5,
                  zorder=3, label="seed ensemble")
    right.scatter(np.sqrt(np.nan_to_num(lap[order])), rows, s=36, color=LAPLACE_COLOR, edgecolor="white",
                  linewidth=1.5, zorder=3, label=f"Laplace ({label})")
    right.set_yticks(rows, [records[i]["image"]["category"].split("_(")[0][:16] for i in order], fontsize=7)
    right.xaxis.set_major_formatter(root)
    right.set_xlabel("1 - mean pairwise F-score of the 4 draws  (√ scale)", color=INK, fontsize=9)
    right.set_ylim(-1, len(order))
    right.set_title("Each input's two scores, sorted by the ensemble", fontsize=10, color=INK, loc="left")
    right.legend(loc="lower right", fontsize=8, frameon=False)
    _style(right)
    right.grid(axis="y", visible=False)
    fig.tight_layout()
    fig.savefig(directory_out / "compare_scores.png", dpi=130)
    plt.close(fig)

    # Cases: two inputs per quadrant, with each method's draws and where they disagree.
    quadrants = [("both high", (ens > ens_cut) & (lap > lap_cut), ens + np.nan_to_num(lap)),
                 ("ensemble only", (ens > ens_cut) & ~(lap > lap_cut), ens - np.nan_to_num(lap)),
                 ("Laplace only", ~(ens > ens_cut) & (lap > lap_cut), np.nan_to_num(lap) - ens),
                 ("both low", ~(ens > ens_cut) & ~(lap > lap_cut), -(ens + np.nan_to_num(lap)))]
    cases = []
    for title, mask, strength in quadrants:
        chosen = [i for i in np.argsort(-strength) if mask[i]][:2]
        cases += [(title, i) for i in chosen]
    columns = 11
    fig, axes = plt.subplots(len(cases), columns, figsize=(columns * 1.15, len(cases) * 1.3 + 0.6))
    axes = np.atleast_2d(axes)
    headers = (["input"] + ["ensemble: where draws disagree"] + [f"seed {k}" for k in range(4)]
               + ["Laplace: where draws disagree"] + [f"weights {k}" for k in range(4)])
    for row, (title, i) in enumerate(cases):
        record = records[i]
        panels = [(Image.open(output / record["image"]["path"]), None),
                  (_agreement_map(record["ensemble_samples"], axis), ENSEMBLE_COLOR)]
        panels += [(_silhouette(sample, axis), None) for sample in record["ensemble_samples"][:4]]
        panels += [(_agreement_map(record["samples"], axis), LAPLACE_COLOR)]
        panels += [(_silhouette(sample, axis), None) for sample in record["samples"][:4]]
        for col, (picture, frame) in enumerate(panels):
            ax = axes[row, col]
            ax.imshow(picture, cmap="gray", vmin=0, vmax=255)
            ax.set_xticks([]), ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_color(frame or GRID)
                spine.set_linewidth(2 if frame else 0.6)
            if row == 0:
                ax.set_title(headers[col], fontsize=7, color=INK)
        axes[row, 1].set_xlabel(f"{ens[i]:.3f}", fontsize=8, color=ENSEMBLE_COLOR, labelpad=1)
        axes[row, 6].set_xlabel(f"{0 if np.isnan(lap[i]) else lap[i]:.3f}", fontsize=8, color=LAPLACE_COLOR,
                                labelpad=1)
        axes[row, 0].set_ylabel(f"{title}\n{record['image']['category'].split('_(')[0][:14]}", fontsize=8,
                                color=INK)
    fig.suptitle("Ensemble (blue) varies the noise seed with the weights fixed; Laplace (orange) varies the "
                 "shape-head weights with the seed fixed.\nDraws: stage-1 occupancy seen along one grid axis, "
                 "darker = nearer. Maps: gray = all 4 draws agree within 1 voxel, violet = share of draws "
                 "missing it. Score under each map.", fontsize=8, color=MUTED)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(directory_out / "compare_cases.png", dpi=120)
    plt.close(fig)


VIEWS = "figures/views_uncertainty"


INFORMATION_COLOR = "#1baf7a"  # categorical slot 3


def _information_map(per_token, occupancy, axis):
    """Per-token information (16^3) summed along ``axis``, upsampled to the 64^3 projection and shown
    where ``occupancy`` is present; light = much information, dark green = little, white = empty."""
    from matplotlib.colors import LinearSegmentedColormap

    grid = per_token.reshape(16, 16, 16).sum(axis=axis)
    grid = np.kron(grid, np.ones((4, 4)))
    present = occupancy.any(axis=axis)
    ramp = LinearSegmentedColormap.from_list("information", ["#0b4d34", "#1baf7a", "#d4f2e6"])
    scaled = np.log1p(grid) / max(np.log1p(grid[present]).max(), 1e-9) if present.any() else grid
    image = ramp(np.clip(scaled, 0, 1))[..., :3]
    image[~present] = 1.0
    return np.flipud(np.moveaxis(image, 0, 1))


def _view_records(output, uids, seeds=4, requires="information.npz"):
    """{(category, uid): [per input view: image, folder, 4-seed ensemble samples and score, and the
    information and normalized mean-gap where present]} for the inputs with stage1.npz and ``requires``."""
    from reconstruction.stage1_uncertainty import disagreement

    db = store.connect(output)
    rows = [(image, directory) for image, directory in _reconstructions(db) if not uids or image["uid"] in uids]
    db.close()
    objects = {}
    for image, directory in rows:
        folder = output / directory
        if not ((folder / "stage1.npz").is_file() and (folder / requires).is_file()):
            continue
        samples = samples_of(load_stage1(folder / "stage1.npz")[1], "ensemble")[:seeds]
        record = {"image": image, "folder": folder, "angle": (float(image["azimuth"]), float(image["elevation"])),
                  "ensemble": disagreement(samples), "samples": samples}
        if (folder / "information.npz").is_file():
            info = np.load(folder / "information.npz")
            record["information"] = json.loads(str(info["scalars"]))["information"]["information"]
            record["information_map"] = info["per_token"]
        if (folder / "meangap.npz").is_file():
            gap = np.load(folder / "meangap.npz")
            record["ratio"] = json.loads(str(gap["scalars"])).get(PRIMARY_KEY)
            if "mean_unguided" in gap:
                record["mean"] = np.unpackbits(gap["mean_unguided"], axis=-1).astype(bool)
        objects.setdefault((image["category"], image["uid"]), []).append(record)
    return objects


AFTER = "figures/after_ensemble"
AFTER_CONTROLS = ("identity", "random", "gt_random", "gt_oracle", "oracle", "detector_gap")
# Summary bars: before, label-free margin selection, and the two label-using upper bounds.
AFTER_COMPARE = {"input": "input", "detector_gap": "SAM3D max top-2 gap",
                 "gt_oracle": "ground-truth oracle", "oracle": "SAM3D oracle"}


def _detection_metrics(picks, scores, vocabulary_size):
    """Pooled detection quality over ``picks`` [(category, selection row)] for one selector.

    Precision, recall and F1 at the detection threshold: one possible true positive per input
    (``tp``, mask IoU checked), false positives = wrong categories scored above threshold.
    AP ranks every (input, category) score of the chosen views, with the true category as the
    positive; an abstention (no ``pick_id``) scores nothing, so its positive is never recalled.
    """
    tp = sum(row["tp"] for _, row in picks)
    fp = sum(row["fp_rate"] * (vocabulary_size - 1) for _, row in picks)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / len(picks)
    ranked = sorted(((score, prompt == category) for category, row in picks if row["pick_id"] is not None
                     for prompt, score in scores[row["pick_id"]].items()), key=lambda pair: -pair[0])
    hits = np.cumsum([positive for _, positive in ranked])
    ap = sum(hits[k] / (k + 1) for k, (_, positive) in enumerate(ranked) if positive) / len(picks)
    return {"precision": precision, "recall": recall,
            "F1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0, "AP": ap}


def _ensemble4(folder):
    """4-seed ensemble disagreement (the calibrated setting) from stage1.npz; None if absent."""
    from reconstruction.stage1_uncertainty import disagreement

    if not (folder / "stage1.npz").is_file():
        return None
    scalars, data = load_stage1(folder / "stage1.npz")
    samples = samples_of(data, "ensemble")
    return scalars["ensemble"]["disagreement"] if len(samples) == 4 else disagreement(samples[:4])


def after(config, lam, selector, uids, mesh_f, tau):
    """Before/after figures for: abstain when the input's 4-seed ensemble disagreement > ``lam``,
    otherwise classify the view ``selector`` moves to -> figures/after_ensemble<lam>/.

    Bars compare all inputs with the kept ones; "filter -> <selector>" scores every input with an
    abstention counted as no answer (not correct, no false positive). With ``tau``, the detector_gap bars
    abstain when SAM3's top-2 gap on its view is below ``tau`` (-> figures/after_ensemble<lam>_gap<tau>/). Needs the analyze stage
    (selections table) and stage1.npz; inputs without an ensemble are left out. Per-object
    grids for ``uids`` (default: one object per category).
    """
    from study.analyze import SUBSETS, summarize
    from study.figures import Heatmap, Images, _mean_map, margin_heatmap, save_bars, save_margin_comparison, save_panels

    output = Path(config["output_dir"])
    db = store.connect(output)
    selections = {}
    for input_id, name, *values in db.execute(
            "SELECT input_id, selector, pick_id, margin, correct, fp_rate, any_fp, tp FROM selections"):
        selections.setdefault(input_id, {})[name] = dict(
            zip(("pick_id", "margin", "correct", "fp_rate", "any_fp", "tp"), values))
    rows = {row["image_id"]: row for row in store.images(db, "source='gt'")}
    directories = dict(db.execute("SELECT input_id, directory FROM reconstructions WHERE status='ok'"))
    prompt_scores = store.prompt_scores(db)
    db.close()
    vocabulary_size = len(config["dataset"]["categories"])
    names = ["input", selector] + [name for name in AFTER_CONTROLS if name != selector]
    score = {}
    for input_id, rows_by_name in selections.items():
        if input_id in directories and all(name in rows_by_name for name in names):
            value = _ensemble4(output / directories[input_id])
            if value is not None:
                score[input_id] = value
    if not score:
        raise SystemExit("no inputs with selections and an ensemble score")
    kept = {i for i, value in score.items() if value <= lam}
    objects = {i: rows[i]["uid"] for i in score}
    directory_out = output / (f"{AFTER}{lam:g}" + ("" if tau is None else f"_gap{tau:g}"))
    combined = f"filter -> {selector}"
    abstain = {"pick_id": None, "margin": float("nan"), "correct": 0.0, "fp_rate": 0.0, "any_fp": 0.0, "tp": 0.0}
    full = {i: {**{n: selections[i][n] for n in names},
                combined: selections[i][selector] if i in kept else abstain} for i in score}
    from study.selectors import detector_gap

    compare = {name: (name, label) for name, label in AFTER_COMPARE.items()}
    abstaining = (combined,)
    if tau is not None:
        gated = f"gap >= {tau:g} -> detector_gap"
        for i in score:
            pick = selections[i]["detector_gap"]
            full[i][gated] = pick if detector_gap(prompt_scores[pick["pick_id"]]) >= tau else abstain
        compare["detector_gap"] = (gated, f"SAM3D top-2 gap >= {tau:g} (all)")
        abstaining += (gated,)
    lines = [f"# Ensemble filter (4-seed disagreement <= {lam:g}) then `{selector}`", "",
             f"{len(score)} inputs with an ensemble score ({len(set(objects.values()))} objects); "
             f"{len(kept)} kept ({len(kept) / len(score):.0%}). `{combined}`: every input, an abstention "
             "counts as no answer (not correct, no false positive).", ""]
    azimuths, elevations = config["dataset"]["azimuths"], config["dataset"]["elevations"]
    for subset, keep in SUBSETS.items():
        ids = [i for i in score if keep(full[i]["input"])]
        if not ids:
            continue
        ids_kept = [i for i in ids if i in kept]
        slug = subset.split(" (")[0].replace(" ", "_")
        everything = summarize({i: full[i] for i in ids}, objects, selector)["subsets"]["all inputs"]["selectors"]
        groups = {f"{name} (all)": everything[name] for name in ("input", selector, "gt_oracle", "oracle", "detector_gap")}
        filtered = {}
        if ids_kept:
            filtered = summarize({i: {n: full[i][n] for n in names} for i in ids_kept}, objects,
                                 selector)["subsets"]["all inputs"]["selectors"]
            groups.update({f"{name} (kept)": filtered[name] for name in names})
        groups.update({f"{name} (all)": everything[name] for name in abstaining})
        coverage = len(ids_kept) / len(ids)
        compared = {**{label: everything[name] for name, label in compare.values()},
                    f"{combined} (all)": everything[combined]}
        save_bars({label: {"top-1 accuracy": row["correct"], "any false positive": row["any_fp"]}
                   for label, row in compared.items()},
                  directory_out / f"summary_{slug}.png",
                  title=f"{subset}: {len(ids)} inputs; kept by ensemble <= {lam:g}: {len(ids_kept)} ({coverage:.0%})",
                  ylabel="fraction of inputs")
        if filtered:
            save_margin_comparison(filtered, directory_out / f"margins_{slug}.png",
                                   title=f"Confidence margin on kept inputs ({subset}, n={len(ids_kept)} of {len(ids)})",
                                   primary=selector)
        lines += [f"## {subset}: {len(ids)} inputs, {len(ids_kept)} kept ({coverage:.0%})", "",
                  "| | top-1 accuracy | any false positive | mean margin |", "|---|---|---|---|"]
        lines += [f"| {label} | {row['correct']:.1%} | {row['any_fp']:.1%} | "
                  + ("" if label.startswith(abstaining) else f"{row['margin']:+.3f}") + " |"
                  for label, row in groups.items()] + [""]
        detection = {label: _detection_metrics([(rows[i]["category"], full[i][name]) for i in ids],
                                               prompt_scores, vocabulary_size)
                     for name, label in [*compare.values(), (combined, f"{combined} (all)")]}
        save_bars(detection, directory_out / f"detection_{slug}.png",
                  title=f"{subset}: SAM3 detection on the chosen view ({len(ids)} inputs)", ylabel="score")
        lines += [f"Detection on the chosen view (score threshold {config['detection']['score_threshold']:g}; "
                  "abstention = nothing detected):", "", "| | precision | recall | F1 | AP |", "|---|---|---|---|---|"]
        lines += [f"| {label} | " + " | ".join(f"{row[k]:.3f}" for k in ("precision", "recall", "F1", "AP")) + " |"
                  for label, row in detection.items()] + [""]
    members = [rows[i] for i in score]
    save_panels(
        [Heatmap(_mean_map(members, {i: float(score[i] > lam) for i in score}),
                 f"share rejected (disagreement > {lam:g})", 0, 1, "Reds", annotate=True),
         Heatmap(_mean_map(members, {i: full[i]["input"]["any_fp"] for i in score}),
                 "any false positive: input", 0, 1, "magma", annotate=True),
         Heatmap(_mean_map(members, {i: full[i][selector]["any_fp"] for i in score}),
                 f"any false positive: {selector}", 0, 1, "magma", annotate=True),
         Heatmap(_mean_map(members, {i: full[i][combined]["any_fp"] for i in score}),
                 f"any false positive: {combined} (abstain = none)", 0, 1, "magma", annotate=True)],
        azimuths, elevations, directory_out / "false_positives.png", ncols=2,
        title=f"False-positive rate by input view, ensemble filter <= {lam:g}")

    by_object = {}
    for i in score:
        by_object.setdefault((rows[i]["category"], rows[i]["uid"]), []).append(i)
    chosen = sorted(key for key in by_object if key[1] in uids) if uids else \
        [min(k for k in by_object if k[0] == category) for category in sorted({k[0] for k in by_object})]
    limit = max(abs(full[i][n]["margin"]) for i in score for n in ("input", selector))
    for category, uid in chosen:
        ids = by_object[(category, uid)]
        angle = {i: (float(rows[i]["azimuth"]), float(rows[i]["elevation"])) for i in ids}
        rejected = {angle[i] for i in ids if i not in kept}
        panels = [Images({angle[i]: output / rows[i]["path"] for i in ids},
                         f"input views (red = rejected: disagreement > {lam:g})", rejected, "red"),
                  Heatmap({angle[i]: score[i] for i in ids}, "4-seed ensemble disagreement", 0,
                          max(0.5, max(score[i] for i in ids)), "Blues", "disagreement",
                          annotate=True, marks=rejected, mark_color="red"),
                  margin_heatmap({angle[i]: full[i]["input"]["margin"] for i in ids},
                                 "SAM3 margin on the input view (before)", limit, annotate=True,
                                 marks=rejected, mark_color="red"),
                  margin_heatmap({angle[i]: full[i][selector]["margin"] for i in ids},
                                 f"SAM3 margin after moving to the {selector} view", limit, annotate=True,
                                 marks=rejected, mark_color="red")]
        fscore = {angle[i]: json.loads((output / directories[i] / "gt_error.json").read_text())["fscore_0.05_scaled"]
                  for i in ids if (output / directories[i] / "gt_error.json").is_file()}
        if fscore:
            panels.append(Heatmap(fscore, f"reconstruction mesh F-score@0.05 (bad < {mesh_f:g})", 0, 1, "RdYlGn",
                                  "F-score", annotate=True, marks=rejected, mark_color="red"))
        input_fp = sum(full[i]["input"]["any_fp"] for i in ids)
        after_fp = sum(full[i][combined]["any_fp"] for i in ids)
        save_panels(panels, azimuths, elevations, directory_out / "objects" / f"{category}_{uid[:8]}.png",
                    ncols=1, title=f"{category} {uid[:8]}: ensemble filter <= {lam:g}, then {selector}",
                    note=f"{len(rejected)}/{len(ids)} views rejected; views with a false positive: "
                         f"{input_fp:.0f} before -> {after_fp:.0f} after filter + {selector}")
    lines += _pick_comparison(output, directory_out, full, rows, score, kept, lam, selector, prompt_scores,
                              config["dataset"]["categories"])
    (directory_out / "after.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines), flush=True)


PICK_EXAMPLES = 6


def _pick_comparison(output, directory_out, full, rows, score, kept, lam, selector, prompt_scores, vocabulary):
    """The view and SAM3 scores chosen by ``detector_gap`` (free over the SAM3D candidates) vs the
    ensemble-gated ``selector`` view: margin/true-score scatters on the kept inputs, and per category
    a gallery of false-positive-before inputs. Returns markdown lines."""
    import matplotlib.pyplot as plt

    gap, gated = "detector_gap", f"filter -> {selector}"

    def scores_of(row):
        return prompt_scores[row["pick_id"]] if row["pick_id"] is not None else None

    def true_score(i, name):
        return scores_of(full[i][name])[rows[i]["category"]]

    shared = sorted(kept)
    figure, axes = plt.subplots(1, 2, figsize=(9, 4.2))
    for ax, (label, value) in zip(axes, [("SAM3 margin", lambda i, n: full[i][n]["margin"]),
                                         ("SAM3 true-category score", true_score)]):
        x = np.array([value(i, gap) for i in shared])
        y = np.array([value(i, selector) for i in shared])
        ax.scatter(x, y, s=4, alpha=0.3)
        low, high = min(x.min(), y.min()), max(x.max(), y.max())
        ax.plot([low, high], [low, high], "k--", lw=0.8)
        ax.set_xlabel(f"{gap} (24 SAM3D candidates)")
        ax.set_ylabel(f"{selector} (kept inputs)")
        ax.set_title(f"{label}\nhigher: {gap} {np.mean(x > y):.0%}, {selector} {np.mean(y > x):.0%}", fontsize=9)
    figure.suptitle(f"Chosen-view SAM3 scores on the {len(shared)} inputs kept by ensemble <= {lam:g}", fontsize=10)
    figure.tight_layout()
    figure.savefig(directory_out / "picks_scatter.png", dpi=130)
    plt.close(figure)

    db = store.connect(output)
    needed = {full[i][n]["pick_id"] for i in score for n in (gap, gated)} - {None}
    paths = {row["image_id"]: row["path"] for row in store.images(db, "i.source='sam3d'") if row["image_id"] in needed}
    db.close()
    paths.update({i: rows[i]["path"] for i in rows})

    def panel(ax, image_id, scores, category, heading):
        ax.imshow(Image.open(output / paths[image_id]).convert("RGB"))
        ax.set_xticks([]), ax.set_yticks([])
        wrong = max((p for p in vocabulary if p != category), key=lambda p: scores[p])
        margin = scores[category] - scores[wrong]
        ax.set_title(f"{heading}\n{category} {scores[category]:.2f} | {wrong} {scores[wrong]:.2f}",
                     fontsize=7, color="green" if margin > 0 else "red")

    rng = np.random.default_rng(0)
    (directory_out / "picks").mkdir(parents=True, exist_ok=True)
    for category in sorted({rows[i]["category"] for i in score}):
        pool = sorted(i for i in score if rows[i]["category"] == category and full[i]["input"]["any_fp"] > 0)
        if not pool:
            continue
        shown = sorted(rng.choice(pool, min(PICK_EXAMPLES, len(pool)), replace=False).tolist())
        figure, axes = plt.subplots(len(shown), 3, figsize=(6.6, 2.4 * len(shown)), squeeze=False)
        for (left, middle, right), i in zip(axes, shown):
            row = rows[i]
            panel(left, i, prompt_scores[i], category,
                  f"input {row['uid'][:6]} az {row['azimuth']:g} el {row['elevation']:g}, u={score[i]:.2f}")
            panel(middle, full[i][gap]["pick_id"], scores_of(full[i][gap]), category, f"{gap} pick")
            if i in kept:
                pick = rows[full[i][selector]["pick_id"]]
                panel(right, pick["image_id"], scores_of(full[i][selector]), category,
                      f"{selector} az {pick['azimuth']:g} el {pick['elevation']:g}")
            else:
                right.axis("off")
                right.text(0.5, 0.5, f"abstain\n(u = {score[i]:.2f} > {lam:g})", ha="center", va="center",
                           color="gray", transform=right.transAxes)
        figure.suptitle(f"{category}: false-positive-before inputs; SAM3 true score | strongest rival "
                         "(green = positive margin)", fontsize=9)
        figure.tight_layout()
        figure.savefig(directory_out / "picks" / f"{category}.png", dpi=110)
        plt.close(figure)

    def mean(name, ids, key):
        return np.mean([true_score(i, name) if key == "true" else full[i][name]["margin"] for i in ids])

    rejected = sorted(set(score) - kept)
    lines = [f"## Chosen view: `{gap}` vs `{gated}`", "",
             "| inputs | n | mean true score: gap / gated | mean margin: gap / gated | margin higher: gap / gated |",
             "|---|---|---|---|---|"]
    margin_gap = np.array([full[i][gap]["margin"] for i in shared])
    margin_gated = np.array([full[i][selector]["margin"] for i in shared])
    lines += [f"| kept (u <= {lam:g}) | {len(shared)} | {mean(gap, shared, 'true'):.3f} / {mean(selector, shared, 'true'):.3f}"
              f" | {margin_gap.mean():+.3f} / {margin_gated.mean():+.3f} | {np.mean(margin_gap > margin_gated):.0%} / "
              f"{np.mean(margin_gated > margin_gap):.0%} |",
              f"| rejected (abstain) | {len(rejected)} | {mean(gap, rejected, 'true'):.3f} / - | "
              f"{mean(gap, rejected, 'margin'):+.3f} / - | - |", ""]
    return lines


REJECTION = "figures/rejection_grids"


def rejection(config, uids, lambdas, mesh_f):
    """Per object and lambda, over all its input views: the inputs, the mesh F-score@0.05 of each
    reconstruction and the 4-seed ensemble disagreement; views rejected (disagreement > lambda) in red."""
    from study.figures import Heatmap, Images, save_panels

    output = Path(config["output_dir"])
    objects = _view_records(output, uids, requires="gt_error.json")
    if not objects:
        raise SystemExit("no input views with both stage1.npz and gt_error.json")
    directory_out = output / REJECTION
    lines = [f"# Views rejected by the ensemble threshold (disagreement > lambda); bad = mesh F-score@0.05 < {mesh_f:g}",
             "", "| object | views | bad | " + " | ".join(
                 f"lambda {value:g}: rejected / bad kept / good rejected" for value in lambdas) + " |",
             "|---|---|---|" + "---|" * len(lambdas)]
    for (category, uid), records in sorted(objects.items()):
        inputs = {r["angle"]: output / r["image"]["path"] for r in records}
        ensemble = {r["angle"]: r["ensemble"] for r in records}
        fscore = {r["angle"]: json.loads((r["folder"] / "gt_error.json").read_text())["fscore_0.05_scaled"]
                  for r in records}
        bad = {angle for angle, value in fscore.items() if value < mesh_f}
        cells = [f"{category} {uid[:8]}", str(len(records)), str(len(bad))]
        for value in lambdas:
            rejected = {angle for angle, score in ensemble.items() if score is None or score > value}
            cells.append(f"{len(rejected)} / {len(bad - rejected)} / {len(rejected - bad)}")
            save_panels(
                [Images(inputs, f"input views (red = rejected: disagreement > {value:g})", rejected, "red"),
                 Heatmap(fscore, "reconstruction mesh F-score@0.05 vs GT (higher = better)", 0, 1, "RdYlGn",
                         "F-score", annotate=True, marks=rejected, mark_color="red"),
                 Heatmap(ensemble, "4-seed ensemble disagreement", 0,
                         max(0.5, max(v for v in ensemble.values() if v is not None)), "Blues", "disagreement",
                         annotate=True, marks=rejected, mark_color="red")],
                config["dataset"]["azimuths"], config["dataset"]["elevations"],
                directory_out / f"{category}_{uid[:8]}_lambda{value:g}.png", ncols=1,
                title=f"{category} {uid[:8]}, lambda = {value:g}",
                note=f"rejected {len(rejected)}/{len(records)} views; bad reconstructions (F < {mesh_f:g}): "
                     f"{len(bad)}, of which {len(bad - rejected)} kept; good reconstructions rejected: "
                     f"{len(rejected - bad)}")
        lines.append("| " + " | ".join(cells) + " |")
    (directory_out / "rejection.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines), flush=True)


def views(config, uids, axis, second="information"):
    """Per object, over all its input views: inputs, SAM3D re-renders, where the 4-seed ensemble
    disagrees, and heatmaps of the ensemble and of ``second``: the image-shape information
    ("information") or the normalized mean-gap ("ratio")."""
    from study.figures import Heatmap, Images, save_panels

    output = Path(config["output_dir"])
    objects = _view_records(output, uids, requires="information.npz" if second == "information" else "meangap.npz")
    directory_out = output / VIEWS
    if second == "information":
        label, other = "-MI", "image-shape information (nats)"
        sign = -1
    else:
        label, other = "ratio", "unexplained-fraction ratio"
        sign = 1
    lines = [f"# Seed ensemble vs {other} across the views of one object", "",
             f"Spearman ρ within each object between ensemble disagreement and {label} "
             "(both: higher = less certain).", "",
             f"| object | views | ensemble | {other} | ρ(ensemble, {label}) |", "|---|---|---|---|---|"]
    for (category, uid), records in sorted(objects.items()):
        inputs, renders, ensemble_maps, second_maps = {}, {}, {}, {}
        for r in records:
            (r["folder"] / "stage1").mkdir(exist_ok=True)
            inputs[r["angle"]] = output / r["image"]["path"]
            renders[r["angle"]] = r["folder"] / "views" / "azimuth_180_elevation_0.png"
            ensemble_maps[r["angle"]] = r["folder"] / "stage1" / f"ensemble4_axis{axis}.png"
            Image.fromarray((_agreement_map(r["samples"], axis) * 255).astype(np.uint8)).resize(
                (128, 128), Image.NEAREST).save(ensemble_maps[r["angle"]])
            second_maps[r["angle"]] = r["folder"] / "stage1" / f"{second}_axis{axis}.png"
            picture = (_information_map(r["information_map"], r["samples"][0], axis) if second == "information"
                       else _agreement_map([r["samples"][0], r["mean"]], axis))
            Image.fromarray((picture * 255).astype(np.uint8)).resize((128, 128), Image.NEAREST).save(
                second_maps[r["angle"]])
        ensemble = {r["angle"]: r["ensemble"] for r in records}
        scores = {r["angle"]: r[second] for r in records}
        finite = [v for v in scores.values() if v is not None]
        if second == "information":
            map_title = "information: where the image shaped the reconstruction (light)"
            heat = Heatmap(scores, "image-shape information, nats (darker = less informative image)",
                           min(finite), max(finite), "Greens_r", "nats")
        else:
            map_title = "reconstruction vs the one-step mean given the image (violet = differ)"
            heat = Heatmap(scores, "unexplained fraction |x - mean(image)|^2 / |x - mean(prior)|^2 "
                           "(higher = less certain)", min(finite), max(finite), "Oranges", "ratio")
        save_panels(
            [Images(inputs, "input view"),
             Images(ensemble_maps, "seed ensemble: where the 4 seeds disagree (violet)"),
             Images(second_maps, map_title),
             Images(renders, "SAM3D reconstruction, re-rendered from the input view"),
             Heatmap(ensemble, "seed ensemble: 1 - mean pairwise F-score (higher = less certain)", 0,
                     max(v for v in ensemble.values() if v is not None), "Blues", "disagreement"),
             heat],
            config["dataset"]["azimuths"], config["dataset"]["elevations"],
            directory_out / f"{category}_{uid[:8]}{'' if second == 'information' else '_ratio'}.png", ncols=3,
            title=f"{category} {uid[:8]}: seed ensemble vs {other} over {len(records)} input views",
            note="Maps show SAM3D's stage-1 occupancy along one axis of its object grid (not the camera); "
                 "gray = agree within 1 voxel; white = empty.")
        values = [r["ensemble"] for r in records]
        others = [np.nan if r[second] is None else r[second] for r in records]
        lines.append(f"| {category} {uid[:8]} | {len(records)} | {min(values):.2f}-{max(values):.2f} | "
                     f"{np.nanmin(others):.3g}-{np.nanmax(others):.3g} | "
                     f"{_spearman(values, [sign * v for v in others]):+.2f} |")
    print("\n".join(lines), flush=True)
    suffix = "" if second == "information" else "_ratio"
    lines += [""] + [f"![{c}]({c}_{u[:8]}{suffix}.png)" for c, u in sorted(objects)]
    (directory_out / f"views{suffix}.md").write_text("\n".join(lines) + "\n")
    print(f"[views] -> {directory_out}", flush=True)


def _explain_picks(records):
    """Three contrasting views of one object: the most ambiguous (highest ensemble), the least
    informative (lowest MI), and the clearest (best combined rank)."""
    ens = np.argsort(np.argsort([r["ensemble"] for r in records]))
    info = np.argsort(np.argsort([r["information"] for r in records]))
    order = {"ambiguous": np.argsort(-ens), "uninformative": np.argsort(info), "clear": np.argsort(ens - info)}
    picks, used = [], set()
    for label, ranked in order.items():
        index = next(int(i) for i in ranked if int(i) not in used)
        used.add(index)
        picks.append((label, records[index]))
    return picks


def prior(config, uids, seeds=4):
    """SAM3D's unconditional shape prior for each object's explainer views -> <reconstruction>/prior.npz."""
    from tools.envs import use_gpu
    use_gpu(config["gpu_id"])
    from reconstruction.sam3d import SAM3D

    output = Path(config["output_dir"])
    pending = [r for records in _view_records(output, uids).values() for _, r in _explain_picks(records)
               if not (r["folder"] / "prior.npz").is_file()]
    print(f"[prior] {len(pending)} views", flush=True)
    if not pending:
        return
    model = SAM3D(checkpoint_dir=config["checkpoints"]["sam3d"])
    base = config["reconstruction"]["seed"]
    for count, r in enumerate(pending, 1):
        occupancy = model.stage1_prior(output / r["image"]["path"], output / r["image"]["mask_path"],
                                       seeds=[base + 1000 * k for k in range(seeds)])
        np.savez_compressed(r["folder"] / "prior.npz", samples=np.packbits(occupancy, axis=-1), seeds=seeds)
        print(f"[prior] {count}/{len(pending)} {r['image']['key']}", flush=True)


def explain(config, uids, axis):
    """How the two scores work, on three contrasting views per object: the 4 seeds and where they
    disagree (ensemble); the shape prior vs the image-conditioned shape from the same seed and where
    the image informed it (information)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output = Path(config["output_dir"])
    rows = [(category, uid, label, r) for (category, uid), records in sorted(_view_records(output, uids).items())
            for label, r in _explain_picks(records) if (r["folder"] / "prior.npz").is_file()]
    if not rows:
        raise SystemExit("no explainer views with prior.npz; run the prior stage first")
    headers = ["input view", "seed 0", "seed 1", "seed 2", "seed 3", "4 seeds overlaid",
               "prior (no image), seed 0", "with the image, seed 0", "where the image informed it"]
    height = len(rows) * 1.6 + 1.0
    fig, axes = plt.subplots(len(rows), len(headers), figsize=(len(headers) * 1.45, height))
    axes = np.atleast_2d(axes)
    for row, (category, uid, label, r) in enumerate(rows):
        prior_samples = np.unpackbits(np.load(r["folder"] / "prior.npz")["samples"], axis=-1).astype(bool)
        panels = [(Image.open(output / r["image"]["path"]), None)]
        panels += [(_silhouette(sample, axis), None) for sample in r["samples"]]
        panels += [(_agreement_map(r["samples"], axis), ENSEMBLE_COLOR),
                   (_silhouette(prior_samples[0], axis), None), (_silhouette(r["samples"][0], axis), None),
                   (_information_map(r["information_map"], r["samples"][0], axis), INFORMATION_COLOR)]
        for col, (picture, frame) in enumerate(panels):
            ax = axes[row, col]
            ax.imshow(picture, cmap="gray", vmin=0, vmax=255)
            ax.set_xticks([]), ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_color(frame or GRID)
                spine.set_linewidth(2 if frame else 0.6)
            if row == 0:
                ax.set_title(headers[col], fontsize=7.5, color=INK)
        axes[row, 0].set_ylabel(f"{category}\n{label}\n({r['angle'][0]:g}, {r['angle'][1]:g})", fontsize=8,
                                color=INK)
        axes[row, 5].set_xlabel(f"disagreement {r['ensemble']:.2f}", fontsize=8, color=ENSEMBLE_COLOR, labelpad=1)
        axes[row, 8].set_xlabel(f"{r['information']:.0f} nats", fontsize=8, color=INFORMATION_COLOR, labelpad=1)
    fig.suptitle("Seed ensemble (columns 2-6): the same image from 4 noise seeds; disagreement = how much the "
                 "shapes differ (gray = agree within 1 voxel, violet = share of seeds missing it).\n"
                 "Image-shape information (columns 7-9): the same seed with and without the image; information "
                 "= how far the image moved the shape from SAM3D's prior (light = informed, dark = little).\n"
                 "Shapes: stage-1 occupancy along one axis of SAM3D's object grid, darker = nearer. The prior "
                 "ignores the image, so the same seed gives the same prior shape in every row.",
                 fontsize=8.5, color=INK, y=1 - 0.08 / height, va="top")
    fig.tight_layout(rect=(0, 0, 1, 1 - 0.7 / height))
    directory_out = output / VIEWS
    directory_out.mkdir(parents=True, exist_ok=True)
    fig.savefig(directory_out / "explainer.png", dpi=125)
    plt.close(fig)
    print(f"[explain] -> {directory_out / 'explainer.png'}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("stage", choices=["stage1", "fisher", "laplace", "information", "prior", "meangap",
                                          "figures", "threshold", "agreement", "rejection", "after", "gt_error", "poc", "views", "explain",
                                          "meangap_test"])
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=None,
                        help="laplace: weight draws (default 4); "
                             "information: log-SNR draws (default 32)")
    parser.add_argument("--ensemble", type=int, default=3, help="stage1: seeds including the reconstruction's")
    parser.add_argument("--fisher-inputs", type=int, default=256, help="fisher: reconstructions used")
    parser.add_argument("--fisher-times", type=int, default=4, help="fisher: flow times per input")
    parser.add_argument("--fisher-split", default="train", help="fisher: manifest split of the inputs")
    parser.add_argument("--prior", type=float, default=None,
                        help="laplace: prior precision (default: maximize the Laplace evidence)")
    # The true SAM3D training size (~1e6+) collapses the last-layer posterior (occupancy
    # spread exactly 0); input rankings were stable for N in 1..1e4 on a pilot.
    parser.add_argument("--data-size", type=float, default=100.0,
                        help="laplace: effective training-set size scaling the mean Fisher (a temperature)")
    parser.add_argument("--noise", type=float, default=1e-3, help="laplace: entropy sigma^2 (paper: 1e-3)")
    parser.add_argument("--name", default="laplace.npz",
                        help="laplace: output file per reconstruction (figures and threshold read laplace.npz)")
    parser.add_argument("--names", nargs="+", default=["laplace.npz"],
                        help="poc: laplace output files to compare; the first one ranks the inputs")
    parser.add_argument("--top", type=int, default=8, help="poc: inputs shown at each extreme")
    parser.add_argument("--uids", nargs="+", help="stage1, laplace, information, views: only these objects")
    parser.add_argument("--elevations", type=float, nargs="+",
                        help="stage1, laplace, information: only these input elevations")
    parser.add_argument("--fisher", type=Path,
                        help="laplace: Fisher file from another study (default: <output_dir>/laplace_fisher.npz)")
    parser.add_argument("--second", default="information", choices=["information", "ratio"],
                        help="views: score shown next to the ensemble")
    parser.add_argument("--reference-seeds", type=int, default=8, choices=[4, 8],
                        help="meangap_test: 8 = seeds 4-7 of 8-seed runs, 4 = seeds 1-3 of 4-seed runs")
    parser.add_argument("--requires", default="laplace.npz",
                        help="gt_error: only reconstructions with this file ('' for all)")
    parser.add_argument("--axis", type=int, default=0, help="figures: occupancy-grid projection axis")
    parser.add_argument("--shard", type=int, nargs=2, default=(0, 1), metavar=("INDEX", "COUNT"))
    parser.add_argument("--gpu", type=int, help="override the config's gpu_id")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--objects-per-category", type=int,
                        help="stage1, laplace: only the first N objects (by uid) of each category")
    parser.add_argument("--method", default="ensemble",
                        help="threshold: score to calibrate (ensemble, laplace, laplace_entropy)")
    parser.add_argument("--criterion", default="disagree", choices=sorted(CRITERIA),
                        help="threshold: what makes a reconstruction not useful")
    parser.add_argument("--mesh-f", type=float, default=0.8,
                        help="threshold: mesh criterion, not useful below this F-score@0.05")
    parser.add_argument("--alphas", type=float, nargs="+", default=[0.01, 0.05, 0.1],
                        help="threshold: target not-useful rates among kept reconstructions")
    parser.add_argument("--sweep", type=float, nargs="+", default=list(np.round(np.arange(0.1, 0.301, 0.02), 2)),
                        help="threshold: lambdas reported on all inputs")
    parser.add_argument("--lambdas", type=float, nargs="+", default=[0.1, 0.2, 0.3],
                        help="rejection: ensemble thresholds to show; after: the first one")
    parser.add_argument("--selector", default="gt_vqf", help="after: view selector applied to kept inputs")
    parser.add_argument("--tau", type=float,
                        help="after: detector_gap abstains when the top-2 gap of its view is below this")
    parser.add_argument("--delta", type=float, default=0.1, help="threshold: allowed failure probability")
    parser.add_argument("--max-brightness", type=float,
                        help="threshold: drop inputs whose object pixels average brighter (white-on-white)")
    parser.add_argument("--splits", type=int, default=200, help="threshold: random category splits")
    args = parser.parse_args()
    config = settings_io.load(args.config)
    if args.gpu is not None:
        config["gpu_id"] = args.gpu
    if args.stage == "stage1":
        stage1(config, args.ensemble, args.shard, args.limit,
               args.objects_per_category, set(args.uids or ()), args.elevations)
    elif args.stage == "fisher":
        fisher(config, args.fisher_inputs, args.fisher_times, args.fisher_split, args.limit)
    elif args.stage == "laplace":
        laplace(config, args.prior, args.data_size, args.samples or 4, args.noise, args.shard, args.limit,
                args.objects_per_category, args.name, set(args.uids or ()), args.elevations, args.fisher)
    elif args.stage == "gt_error":
        gt_error(config, args.requires or None, args.shard, args.limit)
    elif args.stage == "information":
        information(config, args.samples or 32, args.shard, args.limit, args.objects_per_category,
                    set(args.uids or ()), args.elevations)
    elif args.stage == "prior":
        prior(config, set(args.uids or ()))
    elif args.stage == "explain":
        explain(config, set(args.uids or ()), args.axis)
    elif args.stage == "meangap":
        meangap(config, args.shard, args.limit, set(args.uids or ()), args.elevations, args.objects_per_category)
    elif args.stage == "meangap_test":
        meangap_test(config, set(args.uids or ()), args.reference_seeds)
    elif args.stage == "views":
        views(config, set(args.uids or ()), args.axis, args.second)
    elif args.stage == "poc":
        poc(config, args.names, args.top, args.axis)
    elif args.stage == "after":
        after(config, args.lambdas[0], args.selector, set(args.uids or ()), args.mesh_f, args.tau)
    elif args.stage == "rejection":
        rejection(config, set(args.uids or ()), args.lambdas, args.mesh_f)
    elif args.stage == "agreement":
        agreement(config)
    elif args.stage == "figures":
        figures(config, args.axis)
    else:
        threshold(config, args.method, args.criterion, args.alphas, args.delta, args.splits,
                  config["picks"]["seed"] if config.get("picks") else 0, args.mesh_f, args.sweep,
                  args.max_brightness)


if __name__ == "__main__":
    main()
