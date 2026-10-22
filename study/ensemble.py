"""Calibrate the ensemble filter's threshold on a study with ground-truth meshes.

    # CPU, sam3d env: shape error of each reconstruction -> <reconstruction>/gt_error.json
    sam3d-env sam3d python -m study.ensemble gt_error --config configs/lvis_uncertainty.yaml [--shard 0 4]
    # core env: risk-coverage curve and Learn-then-Test threshold -> figures/ensemble/
    sam3d-env core python -m study.ensemble calibrate --config configs/lvis_uncertainty.yaml \\
        --mesh-f 0.5 --max-brightness 0.9

A reconstruction is *not useful* when its scale-searched F-score@0.05 against the
ground-truth mesh is below ``--mesh-f``; it is trusted when its stage-1 ensemble
disagreement (``stage1.npz``) is <= lambda. Lambda is chosen by Learn-then-Test on
calibration categories and evaluated on held-out ones (the ViewActive unseen
categories, and random half/half category splits), so the threshold is not tuned
to the test categories. Result on 1,800 LVIS objects (textured, F < 0.5):
lambda 0.10-0.20 keeps ~99% of useful reconstructions.
"""
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

from study import config as settings_io
from study import db as store
from reconstruction.stage1_uncertainty import load_ensemble

FIGURES = "figures/ensemble"


def _reconstructions(db):
    rows = db.execute("SELECT input_id, directory FROM reconstructions WHERE status='ok' "
                      "ORDER BY input_id").fetchall()
    inputs = {image["image_id"]: image for image in store.images(db, "source='gt'")}
    return [(inputs[input_id], directory) for input_id, directory in rows]


def gt_error(config, shard, limit=None):
    """Shape error against the ground-truth mesh -> <reconstruction>/gt_error.json."""
    from harness.meshio import ObjaverseStore
    from reconstruction.gt_error import shape_error

    output = Path(config["output_dir"])
    db = store.connect(output)
    index, count = shard
    pending = [(image, directory) for position, (image, directory) in enumerate(_reconstructions(db))
               if position % count == index and not (output / directory / "gt_error.json").is_file()][:limit]
    db.close()
    print(f"[gt_error] {len(pending)} reconstructions", flush=True)
    meshes = ObjaverseStore(config["dataset"]["root"], config["dataset"]["canonical"])
    for done, (image, directory) in enumerate(pending, 1):
        truth = meshes.load(image["uid"], device="cpu")
        if truth is None:
            print(f"[gt_error] {image['key']}: no ground-truth mesh", flush=True)
            continue
        predicted = np.load(output / directory / "mesh.npz")
        error = shape_error((predicted["vertices"], predicted["faces"]),
                            (truth.verts.numpy(), truth.faces.numpy()), image["azimuth"], image["elevation"])
        (output / directory / "gt_error.json").write_text(json.dumps(error) + "\n")
        print(f"[gt_error] {done}/{len(pending)} {image['key']} F@0.05={error['fscore_0.05']:.3f}", flush=True)


def _auroc(scores, labels):
    """P(score of a positive > score of a negative), ties count half; NaN without both classes."""
    from scipy.stats import rankdata
    scores, labels = np.asarray(scores, float), np.asarray(labels, bool)
    positives, negatives = labels.sum(), (~labels).sum()
    if not positives or not negatives:
        return float("nan")
    ranks = rankdata(scores)
    return float((ranks[labels].sum() - positives * (positives + 1) / 2) / (positives * negatives))


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


def records(config, mesh_f):
    """Per reconstruction with both files: category, disagreement, not-useful label, brightness."""
    output = Path(config["output_dir"])
    db = store.connect(output)
    members = _reconstructions(db)
    db.close()
    rows = []
    for image, directory in members:
        folder = output / directory
        if not (folder / "stage1.npz").is_file() or not (folder / "gt_error.json").is_file():
            continue
        score = load_ensemble(folder / "stage1.npz")["ensemble"]["disagreement"]
        mesh = json.loads((folder / "gt_error.json").read_text())
        if score is None or "fscore_0.05_scaled" not in mesh:
            continue
        with Image.open(output / image["mask_path"]) as source:
            mask = np.asarray(source.convert("L")) > 127
        with Image.open(output / image["path"]) as source:
            pixels = np.asarray(source.convert("RGB"))[mask]
        rows.append({"uid": image["uid"], "category": image["category"], "score": score,
                     "bad": mesh["fscore_0.05_scaled"] < mesh_f,
                     "brightness": float(pixels.mean() / 255) if pixels.size else 1.0})
    return rows


def calibrate(config, mesh_f, alphas, delta, splits, seed, sweep, max_brightness=None):
    """Risk-coverage sweep and Learn-then-Test thresholds -> figures/ensemble/threshold.md + .png.

    ``max_brightness`` drops inputs whose object pixels average brighter than it:
    untextured models rendered white on white, where SAM3D sees little but a silhouette.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = records(config, mesh_f)
    dropped = 0
    if max_brightness is not None:
        dropped = sum(r["brightness"] > max_brightness for r in rows)
        rows = [r for r in rows if r["brightness"] <= max_brightness]
    if not rows:
        raise SystemExit("no reconstructions with both stage1.npz and gt_error.json")
    manifest = json.loads(Path(config["dataset"]["manifest"]).read_text())
    unseen = {r["category"] for r in rows if manifest[r["uid"]]["split"] == "test_unseen"}
    scores = np.array([r["score"] for r in rows])
    categories = np.array([r["category"] for r in rows])
    bad = np.array([r["bad"] for r in rows])
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
        return {"lambda": float(value), **rates(test & (scores <= value), test), "base_risk": bad[test].mean()}

    everything = np.ones(len(rows), bool)
    lines = [f"Trust the reconstruction when ensemble disagreement <= lambda. Not useful = mesh "
             f"F-score@0.05 < {mesh_f:g}. {len(rows)} inputs (one per object), {len(set(categories))} "
             f"categories, {bad.mean():.1%} not useful"
             + (f"; {dropped} near-white inputs (brightness > {max_brightness:g}) dropped" if dropped else "")
             + ". Risk = not useful among kept; FPR = share of not-useful reconstructions kept; "
             "TPR = share of useful ones kept.", "",
             "## Sweep, all inputs", "", f"| lambda | coverage | risk | risk {1 - delta:.0%} upper bound | FPR | TPR |",
             "|---|---|---|---|---|---|"]
    for value in sweep:
        row = rates(scores <= value, everything)
        lines.append(f"| {value:.2f} | {row['coverage']:.0%} | {row['risk']:.1%} | "
                     f"{_upper_bound(bad, scores <= value, delta):.1%} | {row['fpr']:.1%} | {row['tpr']:.1%} |")
    held_out = np.isin(categories, sorted(unseen))
    lines += ["", f"## Learn-then-Test (delta = {delta:g}, Bonferroni over lambda = 0.01..0.50), held-out "
              f"ViewActive unseen categories ({held_out.sum()} inputs: {', '.join(sorted(unseen))})", "",
              "| alpha | lambda | coverage | risk | FPR | TPR | no-filter risk |", "|---|---|---|---|---|---|---|"]
    for alpha in alphas:
        row = evaluate(~held_out, held_out, alpha)
        lines.append(f"| {alpha:g} | none certified | | | | | {bad[held_out].mean():.1%} |" if row is None else
                     f"| {alpha:g} | {row['lambda']:.3f} | {row['coverage']:.0%} | {row['risk']:.1%} | "
                     f"{row['fpr']:.1%} | {row['tpr']:.1%} | {row['base_risk']:.1%} |")
    rng = np.random.default_rng(seed)
    names = np.unique(categories)
    tests = [np.isin(categories, rng.choice(names, len(names) // 2, replace=False)) for _ in range(splits)]
    lines += ["", f"## {splits} random half/half category splits", "",
              "| alpha | certified | median lambda | mean coverage | mean risk | P(risk > alpha) | mean FPR |",
              "|---|---|---|---|---|---|---|"]
    for alpha in alphas:
        valid = [r for r in (evaluate(~test, test, alpha) for test in tests) if r is not None]
        if not valid:
            lines.append(f"| {alpha:g} | 0% | | | | | |")
            continue
        mean = lambda key: np.nanmean([r[key] for r in valid])
        lines.append(f"| {alpha:g} | {len(valid) / splits:.0%} | {np.median([r['lambda'] for r in valid]):.3f} | "
                     f"{mean('coverage'):.0%} | {mean('risk'):.1%} | {np.mean([r['risk'] > alpha for r in valid]):.0%} | "
                     f"{mean('fpr'):.1%} |")
    within = [_auroc(scores[categories == c], bad[categories == c]) for c in names]
    lines += ["", f"AUROC for not useful: {_auroc(scores, bad):.2f} pooled, "
              f"{np.nanmedian(within):.2f} median within category."]

    directory = Path(config["output_dir"]) / FIGURES
    directory.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 4))
    order = np.argsort(scores, kind="stable")
    kept = np.arange(1, len(order) + 1)
    risk = np.cumsum(bad[order]) / kept
    ax.plot(kept / len(order), risk, label="ensemble disagreement")
    for alpha in alphas:
        ax.axhline(alpha, color="k", lw=0.6, ls=":")
        ax.text(1, alpha, f" alpha {alpha:g}", fontsize=6, va="center", ha="left")
    for index, value in enumerate(sweep[::max(1, len(sweep) // 5)]):
        chosen = scores <= value
        ax.plot(chosen.mean(), bad[chosen].mean(), "o", color="C0", ms=4)
        ax.annotate(f"lambda {value:.2f}\n{chosen.mean():.0%} kept, {bad[chosen].mean():.1%} not useful",
                    (chosen.mean(), bad[chosen].mean()), textcoords="offset points",
                    xytext=(0, 10 if index % 2 else -24), fontsize=6, ha="center")
    top = risk[kept >= 0.05 * len(order)].max()
    ax.set(xlabel="coverage (share of inputs kept)", ylabel="not useful among kept",
           title=f"mesh F-score@0.05 < {mesh_f:g} ({len(rows)} inputs)", xlim=(0.05, 1),
           ylim=(0, 1.3 * max(top, max(alphas))))
    ax.title.set_fontsize(8)
    ax.legend(fontsize=7)
    fig.tight_layout()
    suffix = f"f{mesh_f:g}" + ("_textured" if max_brightness is not None else "")
    fig.savefig(directory / f"risk_coverage_{suffix}.png", dpi=110)
    plt.close(fig)
    (directory / f"threshold_{suffix}.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("stage", choices=("gt_error", "calibrate"))
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--shard", type=int, nargs=2, default=(0, 1), metavar=("INDEX", "COUNT"),
                        help="gt_error: CPU processes (index, count)")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--mesh-f", type=float, default=0.5, help="not useful below this F-score@0.05")
    parser.add_argument("--alphas", type=float, nargs="+", default=[0.01, 0.05, 0.1],
                        help="target not-useful rates among kept reconstructions")
    parser.add_argument("--delta", type=float, default=0.1, help="allowed failure probability")
    parser.add_argument("--splits", type=int, default=200, help="random category splits")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--sweep", type=float, nargs="+", default=list(np.round(np.arange(0.1, 0.301, 0.02), 2)),
                        help="lambdas reported on all inputs")
    parser.add_argument("--max-brightness", type=float,
                        help="drop inputs whose object pixels average brighter (white-on-white)")
    args = parser.parse_args()
    config = settings_io.load(args.config)
    if args.stage == "gt_error":
        gt_error(config, args.shard, args.limit)
    else:
        calibrate(config, args.mesh_f, args.alphas, args.delta, args.splits, args.seed, args.sweep,
                  args.max_brightness)


if __name__ == "__main__":
    main()
