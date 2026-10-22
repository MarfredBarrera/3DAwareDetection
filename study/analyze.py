"""Stage: before/after analysis from stored detections and ambiguity scores.

Before = SAM3 on the ground-truth view. After = SAM3 on the SAM3D candidate
chosen by each selector (see study/selectors.py). Writes the ``selections``
table, ``summary.json``/``summary.md``, and figures under ``figures/``.
"""
import argparse
import json
import math
from pathlib import Path

import numpy as np

from study import config as settings_io
from study import db as store
from study import figures
from study.labels import outcomes
from study.selectors import selector_names, weights

FIELDS = ("margin", "correct", "fp_rate", "any_fp", "tp")
SUBSETS = {"all inputs": lambda before: True,
           "misleading before (margin < 0)": lambda before: before["margin"] < 0,
           "false positive before": lambda before: before["any_fp"] > 0}
BOOTSTRAP = 1000


def _values(outcome, vocabulary_size):
    return {"margin": outcome["margin"], "correct": float(outcome["correct"]),
            "fp_rate": outcome["fp"] / (vocabulary_size - 1), "any_fp": float(outcome["any_fp"]),
            "tp": float(outcome["tp"])}


def uncertainties(db, config):
    """{input_id: stage-1 ensemble disagreement} for every reconstruction (None = undefined)."""
    from reconstruction.stage1_uncertainty import load_ensemble

    output = Path(config["output_dir"])
    rows = db.execute("SELECT input_id, directory FROM reconstructions WHERE status='ok'").fetchall()
    missing = [d for _, d in rows if not (output / d / "stage1.npz").is_file()]
    if missing:
        raise ValueError(f"analysis.ensemble_threshold needs stage1.npz; {len(missing)} reconstructions "
                         f"lack it (e.g. {missing[0]}); reconstruct with reconstruction.ensemble > 1")
    return {i: load_ensemble(output / d / "stage1.npz")["ensemble"]["disagreement"] for i, d in rows}


def select_all(db, config):
    """Rebuild the ``selections`` table; returns {input_id: {selector: row}}.

    With ``analysis.ensemble_threshold``, inputs whose reconstruction the ensemble
    filter distrusts keep the input view for every selector.
    """
    from perception.select import trusted

    vocabulary = config["dataset"]["categories"]
    detection = config["detection"]
    metrics = list(config["ambiguity"])
    names = selector_names(metrics, vqf=config.get("picks") is not None)
    before = outcomes(db, vocabulary, detection, "i.source='gt'")
    after = outcomes(db, vocabulary, detection, "i.source='sam3d'")
    scores = store.view_scores(db, "i.source='sam3d'")
    prompt_scores = store.prompt_scores(db, "i.source='sam3d'")
    vqf_picks = dict(db.execute("SELECT input_id, pick_id FROM picks WHERE selector='vqf'"))
    threshold = config["analysis"].get("ensemble_threshold")
    uncertainty = uncertainties(db, config) if threshold is not None else {}
    children, angles = {}, {}
    for row in store.images(db, "source='sam3d'"):
        if row["image_id"] in after:
            children.setdefault(row["parent_id"], []).append(row["image_id"])
            angles[row["image_id"]] = (row["azimuth"], row["elevation"])
    size = len(vocabulary)
    result = {}
    for input_id, candidates in children.items():
        if input_id not in before:
            continue
        rows = {"input": {**_values(before[input_id], size), "pick_id": input_id}}
        if not trusted(uncertainty.get(input_id), threshold):
            rows.update({name: rows["input"] for name in names})
            result[input_id] = rows
            continue
        margins = {i: after[i]["margin"] for i in candidates}
        for name in names:
            weight = weights(name, candidates, scores, margins, metrics, angles, prompt_scores,
                             config["analysis"].get("combined_weights"), vqf_picks.get(input_id))
            if weight is None:
                continue
            values = {field: sum(w * _values(after[i], size)[field] for i, w in weight.items())
                      for field in FIELDS}
            rows[name] = {**values, "pick_id": max(weight, key=weight.get)}
        result[input_id] = rows
    with db:
        db.execute("DELETE FROM selections")
        db.executemany("INSERT INTO selections VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                       [(input_id, name, row["pick_id"], *(row[f] for f in FIELDS))
                        for input_id, rows in result.items() for name, row in rows.items()])
    return result


def _bootstrap(values, groups, seed=0):
    """95% interval of the mean, resampling objects (views of one object are correlated)."""
    members = {}
    for value, group in zip(values, groups):
        members.setdefault(group, []).append(value)
    blocks = [np.asarray(v, dtype=float) for v in members.values()]
    rng = np.random.default_rng(seed)
    means = []
    for _ in range(BOOTSTRAP):
        sample = [blocks[i] for i in rng.integers(0, len(blocks), len(blocks))]
        means.append(np.concatenate(sample).mean())
    return [float(v) for v in np.percentile(means, [2.5, 97.5])]


def summarize(selections, objects, primary):
    """Mean outcome per selector and subset, with object-bootstrap intervals."""
    names = list(dict.fromkeys(name for rows in selections.values() for name in rows))
    complete = {i: rows for i, rows in selections.items() if all(n in rows for n in names)}
    summary = {"inputs": len(complete), "objects": len({objects[i] for i in complete}),
               "selectors": names, "subsets": {}}
    for subset, keep in SUBSETS.items():
        ids = [i for i, rows in complete.items() if keep(rows["input"])]
        if not ids:
            continue
        groups = [objects[i] for i in ids]
        table = {}
        for name in names:
            table[name] = {"n": len(ids)}
            for field in FIELDS:
                values = [complete[i][name][field] for i in ids]
                table[name][field] = float(np.mean(values))
                table[name][f"{field}_ci95"] = _bootstrap(values, groups)
        paired = {}
        for reference in ("input", "identity", "random"):
            if reference in names and primary in names and reference != primary:
                difference = np.array([complete[i][primary]["margin"] - complete[i][reference]["margin"]
                                       for i in ids])
                paired[reference] = {"improved": float(np.mean(difference > 1e-9)),
                                     "worse": float(np.mean(difference < -1e-9)),
                                     "mean_margin_change": float(difference.mean())}
        result = {"selectors": table, f"{primary} vs": paired}
        summary["subsets"][subset] = result
    return summary


def write_markdown(summary, before, primary, path):
    lines = ["# Viewpoint study summary", "",
             f"Before (ground-truth views): {before['views']} visible views of {before['objects']} objects; "
             f"top-1 accuracy {before['correct']:.1%}, views with any false positive {before['any_fp']:.1%}, "
             f"mean confidence margin {before['margin']:+.3f}.", ""]
    if "ensemble_filter" in summary:
        ensemble = summary["ensemble_filter"]
        lines += [f"Ensemble filter on: disagreement <= {ensemble['threshold']:g} trusts "
                  f"{ensemble['trusted']:.1%} of reconstructions; the rest keep the input view.", ""]
    if not summary["subsets"]:
        lines += ["No reconstructed inputs have complete detections and scores yet.", ""]
    for subset, data in summary["subsets"].items():
        lines += [f"## After: {subset}", "",
                  f"{summary['inputs']} reconstructed inputs in total; this subset has "
                  f"{next(iter(data['selectors'].values()))['n']}. "
                  "Intervals resample objects (95%).", "",
                  "| selector | margin | top-1 acc | any FP | FP rate | TP recall |",
                  "|---|---|---|---|---|---|"]
        for name, row in data["selectors"].items():
            cells = [f"{row['margin']:+.3f}"] + [
                f"{row[f]:.1%} [{row[f + '_ci95'][0]:.0%}, {row[f + '_ci95'][1]:.0%}]"
                for f in ("correct", "any_fp")] + [f"{row['fp_rate']:.1%}", f"{row['tp']:.1%}"]
            lines.append(f"| {name} | " + " | ".join(cells) + " |")
        lines += [""]
        for reference, paired in data[f"{primary} vs"].items():
            lines.append(f"- `{primary}` vs `{reference}`: margin improved on {paired['improved']:.1%}, "
                         f"worse on {paired['worse']:.1%} (mean change {paired['mean_margin_change']:+.3f}).")
        lines.append("")
    lines += ["`input` is SAM3 on the original view; `identity` re-renders the reconstruction from "
              "the input view; `top2gap` and `vqf` are the label-free selectors of `perception`; "
              "`oracle` picks the best candidate using SAM3 itself (upper bound; see study/selectors.py).", ""]
    Path(path).write_text("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--settings", type=Path, required=True)
    parser.add_argument("--no-object-figures", action="store_true",
                        help="skip the per-object figures (the slow part)")
    parser.add_argument("--workers", type=int, default=1, help="processes for per-object figures")
    args = parser.parse_args()
    config = settings_io.read(args.settings)
    output = Path(config["output_dir"])
    primary = config["analysis"]["primary_selector"]
    db = store.connect(output)
    vocabulary = config["dataset"]["categories"]
    gt = outcomes(db, vocabulary, config["detection"], "i.source='gt'")
    if not gt:
        raise ValueError("No ground-truth detections; run the render and detect stages first")
    gt_rows = {row["image_id"]: row for row in store.images(db, "source='gt'")}
    before = {"views": len(gt), "objects": len({gt_rows[i]["uid"] for i in gt}),
              **{field: float(np.mean([_values(o, len(vocabulary))[field] for o in gt.values()]))
                 for field in FIELDS}}
    selections = select_all(db, config)
    summary = summarize(selections, {i: gt_rows[i]["uid"] for i in selections}, primary)
    summary = {"before": before, "primary_selector": primary, "detection": config["detection"], **summary}
    threshold = config["analysis"].get("ensemble_threshold")
    if threshold is not None:
        from perception.select import trusted
        kept = [trusted(u, threshold) for i, u in uncertainties(db, config).items() if i in selections]
        summary["ensemble_filter"] = {"threshold": threshold, "trusted": float(np.mean(kept)) if kept else None}
    (output / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    write_markdown(summary, before, primary, output / "summary.md")
    paths = figures.study_figures(db, config, gt, selections, summary,
                                  object_figures=not args.no_object_figures,
                                  workers=args.workers)
    db.close()
    print(f"[analyze] {len(selections)} reconstructed inputs; {len(paths)} figures; "
          f"summary -> {output / 'summary.md'}", flush=True)


if __name__ == "__main__":
    main()
