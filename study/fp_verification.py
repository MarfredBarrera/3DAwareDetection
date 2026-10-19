"""Verify an open-vocabulary detection using scores from SAM3D rendered views.

The input detector must first fire. A verifier only vetoes that claim; it never
creates a detection. Views are rendered from the same reconstructed object and
scored with the same prompt. No category-specific shape template is used.
"""
import argparse
from itertools import groupby
import json
import math
from pathlib import Path

import numpy as np

from study import config as settings_io
from study import db as store
from study.labels import outcome, true_positive
from tools.cameras import SAM3D_INPUT_VIEW

METHODS = ("majority", "lower_quartile", "relative_margin", "peak_margin")


def verify_claim(prompt, input_score, views, *, method="majority",
                 score_threshold=0.5, min_views=3, margin_threshold=0.2):
    """Return whether an existing detection survives reconstructed views.

    ``majority`` requires more than half the views to exceed the detector's
    threshold. ``lower_quartile`` requires the 25th percentile score to exceed
    it, making one lucky view insufficient. ``relative_margin`` requires the
    claim to score above threshold and beat all supplied rival prompts on
    more than half the views. The
    ``peak_margin`` accepts if any view scores the claim above threshold and
    beats its strongest rival by at least ``margin_threshold``. The last two
    rules need a comparison vocabulary; the first two work with one prompt.
    Incomplete or non-finite evidence fails closed.
    """
    if method not in METHODS:
        raise ValueError(f"Unknown verification method: {method}")
    if not 0 <= score_threshold <= 1 or min_views < 1 or not 0 <= margin_threshold <= 1:
        raise ValueError("Invalid score threshold or minimum view count")
    if not math.isfinite(input_score) or input_score < score_threshold:
        return False
    views = list(views)
    if len(views) < min_views:
        return False
    scores = []
    margins = []
    for view in views:
        score = view.get(prompt)
        if score is None or not math.isfinite(score):
            return False
        scores.append(score)
        if method in ("relative_margin", "peak_margin"):
            rivals = [value for name, value in view.items() if name != prompt]
            if not rivals or not all(math.isfinite(value) for value in rivals):
                return False
            margins.append(score - max(rivals))
    if method == "majority":
        return sum(score >= score_threshold for score in scores) > len(scores) / 2
    if method == "lower_quartile":
        return float(np.quantile(scores, 0.25)) >= score_threshold
    if method == "peak_margin":
        return any(score >= score_threshold and margin >= margin_threshold
                   for score, margin in zip(scores, margins))
    return sum(score >= score_threshold and margin > 0
               for score, margin in zip(scores, margins)) > len(margins) / 2


def _child_groups(db):
    """Stream candidate score maps and true-prompt masks by input."""
    rows = db.execute(
        "SELECT i.parent_id, i.image_id, i.azimuth, i.elevation, i.visible_px, "
        "d.prompt, d.score, d.candidates "
        "FROM images i JOIN detections d ON d.image_id=i.image_id "
        "WHERE i.source='sam3d' ORDER BY i.parent_id, i.image_id")
    for parent_id, family in groupby(rows, key=lambda row: row[0]):
        views = []
        for _, detection_rows in groupby(family, key=lambda row: row[1]):
            found = list(detection_rows)
            payload = next((row[7] for row in found if row[7] is not None), None)
            views.append({"azimuth": found[0][2], "elevation": found[0][3],
                          "visible_px": found[0][4],
                          "scores": {row[5]: row[6] for row in found},
                          "candidates": store.unpack(payload) if payload is not None else None})
        yield parent_id, views


def evaluate(db, config):
    """Evaluate claim-level FP rejection on originally detected study prompts.

    Wrong-category prompts are semantic false positives. The true prompt counts
    as a positive only when its original mask meets the study's IoU criterion.
    True-prompt detections with poor localization are excluded from this semantic
    test because their label may still be correct.
    """
    vocabulary = config["dataset"]["categories"]
    detection = config["detection"]
    threshold = detection["score_threshold"]
    iou_threshold = detection["iou_threshold"]
    from study.analyze import _values, summarize
    inputs = {row["image_id"]: row for row in store.images(db, "i.source='gt'")}
    original = store.prompt_scores(db, "i.source='gt'")
    candidates = store.true_candidates(db, "i.source='gt'")
    counts = {name: {"tp": 0, "fp": 0} for name in
              ("input", "identity", "oracle", *METHODS)}
    object_counts = {}
    outcomes_by_input = {}
    object_ids = {}
    eligible_inputs = 0
    for parent_id, raw_views in _child_groups(db):
        image = inputs.get(parent_id)
        input_scores = original.get(parent_id, {})
        if (image is None or image["visible_px"] <= 0
                or set(input_scores) != set(vocabulary)):
            continue
        if len(raw_views) < 3 or any(set(view["scores"]) != set(vocabulary)
                                    for view in raw_views):
            continue
        complete_views = [view["scores"] for view in raw_views]
        identity = next((view for view in raw_views
                         if math.isclose(view["azimuth"], SAM3D_INPUT_VIEW[0], abs_tol=1e-6)
                         and math.isclose(view["elevation"], SAM3D_INPUT_VIEW[1], abs_tol=1e-6)), None)
        if identity is None:
            continue
        category = image["category"]
        visible_views = [view for view in raw_views if view["visible_px"] > 0]
        if not visible_views:
            continue
        oracle = max(visible_views, key=lambda view: (
            view["scores"][category]
            - max(score for name, score in view["scores"].items() if name != category),
            view["scores"][category]))
        before_outcome = outcome(image, input_scores, candidates.get(parent_id),
                                 vocabulary, detection)
        identity_outcome = outcome({**image, "visible_px": identity["visible_px"]},
                                   identity["scores"], identity["candidates"],
                                   vocabulary, detection)
        oracle_outcome = outcome({**image, "visible_px": oracle["visible_px"]},
                                 oracle["scores"], oracle["candidates"],
                                 vocabulary, detection)
        if identity_outcome is None or oracle_outcome is None:
            continue
        result_rows = {"input": _values(before_outcome, len(vocabulary)),
                       "identity": _values(identity_outcome, len(vocabulary)),
                       "oracle": _values(oracle_outcome, len(vocabulary))}
        for method in METHODS:
            retained_scores = dict(input_scores)
            kept_true = True
            for prompt, input_score in input_scores.items():
                if input_score < threshold:
                    continue
                if not verify_claim(prompt, input_score, complete_views, method=method,
                                    score_threshold=threshold):
                    retained_scores[prompt] = 0.0
                    if prompt == category:
                        kept_true = False
            value = outcome(image, retained_scores,
                            candidates.get(parent_id) if kept_true else None,
                            vocabulary, detection)
            result_rows[method] = _values(value, len(vocabulary))
        outcomes_by_input[parent_id] = result_rows
        object_ids[parent_id] = image["uid"]
        eligible_inputs += 1
        local = object_counts.setdefault(image["uid"], {
            name: {"tp": 0, "fp": 0} for name in counts})
        localized = true_positive(candidates.get(parent_id), threshold, iou_threshold)
        for prompt, input_score in input_scores.items():
            if input_score < threshold:
                continue
            if prompt == image["category"] and not localized:
                continue
            label = "tp" if prompt == image["category"] else "fp"
            counts["input"][label] += 1
            local["input"][label] += 1
            if identity["scores"][prompt] >= threshold:
                counts["identity"][label] += 1
                local["identity"][label] += 1
            if oracle["scores"][prompt] >= threshold:
                counts["oracle"][label] += 1
                local["oracle"][label] += 1
            for method in METHODS:
                if verify_claim(prompt, input_score, complete_views, method=method,
                                score_threshold=threshold):
                    counts[method][label] += 1
                    local[method][label] += 1
    base = counts["input"]
    summary = {}
    negative_prompts = eligible_inputs * (len(vocabulary) - 1)
    for method, result in counts.items():
        total = result["tp"] + result["fp"]
        false_negatives = eligible_inputs - result["tp"]
        true_negatives = negative_prompts - result["fp"]
        summary[method] = {**result,
                           "tn": true_negatives, "fn": false_negatives,
                           "tn_rate": true_negatives / negative_prompts if negative_prompts else None,
                           "fp_rate": result["fp"] / negative_prompts if negative_prompts else None,
                           "f1": 2 * result["tp"] / (2 * result["tp"] + result["fp"] + false_negatives)
                           if eligible_inputs else None,
                           "precision": result["tp"] / total if total else None,
                           "fp_rejection": 1 - result["fp"] / base["fp"] if base["fp"] else None,
                           "tp_retention": result["tp"] / base["tp"] if base["tp"] else None}
    objects = list(object_counts.values())
    rng = np.random.default_rng(0)
    paired = {"tp_retention_change": [], "fp_rejection_change": []}
    for _ in range(2000 if objects else 0):
        sample = [objects[index] for index in rng.integers(0, len(objects), len(objects))]
        for label, field, sign in (("tp", "tp_retention_change", 1),
                                   ("fp", "fp_rejection_change", -1)):
            denominator = sum(obj["input"][label] for obj in sample)
            change = sum(obj["peak_margin"][label] - obj["identity"][label]
                         for obj in sample)
            paired[field].append(sign * change / denominator if denominator else float("nan"))
    comparison = {
        "tp_retention_change": ((counts["peak_margin"]["tp"] - counts["identity"]["tp"])
                                / base["tp"] if base["tp"] else None),
        "fp_rejection_change": ((counts["identity"]["fp"] - counts["peak_margin"]["fp"])
                                / base["fp"] if base["fp"] else None),
        "object_bootstrap_ci95": {
            field: ([float(x) for x in np.percentile(finite, [2.5, 97.5])]
                    if (finite := [value for value in values if math.isfinite(value)]) else None)
            for field, values in paired.items()},
    }
    study_metrics = summarize(outcomes_by_input, object_ids, "relative_margin")
    return {"eligible_inputs": eligible_inputs, "objects": len(objects),
            "vocabulary_size": len(vocabulary),
            "detection": detection,
            "view_policy": "all rendered views, including the reconstructed identity view",
            "methods": summary, "peak_margin_vs_identity": comparison,
            "study_metrics": study_metrics}


def compare_with_existing(result, output_dir):
    """Append verification rules and prompt-level F1 to study selector tables."""
    path = Path(output_dir) / "summary.json"
    fusion = METHODS
    current = result["study_metrics"]
    if path.is_file():
        previous = json.loads(path.read_text())
        if (previous["inputs"] != current["inputs"]
                or previous["objects"] != current["objects"]
                or previous["detection"] != result["detection"]
                or set(previous["subsets"]) != set(current["subsets"])):
            raise ValueError("Existing study summary does not match this verification run; "
                             "rerun the analyze stage")
        names = [name for name in previous["selectors"] if name != "oracle"]
        names += [*fusion, "oracle"]
        source = "study summary plus verification rules"
    else:
        previous = None
        names = ["input", "identity", *fusion, "oracle"]
        source = "verification rules and references only; run analyze for other selectors"
    subsets = {}
    for subset, data in current["subsets"].items():
        fusion_table = data["selectors"]
        if previous is None:
            table = {name: fusion_table[name] for name in names}
        else:
            old_table = previous["subsets"][subset]["selectors"]
            if (next(iter(old_table.values()))["n"]
                    != next(iter(fusion_table.values()))["n"]):
                raise ValueError(f"Existing study summary subset differs: {subset}")
            table = {**old_table, **{name: fusion_table[name] for name in fusion}}
            table = {name: table[name] for name in names}
        enriched = {}
        for name, row in table.items():
            recall = row["tp"]
            fp_per_input = row["fp_rate"] * (result["vocabulary_size"] - 1)
            denominator = recall + fp_per_input
            negative_prompts = row["n"] * (result["vocabulary_size"] - 1)
            fp_count = row["fp_rate"] * negative_prompts
            enriched[name] = {**row,
                              "tn_rate": 1 - row["fp_rate"],
                              "tn_count": negative_prompts - fp_count,
                              "fp_count": fp_count,
                              "prompt_precision": recall / denominator if denominator else None,
                              "f1": 2 * recall / (1 + recall + fp_per_input)}
        subsets[subset] = {"selectors": enriched}
    return {"source": source, "selectors": names, "subsets": subsets}


def save_plots(result, output_dir):
    """Plot the fusion rules beside all available viewpoint study selectors."""
    from study.figures import save_bars, save_margin_comparison

    figures = Path(output_dir) / "figures" / "fp_verification"
    comparison = result.get("comparison_metrics") or compare_with_existing(result, output_dir)
    config_path = Path(output_dir) / "config.json"
    figure_selectors = (json.loads(config_path.read_text()).get("analysis", {}).get("figure_selectors")
                        if config_path.is_file() else None)
    paths = []
    for subset, data in comparison["subsets"].items():
        table = {name: row for name, row in data["selectors"].items()
                 if figure_selectors is None or name in figure_selectors or name in METHODS}
        slug = subset.split(" (")[0].replace(" ", "_")
        title = f"FP verification ({subset}, n={next(iter(table.values()))['n']})"
        paths.append(save_bars(
            {name: {"top-1 accuracy": row["correct"],
                    "any false positive": row["any_fp"]} for name, row in table.items()},
            figures / f"summary_{slug}.png", title=title, ylabel="fraction of inputs"))
        paths.append(save_margin_comparison(
            table, figures / f"margins_{slug}.png",
            title=f"Confidence margin ({subset})", primary="relative_margin",
            baseline=data["selectors"]["input"]["margin"]))
        paths.append(save_bars(
            {name: {"FP rate": row["fp_rate"], "TP recall": row["tp"]}
             for name, row in table.items()},
            figures / f"detection_{slug}.png", title=title,
            ylabel="fraction of prompts / inputs"))
        paths.append(save_bars(
            {name: {"inputs with >=1 FP": row["any_fp"], "TP recall": row["tp"]}
             for name, row in table.items()},
            figures / f"detection_any_fp_{slug}.png", title=title,
            ylabel="fraction of inputs"))
        paths.append(save_bars(
            {name: {"micro-F1": row["f1"]} for name, row in table.items()},
            figures / f"f1_{slug}.png", title=f"Prompt-level F1 ({subset})",
            ylabel="micro-F1"))
    return paths


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True, help="study YAML")
    args = parser.parse_args()
    config = settings_io.load(args.config)
    output = Path(config["output_dir"])
    db = store.connect(output)
    try:
        result = evaluate(db, config)
    finally:
        db.close()
    result["comparison_metrics"] = compare_with_existing(result, output)
    path = output / "fp_verification.json"
    path.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    plots = save_plots(result, output)
    print(f"Evaluated {result['eligible_inputs']} reconstructed inputs -> {path}; "
          f"{len(plots)} plots -> {plots[0].parent}")


if __name__ == "__main__":
    main()
