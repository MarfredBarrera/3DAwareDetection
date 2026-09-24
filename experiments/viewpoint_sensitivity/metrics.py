"""Pure scoring and aggregation for the SAM3 viewpoint experiment.

The primary measure mirrors the paper: top-1 accuracy over a fixed vocabulary.
TP/TN/FP/FN remain secondary, threshold-dependent detection diagnostics.  This
module deliberately has no torch or plotting dependency.
"""
from __future__ import annotations

import math

import numpy as np


def mask_iou(a_bool, b_bool) -> float:
    """Intersection-over-union of two boolean masks; 0.0 when the union is empty."""
    a = np.asarray(a_bool, dtype=bool)
    b = np.asarray(b_bool, dtype=bool)
    union = int(np.count_nonzero(a | b))
    if union == 0:
        return 0.0
    intersection = int(np.count_nonzero(a & b))
    return intersection / union


def bbox_iou(box_a, box_b) -> float:
    """Intersection-over-union of two xyxy boxes; 0.0 when they do not overlap."""
    ax0, ay0, ax1, ay1 = (float(v) for v in box_a)
    bx0, by0, bx1, by1 = (float(v) for v in box_b)
    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    iw, ih = max(0.0, ix1 - ix0), max(0.0, iy1 - iy0)
    inter = iw * ih
    area_a = max(0.0, ax1 - ax0) * max(0.0, ay1 - ay0)
    area_b = max(0.0, bx1 - bx0) * max(0.0, by1 - by0)
    union = area_a + area_b - inter
    if union <= 0:
        return 0.0
    return inter / union


def classify_correct(candidate_scores, candidate_ious, tau_s, tau_iou, modal_area):
    """Label a correct-category query against the ground-truth modal mask.

    A view where the object is not visible (``modal_area == 0``) is SKIP: there
    is no target to hit. Otherwise a true positive needs a candidate that is both
    confident (score >= ``tau_s``) and correctly localized (IoU >= ``tau_iou``);
    a confident detection in the wrong place is a miss, not a hit, so it is FN.
    """
    if modal_area is None or modal_area <= 0:
        return "SKIP"
    scores = np.asarray(candidate_scores, dtype=float).reshape(-1)
    ious = np.asarray(candidate_ious, dtype=float).reshape(-1)
    if scores.size != ious.size:
        raise ValueError("candidate_scores and candidate_ious must be the same length")
    hit = np.any((scores >= tau_s) & (ious >= tau_iou)) if scores.size else False
    return "TP" if bool(hit) else "FN"


def classify_incorrect(best_score, tau_s):
    """Label an incorrect-category query: any confident detection anywhere is FP.

    The wrong category has no ground-truth object, so localization is irrelevant;
    a confident detection is a hallucination (FP) and its absence is a correct
    rejection (TN).
    """
    return "FP" if float(best_score) >= tau_s else "TN"


def _rate(numerator, denominator):
    """Fraction, or NaN when the denominator is empty (undefined, not zero)."""
    return numerator / denominator if denominator > 0 else math.nan


def top1_prediction(prompt_scores, category_order):
    """Return the highest-scoring category, its score, runner-up, and margin.

    SAM3 is a prompted detector rather than a classifier.  We turn it into the
    paper's closed-vocabulary recognizer by assigning each category its maximum
    candidate confidence and taking the argmax.  ``category_order`` provides a
    deterministic tie break.
    """
    categories = list(category_order)
    if not categories:
        raise ValueError("category_order must not be empty")
    if set(prompt_scores) != set(categories):
        raise ValueError("prompt_scores must contain exactly the evaluation vocabulary")
    ranked = sorted(((float(prompt_scores[c]), -i, c) for i, c in enumerate(categories)),
                    reverse=True)
    top_score, _, prediction = ranked[0]
    runner_up = ranked[1][0] if len(ranked) > 1 else math.nan
    margin = top_score - runner_up if len(ranked) > 1 else math.nan
    return prediction, top_score, runner_up, margin


def aggregate_viewpoints(rows):
    """Aggregate one row per rendered view into paper-style accuracy cells.

    Invisible renders are skipped.  Each visible view contributes exactly one
    top-1 correctness trial, regardless of vocabulary size.  Its binary
    confusion counts are summed separately and therefore must not be interpreted
    as a multiclass confusion matrix.
    """
    cells = {}
    for row in rows:
        key = (row["category"], float(row["azimuth"]), float(row["elevation"]))
        cell = cells.setdefault(key, {
            "correct": 0, "incorrect": 0, "SKIP": 0,
            "TP": 0, "TN": 0, "FP": 0, "FN": 0,
        })
        if row.get("top1_correct") is None:
            cell["SKIP"] += 1
            continue
        cell["correct" if row["top1_correct"] else "incorrect"] += 1
        confusion = row.get("binary_confusion", {})
        for label in ("TP", "TN", "FP", "FN"):
            cell[label] += int(confusion.get(label, 0))
    for cell in cells.values():
        cell["n"] = cell["correct"] + cell["incorrect"]
        cell["accuracy"] = _rate(cell["correct"], cell["n"])
        cell["recall"] = _rate(cell["TP"], cell["TP"] + cell["FN"])
        cell["fp_rate"] = _rate(cell["FP"], cell["FP"] + cell["TN"])
    return cells
