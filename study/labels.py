"""Turn stored SAM3 prompt scores into recognition outcomes. Pure NumPy.

SAM3 is a prompted detector. Each vocabulary prompt's maximum candidate
confidence is its category score; the argmax is the closed-vocabulary top-1
prediction. The primary measure is the confidence margin:

    margin = true-category score - strongest wrong-category score

Negative margins are misleading views. Thresholded detection outcomes are
prompt-level: the true prompt is TP if a candidate passes both the score and
mask-IoU thresholds (else FN); each wrong prompt is FP if its score passes the
score threshold (else TN). Invisible targets are skipped.
"""
import math
import re

import numpy as np


def prompt_text(category):
    """SAM3 text for an LVIS-style category name: first synonym, no
    disambiguating parenthetical, spaces for underscores ("car_(automobile)" -> "car")."""
    name = re.sub(r"_?\(.*?\)", "", category.split(" ")[0])
    return name.replace("_", " ").strip()


def mask_iou(a, b):
    """IoU of two boolean masks; 0.0 when the union is empty."""
    a, b = np.asarray(a, dtype=bool), np.asarray(b, dtype=bool)
    union = int(np.count_nonzero(a | b))
    return int(np.count_nonzero(a & b)) / union if union else 0.0


def top1(scores, vocabulary):
    """Highest-scoring prompt; vocabulary order breaks ties deterministically."""
    if set(scores) != set(vocabulary):
        raise ValueError("scores must cover exactly the vocabulary")
    return max(vocabulary, key=lambda prompt: (scores[prompt], -vocabulary.index(prompt)))


def true_positive(candidates, score_threshold, iou_threshold):
    """Whether any true-prompt candidate is both confident and correctly localized."""
    if not candidates:
        return False
    scores = np.asarray(candidates["scores"], dtype=float)
    ious = np.asarray(candidates["ious"], dtype=float)
    if scores.shape != ious.shape:
        raise ValueError("candidate scores and ious must have the same length")
    return bool(np.any((scores >= score_threshold) & (ious >= iou_threshold)))


def outcome(image, scores, candidates, vocabulary, detection):
    """Recognition outcome for one image, or None when the target is invisible."""
    if image["visible_px"] <= 0:
        return None
    category = image["category"]
    wrong = {prompt: score for prompt, score in scores.items() if prompt != category}
    wrong_prompt = max(vocabulary, key=lambda p: (wrong.get(p, -math.inf), -vocabulary.index(p)))
    prediction = top1(scores, vocabulary)
    false_positives = sum(score >= detection["score_threshold"] for score in wrong.values())
    return {
        "true_score": scores[category],
        "wrong_prompt": wrong_prompt,
        "wrong_score": wrong[wrong_prompt],
        "margin": scores[category] - wrong[wrong_prompt],
        "prediction": prediction,
        "correct": prediction == category,
        "tp": true_positive(candidates, detection["score_threshold"], detection["iou_threshold"]),
        "fp": false_positives,
        "any_fp": false_positives > 0,
    }


def outcomes(db, vocabulary, detection, where="1", params=()):
    """{image_id: outcome} for every selected image with complete detections."""
    from study import db as store

    rows = {row["image_id"]: row for row in store.images(db, where, params)}
    scores = store.prompt_scores(db, where, params)
    candidates = store.true_candidates(db, where, params)
    result = {}
    for image_id, image in rows.items():
        if set(scores.get(image_id, ())) != set(vocabulary):
            continue
        value = outcome(image, scores[image_id], candidates.get(image_id), vocabulary, detection)
        if value is not None:
            result[image_id] = value
    return result
