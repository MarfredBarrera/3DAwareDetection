"""Query every rendered view with the complete SAM3 category vocabulary.

Each output row is one recognition trial.  SAM3's maximum candidate confidence
for every category becomes that category's score, and argmax is the top-1 class.
This directly supports the paper-style accuracy heatmap.  Thresholded, localized
TP/TN/FP/FN counts are stored only as interpretable detection diagnostics.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from experiments.viewpoint_sensitivity import store

# Set before NumPy is imported (including indirectly by torch). Repeated large
# mask allocations otherwise request transparent huge pages. On fragmented
# shared hosts, synchronous compaction can stall this cgroup long enough for
# systemd-oomd to kill the whole container despite low resident memory.
# This affects allocation only, not SAM3 scores or experiment thresholds.
os.environ.setdefault("NUMPY_MADVISE_HUGEPAGE", "0")

PROGRESS_EVERY = 50


def _result_key(record):
    return (record["uid"], float(record["azimuth"]), float(record["elevation"]))


def configure_gpu(config):
    """Restrict this worker to one Compose-exposed GPU before CUDA is imported."""
    os.environ["CUDA_VISIBLE_DEVICES"] = str(config["gpu_id"])
    print(f"Using configured GPU {config['gpu_id']} (visible as cuda:0)", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True,
                        help="Resolved settings.json written by run.py")
    parser.add_argument("--restart", action="store_true",
                        help="clear saved trials and start the experiment database over")
    args = parser.parse_args()

    config = json.loads(Path(args.config).read_text())
    configure_gpu(config)

    import numpy as np
    from PIL import Image
    from reconstruction.segmentation import SAM3Preprocessor
    from experiments.viewpoint_sensitivity import metrics

    render_dir = Path(config["render_dir"])
    index_path = render_dir / "index.jsonl"
    if not index_path.is_file():
        raise FileNotFoundError(f"Missing render index: {index_path}. Run the render stage first.")
    output = Path(config["output_dir"])
    results_path = store.database_path(output)

    tau_s = float(config["detection"]["score_threshold"])
    tau_iou = float(config["detection"]["iou_threshold"])
    categories = config["categories"]
    db = store.connect(results_path)
    if args.restart:
        store.clear_trials(db)
    store.ensure_vocabulary(db, categories)
    store.ensure_query_config(db, config)
    completed = store.completed_keys(db)
    if completed:
        print(f"[query] resuming after {len(completed)} completed views", flush=True)

    # threshold=0.0 keeps every candidate; scoring applies tau_s ourselves so the
    # raw scores stay on record. max_masks is unused by detect_many.
    segmenter = SAM3Preprocessor(checkpoint=config["checkpoint"], threshold=0.0, max_masks=1)

    n_rows, n_views = len(completed), 0
    with open(index_path) as index:
        for line in index:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if _result_key(rec) in completed:
                continue
            rgb_path = render_dir / rec["files"]["rgb"]
            if not rgb_path.is_file():
                raise FileNotFoundError(f"render index references missing RGB image: {rgb_path}")
            with Image.open(rgb_path) as source:
                image = source.convert("RGB")
            category = rec["category"]
            modal_area = int(rec.get("modal_area", 0))

            common = dict(uid=rec["uid"], category=category,
                          azimuth=float(rec["azimuth"]), elevation=float(rec["elevation"]),
                          truncated=bool(rec.get("truncated", False)), modal_area=modal_area)

            # Encode the image once and score the full closed vocabulary.  This
            # is the SAM3 analogue of CLIP zero-shot classification in the paper.
            detections = segmenter.detect_many(
                image, categories, spatial_queries={category})
            prompt_scores = {}
            prompt_results = {}
            for prompt, detection in detections.items():
                finite = detection["scores"][np.isfinite(detection["scores"])]
                best_score = float(finite.max()) if finite.size else 0.0
                prompt_scores[prompt] = best_score
                prompt_results[prompt] = {
                    "best_score": best_score,
                    "candidate_count": int(detection["scores"].size),
                    "detected": bool(best_score >= tau_s),
                }
            prediction, top1_score, runner_up_score, margin = metrics.top1_prediction(
                prompt_scores, categories)

            # The ground-truth mask is used only for binary localization
            # diagnostics; top-1 recognition never gets access to it.
            correct = detections[category]
            scores = correct["scores"]
            if modal_area > 0:
                modal_path = render_dir / rec["files"]["modal"]
                with Image.open(modal_path) as gt:
                    gt_mask = np.asarray(gt.convert("L")) > 127
                ious = np.array([metrics.mask_iou(m, gt_mask) for m in correct["masks"]],
                                dtype=float)
                confident = scores >= tau_s
                best_iou = float(ious[confident].max()) if bool(confident.any()) else 0.0
                label = metrics.classify_correct(scores, ious, tau_s, tau_iou, modal_area)
            else:
                ious = np.empty(0, dtype=float)
                best_iou = 0.0
                label = metrics.classify_correct(scores, np.empty(0), tau_s, tau_iou, modal_area)
            prompt_results[category].update({
                "candidate_scores": [float(s) for s in scores.tolist()],
                "candidate_ious": [float(i) for i in ious.tolist()],
                "best_iou_above_threshold": best_iou,
                "binary_label": label,
            })

            confusion = {"TP": 0, "TN": 0, "FP": 0, "FN": 0}
            if modal_area > 0:
                confusion[label] += 1
                for prompt in categories:
                    if prompt == category:
                        continue
                    wrong_label = metrics.classify_incorrect(prompt_scores[prompt], tau_s)
                    confusion[wrong_label] += 1

            row = {
                **common,
                "schema_version": 2,
                "vocabulary": categories,
                "prompt_results": prompt_results,
                "prediction": prediction,
                "top1_score": top1_score,
                "runner_up_score": runner_up_score,
                "margin": margin,
                "top1_correct": (prediction == category) if modal_area > 0 else None,
                "binary_confusion": confusion,
            }
            # Each trial is committed before moving to the next view.
            store.insert_trial(db, row)
            del detections, correct, scores, ious, image
            n_rows += 1

            n_views += 1
            if n_views % PROGRESS_EVERY == 0:
                print(f"[query] {n_views} new views, {n_rows} total rows", flush=True)

    db.close()
    print(f"[query] wrote {results_path}: {n_views} new views, {n_rows} total rows", flush=True)


if __name__ == "__main__":
    main()
