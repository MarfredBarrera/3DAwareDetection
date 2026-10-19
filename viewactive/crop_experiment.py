"""Compare full-frame and object-cropped ImageNet accuracy on cached ViewActive views.

Run in the sam3d container environment after the base ``classify`` stage::

    python -m viewactive.crop_experiment --config configs/viewactive_pilot.yaml

The cropped top-1 arrays are cached under ``eval/crop_experiment/``. This stage
does not alter the original classifier caches, renders, labels, or model.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from study.config import load_viewactive
from tools.envs import use_gpu
from viewactive.__main__ import Run, mask_images, rgb_images
from viewactive.classify import classify_views, correct
from viewactive.cohort import load_class_map
from viewactive.grid import combine


def score(top1, categories, class_map, quality):
    """Per-object accuracy on all views, the best-quality view, and any view."""
    accuracy = np.stack([correct(row, category, class_map)
                         for row, category in zip(top1, categories)])
    quality = combine(quality).reshape(len(quality), -1)
    best = np.argmax(np.where(np.isfinite(quality), quality, -np.inf), axis=1)
    flat = accuracy.reshape(len(accuracy), -1)
    return {"all_views": flat.mean(axis=1),
            "best_quality_view": flat[np.arange(len(flat)), best].astype(float),
            "any_view": flat.any(axis=1).astype(float)}


def summarize(full, cropped, splits):
    """Paired object-level comparison, separated into seen and unseen tests."""
    report = {}
    for split, name in (("test", "seen"), ("test_unseen", "unseen")):
        keep = splits == split
        if not keep.any():
            continue
        metrics = {}
        for metric in full:
            before, after = full[metric][keep], cropped[metric][keep]
            delta = after - before
            rng = np.random.default_rng(0)
            samples = delta[rng.integers(len(delta), size=(1000, len(delta)))].mean(axis=1)
            metrics[metric] = {"full": float(before.mean()), "cropped": float(after.mean()),
                               "difference": float(delta.mean()),
                               "difference_ci95": [float(x) for x in np.percentile(samples, [2.5, 97.5])]}
        report[name] = {"objects": int(keep.sum()), "metrics": metrics}
    return report


def experiment(run, models=("ResNet50", "ViT"), margin=0.15, device="cuda", log=print):
    data = run.labelled()
    evaluated = np.isin(data["splits"], ["test", "test_unseen"])
    uids = data["uids"][evaluated]
    categories = data["categories"][evaluated]
    splits = data["splits"][evaluated]
    quality = data["norm"][evaluated]
    images, masks = rgb_images(run, uids), mask_images(run, uids)
    class_map = load_class_map(run.config["cohort"]["class_map"])
    out = run.dir / "eval" / "crop_experiment"
    out.mkdir(parents=True, exist_ok=True)
    report = {"crop_margin": margin, "models": {}}
    for name in models:
        if name not in run.config["classifiers"]:
            raise ValueError(f"Unknown classifier {name!r}; choices: {list(run.config['classifiers'])}")
        with np.load(run.dir / "classifier" / f"{name}.npz") as cache:
            if not np.array_equal(cache["uids"], uids):
                raise ValueError(f"{name} full-frame classifier cache has different objects")
            full = cache["top1"]
        path = out / f"{name}_margin_{margin:g}.npz"
        if path.is_file():
            with np.load(path) as cache:
                if not np.array_equal(cache["uids"], uids):
                    raise ValueError(f"{name} cropped classifier cache has different objects")
                cropped = cache["top1"]
        else:
            cropped = classify_views(run.config["classifiers"][name], images, masks=masks,
                                     crop_margin=margin, device=device, log=log)
            np.savez(path, uids=uids, top1=cropped, model=run.config["classifiers"][name],
                     crop_margin=margin)
        report["models"][name] = summarize(score(full, categories, class_map, quality),
                                            score(cropped, categories, class_map, quality), splits)
        (out / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        log(f"[crop] {name} done")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--models", nargs="+", default=["ResNet50", "ViT"])
    parser.add_argument("--crop-margin", type=float, default=0.15)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    args = parser.parse_args()
    if not np.isfinite(args.crop_margin) or args.crop_margin < 0:
        parser.error("--crop-margin must be nonnegative and finite")
    config = load_viewactive(args.config)
    if args.device == "cuda":
        use_gpu(config["gpu_id"])
    report = experiment(Run(config), args.models, args.crop_margin, args.device)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
