"""Report the classifier ceiling by LVIS category for a ViewActive run."""
import argparse
import json
from pathlib import Path

import numpy as np

from viewactive.classify import correct
from viewactive.cohort import load_class_map


def audit(run_dir, class_map):
    run_dir = Path(run_dir)
    with np.load(run_dir / "vqf.npz") as data:
        uids = data["uids"]
        categories = data["categories"]
        splits = data["splits"]
        quality = np.nanmean(data["norm"], axis=1)
    index = {uid: i for i, uid in enumerate(uids)}
    mapped = load_class_map(class_map)
    rows = {}
    for path in sorted((run_dir / "classifier").glob("*.npz")):
        with np.load(path) as data:
            for uid, top1 in zip(data["uids"], data["top1"]):
                i = index[uid]
                category = str(categories[i])
                classification = correct(top1, category, mapped)
                best = int(np.nanargmax(quality[i]))
                row = rows.setdefault(str(splits[i]), {}).setdefault(category, {}).setdefault(
                    path.stem, {"all_views": [], "best_quality_view": [], "any_view": []})
                row["all_views"].append(float(classification.mean()))
                row["best_quality_view"].append(float(classification.flat[best]))
                row["any_view"].append(float(classification.any()))
    return {split: {category: {model: {"objects": len(values["any_view"]),
                                       **{key: float(np.mean(value)) for key, value in values.items()}}
                               for model, values in models.items()}
                    for category, models in classes.items()}
            for split, classes in rows.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--class-map", type=Path, default=Path("configs/viewactive_imagenet_map.yaml"))
    args = parser.parse_args()
    report = audit(args.run_dir, args.class_map)
    path = args.run_dir / "eval" / "class_audit.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2) + "\n")
    for split, classes in report.items():
        print(split)
        for category, models in sorted(classes.items(), key=lambda item: np.mean(
                [m["any_view"] for m in item[1].values()])):
            print(category, "objects", next(iter(models.values()))["objects"],
                  "any_view", round(float(np.mean([m["any_view"] for m in models.values()])), 3),
                  "all_views", round(float(np.mean([m["all_views"] for m in models.values()])), 3))


if __name__ == "__main__":
    main()
