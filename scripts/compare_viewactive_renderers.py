"""Paired classifier comparison on matching Objaverse objects and camera views."""
import argparse
import json
from pathlib import Path

import numpy as np

from viewactive.classify import correct
from viewactive.cohort import load_class_map


def compare(source, target, class_map):
    source, target = Path(source), Path(target)
    objects = json.loads((target / "objects.json").read_text())
    mapping = load_class_map(class_map)
    report = {}
    for path in sorted((target / "classifier").glob("*.npz")):
        old_path = source / "classifier" / path.name
        if not old_path.is_file():
            continue
        with np.load(path) as new, np.load(old_path) as old:
            previous = dict(zip(old["uids"], old["top1"]))
            paired = {"seen": [], "unseen": []}
            for uid, top1 in zip(new["uids"], new["top1"]):
                if uid not in previous:
                    continue
                category = objects[uid]["category"]
                before = correct(previous[uid], category, mapping).mean()
                after = correct(top1, category, mapping).mean()
                paired["seen" if objects[uid]["split"] == "test" else "unseen"].append((before, after))
        report[path.stem] = {}
        for split, pairs in paired.items():
            if not pairs:
                continue
            values = np.asarray(pairs)
            delta = values[:, 1] - values[:, 0]
            samples = np.random.default_rng(0).choice(delta, (2000, len(delta))).mean(axis=1)
            report[path.stem][split] = {"objects": len(pairs), "source": float(values[:, 0].mean()),
                                        "blender": float(values[:, 1].mean()),
                                        "difference": float(delta.mean()),
                                        "difference_ci95": [float(x) for x in np.percentile(samples, (2.5, 97.5))]}
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("target", type=Path)
    parser.add_argument("--class-map", type=Path, default=Path("configs/viewactive_imagenet_map.yaml"))
    args = parser.parse_args()
    report = compare(args.source, args.target, args.class_map)
    path = args.target / "eval" / "renderer_comparison.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
