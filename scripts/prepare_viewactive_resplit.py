"""Reuse immutable ViewActive caches with a corrected object-level split.

Example::

    python -m scripts.prepare_viewactive_resplit \
      --source outputs/viewactive/full --config configs/viewactive_fidelity.yaml

The new run gets its own objects.json, VQF split labels, classifier results,
model, and evaluation. Large render and feature caches are linked read-only by
convention; the pipeline never modifies them after their stages complete.
"""
import argparse
import json
import os
from pathlib import Path

import numpy as np

from study.config import load_viewactive
from viewactive.cohort import finalize


def prepare(source, config):
    source = Path(source).resolve()
    target = Path(config["output_dir"])
    if source == target:
        raise ValueError("source and target output directories must differ")
    choices = json.loads((source / "candidates.json").read_text())
    canonical = json.loads((source / "canonical.json").read_text())["objects"]
    previous = json.loads((source / "objects.json").read_text())
    objects = finalize(config["cohort"], [tuple(pair) for pair in choices["pairs"]], canonical,
                       choices["seen"], choices["unseen"], config["objaverse"]["max_faces"],
                       drop_suspect=not config["objaverse"]["trust_file_up"])
    if set(objects) != set(previous) or any(objects[uid]["category"] != previous[uid]["category"]
                                             for uid in objects):
        raise ValueError("the new cohort changes object IDs or categories; caches cannot be reused")

    target.mkdir(parents=True, exist_ok=True)
    records = target / "objects.json"
    if records.exists() and json.loads(records.read_text()) != objects:
        raise ValueError(f"{records} already contains a different cohort")
    records.write_text(json.dumps(objects, indent=1) + "\n")
    for name in ("renders", "features", "canonical.json", "store"):
        link = target / name
        original = source / name
        if not original.exists():
            raise FileNotFoundError(original)
        if link.is_symlink() and link.resolve() == original:
            continue
        if link.exists() or link.is_symlink():
            raise ValueError(f"{link} already exists and does not link to {original}")
        link.symlink_to(Path(os.path.relpath(original, link.parent)))

    vqf = target / "vqf.npz"
    with np.load(source / "vqf.npz") as data:
        arrays = {key: data[key] for key in data.files}
    if set(arrays["uids"]) != set(objects):
        raise ValueError("VQF cache has different objects")
    arrays["splits"] = np.array([objects[uid]["split"] for uid in arrays["uids"]])
    if vqf.exists():
        with np.load(vqf) as data:
            if not np.array_equal(data["uids"], arrays["uids"]) or not np.array_equal(data["splits"], arrays["splits"]):
                raise ValueError(f"{vqf} already contains a different split")
    else:
        np.savez(vqf, **arrays)
    counts = {split: int((arrays["splits"] == split).sum())
              for split in ("train", "val", "test", "test_unseen")}
    (target / "resplit.json").write_text(json.dumps({"source": str(source), "counts": counts,
                                                      "reused": ["renders", "features", "labels"]}, indent=2) + "\n")
    return counts


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    print(prepare(args.source, load_viewactive(args.config)))


if __name__ == "__main__":
    main()
