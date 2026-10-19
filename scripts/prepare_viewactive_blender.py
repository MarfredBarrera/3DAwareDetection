"""Reuse selected objects and GLBs for an independent Blender render run.

The new run keeps its own renders, labels, features, classifiers, and model.
Only cohort metadata and immutable GLBs are shared with the corrected split.
"""
import argparse
import json
import os
from pathlib import Path

from study.config import load_viewactive


def prepare(source, split, config):
    source, split = Path(source).resolve(), Path(split).resolve()
    target = Path(config["output_dir"])
    if config["render"]["backend"] != "blender":
        raise ValueError("target config must select render.backend: blender")
    expected = json.loads((split / "objects.json").read_text())
    selected = json.loads((source / "objects.json").read_text())
    if (set(expected) != set(selected) or
            any(expected[uid]["category"] != selected[uid]["category"] for uid in expected)):
        raise ValueError("corrected split has different object IDs or categories")
    split_config = json.loads((split / "config.json").read_text())
    if split_config["cohort"] != config["cohort"] or split_config["objaverse"] != config["objaverse"]:
        raise ValueError("target cohort/Objaverse settings differ from the corrected split")
    target.mkdir(parents=True, exist_ok=True)
    records = target / "objects.json"
    if records.exists() and json.loads(records.read_text()) != expected:
        raise ValueError(f"{records} has different objects")
    records.write_text(json.dumps(expected, indent=1) + "\n")
    for name in ("candidates.json", "download.json", "canonical.json", "store"):
        original = source / name
        link = target / name
        if not original.exists():
            raise FileNotFoundError(original)
        if link.is_symlink() and link.resolve() == original:
            continue
        if link.exists() or link.is_symlink():
            raise ValueError(f"{link} already exists and does not link to {original}")
        link.symlink_to(Path(os.path.relpath(original, link.parent)))
    return {split_name: sum(row["split"] == split_name for row in expected.values())
            for split_name in ("train", "val", "test", "test_unseen")}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    print(prepare(args.source, args.split, load_viewactive(args.config)))


if __name__ == "__main__":
    main()
