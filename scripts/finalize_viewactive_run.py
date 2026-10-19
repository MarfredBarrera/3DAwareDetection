"""Write the fixed paper comparison and per-class recognition ceiling."""
import argparse
import csv
import json
import os
from pathlib import Path

from scripts.audit_viewactive_classes import audit
from scripts.compare_viewactive_paper import compare
from scripts.compare_viewactive_renderers import compare as compare_renderers


def normalize_blender_indexes(run_dir):
    config = json.loads((run_dir / "config.json").read_text())
    if config["render"].get("backend", "pytorch3d") != "blender":
        return
    from harness.upright import verify_cameras
    from viewactive.blender import _camera

    for index in sorted((run_dir / "renders").glob("shard-*/index.jsonl")):
        pending = index.with_suffix(".jsonl.tmp")
        with index.open() as source, pending.open("w") as target:
            for line in source:
                row = json.loads(line)
                row["R"], row["T"] = _camera(row["azimuth"], row["elevation"], row["distance"])
                row["camera_convention"] = "blender_local_minus_z"
                target.write(json.dumps(row) + "\n")
        os.replace(pending, index)
        number = int(index.parent.name.split("-")[-1])
        verify_cameras(index, run_dir / "logs" / f"camera-verification-{number:03d}.json")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--class-map", type=Path, default=Path("configs/viewactive_imagenet_map.yaml"))
    parser.add_argument("--source", type=Path, help="compare paired classifier views with this run")
    args = parser.parse_args()
    output = args.run_dir / "eval"
    output.mkdir(parents=True, exist_ok=True)
    normalize_blender_indexes(args.run_dir)
    rows = compare(output / "fig4.json")
    fields = ("split", "metric", "paper_step0", "measured_step0", "paper_step9",
              "measured_step9", "step9_gap", "measured_gain")
    with (output / "paper_comparison.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    (output / "class_audit.json").write_text(
        json.dumps(audit(args.run_dir, args.class_map), indent=2) + "\n")
    if args.source:
        (output / "renderer_comparison.json").write_text(
            json.dumps(compare_renderers(args.source, args.run_dir, args.class_map), indent=2) + "\n")
    print(f"Finalized {args.run_dir}")


if __name__ == "__main__":
    main()
