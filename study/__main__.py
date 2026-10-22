"""Run the viewpoint study: render -> reconstruct -> score -> detect -> analyze.

Run inside the container (check nvidia-smi and set gpu_id first):

    sam3d-env core python -m study --config configs/upright.yaml
    sam3d-env core python -m study --config configs/upright.yaml --stages analyze

Every stage only fills in what is missing, so rerunning the same command
resumes after an interruption. Stage logs are appended to <output_dir>/logs/.
"""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path

import numpy as np

from tools.envs import run_in_env
from harness.upright import verify_cameras
from study import config as settings_io
from study import db as store

RENDER_DIR = "renders/gt"
MAX_FACES = 300_000  # Heavier meshes exceed the renderer's memory budget.


def cohort(config):
    """Deterministic object sample: the first N eligible uids per category."""
    dataset = config["dataset"]
    objects = json.loads(Path(dataset["canonical"]).read_text())["objects"]
    manifest = (json.loads(Path(dataset["manifest"]).read_text())
                if dataset["manifest"] is not None else None)
    if manifest is not None:
        if not isinstance(manifest, dict):
            raise ValueError("dataset.manifest must map UIDs to category and split")
        splits = dataset["split"] if isinstance(dataset["split"], list) else [dataset["split"]]
        selected = {uid: row for uid, row in manifest.items() if row.get("split") in splits}
        if not selected:
            raise ValueError(f"No objects in split {dataset['split']!r}")
        missing = set(selected) - set(objects)
        if missing:
            raise ValueError(f"Split objects missing from canonical index: {sorted(missing)[:4]}")
        mismatched = [uid for uid, row in selected.items()
                      if row["category"] != objects[uid]["category"]]
        if mismatched:
            raise ValueError(f"Split category differs from canonical index: {mismatched[:4]}")
    else:
        selected = objects
    chosen = {}
    for uid in sorted(selected):
        record = objects[uid]
        category = record["category"]
        if category not in dataset["categories"] or record["face_count"] > MAX_FACES:
            continue
        if (dataset["require_upright_verified"]
                and record.get("upright_verification", {}).get("status") != "verified"):
            continue
        members = chosen.setdefault(category, [])
        if len(members) < dataset["objects_per_category"]:
            members.append(uid)
    missing = set(dataset["categories"]) - set(chosen)
    if missing:
        raise ValueError(f"No eligible objects for categories {sorted(missing)}")
    return chosen


def register_external(db, output, rows):
    """Register completed external renders by reference, without copying images."""
    count = 0
    with db:
        for index, row in rows:
            shard = index.parent
            rgb = shard / row["files"]["rgb"]
            mask = shard / row["files"]["modal"]
            if not rgb.is_file() or not mask.is_file():
                continue
            store.add_image(
                db, key=gt_key(row["uid"], row["azimuth"], row["elevation"]),
                source="gt", uid=row["uid"], category=row["category"],
                azimuth=row["azimuth"], elevation=row["elevation"],
                path=os.path.relpath(rgb, output),
                mask_path=os.path.relpath(mask, output),
                visible_px=row["modal_area"],
                meta={key: row[key] for key in
                      ("distance", "fov", "image_size", "intrinsics", "R", "T",
                       "modal_bbox", "truncated") if key in row})
            count += 1
    return count


def object_views(uid, azimuths, elevations, count):
    """``count`` (azimuth, elevation) views of an object, fixed by a hash of its uid."""
    grid = sorted((a, e) for a in azimuths for e in elevations)
    rng = np.random.default_rng(int(hashlib.sha256(uid.encode()).hexdigest()[:16], 16))
    return {grid[i] for i in rng.choice(len(grid), size=min(count, len(grid)), replace=False)}


def sample_views(chosen, azimuths, elevations, count):
    """Keep the hash-chosen views of each object among {(uid, az, el): render}."""
    keep = {}
    for uid in {uid for uid, _, _ in chosen}:
        keep.update({(uid, a, e): chosen[(uid, a, e)]
                     for a, e in object_views(uid, azimuths, elevations, count)
                     if (uid, a, e) in chosen})
    return keep


def import_renders(config, groups):
    """Import available views from a running ViewActive render job incrementally."""
    output = Path(config["output_dir"])
    source_run = Path(config["dataset"]["source_run"])
    expected_uids = {uid for members in groups.values() for uid in members}
    azimuths = set(config["dataset"]["azimuths"])
    elevations = set(config["dataset"]["elevations"])
    chosen = {}
    for index in sorted((source_run / "renders").glob("shard-*/index.jsonl")):
        with index.open() as stream:
            for line in stream:
                row = json.loads(line)
                uid = row["uid"]
                if uid in expected_uids and row["azimuth"] in azimuths and row["elevation"] in elevations:
                    key = (uid, row["azimuth"], row["elevation"])
                    if key in chosen:
                        raise ValueError(f"Duplicate external render {key}")
                    if row["category"] not in groups or uid not in groups[row["category"]]:
                        raise ValueError(f"External render category differs from split: {uid}")
                    chosen[key] = (index, row)
    if config["dataset"]["views_per_object"] is not None:
        chosen = sample_views(chosen, azimuths, elevations, config["dataset"]["views_per_object"])
    with store.connect(output) as db:
        count = register_external(db, output, chosen.values())
    expected = len(expected_uids) * (config["dataset"]["views_per_object"]
                                     or len(azimuths) * len(elevations))
    print(f"[render] imported {count}/{expected} Blender views from {source_run}; "
          f"{len({uid for uid, _, _ in chosen})}/{len(expected_uids)} objects have started",
          flush=True)
    return count


def render(config, log):
    output = Path(config["output_dir"])
    dataset = config["dataset"]
    groups = cohort(config)
    if dataset["source_run"] is not None:
        import_renders(config, groups)
        return
    uids = sorted(uid for members in groups.values() for uid in members)
    index = output / RENDER_DIR / "index.jsonl"
    if not index.is_file():
        run_in_env("sam3d", "harness.build_dataset", [
            "--root", dataset["root"], "--canonical", dataset["canonical"],
            "--out", output / RENDER_DIR, "--uids", *uids,
            "--azimuths", *dataset["azimuths"], "--elevations", *dataset["elevations"],
            "--occlusion-levels", "0.0", "--distance", dataset["distance"],
            "--image-size", dataset["image_size"],
            *(["--require-upright-verified"] if dataset["require_upright_verified"] else []),
        ], gpu_id=config["gpu_id"], log=log)
    expected = {(uid, az, el) for uid in uids
                for az in dataset["azimuths"] for el in dataset["elevations"]}
    verify_cameras(index, output / "logs" / "camera-verification.json", expected)
    with store.connect(output) as db:
        count = register(db, index)
    print(f"[render] {count} ground-truth views of {len(uids)} objects: "
          + ", ".join(f"{c}={len(m)}" for c, m in sorted(groups.items())), flush=True)


def register(db, index):
    """Insert every unoccluded index row as a ``gt`` image; return the row count."""
    rows = [json.loads(line) for line in Path(index).read_text().splitlines() if line.strip()]
    rows = [row for row in rows if float(row.get("requested_occlusion", 0)) == 0]
    with db:
        for row in rows:
            add_gt_image(db, row)
    return len(rows)


def add_gt_image(db, row):
    keep = ("distance", "fov", "image_size", "intrinsics", "R", "T", "modal_bbox",
            "truncated", "semantic_front_verified")
    return store.add_image(
        db, key=gt_key(row["uid"], row["azimuth"], row["elevation"]), source="gt",
        uid=row["uid"], category=row["category"], azimuth=row["azimuth"],
        elevation=row["elevation"], path=f"{RENDER_DIR}/{row['files']['rgb']}",
        mask_path=f"{RENDER_DIR}/{row['files']['modal']}", visible_px=row["modal_area"],
        meta={key: row[key] for key in keep if key in row})


def gt_key(uid, azimuth, elevation):
    return f"gt/{uid}/a{float(azimuth):g}_e{float(elevation):g}"


STAGES = ("render", "reconstruct", "score", "pick", "detect", "analyze")
WORKERS = {"reconstruct": "sam3d", "score": "sam3d", "pick": "core", "detect": "core", "analyze": "core"}
CHEAP = {"score": "view_scores", "pick": "picks"}  # Stored results cleared when settings change.


def check_fingerprints(db, config, stages):
    for stage in set(stages) & set(settings_io.FINGERPRINTS):
        value = settings_io.fingerprint(config, stage)
        key = f"fingerprint:{stage}"
        stored = store.get_meta(db, key)
        if stored is None or stored == value:
            store.set_meta(db, key, value)
        elif stage in CHEAP:
            # Scores and picks are cheap to recompute; everything else is not.
            print(f"[study] {stage} settings changed; clearing {CHEAP[stage]}", flush=True)
            with db:
                db.execute(f"DELETE FROM {CHEAP[stage]}")
            store.set_meta(db, key, value)
        else:
            raise ValueError(f"{stage} settings changed since this study was created; "
                             "use a new output_dir (stored results would be inconsistent)")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--stages", nargs="+", choices=STAGES, default=list(STAGES))
    parser.add_argument("--limit", type=int, help="limit new reconstructions or detections in those reconstructions")
    parser.add_argument("--workers", type=int, default=4, help="processes for the score stage and analyze figures")
    parser.add_argument("--no-object-figures", action="store_true")
    args = parser.parse_args()
    config = settings_io.load(args.config)
    output = Path(config["output_dir"])
    (output / "logs").mkdir(parents=True, exist_ok=True)
    settings = output / "config.json"
    settings.write_text(json.dumps(config, indent=2) + "\n")
    with open(output / "logs" / "study.lock", "w") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise SystemExit(f"Another study driver is running on {output}") from None
        with store.connect(output) as db:
            store.ensure_meta(db, "vocabulary", config["dataset"]["categories"],
                              "Use a new output_dir for a different vocabulary.")
            check_fingerprints(db, config, args.stages)
        for stage in STAGES:
            if stage not in args.stages:
                continue
            log = output / "logs" / f"{stage}.log"
            print(f"[study] stage {stage}", flush=True)
            if stage == "render":
                render(config, log)
                continue
            extra = []
            if stage in ("reconstruct", "detect") and args.limit is not None:
                extra = ["--limit", args.limit]
            elif stage == "score":
                extra = ["--workers", args.workers]
            elif stage == "analyze":
                extra = ["--workers", args.workers] + (["--no-object-figures"] if args.no_object_figures else [])
            module = "study.analyze" if stage == "analyze" else "study.workers"
            worker_args = (["--stage", stage] if stage != "analyze" else [])
            run_in_env(WORKERS[stage], module, [*worker_args, "--settings", settings, *extra],
                       gpu_id=config["gpu_id"], log=log)
    print(f"[study] done -> {output}", flush=True)


if __name__ == "__main__":
    main()
