"""Run the viewpoint study: render -> reconstruct -> score -> detect -> analyze.

Run inside the container (check nvidia-smi and set gpu_id first):

    sam3d-env core python -m study --config configs/upright.yaml
    sam3d-env core python -m study --config configs/upright.yaml --stages analyze

Every stage only fills in what is missing, so rerunning the same command
resumes after an interruption. Stage logs are appended to <output_dir>/logs/.
"""
import argparse
import fcntl
import json
from pathlib import Path

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
    chosen = {}
    for uid in sorted(objects):
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


def render(config, log):
    output = Path(config["output_dir"])
    dataset = config["dataset"]
    groups = cohort(config)
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


STAGES = ("render", "reconstruct", "score", "detect", "analyze")
WORKERS = {"reconstruct": "sam3d", "score": "sam3d", "detect": "core", "analyze": "core"}


def check_fingerprints(db, config, stages):
    for stage in set(stages) & set(settings_io.FINGERPRINTS):
        value = settings_io.fingerprint(config, stage)
        key = f"fingerprint:{stage}"
        stored = store.get_meta(db, key)
        if stored is None or stored == value:
            store.set_meta(db, key, value)
        elif stage == "score":
            # Ambiguity scores are cheap to recompute; everything else is not.
            print("[study] ambiguity settings changed; clearing stored scores", flush=True)
            with db:
                db.execute("DELETE FROM view_scores")
            store.set_meta(db, key, value)
        else:
            raise ValueError(f"{stage} settings changed since this study was created; "
                             "use a new output_dir (stored results would be inconsistent)")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--stages", nargs="+", choices=STAGES, default=list(STAGES))
    parser.add_argument("--limit", type=int, help="reconstruct at most this many new inputs")
    parser.add_argument("--workers", type=int, default=4, help="score-stage processes")
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
            if stage == "reconstruct" and args.limit is not None:
                extra = ["--limit", args.limit]
            elif stage == "score":
                extra = ["--workers", args.workers]
            elif stage == "analyze" and args.no_object_figures:
                extra = ["--no-object-figures"]
            module = "study.analyze" if stage == "analyze" else "study.workers"
            worker_args = (["--stage", stage] if stage != "analyze" else [])
            run_in_env(WORKERS[stage], module, [*worker_args, "--settings", settings, *extra],
                       gpu_id=config["gpu_id"], log=log)
    print(f"[study] done -> {output}", flush=True)


if __name__ == "__main__":
    main()
