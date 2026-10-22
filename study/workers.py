"""SAM3D reconstruction, ambiguity scoring, and SAM3 detection workers.

Each stage runs in its configured model environment and resumes stored work.
"""
import argparse
import multiprocessing
import time
from pathlib import Path

import numpy as np
from PIL import Image

from tools.envs import use_gpu
from study import config as settings_io
from study import db as store

PROGRESS_EVERY = 100


def inputs(db, config):
    """Visible ground-truth images of the first N objects per category, in order."""
    recon = config["reconstruction"]
    rows = store.images(db, "source='gt' AND visible_px > 0")
    objects = {}
    for row in rows:
        members = objects.setdefault(row["category"], [])
        if row["uid"] not in members:
            members.append(row["uid"])
    keep = {uid for members in objects.values() for uid in members[:recon["objects_per_category"]]}
    elevations = recon["input_elevations"]
    return [row for row in rows if row["uid"] in keep
            and (elevations is None or row["elevation"] in elevations)]


def directory_for(image):
    return f"renders/sam3d/{image['uid']}/a{image['azimuth']:g}_e{image['elevation']:g}"


def _score(task):
    """Worker: all metrics for one reconstruction's candidates; returns (directory, rows, error)."""
    from study.ambiguity import score_candidates

    output, directory, candidates, metrics, distance = task
    output = Path(output)
    views = [{"azimuth": c["azimuth"], "elevation": c["elevation"],
              "image": output / c["path"], "mask": output / c["mask_path"]} for c in candidates]
    try:
        values = score_candidates(output / directory, views, metrics, distance, (0.0, 0.0, 0.0))
    except Exception as error:  # One bad mesh must not stop the stage.
        return directory, [], f"{type(error).__name__}: {error}"
    rows = [(candidate["image_id"], metric, score)
            for candidate, value in zip(candidates, values) for metric, score in value.items()]
    return directory, rows, None


def reconstruct(config, limit=None):
    use_gpu(config["gpu_id"])
    from reconstruction.sam3d import SAM3D

    output = Path(config["output_dir"])
    recon = config["reconstruction"]
    db = store.connect(output)
    done = {row[0] for row in db.execute("SELECT input_id FROM reconstructions WHERE status='ok'")}
    pending = [image for image in inputs(db, config) if image["image_id"] not in done]
    if limit is not None:
        pending = pending[:limit]
    print(f"[reconstruct] {len(done)} done, {len(pending)} to reconstruct", flush=True)
    if not pending:
        return
    model = SAM3D(checkpoint_dir=config["checkpoints"]["sam3d"])
    for count, image in enumerate(pending, 1):
        relative = directory_for(image)
        started = time.time()
        try:
            candidates = model.render_views(
                output / image["path"], output / image["mask_path"], output / relative,
                recon["candidate_azimuths"], recon["candidate_elevations"],
                input_elevation=image["elevation"], distance=recon["candidate_distance"],
                resolution=recon["resolution"], seed=recon["seed"], ensemble=recon["ensemble"])
            if not (output / relative / "mesh.npz").is_file():
                raise RuntimeError("SAM3D returned no mesh; mesh ambiguity metrics need one")
        except Exception as error:  # Recorded per input; the remaining inputs continue.
            with db:
                db.execute("INSERT OR REPLACE INTO reconstructions VALUES (?, 'error', ?, ?, ?)",
                           (image["image_id"], relative, time.time() - started,
                            f"{type(error).__name__}: {error}"))
            print(f"[reconstruct] {count}/{len(pending)} {image['key']} failed: {error}", flush=True)
            continue
        with db:
            for candidate in candidates:
                with Image.open(output / relative / "views" / candidate["mask"]) as mask:
                    visible = int((np.asarray(mask) > 0).sum())
                store.add_image(
                    db, key=f"sam3d/{image['image_id']}/a{candidate['azimuth']:g}_e{candidate['elevation']:g}",
                    source="sam3d", parent_id=image["image_id"], uid=image["uid"],
                    category=image["category"], azimuth=candidate["azimuth"],
                    elevation=candidate["elevation"],
                    path=f"{relative}/views/{candidate['image']}",
                    mask_path=f"{relative}/views/{candidate['mask']}", visible_px=visible)
            db.execute("INSERT OR REPLACE INTO reconstructions VALUES (?, 'ok', ?, ?, NULL)",
                       (image["image_id"], relative, time.time() - started))
        print(f"[reconstruct] {count}/{len(pending)} {image['key']} "
              f"{time.time() - started:.1f}s", flush=True)
    db.close()


def score(config, workers=4):
    """Ambiguity metrics for every SAM3D candidate, on the reconstructed mesh."""
    output = Path(config["output_dir"])
    metrics = config["ambiguity"]
    wanted = len(metrics)
    db = store.connect(output)
    tasks = []
    distance = config["reconstruction"]["candidate_distance"]
    for input_id, directory in db.execute(
            "SELECT input_id, directory FROM reconstructions WHERE status='ok' ORDER BY input_id"):
        candidates = store.images(db, "parent_id=?", (input_id,))
        scored = db.execute(
            "SELECT COUNT(*) FROM view_scores s JOIN images i USING (image_id)"
            " WHERE i.parent_id=?", (input_id,)).fetchone()[0]
        if scored < wanted * len(candidates):
            tasks.append((str(output), directory, candidates, metrics, distance))
    print(f"[score] {len(tasks)} reconstructions to score with {sorted(metrics)}", flush=True)
    if not tasks:
        return
    failures = 0
    context = multiprocessing.get_context("spawn")
    with context.Pool(max(1, min(workers, len(tasks)))) as pool:
        for done, (directory, rows, error) in enumerate(pool.imap_unordered(_score, tasks), 1):
            if error:
                failures += 1
                print(f"[score] {directory} failed: {error}", flush=True)
            else:
                store.add_scores(db, rows)
            if done % 50 == 0 or done == len(tasks):
                print(f"[score] {done}/{len(tasks)} ({failures} failed)", flush=True)
    db.close()


def pick(config):
    """Pre-detect picks for every reconstructed input (``picks`` config): the learned
    ViewActive ``vqf`` candidate, the ``identity`` candidate and one seeded ``random`` one."""
    settings = config.get("picks")
    if settings is None:
        print("[pick] no picks configured", flush=True)
        return
    use_gpu(config["gpu_id"])
    from perception.select import vqf_pick
    from tools.cameras import SAM3D_INPUT_VIEW
    from viewactive.grid import sam3d_to_viewactive
    from viewactive.model import ViewActive

    output = Path(config["output_dir"])
    db = store.connect(output)
    done = {row[0] for row in db.execute("SELECT DISTINCT input_id FROM picks")}
    pending = [image for image in store.images(db, "i.image_id IN (SELECT input_id FROM reconstructions"
                                                " WHERE status='ok')") if image["image_id"] not in done]
    print(f"[pick] {len(pending)} inputs to pick", flush=True)
    if not pending:
        return
    model = ViewActive.load(settings["viewactive"])
    for count, image in enumerate(pending, 1):
        candidates = store.images(db, "i.parent_id=?", (image["image_id"],))
        with Image.open(output / image["path"]) as rgb, Image.open(output / image["mask_path"]) as mask:
            field = model.predict([rgb.convert("RGB")], [mask.convert("L")])[0]
        angles = [sam3d_to_viewactive(c["azimuth"], c["elevation"], image["elevation"]) for c in candidates]
        quality = model.score_views(field, *zip(*angles))
        best = vqf_pick({n: (float(q), e) for n, (q, (_, e)) in enumerate(zip(quality, angles))},
                        settings["elevation_range"])
        rng = np.random.default_rng([settings["seed"], image["image_id"]])
        chosen = {"vqf": (candidates[best], {"quality": float(quality[best]), "absolute": list(angles[best])}),
                  "random": (candidates[int(rng.integers(len(candidates)))], {})}
        chosen.update({"identity": (c, {}) for c in candidates
                       if (c["azimuth"], c["elevation"]) == SAM3D_INPUT_VIEW})
        with db:
            for selector, (candidate, meta) in chosen.items():
                db.execute("INSERT OR REPLACE INTO picks VALUES (?, ?, ?, ?)",
                           (image["image_id"], selector, candidate["image_id"], store.pack(meta)))
        if count % PROGRESS_EVERY == 0 or count == len(pending):
            print(f"[pick] {count}/{len(pending)}", flush=True)
    db.close()


def detect(config, limit=None):
    use_gpu(config["gpu_id"])

    from reconstruction.segmentation import SAM3Preprocessor
    from study.labels import mask_iou, prompt_text

    output = Path(config["output_dir"])
    vocabulary = config["dataset"]["categories"]
    # Detections stay keyed by category; SAM3 sees the cleaned prompt text.
    prompts = {category: prompt_text(category) for category in vocabulary}
    if len(set(prompts.values())) != len(prompts):
        raise ValueError(f"categories share a SAM3 prompt: {prompts}")
    db = store.connect(output)
    condition = "image_id NOT IN (SELECT image_id FROM detections)"
    if (config.get("picks") or {}).get("detect_only"):
        condition += " AND (i.source='gt' OR i.image_id IN (SELECT pick_id FROM picks))"
    if limit is not None:
        # A bounded pilot should cover one reconstructed input and its views,
        # rather than spend its entire budget on the GT rows imported first.
        condition += (" AND (i.parent_id IN (SELECT input_id FROM reconstructions"
                      " WHERE status='ok') OR i.image_id IN"
                      " (SELECT input_id FROM reconstructions WHERE status='ok'))")
    pending = store.images(db, condition)
    if limit is not None:
        pending = pending[:limit]
    print(f"[detect] {len(pending)} images without detections", flush=True)
    if not pending:
        return
    # threshold=0 keeps every candidate; thresholds are applied at analysis time.
    model = SAM3Preprocessor(checkpoint=config["checkpoints"]["sam3"], threshold=0.0)
    for done, image in enumerate(pending, 1):
        with Image.open(output / image["path"]) as source:
            rgb = source.convert("RGB")
        category = image["category"]
        found = model.detect_many(rgb, list(prompts.values()), spatial_queries={prompts[category]})
        found = {name: found[prompts[name]] for name in vocabulary}
        rows = {}
        for prompt, detection in found.items():
            scores = detection["scores"][np.isfinite(detection["scores"])]
            rows[prompt] = (float(scores.max()) if scores.size else 0.0, len(scores), None)
        true = found[category]
        if image["mask_path"]:
            with Image.open(output / image["mask_path"]) as source:
                mask = np.asarray(source.convert("L")) > 127
            ious = [mask_iou(candidate, mask) for candidate in true["masks"]]
        else:
            ious = [0.0] * len(true["scores"])
        rows[category] = rows[category][:2] + (
            {"scores": [float(s) for s in true["scores"]], "ious": ious},)
        store.add_detections(db, image["image_id"], rows)
        if done % PROGRESS_EVERY == 0 or done == len(pending):
            print(f"[detect] {done}/{len(pending)}", flush=True)
    db.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", required=True, choices=("reconstruct", "score", "pick", "detect"))
    parser.add_argument("--settings", type=Path, required=True)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    config = settings_io.read(args.settings)
    if args.stage == "reconstruct":
        reconstruct(config, args.limit)
    elif args.stage == "score":
        score(config, args.workers)
    elif args.stage == "pick":
        pick(config)
    else:
        detect(config, args.limit)


if __name__ == "__main__":
    main()
