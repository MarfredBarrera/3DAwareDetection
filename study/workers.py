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


_STORE = None


def _canonical_mesh(root, canonical, uid):
    """(vertices, faces) of a ground-truth object in the reviewed render frame."""
    global _STORE
    from harness.meshio import ObjaverseStore

    if _STORE is None or (_STORE.root, _STORE.canonical_path) != (root, canonical):
        _STORE = ObjaverseStore(root, canonical)
        _STORE.canonical_path = canonical
    mesh = _STORE.load(uid, device="cpu")
    if mesh is None:
        raise RuntimeError(f"unloadable mesh {uid}")
    return mesh.verts.numpy().astype(np.float64), mesh.faces.numpy().astype(np.int32)


def _score(task):
    """Worker: all metrics for one reconstruction, or for all ground-truth views of one
    object (``source = (root, canonical, uid)``); returns (label, rows, error)."""
    from study.ambiguity import score_candidates

    output, label, source, candidates, metrics, distance, target = task
    output = Path(output)
    views = [{"azimuth": c["azimuth"], "elevation": c["elevation"],
              "image": output / c["path"], "mask": output / c["mask_path"]} for c in candidates]
    try:
        mesh = _canonical_mesh(*source) if isinstance(source, tuple) else output / source
        values = score_candidates(mesh, views, metrics, distance, target)
    except Exception as error:  # One bad mesh must not stop the stage.
        return label, [], f"{type(error).__name__}: {error}"
    rows = [(candidate["image_id"], metric, score)
            for candidate, value in zip(candidates, values) for metric, score in value.items()]
    return label, rows, None


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
                resolution=recon["resolution"], seed=recon["seed"],
                view_grid=(recon["candidate_azimuths"], recon["candidate_elevations"]),
                save_ply=False, save_gaussians=recon["save_gaussians"],
                view_distance=recon["candidate_distance"],
                input_elevation=image["elevation"])
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
    """Ambiguity metrics for every SAM3D candidate (reconstructed mesh) and every
    visible ground-truth view (canonical mesh, harness camera)."""
    from tools.cameras import HARNESS_LOOK_AT

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
            tasks.append((str(output), directory, directory, candidates, metrics, distance, (0.0, 0.0, 0.0)))
    reconstructions = len(tasks)
    dataset = config["dataset"]
    scored = {row[0] for row in db.execute(
        "SELECT s.image_id FROM view_scores s JOIN images i USING (image_id) WHERE i.source='gt'"
        " GROUP BY s.image_id HAVING COUNT(*) >= ?", (wanted,))}
    views = {}
    for image in store.images(db, "source='gt' AND visible_px > 0"):
        views.setdefault(image["uid"], []).append(image)
    for uid, images in views.items():
        if any(image["image_id"] not in scored for image in images):
            tasks.append((str(output), f"gt/{uid}", (dataset["root"], dataset["canonical"], uid), images,
                          metrics, dataset["distance"], tuple(HARNESS_LOOK_AT)))
    print(f"[score] {reconstructions} reconstructions and {len(tasks) - reconstructions} ground-truth "
          f"objects to score with {sorted(metrics)}", flush=True)
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


def detect(config, limit=None):
    use_gpu(config["gpu_id"])

    from reconstruction.segmentation import SAM3Preprocessor
    from study.labels import mask_iou

    output = Path(config["output_dir"])
    vocabulary = config["dataset"]["categories"]
    db = store.connect(output)
    condition = "image_id NOT IN (SELECT image_id FROM detections)"
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
        found = model.detect_many(rgb, vocabulary, spatial_queries={category})
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
    parser.add_argument("--stage", required=True, choices=("reconstruct", "score", "detect"))
    parser.add_argument("--settings", type=Path, required=True)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    config = settings_io.read(args.settings)
    if args.stage == "reconstruct":
        reconstruct(config, args.limit)
    elif args.stage == "score":
        score(config, args.workers)
    else:
        detect(config, args.limit)


if __name__ == "__main__":
    main()
