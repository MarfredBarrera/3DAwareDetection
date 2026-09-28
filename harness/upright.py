"""Review object frames and build a separate, explicitly verified canonical file.

Review coordinates refer to the OLD canonical frame. A front direction is
optional for symmetric/ambiguous assets; upright verification is mandatory.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

UP_DIRECTIONS = {"+X": [1, 0, 0], "-X": [-1, 0, 0],
                 "+Y": [0, 1, 0], "-Y": [0, -1, 0],
                 "+Z": [0, 0, 1], "-Z": [0, 0, -1]}


def candidate_sheets(source, index, root, out):
    """Render six possible up directions for visual review; no classifier bias."""
    import torch
    from PIL import Image, ImageDraw
    from harness.meshio import ObjaverseStore, CanonicalMesh
    from harness.render import Harness
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    objects = {r["uid"]: r["category"] for r in read_index(index)}
    store = ObjaverseStore(root, source)
    harness = Harness(image_size=192, device="cuda")
    for category in sorted(set(objects.values())):
        uids = sorted(uid for uid, cat in objects.items() if cat == category)
        canvas = Image.new("RGB", (192 * 6, len(uids) * 220), "white")
        draw = ImageDraw.Draw(canvas)
        for i, uid in enumerate(uids):
            mesh = store.load(uid)
            if mesh is None:
                raise ValueError(f"Cannot load {uid}")
            draw.text((2, i * 220), f"{i}: {uid}", fill="black")
            for j, (label, up) in enumerate(UP_DIRECTIONS.items()):
                rotation = torch.tensor(frame_rotation(up), device="cuda", dtype=mesh.verts.dtype)
                vertices = mesh.verts @ rotation.T
                lo, hi = vertices.min(0).values, vertices.max(0).values
                shift = (lo + hi) / 2
                shift[1] = lo[1]
                candidate = CanonicalMesh(uid, category, vertices - shift, mesh.faces, mesh.vert_colors)
                result = harness.render(candidate, [], 2.2, 20, 45)
                pixels = (result["rgb"].cpu().numpy() * 255).clip(0, 255).astype("uint8")
                canvas.paste(Image.fromarray(pixels), (j * 192, i * 220 + 28))
                draw.text((j * 192 + 3, i * 220 + 14), f"old up {label}", fill="black")
        canvas.save(out / f"{category}.png")
        print(f"Candidates: {category}", flush=True)


def read_index(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def contact_sheets(render_dir, out, elevation=22.5):
    """Diagnostic plots: four cardinal views per object, retaining UID labels."""
    from PIL import Image, ImageDraw, ImageOps
    render_dir, out = Path(render_dir), Path(out)
    out.mkdir(parents=True, exist_ok=True)
    grouped = {}
    for row in read_index(render_dir / "index.jsonl"):
        if row["elevation"] == elevation and row["azimuth"] in (0, 90, 180, 270):
            grouped.setdefault(row["category"], {}).setdefault(row["uid"], {})[row["azimuth"]] = row
    if not grouped:
        raise ValueError(f"No cardinal views at elevation {elevation}")
    for category, objects in sorted(grouped.items()):
        canvas = Image.new("RGB", (4 * 160, len(objects) * 180), "white")
        draw = ImageDraw.Draw(canvas)
        for i, (uid, views) in enumerate(sorted(objects.items())):
            if len(views) != 4:
                raise ValueError(f"Incomplete diagnostic views for {uid}")
            for j, az in enumerate((0, 90, 180, 270)):
                with Image.open(render_dir / views[az]["files"]["rgb"]) as im:
                    tile = ImageOps.contain(im.convert("RGB"), (160, 154))
                canvas.paste(tile, (j * 160, i * 180 + 24))
                draw.text((j * 160 + 2, i * 180 + 12), f"az {az} el {elevation}", fill="black")
            draw.text((2, i * 180), f"{i}: {uid}", fill="black")
        canvas.save(out / f"{category}.png")
    (out / "objects.json").write_text(json.dumps(
        {cat: sorted(objects) for cat, objects in sorted(grouped.items())}, indent=2) + "\n")


def frame_rotation(up, front=None):
    """Proper rotation mapping reviewed up to +Y and front (if known) to +Z.

    +Z is the direction of the camera at azimuth=0 in our renderer.
    """
    up = np.asarray(up, dtype=float)
    if up.shape != (3,) or not np.isfinite(up).all() or np.linalg.norm(up) < 1e-8:
        raise ValueError("up must be a finite nonzero 3-vector")
    up /= np.linalg.norm(up)
    if front is None:
        front = [0, 0, 1] if abs(up[2]) < 0.9 else [1, 0, 0]
    front = np.asarray(front, dtype=float)
    if front.shape != (3,) or not np.isfinite(front).all():
        raise ValueError("front must be a finite 3-vector")
    front = front - np.dot(front, up) * up
    if np.linalg.norm(front) < 1e-8:
        raise ValueError("front must not be parallel to up")
    front /= np.linalg.norm(front)
    right = np.cross(up, front)
    return np.stack((right, up, front))


def corrected_record(record, vertices, review):
    """Compose review with the old raw-mesh transform; preserve physical scale."""
    rotation = frame_rotation(review["up"], review.get("front"))
    vertices = np.asarray(vertices) @ rotation.T
    lo, hi = vertices.min(0), vertices.max(0)
    shift = np.array([-(lo[0] + hi[0]) / 2, -lo[1], -(lo[2] + hi[2]) / 2])
    return {**record,
            "rotation": (rotation @ np.asarray(record["rotation"])).tolist(),
            "translation": (rotation @ np.asarray(record["translation"]) + shift).tolist(),
            "extents_canonical": (hi - lo).tolist(),
            "upright_verification": review,
            "semantic_front_verified": review.get("front") is not None}


def build_canonical(source, index, reviews, root, out):
    from harness.meshio import ObjaverseStore
    source, out = Path(source), Path(out)
    if out.resolve() == source.resolve():
        raise ValueError("Use a new canonical path; the source must be preserved")
    original = json.loads(source.read_text())
    decisions = json.loads(Path(reviews).read_text())["objects"]
    selected = {row["uid"]: row["category"] for row in read_index(index)}
    if set(decisions) != set(selected):
        raise ValueError("Reviews must cover exactly the selected object IDs")
    store = ObjaverseStore(root, str(source))
    records = {}
    for uid, category in sorted(selected.items()):
        review = decisions[uid]
        if review.get("status") == "excluded":
            continue
        if review.get("status") != "verified" or not review.get("evidence"):
            raise ValueError(f"Unverified object: {uid}")
        mesh = store.load(uid, device="cpu")
        if mesh is None:
            raise ValueError(f"Cannot load reviewed object {uid}")
        records[uid] = corrected_record(original["objects"][uid], mesh.verts.numpy(), review)
    payload = {**original, "objects": records,
               "convention": {**original["convention"], "forward": "+Z when reviewed; otherwise geometric",
                              "upright": "explicit per-object visual review"},
               "source_canonical": str(source), "orientation_reviews": str(reviews)}
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2) + "\n")
    write_review_report(decisions, records, out.with_name("orientation-report.json"))
    print(f"Wrote {len(records)} reviewed objects to {out}", flush=True)


def write_review_report(decisions, records, out):
    counts = {}
    for rec in records.values():
        counts[rec["category"]] = counts.get(rec["category"], 0) + 1
    report = {
        "review_method": "Codex visual inspection of six up candidates and four corrected cardinal views",
        "reviewed_objects": len(decisions), "retained_objects": len(records),
        "category_counts": counts,
        "up_corrections": sum(not np.allclose(r["up"], [0, 1, 0])
                              for r in decisions.values() if r["status"] == "verified"),
        "semantic_front_verified": sum(r["semantic_front_verified"] for r in records.values()),
        "excluded": {uid: r for uid, r in decisions.items() if r["status"] == "excluded"},
    }
    Path(out).write_text(json.dumps(report, indent=2) + "\n")


def summarize_reviews(canonical, out):
    payload = json.loads(Path(canonical).read_text())
    decisions = json.loads(Path(payload["orientation_reviews"]).read_text())["objects"]
    write_review_report(decisions, payload["objects"], out)


def verify_cameras(index, out, expected=None):
    """Check actual extrinsics: spherical position, proper rotation, no roll.

    ``expected`` optionally lists every (uid, azimuth, elevation) the index must contain.
    """
    from tools.cameras import HARNESS_LOOK_AT, direction
    rows = read_index(index)
    errors = []
    centers = []
    keys = [(r["uid"], r["azimuth"], r["elevation"]) for r in rows]
    if len(keys) != len(set(keys)):
        errors.append("duplicate object/viewpoint records")
    if expected is not None and set(keys) != set(expected):
        errors.append(f"cohort/grid mismatch: missing={len(set(expected) - set(keys))}, "
                      f"extra={len(set(keys) - set(expected))}")
    for row in rows:
        for relative in row["files"].values():
            path = Path(index).parent / relative
            if not path.is_file() or path.stat().st_size == 0:
                errors.append(f"missing or empty rendered file: {path}")
        rotation, translation = np.asarray(row["R"]), np.asarray(row["T"])
        # PyTorch3D uses row vectors: view = world @ R + T.
        center = -rotation @ translation
        position = np.asarray(HARNESS_LOOK_AT) + row["distance"] * direction(
            row["azimuth"], row["elevation"])
        if not np.allclose(center, position, atol=2e-5):
            errors.append(f"camera position: {row['uid']} {row['azimuth']} {row['elevation']}")
        if not np.allclose(rotation.T @ rotation, np.eye(3), atol=2e-5) or np.linalg.det(rotation) < 0:
            errors.append(f"invalid camera rotation: {row['uid']}")
        projected_up = np.array([0, 1, 0]) @ rotation
        if abs(projected_up[0]) > 2e-5 or projected_up[1] <= 0:
            errors.append(f"camera roll: {row['uid']}")
        centers.append(center.tolist())
    report = {"views": len(rows), "errors": errors, "passed": not errors,
              "objects": len({r["uid"] for r in rows}),
              "invisible_views": sum(r["modal_area"] == 0 for r in rows),
              "truncated_views": sum(r["truncated"] for r in rows),
              "camera_center_min": np.min(centers, axis=0).tolist(),
              "camera_center_max": np.max(centers, axis=0).tolist(),
              "note": "Geometric camera verification; semantic orientation requires visual review."}
    Path(out).write_text(json.dumps(report, indent=2) + "\n")
    if errors:
        raise ValueError(f"Camera verification failed: {errors[:4]}")
    print(f"Verified {len(rows)} camera poses: correct centers and no unintended roll.", flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    sheets = sub.add_parser("sheets")
    sheets.add_argument("--render-dir", type=Path, required=True)
    sheets.add_argument("--out", type=Path, required=True)
    sheets.add_argument("--elevation", type=float, default=22.5)
    apply = sub.add_parser("apply")
    apply.add_argument("--source", required=True)
    apply.add_argument("--index", required=True)
    apply.add_argument("--reviews", required=True)
    apply.add_argument("--root", default="/data/objaverse_nas")
    apply.add_argument("--out", required=True)
    candidates = sub.add_parser("candidates")
    candidates.add_argument("--source", required=True)
    candidates.add_argument("--index", required=True)
    candidates.add_argument("--root", default="/data/objaverse_nas")
    candidates.add_argument("--out", required=True)
    verify = sub.add_parser("verify-cameras")
    verify.add_argument("--index", required=True)
    verify.add_argument("--out", required=True)
    summary = sub.add_parser("summary")
    summary.add_argument("--canonical", required=True)
    summary.add_argument("--out", required=True)
    args = vars(p.parse_args())
    command = args.pop("command")
    if command == "sheets":
        contact_sheets(**args)
    elif command == "apply":
        build_canonical(**args)
    elif command == "candidates":
        candidate_sheets(**args)
    elif command == "verify-cameras":
        verify_cameras(**args)
    else:
        summarize_reviews(**args)


if __name__ == "__main__":
    main()
