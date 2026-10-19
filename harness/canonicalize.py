"""Stage 1: derive a canonical frame per Objaverse object.

Objaverse ships no canonical pose, so viewpoint experiments have no zero to
measure from. This pass emits a per-object similarity transform plus quality
flags; it never rewrites a mesh. Measured on this slice: the glTF Y-up
convention holds for ~85-90% of objects (doormat is majority Z-up), chair
azimuths concentrate at R=0.864, and bbox diagonals span 93x.
"""
from __future__ import annotations

import argparse
import gzip
import json
import os
from pathlib import Path
from collections import defaultdict

import numpy as np
from harness.meshio import load_mesh

UP = 1  # canonical up axis (Y), per the glTF spec


def height_ratio(extents: np.ndarray, up: int) -> float:
    """Height along `up` relative to the larger horizontal side."""
    horiz = [extents[i] for i in range(3) if i != up]
    return float(extents[up] / max(max(horiz), 1e-9))


def axis_swap(up: int) -> np.ndarray:
    """Rotation carrying axis `up` onto +Y, as a 3x3."""
    if up == UP:
        return np.eye(3)
    R = np.zeros((3, 3))
    other = [i for i in range(3) if i != up]
    R[1, up] = 1.0           # chosen up -> Y
    R[0, other[0]] = 1.0
    R[2, other[1]] = 1.0
    if np.linalg.det(R) < 0:  # keep it a proper rotation
        R[[0, 2]] = R[[2, 0]]
    return R


def azimuth_frame(v: np.ndarray) -> tuple[np.ndarray, float]:
    """Deterministic yaw: dominant horizontal PCA axis -> +X, upper mass -> -Z.

    Returns the 3x3 yaw and a confidence in [0, 1]; low confidence means the
    object is too symmetric for the 180-degree flip to be meaningful.
    """
    xz = v[:, [0, 2]] - v[:, [0, 2]].mean(0)
    if len(xz) > 20000:
        xz = xz[np.random.default_rng(0).choice(len(xz), 20000, replace=False)]
    cov = np.cov(xz.T)
    if not np.all(np.isfinite(cov)):
        return np.eye(3), 0.0
    w, vecs = np.linalg.eigh(cov)
    principal = vecs[:, int(np.argmax(w))]
    theta = np.arctan2(principal[1], principal[0])       # angle in XZ
    c, s = np.cos(-theta), np.sin(-theta)
    yaw = np.array([[c, 0, -s], [0, 1, 0], [s, 0, c]])   # rotate about Y

    r = v @ yaw.T
    ext = r.max(0) - r.min(0)
    ctr = (r.max(0) + r.min(0)) / 2
    top = r[r[:, 1] > ctr[1] + 0.15 * ext[1]]
    conf = 0.0
    if len(top) >= 20:
        off = (top.mean(0) - ctr)[[0, 2]] / (ext[[0, 2]] + 1e-9)
        conf = float(min(np.linalg.norm(off) / 0.25, 1.0))
        if off[1] > 0:            # push the tall/asymmetric side to -Z
            flip = np.diag([-1.0, 1.0, -1.0])
            yaw = flip @ yaw
    return yaw, conf


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", default="/ICONAS/users/marfred/objaverse")
    ap.add_argument("--out", default=None, help="default: <root>/canonical.json")
    ap.add_argument("--max-mb", type=float, default=64.0, help="skip GLBs larger than this")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--trust-file-up", action="store_true",
                    help="keep the glTF +Y up axis; the audit is still recorded in up_axis_detected")
    args = ap.parse_args()

    root = args.root
    ver = os.path.join(root, "hf-objaverse-v1")
    out_path = args.out or os.path.join(root, "canonical.json")
    max_bytes = int(args.max_mb * 2**20)

    with gzip.open(os.path.join(ver, "object-paths.json.gz"), "rb") as f:
        object_paths = json.load(f)
    u2c = json.loads(Path(root, "manifest.json").read_text())["uid_to_categories"]
    uids = sorted(u2c)
    if args.limit:
        uids = uids[: args.limit]

    # --- pass 1: measure each mesh under the assumed Y-up frame -------------
    raw, skipped = {}, {}
    for n, uid in enumerate(uids, 1):
        rel = object_paths.get(uid)
        g = load_mesh(os.path.join(ver, rel), max_bytes) if rel else None
        if g is None:
            skipped[uid] = "unloadable_or_oversize"
            continue
        v = np.asarray(g.vertices, dtype=np.float64)
        ext = v.max(0) - v.min(0)
        if not np.all(np.isfinite(ext)) or np.any(ext <= 0):
            skipped[uid] = "degenerate_bbox"
            continue
        raw[uid] = dict(
            ratios=[height_ratio(ext, a) for a in range(3)],
            faces=int(len(g.faces)),
            verts=int(len(v)),
            watertight=bool(g.is_watertight),
        )
        if n % 250 == 0:
            print(f"  measured {n}/{len(uids)}", flush=True)

    # --- category medians of the height ratio, assuming Y-up ---------------
    per_cat = defaultdict(list)
    for uid, r in raw.items():
        per_cat[u2c[uid][0]].append(r["ratios"][UP])
    cat_median = {c: float(np.median(v)) for c, v in per_cat.items()}

    # --- pass 2: audit the up axis against the category, then build frames --
    records, suspect = {}, 0
    for n, (uid, r) in enumerate(raw.items(), 1):
        cat = u2c[uid][0]
        target = cat_median[cat]
        best = int(np.argmin([abs(r["ratios"][a] - target) for a in range(3)]))
        R = axis_swap(UP if args.trust_file_up else best)

        # Reload: extents must come from every vertex. azimuth_frame may
        # subsample internally, since PCA is statistical, but the bbox may not.
        g = load_mesh(os.path.join(ver, object_paths[uid]), max_bytes)
        if g is None:
            skipped[uid] = "unloadable_on_second_pass"
            continue
        v_full = np.asarray(g.vertices, dtype=np.float64)
        yaw, conf = azimuth_frame(v_full @ R.T)
        R = yaw @ R

        v_c = v_full @ R.T
        ext = v_c.max(0) - v_c.min(0)
        scale = 1.0 / float(np.linalg.norm(ext))
        lo, hi = v_c.min(0), v_c.max(0)
        # centre XZ, rest the base on y=0, in post-scale units
        t = np.array([-(lo[0] + hi[0]) / 2 * scale, -lo[1] * scale, -(lo[2] + hi[2]) / 2 * scale])

        if best != UP:
            suspect += 1
        if n % 250 == 0:
            print(f"  framed {n}/{len(raw)}", flush=True)
        records[uid] = dict(
            category=cat,
            rotation=R.tolist(),
            scale=scale,
            translation=t.tolist(),
            extents_canonical=(ext * scale).tolist(),
            up_axis_detected="XYZ"[best],
            up_axis_suspect=bool(best != UP),
            azimuth_confidence=round(conf, 3),
            face_count=r["faces"],
            vertex_count=r["verts"],
            is_watertight=r["watertight"],
        )

    payload = dict(
        convention=dict(up="+Y", forward="-Z", scale="unit bbox diagonal", base="y=0"),
        category_height_ratio_median=cat_median,
        objects=records,
        skipped=skipped,
    )
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text(json.dumps(payload, indent=2) + "\n")

    lowconf = sum(1 for r in records.values() if r["azimuth_confidence"] < 0.35)
    nonwt = sum(1 for r in records.values() if not r["is_watertight"])
    print(f"\nwrote {out_path}")
    print(f"  canonicalised   {len(records)}")
    print(f"  skipped         {len(skipped)}")
    print(f"  up-axis suspect {suspect}")
    print(f"  low azimuth confidence (<0.35) {lowconf}")
    print(f"  non-watertight  {nonwt}  (Chamfer/IoU unreliable on these)")


if __name__ == "__main__":
    main()
