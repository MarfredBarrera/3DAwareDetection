"""Driver: canonical.json + GLBs -> a rendered viewpoint x occlusion dataset.

Run inside the container, where PyTorch3D and the GPU live:
  sam3d-env sam3d python -m harness.build_dataset --help
"""
from __future__ import annotations

import argparse
import json
import os
import random
import time
from pathlib import Path

import numpy as np
import torch

from harness.meshio import ObjaverseStore
from harness.render import Harness


def select_uids(pool, objects, *, limit, objects_per_category=None, categories=None):
    """Select either a global prefix or a balanced prefix for every category.

    ``pool`` is expected to have already been shuffled, so this helper is both
    deterministic for a seeded caller and independent of dictionary ordering.
    """
    if objects_per_category is None:
        return pool[:limit]
    grouped = {}
    for uid in pool:
        grouped.setdefault(objects[uid]["category"], []).append(uid)
    expected = sorted(categories if categories is not None else grouped)
    short = {category: len(grouped.get(category, [])) for category in expected
             if len(grouped.get(category, [])) < objects_per_category}
    if short:
        detail = ", ".join(f"{category}={count}" for category, count in short.items())
        raise ValueError(
            f"not enough eligible objects for --objects-per-category="
            f"{objects_per_category}: {detail}")
    return [uid for category in expected
            for uid in grouped[category][:objects_per_category]]


def save_png(t: torch.Tensor, path: str) -> None:
    from PIL import Image
    a = t.detach().cpu().numpy()
    if a.ndim == 2:
        Image.fromarray((a.astype(np.uint8) * 255)).save(path)
    else:
        Image.fromarray((a * 255).clip(0, 255).astype(np.uint8)).save(path)


def sample_id(azimuth, elevation, occlusion):
    """Preserve fractional settings so distinct views never overwrite one another."""
    return f"a{float(azimuth)}_e{float(elevation)}_o{float(occlusion)}"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--root", default="/data/objaverse_nas",
                   help="objaverse root as mounted in the container")
    p.add_argument("--canonical", default=None, help="default: <root>/canonical.json")
    p.add_argument("--out", default="/outputs/viewocc")
    p.add_argument("--categories", nargs="*", default=None)
    p.add_argument("--limit", type=int, default=20, help="objects to render")
    p.add_argument("--objects-per-category", type=int, default=None,
                   help="balanced category sample; overrides the global --limit")
    p.add_argument("--azimuths", type=float, nargs="*",
                   default=[0, 45, 90, 135, 180, 225, 270, 315])
    p.add_argument("--elevations", type=float, nargs="*", default=[0, 20, 40])
    p.add_argument("--distance", type=float, default=2.2)
    p.add_argument("--occlusion-levels", type=float, nargs="*",
                   default=[0.0, 0.15, 0.30, 0.45, 0.60, 0.75])
    p.add_argument("--occluders-per-view", type=int, default=1)
    p.add_argument("--occluder-scale", type=float, default=1.0,
                   help="multiplier on the auto scale that reaches the sight line")
    p.add_argument("--along", type=float, default=0.5,
                   help="occluder standoff: 0=at camera, 1=at target")
    p.add_argument("--image-size", type=int, default=512)
    p.add_argument("--max-faces", type=int, default=300_000,
                   help="skip heavy meshes; the slice runs to 211 MiB at the tail")
    p.add_argument("--min-azimuth-confidence", type=float, default=0.0)
    p.add_argument("--exclude-up-suspect", action="store_true")
    p.add_argument("--require-upright-verified", action="store_true",
                   help="refuse objects without explicit upright review")
    p.add_argument("--save-depth", action="store_true")
    p.add_argument("--sweep-steps", type=int, default=13)
    p.add_argument("--tolerance", type=float, default=0.03)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda")
    args = p.parse_args()
    if (args.limit < 1 or args.image_size < 16 or args.sweep_steps < 2
            or (args.objects_per_category is not None and args.objects_per_category < 1)):
        p.error("limit, objects-per-category >= 1; image-size >= 16; "
                "sweep-steps >= 2 are required")
    if not args.occlusion_levels or any(not 0 <= level < 1 for level in args.occlusion_levels):
        p.error("occlusion-levels must be in [0, 1)")
    if not 0 < args.along < 1 or args.distance <= 0 or args.occluder_scale <= 0:
        p.error("along must be in (0, 1); distance and occluder-scale must be positive")
    if args.occluders_per_view < 0 or args.tolerance < 0:
        p.error("occluders-per-view and tolerance must be nonnegative")
    args.occlusion_levels = sorted(set(args.occlusion_levels))

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise SystemExit("CUDA is not available; this harness is GPU-only by design.")

    rng = random.Random(args.seed)
    os.makedirs(args.out, exist_ok=True)
    store = ObjaverseStore(args.root, args.canonical)
    harness = Harness(image_size=args.image_size, device=args.device)

    pool = store.uids(categories=args.categories, max_faces=args.max_faces,
                      min_azimuth_confidence=args.min_azimuth_confidence,
                      exclude_up_suspect=args.exclude_up_suspect)
    if not pool:
        raise SystemExit("no objects matched the filters")
    rng.shuffle(pool)
    chosen = select_uids(pool, store.objects, limit=args.limit,
                         objects_per_category=args.objects_per_category,
                         categories=args.categories)
    if args.require_upright_verified:
        unverified = [uid for uid in chosen
                      if store.objects[uid].get("upright_verification", {}).get("status")
                      != "verified"]
        if unverified:
            raise ValueError(f"Objects need upright verification: {unverified}")
    occ_pool = store.uids(max_faces=min(args.max_faces, 150_000))
    positive_levels = [level for level in args.occlusion_levels if level > 0]

    index_path = os.path.join(args.out, "index.jsonl")
    meta = dict(vars(args), convention=store.convention, n_candidates=len(pool))
    Path(args.out, "config.json").write_text(json.dumps(meta, indent=2) + "\n")

    n_samples, n_missed, t0 = 0, 0, time.time()
    with open(index_path, "w") as index:
        for i, uid in enumerate(chosen, 1):
            tgt = store.load(uid, device=args.device)
            if tgt is None:
                continue
            sub = os.path.join(args.out, "samples", uid)
            os.makedirs(sub, exist_ok=True)

            for elev in args.elevations:
                for azim in args.azimuths:
                    occs = []
                    for _ in range(args.occluders_per_view if positive_levels and occ_pool else 0):
                        cand = rng.choice(occ_pool)
                        for _ in range(8):   # an occluder of another category reads as a scene
                            if store.objects[cand]["category"] != tgt.category:
                                break
                            cand = rng.choice(occ_pool)
                        o = store.load(cand, device=args.device)
                        if o is not None:
                            occs.append(o)
                    picks, _ = harness.sweep_occlusion(
                        tgt, occs, args.distance, elev, azim,
                        targets=positive_levels,
                        occluder_scale=args.occluder_scale, along=args.along,
                        steps=args.sweep_steps, tol=args.tolerance)

                    wanted = ([(0.0, None)] if 0 in args.occlusion_levels else [])
                    wanted += [(level, picks[level]["offset"]) for level in sorted(picks)]
                    n_missed += len(positive_levels) - len(picks)

                    for level, off in wanted:
                        r = harness.render(
                            tgt, [] if off is None else occs,
                            args.distance, elev, azim, offset=off or 0.0,
                            occluder_scale=args.occluder_scale, along=args.along,
                            want_rgb=True, want_depth=args.save_depth)
                        sid = sample_id(azim, elev, level)
                        save_png(r["rgb"], os.path.join(sub, f"{sid}.rgb.png"))
                        save_png(r["amodal"], os.path.join(sub, f"{sid}.amodal.png"))
                        save_png(r["modal"], os.path.join(sub, f"{sid}.modal.png"))
                        if args.save_depth:
                            np.save(os.path.join(sub, f"{sid}.depth.npy"),
                                    r["depth"].cpu().numpy())
                        rec = dict(r["gt"], sample_id=sid, requested_occlusion=level,
                                   files=dict(rgb=f"samples/{uid}/{sid}.rgb.png",
                                              amodal=f"samples/{uid}/{sid}.amodal.png",
                                              modal=f"samples/{uid}/{sid}.modal.png"))
                        if "upright_verification" in store.objects[uid]:
                            rec["upright_verification"] = store.objects[uid]["upright_verification"]
                            rec["semantic_front_verified"] = store.objects[uid]["semantic_front_verified"]
                            rec["view_regime"] = "realistic" if 0 <= elev <= 60 else "extreme"
                        if args.save_depth:
                            rec["files"]["depth"] = f"samples/{uid}/{sid}.depth.npy"
                        index.write(json.dumps(rec) + "\n")
                        n_samples += 1
                    index.flush()
            print(f"[{i}/{len(chosen)}] {uid[:8]} {tgt.category:14s} "
                  f"samples={n_samples} ({time.time()-t0:.0f}s)", flush=True)

    print(f"\nwrote {index_path}")
    print(f"  samples          {n_samples}")
    print(f"  unreachable levels {n_missed}  (occluder could not hit the ratio)")


if __name__ == "__main__":
    main()
