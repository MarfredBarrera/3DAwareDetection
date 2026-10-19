"""CPU-only audit of cached VQF labels, classifiers, and exact-field navigation.

Run inside the dev container: python -m scripts.diagnose_viewactive_vqf
These are diagnostic ablations, not newly selected evaluation settings.
"""
import json
from pathlib import Path

import numpy as np
from scipy.stats import rankdata

from viewactive.cohort import load_class_map
from viewactive.grid import Grid, combine, reachable, nearest


def main():
    root = Path("outputs/viewactive/blender_full_viz/run")
    out = Path("outputs/viewactive/vqf_audit")
    out.mkdir(exist_ok=True, parents=True)
    data = dict(np.load(root / "vqf.npz"))
    config = json.loads((root / "config.json").read_text())
    grid = Grid.from_config(config["render"])
    mapping = load_class_map("configs/viewactive_imagenet_map.yaml")
    correct = {}
    for path in sorted((root / "classifier").glob("*.npz")):
        stored = np.load(path)
        lookup = dict(zip(stored["uids"], stored["top1"]))
        correct[path.stem] = np.array([
            np.isin(lookup[uid], mapping[cat]) if uid in lookup else np.zeros(grid.shape, bool)
            for uid, cat in zip(data["uids"], data["categories"])])
    # Translation-invariant transition table for the existing exact-field policy.
    E, A = grid.shape
    transition = np.empty((E, E, A, 2), int)
    for e in range(E):
        mask = reachable(grid, e, config["optimize"]["reach_cells"])
        for te in range(E):
            for ta in range(A):
                transition[e, te, ta] = (te, ta) if mask[te, ta] else nearest(grid, mask, te, ta)
    report = {}
    for split in ("test", "test_unseen"):
        ids = np.flatnonzero(data["splits"] == split)
        raw, norm = data["raw"][ids], data["norm"][ids]
        ok = {k: v[ids].reshape(len(ids), -1) for k, v in correct.items()}
        scores = {
            "normalized_combined": combine(norm), "raw_scaled_combined": combine(raw),
            **{str(c): norm[:, i] for i, c in enumerate(data["channels"])},
        }
        best = {}
        for name, score in scores.items():
            choice = np.nanargmax(score.reshape(len(ids), -1), axis=1)
            best[name] = {"accuracy": {k: float(v[np.arange(len(ids)), choice].mean()) for k, v in ok.items()},
                          "mean_abs_elevation": float(np.abs(data["elevations"][choice // A]).mean()),
                          "extreme_elevation_fraction": float((np.abs(data["elevations"][choice // A]) >= 60).mean())}
        combined = combine(norm).reshape(len(ids), -1)
        target = np.nanargmax(combined, axis=1)
        e = np.broadcast_to(np.arange(E*A)[None] // A, (len(ids), E*A)).copy()
        a = np.broadcast_to(np.arange(E*A)[None] % A, (len(ids), E*A)).copy()
        paths = []
        for step in range(101):
            cell = e*A+a
            if step in (0, 9, 100):
                paths.append({"step": step,
                    "accuracy": {k: float(v[np.arange(len(ids))[:, None], cell].mean()) for k,v in ok.items()},
                    "target_reached": float((cell == target[:, None]).mean()),
                    "extreme_elevation_fraction": float((np.abs(data["elevations"][e]) >= 60).mean()),
                    "mean_quality": float(np.nanmean(combined[np.arange(len(ids))[:, None], cell]))})
            update = transition[e, target[:, None] // A, (target[:, None] % A - a) % A]
            e, a = update[..., 0], (a + update[..., 1]) % A
        corr = {}
        for name, score in scores.items():
            ranks = rankdata(score.reshape(len(ids), -1), axis=1, nan_policy="omit")
            ranks -= np.nanmean(ranks, axis=1, keepdims=True)
            corr[name] = {}
            for k, v in ok.items():
                y = v - v.mean(1, keepdims=True)
                denom = np.sqrt(np.nansum(ranks*ranks, 1)*np.sum(y*y, 1))
                valid = denom > 0
                corr[name][k] = {"rho": float(np.mean(np.nansum(ranks*y, 1)[valid]/denom[valid])),
                                 "variable_objects": int(valid.sum())}
        categories = []
        for cat in sorted(set(data["categories"][ids])):
            ix = data["categories"][ids] == cat
            categories.append({"category": str(cat), "objects": int(ix.sum()),
                "accuracy": {k: float(v[ix].mean()) for k,v in ok.items()},
                "any_correct_view": {k: float(v[ix].any(1).mean()) for k,v in ok.items()}})
        report[split] = {"objects": len(ids), "global_best": best, "exact_field_all_start_rollouts": paths,
            "all_view_accuracy": {k: float(v.mean()) for k,v in ok.items()},
            "any_correct_view": {k: float(v.any(1).mean()) for k,v in ok.items()},
            "within_object_spearman": corr,
            "elevations": [{"elevation": float(el),
                "accuracy": {k: float(v.reshape(-1,E,A)[:,i].mean()) for k,v in ok.items()},
                "raw_quality": np.nanmean(raw[:,:,i,:], axis=(0,2)).tolist(),
                "normalized_quality": np.nanmean(norm[:,:,i,:], axis=(0,2)).tolist()}
                for i,el in enumerate(data["elevations"])],
            "raw_channel_span_quantiles": np.nanquantile(np.nanmax(raw,axis=(2,3))-np.nanmin(raw,axis=(2,3)),[0,.1,.5,.9,1],axis=0).tolist(),
            "categories": categories}
    (out / "cached_audit.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k:{n:v for n,v in r.items() if n in ("objects","global_best","exact_field_all_start_rollouts","any_correct_view","raw_channel_span_quantiles")} for k,r in report.items()}, indent=2))


if __name__ == "__main__":
    main()
