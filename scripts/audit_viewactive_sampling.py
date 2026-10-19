"""Diagnostic ten-points-per-face relabeling; production labels stay untouched.

The paper leaves partial-face area handling ambiguous, so retain both fractional
visible area and fully-visible-face area. Histograms use one vote per fully
visible face. A dense area-weighted control isolates sampling density.
"""
import json
import math
from pathlib import Path
import time

import numpy as np

from harness.meshio import load_mesh
from study.ambiguity import build_context, visible_samples
from tools.cameras import HARNESS_LOOK_AT
from viewactive.labels import normalize
from viewactive.cohort import load_class_map

ROOT = Path("outputs/viewactive/vqf_audit")
RUN = Path("outputs/viewactive/blender_full_viz/run")


def entropy(normals, forward, weights=None):
    keep = normals @ forward >= 0
    normals = normals[keep]
    if not len(normals):
        return np.nan
    weights = None if weights is None else weights[keep]
    reference = np.array([0.,1.,0.])
    if abs(forward @ reference) > .99:
        reference = np.array([0.,0.,1.])
    right = np.cross(reference, forward)
    right /= np.linalg.norm(right)
    up = np.cross(forward, right)
    hist = np.histogram2d(np.arccos(np.clip(normals @ forward,0,1)),
                          np.arctan2(normals @ up,normals @ right),bins=(9,18),
                          range=((0,np.pi/2),(-np.pi,np.pi)),weights=weights)[0]
    nz = hist[hist>0]
    p = nz/nz.sum()
    return len(nz)/162 * -np.sum(p*np.log2(p))/math.log2(162)


def main():
    data = dict(np.load(RUN / "vqf.npz"))
    canonical = json.loads((ROOT/"canonical.json").read_text())["objects"]
    paths = json.loads((ROOT/"download.json").read_text())["glbs"]
    eligible = np.flatnonzero(data["splits"] == "test")
    ids = np.sort(np.random.default_rng(20261017).choice(eligible, 50, replace=False))
    destination = ROOT/"sampling"
    destination.mkdir(exist_ok=True)
    start = time.monotonic()
    for index, o in enumerate(ids):
        uid = str(data["uids"][o])
        output = destination/f"{uid}.npz"
        if output.exists():
            continue
        rec = canonical[uid]
        mesh = load_mesh(paths[uid])
        v = (np.asarray(mesh.vertices) @ np.array(rec["rotation"]).T)*rec["scale"] + rec["translation"]
        f = np.asarray(mesh.faces)
        triangles = v[f]
        crosses = np.cross(triangles[:,1]-triangles[:,0], triangles[:,2]-triangles[:,0])
        lengths = np.linalg.norm(crosses,axis=1)
        keep = lengths>1e-12
        f,triangles,crosses,lengths=f[keep],triangles[keep],crosses[keep],lengths[keep]
        areas=lengths/2
        normals=crosses/lengths[:,None]
        context = build_context(v,f,samples=1,seed=0,target=HARNESS_LOOK_AT,distance=1.4,two_sided=True,nthreads=2)
        rng=np.random.default_rng(0)
        u=np.sqrt(rng.random((len(f),10,1))); w=rng.random((len(f),10,1))
        points=(1-u)*triangles[:,None,0]+u*(1-w)*triangles[:,None,1]+u*w*triangles[:,None,2]
        context.points=points.reshape(-1,3).astype(np.float32)
        context.normals=np.repeat(normals,10,axis=0).astype(np.float32)
        dense=np.array(data["raw"][o],copy=True)
        face=dense.copy(); full=dense.copy()
        # Separate finite-sampling perturbation from estimator change.
        repeat_context = build_context(v,f,samples=20000,seed=1,target=HARNESS_LOOK_AT,distance=1.4,two_sided=True,nthreads=2)
        repeat=dense.copy()
        for e,el in enumerate(data["elevations"]):
            for a,az in enumerate(data["azimuths"]):
                cam={"azimuth":float(az),"elevation":float(el)}
                visible,forward,oriented=visible_samples(context,cam)
                byface=visible.reshape(-1,10)
                visible_fraction=byface.mean(1)
                fullface=byface.all(1)
                dense[0,e,a]=face[0,e,a]=np.sum(areas*visible_fraction)/areas.sum()
                full[0,e,a]=np.sum(areas*fullface)/areas.sum()
                dense[1,e,a]=entropy(oriented[visible],forward,np.repeat(areas/10,10)[visible])
                # All ten normals have identical orientation except edge-on two-sided faces.
                oriented_faces=oriented.reshape(-1,10,3)[:,0]
                full[1,e,a]=face[1,e,a]=entropy(oriented_faces[fullface],forward)
                visible2,forward2,normals2=visible_samples(repeat_context,cam)
                repeat[0,e,a]=visible2.mean()
                repeat[1,e,a]=entropy(normals2[visible2],forward2)
        np.savez_compressed(output,uid=uid,index=o,dense=dense,face=face,full=full,repeat=repeat,faces=len(f))
        print(f"{index+1}/50 {uid} {data['categories'][o]} faces={len(f)} elapsed={time.monotonic()-start:.1f}s",flush=True)
    mapping=load_class_map("configs/viewactive_imagenet_map.yaml")
    correct={}
    for path in sorted((RUN/"classifier").glob("*.npz")):
        d=np.load(path); lookup=dict(zip(d["uids"],d["top1"]))
        correct[path.stem]=np.array([np.isin(lookup[data["uids"][o]],mapping[data["categories"][o]]) for o in ids]).reshape(len(ids),-1)
    from scipy.stats import spearmanr
    old=data["raw"][ids]
    results={"objects":len(ids),"uids":data["uids"][ids].tolist(),"variants":{}}
    variants={"original":old}
    for name in ("dense","face","full","repeat"):
        variants[name]=np.stack([np.load(destination/f"{data['uids'][o]}.npz")[name] for o in ids])
    oldscore=normalize(old).mean(1).reshape(len(ids),-1)
    for name,raw in variants.items():
        score=normalize(raw).mean(1).reshape(len(ids),-1)
        choice=np.nanargmax(score,axis=1)
        result={"accuracy":{k:float(v[np.arange(len(ids)),choice].mean()) for k,v in correct.items()},
                "raw_mae":np.nanmean(np.abs(raw-old),axis=(0,2,3)).tolist(),
                "changed_best_fraction":float((choice!=np.nanargmax(oldscore,axis=1)).mean()),
                "combined_rank_correlation_to_original":float(np.nanmean([spearmanr(x,y,nan_policy='omit').statistic for x,y in zip(score,oldscore)])),
                "extreme_elevation_fraction":float((np.abs(data['elevations'][choice//12])>=60).mean())}
        for k,v in correct.items():
            baseline=v[np.arange(len(ids)),np.nanargmax(oldscore,axis=1)]
            delta=v[np.arange(len(ids)),choice].astype(float)-baseline.astype(float)
            boot=np.random.default_rng(0).choice(delta,(5000,len(ids))).mean(1)
            result.setdefault("accuracy_delta_ci95",{})[k]=np.quantile(boot,[.025,.975]).tolist()
        results["variants"][name]=result
    (ROOT/"sampling_audit.json").write_text(json.dumps(results,indent=2)+"\n")
    print(json.dumps(results["variants"],indent=2))


if __name__ == "__main__":
    main()
