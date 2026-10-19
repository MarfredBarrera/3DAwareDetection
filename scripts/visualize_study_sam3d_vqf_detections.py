"""SAM3 on the input views vs. on the SAM3D view chosen by the reconstruction's VQF.

python -m scripts.visualize_study_sam3d_vqf_detections plan --study outputs/upright --elevations 0 45
python -m scripts.visualize_study_sam3d_vqf_detections figures --study outputs/upright --elevations 0 45

For each ground-truth input view, the VQF is computed on its SAM3D
reconstruction's candidate views exactly as for the ground-truth views
(per metric, quality = 1 - stored ambiguity, min-max normalized over that
input's candidates; equal mean). Candidate angles are relative to the input
camera; the elevation band applies to the absolute elevation obtained with the
input view's known elevation (``viewactive.grid.sam3d_to_viewactive``), standing
in for a known gravity direction. The chosen view is the band's VQF maximum.

SAM3D candidates are rendered with gravity up (``tools.cameras.sam3d_gravity``),
so the chosen views are level.

``plan`` writes ``selections.json`` and the archive members to extract
(``missing_members.txt``, relative to the study directory); ``figures`` writes
one page per object (input grid above, chosen SAM3D views below), ``objects.csv``
and ``objects_scatter.png`` under ``<study>/figures/sam3d_vqf_detections_el<LOW>_<HIGH>/``.
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np

from scripts import visualize_viewactive_gt_maxima as maxima
from scripts.visualize_study_vqf_detections import (FP_COLOR, VQF_COLOR, detections, label, scatter,
                                                     view_detections)
from scripts.visualize_study_vqf_views import crop_box, thumbnail
from viewactive.grid import Grid, combine, elevation_rows, sam3d_to_viewactive
from viewactive.labels import normalize

METRICS = maxima.STUDY_METRICS


def output_dir(study, limits):
    return study / "figures" / f"sam3d_vqf_detections_el{limits[0]:g}_{limits[1]:g}"


def plan(study, limits):
    from study import config as settings_io
    from study import db as store

    config = settings_io.read(study / "config.json")
    recon = config["reconstruction"]
    azimuths = [float(a) for a in recon["candidate_azimuths"]]
    elevations = [float(e) for e in recon["candidate_elevations"]]
    db = store.connect(study)
    scores = store.view_scores(db, "i.source='sam3d'")
    detected = view_detections(db, config, "sam3d")
    inputs = {r["image_id"]: r for r in store.images(db, "source='gt'")}
    children = {}
    for row in store.images(db, "source='sam3d'"):
        children.setdefault(row["parent_id"], []).append(row)
    db.close()
    selections, members = {}, []
    for input_id, candidates in children.items():
        source = inputs[input_id]
        raw = np.full((len(METRICS), len(elevations), len(azimuths)), np.nan)
        absolute = np.full((len(elevations), len(azimuths)), np.nan)
        by_cell = {}
        for c in candidates:
            e, a = elevations.index(c["elevation"]), azimuths.index(c["azimuth"])
            by_cell[(e, a)] = c
            absolute[e, a] = sam3d_to_viewactive(c["azimuth"], c["elevation"], source["elevation"])[1]
            for m, metric in enumerate(METRICS):
                value = scores.get(c["image_id"], {}).get(metric)
                if value is not None:
                    raw[m, e, a] = 1 - value
        field = combine(normalize(raw))
        usable = np.array([[(e, a) in by_cell and by_cell[(e, a)]["image_id"] in detected
                            for a in range(len(azimuths))] for e in range(len(elevations))])
        keep = usable & (absolute >= limits[0] - 1e-6) & (absolute <= limits[1] + 1e-6) & np.isfinite(field)
        if not keep.any():
            continue
        e, a = map(int, np.unravel_index(np.argmax(np.where(keep, field, -np.inf)), field.shape))
        pick = by_cell[(e, a)]
        selections[input_id] = {"pick_id": pick["image_id"], "path": pick["path"], "mask_path": pick["mask_path"],
                                "relative": [pick["azimuth"], pick["elevation"]],
                                "absolute_elevation": float(absolute[e, a]), "vqf": float(field[e, a]),
                                "band_candidates": int(keep.sum())}
        members += [pick["path"], pick["mask_path"]]
    out = output_dir(study, limits)
    out.mkdir(parents=True, exist_ok=True)
    (out / "selections.json").write_text(json.dumps({"elevation_range": list(limits),
                                                     "selections": selections}, indent=1) + "\n")
    missing = [m for m in dict.fromkeys(members) if not (study / m).is_file()]
    (out / "missing_members.txt").write_text("".join(f"{m}\n" for m in missing))
    print(f"{len(selections)}/{len(children)} inputs have a candidate in the band; "
          f"{len(missing)} images to extract -> {out / 'missing_members.txt'}", flush=True)


def figures(study, limits):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    from study import config as settings_io
    from study import db as store

    maxima.STUDY = study
    config = settings_io.read(study / "config.json")
    data = maxima.load_data()
    grid = Grid(tuple(map(float, data["azimuths"])), tuple(map(float, data["elevations"])))
    rows, cols = grid.shape
    out = output_dir(study, limits)
    out.mkdir(parents=True, exist_ok=True)
    selections = {int(k): v for k, v in json.loads((out / "selections.json").read_text())["selections"].items()}
    db = store.connect(study)
    sam3d = view_detections(db, config, "sam3d")
    ids = {(r["uid"], grid.elevations.index(r["elevation"]), grid.azimuths.index(r["azimuth"])): r["image_id"]
           for r in store.images(db, "source='gt'")}
    db.close()
    gt = detections(study, config)
    allowed = elevation_rows(grid, limits)[:, None]
    band = f"elevations {limits[0]:+.0f}..{limits[1]:+.0f}°"
    threshold = config["detection"]["score_threshold"]
    order = sorted(range(len(data["uids"])), key=lambda o: (data["categories"][o], data["uids"][o]))
    table = []
    with PdfPages(out / "objects.pdf") as pdf:
        for o in order:
            uid, category = str(data["uids"][o]), str(data["categories"][o])
            image = lambda e, a: study / maxima.member({"uid": uid}, (e, a), data)
            box = crop_box([image(e, a) for e in range(rows) for a in range(cols)])
            cells = [(e, a) for e in range(rows) for a in range(cols) if (uid, e, a) in gt]
            picked = {(e, a): selections[ids[(uid, e, a)]] for e, a in cells if ids[(uid, e, a)] in selections}
            inputs = [gt[(uid, *c)] for c in picked]
            chosen = [sam3d[picked[c]["pick_id"]] for c in picked]
            combined = combine(data["norm"][o])
            gt_vqf = tuple(map(int, np.unravel_index(np.nanargmax(np.where(allowed, combined, np.nan)),
                                                      combined.shape)))
            gt_view = gt[(uid, *gt_vqf)]
            mean = lambda items, key: float(np.mean([d[key] for d in items]))
            share = lambda items, test: float(np.mean([test(d) for d in items]))
            summary = {"category": category, "uid": uid, "inputs": len(picked),
                       "input_mean_margin": mean(inputs, "margin"), "input_mean_fp_rate": mean(inputs, "fp_rate"),
                       "input_any_fp": share(inputs, lambda d: d["fp"] > 0),
                       "input_negative_margin": share(inputs, lambda d: d["margin"] < 0),
                       "vqf_margin": mean(chosen, "margin"), "vqf_fp_rate": mean(chosen, "fp_rate"),
                       "vqf_any_fp": share(chosen, lambda d: d["fp"] > 0),
                       "vqf_negative_margin": share(chosen, lambda d: d["margin"] < 0),
                       "gt_vqf_margin": gt_view["margin"], "gt_vqf_fp_rate": gt_view["fp_rate"]}
            table.append(summary)

            fig = plt.figure(figsize=(28, 17))
            outer = fig.add_gridspec(2, 2, width_ratios=[cols, 3.4], wspace=.04, hspace=.16,
                                     left=.01, right=.99, top=.94, bottom=.04)
            chosen_title = f"SAM3D view chosen by the reconstruction's VQF ({band}, absolute)"
            for block, title in ((0, "Input views (ground truth)"), (1, chosen_title)):
                sub = outer[block, 0].subgridspec(rows, cols, wspace=.06, hspace=.42)
                for e in range(rows):
                    for a in range(cols):
                        ax = fig.add_subplot(sub[rows - 1 - e, a])
                        if block == 0:
                            d = gt.get((uid, e, a))
                            thumbnail(ax, image(e, a), f"az {grid.azimuths[a]:.0f} el {grid.elevations[e]:+.0f}\n"
                                      f"{label(d)}", fontsize=5.5, box=box,
                                      title_color=FP_COLOR if d and d["fp"] else "black")
                        elif (e, a) in picked:
                            s = picked[(e, a)]
                            d = sam3d[s["pick_id"]]
                            path = study / s["path"]
                            own = crop_box([path], masks=[study / s["mask_path"]])
                            thumbnail(ax, path, f"rel {s['relative'][0]:.0f}/{s['relative'][1]:.0f} "
                                      f"→ el {s['absolute_elevation']:+.0f}\n{label(d)}", fontsize=5.5,
                                      box=own, title_color=FP_COLOR if d["fp"] else "black")
                        else:
                            ax.axis("off")
                        if e == rows - 1 and a == 0:
                            ax.text(0, 1.55, title, transform=ax.transAxes, fontsize=13, weight="bold")
            ax = fig.add_subplot(outer[:, 1]); ax.axis("off")
            ax.text(0, .98, f"{'':22s}{'inputs':>9s}{'SAM3D VQF':>11s}{'GT VQF':>9s}\n"
                    f"{'mean margin':22s}{summary['input_mean_margin']:>+9.3f}{summary['vqf_margin']:>+11.3f}"
                    f"{gt_view['margin']:>+9.3f}\n"
                    f"{'mean FP rate':22s}{summary['input_mean_fp_rate']:>9.1%}{summary['vqf_fp_rate']:>11.1%}"
                    f"{gt_view['fp_rate']:>9.1%}\n"
                    f"{'views with any FP':22s}{summary['input_any_fp']:>9.1%}{summary['vqf_any_fp']:>11.1%}"
                    f"{float(gt_view['fp'] > 0):>9.0%}\n"
                    f"{'views with margin < 0':22s}{summary['input_negative_margin']:>9.1%}"
                    f"{summary['vqf_negative_margin']:>11.1%}{float(gt_view['margin'] < 0):>9.0%}\n\n"
                    f"inputs: {len(picked)} ground-truth views with a SAM3D\ncandidate in the band.\n"
                    f"SAM3D VQF: mean over the views chosen\nfor each input (bottom grid).\n"
                    f"GT VQF: the single constrained ground-truth\nVQF view "
                    f"(az {grid.azimuths[gt_vqf[1]]:.0f}°, el {grid.elevations[gt_vqf[0]]:+.0f}°).",
                    family="monospace", fontsize=11, va="top", transform=ax.transAxes)
            fig.suptitle(f"{category} | {uid} — SAM3 on every input view vs. its SAM3D VQF view", fontsize=16)
            fig.text(.01, .012, f"m = SAM3 margin | FP = wrong prompts scoring ≥ {threshold} (red titles) | "
                     "rel = candidate azimuth/elevation relative to the input camera; el = absolute elevation | "
                     "input views share one crop box; SAM3D views are cropped to their own mask", fontsize=10)
            path = out / category / f"{uid}.png"
            path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(path, dpi=100)
            pdf.savefig(fig)
            plt.close(fig)
    with (out / "objects.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(table[0]))
        writer.writeheader()
        writer.writerows(table)
    scatter(table, out / "objects_scatter.png", ylabel="Mean over SAM3D VQF views")
    print(f"Saved {len(table)} object pages, objects.csv and objects_scatter.png under {out}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("plan", "figures"))
    parser.add_argument("--study", type=Path, required=True)
    parser.add_argument("--elevations", nargs=2, type=float, metavar=("LOW", "HIGH"), required=True)
    args = parser.parse_args()
    limits = tuple(args.elevations)
    {"plan": plan, "figures": figures}[args.stage](args.study, limits)
