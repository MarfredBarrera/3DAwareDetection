"""SAM3 on the input views vs. on the constrained ground-truth VQF maximum.

python -m scripts.visualize_study_vqf_detections --study outputs/upright --elevations 0 45

Per object: every ground-truth input view with its SAM3 margin and false
positives, next to the constrained VQF maximum (searched in the elevation band;
built as in ``scripts.visualize_study_vqf_views``) with the same numbers. A
false positive is a wrong vocabulary prompt scoring at least
``detection.score_threshold``; FP rate = false positives / (vocabulary - 1).
Also writes ``objects.csv`` (input-grid means vs. the VQF view per object) and
``objects_scatter.png``. Outputs go to ``<study>/figures/gt_vqf_detections_el<LOW>_<HIGH>/``.
"""
import argparse
import csv
from pathlib import Path

import numpy as np

from scripts import visualize_viewactive_gt_maxima as maxima
from scripts.visualize_study_vqf_views import crop_box, thumbnail
from viewactive.grid import Grid, combine, elevation_rows

VQF_COLOR = maxima.MAX_COLOR
FP_COLOR = "#cc0000"


def view_detections(db, config, source):
    """{image_id: {"margin", "fp", "fp_rate", "fp_prompts", "correct"}} for one image source."""
    from study import db as store
    from study.labels import outcomes

    vocabulary = config["dataset"]["categories"]
    threshold = config["detection"]["score_threshold"]
    where = f"i.source='{source}'"
    results = outcomes(db, vocabulary, config["detection"], where)
    prompts = store.prompt_scores(db, where)
    categories = dict(db.execute(f"SELECT i.image_id, i.category FROM images i WHERE {where}"))
    out = {}
    for i, o in results.items():
        wrong = sorted((p for p, s in prompts[i].items() if p != categories[i] and s >= threshold),
                       key=lambda p: -prompts[i][p])
        out[i] = {"margin": o["margin"], "fp": o["fp"], "fp_rate": o["fp"] / (len(vocabulary) - 1),
                  "fp_prompts": wrong, "correct": bool(o["correct"])}
    return out


def detections(study, config):
    """{(uid, e, a): view_detections entry} for the ground-truth views."""
    from study import db as store

    data = config["dataset"]
    db = store.connect(study)
    found = view_detections(db, config, "gt")
    rows = {r["image_id"]: r for r in store.images(db, "source='gt'")}
    db.close()
    azimuths = [float(a) for a in data["azimuths"]]
    elevations = [float(e) for e in data["elevations"]]
    return {(rows[i]["uid"], elevations.index(rows[i]["elevation"]), azimuths.index(rows[i]["azimuth"])): d
            for i, d in found.items()}


def label(d):
    if d is None:
        return "no detection data"
    fps = ", ".join(d["fp_prompts"]) if d["fp_prompts"] else "none"
    return f"m {d['margin']:+.2f} | FP: {fps}"


def scatter(rows, path, ylabel="Constrained VQF view"):
    import matplotlib.pyplot as plt

    categories = sorted({r["category"] for r in rows})
    colors = plt.get_cmap("tab20")(np.linspace(0, 1, 20))[:len(categories)]
    fig, axes = plt.subplots(1, 2, figsize=(15, 7))
    for ax, key, title in ((axes[0], "margin", "SAM3 margin"), (axes[1], "fp_rate", "FP rate")):
        values = [r[f"input_mean_{key}"] for r in rows] + [r[f"vqf_{key}"] for r in rows]
        low, high = min(values), max(values)
        pad = 0.05 * (high - low or 1)
        ax.plot([low - pad, high + pad], [low - pad, high + pad], color="#999999", lw=1, zorder=1)
        for category, color in zip(categories, colors):
            members = [r for r in rows if r["category"] == category]
            ax.scatter([r[f"input_mean_{key}"] for r in members], [r[f"vqf_{key}"] for r in members],
                       s=28, color=color, label=category, zorder=2, edgecolor="white", linewidth=.5)
        ax.set_xlim(low - pad, high + pad); ax.set_ylim(low - pad, high + pad)
        ax.set_aspect("equal")
        ax.set_xlabel(f"Mean over the object's input views — {title}")
        ax.set_ylabel(f"{ylabel} — {title}")
        ax.set_title(f"{title} per object (one point per object; grey line: equal)")
        ax.grid(alpha=.3)
    axes[1].legend(loc="center left", bbox_to_anchor=(1.02, .5), frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, required=True)
    parser.add_argument("--elevations", nargs=2, type=float, metavar=("LOW", "HIGH"), required=True)
    args = parser.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    from study import config as settings_io

    maxima.STUDY = args.study
    limits = tuple(args.elevations)
    config = settings_io.read(args.study / "config.json")
    data = maxima.load_data()
    found = detections(args.study, config)
    grid = Grid(tuple(map(float, data["azimuths"])), tuple(map(float, data["elevations"])))
    rows, cols = grid.shape
    allowed = elevation_rows(grid, limits)[:, None]
    out = args.study / "figures" / f"gt_vqf_detections_el{limits[0]:g}_{limits[1]:g}"
    band = f"elevations {limits[0]:+.0f}..{limits[1]:+.0f}°"
    threshold = config["detection"]["score_threshold"]
    order = sorted(range(len(data["uids"])), key=lambda o: (data["categories"][o], data["uids"][o]))
    table = []
    out.mkdir(parents=True, exist_ok=True)
    with PdfPages(out / "objects.pdf") as pdf:
        for o in order:
            uid, category = str(data["uids"][o]), str(data["categories"][o])
            combined = combine(data["norm"][o])
            field = np.where(allowed, combined, np.nan)
            vqf = tuple(map(int, np.unravel_index(np.nanargmax(field), field.shape)))
            image = lambda e, a: args.study / maxima.member({"uid": uid}, (e, a), data)
            box = crop_box([image(e, a) for e in range(rows) for a in range(cols)])
            inputs = [found[(uid, e, a)] for e in range(rows) for a in range(cols) if (uid, e, a) in found]
            chosen = found.get((uid, *vqf))
            summary = {"category": category, "uid": uid, "input_views": len(inputs),
                       "vqf_azimuth": grid.azimuths[vqf[1]], "vqf_elevation": grid.elevations[vqf[0]],
                       "input_mean_margin": float(np.mean([d["margin"] for d in inputs])),
                       "input_mean_fp_rate": float(np.mean([d["fp_rate"] for d in inputs])),
                       "input_any_fp": float(np.mean([d["fp"] > 0 for d in inputs])),
                       "input_negative_margin": float(np.mean([d["margin"] < 0 for d in inputs])),
                       "vqf_margin": chosen["margin"], "vqf_fp_rate": chosen["fp_rate"],
                       "vqf_fp": chosen["fp"], "vqf_fp_prompts": " ".join(chosen["fp_prompts"])}
            table.append(summary)

            fig = plt.figure(figsize=(28, 9.5))
            outer = fig.add_gridspec(1, 2, width_ratios=[cols, 3.4], wspace=.04,
                                     left=.01, right=.99, top=.90, bottom=.07)
            cells = outer[0].subgridspec(rows, cols, wspace=.06, hspace=.36)
            for e in range(rows):
                for a in range(cols):
                    d = found.get((uid, e, a))
                    thumbnail(fig.add_subplot(cells[rows - 1 - e, a]), image(e, a),
                              f"az {grid.azimuths[a]:.0f} el {grid.elevations[e]:+.0f}\n{label(d)}",
                              VQF_COLOR if (e, a) == vqf else None, width=3, fontsize=5.5,
                              title_color=FP_COLOR if d and d["fp"] else "black", box=box)
            side = outer[1].subgridspec(2, 1, height_ratios=[2.2, 1], hspace=.12)
            e, a = vqf
            thumbnail(fig.add_subplot(side[0]), image(e, a),
                      f"Constrained VQF maximum ({band})\naz {grid.azimuths[a]:.0f}°, "
                      f"el {grid.elevations[e]:+.0f}° | VQF Q = {combined[e, a]:.3f}\n{label(chosen)}",
                      VQF_COLOR, width=5, fontsize=12,
                      title_color=FP_COLOR if chosen["fp"] else "black", box=box)
            ax = fig.add_subplot(side[1]); ax.axis("off")
            ax.text(0, 1, f"{'':22s}{'input views':>13s}{'VQF view':>11s}\n"
                    f"{'mean margin':22s}{summary['input_mean_margin']:>+13.3f}{summary['vqf_margin']:>+11.3f}\n"
                    f"{'mean FP rate':22s}{summary['input_mean_fp_rate']:>13.1%}{summary['vqf_fp_rate']:>11.1%}\n"
                    f"{'views with any FP':22s}{summary['input_any_fp']:>13.1%}{float(chosen['fp'] > 0):>11.0%}\n"
                    f"{'views with margin < 0':22s}{summary['input_negative_margin']:>13.1%}"
                    f"{float(chosen['margin'] < 0):>11.0%}\n\n"
                    f"input views: all {len(inputs)} ground-truth views",
                    family="monospace", fontsize=11, va="top", transform=ax.transAxes)
            fig.suptitle(f"{category} | {uid} — SAM3 on every input view vs. the constrained VQF view",
                         fontsize=16)
            fig.text(.01, .02, f"m = SAM3 margin (true category minus strongest wrong category) | FP = wrong "
                     f"prompts scoring ≥ {threshold} (red titles have at least one) | FP rate = FP / "
                     f"{len(config['dataset']['categories']) - 1} wrong prompts | red frame: VQF view | "
                     "all views cropped to one common box", fontsize=10)
            path = out / category / f"{uid}.png"
            path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(path, dpi=110)
            pdf.savefig(fig)
            plt.close(fig)
    with (out / "objects.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(table[0]))
        writer.writeheader()
        writer.writerows(table)
    scatter(table, out / "objects_scatter.png")
    print(f"Saved {len(table)} object pages, objects.csv and objects_scatter.png under {out}", flush=True)


if __name__ == "__main__":
    main()
