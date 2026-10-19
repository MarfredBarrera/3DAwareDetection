"""Learned vs. ground-truth VQF on the 12-object sample of the GT maxima figures.

python -m scripts.visualize_viewactive_learned_vs_gt plan --elevations 0 45
python -m scripts.visualize_viewactive_learned_vs_gt figures --elevations 0 45

Both policies run Algorithm 1 (reach and steps from the run config) from the
same starts with the same optimize.elevation_range. The learned field is the
model's prediction from the start view, rotated to the object frame; the
learned rollout re-predicts at every step. Extract the archive members listed in
``missing_members.txt`` into the output directory before making figures.
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np

from scripts.visualize_viewactive_gt_maxima import member
from viewactive.grid import Grid, combine, elevation_rows, to_absolute, to_relative
from viewactive.optimize import field_policy, rollout

BASE = Path("outputs/viewactive/blender_full_viz")
RUN = BASE / "run"
OUT = Path("outputs/viewactive/learned_vs_gt")
CHANNELS = ("Visible ratio", "Normal entropy", "Visual entropy", "Combined")
COLORS = {"start": "#00bfff", "gt": "#ff3355", "learned": "#b04dff"}
LIMITS = None


def sources():
    root = BASE.parent
    return (OUT, BASE, root/"gt_maxima_viz", root/"entropy_weight_sweep",
            root/"entropy_weight_sweep_visibility1", *root.glob("gt_maxima_viz_el*"), *root.glob("learned_vs_gt*"))


def image_path(name):
    for directory in sources():
        if (directory/name).is_file():
            return directory/name
    raise FileNotFoundError(name)


def ranks(x):
    return np.argsort(np.argsort(x)).astype(float)


def spearman(a, b):
    keep = np.isfinite(a) & np.isfinite(b)
    return float(np.corrcoef(ranks(a[keep]), ranks(b[keep]))[0, 1])


def plan():
    import torch
    from viewactive.evaluate import predict_all
    from viewactive.model import ViewActiveNet

    OUT.mkdir(parents=True, exist_ok=True)
    config = json.loads((RUN/"config.json").read_text())
    grid = Grid.from_config(config["render"])
    reach, steps = config["optimize"]["reach_cells"], config["optimize"]["steps"]
    data = dict(np.load(RUN/"vqf.npz"))
    features = np.load(next((RUN/"features").glob("*[!e].npy")), mmap_mode="r")
    state = torch.load(RUN/"model/best.pt", map_location="cpu", weights_only=False)
    net = ViewActiveNet(state["dim"], state["hidden"], grid.shape)
    net.load_state_dict(state["state_dict"])
    net.eval()
    lookup = {str(uid): o for o, uid in enumerate(data["uids"])}
    allowed = elevation_rows(grid, LIMITS)
    rows, missing, fields = [], [], {}
    for number, old in enumerate(json.loads((BASE/"plan.json").read_text()), 1):
        o = lookup[old["uid"]]
        start = tuple(old["views"][0])
        truth = data["norm"][o]                                            # [3, E, A] absolute
        truth_combined = combine(truth)
        pred = predict_all(net, features[o:o + 1], device="cpu")[0]        # [E, A, 3, E, A] relative
        pred_combined = combine(pred)
        learned_start = to_absolute(pred[start], start[1])                 # [3, E, A] absolute
        paths = {
            "gt": rollout(field_policy(lambda e, a: to_relative(truth_combined, a), grid, reach, LIMITS),
                          start, steps),
            "learned": rollout(field_policy(lambda e, a: pred_combined[e, a], grid, reach, LIMITS),
                               start, steps),
        }
        band = np.where(allowed[:, None], truth_combined, np.nan)
        entry = {k: old[k] for k in ("uid", "category", "split", "shard")}
        entry.update(number=number, index=o, start=list(start),
                     band_best=list(map(int, np.unravel_index(np.nanargmax(band), band.shape))),
                     band_best_quality=float(np.nanmax(band)),
                     spearman_start=[spearman(learned_start[c].ravel(), truth[c].ravel()) for c in range(3)]
                     + [spearman(combine(learned_start).ravel(), truth_combined.ravel())])
        for name, views in paths.items():
            views = [list(map(int, v)) for v in views]
            final = tuple(views[-1])
            entry[name] = {"views": views, "final": list(final),
                           "angles": [float(grid.azimuths[final[1]]), float(grid.elevations[final[0]])],
                           "quality": float(truth_combined[final]),
                           "member": member(old, final, data)}
        entry["start_member"] = member(old, start, data)
        entry["start_quality"] = float(truth_combined[start])
        entry["start_angles"] = [float(grid.azimuths[start[1]]), float(grid.elevations[start[0]])]
        for path in (entry["start_member"], entry["gt"]["member"], entry["learned"]["member"]):
            if not any((d/path).exists() for d in sources()):
                missing.append(path)
        fields[old["uid"]] = learned_start
        rows.append(entry)
    (OUT/"plan.json").write_text(json.dumps({"elevation_range": LIMITS, "reach_cells": reach,
                                             "steps": steps, "rows": rows}, indent=1) + "\n")
    np.savez_compressed(OUT/"learned_start_fields.npz", **fields)
    missing = list(dict.fromkeys(missing))
    (OUT/"missing_members.txt").write_text("\n".join(missing) + ("\n" if missing else ""))
    with (OUT/"scores.csv").open("w") as stream:
        writer = csv.writer(stream)
        writer.writerow(["number", "category", "uid", "start_Q", "gt_final_Q", "learned_final_Q",
                         "band_best_Q", "gt_final_az", "gt_final_el", "learned_final_az", "learned_final_el",
                         *[f"spearman_{c}" for c in ("visible", "normal", "visual", "combined")]])
        for r in rows:
            writer.writerow([r["number"], r["category"], r["uid"], r["start_quality"], r["gt"]["quality"],
                             r["learned"]["quality"], r["band_best_quality"], *r["gt"]["angles"],
                             *r["learned"]["angles"], *r["spearman_start"]])
    print(json.dumps({"objects": len(rows), "missing_images": len(missing),
                      "mean_Q": {k: float(np.mean([r[k]["quality"] for r in rows])) for k in ("gt", "learned")},
                      "mean_spearman_combined": float(np.mean([r["spearman_start"][3] for r in rows]))}, indent=1))


def image_panel(ax, row, which):
    from PIL import Image
    from matplotlib.patches import Rectangle
    name = row["start_member"] if which == "start" else row[which]["member"]
    with Image.open(image_path(name)) as img:
        img = img.convert("RGB")
        img.thumbnail((960, 840))
        ax.imshow(img)
    title = {"start": "Input start", "gt": "Ground-truth VQF policy", "learned": "Learned VQF policy"}[which]
    az, el = row["start_angles"] if which == "start" else row[which]["angles"]
    quality = row["start_quality"] if which == "start" else row[which]["quality"]
    ax.set_title(f"{title}\naz {az:.0f}°, el {el:+.0f}° | true Q = {quality:.3f}", fontsize=11)
    ax.set_xticks([]); ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.add_patch(Rectangle((0, 0), 1, 1, transform=ax.transAxes, fill=False,
                           edgecolor=COLORS[which], linewidth=4))


def field_panel(ax, values, row, grid, title, paths, numbers=False):
    from matplotlib.patches import Rectangle
    ax.imshow(values, origin="lower", cmap="viridis", vmin=0, vmax=1, aspect="auto")
    ax.set_title(title, fontsize=11)
    ax.set_xticks(np.arange(len(grid.azimuths)), labels=[f"{v:.0f}" for v in grid.azimuths], fontsize=7)
    ax.set_yticks(np.arange(len(grid.elevations)), labels=[f"{v:+.0f}" for v in grid.elevations], fontsize=7)
    ax.set_xlabel("Azimuth (°)", fontsize=8)
    ax.set_ylabel("Elevation (°)", fontsize=8)
    if numbers:
        for e in range(values.shape[0]):
            for a in range(values.shape[1]):
                v = values[e, a]
                ax.text(a, e, f"{v:.2f}" if np.isfinite(v) else "—", ha="center", va="center",
                        fontsize=6, color="black" if v > .55 else "white")
    if LIMITS is not None:
        rows = np.flatnonzero(elevation_rows(grid, LIMITS))
        for y in (rows[0] - .5, rows[-1] + .5):
            ax.axhline(y, color="white", linestyle="--", linewidth=1.5, zorder=4)
    for which in paths:
        views = np.array(row[which]["views"])
        for (e0, a0), (e1, a1) in zip(views, views[1:]):
            if abs(a1 - a0) <= 1:   # skip drawing across the azimuth wrap
                ax.plot([a0, a1], [e0, e1], color=COLORS[which], linewidth=1.6, alpha=.9, zorder=5)
    marks = [("start", row["start"], 0)] + [(w, row[w]["final"], .08 + .08 * i) for i, w in enumerate(paths)]
    for which, (e, a), inset in marks:
        ax.add_patch(Rectangle((a - .5 + inset, e - .5 + inset), 1 - 2 * inset, 1 - 2 * inset,
                               fill=False, edgecolor=COLORS[which], linewidth=2.3, zorder=6))


def figures():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    from matplotlib.lines import Line2D

    stored = json.loads((OUT/"plan.json").read_text())
    rows = stored["rows"]
    grid = Grid.from_config(json.loads((RUN/"config.json").read_text())["render"])
    data = dict(np.load(RUN/"vqf.npz"))
    learned = dict(np.load(OUT/"learned_start_fields.npz"))
    band = ("all elevations" if LIMITS is None else f"elevations {LIMITS[0]:+.0f}..{LIMITS[1]:+.0f}° (dashed)")
    legend = [Line2D([0], [0], color=COLORS["start"], lw=3, label="Input start"),
              Line2D([0], [0], color=COLORS["gt"], lw=3, label="Ground-truth VQF policy path / final view"),
              Line2D([0], [0], color=COLORS["learned"], lw=3, label="Learned VQF policy path / final view")]
    footer = (f"Algorithm 1, reach {stored['reach_cells']}, {stored['steps']} steps, targets in {band} | "
              "learned field = prediction from the start view (the policy re-predicts each step) | "
              "true Q = mean of normalized ground-truth channels")
    individual = OUT/"objects"
    individual.mkdir(exist_ok=True)
    links = []
    with PdfPages(OUT/"learned_vs_gt.pdf") as pdf:
        for row in rows:
            fig, axes = plt.subplots(2, 7, figsize=(30, 9),
                                     gridspec_kw={"width_ratios": [1, 1, 1, 1.15, 1.15, 1.15, 1.15]})
            for ax, which in zip(axes[0, :3], ("start", "gt", "learned")):
                image_panel(ax, row, which)
            truth = data["norm"][row["index"]]
            pred = learned[row["uid"]]
            for r, (fields, label, path) in enumerate(((truth, "Ground truth", "gt"), (pred, "Learned", "learned"))):
                for ax, field, title in zip(axes[r, 3:], [*fields, combine(fields)], CHANNELS):
                    field_panel(ax, field, row, grid, f"{label}: {title}", (path,), numbers=True)
            for ax in axes[1, :3]:
                ax.axis("off")
            s = row["spearman_start"]
            axes[1, 0].text(0, .95,
                "Spearman, learned field from start vs. ground truth\n\n"
                + "\n".join(f"{n}: {v:+.2f}" for n, v in zip(CHANNELS, s))
                + f"\n\nTrue Q: start {row['start_quality']:.3f}\n"
                f"  ground-truth policy {row['gt']['quality']:.3f}\n"
                f"  learned policy {row['learned']['quality']:.3f}\n"
                f"  best in band {row['band_best_quality']:.3f}",
                ha="left", va="top", fontsize=11, transform=axes[1, 0].transAxes)
            name = f"{row['number']:02d}_{row['category'].replace('/', '_')}_{row['uid'][:8]}"
            links.append(f"- [{row['number']:02d}. {row['category']} — {row['uid'][:8]}](<objects/{name}.png>)")
            split = "unseen" if row["split"] == "test_unseen" else "seen"
            fig.suptitle(f"{row['number']:02d}. {row['category']} | {split} | {row['uid']}", fontsize=16)
            fig.legend(handles=legend, loc="lower left", bbox_to_anchor=(.01, .035), ncol=3, frameon=False, fontsize=11)
            fig.text(.02, .015, footer, fontsize=9)
            fig.subplots_adjust(left=.02, right=.97, top=.88, bottom=.11, wspace=.30, hspace=.40)
            cb = fig.colorbar(plt.cm.ScalarMappable(norm=plt.Normalize(0, 1), cmap="viridis"),
                              ax=axes[:, 3:].ravel().tolist(), fraction=.012, pad=.015)
            cb.set_label("Quality (higher is better)")
            fig.savefig(individual/f"{name}.png", dpi=130)
            pdf.savefig(fig)
            plt.close(fig)
    pages = (len(rows) + 2) // 3
    for page in range(pages):
        chosen = rows[page * 3:(page + 1) * 3]
        fig, axes = plt.subplots(3, 5, figsize=(24, 12), gridspec_kw={"width_ratios": [1, 1, 1, 1.15, 1.15]})
        for panels, row in zip(axes, chosen):
            for ax, which in zip(panels[:3], ("start", "gt", "learned")):
                image_panel(ax, row, which)
            field_panel(panels[3], combine(data["norm"][row["index"]]), row, grid,
                        "Ground-truth combined VQF", ("gt", "learned"))
            field_panel(panels[4], combine(learned[row["uid"]]), row, grid,
                        f"Learned combined VQF (ρ = {row['spearman_start'][3]:+.2f})", ("gt", "learned"))
            panels[0].text(0, 1.30, f"{row['number']:02d}. {row['category']} | {row['uid'][:8]}",
                           transform=panels[0].transAxes, fontsize=13, weight="bold")
        fig.suptitle(f"Learned vs. ground-truth VQF — {page + 1}/{pages}", fontsize=18)
        fig.legend(handles=legend, loc="lower left", bbox_to_anchor=(.01, .035), ncol=3, frameon=False, fontsize=11)
        fig.text(.02, .015, footer, fontsize=9)
        fig.subplots_adjust(left=.025, right=.98, top=.88, bottom=.10, wspace=.28, hspace=.65)
        fig.savefig(OUT/f"overview_{page + 1}.png", dpi=130)
        plt.close(fig)
    mean = lambda f: np.mean([f(r) for r in rows])
    (OUT/"README.md").write_text(
        "# Learned vs. ground-truth VQF\n\n"
        f"Same 12 objects and starts as `gt_maxima_viz`. {footer}.\n\n"
        f"Mean true Q of the final view: ground-truth policy {mean(lambda r: r['gt']['quality']):.3f}, "
        f"learned policy {mean(lambda r: r['learned']['quality']):.3f}, start {mean(lambda r: r['start_quality']):.3f}. "
        f"Mean Spearman (combined, from start view): {mean(lambda r: r['spearman_start'][3]):+.2f}.\n\n"
        "[All objects (PDF)](learned_vs_gt.pdf) | [Scores (CSV)](scores.csv) | [Plan (JSON)](plan.json)\n\n"
        + "\n".join(f"[Overview {i}](overview_{i}.png)" for i in range(1, pages + 1))
        + "\n\n## Detailed figures\n\n" + "\n".join(links) + "\n")
    print(f"Saved {len(rows)} detailed figures and {pages} overview pages in {OUT}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("plan", "figures"))
    parser.add_argument("--elevations", nargs=2, type=float, metavar=("LOW", "HIGH"),
                        help="optimizer targets only in this elevation band (degrees)")
    args = parser.parse_args()
    if args.elevations:
        LIMITS = tuple(args.elevations)
        OUT = OUT.with_name(f"{OUT.name}_el{LIMITS[0]:g}_{LIMITS[1]:g}")
    {"plan": plan, "figures": figures}[args.stage]()
