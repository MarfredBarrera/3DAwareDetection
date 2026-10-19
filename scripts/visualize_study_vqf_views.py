"""Per object of a viewpoint study: every ground-truth input view, the
constrained ground-truth VQF maximum, and the best SAM3-margin view.

python -m scripts.visualize_study_vqf_views --study outputs/upright --elevations 0 45

The VQF is built as in ``scripts.visualize_viewactive_gt_maxima --study``
(stored ambiguity scores of the ground-truth views). Both selected views are
searched only in the elevation band. The margin view uses SAM3 labels
(true-category score minus strongest wrong category). Writes one PNG per object
under ``<study>/figures/gt_vqf_views_el<LOW>_<HIGH>/<category>/`` and one PDF.
"""
import argparse
from pathlib import Path

import numpy as np

from scripts import visualize_viewactive_gt_maxima as maxima
from viewactive.grid import Grid, combine, elevation_rows

VQF_COLOR = maxima.MAX_COLOR
MARGIN_COLOR = "#22aa55"


def margins(study, config):
    """{(uid, elevation index, azimuth index): SAM3 margin} for the ground-truth views."""
    from study import db as store
    from study.labels import outcomes

    data = config["dataset"]
    db = store.connect(study)
    results = outcomes(db, data["categories"], config["detection"], "i.source='gt'")
    rows = {r["image_id"]: r for r in store.images(db, "source='gt'")}
    db.close()
    azimuths = [float(a) for a in data["azimuths"]]
    elevations = [float(e) for e in data["elevations"]]
    return {(rows[i]["uid"], elevations.index(rows[i]["elevation"]), azimuths.index(rows[i]["azimuth"])):
            o["margin"] for i, o in results.items()}


def crop_box(paths, pad=0.08, masks=None):
    """One square box around the union of the objects' masks in all ``paths`` (rgb
    renders with ``.modal.png`` masks, or the given ``masks``), so every view keeps the same scale."""
    from PIL import Image

    union = None
    for n, path in enumerate(paths):
        mask_path = masks[n] if masks else path.with_name(path.name.replace(".rgb.png", ".modal.png"))
        if not mask_path.is_file():
            continue
        with Image.open(mask_path) as mask:
            box = mask.convert("L").getbbox()
            size = mask.size
        if box:
            union = box if union is None else (min(union[0], box[0]), min(union[1], box[1]),
                                               max(union[2], box[2]), max(union[3], box[3]))
    if union is None:
        return None
    cx, cy = (union[0] + union[2]) / 2, (union[1] + union[3]) / 2
    half = max(union[2] - union[0], union[3] - union[1]) * (0.5 + pad)
    half = min(half, max(size) / 2)
    cx, cy = min(max(cx, half), size[0] - half), min(max(cy, half), size[1] - half)
    return tuple(int(round(v)) for v in (cx - half, cy - half, cx + half, cy + half))


def thumbnail(ax, path, title, color=None, width=2, fontsize=6, title_color="black", box=None):
    from PIL import Image
    from matplotlib.patches import Rectangle

    if path.is_file():
        with Image.open(path) as image:
            image = image.convert("RGB")
            ax.imshow(image.crop(box) if box else image)
    ax.set_xticks([]); ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_title(title, fontsize=fontsize, pad=2, color=title_color)
    if color:
        ax.add_patch(Rectangle((0, 0), 1, 1, transform=ax.transAxes, fill=False,
                               edgecolor=color, linewidth=width))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, required=True)
    parser.add_argument("--elevations", nargs=2, type=float, metavar=("LOW", "HIGH"), required=True)
    args = parser.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    from matplotlib.lines import Line2D
    from study import config as settings_io

    maxima.STUDY = args.study
    limits = tuple(args.elevations)
    config = settings_io.read(args.study / "config.json")
    data = maxima.load_data()
    margin = margins(args.study, config)
    grid = Grid(tuple(map(float, data["azimuths"])), tuple(map(float, data["elevations"])))
    rows, cols = grid.shape
    allowed = elevation_rows(grid, limits)[:, None]
    out = args.study / "figures" / f"gt_vqf_views_el{limits[0]:g}_{limits[1]:g}"
    band = f"elevations {limits[0]:+.0f}..{limits[1]:+.0f}°"
    legend = [Line2D([0], [0], color=VQF_COLOR, lw=3, label=f"Constrained VQF maximum ({band})"),
              Line2D([0], [0], color=MARGIN_COLOR, lw=3, label=f"Best SAM3 margin ({band}; uses labels)")]
    order = sorted(range(len(data["uids"])), key=lambda o: (data["categories"][o], data["uids"][o]))
    with PdfPages(out.with_suffix(".pdf")) as pdf:
        for o in order:
            uid, category = str(data["uids"][o]), str(data["categories"][o])
            combined = combine(data["norm"][o])
            field = np.where(allowed, combined, np.nan)
            vqf = tuple(map(int, np.unravel_index(np.nanargmax(field), field.shape)))
            values = np.array([[margin.get((uid, e, a), np.nan) for a in range(cols)] for e in range(rows)])
            values = np.where(allowed, values, np.nan)
            best = tuple(map(int, np.unravel_index(np.nanargmax(values), values.shape)))
            image = lambda e, a: args.study / maxima.member({"uid": uid}, (e, a), data)
            box = crop_box([image(e, a) for e in range(rows) for a in range(cols)])

            fig = plt.figure(figsize=(26, 9.5))
            outer = fig.add_gridspec(1, 2, width_ratios=[cols, 3.2], wspace=.04,
                                     left=.01, right=.99, top=.90, bottom=.07)
            cells = outer[0].subgridspec(rows, cols, wspace=.04, hspace=.32)
            for e in range(rows):
                for a in range(cols):
                    ax = fig.add_subplot(cells[rows - 1 - e, a])     # highest elevation on top
                    m = margin.get((uid, e, a))
                    color = VQF_COLOR if (e, a) == vqf else MARGIN_COLOR if (e, a) == best else None
                    thumbnail(ax, image(e, a),
                              f"az {grid.azimuths[a]:.0f} el {grid.elevations[e]:+.0f}\n"
                              + ("m = —" if m is None else f"m = {m:+.2f}"),
                              color, width=3, title_color="black" if m is None or m >= 0 else "#cc0000",
                              box=box)
                    if (e, a) == vqf and (e, a) == best:
                        ax.patches[-1].set_edgecolor(VQF_COLOR)
                        ax.patches[-1].set_linestyle((0, (4, 4)))
                        from matplotlib.patches import Rectangle
                        ax.add_patch(Rectangle((.04, .04), .92, .92, transform=ax.transAxes, fill=False,
                                               edgecolor=MARGIN_COLOR, linewidth=3))
            side = outer[1].subgridspec(2, 1, hspace=.18)
            for slot, (cell, title, color) in enumerate((
                    (vqf, "Constrained VQF maximum", VQF_COLOR),
                    (best, "Best SAM3 margin", MARGIN_COLOR))):
                e, a = cell
                thumbnail(fig.add_subplot(side[slot]), image(e, a),
                          f"{title}\naz {grid.azimuths[a]:.0f}°, el {grid.elevations[e]:+.0f}° | "
                          f"VQF Q = {combined[e, a]:.3f} | margin {values[e, a]:+.2f}",
                          color, width=5, fontsize=12, box=box)
            fig.suptitle(f"{category} | {uid} — input views (rows: elevation, columns: azimuth)", fontsize=16)
            fig.legend(handles=legend, loc="lower left", ncol=2, frameon=False, fontsize=11)
            fig.text(.36, .02, "m = SAM3 margin (true category minus strongest wrong category; red if negative) | "
                     "VQF = equal mean of per-object normalized visible ratio, normal entropy, visual entropy | "
                     "all views of an object cropped to one common box",
                     fontsize=10)
            path = out / category / f"{uid}.png"
            path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(path, dpi=110)
            pdf.savefig(fig)
            plt.close(fig)
    print(f"Saved {len(order)} object pages under {out} and {out.with_suffix('.pdf')}", flush=True)


if __name__ == "__main__":
    main()
