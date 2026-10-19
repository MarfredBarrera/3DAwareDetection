"""Reusable azimuth x elevation figures.

Grid panels share angle coordinates; wrong-category bars summarize prompt
scores across the same views. This draws ground-truth margins, SAM3D ambiguity
scores, and before/after comparisons::

    save_panels([margin_heatmap(before, "before"), Images(rgb_paths, "views")],
                azimuths, elevations, "figure.png")
"""
import argparse
from dataclasses import dataclass, field
import math
import multiprocessing
import shutil
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # The container is headless.
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
from PIL import Image, ImageOps

from study import config as settings_io
from study import db as store
from study.labels import outcomes
from study.selectors import selector_names, weights

TILE = 96
MISSING = "#e9ece9"


@dataclass
class Heatmap:
    values: dict                 # (azimuth, elevation) -> float or None
    title: str
    vmin: float | None = None
    vmax: float | None = None
    cmap: str = "magma"
    label: str = ""
    annotate: bool = False       # print each value in its cell
    marks: set = field(default_factory=set)  # cells to outline
    rivals: dict = field(default_factory=dict)  # angle -> wrong categories scoring above true
    codes: dict = field(default_factory=dict)   # category -> compact plot label


@dataclass
class Images:
    paths: dict                  # (azimuth, elevation) -> image path
    title: str
    marks: set = field(default_factory=set)


@dataclass
class WrongCategoryBars:
    margins: dict                # wrong category -> mean (true - wrong) on views it wins
    title: str


def margin_heatmap(values, title, limit=None, **kwargs):
    """Confidence margin (true - strongest wrong): red = misleading, blue = correct."""
    finite = [abs(v) for v in values.values() if v is not None and math.isfinite(v)]
    limit = limit or max(finite, default=1.0) or 1.0
    return Heatmap(values, title, -limit, limit, "coolwarm_r", "confidence margin", **kwargs)


def to_grid(values, azimuths, elevations):
    """(elevation, azimuth) array with NaN for missing or undefined cells."""
    grid = np.full((len(elevations), len(azimuths)), np.nan)
    for row, elevation in enumerate(elevations):
        for col, azimuth in enumerate(azimuths):
            value = values.get((azimuth, elevation))
            if value is not None and math.isfinite(value):
                grid[row, col] = value
    return grid


def montage(paths, azimuths, elevations, tile=TILE):
    """One image with a tile per view; high elevations on top like the heatmaps."""
    canvas = Image.new("RGB", (tile * len(azimuths), tile * len(elevations)), MISSING)
    for row, elevation in enumerate(reversed(elevations)):
        for col, azimuth in enumerate(azimuths):
            path = paths.get((azimuth, elevation))
            if path is not None:
                with Image.open(path) as source:
                    canvas.paste(ImageOps.fit(source.convert("RGB"), (tile, tile)),
                                 (col * tile, row * tile))
    return np.asarray(canvas)


def _axes(ax, azimuths, elevations, title):
    ax.set_xticks(range(len(azimuths)), [f"{a:g}" for a in azimuths])
    ax.set_yticks(range(len(elevations)), [f"{e:g}" for e in elevations])
    ax.set_xticks(np.arange(-0.5, len(azimuths), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(elevations), 1), minor=True)
    ax.grid(which="minor", color="black", linewidth=0.5, alpha=0.25)
    ax.tick_params(which="minor", bottom=False, left=False)
    ax.set_xlabel("azimuth (deg)")
    ax.set_ylabel("elevation (deg)")
    ax.set_title(title, fontsize=10)


def _outline(ax, marks, azimuths, elevations):
    for azimuth, elevation in marks:
        if azimuth in azimuths and elevation in elevations:
            ax.add_patch(Rectangle((azimuths.index(azimuth) - 0.5, elevations.index(elevation) - 0.5),
                                   1, 1, fill=False, edgecolor="lime", linewidth=2))


def draw_heatmap(ax, panel, azimuths, elevations):
    grid = to_grid(panel.values, azimuths, elevations)
    cmap = plt.get_cmap(panel.cmap).copy()
    cmap.set_bad(MISSING)
    image = ax.imshow(grid, origin="lower", aspect="auto", cmap=cmap,
                      vmin=panel.vmin, vmax=panel.vmax)
    if panel.annotate or panel.rivals:
        for row in range(len(elevations)):
            for col in range(len(azimuths)):
                angle = (azimuths[col], elevations[row])
                category_codes = [panel.codes.get(name, name) for name in panel.rivals.get(angle, ())]
                lines = ([f"{grid[row, col]:.2f}"]
                         if panel.annotate and np.isfinite(grid[row, col]) else [])
                lines += [",".join(category_codes[i:i + 4])
                          for i in range(0, len(category_codes), 4)]
                if lines:
                    ax.text(col, row, "\n".join(lines), ha="center", va="center",
                            fontsize=(7.5 if len(category_codes) <= 4 else 6)
                            if category_codes else 7, color="black",
                            bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "none", "pad": 0.4}
                            if category_codes else None)
    _axes(ax, azimuths, elevations, panel.title)
    _outline(ax, panel.marks, azimuths, elevations)
    plt.colorbar(image, ax=ax, shrink=0.8, label=panel.label)


def draw_images(ax, panel, azimuths, elevations):
    ax.imshow(montage(panel.paths, azimuths, elevations),
              extent=(-0.5, len(azimuths) - 0.5, -0.5, len(elevations) - 0.5),
              origin="upper", aspect="auto")
    _axes(ax, azimuths, elevations, panel.title)
    _outline(ax, panel.marks, azimuths, elevations)


def draw_wrong_categories(ax, panel):
    ranked = sorted(panel.margins.items(), key=lambda item: (item[1], item[0]))
    if ranked:
        names, margins = zip(*ranked)
        ax.bar(names, margins, color="#c44e52")
        ax.set_ylim(min(margins) * 1.15, 0)
        ax.tick_params(axis="x", labelrotation=35, labelsize=8)
        for label in ax.get_xticklabels():
            label.set_ha("right")
    else:
        ax.text(0.5, 0.5, "No wrong category scored above the true category",
                ha="center", va="center", transform=ax.transAxes)
        ax.set_xticks([])
        ax.set_ylim(-1, 0)
    ax.set_xlabel("wrong object category")
    ax.set_ylabel("mean confidence margin (true − wrong)")
    ax.set_title(panel.title, fontsize=10)
    ax.grid(axis="y", alpha=0.25)


def save_panels(panels, azimuths, elevations, path, *, title=None, note=None, ncols=None, dpi=110):
    """Draw panels in a grid of ``ncols`` columns and save one PNG."""
    azimuths, elevations = [float(a) for a in azimuths], [float(e) for e in elevations]
    ncols = ncols or len(panels)
    nrows = math.ceil(len(panels) / ncols)
    width, height = 2.4 + 0.8 * len(azimuths), 1.2 + 0.6 * len(elevations)
    fig, axes = plt.subplots(nrows, ncols, figsize=(width * ncols, height * nrows), squeeze=False)
    for ax, panel in zip(axes.flat, panels):
        if isinstance(panel, Heatmap):
            draw_heatmap(ax, panel, azimuths, elevations)
        elif isinstance(panel, Images):
            draw_images(ax, panel, azimuths, elevations)
        elif isinstance(panel, WrongCategoryBars):
            draw_wrong_categories(ax, panel)
        else:
            raise TypeError(f"Unsupported figure panel: {type(panel).__name__}")
    for ax in axes.flat[len(panels):]:
        ax.axis("off")
    if title:
        fig.suptitle(title)
    if note:
        fig.text(0.5, 0.005, note, ha="center", va="bottom", fontsize=9, style="italic")
    fig.tight_layout(rect=(0, 0.03 + 0.02 * note.count("\n") if note else 0, 1, 1))
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi)
    plt.close(fig)
    return Path(path)


def save_bars(groups, path, *, title, ylabel, errors=None):
    """Grouped bars: ``groups`` maps x label -> {series: value}; optional matching errors."""
    labels = list(groups)
    series = list(dict.fromkeys(name for values in groups.values() for name in values))
    width = 0.8 / max(len(series), 1)
    fig, ax = plt.subplots(figsize=(1.2 + 1.1 * len(labels) * max(1, len(series) / 2), 3.6))
    for index, name in enumerate(series):
        values = [groups[label].get(name, np.nan) for label in labels]
        spread = None if errors is None else [errors[label].get(name, 0) for label in labels]
        ax.bar(np.arange(len(labels)) + (index - (len(series) - 1) / 2) * width, values, width,
               yerr=spread, label=name, capsize=2)
    ax.set_xticks(range(len(labels)), labels, rotation=20, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.3)
    if len(series) > 1:
        ax.legend(fontsize=8)
    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return Path(path)


def save_margin_comparison(table, path, *, title, primary, baseline=None):
    """Mean SAM3 margin and its paired mean change from the original input."""
    names = list(table)
    margins = np.array([table[name]["margin"] for name in names])
    baseline = table["input"]["margin"] if baseline is None else baseline
    interval = np.array([table[name]["margin_ci95"] for name in names])
    errors = np.vstack((margins - interval[:, 0], interval[:, 1] - margins))
    colors = ["#555555" if name == "input" else "#dd8452" if name == primary else "#4c72b0"
              for name in names]
    x = np.arange(len(names))
    fig, (absolute, change) = plt.subplots(2, 1, figsize=(max(9, 1.5 * len(names)), 7),
                                            sharex=True)
    absolute.bar(x, margins, color=colors, yerr=errors, capsize=3)
    absolute.axhline(0, color="black", linewidth=0.8)
    absolute.axhline(baseline, color="#555555", linestyle="--", linewidth=1,
                     label="original input mean")
    absolute.set_ylabel("mean confidence margin")
    absolute.set_title("Mean margin (95% intervals resample objects)")
    absolute.legend(fontsize=8)
    change.bar(x, margins - baseline, color=colors)
    change.axhline(0, color="black", linewidth=0.8)
    change.set_ylabel("change vs original input")
    change.set_title("Mean margin change on the same input views")
    change.set_xticks(x, names, rotation=25, ha="right")
    for ax in (absolute, change):
        ax.grid(axis="y", alpha=0.25)
        ax.set_axisbelow(True)
    fig.suptitle(title)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return Path(path)


NOTE = "gt azimuth: object frame (semantic front where reviewed); after-maps are indexed by input view"


def _angle(row):
    return (row["azimuth"], row["elevation"])


def _mean_map(rows, values):
    """{(az, el): mean of values over rows at that angle} for rows with a value."""
    cells = {}
    for row in rows:
        if row["image_id"] in values:
            cells.setdefault(_angle(row), []).append(values[row["image_id"]])
    return {angle: float(np.mean(v)) for angle, v in cells.items()}


def _higher_scoring_categories(scores, true_category):
    """Every wrong category with a strictly higher SAM3 score, highest first."""
    true_score = scores.get(true_category)
    if true_score is None:
        return ()
    return tuple(sorted((name for name, score in scores.items()
                         if name != true_category and score > true_score),
                        key=lambda name: (-scores[name], name)))


def _category_key(codes):
    entries = [f"{code}={name}" for name, code in codes.items()]
    return "Above true (code=category): " + "  ".join(entries[:6]) + (
        "\n" + "  ".join(entries[6:]) if len(entries) > 6 else "")


def _wrong_category_bars(members, scores, true_category):
    margins = {}
    for row in members:
        view_scores = scores.get(row["image_id"], {})
        if true_category not in view_scores:
            continue
        for name in _higher_scoring_categories(view_scores, true_category):
            margins.setdefault(name, []).append(view_scores[true_category] - view_scores[name])
    return WrongCategoryBars({name: float(np.mean(values)) for name, values in margins.items()},
                             "wrong categories above true score (winning views only)")


def study_figures(db, config, gt, selections, summary, object_figures=True, workers=1):
    output = Path(config["output_dir"])
    figures = output / "figures"
    dataset = config["dataset"]
    azimuths, elevations = dataset["azimuths"], dataset["elevations"]
    primary = config["analysis"]["primary_selector"]
    allowed = config["analysis"].get("figure_selectors")
    rows = store.images(db, "source='gt'")
    paths = []

    # Goal 1: where do ground-truth views mislead SAM3?
    groups = {"all categories": rows}
    for category in dataset["categories"]:
        groups[category] = [row for row in rows if row["category"] == category]
    for name, members in groups.items():
        accuracy = _mean_map(members, {i: float(o["correct"]) for i, o in gt.items()})
        false_positive = _mean_map(members, {i: float(o["any_fp"]) for i, o in gt.items()})
        margin = _mean_map(members, {i: o["margin"] for i, o in gt.items()})
        paths.append(save_panels(
            [Heatmap(accuracy, "top-1 accuracy", 0, 1, "viridis", annotate=True),
             Heatmap(false_positive, "views with a false positive", 0, 1, "magma", annotate=True),
             margin_heatmap(margin, "mean confidence margin", annotate=True)],
            azimuths, elevations, figures / "before" / f"{name.replace(' ', '_')}.png",
            title=f"Before: SAM3 on ground-truth views ({name})", note=NOTE))

    # Goal 2: does choosing a SAM3D view help, and where?
    if selections:
        by_input = {row["image_id"]: row for row in rows}
        reconstructed = [by_input[i] for i in selections]
        panels = []
        for name in ("input", "identity", primary):
            if allowed is not None and name not in allowed:
                continue
            values = {i: s[name]["any_fp"] for i, s in selections.items() if name in s}
            panels.append(Heatmap(_mean_map(reconstructed, values), f"any false positive: {name}",
                                  0, 1, "magma", annotate=True))
        paths.append(save_panels(panels, azimuths, elevations, figures / "after" / "false_positives.png",
                                 title="False-positive rate by input view", note=NOTE))
        for subset, data in summary["subsets"].items():
            table = {name: row for name, row in data["selectors"].items()
                     if allowed is None or name in allowed}
            slug = subset.split(" (")[0].replace(" ", "_")
            paths.append(save_bars(
                {name: {"top-1 accuracy": row["correct"], "any false positive": row["any_fp"]}
                 for name, row in table.items()},
                figures / "after" / f"summary_{slug}.png",
                title=f"Before vs after ({subset}, n={next(iter(table.values()))['n']})",
                ylabel="fraction of inputs"))
            paths.append(save_margin_comparison(
                table, figures / "after" / f"margins_{slug}.png",
                title=f"Confidence margin ({subset}, n={next(iter(table.values()))['n']})",
                primary=primary, baseline=data["selectors"]["input"]["margin"]))

    if object_figures:
        if allowed is not None:
            omitted = set(SELECTOR_FIGURES) - set(allowed)
            for category in dataset["categories"]:
                for selector in omitted:
                    old = figures / "objects" / category / selector
                    if old.is_dir():
                        shutil.rmtree(old)
        margins = {i: o["margin"] for i, o in gt.items()}
        finite = [abs(v) for v in margins.values()]
        limit = max(finite, default=1.0)
        tasks = [(config, category, uid, members, {r["image_id"]: margins.get(r["image_id"]) for r in members},
                  {r["image_id"]: selections[r["image_id"]] for r in members if r["image_id"] in selections},
                  limit, figures / "objects" / category)
                 for (category, uid), members in _objects(rows).items()]
        if workers > 1 and len(tasks) > 1:
            with multiprocessing.get_context("spawn").Pool(min(workers, len(tasks))) as pool:
                for saved in pool.imap_unordered(_object_task, tasks):
                    paths += saved
        else:
            paths += [path for task in tasks for path in _object_task(task, db)]
    return paths


def _object_task(task, db=None):
    """One object's figures; worker processes open their own read connection."""
    own = db is None
    db = store.connect(task[0]["output_dir"]) if own else db
    try:
        return object_figure(db, *task)
    finally:
        if own:
            db.close()


def _objects(rows):
    grouped = {}
    for row in rows:
        grouped.setdefault((row["category"], row["uid"]), []).append(row)
    return grouped


SELECTOR_FIGURES = {"view_relative_margin": "largest true-category margin over rivals (uses labels)",
                    "oracle": "best SAM3D view per input (uses labels)",
                    "detector_entropy": "lowest SAM3 score entropy per input",
                    "detector_gap": "largest SAM3 top-1 minus top-2 gap per input"}


def save_view_comparison(config, category, uid, members, before, picked_paths,
                         after, selector, path, *, limit=None):
    """One standard two-column comparison, indexed by each input camera."""
    angles = [_angle(row) for row in members]
    before_values = {angle: before.get(angle) for angle in angles}
    after_values = {angle: after.get(angle) for angle in angles}
    finite = [abs(value) for value in (*before_values.values(), *after_values.values())
              if value is not None and math.isfinite(value)]
    limit = limit or max(finite, default=1.0) or 1.0
    output = Path(config["output_dir"])
    panels = [
        Images({_angle(row): output / row["path"] for row in members}, "input images"),
        Images(picked_paths, f"optimized viewpoints: {selector}"),
        margin_heatmap(before_values, "input confidence margin", limit, annotate=True),
        margin_heatmap(after_values, "optimized confidence margin", limit, annotate=True),
    ]
    dataset = config["dataset"]
    return save_panels(panels, dataset["azimuths"], dataset["elevations"], path,
                       title=f"{category} / {uid}: {selector}",
                       note="Cells align by input camera; chosen viewpoints use the reconstruction frame",
                       ncols=2)


def object_figure(db, config, category, uid, members, margins, selections, limit, directory):
    """Per-object figures under ``directory``; returns their paths.

    ``before/<uid>.png``: before margin, wrong-category bars, ground-truth views.
    With reconstructions, also ``after/<uid>.png`` (identity and primary-selector
    margins and picks) and ``<selector>/<uid>.png`` for each of ``SELECTOR_FIGURES``
    (input and chosen views on top, before and after margins below).
    """
    directory = Path(directory)
    dataset = config["dataset"]
    primary = config["analysis"]["primary_selector"]
    done = [r for r in members if r["image_id"] in selections]
    ids = sorted(r["image_id"] for r in members)
    scores = store.prompt_scores(db, f"i.image_id IN ({','.join('?' for _ in ids)})", ids)
    before = {_angle(r): margins.get(r["image_id"]) for r in members}
    panels = [margin_heatmap(before, "before: ground-truth view", limit),
              _wrong_category_bars(members, scores, category),
              Images({_angle(r): Path(config["output_dir"]) / r["path"] for r in members},
                     "ground-truth views")]
    saved = [save_panels(panels, dataset["azimuths"], dataset["elevations"], directory / "before" / f"{uid}.png",
                        title=f"{category} / {uid}", note="GT angles use the object frame", ncols=1)]
    if done:
        after_panels = []
        for name in ("identity", primary):
            values = {_angle(r): selections[r["image_id"]][name]["margin"]
                      for r in done if name in selections[r["image_id"]]}
            after_panels.append(margin_heatmap(values, f"after: {name}", limit))
        after_panels.append(Images(_picks(db, config, done, selections, primary),
                                   f"after: views chosen by {primary}"))
        saved.append(save_panels(after_panels, dataset["azimuths"], dataset["elevations"],
                                 directory / "after" / f"{uid}.png",
                                 title=f"{category} / {uid}: reconstructed views",
                                 note="After maps indexed by input view", ncols=1))
        allowed = config["analysis"].get("figure_selectors") or list(SELECTOR_FIGURES)
        for name in allowed:
            if name == "input":
                continue
            after = {_angle(r): selections[r["image_id"]][name]["margin"]
                     for r in done if name in selections[r["image_id"]]}
            saved.append(save_view_comparison(
                config, category, uid, members, before,
                _picks(db, config, done, selections, name), after, name,
                directory / name / f"{uid}.png", limit=limit))
    return saved


def _picks(db, config, inputs, selections, selector):
    """{input angle: path of the SAM3D candidate ``selector`` chose for that input}."""
    pick_ids = {r["image_id"]: selections[r["image_id"]][selector]["pick_id"]
                for r in inputs if selector in selections[r["image_id"]]}
    pick_rows = {row["image_id"]: row for row in store.images(
        db, f"image_id IN ({','.join(map(str, pick_ids.values())) or 'NULL'})")}
    return {_angle(r): Path(config["output_dir"]) / pick_rows[pick_ids[r["image_id"]]]["path"]
            for r in inputs if r["image_id"] in pick_ids}


def reconstruction_figure(db, config, input_key, path):
    """Every candidate of one reconstruction: SAM3 margin, each ambiguity metric, picks."""
    recon = config["reconstruction"]
    metrics = list(config["ambiguity"])
    (source,) = store.images(db, "key=?", (input_key,)) or [None]
    if source is None:
        raise ValueError(f"Unknown input image key {input_key!r}")
    candidates = store.images(db, "parent_id=?", (source["image_id"],))
    if not candidates:
        raise ValueError(f"{input_key} has no reconstruction")
    result = outcomes(db, config["dataset"]["categories"], config["detection"],
                      "i.parent_id=?", (source["image_id"],))
    scores = store.view_scores(db, "i.parent_id=?", (source["image_id"],))
    prompt_scores = store.prompt_scores(db, "i.parent_id=?", (source["image_id"],))
    margins = {i: o["margin"] for i, o in result.items()}
    angle = {c["image_id"]: _angle(c) for c in candidates}
    codes = {name: str(index + 1) for index, name in enumerate(config["dataset"]["categories"])}
    rivals = {angle[i]: _higher_scoring_categories(prompt_scores.get(i, {}), source["category"])
              for i in result}
    marks = {}
    for name in selector_names(metrics):
        picked = weights(name, list(result), scores, margins, metrics, angle, prompt_scores,
                         config["analysis"].get("combined_weights")) or {}
        marks[name] = {angle[i] for i in picked}
    primary = config["analysis"]["primary_selector"]
    panels = [margin_heatmap({angle[i]: m for i, m in margins.items()}, "SAM3 margin (oracle marked)",
                             marks=marks["oracle"], annotate=True, rivals=rivals, codes=codes)]
    for metric in metrics:
        values = {angle[i]: s.get(metric) for i, s in scores.items()}
        panels.append(Heatmap(values, f"{metric} (lower = less ambiguous)", 0, 1,
                              marks=marks[f"ambiguity:{metric}"], annotate=True))
    panels.append(Images({angle[c["image_id"]]: Path(config["output_dir"]) / c["path"] for c in candidates},
                         f"candidates ({primary} marked)", marks=marks.get(primary, set())))
    return save_panels(panels, recon["candidate_azimuths"], recon["candidate_elevations"], path,
                       title=f"{source['category']} {input_key}", ncols=3,
                       note="candidate angles are relative to the input camera; (180, 0) is the input view"
                            "\n" + _category_key(codes))


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, required=True, help="study YAML")
    parser.add_argument("kind", choices=("object", "reconstruction"))
    parser.add_argument("key", help="object uid, or input image key such as gt/<uid>/a30_e15")
    parser.add_argument("--out", type=Path, help="default: <output_dir>/figures/explore/ "
                        "(a directory for object figures)")
    args = parser.parse_args()
    config = settings_io.load(args.config)
    output = Path(config["output_dir"])
    db = store.connect(output)
    name = args.key.replace("/", "_") + ".png"
    path = args.out or output / "figures" / "explore" / f"{args.kind}_{name}"
    if args.kind == "reconstruction":
        print(reconstruction_figure(db, config, args.key, path))
        return
    members = store.images(db, "source='gt' AND uid=?", (args.key,))
    if not members:
        raise ValueError(f"Unknown object {args.key!r}")
    gt = outcomes(db, config["dataset"]["categories"], config["detection"], "i.uid=? AND i.source='gt'",
                  (args.key,))
    selections = {}
    for row in db.execute("SELECT * FROM selections WHERE input_id IN (SELECT image_id FROM images"
                          " WHERE uid=? AND source='gt')", (args.key,)):
        selections.setdefault(row["input_id"], {})[row["selector"]] = dict(row)
    margins = {i: o["margin"] for i, o in gt.items()}
    directory = args.out or output / "figures" / "explore" / f"object_{args.key}"
    for saved in object_figure(db, config, members[0]["category"], args.key, members, margins,
                               selections, max((abs(v) for v in margins.values()), default=1.0),
                               directory):
        print(saved)


if __name__ == "__main__":
    main()
