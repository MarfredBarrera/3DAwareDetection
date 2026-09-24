"""Plot per-object recognition score margins across viewpoints.

Each heatmap cell is the strongest incorrect category score minus the true
category score for one object. Positive values indicate misleading views.
"""
from __future__ import annotations

# Force a headless backend before pyplot: the container sets EGL_PLATFORM=
# surfaceless and has no display, so any interactive backend would fail on import.
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAVE_MPL = True
except Exception as error:  # pragma: no cover - depends on the runtime image
    HAVE_MPL = False
    _MPL_ERROR = error

import argparse
import io
import json
import math
import re
import shutil
from pathlib import Path

import numpy as np

from experiments.viewpoint_sensitivity import metrics, store
from experiments.viewpoint_sensitivity.summarize_outputs import write_summary

AZIMUTH_NOTE = "azimuth = dataset frame; semantic front only where explicitly verified"
PLOT_CMAP = "magma"
RGB_THUMBNAIL_SIZE = 96
COLLECTION_DIRNAME = "sam3d_inputs"


def _read_rows(path):
    rows = []
    with open(path) as stream:
        for line in stream:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _grid(cells, categories, az_values, el_values, field):
    """Build an (n_categories?, n_el, n_az) value grid for one metric field."""
    az_index = {a: i for i, a in enumerate(az_values)}
    el_index = {e: i for i, e in enumerate(el_values)}
    grid = np.full((len(el_values), len(az_values)), np.nan, dtype=float)
    for (cat, az, el), cell in cells.items():
        if cat in categories and az in az_index and el in el_index:
            grid[el_index[el], az_index[az]] = cell[field]
    return grid


def _object_score_grids(rows, az_values, el_values):
    """One signed score-margin grid per object, without cross-object averaging."""
    az_index = {az: i for i, az in enumerate(az_values)}
    el_index = {el: i for i, el in enumerate(el_values)}
    grids = {}
    seen = set()
    counts = {}
    for row in rows:
        category, uid = row["category"], row["uid"]
        az, el = float(row["azimuth"]), float(row["elevation"])
        key = (category, uid)
        view = (key, az, el)
        if view in seen:
            raise ValueError(f"duplicate object viewpoint: {view}")
        seen.add(view)
        counts[key] = counts.get(key, 0) + 1
        grid = grids.setdefault(key, np.full((len(el_values), len(az_values)), np.nan))
        if row.get("top1_correct") is None:
            continue  # Invisible targets have no meaningful recognition margin.
        _, incorrect_score, true_score = _highest_scoring_incorrect(row)
        grid[el_index[el], az_index[az]] = incorrect_score - true_score
    expected = len(az_values) * len(el_values)
    for key in grids:
        count = counts[key]
        if count != expected:
            raise ValueError(f"incomplete viewpoint grid for object {key}: {count}/{expected}")
    return grids


def _object_rgb_records(render_dir, object_keys, az_values, el_values):
    """Find the matching RGB image for every plotted object and viewpoint."""
    wanted = {(az, el) for el in el_values for az in az_values}
    records = {key: {} for key in object_keys}
    index_path = render_dir / "index.jsonl"
    if not index_path.is_file():
        raise FileNotFoundError(f"Missing render index for RGB grids: {index_path}")
    for record in _read_rows(index_path):
        key = (record["category"], record["uid"])
        if key not in records:
            continue
        angle = (float(record["azimuth"]), float(record["elevation"]))
        if angle in wanted:
            if angle in records[key]:
                raise ValueError(f"duplicate RGB viewpoint for object {key}: {angle}")
            records[key][angle] = render_dir / record["files"]["rgb"]
    for key, views in records.items():
        missing = wanted - views.keys()
        if missing:
            raise ValueError(f"RGB grid for object {key} is missing {sorted(missing)[:4]}")
    return records


def _save_object_score_margin(grid, rgb_paths, category, uid, az_values, el_values, path,
                              color_limit):
    fig, (ax, rgb_ax) = plt.subplots(
        1, 2, figsize=(2.4 + 1.4 * len(az_values), 1.5 + 0.7 * len(el_values)))
    im = _heatmap(ax, grid, az_values, el_values,
                  f"{category} / {uid}: incorrect − true score",
                  vmin=-color_limit, vmax=color_limit, cmap="coolwarm")
    fig.colorbar(im, ax=ax, shrink=0.8, label="score_minus_true_category")
    _rgb_grid(rgb_ax, _rgb_montage(rgb_paths, az_values, el_values),
              az_values, el_values, f"RGB views: {uid}")
    fig.text(0.5, 0.005, AZIMUTH_NOTE, ha="center", fontsize=8, style="italic")
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(path, dpi=120)
    plt.close(fig)


def _overall_cells(rows):
    """Collapse categories: aggregate every row under a single 'overall' label."""
    collapsed = [dict(row, category="overall") for row in rows]
    return metrics.aggregate_viewpoints(collapsed)


def _rank_viewpoint_bands(rows, cells, category, count):
    """Rank disjoint viewpoint bands by mean strongest-wrong minus true score."""
    margins = {}
    for row in rows:
        if row["category"] != category or row.get("top1_correct") is None:
            continue
        angle = (float(row["azimuth"]), float(row["elevation"]))
        _, incorrect_score, true_score = _highest_scoring_incorrect(row)
        margins.setdefault(angle, []).append(incorrect_score - true_score)
    ranked = []
    for (cat, az, el), cell in cells.items():
        if cat != category:
            continue
        values = margins.get((float(az), float(el)), [])
        if values:
            ranked.append((sum(values) / len(values), float(az), float(el), cell))
    if len(ranked) < 2 * count:
        raise ValueError(
            f"{category} has {len(ranked)} scored viewpoints; need at least "
            f"{2 * count} to select {count} disjoint high/low viewpoints")
    ascending = sorted(ranked, key=lambda item: (item[0], item[2], item[1]))
    high_accuracy = ascending[:count]  # smaller wrong-minus-true margin is better
    high_angles = {(item[1], item[2]) for item in high_accuracy}
    low_accuracy = [item for item in reversed(ascending)
                    if (item[1], item[2]) not in high_angles][:count]
    return {"high_accuracy": high_accuracy, "low_accuracy": low_accuracy}


def _safe_component(value):
    """Make category/UID values safe as one portable path component."""
    component = re.sub(r"[^A-Za-z0-9._-]+", "_", str(value)).strip("._")
    return component or "unnamed"


def _highest_scoring_incorrect(row):
    """Return the strongest non-ground-truth prompt using query-stage scores."""
    prompt_results = row.get("prompt_results")
    if not isinstance(prompt_results, dict):
        raise ValueError("result row is missing prompt_results needed for confusion metadata")
    category = row["category"]
    vocabulary = row.get("vocabulary", list(prompt_results))
    candidates = []
    for prompt in vocabulary:
        if prompt == category:
            continue
        try:
            score = float(prompt_results[prompt]["best_score"])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"invalid prompt score for {prompt!r}") from error
        if math.isfinite(score):
            candidates.append((prompt, score))
    if not candidates:
        raise ValueError(f"result row for {category!r} has no incorrect-category scores")
    # max() preserves vocabulary order for ties, matching top1_prediction.
    incorrect_category, incorrect_score = max(candidates, key=lambda item: item[1])
    try:
        true_score = float(prompt_results[category]["best_score"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"result row has no valid true-category score for {category!r}") from error
    return incorrect_category, incorrect_score, true_score


def _collect_sam3d_inputs(rows, cells, render_dir, output, categories,
                          viewpoints_per_band):
    """Copy all sampled-object RGBs at the best/worst category viewpoints.

    Selection uses the category mean of strongest incorrect score minus true
    category score. Smaller means the true category scores higher. Every sampled
    object at each selected viewpoint is copied with its RGB unchanged.
    """
    render_records = {}
    index_path = render_dir / "index.jsonl"
    if not index_path.is_file():
        raise FileNotFoundError(f"Missing render index for SAM3D inputs: {index_path}")
    for record in _read_rows(index_path):
        key = (record["uid"], float(record["azimuth"]), float(record["elevation"]))
        if key in render_records:
            raise ValueError(f"render index contains duplicate RGB view {key}")
        render_records[key] = record

    destination = output / COLLECTION_DIRNAME
    staging = output / f".{COLLECTION_DIRNAME}.tmp"
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)
    manifest = []
    confusion_records = {"high_accuracy": [], "low_accuracy": []}
    selections = {}
    try:
        for category in categories:
            bands = _rank_viewpoint_bands(
                rows, cells, category, viewpoints_per_band)
            selections[category] = {}
            category_component = _safe_component(category)
            for band, ranked in bands.items():
                selections[category][band] = []
                selected_angles = {}
                for rank, (score_margin, azimuth, elevation, cell) in enumerate(ranked, 1):
                    angle = (azimuth, elevation)
                    selected_angles[angle] = (rank, score_margin, cell)
                    selections[category][band].append({
                        "rank": rank, "azimuth": azimuth, "elevation": elevation,
                        "score_minus_true_category": score_margin,
                        "accuracy": float(cell["accuracy"]), "correct": int(cell["correct"]),
                        "incorrect": int(cell["incorrect"]), "n": int(cell["n"]),
                    })

                band_rows = [row for row in rows
                             if row["category"] == category
                             and (float(row["azimuth"]), float(row["elevation"]))
                             in selected_angles]
                for row in sorted(band_rows, key=lambda item: (
                        selected_angles[(float(item["azimuth"]),
                                         float(item["elevation"]))][0], item["uid"])):
                    azimuth, elevation = float(row["azimuth"]), float(row["elevation"])
                    rank, score_margin, cell = selected_angles[(azimuth, elevation)]
                    key = (row["uid"], azimuth, elevation)
                    if key not in render_records:
                        raise ValueError(f"result has no matching render RGB view: {key}")
                    source_rel = render_records[key]["files"]["rgb"]
                    source = render_dir / source_rel
                    if not source.is_file():
                        raise FileNotFoundError(f"render index references missing RGB: {source}")
                    viewpoint_dir = (
                        staging / category_component / _safe_component(row["uid"]) / band
                        / f"viewpoint_(azimuth_{azimuth:g}_elevation_{elevation:g})"
                    )
                    viewpoint_dir.mkdir(parents=True)
                    target = viewpoint_dir / f"rgb{source.suffix.lower()}"
                    shutil.copy2(source, target)
                    incorrect_category, incorrect_score, true_score = (
                        _highest_scoring_incorrect(row))
                    metadata = {
                        "schema_version": 1,
                        "quality_band": band,
                        "object": {
                            "id": row["uid"],
                            "true_category": category,
                            "true_category_score": true_score,
                        },
                        "viewpoint": {
                            "azimuth_degrees": azimuth,
                            "elevation_degrees": elevation,
                            "coordinate_frame": (
                                "azimuth 0 faces reviewed semantic front; +Y up"
                                if render_records[key].get("semantic_front_verified") else
                                "geometric azimuth; semantic front unverified; +Y up"
                            ),
                            "rank_within_band": rank,
                            "regime": render_records[key].get("view_regime", "unspecified"),
                        },
                        "orientation": {
                            "review": render_records[key].get("upright_verification"),
                            "semantic_front_verified": render_records[key].get("semantic_front_verified", False),
                        },
                        "viewpoint_selection": {
                            "metric": "score_minus_true_category",
                            "mean_score_minus_true_category": score_margin,
                            "lower_is_better": True,
                        },
                        "viewpoint_accuracy": {
                            "top1_accuracy": float(cell["accuracy"]),
                            "correct_objects": int(cell["correct"]),
                            "incorrect_objects": int(cell["incorrect"]),
                            "total_objects": int(cell["n"]),
                        },
                        "highest_scoring_incorrect_object": {
                            "category": incorrect_category,
                            "score": incorrect_score,
                            "score_minus_true_category": incorrect_score - true_score,
                        },
                        "top1_result": {
                            "prediction": row["prediction"],
                            "score": row["top1_score"],
                            "runner_up_score": row.get("runner_up_score"),
                            "margin": row["margin"],
                            "correct": row["top1_correct"],
                        },
                        "all_category_scores": {
                            prompt: result["best_score"]
                            for prompt, result in row["prompt_results"].items()
                        },
                        "rgb": {
                            "file": target.name,
                            "source_file": str(source),
                            "copied_without_modification": True,
                        },
                    }
                    metadata_path = viewpoint_dir / "metadata.json"
                    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
                    manifest_record = {
                        "schema_version": 1,
                        "category": category, "quality_band": band,
                        "viewpoint_rank": rank,
                        "viewpoint_score_minus_true_category": score_margin,
                        "viewpoint_accuracy": float(cell["accuracy"]),
                        "viewpoint_correct": int(cell["correct"]),
                        "viewpoint_incorrect": int(cell["incorrect"]),
                        "viewpoint_n": int(cell["n"]),
                        "uid": row["uid"], "azimuth": azimuth,
                        "elevation": elevation,
                        "top1_correct": row["top1_correct"],
                        "prediction": row["prediction"],
                        "top1_score": row["top1_score"], "margin": row["margin"],
                        "rgb": str(target.relative_to(staging)),
                        "metadata": str(metadata_path.relative_to(staging)),
                        "source_rgb": str(source),
                    }
                    manifest.append(manifest_record)
                    confusion_records[band].append({
                        "schema_version": 1,
                        "quality_band": band,
                        "category": category, "uid": row["uid"],
                        "azimuth": azimuth, "elevation": elevation,
                        "viewpoint_rank": rank,
                        "viewpoint_score_minus_true_category": score_margin,
                        "viewpoint_accuracy": float(cell["accuracy"]),
                        "rgb": manifest_record["rgb"],
                        "metadata": manifest_record["metadata"],
                        "highest_scoring_incorrect_category": incorrect_category,
                        "highest_scoring_incorrect_score": incorrect_score,
                        "true_category_score": true_score,
                        "incorrect_minus_true_score": incorrect_score - true_score,
                        "top1_prediction": row["prediction"],
                        "top1_was_incorrect": row["prediction"] != category,
                    })

        with open(staging / "manifest.jsonl", "w") as stream:
            for record in manifest:
                stream.write(json.dumps(record) + "\n")
        for band, records in confusion_records.items():
            with open(staging / f"{band}_confusions.jsonl", "w") as stream:
                for record in records:
                    stream.write(json.dumps(record) + "\n")
        (staging / "selection.json").write_text(json.dumps({
            "schema_version": 1,
            "selection_metric": "per-category mean score_minus_true_category (strongest incorrect score minus true category score); smaller is better",
            "viewpoints_per_band": viewpoints_per_band,
            "images": len(manifest),
            "confusion_records": {
                band: len(records) for band, records in confusion_records.items()
            },
            "categories": selections,
        }, indent=2) + "\n")
        if destination.exists():
            shutil.rmtree(destination)
        staging.replace(destination)
    except Exception:
        if staging.exists():
            shutil.rmtree(staging)
        raise
    return manifest


def _representative_rgb_records(render_dir, categories, az_values, el_values):
    """Choose one object's RGB view at every angle for each category.

    Metric cells may average several objects.  Keeping one UID fixed throughout
    the companion grid makes changes across viewpoints visually comparable.
    """
    wanted_angles = {(float(az), float(el)) for el in el_values for az in az_values}
    representatives = {}
    records = {category: {} for category in categories}
    index_path = render_dir / "index.jsonl"
    if not index_path.is_file():
        raise FileNotFoundError(f"Missing render index for RGB grids: {index_path}")

    with open(index_path) as stream:
        for line in stream:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            category = record.get("category")
            if category not in records:
                continue
            representatives.setdefault(category, record["uid"])
            if record["uid"] != representatives[category]:
                continue
            angle = (float(record["azimuth"]), float(record["elevation"]))
            if angle in wanted_angles:
                records[category][angle] = render_dir / record["files"]["rgb"]

    for category in categories:
        missing = wanted_angles - records[category].keys()
        if missing:
            preview = ", ".join(f"({az:g}, {el:g})" for az, el in sorted(missing)[:4])
            raise ValueError(f"representative RGB grid for {category} is missing: {preview}")
    return records, representatives


def _rgb_montage(paths, az_values, el_values):
    """Load a viewpoint grid into one compact image with angle-aligned cells."""
    from PIL import Image, ImageOps

    size = RGB_THUMBNAIL_SIZE
    montage = Image.new("RGB", (size * len(az_values), size * len(el_values)), "black")
    resampling = getattr(Image, "Resampling", Image).LANCZOS
    # Pillow rows run top-down, while the plots show high elevations at the top.
    for row, elevation in enumerate(reversed(el_values)):
        for col, azimuth in enumerate(az_values):
            path = paths[(float(azimuth), float(elevation))]
            with Image.open(path) as source:
                tile = ImageOps.fit(source.convert("RGB"), (size, size), method=resampling)
            montage.paste(tile, (col * size, row * size))
    return np.asarray(montage)


def _heatmap(ax, grid, az_values, el_values, title, vmin=0.0, vmax=1.0,
             cmap=PLOT_CMAP):
    im = ax.imshow(grid, origin="lower", aspect="auto", vmin=vmin, vmax=vmax, cmap=cmap)
    ax.set_xticks(range(len(az_values)))
    ax.set_xticklabels([f"{a:g}" for a in az_values])
    ax.set_yticks(range(len(el_values)))
    ax.set_yticklabels([f"{e:g}" for e in el_values])
    ax.set_xlabel("azimuth (deg)")
    ax.set_ylabel("elevation (deg)")
    ax.set_title(title)
    return im


def _rgb_grid(ax, montage, az_values, el_values, title):
    ax.imshow(montage, extent=(-0.5, len(az_values) - 0.5,
                              -0.5, len(el_values) - 0.5),
              origin="upper", aspect="auto")
    ax.set_xticks(range(len(az_values)))
    ax.set_xticklabels([f"{a:g}" for a in az_values])
    ax.set_yticks(range(len(el_values)))
    ax.set_yticklabels([f"{e:g}" for e in el_values])
    ax.set_xticks(np.arange(-0.5, len(az_values), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(el_values), 1), minor=True)
    ax.grid(which="minor", color="black", linewidth=0.5, alpha=0.25)
    ax.tick_params(which="minor", bottom=False, left=False)
    ax.set_xlabel("azimuth (deg)")
    ax.set_ylabel("elevation (deg)")
    ax.set_title(title)


def _save_metric(cells, categories, az_values, el_values, field, path, title,
                 rgb_panel):
    grid = _grid(cells, categories, az_values, el_values, field)
    fig, (ax, rgb_ax) = plt.subplots(
        1, 2, figsize=(2.4 + 1.4 * len(az_values), 1.5 + 0.7 * len(el_values)))
    im = _heatmap(ax, grid, az_values, el_values, title)
    fig.colorbar(im, ax=ax, shrink=0.8)
    _rgb_grid(rgb_ax, rgb_panel[0], az_values, el_values, rgb_panel[1])
    fig.text(0.5, 0.005, AZIMUTH_NOTE, ha="center", fontsize=8, style="italic")
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(path, dpi=120)
    plt.close(fig)


def _save_confusion(cells, category, az_values, el_values, path, rgb_panel):
    """Binary prompt-detection counts; these are not multiclass confusion cells."""
    fig = plt.figure(figsize=(3.0 + 2.1 * len(az_values), 2.0 + 1.4 * len(el_values)))
    spec = fig.add_gridspec(2, 3, width_ratios=(1, 1, 2.05))
    axes = [fig.add_subplot(spec[row, col]) for row in range(2) for col in range(2)]
    for ax, field in zip(axes, ("TP", "FN", "FP", "TN")):
        grid = _grid(cells, {category}, az_values, el_values, field)
        vmax = np.nanmax(grid) if np.isfinite(grid).any() and np.nanmax(grid) > 0 else 1.0
        im = _heatmap(ax, grid, az_values, el_values, f"{field} count",
                      vmin=0.0, vmax=vmax)
        fig.colorbar(im, ax=ax, shrink=0.7)
    rgb_ax = fig.add_subplot(spec[:, 2])
    _rgb_grid(rgb_ax, rgb_panel[0], az_values, el_values, rgb_panel[1])
    fig.suptitle(f"{category}: label counts  ({AZIMUTH_NOTE})")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(path, dpi=110)
    plt.close(fig)


def _jsonable(cells):
    """Convert tuple-keyed cells to a JSON-friendly list, NaN rates -> None."""
    out = []
    for (cat, az, el), cell in sorted(cells.items(), key=lambda kv: (kv[0][0], kv[0][2], kv[0][1])):
        record = {"category": cat, "azimuth": az, "elevation": el}
        for key, value in cell.items():
            record[key] = None if isinstance(value, float) and math.isnan(value) else value
        out.append(record)
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True,
                        help="Resolved settings.json written by run.py")
    args = parser.parse_args()

    config = json.loads(Path(args.config).read_text())
    output = Path(config["output_dir"])
    db = store.connect(store.database_path(output))
    rows = store.load_trials(db)
    if not rows:
        raise ValueError("experiment.sqlite contains no recognition trials")
    if any(row.get("schema_version") != 2 for row in rows):
        raise ValueError("experiment.sqlite uses an obsolete trial schema; rerun the query stage")

    # Every object uses the configured grid axes; comparisons stay within objects.
    az_values = sorted({float(a) for a in config["azimuths"]})
    el_values = sorted({float(e) for e in config["elevations"]})
    categories = sorted(config["categories"])
    expected_per_cell = int(config["objects_per_category"])
    row_counts = {}
    for row in rows:
        key = (row["category"], float(row["azimuth"]), float(row["elevation"]))
        row_counts[key] = row_counts.get(key, 0) + 1
    incomplete = []
    for cat in categories:
        expected_per_cell = config.get("category_object_counts", {}).get(
            cat, int(config["objects_per_category"]))
        for el in el_values:
            for az in az_values:
                count = row_counts.get((cat, az, el), 0)
                if count != expected_per_cell:
                    incomplete.append(f"{cat}@({az:g},{el:g})={count}")
    if incomplete:
        preview = ", ".join(incomplete[:8])
        suffix = " ..." if len(incomplete) > 8 else ""
        raise ValueError(
            "incomplete grid (counts must match the configured category cohort): "
            f"{preview}{suffix}")

    object_grids = _object_score_grids(rows, az_values, el_values)
    cells = metrics.aggregate_viewpoints(rows)
    overall = _overall_cells(rows)

    collection = _collect_sam3d_inputs(
        rows, cells, Path(config["render_dir"]), output, categories,
        int(config.get("sam3d_collection", {}).get("viewpoints_per_band", 3)),
    )

    summary = {}
    for cat in categories:
        grid = _grid(cells, {cat}, az_values, el_values, "accuracy")
        finite = grid[np.isfinite(grid)]
        summary[cat] = {
            "mean_viewpoint_accuracy": float(finite.mean()) if finite.size else None,
            "median_viewpoint_accuracy": float(np.median(finite)) if finite.size else None,
            "max_viewpoint_accuracy": float(finite.max()) if finite.size else None,
            "max_minus_mean": float(finite.max() - finite.mean()) if finite.size else None,
        }

    aggregate = {
        "schema_version": 2,
        "azimuths": az_values, "elevations": el_values,
        "categories": categories, "azimuth_note": AZIMUTH_NOTE,
        "primary_metric": "score_minus_true_category: strongest incorrect score minus true category score per object",
        "binary_diagnostic": (
            "TP/FN are the true prompt with score and mask-IoU thresholds; "
            "FP/TN are every other vocabulary prompt with the score threshold only"
        ),
        "detection": config["detection"],
        "summary": summary,
        "per_object": [
            {"category": category, "uid": uid, "score_minus_true_category":
             [[None if not math.isfinite(value) else float(value) for value in line]
              for line in object_grids[(category, uid)]]}
            for category, uid in sorted(object_grids)
        ],
        "per_category": _jsonable(cells), "overall": _jsonable(overall),
    }
    store.put_artifact(db, "aggregate.json", aggregate)

    # Store NumPy grids for optional export without re-reading trial payloads.
    object_keys = sorted(object_grids)
    npz = {"azimuths": np.asarray(az_values), "elevations": np.asarray(el_values),
           "object_categories": np.asarray([key[0] for key in object_keys]),
           "object_uids": np.asarray([key[1] for key in object_keys]),
           "score_minus_true_category": np.stack([object_grids[key] for key in object_keys])}
    for cat in categories:
        for field in ("accuracy", "correct", "incorrect", "recall", "fp_rate",
                      "TP", "TN", "FP", "FN", "SKIP", "n"):
            npz[f"{cat}__{field}"] = _grid(cells, {cat}, az_values, el_values, field)
    for field in ("accuracy", "correct", "incorrect", "recall", "fp_rate",
                  "TP", "TN", "FP", "FN", "SKIP", "n"):
        npz[f"overall__{field}"] = _grid(overall, {"overall"}, az_values, el_values, field)
    npz_buffer = io.BytesIO()
    np.savez_compressed(npz_buffer, **npz)
    store.put_artifact(db, "aggregate.npz", npz_buffer.getvalue())
    db.close()

    if not HAVE_MPL:
        print(f"[analyze] matplotlib unavailable ({_MPL_ERROR}); stored analysis in experiment.sqlite "
              f"and {len(collection)} SAM3D inputs; skipped plots", flush=True)
        write_summary(output.parent)
        return

    plots = output / "plots" / "per_object"
    plots.mkdir(parents=True, exist_ok=True)
    rgb_records = _object_rgb_records(Path(config["render_dir"]), object_keys,
                                      az_values, el_values)
    finite = np.concatenate([np.abs(grid[np.isfinite(grid)])
                             for grid in object_grids.values()])
    color_limit = max(float(finite.max()), 1e-12) if finite.size else 1.0
    for index, (category, uid) in enumerate(object_keys):
        _save_object_score_margin(
            object_grids[(category, uid)], rgb_records[(category, uid)],
            category, uid, az_values, el_values,
            plots / f"score_minus_true_category_{index:04d}_{_safe_component(category)}_{_safe_component(uid)}.png",
            color_limit)
    print(f"[analyze] stored analysis in experiment.sqlite, {len(collection)} SAM3D inputs, and "
          f"{len(object_keys)} per-object heatmaps to {output}", flush=True)
    write_summary(output.parent)


if __name__ == "__main__":
    main()
