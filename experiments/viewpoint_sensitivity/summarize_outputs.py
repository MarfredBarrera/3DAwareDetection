"""Write a compact human-readable index for completed viewpoint runs."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from experiments.viewpoint_sensitivity import store


def _read_json(path):
    return json.loads(path.read_text()) if path.is_file() else None


def _percent(numerator, denominator):
    return f"{100 * numerator / denominator:.1f}%" if denominator else "—"


def write_summary(outputs_dir):
    outputs_dir = Path(outputs_dir)

    def linked(path, label):
        return f"[{label}]({path})" if (outputs_dir / path).exists() else "—"
    runs = sorted(path for path in outputs_dir.glob("viewpoint-sensitivity-*") if path.is_dir())
    lines = [
        "# Experiment results", "",
        "The tables below summarize the saved output files. This page is regenerated after analysis or a batch run.",
        "Each run stores recognition trials and derived analysis in one SQLite database. Use the export command below for JSONL, JSON, or NumPy files.",
        "", "## Runs", "",
        "| Run | Stage | Objects | Scored views | Top-1 accuracy | Mean score − true | SAM3D inputs |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    summaries = {}
    for root in runs:
        name = root.name.removeprefix("viewpoint-sensitivity-")
        database = store.database_path(root)
        if database.is_file():
            with store.connect(database) as db:
                aggregate = store.get_analysis(db)
        else:
            aggregate = _read_json(root / "aggregate.json")
        state = _read_json(root / "run/batch-status.json")
        selection = _read_json(root / "sam3d_inputs/selection.json")
        if aggregate:
            cells = aggregate["overall"]
            correct = sum(cell["correct"] for cell in cells)
            total = sum(cell["n"] for cell in cells)
            margins = [value for obj in aggregate["per_object"]
                       for row in obj["score_minus_true_category"] for value in row
                       if value is not None and math.isfinite(value)]
            mean_margin = f"{sum(margins) / len(margins):+.3f}" if margins else "—"
            objects = len(aggregate["per_object"])
            stage = "analyzed"
            summaries[name] = aggregate["summary"]
        else:
            correct = total = objects = 0
            mean_margin = "—"
            stage = "rendered" if (root / "render/index.jsonl").exists() else "pending"
        if state and state.get("status") == "failed":
            stage = "failed: " + str(state.get("stage", "unknown"))
        lines.append(
            f"| [{name}]({root.name}/) | {stage} | {objects or '—'} | {total or '—'} | "
            f"{_percent(correct, total)} | {mean_margin} | "
            f"{selection['images'] if selection else '—'} |"
        )
    lines += [
        "", "Mean score − true is the strongest incorrect category score minus the true category score,",
        "averaged over scored object views. Smaller values mean the true category scores higher.",
        "The full and reviewed upright runs use different object cohorts and elevation grids; compare them with that in mind.",
        "", "## Category accuracy", "",
    ]
    if summaries:
        names = list(summaries)
        lines += [
            "| Category | " + " | ".join(names) + " |",
            "| --- | " + " | ".join("---:" for _ in names) + " |",
        ]
        categories = sorted({cat for summary in summaries.values() for cat in summary})
        for category in categories:
            values = [f"{100 * summaries[name][category]['mean_viewpoint_accuracy']:.1f}%"
                      if category in summaries[name] else "—" for name in names]
            lines.append("| " + category + " | " + " | ".join(values) + " |")
    lines += ["", "## Where to look", ""]
    for root in runs:
        name = root.name.removeprefix("viewpoint-sensitivity-")
        relative = Path(root.name)
        lines += [
            f"### {name}", "",
            f"- Durable data: {linked(relative / store.DATABASE_NAME, 'trials and analysis database')}; "
            f"{linked(relative / 'plots/per_object', 'object plots')}",
            f"- Inputs: {linked(relative / 'render/index.jsonl', 'render index')}; "
            f"{linked(relative / 'settings.json', 'resolved settings')}",
            f"- SAM3D collection: {linked(relative / 'sam3d_inputs/selection.json', 'selection and metric')}; "
            f"{linked(relative / 'sam3d_inputs/manifest.jsonl', 'image manifest')}",
            f"- Run records: {linked(relative / 'run/batch-status.json', 'stage status')}; "
            f"{linked(relative / 'run/camera-verification.json', 'camera verification')}; "
            f"{linked(relative / 'run', 'logs')}",
            "",
        ]
    upright = outputs_dir / "viewpoint-sensitivity-upright"
    if upright.exists():
        lines += [
            "The reviewed run also keeps its [canonical object frames](viewpoint-sensitivity-upright/canonical.json), "
            "[orientation report](viewpoint-sensitivity-upright/orientation-report.json), "
            "and the `audit-before/`, `candidates/`, and `audit-after/` visual review evidence.", "",
        ]
    lines += [
        "To export legacy files for a run: `python -m experiments.viewpoint_sensitivity.store export "
        "outputs/viewpoint-sensitivity-full --target /tmp/full-export`.", "",
    ]
    correction = outputs_dir / "sam3d_correction"
    if correction.exists():
        lines += [
            "## SAM3D reconstruction example", "",
            f"The separate [{correction.name}]({correction.name}/) output contains "
            f"[_run settings_]({correction.name}/run.json), masks, a mesh, and rendered views.", "",
        ]
    target = outputs_dir / "RESULTS.md"
    target.write_text("\n".join(lines).rstrip() + "\n")
    return target


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("outputs_dir", nargs="?", type=Path, default=Path("outputs"))
    print(write_summary(parser.parse_args().outputs_dir))


if __name__ == "__main__":
    main()
