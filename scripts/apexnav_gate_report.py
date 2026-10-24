"""Side-by-side figures for ApexNav end-to-end runs of several modes (baseline and gate configs).

A results dir holds ``results.csv`` (one row per episode: episode, goal, and per mode
``<mode>_committed_to``, ``<mode>_result``; optional ``<mode>_verified``,
``<mode>_checks``, ``<mode>_rejected``) and ``frames/<mode>_epi<N>_commit.jpg``
(committed detection with its mask, written by ``scripts/apexnav_commit_frames.py``
inside the apexnav container). Modes are read from the ``*_result`` columns in
order. Object names in results.csv come from visual inspection, never from HM3D
semantic labels.
``<mode>_verified`` is the outcome judged from the commit and final frames; when
present it is shown instead of HM3D's ``<mode>_result`` (with HM3D's where they
differ). "goal not scored" means the agent stopped at a real goal-category object
that HM3D does not score as a success.

    sam3d-env core python -m scripts.apexnav_gate_report outputs/apexnav_gate/configs_2026-10-23 \
        --label baseline=Baseline --label sam3="SAM3 only"

Writes ``summary_table[_<page>].png`` (episode, goal, each mode's outcome, committed
object and gate checks), ``comparison[_<page>].png`` (committed detections side by
side) and ``outcomes.png`` (outcome counts per mode, HM3D and verified).
"""
import argparse
import csv
import textwrap
from pathlib import Path

import matplotlib.pyplot as plt
from PIL import Image

TEXT, MUTED, SURFACE, GRID = "#0b0b0b", "#52514e", "#fcfcfb", "#e4e3df"
GOOD, WARNING, CRITICAL, NEUTRAL = "#0ca30c", "#fab219", "#d03b3b", "#a8a6a0"


def status(result):
    """(color, icon) for an episode result; endings without an object stop (stuck, no frontier) are neutral."""
    if result == "success":
        return GOOD, "✓"
    if result == "false positive":
        return CRITICAL, "✗"
    if result in ("goal not scored", "last-mile failure"):
        return WARNING, "~"
    return NEUTRAL, "–"


def outcome(row, mode):
    """(shown outcome, HM3D outcome or None when they agree)."""
    hm3d = row[f"{mode}_result"]
    shown = row.get(f"{mode}_verified") or hm3d
    return shown, (hm3d if hm3d != shown else None)


def modes_of(rows):
    return [key[:-len("_result")] for key in rows[0] if key.endswith("_result") and not key.endswith("_raw_result")]


def comparison(directory, rows, labels, per_page):
    modes = modes_of(rows)
    pages = [rows[i:i + per_page] for i in range(0, len(rows), per_page)]
    for number, page in enumerate(pages, 1):
        fig, axes = plt.subplots(len(page), len(modes), figsize=(4.25 * len(modes), 3.35 * len(page)),
                                 facecolor=SURFACE, squeeze=False)
        for i, row in enumerate(page):
            for j, mode in enumerate(modes):
                ax = axes[i, j]
                frame = directory / "frames" / f"{mode}_epi{row['episode']}_commit.jpg"
                if frame.exists():
                    ax.imshow(Image.open(frame))
                else:
                    ax.set_facecolor(GRID)
                    ax.text(0.5, 0.5, "no object committed", ha="center", va="center", color=MUTED,
                            fontsize=11, transform=ax.transAxes)
                ax.set_xticks([]), ax.set_yticks([])
                for spine in ax.spines.values():
                    spine.set_color(GRID)
                if i == 0:
                    ax.set_title(labels.get(mode, mode), fontsize=12, color=TEXT, pad=8)
                result, hm3d = outcome(row, mode)
                color, icon = status(result)
                note = f" (HM3D: {hm3d})" if hm3d else ""
                ax.set_xlabel(f"{row[f'{mode}_committed_to']}\n{icon} {result}{note}", fontsize=9.5, color=TEXT)
                ax.xaxis.label.set_bbox(dict(facecolor=SURFACE, edgecolor=color, boxstyle="round,pad=0.3", lw=1.5))
            axes[i, 0].set_ylabel(f"episode {row['episode']}\ngoal: {row['goal']}", fontsize=12, color=TEXT, labelpad=10)
        suffix = f" ({number}/{len(pages)})" if len(pages) > 1 else ""
        fig.suptitle(f"ApexNav false-positive episodes: committed detection per mode{suffix} (red = committed mask)",
                     fontsize=13, color=TEXT, y=0.998)
        fig.tight_layout()
        name = f"comparison_{number:02d}.png" if len(pages) > 1 else "comparison.png"
        fig.savefig(directory / name, dpi=110, facecolor=SURFACE)
        plt.close(fig)


def table(directory, rows, labels, caption, per_page):
    """Per episode and mode: the outcome badge, then the committed object and gate checks (rejected)."""
    modes = modes_of(rows)
    width = (1 - 0.15) / len(modes)
    x = [0, 0.045] + [0.15 + k * width for k in range(len(modes))]
    pages = [rows[i:i + per_page] for i in range(0, len(rows), per_page)]
    for number, page in enumerate(pages, 1):
        fig, ax = plt.subplots(figsize=(3.4 * len(modes) + 2.5, 0.75 * (len(page) + 1) + 1.1), facecolor=SURFACE)
        ax.set_xlim(0, 1), ax.set_ylim(len(page) + 0.9, -0.9), ax.axis("off")
        for k, header in enumerate(["Ep.", "Goal"] + [labels.get(m, m) for m in modes]):
            ax.text(x[k] + 0.004, -0.35, header, fontsize=11, color=MUTED, weight="bold", va="center")
        ax.axhline(0.1, color=GRID, lw=1)
        for i, row in enumerate(page):
            y = i + 0.6
            ax.text(x[0] + 0.004, y, row["episode"], fontsize=10.5, color=TEXT, va="center")
            ax.text(x[1] + 0.004, y, row["goal"], fontsize=10.5, color=TEXT, va="center")
            for k, mode in enumerate(modes, 2):
                result, hm3d = outcome(row, mode)
                color, icon = status(result)
                ax.text(x[k] + 0.008, y - 0.15, f"{icon} {result}", fontsize=9.5, color=TEXT, va="center",
                        bbox=dict(facecolor=SURFACE, edgecolor=color, lw=1.4, boxstyle="round,pad=0.3"))
                detail = [row[f"{mode}_committed_to"]] if row[f"{mode}_committed_to"] != "-" else []
                if row.get(f"{mode}_checks") not in (None, ""):
                    detail.append(f"{row[f'{mode}_checks']} ({row[f'{mode}_rejected']}) checks")
                if hm3d:
                    detail.append(f"HM3D: {hm3d}")
                ax.text(x[k] + 0.008, y + 0.27, textwrap.shorten(" · ".join(detail), int(330 * width)), fontsize=8, color=MUTED,
                        va="center")
            ax.axhline(y + 0.5, color=GRID, lw=0.8)
        ax.text(0.004, len(page) + 0.5, textwrap.fill(caption, 40 * len(modes) + 40), fontsize=9, color=MUTED,
                va="center")
        fig.tight_layout()
        name = f"summary_table_{number:02d}.png" if len(pages) > 1 else "summary_table.png"
        fig.savefig(directory / name, dpi=130, facecolor=SURFACE)
        plt.close(fig)


def outcomes(directory, rows, labels):
    """Stacked bars of episode outcomes per mode, as HM3D scores them and (if present) as verified visually."""
    groups = [("success", GOOD), ("false positive", CRITICAL), ("goal not scored", WARNING), ("no object stop", NEUTRAL)]
    bars = []
    for mode in modes_of(rows)[::-1]:
        label = labels.get(mode, mode)
        if f"{mode}_verified" in rows[0]:
            bars.append((f"{label}, verified", [row[f"{mode}_verified"] for row in rows]))
        bars.append((f"{label}, HM3D", [row[f"{mode}_result"] for row in rows]))
    fig, ax = plt.subplots(figsize=(10, 0.65 * len(bars) + 1.3), facecolor=SURFACE)
    for y, (label, results) in enumerate(bars):
        results = ["goal not scored" if r == "last-mile failure" else r for r in results]
        counts = [sum(r == name for r in results) for name, _ in groups[:3]]
        counts.append(len(results) - sum(counts))
        left = 0
        for (name, color), n in zip(groups, counts):
            if n:
                ax.barh(y, n, left=left, height=0.62, color=color, edgecolor=SURFACE, linewidth=2)
                ax.text(left + n / 2, y, str(n), ha="center", va="center", fontsize=10,
                        color=SURFACE if color in (GOOD, CRITICAL) else TEXT)
            left += n
    ax.set_yticks(range(len(bars)), [label for label, _ in bars], fontsize=11, color=TEXT)
    ax.set_xlim(0, len(rows)), ax.set_xlabel("episodes", color=MUTED)
    ax.tick_params(colors=MUTED, length=0)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.set_facecolor(SURFACE)
    handles = [plt.Rectangle((0, 0), 1, 1, color=color) for _, color in groups]
    ax.legend(handles, [name for name, _ in groups], ncol=4, frameon=False, loc="lower left",
              bbox_to_anchor=(0, 1.0), fontsize=10, labelcolor=TEXT)
    ax.set_title(f"Outcomes on {len(rows)} ApexNav episodes with a verified baseline false positive",
                 fontsize=12, color=TEXT, loc="left", pad=30)
    fig.tight_layout()
    fig.savefig(directory / "outcomes.png", dpi=130, facecolor=SURFACE)
    plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--per-page", type=int, default=10, help="episodes per comparison page")
    parser.add_argument("--per-table", type=int, default=25, help="episodes per summary-table page")
    parser.add_argument("--label", action="append", default=[], metavar="MODE=LABEL",
                        help="display name of a mode (default: the mode)")
    parser.add_argument("--caption", default="Objects and outcomes from visual inspection of the commit and final "
                        "frames (HM3D outcome shown where it differs).")
    args = parser.parse_args(argv)
    labels = dict(item.split("=", 1) for item in args.label)
    rows = list(csv.DictReader(open(args.directory / "results.csv")))
    comparison(args.directory, rows, labels, args.per_page)
    table(args.directory, rows, labels, args.caption, args.per_table)
    outcomes(args.directory, rows, labels)


if __name__ == "__main__":
    main()
