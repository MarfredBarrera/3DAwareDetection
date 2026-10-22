"""The perception pipeline on ApexNav's false-positive episodes (HM3D-v2 val).

Each record (``data/apexnav/records/<episode>``, an ApexNav EpisodeRecorder dir whose
``meta.json`` result is "false positive") gives one input: the frame and mask that best
show the object the planner committed to (ApexNav's ``find_commit_evidence``). ``run``
passes it through ``perception.Verifier`` with every candidate rendered and classified,
so ``figure`` can compare all selectors offline. An episode is *resolved* when the
top-1 category on the judged view is not the episode goal, i.e. the committed
detection would have been rejected.

    sam3d-env core python -m scripts.apexnav_fp run --gpu 0
    sam3d-env core python -m scripts.apexnav_fp figure
    sam3d-env core python -m scripts.apexnav_fp grids

Outputs: ``outputs/apexnav_fp/<episode>/`` (input, mask, SAM3D views, episode.json),
``results.json`` and ``apexnav_fp_resolved.png``; ``grids`` writes one page per
episode under ``grids/`` and ``grids.pdf``.
"""
import argparse
import json
import math
import shutil
import sys
from pathlib import Path

import numpy as np
from PIL import Image

from perception import select
from perception.client import HOUSEHOLD

RECORDS = Path("data/apexnav/records")
# The user's manual verification: episodes whose committed object is truly not the goal
# (copied from ApexNav replays/manually_verified/replays). The rest are goal-class objects.
VERIFIED = Path("data/apexnav/verified.txt")
# A second, independent pass (fp | goal | ambiguous + note); reported, not used for the split.
REVIEW = Path("data/apexnav/review.tsv")
APEXNAV_SRC = Path("data/apexnav/apexnav_src")  # copy of basic_utils/record_episode/episode_visualization.py
OUTPUT = Path("outputs/apexnav_fp")
AZIMUTHS = [0, 45, 90, 135, 180, 225, 270, 315]
ELEVATIONS = [0, 30, 60]
# Fixed before looking at results (calibrated on LVIS meshes, mesh F < 0.5).
LAMBDA = 0.15
SELECTORS = ["input frame", "identity (180,0)", "random view", "top-2 gap view", "VQF view",
             "VQF + ensemble (abstain)", "VQF + ensemble (else input)", "gap + ensemble (else input)"]


def goal(meta):
    return meta["target_label"].replace("_", " ")


def prepare(record):
    """Copy the committed-object frame and mask into the episode's output dir."""
    sys.path.insert(0, str(APEXNAV_SRC.resolve()))
    from episode_visualization import find_commit_evidence

    out = OUTPUT / record.name
    if (out / "input.json").is_file():
        return json.loads((out / "input.json").read_text())
    evidence = find_commit_evidence(str(record))
    if evidence is None or evidence["mask_path"] is None:
        return None
    steps = {int(e["step"]): e for e in map(json.loads, (record / "steps.jsonl").open())}
    pitch = float(steps.get(evidence["step"], {}).get("camera_pitch") or 0.0)
    out.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(record / "rgb" / f"{evidence['step']:06d}.jpg", out / "input.jpg")
    with Image.open(evidence["mask_path"]) as mask:
        Image.fromarray(((np.asarray(mask.convert("L")) > 0) * 255).astype(np.uint8)).save(out / "mask.png")
    meta = json.loads((record / "meta.json").read_text())
    info = {"episode": record.name, "goal": goal(meta), "step": evidence["step"],
            "matched_by": evidence["matched_by"], "area_fraction": evidence["area_fraction"],
            # Habitat pitch is negative looking down; study elevation is positive above.
            "input_elevation": -math.degrees(pitch)}
    (out / "input.json").write_text(json.dumps(info, indent=1))
    return info


def run(args):
    from perception import Verifier

    records = sorted(p for p in RECORDS.iterdir() if (p / "meta.json").is_file())
    with Verifier(selector="vqf", all_views=True, ensemble=True, ensemble_threshold=LAMBDA, gpu=args.gpu,
                  azimuths=AZIMUTHS, elevations=ELEVATIONS, log_dir=OUTPUT / "logs") as verifier:
        for count, record in enumerate(records, 1):
            info = prepare(record)
            out = OUTPUT / record.name
            if info is None or (out / "episode.json").is_file():
                continue
            vocabulary = list(dict.fromkeys([*HOUSEHOLD, info["goal"]]))
            verdict = verifier.verify(out / "input.jpg", out / "mask.png", info["goal"],
                                      info["input_elevation"], vocabulary)
            (out / "views").mkdir(exist_ok=True)
            views = []
            for view in verdict.views:
                name = f"azimuth_{view.azimuth:g}_elevation_{view.elevation:g}.png"
                view.image.save(out / "views" / name)
                views.append({"azimuth": view.azimuth, "elevation": view.elevation, "image": name,
                              "quality": view.quality, "vqf_elevation": view.vqf_elevation,
                              "scores": view.scores})
            episode = {"input": verifier.classify(out / "input.jpg", out / "mask.png", vocabulary),
                       "uncertainty": verdict.uncertainty, "views": views}
            (out / "episode.json").write_text(json.dumps(episode))
            print(f"[run] {count}/{len(records)} {record.name} u={verdict.uncertainty}", flush=True)


def outcomes(episode, target, lam=LAMBDA):
    """{selector: True if its view's top-1 category is not the goal (FP rejected)}.

    "+ ensemble" selectors trust the reconstruction only when the disagreement
    <= ``lam``; otherwise they abstain (reject the commit) or fall back to the input frame.
    """
    views = {(v["azimuth"], v["elevation"]): v for v in episode["views"]}
    scores = {key: v["scores"] for key, v in views.items()}
    rejects = lambda key: select.top1(scores[key])[0] != target
    gap = select.top2gap_pick(scores)
    vqf = select.vqf_pick({key: (v["quality"], v["vqf_elevation"]) for key, v in views.items()})
    rejected = {
        "input frame": select.top1(episode["input"])[0] != target,
        "identity (180,0)": rejects((180.0, 0.0)),
        "random view": float(np.mean([rejects(key) for key in views])),
        "top-2 gap view": rejects(gap),
        "VQF view": rejects(vqf),
    }
    trusted = select.trusted(episode["uncertainty"], lam)
    rejected["VQF + ensemble (abstain)"] = rejected["VQF view"] if trusted else True
    rejected["VQF + ensemble (else input)"] = rejected["VQF view"] if trusted else rejected["input frame"]
    rejected["gap + ensemble (else input)"] = rejected["top-2 gap view"] if trusted else rejected["input frame"]
    key = lambda angle: f"{angle[0]:g},{angle[1]:g}"
    return {**rejected, "_picked": key(gap), "_picked_top1": select.top1(scores[gap])[0],
            "_vqf_picked": key(vqf), "_vqf_top1": select.top1(scores[vqf])[0],
            "_input_top1": select.top1(episode["input"])[0], "_uncertainty": episode["uncertainty"],
            "_trusted": trusted}


def figure(args):
    """Resolved counts over the manually verified false positives (``VERIFIED``).

    Episodes whose committed object really is the goal class (unannotated instance or a
    stopping-distance failure) are not detector false positives; on those, "resolving"
    means wrongly rejecting a correct object, so they are reported as the cost.
    """
    import csv

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    with REVIEW.open() as handle:
        review = {row["episode"]: row for row in csv.DictReader(handle, delimiter="\t")}
    verified = set(VERIFIED.read_text().split())
    for name, row in review.items():
        row["second_pass"] = row.pop("verdict")
        row["verdict"] = "fp" if name in verified else "goal"
    rows, missing = [], []
    for record in sorted(p for p in RECORDS.iterdir() if (p / "meta.json").is_file()):
        out = OUTPUT / record.name
        if not (out / "episode.json").is_file():
            missing.append({"episode": record.name, "verdict": review[record.name]["verdict"]})
            continue
        info = json.loads((out / "input.json").read_text())
        rows.append({**info, **review[record.name],
                     **outcomes(json.loads((out / "episode.json").read_text()), info["goal"])})
    groups = {"fp": "verified false positives", "goal": "excluded: object is the goal class"}
    counts = {v: {"total": sum(r["verdict"] == v for r in review.values()),
                  "processed": sum(r["verdict"] == v for r in rows),
                  **{s: float(sum(r[s] for r in rows if r["verdict"] == v)) for s in SELECTORS}}
              for v in groups}
    fp = [r for r in rows if r["verdict"] == "fp"]
    summary = {"counts": counts, "missing": missing,
               "gap_fixes_input_misses_fp": sum(r["top-2 gap view"] and not r["input frame"] for r in fp),
               "gap_breaks_input_rejections_fp": sum(r["input frame"] and not r["top-2 gap view"] for r in fp),
               "lambda": LAMBDA,
               "trusted": {v: sum(r["_trusted"] for r in rows if r["verdict"] == v) for v in groups},
               # Abstain-variant counts over a lambda sweep: (fp rejected, goal wrongly rejected).
               "lambda_sweep": {f"{lam:.2f}": [sum(r["VQF view"] if select.trusted(r["_uncertainty"], lam) else 1
                                                    for r in rows if r["verdict"] == v) for v in groups]
                                for lam in np.arange(0.0, 0.41, 0.05)},
               "vocabulary": list(HOUSEHOLD), "rows": rows}
    (OUTPUT / "results.json").write_text(json.dumps(summary, indent=1, default=float))

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    colors = ["#9aa5b1"] * 3 + ["#2a6fdb"] * (len(SELECTORS) - 3)
    for ax, verdict, label in ((axes[0], "fp", "Real false positives resolved (top-1 != goal)"),
                               (axes[1], "goal", "Correct objects wrongly rejected (top-1 != goal)")):
        n = counts[verdict]["processed"]
        values = [counts[verdict][s] for s in SELECTORS][::-1]
        bars = ax.barh(SELECTORS[::-1], values, color=colors[::-1] if verdict == "fp"
                       else ["#d1495b" if c == "#2a6fdb" else c for c in colors[::-1]])
        for bar, value in zip(bars, values):
            ax.text(bar.get_width() + n * 0.01, bar.get_y() + bar.get_height() / 2,
                    f"{value:.0f} / {n}" if value == int(value) else f"{value:.1f} / {n}", va="center")
        ax.set_xlim(0, n * 1.15)
        ax.set_xlabel(label)
        ax.set_title(f"{groups[verdict]} (n={n} of {counts[verdict]['total']} reviewed)", fontsize=10)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    fig.suptitle(f"ApexNav HM3D-v2 val 'false positive' episodes: {len(review)} recorded, "
                 f"{counts['fp']['total']} manually verified as real FPs", fontsize=10)
    fig.tight_layout()
    fig.savefig(OUTPUT / "apexnav_fp_resolved.png", dpi=150)
    print(json.dumps({k: v for k, v in summary.items() if k not in ("rows", "vocabulary")}, indent=1))


GAP_COLOR, VQF_COLOR, IDENTITY_COLOR = "#2a6fdb", "#b04dff", "#888888"
GOAL_COLOR, OTHER_COLOR = "#cc0000", "#1a8a3a"


def _ranking(scores, n=3):
    return " ".join(f"{k} {v:.2f}" for k, v in select.ranked(scores)[:n])


def _crop_box(paths, pad=0.08):
    """One square box around the non-white pixels of all ``paths`` (white-background
    renders), so every view keeps the same scale."""
    union, size = None, None
    for path in paths:
        with Image.open(path) as image:
            pixels = np.asarray(image.convert("L"))
        size = pixels.shape[::-1]
        box = Image.fromarray(((pixels < 250) * 255).astype(np.uint8)).getbbox()
        if box:
            union = box if union is None else (min(union[0], box[0]), min(union[1], box[1]),
                                               max(union[2], box[2]), max(union[3], box[3]))
    if union is None:
        return None
    cx, cy = (union[0] + union[2]) / 2, (union[1] + union[3]) / 2
    half = min(max(union[2] - union[0], union[3] - union[1]) * (0.5 + pad), max(size) / 2)
    cx, cy = min(max(cx, half), size[0] - half), min(max(cy, half), size[1] - half)
    return tuple(int(round(v)) for v in (cx - half, cy - half, cx + half, cy + half))


def _thumbnail(ax, path, title, color=None, title_color="black", box=None):
    from matplotlib.patches import Rectangle

    with Image.open(path) as image:
        image = image.convert("RGB")
        ax.imshow(image.crop(box) if box else image)
    ax.set_xticks([]); ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_title(title, fontsize=7, pad=2, color=title_color)
    if color:
        ax.add_patch(Rectangle((0, 0), 1, 1, transform=ax.transAxes, fill=False, edgecolor=color, linewidth=3))


def grids(args):
    """Per episode: nominal SAM3 on the input frame, the SAM3D candidate grid with SAM3's
    top-1 on each view, and the top-2 gap and VQF picks."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    rows = json.loads((OUTPUT / "results.json").read_text())["rows"]
    rows.sort(key=lambda r: (r["verdict"] != "fp", r["episode"]))
    verdicts = {"fp": "verified false positive", "goal": "excluded: object is the goal class"}
    yes = lambda rejected: "REJECTED" if rejected else "accepted (goal)"
    out_dir = OUTPUT / "grids"
    with PdfPages(OUTPUT / "grids.pdf") as pdf:
        for r in rows:
            out = OUTPUT / r["episode"]
            episode = json.loads((out / "episode.json").read_text())
            views = {f"{v['azimuth']:g},{v['elevation']:g}": v for v in episode["views"]}
            box = _crop_box([out / "views" / v["image"] for v in views.values()])
            fig = plt.figure(figsize=(26, 11))
            outer = fig.add_gridspec(1, 3, width_ratios=[5, len(AZIMUTHS), 3.6], wspace=.04,
                                     left=.01, right=.99, top=.88, bottom=.05)
            ax = fig.add_subplot(outer[0])
            with Image.open(out / "input.jpg") as image, Image.open(out / "mask.png") as mask:
                ax.imshow(image.convert("RGB"))
                ax.contour(np.asarray(mask.convert("L")) > 127, levels=[.5], colors=["#ffd400"], linewidths=1.5)
            ax.set_xticks([]); ax.set_yticks([])
            ax.set_title(f"Nominal SAM3 on the input frame (committed mask in yellow)\n{_ranking(episode['input'])}",
                         fontsize=10, color=GOAL_COLOR if not r["input frame"] else OTHER_COLOR)
            sub = outer[1].subgridspec(len(ELEVATIONS), len(AZIMUTHS), wspace=.05, hspace=.35)
            for e, elevation in enumerate(ELEVATIONS):
                for a, azimuth in enumerate(AZIMUTHS):
                    key = f"{azimuth:g},{elevation:g}"
                    ax = fig.add_subplot(sub[len(ELEVATIONS) - 1 - e, a])
                    view = views[key]
                    top, gap = select.top1(view["scores"])
                    color = (GAP_COLOR if key == r["_picked"] else VQF_COLOR if key == r["_vqf_picked"]
                             else IDENTITY_COLOR if key == "180,0" else None)
                    _thumbnail(ax, out / "views" / view["image"],
                               f"{azimuth}/{elevation} (el {view['vqf_elevation']:+.0f}) q {view['quality']:.2f}\n"
                               f"{top} {view['scores'][top]:.2f} gap {gap:.2f}",
                               color=color, title_color=GOAL_COLOR if top == r["goal"] else OTHER_COLOR, box=box)
                    if key == r["_vqf_picked"] and key == r["_picked"]:
                        ax.add_patch(matplotlib.patches.Rectangle((.04, .04), .92, .92, transform=ax.transAxes,
                                                                  fill=False, edgecolor=VQF_COLOR, linewidth=3))
            ax = fig.add_subplot(outer[2]); ax.axis("off")
            uncertainty = "undefined" if r["_uncertainty"] is None else f"{r['_uncertainty']:.3f}"
            ax.text(0, .98,
                    f"goal: {r['goal']}\n"
                    f"manual label: {verdicts[r['verdict']]}\n"
                    f"second pass: {r['second_pass']} ({r['object']})\n\n"
                    f"{'selector':24s}{'top-1':>14s}  outcome\n"
                    f"{'nominal SAM3 (input)':24s}{r['_input_top1']:>14s}  {yes(r['input frame'])}\n"
                    f"{'top-2 gap view':24s}{r['_picked_top1']:>14s}  {yes(r['top-2 gap view'])}\n"
                    f"{'VQF view':24s}{r['_vqf_top1']:>14s}  {yes(r['VQF view'])}\n"
                    f"{'ens. + VQF (abstain)':24s}{'':>14s}  {yes(r['VQF + ensemble (abstain)'])}\n"
                    f"{'ens. + VQF (else input)':24s}{'':>14s}  {yes(r['VQF + ensemble (else input)'])}\n\n"
                    f"ensemble disagreement u = {uncertainty}\n"
                    f"(lambda {LAMBDA}: {'trusted' if r['_trusted'] else 'ABSTAIN'})\n"
                    f"top-2 gap pick: {r['_picked']}\n"
                    f"VQF pick: {r['_vqf_picked']} (abs. el {select.VQF_ELEVATIONS[0]:g}.."
                    f"{select.VQF_ELEVATIONS[1]:g})\n\n"
                    f"For a real FP, REJECTED is the fix;\nfor a goal object it is an error.",
                    family="monospace", fontsize=10.5, va="top", transform=ax.transAxes)
            fig.suptitle(f"{r['episode']} | goal {r['goal']} | {verdicts[r['verdict']]}", fontsize=15)
            fig.text(.01, .012, f"SAM3D candidates: azimuth/elevation relative to the input camera (180/0 = input view, "
                     f"gray box); el = absolute elevation; q = ViewActive VQF quality | box: blue = top-2 gap pick, "
                     f"purple = VQF pick | titles: red = top-1 is the goal, green = another category", fontsize=10)
            path = out_dir / r["verdict"] / f"{r['episode']}.png"
            path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(path, dpi=90)
            pdf.savefig(fig)
            plt.close(fig)
    print(f"Saved {len(rows)} episode pages under {out_dir} and {OUTPUT / 'grids.pdf'}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("stage", choices=("run", "figure", "grids"))
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()
    {"run": run, "figure": figure, "grids": grids}[args.stage](args)


if __name__ == "__main__":
    main()
