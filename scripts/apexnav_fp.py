"""SAM3D reconstruction + SAM3 top-2-gap view selection on ApexNav false-positive episodes.

Each record (``data/apexnav/records/<episode>``, an ApexNav EpisodeRecorder dir whose
``meta.json`` result is "false positive") gives one input: the frame and mask that best
show the object the planner committed to (ApexNav's ``find_commit_evidence``). SAM3D
reconstructs it and renders the study's candidate grid; SAM3 scores a fixed household
vocabulary on the raw frame and on every candidate. An episode is *resolved* when the
top-1 category on the selected view is not the episode goal, i.e. the detection the
planner committed to would have been rejected.

    sam3d-env sam3d python -m scripts.apexnav_fp reconstruct --gpu 0
    sam3d-env core  python -m scripts.apexnav_fp detect --gpu 0
    sam3d-env sam3d python -m scripts.apexnav_fp ensemble --gpu 0
    sam3d-env sam3d python -m scripts.apexnav_fp vqf --gpu 0
    sam3d-env core  python -m scripts.apexnav_fp figure
    sam3d-env core  python -m scripts.apexnav_fp grids

``ensemble`` writes the SAM3D stage-1 seed-ensemble disagreement (``ensemble.json``);
``vqf`` writes the learned ViewActive quality of each candidate (``vqf.json``). Both
only add selectors over the stored SAM3 scores.

Outputs: ``outputs/apexnav_fp/<episode>/`` (input, mask, SAM3D views, scores.json),
``results.json`` and ``apexnav_fp_resolved.png``; ``grids`` (after ``figure``) writes one
page per episode under ``grids/`` and ``grids.pdf``.
"""
import argparse
import json
import math
import os
import shutil
import sys
from pathlib import Path

# Before numpy is imported: huge-page advice stalls the shared host until systemd-oomd
# kills the container (see tools.envs.no_huge_pages).
os.environ.setdefault("NUMPY_MADVISE_HUGEPAGE", "0")

import numpy as np
from PIL import Image

from tools.envs import no_huge_pages, use_gpu

RECORDS = Path("data/apexnav/records")
# The user's manual verification: episodes whose committed object is truly not the goal
# (copied from ApexNav replays/manually_verified/replays). The rest are goal-class objects.
VERIFIED = Path("data/apexnav/verified.txt")
# A second, independent pass (fp | goal | ambiguous + note); reported, not used for the split.
REVIEW = Path("data/apexnav/review.tsv")
APEXNAV_SRC = Path("data/apexnav/apexnav_src")  # copy of basic_utils/record_episode/episode_visualization.py
OUTPUT = Path("outputs/apexnav_fp")
SAM3D_CHECKPOINT = "/data/checkpoints/sam3d/hf"
SAM3_CHECKPOINT = "/data/checkpoints/sam3/sam3.pt"
# Study candidate grid (configs/lvis_uncertainty.yaml).
AZIMUTHS = [0, 45, 90, 135, 180, 225, 270, 315]
ELEVATIONS = [0, 30, 60]
DISTANCE, RESOLUTION, SEED = 2.0, 320, 42
# ApexNav's six HM3D goals plus the object classes of mpcat40 (structure and catch-all
# classes dropped; "plant" dropped as it duplicates "potted plant"). Fixed before
# looking at results.
GOALS = ["chair", "couch", "bed", "toilet", "tv", "potted plant"]
VOCABULARY = GOALS + ["door", "table", "picture", "cabinet", "cushion", "window", "curtain",
                      "chest of drawers", "sink", "stool", "towel", "mirror", "shower", "bathtub",
                      "counter", "fireplace", "lamp", "shelf", "blinds", "exercise equipment",
                      "appliance", "clothes"]
# Stage-1 seed ensemble and abstention threshold (calibrated on LVIS meshes, mesh F < 0.5:
# lambda 0.10-0.20 keeps ~99% useful reconstructions).
ENSEMBLE, LAMBDA = 4, 0.15
# Learned ViewActive predictor and its optimize.elevation_range.
VIEWACTIVE = Path("outputs/viewactive/blender_full_el0_45/model/best.pt")
VQF_ELEVATIONS = (0.0, 45.0)
# A raw-frame detection counts for the committed object when its mask overlaps it this much.
INPUT_IOU = 0.3


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


def reconstruct(args):
    from reconstruction.sam3d import SAM3D

    model = SAM3D(checkpoint_dir=SAM3D_CHECKPOINT)
    records = sorted(p for p in RECORDS.iterdir() if (p / "meta.json").is_file())
    for count, record in enumerate(records, 1):
        info = prepare(record)
        out = OUTPUT / record.name
        if info is None or (out / "candidates.json").is_file() or (out / "error.txt").is_file():
            continue
        try:
            candidates = model.render_views(out / "input.jpg", out / "mask.png", out,
                                            resolution=RESOLUTION, seed=SEED,
                                            view_grid=(AZIMUTHS, ELEVATIONS), save_ply=False,
                                            view_distance=DISTANCE,
                                            input_elevation=info["input_elevation"])
        except Exception as error:  # Recorded per episode; the rest continue.
            if "out of memory" in str(error):
                raise  # Transient (shared GPU); the supervisor retries, the stage resumes.
            (out / "error.txt").write_text(f"{type(error).__name__}: {error}")
            print(f"[reconstruct] {count}/{len(records)} {record.name} failed: {error}", flush=True)
            continue
        (out / "candidates.json").write_text(json.dumps(candidates))
        print(f"[reconstruct] {count}/{len(records)} {record.name}", flush=True)
    left = [r.name for r in records if not any((OUTPUT / r.name / f).is_file()
                                              for f in ("candidates.json", "error.txt"))]
    if left:
        sys.exit(f"[reconstruct] {len(left)} episodes unfinished")


def ensemble(args):
    from reconstruction.sam3d import SAM3D

    model = SAM3D(checkpoint_dir=SAM3D_CHECKPOINT)
    todo = sorted(p for p in OUTPUT.iterdir() if (p / "candidates.json").is_file()
                  and not (p / "ensemble.json").is_file())
    for count, out in enumerate(todo, 1):
        summary = model.stage1_ensemble(out / "input.jpg", out / "mask.png", seed=SEED, ensemble=ENSEMBLE)
        (out / "ensemble.json").write_text(json.dumps(
            {"disagreement": summary["disagreement"], "ensemble": ENSEMBLE}))
        print(f"[ensemble] {count}/{len(todo)} {out.name} u={summary['disagreement']:.3f}", flush=True)


def vqf(args):
    from viewactive.grid import sam3d_to_viewactive
    from viewactive.model import ViewActive

    model = ViewActive.load(VIEWACTIVE)
    for out in sorted(p for p in OUTPUT.iterdir() if (p / "candidates.json").is_file()):
        info = json.loads((out / "input.json").read_text())
        with Image.open(out / "input.jpg") as image, Image.open(out / "mask.png") as mask:
            field = model.predict([image.convert("RGB")], [mask.convert("L")])[0]
        keys, angles = [], []
        for candidate in json.loads((out / "candidates.json").read_text()):
            keys.append(f"{candidate['azimuth']:g},{candidate['elevation']:g}")
            angles.append(sam3d_to_viewactive(candidate["azimuth"], candidate["elevation"],
                                              info["input_elevation"]))
        quality = model.score_views(field, *zip(*angles))
        (out / "vqf.json").write_text(json.dumps({k: {"quality": float(q), "elevation": a[1]}
                                                  for k, q, a in zip(keys, quality, angles)}))
    print(f"[vqf] done", flush=True)


def _iou(a, b):
    union = np.logical_or(a, b).sum()
    return float(np.logical_and(a, b).sum() / union) if union else 0.0


def detect(args):
    from reconstruction.segmentation import SAM3Preprocessor

    model = SAM3Preprocessor(checkpoint=SAM3_CHECKPOINT, threshold=0.0)
    for count, out in enumerate(sorted(p for p in OUTPUT.iterdir() if (p / "candidates.json").is_file()), 1):
        if (out / "scores.json").is_file():
            continue
        with Image.open(out / "input.jpg") as source:
            frame = source.convert("RGB")
        with Image.open(out / "mask.png") as source:
            mask = np.asarray(source.convert("L")) > 127
        # Raw frame: the best detection of each category that overlaps the committed object.
        found = model.detect_many(frame, VOCABULARY)
        scores = {"input": {name: max([float(s) for s, m in zip(d["scores"], d["masks"])
                                       if np.isfinite(s) and _iou(m, mask) >= INPUT_IOU] or [0.0])
                            for name, d in found.items()}}
        for candidate in json.loads((out / "candidates.json").read_text()):
            with Image.open(out / "views" / candidate["image"]) as source:
                found = model.detect_many(source.convert("RGB"), VOCABULARY, spatial_queries=set())
            scores[f"{candidate['azimuth']:g},{candidate['elevation']:g}"] = {
                name: float(np.nanmax(d["scores"])) if np.isfinite(d["scores"]).any() else 0.0
                for name, d in found.items()}
        (out / "scores.json").write_text(json.dumps(scores))
        print(f"[detect] {count} {out.name}", flush=True)
    left = [p.name for p in OUTPUT.iterdir() if (p / "candidates.json").is_file()
            and not (p / "scores.json").is_file()]
    if left:
        sys.exit(f"[detect] {len(left)} episodes unfinished")


def _top(scores):
    ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    return ranked[0][0], ranked[0][1] - ranked[1][1]


def outcomes(scores, target, quality, uncertainty, lam=LAMBDA):
    """{selector: True if its view's top-1 category is not the goal (FP rejected)}.

    ``quality`` = {view: ViewActive quality, elevation}; ``uncertainty`` = ensemble
    disagreement. "+ ensemble" selectors trust the reconstruction only when
    disagreement <= ``lam``; otherwise they abstain (reject the commit) or fall back
    to the input frame.
    """
    views = {k: v for k, v in scores.items() if k != "input"}
    gaps = {k: _top(v)[1] for k, v in views.items()}
    picked = max(gaps, key=gaps.get)
    allowed = [k for k in views if VQF_ELEVATIONS[0] <= quality[k]["elevation"] <= VQF_ELEVATIONS[1]]
    vqf_pick = max(allowed or views, key=lambda k: quality[k]["quality"])
    rejected = {
        "input frame": _top(scores["input"])[0] != target,
        "identity (180,0)": _top(views["180,0"])[0] != target,
        "random view": np.mean([_top(views[k])[0] != target for k in views]),
        "top-2 gap view": _top(views[picked])[0] != target,
        "VQF view": _top(views[vqf_pick])[0] != target,
    }
    trusted = uncertainty <= lam
    rejected["VQF + ensemble (abstain)"] = rejected["VQF view"] if trusted else True
    rejected["VQF + ensemble (else input)"] = rejected["VQF view"] if trusted else rejected["input frame"]
    rejected["gap + ensemble (else input)"] = rejected["top-2 gap view"] if trusted else rejected["input frame"]
    return {**rejected, "_picked": picked, "_picked_top1": _top(views[picked])[0],
            "_vqf_picked": vqf_pick, "_vqf_top1": _top(views[vqf_pick])[0],
            "_input_top1": _top(scores["input"])[0], "_uncertainty": uncertainty, "_trusted": trusted}


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
        if not (out / "scores.json").is_file():
            missing.append({"episode": record.name, "verdict": review[record.name]["verdict"],
                            "reason": (out / "error.txt").read_text() if (out / "error.txt").is_file()
                            else "not run"})
            continue
        info = json.loads((out / "input.json").read_text())
        rows.append({**info, **review[record.name],
                     **outcomes(json.loads((out / "scores.json").read_text()), info["goal"],
                                json.loads((out / "vqf.json").read_text()),
                                json.loads((out / "ensemble.json").read_text())["disagreement"])})
    selectors = ["input frame", "identity (180,0)", "random view", "top-2 gap view", "VQF view",
                 "VQF + ensemble (abstain)", "VQF + ensemble (else input)", "gap + ensemble (else input)"]
    groups = {"fp": "verified false positives", "goal": "excluded: object is the goal class"}
    counts = {v: {"total": sum(r["verdict"] == v for r in review.values()),
                  "processed": sum(r["verdict"] == v for r in rows),
                  **{s: float(sum(r[s] for r in rows if r["verdict"] == v)) for s in selectors}}
              for v in groups}
    fp = [r for r in rows if r["verdict"] == "fp"]
    summary = {"counts": counts, "failed": missing,
               "gap_fixes_input_misses_fp": sum(r["top-2 gap view"] and not r["input frame"] for r in fp),
               "gap_breaks_input_rejections_fp": sum(r["input frame"] and not r["top-2 gap view"] for r in fp),
               "lambda": LAMBDA,
               "trusted": {v: sum(r["_trusted"] for r in rows if r["verdict"] == v) for v in groups},
               # Abstain-variant counts over a lambda sweep: (fp rejected, goal wrongly rejected).
               "lambda_sweep": {f"{lam:.2f}": [sum(r["VQF view"] if r["_uncertainty"] <= lam else 1
                                                    for r in rows if r["verdict"] == v) for v in groups]
                                for lam in np.arange(0.0, 0.41, 0.05)},
               "vocabulary": VOCABULARY, "rows": rows}
    (OUTPUT / "results.json").write_text(json.dumps(summary, indent=1, default=float))

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    colors = ["#9aa5b1", "#9aa5b1", "#9aa5b1"] + ["#2a6fdb"] * (len(selectors) - 3)
    for ax, verdict, label in ((axes[0], "fp", "Real false positives resolved (top-1 != goal)"),
                               (axes[1], "goal", "Correct objects wrongly rejected (top-1 != goal)")):
        n = counts[verdict]["processed"]
        values = [counts[verdict][s] for s in selectors][::-1]
        bars = ax.barh(selectors[::-1], values, color=colors[::-1] if verdict == "fp"
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
    return " ".join(f"{k} {v:.2f}" for k, v in sorted(scores.items(), key=lambda kv: -kv[1])[:n])


def grids(args):
    """Per episode: nominal SAM3 on the input frame, the SAM3D candidate grid with SAM3's
    top-1 on each view, and the top-2 gap and ensemble + VQF picks."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    from scripts.visualize_study_vqf_views import crop_box, thumbnail

    rows = json.loads((OUTPUT / "results.json").read_text())["rows"]
    rows.sort(key=lambda r: (r["verdict"] != "fp", r["episode"]))
    verdicts = {"fp": "verified false positive", "goal": "excluded: object is the goal class"}
    yes = lambda rejected: "REJECTED" if rejected else "accepted (goal)"
    out_dir = OUTPUT / "grids"
    with PdfPages(OUTPUT / "grids.pdf") as pdf:
        for r in rows:
            out = OUTPUT / r["episode"]
            scores = json.loads((out / "scores.json").read_text())
            quality = json.loads((out / "vqf.json").read_text())
            names = {f"{c['azimuth']:g},{c['elevation']:g}": out / "views" / c["image"]
                     for c in json.loads((out / "candidates.json").read_text())}
            fig = plt.figure(figsize=(26, 11))
            outer = fig.add_gridspec(1, 3, width_ratios=[5, len(AZIMUTHS), 3.6], wspace=.04,
                                     left=.01, right=.99, top=.88, bottom=.05)
            ax = fig.add_subplot(outer[0])
            with Image.open(out / "input.jpg") as image, Image.open(out / "mask.png") as mask:
                ax.imshow(image.convert("RGB"))
                ax.contour(np.asarray(mask.convert("L")) > 127, levels=[.5], colors=["#ffd400"], linewidths=1.5)
            ax.set_xticks([]); ax.set_yticks([])
            ax.set_title(f"Nominal SAM3 on the input frame (committed mask in yellow)\n{_ranking(scores['input'])}",
                         fontsize=10, color=GOAL_COLOR if not r["input frame"] else OTHER_COLOR)
            sub = outer[1].subgridspec(len(ELEVATIONS), len(AZIMUTHS), wspace=.05, hspace=.35)
            for e, elevation in enumerate(ELEVATIONS):
                for a, azimuth in enumerate(AZIMUTHS):
                    key = f"{azimuth:g},{elevation:g}"
                    ax = fig.add_subplot(sub[len(ELEVATIONS) - 1 - e, a])
                    if key not in scores:
                        ax.axis("off")
                        continue
                    top, gap = _top(scores[key])
                    color = (GAP_COLOR if key == r["_picked"] else VQF_COLOR if key == r["_vqf_picked"]
                             else IDENTITY_COLOR if key == "180,0" else None)
                    path = names[key]
                    thumbnail(ax, path, f"{azimuth}/{elevation} (el {quality[key]['elevation']:+.0f}) "
                              f"q {quality[key]['quality']:.2f}\n{top} {scores[key][top]:.2f} gap {gap:.2f}",
                              color=color, width=3, fontsize=7,
                              title_color=GOAL_COLOR if top == r["goal"] else OTHER_COLOR,
                              box=crop_box([path], masks=[path.with_name(path.name.replace(".png", ".mask.png"))]))
                    if key == r["_vqf_picked"] and key == r["_picked"]:
                        ax.add_patch(matplotlib.patches.Rectangle((.04, .04), .92, .92, transform=ax.transAxes,
                                                                  fill=False, edgecolor=VQF_COLOR, linewidth=3))
            ax = fig.add_subplot(outer[2]); ax.axis("off")
            trusted = "trusted" if r["_trusted"] else "ABSTAIN"
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
                    f"ensemble disagreement u = {r['_uncertainty']:.3f}\n"
                    f"(lambda {LAMBDA}: {trusted})\n"
                    f"top-2 gap pick: {r['_picked']}\n"
                    f"VQF pick: {r['_vqf_picked']} (abs. el {VQF_ELEVATIONS[0]:g}..{VQF_ELEVATIONS[1]:g})\n\n"
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
    parser.add_argument("stage", choices=("reconstruct", "detect", "ensemble", "vqf", "figure", "grids"))
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()
    use_gpu(args.gpu)
    no_huge_pages()
    {"reconstruct": reconstruct, "detect": detect, "ensemble": ensemble, "vqf": vqf,
     "figure": figure, "grids": grids}[args.stage](args)


if __name__ == "__main__":
    main()
