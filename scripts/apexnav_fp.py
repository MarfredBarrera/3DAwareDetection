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

``--gate DIR`` instead replays the requests an online gate run sent to the perception
server (``DIR/verification/<episode>/c<cluster>_s<step>/``: input, mask, decision.json),
so every gate config is compared on identical inputs, goal objects included. ``compare``
scores them against ``DIR/request_labels.tsv`` (case, label goal|other|ambiguous, object;
from visual inspection) and writes ``compare.{json,csv,png}`` and ``compare_table.png`` under
``outputs/apexnav_fp/gate_<DIR name>/``:

    sam3d-env core python -m scripts.apexnav_fp run --gpu 0 --gate outputs/apexnav_gate/verified_2026-10-22
    sam3d-env core python -m scripts.apexnav_fp compare --gate outputs/apexnav_gate/verified_2026-10-22
    sam3d-env core python -m scripts.apexnav_fp views --gate outputs/apexnav_gate/verified_2026-10-22 --goal couch

``views`` draws, per request with that goal, the input frame, every SAM3D candidate with
SAM3's top-1, and the top2gap/VQF picks (``views_<goal>/``, plus ``summary.png``).
"""
import argparse
import json
import math
import shutil
import textwrap
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


def prepare_request(case, out):
    """Copy one logged gate request (frame, mask, target, elevation, online verdict) into ``out``."""
    if (out / "input.json").is_file():
        return json.loads((out / "input.json").read_text())
    decision = json.loads((case / "decision.json").read_text())
    out.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(case / "input.jpg", out / "input.jpg")
    with Image.open(case / "mask.png") as mask:
        Image.fromarray(((np.asarray(mask.convert("L")) > 0) * 255).astype(np.uint8)).save(out / "mask.png")
    verdict = decision["verdict"]
    info = {"episode": out.name, "run_index": decision["run_index"], "goal": decision["target"],
            "step": decision["evidence"]["step"], "input_elevation": decision["evidence"]["input_elevation"],
            "online": {"accepted": verdict["accepted"], "label": verdict["label"], "judged": verdict["judged"],
                       "uncertainty": verdict["uncertainty"], "server": decision["server"]}}
    (out / "input.json").write_text(json.dumps(info, indent=1))
    return info


def gate_cases(gate):
    """(output dir, case name -> request dir) for the server requests logged by a gate run."""
    requests = sorted(d.parent for d in (gate / "verification").glob("*/c*_s*/decision.json")
                      if (d.parent / "input.jpg").is_file())
    return OUTPUT / f"gate_{gate.name}", {f"{d.parent.name}_{d.name}": d for d in requests}


def run(args):
    from perception import Verifier

    if args.gate:
        root, requests = gate_cases(args.gate)
        cases = [(root / name, lambda out, d=d: prepare_request(d, out)) for name, d in requests.items()]
    else:
        root = OUTPUT
        cases = [(OUTPUT / r.name, lambda out, r=r: prepare(r))
                 for r in sorted(p for p in RECORDS.iterdir() if (p / "meta.json").is_file())]
    with Verifier(selector="vqf", all_views=True, ensemble=True, ensemble_threshold=LAMBDA, gpu=args.gpu,
                  azimuths=AZIMUTHS, elevations=ELEVATIONS, log_dir=root / "logs") as verifier:
        for count, (out, make) in enumerate(cases, 1):
            info = make(out)
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
                              "quality": view.quality, "absolute_elevation": view.absolute_elevation,
                              "scores": view.scores})
            episode = {"input": verifier.classify(out / "input.jpg", out / "mask.png", vocabulary),
                       "uncertainty": verdict.uncertainty, "views": views}
            (out / "episode.json").write_text(json.dumps(episode))
            print(f"[run] {count}/{len(cases)} {out.name} u={verdict.uncertainty}", flush=True)


def outcomes(episode, target, lam=LAMBDA):
    """{selector: True if its view's top-1 category is not the goal (FP rejected)}.

    "+ ensemble" selectors trust the reconstruction only when the disagreement
    <= ``lam``; otherwise they abstain (reject the commit) or fall back to the input frame.
    """
    vocabulary = select.competitors(episode["input"], target)
    keep = lambda scores: {k: v for k, v in scores.items() if k in vocabulary}
    views = {(v["azimuth"], v["elevation"]): v for v in episode["views"]}
    scores = {key: keep(v["scores"]) for key, v in views.items()}
    rejects = lambda key: select.top1(scores[key])[0] != target
    if target in select.FLAT:  # judged on the raw frame only (perception.Verifier skips SAM3D)
        rejects = lambda key: select.top1(keep(episode["input"]))[0] != target
    gap = select.top2gap_pick(scores, {key: v["absolute_elevation"] for key, v in views.items()})
    vqf = select.vqf_pick({key: (v["quality"], v["absolute_elevation"]) for key, v in views.items()})
    rejected = {
        "input frame": select.top1(keep(episode["input"]))[0] != target,
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
            "_input_top1": select.top1(keep(episode["input"]))[0], "_uncertainty": episode["uncertainty"],
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
                               f"{azimuth}/{elevation} (el {view['absolute_elevation']:+.0f}) q {view['quality']:.2f}\n"
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
                    f"VQF pick: {r['_vqf_picked']}\n(both picks: abs. el {select.VIEW_ELEVATIONS[0]:g}.."
                    f"{select.VIEW_ELEVATIONS[1]:g})\n\n"
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


# Gate configs compared by ``compare``: (name, outcomes() keys that must all accept; () = no gate).
GATE_CONFIGS = [("baseline (no gate)", ()), ("SAM3 on input frame", ("input frame",)),
                ("top2gap", ("top-2 gap view",)), ("vqf", ("VQF view",)),
                ("input AND top2gap", ("input frame", "top-2 gap view")),
                ("input AND vqf", ("input frame", "VQF view"))]


# Category colors for ``views`` (top-1 label on a view): the goal, its two main confusers, the rest.
LABEL_COLORS = ("#2a78d6", "#eb6834", "#1baf7a")
OTHER_LABEL = "#a8a6a0"


def views(args):
    """Why do SAM3D views of one goal category fail? Per logged gate request with that goal:
    the input frame and SAM3's ranking on it, every SAM3D candidate (top-1 label, goal score;
    top2gap and VQF picks boxed, views outside the elevation band dimmed), and the picks'
    rankings. ``summary.png``: top-1 labels over all candidates per request, and how many
    requests each judge accepts with and without each confuser in the vocabulary."""
    import csv
    from collections import Counter

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    root, requests = gate_cases(args.gate)
    with (args.gate / "request_labels.tsv").open() as handle:
        labels = {row["case"]: row for row in csv.DictReader(handle, delimiter="\t")}
    cases = []
    for name in requests:
        out = root / name
        info = json.loads((out / "input.json").read_text())
        if info["goal"] == args.goal and labels[name]["label"] != "ambiguous" and (out / "episode.json").is_file():
            cases.append((name, info, json.loads((out / "episode.json").read_text()), labels[name]))
    cases.sort(key=lambda c: (c[3]["label"] != "goal", c[0]))
    tops = Counter(select.top1(v["scores"])[0] for _, _, e, _ in cases for v in e["views"])
    confusers = [k for k, _ in tops.most_common() if k != args.goal][:2]
    palette = dict(zip([args.goal, *confusers], LABEL_COLORS))
    color = lambda label: palette.get(label, OTHER_LABEL)
    out_dir = root / f"views_{args.goal.replace(' ', '_')}"
    out_dir.mkdir(exist_ok=True)

    def judged(episode, drop=()):
        """{judge: top-1 label} for the raw frame and the top2gap and VQF picks."""
        keep = lambda scores: {k: v for k, v in scores.items() if k not in drop}
        views = {(v["azimuth"], v["elevation"]): v for v in episode["views"]}
        scores = {key: keep(v["scores"]) for key, v in views.items()}
        gap = select.top2gap_pick(scores, {key: v["absolute_elevation"] for key, v in views.items()})
        vqf = select.vqf_pick({key: (v["quality"], v["absolute_elevation"]) for key, v in views.items()})
        return {"raw frame": select.top1(keep(episode["input"]))[0], "top2gap": select.top1(scores[gap])[0],
                "vqf": select.top1(scores[vqf])[0]}, gap, vqf

    def bars(ax, scores, title):
        order = select.ranked(scores)[:5][::-1]
        ax.barh([k for k, _ in order], [v for _, v in order], color=[color(k) for k, _ in order], height=0.6)
        for i, (_, v) in enumerate(order):
            ax.text(v + 0.02, i, f"{v:.2f}", va="center", fontsize=8)
        ax.set_xlim(0, 1.15), ax.tick_params(labelsize=8, length=0)
        ax.set_title(title, fontsize=9, loc="left")
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)

    for name, info, episode, label in cases:
        out = root / name
        views = {(v["azimuth"], v["elevation"]): v for v in episode["views"]}
        decision, gap, vqf = judged(episode)
        low, high = select.VIEW_ELEVATIONS
        box = _crop_box([out / "views" / v["image"] for v in views.values()])
        fig = plt.figure(figsize=(26, 10.5))
        outer = fig.add_gridspec(1, 3, width_ratios=[5, len(AZIMUTHS), 3.4], wspace=.08, left=.04, right=.99,
                                 top=.88, bottom=.08)
        left = outer[0].subgridspec(3, 1, height_ratios=[3, 2, 1.6], hspace=.3)
        ax = fig.add_subplot(left[0])
        with Image.open(out / "input.jpg") as image, Image.open(out / "mask.png") as mask:
            frame, silhouette = image.convert("RGB"), np.asarray(mask.convert("L")) > 127
        ax.imshow(frame), ax.contour(silhouette, levels=[.5], colors=["#ffd400"], linewidths=1.5)
        ax.set_xticks([]), ax.set_yticks([])
        ax.set_title(f"Input frame (mask {silhouette.mean():.1%} of the frame)", fontsize=10)
        ys, xs = np.nonzero(silhouette)
        pad = 30
        crop = frame.crop((max(xs.min() - pad, 0), max(ys.min() - pad, 0), min(xs.max() + pad, frame.width),
                           min(ys.max() + pad, frame.height)))
        ax = fig.add_subplot(left[1])
        ax.imshow(crop), ax.set_xticks([]), ax.set_yticks([]), ax.set_title("Masked object", fontsize=10)
        bars(fig.add_subplot(left[2]), episode["input"], f"SAM3 on the raw frame -> {decision['raw frame']}")
        grid = outer[1].subgridspec(len(ELEVATIONS), len(AZIMUTHS), wspace=.05, hspace=.35)
        for e, elevation in enumerate(ELEVATIONS):
            for a, azimuth in enumerate(AZIMUTHS):
                key = (float(azimuth), float(elevation))
                view = views[key]
                top = select.top1(view["scores"])[0]
                ax = fig.add_subplot(grid[len(ELEVATIONS) - 1 - e, a])
                edge = GAP_COLOR if key == gap else VQF_COLOR if key == vqf else None
                _thumbnail(ax, out / "views" / view["image"],
                           f"{azimuth}/{elevation} (abs {view['absolute_elevation']:+.0f})\n"
                           f"{top} {view['scores'][top]:.2f} | {args.goal} {view['scores'][args.goal]:.2f}",
                           color=edge, title_color=color(top), box=box)
                if key == gap and key == vqf:
                    ax.add_patch(matplotlib.patches.Rectangle((.04, .04), .92, .92, transform=ax.transAxes,
                                                              fill=False, edgecolor=VQF_COLOR, linewidth=3))
                if not low <= view["absolute_elevation"] <= high:
                    ax.add_patch(matplotlib.patches.Rectangle((0, 0), 1, 1, transform=ax.transAxes,
                                                              facecolor="white", alpha=.55, edgecolor="none"))
        right = outer[2].subgridspec(3, 1, height_ratios=[1.6, 1.6, 2.2], hspace=.45)
        bars(fig.add_subplot(right[0]), views[gap]["scores"], f"top2gap pick {gap[0]:g}/{gap[1]:g} -> {decision['top2gap']}")
        bars(fig.add_subplot(right[1]), views[vqf]["scores"], f"VQF pick {vqf[0]:g}/{vqf[1]:g} -> {decision['vqf']}")
        ax = fig.add_subplot(right[2]); ax.axis("off")
        without = {c: judged(episode, (c,))[0] for c in confusers}
        verdict = lambda top: "accept" if top == args.goal else "reject"
        lines = [f"{'judge':10s}{'top-1':>10s}  decision"] + [f"{j:10s}{t:>10s}  {verdict(t)}" for j, t in decision.items()]
        for c, d in without.items():
            lines += ["", f"without '{c}' in the vocabulary:"] + [f"{j:10s}{t:>10s}  {verdict(t)}" for j, t in d.items()]
        ax.text(0, 1, "\n".join(lines), family="monospace", fontsize=9.5, va="top", transform=ax.transAxes)
        truth = "a real " + args.goal if label["label"] == "goal" else f"NOT a {args.goal}"
        fig.suptitle(f"{name} | goal {args.goal} | object: {label['object']} ({truth})", fontsize=15)
        fig.text(.01, .01, "SAM3D candidates: azimuth/elevation relative to the input camera (180/0 = input view); "
                 f"abs = elevation in the gravity frame (views outside {low:g}-{high:g} are dimmed: neither selector "
                 "may pick them) | title: SAM3 top-1 and its score, then the goal's score; colors: "
                 + ", ".join(f"{k} = {n}" for k, n in zip(palette, ("blue", "orange", "green"))) + ", gray = other"
                 "\nbox: blue = top2gap pick, purple = VQF pick", fontsize=10)
        fig.savefig(out_dir / f"{label['label']}_{name}.png", dpi=80)
        plt.close(fig)

    # Summary: top-1 label shares over every candidate, then acceptance with and without each confuser.
    fig, axes = plt.subplots(1, 2, figsize=(15, 0.3 * len(cases) + 2.2), gridspec_kw={"width_ratios": [3, 2]},
                             facecolor="#fcfcfb")
    ax = axes[0]
    names = [f"{c[3]['object'][:24]} ({'goal' if c[3]['label'] == 'goal' else 'not goal'}) {c[0].split('_')[0]}"
             for c in cases]
    for i, (_, _, episode, _) in enumerate(cases):
        count = Counter(select.top1(v["scores"])[0] for v in episode["views"])
        left = 0
        for category in [*palette, "other"]:
            n = sum(v for k, v in count.items() if (k == category if category != "other" else k not in palette))
            if n:
                ax.barh(i, n, left=left, color=color(category) if category != "other" else OTHER_LABEL,
                        edgecolor="#fcfcfb", linewidth=1, height=.75)
            left += n
    ax.set_yticks(range(len(cases)), names, fontsize=8), ax.invert_yaxis()
    split = sum(c[3]["label"] == "goal" for c in cases)
    ax.axhline(split - .5, color="#0b0b0b", lw=1)
    ax.set_xlim(0, len(AZIMUTHS) * len(ELEVATIONS)), ax.set_xlabel("SAM3D candidate views (top-1 label)")
    ax.legend([plt.Rectangle((0, 0), 1, 1, color=c) for c in [*palette.values(), OTHER_LABEL]], [*palette, "other"],
              ncol=4, frameon=False, loc="lower left", bbox_to_anchor=(0, 1), fontsize=9)
    ax = axes[1]
    groups = [("goal", f"real {args.goal}s accepted"), ("other", f"non-{args.goal}s accepted (false accepts)")]
    settings = [("full vocabulary", ())] + [(f"without '{c}'", (c,)) for c in confusers]
    judges = ["raw frame", "top2gap", "vqf"]
    rows, ticks = [], []
    for group, title in groups:
        members = [c for c in cases if c[3]["label"] == group]
        for setting, drop in settings:
            accepted = Counter(j for c in members for j, t in judged(c[2], drop)[0].items() if t == args.goal)
            rows.append([accepted[j] for j in judges])
            ticks.append(f"{title}\n{setting} (n={len(members)})")
    y = np.arange(len(rows))
    for k, (judge, c) in enumerate(zip(judges, ("#52514e", GAP_COLOR, VQF_COLOR))):
        ax.barh(y + (k - 1) * .26, [r[k] for r in rows], height=.25, color=c, label=judge)
        for yy, r in zip(y, rows):
            ax.text(r[k] + .2, yy + (k - 1) * .26, str(r[k]), va="center", fontsize=8)
    ax.set_yticks(y, ticks, fontsize=8), ax.invert_yaxis(), ax.legend(fontsize=9, frameon=False)
    ax.set_xlabel("requests accepted as " + args.goal)
    for a in axes:
        a.set_facecolor("#fcfcfb")
        for side in ("top", "right"):
            a.spines[side].set_visible(False)
    fig.suptitle(f"Gate requests with goal '{args.goal}' ({args.gate.name}; visual labels; ambiguous excluded)",
                 fontsize=12)
    fig.tight_layout()
    fig.savefig(out_dir / "summary.png", dpi=130, facecolor="#fcfcfb")
    print(f"{len(cases)} requests -> {out_dir}", flush=True)


def compare(args):
    """Accept/reject of every gate config on the logged requests, scored against visual labels."""
    import csv

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    root, requests = gate_cases(args.gate)
    with (args.gate / "request_labels.tsv").open() as handle:
        labels = {row["case"]: row for row in csv.DictReader(handle, delimiter="\t")}
    rows = []
    for name in requests:
        out = root / name
        if not (out / "episode.json").is_file():
            continue
        info = json.loads((out / "input.json").read_text())
        rejected = outcomes(json.loads((out / "episode.json").read_text()), info["goal"])
        row = {"case": name, "goal": info["goal"], "label": labels[name]["label"], "object": labels[name]["object"],
               "online_accepted": info["online"]["accepted"], "uncertainty": rejected["_uncertainty"]}
        for config, keys in GATE_CONFIGS:
            row[config] = not any(rejected[key] for key in keys)
        rows.append(row)
    groups = {"other": "wrong objects rejected", "goal": "goal objects accepted"}
    counts = {g: {"n": sum(r["label"] == g for r in rows),
                  **{c: sum((r[c] if g == "goal" else not r[c]) for r in rows if r["label"] == g)
                     for c, _ in GATE_CONFIGS}} for g in groups}
    summary = {"requests": len(requests), "processed": len(rows),
               "ambiguous": sum(r["label"] == "ambiguous" for r in rows), "counts": counts,
               "elevation_range": select.VIEW_ELEVATIONS}
    (root / "compare.json").write_text(json.dumps({**summary, "rows": rows}, indent=1, default=float))
    episodes = {}
    for r in rows:
        if r["label"] != "ambiguous":
            episodes.setdefault(r["case"].split("_")[0], []).append(r)
    n_other, n_goal = counts["other"]["n"], counts["goal"]["n"]
    headers = ["Gate", f"Wrong objects\naccepted (of {n_other})", f"Episodes with a\nwrong accept (of {len(episodes)})",
               f"Goals accepted\n(of {n_goal})", "Precision"]
    table = []
    for config, _ in GATE_CONFIGS:
        wrong, goals = n_other - counts["other"][config], counts["goal"][config]
        bad = sum(any(r[config] for r in v if r["label"] == "other") for v in episodes.values())
        table.append([config, str(wrong), str(bad), str(goals), f"{goals / max(goals + wrong, 1):.0%}"])
    caption = (f"{len(rows)} verification requests of {args.gate.name} (same frames and masks; labels by visual "
               f"inspection, {summary['ambiguous']} ambiguous excluded). Views from absolute elevations "
               f"{select.VIEW_ELEVATIONS[0]:g}-{select.VIEW_ELEVATIONS[1]:g}; target parts do not compete "
               f"({', '.join(f'{p} for {t}' for t, ps in select.PARTS.items() for p in ps)}); flat targets "
               f"({', '.join(select.FLAT)}) are judged on the raw frame only. The requests are those the online gate "
               "run sent, so this shows what each gate filters, not closed-loop navigation.")
    x = [0.0, 0.27, 0.46, 0.67, 0.85]
    fig, ax = plt.subplots(figsize=(11, 0.5 * len(table) + 2.2), facecolor="#fcfcfb")
    ax.set_xlim(0, 1), ax.set_ylim(len(table) + 1.6, -1.1), ax.axis("off")
    for k, header in enumerate(headers):
        ax.text(x[k] + 0.01, -0.45, header, fontsize=10.5, color="#52514e", weight="bold", va="center")
    ax.axhline(0.05, color="#e4e3df", lw=1)
    for i, line in enumerate(table):
        y = i + 0.55
        for k, value in enumerate(line):
            ax.text(x[k] + 0.01, y, value, fontsize=11, color="#0b0b0b", va="center")
        ax.axhline(y + 0.5, color="#e4e3df", lw=0.8)
    ax.text(0.01, len(table) + 0.95, textwrap.fill(caption, 150), fontsize=8.5, color="#52514e", va="center")
    fig.tight_layout()
    fig.savefig(root / "compare_table.png", dpi=150, facecolor="#fcfcfb")
    plt.close(fig)
    with (root / "compare.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(), writer.writerows(rows)

    names = [c for c, _ in GATE_CONFIGS]
    fig, axes = plt.subplots(1, 2, figsize=(12, 3.8), facecolor="#fcfcfb", sharey=True)
    for ax, (group, title), color in zip(axes, groups.items(), ("#d03b3b", "#0ca30c")):
        n = counts[group]["n"]
        values = [counts[group][c] for c in names][::-1]
        bars = ax.barh(names[::-1], values, color=color, height=0.6)
        for bar, value in zip(bars, values):
            ax.text(bar.get_width() + n * 0.01, bar.get_y() + bar.get_height() / 2, f"{value} / {n}",
                    va="center", fontsize=10, color="#0b0b0b")
        ax.set_xlim(0, n * 1.18), ax.set_title(f"{title} (n={n})", fontsize=11, loc="left", color="#0b0b0b")
        ax.set_facecolor("#fcfcfb"), ax.tick_params(colors="#52514e", length=0)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    fig.suptitle(f"Gate configs on the {len(rows)} verification requests of {args.gate.name} "
                 f"(same frames and masks; labels by visual inspection, {summary['ambiguous']} ambiguous excluded)",
                 fontsize=11, color="#0b0b0b")
    fig.tight_layout()
    fig.savefig(root / "compare.png", dpi=140, facecolor="#fcfcfb")
    print(json.dumps(summary, indent=1))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("stage", choices=("run", "figure", "grids", "compare", "views"))
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--gate", type=Path, help="replay the requests logged by this gate results dir")
    parser.add_argument("--goal", default="couch", help="views: the goal category to study")
    args = parser.parse_args()
    if args.stage in ("compare", "views") and not args.gate:
        parser.error(f"{args.stage} needs --gate")
    {"run": run, "figure": figure, "grids": grids, "compare": compare, "views": views}[args.stage](args)


if __name__ == "__main__":
    main()
