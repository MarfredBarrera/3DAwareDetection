"""Summarize the sync_step comparison of ApexNav verification strategies.

Runs in the apexnav container (numpy + matplotlib), reading the run dirs written by
ApexNav's scripts/run_strategy.sh (formerly videos/sync_runs/run_strategy.sh):

    docker exec -i apexnav bash -c "source /opt/apexnav/apexnav-env.sh; use_core; cd /workspace/ApexNav; \
        python - videos/sync_runs/summary videos/sync_runs/sample_200_seed20261025.json \
        baseline=videos/sync_baseline/baseline sam3=videos/sync_sam3/gate \
        active=videos/sync_active/gate" < scripts/apexnav_sync_summary.py

Writes results.csv (one row per episode x strategy), summary.json (per-strategy metrics
on the episodes every strategy finished, paired differences vs baseline with a paired
bootstrap over episodes, per-category rates) and table.png. The table title names the
sample's dataset and split (samples without a split are HM3D-v2 val).
"""
import csv
import json
import os
import sys

import numpy as np

out, sample_file = sys.argv[1], sys.argv[2]
runs = dict(arg.split("=", 1) for arg in sys.argv[3:])
os.makedirs(out, exist_ok=True)
sample = json.load(open(sample_file))
category = {e["index"]: e["category"] for e in sample["episodes"]}
episodes = [e["index"] for e in sample["episodes"]]
DATASETS = {"hm3dv1": "HM3D-v1", "hm3dv2": "HM3D-v2", "mp3d": "MP3D", "ovon": "HM3D-OVON"}
title = (f"{DATASETS.get(sample['dataset'], sample['dataset'])} {sample['split']}" if "split" in sample
         else "HM3D-v2 val")


def decisions(root, n):
    path = f"{root}/epi{n}/verification/decisions.jsonl"
    if not os.path.exists(path):
        return []
    return [json.loads(line) for line in open(path) if line.strip()]


rows = []
for name, root in runs.items():
    for n in episodes:
        path = f"{root}/epi{n}/episodes.jsonl"
        if not os.path.exists(path):
            continue
        e = json.loads(open(path).readline())
        v = e.get("verification") or {}
        ds = decisions(root, n)
        gt = [(d.get("ground_truth") or {}).get("is_correct_instance") for d in ds]
        states = [d.get("state") for d in ds]
        rows.append({
            "strategy": name, "episode": n, "category": category.get(n), "scene": e["scene"],
            "episode_id": e["episode_id"], "result": e["result"], "success": float(e["success"]),
            "spl": float(e["spl"]), "false_positive": int(e["result"] == "false positive"),
            "count_steps": e["count_steps"], "elapsed_s": e.get("elapsed_s"),
            "requests": v.get("requests", 0), "accepted": v.get("accepted", 0),
            "rejected": v.get("rejected", 0), "errors": v.get("errors", 0),
            "no_evidence": v.get("no_evidence", 0), "cached": v.get("cached", 0),
            "reopened": v.get("reopened", 0),
            # active verification (strategy "active"): insufficient observations, viewpoint
            # moves and how the moves ended
            "insufficient": v.get("insufficient", 0), "moves": v.get("moves", 0),
            "unreachable": v.get("unreachable", 0), "aborted": v.get("aborted", 0),
            "not_redetected": v.get("not_redetected", 0), "fallback": v.get("fallback", 0),
            # ground truth of the judged evidence view (semantic mask overlap with a goal object)
            "reject_goal": sum(s == -1 and g is True for s, g in zip(states, gt)),
            "reject_nongoal": sum(s == -1 and g is False for s, g in zip(states, gt)),
            "accept_goal": sum(s == 1 and g is True for s, g in zip(states, gt)),
            "accept_nongoal": sum(s == 1 and g is False for s, g in zip(states, gt)),
        })

with open(f"{out}/results.csv", "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)

by = {(r["strategy"], r["episode"]): r for r in rows}
common = [n for n in episodes if all((s, n) in by for s in runs)]
rng = np.random.default_rng(0)
boot = rng.integers(0, len(common), size=(10000, len(common))) if common else None


def metric(s, key):
    return np.array([by[(s, n)][key] for n in common], dtype=float)


summary = {"episodes_all_strategies": len(common), "episodes_per_strategy": {
    s: sum(1 for n in episodes if (s, n) in by) for s in runs}, "strategies": {}}
for s in runs:
    entry = {}
    for key in ("success", "spl", "false_positive"):
        x = metric(s, key)
        entry[key] = float(x.mean())
        if s != "baseline" and "baseline" in runs:
            d = x - metric("baseline", key)
            means = d[boot].mean(axis=1)
            entry[f"{key}_diff"] = float(d.mean())
            entry[f"{key}_diff_ci95"] = [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))]
    for key in ("requests", "accepted", "rejected", "errors", "no_evidence", "cached", "reopened",
                "reject_goal", "reject_nongoal", "accept_goal", "accept_nongoal", "insufficient", "moves",
                "unreachable", "aborted", "not_redetected", "fallback", "count_steps"):
        entry[key] = float(metric(s, key).sum()) if key != "count_steps" else float(metric(s, key).mean())
    entry["minutes_per_episode"] = float(np.mean([by[(s, n)]["elapsed_s"] or 0 for n in common]) / 60)
    if s != "baseline" and "baseline" in runs:
        b, x = metric("baseline", "success"), metric(s, "success")
        entry["paired_success"] = {"gained": int(((x == 1) & (b == 0)).sum()),
                                   "lost": int(((x == 0) & (b == 1)).sum())}
        bf, xf = metric("baseline", "false_positive"), metric(s, "false_positive")
        entry["paired_fp"] = {"fp_removed": int(((xf == 0) & (bf == 1)).sum()),
                              "fp_added": int(((xf == 1) & (bf == 0)).sum())}
        changed = [n for n in common if by[(s, n)]["result"] != by[("baseline", n)]["result"]]
        entry["outcome_changed"] = {str(n): [by[("baseline", n)]["result"], by[(s, n)]["result"]]
                                    for n in changed}
    entry["per_category_success"] = {c: float(np.mean([by[(s, n)]["success"] for n in common if category[n] == c]))
                                     for c in sorted(set(category[n] for n in common))}
    summary["strategies"][s] = entry
json.dump(summary, open(f"{out}/summary.json", "w"), indent=1)

# PNG table
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def pct(v):
    return f"{100 * v:.1f}"

def ci(e, key):
    if f"{key}_diff" not in e:
        return "-"
    lo, hi = e[f"{key}_diff_ci95"]
    return f"{100 * e[f'{key}_diff']:+.1f} [{100 * lo:+.1f}, {100 * hi:+.1f}]"

header = ["strategy", "success %", "SPL %", "FP %", "d succ pp\n[95% CI]", "d FP pp\n[95% CI]",
          "succ\ngained/lost", "FP\nremoved/added", "verif.\nrequests", "rejected", "rejected\ngoal/non-goal",
          "moves\n(unreach./aborted)", "steps/epi", "min/epi\n(sim loop)"]
table = []
for s, e in summary["strategies"].items():
    ps, pf = e.get("paired_success"), e.get("paired_fp")
    table.append([s, pct(e["success"]), pct(e["spl"]), pct(e["false_positive"]), ci(e, "success"),
                  ci(e, "false_positive"), f"{ps['gained']}/{ps['lost']}" if ps else "-",
                  f"{pf['fp_removed']}/{pf['fp_added']}" if pf else "-", f"{e['requests']:.0f}",
                  f"{e['rejected']:.0f}", f"{e['reject_goal']:.0f} / {e['reject_nongoal']:.0f}",
                  f"{e['moves']:.0f} ({e['unreachable']:.0f}/{e['aborted']:.0f})", f"{e['count_steps']:.0f}",
                  f"{e['minutes_per_episode']:.1f}"])
fig, ax = plt.subplots(figsize=(20, 0.5 + 0.45 * (len(table) + 1)))
ax.axis("off")
t = ax.table(cellText=table, colLabels=header, loc="center", cellLoc="center")
t.auto_set_font_size(False)
t.set_fontsize(9)
t.scale(1, 2.2)
for j in range(len(header)):
    t[0, j].set_facecolor("#e8e8e8")
ax.set_title(f"ApexNav sync_step, {title}, {len(common)} paired episodes "
             f"(seed {sample['seed']}); diffs vs baseline, paired bootstrap", fontsize=10)
fig.savefig(f"{out}/table.png", dpi=150, bbox_inches="tight")
print(json.dumps({s: {k: e[k] for k in ("success", "spl", "false_positive", "requests", "rejected", "moves")}
                  for s, e in summary["strategies"].items()}, indent=1))
print(f"{len(common)} episodes common to all strategies")
