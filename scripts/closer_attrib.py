"""CLOSER-style failure attribution computed only from simulator data (no eye labels).

  python3 scripts/closer_attrib.py ARM [ARM ...] --episodes FILE [--per-episode OUT.tsv]
            [--cand all|confident] [--min-goal-px N] [--merge-m 1.0]

ARM is NAME=run1+run2 (run dirs under ~/codes/ApexNav/videos; the first run holding an episode
wins; an episode counts only once its episodes.jsonl has a final line with "result").

Every failed episode (Habitat success == 0) gets exactly one cause, in CLOSER's priority order
(docs/papers/CLOSER_2610.28476v1.pdf, Sec. III-A and IV-A):
  FS   the episode ends with a STOP (final_state == REACH_OBJECT) that accepted a non-target.
  FR   >= 1 true-target candidate was detected and then rejected / abandoned (event ended, no stop).
  EXPL no true target candidate was detected at all.
  OTHER a true target was detected but no FS/FR: candidate still pending when the episode ended
        (timeout / collision) or the STOP was on a true target but outside Habitat's 1 m radius.
"True target" = the candidate's mask overlaps the simulator's semantic instance ids of the target
category (HM3D annotations, logged per detection as `semantic_instances`).  See README block at the
bottom of this file and docs in the module functions for per-arm field usage.
"""
import argparse, csv, glob, json, math, os, sys
from collections import Counter
from math import comb

V = os.path.expanduser("~/codes/ApexNav/videos/")
LEDGER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "outputs", "apexnav_online", "labels", "ledger.tsv")
STOP = 4  # FINAL_RESULT.REACH_OBJECT
CAUSES = ["S", "FS", "FR", "EXPL", "OTHER"]


def jl(path):
    out = []
    try:
        for l in open(path):
            l = l.strip()
            if l:
                try:
                    out.append(json.loads(l))
                except ValueError:
                    pass  # partially written line of a running worker
    except OSError:
        pass
    return out


def find_episode(runs, ep):
    for run in runs:
        for d in glob.glob(f"{V}{run}/*/epi{ep}"):
            rows = jl(d + "/episodes.jsonl")
            if rows and "result" in rows[-1]:
                return run, d, rows[-1]
    return None


def is_active(c):
    """Cluster the planner is still pursuing / waiting on: confident and not rejected, or the
    gate accepted (1), deferred (2) or is approaching (3) it."""
    v = c.get("verification", 0)
    return v in (1, 2, 3) or (v == 0 and c.get("is_confident"))


def analyse(d, E, cand="all", min_goal_px=0, merge_m=1.0):
    """Return dict(cause, flags...) for one episode directory."""
    meta = {}
    sp = glob.glob(d + "/records/*/*/steps.jsonl")
    steps = jl(sp[0]) if sp else []
    if sp:
        try:
            meta = json.load(open(os.path.join(os.path.dirname(sp[0]), "meta.json")))
        except (OSError, ValueError):
            pass
    goal_ids = {str(g["object_id"]) for g in meta.get("ground_truth_goals", []) if g.get("object_id") is not None}
    goal_ids |= {str(x) for x in (E.get("commit_ground_truth") or {}).get("goal_object_ids", [])}
    thr = meta.get("ground_truth_semantic_overlap", 0.5)
    dec = jl(d + "/verification/decisions.jsonl")
    gate_type = bool(dec)

    # ---- per-cluster semantic evidence from the logged detections (label_idx 0 = target) ----
    cl = {}  # id -> dict(goal_px, sem_px, ever_conf, ever_ver, last, last_step)
    for r in steps:
        cs = r["object_fusion"]["clusters"]
        t0 = [c for c in cs if c["best_label"] == 0]
        for c in cs:
            if c["best_label"] == 0 or c["id"] in cl:
                s = cl.setdefault(c["id"], dict(goal_px=0, sem_px=0, ever_conf=False, ever_ver=False))
                s["last"], s["last_step"] = c, r["step"]
                s["ever_conf"] |= bool(c.get("is_confident"))
                s["ever_ver"] |= c.get("verification", 0) != 0
        for det in r["detections"]:
            if det["label_idx"] != 0 or det.get("world_centroid") is None or not t0:
                continue
            c = min(t0, key=lambda c: math.dist(det["world_centroid"], c["centroid"]))
            if math.dist(det["world_centroid"], c["centroid"]) > 1.0:
                continue
            si = det.get("semantic_instances") or []
            cl[c["id"]]["sem_px"] += sum(i["pixels"] for i in si)
            cl[c["id"]]["goal_px"] += sum(i["pixels"] for i in si if str(i["id"]) in goal_ids)
    dec_true = {}
    for x in dec:
        gt = x.get("ground_truth")
        if gt and x.get("cluster"):
            dec_true.setdefault(x["cluster"]["id"], []).append(bool(gt.get("is_correct_instance")))
    for cid, s in cl.items():
        s["true"] = (s["sem_px"] > 0 and s["goal_px"] >= min_goal_px and s["goal_px"] / s["sem_px"] >= thr) \
            or any(dec_true.get(cid, []))
        s["cand"] = cand == "all" or s["ever_conf"] or s["ever_ver"]

    last = steps[-1] if steps else None
    stopped = E.get("final_state") == STOP
    # ---- which candidate did the final STOP accept, and is it a target? ----
    acc_cid, acc_true, acc_src = None, False, "none"
    if stopped:
        accepted = [x for x in dec if (x.get("verdict") or {}).get("accepted") or x.get("status") == "correct"]
        if gate_type and accepted:
            x = accepted[-1]
            acc_cid = x["cluster"]["id"]
            gt = x.get("ground_truth")
            if gt is not None:
                acc_true, acc_src = bool(gt.get("is_correct_instance")), "decision_gt"
            elif acc_cid in cl:
                acc_true, acc_src = cl[acc_cid]["true"], "cluster_gt"
            else:
                acc_src = "unresolved"
        else:
            pcs = [r["planner_commit"] for r in steps if r.get("planner_commit")]
            if pcs:
                acc_cid = pcs[-1]["cluster_id"]
            cg = E.get("commit_ground_truth")
            if cg is not None and cg.get("is_correct_instance") is not None:
                acc_true, acc_src = bool(cg["is_correct_instance"]), "commit_gt"
            elif acc_cid in cl:
                acc_true, acc_src = cl[acc_cid]["true"], "cluster_gt"
            else:
                acc_src = "unresolved"  # fallback stop with no target cluster / no semantic evidence
        # ApexNav also ends with STOP after SEARCH_EXTREME (expl_result 6 / no frontier left): it gives
        # up, it does not accept a candidate.  Only the other planner modes accept an object.
        if acc_src == "unresolved" and E.get("expl_result") not in (1, 2, 3):
            acc_src = "gave_up"

    # ---- fate of every true candidate at the end of the episode ----
    true_ids = [i for i, s in cl.items() if s["true"] and s["cand"]]
    active = {c["id"]: c for c in (last["object_fusion"]["clusters"] if last else []) if c["best_label"] == 0 and is_active(c)}
    keep = [cl[i]["last"]["centroid"] for i in true_ids if i in active or i == acc_cid]
    pending = ended = 0
    for i in true_ids:
        if i == acc_cid:
            continue
        if i in active:
            pending += 1
        elif not any(math.dist(cl[i]["last"]["centroid"], k) <= merge_m for k in keep):
            ended += 1  # rejected by gate (verification -1), never confirmed, or lost: reject / abandon
    true_detected = bool(true_ids) or (stopped and acc_true)

    if E.get("success"):
        cause = "S"
    elif stopped and acc_src != "gave_up" and not acc_true:
        cause = "FS"
    elif ended:
        cause = "FR"
    elif not true_detected:
        cause = "EXPL"
    else:
        cause = "OTHER"
    return dict(cause=cause, stopped=int(stopped), acc_src=acc_src, acc_true=int(acc_true),
                n_true=len(true_ids), n_ended=ended, n_pending=pending, gate=int(gate_type),
                final_state=E.get("final_state"), steps=E.get("count_steps"), spl=E.get("spl", 0.0) or 0.0,
                success=int(bool(E.get("success"))), result=E.get("result"), run=None)


def mcnemar(b, c):
    n = b + c
    return 1.0 if n == 0 else min(1.0, 2 * sum(comb(n, i) for i in range(min(b, c) + 1)) / 2 ** n)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("arms", nargs="+", help="NAME=run1+run2")
    ap.add_argument("--episodes", required=True, help="whitespace/line separated episode ids")
    ap.add_argument("--per-episode", help="write a per-episode TSV here")
    ap.add_argument("--cand", choices=["all", "confident"], default="all",
                    help="all: any target-label object-map cluster is a candidate (default); "
                         "confident: only clusters that became confident or reached the gate")
    ap.add_argument("--ledger", action="store_true",
                    help="report-only: confusion of automatic cause vs eye label in labels/ledger.tsv (never an input)")
    ap.add_argument("--min-goal-px", type=int, default=0, help="min summed goal pixels for a true cluster")
    ap.add_argument("--merge-m", type=float, default=1.0, help="dedupe radius for rejected clusters near a live true one")
    a = ap.parse_args()
    eps = [int(x) for x in open(a.episodes).read().split()]
    R = {}
    for arm in a.arms:
        name, runs = arm.split("=", 1)
        R[name] = {}
        for ep in eps:
            f = find_episode(runs.split("+"), ep)
            if f:
                res = analyse(f[1], f[2], a.cand, a.min_goal_px, a.merge_m)
                res["run"] = f[0]
                R[name][ep] = res
    print(f"episodes file: {a.episodes} ({len(eps)} ids); cand={a.cand} min_goal_px={a.min_goal_px} merge_m={a.merge_m}")
    print(f"{'arm':10s} {'n':>5s} {'S':>12s} {'FS':>11s} {'FR':>11s} {'Expl':>12s} {'Other':>11s} {'FS+FR':>12s} {'SPL':>6s}")
    for name, d in R.items():
        n = len(d)
        c = Counter(r["cause"] for r in d.values())
        p = lambda k: f"{c[k]:4d} ({100 * c[k] / max(n, 1):4.1f})"
        spl = 100 * sum(r["spl"] for r in d.values()) / max(n, 1)
        fsfr = c["FS"] + c["FR"]
        print(f"{name:10s} {n:5d} {p('S'):>12s} {p('FS'):>11s} {p('FR'):>11s} {p('EXPL'):>12s} {p('OTHER'):>11s} "
              f"{fsfr:4d} ({100 * fsfr / max(n, 1):4.1f}) {spl:6.1f}")
    names = list(R)
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            x, y = names[i], names[j]
            E = [e for e in R[x] if e in R[y]]
            b = sum(1 for e in E if R[x][e]["success"] and not R[y][e]["success"])
            c = sum(1 for e in E if R[y][e]["success"] and not R[x][e]["success"])
            print(f"McNemar S {x} vs {y}: n={len(E)} {x}-only={b} {y}-only={c} p={mcnemar(b, c):.3f}")
    if a.ledger:
        led = {(r["run"], int(r["episode"])): r["label"]
               for r in csv.DictReader(open(LEDGER), delimiter="\t")}
        for name, d in R.items():
            cnt = Counter((r["cause"], led[(r["run"], e)]) for e, r in d.items() if (r["run"], e) in led)
            labs = sorted({k[1] for k in cnt})
            print(f"\nagreement {name}: rows = automatic cause, cols = eye label (ledger), episodes with a ledger row: "
                  f"{sum(cnt.values())}/{len(d)}")
            print(f"{'':6s}" + "".join(f"{l:>9s}" for l in labs))
            for c in CAUSES:
                print(f"{c:6s}" + "".join(f"{cnt[(c, l)]:9d}" for l in labs))
    if a.per_episode:
        cols = ["arm", "episode", "run", "cause", "success", "result", "final_state", "steps", "stopped", "acc_src",
                "acc_true", "n_true", "n_ended", "n_pending", "gate", "spl"]
        with open(a.per_episode, "w") as f:
            w = csv.writer(f, delimiter="\t")
            w.writerow(cols)
            for name, d in R.items():
                for ep, r in sorted(d.items()):
                    w.writerow([name, ep] + [r[k] for k in cols[2:]])


if __name__ == "__main__":
    main()
