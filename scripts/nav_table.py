"""Outcome table from the eye-verified ledger (outputs/apexnav_online/labels/ledger.tsv).
  python scripts/nav_table.py RUN [RUN ...] [--episodes FILE | --common] [--md]
RUN is a videos/ run name, or an arm: NAME=run1+run2 (or just run1+run2); first run holding the episode wins.
Arm aliases can also be defined in labels/arms.tsv (name<TAB>run1+run2+...).  --episodes FILE: whitespace/line separated ids.
Only verified=eye rows count in the main columns; verified=auto rows are only reported in the 'unverif.' column; 'flagged' = eye rows with a doubt flag (borderline_1m, annotation_doubt, label_unsure).
Two runs: McNemar on S (exact) and the discordant episodes."""
import argparse, csv, os, sys
from math import comb
LD = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "outputs", "apexnav_online", "labels")
COLS = ["n", "S", "FS-perc", "FR", "perc", "FS-nav", "NAV", "nav", "EXPL", "F?", "unverif.", "flagged", "S %"]

def load():
    led = {}
    for r in csv.DictReader(open(os.path.join(LD, "ledger.tsv")), delimiter="\t"):
        led[(r["run"], int(r["episode"]))] = r
    return led

def arms_file():
    p = os.path.join(LD, "arms.tsv"); d = {}
    if os.path.exists(p):
        for r in csv.reader(open(p), delimiter="\t"):
            if len(r) >= 2 and not r[0].startswith("#"): d[r[0]] = r[1]
    return d

def parse_arm(a, known):
    if "=" in a: name, runs = a.split("=", 1)
    elif a in known: name, runs = a, known[a]
    else: name, runs = a, a
    return name, runs.split("+")

def arm_rows(led, runs):
    out = {}
    for run in runs:
        for (r, ep), row in led.items():
            if r == run and ep not in out: out[ep] = row
    return out

def mcnemar(b, c):
    n = b + c
    return 1.0 if n == 0 else min(1.0, 2 * sum(comb(n, i) for i in range(min(b, c) + 1)) / 2 ** n)

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("runs", nargs="+"); ap.add_argument("--episodes"); ap.add_argument("--common", action="store_true")
    ap.add_argument("--md", action="store_true"); a = ap.parse_args()
    led = load(); known = arms_file()
    arms = [parse_arm(x, known) for x in a.runs]
    rows = {name: arm_rows(led, runs) for name, runs in arms}
    eps = None
    if a.episodes: eps = {int(x) for x in open(a.episodes).read().split()}
    if a.common: eps = set.intersection(*(set(r) for r in rows.values())) if eps is None else eps & set.intersection(*(set(r) for r in rows.values()))
    tab = []
    for name, _ in arms:
        R = {e: r for e, r in rows[name].items() if eps is None or e in eps}
        eye = [r for r in R.values() if r["verified"] == "eye"]
        c = {k: sum(r["label"] == k for r in eye) for k in ("S", "FS-perc", "FR", "FS-nav", "NAV", "EXPL", "F?")}
        n = len(eye)
        tab.append([name, n, c["S"], c["FS-perc"], c["FR"], c["FS-perc"] + c["FR"], c["FS-nav"], c["NAV"], c["FS-nav"] + c["NAV"], c["EXPL"], c["F?"],
                    len(R) - n, sum(bool(r.get("flag")) for r in eye), f"{100 * c['S'] / n:.1f}" if n else "-"])
    hdr = ["arm"] + COLS
    if a.md:
        print("| " + " | ".join(hdr) + " |\n|" + "---|" * len(hdr))
        for t in tab: print("| " + " | ".join(map(str, t)) + " |")
    else:
        w = [max(len(str(x)) for x in col) for col in zip(hdr, *tab)]
        for t in [hdr] + tab: print("  ".join(str(x).rjust(wi) if i else str(x).ljust(wi) for i, (x, wi) in enumerate(zip(t, w))))
    if len(arms) == 2:
        (n1, _), (n2, _) = arms; A, B = rows[n1], rows[n2]
        both = [e for e in sorted(set(A) & set(B)) if (eps is None or e in eps) and A[e]["verified"] == "eye" and B[e]["verified"] == "eye"]
        b = [e for e in both if A[e]["label"] == "S" and B[e]["label"] != "S"]; c = [e for e in both if A[e]["label"] != "S" and B[e]["label"] == "S"]
        print(f"\nMcNemar on S over {len(both)} common eye-verified eps: {n1}-only {len(b)}, {n2}-only {len(c)}, exact p={mcnemar(len(b), len(c)):.3f}")
        for e in b + c: print(f"  ep {e}: {n1}={A[e]['label']}  {n2}={B[e]['label']}   | {n1}: {A[e]['note'][:70]} | {n2}: {B[e]['note'][:70]}")

main()
