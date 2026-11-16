#!/usr/bin/env python3
"""Reproducible, resumable three-way Qwen verification screening study.

Stages: prepare (freeze cohort/gate), reconstruct (grid assets for flagged rows),
infer (Qwen diagnostic arms), analyze (episode-clustered metrics/report).
Heavy model/worker imports are deliberately lazy.
"""
import argparse
import csv
import hashlib
import json
import math
import re
import random
import statistics
import time
import sys
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
DATA = ROOT / "outputs/apexnav_offline/sufficiency_2026-10-28"
DEFAULT_OUT = ROOT / "outputs/apexnav_offline/qwen_verification_pilot"
SEED = 20261103


def read_csv(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = list(dict.fromkeys(k for row in rows for k in row))
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, keys, extrasaction="ignore")
        w.writeheader(); w.writerows(rows)


def digest(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""): h.update(chunk)
    return h.hexdigest()


def validate_frozen(out):
    manifest=json.loads((out/"manifest.json").read_text())
    if manifest.get("source_sha256") != digest(DATA/"requests.csv"):
        raise RuntimeError("source requests changed after cohort freeze")
    if manifest.get("gate_sha256") != digest(out/"frozen_gate.json"):
        raise RuntimeError("frozen gate changed after cohort freeze")
    if manifest.get("model")!="Qwen/Qwen3-VL-8B-Instruct" or manifest.get("prompt_version")!="three-way-v1":
        raise RuntimeError("model/prompt fingerprint mismatch")
    from perception.qwen import prompt
    prompt_hash=hashlib.sha256(prompt("<target>",True).encode()).hexdigest()
    if manifest.get("prompt_template_sha256") != prompt_hash:
        raise RuntimeError("frozen prompt template changed")
    return manifest


def scalar_metrics_for_labels(selected, preds, labels):
    def label(r): return labels.get(str(r.get("source_index", "")), "unknown")
    clear=[r for r in selected if label(r) in ("goal","not_goal")]
    decided=[r for r in clear if preds.get(r["name"]) in ("correct","wrong")]
    goal=[r for r in clear if label(r)=="goal"]
    not_goal=[r for r in clear if label(r)=="not_goal"]
    ct=[r for r in selected if label(r)=="cant_tell"]
    rate=lambda n,d: n/d if d else None
    correct=sum(preds.get(r["name"])==("correct" if label(r)=="goal" else "wrong") for r in decided)
    return {"n":len(selected),"clear_n":len(clear),"decided_clear_n":len(decided),
        "decided_coverage_clear":rate(len(decided),len(clear)),
        "risk_among_decided_clear":rate(len(decided)-correct,len(decided)),
        "known_false_accepts":sum(preds.get(r["name"])=="correct" for r in not_goal),
        "known_false_rejects":sum(preds.get(r["name"])=="wrong" for r in goal),
        "known_false_accept_rate":rate(sum(preds.get(r["name"])=="correct" for r in not_goal),len(not_goal)),
        "known_false_reject_rate":rate(sum(preds.get(r["name"])=="wrong" for r in goal),len(goal)),
        "true_goal_accepts":sum(preds.get(r["name"])=="correct" for r in goal),
        "true_not_goal_rejections":sum(preds.get(r["name"])=="wrong" for r in not_goal),
        "cant_tell_n":len(ct),"cant_tell_deferral_recall":rate(sum(preds.get(r["name"])=="insufficient" for r in ct),len(ct)),
        "cant_tell_unsupported_accepts":sum(preds.get(r["name"])=="correct" for r in ct),
        "cant_tell_unsupported_rejects":sum(preds.get(r["name"])=="wrong" for r in ct),
        "cant_tell_unsupported_accept_rate":rate(sum(preds.get(r["name"])=="correct" for r in ct),len(ct)),
        "cant_tell_unsupported_reject_rate":rate(sum(preds.get(r["name"])=="wrong" for r in ct),len(ct)),
        "accept_precision_clear":rate(sum(label(r)=="goal" for r in selected if preds.get(r["name"])=="correct"),sum(preds.get(r["name"])=="correct" and label(r) in ("goal","not_goal") for r in selected)),
        "reject_precision_clear":rate(sum(label(r)=="not_goal" for r in selected if preds.get(r["name"])=="wrong"),sum(preds.get(r["name"])=="wrong" and label(r) in ("goal","not_goal") for r in selected)),
        "review_rate":rate(sum(r["gate_review"]=="True" for r in selected),len(selected)),
        "raw_errors_repaired":sum(label(r) in ("goal","not_goal") and
            ("correct" if r["raw_accept"]=="True" else "wrong") != ("correct" if label(r)=="goal" else "wrong") and
            preds.get(r["name"]) == ("correct" if label(r)=="goal" else "wrong") for r in selected),
        "raw_correct_introduced_errors":sum(label(r) in ("goal","not_goal") and
            ("correct" if r["raw_accept"]=="True" else "wrong") == ("correct" if label(r)=="goal" else "wrong") and
            preds.get(r["name"]) in ("correct","wrong") and preds.get(r["name"]) != ("correct" if label(r)=="goal" else "wrong") for r in selected),
        "defer_rate":rate(sum(preds.get(r["name"])=="insufficient" for r in selected),len(selected))}


def episode_id(row):
    # Underlying IDs are strategy-prefixed online, but the terminal episode token
    # is the independent unit; offline IDs are kept intact.
    found = re.search(r"epi(\d+)", row["episode"] or row["name"])
    return f"epi{found.group(1)}" if found else row["episode"]


def episode_disjoint_cohort(online, calibration_episodes):
    blocked=set(calibration_episodes)
    return [r for r in online if episode_id(r) not in blocked]


def merge_label_sources(label_sources):
    return {str(i):v.get("review",v.get("b",v.get("a","unknown")))
            for i,v in label_sources.items()}


def required_arms(rows):
    arms={(r["name"],"input") for r in rows}
    arms|={(r["name"],arm) for r in rows if r["gate_review"]=="True"
           for arm in ("matching_grid","mismatch_grid")}
    return arms


def require_complete_inference(rows, decisions):
    missing=required_arms(rows)-set(decisions)
    if missing:raise RuntimeError(f"inference incomplete: {len(missing)} required arms missing; first={sorted(missing)[:8]}")


def write_false_accept_review(rows, labels, methods, decisions, out):
    from PIL import Image, ImageDraw
    from perception.qwen import outlined
    examples=[r for r in rows if r["gate_review"]=="True"
        and labels.get(str(r["source_index"]))=="not_goal"
        and r["raw_accept"]=="False"
        and methods["gate_qwen_grid"].get(r["name"])=="correct"]
    if not examples:
        return 0
    panel_w,panel_h=640,336
    margin,title_h=16,40
    row_h=title_h+panel_h+margin
    sheet=Image.new("RGB",(margin*3+panel_w+896,margin+(row_h+margin)*len(examples)),"white")
    draw=ImageDraw.Draw(sheet)
    for i,r in enumerate(examples):
        y=margin+i*(row_h+margin)
        d=decisions[(r["name"],"matching_grid")]
        caption=f"{r['name']} | target={r['target']} | provisional eye label=not_goal | raw SAM3=wrong | Qwen grid={d['status']}"
        draw.text((margin,y),caption,fill="black")
        folder=ROOT/r["folder"]
        cam=outlined(Image.open(folder/"input.jpg"),Image.open(folder/"mask.png"))
        cam.thumbnail((panel_w,panel_h))
        grid=Image.open(out/"grids"/f"{r['name']}.jpg").convert("RGB")
        grid.thumbnail((896,panel_h))
        sheet.paste(cam,(margin,y+title_h))
        sheet.paste(grid,(margin*2+panel_w,y+title_h))
    sheet.save(out/"review_false_accepts.jpg",quality=90)
    return len(examples)


def prepare(out=DEFAULT_OUT):
    from perception.qwen import UncertaintyGate
    all_rows = read_csv(DATA / "requests.csv")
    offline = [r for r in all_rows if r["set"] == "offline"]
    online = [r for r in all_rows if r["set"] == "online"]
    gate = UncertaintyGate.fit([{"margin": float(r["margin"]), "area": float(r["area"]),
        "view_flip": r["view_flip"] or None} for r in offline])
    blocked = {episode_id(r) for r in offline}
    eligible = episode_disjoint_cohort(online,blocked)
    chosen = {episode_id(r) for r in eligible}
    source_index = {r["name"]: i for i, r in enumerate(all_rows)}
    rows = [dict(r, episode_id=episode_id(r), source_index=source_index[r["name"]]) for r in eligible]
    for r in rows:
        assessment = gate.assess(float(r["margin"]), float(r["area"]),
                                 None if not r["view_flip"] else float(r["view_flip"]))
        r["gate_review"], r["gate_score"], r["gate_reason"] = assessment["review"], assessment["score"], assessment["reason"]
    out.mkdir(parents=True, exist_ok=True)
    if (out / "manifest.csv").exists() or (out / "decisions.jsonl").exists():
        raise FileExistsError(f"refusing to replace frozen study artifacts in {out}")
    gate_json = {"references": gate.references, "thresholds": gate.thresholds,
                 "calibration_n": len(offline), "review_fraction": 1/3,
                 "feature_order": ["-abs(margin)", "-area", "view_flip"], "seed": SEED}
    (out / "frozen_gate.json").write_text(json.dumps(gate_json, indent=2) + "\n")
    write_csv(out / "manifest.csv", rows)
    from perception.qwen import prompt
    (out / "manifest.json").write_text(json.dumps({"seed": SEED, "selected_episode_ids": sorted(chosen),
        "selected_rows": len(rows), "sampling": "all episode-disjoint eligible online rows",
        "excluded_episode_ids": sorted(blocked), "source": str(DATA / "requests.csv"),
        "source_sha256": digest(DATA / "requests.csv"), "gate_sha256": digest(out / "frozen_gate.json"),
        "model":"Qwen/Qwen3-VL-8B-Instruct","prompt_version":"three-way-v1",
        "prompt_template_sha256":hashlib.sha256(prompt("<target>",True).encode()).hexdigest()}, indent=2)+"\n")
    return rows


def reconstruct(out=DEFAULT_OUT,gpu=0):
    """Render complete flagged grids through the existing SAM3D worker.

    Worker-specific scene paths are kept in the input table; unsupported/missing
    source rows remain explicit errors and are never replaced with a view subset.
    """
    import base64
    from PIL import Image
    from perception.client import _Worker, default_launcher
    from perception.images import encode
    rows = [r for r in read_csv(out / "manifest.csv") if r["gate_review"] == "True"]
    worker = _Worker("sam3d", ["--role", "sam3d", "--sam3d", "/data/checkpoints/sam3d/hf", "--gpu", str(gpu)],
                     default_launcher(), out / "sam3d_worker.log")
    error_path=out/"reconstruction_errors.jsonl"
    previous_errors={}
    if error_path.exists():
        for line in error_path.read_text().splitlines():
            if line.strip():
                x=json.loads(line); previous_errors[x["name"]]=previous_errors.get(x["name"],0)+1
    try:
        worker.wait_ready()
        for i, r in enumerate(rows, 1):
            dest = out / "grids" / f"{r['name']}.jpg"
            timing_file = out / "reconstruction.jsonl"
            if dest.exists(): continue
            folder = ROOT / r["folder"]
            image, mask = Image.open(folder / "input.jpg"), Image.open(folder / "mask.png")
            decision = json.loads((folder / "decision.json").read_text())
            input_elevation = decision.get("evidence", {}).get("input_elevation", 0.0)
            started=time.perf_counter()
            try:
                result = worker.call("reconstruct", image=encode(image), mask=encode(mask, "L"),
                    input_elevation=input_elevation, azimuths=list(range(0, 360, 45)), elevations=[0, 30, 60],
                    distance=2.0, resolution=320, seed=42, ensemble=1, max_uncertainty=None)
            except Exception as exc:
                with open(error_path,"a") as f:
                    f.write(json.dumps({"name":r["name"],"attempt":previous_errors.get(r["name"],0)+1,
                        "error":f"{type(exc).__name__}: {exc}"})+"\n")
                previous_errors[r["name"]]=previous_errors.get(r["name"],0)+1
                print(f"reconstruction failed {i}/{len(rows)} {r['name']}: {exc}",flush=True)
                continue
            views = {(int(v["azimuth"]), int(v["elevation"])):
                     Image.open(__import__("io").BytesIO(base64.b64decode(v["image"])))
                     for v in result["views"]}
            if len(views) != 24: raise RuntimeError(f"{r['name']}: expected 24 renders, got {len(views)}")
            from perception.qwen import render_grid
            dest.parent.mkdir(parents=True, exist_ok=True)
            render_grid(views).save(dest, quality=90)
            with open(timing_file, "a") as f:
                f.write(json.dumps({"name":r["name"],"gpu":gpu,"timing":result.get("timing"),"n_views":len(views)})+"\n")
            if previous_errors.get(r["name"],0):
                with open(out/"reconstruction_retry.jsonl","a") as f:
                    f.write(json.dumps({"name":r["name"],"attempts":previous_errors[r["name"]]+1,
                      "retry_status":"succeeded","same_settings":True,
                      "retry_cost_s":round(time.perf_counter()-started,3),
                      "settings":{"azimuths":list(range(0,360,45)),"elevations":[0,30,60],"distance":2.0,
                                  "resolution":320,"seed":42,"ensemble":1}})+"\n")
            print(f"reconstructed {i}/{len(rows)} {r['name']}", flush=True)
    finally:
        worker.close()


def infer(out=DEFAULT_OUT, smoke=False,gpu=0):
    os.environ["CUDA_VISIBLE_DEVICES"]=str(gpu)
    from PIL import Image
    from perception.qwen import QwenJudge, outlined, prompt
    frozen=validate_frozen(out)
    rows = read_csv(out / "manifest.csv")
    recon_errors={}
    ep=out/"reconstruction_errors.jsonl"
    if ep.exists():
        for line in ep.read_text().splitlines():
            if line.strip():
                item=json.loads(line); recon_errors[item["name"]]=item
    missing=[]
    for r in rows:
        folder=ROOT/r["folder"]
        for p in (folder/"input.jpg",folder/"mask.png"):
            if not p.exists(): missing.append(str(p))
        if r["gate_review"]=="True" and not (out/"grids"/f"{r['name']}.jpg").exists() and r["name"] not in recon_errors:
            missing.append(str(out/"grids"/f"{r['name']}.jpg"))
    if missing:
        raise FileNotFoundError("missing required inference assets:\n"+"\n".join(missing[:30]))
    judge = QwenJudge(frozen["model"])
    done_path = out / "decisions.jsonl"
    done = {}
    if done_path.exists():
        for line in done_path.read_text().splitlines():
            if line.strip():
                x=json.loads(line); done[(x["name"], x["arm"])]=x
    for x in done.values():
        if x.get("model") != frozen["model"] or x.get("prompt_version") != frozen["prompt_version"]:
            raise RuntimeError("existing inference records use a different model or prompt")
        target=next(r["target"] for r in rows if r["name"]==x["name"])
        with_grid=x["arm"]!="input"
        expected_hash=hashlib.sha256(prompt(target,with_grid).encode()).hexdigest()
        if x.get("prompt_sha256") != expected_hash:
            raise RuntimeError("existing inference record prompt hash does not match frozen study prompt")
    with open(done_path, "a") as sink:
        count=0
        for r in rows:
            name=r["name"]
            # Existing sufficiency archive has the original camera frame/mask.
            folder=ROOT / r["folder"]
            image_path=folder / "input.jpg"
            mask_path=folder / "mask.png"
            image=outlined(Image.open(image_path), Image.open(mask_path))
            grid_path=out / "grids" / f"{name}.jpg"
            arms=[("input",None)]
            if r["gate_review"]=="True":
                if grid_path.exists(): arms += [("matching_grid",Image.open(grid_path))]
                # deterministic cyclic mismatch, guaranteeing a different source episode
                flagged=[x for x in rows if x["gate_review"]=="True" and (out/"grids"/f"{x['name']}.jpg").exists()]
                pos=next((i for i,x in enumerate(flagged) if x["name"]==name),None)
                cycle=flagged[pos+1:]+flagged[:pos] if pos is not None else flagged
                other=next((x for x in cycle if x["episode_id"]!=r["episode_id"]),None)
                if other is not None: arms += [("mismatch_grid",Image.open(out/"grids"/f"{other['name']}.jpg"))]
            for arm,grid in arms:
                if (name,arm) in done and not done[(name,arm)].get("error"): continue
                try:
                    result=judge.judge(image,r["target"],grid)
                except Exception as exc:
                    result={"status":None,"probabilities":{"correct":None,"wrong":None,"insufficient":None},
                            "invalid_answer":False,"top_token":None,"latency_s":None,"n_tokens":None,
                            "error":f"{type(exc).__name__}: {exc}"}
                item={"name":name,"episode_id":r["episode_id"],"arm":arm,
                      "model":"Qwen/Qwen3-VL-8B-Instruct","prompt_version":"three-way-v1",
                      "prompt_sha256":hashlib.sha256(prompt(r["target"],grid is not None).encode()).hexdigest(),
                      "transformers_version":__import__("transformers").__version__,"gpu":gpu,**result}
                sink.write(json.dumps(item)+"\n"); sink.flush(); done[(name,arm)]=item
                count+=1
                if smoke and count>=1: return
            if r["gate_review"]=="True":
                for arm in ("matching_grid","mismatch_grid"):
                    if (name,arm) in done:continue
                    err=recon_errors.get(name,{}).get("error","no different-episode reconstruction available")
                    item={"name":name,"episode_id":r["episode_id"],"arm":arm,"model":frozen["model"],
                         "prompt_version":frozen["prompt_version"],"prompt_sha256":hashlib.sha256(prompt(r["target"],True).encode()).hexdigest(),
                         "transformers_version":__import__("transformers").__version__,"gpu":gpu,"status":None,
                         "probabilities":{"correct":None,"wrong":None,"insufficient":None},"invalid_answer":False,
                         "top_token":None,"latency_s":None,"n_tokens":None,"error":"reconstruction unavailable: "+err}
                    sink.write(json.dumps(item)+"\n");sink.flush();done[(name,arm)]=item
            print(f"inference complete {name} ({len(done)} arms recorded)", flush=True)


def analyze(out=DEFAULT_OUT, draws=2000):
    validate_frozen(out)
    rows=read_csv(out / "manifest.csv")
    fmt=lambda x: "undefined" if x is None else f"{x:.3f}"
    decisions={}
    p=out / "decisions.jsonl"
    if p.exists():
        for line in p.read_text().splitlines():
            if line.strip():
                x=json.loads(line); decisions[(x["name"],x["arm"])]=x
    # Agent labels are joined by the global source row index; review overrides
    # take precedence over both independent initial label files.
    labels={}; label_sources={}
    for fn in ("eye_labels_a.tsv","eye_labels_b.tsv","eye_labels_review.tsv"):
        path=DATA/fn
        if not path.exists(): continue
        with open(path) as f:
            for x in csv.DictReader(f,delimiter="\t"):
                label_sources.setdefault(x["index"],{})[fn.removeprefix("eye_labels_").removesuffix(".tsv")]=x["label"]
    labels.update(merge_label_sources(label_sources))
    require_complete_inference(rows,decisions)
    methods={"sam3_raw":{},"gate_defer":{},"gate_qwen_input":{},"gate_qwen_grid":{},"qwen_input_all":{}}
    for r in rows:
        raw=r["raw_accept"]=="True"; base="correct" if raw else "wrong"
        methods["sam3_raw"][r["name"]]=base
        methods["gate_defer"][r["name"]]="insufficient" if r["gate_review"]=="True" else base
        d_input=decisions.get((r["name"],"input"))
        d_grid=decisions.get((r["name"],"matching_grid"))
        methods["qwen_input_all"][r["name"]]=d_input["status"] if d_input else None
        methods["gate_qwen_input"][r["name"]]=(d_input["status"] if r["gate_review"]=="True" else base)
        methods["gate_qwen_grid"][r["name"]]=(d_grid["status"] if r["gate_review"]=="True" else base)
    output={"n_requests":len(rows),"n_episodes":len({r['episode_id'] for r in rows}),"bootstrap_draws":draws,"seed":SEED,"label_caveat":"agent-produced labels with agent review; provisional, not independently validated human ground truth","methods":{}}
    for method,preds in methods.items():
        counts={y:{p:0 for p in ("correct","wrong","insufficient","missing")} for y in ("goal","not_goal","cant_tell","unknown")}
        for r in rows:
            y=labels.get(str(r.get("source_index","")),"unknown"); pred=preds[r["name"]] or "missing"
            if y not in counts:y="unknown"
            counts[y][pred]+=1
        metrics=scalar_metrics_for_labels(rows,preds,labels)
        episode_groups={}
        for r in rows: episode_groups.setdefault(r["episode_id"],[]).append(r)
        rng=random.Random(SEED); samples={k:[] for k in ("decided_coverage_clear","risk_among_decided_clear","known_false_accept_rate","known_false_reject_rate","cant_tell_deferral_recall","cant_tell_unsupported_accept_rate","cant_tell_unsupported_reject_rate","accept_precision_clear","reject_precision_clear","review_rate","defer_rate")}
        eps=list(episode_groups)
        for _ in range(draws):
            draw=[r for epi in (rng.choice(eps) for _ in eps) for r in episode_groups[epi]]
            m=scalar_metrics_for_labels(draw,preds,labels)
            for k in samples:
                if m[k] is not None:samples[k].append(m[k])
        metrics["bootstrap_95_ci"]={k:([sorted(v)[int(.025*(len(v)-1))],sorted(v)[int(.975*(len(v)-1))]] if v else None) for k,v in samples.items()}
        output["methods"][method]={"counts":counts,"metrics":metrics}
    # Paired episode bootstrap for full-cohort policy arms and flagged-only
    # diagnostic controls. Each draw resamples the same episode IDs in both arms.
    selected_gate=[r for r in rows if r["gate_review"]=="True"]
    def paired_comparison(name, selected, pred_a, pred_b):
        groups={}
        for r in selected:groups.setdefault(r["episode_id"],[]).append(r)
        rng=random.Random(SEED); result={}
        for field in ("risk_among_decided_clear","decided_coverage_clear","accept_precision_clear","reject_precision_clear"):
            vals=[]; epnames=list(groups)
            for _ in range(draws):
                sample=[r for epi in (rng.choice(epnames) for _ in epnames) for r in groups[epi]]
                ma=scalar_metrics_for_labels(sample,pred_a,labels)[field]
                mb=scalar_metrics_for_labels(sample,pred_b,labels)[field]
                if ma is not None and mb is not None:vals.append(ma-mb)
            a=scalar_metrics_for_labels(selected,pred_a,labels)[field]
            b=scalar_metrics_for_labels(selected,pred_b,labels)[field]
            s=sorted(vals)
            result[field]={"a_minus_b":None if a is None or b is None else a-b,
                "ci_95":([s[int(.025*(len(s)-1))],s[int(.975*(len(s)-1))]] if s else None),"n_draws":len(vals)}
        paired[name]=result
    paired={}
    paired_comparison("gate_qwen_grid_minus_input_policy_all_requests",rows,methods["gate_qwen_grid"],methods["gate_qwen_input"])
    policy_grid={r["name"]:decisions[(r["name"],"matching_grid")]["status"] for r in selected_gate}
    policy_input={r["name"]:decisions[(r["name"],"input")]["status"] for r in selected_gate}
    mismatch={r["name"]:decisions[(r["name"],"mismatch_grid")]["status"] for r in selected_gate}
    paired_comparison("matching_grid_minus_input_flagged",selected_gate,policy_grid,policy_input)
    paired_comparison("matching_grid_minus_mismatch_grid_flagged",selected_gate,policy_grid,mismatch)
    output["paired_deltas"]=paired
    output["mismatch_control_flagged"]={"matching_grid":scalar_metrics_for_labels(selected_gate,policy_grid,labels),
        "mismatch_grid":scalar_metrics_for_labels(selected_gate,mismatch,labels),
        "status_disagreement_n":sum(policy_grid[r["name"]]!=mismatch[r["name"]] for r in selected_gate),
        "n_flagged":len(selected_gate)}
    output["flagged_arm_status_counts"]={arm:{status:sum(decisions[(r["name"],arm)]["status"]==status for r in selected_gate)
        for status in ("correct","wrong","insufficient",None)} for arm in ("matching_grid","mismatch_grid")}
    output["at_stop"]={"n_requests":sum(r["at_stop"]=="True" for r in rows),"available":any(r["at_stop"]=="True" for r in rows)}
    output["navigation_diagnostics"]={"episodes_by_policy":{}}
    for method,preds in methods.items():
        output["navigation_diagnostics"]["episodes_by_policy"][method]={
          "with_known_false_accept":len({r["episode_id"] for r in rows if labels.get(str(r["source_index"]))=="not_goal" and preds[r["name"]]=="correct"}),
          "with_only_unresolved_checks":len({epi for epi in {r["episode_id"] for r in rows} if all(preds[r["name"]]=="insufficient" for r in rows if r["episode_id"]==epi)})}
    full_rows=read_csv(DATA/"requests.csv")
    grouped={}
    pat=re.compile(r"^(?P<strategy>[^_]+)_(?P<episode>epi\d+)_.*?_c(?P<cluster>\d+)_s(?P<step>\d+)$")
    for fr in full_rows:
        if fr["set"]!="online":continue
        m=pat.match(fr["name"])
        if not m:continue
        key=(m["strategy"],m["episode"],fr["target"],m["cluster"])
        grouped.setdefault(key,[]).append((int(m["step"]),fr))
    future_counts={"unresolved":0,"with_later_distinct_real_image_and_higher_evidence_step":0,
                   "resolved_known_later_eye_label":0,"later_policy_agrees_with_later_eye_label":0,
                   "later_policy_decision_insufficient":0,"censored_no_qualifying_later_observation":0}
    all_index={r["name"]:i for i,r in enumerate(full_rows)}
    def evidence_step(r):
        dpath=ROOT/r["folder"]/"decision.json"
        if not dpath.exists():return None
        try:return json.loads(dpath.read_text()).get("evidence",{}).get("step")
        except (ValueError,OSError):return None
    for r in rows:
        if methods["gate_qwen_grid"][r["name"]]!="insufficient":continue
        m=pat.match(r["name"])
        if not m:continue
        future_counts["unresolved"]+=1
        key=(m["strategy"],m["episode"],r["target"],m["cluster"])
        curr_path=ROOT/r["folder"]/"input.jpg"
        curr_hash=digest(curr_path) if curr_path.exists() else None
        current_evidence=evidence_step(r)
        if current_evidence is None:
            future_counts["censored_no_qualifying_later_observation"]+=1;continue
        candidates=[]
        for step,fr in grouped.get(key,[]):
            later_evidence=evidence_step(fr)
            if later_evidence is None or later_evidence<=current_evidence:continue
            fpath=ROOT/fr["folder"]/"input.jpg"
            if not fpath.exists():continue
            if curr_hash and digest(fpath)==curr_hash:continue
            candidates.append((step,fr))
        if not candidates:
            future_counts["censored_no_qualifying_later_observation"]+=1;continue
        future_counts["with_later_distinct_real_image_and_higher_evidence_step"]+=1
        step,fr=min(candidates,key=lambda pair:pair[0])
        lab=labels.get(str(all_index.get(fr["name"],"")),"unknown")
        if lab in ("goal","not_goal"):
            future_counts["resolved_known_later_eye_label"]+=1
            pred=methods["gate_qwen_grid"].get(fr["name"])
            if pred is None:
                pred="correct" if fr["raw_accept"]=="True" else "wrong"
            if pred=="insufficient":future_counts["later_policy_decision_insufficient"]+=1
            elif pred==("correct" if lab=="goal" else "wrong"):
                future_counts["later_policy_agrees_with_later_eye_label"]+=1
    output["navigation_diagnostics"]["future_observation_counts"]=future_counts
    output["navigation_diagnostics"]["future_view_resolution"]="computed from saved decision evidence steps and distinct source images"
    output["navigation_diagnostics"]["interpretation"]="Observed later same-strategy, same-episode, same-cluster requests with a different saved image are diagnostic only; no counterfactual recovery or success is inferred."
    latencies={arm:[d["latency_s"] for d in decisions.values() if d["arm"]==arm and d.get("latency_s") is not None]
               for arm in ("input","matching_grid","mismatch_grid")}
    output["qwen_latency_s"]={arm:{"n":len(v),"median":statistics.median(v) if v else None,
        "mean":statistics.mean(v) if v else None,"p95":sorted(v)[int(.95*(len(v)-1))] if v else None}
        for arm,v in latencies.items()}
    output["qwen_protocol"]={arm:{"n":sum(d["arm"]==arm for d in decisions.values()),
        "invalid_answer_count":sum(d["arm"]==arm and d.get("invalid_answer") is True for d in decisions.values()),
        "semantic_insufficient_count":sum(d["arm"]==arm and d.get("status")=="insufficient" and not d.get("invalid_answer",False) for d in decisions.values()),
        "logged_error_count":sum(d["arm"]==arm and d.get("error") is not None for d in decisions.values())}
        for arm in ("input","matching_grid","mismatch_grid")}
    retry_path=out/"reconstruction_retry.jsonl"
    retries=[json.loads(x) for x in retry_path.read_text().splitlines() if x.strip()] if retry_path.exists() else []
    output["reconstruction_retries"]={"recoveries":retries,"final_successful_grids":len(list((out/"grids").glob("*.jpg"))),
        "logged_unrecovered_errors":len((out/"reconstruction_errors.jsonl").read_text().splitlines()) if (out/"reconstruction_errors.jsonl").exists() else 0}
    retry_notes=[]
    for retry in retries:
        retry_cost=retry.get("retry_cost_s","not measured separately")
        cost_text=f"retry time {retry_cost} s" if isinstance(retry_cost,(int,float)) else f"retry time {retry_cost}"
        retry_notes.append(f"First pass for `{retry['name']}` failed: {retry.get('initial_error', 'reconstruction error')}. Retry status: {retry.get('retry_status','unknown')}; same settings: {retry.get('same_settings','unknown')}; {cost_text}.")
    if output["reconstruction_retries"]["logged_unrecovered_errors"]:
        retry_notes.append(f"{output['reconstruction_retries']['logged_unrecovered_errors']} unrecovered attempts are logged in `reconstruction_errors.jsonl`.")
    if not retry_notes: retry_notes.append("No reconstruction retries or unrecovered errors were logged.")
    retry_report=" ".join(retry_notes)+" The original worker traceback is preserved in `sam3d_worker.log`."
    rt=[json.loads(x).get("timing",{}) for x in (out/"reconstruction.jsonl").read_text().splitlines() if x.strip()]
    total=[x.get("model_seconds",0)+x.get("render_seconds",0) for x in rt]
    output["reconstruction_timing_s"]={"n":len(total),"median_model_plus_render":statistics.median(total) if total else None,
        "mean_model_plus_render":statistics.mean(total) if total else None,"sum_model_plus_render":sum(total),
        "mean_model":statistics.mean([x.get("model_seconds",0) for x in rt]) if rt else None,
        "mean_render":statistics.mean([x.get("render_seconds",0) for x in rt]) if rt else None}
    per_request=[]
    for r in rows:
        item={**r,"eye_label":labels.get(str(r["source_index"]),"unknown"),"label_sources":json.dumps(label_sources.get(str(r["source_index"]),{})),
              "raw_action":methods["sam3_raw"][r["name"]],"gate_action":methods["gate_defer"][r["name"]],
              "gate_qwen_input":methods["gate_qwen_input"][r["name"]],"gate_qwen_grid":methods["gate_qwen_grid"][r["name"]],
              "qwen_input_all":methods["qwen_input_all"][r["name"]],
              "qwen_input_result":json.dumps(decisions[(r["name"],"input")]),
              "qwen_grid_result":json.dumps(decisions[(r["name"],"matching_grid")]) if r["gate_review"]=="True" else "",
              "qwen_mismatch_result":json.dumps(decisions[(r["name"],"mismatch_grid")]) if r["gate_review"]=="True" else "",
              "qwen_grid_latency_s":decisions[(r["name"],"matching_grid")]["latency_s"] if r["gate_review"]=="True" else ""}
        per_request.append(item)
    write_csv(out/"per_request.csv",per_request)
    output["false_accept_review_examples"]=write_false_accept_review(rows,labels,methods,decisions,out)
    (out/"summary.json").write_text(json.dumps(output,indent=2)+"\n")
    report=["# Three-way Qwen verification screening","",
      f"Selected {output['n_requests']} requests across {output['n_episodes']} disjoint online episodes. The gate flags {sum(r['gate_review']=='True' for r in rows)} requests.","",
      "## Decision metrics","","| Policy | True goals accepted | True not-goals rejected | False accepts n/rate | False rejects n/rate | Clear decided coverage | Clear risk among decided | Cant-tell deferral recall | Unsupported cant-tell accepts / rejects | Review rate | Total defer rate |","|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for method,entry in output["methods"].items():
        m=entry["metrics"]
        report.append(f"| {method} | {m['true_goal_accepts']} | {m['true_not_goal_rejections']} | {m['known_false_accepts']} / {fmt(m['known_false_accept_rate'])} | {m['known_false_rejects']} / {fmt(m['known_false_reject_rate'])} | {fmt(m['decided_coverage_clear'])} | {fmt(m['risk_among_decided_clear'])} | {fmt(m['cant_tell_deferral_recall'])} | {m['cant_tell_unsupported_accepts']} / {m['cant_tell_unsupported_rejects']} | {fmt(m['review_rate'])} | {fmt(m['defer_rate'])} |")
    mm=output["mismatch_control_flagged"]
    risk_delta=output["paired_deltas"]["gate_qwen_grid_minus_input_policy_all_requests"]["risk_among_decided_clear"]
    risk_ci=risk_delta["ci_95"]
    risk_ci_text=("includes zero" if risk_ci and risk_ci[0]<=0<=risk_ci[1] else "does not include zero")
    report += ["","Episode bootstrap 95% intervals for primary rates, known error rates, and clear accept/reject precision are in `summary.json`. Paired episode deltas compare the full deployed policies across all requests, plus flagged-only input and mismatch controls. Counts accompany rates; undefined denominators are null.",
      f"On the {mm['n_flagged']} flagged requests, matching-grid vs mismatch-grid status differed on {mm['status_disagreement_n']} requests. Their full three-way counts and clear-case metrics are in `summary.json`.","",
      "## Controls and limitations","",
      f"Qwen median latency: input {fmt(output['qwen_latency_s']['input']['median'])} s, matching grid {fmt(output['qwen_latency_s']['matching_grid']['median'])} s, mismatch grid {fmt(output['qwen_latency_s']['mismatch_grid']['median'])} s. SAM3D-reported mean model+render time per flagged request was {fmt(output['reconstruction_timing_s']['mean_model_plus_render'])} s (median {fmt(output['reconstruction_timing_s']['median_model_plus_render'])} s); details are in `reconstruction.jsonl`.",
      retry_report,
      "Input-only was run on all selected requests. Matching and deterministic different-episode mismatch grids were run on flagged requests. Logged view-flip rates remain the frozen gate signal; regenerated grids were not rescored by SAM3, so this does not measure render-to-logged-source mismatch.",
      "Gate cost was not measured. Computing view flip requires SAM3D plus SAM3 over the candidate views even when Qwen is bypassed; this pilot does not show reconstruction savings. Qwen latency excludes model startup and uncertainty-gate computation.",
      f"Qwen protocol invalid-answer counts by arm are in `summary.json`; invalid responses remain unresolved and are counted separately from semantic insufficiency. Qwen inference exceptions: {sum(v['logged_error_count'] for v in output['qwen_protocol'].values())}; unrecovered reconstruction attempts: {output['reconstruction_retries']['logged_unrecovered_errors']}.",
      "## Navigation diagnostics","","| Policy | Episodes with known false accept | Episodes with only unresolved checks |","|---|---:|---:|"]
    for method,counts in output["navigation_diagnostics"]["episodes_by_policy"].items():
        report.append(f"| {method} | {counts['with_known_false_accept']} | {counts['with_only_unresolved_checks']} |")
    fv=output["navigation_diagnostics"]["future_observation_counts"]
    report += ["",f"Of {fv['unresolved']} grid-policy insufficient checks, {fv['with_later_distinct_real_image_and_higher_evidence_step']} had a later recorded distinct image with a strictly higher evidence step; {fv['resolved_known_later_eye_label']} had a clear later eye label, and the later policy agreed on {fv['later_policy_agrees_with_later_eye_label']}. {fv['censored_no_qualifying_later_observation']} had no qualifying later observation. These are observed diagnostics, not counterfactual recoveries.","",
      "## Interpretation","",
      f"On flagged requests, matching-grid Qwen returned {output['flagged_arm_status_counts']['matching_grid']['correct']} correct, {output['flagged_arm_status_counts']['matching_grid']['wrong']} wrong, and {output['flagged_arm_status_counts']['matching_grid']['insufficient']} insufficient verdicts. The correct verdicts comprised {output['mismatch_control_flagged']['matching_grid']['true_goal_accepts']} known goals, {output['mismatch_control_flagged']['matching_grid']['known_false_accepts']} known not-goals, and {output['mismatch_control_flagged']['matching_grid']['cant_tell_unsupported_accepts']} cant-tell rows. It introduced {output['methods']['gate_qwen_grid']['metrics']['raw_correct_introduced_errors']} known false accepts and repaired {output['methods']['gate_qwen_grid']['metrics']['raw_errors_repaired']} known raw errors by a correct opposite decision, while accepting {output['methods']['gate_qwen_grid']['metrics']['true_goal_accepts']-output['methods']['gate_defer']['metrics']['true_goal_accepts']} more true goals than gate-defer. The mismatch control was similarly conservative ({output['mismatch_control_flagged']['mismatch_grid']['true_goal_accepts']} known goals accepted; {output['mismatch_control_flagged']['mismatch_grid']['known_false_accepts']} false accepts).",
      f"Full-policy matching-grid minus input risk among decided clear cases was {fmt(risk_delta['a_minus_b'])} (95% CI {risk_ci}); the interval {risk_ci_text} while clear decided coverage fell by {abs(output['paired_deltas']['gate_qwen_grid_minus_input_policy_all_requests']['decided_coverage_clear']['a_minus_b']):.3f}. Matching and mismatch statuses differed on {output['mismatch_control_flagged']['status_disagreement_n']}/{output['mismatch_control_flagged']['n_flagged']} flagged requests. The pilot supports further triage testing but does not establish grid-specific benefit.",
      f"Review the {output['false_accept_review_examples']} introduced false accepts in [the visual audit sheet](review_false_accepts.jpg); labels remain provisional.","",
      "The rows were included in the earlier sufficiency study that motivated the margin/area/view-flip features. This is retrospective development screening, not an untouched confirmatory test. The calibration and inference cohort is episode-disjoint, but may overlap in scene. Labels are provisional agent-produced eye labels with partial agent review, not independently validated human ground truth. Qwen's option scores are not calibrated confidence. All requests have at-STOP=false, so no STOP subset is available.",
      "This screening study makes no closed-loop success, recovery, or SPL claim. Insufficient is retained as unresolved; it is never coerced into accept/reject. See `per_request.csv`, `manifest.csv`, `frozen_gate.json`, and `summary.json`.","",
      "## Reproduction commands","","```sh",
      "docker compose exec -T dev sam3d-env core python scripts/qwen_verification_study.py prepare --out outputs/apexnav_offline/qwen_verification_pilot_repro",
      "# Check nvidia-smi and choose one idle physical GPU before each model stage.",
      "docker compose exec -T dev sam3d-env core python scripts/qwen_verification_study.py reconstruct --out outputs/apexnav_offline/qwen_verification_pilot_repro --gpu 0",
      "docker compose exec -T dev /data/envs/qwen/bin/python scripts/qwen_verification_study.py infer --out outputs/apexnav_offline/qwen_verification_pilot_repro --gpu 0",
      "docker compose exec -T dev sam3d-env core python scripts/qwen_verification_study.py analyze --out outputs/apexnav_offline/qwen_verification_pilot_repro",
      "docker compose exec -T dev sam3d-env core python -m unittest tests.test_qwen_verification tests.test_perception -v",
      "```",""]
    (out/"report.md").write_text("\n".join(report))
    return output


def main():
    ap=argparse.ArgumentParser(); ap.add_argument("stage",choices=("prepare","reconstruct","infer","analyze")); ap.add_argument("--out",type=Path,default=DEFAULT_OUT); ap.add_argument("--smoke",action="store_true"); ap.add_argument("--gpu",type=int,default=0); a=ap.parse_args()
    if a.stage=="infer":infer(a.out,a.smoke,a.gpu)
    elif a.stage=="reconstruct":reconstruct(a.out,a.gpu)
    else:{"prepare":prepare,"analyze":analyze}[a.stage](a.out)

if __name__=="__main__": main()
