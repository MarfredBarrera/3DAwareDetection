"""Extract each run's committed detection (mask overlay) and final frame, for apexnav_gate_report.

Runs in the apexnav container (uses ApexNav's find_commit_evidence); a run dir holds
``epi<N>/records/<result>/<record>/`` as written by ApexNav's scripts/run_gate_e2e.sh:

    docker exec -i apexnav bash -c "source /opt/apexnav/apexnav-env.sh; use_core; cd /workspace/ApexNav; \
        python - /tmp/frames baseline=videos/gate_verified/baseline sam3=videos/gate_sam3/gate 59 63 133" \
        < scripts/apexnav_commit_frames.py

Writes ``<mode>_epi<N>_{commit,raw,final}.jpg``, ``<mode>_epi<N>_mask.png`` and
``summary.json`` ({"<mode>_<N>": result, target, commit step, verification counts, ...}).
"""
import glob
import json
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, "/workspace/ApexNav")
from basic_utils.record_episode.episode_visualization import find_commit_evidence

out = sys.argv[1]
runs = dict(arg.split("=", 1) for arg in sys.argv[2:] if "=" in arg)
episodes = [int(arg) for arg in sys.argv[2:] if "=" not in arg]
os.makedirs(out, exist_ok=True)
summary = {}
for mode, root in runs.items():
    for e in episodes:
        records = glob.glob(f"{root}/epi{e}/records/*/*")
        if not records:
            summary[f"{mode}_{e}"] = {"result": "missing"}
            print(mode, e, "missing")
            continue
        record = records[0]
        meta = json.load(open(f"{record}/meta.json"))
        steps = [json.loads(line) for line in open(f"{record}/steps.jsonl")]
        evidence = find_commit_evidence(record, steps)
        last = steps[-1]["step"]
        info = {"result": meta["result"], "target": meta["target_label"], "scene": meta["scene"],
                "episode_id": meta["episode_id"], "distance_to_goal": meta["distance_to_goal"],
                "final_step": last, "record": record, "commit_step": None, "evidence_step": None,
                "matched_by": None, "verification": meta.get("verification")}
        cv2.imwrite(f"{out}/{mode}_epi{e}_final.jpg", cv2.imread(f"{record}/rgb/{last:06d}.jpg"))
        if evidence:
            info.update(commit_step=evidence["commit_step"], matched_by=evidence["matched_by"])
        if evidence and evidence.get("mask_path"):
            image = cv2.imread(f"{record}/rgb/{evidence['step']:06d}.jpg")
            mask = cv2.imread(evidence["mask_path"], cv2.IMREAD_GRAYSCALE) > 0
            overlay = image.astype(float)
            overlay[mask] = 0.45 * overlay[mask] + 0.55 * np.array([0, 0, 255])
            contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            overlay = overlay.astype(np.uint8)
            cv2.drawContours(overlay, contours, -1, (0, 0, 255), 2)
            cv2.imwrite(f"{out}/{mode}_epi{e}_commit.jpg", overlay)
            cv2.imwrite(f"{out}/{mode}_epi{e}_raw.jpg", image)
            cv2.imwrite(f"{out}/{mode}_epi{e}_mask.png", mask.astype(np.uint8) * 255)
            info["evidence_step"] = evidence["step"]
        summary[f"{mode}_{e}"] = info
        print(mode, e, info["result"], info["target"], "commit", info["commit_step"], "evidence", info["evidence_step"])
json.dump(summary, open(f"{out}/summary.json", "w"), indent=1)
