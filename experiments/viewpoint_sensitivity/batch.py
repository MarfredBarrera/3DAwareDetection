"""Durable server-side driver for a reviewed dataset, with logs and resume.

Run using docker compose exec -d ... so the worker survives client disconnects.
Re-running skips successful stages and resumes the query stage's JSONL progress.
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from experiments.viewpoint_sensitivity.run import load_config
from experiments.viewpoint_sensitivity.summarize_outputs import write_summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--render-only", action="store_true",
                        help="render and verify without recognition scoring")
    args = parser.parse_args()
    config = load_config(args.config)
    output = Path(config["output_dir"])
    output.mkdir(parents=True, exist_ok=True)
    fingerprint = hashlib.sha256(args.config.read_bytes() + Path(config["canonical"]).read_bytes()).hexdigest()
    run_dir = output / "run"
    run_dir.mkdir(exist_ok=True)
    state_path = run_dir / "batch-status.json"
    with open(run_dir / "batch.lock", "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        state = json.loads(state_path.read_text()) if state_path.exists() else {
            "fingerprint": fingerprint, "completed_stages": []}
        if state["fingerprint"] != fingerprint:
            raise ValueError("Config/canonical changed: use a new output directory")

        def status(stage, value, **extra):
            state.update(stage=stage, status=value,
                         updated_at=datetime.now(timezone.utc).isoformat(), **extra)
            temp = state_path.with_suffix(".tmp")
            temp.write_text(json.dumps(state, indent=2) + "\n")
            temp.replace(state_path)

        commands = {
            stage: [sys.executable, "-m", "experiments.viewpoint_sensitivity.run",
                    "--config", str(args.config), "--stage", stage]
            for stage in ("render", "query", "analyze")
        }
        commands["verify"] = ["sam3d-env", "sam3d", "python", "-m",
                              "experiments.viewpoint_sensitivity.upright", "verify-cameras",
                              "--index", str(output / "render/index.jsonl"),
                              "--out", str(run_dir / "camera-verification.json"),
                              "--settings", str(output / "settings.json")]
        stages = ("render", "verify") if args.render_only else ("render", "verify", "query", "analyze")
        for stage in stages:
            if stage in state["completed_stages"]:
                continue
            status(stage, "running")
            with open(run_dir / f"{stage}.log", "a", buffering=1) as log:
                result = subprocess.run(commands[stage], stdout=log, stderr=subprocess.STDOUT)
            if result.returncode:
                status(stage, "failed", returncode=result.returncode)
                raise SystemExit(result.returncode)
            state["completed_stages"].append(stage)
            status(stage, "completed", returncode=0)
        status("render-only" if args.render_only else "all", "completed")
        write_summary(output.parent)


if __name__ == "__main__":
    main()
