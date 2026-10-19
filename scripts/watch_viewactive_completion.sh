#!/usr/bin/env bash
# Detached report writer for a run already being supervised by an older driver.
set -euo pipefail
cd /workspace
run_dir=$1
source_run=${2:-}
log="$run_dir/logs/finalize.log"
for ((attempt = 0; attempt < 864; attempt++)); do  # up to 72 hours, every five minutes
    if [[ -s "$run_dir/model/best.pt" && -s "$run_dir/eval/fields.json" &&
          -s "$run_dir/eval/fig4.json" ]]; then
        if [[ -n "$source_run" ]]; then
            flock -x "$run_dir/logs/viewactive.lock" \
                sam3d-env sam3d python -m scripts.finalize_viewactive_run "$run_dir" \
                --source "$source_run" >> "$log" 2>&1
        else
            flock -x "$run_dir/logs/viewactive.lock" \
                sam3d-env sam3d python -m scripts.finalize_viewactive_run "$run_dir" \
                >> "$log" 2>&1
        fi
        exit 0
    fi
    sleep 300
done
printf 'ViewActive run did not complete within 72 hours: %s\n' "$run_dir" >> "$log"
exit 1
