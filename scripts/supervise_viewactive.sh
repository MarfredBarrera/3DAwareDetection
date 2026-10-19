#!/usr/bin/env bash
# Resume the full ViewActive pipeline after transient failures. Run detached in
# the dev container; stage outputs and the driver's lock make reruns idempotent.
set -u
cd /workspace

config=${1:-configs/viewactive.yaml}
run_dir=${2:-outputs/viewactive/full}
source_run=${3:-}
log="$run_dir/logs/supervisor.log"
mkdir -p "$run_dir/logs"

stamp() { date -u '+%Y-%m-%d %H:%M:%S UTC'; }
complete() {
    test -s "$run_dir/model/best.pt" &&
    test -s "$run_dir/eval/fields.json" &&
    test -s "$run_dir/eval/fig4.json"
}
finalize() {
    if [[ -n "$source_run" ]]; then
        sam3d-env sam3d python -m scripts.finalize_viewactive_run "$run_dir" \
            --source "$source_run" >> "$log" 2>&1
    else
        sam3d-env sam3d python -m scripts.finalize_viewactive_run "$run_dir" >> "$log" 2>&1
    fi
}

# The current driver holds this lock. Wait for it to finish or exit before
# deciding whether a restart is needed.
printf '%s waiting for current driver\n' "$(stamp)" >> "$log"
flock -x "$run_dir/logs/viewactive.lock" true

for attempt in 1 2 3; do
    if complete; then
        if ! finalize; then
            printf '%s final analysis failed\n' "$(stamp)" >> "$log"
            exit 1
        fi
        printf '%s full run complete\n' "$(stamp)" >> "$log"
        exit 0
    fi
    # Prefer the config's gpu_id; otherwise take whichever GPU has the most free
    # memory. Wait only if no GPU has 16 GB free.
    preferred=$(awk '$1 == "gpu_id:" {print $2}' "$config")
    while true; do
        gpu=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits 2>/dev/null |
              awk -F', *' -v p="$preferred" '$2 >= 16000 {if ($1 == p) {pick = p; exit}
                                                       if ($2 > best) {best = $2; pick = $1}}
                                             END {print pick}')
        [[ "$gpu" =~ ^[0-9]+$ ]] && break
        printf '%s no GPU has 16000 MiB free; waiting\n' "$(stamp)" >> "$log"
        sleep 300
    done
    printf '%s resume attempt %s on GPU %s\n' "$(stamp)" "$attempt" "$gpu" >> "$log"
    sam3d-env sam3d python -m viewactive --config "$config" --gpu "$gpu" --workers 4 \
        >> "$run_dir/logs/driver.log" 2>&1
    status=$?
    if complete; then
        if ! finalize; then
            printf '%s final analysis failed\n' "$(stamp)" >> "$log"
            exit 1
        fi
        printf '%s full run complete\n' "$(stamp)" >> "$log"
        exit 0
    fi
    printf '%s driver exited %s before completion\n' "$(stamp)" "$status" >> "$log"
    if (( attempt < 3 )); then
        sleep 120
    fi
done
printf '%s stopped after three failed resume attempts; inspect driver.log\n' "$(stamp)" >> "$log"
exit 1
