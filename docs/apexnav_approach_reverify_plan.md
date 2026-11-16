# Plan: D1 closed loop in ApexNav (sufficiency check -> SAM3 / Qwen+grid -> approach)

Rewritten 2026-11-03 (replaces the earlier approach-and-reverify draft). Target metric: **FS+FR**
(CLOSER episode attribution, `outputs/apexnav_online/failure_causes_2026-11-03/`). No top2gap, no vqf.

## 1. Design (user, 2026-11-03)
For every verification request (frame, mask, target):
1. **Sufficiency check** on the frame: rank mean of -|SAM3 raw margin| and -mask area, against the 131
   offline eye-labelled requests; insufficient if score >= 0.546 (fit on those 131, never on test data).
   **Sufficient -> SAM3 on the raw frame decides** (accept / reject). No SAM3D, no Qwen.
2. **Insufficient -> Qwen3-VL-8B (int8) three-way** on the outlined frame + SAM3D 24-view grid:
   correct -> accept, wrong -> reject, insufficient -> step 3.
3. **Approach for a better view**; every new view of the object re-enters step 1. Failure -> reject.

Offline evidence (`outputs/apexnav_offline/qwen_all_requests_2026-11-03/`, 295 eye-labelled HM3D-v2 requests):
false accepts 14 -> 7 of 45 clear not-goals, false rejects 3 -> 0 of 160 clear goals, 37% of requests go to
approach (66/90 poor views, 21 clear goals, 22 clear not-goals). No-gate baseline accepts all 45.
Remaining false accepts are mostly armchair-as-couch. int8 vs bf16 Qwen: 115/117 identical, 2 accept ->
insufficient (`qwen_verification_pilot/int8_parity/`).

Why approach (CLOSER, arXiv 2610.28476, Table IV): a tri-state verdict without moving lowers FS but raises FR
(13.8 -> 15.1); approaching lowers both (FR 9.3). Our pilot: 89/93 "insufficient" checks never got a better
view on their own.

## 2. Server side (3DAwareVerification, branch apexnav-gate)
Build on the uncommitted `qwen_grid` selector (perception/client.py, qwen.py, server.py, worker.py;
design notes in `docs/qwen_pipeline_architecture.md`) rather than adding a parallel path:
- **Gate:** 2-signal mode (margin, area); no view flip, so SAM3D runs only for insufficient requests.
  Freeze references + threshold in `outputs/apexnav_offline/qwen_all_requests_2026-11-03/d1_gate.json`
  (from `_compare.py fit_thresholds`). Selector name `d1` (or `qwen_grid --gate d1_gate.json`).
- **Qwen:** int8, three-way-v1 prompt, frame outlined + grid (azimuths 0..315 step 45, elevations 0/30/60,
  distance 2.0, 320 px, seed 42, ensemble 1: the offline settings). Invalid answer or failed reconstruction
  -> insufficient, logged with the reason.
- **Reply:** `verdict.status` in {correct, wrong, insufficient}, `verdict.source` in {sam3, qwen}, the gate
  score, Qwen probabilities, and the grid image for logging. Stateless: one request in, one verdict out.
- **Tests:** gate thresholds reproduce `_compare.py` D1 decisions on the 295 requests (CPU replay with stored
  scores + Qwen answers); worker lazy start (no SAM3D/Qwen until first insufficient request); protocol.

## 3. ApexNav side (new branch `d1-approach` off `ovon-eval`)
### 3.1 Verdict mapping (perception_gate/gate.py `handle`)
correct -> ACCEPTED (1), wrong -> REJECTED (-1), insufficient -> **APPROACH (3)** with the object point
(the evidence centroid, i.e. the detection's back-projected mask) and the camera position of the view.
At the stop backstop (`at_stop`), insufficient never stops the agent (see 3.4).

### 3.2a Approach = ApexNav native object navigation (user, 2026-11-03; supersedes 3.2)
First smoke (frame-only): the ported viewpoint search found no reachable viewpoint in 10 of 12 approaches
(5 m path cap from 1d3436e, bearings from the evidence camera, no straight-on candidate under 1.75 m).
New design: insufficient -> the cluster becomes the forced planner target and ApexNav's own object path
(searchObjectPath, approach point 0.5-0.85 m, per-step replanning) drives toward it; new views en route are
verified at once; at REACH_OBJECT one face-turn + verify; still insufficient -> reject. Budgets unchanged.

### 3.2 Approach module (superseded by 3.2a) (C++ planner, reusing ApexNav `1d3436e` on `backup/2026-10-26-active-timing`)
Reused: `startViewpointMove`, `planViewpointAction`, `endViewpointMove`, `searchViewpointPath`
(known-free + line-of-sight + NORMAL A*), the `vp_*` FSM fields, arrival / face-object / abort logic,
`VerificationRequest.viewpoint_status`. New or changed:
- **Target:** object point from the evidence (not SAM3D guides, not the fused cluster centroid).
- **Viewpoint candidates:** bearing = current line of sight (object -> camera) with jitter 0, +-20, +-40 deg;
  radius 0.8-1.5 m (step 0.25); cost = path length + 1.0 m x |jitter| rank. If the camera is already within
  1.5 m, use only the +-20/+-40 bearings (a side view: the object is likely cut off or occluded, not far).
- **Budgets:** max 2 approaches per cluster, 40 steps per approach, 120 approach steps per episode; all count
  against the 500-step limit. Turns while approaching don't count toward `stucking_action_count_`.
- **Re-verification:** while approaching (and on arrival, after turning to face the object), the first new
  label-0 detection of the cluster that is >= 0.5 m closer or >= 30 deg from every view already judged is
  sent as a new request (`approach=true`), bypassing the 25-step cooldown (reopen guards in
  exploration_fsm.cpp ~F:385 and map_ros.cpp ~281 accept state 3). The verdict cache must not return the
  old view (new evidence is required by construction).
- **Outcomes:** accept -> continue to the object and stop as today; reject -> state -1, resume exploration;
  insufficient -> next approach if budget remains, else reject. Unreachable (no candidate), aborted (budget,
  no progress, viewpoint occupied, no path) or not re-detected on arrival -> reject (user decision).
  A rejected cluster keeps today's reopen rule (cooldown + new view).
- **One cluster at a time:** while approaching, other detections are stored as evidence but not verified;
  the backstop's `nearestCluster` id is pinned to the approached cluster.

### 3.3 Determinism (sync_step)
The approach issues exactly one action per processed step through `planViewpointAction`; no wall-clock
budgets; gate timeout aborts as today. Qwen decisions are argmax over logits (deterministic up to GPU
kernels; SAM3D/SAM3 already show rare near-tie flips, sync README).

### 3.4 Stop backstop
REACH_OBJECT on a cluster that is not accepted: turn to face the object (one action), verify the fresh view.
Insufficient at the stop -> one side-view approach (+-40 deg, within 1.5 m) if budget remains, else reject.
Never stop on an insufficient verdict.

### 3.5 Logging
`decisions.jsonl`: gate score + features, source (sam3/qwen), Qwen probabilities, status, approach id,
approach status (arrived / early / unreachable / aborted / not_redetected), steps used, viewpoint. Grid images
per Qwen decision. `results.csv`: requests, sam3 decisions, qwen decisions, approaches, approach outcomes.
`failure_causes_2026-11-03/_causes.py` extended to the new runs (FS, FR-gate, FR-other, explore, other).

## 4. Validation before the full runs
1. Unit tests (both repos): verdict mapping, approach budgets, reopen guards for state 3, cache bypass.
2. Reviewer agent on both diffs (correctness, determinism, failure handling).
3. **GPU memory (one GPU only, user 2026-11-03):** SAM3, SAM3D, Qwen int8 and ApexNav share one GPU.
   Requests are served one at a time, so compute is already staggered. Record the peak in the smoke run; if
   memory is tight, load Qwen/SAM3D lazily and release between calls on the same GPU. Never a second GPU.
   If memory is still an issue, drop the grid: Qwen judges the outlined frame alone (server flag
   `--d1-evidence input`, SAM3D never started) (user, 2026-11-03).
4. Determinism: 20 sample episodes twice in sync_step -> identical steps.jsonl (`compare_runs.py`).
5. Smoke: 10 FR-gate + 10 FS episodes from `failure_causes_2026-11-03/episodes.tsv` (HM3D-v2), every
   approach checked by eye (frames per approach step).

## 5. Runs
| run | episodes | strategies | est. GPU time |
|---|---|---|---|
| HM3D-v2 | 200, `sync_2026-10-25` sample | **d1** (baseline, sam3 reused) | ~6-8 h |
| OVON val_unseen | 200, `ovon_unseen_2026-10-27` sample | **d1** (baseline, sam3 reused) | ~8-10 h |
Same rules and flags as those studies otherwise; if shared code changes (e.g. the backstop face-object
turn), rerun baseline and sam3 on the first 20 episodes to confirm unchanged outcomes, else rerun them fully.
One GPU at a time, servers stopped after each run, archives to the NAS.

## 6. Analysis
FS, FR-gate, FR-other, FS+FR and success with paired bootstrap CIs vs baseline and sam3; eye labels for
every cell whose outcome differs from sam3 (OVON: from the eye table). Approach stats: triggers, arrivals,
early decisions, outcomes, steps used, decisions by source. Success criteria: FS+FR below baseline and
below sam3 on both datasets (HM3D-v2 baseline 10.0 / sam3 7.5; OVON eye 14.0 / 15.0), FS not above sam3.

## 7. Decisions (user, 2026-11-03)
1. One GPU only; assume everything fits, stagger compute if needed.
2. Keep the current architecture: build on the uncommitted `qwen_grid` work in perception/.
3. Workflow as for OVON: one agent implements per repo, one reviews; GPU smoke and full runs only after review.

## 8. Interface contract between the two repos
- Server reply `verdict`: existing fields plus `status` in {"correct","wrong","insufficient"}, `source` in
  {"sam3","qwen"}, `gate` {score, threshold, margin, area, insufficient}, `qwen` {status, probabilities,
  invalid_answer, error} or null. `accepted == (status == "correct")`. Server selector: `--selector d1`.
- ApexNav gate states in `VerificationResult.msg`: 1 accept, -1 reject, 2 defer (unused by d1), 0 reopen,
  **3 approach** (+ object_x/y = evidence centroid, camera_x/y = evidence camera).
- `VerificationRequest`: `approach_id` (0 = none), `viewpoint_status` (existing: 1 arrived, 2 unreachable,
  3 aborted) plus 4 = early (new view during the approach).
