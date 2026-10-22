# ApexNav x perception.Verifier integration plan

Goal: when ApexNav believes it has detected the target, the detection must pass
`perception.Verifier`; a failed check is a non-detection and the agent keeps exploring.

## 0. Facts that shape the design

- **Detection** runs in Python: `habitat_evaluation.py:167` `get_object` (YOLOv7/GroundingDINO + MobileSAM;
  `label_list==0` is the target). Point clouds go to the C++ planner (`get_object_point_cloud`, l.184).
- **"Believed" is decided in C++.** `ObjectMap2D::isConfidenceObject` (`src/planner/plan_env/src/object_map2d.cpp:704`)
  requires fused confidence >= the LLM threshold and >= `min_observation_num` observations.
  `getTopConfidenceObjectCloud` (l.606) feeds `planNextBestPoint` (`exploration_manager.cpp:96-170`), which
  returns `SEARCH_BEST_OBJECT`, `SUSPICIOUS` (no threshold) or `EXTREME`. Every `SEARCH_*` result becomes
  `FINAL_RESULT::SEARCH_OBJECT`; the stop (`REACH_OBJECT`, `exploration_fsm.cpp:188-204`) happens in a later
  `PLAN_ACTION` cycle.
- **Recorded FPs (89 records):** 68 via `SEARCH_BEST_OBJECT`, 21 via `SEARCH_SUSPICIOUS_OBJECT` (19 never had a
  confident cluster), 0 via over-depth. A gate only at the confidence threshold would miss ~24% of FPs.
- **Cluster ids are stable** within an episode (index into `objects_`, never erased, reset per episode), so the
  cluster is the cache key.
- **ApexNav `verification/`** is the user's untracked offline review tool (SAM3 on recorded frames -> GIFs). Not an
  online hook. Leave it; new code goes in a new package `perception_gate/`.
- **Environments:** ApexNav env A py3.12 / torch cu126 / ROS 2 Jazzy + Habitat; env B py3.9 (VLM servers). Ours:
  `core` py3.12 / torch 2.8 cu126, `sam3d` py3.11 / torch 2.5 cu121 + PyTorch3D. SAM3D can't go into ApexNav's
  image, so the Verifier runs as a separate service.
- **Docker is rootless.** `apexnav:jazzy` is not loaded; it is only in `ApexNav/apexnav-jazzy-image-2026-10-17.tar`
  (31 GB, NAS). ApexNav compose mounts `${APEXNAV_DATA:-~/nas/apexnav_data}`, which doesn't exist; set it to
  `/ICONAS/users/marfred/apexnav_data` (datasets stay on the NAS).
- **Uncommitted ApexNav changes** (beyond the 4 known): modified `requirements-core.txt`,
  `docs/DOCKER_JAZZY_FOXGLOVE.md`, `scripts/verify_install.sh`; deleted `src/CMakeLists.txt` (catkin symlink);
  untracked `verification/` and the image tar. `find_commit_evidence`, `_view_quality`, `_mask_touches_border`
  exist only in the uncommitted `episode_visualization.py`.

## 1. Hook: verify the cluster the planner is about to go to, before it can stop

When `planNextBestPoint` selects an unverified target cluster (best, suspicious or extreme), the FSM sends a
verification request and waits (no action published, no sim step). Python verifies the best stored view of that
cluster and replies. Accepted: cluster marked accepted for good. Rejected: cluster excluded from every target list;
the agent replans as if it was never detected. Accepted: the action already planned is published (no replan, so
the stuck counters are not fed a fake no-move step); rejected: replan with `skip_stuck_check_once_`, which also
skips the path-stability heuristic. Gate off = exact baseline. Verifying every detection in `get_object` (~11 s
each per frame) is rejected.

Stop backstop (review 2026-10-22): over-depth targets and the extreme "others" cloud have no cluster id, so the
FSM also checks at the stop itself. When `callActionPlanner` returns `REACH_OBJECT` and the target cluster is not
accepted, it resolves the nearest non-rejected cluster with a cell within 0.75 m of the target object point
(`ExplorationData::target_object_pos_`, the nearest point of the target cloud; `next_pos_` is the approach point
0.5-0.85 m away) and requests it (`at_stop`); with no such cluster it requests the bare position
(`cluster_id = -1`, centroid = object point). Accept -> stop; reject -> no stop, the cluster is rejected and the
position is cut out of over-depth target clouds (`ObjectMap2D::suppressPosition`, 0.75 m), then replan. With
the gate on, the extreme "others" fallback (other-class clusters) is never a target. Request seqs never reset;
Python drops requests with seq <= the last one seen when the episode started.

Re-check (user decision 2026-10-22: rejection is not permanent; use a temporal/spatial cooldown). Python
reopens a rejected cluster (sends `state=0`, so the next selection re-requests it) only when all hold:
at least `recheck_cooldown_steps` sim steps since the last check, a new evidence view whose camera position is
>= `recheck_min_distance_m` from every previously used view of that cluster (or viewing direction differs by
>= `recheck_min_angle_deg`), and fewer than `max_checks` checks so far. Reopen checks run in `observe()`, so
they cost nothing until a new view qualifies. Re-spawned clusters: a new cluster within 0.75 m of a rejected one
inherits the rejection (check count, cooldown) without a server call, and verdicts are cached per evidence view
(step, index), so a cluster whose best view was already judged reuses that verdict.

C++ changes (ApexNav):
1. `src/planner/plan_env/msg/VerificationRequest.msg`
   (`uint32 seq, int32 cluster_id, float32 centroid_x, centroid_y, float32 confidence, int32 observation_num, int32 expl_result, bool at_stop`)
   and `VerificationResult.msg` (`uint32 seq, int32 cluster_id, int8 state`); add to `rosidl_generate_interfaces`
   in `plan_env/CMakeLists.txt:32`.
2. `object_map2d.h/.cpp`: `int verification_ = 0` on `ObjectCluster` (-1 rejected, 0 unverified, 1 accepted);
   `setVerification`/`getVerification`; `getTopConfidenceObjectCloud` gains `vector<int>* ids = nullptr` and skips
   `-1` in all three branches (incl. the extreme "others" cloud); `isConfidenceObject` also requires
   `verification_ != -1`; add `int8[] verification` to `ObjectFusionState.msg` and fill it.
3. `exploration_data.h`: `ExplorationData::target_cluster_id_` (-1 for over-depth / "others");
   `FSMData::skip_stuck_check_once_`, `verify_seq_`, `verify_start_`; append `"WAIT_VERIFICATION"` to `state_str_`.
4. `exploration_manager.cpp` `planNextBestPoint`: record the cluster id each `return SEARCH_*` used.
5. `exploration_fsm.h:75`: append `WAIT_VERIFICATION` to `ROS_STATE`; mirror in `params.py` as `= 6`.
6. `exploration_fsm.cpp`: params `verification.enabled` (false), `verification.timeout_s` (300); pub/sub.
   In `PLAN_ACTION` (after l.135): if enabled, `final_result_ == SEARCH_OBJECT`, `target_cluster_id_ >= 0` and
   unverified -> publish request, go to `WAIT_VERIFICATION`. Result callback (matching seq): `setVerification`;
   on reject set `final_result_ = EXPLORE` (else the reach check at l.188 could stop at it) and
   `skip_stuck_check_once_` (stuck detector at l.198); `publishObjectFusionState()`; back to `PLAN_ACTION`.
   Timeout: fail open, log ERROR, replan. Episode reset during `WAIT_VERIFICATION` -> `WAIT_TRIGGER`.
7. Launch: `verification_gate` arg (default false) -> `verification.enabled` in `algorithm.launch.py` and
   `exploration.launch.py` (same plumbing as `multiview_fusion`); `scripts/start_session.sh` l.103 appends
   `verification_gate:=${VERIFICATION_GATE:-false}`.

## 2. Architecture: our container serves `perception.Verifier` over HTTP

- New `perception/server.py`: stdlib `http.server.HTTPServer`, single-threaded (serialized), one `Verifier` with
  fixed settings per run.
  `sam3d-env core python -m perception.server --host 0.0.0.0 --port 8770 --gpu G --selector vqf --threshold 0.15 [--no-ensemble] --on-uncertain input --log-dir outputs/apexnav_gate/server`
- `GET /health` -> `{ok, protocol: 1, settings}`. `POST /verify` takes
  `{protocol, request_id, image (b64), mask (b64 png), target, input_elevation, vocabulary?}` and returns
  `{ok, verdict, view_image (b64|null)}` or 400/500 `{ok: false, error}`.
- CLI: `python -m perception.server ping URL`, `... verify URL --image --mask --target --elevation`.
- `Verdict.summary()` in `perception/client.py` (JSON-safe, views without images); `perception/__main__.py` reuses it.
- Docs: README "Use from a navigation planner", CLAUDE.md Layout.
- Transport: `docker network connect 3dawareverification_default apexnav`; ApexNav uses
  `http://3dawareverification-dev-1:8770`. No host port, no compose changes.
- ApexNav client `perception_gate/client.py`: ~40 lines, urllib + cv2, no import from our repo; checks protocol via `/health`.

## 3. Inputs

| Verifier input | ApexNav source |
|---|---|
| rgb | `rgb_raw` (`habitat_evaluation.py:154`); copy when gate on (`get_object` draws on it) |
| mask | `object_masks_list[i]` with `label_list[i] == 0` (480x640 uint8) |
| target | `label.split('\|')[0].replace('_', ' ')`; vocabulary `HOUSEHOLD` + target (as `apexnav_fp.goal`) |
| input_elevation | `-degrees(camera_pitch)` at the evidence step |
| intrinsics | not needed (SAM3D estimates; hfov 79, 640x480); logged only |

Evidence: `VerificationGate.observe()` stores each label-0 detection (step, rgb, mask, score, world centroid via
`get_object_point_cloud(..., return_centroids=True)`, camera pose, pitch, semantic instance, that step's cluster
centroids) in `perception_gate/evidence.py`'s `EvidenceStore`: per 0.75 m cell of the world centroid, the 8 most
readable views plus the newest; past 256 views the most crowded cell loses its worst non-newest view (a place's
last view is never dropped). A request picks the best view within 0.75 m of the cluster centroid (as it stood at
the view's step), else the nearest within 1.5 m, else `on_no_evidence` (default accept, logged). The ranking
(`view_quality`, `mask_touches_border`) is a copy of `episode_visualization.py`'s, so `perception_gate` runs on a
clean checkout. Intrinsics (hfov, width, height) are logged with every decision.

## 4. Python wiring, config, logging (ApexNav)

- `perception_gate/gate.py`: `VerificationGate(cfg)` with `start_episode`, `observe`, `handle(request, fusion) -> state`,
  `finish_episode` (counts). `habitat_evaluation.py` imports it only when `verification.enabled` (baseline has no new
  import). Startup pings `/health` (60 s timeout); `/health` reports worker liveness and the server exits non-zero
  when a worker dies. `max_consecutive_errors` (3) server errors in a row raise and abort the run, so a run never
  continues silently ungated. C++ timeouts (300 s) are only logged by the planner: Python's own HTTP timeout
  (120 s) fires first and is logged in `decisions.jsonl`, so a C++ timeout means Python itself was stuck.
- `habitat2ros/eval_node.py`: subscribe `/object/verification_request` into a latch; `publish_verification_result`;
  add `verification.enabled` to `query_planner_fusion_config` (l.173); `main()` aborts if Python and C++ flags differ.
- `habitat_evaluation.py`: in `run_episode` after l.115 handle a latched request, publish, `continue`; after l.184
  `gate.observe(...)`; `recorder.log_step(..., verification=...)` (new optional kwarg, `episode_recorder.py:301`);
  build the gate near l.276, `start_episode` next to `recorder.start` (l.325); counts into `recorder.finish` extras.
- `verification:` section in `config/habitat_eval_{hm3dv1,hm3dv2,mp3d}.yaml`:
  ```yaml
  verification:
    enabled: false
    url: http://3dawareverification-dev-1:8770
    timeout_s: 120
    on_error: accept
    on_no_evidence: accept
    max_consecutive_errors: 3
    ping_timeout_s: 60
    max_checks: 3
    recheck_cooldown_steps: 25
    recheck_min_distance_m: 1.0
    recheck_min_angle_deg: 45
    match_distance_m: 0.75
    log_dir: "{video_output_path}/verification"
    save_images: true
  ```
  Selector/ensemble/threshold/on_uncertain are server flags, copied from `/health` into every log line.
- `<log_dir>/decisions.jsonl`, one line per decision: run_index, scene, episode_id, target, cluster
  id/centroid/confidence/obs/check, expl_result, at_stop, source (server, cached, inherited), camera intrinsics, evidence (step, index, score, area, match distance, pitch, elevation),
  server settings, verdict summary, latency, error, and **ground truth** (evidence mask's semantic instance is a
  goal object, via `semantic_instances_for_masks` vs `episode.goals[*].object_id`).
  Images: `<log_dir>/epi<run>_<label>/c<id>_s<step>/{input.jpg, mask.png, view.png, decision.json}`.

## 5. Resource usage (estimates; measure in the smoke test)

- GPU: our server SAM3 ~4-6 GB + SAM3D/ViewActive ~10-14 GB (peak in 4-seed ensemble); ApexNav BLIP-2 ITM 5-8,
  GroundingDINO ~1.5, YOLOv7-e6e ~1.5, MobileSAM <1, habitat-sim 1-2 GB; ollama qwen3:8b +6 GB (skip if
  `llm/answers` cache covers targets). Total ~25-35 GB on one idle L40S (46 GB), no sharding: `VLM_GPU=G` and
  server `--gpu G`. Check `nvidia-smi -L` UUIDs match in both containers.
- CPU: ~6-10 busy cores. Our container exports OMP/MKL/OPENBLAS=2; in ApexNav export the same in every
  `start_session.sh` pane; skip Foxglove; optionally `docker update --cpus 8 apexnav`, `--cpus 4` ours.
- Time: ~11 s per verification (7 s `--no-ensemble`), typically 1-3 per episode.
- Before every run: `nvidia-smi; uptime`; pick a GPU with >= 40 GB free and low utilization.

## 6. Test plan

- Unit (`tests/test_perception.py`, `ServerTests`): server on port 0 in a thread with `perception.client._Worker`
  patched to `FakeWorker`; `/health`, `/verify` accepted/rejected round-trip, 400 on missing field, `ok: false`
  on worker error, protocol mismatch, `Verdict.summary()` JSON-safe.
  `docker exec 3dawareverification-dev-1 sam3d-env core python -m unittest tests.test_perception`, then full discover.
- Smoke (no ApexNav):
  ```bash
  nvidia-smi; uptime
  docker exec -d 3dawareverification-dev-1 bash -lc 'sam3d-env core python -m perception.server --host 0.0.0.0 --port 8770 --gpu G --log-dir outputs/apexnav_gate/server > outputs/apexnav_gate/server.out 2>&1'
  docker exec 3dawareverification-dev-1 sam3d-env core python -m perception.server ping http://localhost:8770
  docker exec 3dawareverification-dev-1 sam3d-env core python -m perception.server verify http://localhost:8770 \
    --image outputs/apexnav_fp/epi324_00869-MHPLjHsuG27_17_tv/input.jpg --mask outputs/apexnav_fp/epi324_00869-MHPLjHsuG27_17_tv/mask.png --target tv --elevation 30
  ```
  Compare with `outputs/apexnav_fp/results.json`; record peak GPU memory.
- ApexNav setup + ROS checks:
  ```bash
  docker load -i /ICONAS/users/marfred/ApexNav/apexnav-jazzy-image-2026-10-17.tar
  APEXNAV_DATA=/ICONAS/users/marfred/apexnav_data docker compose -f docker/docker-compose.yml up -d
  docker network connect 3dawareverification_default apexnav
  docker exec apexnav bash -lc './scripts/build_ws.sh && python -c "from perception_gate.client import ping; print(ping(\"http://3dawareverification-dev-1:8770\"))"'
  ```
  `ros2 interface show plan_env/msg/VerificationRequest`.
- End-to-end (modest): 3 verified-FP + 2 goal-class records (`data/apexnav/records/*/meta.json` run_index,
  `verified.txt`). Baseline (`VERIFICATION_GATE=false`, `test_epi_num=N`) to confirm the FP reproduces; then gate on
  (`VLM_GPU=G VERIFICATION_GATE=true ./scripts/start_session.sh`;
  `python habitat_evaluation.py --dataset hm3dv2 test_epi_num=$N verification.enabled=true video_output_path=videos/gate_vqf/{split}`).
  Then 10 non-FP val episodes for SR/SPL cost. Larger sweeps (`vqf`, `top2gap`) only after user approval.

## 7. Risks / open questions

1. Uncommitted ApexNav work (user decision: leave them): new branch from the current worktree; no stash/checkout/reset; `git add` explicit paths.
2. Over-depth targets are not verified (0/89 FP records ended that way).
3. Wrong rejections: mitigated by the cooldown re-check; tune cooldown values on the FP records.
4. Failure policy: fail-open (user decision), always logged.
5. After a rejection the planner may chase lower-confidence clusters, each costing a verification.
6. One Verifier config per server run.
7. 31 GB image load from the NAS is slow; GPUs are busy.
8. ApexNav `verification/` may be superseded by `scripts/apexnav_fp.py`; untouched here.
