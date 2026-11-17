# nav: planner-agnostic gate, approach and last-mile navigation

Everything an ObjectNav planner needs to put the Verifier in its loop, with no ROS and no planner imports. The planner
keeps its exploration and its own detector; this package decides **whether a candidate detection is the target**
(accept / reject / defer), drives the **approach** to get a better view when the first one is insufficient, and
optionally replaces the planner's motion toward an accepted target (**split navigation**). ApexNav is the first
adapter (`planners/ApexNav`, `habitat_evaluation.py`); a VLFM adapter implements the same calls.

```
nav/gate      client.py (HTTP to perception.server), evidence.py, gate.py (VerificationGate), approach.py (LiveApproach),
              ground_truth.py (logging helpers)
nav/approach  mppi.py (MppiV15 + closer.py warm start), decide.py (belief + fused SAM3 verdict), size_opinion.py,
              local_nav.py, geom.py, scene.py
nav/lastmile  split.py (SplitNav), medial.py (A* on the agent's state lattice), collision_map.py, body_map.py
nav/eval      closer_attrib.py, nav_table.py (outcome scoring of run directories)
```

Requirements: numpy, scipy, cv2, torch (CPU) and, for the gate, the Verifier server (`python -m perception.server --selector
jev_sam3`, `perception.jev` and `perception.select` are imported by `nav.approach.decide`). Put the repo root on `PYTHONPATH`.
The agent body is a `habitat_sim.Simulator` (`sim.get_agent(0).get_state()`, sensor `"rgb"`; `nav.approach.scene`
reads pose and camera from it, numpy-quaternion needed), with ApexNav's embodiment constants (640x480, hfov 79, camera
0.88 m, radius 0.18, forward 0.25 m, turns 30 deg, no sliding, camera pitch 0 during an approach). Another simulator or
robot needs its own `scene.agent_pose` / `scene.camera`.

## What the planner provides

Frames and poses of the **planner's own** frame: `pose = (x, y, yaw)` in its world (x, y plane), detections as
target-class masks (`H x W`, nonzero on the object) with their world centroids `(x, y)`, RGB `H x W x 3 uint8`, depth in
metres `H x W float32` (sensor range 5 m), `label` the target category.

| Call | When | Inputs | Output |
|---|---|---|---|
| `VerificationGate(cfg, log_dir_root, camera={hfov, width, height, position})` | once | `cfg` = the `verification.*` keys (`gate.DEFAULTS`; `enabled`, `url`, `approach`, `on_error=reject`, `multiview*`, `approach_planner=mppi_v15`, `on_no_evidence`) | pings the server; raises on a wrong URL / selector |
| `gate.start_episode(run_index, scene, episode_id, label, goal_object_ids, max_episode_steps)` | episode start | | |
| `gate.observe(step, rgb, masks, labels, scores, centroids, semantic_instances, pose, camera_pitch, fusion, external_approach)` | after every executed step | label `0` = target class, `fusion = {"clusters": [{"id", "centroid"}]}` (the planner's object clusters) | ids of rejected clusters to reopen (the planner forgets their verdict) |
| `gate.handle(request, fusion, step, frame)` | the planner selected a candidate cluster and **waits** | `request`: `seq, cluster_id, centroid_x, centroid_y, confidence, observation_num, expl_result, at_stop`, and for a re-request `approach_id, viewpoint_status, approach_steps`; `frame` from the last step (`depth, masks, centroids, cam`) | `1` accept, `-1` reject, `2` defer (treat as a non-detection, keep exploring), `3` approach |
| `gate.pending_approach(cluster_id)`, `gate.approach_close(request, last_step)`, `gate.approach_handoff(request, last_step)` | state `3` | | the approach issued; whether the object is close and in view; whether the planner's own navigation keeps approaching (`"native"`) or the gate's planner takes over (`"planner"`) |
| `LiveApproach(sim, url, target, object_pf, budget, "mppi_v15", log_dir, tag, bank=gate.approach_bank(cluster_id))` then `approach.step(rgb, depth, masks, centroids, pose)` | the gate's planner drives the approach (agent pitch 0) | the frame of the step just executed | the next action (`move_forward` / `turn_left` / `turn_right`) or `None` when committed |
| `gate.finish_external_approach(request, step, approach.result, approach_log)`; `gate.remember_object(approach.object_points(), state)` | the approach ended | | the verdict (state `1`, `-1`, `2`) to return to the planner |
| `EpisodeCollisionMap(sim).update(depth, moved_forward, position, yaw)` | every step (split navigation) | | the map `SplitNav` plans on (`.lmap`) |
| `SplitNav().act(native_action, target, sim, pose_pf, gate, body_map)` | before executing a motion action | `target = {cluster_id, x, y, accepted, heading_to_object, log}`: the planner's current target and whether it is gate-accepted (or an undecided object it navigates to) | `(override action or None, info)`: an override replaces the planner's action, `None` keeps it; the planner's STOP is never overridden |
| `gate.finish_episode()`, `gate.pop_events()` | episode end / per step | | counts, decision events for the step record |

The decision flow of a request: JEV judges only the uncertainty of the first frame; a sure frame is decided by SAM3's raw
label; an unsure one makes the planner approach, and the gate's planner (`MppiV15` on the agent's own depth map) collects
views (`decide.py`: JEV naming belief, new viewpoints only) until P(consistent) >= 0.8, then the fused SAM3 verdict over the
nearest views (with the metric-size veto) accepts or rejects. No evidence is a deferral, never a verdict. After an accept the
planner's own navigation (optionally split navigation) goes on to the object.

## ApexNav wiring (`planners/ApexNav/habitat_evaluation.py`)

- `build_verification_gate`: one `VerificationGate` per run from the `verification:` config (`url: ${oc.env:VERIFIER_URL}`).
- `execute_step`: `gate.observe` after every step with the planner's target detections; the reopened ids ride with the
  `StepObservation` to the C++ planner (`sync_step`: the planner plans only after each step's inputs).
- `run_episode`: a latched `VerificationRequest` -> `gate.handle`; state `3` -> `gate.approach_handoff` / `approach_close`
  decide between ApexNav's own object navigation and `run_gate_approach` (a `LiveApproach` loop through `execute_step`
  with `external_approach=True`); the verdict goes back as `VerificationResult`.
- `NAV_SPLIT=true`: `run_split` builds `target` from the `StepAction` (`target_accepted`, `target_cluster_id`, `target_x/y`,
  `final_state == SEARCH_OBJECT` with an object-search `expl_result`) and calls `SplitNav.act`; the episode collision map is fed in
  `execute_step`. Sliding must be off.
- Runs: `planners/ApexNav/scripts/run_pool.sh` (arms `baseline`, `split`), outputs in `planners/ApexNav/videos/<run>`; score
  with `python -m nav.eval.closer_attrib` / `nav.eval.nav_table`.

## Writing another adapter (e.g. VLFM)

1. Map the planner's frames to the conventions above (pose, target masks and centroids, depth in metres).
2. Hold the planner when it selects a candidate cluster, call `gate.handle`, apply the state (accept: stop / go on; reject and
   defer: drop the candidate; approach: `approach_handoff` then `LiveApproach.step` until `None`).
3. Call `gate.observe` after every step; forget the verdict of the ids it returns.
4. Optionally call `SplitNav.act` before each motion action once the planner heads to a gate-accepted target.
5. Provide `scene.agent_pose` / `scene.camera` for the simulator in use.
Tests: `python -m unittest discover -s nav -t .` (core env, `PYTHONPATH` = repo root).
