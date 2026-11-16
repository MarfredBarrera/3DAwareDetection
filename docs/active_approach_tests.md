# Active-approach tests (controlled, Habitat only)

Question: starting from a poor view of an object, how well does an approach policy move the
agent to views that let the perception decide correctly (accept a target, reject a look-alike),
and at what path cost? No ROS planner: Habitat-sim with ApexNav's embodiment (640x480, hfov 79,
camera 0.88 m, forward 0.25 m, turns 30 deg, radius 0.18), so policies and judges can be swapped
and every run is deterministic.

Code: `~/codes/ApexNav/active_approach/` (new package; Habitat lives in the `apexnav` container).
Outputs: `~/codes/ApexNav/videos/active_approach/` (the container cannot see this repo), copied to
`outputs/apexnav_online/active_approach_<date>/` when a run is final.

## Scenarios (`python -m active_approach.scenarios`)
HM3D-v2 val (36 scenes; `--dataset ovon` reads OVON val_unseen goals, untested). Per scene and goal
category: up to 2 goal instances (**target**) and up to 2 non-goal look-alikes (**confuser**, e.g.
stool/bench for chair, armchair for couch, sink/bathtub for toilet, picture/mirror for tv, vase
for plant; `LOOKALIKES`). Per object, 48 start poses on its floor (0.5-6 m away; 60% facing it,
40% with it 25-42 deg off-centre), rendered and measured with **GT visibility**:
over the object's camera-facing surface samples (from the HM3D semantic mesh), the fraction
visible / outside the frame (`trunc_frac`) / hidden by another surface (`occ_frac`), plus visible
pixels, area, distance and `facing_cos` (mean |cos| of visible normals vs. rays).
A start must show >= 400 px of the object but not be GT-sufficient
(vis >= 0.6, trunc <= 0.15, area >= 1%). Tags (first match): occluded (occ >= 0.4), truncated
(trunc >= 0.35), far (>= 3.5 m, area < 1%), oblique (facing_cos < 0.35), poor (other), plus a few
clear (sufficient) controls. One start per (object, tag), at most 30 per (role, tag).
Files: `scenarios.jsonl` (scene, object id/name, role, target label, start pose, floor, GT
visibility, tags), `starts/<id>.jpg` (object outlined), `sheet_<role>_<tag>_NN.jpg` for eye checks.
The semantic labels are noisy: eye-check the sheets and drop/relabel bad scenarios before using
numbers.

## Uncertain starts only (JEV, 2026-11-05)
Approach is tested only from starts the agent itself would doubt. Signal (the user's choice;
`outputs/apexnav_offline/reliability_2026-11-05` README, `_vis_conf.py`, `_wrong_only.py`): JEV-27B-VL naming on the
red-outlined frame, u = mean percentile (vs the 131 offline requests) of naming entropy H/log16 and look-alike
contention min(p_target, max confuser p); uncertain iff u >= 0.4664, the online 45%-approach cut-off. Frozen by
`outputs/apexnav_online/active_approach_2026-11-05/_jev_gate.py` -> `jev_gate.json`; served by
`scripts/jev_server.py` (container `jev-judge`, int8, ~29 GiB on one GPU; output matches `jevgpu.jsonl`,
|du| <= 0.002) and used by the harness as `--judge jev` (JEV naming top-1 decides).
- **Real starts** (`source: jev_real`): the 216 of 444 eye-labelled requests JEV flags.
  `_real_requests.py` (host) takes each request's episode record and logged poses; `active_approach.real`
  re-renders the exact logged frame (the episode is the k-th of the scene's content file, k from the record
  name; when the request step does not match, the best-matching earlier logged pose, i.e. the evidence frame)
  and takes the HM3D instance with most pixels under the logged mask (walls/floors excluded). Truth: eye label,
  for cant_tell the prior label. `compare/` shows logged | re-rendered for every one; eye-checked, drops in
  `scenarios_jev_real/eye_drops.tsv`; objects labelled goal in one run and not_goal in another are dropped.
- **Synthetic starts**: the 357 generated plus a larger pool (`--seed 1 --per-category 4 --samples 64`), each
  start re-rendered with the GT object mask, scored by `active_approach.jev_filter`; only JEV-uncertain kept,
  up to 30 per (role, tag).
Result: `scenarios_uncertain/scenarios.jsonl` (+ `jev_scores.jsonl`, `dropped.tsv`, sheets).

## Runner (`python -m active_approach.run`, 2026-11-06)
This is the deployed decision flow (ApexNav `perception_gate`: approach plus view bank), with the
user's embodiment constraints.

**Constraints**
- **Pitch is fixed at 0 for the whole run.** ApexNav itself starts at -30 (it looks down, spins 12
  turns, then looks up) and then navigates at 0. So 0 is its real operating pitch.
- **The 9 kept real starts logged at -30** (12 before the eye drops) are re-rendered at pitch 0 from
  the same pose. They are flagged `pitch_reset` and reported separately. The object may be cut off or
  out of frame at 0; if so, the start is not judgeable, the bank stays empty, and the run rejects.
- **Non-holonomic base.** The only actions are `move_forward` 0.25 m and `turn_left`/`turn_right`
  30 deg (ApexNav's `habitat_eval_hm3dv2.yaml`), with `allow_sliding: false` as in ApexNav. There is
  no strafing, teleporting or tilting.
- **Every action counts** against the budget of 40 (ApexNav `approach_max_steps`). Every reachable
  yaw is therefore yaw0 + k * 30 deg.

**Controller**
- It follows the navmesh geodesic on the lattice: aim at the path point 0.5 m ahead, turn only when
  more than 25 deg off, and step forward if `try_step_no_sliding` is free.
- When blocked, habitat's greedy follower takes 4 actions. The greedy follower alone zig-zags, turning
  every step on off-lattice paths; the line-follow/greedy mix once dithered L/R, and the 4-action hand-off fixed it.
- It finishes with a face turn (within 15 deg).

**Flow**
1. **Start frame.** The judge's `first` call runs on it (ApexNav's first `/verify`). A decision ends
   the run.
2. **Unsure start.** The policy gives viewpoints. The judge is asked only when the agent is
   stationary at a viewpoint and faces the object: |yaw - bearing to the cloud center| <= 20 deg and
   at least 200 mask px. The frame joins the bank, and `/verify_views` runs unforced ("early").
3. **The policy has nothing left:** a forced "arrival" call (M >= 0, V >= 0.5).
4. **The budget runs out, no path, or stuck:** a forced "failure" call (M >= 0.15).
5. **An empty bank rejects.**

- **En route (S2/S3 only).** A frame right after a forward step can also be judged unforced, if it
  is judgeable, the object is not clipped at the sides, and it is new by ApexNav's `approach_view`
  rule (0.5 m closer than, or 30 deg around from, every judged view).
- **Map.** The agent's map is seeded with a 12-turn depth panorama at the start, which is not
  counted: ApexNav has its exploration map when it triggers. The 2.5D height map
  (`geom.HeightMap`) grows with every frame. The object cloud comes from the GT mask and depth.
- **Logs.** Each step writes pose, action, GT visibility, the viewpoint (with planner terms), the
  judge reply and SVPTO pose validity. Frames are saved only when judged, plus the last one (home
  quota). `result.json` holds the outcome.

**Judges** (`judges.py`)
- **`jev_sam3`.** `perception.server --selector jev_sam3`, with the same thresholds as the
  closed-loop run.
  - `first` -> `/verify`. It is cached per scenario in `<run>/first_jev_sam3.jsonl`, since every
    policy starts from the same frame.
  - `views(bank, mode)` -> `/verify_views` over ApexNav's own `select_views` (<= 6 views, dup 0.3 m /
    15 deg, view_quality with a constant detector score of 0.5), `forced = mode != early`, margin 0.0
    (early/arrival) or 0.15 (failure), and vote 0.5.
- **`gt`.** GT sufficiency, for Habitat-only dry runs. `HttpJudge` stays only for `jev_filter`.

**Policies** (`policies.py`, `svpto.py`, geometry in `geom.py`). Run-time inputs only, except the
oracle.
- `stay`: the identity control.
- `native`: 1.5 m standoff, then 0.75 m.
- `random_ring`: 2 random lattice-aligned viewpoints at 1-2.5 m.
- `oracle`: a GT ring.
- `fba`: M1.
- `svr`: M2a, the GT-free product.
- `*_tour`: the M3 wrapper (K = 3, at least 45 deg apart).
- `s2`: S2 + M3. Ablations: `s2_nofm` (w_FM = 0) and `s2_wedge` (beta_FM = beta_h).
- `s3`: M2 + M3 driven by the visibility MPC. Its ablation `s3_abl` (s = 0) is MPC-driven M2 + M3.

**Metrics** (`python -m active_approach.report <run> [--eye eye.tsv]`)
- `summary.csv`: the wrong count (FA + FR), unsure starts, resolved (sure at a stop or en route)
  against forced, wrong resolved against wrong forced, and actions/path/stops on unsure starts.
- `paired.csv`: exact McNemar against stay and against random_ring.
- `sheets/before_after_*.jpg` and `sheets/wrong_<judge>_<policy>_NN.jpg`.

**Run**
```
docker exec apexnav bash -c 'source /opt/apexnav/apexnav-env.sh; use_core; cd /workspace/ApexNav
  python -m active_approach.run --scenarios videos/active_approach/scenarios_uncertain/scenarios.jsonl \
    --ids-file videos/active_approach/subset120.txt --policies stay native ... --judge jev_sam3 \
    --out videos/active_approach/runs/<name> --gpu $GPU'
```
The server goes in the dev container on the same GPU:
`sam3d-env core python -m perception.server --host 0.0.0.0 --port 8770 --gpu $GPU --selector jev_sam3 --log-dir ...`.
`zeta_max` is frozen once: `python -m active_approach.svpto zeta-max --scenarios ...` writes
`videos/active_approach/zeta_max.json` (1916; the median start is 722).
`subset120.txt` holds 112 starts: up to 5 per source x role x tag, drawn from the 467 that jev_sam3's
gate calls unsure (seed 0).

## Status (2026-11-05)
Scenario set `outputs/apexnav_online/active_approach_2026-11-05/scenarios_hm3dv2/` (357 scenarios,
36 scenes, 30 per role x tag; 27 confuser/clear). Smoke `runs/smoke_2026-11-05/` (10 scenarios,
stay/native/oracle x gt/sam3, sheets in `sheets/`), eye-checked: harness works; failure modes
seen: native goes too close to a low object (stool cut off at 0.9 m), a decision taken mid-turn
on a half-visible object (oracle/sam3 rejected an armchair at the frame edge), SAM3 stays
"insufficient" on a small object even from a clear GT view. Not yet eye-checked: the full
scenario sheets. Open choices: GT-sufficiency thresholds (a large couch at 1.3 m with trunc 0.28
counts as "poor" though it is clear by eye); whether to judge only when the agent is stationary
and facing the object.

### Uncertain set (2026-11-05)
`outputs/apexnav_online/active_approach_2026-11-05/scenarios_uncertain/scenarios.jsonl`: 517 starts, all eye-checked.
- **Real (169):** 216 of 444 requests are JEV-uncertain. 197 were re-created: 17 have no goal/not_goal truth,
  1 has no instance under the mask, 1 had no matching frame. Every re-created frame is the exact logged frame
  (corr 1.00). Of the 197, 20 were dropped by eye (13 wrong instance, 7 mesh artefact or no object) and 8 for
  conflicting labels. 11 kept starts have an ambiguous match (flagged). With the GT mask instead of the logged
  detector mask, JEV still flags 145 of the 169.
- **Synthetic (348):** 704 starts were scored and 352 were uncertain. 64 were dropped by eye (mostly object not
  visible, mesh artefacts, or an HM3D label that does not match the object). Groups hold 25–30 starts
  (target/oblique 25 and confuser/clear 27: the pool ran out). The set spans 29 scenes, but four of them hold
  42% of it, because groups filled in a seeded scene order.
`--judge jev` works in the runner (`runs/jev_check_2026-11-05`). The JEV container is stopped after use.

## jev_sam3 as the harness judge: review (2026-11-06)

Read: `perception/{select,jev,server,worker,client}.py` (uncommitted), ApexNav `perception_gate/gate.py`
(`_judge`, `_judge_views`, `_decide_three_way`) and the sync_jev_sam3 run logs (176 episodes so far). No
jev_sam3 code was changed.

Semantics the harness copies:
- **/verify (first request).** JEV names the outlined object. The frozen gate (`JevGate`: mean midrank
  percentile of H/log16 and min(p_target, max confuser p), threshold **0.5076**) says unsure, which
  means approach. Otherwise SAM3's raw top-1 on the masked frame decides, with no margin.
- **/verify_views (bank).**
  - JEV's gate runs on the *mean* distribution over the sent views.
  - Unforced (`early`) and unsure: insufficient.
  - Otherwise SAM3 runs `aggregate_views`: accept iff M >= `sam3_margin` and V >= 0.5.
  - The gate sends margin 0.0 on early and arrival requests, and 0.15 on failure (unreachable or
    aborted). So a JEV-sure early view and a forced arrival use the same SAM3 rule.
  - The views are ApexNav's `select_views` (<= 6, by quality, de-duplicated), and `input_elevation`
    is ignored.

Findings (in order of impact on the harness):
1. **Threshold mismatch in the scenario set.** The 517 "uncertain" starts were selected with
   `scripts/jev_server.py` / `active_approach_2026-11-05/jev_gate.json`, threshold **0.4664**. That
   is the 45% quantile over the *online* rows only. `perception.jev` (jev_sam3) uses **0.5076**, the
   45% quantile over all 444 rows. The references are identical. Re-scoring the stored JEV signals with
   the jev_sam3 gate makes **50 of 517 starts "sure"** (33 real, 17 synthetic), so jev_sam3 decides
   them at step 0. The harness treats "unsure" as what jev_sam3 says at run time, and the evaluation
   subset leaves those 50 out.
2. **Latent bias: SAM3 matching nothing on the single-frame path.**
   - `worker.classify` returns 0 for every category when no detection overlaps the mask (IoU < 0.3).
     `select.top1` then returns the first vocabulary entry. With `HOUSEHOLD` that is **"chair"**, so
     a chair target is *accepted* with no evidence, and any other target is rejected.
   - This affects `_jev_sam3` (and `_d1` / `input`). In the closed-loop run it happened 0 times in
     107 sure single-frame decisions. GT masks (the harness), occluded objects, or small masks make it
     more likely.
   - Fix (not applied): all-zero scores -> `wrong`, with label None, or `insufficient`.
   - The multi-view path is safe: `aggregate_views` drops all-zero views, and none left -> reject.
3. **Dilution in /verify_views.**
   - The JEV gate judges the mean distribution of every selected view. That includes the far,
     unsure trigger view and older bank views.
   - So one clear close view can stay "unsure" until it outweighs them, and fewer approaches resolve
     unforced than single-view sureness would suggest.
   - SAM3's aggregate, by contrast, down-weights small views (area / 0.10, clipped x 0.7).
   - This is design, not a bug, but it shapes what a planner must produce: several clear views, not
     one. The harness keeps ApexNav's `select_views` unchanged, so results transfer.
4. **Image and mask domain.** The gate references and threshold were fit on logged `input.jpg`
   (cv2 JPEG q95) with ApexNav detector masks. Live ApexNav and the harness send lossless PNG. The
   harness also uses GT masks: with them, the old gate flags 145 of the 169 real starts, against all
   169 with the logged masks. The mask source is the larger shift; the JPEG effect is unmeasured
   (probably small).
5. **Closed-loop config, not the module.** The sync_jev_sam3 run keeps `on_no_evidence: accept`
   (default). 3 clusters were accepted with no verification at all, which inflates FS slightly.
6. **Minor.**
   - `aggregate_views` labels the target only when M > 0, while acceptance is M >= 0. At M == 0 the
     verdict accepts but the label says otherwise (cosmetic).
   - Forced /verify_views still runs JEV on every view (about 1 s per view, logged only).
   - OVON targets have no confuser list, so their contention is 0, which biases the gate toward
     "sure" (not relevant for HM3D).

## Results: jev_sam3 judge, 112-start subset (2026-11-06)
Run: `~/codes/ApexNav/videos/active_approach/runs/jev_sam3_subset_2026-11-06/`, to be archived to
`/ICONAS/users/marfred/`.
- Settings: same server settings as sync_jev_sam3; pitch 0; the 30-degree action set.
- Subset: 112 starts, at most 5 per source x role x tag, all unsure under jev_sam3's gate.

**Eye check.** Every scenario with at least one wrong decision (42) was checked on the contact sheets.
`eye.tsv` drops 10 of them as X: the HM3D label is wrong or the object cannot be identified by eye.
- 4 flower arrangements in vases, labelled both plant and bouquet;
- a bar chair labelled stool, and a bar stool labelled chair;
- a loveseat labelled armchair;
- 3 unidentifiable "plants" (folded fabric, an ornament, a figurine).

That leaves 102 starts: 52 confusers and 50 targets.

| policy | wrong | FA | FR | resolved unforced | path (m) | stops | vs stay b/c (p) |
|---|---|---|---|---|---|---|---|
| stay (identity) | 17 | 8 | 9 | 0 | 0 | 0 | - |
| native (1.5 m then 0.75 m) | 13 | 8 | 5 | 15 | 2.0 | 0.94 | 10/6 (0.45) |
| random_ring | 16 | 5 | 11 | 10 | 3.9 | 1.29 | 7/6 (1.0) |
| fba_tour (M1+M3) | 15 | 5 | 10 | 15 | 2.0 | 1.30 | 7/5 (0.77) |
| svr_tour (M2a+M3) | 12 | 3 | 9 | 8 | 1.35 | 0.62 | 8/3 (0.23) |

(round-2 `s2`/`s3` were stopped after 24/35 starts: 2 and 6 wrong, against 3 and 5 for stay on the same
starts.)

**Findings.**
1. **Almost every decision is the forced one.** Only 8-15 of 102 unsure starts become sure at a stop.
   The rest are settled by the forced SAM3 aggregate. JEV's gate on the *mean* distribution over up to
   6 bank views stays unsure because the far, unsure trigger view is always in the bank.
   - The trigger frame alone already gets 85/102 right when forced (stay).
   - No planner differs significantly from stay or random_ring (McNemar p >= 0.23).
2. **The vote dilution flips correct views into rejects.** Example:
   `fba_tour/4ok3usBNeis_379_toilet_occluded`.
   - The arrival view shows the toilet clearly (4% area; JEV top-1 "toilet").
   - The forced `/verify_views` over steps [25, 6, 0] gave M = +0.54 but V = 1/3. SAM3 labels were
     toilet, bed and chair; the other two are the occluded start view and a door-edge view. Result:
     reject.
   - The margin is area-weighted, but the vote is unweighted, so tiny far views veto.
   - Changing this (an area-weighted vote, or dropping small views) is a decision-module change and
     needs the user's OK.
3. **Planner failure modes, by eye.**
   - native ends too close or truncated: 75 runs tagged truncated, e.g. stools at 0.8 m, a TV at 0.8 m.
   - random_ring loses the object or runs out of budget in about half its runs.
   - fba_tour gets good, whole-object views (the toilet, couch and plant arrivals are clearly
     identifiable), but the forced aggregate still fails as in 2.
   - svr_tour moves least (1.35 m) and has the fewest FA. Its tour seldom finds a second view
     (`no_candidate` in 93 runs).
4. **Most remaining FAs are genuine look-alikes** that SAM3 accepts on clear views: wingback armchairs
   as couch, a bench / window seat as couch, a kids' stool as chair, a folder rack as bed. Better views
   do not fix these. The decision module is the bottleneck.
