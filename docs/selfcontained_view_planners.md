# Self-contained view-aware approach planners (design, 2026-11-07)

Scope: the approach phase of the verification module (budget 40 actions) on top of *any* ObjectNav
agent. The earlier planners (`docs/active_approach_planners.md`: M1/M2/M3, S2/S3, round 3) use the
habitat navmesh, geodesic costs, `snap_point`, `try_step_no_sliding` and a free 12-turn seed panorama.
This doc drops all of them. It covers two families: ports of the existing planners onto the self-built
map (family P, via `local_nav.LocalPathfinder`), and new self-contained view-aware designs (D1-D9). It is
design only. The harness is `~/codes/ApexNav/active_approach/`,
and the decision flow is `decide.py` with the 2026-11-07 config.

**The honest prior.** On the 312 clean round-3 runs, Seek plus the flow gets 291-296 right under
LOSO CV. The 13-19 remaining errors are mostly perception: unreadable plant meshes, bouquet vs plant,
leather armchair vs couch. About 2-3 are "no usable view" planning failures. A view-aware planner
therefore has a ceiling of about **3-6 corrections out of 312** against a naive approach that already
reaches usable views. Without a navmesh, the naive baseline itself gets weaker: it hits obstacles,
goes too close and gets stuck. That is where geometry has room to help.

Each design below names the failure mode it targets and makes a falsifiable prediction against N1.
A design that cannot move the wrong count should be judged on actions, and on reaching a sufficient
view under the GT proxy (section 4).

---

## 1. Shared self-contained infrastructure: `active_approach/local_nav.py` (exists) plus gaps

**Inputs the planner may read:**
- RGB-D: 640x480, hfov 79, vfov 63.4, camera height 0.88 m, pitch 0;
- the odometry pose;
- the target mask: the trigger mask, then a tracker (the harness uses the GT semantic mask as a
  stand-in);
- collision feedback: a blocked `move_forward` does not move the agent (`allow_sliding` false);
- the per-frame JEV/SAM3 outputs, which the flow computes anyway.

**Already built** (`run.py --selfcontained`):
- **`LocalMap`.** A 10 cm grid fed only by the agent's own depth, via `geom.HeightMap`.
  - The map is centred on the start pose; the floor is the agent's own y; there is no seed panorama.
  - A cell is occupied if >= 2 of its 5 cm sub-cells have a top in 0.12-1.6 m, or it is a bump mark,
    or an object cell. Free means seen and not occupied. Everything else is unknown.
- **`plan`.** 8-connected A*. Obstacles are inflated by 0.25 m; an unknown cell costs 3 against 1 for
  a free cell.
- **`LocalController`.** The `Controller` interface: a 0.5 m look-ahead and 25 deg hysteresis,
  replanning every step and marking bumps.
- **`LocalPathfinder`.** A habitat-PathFinder shim (`snap_point`, `find_path`, `is_navigable`,
  `try_step_no_sliding`, `distance_to_closest_obstacle`, `nav_grid`) answered from `LocalMap`. With
  it, the navmesh-era planners run unchanged (family P, section 2b).

**Gaps found reading the code** (suggested changes; none applied):
1. **GT leak.** In `run.py:225`, `center` falls back to `scenario["object_center"]` when the cloud is
   empty. In self-contained mode, use the trigger-mask bearing at an assumed range, or the last
   cloud centre.
   Also add a guard that raises if a policy reads `sim.pathfinder`, `ctx["surface"]` or `ctx["views"]`
   (except the oracle).
2. **"Unknown" now means navigable, which silently removes two occlusion terms.**
   - `Predictor._occlusion` counts an unknown cell as an occluder (half weight) only if
     `hm.navigable` is false.
   - `svpto.Visibility.esdf_unknown` uses `~seen & ~nav`.
   - Under the shim, unknown cells have a finite cost, so both terms are always 0.
   - So `occ_hat` and S2's FM go optimistic exactly where the object's surroundings are unseen.
   - Fix: give both their own unknown rule. One option is unknown at 0.5 weight when the ray height
     is < 1.0 m, whatever the navigability. Another is an "unknown behind a hit" shadow: cells in the
     depth shadow of an occupied cell.
3. **An unreachable viewpoint ends the run.** `LocalController` raises `no path`, so the run ends
   `no_path` with a forced decision.
   - Goals chosen in unknown space often turn out to be inside furniture once seen.
   - Fix: on no path, mark the goal infeasible and call `policy.next_viewpoint` again, at most 3
     retries, and log `retarget`.
4. **The blind zone.** At pitch 0 the floor is seen only beyond **1.43 m**, and a 0.3 m obstacle only
   beyond 0.94 m. So the area in front of the start is unknown and costs 3 per cell, which makes paths
   bend around the robot's own front.
   - Fix: a hybrid unknown cost. Cells within 1.5 m of a visited pose cost 1.5; cells in the shadow of
     a hit cost 5; all other unknown cells cost 3.
   - Plus an optional look-around: `scan` in {0, 2, 11} turns, all counted.
5. **Thin obstacles.** The ">= 2 sub-cells in one frame-max" rule drops chair legs and plant stems.
   - Fix: accumulate a hit count over frames and call a cell occupied at >= 2 hits from any frames.
   - Also add a reactive guard before every F: depth points with height in [0.12, 1.6] inside the
     0.40 x 0.45 m box ahead mean mark the cells and do not step.
6. **No clearing.** `HeightMap.top` is a running max, so one spurious hit stays forever.
   - This is harmless with perfect odometry, but it matters for the noise ablation.
   - Fix: use log-odds on the 10 cm grid (hit +0.85, ray miss -0.4) and keep bumps sticky.
7. **Turn cost.** The grid A* ignores turns. `geom.action_cost` adds the turns of the returned
   polyline, but the path itself is not turn-optimal.
   - Fix (optional): a lattice A* over states (x, z, k), with F/L/R each costing 1, used for the
     *final* 1 m and for costing candidates. That is where turn count dominates.
8. **No MPC driver.** `MPCDriver` (s3) is refused in self-contained mode. `svpto.step` already goes
   through `try_step_no_sliding`, so pointing it at the shim enables S3. The shim's `try_step` checks
   only the endpoint; it should check the swept 0.25 m segment at 5 cm.

**New shared helpers for the D-designs** (small additions next to `local_nav`):
- **Object track.**
  - Erode the mask by 2 px; reject depth outliers (|d - median| <= max(0.4, 2.5 MAD)); keep the 3D
    component nearest the last centre.
  - Fuse a frame only if the cloud-to-mask reprojection IoU is >= 0.3.
  - `geom.ObjectModel` with `round3.phantom_model`, reused rather than copied.
- **Image cues per frame:**
  - mask area, bbox, `touch[t/b/l/r]`, offset;
  - the **occluder fraction**: hull pixels that are not mask and have depth < the object's median
    depth - 0.15 m, divided by the hull pixels of the reprojected cloud (dilated 15 px). Split it into
    `occ_L` and `occ_R`.
- **Seen-surface normals:** horizontal normals from depth cross-products in the mask, binned into 12
  bearings (`seen_normals`), and the bearings viewed so far (`seen_bearings`).
- **Tracker ablation T1:** mask = the reprojected cloud, kept where it is depth-consistent (0.1 m).
  This measures the dependence on the GT-mask stand-in.
- **Robustness knobs** (proxy only): odometry noise of 2 cm / 1 deg per action, 1% depth noise, T1.
  A design that wins only without them does not transfer.
- **Logging** (on top of steps/frames.jsonl):
  - `collided`, `retarget`, `scan`, the unknown fraction on the plan;
  - grid png at the stops;
  - `design_state`;
  - and, for analysis only, the grid's false-free rate against the navmesh. That rate is the key
    infrastructure check.

---

## 2. Naive baselines

The baselines are run with the same infrastructure and the same flow. Every frame where the target
is in view is a candidate observation for decide.py.

**N0 `native_sc`** is a port of ApexNav's rule: 1.5 m standoff, then 0.75 m, facing. Goals are on the
straight line from c, and the lattice A* replaces `snap_point`. If the goal cell is blocked, slide the
goal along the line toward the agent to the first free cell. N0 is the reference that connects to the
logged native runs. It is expected to reproduce native's truncation at 0.75 m.

**N1 `closer`**, the strong naive baseline and **the primary comparator**:
1. Turn to face c. A* to the straight-line goal at r_stop = clip(d_fit(bearing_now), 0.9, 2.2) m.
   d_fit uses phantoms, so the whole object fits in the frame.
2. Recompute r_stop and the goal every 4 steps as the cloud grows. The goal only moves along the
   current line.
3. At the goal, turn to face it. Then stop; the flow makes its forced call if it has not triggered.
4. If the goal is blocked, take the nearest free cell on the line farther out. If there is no plan,
   approach as far as possible on the line.

N1 has no view choice at all: one line, one stop. It is what a careful engineer writes in a day.

**N2 `closer_recenter`** = N1 plus two cheap habits:
- (a) when the mask touches the left or right border and the offset is > 15 deg, turn first;
- (b) after the stop, one lattice step around: turn +-90, F x n with n = round(r_stop * tan(30 deg) /
  0.25), turn back, a 30 deg arc for a second view. The side is the one with more free cells; it
  costs about 8 actions.

N2 is what "more views" buys without any view reasoning. If a view-aware design cannot beat N2, its
reasoning adds nothing over "take a second view".

---

## 2b. Family P: the existing planners, ported to the self-contained map

Cheapest of all: the shim already answers their pathfinder calls. What each depends on, and its
replacement:

| piece | navmesh dependency | self-contained replacement (status) |
|---|---|---|
| `geom.lattice_candidates` | `snap_point` (tolerance 0.15) | the shim's `snap_point` on the inflated local grid (done). Unknown counts as navigable, so pair it with gap 3 (retarget) |
| `geom.action_cost`, `policies.geodesic` | geodesic + polyline turns | the shim's `find_path` (cost-weighted grid A*, geometric length) (done). Gap 7 for turn-exact costs |
| `geom.Predictor._occlusion` | `hm.navigable` for unknown cells | needs gap 2, otherwise occ_hat = seen-only |
| `geom.HeightMap` | seed panorama, origin at the GT centre | own frames only, start-centred (done) |
| `svpto.Visibility` (S2 FM/EC) | `navigable_grid`, ESDF of unknown & non-navigable | `nav_grid` (done). Needs gap 2 for `esdf_unknown` |
| `svpto.step` (S2 lattice A*, S3 MPC) | `try_step_no_sliding` | shim (endpoint only; gap 8: swept check). MPCDriver refused (gap 8) |
| `policies.Controller` | greedy geodesic follower, `try_step` | `LocalController` (done) |
| `Native`, `Seek`'s native hand-off | `snap_point` | shim (done) = **N0** |
| `Tour3`, `Seek`, `SeekConfident` wrappers | only through the base planner | unchanged |

- **P1 `fba_tour3_seek` (self-contained).**
  - Prediction: the most robust port, because it needs only d_fit, the line and the ring. Wrong is
    within +-1 of its navmesh run, actions +10-20%.
  - Failure: ring candidates in unknown space lead to retargets.
- **P2 `svr_tour3_seek` (sc).** It depends on `occ_hat`.
  - Prediction: without gap 2 it degrades toward FBA, choosing occluded bearings. With gap 2 it is
    within +-1 of its navmesh run.
  - Run it both ways: that is the test of gap 2.
- **P3 `s2_tour3_seek` (sc).**
  - It depends most on map completeness (FM over the whole cone, valid poses).
  - Prediction: the largest drop. `no valid pose` leads to the M1 fallback in > 50% of runs, so it
    converges to P1.
  - Keep it only if it beats P1 on stage A.
- **P4 Seek / SeekConfident wrappers** apply to any P or D policy. Seek (the native hand-off when
  < 2 views) is the default wrapper for every design, so all designs share the same "no usable view"
  rescue.

Navmesh vs self-contained, paired on the same starts, is itself a result: **the cost of
self-containment** per planner. Logged navmesh runs exist for fba/svr/s2_tour3 (112 starts) and their
Seek re-runs.

## 3. View-aware designs (cheap -> ambitious)

The prediction format is: on subset X, metric Y changes by Z against N1, and stays within +-2 on the
other tags. "Wrong" means FA + FR under decide.py with its frozen config; "reach" is the GT proxy
(section 4.2).

### D1. Image-reflex closer (IRC): no 3D model, image cues only
- **Inputs:** the per-frame mask cues (area, touch, offset, occ_L/R).
- **Algorithm** (per step; the first rule that fires wins):
  1. `px < 50`: the object is lost. Turn toward the last known bearing; after 3 turns, use the
     cloud centre.
  2. offset > 20 deg: turn toward the object.
  3. `touch_top` or `touch_bottom`, and area >= 4%: **stop advancing**. Never step closer again,
     even if d_fit says so.
  4. `touch_left/right` with area < 25%: turn toward that side (the offset is wrong, not the
     distance).
  5. occ_L + occ_R > 0.3: hand off to D3's sidestep, once.
  6. area < A_stop (6%): step forward via A* on the line.
  7. Otherwise, stop (arrived).
- **What it fixes:** N1's d_fit is wrong when the first view is clipped, top or side. The phantom
  prior may then be too small (bar chairs, tall plants) or too large. IRC uses what the frame
  actually shows. This targets native's "too close / truncated" failure (75 truncated finals in
  round 2).
- **Cost:** the same as N1, about 6-14 actions.
- **Sweep:** A_stop {3, 6, 10%}; the rule-3 area {2, 4%}.
- **Prediction:** on `truncated` + `far` starts, the GT trunc_frac >= 0.3 rate at counted views drops
  by >= 40% against N1, and reach@1 rises by >= 5 pts. Wrong is unchanged (+-2). If the truncation
  rate drops but wrong does not, truncation was not causing errors, which is evidence against all
  view-aware work.

### D2. Frame-fit servo with lattice-aligned arrival (FFS)
- **Inputs:** the object model with phantoms, and the image cues.
- **Algorithm:**
  - Target fill: choose r so that the predicted bbox height is h_fill x 480 px and the width is
    <= 0.85 x 640. The vertical cut must be <= kappa.
    `r* = max(r_h(h_fill), r_w(0.85), d_fit)`.
  - The goal is the **lattice-ring pose** (p = c - r u(psi_k), psi_k a lattice yaw: arrival heading points exactly at c) at r* nearest to the current bearing (+-15 deg), so
    the arrival heading points exactly at c.
  - Servo the last 1 m on measured cues: if the bbox height / 480 >= h_fill or the mask touches the
    top or bottom, stop. If it is still < 0.6 h_fill at the goal, step forward once more (at most 2).
- **What it fixes:**
  - the facing error of up to 15 deg on N1's straight line, which side-clips wide objects (couch,
    bed, tv);
  - over- and undershoot from a wrong d_fit.
- **Cost:** N1 + 0-3 actions.
- **Sweep:** h_fill {0.35, 0.5, 0.65}; kappa {0, 0.25}.
- **Prediction:** on couch/bed/tv starts, the side-clip rate at the stop falls by >= 50% against N1.
  Wrong falls by >= 2 on the 112 subset, or the design is dropped as a refinement of N1.

### D3. Occlusion-escape sidestep (OES)
- **Inputs:** occ_L and occ_R (section 1), the grid, and the object model.
- **Trigger:** occ = occ_L + occ_R > 0.25 on a judgeable frame, or the GT-free check that the mask
  area grows by < 1.2x per metre of approach.
- **Algorithm.** Side s = argmin(occ_L, occ_R); toward the less occluded side the occluder ends
  sooner. Use the grid to tie-break: the side whose ring candidates have fewer occupied cells on the
  sight line.
  - Candidate poses: the lattice ring at r in {d_fit, d_fit + 0.5}, bearings psi_now +- {30, 60} deg
    on side s.
  - Score: the sight line c->p must be free on the grid (3 lines: centre and both extremes, ignoring
    cells within 0.15 m of the footprint), minus 0.02 x A*-actions.
  - Execute the A* plan, which is a turn-forward-turn when the side is open.
  - If no candidate on side s is free, try the other side once, then fall back to N1.
  - Fire at most twice.
- **What it fixes:** the `occluded` tag (96 starts, the largest). N1 drives straight into the occluder
  and keeps the same occluded bearing.
- **Cost:** 6-14 extra actions.
- **Sweep:** the trigger {0.15, 0.25, 0.4}; the bearing offsets {30, 60}; the side rule (image vs grid).
- **Prediction:** on `occluded` starts, the GT occ_frac at the best counted view is <= 0.2 in >= 15
  pts more runs than N1, and occluded-tag FR falls by >= 2. No change elsewhere, because the trigger
  rarely fires there; measure the false trigger rate on `clear`.

### D4. Facing-normal orbit (FNO): the visible-surface goal
- **Inputs:** `seen_normals`, the PCA footprint, and the category prior of a "readable side":
  - front for chair, couch, toilet and tv;
  - foot or long side for bed;
  - any side for plant.
- **Algorithm.**
  - Front estimate: the dominant seen normal bin if its weight is > 40% of the seen weight.
    Otherwise use the long-axis normal of the footprint (couch, bed, tv) closest to the current
    bearing.
  - Goal: the lattice-ring pose at d_fit with bearing nearest the front normal, within +-30 deg.
  - Accept it only if A* cost <= 20 and the sight line is free on the grid. Otherwise take the
    best-facing reachable bearing at cost <= 20, else N1.
  - Executed views are judged on arrival; frames en route still feed the flow.
- **What it fixes:** the `oblique` tag: a tv seen edge-on, a couch end-on (it reads as an armchair or
  bench), a toilet from behind.
- **Cost:** 8-20 actions.
- **Sweep:** the tolerance {15, 30 deg}; the cost cap {12, 20}; the front rule (normals vs PCA).
- **Prediction:** on `oblique` starts, the GT facing_cos at the best view is > 0.6 in >= 20 pts more
  runs than N1, and wrong falls by >= 2 on the oblique tag (n = 58). Couch/armchair FAs: no change
  expected, because those are look-alikes in face-on views.

### D5. Two-view disparity (TVD): a second, independent view >= 45 deg around
- **Inputs:** the grid, the object model, and `seen_bearings`.
- **Algorithm.**
  1. Run N1 (or D2) to the first stop.
  2. If the flow has not triggered, pick the second view: the lattice-ring pose at d_fit with
     |bearing - every seen bearing| >= 45 deg. Choose by min A*-cost and a free sight line, with cost
     <= remaining - 2.
  3. Arc execution: turn 90 toward the side, F x n, then turn to face. On the lattice this is about
     2 + 0.8 r + 3 actions for 45 deg at r = 1.5 m, so about 7-10.
  4. Stop after the second view, or a third if the flow is still untriggered and budget remains.
- **What it fixes:**
  - the flow's trigger needs consistent evidence; a duplicate view adds little under the diversity
    gate (GATE all also counts duplicates);
  - look-alikes whose shape cue is depth or side profile: armchair vs couch (seat count and width are
    visible from the side), tv vs picture (depth), toilet vs sink.
- **Cost:** +8-12 actions over N1.
- **Sweep:** the separation {45, 60, 90}; K {2, 3}.
- **Prediction:** against **N2** (the 30 deg arc), not just N1: confuser FA falls by >= 2 on
  couch/chair/tv confusers. If TVD matches N2, novelty beyond 30 deg is worthless. Actions are +4
  over N2.

### D6. Confusion-pair view prior (CPV): perception-driven, VLM-free planning
- **Inputs:** the per-frame JEV naming distribution, already computed for the flow (0 extra calls),
  and the object model.
- **Algorithm.**
  - Running pair: (target t, strongest rival c*) from the belief's mean over the counted frames.
  - A fixed table maps each pair to a discriminative view type (written before any data is seen):

    | pair | view type | rule |
    |---|---|---|
    | couch / armchair, chair / armchair | wide-front | bearing on the long-axis normal, r so the width fills 0.85 of the image (the seat count shows) |
    | couch / bench, bed / couch | full-length | the far end of the long axis, 30-45 deg off-axis (back and cushions visible) |
    | tv / picture, mirror, window | grazing | 40-60 deg off the screen normal at 1.2-1.5 m (bezel depth, stand) |
    | plant / vase, flowers | full-height | r = d_fit with kappa 0 (pot plus foliage in frame), any bearing |
    | toilet / sink, bathtub | side | 60-90 deg off the front normal (bowl profile, tank) |
    | chair / stool, table | front-high | the backrest side: front normal, r with kappa 0 |

  - The goal is the reachable lattice-ring pose matching the view type (cost cap 20).
  - Re-evaluate the pair after each counted frame. Switch the goal only if c* changes and the old
    goal is > 6 actions away.
  - Once the flow triggers, stop (the flow commits).
- **What it fixes:** the dominant *remaining* error class, look-alikes. It is the only design aimed at
  it.
- **Cost:** 8-20 actions.
- **Sweep:** table on/off per pair (one-pair ablations); the cost cap.
- **Prediction:** on look-alike scenarios (confusers, plus targets whose start rival is a look-alike),
  wrong falls by >= 3 against N2. The prediction is falsified if any pair's dedicated view is *less*
  accurate than N1's stop view. Check that per pair offline first: the logged frames already cover
  many bearings, so the existing data can test the table before any run (section 4.1).

### D7. Clear-cone standoff (CCS): obstacle-aware stop, SVPTO-lite on the own grid
- **Inputs:** the grid and the object model.
- **Algorithm.**
  - Candidates: lattice-ring poses at r in [d_fit - 0.25, d_fit + 1.25], every bearing.
  - Cost: `A*(p) + w_cone x FM(p) + w_unk x unknown_frac(cone)`.
    - FM is the SVPTO eq. 3 hinge with the cone half-width from the object's angular size plus
      5 deg, read on the own inflated grid, not the navmesh ESDF.
    - unknown_frac is the share of unknown cells in the cone.
  - Pick the min. Replan whenever the cone cells' state changes.
  - The stop distance shrinks in clutter: if the cone at r is blocked, prefer r + 0.25 on the same
    bearing over walking around.
- **What it fixes:**
  - N1/N0 in clutter: collisions, stuck runs, and stops behind chairs or a table edge, where the
    occluder sits between robot and object. This is the obstacle failure of native.
  - It also covers the no-navmesh-specific failure, a plan into unknown space that turns out blocked.
- **Cost:** about N1 (it usually stops earlier).
- **Sweep:** w_cone {0.1, 0.5}, w_unk {0, 2}, and the unknown mode.
- **Prediction:** collisions per run fall by >= 50% and stuck/budget ends by >= 50% against N1, at no
  more actions. Wrong is unchanged, or improves on `occluded` + `poor` by >= 2.

### D8. Object-surface information gain (OIG): NBV on a local voxel map (ambitious)
- **Inputs:** a local 3D voxel grid of 5 cm over the object bbox + 0.4 m, with states object /
  free / unknown, filled by depth ray casting (the same depth rays as the local map, in 3D, only inside the bbox).
- **Algorithm.**
  - Frontier: unknown voxels adjacent to object voxels, plus phantom extents as unknown.
  - Candidates: lattice-ring poses, 3 radii x 12 bearings.
  - Gain(p): cast rays from p through the image grid (stride 16) and count the frontier voxels hit
    before an occupied voxel, weighted by the part prior (top half x1.5 for chair/plant: backrest and
    foliage).
  - Score `gain(p)/G0 - A*(p)/A0`, with A0 = 30.
  - Stop when the best gain is < 15% of the initial frontier, or when the flow triggers.
- **What it fixes:** the `poor` and `occluded` tags, and the partially seen "unreadable" meshes (if
  more surface helps them at all).
- **Cost:** 10-30 actions; the most compute, about 30 ms per candidate on CPU.
- **Sweep:** A0 {15, 30, 60}; the stop fraction; the part prior on/off.
- **Prediction:** reach@2 is +5 pts over D3+D4 combined on occluded + poor, else drop it. Wrong is
  expected to be unchanged: completeness is not identity (see the honest prior).

### D9. Naming-gradient probing (NGP): the cheap perception-feedback controller (control experiment)
- **Inputs:** the per-frame JEV distribution.
- **Algorithm.**
  - At each stop, compare the naming entropy H of the last two counted frames.
  - If H drops by > 0.1 after the last move, repeat the same move type (F, or a +-30 arc).
  - Otherwise try the opposite or the orthogonal move: arc left, arc right, closer, in that order.
  - At most 4 probes.
- **Why include it:** JevPilot (the VLM chose moves) was cheaper but not more accurate. NGP tests
  whether *any* perception-feedback steering helps once the VLM is out of the loop.
- **Prediction:** equal to N2 at +-1 wrong. If it beats N2, perception feedback adds information
  that geometry lacks.

---

## 4. Evaluation protocol

### 4.1 Stage 0: offline, existing logs (CPU, minutes)
- **Grid and tracker sanity.** Replay the logged trajectories' poses through `LocalGrid` on CPU. The
  frames are not stored, so re-render the depth in the harness with `--judge gt`; this is cheap.
  Report:
  - the false-free rate on navmesh-blocked cells, per scenario;
  - the T1 tracker IoU against the GT mask.
- **CPV table check (D6).** From the ~3,700 logged frames (`frames.jsonl` in jev_sam3_r3*, seek and
  seekc), bin each frame by view type relative to the GT footprint axes: angle to the front normal,
  fill, kappa.
  - Per look-alike pair, compare SAM3 top-1 accuracy and JEV contention across bins.
  - The table survives only for pairs where its view-type bin beats the "N1-like" bin (frontal-ish,
    fit distance) by >= 10 pts with n >= 15.
  - Drop the other rows before any run. This is the cheapest falsification in the doc.
- **Perception proxy model.** Fit the per-frame `P(frame correct)` with an L2 logistic, scene-grouped
  5-fold.
  - Label: SAM3 top-1 matches the role (the eye-corrected truth, `decide.EYE_ROLE`) and JEV top-1 is
    consistent.
  - Features: GT log-area, trunc, occ, facing_cos, touches_border, logit(u0).
  - Report AUC and reliability. Use it in stage A only if AUC >= 0.70.

### 4.2 Stage A: GT proxy (habitat only, no VLM; `--judge gt --selfcontained`)
- **Scenarios.** All 517 starts minus `EYE_AMBIGUOUS` and the eye drops, stratified by role x tag.
  The headline is per tag.
- **Resources.**
  - Habitat renders on a GPU context (~1 GB). The cap is ONE GPU total across all jobs: run stage A only on the GPU already held by the user's job (or when none runs), check `nvidia-smi`,
    one process.
  - At most 4 CPU workers; RAM < 8 GB.
  - Frames are not saved except at stops (home quota).
- **Per-run metrics:**
  - **reach@1 / reach@2:** at least 1 (2 diverse, by the decide gate 0.5 m / 20 deg) counted frames
    that are GT-sufficient (vis >= 0.6, trunc <= 0.15, area >= 1%);
  - **p_best:** the proxy model's max P(correct) over the counted frames, and a noisy-OR over the best 2;
  - actions to reach@1 and total actions;
  - collisions, stuck, budget ends, lost (px < 50 at the end), final too_close (< 0.8 m), final trunc
    >= 0.3;
  - the grid false-free hits: a blocked step into a cell the grid called free.
- **Tests:**
  - paired, per scenario, against N1: McNemar on reach@2; Wilcoxon on p_best and actions;
  - Holm correction over the designs.
- **Gate to stage B:** at least one of these, with no tag losing > 3 pts reach@2 and p < 0.05 (Holm)
  on its own primary:
  - (a) the design's predicted tag improves reach@2 by >= 5 pts;
  - (b) the same reach with >= 20% fewer actions;
  - (c) collisions or stuck runs are halved.
- **Robustness:** re-run the gated designs with the robustness knobs of section 1. Wins that vanish are reported, not
  promoted.

### 4.3 Stage B: the real decision flow (GPU, jev_sam3)
- **Set.** `subset120.txt` (112 starts, up to 5 per source x role x tag), plus the 13 known
  remaining-error starts (TEE321, yr17_353, epi175, ziup tv, bxs78, TEE284, ...) as a reported
  sub-panel. Those have the most planning headroom.
- **Run.**
  - `run.py --selfcontained --judge jev_sam3 --log-frames --full`, policies N0, N1, N2, P1 (and P2/P3 if
    they pass stage A) plus the gated designs;
  - **one GPU total**: the perception server and habitat share it, one job, queued behind any running
    job; the server is stopped right after.
  - `first_jev_sam3.jsonl` is shared (the start frame is the same for everyone).
- **Decision.** `decide.py` with the **frozen 2026-11-07 config**:
  - GATE all, VIEWS nearest, TRIGGER consistent, P* 0.8, MARGIN -0.1, REL_AREA 0.1, no veto.
  - It is not re-tuned per policy, so there is no in-sample advantage.
  - Also report the LOSO-CV re-tuned numbers.
- **Metrics:**
  - wrong = FA + FR (undecided = reject), per tag and role;
  - actions to commit; forced-commit rate; collisions;
  - the no-usable-view count.
- **Tests:**
  - exact McNemar against N1 (primary), N2 and N0, plus `stay` for context. Holm over the designs.
  - Family P is also compared, paired, against its own logged navmesh run: this measures the cost of
    self-containment.
  - Power note: at about 6% error, p < 0.05 needs roughly a 9:1 split over >= 10 discordant pairs. A
    planner with a 2-3 error headroom cannot reach significance on 112. Report discordant pairs and
    actions, and promote on (wrong <= N1) AND (actions or no-usable-view clearly better).
- **Eye check.** Inspect every wrong case of every policy, and every discordant pair against N1, on
  contact sheets (crops, plus a GT geometry overlay). Labels must be S/FS/FR/X, never "?".
  Benchmark labels are noisy: report eye-verified numbers only.

### 4.4 Stage C
Run the top 1-2 designs plus N1 on all clean starts, with the same protocol. Then archive the runs to
`/ICONAS/users/marfred/` as one tar.

---

## 5. Ranked shortlist (implement in this order)

0. **The `local_nav` gaps 1-3 (GT leak, unknown-occlusion semantics, retarget), plus N0/N1/N2.** This
   is the real new risk: without a navmesh, N1 may lose more than any view reasoning gains. Stage A on
   N0 against the logged native measures the cost of going self-contained.
1. **Family P, ports P1 and P2** (`fba_tour3_seek`, `svr_tour3_seek` with `--selfcontained`). They
   need almost no new code: the shim exists. They answer directly whether the round-3 planners survive
   without the navmesh. P3 (s2) runs on stage A only.
2. **D7 clear-cone standoff.** It fixes the failure the user saw (native into obstacles, stuck), and
   that failure gets *worse* without a navmesh. It is cheap, and the most likely to show a
   significant effect (collisions and stuck runs).
3. **D1 image-reflex closer**, with D2's lattice arrival folded in as one design: IRC rules plus a
   lattice-ring goal. It costs little and directly attacks native's truncation. If it does not beat
   N1, truncation is not the problem, and most view-aware work is moot.
4. **D3 occlusion escape.** It covers the largest tag (occluded, 96 starts) and has a clear mechanism
   and a narrow trigger, so it should not hurt elsewhere.
5. **D6 confusion-pair views.** It is the only design aimed at the remaining look-alike errors. Gate
   it on the Stage 0 table check, which costs CPU minutes; implement only the rows that survive.
6. **D5 two-view disparity**, as the honest test of "more views" against N2. Run it only if D6's
   table check fails (D6 subsumes it for look-alikes).

Deferred:
- D4: the oblique tag is small; fold it into D6's wide-front and grazing rows.
- D8: high cost; predicted no gain in wrong decisions.
- D9: a control; run it only if D6 wins, to separate perception feedback from geometry.

Expected outcome, stated now: D7 and D1 beat N1 on collisions, truncation and actions; D3 and D6 have
the best chance of a +2-4 wrong-decision gain on their tags. No design is expected to beat N2 by a
significant margin on all 112 starts. If none passes the Stage A gate, conclude that a careful naive
approach is enough, and put the effort into perception (look-alikes, unreadable meshes).

---

## 6. Results (2026-11-07)

Code (ApexNav repo, uncommitted): `active_approach/local_nav.py` (LocalMap, LocalPathfinder shim, LocalController,
`track_mask`), `active_approach/selfcontained.py` (closer, closer_rc, tvd, irc, oes, ccs), `run.py --selfcontained
[--tracker reproj] [--name-usable]`, judge `gt_seek`, `active_approach/proxy.py`, `active_approach/test_local_nav.py`.
Scripts: `videos/active_approach/run_sc_proxy.sh`, `run_sc_jev.sh`, `tmp/sc/{iter.sh,iter_t1.sh,score.py,pair.py,traj_stats.py}`.

### 6.1 Infrastructure iteration (Stage A proxy, native + closer, 112-start subset)
Proxy `reach` = some frame that decide.py could count (facing <= 20 deg, >= 400 px) is GT-sufficient.
The navmesh references are native 80 and fba_tour3_seek 86.

| version | change | native | closer | bumps (native) |
|---|---|---|---|---|
| v1 | first shim (A* line follower) | 43 | 53 | 47 |
| v1b | follower: lattice look-ahead (12 headings x <= 6 steps + Dijkstra cost-to-go); fixed a "stay" tie | 43 | 53 | - |
| v2 | goals snapped to the nearest *reachable* cell; viewpoint leash (path > 2x straight + 1 m or 2x estimate) | 51 | 60 | 203 |
| v3 | hard 0.20 m inflation, a bump marks 3 cells across the heading (the soft own-disc caused repeat bumps) | 55 | 61 | 97 |
| v4 | give-up -> face the object; unknown cost 3 -> 1.3 | 57 | 62 | 104 |
| v5 | **view-keeping**: +0.5 per simulated step that leaves the object > 30 deg off the heading | 66 | 68 | 64 |
| v6/v7 | view cost 1 / 2 / 4: 73 / 75 / 75; unknown cost 1.0 (+1) | 76 | 74 | 31 |

The missing piece was not the map but keeping the object in view while driving. The navmesh line-follower did
that implicitly: 76% of its frames are usable, against 45% for v4. Final defaults: unknown 1.0, view cost 2.0.

### 6.2 All planners, final infrastructure (v8) and the GT-free tracker
| policy | reach | reach2 | actions | bumps | stuck/budget | reach, tracker T1 |
|---|---|---|---|---|---|---|
| nav: native (navmesh) | 80 | 49 | 10.9 | 0 | 7 | - |
| nav: fba_tour3_seek (navmesh) | 86 | 58 | 21.5 | 0 | 11 | - |
| native (sc) | 76 | 44 | 10.8 | 31 | 1 | 74 |
| **closer** (N1) | 74 | 38 | 8.0 | 17 | 0 | 72 |
| **closer_rc** (N2 = N1 + 30 deg arc) | 86 | 57 | 24.1 | 100 | 8 | 85 |
| tvd (D5, >= 45 deg) | 85 | 55 | 26.1 | 127 | 10 | - |
| fba_tour3_seek (sc port, P1) | 88 | 50 | 25.8 | 124 | 20 | 87 |
| svr_tour3_seek (sc port, P2) | 83 | 48 | 24.0 | 127 | 22 | - |
| oes (D3) | 81 | 50 | 19.8 | 113 | 15 | - |
| ccs (D7) | 61 | 23 | 7.6 | 16 | 1 | - |
| irc (D1) | 56 | 25 | 13.0 | 2 | 1 | - |
(s2_tour3_seek: ~10 min per run on the shim; dropped after 27 runs, as predicted it falls back to M1.)

Findings:
- **Self-containment costs about nothing now.** sc fba 88 vs navmesh 86; sc native 76 vs 80. Removing the GT mask
  (T1 reprojection tracker after the trigger frame) costs 1-2 more.
- **A second view is what helps; the view reasoning adds little.** closer -> closer_rc +12 reach, +29 reach2, at
  +16 actions. fba_tour3_seek (geometric NBV) = closer_rc within noise (88 vs 86). tvd (>= 45 deg) is no better
  than 30 deg.
- **ccs and irc stop too early.** On the GT proxy, stopping where the view "should" fit loses reach. irc stops at
  the first top/bottom border touch, which happens at ~1.4 m for every floor object at pitch 0.
- **oes** fires often (occluders in the widened box are common) and costs actions without beating closer_rc.
- The multi-view planners bump 100+ times per 112 runs (about 1 per run). These are low obstacles in the
  pitch-0 blind zone, which a depth camera cannot see within 1.43 m.

### 6.3 Offline information-gain check (CPU, logged frames)
Script: `videos/active_approach/tmp/sc/ig_check.py`; output `tmp/sc/ig/ig_check.json`. Data: 3607 usable non-start
frames from 583 logged trajectories (round-3 tours, Seek, SeekConfident, native, pilot), 32 scenes.
Observation model q = P(view is right) by logistic regression, leave-one-scene-out. Under a noisy-sensor model
the expected information gain is monotone in q, so the max-IG view is the max-q view.

| LOSO AUC | JEV top-1 right | SAM3 right (margin -0.1) |
|---|---|---|
| Predictor features (what a planner knows beforehand) | 0.58 | 0.57 |
| measured mask features (upper bound) | 0.63 | 0.58 |
| distance only | 0.53 | 0.63 |

Picking one view per trajectory (435 with >= 3 candidates), the fraction right:

| pick | JEV | SAM3 |
|---|---|---|
| max q (IG) | .763 | .885 |
| last view (where an approach ends) | .756 | .802 |
| random | .721 | .868 |
| oracle | .892 | .977 |

Reading:
- **Geometry barely predicts whether a view will be read right** (AUC 0.57-0.63). The oracle-random gap shows
  that better views exist inside trajectories, but nothing a planner can compute beforehand finds them. So an
  information-gain planner with a geometric observation model has no signal to exploit for JEV.
- For SAM3, IG beats the last view (58 vs 22 trajectories), but random does almost as well (.868). The gain
  is "do not judge from the closest view" (truncation), not a fine view choice. decide.py's REL_AREA/nearest-6
  aggregation and `closer`'s fit-distance stop already address that.
- The Predictor features include the frame's own cloud (an optimistic leak), so the real AUC would be lower.
- Conclusion: no GPU run for an IG planner. Its ceiling is the observation model, and geometry is not one.

### 6.4 Evidential Active Recognition (Fan et al., CVPR'24) fusion, offline
Scripts: `videos/active_approach/tmp/sc/evidential.py`, `evid_check.py`. Per view an evidential opinion
(e = C p, b = e / S, u = K / S); views fused with the paper's approximate Dempster-Shafer rule. CV is
leave-one-scene-out on the 312 clean round-3 + Seek runs (decide.py's tuning set).

| variant | CV correct | FA | FR | actions |
|---|---|---|---|---|
| today: Dirichlet P(consistent) trigger + SAM3 area-weighted margin | 297 | 1 | 14 | 13.3 |
| fused-JEV u <= U* trigger | 297 | 1 | 14 | 14.9 |
| fused-JEV top-2 gap trigger | 290 | 3 | 19 | 14.4 |
| **today's trigger + fused-SAM3 verify (accept iff the target is the fused top)** | **301** | **0** | **11** | 13.3 |
| both fused | 298 | 3 | 11 | 17.5 |

- **Fused-SAM3 verify is flat** for C_S <= 10 with DS margin <= 0 (301 everywhere), so it is not a tuned spike.
- **Tuning set:** 5 fixed, 1 broken (p = 0.22).
- **Held-out trajectories** (the new self-contained runs, native, SeekConfident, pilot; 486 runs): today 451,
  fused 456; 9 fixed, 4 broken (p = 0.27).
- Fixed: plant targets (bxs78, epi175, TEE321) and the epi445 armchair-as-couch FA. Broken: two chair
  confusers seen by closer_rc and the pilot.
- **Reading:** a consistent +1 point (scenarios repeat across policies, so the true n is smaller). It is
  a decision-module change, not a planning one.
- **Not tried: the paper's learned PPO policy.** Its recognizer has to run every step, and JEV takes ~1 s per
  frame. The paper's own gain over fixation is +2.6.

### 6.5 Vocabulary variants (SAM3 / JEV words), offline
Scripts: `tmp/vocab_rescore.py` (SAM3 re-scored on the 1955 logged frames over the union of HM3D and JEV words),
`tmp/vocab_variants.py`, `tmp/sc/vocab_score.py`, `vocab_why.py`. The 312 clean runs, frozen flow.

| SAM3 vocabulary | today's rule | fused SAM3 |
|---|---|---|
| logged (HM3D 28, original frames) | 297 (FA 1, FR 14) | 301 (FA 0, FR 11) |
| HM3D 28, re-rendered (control) | 295 | 297 |
| HM3D + folds (monitor->tv, armchair->chair, flowers->plant) | 293 | 297 |
| JEV's per-target words | 269 (FR 36) | 269 |
| JEV's words + folds | 274 | 276 |

- **Keep SAM3 on the 28 HM3D classes.** JEV's look-alike words steal SAM3's top-1 from real targets: "vase" on
  potted plants (68 frames in the new false rejects), "sculpture" (21), "bench" on couches.
- Folding synonyms into the target does not help.
- JEV keeps its 16 per-target words; it only decides *when* (it never names the answer).
- Fused SAM3 beats today's rule by +2 to +4 on every HM3D vocabulary.

### 6.6 JEV over HM3D words (16 per target) vs its current words
Script `tmp/sc/jev_vocab.py` (re-render every countable frame; POST /ask with the frozen naming prompt, once per word
list), `jev_vocab_score.py`. HM3D16 = target + its HM3D look-alikes (couch: chair, stool, bed, cushion; chair: stool,
couch, table; bed: couch, stool, table, clothes; tv: picture, mirror, window, cabinet; plant: lamp, curtain; toilet:
sink, bathtub, shower) + HM3D fillers to 15 + "something else". Both arms use the same re-rendered frames and the same
re-rendered HM3D SAM3 scores; 312 clean runs, frozen flow. The start-frame gate is NOT refit (same uncertain starts).

| JEV words | JEV top-1 right (1933 frames) | today's rule | fused SAM3 | actions to commit | triggered |
|---|---|---|---|---|---|
| current 16 | .707 | 295 (FA 3, FR 14) | 297 | 13.4 | 178 |
| HM3D 16 | .830 | 293 (FA 3, FR 16) | 297 | **11.3** | **202** |

- Accuracy unchanged (today 0 fixed / 2 broken; fused 1 / 1).
- With HM3D words JEV names more consistently (no synonym splitting), so the trigger fires in 24 more runs and the
  commit comes 2.1 actions earlier (-16%).
- Open: refit the start-frame gate with HM3D words (needs the 444 reference requests re-named). It decides which
  starts count as uncertain.

### 6.7 Formalized (user decision 2026-11-07): decision module v2
- **Start-frame JEV gate: unchanged** (`jev.name_opts` words, frozen references and threshold).
- **Approach-phase naming** (the belief and its P(consistent) trigger): `perception.jev.approach_opts` (HM3D words,
  table in 6.6). Server `POST /name` takes `words` "approach" (default) or "gate"; `gate` signals only for "gate".
  The harness (`judges.py`) names with "approach" and never feeds those distributions to `/verify_views`'s gate.
- **Multi-view verification**: `perception.select.fuse_views` (evidence 2 x SAM3 score, approximate
  Dempster-Shafer fusion, accept iff the target is the fused top). It replaces margin + vote for selector
  jev_sam3. `aggregate` is still logged, and `sam3_margin` / `sam3_vote` in a request are now a 400. The
  ApexNav gate no longer sends them; its `multiview_*_margin` settings serve the d1 fallback only.
- `active_approach/decide.py` DEFAULTS = this flow (GATE all, nearest 6, REL_AREA 0.1, P* 0.8 consistent, VERIFY
  fused): 301/312 on the round-3 + Seek runs (VERIFY "margin": 297).
- Tests: `tests/test_perception.py` (approach options, /name words, fused verify_views), `tests/test_qwen_verification.py`
  (`fuse_views`); ApexNav `perception_gate/test_approach.py`. All pass.
- The running perception server still has the old code. It is restarted after the planner runs, which keep one
  version.

### 6.8 Stage B: the real decision flow (jev_sam3), 112-start subset
Runs: `ApexNav/videos/active_approach/runs/jev_sam3_sc_2026-11-07/` (`run.py --selfcontained --log-frames --name-usable
--full-trajectory`, JEV over its gate words at run time, judged by the server at that time). Scored offline with decide.py's
formal flow (6.7) by `tmp/sc/score.py`. References: the logged navmesh round-3 + Seek runs and the navmesh native
re-run (`jev_sam3_r3native_2026-11-07`). 104 starts are common and not eye-ambiguous.

| planner | correct | FA | FR | actions to commit | trajectory actions |
|---|---|---|---|---|---|
| navmesh fba_tour3_seek (the preliminary version) | **101** | 0 | 3 | 12.8 | 21.3 |
| navmesh s2 / svr _tour3_seek | 100 / 100 | 0 | 4 | 13.9 / 13.2 | 21.9 / 20.0 |
| navmesh native (ApexNav 1.5 -> 0.75 m) | 98 | 2 | 4 | 6.1 | 10.7 |
| **self-contained closer** | **99** | 1 | 4 | **4.5** | 7.8 |
| self-contained fba_tour3_seek | 95 | 2 | 7 | 12.6 | 25.3 |
| self-contained closer_rc | 94 | 3 | 7 | 11.1 | 23.7 |

Paired exact McNemar:
- **closer vs the navmesh original:** 0/2 (p = 0.5). It loses epi175 (stems in a vase) and epi445 (an armchair
  read as a couch).
- **The self-contained multi-view planners vs the navmesh original:** 0/6-7 (p = 0.02-0.03).
- **closer vs the self-contained multi-view planners:** 4-6/0-1 (p ~ 0.13).
- With the older margin-and-vote rule: the same ordering (98 / 100 / 95 / 94).

Eye check of the discordant cases (`tmp/sc/wrong_final/`): mostly ambiguous objects, where the extra views tip SAM3,
not navigation:
- epi893: the labelled "couch" is an upholstered ottoman or chaise.
- svBbv 380: a high-backed bar stool (a chair confuser) reads "chair" from the arc's far views.
- bxs 136: a child's chair/stool.
- TEE321 / yr17 plants: flower arrangements.
- epi651 / h1zee 615: the multi-view planners drove into close, truncated views (a wingback armchair read as a couch at
  0.5 m; a hide-covered chair seen only top-cropped).

**Conclusion.**
- Without the navmesh, the naive `closer` is the planner to use: it is within noise of the navmesh original (99 vs
  101 of 104) and commits after ~4.5 actions instead of ~13.
- Every view-aware planner tried (geometric NBV ports, arcs, two-view, occlusion escape, clear cone, image reflex, the
  information-gain check, the evidential fusion) is either no better on the proxy or worse on the real flow.
- The remaining errors are perception (look-alikes, flower arrangements, unreadable meshes), consistent with the
  literature's 2-3 point ceiling for active view selection over a centred approach.

### 6.9 MPPI footprint planner (Stage A proxy, GT-free tracker, 2026-11-07)
Code: ApexNav `active_approach/mppi.py` (`act` interface, `--selfcontained --tracker reproj`); runs
`videos/active_approach/runs/mppi_proxy/{v1,v1dial,all}`, launcher `tmp/sc/mppi_proxy.sh`.
Uses only RGB-D + odometry. The object is the fused depth cloud plus geometric phantoms where the frame cuts it.
Collision uses the local inflated grid. Occlusion uses its own 3D voxel map: the 2.5D max-height map let ceilings and
lintels hide everything. Each step samples K F/L/R sequences (H = 8) and scores every pose with
J = in-frame hull area x unoccluded fraction x completeness^kappa x facing. A rollout's value is max_t (log J - 0.02 t)
minus bumps. The update is a softmax over values, and the planner stops when nothing beats the current pose by
0.03. `mppi_dial` adds DIAL-MPC annealing: noise shrinks over 6 iterations and grows along the horizon, and the
temperature goes 0.15 -> 0.03.

| policy | reach | reach2 | truncE | bumps | actions |
|---|---|---|---|---|---|
| closer | 72 | 33 | 15 | 19 | 8.1 |
| native (sc) | 74 | 44 | 59 | 31 | 10.8 |
| closer_rc | 85 | 56 | 35 | 95 | 24.2 |
| **mppi (kappa 2)** | **80** | 38 | 20 | 27 | 11.0 |
| mppi_k4 | 79 | 36 | 19 | 22 | 10.6 |
| mppi_dial | 78 | 40 | 19 | 26 | 11.1 |
| mppi_k0 (pure footprint) | 75 | 40 | 67 | 30 | 15.2 |

Paired comparisons on reach:
- mppi vs closer: 10/2 (p = 0.04).
- mppi vs native: 8/2 (p = 0.11).
- mppi vs closer_rc: 4/9 (p = 0.27).
- DIAL vs plain MPPI: 2/4 (no gain).
- Pure footprint truncates the final view in 67/112 runs.

Eye check of the 12 mppi-vs-closer discordant final frames: the wins are real. Examples:
- a full couch where closer stares at a wall (epi234);
- the stools in full view (epi281);
- the sofa clear of the lamp (wcojb 344).

One failure mode: bxs 78, a plant on a cabinet, ends pressed against the cabinet front. The target is above the frame
at pitch 0, so the tracker loses it.
Stage B (jev_sam3 + decide.py) is not run yet.

**Stage B (jev_sam3 + decide.py fused flow, GT-free tracker, 2026-11-07).**
Runs: `ApexNav/videos/active_approach/runs/jev_sam3_mppi_2026-11-07` (launcher `run_mppi_stageB.sh`).
Score: `tmp/sc/score_mppi`. 104 clean common runs:

| planner | correct | FA | FR | actions to commit | trajectory actions |
|---|---|---|---|---|---|
| closer | **95** | 1 | 8 | 4.0 | 8.0 |
| mppi | 92 | 2 | 10 | 5.7 | 10.9 |

Paired: closer 4 / mppi 1 (p = 0.38).

Eye check (`tmp/stageB_disc.jpg`):
- MPPI's losses are planning failures:
  - it drives too close and loses the object above or across the frame (bxs 78 plant on a cabinet; h1zee 284 hide chair);
  - it stops at once with no usable view (epi696);
  - a full view of bar stools tipped a confuser into a false accept (h1zee 194).
- Its win is the sofa behind the lamp (wcojb 344).

The proxy's +8 reach does not become more correct decisions.

### 6.10 MPPI iterations v2-v10 and the baselines in Stage B (2026-11-08)
Goal (user): iterate on MPPI until it beats every approach strategy. Same conditions for all: Stage B, GT-free
tracker (`--tracker reproj`), jev_sam3 server, decide.py fused flow, 104 clean common runs. Runs:
`ApexNav/videos/active_approach/runs/jev_sam3_mppi_2026-11-07`; score `tmp/sc/score_v10`; eye sheets
`tmp/sc/eye_v8`, `tmp/sc/eye_v10`.

Versions (all in `active_approach/mppi.py`):
- v2: vertical-cut barrier (top / bottom cut from the near-face distance at pitch 0 and the category height prior),
  side-only completeness, cost-to-go to closer's fit pose, stop only on a usable view.
- v3: the user's cost: bbox presence x centring exp(-2c^2), horizontal-extent barrier, anchor-filtered cloud.
- v4-v7: MPPI around closer's drive as the nominal (v4), collision-aware lattice nominal (v5), half free samples and
  back-out hysteresis (v6), a penalty on weak usable frames on the way (v7).
- v8: closer until it would stop, then v1's MPPI (DELTA 0.1) from there, sharing the voxel map.
- v9 (user's idea): one cost, V = (1-w) V_view - w W_ref track - bumps. The reference is closer's drive (the real
  closer's first action, then a lattice descent to its goal). w = distance to closer's goal / 1.5 m (0 once closer
  is done). Half the samples are around the reference, half free; the best sequence's first action is executed.
- v10: v9 with w >= 0.5 until closer is done. In v9, near starts got w = 0 at once, and MPPI ignored closer's
  final facing turn and gambled on a truncated first view (epi574 bed, h1zee 284 chair).

| policy | correct | FA | FR | actions to commit | trajectory actions |
|---|---|---|---|---|---|
| **mppi_v10** | **98** | 1 | 5 | 5.0 | 11.9 |
| mppi_v8 | 97 | 1 | 6 | 5.6 | 12.4 |
| mppi_v9 | 96 | 2 | 6 | 4.9 | 11.3 |
| closer | 95 | 1 | 8 | 4.0 | 8.0 |
| mppi_v5 | 94 | 2 | 8 | 5.8 | 11.6 |
| mppi_v4 / mppi_v6 | 93 | 1-2 | 9-10 | 6.1-6.2 | 11.6-14.4 |
| closer_rc | 93 | 4 | 7 | 8.9 | 23.8 |
| fba_tour3_seek | 93 | 4 | 7 | 9.9 | 26.5 |
| native | 93 | 4 | 7 | 4.9 | 10.6 |
| mppi (v1) | 92 | 2 | 10 | 5.7 | 10.9 |
| mppi_v2 | 90 | 4 | 10 | - | - |

Paired wins/losses of v10, with exact McNemar p:
- closer: 3/0 (p = 0.25).
- closer_rc: 6/1 (p = 0.125).
- fba_tour3_seek: 5/0 (p = 0.06).
- native: 5/0 (p = 0.06).
- v8: 1/0.

No comparison is significant at 0.05 on 104 runs.

Eye check: all discordant labels hold.
- The confusers are a wing armchair for "couch", a leopard pouf for "bed", and bar stools for "chair".
- v10's one loss is a green armchair for "couch" (epi445), which closer_rc rejects.
- v10's own win is q5QZ 413, a daybed-style sofa: a fuller frontal final view than v8's.

Proxy (Stage A, 112 starts):
- v8: reach 82, reach2 46, truncE 22, bumps 28.
- v9: reach 83, reach2 44, truncE 18, bumps 23.

Why v8/v10 work: most of closer's errors were decisions forced at its single stop. Plain MPPI's early deviations,
made from far with a sparse cloud and tracker bleed, gave weak frames the flow committed on. v10 tracks closer while
far, blends in the view score mid-range, and runs pure view-MPPI only after closer is done.
Gate: `APPROACH_PLANNER=mppi_v10` (perception_gate/approach.py PLANNERS).

**v11 (2026-11-08): cut-aware re-plan.** v10 failed the TV on a tall dresser (ziup 179). It started at closer's
fit distance, so it only turned to face, stopped on a top-cut glimpse (41% visible), and the flow committed at
step 1 (fused SAM3 reject).

v11 adds:
- **Trigger** (once per run): the robot faces the object, the mask touches the top image row, and the near-face
  distance is at least 0.2 m short of the distance where the estimated top fits in the frame at pitch 0.
- **Re-plan:** v2's MPPI (vertical-cut barrier) with horizon 14, best sequence executed, no pull toward closer's
  goal, for at most 25 steps.
- **Guard:** while still too close, a turn into a facing view is flipped the other way, so the flow cannot commit
  on the cut glimpse.

Two simpler back-offs failed in smoke tests:
- a straight back-off to a computed goal drove into unmapped space;
- a grid path to that goal ran past the dresser.

Stage B, 104 common runs:
- **v11 99** (FA 1, FR 4), v10 98, v8 97, closer 95, closer_rc / fba_tour3_seek / native 93.
- Paired with v11: v10 0/1, closer 0/4 (p .125), closer_rc 1/7 (p .07), fba_tour3_seek 0/6 (p .03),
  native 0/6 (p .03), v4 0/6 (p .03).
- The one change vs v10 is ziup 179: it turns away, backs off to 2.5 m and accepts on a full view at step 14
  (eye-checked).
- The trigger fired in 13 runs (6 TVs, 6 plants, 1 bed) and changed only that outcome. Mean trajectory actions:
  13.0 vs 11.9.

Next (user: v11 is too hand-crafted): v12 should get this behaviour from the planner itself. Planned ingredients:
- a cut-aware view score;
- horizon about 16, with MPPI warm-started from viewpoint proposals: candidate poses at fit distance, scored and
  connected by lattice paths (Biased-MPPI style);
- v7's poor-frame penalty in place of the guard;
- alternatively, exact lattice search with batched GPU scoring.

### 6.11 v12-v15: one view MPPI without triggers (2026-11-08/09)

Code: `active_approach/mppi.py` `MppiV15` (v12 and v14 were folded into it on 2026-11-09; rejected variants
v12b-e and v13 removed). Gate: `APPROACH_PLANNER=mppi_v15`.

Nomenclature (MppiV15, 2026-11-09):
- **rollout**: an action sequence of horizon H = 16 on the lattice (F 0.25 m, L/R 30 deg, S stay), simulated on the
  agent's own grid to poses p_0..p_H.
- **warm-start trajectory**: the rollout of the naive approach (`closer`: drive to a fit distance and face the
  object), recomputed every step; its goal pose g is also a proposal. ("closer" stays the name of the baseline policy.)
- **view quality** J(p) = sat(size) x completeness^2 x centeredness, **view cost** c(p) = -log J(p):
  - completeness: fraction of the object's area (voxel-thinned cloud + phantom mass for cut extents, sized by
    `round3.PRIORS`) projected inside a Tukey window and unoccluded in the own voxel map;
  - size: in-window projected area / image area, saturated as size / (size + 0.05);
  - centeredness: 1 within 15 deg of the object's bearing, Gaussian falloff beyond.
- **cost** of a rollout (minimized): cost_t = sum_{s<t} q_s R_s c(p_s) + R_t c(p_t) + C_ACT t, the expected view cost
  of the frame the decision flow commits on (q_s = P_COMMIT 0.3 for usable facing frames, R_s = P(no decision
  before s)); cost = min_t cost_t + C_BUMP bumps + w W_REF track(warm-start trajectory), with w = clip(|g - p_0| / 1.5)
  non-increasing over the episode.
- **warm starts** of the MPPI: path-following rollouts to reachable ring proposals (both turn directions), the
  warm-start trajectory, the previous rollout shifted, stay, turn-to-face. Execute the first action of the lowest-cost
  rollout; stop when staying costs less than DELTA = 0.1 more.

History:
- **v12:** naive approach first, then one view MPPI (the score above with a poor-frame penalty). 98/104. Emergent
  back-offs (toilet 139, plant 458, chair 615) but loses ziup 179.
- **v14 (user's design):** no phases; the warm-start trajectory and goal seed one MPPI, and far out a tracking cost
  keeps it on that trajectory. 97/104.
- **v15:** v14 plus four changes, each from a logged v14 failure:
  1. tracking weight non-increasing (couch 690: a back-off grew the distance to g, tracking pulled it forward, 40 actions);
  2. the previous rollout is kept unless another is cheaper by DELTA (switched between equal tours every step);
  3. proposals reachable and faceable within H, warm starts in both turn directions, the 2 N_PROP cheapest kept,
     no 2D line-of-sight prefilter (ziup 179: all proposals 19-26 actions away; the 2D test rejected every view of
     the TV over its wardrobe);
  4. the expected-decision cost (P_COMMIT; measured: the flow commits at 33 % of first admitted frames, 22 % of
     all) replaces the poor-frame penalty (W_POOR, POOR_GAP).

Stage B (decide.py fused flow):

| planner | main (104) | second set (120) | total | traj actions (main) |
|---|---|---|---|---|
| v15 | 99 | 103 | 202 | 11.4 |
| v11 | 99 | 101 | 200 | 13.0 |
| v14 | 97 | 101 | 198 | 11.5 |
| v12 | 98 | 100 | 198 | 12.5 |
| closer | 95 | 102 | 197 | 8.0 |

The second set is `videos/active_approach/subsetB120.txt`: 120 other starts from the same 517, 20 per tag.
No pair is significant (exact McNemar p >= 0.25): only 4-6 starts per pair disagree. v15 gets ziup 179 without a
trigger (turns away, backs off 1.9 -> 2.35 m, whole TV; eye-checked) and keeps the toilet 139 back-off. Remaining
errors: q5QZ 413 bench (fixed pitch cuts it at < 2 m) and clear views the judge misreads (4ok3 269, bxsV 318).

Rejected after tests: size calibrated to the real mask (model size ~3x too large, but the fix pulls the robot in and
kills back-offs); discounting long rollouts by 0.88^t (the measured rate at which a rollout's target view is reached;
also kills back-offs).
Episodes page: claude.ai/artifact/3uJyd8ykVSo5WbYFHTECbg.
