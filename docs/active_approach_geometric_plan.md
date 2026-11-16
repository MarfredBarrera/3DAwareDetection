# Plan: geometric active approach after an "uncertain" verdict (2026-11-05)

Scope: what the ApexNav agent does after the sufficiency signal (black box: yes / no / uncertain per
frame; `docs/observation_sufficiency_plan.md`) says **uncertain**. It replaces today's native approach
(forced cluster -> `searchObjectPath` -> 1.5 m standoff -> face turn -> re-verify;
`docs/apexnav_approach_reverify_plan.md` 3.2a) and keeps the d1mv view bank and forced decision
(`docs/d1_multiview_close_range_plan.md`). No learning, 2D/2.5D map, a ring of candidate viewpoints.

Failures to fix: (F1) too close, object cut off at the top (camera 0.88 m); (F2) cluster centroid drift on
large objects (bed 1.6 m); (F3) occluded views don't improve because the path keeps the same bearing;
(F4) ~2/3 of triggers are non-goals, so every approach step must buy information.

## 1. Reference paper

**H. Wang, X. Zhang, Y. Liu, X. Zhang, Y. Zhuang, "SVPTO: Safe Visibility-Guided Perception-Aware
Trajectory Optimization for Aerial Tracking", IEEE Trans. Industrial Electronics 71(3):2716-2725, 2024,
doi:10.1109/TIE.2023.3270541** (IEEE arnumber 10113878). Closed access, no arXiv; below is from the abstract
and its lineage. Exact equations not verified (need the PDF).

- A quadrotor tracks a moving target; occlusion and collision are handled together by **actively changing the
  relative distance and angle** to the target, not only by avoiding obstacles.
- **FOV-based occlusion metric:** the camera-to-target view cone must be obstacle-free. In the line it builds on
  (Wang et al. 2021, arXiv 2103.06742), the cone is approximated by M balls on the segment robot p -> target c:
  center c_i = p + (i/M)(c - p), radius r_i = rho (i/M) |p - c|, cost sum g(r_i^2 - ESDF(c_i)^2) with
  g(x) = max(0,x)^3 (rho 0.8, M 10). Wang 2021 also has a distance band od_min < |p-c| < od_max (2.5-3.5 m) and
  a yaw-to-target term.
- **Environment-adaptive distance:** environment complexity (obstacle density near the target) changes the
  preferred tracking distance (closer in clutter, which lowers the chance that something gets between).
- **Critical relative angle:** limits the angle between the motion direction and the target bearing, so the camera also
  sees obstacles ahead.
- Time optimization for smoothness.

Takeaway for us: score a viewpoint by (a) a distance band, (b) an obstacle-free view cone to the object, checked with
the distance field, (c) heading toward the target, and let clutter change the distance. All four are cheap on
ApexNav's 2D SDF map. We pick discrete viewpoints, so we don't need its continuous optimizer.

## 2. Related ideas (what we borrow)
| idea | source | borrowed as |
|---|---|---|
| visibility = distance band + yaw + ESDF ball chain | Wang et al. 2021 (2103.06742); Elastic Tracker, Ji et al. 2022 (2109.07111) | `fit` band, `vis` ray/cone test |
| NBV by counting the unknown cells a candidate sees, using ray casting | Isler et al. ICRA 2016; Bircher et al. ICRA 2016 (receding-horizon NBV) | `gain` term (unseen object side) |
| active classification: choose the view that adds the most new information, with occlusion predicted | Patten et al. RA-L 2016; Jayaraman & Grauman ECCV 2016 | novelty in bearing over judged views |
| move closer and re-check before committing | CLOSER (2610.28476); Krantz et al. ICCV 2023 (re-match on approach); Explore-until-Confident, Ren et al. RSS 2024 | the approach loop, early re-verify en route |
| canonical / readable views | Blanz et al. 1999 | yaw toward the object's center, small elevation |

## 3. Shared definitions (all policies)
Inputs that ApexNav has (read-only survey, branch `d1-approach`):
- Object footprint: `ObjectMap2D::getObjectCloud(id)` gives the 2D cells of `ObjectCluster::good_cells_`
  (z=0). Height: `ObjectCluster::box_min3d_/box_max3d_` and `clouds_[0]` (label-0 3D points); use the 5/95 z
  percentiles, not the box. File: `src/planner/plan_env/include/plan_env/object_map2d.h`.
- Map: `SDFMap2D` (`getOccupancy` UNKNOWN/FREE/OCCUPIED, `getInflateOccupancy`, `getDistance` = ESDF), 0.05 m
  cells. Obstacles are depth points with z in [0.5, 0.88] m (`map_ros/filter_min/max_height`), so 2D rays miss
  occluders lower than 0.5 m and higher than 0.88 m.
- Paths: `Astar2D` (`NORMAL`/`OPTIMISTIC`), `ExplorationManager::searchPointPath`; LOS + ring search to reuse:
  `searchViewpointPath` on `1d3436e` (backup branch).
- Last views: gate `observe()` items (mask, area_fraction, `mask_touches_border`, camera, pitch, step) and
  the judged views `state["used"]`/`state["bank"]` (`perception_gate/gate.py`). Depth is in the Habitat obs
  (`habitat_evaluation.py`), not in the gate today.
- Camera: h = 0.88 m, 640x480, hfov 79 deg -> half vfov beta_v = 31.7, half hfov beta_h = 39.5; pitch steps
  of 30 deg (`PITCH_STEP`), turn 30 deg, forward 0.25 m.

Per candidate p (yaw facing the object's visible-cell center, pitch theta in {0, +-30}):
- `d(p)` = distance to the **nearest object cell** (not the centroid; fixes F2).
- **Fit band** `[d_lo, d_hi]` (fixes F1). With margin m = 3 deg, H = z_hi - z_lo, z_b = z_lo + 0.25 H (the
  bottom 25% may be cut off):
  `d_lo = max( (z_hi-h)/tan(theta+beta_v-m) if z_hi>h,  (h-z_b)/tan(beta_v-theta-m) if z_b<h,  (W_perp/2)/tan(beta_h-m) )`
  `d_hi = max(d_lo + 0.5, min(2.5, H/0.22))` (object at least ~20% of the image height).
  If the triggering mask touches the **top border**, z_hi is a lower bound: set
  `z_hi <- max(z_hi, h + d_cam*tan(beta_v)) + 0.2`. Examples at theta=0: bed 0-0.65 m -> 1.3 m; TV 0.8-1.5 -> 1.1;
  toilet 0-0.8 -> 1.25; couch 2.2 m wide (seen side-on) -> 1.5. `fit(p) = 1` inside the band, else `exp(-dist_to_band/0.5)`.
- **Visibility** `vis(p)` = fraction of object cells c (subsample <= 64) whose 2D ray p->c has no OCCUPIED cell
  outside 0.3 m of the object cells, with c inside the horizontal FOV. UNKNOWN cells on the ray count 0.5.
  Cone version (SVPTO-style): also require `ESDF(c_i) >= r_i` on the ball chain to the visible center.
- **Novelty** `nov(p) = min(1, min_k angdiff(phi(p), phi_k) / 60deg)`, phi = bearing object -> camera, k over judged views
  (fixes F3).
- **Gain** `gain(p)` = fraction of the UNKNOWN cells within 0.5 m of the object footprint that p sees by 2D ray
  (the unseen side).
- **Cost** `L(p)` = A* path length (NORMAL, then OPTIMISTIC x1.3); p must be in the map, not OCCUPIED, not inflated;
  UNKNOWN p allowed with x0.5 utility (the 1d3436e search failed 10/12 by requiring known-free).

Candidates: bearings every 15 deg (24) x radius r in {d_lo, d_lo+0.25, ..., d_hi} measured **from the object
surface** along the bearing (`p = support(phi) + r u(phi)`), x theta in {0, +-30} only when the band at 0 is empty.

## 4. Candidate policies (ranked by value / simplicity)

**P1. Fit-aware ring viewpoint (recommended v1).**
`U(p) = fit(p) * vis(p) * (0.5 + 0.5 nov(p))`, choose `argmax U(p) - L(p)/L0` (L0 = 4 m: a 1 m detour costs 0.25 of utility).
Fixes F1-F3; F4 via the early re-verify and the "nothing to gain" stop below. Failure modes: wrong z
extent from noisy fused clouds (mitigated by percentiles and the clipped-top bump); 2D map misses occluders outside
0.5-0.88 m; footprint fragments of one object (use d1mv's sibling merge, `multiview_sibling_m`); viewpoint in
UNKNOWN turns out occupied (replan). Stop: see section 5.

**P2. Adaptive standoff only (smallest change).** Keep native `searchObjectPath`, but replace
`APPROACH_STANDOFF = 1.5` with `d_lo + 0.25` from the nearest cell (`exploration_fsm.h`), and at arrival add the
pitch turn if the band needs theta != 0. Fixes F1, partly F2; not F3. Stop: arrival -> face -> verify. ~30 lines
of C++. A fallback if P1 slips.

**P3. Cause-conditioned primitive.** Diagnose the last view and pick one move: mask clipped top/bottom or
area > 0.35 -> back off to d_lo along the current bearing (or tilt); area < 0.02 -> straight approach to d_hi;
measured occlusion (depth-image fraction of the mask bbox with depth < object depth - 0.3 m) > 0.3 ->
side-step to the nearest bearing with vis >= 0.8 and nov >= 0.5. Simple and explainable, but brittle when the
causes mix; better used as P1's prior (weight the terms by the cause) than on its own.

**P4. NBV with gain.** `U = fit * (0.5 vis + 0.5 gain) * (0.5 + 0.5 nov)`. Useful when only one face of the object was
seen (bed from the foot, chair from behind). Costs more ray casts, and gain is 0 once the area is mapped (the
usual case, since the agent saw it). v2 term for P1, enabled when `gain` > 0.2 anywhere.

**P5. Visibility-keeping path (SVPTO proper).** Add a per-cell A* cost for losing LOS to the object, so every step
en route gives a usable early view. This touches `Astar2D` and is the most invasive. Only if the logs show few early
decisions.

## 5. Recommended v1 (P1) and integration

Interface (pure function, identical in the harness and ApexNav):
```
plan_view(agent: Pose(x, y, yaw, pitch),
          obj: ObjectGeom(cells Nx2, z_lo, z_hi, evidence_xy, judged_views [(cam_xy, step, clipped_top, area)]),
          grid: Grid(occ int8 {-1,0,1}, inflated bool, esdf float, res, origin),
          budget_m: float, params) -> ViewPlan | None
ViewPlan = {viewpoint (x, y, yaw, pitch), path [(x,y)], terms {fit, vis, nov, gain, L}, U, reason,
            candidates (top 10, for logs)}
```
Reference Python (`perception_gate/viewplan.py`, numpy only, unit-tested on hand-made grids); C++ port in
`ExplorationManager::planViewpoint` reusing the LOS lambda and A* of `searchViewpointPath` (1d3436e). Parity
test: the same fixtures through both.

```
on uncertain(cluster):                                   # gate state 3
  plan = plan_view(...)                                  # C++ at startApproach
  if plan is None or plan.U < 0.3: forced_decision()     # nothing worth driving to (F4)
  if U(current pose) >= plan.U - 0.1: forced_decision()  # this view already is about as good: moving won't help
  drive to plan.viewpoint                                # one action per step (planViewpointAction)
  every step: new view of the cluster -> early verify (existing approach_view rule)
  every 5 steps or footprint bbox moved > 0.25 m: replan; switch only if score gain > 0.1
  arrived (<= 0.25 m): turn to plan.yaw (<= 3 turns), tilt to plan.pitch, verify (status 1)
  verify says uncertain: add view to judged_views; second viewpoint with nov >= 0.5 (max 2 per approach)
  budget gone / unreachable: forced decision over the view bank (d1mv), never stop on uncertain
```
Integration points (ApexNav, branch d1-approach; read-only today):
- `exploration_fsm.cpp` `startApproach`: call `planViewpoint` instead of setting `forced_cluster_id_`; store
  `vp_*` fields (from 1d3436e `exploration_data.h`).
- `checkApproach`: arrival = within 0.25 m of the viewpoint (replaces the standoff/REACH_OBJECT test);
  the step budget and stuck checks stay as they are; per-step action from `planViewpointAction` (1d3436e).
- `faceTurn`: face `plan.yaw`, then the pitch step; restore pitch 0 after verifying (2 actions).
- `VerificationResult.msg`: add `clipped_top`, `area`, and `judged_x[]/judged_y[]` (judged camera positions),
  filled by `perception_gate/gate.py` `_issue_approach` from `state["used"]`/bank.
- `gate.py` `_after_approach`: status 2 (unreachable) and P1's "nothing to gain" -> the multiview forced decision,
  as with `multiview_failure`.
- Logs: `ViewPlan` (terms, U, reason, top candidates) per approach in `decisions.jsonl`.

## 6. Metrics
Per uncertain trigger (the unit), split by eye label goal / non-goal:
1. **Resolution rate**: fraction that becomes a confident yes/no before the forced decision, and its accuracy (by eye).
2. **Views to decision** and **actions / path length per trigger**; separately for non-goals (F4 cost).
3. **View quality at arrival**: mask touches top/bottom border (F1), area fraction, depth-measured occlusion
   (trigger view vs arrival view, F3), bearing change from the trigger view.
4. **Planner health**: viewpoint found / reached / replans / abort reasons; the "nothing to gain" rate.
5. Episodes: **FS and FR by eye** (CLOSER attribution), FS+FR, success, SPL, vs d1mv native standoff, paired
   bootstrap.
Harness controls (repo convention): identity (stay and decide), random ring candidate, native standoff, and an
**oracle** (render every ring candidate in Habitat, run the signal, take the best). Report P1's regret vs the oracle.
P1 counts only if it beats identity and random on resolution rate at equal or lower actions.

## 7. Open questions
1. Is the tilt allowed (2 extra actions, and does ApexNav's depth->map projection handle pitch != 0)? If not, the band
   uses theta = 0 only.
2. Object height when the 3D cloud is thin: per-category priors (bed 0.6, TV 0.7, plant 1.0 ...) as a fallback?
3. v1 in C++ (closed loop now) or Python-first in the harness (needs the occupancy grid exported)?
4. Should "nothing to gain" force a decision, or reject outright (it changes FR on goals seen only from bad angles)?
5. Second viewpoint: same approach budget (40) or its own?
