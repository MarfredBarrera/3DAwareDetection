# Active-approach planners, round 1 (2026-11-06)

Goal: after `jev_sam3` says *unsure*, drive to views where it becomes sure **and right**. Harness:
`~/codes/ApexNav/active_approach/` (`docs/active_approach_tests.md`); background and P1-P5:
`docs/active_approach_geometric_plan.md`. This doc proposes the planners. It is design only.

## 0. What the planner must optimize (evidence)

**Embodiment** (`config/habitat_eval_hm3dv2.yaml`): forward 0.25 m, turn 30 deg, camera h = 0.88 m,
640x480, hfov 79 deg, so beta_h = 39.5 deg, beta_v = 31.7 deg, f = 388 px. Pitch is fixed (0; 9 real
starts are at -30 and keep it). There is no strafing and no reversing.

**Yaw lattice.** Forward steps and slides never change yaw, so yaw is always `yaw0 + k*30deg` (12
headings). A viewpoint whose object-center bearing lies on the lattice has zero facing error. Any
other viewpoint has up to 15 deg of error, which eats 15 of the 39.5 deg half-hfov.

**Costs.** Turning 180 deg takes 6 actions. Backing off takes about 12 + n actions (turn around,
walk, turn back). Overshooting toward the object is therefore expensive (F1, native at 0.75 m).

**What JEV's "unsure" reacts to.** I fit this on the 932 JEV-scored synthetic and real starts, using
GT view features (`jev_scores.jsonl` joined with the `scenarios_*` jsonl; script
`/tmp/claude-1018/-home-marfred-codes-3DAwareVerification/57f32fa9-5c45-43d7-b8d3-5643ff58f6be/scratchpad/proxy.py`).

| feature bin | unsure rate |
|---|---|
| area < 0.3% / 1-2% / 4-8% / > 16% | 0.89 / 0.63 / 0.43 / 0.26 |
| occ_frac < 0.05 / 0.4-0.6 / > 0.6 | 0.53 / 0.75 / 0.80 |
| trunc 0.15-0.35 (large objects) / > 0.6 | 0.45 / 0.71 |
| facing_cos > 0.7 vs < 0.25 | 0.42 vs 0.75 |
| distance (at fixed area) | weak, not monotone |
| role: target / confuser | 0.51 / 0.71 |

- **Image area is the main lever** (Spearman -0.42; every other feature is at most 0.19). Mild top or
  bottom truncation is fine when it buys area. Heavy truncation and occlusion of 0.4 or more hurt.
- When JEV is sure on a single frame it is rarely wrong: 11 of 376 sure frames (3%). Wrong
  decisions come from forced decisions and from views where JEV never becomes sure.
- **Proxy for "unsure"** (logistic regression, 5-fold CV AUC 0.79):
  `logit P_unsure = -1.19 - 1.49 log10(area) + 0.43 occ + 0.32 trunc + 0.24 max(0, trunc-0.5) - 0.31 facing_cos - 0.27 min(d,6) + cat`,
  with cat offsets chair -0.25, couch/sofa +0.76/-0.10, plant +0.47..0.54, toilet -0.50, tv +0.49 /
  tv_monitor -0.77, and bed 0 (the base). The implementer should refit it with run-time-predictable
  features only.
- **Ceiling.** Even the ideal view (area 0.1, no occlusion, face-on, 1.5 m) predicts P_unsure of about
  0.4. Being a confuser adds +0.72 logit, and geometry cannot see that. So about a third of the
  approaches will need multi-view aggregation (`/verify_views`, mean JEV distribution) or a forced
  decision. The planners must plan for 2-3 views, not just one.

**Shared run-time geometry** (no GT; the same inputs exist in ApexNav):

- **Object model** from the fused cloud: z_lo and z_hi are the 5 and 95 percentiles; the footprint
  comes from 2D PCA (long axis L, short axis S); the center c is the footprint center.
- **Local 2.5D height map**, 5 cm cells: the max z of every non-object depth point from the frames
  seen so far, plus a seen/unknown flag. Cells that are unknown and not navigable on the navmesh get
  an assumed height of 1.0 m and count half. This catches the occluders below 0.5 m and above
  0.88 m that ApexNav's 2D map drops.
- **`predict(pose)`**: project at most 256 cloud points through the camera model.
  - `area_hat` = the point splat's convex-hull area / image area, times (1 - occ_hat).
  - `trunc_hat` = the fraction of points outside a 3 deg image margin.
  - `occ_hat` = the fraction of rays camera->point that pass a height-map cell higher than the ray
    there, ignoring cells within 0.15 m of the footprint.
  - `facing_hat` = |cos| between the ray and the nearer footprint-axis normal.
- **Fit distance** `d_fit(bearing)` = the smallest radius on a 0.25 m grid where `trunc_hat` <= 0.05
  horizontally and at most kappa = 0.25 of the vertical extent is cut. With pitch theta and margin
  m = 3 deg, this reduces to
  `max((z_hi-h)/tan(theta+beta_v-m), (h-z_lo-kappa*H)/tan(beta_v-theta-m), (W/2)/tan(beta_h-5))`.
  At theta = 0: chair 1.2 m, toilet 1.25, stool 1.3, plant 1.1, bed from the foot 1.45, couch side-on
  1.6, TV 0.8-0.9. With kappa = 0 (floor contact visible), every floor object needs 1.6 m.
- **Judgeable frame** (decide only on these): the yaw is the lattice yaw nearest the bearing to the
  center, with an offset of 20 deg or less; the mask does not touch the left or right border (unless
  W does not fit the hfov); and the last action was not part of a turn sequence still in progress.
  The run then calls `jev_sam3` on that frame. At or after the first planned viewpoint it calls
  `/verify_views` (unforced) over the judgeable bank. Bank views with area < 0.5% are left out, because
  they dilute the mean distribution.
- **Lattice-aligned candidates**: `p = c - r*u(psi_k)` for the 12 lattice yaws psi_k, with
  r in {d_fit, d_fit+0.25, d_fit+0.5}. Snap to the navmesh and drop a candidate if the snap moves it
  more than 0.15 m. Cost `A(p)` = the geodesic / 0.25 plus the turns along the geodesic polyline
  (sum of |delta heading| / 30) plus the final face turn.

## 1. Methods

### M1. Fit-band approach, lattice-faced (FBA)

This is P2 done right. Approach along the current bearing to `r* = d_fit` and stop there. Do not go
closer to "be safe": retreating costs 12 actions or more. The goal is the lattice-aligned candidate
on the current bearing (+-15 deg) at r*. Execution: drive the geodesic with the shared controller,
then turn to the lattice yaw.

Iterations:
- **a.** Plain.
- **b.** If `area_hat(r*)` < 2%, allow kappa = 0.5 (more bottom cut) to get closer for small objects.
- **c.** Re-estimate d_fit every 4 steps from the growing cloud: z_hi rises when the top was clipped
  before.

Termination: sure on a judgeable frame en route or at r*; otherwise one forced `/verify_views`.
Failures: no gain against occlusion (F3) or obliqueness; an underestimated z_hi early on (fixed by c).
Cost: about 60 lines. It is the strongest cheap bet, because area dominates.

### M2. Sure-view ring (SVR): P1 with a JEV proxy and no tilt

Score the lattice-aligned candidates and choose
`argmax  S(p) = (1 - P_unsure(predict(p))) * (0.5 + 0.5 nov(p)) - A(p)/A0`, where:
- `nov` = min(1, min over judged views of |delta bearing| / 60 deg);
- A0 = 60 actions, so 10 actions cost 0.17 (sweep A0 in {30, 60, inf});
- the current pose is a candidate with A = 0. If it wins by a margin of 0.05 or less, stop and force a
  decision ("nothing to gain").

Execution: shared controller, then the lattice face turn, then judge. Replan every 5 steps or when the
cloud bounding box moves more than 0.25 m; switch goals only when S rises by more than 0.1.

Iterations:
- **a.** GT-free product `fit * (1 - occ_hat) * min(1, area_hat/0.04)` instead of the logistic.
- **b.** The fitted proxy.
- **c.** A proxy refit on frames the runs themselves logged (more close views; the starts are biased
  toward bad views).

Failures: the height map is mostly unknown from one start frame, so occlusion behind the object is
guessed. Viewpoints may sit in unknown space. Proxy error: confusers stay unsure.
Cost: about 250 lines (the height map, `predict`, scoring). Fixes F1-F3.

### M3. Budgeted multi-view tour (BMT): the decision wrapper for M1 or M2

Because the ceiling is about 0.4, plan K <= 3 views:
- v1 is the M1 or M2 goal;
- each next view is `argmax (1-P_unsure) * nov - A/A0` from the current pose, with bearing
  separation of at least 45 deg and staying in the fit band;
- the remaining budget must cover the next leg.

After each view, call `/verify_views` (unforced) over the bank and stop when it is sure. After K views,
or when the budget (40) runs out, make a forced decision (M >= 0, V >= 0.5, as the server does).
Iteration: K in {2, 3}; between views on the same ring, order the legs by the cheaper turn direction.
Failures: on a cramped navmesh no second bearing exists, so the tour forces early. Two views that
disagree raise entropy, which is correct but costs actions. Cost: about 80 lines on top of M1 or M2.

### M4. Visibility-keeping lattice path (VKP, SVPTO-like)

Every step should yield a judgeable frame. Run A* over states (x, y, k): positions from simulated
forward steps (`pathfinder.try_step`), with k one of the 12 headings. The step cost is
1 + lambda * [the object is outside +-(beta_h - 8 deg) or occ_hat > 0.4], with lambda = 2. The goal
is the M2 viewpoint. Non-holonomic consequence: keeping the object at offset alpha <= 30 deg while
moving gives a log-spiral, `r(phi) = r0 exp(-phi cot alpha)`. That changes the bearing by only
ln(r0/d_fit)/1.73 rad, about 30 deg going from 3 m to 1.2 m. Larger detours around occluders must
include blind segments, so M4 helps mostly with early decisions on approach and gives little for
going around an occluder. Cost: about 200 lines, plus search time (about 15k states within 4 m).
Implement only if M2 logs show many steps without a judgeable frame.

### M5. Cause-conditioned primitives (CCP; P3, now non-holonomic)

Diagnose the last judged frame and apply one move:
- area < 1%: M1.
- Mask touches the top or bottom border, or d < d_fit - 0.25: back off by a **tangent exit**
  (turn 90 deg, walk s = sqrt(d_fit^2 - d^2), turn to face). That is about 9 actions with +atan(s/d)
  of bearing novelty, against 12 or more for a reverse.
- Mask touches the left or right border: a 1-turn re-face.
- Occluded (more than 30% of the expanded mask box has depth below the object median minus 0.3 m and
  is not mask): swing to the side away from the occluder centroid by
  delta phi = atan((w_occ/2 + W/2)/gap), at radius d_fit.
- Oblique (facing_hat < 0.35, elongated footprint): go to the bearing perpendicular to L.

Explainable and cheap (about 120 lines), but brittle when several causes mix. Use it as a diagnostic
baseline and as M2's tie-breaker.

### M6. Look-alike-discriminative constraints (LAD): an add-on to M2 or M3

JEV's start distribution names the contending confuser `c* = argmax p(confuser)`. Add pair-specific
hard constraints to the candidate filter:

| target vs c* | view constraint |
|---|---|
| couch vs armchair/chair/bench | whole long axis in frame (`(L/2)/tan(34.5deg)` <= d), bearing within 30 deg of perpendicular to L |
| chair vs stool/bench/table | full height (kappa = 0 top and bottom), bearing >= 60 deg from the start (the backrest shows from the side or front) |
| bed vs couch/bench/table | whole footprint in frame, bearing 30-60 deg to L (length, width and headboard) |
| tv vs picture/mirror/window | 35-60 deg to the panel normal (short axis S) to reveal depth and stand; bottom edge in frame |
| toilet vs sink/bathtub/trash can | floor contact visible (kappa = 0, d >= 1.6 m), 30-60 deg to the S-normal for the bowl profile |
| plant vs vase/lamp/flowers | top in frame with 0.2 m extra, area_hat >= 5% |

Use it only when the contention percentile is above 0.5, so the planner targets the look-alike part of
the gate. Failures: wrong PCA axes on thin or fragmented clouds; the "front" of a chair is unknown.
Cost: about 60 lines on M2. This is the only method aimed at the confuser half, which is
0.71 unsure and the source of false accepts.

## 2. Ranking and what to implement first

| rank | method | why | first-round params |
|---|---|---|---|
| 1 | **M1 FBA** (a+c) | area is the dominant signal; cheapest; fixes F1 | kappa 0.25, m 3 deg, re-estimate every 4 steps |
| 2 | **M2 SVR** (a, then b) | handles occlusion, obliqueness and novelty; reuses M1's geometry | 12 lattice bearings x 3 radii, A0 60, nothing-to-gain 0.05 |
| 3 | **M3 BMT** wrapper on M1 and M2 | the predicted ceiling of about 0.4 needs aggregation | K 2-3, separation >= 45 deg, budget 40, forced M >= 0 / V >= 0.5 |
| 4 | M6 LAD | confusers; needs M2 | contention percentile > 0.5 |
| 5 | M5 CCP | diagnostic baseline | as above |
| 6 | M4 VKP | invasive; gain bounded by the spiral limit | lambda 2 |

Implement **M1, M2(a/b) and M3** first. Run M1+M3 and M2+M3 as policies, so the wrapper's gain is
separable.

## 3. Evaluation protocol

- **Set**: all 517 JEV-uncertain starts (`scenarios_uncertain/scenarios.jsonl`), paired across
  policies. Report per role x tag x source (real/synthetic).
- **Judge**: a new harness judge `jev_sam3` with a per-run view bank. On judgeable frames it calls
  `/verify` (selector `jev_sam3`, first frame) and then `/verify_views` (unforced, over the bank). The
  final call is forced. Non-judgeable frames are logged but not judged. The current `--judge jev` lets
  JEV's top-1 decide, which is not the deployed module.
- **Baselines**:
  - `stay`: a forced decision on the start frame.
  - `native`: 1.5 m, then 0.75 m.
  - `random-ring`: a uniform lattice-aligned candidate at r in [1, 2.5] m, 2 views, then forced.
  - `oracle`: GT ring, as now.
  - `jev-oracle` on a stratified 120-start subset: render every lattice candidate and take the nearest
    one where `jev_sam3` is sure and correct. This is the upper bound for regret.
  - A planner counts only if it beats `stay` and `random-ring`.
- **Primary metric**: wrong decisions = FA on confusers + FR on targets (undecided = reject), each
  **eye-verified** on sheets (every wrong one plus 10% of the correct ones; the labels are noisy).
  Use a paired McNemar test against each baseline.
- **Secondary metrics**: actions and path to decision; the fraction of starts that become sure before
  the forced call; views used; JEV calls.
- **View diagnostics**: GT area, trunc, occ and facing at decision time; proxy calibration (predicted
  vs observed unsure at the visited views).
- **Failure tags**: the existing ones plus `nothing_to_gain`, `no_candidate` and `forced`.

## 4. SVPTO-inspired planners (round 2, revised with the paper)

Source: `docs/papers/SVPTO_...pdf` (Wang et al., IEEE TIE 71(3) 2024), Sec. IV and Table I.
**V** marks terms taken verbatim from the paper; **A** marks our adaptations.

### 4.1 SVPTO's formulation

All costs use the hinge `p(x) = max(0, x)`, summed over the trajectory waypoints k. Notation:
`delta_k = t_k - p_k` is the vector from the robot to the target, and `beta` is the FOV half-angle
(the paper treats the FOV as a 2D horizontal sector).

- **(V) FOV metric, the lateral adjustment (eq. 3).**
  - `J_FM = sum_k sum_{i<N_h} p(l_i / gamma(m_{k,i}) - 1)`.
  - The sample points are `m_{k,i} = p_k + (i/N_h) delta_k`, and gamma is the ESDF.
  - `l_i = (i/N_h) |delta_k| tan(beta)` is the FOV cone's half-width at m_i. So the **whole FOV
    wedge** toward the target must be free of obstacles, not just a tube around the sight line.
- **(V) Environment complexity, the longitudinal adjustment (eq. 4).**
  - Split the FOV fan into N_l depth regions and compute `zeta = sum_j eta_j S_j`, where S_j is the
    voxel count in region j (from raycasting the depth image) and nearer regions get a larger eta_j.
  - `rd_max = (zeta_max - zeta)/zeta_max * (od_max - od_min) + od_min`.
  - `J_EC = p(od_min - |delta|)^2 + p(|delta| - rd_max)^2`. Clutter shrinks the maximum distance.
  - The paper does not give the values of eta_j, N_l or zeta_max.
- **(V) Relative angle, obstacle visibility (eq. 5).**
  - `J_OV = p(cos beta - delta.v / (|delta||v|))`, with v the **target's** velocity.
  - It keeps the robot's motion direction inside the FOV, so obstacles ahead are seen.
- **(V) Yaw (eq. 7).** It always points at the target, independently of the translation.
- **(V) Front end.** Hybrid A*, topologically equivalent to the target's path (no obstacle between the
  two paths). The end point is the midpoint toward the target's end point. A b-spline back end is
  solved with NLopt.
- **(V) Parameters.** lambda_FM 0.1, lambda_EC 10, lambda_OV 5; od 2.0-3.5 m; FOV 86 x 57 deg;
  depth range 6 m.
- **(V) Table I.**
  - The lateral term (lambda_FM = 0.1) cuts occlusion time the most: 9.1 -> 2.3%, 13.2 -> 3.9,
    19.4 -> 8.5 and 22.1 -> 9.3%.
  - The complexity term cuts it further: 2.3 -> 0.9, 3.9 -> 1.6, 8.5 -> 3.7 and 9.3 -> 4.7%.
  - The relative-angle term roughly halves the time spent beyond the critical angle.

### 4.2 Our terms

Costs are per pose, and one action costs 1. The paper's weight ratios are kept with scale s = 1:
`J_vis = s (0.1 J_FM + 10 J_EC + 5 J_OV)`. As a check, a blocked sight line gives about 0.5-2, a band
miss of 0.5 m about 2.5, and the target at 60 deg off-axis about 1.4, all comparable to a few actions.

- **FM, the clearance cone (A).** Same form as eq. 3, with N_h = 10 and i from 1 to N_h - 1.
  - The paper's full wedge (beta = 39.5 deg) cannot be satisfied indoors: at 1.5 m its half-width is
    1.2 m. So the cone angle is
    `beta_FM = min(beta_h, atan((W/2 + 0.15)/|delta|) + 5 deg)`, i.e. the object's angular half-width
    plus a margin.
  - Ablation `beta_FM = beta_h` is the paper-faithful version.
  - gamma(m_i) is read from the 2.5D height-map ESDF slice at the ray height
    `z_i = 0.88 + (i/N_h)(z_c - 0.88)`, with slices at z in {0.3, 0.6, 0.9, 1.2} m.
  - Cells within 0.15 m of the object footprint are excluded. Unknown cells count as occupied at
    half weight.
  - Three sight lines, to the object's center and its two horizontal extremes, so the object's whole
    width must be clear.
  - This replaces round 1's ball chain with radius proportional to the object width; the
    differences are the hinge form and the cone's half-width taken from the FOV.
- **EC, distance band from clutter (A).**
  - od_min = d_fit (section 0), od_max = d_fit + 1.0 m. The paper's 2.0-3.5 m suits a quadrotor
    following a person; ours comes from the object's height and the vertical FOV.
  - zeta is computed from the latest depth frame in the 79 deg HFOV fan. Use non-floor and
    non-object points with z in [0.1, 1.8] m, voxelized at 5 cm, out to min(5 m, which is ApexNav's
    max_depth, and the object distance + 0.5 m).
  - N_l = 4 equal depth bins with `eta_j = (N_l - j)/N_l`, j = 0 nearest, so the weights are
    1, 0.75, 0.5 and 0.25.
  - zeta_max = the 95th percentile of zeta over the 517 start frames, frozen; clip zeta to zeta_max.
  - zeta is recomputed at each replan from the current frame, as the paper does. A per-candidate zeta
    from the height map is optional.
- **OV, target in view while moving (A).**
  - Our target is static and our camera axis is the motion direction, so obstacles ahead are always
    seen. The useful analogue is the reverse: keep the **target** inside the FOV while moving.
  - `J_OV = p(cos beta_h - cos(angle(heading, delta)))`, applied after every forward action, with
    beta_h = 39.5 deg. Turns are exempt.
  - This replaces round 1's AO term, which had a 25 deg threshold and a cubic penalty.
- **Yaw (A).** Eq. 7 (yaw always on the target) is impossible while moving sideways on our
  non-holonomic base. The trade-off is therefore heading toward the target (in view, but little change
  in bearing: about ln(r0/d_fit) tan 39.5 deg, roughly 40 deg from 3 m to 1.2 m) against a blind
  sideways detour. Yaw is enforced only where we decide: stationary poses turn to the lattice yaw
  nearest the target bearing.
- **Safety (V/A).** As in [30], with ESDF clearance d_thr = 0.25 m on the navmesh or ApexNav's
  SDFMap2D: `w_C p(d_thr - ESDF)^2 / d_thr^2`, w_C = 10.
- **Valid pose** = J_FM = J_EC = J_OV = 0 at a lattice yaw. These are the poses we judge.

The action model is unchanged from round 1: {F, L, R}, `try_step`, the pruning rules, and states
hashed at 0.1 m x 12 headings.

### S2. Visibility-guided lattice A* (SVPTO front end and lateral/longitudinal adjustment); implement first

Run A* from the current pose to the set of valid poses with `P_unsure <= P_unsure(start) - 0.1`.
- Edge cost: 1 per action plus `J_vis` and the safety cost at the new pose.
- **Hard:** the last T = 4 poses must be valid. This is the paper's "no obstacle between the robot and
  the target" condition, applied to the final approach only. Blind segments elsewhere are what make
  detours possible.
- Heuristic: `max(0, |p - c| - rd_max)/0.25 + ceil(offset/30 deg)`.
- Cap at 20k expansions within a 5 m box.

The detour bearing comes from FM (the lateral term) and the stand-off distance from EC (the
longitudinal term), which mirrors Table I. Replan every 4 steps or when a new frame changes J_FM on the
remaining path; zeta is refreshed from that frame. Termination and failure modes are as in round 1.
Also: a zeta that is noisy from one cluttered frame makes the band jump, so smooth it with an EMA of
0.5. A cost of about 250 lines.

### S3. M2/M3 goal + visibility-MPC (S1's beam with the J_vis terms); second

Beam search over N = 8 actions (beam 64, discount 0.9), executing 2 actions and then replanning. The
terminal cost is the remaining actions to the M2/M3 goal. Frames from valid poses go to the bank, and
a sure verdict ends early. Its ablation is M2+M3 with s = 0. About 120 lines on top of S2's pieces.

S1 (proxy terminal, no goal) remains a free variant.

### 4.3 What changed after reading the paper

- **Clearance radius.** It is now the cone width from the FOV angle, in the paper's ratio-hinge form.
  We cap the angle at the object's angular size (beta_h itself is the ablation), where round 1 used a
  ball radius proportional to the object width.
- **Distance band.** It now comes from SVPTO's eq. 4 complexity rule, with depth-binned voxel counts
  in our 79 deg fan. Round 1 used a 1.5 m clutter disc on the map.
- **Facing term.** OV, re-aimed at keeping the target in view while moving, replaces AO. Weights now
  follow the paper's ratios.
- **Ranking.** Unchanged: S2 first, S3 second. Table I says the lateral term matters most, and S2's
  hard final-approach visibility plus FM is exactly that. The paper gives no evidence for per-pose
  judging, which is S3's addition.
- **Not adopted.**
  - The b-spline/NLopt back end: our actions are discrete.
  - Topological equivalence to the target's path: the target is static, so it reduces to the
    final-approach constraint.
  - The time-allocation cost.

## 5. Round 3: fixes after the GT dry run (112 starts)

Dry run: FBA (M1) beat SVR (M2a), with FR 25 vs 33 and 52 vs 22 of the 92 unsure starts resolved.
FBA+tour came out at 22/57. M1+M3 stays the reference; M2 and S2 count only if they beat it.

**1. M2 quality: score relative gain, not absolute area.** A small or far object never reaches
area 0.04, so every score fell to 0 or below. Use the gain over the **current** view instead:
- `q(p) = clip(log2(area_hat(p)/area_now), 0, 3)/3 * (1 - occ_hat(p)) * (1 - trunc_hat(p))
  + 0.5 * max(0, occ_now - occ_hat(p))`. A 2x area gain gives 0.33 and an 8x gain gives 1; clearing
  occlusion adds directly.
- `S = q * (0.5 + 0.5 nov) - A/A0`, with A0 = 60.
- Nothing to gain: the best q is below 0.15 (less than about a 1.4x gain and no occlusion gain).
- With the proxy (M2b), use `Delta = P_unsure(now) - P_unsure(p)` in place of q. The log-area term
  makes this scale-free in the same way.
- Tour (M3): q is relative to the **best view so far**. A second view qualifies when q >= 0 (no worse)
  and nov >= 0.5, so "no candidate" stops meaning "no candidate better than the impossible absolute
  scale".

**2. Clipped views: unknown extent.** For each image border the trigger mask touches, the object
continues past it. Add phantom points to the cloud before computing d_fit and trunc_hat; occ_hat
ignores them.
- **Top clipped:**
  `z_hi = max(z_hi_obs + 0.2, z_frustum + 0.1, z_lo + H_prior)`, where z_frustum is the height of the
  top image ray at the object's depth.
- **Bottom clipped:** `z_lo = max(floor, min(z_lo_obs - 0.2, z_hi - H_prior))`.
- **Left or right clipped:** extend the footprint on that side, along the image-plane direction, by
  `max(0.2, W_prior - W_obs)`.
- **Priors (H x W, m):** chair 0.9 x 0.55, couch 0.85 x 2.0, bed 0.6 x 2.0, toilet 0.8 x 0.45,
  tv 0.7 x 1.1, plant 1.0 x 0.5. Take the max of the target's prior and the contending confuser c*'s
  prior (conservative, so it ends farther away).
- A phantom side is dropped once a judged frame shows that side unclipped.
- This replaces "halve q if the last frame was clipped".

**3. S2 validity tolerance and band.**
- **Valid** requires all of:
  - all 3 sight lines clear (min ESDF >= 0.05 m over the samples with lambda <= 0.85);
  - FM at or below its 30th percentile over the in-band lattice poses of this plan (a relative
    tolerance, since FM is never 0 indoors);
  - d inside the band;
  - target offset <= 39.5 deg at the arrival lattice yaw.
- **Band:**
  - `od_min = d_fit - 0.25`.
  - `rd_max = max(od_min + 0.75, eq. 4 value)` with `od_max = d_fit + 1.25`. The floor keeps the band
    at least 3 forward steps wide; a band of about 0 m is what made EC the limiting term.
- If no valid pose is within the cap, drop the FM tolerance (keep the clear sight lines). If there is
  still none, hand over to M1.
- Log per-term pass rates for every plan.

**4. S3 MPC parking: cost balance.**
- Cap the per-pose visibility cost at `min(J_vis, 1)` and set the terminal weight w_G = 2 per
  remaining action. One step of progress then always outweighs one step of bad visibility, so parking
  can never win.
- Apply EC only within 1.0 m (geodesic) of the goal. Elsewhere only OV (weight 1) and safety act.
- These two rules replace the progress guard; keep the guard only as a stuck check.

**5. Proxy refit.**
- **Label:** jev_sam3 unsure = score >= 0.5076, recomputed from the logged pct_entropy and
  pct_contention, on every judged frame of every run (all policies, start frames included).
- **Features:** `log10 area, occ, trunc, max(0, trunc-0.5), facing_cos, touches_border, logit(u0)`.
  u0 is the trigger frame's JEV score, which is known at run time. It carries the object's intrinsic
  ambiguity (the confuser effect) without the role or the category. Drop the category offsets.
- **Fit:**
  - Model: L2-logistic with lambda 1e-2.
  - Weighting: each start weighs 1 in total (frames weighted 1/n_frames).
  - Cross-validation: grouped 5-fold by **scene**; report AUC and a reliability plot.
- Fit two versions: on GT features (for analysis) and on `predict()` features logged at each judged
  pose (deployed, because that is what the planner sees). From now on, log `predict()` for every
  judged pose.
- The 50 starts that are sure under jev_sam3 stay in the set and are reported separately, as a
  `stay`-resolvable control.

**6. jev-oracle.** Yes, on **30 starts**: role x tag, 2-3 per cell, half real. That is about 1 h of
GPU, run after the main jev_sam3 runs. It is worth more as **dense training data** (about 1,000 views
labelled by jev_sam3 from controlled poses, much less biased than the starts) than as an upper bound,
since the GT oracle already bounds the views. Use it to (a) refit or validate the proxy and (b) measure
the regret of FBA and SVR on those 30. Skip it if the GPU is contended.

**7. Mean-distribution dilution (`/verify_views`).** The bank mean includes the unsure trigger view
(up to 6 views), so a single clear new view is averaged with a flat distribution and stays unsure.
Implications:
- **Bank:** add only frames from valid or at-band poses. En-route mediocre frames are logged but not
  banked.
- **Tour:** with the trigger in the bank, at least 2 good views are needed to outweigh it. Plan K = 3
  and make the 2nd view cheap and **consistent** rather than novel: the same band, the next lattice
  bearing (30 deg separation instead of 45), usually 3-6 actions. Novelty still breaks ties.
- **Decision-side option** (needs the user's OK; the module is frozen): call `/verify` on each new
  planned viewpoint's frame alone before `/verify_views`, so a clear view can decide on its own.
  Alternatively, drop bank views whose area is below 0.25x the best view's. Either directly removes
  the dilution. Report the planners with and without it.

**Round 3 order:**
1. FBA+tour with fixes 2 and 7: the reference.
2. SVR with fixes 1 and 2 (M2c = relative-gain M2a; M2b after the refit).
3. S2 with fix 3.
4. S3 with fix 4, only if S2 beats FBA+tour.

## 6. Belief update (Dirichlet evidence)

**Scope.** The planner is unchanged. JEV is called once per frame on a single image, on every frame
where the tracked object is in view. Its naming distributions accumulate into a Dirichlet belief, and
the belief decides only **when** to call the unchanged `jev_sam3` module again. Options are the 16
naming options of `name_opts(target)`, folded with `jev.fold`, so K = 16 (15 for chair).

### 6.1 Update

For each frame v with folded JEV distribution p_v:
`alpha_c <- alpha_c + w_v * e_v(c)`. Each frame adds at most w_v of evidence mass, because e_v sums
to at most 1.

**Prior alpha0** (prior mass a = 2; the vacuity in 6.2 is defined relative to it):
- **P-u, uniform:** `alpha0_c = a/K`.
- **P-cat, per category:** `alpha0_c = a * pi_c`, where pi_c is the smoothed JEV top-1 frequency for
  that target over the scored starts (add 0.5 to every count).

**Evidence e_v:**
- **E1, raw:** `e_v = p_v`.
- **E2, tempered:** `e_v proportional to p_v^(1/T)`, normalized, with T fitted (start at T = 2; JEV is
  overconfident).
- **E3, thresholded:** keep the options with p >= 0.05, then renormalize.
- **E4, one-hot:** `e_v = onehot(top1) * (1 - H_v)`, with H_v = H(p_v)/log 16.

**Weight** `w_v = r'_v * n_v * m_v`, in [0, 1]:
- **(a) Reliability.** `r_v = P(JEV right | geometry)`, a logistic regression with the round-3
  features but the label "top-1 == target is correct for the role"; scene-grouped CV.
  `r'_v = max(0, 2 r_v - 1)`, so a coin-flip view adds nothing.
- **(b) Novelty, for correlated frames.** Two forms:
  - **Kernel:** `n_v = 1 - max_j k(v, j)` over the frames already counted, with
    `k = exp(-(dphi/20deg)^2 - (dd/0.5 m)^2 - (doff/15deg)^2)`. Here phi is the bearing from the
    object, d the distance, and off the object's offset from the image center.
  - **View-cell pooling (simpler):** cells are 30 deg bearing bins x distance bins
    {<1.2, 1.2-2, 2-3, >3 m}. A cell's evidence is its frames' weighted mean, with the cell's total
    mass capped at max over its frames of r'_v m_v. Recompute alpha from the cells after every frame.
- **(c) Mask and track consistency.**
  - `m_v = clip((IoU_v - 0.2)/0.3, 0, 1)`, where IoU_v is between the frame mask and the tracked
    cloud reprojected into that frame.
  - m_v = 0 if the mask has fewer than 400 px or its depth median is more than 0.5 m from the cloud
    center.

### 6.2 Belief signals

`S = sum alpha`, `p_hat_c = alpha_c/S`.
- **Vacuity** `u = a/S`. It is 1 before any evidence; u <= 0.4 means about 3 effective views.
- **Conflict between views:** `JS = H(p_bar) - sum_v w~_v H(e_v)`, the weighted Jensen-Shannon
  divergence of the counted views' evidence. p_bar is their weighted mean and w~ the normalized
  weights.
- **Dominance**, with the pair ratio `p_t/(p_t + p_x) ~ Beta(alpha_t, alpha_x)`:
  - accept: `P_acc = P(p_t > p_c*) = 1 - I_0.5(alpha_t, alpha_c*)`, where
    `c* = argmax over contention_confusers(target) of alpha`;
  - reject: `P_rej = I_0.5(alpha_t, alpha_n)`, where n = the most-supported non-target option
    (including "something else").
- **Relation to the frozen gate.** With alpha0 -> 0, w = 1 and E1, p_hat is exactly the mean
  distribution `/verify_views` judges today, so the gate's entropy and contention computed on p_hat
  reproduce the current rule. What is new:
  - vacuity: how much independent evidence there is, which the mean cannot express;
  - the weights: bad and duplicate views stop diluting;
  - the dominance probabilities: a calibrated stop.

### 6.3 Trigger, re-trigger and budget

Call `jev_sam3` when one of these fires (fit the thresholds, below):
- **T-gate:** the frozen gate score on p_hat is below 0.5076 and u <= 0.5. The current gate on the
  better-built mean.
- **T-dom:** `max(P_acc, P_rej) >= 0.95` and `S - a >= 1.5`.
- **T-sprt:** `Lambda = sum_v w_v log((p_v(t) + eps)/(max_{c != t} p_v(c) + eps))`, with eps = 0.02.
  Fire when |Lambda| >= log((1-beta)/alpha) = 2.94 for alpha = beta = 0.05.

What the call is (compare both):
- **C-views:** `/verify_views` over ApexNav's `select_views` from the banked in-view frames.
- **C-best:** `/verify` on the frame with the largest r'_v m_v.

`jev_sam3` re-runs JEV itself; the belief is not passed in.

- **Still "insufficient":** keep accumulating. Re-trigger only after new evidence `S - S_last >= 1.0`
  that includes at least one new view cell.
- **At the end of the budget:** one forced `/verify_views` (forced = true) over the selected views.

### 6.4 Variants and ablations

| name | prior | evidence | weight | trigger |
|---|---|---|---|---|
| **B1** mean+vacuity | P-u | E1 | m_v, cell pooling | T-gate |
| **B2** reliability Dirichlet | P-u | E2 | r' n m (kernel) | T-dom |
| **B3** weighted SPRT | none | p_v | r' n m | T-sprt |

Ablations:
- **today:** alpha0 -> 0, w = 1, `/verify_views` on every new judgeable frame;
- **no weights:** w = 1 with cell pooling;
- **no novelty:** n = 1;
- E1 vs E2 vs E4;
- P-u vs P-cat;
- C-views vs C-best.

### 6.5 Evaluation (offline replay, CPU)

- **Data.** Replay the harness logs of every policy's trajectories (round 3 runs, 112 starts first,
  then all 517).
- **Why this is CPU-only.**
  - `jev_sam3`'s `/verify_views` is a deterministic function of the per-view JEV distributions (gate on
    their mean) and the per-view SAM3 scores (`select.aggregate_views`).
  - `/verify` on one frame needs only that frame's JEV and SAM3 outputs.
  - So logging both per frame lets every trigger and call variant replay without a GPU.
  - `select_views` must be imported from ApexNav's `perception_gate`, not reimplemented.
- **Fit.** Thresholds (u*, P*, the alpha/beta of T-sprt, T, ΔS) and the reliability model are fitted
  with **scene-grouped 5-fold CV** on eye labels. The objective is wrong decisions + 0.02 x calls.
  Report the Pareto curve of wrong decisions against calls.
- **Metrics:**
  - eye-verified wrong decisions (FA on confusers + FR on targets, undecided = reject);
  - jev_sam3 calls per start;
  - frames, actions and path until the decision;
  - forced-decision rate;
  - calibration of P_acc and P_rej against eye labels.
- Each variant counts only if it beats "today" on wrong decisions at an equal or lower number of calls.

### 6.6 Per-frame log fields (harness `steps.jsonl`; required for the replay)

- **ids:** run, scenario, policy, judge, step, action, timestamp.
- **pose:** x, y, z, yaw, pitch; distance and bearing from the cloud center; target offset in the
  image; judgeable flag; arrived flag.
- **mask:** pixels, area, bbox, touches [top, bottom, left, right], depth median, reprojection IoU
  with the tracked cloud, cloud size, rgb/mask paths.
- **predicted geometry:** area_hat, trunc_hat, occ_hat, facing_hat, d_fit, phantom sides
  (round 3 #2).
- **JEV, every in-view frame:**
  - the option list and the raw and folded distributions;
  - entropy, contention, pct_entropy, pct_contention, gate score;
  - top-1;
  - the template version.
- **SAM3:** full-vocabulary scores on the frame plus mask (selector `input`); margin; top-1.
- **jev_sam3 calls:** the views sent, the forced flag, the verdict summary.
- **GT (analysis only):** vis_frac, trunc_frac, occ_frac, area, facing_cos, distance, role, object
  id and the fraction of the mask on it.
