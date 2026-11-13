# View planning for object verification: literature notes (2026-11-07)

Scope: a self-contained "approach" phase (<= 40 actions: forward 0.25 m, turn 30 deg; no
tilt, no strafe) that collects clearer views of one candidate object. Allowed inputs:
RGB-D (640x480, hfov 79, h 0.88 m), odometry, the trigger mask plus a tracker, and
collision feedback. No navmesh, no global map, no GT.

Every paper below was found by web search on 2026-11-07, and the links worked.
Numbers marked (abs) come from the abstract or a summary only. All other numbers
were read from the paper's own tables (HTML/ar5iv).

---

## 1. Active recognition / NBV for recognition

### Look-Ahead Before You Leap: End-to-End Active Recognition by Forecasting the Effect of Motion
- Jayaraman & Grauman, ECCV 2016, arXiv 1605.00164 — https://arxiv.org/abs/1605.00164
- A recurrent actor-aggregator-classifier trained with REINFORCE, plus a "look-ahead"
  regulariser that predicts the next feature from the chosen motion. Viewpoints sit on a
  fixed grid (SUN360 panoramas, GERMS robot hand). There is no navigation.
- Inputs: a fixed view sphere or grid, with labels at training time.
- Margins: SUN360 T=5 gives 66.0% learned vs 55.6% random-recurrent vs 40.1% single view.
  GERMS T=3 gives 48.0% vs 44.2% random. The look-ahead term adds less than 1 pt.
- Use for us: the gain over random is large only when views are cheap and arbitrary. For
  us, motion is costly and constrained, so treat this as an upper bound, not as evidence.

### Learning to Look Around: Intelligently Exploring Unseen Environments for Unknown Tasks
- Jayaraman & Grauman, CVPR 2018, arXiv 1709.00507 — https://arxiv.org/abs/1709.00507
- An RL policy is rewarded for reducing its uncertainty about unobserved parts of a panorama
  or 3D shape ("active observation completion"). The policy is task-agnostic.
- Inputs: a panorama/viewgrid. No navigation.
- Use for us: a task-agnostic "see unseen surface" reward is a reasonable proxy when class
  labels are unreliable. That fits our case, since VLM labels are noisy.

### A Dataset for Developing and Benchmarking Active Vision (AVD)
- Ammirato et al., ICRA 2017, arXiv 1702.08272 — https://arxiv.org/abs/1702.08272
- Dense real RGB-D captures on a discrete motion grid in 9 homes. A REINFORCE next-best-move
  policy for instance classification, with six moves (forward/back/left/right/rot cw/ccw), T <= 5.
- Inputs: the dataset's precomputed motion graph (the equivalent of a navmesh).
- Results (ar5iv table, split 1): 5 moves gives learned .45, random .28, "always forward" .29.
  "Forward" is blind: it does not first turn toward the object.
- They also report that detector performance is "severely impacted by object scale,
  occlusion, and viewing direction".
- Use for us: blind forward motion is a weak baseline. The meaningful naive baseline is
  "center the object, then move forward" (see Fixation below).

### Embodied Amodal Recognition: Learning to Move to Perceive Objects
- Yang et al., ICCV 2019, arXiv 1904.04404 — https://arxiv.org/abs/1904.04404
- Embodied Mask R-CNN with a policy that moves the agent near an occluded target in House3D,
  for classification and amodal box/mask.
- Inputs: egocentric RGB and the initial target box. Training uses simulator GT.
- Claim (abs): learned moves beat passive, random and shortest-path moves. I did not
  verify the margins.
- Use for us: occlusion is the case where moving pays off most. A planner should explicitly
  try to reduce occluded mask area.

### Evidential Active Recognition: Intelligent and Prudent Open-World Embodied Perception
- Fan, Liang, Li, Hua, Wu, CVPR 2024, arXiv 2311.13793 — https://arxiv.org/abs/2311.13793
- Treats recognition as sequential evidence gathering (Dirichlet / evidential combination).
  Per-step uncertainty shapes the RL reward. Habitat-based dataset with distance,
  occlusion and visibility difficulty levels.
- Inputs: RGB, a binary target mask, and the observation history. No map. Actions are
  forward 0.25 m, turn 10 deg, look up/down 10 deg, with T = 10.
- Table 1 (top-1, all): learned 64.4, **Fixation (keep target centred) 61.8**,
  single view 60.9, random 53.4. On the hard split: learned 58.0, fixation 56.1, single 49.7.
- Use for us: **this setup is closest to ours.** The learned policy beats a
  centre-the-target heuristic by only ~2.6 pt. Random motion is worse than not moving.
  The Dirichlet evidence fusion matches our belief design.

### Active Open-Vocabulary Recognition: Let Intelligent Moving Mitigate CLIP Limitations
- Fan, Zhou, Xing, Wu, CVPR 2024, arXiv 2311.17938 — https://arxiv.org/abs/2311.17938
- A class-agnostic policy driven by inter-frame and inter-concept similarities, with
  attention-based feature fusion. CLIP stays frozen.
- Inputs: RGB, a target box, and relative pose change. No map. Habitat actions are the same
  as above, with T = 10.
- Habitat (open top-1): learned 28.0, Fixation 26.2, random 25.1, passive CLIP 24.9.
  ShapeNet (free view sphere) shows much larger gains: 53.3 vs 29.6.
- Use for us: in realistic Habitat motion, learned beats fixation by **~1.8 pt**. Most of
  the gain came from fusing views, not from where the agent moved. This is the strongest
  evidence that "centre and approach" is hard to beat.

### Recurrent 3D attentional network / NBV belief formulations (survey context)
- Xu et al., arXiv 1610.04308 — https://arxiv.org/abs/1610.04308
- A depth-only recurrent attention model for joint recognition and NBV. Its related-work
  section summarises the NBV-for-recognition criteria: mutual information between views and
  classes, entropy reduction over hypotheses, and belief-uncertainty reduction.
- Inputs: a depth view sphere around the object.
- Use for us: expected entropy reduction over class beliefs is the textbook criterion.
  It needs an observation model p(label | view), which is unavailable for unseen views
  without rendering or prediction.

### Deep RL / topology NBV for active classification (arm-mounted)
- Collander, Beksi, Huber, arXiv 2103.02789 — https://arxiv.org/abs/2103.02789
  RL with a topology-based information gain on streaming point clouds.
- Korbach et al., arXiv 2110.06766 — https://arxiv.org/abs/2110.06766
  SAC moving an arm toward the views that are most informative for classification.
- Inputs: object-centric views. No navigation constraints.
- Use for us: only the reward ideas transfer (holes/concavities = unexplained surface).
  The kinematics do not.

---

## 2. Verification inside ObjectNav

### CLOSER: Evidence-Aware Active Visual Verification for Embodied Object-Goal Navigation
- Wu et al. (Peking U.), arXiv 2610.28476 (Oct 2026) — https://arxiv.org/abs/2610.28476
- Pipeline: Grounded-SAM, then ByteTrack/SAM tracking, then a LoRA Qwen2.5-VL-7B tri-state
  label per view (yes/no/uncertain), then a CLIP diversity gate into a K=4 cache, then MLP
  fusion. If the result is still uncertain, a **Diffusion-Transformer approach policy on
  mask + depth only** outputs 8 egocentric waypoints (0.25 m apart), and a low-level
  controller turns them into discrete actions. Near the object (<1.0 m mean mask depth)
  it falls back to the larger logit.
- View sufficiency gate: mask/image >= 0.02, box fill >= 0.40, mean mask depth <= 3.0 m.
- Inputs at deployment: tracked mask + depth (no RGB, no map). Training: an expert on a
  **privileged GT navigability map** (Voronoi medial axis, then Bezier smoothing), with
  behaviour cloning / diffusion.
- Approach ablation (initially uncertain candidates, % correct / extra m / fallback %):
  in-place look-around 32.4 / 0 / 0; **straight-line 61.7 / 2.84 / 37.6**;
  STRIVE rule 70.2 / 2.21 / 25.4; deterministic Voronoi-Bezier BC 76.5 / 2.08 / 18.1;
  diffusion 83.6 / 1.69 / 12.7. VLFM HM3D v2 episodes: native FS 19.4 / FR 13.8, CLOSER
  6.4 / 9.3. Without the approach policy FR rises 9.3 to 15.1. Plugging it into 8 ObjectNav
  agents (incl. ApexNav, VLFM) adds +3.6 SR on average.
- Failure modes reported: close-range occlusion fallbacks (19.7% of those wrong), walls
  blocking line of sight, ByteTrack identity switches to a similar neighbour, and
  confusers (daybed vs sofa) staying ambiguous even close up.
- The paper does not define the straight-line baseline. I found no standoff-ring baseline.
- Use for us: **the direct precedent for our module and our constraints.** It suggests
  (a) the naive straight-line policy is beatable by ~20 pt on hard (uncertain) candidates,
  mostly because of fallbacks (too close or occluded); (b) a map-free mask+depth policy can
  be distilled from a privileged-map expert. Our navmesh planners could serve as that expert.

### STRIVE: Structured Representation Integrating VLM Reasoning for Efficient Object Navigation
- arXiv 2505.06729 — https://arxiv.org/abs/2505.06729
- Re-verification viewpoint rule (App. F-B): sample poses every 0.25 m along the planned path
  to the object. Walk back from the object toward the agent, and take the first pose where
  >95% of the target point cloud is visible, the bbox is larger than at the trigger, and the
  width/height are each >= 80% of their original values (scaled by sqrt of the area ratio).
  Then run one VLM check.
- Inputs: a planned path (map), the object's point cloud, and the trigger-view measurements.
- Ablation: removing VLM verification costs 7.5 SR. Re-verification is not isolated.
- Use for us: an excellent **"closest pose that is still un-truncated and fully visible"**
  rule. It can be computed from a local object point cloud plus the straight line. In
  CLOSER's table it beats plain straight-line by 8.5 pt.

### ApexNav: Adaptive Exploration with Target-centric Semantic Fusion
- Zhang et al., arXiv 2504.14478 (2025) — https://arxiv.org/abs/2504.14478
- Keeps long-term memory of the target and similar-looking objects, with multi-frame,
  context-aware confidence fusion. This reduces misdetection-driven stops.
- Inputs: a global semantic/occupancy map.
- Use for us: this is the host agent. Its fusion is passive (no deliberate view-seeking),
  so our module is complementary.

### VLFM: Vision-Language Frontier Maps for Zero-Shot Semantic Navigation
- Yokoyama et al., ICRA 2024, arXiv 2312.03275 — https://arxiv.org/abs/2312.03275
- After detection, it takes the closest object point from mask + depth as the goal, and a
  learned **PointNav policy (depth + relative goal only, no map)** drives there. It stops
  inside the success radius. There is no verification step.
- Use for us: shows a map-free depth+odometry PointNav controller is good enough to reach
  a standoff point in Habitat.

### 3DGSNav
- Zheng et al., arXiv 2602.12159 (2026) — https://arxiv.org/abs/2602.12159
- A detector flags candidates. A VLM chooses viewpoint switches, and novel views are
  *rendered* from an online 3D Gaussian Splatting memory for re-verification.
- Inputs: an online 3DGS scene model (a global map).
- Use for us: equivalent to our SAM3D-rendering idea. It needs a heavy scene model, and its
  benefit over really moving was not isolated (I did not see a table).

### PIGEON: VLM-Driven Object Navigation via Points of Interest Selection
- Peng et al., arXiv 2511.13207 (2025) — https://arxiv.org/abs/2511.13207
- When detector confidence passes a threshold, an RL-fine-tuned Qwen2.5-VL checks
  multi-view snapshots stored in a PoI memory.
- Use for us: passive multi-view verification from past frames. No active view-seeking.

### ReMemNav
- arXiv 2603.26788 (2026) — https://arxiv.org/abs/2603.26788
- A VLM "target verification mode" re-checks a single view before the final approach. On
  failure it returns to exploration.
- Combined rethinking module: SR 42.3 to 52.8 (Qwen3-VL-4B, HM3D v0.2). Verification is
  not isolated.
- Use for us: the reject-then-keep-exploring semantics match our gate.

### Other verification-gated ObjectNav (abstract-level only)
- ZONDA, arXiv 2607.21025 — https://arxiv.org/abs/2607.21025
  Multi-view, multi-scale VLM cross-check that "significantly" reduces FPs. A search
  snippet mentioned a view-quality score (confidence, size, edge truncation). I could not
  confirm it from the abstract.
- ConsistNav, arXiv 2605.09869 — https://arxiv.org/abs/2605.09869
  An FSM executive with persistent candidate memory that suppresses unverified stops.
  +11.4 SR on MP3D (abs).
- DRIVE-Nav, arXiv 2603.28691 — https://arxiv.org/abs/2603.28691
  Cross-frame verification on HM3D-OVON, 50.2 SR (abs).
- Reliable Semantic Understanding for Real-World ZS-ObjectNav, arXiv 2410.21926 —
  https://arxiv.org/abs/2410.21926
  A validator VLM can send the agent back to re-examine before it commits.
- One Map to Find Them All, arXiv 2409.11764 — https://arxiv.org/abs/2409.11764
  A probabilistic semantic map update to mitigate feature-extraction errors. The search
  snippet claimed an FP-rate ablation (26.6 to 19.5), which I did not verify.
- Use for us: the consensus across these papers is "accumulate multi-view evidence before
  stopping". **Only CLOSER and STRIVE move deliberately to get the views.**

### Self-supervised embodied perception (context)
- Move to See Better, Fang et al., BMVC 2021, arXiv 2012.00057 — https://arxiv.org/abs/2012.00057
  Uses multi-view 3D consistency (2D detector + depth + approximate egomotion) for pseudo-labels.
  Navigation is simple/random.
- SEAL, Chaplot et al., NeurIPS 2021, arXiv 2112.01001 — https://arxiv.org/abs/2112.01001
  An exploration policy plus 3D label propagation.
- Embodied Visual Active Learning, Nilsson et al., AAAI 2021, arXiv 2012.09503 —
  https://arxiv.org/abs/2012.09503
- Use for us: mask+depth+odometry is enough to fuse a target's 3D points across frames,
  and to keep tracking identity by 3D consistency rather than by 2D tracker alone.

---

## 3. Map-free local planning

### The Vector Field Histogram: Fast Obstacle Avoidance for Mobile Robots
- Borenstein & Koren, IEEE T-RA 1991 — https://www.cs.utexas.edu/~pstone/Courses/393Rfall11/readings/Borenstein-tra-91.pdf
- Uses a local histogram grid built from range data, reduced to a polar obstacle-density
  histogram. It steers into the low-density sector closest to the goal direction.
- Inputs: range sensing + odometry only.
- Use for us: a natural fit for "turn 30 deg or forward". A 12-sector polar histogram from
  depth columns (floor-removed, 0.1-1.5 m height band) is exactly our action resolution.

### The Dynamic Window Approach to Collision Avoidance
- Fox, Burgard, Thrun, IEEE RAM 1997 — https://www.ri.cmu.edu/publications/the-dynamic-window-approach-to-collision-avoidance
- Searches the reachable velocity set and scores each option on heading, clearance and
  velocity.
- Use for us: with discrete actions, DWA degenerates into scoring a few short action
  sequences (e.g. F, LF, RF, LLF, ...) by clearance and goal progress. A cheap lookahead.

### DD-PPO
- Wijmans et al., ICLR 2020, arXiv 1911.00357 — https://arxiv.org/abs/1911.00357
- Near-perfect map-free PointNav from RGB-D + GPS/compass.

### Is Mapping Necessary for Realistic PointGoal Navigation?
- Partsey et al., CVPR 2022, arXiv 2206.00997 — https://arxiv.org/abs/2206.00997
- 99.8% success with perfect GPS+compass under sensing/actuation noise, without a map.
- Use for us: reaching a chosen standoff pose (relative goal from odometry) does not need a
  map. A pretrained Habitat PointNav policy (as used by VLFM) could be the low-level
  controller, but it is trained for different action sizes (turn 10/30 deg depending on
  the checkpoint). I have not checked which.

### Volumetric NBV information gain (unknown-voxel ray casting)
- Isler, Sabzevari, Delmerico, Scaramuzza, ICRA 2016, "An Information Gain Formulation for
  Active Volumetric 3D Reconstruction" — https://doi.org/10.1109/ICRA.2016.7487527
- Delmerico et al., Autonomous Robots 2018, "A comparison of volumetric information gain
  metrics for active 3D object reconstruction" — https://doi.org/10.1007/s10514-017-9634-0
- These ray-cast candidate views through a probabilistic voxel map and count unknown,
  occluded, or "rear-side" voxels near the object. The gain is traded off against motion cost.
- Inputs: a local voxel map + pose. No global map.
- Use for us: a small (e.g. 4x4x2 m) voxel grid centred on the object, built from our own
  depth, supports "visible-unknown-object-surface" and "occluder-in-the-way" scores for
  candidate poses.

### Pred-NBV: Prediction-guided Next-Best-View
- Dhami, Sharma, Tokekar, IROS 2023, arXiv 2304.11465 — https://arxiv.org/abs/2304.11465
- Completes the partial object point cloud (PoinTr-C) and plans NBV on the predicted
  shape, with an information-gain vs control-effort trade-off. Reports +25% coverage (abs).
- Use for us: shape completion (or SAM3D, which we already run) can predict where the
  hidden surface is, and so which side to go to.

### Semantic-aware NBV for multi-DoF mobile systems
- arXiv 2404.16507 (2024) — https://arxiv.org/abs/2404.16507
- Adds a semantic gain term (target-class voxels) to the visibility gain for
  search-and-acquisition of a specific object.
- Use for us: weight unknown voxels by "near the target mask" rather than counting all of
  them.

---

## 4. VLM as viewpoint planner

### PIVOT: Iterative Visual Prompting Elicits Actionable Knowledge for VLMs
- Nasiriany et al., ICML 2024, arXiv 2402.07872 — https://arxiv.org/abs/2402.07872
- Draws candidate actions on the image and has the VLM pick among them, iterating.
  The authors state performance is "far from perfect".

### VLMnav: End-to-End Navigation with VLMs
- Goetting, Singh, Loquercio, arXiv 2411.05755 (2024) — https://arxiv.org/abs/2411.05755
- Numbered arrows over depth-feasible directions, and the VLM picks one. They note that VLMs
  struggle with continuous coordinates, so the choice must be discrete and depth-filtered.

### VG-AVS: Toward Ambulatory Vision — Visually-Grounded Active View Selection
- Koo et al., arXiv 2512.13250 (Dec 2025) — https://arxiv.org/abs/2512.13250
- Picks the next viewpoint from the current image alone. Off-the-shelf VLMs needed SFT + RL
  to do this well, and the fine-tuned module improves EQA systems (abs; no zero-shot
  numbers seen).

### ActiView: Evaluating Active Perception Ability for MLLMs
- Wang et al., arXiv 2410.04659 — https://arxiv.org/abs/2410.04659
- Over 30 MLLMs show "a significant gap" in active perception (shifting/zooming the
  view to find missing information).

### See, Act, Adapt (VLM as pose controller trained by perception confidence)
- arXiv 2602.23806 (Feb 2026) — https://arxiv.org/abs/2602.23806
- A VLM is SFT'd on rule-based exploration trajectories, then RL'd with frozen-perception
  confidence as the reward. +14-28% on ReplicaCAD perception tasks (abs; baselines
  unclear).

### Explore until Confident (calibration context)
- Ren et al., RSS 2024, arXiv 2403.15941 — https://arxiv.org/abs/2403.15941
- VLM confidence is miscalibrated, which causes premature stops or over-exploration.
  Multi-step conformal prediction calibrates when to stop.
- Use for us (section 4 overall): zero-shot VLM view choice is weak and slow. Where it is
  used, it is restricted to a few depth-feasible discrete options, and good results need
  fine-tuning. Use the VLM for *judging* views, not for steering. Calibrate the stopping
  rule (conformal / held-out thresholds) instead of trusting raw VLM confidence.

---

## 5. Do simple heuristics compete? (evidence summary)

| Paper | Setting | Heuristic | Learned / NBV | Gap |
|---|---|---|---|---|
| Evidential AR (2311.13793) | Habitat, mask, T=10, fwd/turn10/tilt | Fixation 61.8 | 64.4 | +2.6 |
| Evidential AR, hard split | same | Fixation 56.1 (single view 49.7) | 58.0 | +1.9 |
| AOVR (2311.17938) | Habitat, box, T=10 | Fixation 26.2 | 28.0 | +1.8 |
| AVD (1702.08272) | real grid, T=5 | blind Forward .29 | .45 | +.16 |
| CLOSER (2610.28476) | HM3D, uncertain cands only | straight-line 61.7; STRIVE rule 70.2 | 83.6 (privileged-map-distilled) | +22 / +13 |
| Look-ahead (1605.00164) | view grids, no nav | random views 55.6 | 66.0 | +10 |

Reading:
- When the "heuristic" already turns to keep the target centred, learned policies gain only
  **~2-3 pt** (two independent Habitat studies from the same group).
- Blind forward motion and random motion are weak. Random can be worse than standing still.
- The only large gap in a setting like ours is CLOSER's, measured on hard
  (initially uncertain) candidates. Most of it comes from fewer forced close-range
  fallbacks (37.6% to 12.7%), i.e. from *not driving into the object* and from routing
  around occluders. A geometric "closest un-truncated, fully visible pose" rule
  (STRIVE) recovers ~40% of that gap without learning.
- No paper I found compares against a "fixed standoff ring / circle-around" baseline. That
  baseline is unmeasured in the literature, and we should report it ourselves.

---

## Synthesis: planner ideas under our constraints

Ranked by expected value per unit of effort. All of them use only RGB-D, odometry, the
mask/tracker, and collision feedback.

1. **Strong naive baseline first: "Fixate + approach to sufficiency, never closer".**
   - Turn to centre the mask, then step forward.
   - Stop when STRIVE/CLOSER-style sufficiency holds: mask >= 2% of the image, box fill
     >= 0.4, mean depth <= 3 m, no box edge touching the image border, and the predicted
     next-step box still fits inside the frame (from depth + hfov 79).
   - Then hold at that standoff (typically ~1.2-2 m for furniture) and collect views.
   - This is the bar. The literature says learned policies beat it by only ~2-3 pt unless
     occlusion or truncation dominate.
2. **Collision-aware execution layer (needed by every planner).** A rolling egocentric
   occupancy grid (~6x6 m, 5 cm cells), fused over the last N frames with odometry, plus a
   12-sector VFH polar histogram at 30 deg resolution. A DWA-like lookahead scores 2-3-step
   action sequences. Collision feedback marks the forward cell as blocked. This alone should
   remove most failures where the approach path is blocked.
3. **Object-centric candidate poses on a local ring, scored geometrically (local NBV).**
   - Back-project the mask to a 3D object cloud and keep fusing it across frames with
     odometry. Take its centroid and extent.
   - Generate candidates on rings at the standoff distance(s) where the object fills ~15-40%
     of the image without truncation, at bearings every 30 deg.
   - Keep only candidates whose straight-line path is free in the local grid.
   - Score = predicted mask size/no truncation + **occlusion-free ray fraction** (ray-cast
     from the candidate to object points through the local grid, as in Isler/Delmerico)
     + unseen-object-surface gain (object-weighted unknown voxels, as in semantic NBV) −
     path cost in actions.
   - Re-plan every few steps. This targets occlusion and oblique views, which are exactly
     where CLOSER found the gap.
4. **Distil our navmesh planners into a mask+depth policy (the CLOSER recipe).** Our current
   Habitat-navmesh planners become the privileged expert. Behaviour cloning (or
   diffusion) maps mask+depth (+ odometry history) to discrete actions or waypoints, and
   the result is deployable without the navmesh. This has the highest ceiling, and CLOSER
   gained +7 pt from BC to diffusion. It costs data generation and training, and it only
   pays if the expert really beats idea 1 (verify that first).
5. **Confuser-aware view choice.** Pick the view from the confusion pair, not just
   "clearer":
   - picture vs TV: an oblique 30-60 deg view, plus depth thickness/protrusion from the wall;
   - armchair vs couch: a frontal view at a distance where the full width fits, plus
     metric width from depth;
   - vase/flowers vs potted plant: close enough to see pot vs vase and soil, and the
     height/width ratio.
   Depth gives metric size and thickness for free. Add these as extra evidence in the
   Dirichlet belief alongside the VLM. CLOSER notes that confusers stay ambiguous even
   close up, so geometry may help more than motion here.
6. **Evidence-aware stopping and view admission.** Admit a view to the belief only if it
   passes sufficiency and is *diverse* (bearing or distance differs, as in CLOSER's
   CLIP/diversity gate; we could use the odometry bearing to the object centroid). Stop as
   soon as the belief resolves, and calibrate the thresholds on held-out data
   (Explore-until-Confident). This saves budget for the hard cases.
7. **Lost-object and track-switch recovery.** Keep the 3D centroid in the odometry frame.
   If the tracker loses the object, turn toward the stored bearing and re-segment inside
   the projected 3D box. Reject a re-association whose 3D centroid jumps by more than
   ~0.5 m (this covers CLOSER's ByteTrack switch failure).
8. **SAM3D/shape-completion-guided side choice (Pred-NBV-like).** Use the reconstruction
   we already have to predict which side holds the unseen or discriminative surface, and
   bias ring bearings toward it. Speculative: our own d1mv ablation found SAM3D adds
   nothing there, so the expected value is low.
9. **VLM-as-steerer: lowest priority.** At most, offer 3-5 depth-feasible ring candidates
   rendered as numbered arrows (VLMnav-style) and ask which side would disambiguate the
   confusers. The literature (ActiView, VG-AVS, PIVOT) suggests zero-shot quality is poor
   and latency is high with a 27B model.

Baselines a new planner must beat, and by how much:
- **Fixate + approach-to-sufficiency (idea 1).** Literature gap is about 2-3 pt on
  recognition accuracy. Anything smaller is noise with 517 starts (binomial SE ~2 pt).
- **Straight-line approach without a "not too close" stop.** CLOSER: 61.7% vs 83.6% on
  uncertain candidates. This baseline is easy to beat, so beating it is not a contribution
  by itself.
- **Fixed standoff ring (circle at ~1.5 m, sample every 30-60 deg).** Not reported
  anywhere. Measure it, since it is the natural rival to idea 3.
- **No motion (single view / in-place look-around).** CLOSER 32.4%. Random motion can be
  below this (Evidential AR).

Uncertainty notes:
- CLOSER is a single, very recent preprint. Its straight-line baseline is undefined, and its
  expert used a GT map.
- The Habitat active-recognition numbers come from short horizons (T=10) and fine turns
  (10 deg) with tilt. Our 30 deg turns and no tilt make fine re-centring harder, which
  slightly favours explicit geometric planners over pure fixation.
