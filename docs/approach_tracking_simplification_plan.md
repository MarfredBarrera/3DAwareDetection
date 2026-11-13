# Approach object tracking: one identity rule (plan, 2026-11-09)

Goal: the gate-driven approach (jev_sam3 + mppi_v15, `~/codes/ApexNav/perception_gate/approach.py`
`LiveApproach`) must clearly beat the baseline (native ApexNav: S148 FP8 on 200 HM3D episodes) and
jev_sam3 (S148 FP3 FN2). Today mppi_v15 sam3det is S148 too: it only trades errors for "explore".

## Failures on the hard set (sync_jev_sam3_mppi_v15_sam3det, 19 episodes, eye-checked frames)

| ep | target | outcome | cause |
|---|---|---|---|
| 693 | potted plant | FN (baseline S) | **association**: at step 0 SAM3's only candidate within 1 m (x/z only) was the wall sconce above the orchid; it was taken although it does not overlap the tracked orchid. Its points entered the cloud, the reprojected outline followed the lamp, and later SAM3 picks were the lamp. The closest views (steps 8-9) were never judged because SAM3 skips frames that are not "new" for the pose gate. |
| 259 | potted plant | explore (correct reject; baseline FP) | **association**: request = pedestal with figurine; SAM3 picked the curtain behind (0.84), so the curtain fed the cloud; then MPPI turned away and the outline was empty for 10 steps. |
| 304 (earlier) | bed | fixed in sam3det | **association**: the projected outline of the bed foot fed the decision, and the SAM3 IoU >= 0.3 gate dropped the whole-bed mask. |
| 445 | couch | FP (baseline S) | **planner stop at 2.3 m**: an armchair through a doorway. v15 returned planner_done after 1 action; all seeds were valued equal (0.71), so the view score saturates. Hypothesis: the cloud from the detector mask bleeds onto the background at depth edges and is inflated, so sat(size) saturates (memory: model size about 3x the real size). The forced verify then used only far views: SAM3 couch .79 vs chair .75. |
| 63 | bed | FP (= baseline) | classification: a daybed/window seat; JEV and SAM3 both say bed. Out of scope. |
| 463 | bed | explore | ApexNav's last mile on the right object. Out of scope. |

So three of the five perception errors come from **which mask is the object**. Five rules decide it today:
- the ApexNav mask, matched by its centroid within 1 m;
- the reprojected cloud;
- SAM3 candidates within 1 m in x/z, picked by IoU with the outline (no minimum);
- the `object_pf` EMA;
- the cloud, which also grows from reprojected outlines (self-reinforcing bleed).

## Plan: one identity, one association rule

1. **Object = the fused cloud O** (5 cm voxels). Seed it from the request frame's detector mask. Seed from the
   bank masks too if the bank items carry depth; check this.
2. **Reference outline T** each frame = `local_nav.track_mask(O)`, which is depth-tested, so it is 3D-consistent.
   Before O exists, T = the request mask.
3. **One association rule for every detector mask**, both ApexNav's per-step target masks and SAM3 `/detect`
   candidates:
   - score = overlap coefficient |M ∩ T| / min(|M|, |T|), with ties broken by IoU;
   - accept the best if the score is >= `ASSOC_MIN` (0.5), else no detection this frame.
   - The overlap coefficient is 1 for the whole bed ⊃ the bed-foot outline (304) and 0 for the sconce above the
     orchid (693) and the curtain behind the pedestal (259).
   - This replaces `MATCH_M` centroid gates, `DETECT_RULE` variants and the x/z-only distance.
4. **O grows only from accepted detector masks.** Erode each mask by ~3 px before back-projection to drop
   depth-edge flying pixels. Never add reprojected points. `object_pf` becomes O's centre once O exists; keep it
   only for the find phase.
5. **SAM3 runs on every frame facing the object with |T| >= SAM3_PX.** Drop the pose-gate "fresh" condition, so
   the closest views reach the SAM3 verification (`self.seen` = all usable views). The JEV belief keeps its pose gate.
6. **Then 445:** with a cleaner cloud, check whether v15 still stops at 2.3 m. Log the model size vs the real
   mask size. Change the planner only if needed, and through its cost, not with a trigger (user rule: no if/else
   hacks).

Expected: 693 -> S, 259 stays a correct reject, 304 stays S, 445 maybe -> S; no regressions.

## Evaluation
- Unit tests: `perception_gate` pytest (68). Add tests for the association rule (sconce / curtain / bed-foot
  synthetic masks).
- Hard set, closed loop:
  `EPISODES="37 63 87 95 114 259 292 304 396 445 459 463 510 574 650 681 693 915 943"`,
  new NAME. Score with `outputs/apexnav_online/gate_mppi_v15_2026-11-09/_classify.py <run>`. Eye-check every
  changed episode.
- Then all 82 (`gate_v15_episodes.txt`) and compare against the baseline and jev_sam3 (S, FP, FN).

## Update (18:40): results of the first rule, and the next step

- Hard set with the containment rule (sync_..._assoc, automatic labels): S10, FP3, FN4 (before: S13, FP2, FN1).
  The overlap coefficient rewards containment:
  - it picked a 26k px lamp mask over a 4k px outline, 41k px of clothes over 3.9k, and the whole bed over the
    couch-like bench at its foot (304; the bench is the GT couch);
  - it also picked a 487 px chair fragment over the 8.5k whole chair.

  IoU picks the right mask at every step where the two rules differ. Switched to IoU >= 0.3, same single rule
  (run sync_..._assoc3).
- **How approaches end** (sam3det + assoc, 46 approaches): the JEV belief commits in 6. About 75% end by the
  planner stopping, and the forced fused-SAM3 verdict then decides.
- **Approach verdicts vs automatic GT** (detmask, 80): 10 false accepts, 3 false rejects; 8 of the false accepts
  are forced at a planner stop.
- Example, 445: the best rollout improved the cost by 0.084 < DELTA 0.1, so MPPI stopped at 2.3 m with
  P(consistent) 0.59. The verdict was then SAM3 couch .79 vs chair .75, accepted.
- **Next: an uncertainty-aware stop.** MPPI's stop tolerance DELTA (and the action cost C_ACT) is what an
  extra action is worth relative to a better view. It should shrink while the evidence is undecided:
  DELTA_eff = DELTA · min(1, P(consistent) / P_STAR), with the JEV belief as the uncertainty signal (the chosen
  "needs a clearer view" signal). No trigger: the same cost, rescaled by the information state that LiveApproach
  already has (pass it in Observation.meta). Evaluate offline in Stage B (jev_sam3 judge) and on the closed-loop
  hard set.

## Update (19:10): IoU rule results (sync_..._assoc3)

Hard set: S14 FP2 FN1 explore2. Before: sam3det S13 FP2 FN1; baseline S13 FP4; jev_sam3 S13 FP3 FN2.
- Fixed: 693 and 463 (now S); 87 and 304 are S again.
- The belief commits in 6 of 26 approaches (3 of 23 before).
- 95: the verdict was right both times, but MPPI wandered 32 actions, out of view for about 14 of them, before
  ApexNav got stuck.
- 943 is a new FN: the TV was never re-detected after the hand-off. The find phase turned L/R 12 times while
  already facing it, and "not_found" rejects.
- 63 (a daybed) and 445 (stops at 2.3 m) are still FP.

Next round:
- (a) **No evidence -> defer.** A not_found or empty approach returns the object undecided, not as a reject: the
  absence of a view is no evidence that it is the wrong object.
- (b) **Uncertainty-aware MPPI stop:** DELTA_eff = DELTA·(1-P_STAR)/(1-p).
- (c) **Snapshot of perception_gate/ and active_approach/ per run**, so edits during a run do not leak into it.

## Update (20:05): defer + uncertainty-aware stop + code snapshot (sync_..._assoc4)

Hard set: S14 FP2 FN0 explore3. Before: assoc3 S14 FP2 FN1; baseline S13 FP4; jev_sam3 S13 FP3 FN2.
- 943 is now deferred, not falsely rejected; ApexNav got stuck later, so it counts as explore.
- 445: MPPI now drives to 1.6 m (DELTA_eff 0.049 at p .59) and gets a clear view of the armchair (eye-checked).
  JEV says chair in every frame, but SAM3 still ranks couch above chair (.83 vs .77). This is a classifier
  confusion, not a planning problem.
- Next: all 82 unsure episodes (RESUME into assoc4) for the 200-episode comparison.

## Update (22:15): full 200 episodes (sync_..._assoc4, automatic labels)

| | S | FP | FN | explore |
|---|---|---|---|---|
| baseline (native) | 148 | 8 | 0 | 44 |
| jev_sam3 | 148 | 3 | 2 | 47 |
| **mppi_v15 assoc4** | **150** | **2** | **0** | 48 |

Changed vs jev_sam3 (eye-checked where verdicts were involved):
- 463 FN->S, 650 FP->S (5 chairs correctly rejected, 1 deferred), 731 explore->S (pool table requested as chair, rejected), 303 explore->S (no approach; nav variance).
- 259 FN->explore (correct reject of the pedestal).
- 95, 156 S->explore: verdict correct (accept), then ApexNav stuck in the last mile (156: toilet accepted at 1.9 m in a tight bathroom; 95: MPPI wandered 32 actions).
- Remaining FPs: 63 (daybed, both models say bed) and 445 (SAM3 couch > chair at 1.6 m; JEV says chair).

## Update (22:50): offline JEV x SAM3 verdict fusion (CPU, logged assoc4 + pooled runs)
- Current forced/belief verdict (assoc4, 75 verdicts, automatic GT): 9 false accepts, 1 false reject. 6 of the 9 have SAM3 and JEV both naming the target (wrong instance or GT error); no fusion can change them.
- Additive and product-of-experts fusions (SAM3 margin + λ·JEV margin; Σ log softmax(SAM3/T) + w·log JEV): every λ, w > 0 is worse. On couch vs armchair, JEV is wrong as often as SAM3 (650, 598, 304).
- A mild JEV veto (keep a SAM3 accept only if J = mean p(t) - max other mean p ≥ -0.25) gives FA 9→7, FR unchanged, flat over δ 0.2-0.3. The 3 flips were eye-checked as correct rejects of armchairs (283, 320, 650). Support is only 3 objects.
- 445 cannot be fixed this way: J = -0.14, and catching it needs δ ≤ 0.13, which also rejects a real bed (332).
- Decision: not adopted for now. A veto was rejected earlier, and the gain is small. Scripts are in the session scratchpad `fus/`.

## Update (23:55): post-approach wedging fix (sync_..._assoc5)
- Root cause, from an offline replay of 86 approaches against a recomputed navmesh:
  - With no sliding, a colliding step leaves the agent on the navmesh edge. Often no heading on the 30° lattice can then move (156, 95), so ApexNav's escape fails.
  - The local map called walls seen above 1.6 m free, because occupancy used the cell's max height. Inflation (0.2 m) left the 0.224 m ring open. Partial forward moves were not counted as bumps.
- Fix (Python only):
  - occupancy from the max height below the body top (`HeightMap.low`);
  - metric inflation of radius + half a cell diagonal;
  - every short forward step counts as a bump at its stop position (`LocalMap.forward_feedback`), in all phases.
  - Tests: 82 pass.
- Re-run of 22 episodes: 95 and 156 are S again (eye-checked), with no regressions. assoc4 with these two swapped in gives S152 FP2 FN0. The full 82-episode re-run is in progress.
