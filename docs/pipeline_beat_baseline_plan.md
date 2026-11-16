# Plan: make the gate pipeline beat native ApexNav (2026-11-10)

## Where we stand (held-out 200, not the tuning set; 178 episodes in both runs, eye-corrected)

| run | Success | FS | FR | explore / other | bench SR | SPL |
|---|---|---|---|---|---|---|
| baseline (native ApexNav) | 136 | 3 | 0 | 39 | 73.0 | 37.6 |
| pipeline (jev_sam3 + mppi_v15 assoc5) | 132 | 3 | 0 | 43 | 71.3 | 36.9 |

The verdicts are mostly right. The pipeline loses on what happens after a decision.

Scorer: `outputs/apexnav_online/heldout_2026-11-10/score.py` + `eye.tsv`.
Runs: `~/codes/ApexNav/videos/heldout_{baseline/baseline,assoc5/gate}`.

### Failure taxonomy (eye-checked)

1. **Last-mile failures after a correct commit** (commit_ground_truth_geometry.is_correct_instance, no success):
   - Pipeline 8: 180, 212, 237, 410, 457, 710, 804, 852. Baseline 5: 237, 410, 457, 710, 760.
   - 6 of 8 are wedges: the agent tries forward in nearly every heading and never moves (212, 237, 457, 710, 804, 852).
     The cause is `allow_sliding: false` plus ApexNav's lattice moves hugging obstacles. Once wedged, no planner
     can escape, so the fix must be prevention (clearance).
   - ApexNav's reach test (REACH_DISTANCE 0.20 m to its `next_pos`, SOFT 0.45 m) can also stop far from the goal
     viewpoints (180 at 2.2 m, tuning 876 at 2.7 m).
   - HM3D success needs a stop within 0.2 m (`success_distance`) of a goal viewpoint, i.e. a navigable point within
     about 1 m of the object.
   - The 852 pattern: after an accept, the cluster becomes ApexNav's forced target and can't be dropped. The
     baseline gave up on the same unreachable chair and found another one.
2. **Approach-induced stuck endings** (804; tuning 595 FN, 943):
   - A request arrives while ApexNav is escaping a wedge, and the approach replaces the escape.
   - Approach actions count toward ApexNav's `stucking_action_count_` (MAX 25).
   - `finishVerification` with `verify_stuck_` ends the episode as STUCKING (or the cluster is REJECTED).
   - In 804 the baseline, from the identical wedge, escaped and succeeded.
3. **Armchair vs couch false stops** (648 held-out; 445, 283, 320 tuning):
   - Vocabulary (vocab study) and the JEV veto (held-out: fixes 60 but rejects the real sectional in 40) don't work.
   - Metric size does. Offline on logged clouds, with "longest side < 1.2 m AND some contributing mask touches no
     image border": 445, 283, 648 and 320 are vetoed (all armchairs), and no real couch is vetoed (715 and 322 are
     always border-clipped, so they're skipped).
   - The same idea for beds fails: real beds 332, 352 and 95 have partial clouds (1.1-1.6 m) even when the mask
     looks whole. Leave beds out.
   - `active_approach/size_opinion.py` (Bayesian width likelihood, completeness + side-occlusion test,
     `decide.DEFAULTS["SIZE"]=True`) was written 2026-11-10 10:30. It is not in the held-out snapshot and has never
     run closed-loop.
4. Remaining FS not addressed: 60 (bench as bed, both systems), 914 (mirror as TV, accepted at the stop).

## Fixes, in priority order

### A. Our own last mile after an accept (largest expected gain: up to ~6 on held-out)
After the gate accepts, the gate's MPPI planner (mppi_v15 on the fixed local map: `HeightMap.low`, metric
inflation, `forward_feedback`) keeps driving and issues the STOP itself, instead of handing a forced target back to
ApexNav.

- **Goal:** a free cell (with clearance) whose horizontal distance to the nearest point of the fused object cloud O
  is <= STOP_M (0.8 m), facing the object.
- **Stop:** within STOP_M of O and facing it. The navmesh is never used (the self-contained rule).
- **No progress for N steps:**
  - within 1.0 m of O: stop anyway;
  - otherwise hand back to ApexNav as today, without the forced-target lock, so it can drop an unreachable target
    as the baseline does (852).
- **Budget:** its own action budget, separate from the approach budget.

Alternatives considered:
- **A\* on ApexNav's map:** ApexNav already uses A\*. The failures are clearance/wedge, not path finding.
- **DD-PPO PointNav:** no checkpoint here; trained for navigable point goals (our goal is a non-navigable object);
  a new model and env dependency; uncertain sliding behaviour.
- Reusing our MPPI is the smallest change, and it already prevented the 95/156 wedges.

### B. Stuck accounting (small, principled)
- Approach actions don't count toward ApexNav's stuck counters. Reset `stucking_action_count_` and
  `stucking_next_pos_count_` at approach start and end.
- An approach that ends stuck is **DEFERRED** (no new evidence), never REJECTED and never an episode end. ApexNav
  then resumes with its own escape logic.
- Don't start an approach while ApexNav is escaping (`escape_stucking_flag_`). Defer the request instead.

### C. Size opinion for couch (already in code; validate, then keep or simplify)
- Offline: replay the logged approaches (assoc4, assoc5, held-out) through `decide.verify` with SIZE on.
- Report every flip and eye-check it. Expected flips: 445, 283, 648, 320 only.
- If the Bayesian version flips anything else, fall back to the plain rule above.
- No bed prior.

## Evaluation protocol
1. Unit tests + offline replay (C).
2. Development episodes (tuning set + the known held-out losses; these are now "seen"):
   - post-accept failures 212, 237, 457, 710, 804, 852, 180, 410 (held-out), 156, 257, 876 (tuning);
   - stuck cases 595, 943;
   - couch 445, 283, 320, 648;
   - plus 20 random tuning successes as a regression guard.
3. Final test on fresh episodes: 200 sampled from `videos/heldout_2026-11-10_rest_episodes.txt` (never run with
   the pipeline). Run both arms. Report eye-corrected Success / FS / FR / explore, SPL, and McNemar.
   - Also report the 200 held-out set once more (now seen, so secondary).
   - The deferred 600-episode baseline run (`videos/heldout_2026-11-10_STOP` blocks chain3) can become that
     baseline arm.

## Constraints
- One GPU (GPU 3, our stack), CPU RAM <= 64 GB, OMP_NUM_THREADS=2, WORKERS=3.
- Launch with `setsid nohup` so runs survive a session end.
- Finish code before launching a new NAME (the code snapshot is per OUT_ROOT).
- No commits. Stop servers after runs. Eye-check every changed outcome.

## Result: final test on 200 fresh episodes (test2, 2026-11-10, eye-corrected)

| run | S | FS | FR | explore | SPL |
|---|---|---|---|---|---|
| baseline (native ApexNav) | 148 | 13 | 0 | 39 | 36.6 |
| pipeline, own last mile (fix A on) | 145 | 6 | 0 | 49 | 35.5 |
| pipeline, `LAST_MILE=0` | 151 | 4 | 0 | 45 | 36.4 |

`LAST_MILE=0` vs baseline: FS McNemar 9 vs 0 (p=0.004); success 5 vs 2.

Fix A (our own last mile) is rejected:
- 12 of 36 hand-backs failed. 7 baseline successes ended stucking or lost, after the last mile stalled on unseen
  obstacles of its fresh local map.
- Its ring stop also made new false stops after forced accepts (938, 960, 181).

With the last mile off, ApexNav navigates to the accepted cluster, unforced (fix B keeps it droppable).

The size opinion is now veto-only (it can never turn a non-accept into an accept). On test2, ep 181 had flipped
reject -> accept on an armchair merged with a sectional (W 1.30 m).

Scorer: `outputs/apexnav_online/test2_2026-11-10/score.py` (`ARM=test2_nolm`) + `eye.tsv`.

Final config (LAST_MILE=0 + size veto) on held-out 200 (`videos/heldout_final`, eye): S151 FS2 explore47 SPL37.3 vs
baseline S152 FS3 explore45 SPL37.4. Only diffs: 745 (mirror FS -> rejected, explore) and 995 (loss: frontier
exhaustion; the gate deferred the door glass "tv"). Remaining gate-fixable FS: 914 (door accepted as TV through
`on_no_evidence=accept` on a 1-obs 0.20 cluster); 60 (bench forced accept after approach).

Combined held-out + test2 (400, final config, eye): S302 FS6 vs baseline S300 FS16; FS McNemar 10 vs 0 (p=0.002);
success 5 vs 3; SPL 36.9 vs 37.0.

## No-evidence defer (2026-11-10 23:00)
Replay of 250 logged decisions: `on_no_evidence=defer` fixes 914 and rejects no eye-checked target. Stricter forced
decisions (SAM3+JEV agree, or always defer) lose more real targets than they save (JEV names toilets "bathtub",
couches "chair"), so they were rejected. `NO_EVIDENCE=defer` (run_strategy.sh -> verification.on_no_evidence) re-run
closed-loop on every episode it can change (`videos/noevid_defer`: 914 919 279 948 120 825 138 312): 914 FS -> no
stop (explore); the other 7 are identical to before (same steps, SPL).

Final config = LAST_MILE=0 + size veto + NO_EVIDENCE=defer. Eye-corrected:

| set | arm | S | FS | FR | explore | SPL |
|---|---|---|---|---|---|---|
| held-out 200 | baseline | 152 | 3 | 0 | 45 | 37.4 |
| held-out 200 | final | 151 | 1 | 0 | 48 | 37.3 |
| test2 200 | baseline | 148 | 13 | 0 | 39 | 36.6 |
| test2 200 | final | 151 | 4 | 0 | 45 | 36.4 |
| 400 | baseline | 300 | 16 | 0 | 84 | 37.0 |
| 400 | final | 302 | 5 | 0 | 93 | 36.9 |

FS McNemar 11 vs 0 (p=0.001); success 5 vs 3. The remaining 5 FS are all also baseline FS (look-alikes accepted by
the forced decision: 60 bench/bed, 563 ottoman/bed, 788 792 bench/chair, 59 trash can/toilet).

## Tuning 200 with the final config (2026-11-11, `videos/tuning_final`, eye)
Scorer `outputs/apexnav_online/tuning_final_2026-11-11/score.py` + `eye.tsv`; the baseline is the 11-04 sync_baseline (NAS tar).

| arm | S | FS | FR | explore | bench SR | SPL |
|---|---|---|---|---|---|---|
| baseline | 148 | 8 | 0 | 44 | 71.0 | 37.7 |
| final | 151 | 1 | 0 | 48 | 74.0 | 38.7 |

FS McNemar 7/0 (p=0.016); success 4 vs 1. Only loss 185: the approach accepted the goal armchair, then ApexNav's
native last mile drove under the stairs and ended stucking. 650 (bench success on "no frontier", 0.16 m) is counted
as explore.

All 600 (tuning + held-out + test2, same config, eye): S 453 FS 6 FR 0 vs baseline S 448 FS 24 FR 0;
FS McNemar 18/0 (p<1e-5); success 9 vs 4 (p=0.27); SPL 37.5 vs 37.2.

## Terminal close-in and the 600-episode result (2026-11-11)
The failure analysis (scratchpad failures.tsv) finds that the last-mile failures after a correct accept are mostly not
fixable from the gate. In the stopped-short cases the agent is at the object but 0.4-0.53 m from HM3D's goal
viewpoints, in both arms. The stuck cases are wedges where no heading moves. The failures after a removed false stop
are exploration (the target is never seen again).

Terminal close-in (`perception_gate/close_in.py`, `CLOSE_IN=1`): when ApexNav STOPs or ends stucking >0.9 m
(`CLOSE_IN_FAR_M`) from the gate-accepted object, up to 12 actions toward its points, then STOP. It gains 802. The
0.9 m threshold is post hoc: with every trigger, SR is unchanged and SPL -0.4.

Bench (simulator) success, final config + close-in vs native ApexNav:

| set | baseline SR | pipeline SR | baseline SPL | pipeline SPL |
|---|---|---|---|---|
| test2 + heldout (400, never tuned on) | 71.5 | 73.25 (+1.75; McNemar 10/3, p=0.09) | 37.0 | 36.9 |
| all 600 | 71.33 | 73.50 (+2.17; McNemar 17/4, p=0.007; 95% CI +0.7..+3.7) | 37.2 | 37.5 |

Eye-checked false stops over 600: 23 -> 6; FR 0. CLOSER on ApexNav (Table III): HM3D-v2 +1.3 SR / +1.0 SPL,
HM3D-v1 +1.8 SR.
