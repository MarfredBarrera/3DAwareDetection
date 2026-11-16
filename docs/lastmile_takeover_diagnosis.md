# Last-mile takeover diagnosis — 2026-11-13

Compared the 1,000 matched episodes in `outputs/apexnav_online/takeover1000_2026-11-13/takeover1000.tsv` against `cmp.json` and the original `videos/{takeover600,fresh400_takeover}/gate/epi*/verification/takeover.jsonl` logs. These are reviewed manual 1 m labels, not simulator success. The November 12 score.txt is preliminary and differs from the November 13 reviewed results.

| Arm | Success | False stop | False rejection | Other | SPL % |
|---|---:|---:|---:|---:|---:|
| Baseline | 741 | 54 | 0 | 205 | 38.1 |
| Pipeline/native navigation | 750 | 20 | 5 | 225 | 37.5 |
| Pipeline/takeover | 697 | 42 | 5 | 256 | 36.0 |

Takeover loses 73 pipeline successes and gains 20: net −53 (−5.3 percentage points; paired exact McNemar p=2.92e-8). The losses are 53 other failures and 20 false stops. Gains are 16 former other failures and four former false stops. All five false rejections remain unchanged.

## What goes wrong

1. **The integration often supplies a point rather than an object cloud.** Of 817 takeovers, 572 use `cluster_centroid` (one point), and 245 use `approach_cloud`. In `habitat_evaluation.py:559`, a direct accept constructs a singleton at the transformed cluster centroid and current agent height. Only the external-approach branch passes its cloud. Of the 73 losses, 39 use the centroid and 34 the approach cloud; this is a concrete integration mismatch, but not the sole cause.

2. **STOP measures proximity to the perceived target, not arrival at an accessible real target.** `nav_ddppo.object_distance` projects the perceived points into x/z; radius STOP runs before map/goal evaluation. It does not test floor, wall separation, current semantic confirmation, or orientation. For a singleton, it measures centroid distance. For a cloud, the original default measures the nearest point and is vulnerable to stray/merged points. Of the 73 losses, 37 are radius stops (24 other failures, 13 false stops). Examples from existing eye review: ep35 cloud distance 0.731 m versus real couch 1.03 m; ep40 0.615 m versus real couches 1.48 m; fresh ep744 0.303 m versus rocking chair 1.77 m. Fresh 508/511/528 stop inside while the chairs are across a glass door. These are perceived-goal/arrival errors, not simply a weak learned controller.

3. **Failure to progress becomes terminal.** Fifteen lost successes end with stagnation STOP (11 other failures, four false stops); three end with policy STOP, all false stops. The original stagnation threshold permits stopping up to 1.5 m from the perceived points after 40 steps without sufficient goal-distance improvement. Policy STOP can survive when the snapped-goal distance is small even if actual object distance is large. Eps454/675 illustrate stopping across a wall. Eighteen further lost successes consume the remaining episode budget; 16 of these use a centroid target.

4. **First accept is a permanent commitment.** `run_takeover` at `habitat_evaluation.py:388` takes the entire remaining budget, never returns navigation control, and turns navigator STOP into REACH_OBJECT. `Takeover.obs` keeps supplying the original cloud. Thus later observations do not replace or reverify that target through the gate. A mistaken accept or unreachable target cannot be abandoned in favor of another instance through the native loop. This explains how accepting a wrong object becomes a new false stop where native navigation eventually succeeded; the logs establish the transitions, though they do not isolate each native recovery mechanism causally.

5. **Navigation starts with a fresh local map.** `DDPPONavigator.reset` discards map and recurrent state. Snap candidates only tighten toward the object, and an old snap can survive changed connectivity. Collision patches in `cur` mode persist; recovery and probing are off by default. Existing `lastmile/_tmp/REPORT_slideoff_gap.md` documents invisible navmesh boundary wedges and partial forward steps under sliding-off dynamics. This supports a physical explanation for some caps, but does not prove all 18 lost-success caps are wedges. Individual action/displacement traces are needed for that attribution.

## Does it rescue the intended population?

Among 34 baseline false stops removed by the native pipeline, the pipeline already turns 19 into success and leaves 15 as other failures. Takeover preserves 17 of those successes, loses two, and rescues **none of the 15 remaining failures**. It does rescue four of the 20 false stops still present in the native pipeline, but creates 27 new false stops from episodes previously labeled success or other failure. The five existing false rejections never reach takeover, so a module triggered only on accept cannot recover them directly.

## Version and measurement caveats

Current `lastmile/live.py` contains STOP_FIX (0.5 m radius, percentile distance, grace period, tighter stagnation). Recorded losses include radius stops at 0.7–0.8 m and stagnation at 1.4–1.5 m, so those logs do not reflect those tighter settings. `videos/takeover600/code` does not contain a lastmile snapshot; do not infer the run's behavior from today's file alone. The already-present fix is not validated by this 1,000-episode result. Singleton centroid distances are unaffected by percentile filtering.

The reviewed rule requires within 1 m of a real accessible target; simulator viewpoint success is different. Twelve of the 73 reviewed losses are simulator successes. Some near-threshold cases are scoring-sensitive, but the aggregate includes wrong objects, walls/floors, and caps and cannot be explained solely by the 1 m threshold. The existing 256-other-failure decomposition is not a causal decomposition of the 73 regressions.

## Recommended next experiment

Keep native navigation as the default. Evaluate a bounded fallback only after native progress fails, with handback on failed arrival or exhausted local budget. Before another full run, replay the 73 regressions and 20 gains using the exact accepted poses and budgets, with successful episodes as regression controls:

- Pass the accepted instance's actual fused cloud on both accept paths; record source and point count.
- Treat stagnation/budget exhaustion as failed local navigation and hand back, rather than asserting arrival.
- Revalidate instance identity, current visibility/accessibility, floor, and robust distance before final STOP; update target evidence while navigating. Keep simulator ground truth exclusively in evaluation.
- Compare the existing STOP_FIX against the original settings separately from handback and cloud-source changes. A tighter radius alone may turn early stops into caps.
- Log all effective environment switches and snapshot lastmile code; record per-step poses, goals, blocked forwards, and stop decisions. Repeat sampled-policy comparisons across seeds.

No navigator implementation or experiment configuration was changed during this investigation.
