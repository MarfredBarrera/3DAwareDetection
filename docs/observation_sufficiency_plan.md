# Plan: when is a navigation observation sufficient? (2026-10-28)

## Question
When ApexNav thinks it sees the target, is the current observation good enough for *any*
conclusion (goal / not goal) to be reliable, or is it too uncertain, so the agent should gather
more information first? We want a signal of **observation insufficiency**, not of one detector's
error and not of "another view would fix it".

## Prior results (do not repeat)
- `outputs/apexnav_online/sync_2026-10-25/analysis/sufficiency/`: 3-way logistic gate
  (accept / reject / insufficient at 95% precision per side). Offline (eye labels) AUC 0.89 from
  raw margin alone; online (HM3D labels) 0.64 and every request insufficient. Confident rejects
  are reliable; confident accepts are not (SAM3's confident errors look like goals).
  SAM3D view features and visibility added ~nothing.
- `outputs/apexnav_online/sufficiency_investigation_2026-10-26/`: SAM3D ensemble disagreement
  AUROC 0.44 for error (chance). Dropped.
- `outputs/apexnav_offline/defer_replay_2026-10-26/`: HM3D labels overstate SAM3 errors ~3x;
  mask area / edge gates do not separate errors from goals.

## Labels (nav only)
Unit: one verification request (frame + mask + target). Each request gets an eye label from
the frame alone: `goal`, `not_goal`, or `cant_tell` (frame does not show enough to decide).
HM3D labels are not used as truth. Sources: 122 offline requests (eye labels exist, re-check
for `cant_tell`), 313 sync_2026-10-25 requests, and OVON requests once that run exists.
Contact sheets for labelling; the user spot-checks a random 10% and every `cant_tell`.

## Targets and metrics
A signal `u` flags a request insufficient when `u > t`. On the requests it calls sufficient:
1. **Selective risk, primary**: error rate of SAM3's raw decision on both sides
   (accepted non-goals, rejected goals) vs coverage. Report coverage at <= 5% risk per side.
2. **Decision-maker independence**: the same curve for the VLM's decision. A signal of the
   *observation* should cut risk for both deciders, not only SAM3.
3. **AUROC for `cant_tell`** and **AUROC for SAM3 error** (secondary).
Each signal is also reported **within margin bands** (added value over raw margin), with its
cost per request. Controls: raw |margin| alone, random flagging at equal coverage.
Episode-grouped bootstrap (2000 draws) for all CIs.

## Signals
| family | signals | compute |
|---|---|---|
| free (logged) | SAM3 score, margin, confusable runner-up, open-vocab competitor margin, mask area, edge/bottom contact, depth | none |
| perturbation | SAM3 score and mask IoU spread over crop / scale / flip / small shift jitter (~8 variants) | SAM3, core env |
| temporal | agreement of SAM3 decision and score over the last k frames of the same object cluster | SAM3 on logged approach frames |
| VLM | ApexNav's VLM: "target or runner-up?" plus "can this be decided from this view?"; answer and self-rated sufficiency | ApexNav VLM server |
| SAM3D views | fraction of the 24 candidate views where top-1 flips; top2gap verdict | SAM3D (stored for raw-accepted online requests; new for the rest) |

Open item: temporal needs per-step frames of the approach. Check whether ApexNav logs them;
if not, re-run the labelled episodes with frame dumping (deterministic replay not guaranteed,
so re-label any new request frames).

## Steps
1. Build the request table (one row per request, all logged features, frame/mask paths).
2. Contact sheets and eye labels incl. `cant_tell`; user review.
3. Free signals and controls (no GPU): curves and the margin-band table.
4. GPU passes, one idle GPU, one job at a time, workers stopped afterwards: perturbation,
   VLM, SAM3D views for the missing requests, then temporal.
5. Analysis script and figures; outputs to `outputs/apexnav_offline/sufficiency_<date>/`
   (README, requests.csv, curves PNG, table PNG).

## Decision rule for success
A signal is worth integrating if, at <= 5% risk per side, it raises coverage over raw |margin|
alone with a CI that excludes zero, and it lowers risk for the VLM decision too.
