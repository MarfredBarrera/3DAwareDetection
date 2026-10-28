# Plan: why a SAM3D view helps SAM3 on Objaverse but not in ApexNav on HM3D

Written 2026-10-27 by a planning agent (read-only; nothing run). Constraint: at most one GPU at a time
in total; GPU work only after the OVON run, sequential, servers stopped when idle.

## 0. Checked while planning (CPU, read-only)
1. **Different metrics.** `outputs/upright/study.sqlite` `selections` (`detector_gap` = old name of top2gap):
   any-FP 29.9% (input) -> 18.5% (top2gap), but top-1 77.4% -> 77.8% (re-checked 2026-10-27). `identity` alone
   gives any-FP 24.3% at top-1 72.7%. The view lowers wrong-class scores; it does not improve accept/reject.
2. **Same view error on true objects, different baseline.** Objaverse top2gap view wrong on 22% of true objects,
   input render 22.6%. HM3D view vetoes ~19-28% of true goals, but the raw frame (with context) accepts ~93%
   (pooled_2026-10-25: 275/295). HM3D compares frame-with-context vs render; Objaverse render vs render.
3. **Vocabulary.** Objaverse 12 classes (no cushion, clothes, towel, curtain, picture, potted plant, tv);
   HM3D `HOUSEHOLD` 28. SAM3 scores prompts independently (`detect_many`), so dropping classes from stored scores
   is exact.
4. **Selectors differ.** `study/selectors.py` top2gap has no elevation band; `perception.select.top2gap_pick` keeps
   0-90°. Objaverse inputs span 0-60° evenly; HM3D inputs are ~all 0°.
5. **Richest HM3D data:** `outputs/apexnav_offline/pooled_2026-10-25/` — 659 requests, 596 eye-labelled
   (295 goal / 301 other / 63 ambiguous), 24 views with scores + PNGs. The view earns its keep on couch (removes
   40/42 wrong accepts, mostly armchairs); on bed/chair/toilet it mostly costs recall. The 200-episode closed loop
   had few such errors left after raw SAM3 (base rate).
6. `sync_2026-10-25/analysis/sufficiency/requests/` (313) has inputs, masks, 24 view scores, camera+centroid
   (distance), no view PNGs (maybe in `/ICONAS/users/marfred/ApexNav/sync_runs_2026-10-25.tar`).
7. Ensemble u only in `gate_verified_2026-10-22` (131) and `active_replay/replay.jsonl` (313; AUC 0.44 for SAM3 wrong).

## 1. Data
| id | source | gives | labels |
|---|---|---|---|
| OBJ | `outputs/upright/study.sqlite` | 110 obj x 60 views, 24 candidates each, scores, `visible_px` | GT |
| POOL | `outputs/apexnav_offline/pooled_2026-10-25/` | 659 HM3D requests, 24 views + PNGs | eye (596) |
| GV | `outputs/apexnav_offline/gate_verified_2026-10-22/` + `apexnav_online/verified_2026-10-22/request_labels.tsv` | 131 requests, PNGs, u | eye |
| SYNC | `outputs/apexnav_online/sync_2026-10-25/analysis/sufficiency/requests/` | 313 closed-loop requests | HM3D + some eye |
| DEF | `outputs/apexnav_online/defer_2026-10-26/`, `apexnav_offline/defer_replay_2026-10-26/requests.csv` | defer outcomes, features | eye |
| ACT | `apexnav_offline/active_replay/replay.jsonl` | u for SYNC requests | HM3D |
| LVIS | `outputs/lvis_uncertainty/study.sqlite` | mesh F vs u | GT |

Outputs: HM3D `outputs/apexnav_offline/sam3d_gap_2026-10-28/<inv>/`, Objaverse `outputs/upright/sam3d_gap/<inv>/`;
README + `_<name>.py` + CSVs + figures each. Rules from `perception.select`; plots from `study.figures` and
`scripts/apexnav_fp.py` helpers.

## 2. Hypotheses (ranked by information per cost)
- H1 metric mismatch (any-FP vs top-1) — strong.
- H2 context/baseline strength: render loses context, raw HM3D frame is much stronger — strong.
- H3 base rate & error type: view fixes only shape-confusable errors, rare on HM3D after raw SAM3 — strong.
- H4 vocabulary attractors (clothes, towel, lamp, cushion, curtain) — medium.
- H5 input degradation (truncation, distance, occlusion, mask) — medium.
- H6 reconstruction failures (black slabs, flat, missing support) — medium.
- H7 selection/elevation band — medium-low.
- H8 how the verdict is used (veto vs defer, asymmetric cost) — analytic.
- H9 GPU nondeterminism noise floor (~1-2%).
- H10 scoring asymmetry (raw needs mask IoU >= 0.3; renders count any detection) — low-medium.
- H11 ensemble u measures shape agreement, not label correctness — confirm only.

## 3. Investigations
### Tier A — CPU only
- **A0 Objaverse nav replay (H1, H7, H11), ~30 min.** Gate rule on Objaverse: raw accept iff top-1 == T; view verdicts
  with banded `top2gap_pick` and unbanded study top2gap (+ identity/random/oracle); rules raw, raw AND view,
  unsure->view, defer. Report q_e = P(view no | error), q_g = P(view no | goal) by class, elevation, runner-up.
  Check reconstruct fingerprint is gravity-up. -> `outputs/upright/sam3d_gap/nav_replay/`.
- **A1 Error types (H3), ~1 h + eye pass.** Tag HM3D raw-accept errors (shape-confusable / flat / not an object /
  fragment) and Objaverse (true, pred) pairs; compare mix and q_e per type for chair/couch/bed/toilet.
- **A2 Decomposition (H2, H5-H7), ~30 min.** Target top-1 for goals and errors at raw -> identity (180,0) -> VQF ->
  banded top2gap -> any-view (oracle recall) -> majority, both domains.
- **A3 Vocabulary ablation (H4, H10), ~20 min.** Re-judge stored view scores with fixed subsets: Objaverse-equivalent;
  HOUSEHOLD minus {clothes, towel, cushion, curtain, blinds, picture, lamp}; goals + CONFUSABLE. Confusion table of
  vetoed goals.
- **A4 Conditions (main question), ~1-2 h.** Pre-fixed features (mask area, bbox height, edge contact, solidity,
  components, distance, raw margin, runner-up, confusable, unsure, category, u, view-label stats, A6 render stats);
  outcome +1 FP caught / -1 goal vetoed / 0; pre-fixed bins with episode-cluster bootstrap CIs; mirror on Objaverse.
  No learned fusion.
- **A5 Decision EV (H8), ~1 h.** EV_veto(x) = p q_e L_FP - (1-p) q_g L_veto (L_defer for defer); losses from SYNC/defer
  logs; break-even base rate p* per rule and bin.
- **A6 Render image statistics (H6), ~1-2 h.** Dark fraction, saturation, silhouette area/ratio/solidity, components
  over the 24 PNGs; vetoed goals vs kept goals vs caught errors; calibrate on Objaverse (+ LVIS F-score).
- **A7 Ensemble u across domains (H11), 20 min.**
- **A8 Noise floor (H9), 15 min** from `pooled_2026-10-25/repeat/` and the GV overlap.

### Tier B — GPU, after OVON, one GPU, sequential, servers stopped after each
- **B1 Decontextualized HM3D input (H2), ~0.2 GPU-h:** masked object on neutral background (and blurred-background
  variant) through SAM3 with HOUSEHOLD.
- **B2 Objaverse with HOUSEHOLD vocabulary (H4), ~1.5-2 GPU-h:** detect stage only on chair/sofa/bed/toilet at 0° and
  their candidates (~11.7k images), new config `configs/upright_household.yaml` importing renders/reconstructions
  (small `study/config.py` option).
- **B3 HM3D-style degradation of Objaverse inputs (H5), pilot ~4.3 GPU-h:** 39 objects x 4 azimuths at 0°; clean,
  truncation 50%, distance 1/4, occluder 40% (624 inputs); full grid (~12 GPU-h) only if the pilot shows an effect.
  `configs/upright_degraded.yaml`.
- **B4 optional, ~2.6 GPU-h:** regenerate SYNC view PNGs if not in the NAS tar.

## 4. Visualizations
- **V1 per-request strips:** raw frame + mask | crop | identity render | 8x3 candidate montage (top-1 label + margin,
  green/red border, top2gap and VQF picks boxed, out-of-band greyed) | top-5 bars raw vs pick | outcome badge.
  Sets: view saved an FP (all), view fixed nothing (all), 24 vetoed goals (4/target, seed 0), 12 controls.
  `figures/strips_{saved,missed,vetoed,control}.pdf`.
- **V2 outcome galleries:** 4x6 crop+view thumbnails sorted by mask area, annotated with area, edge contact,
  margin, runner-up, view label.
- **V3 Objaverse vs HM3D pairs** per class (chair, couch, bed, toilet), matched on raw-margin tertile, plus error pairs;
  input, identity, pick, 24-view margin heatmap.
- **V4 class-mean margin heatmaps** (8x3) per class and domain.
- **V5 condition distributions** (area, edge, solidity, margin, u) by group.
- **V6 usefulness map:** per category, area tertile x margin bin, n / q_e / q_g / net, coloured by EV.
- **V7 accuracy waterfall** raw -> decontext -> identity -> VQF -> top2gap -> any-view, both domains.
- **V8 label flow:** target -> view top-1 confusion for vetoed goals vs Objaverse.
- **V9 render-quality panel:** extremes of dark fraction / thinness as montages with outcomes.
- **V10 degradation ladder** (after B3).
- **V11 EV curves** vs base rate with domain operating points.

## 5. Order
A0 -> A2 (+V7) -> A3 (+V8) -> A8, A1 -> A4 (+V5, V6) -> A5 (+V11) -> V1-V4 for user review -> A6 (+V9), A7 ->
after OVON: B1 -> B2 -> B3 pilot (+V10) -> full B3 / B4 only if justified. CPU work <= 4-8 processes, SQLite read-only.
Each step writes its README before the next starts.

## 6. Open questions for the user
1. Restate the Objaverse headline as top-1/decision metrics with any-FP secondary if A0 confirms H1?
2. Is a semantics-fixed vocabulary reduction for the view step acceptable if A3 shows attractor vetoes?
3. ~30-60 min of eye labelling (error types, occlusion/truncation tags)?
4. B3 context compositing: needed, or is truncation/distance/occluder enough?
5. OVON: confirm the run logs all 24 view scores (needed to repeat A4/A5 there without GPU) — decide before it starts.
6. List the NAS tar for SYNC view PNGs, or are POOL/GV images enough?
7. Count SPL/time in L_defer, or only success?
