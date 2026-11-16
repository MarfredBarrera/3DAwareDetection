# Plan: D1 close-range decision over many accumulated views

Proposed file: `~/codes/3DAwareVerification/docs/d1_multiview_close_range_plan.md`. Plan agents can't write files, so the full plan is below for you to save.

## 0. Findings that shape the design (read-only, current code and run data)

1. **Masks come from ApexNav, not SAM3.**
   - `habitat_evaluation.py` calls `gate.observe()` every step with the label-0 detections from ApexNav's own detector (YOLO / GroundingDINO masks), plus world centroids, semantic instances, pose and pitch.
   - SAM3 runs only on the server, once per request: `Verifier._raw` → `classify(image, mask, vocab)`. It keeps only SAM3 instances with IoU ≥ 0.3 against the ApexNav mask.
   - So the per-view mask is free. The per-view SAM3 scores cost about 3–4 s each (median latency of a SAM3-only request is 4.2 s).
2. **Qwen is cheap and SAM3D is the memory risk.**
   - A current Qwen call is 0.37 s for about 1,738 tokens (outlined 640x480 frame ≈ 300 tokens, plus the 1792x672 grid ≈ 1,176 tokens). Qwen-only latency is reported in `verdict.qwen.latency_s`; the full request median is about 8 s, with the SAM3D grid adding the rest.
   - 5 episodes in `videos/sync_d1` have `reconstruct: OutOfMemoryError` inside an "insufficient" verdict.
   - **574's s26 "insufficient" was this OOM, not a Qwen judgement** (qwen.error = OOM, probabilities None). The same holds for both requests in 283.
3. **720 lost good views because of an association rule.**
   - Approach views at s53–55 (range 2.15–1.67 m, area up to 0.25) were dropped by `_closer_to_other`: fusion cluster 9 at (2.67,-5.67) was 0.08 m from cluster 7, i.e. a fragment of the same object.
   - At arrival (s56) there was no detection, so rule C accepted the stale s45 'tv' label.
4. **876: the approach produced no new views.**
   - Only s59–63 exist: range about 2.5 m, about 1,267 px (area ≈ 0.004).
   - Status was aborted after 40 steps (budget used up) with the agent 2.7 m away.
   - Pooling views cannot add information here. Only a fallback rule over the existing views can change this case.
5. **574: SAM3 is confidently wrong at close range.** At arrival s32 it gave bed with margin +0.46 and area 0.154, so the frame counts as "sufficient". Any rule that lets SAM3 decide at close range keeps this false stop. The fix depends on Qwen seeing several clear views.
6. **693:** the bank would hold s65, s66 (about 2 m, tiny) and s73 (1.37 m, area 0.0106). SAM3 on s73 says potted plant with margin 0.51.
7. **Approach outcomes in the live run so far (28 episodes with approaches):**
   - Early views: 18 insufficient → keep approaching, 10 decided by SAM3, 3 decided by Qwen.
   - Arrivals: 9 of 12 went through rule C (not re-detected), 2 through rule D, 1 through SAM3.
   - Aborted: 4.
   - **Rule C is the main arrival path, so a view bank matters more than the arrival frame.**
8. **Records are enough for an offline replay.** Each `records/<outcome>/<epi>/` has:
   - `steps.jsonl`: pose, camera_pitch, detections (score, label_idx, world_centroid, semantic_instances, mask file) and object_fusion clusters;
   - `rgb/%06d.jpg` for every step (`record.flush_on=[all]`);
   - `masks/<step>_<k>.png`.
9. **Planner (no C++ change needed):**
   - `finishVerification` with `approach && !arrival && state==1` means "accepted on the way": it replans toward the forced cluster with no budget. So accepting after a failed approach already works.
   - If the approach ended with the agent stuck (`verify_stuck_`), the episode ends anyway; the verdict only goes to the log.
   - Status 2 lumps "no object path" with "target lost". Without a C++ change the gate cannot tell them apart (see risk 7).
10. **Shared state with the live run (hard constraints):**
   - `run_gate_e2e.sh` starts a new `habitat_evaluation.py` for every episode, so any edit to `perception_gate/*.py` reaches the live 200-episode run at its next episode.
   - `run_strategy.sh` hard-codes port 8770, and its cleanup runs `pkill -f perception.server` / `perception.worker` / `habitat_evaluation.py` / the planner in the shared `3dawareverification-dev-1` and `apexnav` containers.
   - **A second closed-loop run cannot run alongside the live run (live run is on GPU 2).**

## 1. Views: source, assignment, quality, selection, budget

**Source.** The `observe()` items already built in gate.py: rgb, binary mask, detector score, area_fraction, quality, clipped, centroid, camera, pitch, semantic_instances, and the per-step cluster snapshot. No extra perception per step.

**Per-cluster view bank** (gate.py, `state["bank"]`, a list of references to item dicts, no copies):
- **Seeded** when an approach is issued (`_issue_approach`): every EvidenceStore item that matches the cluster (rule below). This covers pre-approach views, including the one that triggered the approach.
- **Filled** in `observe()` for each cluster with a pending approach: every new item that matches it, from `issued_step` on, with no novelty thresholds. This avoids losing views to EvidenceStore eviction (8 per cell plus the newest).
- **Capped** at `multiview_bank` = 24: drop the lowest-quality item, never the newest. Memory is at most 24 × 0.9 MB per cluster, and items are shared with EvidenceStore.

**Match rule** (`_bank_match(item, cluster_id, pending)` → distance or None):
- `d = min(|c − pending.object|, |c − item.clusters[cluster_id]|)`, which must be ≤ `match_distance_m` (0.75).
- **Competitor exclusion is relaxed.** Another cluster nearer the item than `d` excludes it *only if* that cluster's centroid (at the item's step) is more than `multiview_sibling_m` = 0.5 m from this cluster's reference. Fragments such as 720's clusters 7, 8, 9 count as the same object.
- `_approach_view` (the early trigger) keeps its current, stricter rule unchanged.

**Quality** (unchanged, so it stays comparable with offline commit review): `view_quality(score, area, clipped)` = min(1, area/0.10) · max(score, 0.05) · (0.7 if clipped). Range is logged per view and used for ordering.

**Selection** (`select_views(bank, k)`, pure function):
1. Drop near-duplicates: same (step, index), or camera within 0.3 m and bearing within 15° of an already chosen view. In 574, s22/s30 and s23/s29 are exact pose repeats.
2. Pick greedily by quality. Slot 1 is forced to the best view with range ≤ the median range, so the closest clear view is always included.
3. k = `multiview_max_views` = 6.
4. Send views ordered nearest first.

**GPU and latency budget:**
- **Qwen:** 6 outlined 640x480 frames ≈ 1,800 visual tokens + about 300 text tokens, about the same as today's frame + grid (1,738). Transient activations stay well under 1 GB with int8 and `logits_to_keep=1`.
- **No SAM3D call on this path**, so peak memory is lower than the current insufficient path and the reconstruct OOMs cannot occur at close range.
- **Server hard limits:** at most 8 images, each resized to fit 640x480. Anything else is a 400.
- **SAM3 per-view scores** are computed only when Qwen is not decisive, at about 3 s per view, so at most about 20 s at a forced decision (`timeout_s` is 120). The gate caches per-view scores by (step, index) and sends them back so they are not recomputed.
- Guard limit 45.5 GB is unaffected.

## 2. Joint decision

**Qwen multi-image prompt** (`qwen.prompt_views`, `PROMPT_VERSION_VIEWS = "multiview-v1"`). Content is interleaved: `"Image k (about X.X m away):"` followed by the image, for k = 1..n, then:

> Images 1 to {n} are robot camera observations of the SAME physical object, outlined in red in every image, taken from different distances and directions while the robot approached it (Image 1 is the closest). The outline comes from an automatic detector: it may cover only part of the object or spill onto neighbours; judge the single physical object the outlines mark consistently. The requested category is {target!r}. Combine the evidence from all images: a clear, whole, close view outweighs small, distant or cut-off views. Ignore other objects in the room.

This is followed by the existing couch/armchair sentence and the existing A/B/C block, unchanged, including "Lack of evidence for the target is not evidence that it is wrong."

- Each image is `outlined(image, mask)` (same red outline as today).
- Variant `crop_small` (offline ablation only): for views with area < 0.02, also send a crop around the mask bbox, padded by the object size, at least 160 px, resized to at most 448 px (about 200 extra tokens). It targets tiny-object cases like 876 and 693.
- Variant `forced2` (offline ablation only): at forced decisions, a two-way A/B prompt, deciding by p_c/(p_c+p_w) ≥ τ. This is the fallback if Qwen keeps answering C (see risk 1).
- Target definitions in the prompt ("a framed picture/poster/mirror is not a tv; a pouf/ottoman is not a bed") are a separate variant. Judge it on held-out episodes only, because it is fitted to the eye errors.

**SAM3 aggregate** (pure function `select.aggregate_views(per_view, target)`):
- Per view: scores_k, margin_k = s_k[target] − max over other categories, weight w_k = min(1, area_k/0.10) · (0.7 if clipped).
- Views where SAM3 matched nothing (every score 0, the IoU gate missed) are excluded.
- Returns M = Σ w_k·m_k / Σ w_k, V = fraction of views whose top-1 is the target, n_used, and label = target if M > 0, else the top category by summed score.

**Combination (in the gate; the server returns Qwen's status and the SAM3 aggregate):**

| mode | Qwen correct / wrong | Qwen insufficient, error or invalid |
|---|---|---|
| early (status 4) | accept / reject | keep approaching (state 3) |
| arrival (status 1; also replaces rule C and rule D) | accept / reject | SAM3 fallback: accept iff M ≥ τ_arr (0.0) and V ≥ 0.5, else reject |
| failure (status 2 or 3) with ≥ `multiview_min_views` (2) distinct-step views | accept / reject | SAM3 fallback: accept iff M ≥ τ_fail (0.15) and V ≥ 0.5; reject iff M ≤ −τ_fail; else reject (as today; the cluster stays reopenable by the existing re-check rule) |
| failure with < 2 views, or empty bank | reject (as today) | reject |

**SAM3 does not shortcut at close range.** The sufficiency gate is not applied to multi-view requests. The approach exists because SAM3 was unsure, and 574 shows SAM3 "sufficient" yet wrong. A config switch `multiview_sam3_shortcut` (default false) exists only for the offline ablation.

All thresholds (τ_arr, τ_fail, V, k, sibling_m) are config values, calibrated offline on replayed sync_d1 approaches against the eye labels. No learned components are added.

## 3. Code changes

Implementation hygiene:
- Develop in git worktrees, or keep every change additive and **default-off**. The live run imports gate.py again at each episode.
- `/verify` behaviour stays unchanged and PROTOCOL stays 1.
- Coordinate with Codex on gate.py, test_approach.py, server.py, client.py, qwen.py and worker.py.

### 3DAwareVerification

- **`perception/qwen.py`**
  - `prompt_views(target, distances, forced2=False)`.
  - `QwenJudge.judge_views(images, target, distances=None, forced2=False)`: same logits/ABC extraction as `judge`, interleaved content, returns the same dict plus `n_images`.
  - Optional `crop_around(image, mask, pad=1.0, min_px=160, max_px=448)`.
- **`perception/worker.py`**
  - `QwenServer.judge_views(images, masks, target, distances=None, crop_small=False, forced2=False)`: applies `outlined` per view, inside `_on_gpu`.
- **`perception/select.py`**
  - `aggregate_views(per_view, target)` as defined above (pure, unit-tested).
- **`perception/client.py`**
  - `MULTIVIEW_MAX = 8`.
  - New dataclass `ViewsVerdict(status, source, qwen, per_view, aggregate, accepted)` with `summary()`.
  - `Verifier.verify_views(images, masks, target, distances=None, vocabulary=None, sam3=("fallback"|"always"|"never"), cached_scores=None, crop_small=False, forced2=False)`:
    1. Requires selector d1, else ValueError.
    2. Starts the Qwen worker only (`self._start("qwen")`, not SAM3D).
    3. Calls `qwen.judge_views`; failures and invalid answers become insufficient with `qwen["error"]`, the same handling as `_d1`.
    4. If sam3=="always", or "fallback" and the status is insufficient, classifies each view whose (step, index) has no cached scores.
    5. Returns per_view [{step, index, scores_top5, margin, area, clipped, label}] and the aggregate.
  - `d1_accept_min` applies to `correct`, as today.
- **`perception/server.py`**
  - `POST /verify_views` with `{protocol: 1, request_id, target, vocabulary?, sam3, crop_small?, forced2?, views: [{image, mask, input_elevation, step, index, range_m, scores?}]}` → `{ok, verdict: ViewsVerdict.summary()}`.
  - `parse_views_request` returns 400 for: no views, more than 8, any image not exactly 640x480 (do not resize: offline and live inputs must match), an empty mask, or a selector other than d1.
  - The log record gains `n_views`, status, source and the aggregate.
  - `settings["multiview"] = True` when selector is d1, so the gate can check the capability through `/health`.
- **Tests**
  - `tests/test_perception.py`: `D1MultiviewTests` with `FakeWorker`:
    - Qwen correct makes no classify calls (sam3=fallback);
    - insufficient classifies each view, cached scores are skipped, and the aggregate values are checked;
    - a Qwen error becomes insufficient;
    - SAM3D is never started;
    - more than 8 views or a non-d1 selector raises;
    - `/verify_views` round trip and 400 cases;
    - `/health` reports `multiview: true`.
  - `tests/test_qwen_verification.py`: `prompt_views` (image count, captions, version; forced2 drops C) and `aggregate_views` (weights, clipped penalty, all-zero views excluded, empty input → n_used 0).

### ApexNav (branch d1-approach)

- **`perception_gate/client.py`**: `verify_views(url, views, target, sam3="fallback", vocabulary=None, request_id=None, timeout=120.0, **variants)`. Views are dicts {rgb, mask, input_elevation, step, index, range_m, scores?}, PNG-encoded as `verify` does today.
- **`perception_gate/gate.py`**
  - **DEFAULTS:** `multiview: False`, `multiview_max_views: 6`, `multiview_bank: 24`, `multiview_min_views: 2`, `multiview_sibling_m: 0.5`, `multiview_dup_distance_m: 0.3`, `multiview_dup_angle_deg: 15`, `multiview_arrival_margin: 0.0`, `multiview_failure_margin: 0.15`, `multiview_vote: 0.5`, `multiview_failure: True`, `multiview_sam3_shortcut: False`, `multiview_variant: None`.
  - **Validation:** multiview needs `approach`. At ping, `server.settings.multiview` must be true, else raise ValueError.
  - **`__init__`:** `self._verify_views = verify_views or client.verify_views`, injectable like `verify` for tests.
  - **`observe()`:** after `self.evidence.add`, call `self._bank_views(step, new)` (multiview only), which appends matching items to `state["bank"]` of clusters with a pending approach and applies the cap.
  - **`_issue_approach`:** seed `state["bank"]` from EvidenceStore through `_bank_match`.
  - **New:** `_bank_match`, `select_views` (module-level pure function), `_sam3_cache = {(step, index): scores}`, and `_judge_views(record, state, step, mode)`:
    1. build the views;
    2. call the server, with the same `on_error` and consecutive-error handling as `_judge`;
    3. store the returned per-view scores in the cache;
    4. apply the table in §2;
    5. set `record["views"]` = [{step, index, range_m, area, clipped, quality, camera}], `record["verdict"]`, `record["model"]` ("qwen_multi" / "sam3_multi"), and `record["policy"]`, e.g. `multiview,arrival,qwen_wrong` or `multiview,failure,insufficient,sam3_accept(M=0.21,V=0.67,n=5)`;
    6. no verdict cache by (step, index);
    7. `state["last_verdict"]` is not used.
  - **`_after_approach`, when multiview is on:**
    - UNREACHABLE / ABORTED → `_judge_views(..., "failure")` if `multiview_failure` is set and the bank has ≥ min_views distinct steps, else `_approach_failed`;
    - EARLY → `"early"`, where the bank includes `pending["early"]`;
    - ARRIVED → `"arrival"`, where the bank includes the arrival-step views (no thresholds);
    - empty bank → `_approach_failed("not_redetected")`;
    - `_not_redetected` and rule D are not reached in multiview mode.
  - **Logging:** `_log` writes `views/v{k}_s{step}_{index}.jpg`, the matching masks, and `decision.json`.
  - **Counts:** `multiview_qwen`, `multiview_sam3`, `multiview_failure_decided`, `multiview_views` (sum).
- **`perception_gate/test_approach.py`**: new `MultiviewTests` using fakes for `verify` and `verify_views`:
  1. sibling-cluster views (720-like, a cluster 0.08 m away) are banked, while a competitor 1 m away still excludes;
  2. arrival without an arrival-step detection uses the bank and makes no rule-C call; Qwen wrong → REJECTED;
  3. arrival with Qwen insufficient: M/V thresholds accept and reject at the boundaries;
  4. aborted with ≥ 2 views: Qwen correct → ACCEPTED; insufficient with |M| < τ_fail → REJECTED; < 2 views → REJECTED with no server call;
  5. early: insufficient → APPROACH (same approach), correct → ACCEPTED;
  6. `select_views`: duplicates dropped, k cap, nearest first, best close view in slot 1;
  7. bank cap keeps the newest;
  8. multiview off → every existing test unchanged;
  9. config validation (needs approach; server without the capability raises);
  10. the SAM3 score cache is sent back on the second call.
- **`perception_gate/replay.py` (new, offline tool):** `python -m perception_gate.replay --episode-dir videos/sync_d1/gate/epiN --url http://3dawareverification-dev-1:8771 --out videos/d1_mv_replay/epiN [--set verification.multiview=true ...]`.
  - Builds `VerificationGate` (log_dir = out) and calls `start_episode` from meta.json.
  - For each step: `observe(step, rgb jpg, masks, label_idx, score, world_centroid, semantic_instances, pose, camera_pitch, object_fusion)`.
  - At each step that appears in the original `decisions.jsonl`, calls `handle()` with the reconstructed request. Approach re-requests take approach_id, status and steps from `record.approach`; status names map back to 1–4.
  - **Stops scoring at the first decision that differs from the original**, because the rest of the trajectory is counterfactual.
  - Outputs one line per decision, old vs new.
  - Runs inside the `apexnav` container (cv2, basic_utils).
- **`scripts/run_strategy.sh`:** new strategy `d1mv` = d1 plus `verification.multiview=true` in eval_args (optionally `MV_VARIANT=` → `verification.multiview_variant`). No server flag is needed.
- **C++:** none.

## 4. Test protocol

**A. Unit tests** (no GPU):
```
cd ~/codes/ApexNav && python -m unittest perception_gate.test_approach perception_gate.test_gate
docker exec -w /workspace 3dawareverification-dev-1 sam3d-env core python -m pytest tests/test_perception.py tests/test_qwen_verification.py -q
```

**B. Offline replay** (can run before the live run ends, on a free GPU and a separate port, started by the user or implementer, not by me):
```
docker exec -d 3dawareverification-dev-1 bash -lc "cd /workspace && sam3d-env core python -m perception.server --host 0.0.0.0 --port 8771 --gpu <free> --selector d1 --log-dir outputs/apexnav_online/d1_mv_replay/server"
```
1. **Fidelity check:** replay with `multiview=false` on 574, 720, 876, 693. Original SAM3 statuses should reproduce (records are JPEG, live inputs were PNG; Qwen may differ slightly).
2. **Main run:** replay with `multiview=true` on the 28 episodes that have approaches (114, 151, 194, 257, 259, 283, 292, 303, 37, 396, 405, 427, 459, 510, 538, 574, 587, 637, 681, 693, 715, 717, 720, 731, 847, 876, 915, 920), plus 614 and 586 as no-approach sanity checks.
3. **Ablations:** `sam3=always` to log M and V everywhere, `crop_small`, `forced2`, k ∈ {4, 6, 8}.
4. **Score** against `eye_v4/eye_labels.tsv` and `ground_truth.is_correct_instance`. Targets:
   - 574 → reject; 720 → reject; 876 → accept or at least "Qwen correct", and report whether the decision came from the fallback; 693 → accept;
   - 717, 151, 283, 637 must stay accept (eye S).
5. Calibrate τ_arr, τ_fail and V on the episodes other than these 10, and report both groups.
6. **Kill criterion:** if Qwen multi-view answers C on 574 (and 720), switch to the `forced2` variant before any closed-loop run.

**C. Closed loop** (only after the live `videos/sync_d1` run finishes; see §0.10):
```
GPU=<free> ./scripts/run_strategy.sh d1mv d1_multiview_2026-11-05 true 574 720 876 693 614 717 637 151 586 283 888
```
- Output goes to `videos/d1_multiview_2026-11-05/gate/epiN`, a new directory.
- Check:
  - episodes with no approach (586, 614, 888) have identical trajectories and decisions to sync_d1 (determinism under sync_step);
  - outcomes for the 4 target episodes;
  - no regression on 717, 151, 283, 637;
  - per-decision latency and `gpu_guard.log` peak (< 45.5 GB);
  - `views/` folders for eye review.
- Then run the 200-episode set as a new run.

## 5. Risks / what would make it fail

1. **Qwen's "insufficient" bias.** Single-view p(C) was 0.96–0.9999. If multi-view still says C, the SAM3 fallback takes over, and SAM3 says 'bed' for the pouf (574 stays wrong) and 'tv' for the print (720 likely wrong too). Mitigation: the `forced2` variant; decide offline first.
2. **The sibling relaxation can merge different adjacent objects** (pouf next to the real bed, plant on a vanity). Qwen may then see a different object outlined in some views and accept. Mitigation: sibling only within 0.5 m and only views within 0.75 m of the issued point; the replay logs each view's semantic instance ids so this can be audited.
3. **Close-range ApexNav masks can be partial, clipped or multi-object**, so the outlines mislead Qwen. Clipped views are down-weighted but not removed.
4. **876 cannot be fixed by pooling** (no new views). It is fixed only by the failure fallback (M ≥ 0.15) or `crop_small`. Accepting after a failed approach can cause false stops on far, small look-alikes. Watch the failure-accept rate in the replay.
5. **Accept after status 2 "no object path":** the planner replans toward an object it cannot reach (possible loops or stuck). Separating it from "target lost" needs a C++ change (a new viewpoint_status), which is deliberately left out. Option: set `multiview_failure` to apply only to ABORTED.
6. **Latency:** the forced SAM3 fallback is about 3 s × k. Early requests become cheaper (Qwen only, no SAM3D).
7. **Interference with the live run:**
   - gate.py edits reach the next live episode, so changes must be default-off and the file must always import; use a worktree.
   - Server edits apply on the live run's next server (re)start.
   - `run_strategy.sh` cleanup kills all perception servers and episodes in the shared containers, so no concurrent closed-loop run.
   - Codex may be editing the same files.
8. **Overfitting** of the prompt, thresholds and definitions to 4 eye-labelled episodes. Mitigation: calibrate on the other approach episodes and keep `PROMPT_VERSION_VIEWS` versioned.
9. **Not addressed:**
   - 888 (mirror accepted at the initial far request). Possible phase 2: use multi-view at initial requests when the bank has ≥ 2 views.
   - SAM3D OOM at initial requests (5 episodes), which wrongly creates "insufficient" verdicts. Suggest, separately: on a reconstruct OOM, retry Qwen with `d1_evidence=input` instead of returning insufficient.

### Critical files for implementation
- /home/marfred/codes/ApexNav/perception_gate/gate.py
- /home/marfred/codes/ApexNav/perception_gate/test_approach.py
- /home/marfred/codes/3DAwareVerification/perception/client.py
- /home/marfred/codes/3DAwareVerification/perception/qwen.py
- /home/marfred/codes/3DAwareVerification/perception/server.py

Also touched: `/home/marfred/codes/3DAwareVerification/perception/worker.py`, `/home/marfred/codes/3DAwareVerification/perception/select.py`, `/home/marfred/codes/ApexNav/perception_gate/client.py`, the new `/home/marfred/codes/ApexNav/perception_gate/replay.py`, and `/home/marfred/codes/ApexNav/scripts/run_strategy.sh`.
## Addendum (2026-11-04, user): SAM3D moves to close range (strategy d1mv)
1. SAM3 + mask heuristics (sufficiency) -> sufficient: SAM3 raw decides.
2. Insufficient -> Qwen three-way on the frame only (`d1_evidence=input`, no SAM3D): correct/wrong decide, insufficient -> approach.
3. Approach while filling the per-cluster view bank (sections 1-3 above).
4. Close-range decision (arrival / early / failure fallback with >= min views): SAM3D on the best close view -> 24-view grid; one Qwen request with the k past/close outlined images + the grid. The prompt says the reconstruction is automatic from one image and may be inaccurate or misleading from some angles; it is supporting evidence, and real images win on conflict. The SAM3 aggregate fallback is unchanged. Ablation `multiview_grid` on/off. A SAM3D OOM/error falls back to images-only, never to insufficient.
