# Plan: ApexNav closed-loop verification-gate study on HM3D-OVON val_unseen

Written 2026-10-27 by the planning agent; user decisions: val_unseen, 200 episodes, 6 strategies
(baseline, SAM3_gate, +SAM3D_veto top2gap/vqf, SAM3D_defer; SAM3_naive dropped 2026-10-27); be mindful of shared
CPU/GPU and never leave VLM/perception servers idling on a GPU.

## 0. Summary

Replicate `outputs/apexnav_online/sync_2026-10-25` (HM3D-v2 val, 200 episodes, sync_step, paired
bootstrap, eye-verified outcomes) on a 200-episode stratified sample of **OVON val_unseen** (3000 episodes,
49 categories, the same 36 HM3D val scenes) with **5 strategies**: baseline, sam3, top2gap, vqf, defer
(user decisions 2026-10-27: SAM3_naive dropped; Step 0 Option A approved; alias table approved).

Unchanged: planner branch and sync_step, gate config values, server flags, vocabulary (`HOUSEHOLD`),
embodiment (camera 0.88 m, 79° hfov, 640x480, 0.25 m / 30°), 500 steps, success = stop within 0.2 m of a view
point, sample design, blocks of 25, report format.

Changed:
1. A small OVON dataset loader + config in ApexNav (habitat 0.3.3 ObjectNav-v1 cannot load OVON as-is).
2. Labels passed verbatim (no MP3D mapping); LLM answers pre-generated and frozen per dataset.
3. A minimal open-vocabulary extension of `perception.select` (name aliases + a few flat categories), with a
   unit test proving HM3D-v2 verdicts are unchanged (no HM3D rerun needed).

Cost: ~29-40 GPU-h sequential on one GPU, plus ~1.5 h for LLM answers and the smoke test.

## 1. Facts checked while planning (CPU only)

### 1.1 OVON loading in habitat 0.3.3
- Top-level split shells lack `category_to_task_category_id` -> `AttributeError` in `ObjectNavDatasetV1.from_json`.
- Episodes carry `children_object_categories` -> `TypeError` in `ObjectGoalNavEpisode(**episode)`.
- Per-scene content files have empty category maps; `ObjectGoalSensor._get_observation_space` would `max()` an
  empty sequence (ApexNav ignores `objectgoal`, but `objectnav_hm3d` builds the sensor).
- Goal `object_id` is a string `"<category>_<semantic id>"` (e.g. `freezer_2`); HM3D-v2 uses ints. The suffix is
  the line id in `<scene>.semantic.txt` (10476/10476 val_unseen goal ids exist, 9103 with the same name).
  Unfixed, `semantic_goal_overlap` never matches and `check_failure` labels correct commits `false positive`.
- With an in-memory subclass + fixes, all 3 val splits load (3000 eps each, 0 missing scenes, 0 goal-less eps).
  Scenes resolve under `data/scene_datasets/hm3d/val/...`; `hm3d_v0.2` is a symlink to `hm3d` on the NAS, so
  assets and semantics equal HM3D-v2. habitat renumbers `episode_id` per scene file, so use iterator indices
  (`test_epi_num`) as for HM3D.
- `val_unseen/` holds `val_unseen_hard.json.gz` (no `val_unseen.json.gz`); `val_seen_synonyms/` holds
  `val_unseen_easy.json.gz`.
- HM3D-v2 val uses the same 36 scenes; mean geodesic start distance 5.72 m vs 5.80 m.

### 1.2 ApexNav label, LLM and detector path
- `resolve_label` would silently rewrite OVON labels in the MP3D map (`table`, `cushion` -> pillow,
  `picture` -> "picture | framed photograph"); `picture` is in val_unseen, so OVON turns the mapping off.
- `read_answer` appends a queried answer to the file before validating; with `llm_client: none` a missing label
  writes `label: []` then crashes. `data/ollama` on the NAS is empty (qwen3:8b never pulled).
- Detector is chosen automatically (`vlm/utils/get_object_utils.get_object`): YOLOv7 for COCO classes (val_unseen:
  book, microwave, refrigerator, vase), else GroundingDINO with target + LLM similar labels. GroundingDINO keeps a
  target detection only if the phrase equals the label exactly; multi-word labels are at risk.

### 1.3 Repo state
- ApexNav `defer-gate` = 726bbc9 (= `sync-step`) + uncommitted defer gate in 10 files. The built planner
  (10-26 21:06) is newer than those sources and was used for the defer run; no C++ change is planned.
- 3DAV `apexnav-gate` @ c378279: `naive` and `--view-when unsure` are uncommitted in perception/ + tests.
- Drivers are in gitignored `ApexNav/videos/sync_runs/` (`run_strategy.sh` knows all 6 strategies). It kills the
  perception server only on the normal path and never stops the VLM tmux session.
- HM3D min/episode: baseline 1.38, sam3 1.47, vqf 1.71, top2gap 2.02, defer 2.01 (naive ~1.45 assumed).

## 2. Step 0: base for the uncommitted defer/naive work (needs the user's OK)

Never stash, reset, `checkout --` or `git add -A` in either repo.

- **Option A (recommended, needs approval):** commit the 10 ApexNav defer files on `defer-gate`, then
  `git switch -c ovon-eval`; commit the 4 3DAV files on `apexnav-gate`; OVON changes follow as separate commits.
- **Option B:** `git switch -c ovon-eval` from `defer-gate` carrying the uncommitted changes; commit only OVON
  files by path; no 3DAV commits. Write `versions.txt` (HEADs, `git status --short`, planner mtime) and
  `apexnav_uncommitted.patch` / `perception_uncommitted.patch` into the output folder at run start.

Check the planner binary is newer than `src/planner/**`; if not, stop and ask.

## 3. Code changes

### 3.1 3DAV `perception/select.py` and `perception/client.py`
Problem: competitors are `HOUSEHOLD + target` and accept requires top-1 == target, but HOUSEHOLD contains
broader names/synonyms of OVON targets (pillow/cushion, refrigerator/appliance, plant/potted plant, ...).

```python
ALIASES = {
    "cushion": ("pillow",),
    "picture": ("photo", "photo mount"),
    "potted plant": ("flowerpot", "decorative plant"),
    "chest of drawers": ("dresser", "nightstand"),
    "cabinet": ("nightstand",),
    "table": ("nightstand",),
    "counter": ("island",),
    "window": ("window glass",),
    "shower": ("shower glass",),
    "stool": ("ottoman", "footrest"),
    "bed": ("bedframe",),
    "exercise equipment": ("exercise bike",),
    "appliance": ("refrigerator", "freezer", "microwave", "dishwasher", "coffee machine", "boiler"),
}

def same_object(a, b):
    a, b = a.lower(), b.lower()
    return a == b or a.endswith(b) or b.endswith(a) or b in ALIASES.get(a, ()) or a in ALIASES.get(b, ())

def competitors(vocabulary, target):
    return [c for c in dict.fromkeys([*vocabulary, target])
            if c == target or (c not in PARTS.get(target, ()) and not same_object(c, target))]

def flat(target):
    return any(same_object(f, target) for f in FLAT)
```
(The head rule must match whole words: "tv stand" must keep "tv" as a competitor.)

- FLAT += `("rug", "carpet", "calendar", "blinds")`; via aliases photo, photo mount, window glass are flat too.
- client.py: `target in select.FLAT` -> `select.flat(target)`.
- PARTS, CONFUSABLE, vocabulary (`HOUSEHOLD`) unchanged.
- val_unseen population: flat 921 eps (31%), alias 1330 (44%), novel 749 (25%).
- HM3D invariance: for chair, couch, bed, toilet, tv, potted plant, `same_object` removes nothing and `flat()` is
  True only for tv. Tests: `test_hm3d_rules_unchanged`, `test_open_vocabulary_competitors` (pillow lacks cushion,
  refrigerator lacks appliance, plant lacks potted plant, footstool lacks stool, vase keeps potted plant,
  tv stand keeps tv, flat(photo/window glass/rug) True, flat(sideboard) False).
- Before any GPU run write `outputs/apexnav_online/ovon_unseen_<date>/competitors.tsv` (category, n_population,
  group, removed competitors, flat); rules frozen after the smoke test.

### 3.2 ApexNav (branch `ovon-eval`)
1. `basic_utils/eval/ovon_dataset.py`: `@registry.register_dataset(name="OVON-v1") class OVONDatasetV1(ObjectNavDatasetV1)`;
   `from_json` setdefaults category maps, pops `children_object_categories`, sets goal `object_id` to the int
   semantic id, calls super; `__init__` fills both category maps from the episodes' categories.
2. `config/habitat_eval_ovon.yaml` inheriting `habitat_eval_hm3dv2` (dataset type OVON-v1, split val_unseen,
   data_path `data/datasets/ovon/hm3d/{split}/{split}.json.gz`, `label_mapping: none`, llm paths
   `llm_answer_ovon.txt` / `llm_response_ovon.txt`, `llm_client: none`, `video_output_path`). Verify Hydra
   composition on CPU; fall back to a documented full copy only if it does not compose.
3. NAS: symlink `val_unseen/val_unseen.json.gz -> val_unseen_hard.json.gz`, documented in `datasets/ovon/README.md`.
4. `basic_utils/eval/setup.py`: add `ovon` to `--dataset`.
5. `habitat_evaluation.py`: import the loader; `label_mapping` in {mp3d (default), none}; label verbatim for none.
6. `scripts/sample_episodes.py`: import the loader; write `split` into the output JSON.
7. `llm/answers/llm_answer_ovon.txt` (+ `llm_response_ovon.txt`), generated, committed.
8. Promote `videos/sync_runs/run_strategy.sh` to tracked `scripts/run_strategy.sh`: `DATASET` env passed through,
   `trap cleanup EXIT` killing perception server/workers and its gpu_guard, `versions.txt` on start.
9. Untouched: gate.py, run_gate_e2e.sh, C++, existing configs.

### 3.3 3DAV analysis
Move `ApexNav/videos/sync_runs/apexnav_sync_summary.py` to `scripts/apexnav_sync_summary.py`; title from the
sample's dataset/split (default "HM3D-v2 val").

## 4. Study design
- OVON val_unseen, n = 200, `scripts/sample_episodes.py` (proportional by category), seed 20261027. Report by
  group (flat / alias / novel), not per category.

| report name | strategy | perception.server flags / gate |
|---|---|---|
| baseline | baseline | no gate |
| SAM3_gate | sam3 | `--selector input` |
| +SAM3D veto (top2gap) | top2gap | `--selector top2gap --no-ensemble --judge-input` |
| +SAM3D veto (vqf) | vqf | `--selector vqf --no-ensemble --judge-input` |
| SAM3D defer | defer | `--selector top2gap --no-ensemble --judge-input --view-when unsure` + `verification.on_view_reject=defer` |

- Frozen gate rules as in the sync README (inherited from `habitat_eval_hm3dv2.yaml`); perception seed 42.
- 8 blocks of 25 episodes; within a block baseline, sam3, top2gap, vqf, defer.
- Not comparable to the OVON leaderboard (different agent/camera/threshold).

## 5. Procedure
One ApexNav stack at a time (fixed ports 12181-12184, 8770, ROS_DOMAIN_ID 42, global pkill patterns), on one
idle GPU (memory.used < 1000 MiB, no compute apps).

### 5.0 Cleanup (every GPU step)
```bash
cleanup() {
  docker exec 3dawareverification-dev-1 pkill -f perception.server
  docker exec 3dawareverification-dev-1 pkill -f perception.worker
  docker exec apexnav pkill -f habitat_evaluation.py
  docker exec apexnav tmux kill-session -t apexnav 2>/dev/null
  docker exec apexnav pkill -f 'vlm\.(detector|itm|segmentor)'
  docker exec apexnav pkill -f 'exploration.launch.py|exploration_node|tsp_node|foxglove'
  docker exec apexnav pkill -f 'ollama serve'
  pkill -f "gpu_guard.sh $GPU"
  sleep 5
  nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv -i $GPU
  nvidia-smi --query-gpu=memory.used --format=csv,noheader -i $GPU
}
trap cleanup EXIT INT TERM
```
Preflight aborts if the GPU is busy, 1-min load > 96, an `apexnav` tmux session or perception.server already
exists, or < 60 GB free in home. During runs `gpu_guard.sh $GPU 44000 8`; between strategies wait while load > 110.

### 5.1 CPU preparation
Symlink; compose config; draw sample (`--dataset ovon --n 200 --seed 20261027`; accept 3000 eps, 49 categories,
sum 200, split val_unseen); goal-id check (10476/10476); unit tests in both repos; `competitors.tsv`.

### 5.2 LLM answers (~0.5-1 GPU-h, then stopped)
Start `ollama serve` pinned to `$GPU` with `OLLAMA_MODELS=/workspace/ApexNav/data/ollama` before
`scripts/pull_llm_model.sh`; generate all 49 labels via `read_answer`; validate format (float in [0.25, 0.65],
room string, 1-5 similar labels != target; regenerate only on format failure, max 3); never hand-edit; cleanup.

### 5.3 Smoke test (~40 min)
5 sample episodes chosen by rule (flat; YOLO alias; GroundingDINO alias; GroundingDINO multi-word novel; plant);
baseline twice + defer once. Acceptance: 5/5 episodes per run, no ABORT/tracebacks; scene/label match sample;
label verbatim and LLM file unchanged; numeric goal ids; successes have distance <= 0.2 and correct instance;
baseline a/b identical (`compare_runs.py`); defer requests 0 errors, no removed competitor in scores, flat targets
judged on input; target detections present or explained; eye check of commit/final frames; timing within 1.5x;
cleanup leaves no tmux session, no perception.server, GPU at preflight memory. Summary in `smoke.md`.

### 5.4 Full run (~29-40 GPU-h)
`videos/ovon_runs/run_ovon.sh` (from `run_big.sh`) with preflight, trap/cleanup, VLM session start/readiness,
blocks of 25 x 5 strategies, resume. Accept 1000 rows, 0 server errors, no leftover ABORT, GPU free afterwards.

### 5.5 Analysis
`outputs/apexnav_online/ovon_unseen_<date>/`: summary (results.csv, summary.json, table.png), commit frames,
eye labels (formats of `defer_2026-10-26`; sibling categories `goal?` reported strict and lenient), `_compare.py`
(eye-corrected success, SPL, real FP, paired bootstrap vs baseline and sam3, per group, side by side with HM3D-v2),
`_figure.py`, README in the structure of `sync_2026-10-25`, drivers in `drivers/`; tar run records to
`/ICONAS/users/marfred/ApexNav/ovon_runs_<date>.tar` and delete local `videos/ovon_*`.

## 6. Cost
~8.6 min/episode over 5 strategies -> 28.7 GPU-h at HM3D rates; x1.0-1.4 -> 29-40 GPU-h. Disk ~25-30 GB until archived.

## 7. Risks / open questions
1. Step 0: Option A approved by the user (2026-10-27).
2. The alias table is hand-written and makes the gate more lenient for those targets; review `competitors.tsv`.
3. Competitors stay HOUSEHOLD: FPs whose true class is outside it can't be rejected.
4. 31% flat episodes make top2gap/vqf/defer = sam3 there by design; report per group.
5. LLM answers from qwen3:8b (~5 GB pull to the NAS); paired across strategies.
6. GroundingDINO exact-phrase matching may drop multi-word targets; smoke test checks.
7. 13% of goal instances have a semantic name != category; eye correction required.
8. SAM3/SAM3D GPU nondeterminism may flip near-ties.
9. Shared machine: guard trips -> ABORT, resume.
