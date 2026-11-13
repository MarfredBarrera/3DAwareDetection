# Qwen in the verification pipeline: architecture plan (2026-11-03)

## Implemented (2026-11-03): selector `d1`

`Verifier(selector="d1")` / `python -m perception.server --selector d1` implements the D1 design of
`docs/apexnav_approach_reverify_plan.md` (sections 1, 2, 8) as a preset of the plan below
(gate = 2-signal sufficiency, judge = qwen, evidence = grid or input):
- Gate: `perception.qwen.SufficiencyGate`, frozen by `sam3d-env core python -m perception.qwen` into
  `outputs/apexnav_offline/qwen_all_requests_2026-11-03/d1_gate.json` (131 offline requests; threshold
  0.5458, same percentile rule as `_compare.py`, not the midranks of `UncertaintyGate`).
- Sufficient: SAM3 raw top-1 decides (`source="sam3"`); SAM3D and Qwen are not started.
- Insufficient: Qwen (int8 by default) judges the outlined frame + SAM3D grid. SAM3, SAM3D and Qwen all
  start at server launch on the one `--gpu` (ApexNav's gate timeout is 120 s; `--lazy-workers` defers
  SAM3D/Qwen to the first insufficient request, which then also waits for the model load: only with a gate
  timeout well above the load time; runs use the eager default); `--d1-evidence input` drops the grid and never starts SAM3D. Reconstruction / Qwen errors and invalid
  answers are `insufficient` with `verdict.qwen.error`.
- Reply keeps `PROTOCOL = 1` and adds `verdict.status/source/gate/qwen`; `requests.jsonl` logs them as
  `verdict_status`, `source`, `gate`, `qwen` (`status` stays the HTTP status).
- Tests (`tests/test_perception.py` `D1Tests`, `D1ReplayTests`) replay all 295 pilot requests and match
  `_compare.py`'s "D1 sufficiency-first (area+margin)" decisions exactly.

## What exists and what it showed

- Uncommitted on `apexnav-gate`: selector `qwen_grid` in `perception.Verifier`,
  a third worker (role `qwen`, venv `/data/envs/qwen`), `perception/qwen.py`
  (frozen `UncertaintyGate`, three-way `Decision`, prompt, `QwenJudge`), and
  `scripts/qwen_verification_study.py`.
- Offline screening (`outputs/apexnav_offline/qwen_verification_pilot`, 295 requests):
  | policy | false accepts | false rejects | clear coverage |
  |---|---:|---:|---:|
  | SAM3 raw | 14 | 3 | 1.00 |
  | gate -> defer | 2 | 1 | 0.72 |
  | gate -> Qwen input | 8 | 1 | 0.88 |
  | gate -> Qwen grid | 5 | 1 | 0.82 |
  Grid vs input: the CI includes 0. The mismatch-grid control behaves about the
  same. Qwen latency is 0.07 s (input) or 0.28 s (grid). SAM3D is about 3.4 s.
- Conclusion: the uncertainty gate helps most. Qwen recovers coverage from
  deferral. Grid evidence has not yet shown a benefit of its own. The design
  must make the evidence a switch, not a fixed part of the pipeline.

## Target design

Split the monolithic `qwen_grid` selector into three independent stages. Each
stage is a `Verifier` argument and one function:

```
frame, mask, target
  -> 1. raw judge     SAM3 on the masked frame (margin, area, raw label)       [always]
  -> 2. gate          UncertaintyGate.assess(margin, area[, view_flip])        [gate=]
        confident  -> status from raw SAM3 (correct / wrong)
        unsure     -> 3.
  -> 3. evidence      "input" | "view" (vqf/top2gap pick) | "grid"            [evidence=]
  -> 4. judge         "sam3" (current binary path) | "qwen" (three-way)       [judge=]
  -> Verdict(status in {correct, wrong, insufficient}, accepted = status == correct)
```

- `Verifier(selector, judge="sam3"|"qwen", evidence="input"|"view"|"grid", gate=path|None)`.
  Today's selectors are `judge="sam3", evidence="view"`. `qwen_grid` becomes
  `judge="qwen", evidence="grid", gate=...`. The `input` and `naive` selectors stay as they are.
- **Gate without SAM3D**: the frozen gate has a 2-signal mode (margin, area).
  With `evidence="input"`, SAM3D is never started. That makes it the cheap arm
  (about 0.1 s per unsure request instead of about 4 s), and it is the one to beat.
  Using `view_flip` forces SAM3D on every request, so it is only allowed when the evidence needs SAM3D anyway.
- Workers start lazily, only those the configuration needs: `qwen` iff
  `judge="qwen"`, and `sam3d` iff the evidence or gate needs it. Keep the existing `_Worker` protocol.
  Qwen loads int8 by default (~10 GB; parity is checked in `int8_parity/`) to fit the single GPU.
- **Verdict is always three-way.** Binary paths set `status` from `accepted`.
  Invalid Qwen answers and failed reconstructions become `insufficient` and are
  logged with a reason. They are never coerced to accept or reject.
- **Policy lives in the planner, not the server.** The server stays stateless:
  one request in, one verdict out. ApexNav `perception_gate` maps
  correct -> ACCEPTED, wrong -> REJECTED, insufficient -> DEFERRED (already
  state 2). It owns retry budgets, the "new real view required" rule, and STOP blocking.
  Remove `Decision.navigation_action` from `perception/qwen.py`, or move it to
  ApexNav, so that the rule exists in one place.

## Protocol and files

- HTTP `PROTOCOL = 2`: the reply always carries `verdict.status`. `view_image` is the
  evidence image Qwen or SAM3 judged (view or grid). ApexNav refuses to run against protocol 1.
- `requests.jsonl` logs the gate features and score, the evidence kind, Qwen's status,
  probabilities, `invalid_answer`, latency, `PROMPT_VERSION` and the model/int8 setting.
  Every closed-loop run can then be re-analyzed offline without re-running models.
- Move the frozen gate out of `outputs/` (it gets archived) to
  `configs/qwen_gate.json`, with its calibration hash. `QWEN_GATE` points there.
- One study script: fold the earlier `qwen_grid_2026-11-02/_qwen.py` experiment
  into `scripts/qwen_verification_study.py` (input/grid/mismatch arms already
  cover it) and delete the parallel copy.
- `perception/qwen.py` keeps the gate, prompt, image prep and `QwenJudge`. It must not
  import `client`, so that the study script and the worker share it.

## Implementation steps

1. Refactor `Verifier` into raw -> gate -> evidence -> judge. `qwen_grid` becomes a preset.
   Unit tests: every (judge, evidence, gate) combination with fake workers; check
   which workers start; insufficient is never accepted.
2. Protocol 2 in `perception/server.py` and ApexNav `perception_gate/client.py`, plus the
   status -> state mapping in `gate.py` (`test_gate.py` covers it).
3. Move the gate to `configs/` and consolidate the study scripts.
4. Offline replay through the real server on the 131 + 295 request sets. Check the
   numbers match the pilot (regression test of the refactor).
5. Closed loop (paired episodes, same seeds, `scripts/run_strategy.sh`): SAM3 raw,
   gate -> defer, gate -> Qwen input, gate -> Qwen grid. Metrics: success, SPL,
   false stops, deferral count and extra steps. Grid stays only if it beats input
   in closed loop. Otherwise drop it and keep SAM3D out of the Qwen path.

## Open decisions

- Default arm for ApexNav: gate -> Qwen input (cheap) vs grid (5 vs 8 false accepts, not significant).
- Retry budget for insufficient (the pilot assumed 3 new views) and what happens at budget exhaustion
  (explore elsewhere without blacklisting).
- Whether to recalibrate the gate on a 2-signal basis, since the input arm never computes view flip.
