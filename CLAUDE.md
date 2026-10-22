# CLAUDE.md

## Project goals

1. **Prove that suboptimal viewpoints cause false positives in object detectors.**
   Done on the reviewed upright Objaverse cohort (110 objects x 60 views): SAM3
   scores the full category vocabulary on ground-truth renders; per-object
   heatmaps of the confidence margin are the "before" result.
2. **Show that an ambiguity score on SAM3D-reconstructed viewpoints disambiguates
   these views.** Before/after: for each input image, reconstruct with SAM3D,
   render candidate views, choose the least ambiguous one, run SAM3 on it, and
   show that false positives drop. It works on hand-picked cases; the study
   pipeline makes it systematic.

Everything in the repo should serve these goals. Remove outdated code and data
rather than keeping parallel versions. Changes should make experimenting easy:
reusable plots, and all images, angles, and scores stored in the study database.

## Conventions

- **Margin** = true-category score - strongest wrong-category score. Higher is
  better; negative means a misleading view. (Old code used the opposite sign
  under the name `score_minus_true`.)
- **Ambiguity** metrics are in [0, 1], lower is better; `None` means undefined.
- **Cameras**: one y-up az/el convention in `tools/cameras.py`. Ground-truth
  angles are in the reviewed object frame; SAM3D candidate angles are relative
  to the input camera, and (180, 0) re-renders the input view (verified visually).
  SAM3D candidate cameras use gravity as up (`tools.cameras.sam3d_gravity`, from
  the input elevation). Before 2026-10-18 they kept the input camera's up, so
  off-axis candidates of elevated inputs were rolled. That bug is closed:
  candidate grids require the input elevation, and the reconstruct fingerprint
  includes "gravity-up", so a rolled study refuses to resume. Rolled results
  exist only in the NAS archive `upright-rolled-sam3d-2026-10-18.tar`.
- **Controls** for any after-result: `identity` (re-render from the input view),
  `random`, and `oracle`. An ambiguity selector only counts if it beats
  `identity` and `random`.
- Learned metric weights were ~1/3 each and not significantly better than equal
  weights (p 0.2-0.46), so `ambiguity:combined` uses equal-weight ranks.

## Layout and workflow

- `python -m study --config configs/<name>.yaml` runs render -> reconstruct ->
  score -> detect -> analyze. Stages resume. Outputs are in `outputs/<study>/`,
  with `study.sqlite` as the source of truth. See README.md. `outputs/` and
  `data/` stay on the server; archive large finished datasets (e.g. completed
  render runs) to `/ICONAS/users/marfred/3DAwareVerification/` and delete the
  local copy (home quota is 300 GB).
- New metric: register it in `study/ambiguity.py` and add it to the config's
  `ambiguity` section. New selector: `study/selectors.py`. New plot: compose
  `study.figures` panels. None of these needs new GPU work.
- Two conda envs in the container: `core` (SAM3, analysis) and `sam3d` (SAM3D,
  PyTorch3D, Open3D). Run with `sam3d-env <env> python -m ...` inside the
  `dev` container; the repo is mounted at `/workspace`.
- Shared GPU server: check `nvidia-smi` and set `gpu_id` to an idle GPU before
  every GPU run.
- Tests: `sam3d-env core python -m unittest discover -s tests`, plus
  `test_ambiguity.py`, `test_harness.py` and
  `test_stage1_uncertainty.py` in the `sam3d` env.
- Keep config validation in `study/config.py`; don't add new YAML entry points
  outside `configs/`.
