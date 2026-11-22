# Nominal native super-resolution science run

This document describes the **default behavior** when you run `python main.py` from the repository root. The pipeline uses **native BCD cutouts** (no per-frame flux reprojection), a **North-up latent scene** at `PIXEL_SCALE / SUPERSAMPLE_FACTOR`, **PRF convolution** in the forward model with **project-then-convolve** ordering, a **GP-regularized** host scene, a **delta-function transient** (handled in the solver/PRF path, not as scene grid pixels) with **shared floating sky position** and **nonnegative flux shared within one-day epochs** (templates pinned to zero flux), and optional masking helpers for bright cores.

## How to run

```bash
pip install -r requirements.txt
# Set DATA_DIR, OUTPUT_DIR, target coordinates, etc. in src/config.py
python main.py
```

Outputs go under `OUTPUT_DIR` (FITS, CSV) and `DIAGNOSTIC_DIR` (plots, PDFs, JSON). See [Outputs](#outputs).

## Configuration applied in `main.py`

`main.py` wraps `pipeline_fit.run_pipeline_fit_core()` in `_temporary_config(...)` so the nominal science settings apply for that run without manually editing every flag in `config.py`. The mapping is documented inline in `main.py` (`nominal_overrides`).

Key ideas:

| Area | Setting | Role |
|------|---------|------|
| Geometry | `FIT_ON_NATIVE_PIXELS`, `SUPERSAMPLE_FACTOR` | Native stamps; SR=2 scene grid vs native pixels. |
| Cutout | `NATIVE_CUTOUT_SIZE` | After extraction, crop each stamp to a square centered on the transient (e.g. 40 px). `0` = no extra crop. |
| Scene | `USE_SCENE_GP_PRIOR`, `SCENE_INDEPENDENT_RIDGE` | GP interpolation is the production default; independent pixels are diagnostic only. |
| Transient | `FLOAT_TRANSIENT_POSITION` | Fit RA/Dec offset of the transient PRF term (iterated linearization in solver). |
| PRF | `PRF_ORDER_PROJECT_THEN_CONVOLVE` | Forward model order: scene → native grid → PRF. |
| Normal equations | `SOLVER_DENSE_MEMORY_LIMIT_BYTES` | Exact dense likelihood with a preallocation memory guard. Legacy Hessian cap is no longer used. |
| CR mask | Exposure consensus | Compare aligned peer exposures; preserve repeatable nuclear structure. |
| Unmask | `UNMASK_SIGMA_INF_RADIUS_PX=0` | Disabled in production; retain instrument masks. |

To change the nominal run permanently, either edit `nominal_overrides` in `main.py` or set the same keys on `src/config.py` and remove/adjust the temporary overrides.

## Pipeline stages (high level)

1. Discover FITS, build catalog, split science vs template by `SPLIT_DATE_MJD`.
2. Deep template mosaic, median stack, alignment of all frames.
3. Extract **native** analysis cutouts; optional fixed-size crop (`NATIVE_CUTOUT_SIZE`).
4. Exposure-consensus CR rejection preserves repeatable nuclear structure. Nuclear unmasking is disabled in the nominal run.
5. Star list from the deep template (solver uses nearby stars on native stamps).
6. Joint MAP/GLS solve (`solver.run_gls_solve`) with PRF and transient terms as implemented in `src/solver.py`.
7. `main.py` runs reconstruction and diagnostics (light curve CSV/PNG, residual PDFs, stacked residuals with/without transient, fit quality JSON).

## Outputs

Typical files (channel tag e.g. `ch2`):

| Location | File | Description |
|----------|------|-------------|
| `OUTPUT_DIR` | `lightcurve_<chan>.csv`, `lightcurve_<chan>.png` | Per-BCD transient flux, errors, MJD, template flag; plot. |
| `OUTPUT_DIR` | `template_model_scene_<chan>.fits`, `gp_scene_only_<chan>.fits` | Reconstructed scene components. |
| `DIAGNOSTIC_DIR` | `fit_quality_<chan>.json`, `.txt` | Per-frame QA, epoch transient summaries, offsets. |
| `DIAGNOSTIC_DIR` | `STACKED_RESIDUALS_WITH_WITHOUT_TRANSIENT.pdf` | Median stacked residual maps: model without transient vs with; difference panel. |
| `DIAGNOSTIC_DIR` | `DIAGNOSTIC_EPOCH_STACKS.pdf`, `DIAGNOSTIC_DETAILED_RESIDUALS.pdf`, … | Existing diagnostic PDFs/PNGs. |

## GP hyperparameter optimization

When `USE_SCENE_GP_PRIOR` is **False**, `run_pipeline_fit_core` does **not** run `gp_model.optimize_hyperparameters` (that path is only for GP-based runs). Initial `ell`/`var` passed to the solver still come from `config.INIT_*` and the robust variance estimate from data.

## Related modules

- `src/pipeline_fit.py` — cutout crop, unmask, solve entry.
- `src/native_fit_campaign.py` — `crop_cutout_to_size`, `unmask_sigma_inf_in_radius`, CR mask, `_temporary_config`, PDF writers used by campaign scripts.
- `src/diagnostics.py` — plotting and `plot_stacked_residuals_with_without_transient`.
- `scripts/iterative_native_fit.py` — optional staged campaign driver (separate from `main.py`).

## Tests

If `pytest` is available:

```bash
pytest tests/
```

(Add or extend tests when changing solver masks or forward-model order.)

## Review implementation notes (2026-11-20)

The transient is evaluated directly from the detector-integrated PRF on native
pixels. Its amplitude is total model flux in Jy and is independent of the latent
scene SR factor; a cropped stamp is not renormalized. Shared flux entries in the
per-BCD CSV are repeated measurements of the same fitted parameter, not independent
estimates to average again. `Epoch_Gap_Warning` flags neighboring spans over one
and under two days. Backgrounds remain per BCD.

Missing instrument PRFs raise an error. Synthetic campaign mode explicitly selects
an analytic PRF. The experimental central-annulus penalty is disabled. Exact dense
normal equations replace the old diagonal likelihood approximation, with an explicit
memory guard; large runs may require reducing the ROI until a scalable exact solver
is implemented. Curvature errors are local Gaussian approximations, especially near
the nonnegative boundary. See `CODE_REVIEW_2026-11-20.md` for remaining acceptance work.

Run deterministic recovery diagnostics with:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python scripts/validate_internal_consistency.py
```
