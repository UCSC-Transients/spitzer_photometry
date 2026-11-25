# Stage 2: native responses and isolated-star astrometry

Scientific promotion is **unapproved**. The response/centroid interface has
independent synthetic recovery tests; absolute calibrated IRAC flux does not yet
have sufficient evidence. Nothing here introduces a galaxy GP or changes epoch
sharing, template-zero flux, host interpolation, or the production flux bounds.

## Launch and scope

- Supplied predecessor head: `f0069d30042bc0b9c49b8ef4891557dce5d00dbc`.
- Reviewed Stage 0 head: `0880bcad81a5af09e5326d1d86f8ad58e111c754` (PR #22).
- Integrated dependency/launch target `main`:
  `09ddf161a59f135b76bd78d3dd09e28c3249aa1c` (PR #23 merged). Its tree is identical
  to the supplied predecessor head. Both predecessor PRs were merged by the user.
- Run: `20261125-01`; branch `codex/02-prf-astrometry-20261125-01`.
- Author: `.worktrees/author-02-prf-astrometry-20261125-01`; comparison checkout:
  `.worktrees/base-02-prf-astrometry-20261125-01`.
- Original dirty checkout was read only. Launch/source-document hashes are in
  `.test-artifacts/stage2/launch.json` in the author worktree.

## Calibration evidence and blocking flux gate

The local production calibration directory has 25 x100 warm APEX PRFs per
channel. The headers identify postcryogenic data, `PRFDATE=140915`, 2559-square
arrays, x/y oversampling 100, full-detector PRFCOL/PRFROW, and the zero-phase
realization centered on zero-based table entry `[1279,1279]`. The new grid maps
one-based calibration locations 25/77/129/181/233 to zero-based detector positions
24/76/128/180/232. A crop coordinate must never select a full-detector calibration.

The [SSC calibration page](https://irsa.ipac.caltech.edu/data/SPITZER/docs/irac/calibrationfiles/psfprf/)
describes native-pixel-integrated PRFs and a zero-phase radius-10 normalization.
The [centering discussion](https://irsa.ipac.caltech.edu/data/SPITZER/docs/irac/iracinstrumenthandbook/56/)
distinguishes warm native-realization centering from the oversampled centroid.
The [sampling prescription](https://irsa.ipac.caltech.edu/data/SPITZER/docs/irac/iracinstrumenthandbook/59/)
uses native-interval samples; integrating the table over each native pixel again
would blur the response twice. First-moment centroids depend on aperture and
algorithm and are diagnostic, not a second recentering operation.

For raw table T, oversampling s, and fixed center c, the existing native sampler
uses `P(x,y) = sample(T,c+s*(pixel-source)) / [sum(T)/s**2]`. That denominator
integrates across phases and the full rectangular table; it is not the radius-10
zero-phase aperture. Its normalization is preserved for compatibility and now
explicitly documented as a legacy amplitude convention, not certified Jy.

Measured across all 50 tables:

- ch1 raw zero-phase radius-10 sum: 1.13531–1.14118; legacy-normalized sum:
  0.972324–0.976471.
- ch2 raw zero-phase radius-10 sum: 1.14456–1.14801; legacy-normalized sum:
  0.967448–0.969184.
- Central ch2 full-table integral: 1.182829; radius-10 zero-phase sum: 1.145354.
  Its SHA-256 is `d34f398ba1e1305aab4d1df0e898e997f61dbc3a0437c3c1532382b01044a8b8`.

The raw aperture sums disagree with the unit-aperture header claim. This audit
does not resolve that discrepancy or replace it with a fitted correction. The
new optional `zero_phase_aperture` convention divides by one *fixed* native
radius-10 sum. It makes the amplitude gauge explicit, but **does not establish
absolute Jy**. No phase or cropped-stamp sum is forced to unity. The published
[APEX correction table](https://irsa.ipac.caltech.edu/data/SPITZER/docs/dataanalysistools/tools/mopex/mopexusersguide/98/)
labels its correction factors as cryogenic; they are not applied to these warm
files. Warm mean-phase, array-location, color, background-annulus and independent
standard-star comparisons remain necessary. No private FITS content is committed.

## Reusable Stage 3 interface

`src.point_response.NativePRF` snapshots an immutable table, center,
oversampling, channel, mission, SHA-256 and normalization convention. `sample`
returns native response per declared amplitude and `derivatives` returns exact
piecewise-bilinear derivatives in native pixels. The production native sampler
uses the same low-level sampling function, preserving its existing normalization.

`PRFGrid` loads strict warm-table snapshots and blends native responses using
full-detector location. It verifies channel, mission, shape, oversampling,
centering and filename/header position consistency. All nonzero-weight corners
are required. The old production loader's partial-corner behavior remains a
legacy limitation; the new interface never silently replaces a missing corner.
Grid-edge clamping is unvalidated extrapolation. This is not a rotation/distortion
calibration model: those transformations come from the supplied detector WCS.

`StarFrame(data, sigma, wcs, response, mask)` snapshots inputs. Data and sigma are
Jy/native pixel; the fitted amplitude inherits the declared PRF convention.
Real grid providers require explicit `wcs._detector_wcs` provenance. Resampled
images are outside this interface's contract.

`fit_isolated_stars(frames, positions_xy, initial_flux, reference_wcs=...)`
jointly fits nonnegative shared star fluxes, independent star sky centroids,
signed constant backgrounds and detector-frame translations. The positions are
reference-WCS pixels; detector translations are native pixels. Frame zero's
translation is fixed exactly to zero, and multiple stars are required to fit
shared registration. The absolute sky origin remains tied to that reference.
Rotation/distortion are supplied, not estimated. Search-bound solutions and
nonconvergence raise errors. Rank deficiency and active flux bounds produce NaN
local covariance, not false precision or clipped confidence intervals.

`StarFit.predict(frames)` uses the same renderer and rejects geometry or
calibration changes after fitting. The result records joint covariance order,
singular values, rank, active constraints, objective history, optimality,
setup/solve timings and response-call accounting. Covariance is local joint
curvature with known measurement sigma, not validated boundary coverage or
calibration uncertainty. The fitter is opt-in; it does not silently shift the
pipeline's host, nucleus or transient.

The new interface applies neither p-map nor another intrapixel correction. Its
calibration snapshot cache is bounded to eight files, uses file identity
(device/inode/size/mtime/ctime), and stores SHA-256 of exactly the bytes parsed.
This is a local-filesystem invalidation contract, not a claim that stat keys are
content hashes. A change during reading/fitting or before prediction is rejected.
The shared solver fingerprint now checks current file identity before using a
cached operator; it no longer caches directory identity indefinitely.

## Validation and reproducibility

Required PR CI discovers `tests/test_stage2_astrometry.py` through the existing
fast numerical job. It covers independent integrated truth, one/three exposures,
multiple stars, nonnegative flux, independently injected sky and registration
offsets, SR=1/2/4 at fixed sky footprint, rolls, SIP, missing pixels, derivatives,
fit/predict equality, lost wings, phase-dependent response, missing calibration,
and restored-mtime cache invalidation. Real-file tests are explicitly marked.
No science or performance threshold is loosened.

`python scripts/validate_stage2.py --output-dir test-results/stage2
--prf-dir /read-only/calibration/directory` emits separate synthetic and real
bundles through `src.validation_metrics.write_artifact_bundle`. Omit `--prf-dir`
for synthetic-only CI; real availability is then recorded as missing. Bundles
contain strict JSON, arrays, hashes, seed/config metadata and labeled PNGs.
Development seeds are 1729/2718/31415; reserved acceptance seeds remain untouched.

The first noisy run failed the existing optimality requirement (0.001496 > 0.001)
despite an optimizer `xtol` termination. Its log/status are retained under
`.test-artifacts/stage2/evidence*`. Exact bilinear derivatives fixed the failure
without relaxing the convergence test. The completed `evidence-v2` run has:

- Maximum noiseless fractional flux bias: 5.96094e-5; maximum centroid error:
  3.04186e-6 native pixel; registration error: 2.63312e-6 native pixel.
- All 12 SR/seed trials converge; maximum scaled optimality 0.000203.
- Three independent noise seeds give 8/9 local 95% flux intervals covering truth.
  Stars are correlated within a fit, and repeating a seed across SR does not
  increase the independent sample size. This does not establish coverage.

The clean Python 3.11 environment passes `pip check`. Final fast verification:
419 passed, 5 skipped, 12 deselected (60.27 s); the three Stage 2 real-calibration
tests pass separately. The shared environment's bqplot conflicts, an initial
test launched before the clean install had completed (missing astropy), and the
earlier optimizer failure are retained in logs rather than removed from the
record. Dependency versions are frozen in the local evidence directory.

The first paired run (shared read-only environment) gave identical arrays:
cold 35.55 → 37.98 ms; warm 3.72 → 5.53 ms. Median RSS was 192.51 → 189.89 MB
cold and 188.69 → 190.30 MB warm. Fresh calibration-identity checks add warm
overhead. Neither combined engineering budget was crossed. Final clean-environment
measurements are recorded in the PR and `paired-clean/summary.json`; the common
operator implementation is unchanged between these measurements.

Full evidence lives under `.test-artifacts/stage2/` in the author worktree:
`evidence-clean/{synthetic,real-prf}` holds arrays/plots/manifests;
`all-local-prf-files.json` inventories all 50 local FITS files and their hashes;
`bias-summary.json` records per-star mean bias, RMSE and descriptive t intervals
from three seeds. Plots are visually unverified; quantitative checks use arrays.

Use the unchanged Stage 1 runner for paired engineering measurements:
`python scripts/run_stage1_benchmarks.py --base-repo BASE --head-repo HEAD
--output-dir test-results/paired-stage2 --repeats 5 --timeout 45`.
It compares the established common operator path, not the new fitter (which has
no baseline equivalent). No speedup is claimed. Exact numerical arrays must
agree before timings are interpreted.

## Unmet gates and reviewer handoff

Blocking: absolute warm flux calibration and user-declared science tolerance.
Also unvalidated: held-out real BCDs, large-volume fitting, real registration
recovery against external truth, zero-flux interval coverage, variable stars,
rotation/distortion inference, chromatic terms and detector-edge extrapolation.
Source/registration covariance is retained locally; calibration uncertainty is
not modeled. The static-star assumption must be tested before use on real data.

An independent GPT-6 Astra/xhigh reviewer should use `PROMPT_REVIEW_PR.md` at the
exact PR head in a separate detached worktree. Focus on normalization/gauge
interpretation, missing-corner behavior, derivative knots, covariance limits,
cache invalidation, fixed-footprint recovery and the paired arrays/budgets.
Code review approval would not approve scientific promotion. Do not merge.
