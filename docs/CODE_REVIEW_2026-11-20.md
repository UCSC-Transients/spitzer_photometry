# Code review and repair plan — 2026-11-20

## Scope and status

Reviewed the current working tree, production entry point, numerical operators,
preprocessing, GP prior, campaign driver, diagnostics, existing tests and CI.
Experimental scripts were inspected for shared assumptions and entry-point behavior;
this is not a claim that every experimental configuration has been executed.
Findings below distinguish direct reproductions from static code-path findings.
No scientific result is certified by this review. Repairs and regression tests are
implemented in the working tree; the remaining acceptance work is listed below.
Finding locations describe the reviewed baseline, not current line numbers.

Existing staged, modified and untracked work is preserved. In particular, the
native-only cutout changes and sigma-weighting additions predate this review.
`clean_repo_run/` is an untracked historical tree, not the production baseline.

## Repository comparison

Live `git ls-remote` queries to both UCSC-Transients/spitzer_photometry and the
configured origin (astrofoley/spitzer_photometry) returned identical branch tips.
Both HEAD/main refs are `ce99cf5396b658563a096ea31b560dd227a76955`.

- Checked-out `feat/phase1-publication-baseline`: `ce99cf5`, identical to remote main;
  one commit ahead of its own remote feature branch (`9f61a03`).
- Local `main`: `98b8e02`, four commits behind remote main, no unique commits.
- Local `debug/native-core-mismatch-phase0-tests`: `68f0ca6`, seven commits behind
  remote main, no unique commits.
- Local `feature/intrapixel-correction`: `2910f5b`, equal to its remote branch,
  nine commits behind remote main, no unique commits relative to main.
- Remote `feature/nominal-native-science-docs-pr`: `105130b`.

The important unmerged work is the dirty working tree, not a divergent local
committed branch. No reset, checkout, merge, push, issue or PR was performed.

## Intended scientific model

The documented production model fits native BCD measurements in Jy/native pixel.
A shared static host scene and stars are projected into each detector exposure
and convolved with its PRF. Science BCDs have transient flux parameters; template
BCDs have zero transient flux. Backgrounds are fitted per BCD. The transient can
have a common floated sky position. The latent scene is intended to be north-up
at native pixel scale divided by the super-resolution factor. GP regularization,
host Gaussians and a nuclear point source are optional.

User-confirmed production contract: a GP-regularized host interpolates across the
point source; point-source flux is nonnegative and shared by exposures whose total
span is at most one day. A neighboring span over one but under two days is flagged,
not merged. There is no external reference light curve. Acceptance uses internal
consistency, held-out residuals and synthetic injections. The native-only local
extraction and sigma-weighting changes were retained; tuning weights is not itself
scientific validation.

Point-source columns now sample the detector-integrated IRAC PRF directly on the
native detector grid. Their Jy amplitudes do not depend on scene supersampling;
cropped stamps retain missing-wing losses. The absolute calibration convention of
the supplied PRF tables still needs an aperture/encircled-energy audit. The host
uses a separate latent intensity field and GP interpolation.

### Repair status

Implemented with targeted regressions: channel parsing (R01), explicit scene WCS
(R02), common fit/predict component operators (R03), native point-source sampling
(R04), preserved detector WCS (R05), CD/SIP cache keys and per-solve invalidation
(R06), exact likelihood Hessian (R07), scaled constrained optimization (R08), bound
indices and iterative position fitting (R09), raised failures/sparse priors (R10),
invalid-data handling (R11), exposure-consensus CR rejection in the nominal path
(R12), saved model settings/effective uncertainties (R13), fitted star fluxes
(R14), selected covariance solves with unavailable errors represented by NaN
(R15), GP-off prior isolation and disabled experimental annular penalty (R16),
partial PRF loading and explicit missing-calibration failures (R17), PRF scale
(R18), linear convolution boundaries (R19), ACF centering/pair normalization
(R22), campaign metric units (R23), per-BCD backgrounds (R24), GP numerical scale
(R25), bounded memory and cached calibration reads (R26), footprint bounds (R27),
p-map broadcast/cache keys (R28), input checksums (R29), spherical unique-star
matching and robust date parsing (R30), legacy empty/shared-epoch postprocessing
and declared pandas dependency (R31), and deterministic CI
fixtures plus injection artifacts (R32). This list identifies repaired paths;
it does not imply that every subproblem in those findings is closed.

Still requiring work or broader validation:

- R04/R05/R06/R19: real calibration aperture conventions, full distortion/rotation
  sweeps, detector-position variation, and content keys for every operator cache.
- R07/R26: exact dense solves now fail before oversized allocations. A matrix-free
  constrained solver and GP operator are needed for large scenes; do not restore
  the old approximate Hessian to evade this limit.
- R12: consensus masks need real dither/roll tests and injected cosmic rays. With
  fewer than two peer BCDs no consensus rejection is attempted. Campaign legacy
  local-mask paths still need migration.
- R15/R25: reported curvature errors marginalize the scene but do not establish
  coverage at the zero-flux boundary or include GP-hyperparameter uncertainty.
  Hyperparameter fitting remains a heuristic, not the full forward likelihood.
- R16: the experimental symmetric annulus penalty is disabled, not replaced by
  a scientifically approved one-sided prior.
- R20/R21/R23: nominal display masks now reproject to the scene WCS, normalized
  maps use saved fit noise, and dipole scans convert scene displacements through
  WCS. Campaign application of the proposed WCS shift and experimental script
  metrics still need validation. Signed residual dipoles are corrected.
- R29/R30/R33: manifests still need calibration/uncertainty-file hashes and
  full failure provenance; ingestion failure provenance and global configuration
  architecture remain follow-up items.
- Real-data held-out residual tests, larger injection coverage experiments and
  calibrated PRF tests remain acceptance gates. No real light curve is certified.

## Findings: scientific correctness and failure handling

### R01 — P1: wrong PRF channel for standard Spitzer filenames

`src/solver.py:1283`, `src/solver.py:1928`, and diagnostic channel selection use
the literal substring `ch2`, otherwise choosing ch1. Discovery accepts `I2`.
Reproduction: `SPITZER_I2_123_0000_0000_1_cbcd.fits` is accepted as ch2 and then
modeled as ch1. Pass validated channel metadata throughout; reject ambiguity.

### R02 — P1: nominal scene is the first native stamp, not the promised SR grid

`src/pipeline_fit.py:113` passes the first native cutout WCS and a native-shaped
median to `run_gls_solve`; `src/solver.py:1010` and `:1096` adopt that shape/WCS.
There is no call to the campaign footprint-based scene builder in this path.
SR=2 therefore does not establish the documented north-up SR=2 latent grid.
Build an explicit union footprint at the requested scale and verify every BCD's
coverage; separate the display template from the scene geometry.

### R03 — P1: fitting and prediction use different forward operators

In `src/solver.py:1370`, transient/star/nuclear columns are convolved on the scene
then projected. Host columns at `:1446` use the anchor bundle directly. Prediction
at `:2015` applies the selected frame operator to all components, including
project-then-convolve. The nominal configuration selects that latter order.
Use the same component renderer for matrix assembly, position derivatives,
reconstruction and residuals, including exact/anchor modes.

### R04 — P1: integrated-flux normalization is not defined consistently across SR

`src/solver.py:501` samples scene values at native pixel centers using bilinear
interpolation. It neither integrates over native pixel area nor supplies an area
conversion. A scene delta is normalized to sum to its fitted Jy amplitude.
These conventions do not guarantee conserved point-source flux when grid scale
or phase changes. Reprojection also renormalizes each truncated PRF stamp to one
(`:207`), obscuring aperture losses near boundaries. Establish a single units
contract and test total flux, surface brightness, pixel phase and finite support
independently before interpreting fitted amplitudes as calibrated Jy.

### R05 — P1: cropped coordinates select the wrong detector-position PRF

`src/preprocessing.py:482` retains only cutout WCS as both `wcs` and `raw_wcs`.
`crop_cutout_to_size` slices both again. `convolved_delta_column` uses that WCS
to choose calibration-grid coordinates (`src/solver.py:740`). Those coordinates
are stamp positions, not original full-detector positions. Preserve detector
origin/full-detector astrometry separately from likelihood-grid WCS, including
alignment and SIP handling. The same distinction matters for optional p-map gains.

### R06 — P1: cache keys can return scientifically stale operators

Direct reproduction: two otherwise identical WCS objects with different CD
matrices have equal `_wcs_projection_cache_key` values (`src/solver.py:465`).
CD is omitted. SIP reference pixel and additional distortion representations also
need audit. PRF caches use object IDs and omit calibration/config fingerprints
(`:250`, `:333`, `:419`); mutable WCS and object-ID reuse can reuse wrong kernels.
Geometry cache (`:1075`) omits per-frame WCS, masks, epoch IDs and several settings
used to build cached content. Remove unsafe caching first, then reintroduce bounded
content-based caches with invalidation tests and measured benefit.

### R07 — P1: approximate scene Hessian is not the likelihood Hessian

`src/solver.py:1310` retains exact RHS and nuisance cross-blocks but drops scene
off-diagonals outside the full-Hessian cap. The default path substitutes projected
weights, not even the exact diagonal of F-transpose W F. This can destroy positive
semidefiniteness and change the optimum. The exact-diagonal branch also omits
second-scene/cross-scene terms for two-component GP fits. Use the full operator
in an exact matrix-free normal action; approximations belong in a preconditioner.

### R08 — P1: large bounded fits are clipped, not optimized

`src/solver.py:1697` returns clipped unconstrained parameters above 2,000
parameters. Coupled scene/background parameters must be reoptimized when a flux
hits a bound. Small fits can also return a failed optimizer's iterate (`:993`).
Require converged constrained solutions, projected-gradient/KKT checks, and an
explicit unsuccessful outcome otherwise.

### R09 — P1: floated-position pass uses stale host/nuclear bound indices

`src/solver.py:1711` computes indices before the offset columns are inserted;
`:1721` passes those old indices into the rebuilt system. Bounds can be applied
to another parameter while leaving host/nuclear flux unconstrained. Recompute
indices from the returned layout on every assembly pass.

### R10 — P1: solver errors masquerade as successful fits

`src/solver.py:1878` catches all errors and returns a partially initialized result.
The pipeline only checks `results is None`, so failed fits can proceed to output.
Direct reproduction: sparse diagonal precision cannot be assigned to the dense
Hessian at `:1262`; the existing finite-scene test can nevertheless pass because
the returned default scene is zero. Raise typed failures or require an explicit
success/status contract, with finite-objective and valid-result checks.

### R11 — P1: invalid pixels can be overweighted or poison the solve

`load_data`/extraction use `nan_to_num` without explicit infinity policies.
Positive/negative infinite data become huge finite values; negative/zero sigma
can become extremely large weights through clipping. The solver mask at `:1275`
does not require finite data or positive effective sigma, and multiplying zero
weight by NaN still produces NaN. Other masks use different thresholds.
Centralize data-quality masks, retain provenance, sanitize masked operands before
linear algebra, and distinguish valid zero measurements from missing coverage.

### R12 — P1: unmasking undoes cosmic-ray rejection

`src/pipeline_fit.py:135` applies the CR mask then calls unmasking, although its
comment claims only pre-existing bad pixels are restored. The unmasker has no
provenance. Reproduction restored all nine newly masked pixels in a test disk.
Keep instrument, coverage and CR masks separate. Do not recover any class without
an explicit science policy; preserve CR flags after permitted recovery.

### R13 — P1: campaign scoring loses the fitted model configuration

`src/native_fit_campaign.py:1207` exits `_temporary_config` before computing fit
metrics. Similar behavior occurs in `run_stage` and shifted-fit scoring.
Prediction reads global flags, so nuclear/host components, operator order and
effective noise may differ from the fitted model. Restoring only nuclear flags
for PDFs does not restore the whole model. Save an immutable fitted model spec
and use it for every prediction/metric; test predictions after config restoration.

### R14 — P1: post-fit star replacement changes the solution

`src/pipeline_fit.py:209` replaces nonpositive fitted star fluxes with catalog
initial values without refitting scene, backgrounds or transient. Prediction
also silently omits negative stars. Enforce any desired star constraint inside
the optimization, or retain signed fitted coefficients consistently.

### R15 — P1: uncertainty outputs can report false certainty

`src/solver.py:1822` inverts the full Hessian, clips negative covariance diagonals
to zero, and leaves errors at zero if inversion fails. Active bounds, position
linearization and approximate Hessians are not handled by this covariance.
Epoch fluxes are arithmetic means but epoch errors are RMS individual errors
(`:1831`), not uncertainties of that mean; inter-BCD covariance is discarded.
Use covariance of the actual estimator, label conditional/local approximations,
and report unavailable uncertainty explicitly. Validate coverage by simulation.

### R16 — P1: GP-off still receives a central morphology penalty

`src/solver.py:1629` applies the central annular quadratic term without checking
`USE_SCENE_GP_PRIOR`. The default calls it monotonicity, but the initial quadratic
penalizes both signs of the annular difference; it is not a one-sided constraint.
It also acts on only the first block of a two-component scene. Make each prior
explicit, and apply any monotonic condition to the intended total scene.

## Findings: calibration, diagnostics and maintainability

- **R17 / P1 — PRF load failures:** `src/solver.py:124` evaluates an ndarray as
  a boolean when one corner is missing (reproduced). If all corners are missing
  it silently substitutes a Gaussian. Failed reprojection returns zeros. Require
  calibration availability for science mode; any synthetic fallback must be
  explicit and recorded. Cache normalized calibration data with file identity.
- **R18 / P2 — PRF resolution tests/spec disagree:** four of five tests in
  `tests/test_prf_resolution_independence.py` fail. They set
  `PRF_PIXEL_SCALE_ARCSEC`, which the implementation does not read. This proves
  a test/config contract failure; it does not by itself prove real SSC PRFs are
  delta functions. Validate calibration header scale and centering; test CD and
  PC/CDELT, asymmetric PRFs, rotation and encircled energy independently.
- **R19 / P1 — boundary wraparound:** anchor convolution at `src/solver.py:301`
  is circular; `_center_kernel_for_convolution` also rolls edge content.
  Opposite edges can receive source flux, unlike the exact operator. Use padded
  linear convolution with a matched adjoint and physically adequate scene margins.
- **R20 / P2 — sky stacks and masks confuse equal shape with equal WCS:**
  `src/pipeline_fit.py:128` medians differently oriented native arrays directly.
  The newly unified native extraction also makes the legacy CR comparison use
  mismatched grids. Diagnostic/intersection masks in `diagnostics.py` and
  `dipole_chi2_scan.py:27` similarly compare shapes rather than sky transforms.
  Reproject diagnostic copies and masks explicitly; preserve native fit data.
- **R21 / P2 — dipole correction is in the wrong coordinate system:**
  `src/dipole_chi2_scan.py:116` applies a scene-pixel direction/shift directly to
  every native image, ignoring rotation and SR. Its residual centroid uses
  absolute residuals, which can cancel a symmetric signed dipole. Convert a sky
  displacement through each WCS and use a signed/template-derivative statistic.
- **R22 / P2 — ACF statistics are wrong:** `src/residual_metrics.py:8` subtracts
  the valid mean from masked pixels too. A constant valid patch produces a
  nonzero normalized ACF (reproduced). `acf_e90_scale_pix` seeks the first value
  above 0.5 instead of the first downward crossing and returns the largest radius
  when none is above. Correct mask-pair normalization and naming/threshold tests.
- **R23 / P2 — QA units and weights disagree:** `src/fit_metrics.py:76` labels
  sqrt(mean normalized chi-square) as RMSE, then divides by sqrt(Jy), which is not
  a calibrated Poisson uncertainty. Diagnostics/dipole scans often use raw sigma
  while fitting uses adjusted sigma; clipping thresholds differ. Report both raw
  and effective-noise residuals with explicit units; chi-square per pixel is not
  reduced chi-square without an effective-degree-of-freedom calculation.
- **R24 / P2 — output backgrounds are not the fitted BCD backgrounds:**
  `main.py:64` replaces them with epoch averages in the per-BCD CSV. Export
  `bcd_backgrounds`; keep epoch summaries separate and document their estimator.
- **R25 / P2 — GP estimation is not matched to the observation model:**
  `src/gp_model.py:96` fits a covariance directly to one observed stamp without
  PRF, background or mean treatment; small-stamp slicing can select an unintended
  corner. Fixed log-variance bounds and absolute 1e-6 covariance jitter need
  assessment in Jy units. Optimize a likelihood using the actual forward model
  or explicitly label this an initialization heuristic. Check optimizer success.
- **R26 / P2 — memory and repeated work:** full dense H, Q, inverse covariance
  and impulse-built operators scale quadratically/cubically. A single 20,000²
  float64 matrix is 3.2 GB, before workspaces and copies. The supposedly scalable
  GP fallback allocates `np.eye(n_scene)` too. Exact PRFs repeatedly read large
  FITS files. Use matrix-free/sparse operators, factorization reuse, selected
  covariance solves, bounded caches and an explicit memory budget.
- **R27 / P2 — footprint construction:** `native_fit_campaign.py:53` uses a
  small-angle RA approximation with positive RA offsets but negative RA CDELT
  when setting CRPIX. Asymmetric extents, RA wrap and distortion need regression
  coverage. The budget loop can return an over-budget grid after 20 attempts.
  Build bounds in the actual tangent WCS, include edges, and raise on failure.
- **R28 / P2 — p-map cache and broadcasting:** cache key omits `pmap_dir`
  (`src/pmap_correction.py:25`); changing calibration directories reuses old data.
  Scalar x/y with vector fluxes are not broadcast before indexed access; errors
  are swallowed into missing output. Validate/broadcast inputs and include source
  identity in the cache key.
- **R29 / P2 — reproducibility:** manifest contains paths and a dirty boolean,
  but not input/calibration checksums, uncertainty/mask files, applied astrometric
  shifts, rejected-file reasons, optimizer status or the dirty patch identity.
  Write a run-start manifest and completion/failure record in a unique run directory.
- **R30 / P2 — input/catalog robustness:** discovery order is not sorted; dates,
  missing uncertainty files and broad exception catches can silently drop data.
  Catalog identities use basenames and a shared deep-template name without
  channel/config provenance. Spherical source matching should replace Euclidean
  RA/Dec matching, particularly at RA wrap and high declination. Include explicit
  ingestion reports and a deterministic catalog invalidation policy.
- **R31 / P2 — unused legacy module is not robust:** `src/postprocessing.py`
  imports undeclared pandas, fails on empty inputs before its empty check, and
  returns zero flux/error if every uncertainty is unusable. Decide whether this
  remains a supported interface; test or deprecate it explicitly.
- **R32 / P2 — test/CI blind spots:** global fixtures turn off floated positions,
  positivity and multiple PRF anchors; tests often assert finite arrays or inspect
  source strings instead of recovery. Slow tests are excluded entirely and local
  calibration presence changes runtime/behavior. Separate deterministic unit
  fixtures from real-calibration tests; exercise production settings explicitly.
- **R33 / P3 — maintainability:** global mutable config, function attributes used
  as campaign arguments, duplicated channel/mask/model logic, broad warning
  suppression, bare exceptions, stale docstrings, hard-coded experimental paths
  and overlapping drivers obscure contracts. Refactor only after numerical
  regression coverage; add typing at the input/result/model boundaries first.

## Repair sequence and acceptance gates

### Phase A — establish reproducible failure tests and safe execution

1. Preserve the current dirty tree as a reviewed patch/checkpoint; do not bundle
   existing untracked work blindly into a commit. Use remote main as the reference.
2. Add focused reproductions for R01, R06, R09–R12, R17, R22, R24 and R28.
3. Fix validated metadata, mask provenance, partial PRF loading, sparse precision
   handling and failure propagation. A failed fit must not write a success light curve.
4. Remove unsafe cache reuse and record fitted configuration in results.
5. Establish a clean, reproducible Python environment; keep scientific inputs
   outside unit-test discovery and output under per-test temporary directories.

Gate: targeted failures turn green, invalid-input tests raise actionable errors,
and the same seeded results are obtained cold/warm and in reversed test order.

### Phase B — unify the forward model and geometry

1. Resolve units, detector integration, flux-sign and mask-policy questions below.
2. Preserve full-detector calibration coordinates; construct a north-up latent
   grid covering every exposure at a fixed sky footprint and specified SR.
3. Render static scene, point sources, host and positional derivatives through
   the same operator. Include finite-support losses and nonperiodic boundaries.
4. Compare the matrix-free operator and adjoint to a small explicit design matrix.
5. Remove approximate likelihood Hessians; use approximations only for acceleration.
6. Correct native-vs-sky stacks, masks, footprints and dipole displacements.

Gate: numerical adjoint error below 1e-10 on well-conditioned small fixtures;
matrix/renderer equality to floating-point tolerance; conserved flux and constant
surface brightness within a predeclared discretization tolerance. Sweep SR=1,2,4,
rotation, distortion, subpixel phase, detector position and distance from edges.
Do not pass by renormalizing a cropped output stamp.

### Phase C — solve the intended statistical problem

1. Implement a converged bounded solve with scaled variables, proper block
   indices and KKT diagnostics; test both sides of the former 2,000-parameter branch.
2. Iterate position fitting to a stated convergence threshold; verify finite
   differences and recover known offsets jointly with flux and backgrounds.
3. Make all priors explicit, including GP-off, monotonicity and star constraints.
4. Compute nuisance-marginalized flux covariance using selected solves rather
   than a full inverse; propagate covariance into the chosen epoch estimator.
5. Validate GP hyperparameters and uncertainty treatment under active bounds.

Gate: tiny problems agree with an independently assembled weighted least-squares
or constrained reference; all accepted optimizations meet residual/KKT criteria;
injection pulls and interval coverage agree with their intended probabilities
within Monte Carlo uncertainty. Report rank deficiencies rather than zero errors.

### Phase D — diagnostics and real-data validation

1. Fix ACF, dipole, chi-square, RMSE units, per-BCD background exports and all
   configuration-dependent predictions. Recompute campaign scores consistently.
2. Select a small immutable real-data benchmark with both channels, science and
   templates, several rolls/dithers, masks and a bright host. Record checksums.
3. Compare independent-pixel and GP models on identical pixels and fixed noise;
   use held-out BCDs/epochs. Tuning weights to reduce weighted chi-square alone
   cannot demonstrate improved scientific accuracy.
4. Inject positive and zero transient fluxes over a grid
   of phases, backgrounds, host gradients, positions and S/N. Include independent
   analytic/pixel-integrated truth, not only injections through the fitted renderer.
5. Use held-out BCDs and independent analytic injections; no external transient
   reference exists. Calibrator stars may test instrument normalization separately.

Required plots: data/model/residual/normalized-residual maps with WCS and units;
PRF profiles and encircled energy; flux and position bias versus SR, phase and
detector position; injection recovery and pull histograms; interval coverage;
light curve plus within-epoch consistency; residual ACF and lag correlations;
mask overlays; held-out prediction errors; solver convergence; runtime/peak memory
versus scene pixels and BCD count. Save numerical arrays beside plots.

Gate: predeclared tolerances on flux bias and uncertainty coverage, no unexplained
structured residual trend with rotation/phase, and documented comparison to an
independent synthetic truth. Exact scientific thresholds require the user's precision
requirements; they must not be chosen after looking at recovery results.

### Phase E — GitHub CI and delivery

Update `.github/workflows/pytest.yml` after regression fixtures are deterministic:

- Required PR jobs on supported Python versions (start with 3.11 and 3.12):
  install declared dependencies, `pip check`, import/compile checks, focused lint
  checks, unit tests and small physics/injection recovery tests. Use noninteractive
  Matplotlib, temporary config/output paths, bounded BLAS threads and job timeouts.
- Explicit production-setting integration job: floated position, positivity,
  multiple anchors, both operator orders/channels, empty/invalid input and CLI
  failure behavior. Do not inherit the fixture that disables these behaviors.
- Extended regression job on manual dispatch and a chosen periodic schedule:
  full synthetic suite, calibrated PRF tests and real-data benchmark if licensing,
  size and access permit. Verify fixture checksums; never silently substitute a
  Gaussian or skip required calibration tests in this job.
- Upload JUnit XML, coverage, run manifests, numerical diagnostic JSON/arrays and
  PNG/PDF plots even on failure. Document any genuinely optional real-data skips.
- Add duration/memory tracking and evaluate compatibility against a pinned
  reproducibility environment plus a controlled dependency-upgrade job.
- Replace source-string tests and self-tests of NumPy with tests of this package's
  actual behavior. Move cheap PRF resolution assertions into required CI.
- Keep each repair group reviewable; run its regressions before checkpointing.
  Only mark a finding resolved after its behavioral regression and validation pass.

## Scientific decisions and outstanding acceptance criteria

The user confirmed GP interpolation, nonnegative flux, one-day sharing, internal
validation, preservation of useful local edits, and replacement of the bright-core
unmasking workaround with appropriate rejection. Precision requirements remain to
be specified. Both channels are supported separately; a joint color model is not
part of this repair. Optional host/nuclear priors require separate validation.

## Verification performed

- Live remote branch-tip queries and local ancestry/dirty-tree inspection.
- Baseline non-slow suite: 70 passed, 3 skipped, 13 deselected in 1032 seconds.
- Original PRF resolution tests: 4 failed, 1 passed. All five now pass in required CI.
- Latest complete fast run before the final metric/cache updates: 94 passed,
  3 skipped, 8 deselected in 20.25 seconds. Final results are recorded below.
- Added a floated-position recovery test with positivity enabled, and an explicit
  test of Jy residual RMS versus saved effective uncertainty; both pass.
- Nine seeded synthetic trials (three seeds at each SR=1,2,4): maximum absolute
  recovery pull 0.595914. Shared seeds across SR are correlated; this is a regression
  check, not an independent nine-trial uncertainty-coverage measurement.
- Numerical JSON and plots: `output/review_validation/internal_consistency.json`,
  `residual_diagnostics.png`, and `flux_recovery.png`. The arrays support the
  recovery result; plot appearance has not been used as an acceptance criterion.
- Extended tests initially exposed six oversized legacy smoke configurations.
  Exact solves now have a memory guard; synthetic smoke grids were reduced while
  retaining N=1,2,10 BCD cases. Extended rerun status is recorded below.
- `.venv` pytest exits 139 before reporting tests; `.local_env` works (Python
  3.11.15). The broken environment was not silently replaced or repaired.
- CI now runs Python 3.11/3.12 fast tests, dependency consistency, synthetic recovery
  and diagnostic artifacts; manual dispatch adds the extended synthetic suite.
  CI has been edited locally, not executed on GitHub.
- No end-to-end real science fit or calibrated-PRF acceptance is claimed.

### Completed full-suite verification

The complete synthetic/unit suite passed: **108 passed, 3 skipped**, 430.72 seconds.
Three warnings explicitly exercise the smoothed-diagonal GP fallback in the legacy
N=1,2,10 tests; those cases are not evidence for production Matérn interpolation.
The small injection and floated-position tests separately exercise the GP path.
The older extended run also passed all eight tests but took 9175 seconds while its
first subprocess retained real-calibration behavior. No review test processes remain
from those runs. JUnit is saved under `output/review_validation/pytest.xml`.

Nine additional zero-flux trials pass with nonnegative recovered amplitudes; largest
positive pull is 0.595914. Boundary estimates have the expected clipping behavior;
these plots do not establish confidence-interval coverage. Products are under
`output/review_validation/zero_flux/`.

`compileall` and `git diff --check` pass. Local `pip check` reports an unrelated
installed bqplot 0.12.37 requiring NumPy<2 and pandas<2, while `.local_env` has
NumPy2.4.3/pandas3.0.2. No shared-environment packages were changed. CI installs the
repository requirements into clean Python environments and checks them there.

Final fast verification after the ingestion, legacy postprocessing and diagnostic
updates: **101 passed, 3 skipped, 8 deselected**, 34.72 seconds. These final changes
were covered by the fast suite; the 108-test full run preceded the last postprocessing
and ingestion edits. Repairs are checkpointed on `codex/review-scientific-correctness`;
unrelated pre-existing work remains separate. Nothing was pushed.
