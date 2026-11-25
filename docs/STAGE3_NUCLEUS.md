# Stage 3: template nucleus discrimination (experimental)

This adds an opt-in reference experiment. The simplest default is a point nucleus
with a common floated sky centroid. Gaussian and point-core-plus-Gaussian-envelope
alternatives remain explicit. No model is automatically promoted, no production
pipeline default changes, and no flexible local GP or transient parameter is added.
Scientific promotion is **unapproved**. In particular, optimizer failures, local
uncertainty approximations, real-profile quadrature, template provenance, and the
unspecified science-bias tolerance prevent a claim of real-nucleus discrimination.

## Launch and dependencies

- Supplied `BASE_REF`: `d63385e6e0560cb505e6fda3d21184428885de02`.
- Launch `PR_BASE=main`: `74024330cc0189730f9087aad85a0819c9ddb8f2`.
  Stage 1 PR #23 and Stage 2 PR #24 are merged. The supplied base is the reviewed
  Stage 2 head and has exactly the same tree as that merge commit.
- Run `20261125-02`; author branch `codex/03-nucleus-20261125-02`;
  worktree `.worktrees/author-03-nucleus-20261125-02`.
- Separate benchmark base: `.worktrees/base-03-nucleus-20261125-02` at the exact
  supplied base. The original dirty checkout was read only.
- Read the science model, work orders, original review, and Stage 2 independent
  review before writing. That review allowed Stage 3 and required callers to
  reject mixed channels/amplitude conventions; this interface does so.
- The clean Stage 2 review Python 3.11.15 environment is shared **read only**.
  `uv pip check` passes (45 packages); a freeze is saved locally. Every output,
  cache, temporary directory, and test result is under this worktree. No packages
  or caches are written into that environment. No new dependency is introduced.

## Scientific model and interface

`src.nucleus.TemplateFrame` wraps the immutable Stage 2 `StarFrame`, a unique
exposure ID, an explicit verified-template assertion, provenance, a partition,
and independent registration/background calibration. Input amplitudes share one
ch2 mission/normalization convention. The registration covariance must be positive
definite or exactly zero for a fixed gauge. Its measurements must be external or
come from disjoint field-star pixels; passing calibration derived from the same
nuclear pixels would double-count data. No automatic catalog or registration
calibration is inferred by this stage.

`NucleusScene` defines the reference sky WCS, initial nuclear centroid, fixed
catalog star positions, and a controlled broad positive host profile. All physical
amplitudes are nonnegative. Per-frame background corrections may be signed.
Catalog positions and the *shape* of the smooth host are conditional inputs;
propagation of their uncertainty remains a limitation. There is no hidden GP,
noise floor, Poisson surrogate, or residual-driven rescaling of sigma.

`fit_nucleus(training_frames, scene, model)` fits:

- a shared nuclear sky centroid;
- one nonnegative nuclear amplitude for `point` or `gaussian`, or two for
  `core_envelope`, with a common centroid;
- Gaussian sigma x, sigma y, and correlation for resolved candidates;
- one nonnegative shared amplitude per explicit neighbor and for the smooth host;
- detector translations with the independent Gaussian registration likelihood;
- signed background corrections for training BCDs.

The nucleus centroid and frame registration have separate parameters. The sky
reference gauge is fixed by catalog positions and the supplied registration
calibration, not by recentering the whole scene on the nucleus. WCS supplies rolls
and distortion; rotation/distortion inference and variable stars remain untested.

`gaussian_response` integrates the existing native point response over **intrinsic
sky source positions**, using positive Gauss–Hermite weights. It never integrates
a detector PRF over native pixels again, blurs native images after sampling, or
renormalizes a cropped stamp. Each quadrature node is mapped through WCS; a real
`PRFGrid` selects full-detector calibration at that node. The zero-width limit
calls the native point renderer exactly. Analytic derivatives follow the same
bilinear calibration interpolation; the WCS Jacobian alone is differenced. Grid
providers retain finite differences including spatial calibration variation.

The reference implementation has a 2,000,000-element dense-Jacobian limit and
bounded quadrature chunks. It is not certified for production-volume inference.
Gauss–Hermite orders are explicit configuration, with convergence comparisons
saved separately. Real-table tests reveal remaining numerical quadrature error;
all real-profile fitting (including the host integral) therefore requires the
explicit `experimental_real_quadrature=True` opt-in. This is not a science approval.

## Predictive comparison and uncertainty

Training, validation, and final held-out **exposures** are separate and duplicate
IDs are rejected. Predictions use external registration and background calibration
for unseen BCDs. Neither their nuclear pixels nor their star fluxes are refitted.
`compare_models` reports all three models and complexity, validation scores, and
untouched held-out scores. It retains `point` as the default without selecting a
model from lower training chi-square. Descriptive BIC is marked nonregular when
bounds/priors invalidate ordinary asymptotics; it is not a promotion criterion.

Interior covariance includes amplitude, centroid, profile, registration, and
background cross terms. Prediction uses the marginalized global covariance and
adds unseen-frame calibration covariance. External calibration errors are assumed
independent across frames and between backgrounds and registration; a general
correlated calibration posterior is not implemented. The likelihood retains the full
`diag(sigma²) + U Uᵀ` covariance, including covariance **between** held-out
exposures; a Woodbury calculation avoids a dense pixel covariance. A dense small
regression verifies that likelihood. It is a local Gaussian approximation, not
validated interval coverage or integration over model families. At active bounds,
rank loss, or search-domain limits, covariance is unavailable. Conditional scores
are labeled separately and must never be compared as though they had propagated
uncertainty. No clipped symmetric interval is reported at zero flux.

Alternative model predictions are saved as a range, with no invented posterior
model probabilities. That range and all retained candidates must be carried into
later host/transient work if template evidence does not resolve morphology.
There is no Stage 6 conditional/joint transient posterior in this PR.

The convergence gate remains scaled optimality <= 0.001 and an optimizer success
status. Fresh trust-region attempts use the same objective and gate; rejected
attempts retain parameters, Jacobian, objective history, training model, timings,
and explicit failure status. A low chi-square or optimizer `ftol` message alone
never makes a fit successful. Bilinear-PRF knots and weak resolved profiles remain
problematic in some deliberate mismatch/near-point trials.

## Truth cases and interpretation

Synthetic truth independently integrates analytic optical Gaussian components over
native detector pixels. The fit uses separately tabulated, detector-integrated
responses and source-position quadrature. These are labeled **synthetic**, never a
replacement for IRAC calibration. Development seeds are 1729, 2718, and 31415;
reserved Stage 1 acceptance seeds/exposures are untouched.

The four truth families separate a common centroid offset, symmetric finite
extent, an anisotropic detector PRF mismatch affecting both nucleus and stars,
and a genuinely asymmetric two-component sky nucleus. Progression includes one
BCD, same-roll dithers, multiple rolls, and SR=1/2/4 at fixed sky footprint.
The asymmetric case's reported truth centroid is its brightness-weighted centroid.

Interpretation is deliberately limited:

- A common sky centroid error produces an odd derivative pattern; floating that
  centroid removes it in the matched point truth. One BCD cannot generally tell
  such an offset from a shifted PRF or an asymmetric unresolved source.
- Symmetric finite extent leaves predominantly even core/wing structure. In the
  matched resolved truth, a Gaussian improves independent predictions and flux
  recovery. A coincident point core can hit zero, leaving its covariance unavailable.
- Detector-fixed PRF errors affect neighboring stars and change sky orientation
  with roll. Sky asymmetry remains tied to sky geometry. Dithers/rolls and explicit
  stars provide discrimination, but convergence failures prevent a universal
  classifier or a confident morphology claim in the mismatch experiment.
- A sufficiently weak or unresolved envelope and a point source remain ambiguous.
  A smooth-looking residual or an unavailable alternative fit does not establish
  that the point model is physically correct.

Signed derivative coefficients have covariance and an approximate Bonferroni
correction across the declared frame/model family. Only held-out results with
available predictive covariance admit that local Gaussian interpretation;
training statistics and conditional-boundary results are descriptive. Plots of
pointwise normalized residuals are not full whitening. A nearby source is injected
at multiple nonnegative fluxes into the residual experiment. Its response is
projected with a fitted constant background, and nonnegative recovered flux is
reported. This measures possible **fixed-host** contamination, not uncertainty-
propagated joint transient recovery. Failures and conditional coverage are retained.

## Measured development results

Final required fast suite: **438 passed, 5 skipped, 13 deselected** (65.75 s).
Relevant campaign integration: **3 passed** (231.15 s). Real-calibration algebra:
**4 passed** (0.24 s), including the Stage 2 tests. Compile checks and
`git diff --check` pass. Local Python 3.12 was not run; the existing PR CI covers it.

Across four truth families, one noiseless run and three noise seeds, **36 of 48
candidate fits converge and 12 are explicitly rejected**. All 12 additional
single/dither/SR progression fits converge. The stress command exits 1 because of
the rejected fits; this is an unmet robustness gate, not a successful science run.
The original finite-difference experiment rejected 22/48 and the first analytic
version rejected 14/48; those logs remain. No difficulty, sigma, or gate changed.

For the 10 mJy synthetic resolved nucleus, mean recovered nuclear-flux bias over
three noise seeds is **-1.06485 mJy** for a point fit versus **+0.00702 mJy** for a
Gaussian. Local 95% flux intervals cover 0/3 versus 3/3 respectively; three seeds
do not establish coverage. Noiseless held-out nearby-source contamination falls
from **245.33 microJy** (point) to **0.211 microJy** (Gaussian). Under asymmetric
truth it remains **24.21 microJy** even with the symmetric Gaussian, versus
184.14 microJy for the point model. A point fit to noiseless detector PRF mismatch
leaves 268.20 microJy of possible nearby-source contamination. These are signed
projection magnitudes in the synthetic instrument's declared flux units.

Point-truth mean point-fit bias is +0.01071 mJy (3/3 local intervals cover).
Complete bias/RMSE/descriptive confidence intervals, unavailable intervals, failed
fits, per-frame structure statistics, and probe recovery slopes are in
`evidence/summary.json` and `evidence/metrics.json`. These results motivate model
uncertainty and calibration diagnostics, not a universal resolution claim.

## Real inputs and noise

The read-only local inventory contains 516 ch2 CBCDs, of which 234 satisfy the
existing pipeline's date rule. No independent transient-free/registration manifest
was supplied, so these are **candidates**, not verified templates and not fitted
real-template science. The inventory preserves image/uncertainty checksums and
BUNIT, gain, exposure, FLUXCONV, pixel-scale and date metadata. Private FITS bytes
are never committed or uploaded.

The inspected CBCD/CBUNC products both use MJy/sr. A caller must convert both by
`1e6 * native_pixel_solid_angle_sr` under one documented pixel-area convention.
Do not reapply gain, exposure or FLUXCONV after this calibrated-unit conversion.
The archive uncertainty estimate already incorporates pipeline read/Poisson and
propagated terms; it is not a complete detector-systematic or absolute-flux error
model. No extra sqrt(Jy) term or empirical dipole-hiding floor is added. See the
[IRAC noise definition](https://irsa.ipac.caltech.edu/data/SPITZER/docs/irac/iracinstrumenthandbook/50/)
and [pipeline uncertainty propagation](https://irsa.ipac.caltech.edu/data/SPITZER/docs/irac/iracinstrumenthandbook/25/).

Real-table algebra uses the strict warm ch2 calibration with SHA-256
`d34f398ba1e1305aab4d1df0e898e997f61dbc3a0437c3c1532382b01044a8b8`.
The point limit is bitwise identical. Relative norm differences between quadrature
orders 11 and 41 are 0.0552%, 0.2400%, and 0.3175% at widths 0.1, 0.4 and 0.8 native
pixels. These are measured discretization differences, not bounds on truth error.
Absolute warm flux normalization remains the Stage 2 open calibration gate.

## Reproduction and artifacts

Run the required numerical regressions with `pytest -m 'not slow and not
real_calibration'`. The existing required Python 3.11/3.12 CI discovers the Stage 3
tests; no numerical test is excluded and no CI/performance budget is rewritten.
The full stress experiment may exit nonzero when candidate optimizations fail;
its bundle still records the rejected fits and completed comparisons.

```bash
python scripts/validate_stage3.py --output-dir test-results/stage3
python scripts/summarize_stage3.py test-results/stage3
python scripts/audit_stage3_inputs.py --raw-dir /read-only/raw \
  --prf-dir /read-only/prfs --output-dir test-results/stage3-real
python scripts/run_stage1_benchmarks.py --base-repo /exact/base \
  --head-repo /exact/head --output-dir test-results/stage3-paired \
  --repeats 5 --timeout 45
```

Detailed local evidence is in `.test-artifacts/stage3/` of the author worktree:
launch/source hashes, environment freeze, every command/status/log, JUnit, failed
attempts, synthetic NPZ/JSON/plots and checksum manifests, real input audit and
real-table algebra, and paired benchmark samples/arrays. Earlier failed numerical
runs are retained. Plots are not used as quantitative acceptance evidence; metrics
come from their saved arrays. No claim depends on visual inspection.

The unchanged Stage 1 common-operator benchmark is the paired regression measure;
the new fitter has no base equivalent. New-fitter setup/solve timings, renderer
counts, Jacobian sizes and process RSS are reported separately. No speedup is
claimed. Final measurements and exact head are recorded in the PR and local
summary, since the commit cannot contain its own SHA.

## Reviewer handoff and unmet gates

Review the exact head with `PROMPT_REVIEW_PR.md`, GPT-6 Astra/xhigh, in a separate
detached checkout. Focus on native/profile derivative identity, the point limit,
registration gauge, conditional calibration provenance, constrained/degenerate
covariance, cross-frame predictive likelihood, failure retention, real quadrature,
and the full paired arrays. No messages or reviews were posted elsewhere.

Unmet: a declared science-bias tolerance; independently verified real templates;
absolute warm calibration; real-profile numerical integration acceptance; robust
optimization across every mismatch/unresolved case; calibrated boundary/global
model uncertainty and coverage; uncertain host shape/catalog positions; variable
stars; inferred rotation/distortion; and production-volume scaling. Keep model
alternatives explicit and promotion unapproved. The PR does not authorize merge.
