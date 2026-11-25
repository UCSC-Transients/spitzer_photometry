# Stage 4: positive diffuse GP and component identifiability

This is an **experimental small reference posterior**, not a production-default
change or scientific promotion. It answers whether source flux can move into a
diffuse host or nucleus and exposes failures of the static model. The independent
synthetic experiment finds measurable leakage, position sensitivity, and strong
model mismatch. No negligible-leakage claim is made. A science-bias tolerance has
not been supplied.

## Launch and ownership

- Run: `20261125-03`; supplied exact base:
  `2d5bcb40babab49f761496c331e4b28b327f6448`.
- Target `main`, pulled with `--ff-only` in the separate target worktree:
  `9805b979d5e3e6459b0f7f0b915f87a29e892e7f`. Its tree equals the supplied base.
- Stage 1/2/3 PRs #23/#24/#25 are integrated. The independent Stage 3 review of
  the supplied head permits experimental Stage 4. Its nonblocking RV25-01 scoring
  failure-retention follow-up is unchanged; this stage does not call that
  `compare_models` path.
- Author branch: `codex/04-gp-identifiability-20261125-03`; checkout:
  `/Users/rfoley/spitzer_photometry/.worktrees/author-04-gp-identifiability-20261125-03`.
  The separate `base-04-gp-identifiability-20261125-03` checkout pins the exact
  supplied base. The original dirty checkout is untouched.
- The Stage 3 review Python 3.11.15 environment is used read only; 45 installed
  packages pass dependency consistency. All caches, temporary files and results
  are author-worktree-local. No new dependency is introduced.
- The only shared CI edit adds the small Stage 4 artifact consumer to the existing
  required jobs. The Stage 1 benchmark interface and gates are unchanged.

## Physical model and units

`src.gp_identifiability.StaticData` contains masked native measurements, their full
noise covariance, and a matrix `[F, P_stars, P_nucleus]`. Host coefficients have
units Jy/native pixel (native-equivalent surface brightness); source amplitudes
have the integrated-flux convention of the supplied calibrated PRF. The angular
host-knot coordinates are explicit in arcseconds. Matérn 3/2 scale is in arcseconds
and amplitude is a standard deviation in Jy/native pixel. There is no hidden scene
supersampling conversion, diagonal-GP replacement, noise floor or sqrt(Jy) proxy.

`template_design` preserves supplied established host operators. The validation
adapter snapshots `solver._apply_frame_forward_operator`; a regression compares
its full action with the unchanged F. Native stars use the Stage 2 renderer and
the compact nucleus uses the Stage 3 point or intrinsic-Gaussian integral. An
explicit point-core/envelope option is also available. Point responses are never
renormalized to a crop or integrated over detector pixels again. Existing host
operator finite-support behavior is retained and remains a measured limitation.

All physical coefficients are nonnegative. Write the host as `H = m + g`, with
Gaussian perturbation `g`, then condition the **joint prior** on `H >= 0` and all
source amplitudes `>= 0`. A signed perturbation is allowed only while total host
brightness stays nonnegative. A nonnegative interpolation basis preserves this
between grid nodes. The supplied F must preserve positivity, allowing only
floating-point roundoff. Independent signed background calibration is subtracted
and its Gaussian uncertainty enters the full within-exposure noise covariance.
Uncertain registration is rejected explicitly; it must be represented by separate
geometry alternatives rather than silently ignored.

The host mean and half-normal source prior scale are declared inputs. They are
not estimated from recovered transient flux. This experiment uses explicit
forward/noise-aware **marginal likelihood**, not a covariance fit to an observed
stamp or a kernel chosen to hide a nuclear peak.

## Posterior, evidence and uncertainty

Before truncation, the parameter-space Gaussian update integrates the full
`N + A K A.T` observation covariance, retaining every cross term. Let `Z_G` be
that normalized Gaussian evidence, `q_prior` the prior positive-orthant mass and
`q_post` its posterior counterpart. Then

    Z_positive = Z_G * q_post / q_prior.

Source-prior orthant mass is analytic (`2**(-number_of_sources)`); host and
posterior masses use independent Gaussian rejection proposals. Accepted draws
are IID from the truncated posterior, so active bounds do not receive a clipped
Gaussian interval. Two 97.5% Wilson mass bounds give Bonferroni approximate 95%
evidence bounds; these are not simultaneous bounds across an entire candidate
grid. Insufficient accepted counts raise `SamplingFailure` with diagnostic counts.
Proposal counts, seeds, Monte Carlo mean errors and Gaussian normal-equation
residuals are saved. This is sampling/integration, not an optimizer with a KKT
termination criterion.

Template evidence weights cover GP-only, GP+stars, GP+stars+point nucleus and
GP+stars+resolved nucleus on a declared scale/amplitude grid. Prior mass is
explicit; refining a grid must divide existing prior mass rather than reward a
model for having more cells. Failed candidates make a comparison incomplete and
must not be dropped before renormalizing. Predictions include within-component
covariance and the between-model/hyperparameter covariance term. Evidence-weight
Monte Carlo bounds are reported separately; moments use central weights.

The saved diagnostics include component correlations, a prior-scaled weak
likelihood mode, singular values/rank, and a conservative training
posterior-predictive discrepancy. The latter uses a predeclared `1e-4` flag;
it is **not** a calibrated frequentist test or a science-bias tolerance. A model
mixture whose candidates all flag mismatch is explicitly labeled
`all_candidates_misspecified`, even when its conditional weights are concentrated.

Exact coincidence remains prior-dominated with broad, anticorrelated component
uncertainty. Missing/unverified templates fail before selection. Deliberately
variable contamination gives a static-mismatch diagnostic. Constant contamination
indistinguishable from a static source cannot be detected from these pixels alone:
verified-template provenance is a substantive assumption, not a date selection.

## Stage 6 contract: count templates once

For static parameters `theta`, candidate `j`, templates `T`, science `S`, and
nonnegative epoch flux `f`, Stage 6 must use

    L(f) = sum_j p(j | T) integral p(S | theta, f, j) p(theta | T, j) dtheta.

`conditional_log_likelihood` evaluates a single candidate from its positive
posterior draws. `mixture_conditional_likelihood` includes template-derived
candidate weights and requires identical science pixels/noise across candidates.
Low importance ESS is an explicit numerical failure diagnostic. `predictive_evidence`
uses a further Gaussian update plus orthant correction for held-out template
scoring, avoiding importance collapse. Neither interface refits hyperparameters on
science data. Reusing training exposure IDs raises rather than multiplying the
template likelihood twice. Exact small Gaussian joint/sequential equivalence and
analytic truncated-normal evidence ratios are numerical regressions.

Stage 6 still owns nonnegative epoch inference, total-span <=1-day grouping,
template transient flux exactly zero, position uncertainty, boundary detection and
coverage. This stage changes none of those production contracts. No Stage 6 light
curve or scientifically calibrated flux interval is claimed.

## Validation, evidence and limits

`scripts/validate_stage4.py` consumes the Stage 1 artifact writer and Stage 2/3
independent optical truth. It integrates a continuous asymmetric broad host and
point/extended components over native pixels; truth is not drawn from the fitted
GP. Three development noise seeds, four source fluxes, separation, host gradient,
subpixel phase, masks, template counts, rolls, a centroid mismatch, extended
nucleus and contamination are retained. It reports recovery/transfer slopes with
small-N descriptive intervals, Jy bias/RMSE, positive interval coverage, zero-flux
upper limits, covariance, held-out likelihood and prior sensitivity. The reserved
Stage 1 acceptance seeds/exposure IDs are untouched.
Host transfer is the fitted host's point-response projection in the native noise
metric, not its integrated flux over the entire footprint. Transfer slopes may be
signed even though every physical component amplitude remains nonnegative.

The required quick CI run uses the same physical/noise cases with fewer seeds and
grid cells; it includes the near-nucleus and masked/few-template cases. The full
experiment has 72 candidate fits plus 192 injection fits. Repeated flux levels use
paired noise within a seed; three independent seeds do not establish coverage.
Continuous positive-prior equal-tail intervals exclude exactly zero; zero-flux
upper limits are reported separately and their trivial inclusion of zero is not
called detection/coverage validation.

All 72 scale/model candidates with the deliberately small 9x9 host footprint flag
strong model mismatch, including the separated-source case. The contributions
of finite support, prior restrictions and inherited operator approximations have
not been isolated. These results are not validated host recovery.
SR=1/2/4 tests check sky-coordinate-unit invariance with fixed angular GP knots;
they **do not** certify fine-grid or footprint convergence. Full supported volume,
larger footprints and true fine-grid convergence remain Stage 5/acceptance work.
Other open gates: an agreed science-bias threshold, independent real-template
purity/registration manifest, absolute warm PRF normalization, real extended-profile
quadrature, uncertain star/nucleus geometry, continuous shape/kernel priors,
general variable stars and more independent coverage trials. The offset-injection
experiment quantifies one omitted geometry effect; it does not integrate it away.

Real-table tests preserve source-column identity using the actual strict warm ch2
table. They are labeled algebra, never real-template or absolute-flux validation.
No private FITS bytes are committed or uploaded.

Reproduction (use isolated output/cache paths and one numerical thread):

```bash
python -m pytest -m 'not slow and not real_calibration'
python scripts/validate_stage4.py --draws 16384 --output-dir test-results/stage4
python scripts/validate_stage4.py --quick --draws 4096 --output-dir test-results/stage4-ci
python scripts/run_stage1_benchmarks.py --base-repo /exact/base --head-repo /exact/head \
  --output-dir test-results/stage4-paired --repeats 5 --timeout 45
```

Local evidence lives under `.test-artifacts/stage4/` in the author worktree:
source/ref hashes, environment and command/status logs, JUnit, failed first
attempts, full arrays and checksums, leakage/scale/correlation plots, calibration
identity, and paired benchmark samples. The first experiment retained unnecessary
whole posterior arrays through sliced views (2.47 GiB peak RSS); the final runner
copies only required component draws. Both runs remain available. Plot appearance
is not an acceptance criterion; numerical claims come from saved arrays.

The compact [public numerical evidence](STAGE4_EVIDENCE.json) and
[leakage/scale/correlation figure](STAGE4_LEAKAGE.svg) are committed synthetic-only
products, with source hashes and seeds. Complete NPZ draws, covariances and input
matrices remain in the local artifact bundle.

The unchanged Stage 1 common-operator benchmark is the paired regression measure.
The new posterior has no base implementation, so its solve/experiment timings and
memory are reported separately. No speedup is claimed. Exact final measurements
and head SHA are recorded in the PR and local final record.

Independent reviewer handoff: GPT-6 Astra/xhigh, fresh detached checkout at the
PR head, using `PROMPT_REVIEW_PR.md`. Focus on truncation normalizers, full noise
and component covariance, evidence-weight uncertainty, template likelihood reuse,
failure retention, finite-footprint mismatch, prior dependence and the unchanged
paired gates. Code readiness is experimental; scientific promotion stays unapproved.
