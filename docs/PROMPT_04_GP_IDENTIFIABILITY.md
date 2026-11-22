# Stage 04-gp-identifiability: copy this prompt into a new implementation chat

Select **gpt-6-astra / xhigh** in the chat settings first.

```text

You are an implementation agent for UCSC-Transients/spitzer_photometry. First read
/Users/rfoley/spitzer_photometry/docs/STAGED_SCIENCE_MODEL.md and
/Users/rfoley/spitzer_photometry/docs/AGENT_WORK_ORDERS.md, plus applicable repository
instructions and the original CODE_REVIEW_2026-11-20.md. Read these source documents
before creating your worktree if they are not yet in BASE_REF. Follow this work
order; the plan is approved. Work within this stage, through implementation,
verification, a scoped commit and a GitHub PR. Do not merge the PR.

Launch inputs: BASE_REF=<verified approved dependency ref>,
PR_BASE=<GitHub target branch>, RUN_ID=<unique suffix>. Resolve these to exact SHAs
and record them. If a required dependency is missing, do read-only preparation and
ask for the missing ref; do not implement against an invented replacement.

Create your own branch codex/04-gp-identifiability-RUN_ID and worktree
/Users/rfoley/spitzer_photometry/.worktrees/author-04-gp-identifiability-RUN_ID from BASE_REF.
Check branch/path/HEAD and worktree inventory before writes; never reuse an existing
path or mutate the original dirty checkout. Keep outputs/config/caches separate;
read-only data can be shared. Do not copy unrelated local edits. Git metadata or
network operations may require normal tool approval; never bypass it. Subagent
writers require their own worktrees, never your checkout. Use the explicit model/
effort assignments and six-bullet reporting contract in AGENT_WORK_ORDERS.md.

Maintain the nonnegative point-source flux, <=1-day shared epoch, template-zero,
GP-host interpolation and native calibrated-PRF contracts. Never reduce a test's
physical difficulty, inflate errors, substitute a Gaussian calibration, weaken a
performance gate or alter a science threshold solely to obtain a pass. Keep
synthetic and real-calibration results labeled. If absolute science tolerances are
missing, report measured bias/coverage and leave promotion unapproved; proceed
with tests and implementation that do not depend on that decision.

Run targeted regressions, relevant integration tests and paired base/head
benchmarks. Use the same inputs/thread count for base/head; record failures as well
as timings and memory. Validate physics before claiming speedup. Save seeds,
checksums, config, numerical arrays, plots and convergence diagnostics. Document
limitations and untested regimes. Keep numerical tests in required PR CI; use the
Stage 1 benchmark interface rather than independently rewriting CI. Coordinate
necessary shared-file changes through a scoped commit.

Push only your scoped branch and create a PR against PR_BASE, with problem/result,
scientific assumptions, validation and performance comparison, artifact locations,
and unmet acceptance gates. Follow the repository PR format. Do not publish private
FITS data. Return <=250 words with PR URL, head/base SHAs, worktree, tests, quantified
performance, limitations, and the reviewer handoff. Do not post messages elsewhere.

Your assigned work:
Depends on integrated, reviewed Stage 3 and the Stage 1/2 infrastructure. Implement
and validate diffuse-GP + explicit stars + separately modeled compact nucleus.
This stage's primary question is whether the host absorbs point-source flux.

Sweep GP scales/amplitudes in physical units on templates. Compare GP-only, GP+stars,
and GP+stars+nucleus with held-out exposures. Do not use a small global kernel to
force the GP to fit the nuclear peak. Choose template-derived hyperparameter priors
or an explicit marginal-likelihood procedure with forward operator/noise effects
accounted for. Carry hyperparameter/model uncertainty or quantify the omitted term.

Build independent injections spanning source brightness, separation from nucleus,
host gradients, subpixel phase, masks and template count. Quantify recovered-flux
slope, transfer into GP/nucleus, component covariance/near-null modes, and sensitivity
to kernel priors. Include negative-control cases in which templates are absent or
contaminated and precise separation is impossible. Non-identifiability must surface
as a diagnostic/uncertainty/failure, not a seemingly precise point estimate.

Retain uncertainty in the interpolated host beneath point sources. If considering
masked source cores, GP projections/orthogonality or local kernels, treat each as
an explicit model change with bias/coverage tests; none is an automatic cure. Explain
how nonnegative physical host brightness relates to any signed GP perturbation.

Keep science-frame transient recovery out of hyperparameter/model selection. Define
the static-posterior/conditional-likelihood interface for Stage 6 and document how
template likelihood is counted once. Open a PR with leakage/scale/correlation plots,
numerical evidence and paired benchmarks. Do not claim negligible leakage without
confidence intervals and an agreed bias threshold.
```
