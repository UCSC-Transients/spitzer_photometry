# Stage 06-transient: copy this prompt into a new implementation chat

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

Create your own branch codex/06-transient-RUN_ID and worktree
/Users/rfoley/spitzer_photometry/.worktrees/author-06-transient-RUN_ID from BASE_REF.
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
Depends on integrated, reviewed Stages 4 and 5. Implement the final single-band
transient inference and its acceptance experiment with the approved static model.

Propagate the template-conditioned host/nucleus/star/registration uncertainty into
nonnegative transient flux and a shared sky position. A sequential posterior must
not double-count templates; verify equivalence to a small joint calculation.
Share flux only within epochs of total span <=1 day, flag >1 and <2-day candidate
spans and split longer spans. Test boundary dates, gaps and chained exposures.
Repeated per-BCD entries of one epoch flux are not independent measurements.

Run blinded independent injections over flux/SNR, zero flux, host complexity,
nucleus separation, phase/roll, masking, template quantity/contamination and modest
model mismatch. Use multiple independent seeds; repeated seeds across SR are paired
comparisons, not independent coverage trials. Freeze model selection before the
final held-out test set. Assess attenuation, bias, scatter, intervals, upper limits,
failure rate and host-to-transient leakage. Include template-poor non-identifiable
cases and explicitly report them.

Select and validate uncertainty reporting near the nonnegative boundary. Do not
clip symmetric Gaussian errors and claim coverage. State frequentist or posterior
interpretation and propagate/quantify GP/nuclear model uncertainty. Validate residual
distributions with held-out predictive covariance and simulations, not unit-variance
training-residual assumptions. Never tune noise weights just to reduce chi-square.

Deliver full reproducibility manifests, light-curve/upper-limit products, injection
and residual diagnostic plots/arrays, required small CI regressions and extended
coverage jobs. Benchmark end-to-end cost. If the user has not supplied acceptable
bias/coverage tolerances, request them, retain measured results and leave promotion
unapproved rather than inventing a scientific pass. Open the PR.
```
