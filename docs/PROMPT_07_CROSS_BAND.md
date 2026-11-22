# Stage 07-cross-band: copy this prompt into a new implementation chat

Select **gpt-6-astra / high** in the chat settings first.

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

Create your own branch codex/07-cross-band-RUN_ID and worktree
/Users/rfoley/spitzer_photometry/.worktrees/author-07-cross-band-RUN_ID from BASE_REF.
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
Depends on reviewed Stage 6 and explicit single-band acceptance. Implement an
optional multiband extension; keep the validated independent-band path available.

Start with common sky coordinates and registration, while allowing band-specific
PRFs, backgrounds, diffuse morphology, stellar/nuclear amplitudes, and transient
fluxes. Colors need not be constant and nonsimultaneous epochs are not simultaneous
measurements. Then investigate a positive-semidefinite cross-band GP covariance
with independent per-band residual components, only if data constrain it. Use the
zero-correlation limit as a strict equivalence test to independent-band fits.

Inject realistic color gradients, different nuclear/stellar colors, differing PSFs,
missing band/epochs, and color-evolving transients. Demonstrate that sharing improves
held-out predictions without forcing a physically false fixed-color morphology or
transferring transient signal into the other band/host. Assess identifiability and
uncertainty of correlation parameters and calibration nuisances. With insufficient
evidence, deliver a disabled experimental option and retain independent fits.

Document cross-band timing, physical assumptions and promotion gates. Add numerical,
leakage, covariance-PSD and missing-data regressions, diagnostics and paired scaling
benchmarks. Use xhigh effort only for a concrete unresolved inference issue. Open
a scoped PR; do not retroactively call single-band success cross-band validation.
```
