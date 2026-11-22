# Stage 01-validation-ci: copy this prompt into a new implementation chat

Select **gpt-5.6-sol / high** in the chat settings first.

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

Create your own branch codex/01-validation-ci-RUN_ID and worktree
/Users/rfoley/spitzer_photometry/.worktrees/author-01-validation-ci-RUN_ID from BASE_REF.
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
Depends on the reviewed Stage 0 baseline. Build the validation and efficiency
infrastructure before model complexity increases. Own CI/benchmark infrastructure;
coordinate with Stage 2 so it can add fixtures/tests without rewriting your workflow.

Provide deterministic small fixtures with independently generated truth: isolated
stars; shifted and resolved nucleus; smooth/structured host; blended stars;
positive/zero transient; cosmic rays and missing pixels; multiple rolls/phases and
one-day epoch edges. Keep truth generation independent of the fitted renderer
where practical, and label same-renderer algebra tests separately. Reserve seeds
and exposures for untouched acceptance tests.

Define a versioned metrics/artifact interface: flux/component leakage, centroid and
profile bias, whiteness/correlation, held-out prediction, coverage, solver residuals,
time and peak memory. Handle training residual leverage and zero-flux boundaries
correctly; do not demand unit training-residual variance by construction.

Implement paired base/head CI performance tests with identical runner/environment,
BLAS threads, warmup, repeated subprocess medians and dispersion, separate cold/warm
cache timing, bounded timeouts and cleanup. Record matrix sizes and operation/read
counts. Use the proposed budgets in STAGED_SCIENCE_MODEL.md as initial engineering
budgets and validate their noise behavior. Require confirming reruns for suspected
regressions; report inconclusive results honestly. Retain a stable numerical
benchmark when either side lacks a newly introduced API. Avoid brittle hardware-
absolute timing gates. Establish memory/allocation scaling checks too.

Required PR tests should remain small. Manual extended jobs run larger coverage,
real-PRF checks and optional checksum-verified real BCDs with explicit missing-data
status. Never enable fork PR access to private data/secrets or use pull_request_target
to execute untrusted code. Upload numerical and plotted artifacts on failure.
Demonstrate that a deliberate numerical regression and a controlled performance
regression are detected, then remove the deliberate defects. Open the PR.
```
