# Stage 05-scalable-solver: copy this prompt into a new implementation chat

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

Create your own branch codex/05-scalable-solver-RUN_ID and worktree
/Users/rfoley/spitzer_photometry/.worktrees/author-05-scalable-solver-RUN_ID from BASE_REF.
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
Depends on integrated, reviewed Stage 4. Replace the large-scene dense limitation
with a scalable implementation of the accepted objective, preserving the small
exact dense implementation as an independent numerical oracle.

Profile setup, GP precision/covariance action, operator assembly, bounded solve,
position iterations, prediction and nuisance covariance. Choose matrix-free/block/
sparse methods based on measured bottlenecks. Preserve exact likelihood cross terms,
calibrated forward/adjoint operators, bounds and the chosen GP model. If an approximate
GP is unavoidable, expose it as a distinct approximation with convergence/bias gates;
never silently switch to a diagonal host prior or approximate likelihood Hessian.

Use preconditioning for acceleration rather than changing the objective. Require
convergence/KKT and adjoint checks with explicit failures. Handle gauges/rank deficiency,
active constraints, invalid pixels and position linearization. Compute needed flux
uncertainty with selected solves or equivalent methods without a full dense inverse;
validate nuisance marginalization. Restore no old clip-an-unconstrained-fit shortcut.

Compare dense/scalable objective, parameters, predictions, derivatives and uncertainty
on well-conditioned and adversarial fixtures. Sweep image size, SR, BCD count and
anchor count; include multiple GP components only if supported by the approved model.
Record peak RSS, allocation shapes, solver/operator counts and cold/warm runtimes.
Demonstrate the previously memory-guarded use case without exceeding the declared
memory budget and without degrading correctness. Open a PR with equivalence tests,
scaling plots, CI budgets and explicit unsupported regimes.
```
