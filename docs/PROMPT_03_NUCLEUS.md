# Stage 03-nucleus: copy this prompt into a new implementation chat

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

Create your own branch codex/03-nucleus-RUN_ID and worktree
/Users/rfoley/spitzer_photometry/.worktrees/author-03-nucleus-RUN_ID from BASE_REF.
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
Depends on integrated, reviewed Stages 1 and 2. Diagnose nucleus residuals using
transient-free templates, beginning with one BCD and progressing through dithers
and rolls. Do not tune against the unknown transient flux.

Compare nested models: point nucleus with floated common sky centroid; a compact
resolved elliptical Gaussian; point core plus resolved envelope. Include explicit
neighboring stars and a controlled smooth host. Separate relative frame astrometry
from nuclear centroid/morphology. A point nucleus should use the calibrated native
point renderer; extended profiles must be correctly forward-modeled and have a
verified point-source limit. Avoid double integration of the detector PRF.

Inject centroid errors, finite extent, PRF mismatch and true asymmetric structure
as different truth cases. Determine which signed dipole/derivative/phase/roll
signatures distinguish them and where they cannot be distinguished. Fit templates
with training/validation/held-out splits. Compare predictive likelihood/residuals
and complexity, not just lower training chi-square. Propagate centroid/profile
uncertainty and constrain physical brightness/amplitudes to be nonnegative.

Use actual BCD uncertainty definitions; no sqrt(Jy) Poisson proxy and no error
inflation to hide dipoles. Quantify structured residual amplitude and its possible
bias on a nearby injected source. If the available templates cannot distinguish
point from resolved emission, retain model uncertainty and report this limitation.

Deliver model-comparison metrics, nucleus residual/centroid/profile/phase/roll plots,
small physics regressions, real-template diagnostics where available, and benchmark
results. Open a PR with the simplest supported default and alternatives explicit.
Do not introduce a flexible local GP that can absorb point-source flux at this stage.
```
