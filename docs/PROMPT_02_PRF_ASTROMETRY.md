# Stage 02-prf-astrometry: copy this prompt into a new implementation chat

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

Create your own branch codex/02-prf-astrometry-RUN_ID and worktree
/Users/rfoley/spitzer_photometry/.worktrees/author-02-prf-astrometry-RUN_ID from BASE_REF.
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
Depends on reviewed Stage 0; may run alongside Stage 1. Work on native isolated
point sources and astrometry before introducing the galaxy GP.

Audit local PRF files and their mission phase, channel, full-detector coordinates,
centroid definition, oversampling and aperture normalization against the instrument
documentation. The current direct sampler normalizes by the PRF array sum: verify
its relationship to the published aperture-calibrated Jy convention. Test native
pixel phase, SR=1/2/4 with fixed sky footprint, rotation/distortion, detector position,
truncated wings, and missing calibration. Preserve phase-dependent response rather
than normalizing every phase/crop to one. Avoid double pixel integration or duplicate
intrapixel corrections. Explicit synthetic mode cannot count as real-PRF validation.

Fit isolated stars jointly for nonnegative flux and centroid; use a reference-frame
gauge for exposure registration. Distinguish sky-source shifts from detector/WCS
errors. Regressions should recover independently injected offsets and flux, with
shared frame offsets constrained by multiple stars. Verify model derivatives and
fit/predict equality. Audit caches under WCS/config/calibration changes and retain
input immutability. Improve calibration IO only with safe content/invalidation keys.

Deliver PRF radial/encircled-energy and phase/position/roll error plots with arrays,
calibration hashes, native/scene units, and paired performance measurements. Provide
a validated response/centroid interface that Stage 3 can reuse. Mark any unverified
absolute flux calibration as a blocking scientific gate. Open the PR.
```
