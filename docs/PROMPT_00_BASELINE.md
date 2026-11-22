# Stage 00-baseline: copy this prompt into a new implementation chat

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

Create your own branch codex/00-baseline-RUN_ID and worktree
/Users/rfoley/spitzer_photometry/.worktrees/author-00-baseline-RUN_ID from BASE_REF.
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
Package the existing repairs and approved staged design; do not add new scientific
implementation. BASE_REF is 531612b2ccb3f25ed44db2ee1478ed2461715050. Verify GitHub
main live and use it as PR_BASE; inspect remote ownership because the configured
origin and UCSC URLs historically had the same refs. Do not assume that remains true.

Create the isolated worktree at that baseline. Copy only the new STAGED_SCIENCE_MODEL,
AGENT_WORK_ORDERS and PROMPT_*.md documents from the source checkout into this worktree
if absent; inspect their contents. Include the existing review report and repaired
code already in the baseline. Exclude CLAUDE.md, unrelated dirty documentation,
bin/, clean_repo_run/ and unrelated untracked scripts. Inventory the PR diff against
the real base and distinguish earlier local weighting/extraction work retained in
531612b from new review fixes.

Check the baseline tests in an isolated environment, demonstrate the positive/zero
injection scripts, and record the large-scene memory guard and calibration/coverage
limitations. The prior logs said 108 full tests and 101 final fast tests passed,
but counts are historical evidence, not a substitute for your verification.

Open a draft baseline PR when unresolved scientific gates remain. Do not claim the
full pipeline is production-ready. The independent reviewer should inspect this PR
before new science implementation branches start. Return the exact approved
candidate SHA and integration instructions for Stage 1/2.
```
