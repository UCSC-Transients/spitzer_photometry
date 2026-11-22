# Implementation chats and independent PR review

Read `STAGED_SCIENCE_MODEL.md` for the updated physical model. This file is the
launch index; each linked prompt is self-contained and may be pasted into a new
chat. No implementation/review chats or worktrees were created by writing these
instructions. These are work orders, not completed implementation claims.

## Model selection

These recommendations use model IDs advertised by this Codex host, not a claim
about the newest model on all accounts. Select the model/effort in the chat UI
before pasting; prose cannot guarantee that the root model changes. If a requested
model is unavailable, report that and ask for an available substitute; do not
silently downgrade scientific inference or independent review.

- Stage 0, baseline packaging: GPT-5.6 Sol (`gpt-5.6-sol`), high.
- Stage 1, fixtures and performance CI: GPT-5.6 Sol, high.
- Stage 2, PRF and astrometry: GPT-6 Astra (`gpt-6-astra`), high.
- Stage 3, nucleus model discrimination: GPT-6 Astra, xhigh.
- Stage 4, GP/source identifiability: GPT-6 Astra, xhigh.
- Stage 5, scalable constrained inference: GPT-6 Astra, xhigh.
- Stage 6, transient recovery and coverage: GPT-6 Astra, xhigh.
- Stage 7, optional cross-band inference: GPT-6 Astra, high; use xhigh for a
  concrete unresolved covariance/identifiability issue, not as a blanket default.
- Independent reviewer for every PR: GPT-6 Astra, xhigh. Independence means a
  separate chat, checkout and assessment; using the same model family is fine.

These are workload-based judgments. Official [reasoning guidance](https://developers.openai.com/api/docs/guides/reasoning)
explains the speed/quality tradeoff and that supported efforts depend on the model.
There is no project-specific evidence that max/ultra effort is necessary.

## Launch order and dependencies

1. [Stage 0](PROMPT_00_BASELINE.md): publish/review the existing repair baseline and
   these work orders. Base its PR on GitHub main, not on itself. Resolve reviewer
   blockers before downstream implementation.
2. [Stage 1](PROMPT_01_VALIDATION_CI.md) and [Stage 2](PROMPT_02_PRF_ASTROMETRY.md)
   may run in parallel from the approved Stage 0 head. Stage 1 owns CI/benchmark
   infrastructure; Stage 2 owns calibration/astrometry. Both add scoped tests.
   Integrate their reviewed PRs and rerun checks before Stage 3.
3. [Stage 3](PROMPT_03_NUCLEUS.md), then [Stage 4](PROMPT_04_GP_IDENTIFIABILITY.md),
   then [Stage 5](PROMPT_05_SCALABLE_SOLVER.md), then
   [Stage 6](PROMPT_06_TRANSIENT.md). Each starts from an integrated, reviewed
   predecessor. Stage 5 design/profiling can happen earlier in a separate checkout,
   but do not implement against a moving Stage 4 interface or certify stale tests.
4. [Stage 7](PROMPT_07_CROSS_BAND.md) starts only after the single-band acceptance
   gates pass. If scientific bias tolerances are still unspecified, provide a
   disabled experimental path and mark the promotion gate unmet.
5. For **every** PR, send [the reviewer prompt](PROMPT_REVIEW_PR.md) to a separate
   review chat. Copy its correction prompt back to the original author. After a
   fix, give the reviewer the new head SHA. Only the user merges unless separately
   authorized. Rebase descendants onto the actual integrated result, not an old
   pre-review branch. Re-review after conflict resolution changes behavior.

For each launch supply `BASE_REF` (a verified local commit/ref containing all
accepted predecessors), `PR_BASE` (the GitHub branch to compare against), and a
unique `RUN_ID`. Never use an unreviewed moving branch as an implicit dependency.
If a predecessor is not merged, explicitly stacked PRs are possible: target the
predecessor branch and record its exact SHA. Retarget/rebase after integration.

## Worktree isolation

Use one worktree and branch per implementation chat, and a different detached
worktree per reviewer/PR revision. Suggested paths under this repository's writable
root: `.worktrees/author-STAGE-RUN_ID` and `.worktrees/review-PR-SHA-RUN_ID`.
Add `.worktrees/` to the local Git exclude file if needed; do not sweep it into a
commit. Creating Git metadata may require sandbox permission. Never reuse or
force-remove an existing worktree, reset somebody else's branch, or operate on the
original dirty checkout. Verify `git worktree list`, absolute cwd, branch and HEAD
before editing or running Git mutations. Different worktrees prevent filesystem
collisions; they do not eliminate Git merge conflicts.

Every writer subagent also needs its own worktree/branch at the parent checkpoint.
A subagent that cannot change cwd reliably must remain read-only and provide a
patch proposal. Parent cherry-picks reviewed child commits serially, runs combined
checks, and resolves conflicts; children never commit into the parent's checkout.
All outputs, caches, JUnit and config writes are worktree-specific. Shared input
FITS/calibration may be read-only; outputs/source catalogs must never be shared.
Use separate environments or a documented read-only environment with dependencies
checked. The original `.local_env` has unrelated bqplot dependency conflicts;
`.venv` previously crashed. Neither is a certified clean CI environment.

## Concise coordination and subagents

Subagents are optional, for independent bounded work only. Explicitly choose model
and effort when spawning; use a minimal task packet if the runtime requires it to
honor overrides rather than silently inheriting settings. Include the scientific
contract, assigned worktree/cwd, base SHA, owned files, checks, and output limits.

- Physics, PRF/GP mathematics, solver equivalence, uncertainty coverage:
  `gpt-6-astra`, high; xhigh for a demonstrated difficult inference problem.
- Test/benchmark implementation and CI plumbing: `gpt-5.6-sol`, high.
- Bounded inventory, documentation and result aggregation: `gpt-5.6-sol`, medium.
- Read-only physics/correctness reviewer: `gpt-6-astra`, xhigh.
- Read-only performance reviewer: `gpt-6-astra`, high.

Every subagent reports at most six bullets: outcome, files/commit, checks with
measured results, numerical/performance changes, blocker, and handoff. No raw test
logs or repeated narrative. Save detailed data in artifacts. Authors' final replies
are at most 250 words plus PR/artifact links. Review findings must all be reported;
keep each finding concise rather than hiding findings to meet a word limit.

The independent review chat is not the author's own review subagent. It independently
checks the exact PR head, tests and evidence and returns the next author prompt.
