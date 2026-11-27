# Stage 5 predeclared performance acceptance run: 20261127-01

This protocol is committed before the measurements. It authorizes one execution
of the existing PR performance job following this scoped documentation/evidence
push. Its results belong to the exact pushed SHA, recorded by GitHub Actions.
There is no solver, fixture, dependency, benchmark, or workflow change.

## Fixed comparison and execution

- PR: https://github.com/UCSC-Transients/spitzer_photometry/pull/27.
- Author branch: `codex/05-scalable-solver-20261126-05`.
- Author worktree: `.worktrees/author-05-scalable-solver-20261126-05`.
- Reviewed numerical head: `853ea296f3660eeeded8ece9ef6585157873e73c`.
- Base: integrated Stage 4 `05d6a77931c06851f4cc1e04cca255e7e75f764a`, PR target `main`.
- Measurement head: the commit introducing this protocol; verify its numerical
  source, tests, requirements, scripts and workflow match the reviewed head.
- Runner: one fresh GitHub-hosted `ubuntu-latest` performance-job VM. Both refs
  use that same fixed runner and Python 3.11 environment; other CI jobs use separate
  runners. Save run/job IDs, actual runner metadata and dependency provenance.
- No concurrent author benchmarks or tests on that runner. No local timings are
  substituted: local preflight load averages were 18.48/21.57/14.97 on a machine
  previously inventoried as a 10-core M1 Max, with unrelated desktop activity.
  This is a failed isolation precondition, not a benchmark failure or sample.
- The unchanged Stage 1 collector alternates base/head in separate subprocesses,
  with unique runtime/cache directories and one numerical thread. Numerical kernels
  are warmed as specified by the existing workers. All inputs, seeds, masks, physical
  noise, priors and proposal budgets remain identical across each pair.

## Workloads, gates and stopping rule

1. Existing required Stage 1 benchmark: five cold/warm pairs, 45-second worker
   timeout, existing full-array and deterministic allocation checks.
2. If Stage 1 passes, the unchanged required Stage 5 consumer runs `small native9`:
   five cold/warm pairs per fixture, 1,024 proposals, minimum 512 accepted,
   180-second worker timeout and hard 1 GiB process RSS cap for every initial and
   confirmation record. Warm means a same-process rebuild, not a retained fit.
3. Existing numerical gates precede interpretation of performance, including
   Gaussian covariance error <=`1e-8` in marginal-SD product units on comparable
   benchmark cases, six combined Monte Carlo SE for positive moments, and overlap
   of positive-evidence intervals. These ordinary fixtures do not close the P3
   near-null limitation.
4. Preserve the >25% **and** >0.25-second timing budget, >20% **and** >20-MiB memory
   budget, and >10% relative-MAD inconclusive rule. The existing code may collect
   one confirming paired set for a suspected regression. Dispersion alone never
   authorizes another run. Completion, numerical and RSS checks apply to all records.
5. Preserve fail-fast CI behavior: a failed/inconclusive Stage 1 step leaves Stage 5
   unexecuted and acceptance unmet. Save the always-uploaded benchmark artifact,
   logs where accessible, and exit/job status. Do not bypass a failed gate.
6. Do not dispatch or retry this job, select a passing subset, adjust repeat counts,
   modify science/performance thresholds, or rerun until green. If dispersion
   persists, report runner/measurement variability as an unresolved gate. The
   Stage 6 dependency remains unavailable until required acceptance and reviewed
   integration; no merge is authorized.

All earlier original/correction/reviewer measurements remain evidence. In
particular, CI runs `45755742007` and `45758581214` failed performance, and the
reviewer's small/native9 run retained three inconclusive timing cases. Existing
nine-case scaling evidence is retained; this run covers the required CI workloads,
not another full scaling sweep. No speedup or scientific promotion follows merely
from passing an engineering budget. Record the outcome in the PR body so a second
documentation push does not inadvertently become another performance attempt.
