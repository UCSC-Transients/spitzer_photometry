# Stage 1 validation and benchmark interface

This interface is the shared entry point for later staged science work.  Stage 2
and subsequent stages should add fixture consumers, metrics, and tests through
these modules instead of replacing the CI or paired benchmark workflow.

## Versioned contracts

- `src.validation_fixtures` (`stage1-fixture-v1`) generates small detector-native
  fixtures for an isolated star, blended stars, shifted/resolved nucleus,
  smooth plus structured host, positive or zero transient, cosmic rays, missing
  pixels, rolls, pixel phases, templates, and exact one-day epoch edges.
- `src.validation_metrics` (`stage1-metrics-v1`) records component flux and
  leakage, centroid/profile bias, leverage-aware training residuals, predictive
  held-out residuals, lag correlations, constrained zero-flux coverage, solver
  residuals, arrays, checksums, and labeled plots.
- `src.stage1_benchmark` (`stage1-benchmark-v1`) defines the paired engineering
  budgets and comparison states: passed, confirmation required, failed, or
  inconclusive because of dispersion.

Fixture truth is independent of the fitted renderer: point responses use analytic
native-pixel integrals and extended components use fixed quadrature.  It is
explicitly labeled synthetic and is not an IRAC calibration substitute.  Any
same-renderer algebra test added later must carry a separate label.

Required PR tests use development seeds `1729`, `2718`, and `31415`.  Seeds
`104729` and `130363`, together with exposure IDs beginning
`acceptance-roll-031` and `acceptance-roll-109`, are reserved for untouched later
acceptance work.  Do not add them to routine PR CI.

## Required CI

The fast Python 3.11/3.12 jobs run the small numerical tests and write fixture,
metric, JSON, NPZ, checksum, JUnit, and PNG artifacts.  The pull-request
performance job checks out the exact GitHub base and head into separate paths,
installs one environment, forces one BLAS/OpenMP thread, and runs alternating
isolated subprocesses.  Cold and warm operator-cache cases are separate; each
records matrix dimensions, operator calls, calibration function/file reads,
stable numerical outputs, wall time, process peak RSS, traced allocations, and
estimated allocation scaling.

The initial mechanical budgets are exactly those in `STAGED_SCIENCE_MODEL.md`:
more than 25% **and** 0.25 seconds wall-time growth, or more than 20% **and**
20 MiB peak-memory growth.  A suspected regression must repeat in a second paired
run.  Relative median absolute deviation above 10% is inconclusive, not a pass.
Numerical equivalence is checked before any performance claim.  No
hardware-absolute timing limit is enforced.

Local paired run:

```bash
python scripts/run_stage1_benchmarks.py \
  --base-repo /path/to/exact/base \
  --head-repo /path/to/exact/head \
  --output-dir test-results/paired-stage1 \
  --repeats 5 --timeout 45
```

The worker intentionally uses the stable Stage 0 frame-operator API, so a base
checkout does not need the newly introduced Stage 1 modules.  All runtime caches
are per-subprocess temporary directories and are cleaned after each bounded run.

## Extended and real-data status

Manual dispatch runs the larger non-real-calibration suite.  It also invokes
`scripts/check_stage1_real_data.py`, which records `missing` when no private
manifest is available and verifies SHA-256 checksums when one is supplied through
`SPITZER_REAL_DATA_MANIFEST`.  The workflow never uses `pull_request_target`,
does not expose secrets to fork code, and never uploads FITS content.

Real calibrated-PRF normalization, real BCD held-out behavior, larger coverage,
and a declared absolute science-bias tolerance remain unmet promotion gates.
Training residual width is not required to equal one, and zero-flux trials are
reported as constrained intervals/upper limits rather than symmetric pulls.
