# Stage 5: exact finite-grid positive GP inference

This is an opt-in experimental implementation of the accepted Stage 4 posterior.
It does not change production defaults or confer scientific promotion. The dense
implementation in `src/gp_identifiability.py` is unchanged and remains an
independent numerical oracle. Agreement with that reference does not establish
that its physical model is adequate for real BCDs.

## Dependency, ownership and isolation

Accepted Stage 4 reference: `321c8541fc1abec7618bb474c66613d8d487a593`, after
independent re-review resolved RV26-01. PR #26 merged as
`05d6a77931c06851f4cc1e04cca255e7e75f764a`; the two trees are identical. The requested
stack base `codex/04-gp-identifiability-20261125-03` was deleted on the remote after
merge. Following the launch's post-integration instruction, this author branch was
created at the accepted SHA and reconciled onto the actual merge commit before
implementation; the PR targets `main`.

Run `20261126-05` owns branch `codex/05-scalable-solver-20261126-05` and worktree
`.worktrees/author-05-scalable-solver-20261126-05`. The independent paired base is
`.worktrees/base-05-scalable-solver-20261126-05`, detached at the merge commit.
Original and other author/reviewer checkouts are not modified. The Stage 4 re-review
Python 3.11.15 environment is read-only; `uv pip check` verifies all 45 installed
packages. Caches, plots, draws and run outputs are author-local. No dependencies
were added. No private FITS data is published.

## Objective and exactness

For physical host and source coefficients theta, retain the Gaussian model

    theta ~ N(m, K),  y | theta ~ N(A theta, N),  theta >= 0.

Host covariance is precisely the Stage 4 Matérn 3/2 kernel on the supplied angular
grid. Amplitude is SD in Jy/native pixel; scale is in arcseconds. Source priors
have the declared zero-mean Gaussian SD before positive truncation. No host/source
orthogonalization, diagonal GP substitute, covariance taper, low-rank GP, jitter,
error inflation, cropped-point-response renormalization or changed likelihood
cross term is introduced.

The finite host grid is embedded in a larger circulant covariance. Its FFT spectrum
must be strictly positive; otherwise the embedding grows within a declared memory
budget or fails. Restricting this auxiliary Gaussian field to the original grid
recovers the exact finite Matérn covariance. The finite covariance action zero-pads
the input into that embedding and restricts the result. Neither operation adds
periodic coupling to the requested host covariance. Eigenvalues are never clipped.

Condition in observation space:

    B = K A.T,  S = N + A B
    posterior mean = m + B S^-1 (y - A m)
    posterior covariance action(v) = K v - B S^-1 B.T v.

The Schur factorization uses diagonal scaling for numerical conditioning and checks
the original operator residual, with at most two refinement corrections. Failure
above the declared relative tolerance (default `1e-10`) raises. Covariance exports
use selected actions; there is no full host inverse or host covariance allocation.

An independent prior draw z and measurement-noise draw epsilon give an IID
posterior Gaussian draw

    posterior_mean + z - B S^-1 (A z + epsilon).

Jointly reject negative physical coefficients. Independently estimate the prior
host orthant mass; the source contribution is analytic, `2**(-number_of_sources)`.
Positive evidence is Gaussian evidence times posterior/prior orthant mass. These
are the Stage 4 normalization factors, not a MAP or boundary-Laplace surrogate.
Two 97.5% Wilson mass bounds retain the Stage 4 approximate Bonferroni 95% evidence
interval. Evidence/moments have Monte Carlo error, rather than exact integration.
The ordinary accepted-count gate remains 512. Rare orthants explicitly fail.

Accepted draws are saved in bounded independent NPY batches. Selected positive
moments, model mixtures and predictive covariance retain host/source correlation,
active-bound effects, and between-model variance. Conditional likelihood integrates
over the positive template posterior; template IDs cannot be reused. Held-out
template evidence uses a joint/old Gaussian and positive-mass ratio, counting each
exposure once. Low importance ESS remains an explicit failed diagnostic.

## Instrument, nuisance and optimization contracts

`scalable_operators.frozen_host_operator` snapshots the established sparse WCS
projection and anchor kernels for either existing operator order. Later global
configuration or WCS changes do not change the snapshot. Native point and nuclear
responses use the unchanged Stage 2/3 calibrated renderers. Adjoint checks run
before conditioning. Invalid pixels are excluded before arithmetic.

Measured repeated native FFT calls motivated a bounded exact sparse row cache
built during adjoint assembly. Every nonzero coefficient, including numerical
roundoff, is retained. If the cache's peak would exceed the cap it is omitted;
the original exact operator remains in use. This is an acceleration policy, not a
different scientific model.

Independent background calibration is subtracted once and marginalized with
`N_i = diag(sigma_i**2) + tau_i**2 11.T`. Sherman--Morrison retains the full
rank-one covariance and its determinant. `nuisance_moments` exports joint selected
physical/background covariance, including expected conditional background variance
and covariance of conditional means. Background offsets remain signed.

`GaussianPosterior.bounded_mode` is an optional diagnostic of the constrained
objective, separate from posterior integration. Full GP precision actions use
residual-checked PCG; the restricted circulant inverse is only a preconditioner.
L-BFGS-B uses physical nonnegative bounds and variable scaling. A scaled projected
gradient above the declared KKT tolerance raises, irrespective of optimizer status.
Clipping the unconstrained solution is never returned as a constrained fit.

Geometry remains conditional, exactly as in the accepted Stage 4 reference.
Nonzero registration covariance is rejected; uncertain positions require separately
declared geometry alternatives. Source derivative identity is tested, but this
path does not implement floated-position iterations or marginalize continuous
geometry. It does not introduce transient estimation or epoch grouping. Existing
production nonnegativity, <=1-day sharing and exact template-zero contracts remain
unchanged; Stage 6 owns transient boundary/coverage inference.

## Memory model and unsupported regimes

Host scaling is O(p n + n**2 + batch * embedding_pixels) in memory, where p is
host/source parameters and n is valid native pixels. GP actions/sampling use FFTs;
observation factorization remains dense O(n**3). This is deliberately a scalable
host implementation, not a claim of unbounded BCD-count scaling.

The default numerical-array cap is 512 MiB and draw-payload disk cap is 2 GiB.
The paired scaling demonstration additionally declares a 1 GiB process RSS cap,
including Python/imports. Estimates include retained operators, cross covariance,
factorization workspaces, FFTs, selected solves and sparse-cache construction.
Memory/disk failures are explicit and never change prior, data, masks or tolerance.

FFT workspace has a separate 8 MiB target. Columns are chunked internally, with
at least one field required to fit the overall memory cap. The posterior proposal
batch and total draw count stay fixed; host white-noise ordering and the subsequent
source draws are unchanged. Diagnostics record the actual FFT chunk shape. This
policy addresses measured SR=2 workspace growth and an SR=4 allocation refusal,
without changing the finite covariance or accepted-count gate. The original failed
sweep remains in the artifact bundle.

Unsupported: irregular/nonorthogonal host grids; multiple GP components; exact
scene-PRF mode (it is rejected, never silently changed to anchors); a large
observation covariance exceeding the cap; embeddings whose positive spectrum
cannot be established within budget; rare orthants unresolved at the declared
proposal count; arbitrary large full predictive covariance; continuous geometry
uncertainty; real-BCD scientific acceptance; production full-campaign integration.
Proper priors can regularize a likelihood-null direction without making it
data-identifiable. Large fits report the likelihood rank upper bound explicitly;
they do not claim a complete large null-space diagnosis. Coincident-component
tests require broad anticorrelated positive uncertainty.

## Validation and performance protocol

Required numerical tests cover finite-kernel FFT identity, Gaussian draws, full
noise, Gaussian evidence/means/predictions/derivatives, selected covariance,
positive truncated-normal moments/evidence, conditional likelihood, evidence
mixtures, nuisance elimination, active-bound KKT, explicit failures, calibration
gauge, invalid pixels, frozen configuration, SR=1/2/4, both operator orders and
multiple anchors. Real-table tests are separately labeled and never stand in for
absolute calibration or held-out real BCDs.

`scripts/validate_stage5.py` consumes the Stage 1 artifact writer and the corrected
Stage 4 independent optical sky. Positive/zero source fluxes, source/nucleus
separation, masks, one/three templates and three development noise seeds are kept
at their original physical/noise difficulty. The reserved acceptance set is not
touched. Gaussian arrays use Stage 1 identity gates; covariance additionally must
agree to `1e-8` in marginal-SD product units so small Jy-squared values cannot pass
only because of an absolute tolerance. Positive means/covariances compare within
six combined Monte Carlo SE; evidence intervals must overlap. These are numerical
identity criteria, not post-hoc scientific-bias thresholds.

`scripts/run_stage5_benchmarks.py` uses Stage 1's existing alternating isolated
worker runner, numerical comparisons and engineering-budget implementation. The
original Stage 1 benchmark remains required and unchanged. Cold and warm cases
use identical pixels, seeds, priors, calibration, thread count and proposal budgets
on base/head. Suspected regressions get a confirming paired run. >10% relative
MAD is inconclusive; the 25% plus 0.25 s time and 20% plus 20 MiB memory gates are
unchanged. No speed ratio is computed when the dense reference rejects a case.

The sweep varies native size, actual scene SR, BCD count, anchor count and host
footprint. The large case retains 10 BCDs, 800 valid native pixels, a 128x128 host
grid and two sources. Its 8,192-proposal budget retains the 512-accepted gate;
the earlier 1,024-proposal/74-accepted pilot is insufficient under that gate and
is retained as exploratory evidence. No physical difficulty was reduced.

The existing required Python 3.11/3.12 jobs discover all numerical tests and run
the small science comparison consumer. The existing performance job additionally
runs paired small/native9 posterior cases using the same Stage 1 machinery. Full
scaling and larger development experiments are local/manual evidence.

## Artifacts and promotion

The local run bundle is `.test-artifacts/stage5/` in the author worktree. It retains
launch/source-document hashes, environment freeze, initial failure JUnit, later
JUnit, source profiles, real-calibration identity, positive draw batches, JSON/NPZ
science arrays, convergence histories, paired raw records, checksums and scaling
plots. Benchmark failures are saved alongside completed samples. The PR supplies
measured results and exact tested refs. Numerical arrays support conclusions;
plot appearance is not a scientific acceptance criterion.

Scientific promotion stays unapproved: no absolute bias tolerance was supplied;
three development seeds cannot establish coverage; inherited footprint/model
mismatch remains; continuous geometry/hyperparameter uncertainty is conditional
or represented only by declared alternatives; absolute warm calibration and real
held-out BCD acceptance are unmet. The independent reviewer must inspect exact
posterior/evidence preservation, rare-mass failures, nuisance covariance, memory
accounting, numerical gates, and unsupported regimes in a fresh checkout.
