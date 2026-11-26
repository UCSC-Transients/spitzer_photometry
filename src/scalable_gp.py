"""Exact finite-grid positive Matérn inference with bounded host-space memory.

The dense Stage 4 reference remains independent. This implementation uses a
positive circulant embedding to apply/sample the *same finite* Matérn 3/2 kernel,
then conditions in observation space. The periodic extension is an auxiliary
sampling construction: no periodic wrap is introduced into the restricted grid.
No eigenvalues are clipped, GP jitter added, likelihood cross terms discarded, or
positive draws replaced by a MAP/Laplace approximation. Rejection integration has
explicit Monte Carlo error and may fail in rare orthants. Large observation count
is still a bounded dense problem; this is a host-scalable, not universal, solver.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
import json
from pathlib import Path
import tempfile
import time
from typing import Iterator

import numpy as np
from scipy.fft import rfftn, irfftn, next_fast_len
from scipy.linalg import cho_factor, cho_solve
from scipy.optimize import minimize
from scipy.sparse.linalg import LinearOperator, cg
from scipy.sparse import csr_matrix
from scipy.special import logsumexp
from scipy.stats import norm, chi2

from .isolated_astrometry import _freeze
from .scalable_operators import OperatorData, Design, BlockNoise, BackgroundNoise


class NumericalFailure(RuntimeError):
    """Explicit convergence, precision or sampling failure with retained details."""
    def __init__(self, message: str, diagnostics: dict):
        super().__init__(message)
        self.diagnostics = diagnostics


@dataclass(frozen=True)
class Budget:
    """Declared allocation/disk caps, separate from measured process peak RSS.

    ``memory_bytes`` bounds estimated live numerical arrays, excluding interpreter
    and imported libraries. ``disk_bytes`` bounds accepted-draw payload. Sampling
    batch size is fixed before inference. FFT columns are additionally chunked to
    bound workspace, without changing draws. Caps never change the model/tolerances.
    """
    memory_bytes: int = 512*2**20
    disk_bytes: int = 2*2**30
    batch_size: int = 16
    fft_work_bytes: int = 8*2**20
    solve_rtol: float = 1e-10

    def __post_init__(self) -> None:
        if (self.memory_bytes <= 0 or self.disk_bytes <= 0 or self.batch_size < 1 or self.fft_work_bytes <= 0
                or not 0 < self.solve_rtol <= 1e-8):
            raise ValueError('Positive budgets/batch and solve tolerance <= 1e-8 required')

    def guard(self, size: int, phase: str) -> None:
        if size > self.memory_bytes:
            raise MemoryError(f'{phase} requires estimated {size} bytes; declared cap {self.memory_bytes}')


class GridPrior:
    """Matérn 3/2 host on a regular orthogonal angular grid plus source Gaussians.

    Source SD is the *untruncated* half-normal scale. Truncation is applied jointly
    by ``fit_positive`` and normalized there. A successful nonnegative circulant
    spectrum gives exact finite-kernel Gaussian draws, up to floating-point error.
    Failure to find such an embedding within the cap raises; no approximate GP.
    """
    def __init__(self, xy_arcsec: np.ndarray, shape: tuple[int, int], *, scale_arcsec: float,
                 amplitude: float, host_mean: np.ndarray, source_count: int,
                 source_prior_sigma: float, budget: Budget = Budget()):
        start = time.perf_counter()
        xy = np.asarray(xy_arcsec, dtype=float)
        h, w = shape
        if (len(shape) != 2 or min(shape) < 1 or h*w != len(xy) or xy.shape != (h*w, 2)
                or not np.isfinite(xy).all() or source_count < 0
                or not np.isfinite([scale_arcsec, amplitude, source_prior_sigma]).all()
                or min(scale_arcsec, amplitude, source_prior_sigma) <= 0):
            raise ValueError('Regular finite angular grid and positive GP/source scales required')
        hm = np.asarray(host_mean, dtype=float)
        if hm.shape != (h*w,) or not np.isfinite(hm).all() or np.any(hm < 0):
            raise ValueError('Declared nonnegative finite host mean required')
        grid = xy.reshape(h, w, 2)
        ex = (grid[0, -1]-grid[0, 0])/(w-1) if w > 1 else np.array([1., 0.])
        ey = (grid[-1, 0]-grid[0, 0])/(h-1) if h > 1 else np.array([0., 1.])
        # For singleton axes only the existing direction matters.
        if w == 1 and h > 1:
            ex = np.array([ey[1], -ey[0]])
        if h == 1 and w > 1:
            ey = np.array([-ex[1], ex[0]])
        yy, xx = np.indices(shape)
        expected = grid[0, 0]+xx[..., None]*ex+yy[..., None]*ey
        dx, dy = np.linalg.norm(ex), np.linalg.norm(ey)
        if (min(dx, dy) <= 0 or abs(ex@ey) > 1e-10*dx*dy
                or not np.allclose(grid, expected, rtol=0, atol=1e-9*max(dx, dy))):
            raise ValueError('Only regular orthogonal grids supported; no interpolation approximation')
        self.shape, self.host_size, self.size = shape, h*w, h*w+source_count
        self.xy, self.mean = _freeze(xy), _freeze(np.r_[hm, np.zeros(source_count)])
        self.scale, self.amplitude, self.source_sd = scale_arcsec, amplitude, source_prior_sigma
        self.budget = budget
        self.counts = dict(covariance=0, precision_iterations=0, prior_samples=0)
        self.embedding_attempts = []
        base = (next_fast_len(max(2, 2*(h-1))), next_fast_len(max(2, 2*(w-1))))
        for attempt in range(8):
            embed = tuple(next_fast_len(v*2**attempt) for v in base)
            # Factor, real/complex FFT workspaces and batched sampling work.
            cells = int(np.prod(embed))
            fft_batch = min(budget.batch_size, max(1, budget.fft_work_bytes//(64*cells)))
            budget.guard(8*cells*(8*fft_batch+8), 'GP embedding')
            iy = np.minimum(np.arange(embed[0]), embed[0]-np.arange(embed[0]))*dy
            ix = np.minimum(np.arange(embed[1]), embed[1]-np.arange(embed[1]))*dx
            r = np.hypot(iy[:, None], ix[None, :])*np.sqrt(3)/scale_arcsec
            kernel = amplitude**2*(1+r)*np.exp(-r)
            spectrum = rfftn(kernel).real
            minimum = float(spectrum.min())
            self.embedding_attempts.append(dict(shape=embed, minimum_eigenvalue=minimum))
            if minimum > 0:
                self.embedding_shape, self.spectrum = embed, spectrum
                self.fft_batch_size = fft_batch
                break
        else:
            raise NumericalFailure('No positive exact circulant embedding found', dict(attempts=self.embedding_attempts))
        self.sqrt_spectrum = np.sqrt(self.spectrum)
        self.storage_bytes = self.spectrum.nbytes+self.sqrt_spectrum.nbytes+self.mean.nbytes+self.xy.nbytes
        self.setup_seconds = time.perf_counter()-start

    def _host_fft(self, values: np.ndarray, spectrum: np.ndarray) -> np.ndarray:
        columns = np.asarray(values).reshape(self.host_size, -1)
        if columns.shape[1] > self.fft_batch_size:
            out = np.empty_like(columns, dtype=float)
            for i in range(0, columns.shape[1], self.fft_batch_size):
                out[:, i:i+self.fft_batch_size] = self._host_fft(columns[:, i:i+self.fft_batch_size], spectrum)
            return out.reshape(values.shape)
        padded = np.zeros((columns.shape[1],)+self.embedding_shape)
        padded[:, :self.shape[0], :self.shape[1]] = columns.T.reshape((-1,)+self.shape)
        transformed = irfftn(rfftn(padded, axes=(-2, -1))*spectrum,
                             s=self.embedding_shape, axes=(-2, -1))
        return transformed[:, :self.shape[0], :self.shape[1]].reshape(-1, self.host_size).T.reshape(values.shape)

    def action(self, values: np.ndarray) -> np.ndarray:
        """Exact finite covariance K times vectors; no wrap within the host grid."""
        x = np.asarray(values, dtype=float)
        if x.ndim not in (1, 2) or x.shape[0] != self.size or not np.isfinite(x).all():
            raise ValueError('Prior input dimensions/values invalid')
        self.budget.guard(5*x.nbytes+8*int(np.prod(self.embedding_shape))*(8*self.fft_batch_size+8)
                          +self.storage_bytes, 'GP covariance action')
        self.counts['covariance'] += 1 if x.ndim == 1 else x.shape[1]
        out = np.empty_like(x)
        out[:self.host_size] = self._host_fft(x[:self.host_size], self.spectrum)
        out[self.host_size:] = self.source_sd**2*x[self.host_size:]
        return out

    def sample(self, rng: np.random.Generator, count: int) -> np.ndarray:
        """Zero-mean untruncated prior draws as columns, limited to one batch."""
        if not 1 <= count <= self.budget.batch_size:
            raise ValueError('Draw count exceeds declared sampling batch')
        out = np.empty((self.size, count))
        for start in range(0, count, self.fft_batch_size):
            stop = min(count, start+self.fft_batch_size)
            white = rng.standard_normal((stop-start,)+self.embedding_shape)
            fields = irfftn(rfftn(white, axes=(-2, -1))*self.sqrt_spectrum,
                            s=self.embedding_shape, axes=(-2, -1))
            out[:self.host_size, start:stop] = fields[:, :self.shape[0], :self.shape[1]].reshape(stop-start, -1).T
        # Sources follow all host white noise, preserving the RNG stream when
        # only the internal FFT workspace chunk changes.
        out[self.host_size:] = self.source_sd*rng.standard_normal((self.size-self.host_size, count))
        self.counts['prior_samples'] += count
        return out

    def precision(self, values: np.ndarray, *, maxiter: int = 4000) -> np.ndarray:
        """Residual-checked PCG precision action; circulant inverse is only a preconditioner."""
        x = np.asarray(values, dtype=float)
        if x.shape != (self.size,) or not np.isfinite(x).all():
            raise ValueError('Precision requires one finite parameter vector')
        h = self.host_size
        operator = LinearOperator((h, h), matvec=lambda v: self._host_fft(v, self.spectrum))
        preconditioner = LinearOperator((h, h), matvec=lambda v: self._host_fft(v, 1/self.spectrum))
        iterations = [0]
        def callback(_: np.ndarray) -> None:
            iterations[0] += 1
        sol, info = cg(operator, x[:h], M=preconditioner, rtol=self.budget.solve_rtol,
                       atol=0, maxiter=maxiter, callback=callback)
        residual = np.linalg.norm(operator@sol-x[:h])/max(np.linalg.norm(x[:h]), np.finfo(float).tiny)
        self.counts['precision_iterations'] += iterations[0]
        if info != 0 or not np.isfinite(residual) or residual > 5*self.budget.solve_rtol:
            raise NumericalFailure('GP precision PCG did not converge', dict(info=info, iterations=iterations[0], residual=residual))
        return np.r_[sol, x[h:]/self.source_sd**2]


def estimate_memory(data: OperatorData, prior: GridPrior) -> dict:
    """Conservative live-array estimate, including retained operators and FFT batches."""
    n, p = data.design.shape
    batch = prior.budget.batch_size
    fft_cells = int(np.prod(prior.embedding_shape))
    components = dict(cross_covariance=8*n*p, cross_workspace=8*n*p, observation_workspaces=8*6*n*n,
                      batch_workspaces=8*(prior.fft_batch_size*8*fft_cells+batch*(8*p+6*n)),
                      vectors=8*(12*p+12*n), prior=prior.storage_bytes,
                      operator=data.design.storage_bytes, noise=data.noise.storage_bytes)
    return dict(peak_bytes=sum(components.values()), components=components,
                shapes=dict(cross=[p, n], observation_factor=[n, n], fft=[prior.fft_batch_size, *prior.embedding_shape]),
                no_host_square_allocation=True)


class GaussianPosterior:
    """Observation-space Gaussian conditioning with selected covariance actions.

    B = K A.T, S = N + A B. Mean m+B S^-1(y-Am); covariance K-B S^-1 B.T.
    Samples use conditional simulation, preserving all host/source cross terms.
    Only the observation covariance is factorized; no parameter inverse is formed.
    """
    def __init__(self, data: OperatorData, prior: GridPrior):
        start = time.perf_counter()
        if prior.size != data.design.shape[1] or not np.array_equal(prior.xy, data.host_xy_arcsec):
            raise ValueError('Prior and data layouts must match')
        self.data, self.prior = data, prior
        self.memory = estimate_memory(data, prior)
        prior.budget.guard(self.memory['peak_bytes'], 'Gaussian conditioning')
        self.adjoint_error = data.design.check_adjoint()
        n, p = data.design.shape
        self.cross = np.empty((p, n))
        unit = np.zeros(n)
        # Adjoint impulses are already required for K A.T. Retain their exact
        # sparse rows when the declared memory cap permits; cache only, no model
        # approximation. Dense explicit fixtures already have fast batched BLAS.
        row_values, row_indices, row_ends = [], [], [0]
        cache_rows = not getattr(data.design, 'explicit', False)
        cache_bytes = 0
        for j in range(n):
            unit[j] = 1
            adjoint = data.design.adjoint(unit)
            if np.any(adjoint < -64*np.finfo(float).eps*max(1., float(np.max(np.abs(adjoint))))):
                raise ValueError('Host/source operator does not preserve physical nonnegative brightness')
            self.cross[:, j] = prior.action(adjoint)
            if cache_rows:
                nz = np.flatnonzero(adjoint)
                cache_bytes += 16*len(nz)
                # Includes list arrays, concatenation, CSR conversion, frozen
                # copy and Python/list overhead, alongside peak FFT workspaces.
                if self.memory['peak_bytes']+5*cache_bytes+1024*n > prior.budget.memory_bytes:
                    cache_rows = False; row_values.clear(); row_indices.clear()
                else:
                    row_values.append(adjoint[nz]); row_indices.append(nz)
                    row_ends.append(row_ends[-1]+len(nz))
            unit[j] = 0
        self.operator_cache_bytes = 0
        self.assembly_operator_counts = dict(data.design.counts)
        if cache_rows:
            matrix = csr_matrix((np.concatenate(row_values), np.concatenate(row_indices), np.array(row_ends)), shape=(n, p))
            cached = Design.from_sparse(matrix, provenance=data.design.provenance+'; exact sparse row cache')
            self.operator_cache_bytes = cached.storage_bytes
            self.memory['peak_bytes'] += 5*cache_bytes+1024*n
            self.memory['components']['exact_operator_cache_peak'] = 5*cache_bytes+1024*n
            self.data = data = replace(data, design=cached)
            del matrix, row_values, row_indices
        self.assembly_seconds = time.perf_counter()-start
        covariance = np.empty((n, n))
        batch = prior.budget.batch_size
        for j in range(0, n, batch):
            k = min(n, j+batch)
            identity = np.zeros((n, k-j)); identity[np.arange(j, k), np.arange(k-j)] = 1
            covariance[:, j:k] = data.design.forward(self.cross[:, j:k])+data.noise.action(identity)
        scale = np.sqrt(np.diag(covariance))
        if not np.isfinite(scale).all() or np.any(scale <= 0):
            raise NumericalFailure('Nonpositive observation covariance diagonal', {})
        covariance /= scale[:, None]; covariance /= scale[None, :]
        symmetry_error = np.max(np.abs(covariance-covariance.T))
        if symmetry_error > 1e-10:
            raise NumericalFailure('Observation covariance is not symmetric', dict(error=float(symmetry_error)))
        covariance = (covariance+covariance.T)/2
        factor_start = time.perf_counter()
        try:
            self.factor = cho_factor(covariance, lower=True, check_finite=True)
        except np.linalg.LinAlgError as exc:
            raise NumericalFailure('Observation covariance factorization failed; no jitter added', {}) from exc
        self.scale = scale
        self.factor_seconds = time.perf_counter()-factor_start
        self.logdet = float(2*(np.log(scale).sum()+np.log(np.diag(self.factor[0])).sum()))
        self.solve_count = 0
        self.maximum_solve_residual = 0.
        self.residual_history: list[float] = []
        residual = data.y-data.design.forward(prior.mean)
        alpha = self.solve(residual)
        self.mean = prior.mean+self.cross@alpha
        self.log_evidence = float(-.5*(residual@alpha+self.logdet+n*np.log(2*np.pi)))
        self.seconds = time.perf_counter()-start

    def solve(self, rhs: np.ndarray) -> np.ndarray:
        """Solve S x=rhs and check the original operator residual, with refinement."""
        self.solve_count += 1
        r = np.asarray(rhs)
        scales = self.scale.reshape((-1,)+(1,)*(r.ndim-1))
        solution = cho_solve(self.factor, r/scales)/scales
        for _ in range(3):
            residual = self.data.noise.action(solution)+self.data.design.forward(self.cross@solution)-r
            relative = float(np.linalg.norm(residual/scales)/max(np.linalg.norm(r/scales), np.finfo(float).tiny))
            if relative <= self.prior.budget.solve_rtol:
                break
            solution -= cho_solve(self.factor, residual/scales)/scales
        self.maximum_solve_residual = max(self.maximum_solve_residual, relative)
        self.residual_history.append(relative)
        if not np.isfinite(relative) or relative > self.prior.budget.solve_rtol:
            raise NumericalFailure('Gaussian solve residual exceeds declared tolerance', dict(relative=relative))
        return solution

    def covariance_action(self, values: np.ndarray) -> np.ndarray:
        """Marginalized covariance action; sources retain host and nuisance uncertainty."""
        return self.prior.action(values)-self.cross@self.solve(self.cross.T@values)

    def selected_covariance(self, indices: np.ndarray) -> np.ndarray:
        """Selected physical-parameter covariance without a full dense inverse."""
        idx = _indices(indices, self.prior.size)
        self.prior.budget.guard(self.memory['peak_bytes']+24*self.prior.size*len(idx), 'Selected covariance')
        out = np.empty((len(idx), len(idx)))
        for j in range(0, len(idx), self.prior.budget.batch_size):
            chosen = idx[j:j+self.prior.budget.batch_size]
            unit = np.zeros((self.prior.size, len(chosen))); unit[chosen, np.arange(len(chosen))] = 1
            out[:, j:j+len(chosen)] = self.covariance_action(unit)[idx]
        return (out+out.T)/2

    def sample(self, rng: np.random.Generator, count: int) -> np.ndarray:
        """IID untruncated posterior draws via exact conditional simulation."""
        z = self.prior.sample(rng, count)
        measurement = self.data.design.forward(z)+self.data.noise.sample(rng, count)
        return self.mean[:, None]+z-self.cross@self.solve(measurement)

    def objective_gradient(self, parameters: np.ndarray) -> tuple[float, np.ndarray]:
        """Negative log posterior up to a constant, with exact prior precision."""
        delta = parameters-self.prior.mean
        precision = self.prior.precision(delta)
        residual = self.data.design.forward(parameters)-self.data.y
        weighted = self.data.noise.solve(residual)
        return float(.5*(delta@precision+residual@weighted)), precision+self.data.design.adjoint(weighted)

    def bounded_mode(self, *, maxiter: int = 2000, kkt_tolerance: float = 1e-6) -> dict:
        """Optional nonnegative mode diagnostic; never used as posterior uncertainty.

        Diagonal variable scaling is preconditioning only. Every objective and
        derivative uses the full prior/likelihood. A failed KKT check raises even
        if the optimizer claims success; no clipped unconstrained solution returns.
        """
        if maxiter < 1 or not 0 < kkt_tolerance <= 1e-5:
            raise ValueError('Positive iteration budget and KKT tolerance <=1e-5 required')
        scale = np.r_[np.full(self.prior.host_size, self.prior.amplitude),
                      np.full(self.prior.size-self.prior.host_size, self.prior.source_sd)]
        history = []
        def objective(u: np.ndarray) -> tuple[float, np.ndarray]:
            value, gradient = self.objective_gradient(u*scale)
            history.append(value)
            return value, gradient*scale
        result = minimize(objective, np.maximum(self.mean, 0)/scale, jac=True, method='L-BFGS-B',
                          bounds=[(0., None)]*self.prior.size,
                          options=dict(maxiter=maxiter, maxls=50, ftol=1e-15, gtol=kkt_tolerance*.1, maxcor=30))
        value, gradient = objective(result.x)
        projected = np.where((result.x <= 1e-10) & (gradient > 0), 0., gradient)
        kkt = float(np.max(np.abs(projected)))
        report = dict(parameters=result.x*scale, objective=value, projected_gradient=kkt,
                      iterations=int(result.nit), optimizer_success=bool(result.success),
                      active_indices=np.flatnonzero(result.x <= 1e-10).tolist(), objective_history=history)
        if not np.isfinite(kkt) or kkt > kkt_tolerance:
            raise NumericalFailure('Bounded mode failed KKT convergence', report)
        return report


def _indices(indices: np.ndarray, size: int) -> np.ndarray:
    idx = np.asarray(indices)
    if idx.ndim != 1 or not len(idx) or not np.issubdtype(idx.dtype, np.integer) or np.any(idx < 0) or np.any(idx >= size):
        raise ValueError('Nonempty valid integer component indices required')
    return idx


def _mass_interval(k: int, n: int) -> tuple[float, float]:
    """97.5% Wilson bounds; two mass bounds form an approximate Bonferroni 95%."""
    z = norm.ppf(.9875); p = k/n
    center = (p+z*z/(2*n))/(1+z*z/n)
    radius = z*np.sqrt(p*(1-p)/n+z*z/(4*n*n))/(1+z*z/n)
    return float(max(0., center-radius)), float(min(1., center+radius))


@dataclass
class PositivePosterior:
    """On-disk IID positive draws; arbitrary selected moments retain covariance.

    No full host covariance is materialized. ``moments`` and ``predict`` compute
    requested covariance from the same positive draws, including active bounds.
    Gaussian curvature is accessible separately and is never reported as a
    positive-posterior interval. Files are retained in the caller's run directory.
    """
    gaussian: GaussianPosterior
    directory: Path
    chunks: tuple[Path, ...]
    mean: np.ndarray
    variance: np.ndarray
    log_evidence: float
    diagnostics: dict

    @property
    def data(self) -> OperatorData:
        return self.gaussian.data

    def batches(self) -> Iterator[np.ndarray]:
        """Yield accepted draws as parameter columns, at most one bounded batch."""
        for path in self.chunks:
            values = np.load(path, allow_pickle=False)
            if values.ndim != 2 or values.shape[0] != self.gaussian.prior.size or not np.isfinite(values).all() or np.any(values < 0):
                raise NumericalFailure('Invalid stored positive draws', dict(path=str(path)))
            yield values

    def moments(self, indices: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Selected posterior moments with all unselected components marginalized."""
        idx = _indices(indices, len(self.mean))
        self.gaussian.prior.budget.guard(self.gaussian.memory['peak_bytes']+16*len(idx)**2, 'Positive selected moments')
        covariance = np.zeros((len(idx), len(idx)))
        for values in self.batches():
            delta = values[idx]-self.mean[idx, None]
            covariance += delta@delta.T
        return self.mean[idx].copy(), covariance/(self.diagnostics['accepted']-1)

    def check_prediction(self, data: OperatorData) -> None:
        if set(data.exposure_ids) & set(self.data.exposure_ids):
            raise ValueError('Template likelihood would be counted twice')
        if data.source_names != self.data.source_names or not np.array_equal(data.host_xy_arcsec, self.data.host_xy_arcsec):
            raise ValueError('Prediction must preserve component layout and angular host grid')
        if data.calibration_gauge != self.data.calibration_gauge:
            raise ValueError('Prediction calibration gauge differs from training templates')

    def predict(self, data: OperatorData, *, full_covariance: bool = False) -> tuple[np.ndarray, np.ndarray]:
        """Mean and measurement-plus-positive-static covariance (or its diagonal)."""
        self.check_prediction(data)
        n = len(data.y)
        extra = 24*n*n if full_covariance else 24*n
        self.gaussian.prior.budget.guard(self.gaussian.memory['peak_bytes']+extra+data.design.storage_bytes
                                         +data.noise.storage_bytes+32*n*self.gaussian.prior.budget.batch_size,
                                         'Positive prediction')
        mean = data.design.forward(self.mean)
        covariance = np.zeros((n, n) if full_covariance else n)
        for values in self.batches():
            delta = data.design.forward(values)-mean[:, None]
            covariance += delta@delta.T if full_covariance else np.sum(delta*delta, axis=1)
        covariance /= self.diagnostics['accepted']-1
        # Request only needed noise columns; diagonal-only prediction stays O(n).
        for j in range(n):
            unit = np.zeros(n); unit[j] = 1
            column = data.noise.action(unit)
            if full_covariance:
                covariance[:, j] += column
            else:
                covariance[j] += column[j]
        return mean, covariance

    def log_predictive(self, data: OperatorData, offset: np.ndarray | None = None) -> tuple[float, float]:
        """Integrate the new likelihood over positive draws, returning score and ESS."""
        self.check_prediction(data)
        self.gaussian.prior.budget.guard(self.gaussian.memory['peak_bytes']+data.design.storage_bytes
                                         +data.noise.storage_bytes+32*len(data.y)*self.gaussian.prior.budget.batch_size,
                                         'Conditional likelihood')
        shift = np.zeros_like(data.y) if offset is None else np.asarray(offset)
        if shift.shape != data.y.shape or not np.isfinite(shift).all():
            raise ValueError('Finite native prediction offset required')
        log_total = log_squares = -np.inf
        for values in self.batches():
            residual = (data.y-shift)[:, None]-data.design.forward(values)
            score = -.5*(np.sum(residual*data.noise.solve(residual), axis=0)
                          +data.noise.logdet+len(data.y)*np.log(2*np.pi))
            log_total = np.logaddexp(log_total, logsumexp(score))
            log_squares = np.logaddexp(log_squares, logsumexp(2*score))
        return float(log_total-np.log(self.diagnostics['accepted'])), float(np.exp(2*log_total-log_squares))

    def predictive_evidence(self, data: OperatorData, *, seed: int, draws: int = 16384) -> dict:
        """New-template evidence by joint/old Gaussian and positive-mass ratios.

        Independent new exposures appear exactly once in the joint problem. The
        common prior orthant normalization cancels algebraically, including the
        source half-normal factors. This does not update template selection weights.
        """
        self.check_prediction(data)
        if data.partition not in ('validation', 'heldout') or data.verified is not True:
            raise ValueError('Predictive model comparison uses verified template partitions only')
        if not isinstance(draws, int) or not 512 <= draws <= 262144:
            raise ValueError('Invalid bounded predictive sampling budget')
        old = self.data
        def forward(x: np.ndarray) -> np.ndarray:
            return np.concatenate((old.design.forward(x), data.design.forward(x)))
        def adjoint(x: np.ndarray) -> np.ndarray:
            return old.design.adjoint(x[:len(old.y)])+data.design.adjoint(x[len(old.y):])
        design = Design((len(old.y)+len(data.y), old.design.shape[1]), forward, adjoint,
                        provenance=old.design.provenance+'; '+data.design.provenance,
                        storage_bytes=old.design.storage_bytes+data.design.storage_bytes)
        joint = OperatorData(np.r_[old.y, data.y], BlockNoise((old.noise, data.noise)), design,
                             old.host_xy_arcsec, old.source_names, old.exposure_ids+data.exposure_ids,
                             'train', True, 'joint independent templates', old.calibration_gauge)
        self.gaussian.prior.budget.guard(estimate_memory(joint, self.gaussian.prior)['peak_bytes']
                                         +self.gaussian.cross.nbytes+self.gaussian.factor[0].nbytes,
                                         'Joint held-out conditioning')
        conditioned = GaussianPosterior(joint, self.gaussian.prior)
        rng = np.random.default_rng(seed)
        accepted = 0
        for begin in range(0, draws, self.gaussian.prior.budget.batch_size):
            samples = conditioned.sample(rng, min(draws-begin, self.gaussian.prior.budget.batch_size))
            accepted += int(np.all(samples >= 0, axis=0).sum())
        if accepted < 32:
            raise NumericalFailure('Predictive orthant mass unresolved', dict(accepted=accepted, proposals=draws))
        bounds = _mass_interval(accepted, draws)
        old_bounds = self.diagnostics['posterior_mass_95']
        gaussian_score = conditioned.log_evidence-self.gaussian.log_evidence
        return dict(log_score=float(gaussian_score+np.log(accepted/draws/(self.diagnostics['accepted']/self.diagnostics['proposals']))),
                    log_score_95=[float(gaussian_score+np.log(bounds[0]/old_bounds[1])),
                                  float(gaussian_score+np.log(bounds[1]/old_bounds[0]))],
                    accepted=accepted, proposals=draws)

    def nuisance_moments(self, indices: np.ndarray) -> dict:
        """Selected source/host plus signed calibrated-background joint covariance.

        Rao--Blackwellization adds E[Var(background|theta,y)] as well as covariance
        of conditional means. No fitted-pixel background prior is counted twice.
        """
        noise = self.data.noise
        if not isinstance(noise, BackgroundNoise):
            raise ValueError('Explicit background decomposition required for nuisance export')
        idx = _indices(indices, len(self.mean))
        _, conditional_variance = noise.conditional_background(np.zeros(len(self.data.y)))
        size = len(idx)+len(conditional_variance)
        self.gaussian.prior.budget.guard(self.gaussian.memory['peak_bytes']+24*size*size, 'Nuisance moments')
        count, mean, centered = 0, np.zeros(size), np.zeros((size, size))
        for values in self.batches():
            residual = self.data.y[:, None]-self.data.design.forward(values)
            backgrounds, _ = noise.conditional_background(residual)
            sample = np.vstack((values[idx], backgrounds))
            count, mean, centered = _combine_moments(count, mean, centered, sample)
        covariance = centered/(count-1)
        covariance[len(idx):, len(idx):] += np.diag(conditional_variance)
        return dict(mean=mean, covariance=covariance, parameter_indices=idx, background_exposures=self.data.exposure_ids)


def _combine_moments(count: int, mean: np.ndarray, m2: np.ndarray, batch: np.ndarray) -> tuple:
    """Stable parallel-variance update, for a diagonal or selected full covariance."""
    k = batch.shape[1]
    if k == 0:
        return count, mean, m2
    block_mean = batch.mean(axis=1)
    delta = block_mean-mean
    centered = batch-block_mean[:, None]
    if m2.ndim == 1:
        update = np.sum(centered*centered, axis=1)+delta*delta*count*k/(count+k)
    else:
        update = centered@centered.T+np.outer(delta, delta)*count*k/(count+k)
    return count+k, mean+delta*k/(count+k), m2+update


def fit_positive(data: OperatorData, prior: GridPrior, *, output_dir: Path, seed: int,
                 draws: int = 16384, minimum_accepted: int = 512) -> PositivePosterior:
    """Normalized Stage 4 positive posterior with streamed IID rejection draws.

    Proposal counts, all accepted samples, evidence/mass bounds and convergence
    diagnostics survive a sampling failure. The declared draw budget is never
    increased adaptively to hide failure. Priors/geometry are not learned on science.
    """
    if data.partition != 'train' or data.verified is not True:
        raise ValueError('Only independently verified training templates may select the static model')
    if not isinstance(draws, int) or not 512 <= draws <= 262144 or not 32 <= minimum_accepted <= draws:
        raise ValueError('Invalid bounded sampling budget')
    if draws*prior.size*8 > prior.budget.disk_bytes:
        raise MemoryError('Declared posterior draw disk budget exceeded')
    directory = Path(tempfile.mkdtemp(prefix='positive-', dir=output_dir))
    start = time.perf_counter()
    gaussian = GaussianPosterior(data, prior)
    data = gaussian.data
    prior_rng, posterior_rng = (np.random.default_rng(s) for s in np.random.SeedSequence(seed).spawn(2))
    count, prior_good = 0, 0
    mean, m2 = np.zeros(prior.size), np.zeros(prior.size)
    chunks = []
    discrepancy = []
    sample_start = time.perf_counter()
    for begin in range(0, draws, prior.budget.batch_size):
        n = min(draws-begin, prior.budget.batch_size)
        prior_draw = prior.sample(prior_rng, n)[:prior.host_size]+prior.mean[:prior.host_size, None]
        prior_good += int(np.all(prior_draw >= 0, axis=0).sum())
        proposals = gaussian.sample(posterior_rng, n)
        positive = proposals[:, np.all(proposals >= 0, axis=0)]
        if positive.shape[1]:
            path = directory/f'draws-{begin:08d}.npy'
            np.save(path, positive, allow_pickle=False); chunks.append(path)
            count, mean, m2 = _combine_moments(count, mean, m2, positive)
            if sum(len(v) for v in discrepancy) < 512:
                residual = data.y[:, None]-data.design.forward(positive)
                discrepancy.append(np.sum(residual*data.noise.solve(residual), axis=0))
    pi, qi = _mass_interval(prior_good, draws), _mass_interval(count, draws)
    diagnostics = dict(seed=seed, proposals=draws, accepted=count, prior_host_accepted=prior_good,
                       prior_host_mass_95=pi, posterior_mass_95=qi, mass_interval_confidence=.975,
                       sampling='IID conditional Gaussian simulation and rejection; no MCMC',
                       approximation='none; finite Matérn 3/2 covariance, conditional geometry',
                       embedding_attempts=prior.embedding_attempts, memory_estimate=gaussian.memory,
                       memory_budget_bytes=prior.budget.memory_bytes, disk_budget_bytes=prior.budget.disk_bytes,
                       sample_batch_size=prior.budget.batch_size, fft_batch_size=prior.fft_batch_size,
                       fft_work_target_bytes=prior.budget.fft_work_bytes, adjoint_error=gaussian.adjoint_error,
                       maximum_solve_residual=gaussian.maximum_solve_residual,
                       solve_residual_history=gaussian.residual_history,
                       operator_counts=dict(data.design.counts), prior_counts=dict(prior.counts),
                       assembly_operator_counts=gaussian.assembly_operator_counts,
                       exact_sparse_operator_cache_bytes=gaussian.operator_cache_bytes,
                       solve_count=gaussian.solve_count, conditioning_exposures=data.exposure_ids,
                       geometry_uncertainty='conditional; no position iterations or registration integration',
                       timings=dict(gp_setup=prior.setup_seconds, assembly=gaussian.assembly_seconds,
                                    gaussian=gaussian.seconds, factor=gaussian.factor_seconds,
                                    positive_sampling=time.perf_counter()-sample_start, total=time.perf_counter()-start),
                       parameters=prior.size, likelihood_rank_upper_bound=min(data.design.shape),
                       likelihood_rank_deficient=True if len(data.y) < prior.size else None,
                       likelihood_rank_note='upper bound only unless explicit small rank diagnostic requested',
                       science_promotion_approved=False)
    if prior_good < minimum_accepted or count < minimum_accepted:
        diagnostics['status'] = 'insufficient_positive_mass'
        (directory/'diagnostics.json').write_text(json.dumps(diagnostics, indent=2))
        raise NumericalFailure('Positive mass insufficient; no interval/evidence manufactured', diagnostics)
    logz = gaussian.log_evidence+np.log(count/prior_good)+len(data.source_names)*np.log(2)
    diagnostics['log_evidence_95'] = [float(gaussian.log_evidence+np.log(qi[0]/pi[1])+len(data.source_names)*np.log(2)),
                                     float(gaussian.log_evidence+np.log(qi[1]/pi[0])+len(data.source_names)*np.log(2))]
    variance = m2/(count-1)
    if not np.isfinite(variance).all() or np.any(variance <= 0):
        raise NumericalFailure('Unresolved positive posterior variance', diagnostics)
    diagnostics['mean_mc_se'] = np.sqrt(variance/count).tolist()
    ppc = float(np.mean(chi2.sf(np.concatenate(discrepancy)[:512], len(data.y))))
    diagnostics.update(status='sampled', training_posterior_predictive_tail=ppc, static_mismatch=ppc < 1e-4,
                       mismatch_threshold=1e-4, mismatch_note='Stage 4 conservative diagnostic; not science acceptance')
    (directory/'diagnostics.json').write_text(json.dumps(diagnostics, indent=2))
    return PositivePosterior(gaussian, directory, tuple(chunks), mean, variance, float(logz), diagnostics)


def conditional_log_likelihood(posterior: PositivePosterior, science: OperatorData,
                               transient_column: np.ndarray, flux: float) -> tuple[float, float]:
    """Stage 6 interface: template posterior expectation, with nonnegative flux."""
    p = np.asarray(transient_column)
    if (science.partition != 'science' or p.shape != science.y.shape or not np.isfinite(p).all()
            or np.any(p < 0) or not np.isfinite(flux) or flux < 0):
        raise ValueError('Science partition, nonnegative native transient response and flux required')
    return posterior.log_predictive(science, p*flux)


def model_weights(posteriors: tuple[PositivePosterior, ...], prior_mass: np.ndarray) -> dict:
    """Evidence-normalized candidate weights with orthant-integration uncertainty."""
    p = np.asarray(prior_mass, dtype=float)
    if (not posteriors or p.shape != (len(posteriors),) or np.any(p <= 0) or not np.isfinite(p).all()
            or len({f.data.likelihood_digest() for f in posteriors}) != 1):
        raise ValueError('All completed candidates, positive prior masses and identical template likelihood required')
    scores = np.array([f.log_evidence for f in posteriors])+np.log(p/p.sum())
    bounds = np.array([f.diagnostics['log_evidence_95'] for f in posteriors])+np.log(p/p.sum())[:, None]
    intervals = []
    for j in range(len(p)):
        lo, hi = bounds[:, 1].copy(), bounds[:, 0].copy()
        lo[j], hi[j] = bounds[j, 0], bounds[j, 1]
        intervals.append([np.exp(lo[j]-logsumexp(lo)), np.exp(hi[j]-logsumexp(hi))])
    return dict(weights=np.exp(scores-logsumexp(scores)), orthant_mc_weight_bounds=np.array(intervals),
                status='all_candidates_misspecified' if all(f.diagnostics['static_mismatch'] for f in posteriors) else 'conditional_comparison')


def mixture_moments(posteriors: tuple[PositivePosterior, ...], weights: np.ndarray,
                    components: tuple[int | str, ...]) -> dict:
    """Selected host indices/source names with within- and between-model covariance.

    Missing source components contribute exactly zero as in Stage 4. Avoids a full
    host inverse/covariance. All candidates must share the same angular host grid.
    """
    p = _weights(posteriors, weights)
    means, covariances = [], []
    for fit in posteriors:
        if not np.array_equal(fit.data.host_xy_arcsec, posteriors[0].data.host_xy_arcsec):
            raise ValueError('Mixture host grids must match')
        positions, indices = [], []
        for j, component in enumerate(components):
            if isinstance(component, int):
                if not 0 <= component < fit.gaussian.prior.host_size:
                    raise ValueError('Host component index out of range')
                positions.append(j); indices.append(component)
            elif component in fit.data.source_names:
                positions.append(j); indices.append(fit.gaussian.prior.host_size+fit.data.source_names.index(component))
        m, c = np.zeros(len(components)), np.zeros((len(components), len(components)))
        if indices:
            selected_mean, selected_cov = fit.moments(np.array(indices))
            m[positions] = selected_mean; c[np.ix_(positions, positions)] = selected_cov
        means.append(m); covariances.append(c)
    means = np.array(means); mean = p@means
    within = np.einsum('i,ijk->jk', p, covariances)
    delta = means-mean; between = np.einsum('i,ij,ik->jk', p, delta, delta)
    return dict(mean=mean, covariance=within+between, within_covariance=within, between_covariance=between)


def _weights(posteriors: tuple, weights: np.ndarray) -> np.ndarray:
    p = np.asarray(weights, dtype=float)
    if not posteriors or p.shape != (len(posteriors),) or not np.isfinite(p).all() or np.any(p < 0) or not np.isclose(p.sum(), 1):
        raise ValueError('Normalized nonnegative weights required')
    return p


def mixture_conditional_likelihood(posteriors: tuple[PositivePosterior, ...],
                                   science_by_model: tuple[OperatorData, ...], weights: np.ndarray,
                                   transient_column: np.ndarray, flux: float) -> dict:
    """Static-model mixture retaining template weights and low-ESS failure status."""
    p = _weights(posteriors, weights)
    if len(science_by_model) != len(posteriors) or len({d.likelihood_digest() for d in science_by_model}) != 1:
        raise ValueError('Every candidate must evaluate identical science data/noise')
    scores = np.array([conditional_log_likelihood(f, d, transient_column, flux)
                       for f, d in zip(posteriors, science_by_model)])
    use = p > 0
    return dict(log_likelihood=float(logsumexp(np.log(p[use])+scores[use, 0])),
                component_importance_ess=scores[:, 1].tolist(),
                status='insufficient_predictive_samples' if np.any(scores[use, 1] < 32) else 'evaluated',
                interpretation='likelihood only; no confidence interval or science promotion')
