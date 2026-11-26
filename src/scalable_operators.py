"""Frozen native operators and exact calibrated-background noise for Stage 5.

No dense-reference implementation is called here. The host actions snapshot the
established instrument operators; point columns use the native Stage 2/3 renderers.
Geometry is conditional, just as in Stage 4. Units are Jy/native pixel for host
coefficients and the supplied calibration's integrated-flux convention for sources.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import Callable, Protocol

import numpy as np
from astropy.wcs import WCS
from scipy.linalg import cho_solve
from scipy.sparse import csr_matrix

from .isolated_astrometry import _freeze, _frame_identity, star_response
from .nucleus import TemplateFrame, gaussian_response, validate_frames


class Noise(Protocol):
    """SPD observation covariance, acting on columns, with normalized density."""
    size: int
    logdet: float
    storage_bytes: int
    def action(self, x: np.ndarray) -> np.ndarray: ...
    def solve(self, x: np.ndarray) -> np.ndarray: ...
    def sample(self, rng: np.random.Generator, count: int) -> np.ndarray: ...
    def digest(self) -> str: ...


class DenseNoise:
    """Small arbitrary full-noise adapter; no diagonal approximation is made."""
    def __init__(self, covariance: np.ndarray):
        a = np.asarray(covariance, dtype=float)
        if (a.ndim != 2 or a.shape[0] != a.shape[1] or not len(a)
                or not np.isfinite(a).all() or not np.allclose(a, a.T, rtol=1e-12, atol=0)):
            raise ValueError('Finite symmetric noise covariance required')
        self.covariance = _freeze(a)
        self.root = np.linalg.cholesky(a)
        self.size = len(a)
        self.logdet = float(2*np.log(np.diag(self.root)).sum())
        self.storage_bytes = 2*a.nbytes

    def action(self, x: np.ndarray) -> np.ndarray:
        return self.covariance@x

    def solve(self, x: np.ndarray) -> np.ndarray:
        return cho_solve((self.root, True), x)

    def sample(self, rng: np.random.Generator, count: int) -> np.ndarray:
        return self.root@rng.standard_normal((self.size, count))

    def digest(self) -> str:
        return hashlib.sha256(self.covariance.tobytes()).hexdigest()


class NoiseSolveFailure(RuntimeError):
    """The full background-noise equations could not be resolved in float64."""
    def __init__(self, diagnostics: dict):
        super().__init__('Background noise solve residual exceeds 1e-10; precision unresolved')
        self.diagnostics = diagnostics


class BackgroundNoise:
    """Independent native variances plus one calibrated background per exposure.

    N_i = diag(sigma_i**2) + tau_i**2 11.T. Weighted common-mode separation and
    the determinant lemma retain the entire rank-one term. Backgrounds may be signed.
    Their external calibration has already been subtracted from the measurements.
    """
    def __init__(self, variances: tuple[np.ndarray, ...], background_sd: tuple[float, ...]):
        if not variances or len(variances) != len(background_sd):
            raise ValueError('One variance vector and background SD per exposure required')
        self.variances = tuple(_freeze(v) for v in variances)
        self.tau = _freeze(np.asarray(background_sd, dtype=float))
        if (any(v.ndim != 1 or not len(v) or not np.isfinite(v).all() or np.any(v <= 0)
                for v in self.variances) or self.tau.shape != (len(self.variances),)
                or not np.isfinite(self.tau).all() or np.any(self.tau < 0)):
            raise ValueError('Positive native variances and nonnegative finite background SD required')
        edges = np.r_[0, np.cumsum([len(v) for v in self.variances])]
        self.slices = tuple(slice(int(a), int(b)) for a, b in zip(edges[:-1], edges[1:]))
        self.size = int(edges[-1])
        self.logdet = float(sum(np.log(v).sum()+np.log1p(t*t*np.sum(1/v))
                               for v, t in zip(self.variances, self.tau)))
        self.storage_bytes = 8*(self.size+len(self.tau))
        self.solve_count = 0
        self.maximum_solve_residual = 0.

    def action(self, x: np.ndarray) -> np.ndarray:
        columns = np.asarray(x, dtype=float).reshape(self.size, -1)
        out = np.empty_like(columns)
        for sl, v, t in zip(self.slices, self.variances, self.tau):
            out[sl] = v[:, None]*columns[sl]+t*t*columns[sl].sum(axis=0)
        return out.reshape(x.shape)

    def solve(self, x: np.ndarray) -> np.ndarray:
        """Separate common and differential modes, checking every RHS afterwards.

        Subtracting two precision-weighted vectors loses the common mode when
        tau dominates native noise, even for a perfectly conditioned scalar.
        Center before weighting instead. Very ill-conditioned multi-pixel cases
        can still be unrepresentable; refuse them rather than report a false fit.
        """
        columns = np.asarray(x, dtype=float).reshape(self.size, -1)
        if not np.isfinite(columns).all():
            raise ValueError('Finite noise RHS required')
        out = np.empty_like(columns)
        for sl, v, t in zip(self.slices, self.variances, self.tau):
            if t == 0:
                out[sl] = columns[sl]/v[:, None]
                continue
            weights = v.min()/v
            centered = columns[sl]-columns[sl.start]
            average = np.sum(weights[:, None]*centered, axis=0)/weights.sum()
            common = columns[sl.start]+average
            out[sl] = ((centered-average)/v[:, None]
                       + weights[:, None]*(common/(v.min()+t*t*weights.sum())))
        residual = self.action(out)-columns
        # Per-exposure, per-RHS checks prevent a large column or a bright exposure
        # from hiding another's failure. Max norms avoid squaring/overflow.
        per_block = np.array([np.max(np.abs(residual[sl]), axis=0)/np.maximum(
            np.max(np.abs(columns[sl]), axis=0), np.finfo(float).tiny) for sl in self.slices])
        relative = np.max(per_block, axis=0)
        worst = float(np.max(relative))
        self.solve_count += 1
        self.maximum_solve_residual = max(self.maximum_solve_residual, worst)
        if not np.isfinite(out).all() or not np.isfinite(worst) or worst > 1e-10:
            raise NoiseSolveFailure(dict(relative_residual=worst, per_rhs=relative.tolist(),
                                         per_exposure_rhs=per_block.tolist(),
                                         tolerance=1e-10, solve_count=self.solve_count))
        return out.reshape(x.shape)

    def sample(self, rng: np.random.Generator, count: int) -> np.ndarray:
        out = np.empty((self.size, count))
        for sl, v, t in zip(self.slices, self.variances, self.tau):
            out[sl] = np.sqrt(v[:, None])*rng.standard_normal((len(v), count))
            out[sl] += t*rng.standard_normal(count)
        return out

    def conditional_background(self, residual: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Conditional background means and variances given physical parameters."""
        columns = np.asarray(residual, dtype=float).reshape(self.size, -1)
        means, variances = [], []
        for sl, v, t in zip(self.slices, self.variances, self.tau):
            variance = t*t/(1+t*t*np.sum(1/v))
            means.append(variance*np.sum(columns[sl]/v[:, None], axis=0))
            variances.append(variance)
        return np.array(means), np.array(variances)

    def digest(self) -> str:
        h = hashlib.sha256(self.tau.tobytes())
        for v in self.variances:
            h.update(str(v.shape).encode()); h.update(v.tobytes())
        return h.hexdigest()


class BlockNoise:
    """Independent exposure/partition noise blocks, including full dense blocks."""
    def __init__(self, blocks: tuple[Noise, ...]):
        self.blocks = blocks
        edges = np.r_[0, np.cumsum([b.size for b in blocks])]
        self.slices = tuple(slice(int(a), int(b)) for a, b in zip(edges[:-1], edges[1:]))
        self.size = int(edges[-1])
        self.logdet = sum(b.logdet for b in blocks)
        self.storage_bytes = sum(b.storage_bytes for b in blocks)

    def action(self, x: np.ndarray) -> np.ndarray:
        return np.concatenate([b.action(x[sl]) for b, sl in zip(self.blocks, self.slices)])

    def solve(self, x: np.ndarray) -> np.ndarray:
        return np.concatenate([b.solve(x[sl]) for b, sl in zip(self.blocks, self.slices)])

    def sample(self, rng: np.random.Generator, count: int) -> np.ndarray:
        return np.vstack([b.sample(rng, count) for b in self.blocks])

    def digest(self) -> str:
        return hashlib.sha256(repr(tuple(b.digest() for b in self.blocks)).encode()).hexdigest()


class Design:
    """Finite linear map with explicit Euclidean adjoint and operation counters.

    Callables must be frozen and input-immutable. Batched calls are columns.
    The native factory below constructs those snapshots; ``from_matrix`` serves
    independent algebra tests. Arbitrary callables require caller provenance.
    """
    def __init__(self, shape: tuple[int, int], forward: Callable, adjoint: Callable,
                 *, provenance: str, storage_bytes: int = 0):
        if len(shape) != 2 or min(shape) < 1 or not provenance:
            raise ValueError('Nonempty design and explicit operator provenance required')
        self.shape, self._forward, self._adjoint = shape, forward, adjoint
        self.provenance, self.storage_bytes = provenance, storage_bytes
        self.counts = dict(forward=0, adjoint=0)

    @classmethod
    def from_matrix(cls, matrix: np.ndarray) -> Design:
        a = _freeze(matrix)
        if a.ndim != 2 or not np.isfinite(a).all() or np.any(a < -64*np.finfo(float).eps):
            raise ValueError('Finite nonnegative brightness matrix required')
        digest = hashlib.sha256(a.tobytes()).hexdigest()
        result = cls(a.shape, lambda x: a@x, lambda x: a.T@x,
                     provenance='explicit matrix '+digest, storage_bytes=a.nbytes)
        result.explicit = True
        return result

    @classmethod
    def from_sparse(cls, matrix: csr_matrix, *, provenance: str) -> Design:
        """Freeze an exact sparse design, retaining even tiny nonzero coefficients."""
        a = matrix.copy()
        if not np.isfinite(a.data).all():
            raise ValueError('Finite sparse coefficients required')
        for value in (a.data, a.indices, a.indptr):
            value.setflags(write=False)
        result = cls(a.shape, lambda x: a@x, lambda x: a.T@x, provenance=provenance,
                     storage_bytes=sum(v.nbytes for v in (a.data, a.indices, a.indptr)))
        result.explicit = True
        return result

    def _apply(self, x: np.ndarray, transpose: bool) -> np.ndarray:
        value = np.asarray(x, dtype=float)
        side = int(transpose)
        if value.ndim not in (1, 2) or value.shape[0] != self.shape[1-side] or not np.isfinite(value).all():
            raise ValueError('Finite operator input with matching dimensions required')
        self.counts['adjoint' if transpose else 'forward'] += 1 if value.ndim == 1 else value.shape[1]
        fn = self._adjoint if transpose else self._forward
        out = np.asarray(fn(value), dtype=float)
        if out.shape != (self.shape[side],)+value.shape[1:] or not np.isfinite(out).all():
            raise ValueError('Invalid forward/adjoint output')
        return out

    def forward(self, x: np.ndarray) -> np.ndarray:
        return self._apply(x, False)

    def adjoint(self, x: np.ndarray) -> np.ndarray:
        return self._apply(x, True)

    def check_adjoint(self, *, tolerance: float = 1e-10) -> float:
        rng = np.random.default_rng(1729)
        errors = []
        for _ in range(3):
            x, y = rng.normal(size=self.shape[1]), rng.normal(size=self.shape[0])
            ax, aty = self.forward(x), self.adjoint(y)
            errors.append(abs(ax@y-x@aty)/max(np.linalg.norm(ax)*np.linalg.norm(y),
                                             np.linalg.norm(x)*np.linalg.norm(aty), np.finfo(float).tiny))
        error = max(errors)
        if error > tolerance:
            raise ValueError(f'Adjoint check failed: {error:.3g} > {tolerance:.3g}')
        return error


@dataclass(frozen=True)
class OperatorData:
    """Native pixels, full noise, component layout and template/calibration gauge."""
    y: np.ndarray
    noise: Noise
    design: Design
    host_xy_arcsec: np.ndarray
    source_names: tuple[str, ...]
    exposure_ids: tuple[str, ...]
    partition: str
    verified: bool
    provenance: str
    calibration_gauge: tuple = ('synthetic-linear-algebra',)

    def __post_init__(self) -> None:
        y, xy = _freeze(self.y), _freeze(self.host_xy_arcsec)
        if (y.ndim != 1 or not len(y) or not np.isfinite(y).all() or xy.ndim != 2
                or xy.shape[1] != 2 or not len(xy) or not np.isfinite(xy).all()
                or self.noise.size != len(y) or self.design.shape != (len(y), len(xy)+len(self.source_names))):
            raise ValueError('Finite native data, host coordinates and matching operator/noise required')
        if (not self.exposure_ids or len(set(self.exposure_ids)) != len(self.exposure_ids)
                or len(set(self.source_names)) != len(self.source_names) or not self.provenance
                or self.partition not in ('train', 'validation', 'heldout', 'science')):
            raise ValueError('Unique IDs, provenance and declared partition required')
        object.__setattr__(self, 'y', y); object.__setattr__(self, 'host_xy_arcsec', xy)

    def likelihood_digest(self) -> str:
        """Model-independent identity for candidate evidence comparisons."""
        h = hashlib.sha256(self.y.tobytes())
        h.update(self.noise.digest().encode())
        h.update(repr((self.exposure_ids, self.partition, self.calibration_gauge)).encode())
        return h.hexdigest()


def from_dense(data: object) -> OperatorData:
    """Adapt a Stage 4 fixture without invoking its inference implementation."""
    return OperatorData(data.y, DenseNoise(data.noise), Design.from_matrix(data.design),
                        data.host_xy_arcsec, data.source_names, data.exposure_ids,
                        data.partition, data.verified, data.provenance, data.calibration_gauge)


def frozen_host_operator(scene_wcs: WCS, native_wcs: WCS, scene_shape: tuple[int, int],
                         native_shape: tuple[int, int], channel: str, *,
                         is_full_array: bool = False) -> Design:
    """Snapshot the selected established F/adjoint; never silently change PRF mode.

    Project-then-convolve and scene-anchor orders are supported. Dense exact-scene
    PRFs are rejected rather than replaced with anchors. Configure calibration
    before construction; subsequent config/WCS changes cannot alter this operator.
    """
    from . import solver, config
    project_first = bool(config.PRF_ORDER_PROJECT_THEN_CONVOLVE)
    if not project_first and solver._prf_operator_mode(scene_shape) == 'exact':
        raise ValueError('Exact scene-PRF mode has no scalable implementation; no anchor substitution')
    projection = solver._projection_matrix(scene_wcs, native_wcs, scene_shape, native_shape).copy()
    bundle = (solver._get_prf_operator_bundle_native(native_wcs, native_shape, channel, is_full_array)
              if project_first else solver._get_prf_operator_bundle(scene_wcs, native_wcs, scene_shape, channel, is_full_array))
    kernels, weights, wsum = tuple(_freeze(k) for k in bundle[0]), tuple(_freeze(w) for w in bundle[1]), _freeze(bundle[2])
    def one(x: np.ndarray, adj: bool) -> np.ndarray:
        if project_first:
            if adj:
                z = solver._apply_prf_adjoint_from_bundle(x.reshape(native_shape), kernels, weights, wsum)
                return projection.T@z.ravel()
            z = (projection@x).reshape(native_shape)
            return solver._apply_prf_operator_from_bundle(z, kernels, weights, wsum).ravel()
        if adj:
            z = (projection.T@x).reshape(scene_shape)
            return solver._apply_prf_adjoint_from_bundle(z, kernels, weights, wsum).ravel()
        z = solver._apply_prf_operator_from_bundle(x.reshape(scene_shape), kernels, weights, wsum)
        return projection@z.ravel()
    def apply(x: np.ndarray, adj: bool) -> np.ndarray:
        return one(x, adj) if x.ndim == 1 else np.column_stack([one(v, adj) for v in x.T])
    digest = hashlib.sha256(repr((scene_shape, native_shape, channel, project_first)).encode())
    for a in (*kernels, *weights, wsum, projection.data, projection.indices, projection.indptr):
        digest.update(a.tobytes())
    storage = sum(a.nbytes for a in (*kernels, *weights, wsum, projection.data, projection.indices, projection.indptr))
    return Design((int(np.prod(native_shape)), int(np.prod(scene_shape))),
                  lambda x: apply(x, False), lambda x: apply(x, True),
                  provenance='frozen established host F '+digest.hexdigest(), storage_bytes=storage)


def native_template_data(frames: tuple[TemplateFrame, ...], reference: WCS,
                         host_xy_arcsec: np.ndarray, host_operators: tuple[Design, ...], *,
                         stars_xy: np.ndarray, nucleus_xy: np.ndarray | None = None,
                         nucleus_profile: np.ndarray | None = None, core_envelope: bool = False,
                         quadrature_order: int = 21, experimental_real_quadrature: bool = False) -> OperatorData:
    """Native Stage 4 design/noise semantics with matrix-free host columns.

    Registration covariance is explicitly unsupported, preserving the conditional
    reference model. Invalid pixels are removed before any linear algebra; full
    detector coordinates and calibrated finite-stamp source losses are preserved.
    """
    validate_frames(frames)
    if len(frames) != len(host_operators) or len({f.split for f in frames}) != 1:
        raise ValueError('One host operator per exposure and one partition required')
    if any(np.any(f.registration_covariance) for f in frames):
        raise ValueError('Conditional geometry only: supply separate registration alternatives')
    stars = np.asarray(stars_xy, dtype=float).reshape(-1, 2)
    names = tuple(f'star_{i}' for i in range(len(stars)))
    if nucleus_xy is None and (nucleus_profile is not None or core_envelope):
        raise ValueError('Nuclear profile requires a centroid')
    if nucleus_xy is not None:
        names += ('nucleus_core', 'nucleus_envelope') if core_envelope else ('nucleus',)
    host_count = len(host_xy_arcsec)
    sources, masks, values, variances, identities = [], [], [], [], []
    for frame, host in zip(frames, host_operators):
        image = frame.image
        if host.shape != (image.data.size, host_count):
            raise ValueError('Host operator shape mismatch')
        columns = [star_response(image, reference, p, frame.registration_xy).ravel() for p in stars]
        if nucleus_xy is not None:
            if nucleus_profile is None or core_envelope:
                columns.append(star_response(image, reference, nucleus_xy, frame.registration_xy).ravel())
            if nucleus_profile is not None:
                if not getattr(image.response, 'synthetic', False) and not experimental_real_quadrature:
                    raise ValueError('Real extended-profile quadrature requires explicit experimental opt-in')
                columns.append(gaussian_response(image, reference, nucleus_xy, nucleus_profile,
                                                frame.registration_xy, order=quadrature_order).ravel())
            elif core_envelope:
                raise ValueError('Core/envelope requires a resolved envelope profile')
        mask = image.valid.ravel().copy()
        source = np.column_stack(columns)[mask] if columns else np.empty((int(mask.sum()), 0))
        if not np.isfinite(source).all() or np.any(source < -64*np.finfo(float).eps):
            raise ValueError('Invalid physical source response')
        sources.append(_freeze(source)); masks.append(mask)
        values.append((image.data-frame.background).ravel()[mask])
        variances.append(image.sigma.ravel()[mask]**2)
        identities.append((_frame_identity(image), host.provenance, frame.registration_xy.tolist()))
    edges = np.r_[0, np.cumsum([len(y) for y in values])]
    slices = tuple(slice(int(a), int(b)) for a, b in zip(edges[:-1], edges[1:]))
    def forward(x: np.ndarray) -> np.ndarray:
        return np.concatenate([host.forward(x[:host_count])[mask]+source@x[host_count:]
                               for host, mask, source in zip(host_operators, masks, sources)])
    def adjoint(x: np.ndarray) -> np.ndarray:
        out = np.zeros((host_count+len(names),)+x.shape[1:])
        for sl, host, mask, source in zip(slices, host_operators, masks, sources):
            native = np.zeros((host.shape[0],)+x.shape[1:]); native[mask] = x[sl]
            out[:host_count] += host.adjoint(native)
            out[host_count:] += source.T@x[sl]
        return out
    provenance = 'native calibrated components; '+repr(identities)
    design = Design((int(edges[-1]), host_count+len(names)), forward, adjoint,
                    provenance=provenance, storage_bytes=sum(h.storage_bytes for h in host_operators)
                    +sum(s.nbytes+m.nbytes for s, m in zip(sources, masks)))
    response = frames[0].image.response
    return OperatorData(np.concatenate(values), BackgroundNoise(tuple(variances), tuple(f.background_sigma for f in frames)),
                        design, host_xy_arcsec, names, tuple(f.exposure_id for f in frames),
                        frames[0].split, True, provenance+'; '+'; '.join(f.provenance for f in frames),
                        (response.channel, response.mission, response.convention, getattr(response, 'synthetic', False)))
