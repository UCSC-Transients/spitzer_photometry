"""Joint isolated-star flux/centroid and exposure-translation inference.

This is a Stage 2 component interface, not a host/GP/transient solver. Stars have
one nonnegative amplitude each. Their sky positions are expressed in reference
WCS pixels; translations are detector pixels and frame zero is the fixed gauge.
No Gaussian calibration, intrapixel correction or global mutable config is used.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
import hashlib
import time

import numpy as np
from astropy.wcs import WCS
from scipy.optimize import least_squares

from .point_response import NativePRF, PRFGrid


def _freeze(values: np.ndarray) -> np.ndarray:
    a = np.asarray(values, dtype=float)
    return np.frombuffer(a.tobytes(), dtype=float).reshape(a.shape)


def _frame_identity(frame: StarFrame) -> tuple:
    """Capture geometry and calibration, including SIP/lookup distortion arrays."""
    digest = hashlib.sha256()
    for w in (frame.wcs, getattr(frame.wcs, "_detector_wcs", frame.wcs)):
        for hdu in w.to_fits(relax=True):
            digest.update(hdu.header.tostring().encode())
            if hdu.data is not None:
                digest.update(np.asarray(hdu.data).tobytes())
    response = frame.response
    calibration = (response.identity() if isinstance(response, PRFGrid) else
                   (response.sha256, response.oversample, response.convention,
                    response.center_xy, response.normalization, response.synthetic))
    return (frame.data.shape, digest.hexdigest(), calibration)


@dataclass(frozen=True)
class StarFrame:
    """Snapshot native Jy/pixel data, sigma, WCS and an explicit PRF provider.

    WCS must describe native detector sampling (including SIP), not a resampled
    image. ``_detector_wcs`` is required for cropped real-calibration frames.
    Invalid data/sigma and false mask pixels are excluded without changing inputs.
    """
    data: np.ndarray
    sigma: np.ndarray
    wcs: WCS
    response: NativePRF | PRFGrid
    mask: np.ndarray | None = None
    valid: np.ndarray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        data, sigma = _freeze(self.data), _freeze(self.sigma)
        if data.ndim != 2 or data.shape != sigma.shape:
            raise ValueError("Data and sigma must have the same 2D shape")
        valid = np.isfinite(data) & np.isfinite(sigma) & (sigma > 0)
        if self.mask is not None:
            if np.shape(self.mask) != data.shape:
                raise ValueError("Mask shape mismatch")
            valid &= np.asarray(self.mask, dtype=bool)
        if not np.any(valid):
            raise ValueError("Frame has no valid measurements")
        w = self.wcs.deepcopy()
        detector = getattr(self.wcs, "_detector_wcs", None)
        if isinstance(self.response, PRFGrid) and detector is None:
            raise ValueError("Real calibration requires explicit full-detector WCS provenance")
        if detector is not None:
            w._detector_wcs = detector.deepcopy()
        object.__setattr__(self, "data", data)
        object.__setattr__(self, "sigma", sigma)
        object.__setattr__(self, "wcs", w)
        object.__setattr__(self, "valid", np.frombuffer(valid.tobytes(), dtype=bool).reshape(valid.shape))


def star_response(frame: StarFrame, reference_wcs: WCS, position_xy: np.ndarray,
                  translation_xy: np.ndarray) -> np.ndarray:
    """Map a sky source through each WCS, then apply its detector registration.

    A source shift is shared in the reference sky frame; a translation belongs
    to one detector exposure. Both affect the actual full-detector PRF location.
    The same function is used for objective, derivatives, and predictions.
    """
    ra, dec = reference_wcs.pixel_to_world_values(*position_xy)
    x, y = np.asarray(frame.wcs.world_to_pixel_values(ra, dec)) + translation_xy
    if isinstance(frame.response, NativePRF):
        return frame.response.sample(frame.data.shape, float(x), float(y))
    detector = frame.wcs._detector_wcs
    dx, dy = np.asarray(detector.world_to_pixel_values(ra, dec)) + translation_xy
    return frame.response.sample(frame.data.shape, float(x), float(y), float(dx), float(dy))


def response_derivatives(frame: StarFrame, reference_wcs: WCS, position_xy: np.ndarray,
                         translation_xy: np.ndarray, *, step: float = 1e-3) -> tuple[np.ndarray, np.ndarray]:
    """Derivatives per reference pixel and per detector pixel (axis first).

    Bilinear calibration interpolation is piecewise differentiable; derivatives
    at knots are symmetric derivatives. ``step`` is independent of scene SR.
    """
    if not np.isfinite(step) or step <= 0:
        raise ValueError("Derivative step must be positive")
    if isinstance(frame.response, NativePRF):
        def native_position(pos: np.ndarray) -> np.ndarray:
            return np.asarray(frame.wcs.world_to_pixel_values(*reference_wcs.pixel_to_world_values(*pos)))

        x, y = native_position(position_xy)+translation_xy
        detector = frame.response.derivatives(frame.data.shape, x, y)
        geometry = np.array([(native_position(position_xy+delta)-native_position(position_xy-delta))/(2*step)
                             for delta in np.eye(2)*step])
        return np.einsum("ab,bij->aij", geometry, detector), detector
    sky, detector = [], []
    for delta in np.eye(2)*step:
        sky.append((star_response(frame, reference_wcs, position_xy+delta, translation_xy)
                    - star_response(frame, reference_wcs, position_xy-delta, translation_xy))/(2*step))
        detector.append((star_response(frame, reference_wcs, position_xy, translation_xy+delta)
                         - star_response(frame, reference_wcs, position_xy, translation_xy-delta))/(2*step))
    return np.asarray(sky), np.asarray(detector)


@dataclass(frozen=True)
class StarFit:
    """Joint solution; covariance is local curvature only, never boundary coverage."""
    flux: np.ndarray
    positions_xy: np.ndarray
    translations_xy: np.ndarray
    backgrounds: np.ndarray
    covariance: np.ndarray
    diagnostics: dict[str, Any]
    reference_wcs: WCS
    frame_identities: tuple

    def predict(self, frames: tuple[StarFrame, ...]) -> tuple[np.ndarray, ...]:
        """Predict in the same exposure order using the fitted component renderer."""
        if len(frames) != len(self.backgrounds):
            raise ValueError("Prediction frames must match fitted exposure order")
        if tuple(_frame_identity(f) for f in frames) != self.frame_identities:
            raise ValueError("Geometry or calibration changed since fitting; refit required")
        return tuple(self.backgrounds[i] + sum(
            flux * star_response(frame, self.reference_wcs, pos, self.translations_xy[i])
            for flux, pos in zip(self.flux, self.positions_xy)) for i, frame in enumerate(frames))


def fit_isolated_stars(
    frames: tuple[StarFrame, ...], positions_xy: np.ndarray, initial_flux: np.ndarray,
    *, reference_wcs: WCS | None = None, fit_registration: bool = True,
    position_radius: float = 2., translation_radius: float = 2.,
    derivative_step: float = 1e-3, max_nfev: int = 150,
) -> StarFit:
    """Fit fixed stars jointly with signed constant backgrounds and translations.

    ``positions_xy`` are starting source locations in reference-WCS pixels;
    ``initial_flux`` supplies positive scale guesses in the declared PRF units.
    Multiple stars are required for shared registration. Exposure zero has
    translation exactly zero. Rotation/distortion are supplied by WCS, not fitted.
    Search radii are numerical domains; a solution touching them is rejected.
    A rank-deficient or bound-active fit has unavailable (NaN) local covariance.
    No confidence interval at zero flux or absolute calibration is claimed.
    """
    setup_start = time.perf_counter()
    frames = tuple(frames)
    identities = tuple(_frame_identity(f) for f in frames)
    positions = np.asarray(positions_xy, dtype=float)
    initial = np.asarray(initial_flux, dtype=float)
    ns, nf = len(initial), len(frames)
    if not nf or not ns or positions.shape != (ns, 2) or not np.isfinite(positions).all():
        raise ValueError("Need frames and finite (n_star,2) reference positions")
    if not np.isfinite(initial).all() or np.any(initial <= 0):
        raise ValueError("Initial flux scales must be finite and positive")
    if not np.isfinite([position_radius, translation_radius, derivative_step]).all() or min(position_radius, translation_radius, derivative_step) <= 0:
        raise ValueError("Search radii and derivative step must be positive")
    if fit_registration and nf > 1 and ns < 2:
        raise ValueError("Shared exposure registration requires multiple stars")
    reference = (reference_wcs or frames[0].wcs).deepcopy()
    nreg = 2*(nf-1) if fit_registration else 0
    bg0 = 3*ns+nreg
    scale = float(np.max(initial))
    start = np.r_[initial/scale, positions.ravel(), np.zeros(nreg),
                  [np.median(f.data[f.valid])/scale for f in frames]]
    lower = np.r_[np.zeros(ns), (positions-position_radius).ravel(), np.full(nreg, -translation_radius), np.full(nf, -np.inf)]
    upper = np.r_[np.full(ns, np.inf), (positions+position_radius).ravel(), np.full(nreg, translation_radius), np.full(nf, np.inf)]

    def unpack(p: np.ndarray) -> tuple:
        shifts = np.zeros((nf, 2))
        if nreg:
            shifts[1:] = p[3*ns:bg0].reshape(nf-1, 2)
        return p[:ns]*scale, p[ns:3*ns].reshape(ns, 2), shifts, p[bg0:]*scale

    objectives: list[float] = []
    calls = {"response": 0, "native_derivative": 0, "residual": 0, "jacobian": 0}

    def evaluate(p: np.ndarray, jacobian: bool = False) -> np.ndarray:
        flux, pos, shifts, bg = unpack(p)
        blocks = []
        for i, frame in enumerate(frames):
            mask, sigma = frame.valid, frame.sigma[frame.valid]
            model = np.full(frame.data.shape, bg[i])
            jac = np.zeros((int(mask.sum()), len(p))) if jacobian else None
            for j in range(ns):
                response = star_response(frame, reference, pos[j], shifts[i])
                model += flux[j]*response
                if jacobian:
                    sky, detector = response_derivatives(frame, reference, pos[j], shifts[i], step=derivative_step)
                    jac[:, j] = scale*response[mask]/sigma
                    jac[:, ns+2*j:ns+2*j+2] = (flux[j]*sky[:, mask]/sigma).T
                    if nreg and i:
                        jac[:, 3*ns+2*(i-1):3*ns+2*i] += (flux[j]*detector[:, mask]/sigma).T
            if jacobian:
                jac[:, bg0+i] = scale/sigma
                blocks.append(jac)
            else:
                blocks.append((model[mask]-frame.data[mask])/sigma)
        calls["jacobian" if jacobian else "residual"] += 1
        calls["response"] += ns*sum(9 if jacobian and isinstance(f.response, PRFGrid) else 1 for f in frames)
        calls["native_derivative"] += ns*sum(jacobian and isinstance(f.response, NativePRF) for f in frames)
        output = np.vstack(blocks) if jacobian else np.concatenate(blocks)
        if not jacobian:
            objectives.append(float(output @ output))
        return output

    solve_start = time.perf_counter()
    result = least_squares(evaluate, start, jac=lambda p: evaluate(p, True), bounds=(lower, upper),
                           x_scale="jac", ftol=1e-11, xtol=1e-11, gtol=1e-8, max_nfev=max_nfev)
    solve_seconds = time.perf_counter()-solve_start
    if tuple(_frame_identity(f) for f in frames) != identities:
        raise RuntimeError("Geometry or calibration changed during fitting")
    if not result.success or not np.isfinite(result.fun).all() or result.optimality > 1e-3:
        raise RuntimeError(f"Astrometry fit did not converge: {result.message}; optimality={result.optimality}")
    if np.any(result.active_mask[ns:bg0]):
        raise RuntimeError("Astrometry solution reached the specified search domain")
    _, singular, vt = np.linalg.svd(result.jac, full_matrices=False)
    rank = int(np.sum(singular > singular[0]*max(result.jac.shape)*np.finfo(float).eps))
    cov = np.full((len(start), len(start)), np.nan)
    if rank == len(start) and not np.any(result.active_mask):
        cov = (vt.T / singular**2) @ vt
        units = np.r_[np.full(ns, scale), np.ones(2*ns+nreg), np.full(nf, scale)]
        cov *= units[:, None]*units[None, :]
    flux, pos, shifts, bg = unpack(result.x)
    diagnostics = {"success": True, "message": result.message, "nfev": result.nfev,
                   "njev": result.njev, "chi2": float(result.fun @ result.fun),
                   "optimality_scaled": float(result.optimality), "rank": rank,
                   "n_parameters": len(start), "singular_values": singular.tolist(),
                   "active_mask": result.active_mask.tolist(), "reference_frame": 0,
                   "absolute_calibration_verified": False,
                   "objective_history": objectives, "calls": calls,
                   "setup_seconds": solve_start-setup_start, "solve_seconds": solve_seconds,
                   "covariance_order": "flux, source_xy, nonreference_translation_xy, background",
                   "uncertainty": "local joint curvature; unavailable at bounds/rank deficiency"}
    return StarFit(*map(_freeze, (flux, pos, shifts, bg, cov)), diagnostics, reference, identities)
