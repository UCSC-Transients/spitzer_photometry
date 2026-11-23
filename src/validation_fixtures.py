"""Deterministic Stage 1 synthetic fixtures with independent injected truth.

The fixtures in this module are deliberately detector-native and synthetic.  They
do not use :mod:`src.solver`, its point renderer, or any IRAC calibration file.
Point sources are integrated analytically over native pixels and extended
components use fixed sub-pixel quadrature.  Consequently these fixtures test
component bookkeeping and recovery without claiming absolute IRAC calibration.

Development and acceptance seeds/exposures are disjoint.  Required PR tests must
use the development partition; the acceptance partition is reserved for a later
independent promotion decision.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Mapping

import numpy as np
from scipy.special import erf


FIXTURE_SCHEMA_VERSION = "stage1-fixture-v1"
DEVELOPMENT_SEEDS = (1729, 2718, 31415)
ACCEPTANCE_SEEDS = (104729, 130363)
RESERVED_ACCEPTANCE_EXPOSURES = (
    "acceptance-roll-031",
    "acceptance-roll-109",
)


@dataclass(frozen=True)
class ExposureSpec:
    """Immutable geometry and epoch assignment for one native exposure."""

    exposure_id: str
    mjd: float
    roll_deg: float
    phase_x: float
    phase_y: float
    is_template: bool
    epoch_id: int | None
    partition: str = "development"


@dataclass(frozen=True)
class ValidationFixture:
    """A complete detector-native synthetic validation fixture.

    Arrays have shape ``(n_exposure, height, width)``.  ``truth_components``
    stores the noise-free contribution of every named physical component plus
    ``total``.  Cosmic-ray values remain in ``data`` and are identified by
    ``cosmic_ray_mask``; missing pixels are NaN with infinite uncertainty.
    """

    data: np.ndarray
    uncertainty: np.ndarray
    valid_mask: np.ndarray
    cosmic_ray_mask: np.ndarray
    truth_components: Mapping[str, np.ndarray]
    exposures: tuple[ExposureSpec, ...]
    metadata: Mapping[str, object]

    def checksum(self) -> str:
        """Return a stable SHA-256 over arrays, exposure metadata, and schema."""

        digest = hashlib.sha256()
        digest.update(FIXTURE_SCHEMA_VERSION.encode("ascii"))
        for name, value in (
            ("data", self.data),
            ("uncertainty", self.uncertainty),
            ("valid_mask", self.valid_mask),
            ("cosmic_ray_mask", self.cosmic_ray_mask),
        ):
            array = np.ascontiguousarray(value)
            digest.update(name.encode("ascii"))
            digest.update(str(array.dtype).encode("ascii"))
            digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
            digest.update(array.tobytes())
        for name in sorted(self.truth_components):
            digest.update(name.encode("utf-8"))
            digest.update(np.ascontiguousarray(self.truth_components[name]).tobytes())
        exposure_payload = [exposure.__dict__ for exposure in self.exposures]
        digest.update(json.dumps(exposure_payload, sort_keys=True).encode("utf-8"))
        digest.update(json.dumps(dict(self.metadata), sort_keys=True).encode("utf-8"))
        return digest.hexdigest()


def development_exposures() -> tuple[ExposureSpec, ...]:
    """Return the fixed development exposure set.

    Science pairs span exactly one day.  The gap between the pairs is just over
    one day, so a single-linkage implementation would incorrectly chain them.
    Templates have no transient flux by construction.
    """

    return (
        ExposureSpec("dev-science-00", 59000.0, 0.0, 0.10, 0.20, False, 0),
        ExposureSpec("dev-science-01", 59001.0, 17.0, 0.45, 0.55, False, 0),
        ExposureSpec("dev-science-02", 59002.000001, 43.0, 0.80, 0.15, False, 1),
        ExposureSpec("dev-science-03", 59003.000001, 71.0, 0.25, 0.85, False, 1),
        ExposureSpec("dev-template-00", 58990.0, 103.0, 0.60, 0.35, True, None),
        ExposureSpec("dev-template-01", 58991.4, 149.0, 0.05, 0.70, True, None),
    )


def _rotate_sky_to_detector(x: float, y: float, exposure: ExposureSpec) -> tuple[float, float]:
    """Rotate a sky-plane offset into detector coordinates and apply pixel phase."""

    theta = np.deg2rad(exposure.roll_deg)
    c, s = float(np.cos(theta)), float(np.sin(theta))
    return c * x + s * y + exposure.phase_x, -s * x + c * y + exposure.phase_y


def _integrated_axis_gaussian(
    shape: tuple[int, int], x0: float, y0: float, sigma: float, flux: float
) -> np.ndarray:
    """Analytically integrate a circular Gaussian over native pixel boundaries."""

    height, width = shape
    x_edge = np.arange(width + 1, dtype=np.float64) - width / 2.0
    y_edge = np.arange(height + 1, dtype=np.float64) - height / 2.0
    scale = np.sqrt(2.0) * float(sigma)
    x_int = 0.5 * (erf((x_edge[1:] - x0) / scale) - erf((x_edge[:-1] - x0) / scale))
    y_int = 0.5 * (erf((y_edge[1:] - y0) / scale) - erf((y_edge[:-1] - y0) / scale))
    return float(flux) * np.outer(y_int, x_int)


def _quadrature_component(
    shape: tuple[int, int], evaluator: object, *, samples_per_axis: int = 5
) -> np.ndarray:
    """Integrate an independent analytic surface-brightness function by quadrature."""

    height, width = shape
    offsets = (np.arange(samples_per_axis, dtype=np.float64) + 0.5) / samples_per_axis - 0.5
    yy, xx = np.indices(shape, dtype=np.float64)
    xx = xx - width / 2.0 + 0.5
    yy = yy - height / 2.0 + 0.5
    result = np.zeros(shape, dtype=np.float64)
    for dy in offsets:
        for dx in offsets:
            result += evaluator(xx + dx, yy + dy)  # type: ignore[operator]
    return result / float(samples_per_axis**2)


def _host_component(shape: tuple[int, int], exposure: ExposureSpec) -> np.ndarray:
    """Positive smooth plus structured host, evaluated in a common sky frame."""

    theta = np.deg2rad(exposure.roll_deg)
    c, s = float(np.cos(theta)), float(np.sin(theta))

    def surface(x_det: np.ndarray, y_det: np.ndarray) -> np.ndarray:
        x_shift = x_det - exposure.phase_x
        y_shift = y_det - exposure.phase_y
        x_sky = c * x_shift - s * y_shift
        y_sky = s * x_shift + c * y_shift
        smooth = 1.8e-4 * np.exp(-0.5 * ((x_sky / 5.8) ** 2 + (y_sky / 4.3) ** 2))
        structured = 3.0e-5 * (1.0 + np.sin(0.72 * x_sky + 0.31 * y_sky))
        gradient = 1.5e-5 * (1.0 + (x_sky + 8.0) / 16.0)
        return smooth + structured + np.maximum(gradient, 0.0)

    return _quadrature_component(shape, surface)


def _resolved_nucleus(shape: tuple[int, int], exposure: ExposureSpec) -> np.ndarray:
    """Integrate a shifted, resolved elliptical nuclear component."""

    center_x, center_y = _rotate_sky_to_detector(0.35, -0.28, exposure)
    phi = np.deg2rad(25.0 - exposure.roll_deg)
    cp, sp = float(np.cos(phi)), float(np.sin(phi))
    sigma_major, sigma_minor, flux = 1.25, 0.72, 6.0e-3
    normalization = flux / (2.0 * np.pi * sigma_major * sigma_minor)

    def surface(x_det: np.ndarray, y_det: np.ndarray) -> np.ndarray:
        dx, dy = x_det - center_x, y_det - center_y
        major = cp * dx + sp * dy
        minor = -sp * dx + cp * dy
        return normalization * np.exp(-0.5 * ((major / sigma_major) ** 2 + (minor / sigma_minor) ** 2))

    return _quadrature_component(shape, surface, samples_per_axis=7)


def make_validation_fixture(
    *,
    seed: int = DEVELOPMENT_SEEDS[0],
    transient_flux_jy: float = 2.0e-3,
    shape: tuple[int, int] = (17, 17),
    partition: str = "development",
) -> ValidationFixture:
    """Build the deterministic Stage 1 fixture.

    Parameters
    ----------
    seed
        A reserved development seed.  Acceptance seeds require
        ``partition="acceptance"`` so routine tests cannot consume them silently.
    transient_flux_jy
        Nonnegative transient flux shared within each science epoch and exactly
        zero in templates.
    shape
        Native detector shape.  Both dimensions must be at least 13 pixels.
    partition
        ``"development"`` for required PR tests or ``"acceptance"`` for the
        untouched later acceptance partition.

    Notes
    -----
    The point response is a labeled synthetic analytic Gaussian, not a substitute
    for native calibrated-PRF validation.  Real-calibration acceptance remains a
    separate manual gate.
    """

    if partition not in {"development", "acceptance"}:
        raise ValueError("partition must be 'development' or 'acceptance'")
    if partition == "development" and seed not in DEVELOPMENT_SEEDS:
        raise ValueError("required PR fixtures must use a registered development seed")
    if partition == "acceptance" and seed not in ACCEPTANCE_SEEDS:
        raise ValueError("acceptance fixtures must use a reserved acceptance seed")
    if not np.isfinite(transient_flux_jy) or transient_flux_jy < 0.0:
        raise ValueError("transient flux must be finite and nonnegative")
    if min(shape) < 13:
        raise ValueError("fixture dimensions must be at least 13 native pixels")

    exposures = development_exposures()
    if partition == "acceptance":
        exposures = tuple(
            ExposureSpec(
                RESERVED_ACCEPTANCE_EXPOSURES[index % len(RESERVED_ACCEPTANCE_EXPOSURES)] + f"-{index}",
                exposure.mjd + 200.0,
                (exposure.roll_deg + 31.0) % 180.0,
                (exposure.phase_x + 0.17) % 1.0,
                (exposure.phase_y + 0.29) % 1.0,
                exposure.is_template,
                exposure.epoch_id,
                "acceptance",
            )
            for index, exposure in enumerate(exposures)
        )

    n_exposure = len(exposures)
    component_names = ("host", "nucleus", "isolated_star", "blended_stars", "transient", "background")
    components = {name: np.zeros((n_exposure, *shape), dtype=np.float64) for name in component_names}
    point_sigma = 0.78
    for index, exposure in enumerate(exposures):
        components["host"][index] = _host_component(shape, exposure)
        components["nucleus"][index] = _resolved_nucleus(shape, exposure)
        iso_x, iso_y = _rotate_sky_to_detector(-4.2, 3.1, exposure)
        components["isolated_star"][index] = _integrated_axis_gaussian(
            shape, iso_x, iso_y, point_sigma, 4.0e-3
        )
        blend = np.zeros(shape, dtype=np.float64)
        for sky_x, sky_y, flux in ((-1.1, -2.7, 2.5e-3), (0.15, -2.55, 1.4e-3)):
            det_x, det_y = _rotate_sky_to_detector(sky_x, sky_y, exposure)
            blend += _integrated_axis_gaussian(shape, det_x, det_y, point_sigma, flux)
        components["blended_stars"][index] = blend
        transient_x, transient_y = _rotate_sky_to_detector(2.1, -1.4, exposure)
        transient_flux = 0.0 if exposure.is_template else float(transient_flux_jy)
        components["transient"][index] = _integrated_axis_gaussian(
            shape, transient_x, transient_y, point_sigma, transient_flux
        )
        background = 6.0e-5 * (index - (n_exposure - 1) / 2.0)
        components["background"][index].fill(background)

    total = np.sum(np.stack([components[name] for name in component_names]), axis=0)
    rng = np.random.default_rng(seed)
    uncertainty = np.full_like(total, 1.2e-5)
    data = total + rng.normal(0.0, uncertainty)
    cosmic_ray_mask = np.zeros_like(total, dtype=bool)
    cosmic_positions = ((2, 13), (13, 3), (5, 5), (11, 12), (4, 14), (14, 7))
    for index, (y_pos, x_pos) in enumerate(cosmic_positions):
        cosmic_ray_mask[index, y_pos % shape[0], x_pos % shape[1]] = True
        data[index, y_pos % shape[0], x_pos % shape[1]] += 2.5e-2

    valid_mask = np.ones_like(total, dtype=bool)
    for index in range(n_exposure):
        missing = ((0, index % shape[1]), (shape[0] - 1, (3 * index + 1) % shape[1]))
        for y_pos, x_pos in missing:
            valid_mask[index, y_pos, x_pos] = False
            data[index, y_pos, x_pos] = np.nan
            uncertainty[index, y_pos, x_pos] = np.inf

    truth_components: dict[str, np.ndarray] = {name: value for name, value in components.items()}
    truth_components["total"] = total
    metadata: dict[str, object] = {
        "schema_version": FIXTURE_SCHEMA_VERSION,
        "partition": partition,
        "seed": int(seed),
        "truth_renderer": "independent_analytic_pixel_integrals_and_fixed_quadrature",
        "calibration_status": "synthetic_analytic_response_not_real_prf_calibration",
        "point_flux_constraint": "nonnegative",
        "template_transient_flux_jy": 0.0,
        "science_transient_flux_jy": float(transient_flux_jy),
        "epoch_rule": "shared only when total span <= 1 day; no single-linkage chaining",
        "reserved_acceptance_seeds": list(ACCEPTANCE_SEEDS),
        "reserved_acceptance_exposures": list(RESERVED_ACCEPTANCE_EXPOSURES),
    }
    return ValidationFixture(
        data=data,
        uncertainty=uncertainty,
        valid_mask=valid_mask,
        cosmic_ray_mask=cosmic_ray_mask,
        truth_components=truth_components,
        exposures=exposures,
        metadata=metadata,
    )
