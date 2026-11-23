"""Behavioral regressions for the seventh independent review of PR #22."""
from __future__ import annotations

import numpy as np
import pytest
import sep
from astropy.coordinates import SkyCoord
from astropy.table import Table
from astropy.wcs import WCS
from scipy.special import erf

from src import config, native_fit_campaign, pipeline_fit, preprocessing, solver


def _wcs(
    size: int,
    phase: float = 0.0,
    phase_y: float | None = None,
) -> WCS:
    phase_y = phase if phase_y is None else phase_y
    transform = WCS(naxis=2)
    transform.wcs.crpix = [
        (size + 1) / 2 + phase,
        (size + 1) / 2 + phase_y,
    ]
    transform.wcs.crval = [config.TRANSIENT_RA, config.TRANSIENT_DEC]
    transform.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    transform.wcs.cdelt = [-1.22 / 3600.0, 1.22 / 3600.0]
    return transform


def _frame(transform: WCS, size: int, index: int) -> dict:
    return {
        "data": np.full((size, size), 1.0e-3),
        "sigma": np.full((size, size), 1.0e-6),
        "wcs": transform,
        "raw_wcs": transform,
        "epoch_id": 0,
        "is_template": True,
        "filename": f"review27_{index}_ch2.fits",
        "mjd": index / 10.0,
    }


def _integrated_gaussian(
    shape: tuple[int, int],
    x0: float,
    y0: float,
    width: float = 0.9,
) -> np.ndarray:
    yy, xx = np.indices(shape, dtype=float)
    scale = np.sqrt(2.0) * width
    return 0.25 * (
        erf((xx - x0 + 0.5) / scale) - erf((xx - x0 - 0.5) / scale)
    ) * (
        erf((yy - y0 + 0.5) / scale) - erf((yy - y0 - 0.5) / scale)
    )


@pytest.mark.parametrize("gradient", [-1.0e-5, 0.0, 1.0e-5])
@pytest.mark.parametrize("mask_mode", ["right_core", "left_annulus"])
def test_catalog_affine_measurement_matches_asymmetric_support(
    monkeypatch,
    gradient,
    mask_mode,
):
    size = 49
    transform = _wcs(size)
    source = SkyCoord(
        *transform.pixel_to_world_values(24.0, 24.0), unit="deg",
    )
    response = solver.native_point_source_column(
        transform,
        transform,
        (size, size),
        (size, size),
        "ch2",
        source.ra.deg,
        source.dec.deg,
    ).reshape(size, size)
    yy, xx = np.indices((size, size), dtype=float)
    raw_frames = []
    for index in range(3):
        data = 0.01 + 0.002 * response + gradient * (xx - 24.0)
        sigma = np.full((size, size), 1.0e-6)
        if mask_mode == "right_core":
            sigma[(xx >= 25) & (xx <= 27) & (yy >= 22) & (yy <= 26)] = np.inf
        else:
            radius = np.hypot(xx - 24.0, yy - 24.0)
            sigma[(xx < 24) & (radius >= 6) & (radius <= 10)] = np.inf
        raw_frames.append({
            "data": data / config.MJY_SR_TO_JY,
            "sigma": sigma / config.MJY_SR_TO_JY,
            "wcs": transform,
            "filename": f"review27_catalog_{index}_ch2.fits",
            "mjd": index / 10.0,
            "is_full_array": False,
        })
    monkeypatch.setattr(
        preprocessing,
        "load_data",
        lambda _entry: raw_frames.pop(0),
    )
    processed = preprocessing.reproject_to_grid(
        [{} for _ in range(3)], transform, (size, size),
    )
    median, _ = preprocessing.create_median_stack(
        np.asarray([entry["data"] for entry in processed]),
    )
    aperture_radius = 4.88 / 1.22
    flux = sep.sum_circle(
        np.ascontiguousarray(median, dtype=np.float64),
        np.asarray([24.0]),
        np.asarray([24.0]),
        aperture_radius,
        bkgann=(1.5 * aperture_radius, 2.5 * aperture_radius),
        subpix=5,
    )[0][0] * preprocessing._scene_sample_to_integrated_jy_factor(transform)
    row = Table({
        "x": [24.0],
        "y": [24.0],
        "ra": [source.ra.deg],
        "dec": [source.dec.deg],
        "flux": [flux],
        "catalog_aperture_flux_jy": [flux],
        "catalog_aperture_radius_arcsec": [4.88],
        "catalog_aperture_annulus_inner_arcsec": [7.32],
        "catalog_aperture_annulus_outer_arcsec": [12.2],
        "catalog_aperture_subpix": [5],
    })[0]

    prior, fraction = pipeline_fit._catalog_prior_flux_jy(
        row,
        source,
        [],
        transform,
        (size, size),
        "ch2",
        catalog_frames=processed,
        catalog_wcs=transform,
        catalog_shape=(size, size),
    )

    assert fraction > 0.0
    np.testing.assert_allclose(prior, 0.002, rtol=5.0e-6)


@pytest.mark.parametrize(
    "widths",
    [(.5, 1.8, 1.8), (.5, 1.2, 1.2, 1.8), (.5, 1.8, 1.8, 1.2)],
)
@pytest.mark.parametrize("holes", [False, True])
def test_consensus_cr_clean_response_support_has_exact_empty_masks(widths, holes):
    size = 128
    cutouts = []
    for index, width in enumerate(widths):
        transform = _wcs(size, (index - 1) * 20.0 + 0.1, 0.1)
        x0, y0 = transform.world_to_pixel_values(
            config.TRANSIENT_RA, config.TRANSIENT_DEC,
        )
        cutout = _frame(transform, size, index)
        cutout["data"] = 0.001 + 0.002 * _integrated_gaussian(
            (size, size), x0, y0, width,
        )
        if holes:
            yy, xx = np.indices((size, size))
            radius = np.hypot(xx - np.rint(x0), yy - np.rint(y0))
            cutout["sigma"][(radius >= 5) & (radius < 6)] = np.inf
        cutouts.append(cutout)

    native_fit_campaign.apply_consensus_cr_masks(cutouts)

    assert [int(cutout["cr_mask"].sum()) for cutout in cutouts] == [0] * len(widths)


@pytest.mark.parametrize("sampling", ["same", "phase", "roll"])
@pytest.mark.parametrize(
    "amplitudes_sigma",
    [(200.0, 200.0), (200.0, 100.0), (2000.0, 1000.0)],
)
def test_consensus_cr_localizes_adjacent_artifacts(sampling, amplitudes_sigma):
    size = 31
    cutouts = []
    for index in range(3):
        transform = _wcs(size, 0.1 if index and sampling == "phase" else 0.0)
        if index and sampling == "roll":
            angle = np.deg2rad(15.0)
            transform.wcs.pc = [
                [np.cos(angle), -np.sin(angle)],
                [np.sin(angle), np.cos(angle)],
            ]
        cutouts.append(_frame(transform, size, index))
    cutouts[0]["data"][15, 15] += amplitudes_sigma[0] * 1.0e-6
    cutouts[0]["data"][15, 16] += amplitudes_sigma[1] * 1.0e-6

    native_fit_campaign.apply_consensus_cr_masks(cutouts)

    expected = np.zeros((size, size), dtype=bool)
    expected[15, 15:17] = True
    np.testing.assert_array_equal(cutouts[0]["cr_mask"], expected)
    assert [int(cutout["cr_mask"].sum()) for cutout in cutouts] == [2, 0, 0]


@pytest.mark.parametrize("phase", [0.0, 0.25, 0.5])
@pytest.mark.parametrize("amplitude_sigma", [0.0, 20.0, 80.0, 200.0, 2000.0])
def test_consensus_cr_localizes_offcenter_source_artifact(phase, amplitude_sigma):
    size = 25
    cutouts = []
    for index in range(3):
        transform = _wcs(size, phase if index else 0.0)
        x0, y0 = transform.world_to_pixel_values(
            config.TRANSIENT_RA, config.TRANSIENT_DEC,
        )
        cutout = _frame(transform, size, index)
        cutout["data"] += 0.002 * _integrated_gaussian(
            (size, size), x0, y0,
        )
        cutouts.append(cutout)
    cutouts[0]["data"][12, 13] += amplitude_sigma * 1.0e-6

    native_fit_campaign.apply_consensus_cr_masks(cutouts)

    expected = np.zeros((size, size), dtype=bool)
    if amplitude_sigma > 0.0:
        expected[12, 13] = True
    np.testing.assert_array_equal(cutouts[0]["cr_mask"], expected)
    assert [int(cutout["cr_mask"].sum()) for cutout in cutouts] == [
        int(amplitude_sigma > 0.0), 0, 0,
    ]
