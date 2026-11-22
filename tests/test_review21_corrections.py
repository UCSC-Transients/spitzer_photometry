"""Behavioral regressions for the independent review of PR #21."""
from __future__ import annotations

from math import erf

import numpy as np
import pytest
from astropy.coordinates import SkyCoord
from astropy.wcs import WCS

from src import config, native_fit_campaign as campaign, pipeline_fit, preprocessing, solver


def _wcs(n: int, *, scale_arcsec: float = 1.22, phase: float = 0.0, roll_deg: float = 0.0) -> WCS:
    wcs = WCS(naxis=2)
    wcs.wcs.crpix = [(n + 1) / 2 + phase, (n + 1) / 2 + phase]
    wcs.wcs.crval = [config.TRANSIENT_RA, config.TRANSIENT_DEC]
    wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    theta = np.deg2rad(roll_deg)
    wcs.wcs.cd = np.array([
        [-np.cos(theta), np.sin(theta)],
        [np.sin(theta), np.cos(theta)],
    ]) * scale_arcsec / 3600.0
    return wcs


def _pixel_integrated_gaussian(shape: tuple[int, int], x0: float, y0: float, sigma_px: float) -> np.ndarray:
    yy, xx = np.indices(shape, dtype=float)
    root_two_sigma = np.sqrt(2.0) * sigma_px
    int_x = 0.5 * np.vectorize(erf)((xx - x0 + 0.5) / root_two_sigma)
    int_x -= 0.5 * np.vectorize(erf)((xx - x0 - 0.5) / root_two_sigma)
    int_y = 0.5 * np.vectorize(erf)((yy - y0 + 0.5) / root_two_sigma)
    int_y -= 0.5 * np.vectorize(erf)((yy - y0 - 0.5) / root_two_sigma)
    return int_x * int_y


@pytest.mark.parametrize("sr", [1, 2, 4])
def test_catalog_to_native_fit_uses_integrated_jy_at_fixed_footprint(monkeypatch, sr):
    """Catalog aperture, renderer correction, prior, and fit share one Jy convention."""
    truth_jy = 0.002
    native_n = 15
    catalog_n = 65 * sr
    fit_n = 7 * sr
    reference = _wcs(65)
    catalog_scene = _wcs(catalog_n, scale_arcsec=config.PIXEL_SCALE / sr)
    source_ra, source_dec = reference.pixel_to_world_values(14.0, 32.0)
    source = SkyCoord(source_ra, source_dec, unit="deg")
    fit_scene = _wcs(fit_n, scale_arcsec=config.PIXEL_SCALE / sr)
    fit_scene.wcs.crval = [source.ra.deg, source.dec.deg]

    # Independent analytic template image.  Values retain Jy/native-pixel after
    # reprojection, so the integral needs the scene/native solid-angle factor.
    x_scene, y_scene = catalog_scene.world_to_pixel_values(source.ra.deg, source.dec.deg)
    response_scene = _pixel_integrated_gaussian(
        (catalog_n, catalog_n), float(x_scene), float(y_scene), 0.9 * sr,
    )
    response_scene /= response_scene.sum()
    area_factor = 1.0 / sr**2
    yy, xx = np.indices((catalog_n, catalog_n), dtype=float)
    center = (catalog_n - 1) / 2.0
    host = 1.0e-3 + 2.0e-3 * np.exp(-((xx - center) ** 2 + (yy - center) ** 2) / (8.0 * sr**2))
    template = host + truth_jy * response_scene / area_factor
    table = preprocessing.detect_sources(
        template,
        catalog_scene,
        is_template=True,
        mask_nucleus=True,
        integrated_jy_catalog=True,
    )
    assert table is not None and len(table) >= 1
    coordinates = SkyCoord(table["ra"], table["dec"], unit="deg")
    row = table[int(np.argmin(coordinates.separation(source).arcsec))]

    cutouts = []
    for i, phase in enumerate((0.0, 0.25, 0.5)):
        frame_wcs = _wcs(native_n, phase=phase)
        frame_wcs.wcs.crval = [source.ra.deg, source.dec.deg]
        response = solver.native_point_source_column(
            fit_scene,
            frame_wcs,
            (fit_n, fit_n),
            (native_n, native_n),
            "ch2",
            source.ra.deg,
            source.dec.deg,
        ).reshape(native_n, native_n)
        cutouts.append({
            "data": 1.0e-4 + truth_jy * response,
            "sigma": np.full((native_n, native_n), 2.0e-6),
            "wcs": frame_wcs,
            "raw_wcs": frame_wcs,
            "epoch_id": 0,
            "is_template": True,
            "filename": f"catalog_sr{sr}_ch2_{i}.fits",
            "mjd": float(i),
        })

    prior_jy, aperture_fraction = pipeline_fit._catalog_prior_flux_jy(
        row, source, cutouts, fit_scene, (fit_n, fit_n), "ch2",
    )
    assert 0.0 < aperture_fraction <= 1.01
    assert prior_jy == pytest.approx(truth_jy, rel=0.08)

    with campaign._temporary_config({
        "SUPERSAMPLE_FACTOR": sr,
        "USE_SCENE_GP_PRIOR": False,
        "SCENE_INDEPENDENT_RIDGE": 1.0e10,
        "ENFORCE_GP_CENTRAL_MONOTONICITY": False,
        "FLOAT_TRANSIENT_POSITION": False,
        "TRANSIENT_NONNEGATIVE": True,
        "USE_HOST_GAUSSIAN_CORE": False,
        "USE_NUCLEAR_POINT_SOURCE": False,
        "PRF_ORDER_PROJECT_THEN_CONVOLVE": True,
        "PRF_OPERATOR_MODE": "anchor",
        "PRF_SPATIAL_ANCHORS_PER_AXIS": 1,
    }):
        result = solver.run_gls_solve(
            cutouts, [source], [prior_jy], {}, (1.0, 1.0e-7),
            np.zeros((fit_n, fit_n)), fit_scene, 1,
        )
    assert result["star_fluxes"][0] == pytest.approx(truth_jy, rel=0.08)
    assert result["star_prior_fluxes"][0] == pytest.approx(prior_jy)
    assert result["star_prior_equivalent_exposures"] == pytest.approx(5.0)
    assert np.isfinite(result["star_prior_precision"][0])


def test_consensus_cr_abstains_on_phase_roll_but_recovers_injected_cr():
    n = 17
    truth_jy = 0.002
    noise_jy = 1.0e-6
    cutouts = []
    for i, (phase, roll) in enumerate(((0.0, 0.0), (0.5, 27.0), (0.25, 73.0), (0.75, 121.0))):
        wcs = _wcs(n, phase=phase, roll_deg=roll)
        x0, y0 = wcs.world_to_pixel_values(config.TRANSIENT_RA, config.TRANSIENT_DEC)
        response = _pixel_integrated_gaussian((n, n), float(x0), float(y0), 0.9)
        data = 1.0e-3 + truth_jy * response
        cutouts.append({
            "data": data,
            "sigma": np.full((n, n), noise_jy),
            "wcs": wcs,
            "raw_wcs": wcs,
            "epoch_id": 0,
            "is_template": True,
            "filename": f"phase_roll_ch2_{i}.fits",
            "mjd": i / 10.0,
        })
    injected = (4, 12)
    cutouts[0]["data"][injected] += 2.0e-4

    campaign.apply_consensus_cr_masks(cutouts)

    assert cutouts[0]["cr_mask"][injected]
    for entry in cutouts:
        x0, y0 = entry["wcs"].world_to_pixel_values(config.TRANSIENT_RA, config.TRANSIENT_DEC)
        yy, xx = np.indices((n, n))
        source_footprint = (xx - x0) ** 2 + (yy - y0) ** 2 <= 3.0**2
        assert not np.any(entry["cr_mask"] & source_footprint)
