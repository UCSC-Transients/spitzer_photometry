"""Behavioral regressions for the sixth independent review of PR #22."""
from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pytest
import sep
from astropy.coordinates import SkyCoord
from astropy.table import Table
from astropy.wcs import WCS
from scipy.special import erf

from src import (
    config,
    diagnostics,
    native_fit_campaign,
    pipeline_fit,
    preprocessing,
    run_manifest,
    solver,
)


def _wcs(
    size: int,
    phase_x: float = 0.0,
    phase_y: float | None = None,
    scale_arcsec: float = 1.22,
) -> WCS:
    phase_y = phase_x if phase_y is None else phase_y
    transform = WCS(naxis=2)
    transform.wcs.crpix = [
        (size + 1) / 2 + phase_x,
        (size + 1) / 2 + phase_y,
    ]
    transform.wcs.crval = [config.TRANSIENT_RA, config.TRANSIENT_DEC]
    transform.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    transform.wcs.cdelt = [-scale_arcsec / 3600.0, scale_arcsec / 3600.0]
    return transform


def _frame(transform: WCS, size: int, index: int) -> dict:
    return {
        "data": np.full((size, size), 1.0e-3),
        "sigma": np.full((size, size), 1.0e-6),
        "wcs": transform,
        "raw_wcs": transform,
        "epoch_id": 0,
        "is_template": True,
        "filename": f"review26_{index}_ch2.fits",
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


@pytest.mark.parametrize("phase", [0.0, 1.0e-4, 0.1, 0.25])
@pytest.mark.parametrize("noise", [False, True])
@pytest.mark.parametrize("peer_width", [1.2, 1.8])
def test_consensus_cr_preserves_perturbed_spatial_prf_controls(
    phase,
    noise,
    peer_width,
):
    size = 128
    generator = np.random.default_rng(42)
    frames = []
    for index, (shift, width) in enumerate(
        zip((-30.0, 0.0, 30.0), (0.5, peer_width, peer_width)),
    ):
        transform = _wcs(size, shift + phase, phase)
        x0, y0 = transform.world_to_pixel_values(
            config.TRANSIENT_RA, config.TRANSIENT_DEC,
        )
        entry = _frame(transform, size, index)
        entry["data"] += 0.002 * _integrated_gaussian(
            (size, size), x0, y0, width,
        )
        if noise:
            entry["data"] += generator.normal(0.0, 1.0e-6, (size, size))
        frames.append(entry)

    native_fit_campaign.apply_consensus_cr_masks(frames)

    assert [int(frame["cr_mask"].sum()) for frame in frames] == [0, 0, 0]


@pytest.mark.parametrize(
    "amplitudes_sigma",
    [(200.0, 200.0), (200.0, 100.0), (2000.0, 1000.0)],
)
def test_consensus_cr_recovers_every_adjacent_artifact_pixel(amplitudes_sigma):
    frames = [_frame(_wcs(31), 31, index) for index in range(3)]
    frames[0]["data"][15, 15] += amplitudes_sigma[0] * 1.0e-6
    frames[0]["data"][15, 16] += amplitudes_sigma[1] * 1.0e-6

    native_fit_campaign.apply_consensus_cr_masks(frames)

    assert frames[0]["cr_mask"][15, 15]
    assert frames[0]["cr_mask"][15, 16]
    assert [int(frame["cr_mask"].sum()) for frame in frames] == [2, 0, 0]


@pytest.mark.parametrize("sampling", ["phase", "roll"])
@pytest.mark.parametrize("amplitude_sigma", [20.0, 80.0, 200.0, 2000.0])
def test_consensus_cr_retains_source_cr_sensitivity(sampling, amplitude_sigma):
    size = 25
    frames = []
    for index in range(3):
        transform = _wcs(size, 0.1 if index and sampling == "phase" else 0.0)
        if index and sampling == "roll":
            angle = np.deg2rad(15.0)
            transform.wcs.pc = [
                [np.cos(angle), -np.sin(angle)],
                [np.sin(angle), np.cos(angle)],
            ]
        x0, y0 = transform.world_to_pixel_values(
            config.TRANSIENT_RA, config.TRANSIENT_DEC,
        )
        entry = _frame(transform, size, index)
        entry["data"] += 0.002 * _integrated_gaussian((size, size), x0, y0)
        frames.append(entry)
    frames[0]["data"][12, 12] += amplitude_sigma * 1.0e-6

    native_fit_campaign.apply_consensus_cr_masks(frames)

    assert frames[0]["cr_mask"][12, 12]
    assert [int(frame["cr_mask"].sum()) for frame in frames] == [1, 0, 0]


def _catalog_row(image: np.ndarray, support: np.ndarray, transform: WCS, source):
    x, y = transform.world_to_pixel_values(source.ra.deg, source.dec.deg)
    scale = 3600.0 * np.sqrt(abs(np.linalg.det(transform.pixel_scale_matrix)))
    radius = 4.88 / scale
    flux = sep.sum_circle(
        np.ascontiguousarray(image, dtype=np.float64),
        np.asarray([x]),
        np.asarray([y]),
        radius,
        mask=~np.asarray(support, dtype=bool),
        bkgann=(1.5 * radius, 2.5 * radius),
        subpix=5,
    )[0][0] * preprocessing._scene_sample_to_integrated_jy_factor(transform)
    return Table({
        "x": [x],
        "y": [y],
        "ra": [source.ra.deg],
        "dec": [source.dec.deg],
        "flux": [flux],
        "catalog_aperture_flux_jy": [flux],
        "catalog_aperture_radius_arcsec": [4.88],
        "catalog_aperture_annulus_inner_arcsec": [7.32],
        "catalog_aperture_annulus_outer_arcsec": [12.2],
        "catalog_aperture_subpix": [5],
    })[0]


def _catalog_frames(
    monkeypatch,
    *,
    sr: int,
    gradient: float,
    noise: bool = False,
    mask_mode: str = "none",
    staggered_cr: bool = False,
):
    truth = 0.002
    native_size = 49
    source = SkyCoord(config.TRANSIENT_RA, config.TRANSIENT_DEC, unit="deg")
    phases = (-0.2, 0.0, 80.0 if mask_mode == "off_footprint" else 0.23)
    native_wcs = [_wcs(native_size, phase, -phase / 2.0) for phase in phases]
    detector_x = np.asarray([
        float(np.asarray(w.world_to_pixel_values(source.ra.deg, source.dec.deg)[0]).ravel()[0])
        for w in native_wcs
    ])

    def supplied_prf(_channel, x_det=None, y_det=None):
        del y_det
        index = int(np.argmin(np.abs(detector_x - float(x_det))))
        yy, xx = np.mgrid[-100:101, -100:101] / 20.0
        width = (0.65, 0.9, 1.25)[index]
        response = np.exp(-(xx**2 + yy**2) / (2.0 * width**2))
        return response / response.sum()

    monkeypatch.setattr(config, "PRF_OVERSAMPLE", 20)
    monkeypatch.setattr(config, "PRF_PIXEL_SCALE_ARCSEC", 1.22 / 20.0)
    monkeypatch.setattr(config, "PRF_PREBLUR_IN_OVERSAMPLE_PIXELS", 0.0)
    monkeypatch.setattr(config, "PRF_APPLY_PMAP_GAIN", False)
    monkeypatch.setattr(solver, "load_prf", supplied_prf)
    catalog_size = 49 * sr
    catalog_wcs = _wcs(catalog_size, scale_arcsec=1.22 / sr)
    responses = [
        solver.native_point_source_column(
            catalog_wcs,
            transform,
            (catalog_size, catalog_size),
            (native_size, native_size),
            "ch2",
            source.ra.deg,
            source.dec.deg,
        ).reshape(native_size, native_size)
        for transform in native_wcs
    ]
    generator = np.random.default_rng(42)
    raw_frames = []
    yy, xx = np.indices((native_size, native_size), dtype=float)
    for index, (transform, response) in enumerate(zip(native_wcs, responses)):
        x0, y0 = transform.world_to_pixel_values(source.ra.deg, source.dec.deg)
        data = (0.001, 0.003, 0.002)[index] + truth * response
        data += gradient * (xx - float(np.asarray(x0).ravel()[0]))
        if noise:
            data += generator.normal(0.0, 1.0e-6, data.shape)
        sigma = np.full(data.shape, 1.0e-6)
        if mask_mode == "full" and index == 2:
            sigma[:] = np.inf
        elif mask_mode == "partial" and index == 2:
            sigma[:, :8] = np.inf
        if staggered_cr:
            dy, dx = ((0, 2), (2, 0), (0, -2))[index]
            cr_y = int(round(float(y0))) + dy
            cr_x = int(round(float(x0))) + dx
            if 0 <= cr_y < native_size and 0 <= cr_x < native_size:
                data[cr_y, cr_x] += 200.0e-6
        raw_frames.append({
            "data": data / config.MJY_SR_TO_JY,
            "sigma": sigma / config.MJY_SR_TO_JY,
            "wcs": transform,
            "filename": f"review26_catalog_{index}_ch2.fits",
            "mjd": float(index),
            "is_full_array": False,
        })
    iterator = iter(raw_frames)
    monkeypatch.setattr(preprocessing, "load_data", lambda _entry: next(iterator))
    processed = preprocessing.reproject_to_grid(
        [{} for _ in raw_frames], catalog_wcs, (catalog_size, catalog_size),
    )
    median, _ = preprocessing.create_median_stack(
        np.asarray([entry["data"] for entry in processed]),
    )
    support = np.any(np.asarray([
        entry["projected_valid_mask"] for entry in processed
    ]), axis=0)
    row = _catalog_row(median, support, catalog_wcs, source)
    return truth, source, row, processed, catalog_wcs


@pytest.mark.parametrize("sr", [1, 2, 4])
@pytest.mark.parametrize("gradient", [-1.0e-5, 1.0e-5])
@pytest.mark.parametrize("noise", [False, True])
def test_catalog_affine_background_is_preserved_through_joint_fit(
    monkeypatch,
    sr,
    gradient,
    noise,
):
    truth, source, row, frames, catalog_wcs = _catalog_frames(
        monkeypatch,
        sr=sr,
        gradient=gradient,
        noise=noise,
        mask_mode="partial",
    )
    prior, fraction = pipeline_fit._catalog_prior_flux_jy(
        row,
        source,
        [],
        catalog_wcs,
        (49 * sr, 49 * sr),
        "ch2",
        catalog_frames=frames,
        catalog_wcs=catalog_wcs,
        catalog_shape=(49 * sr, 49 * sr),
    )
    assert fraction > 0.0
    np.testing.assert_allclose(
        prior,
        truth,
        rtol=5.0e-3 if not noise else 0.0,
        atol=0.0 if not noise else 3.0e-5,
    )

    fit_wcs = _wcs(9)
    response = solver.native_point_source_column(
        fit_wcs,
        fit_wcs,
        (9, 9),
        (9, 9),
        "ch2",
        source.ra.deg,
        source.dec.deg,
    ).reshape(9, 9)
    fit_frame = _frame(fit_wcs, 9, 0)
    fit_frame["data"] += truth * response
    with native_fit_campaign._temporary_config({
        "USE_SCENE_GP_PRIOR": False,
        "SCENE_INDEPENDENT_RIDGE": 1.0e10,
        "ENFORCE_GP_CENTRAL_MONOTONICITY": False,
        "FLOAT_TRANSIENT_POSITION": False,
        "TRANSIENT_NONNEGATIVE": False,
        "USE_HOST_GAUSSIAN_CORE": False,
        "USE_NUCLEAR_POINT_SOURCE": False,
        "PRF_ORDER_PROJECT_THEN_CONVOLVE": True,
        "PRF_OPERATOR_MODE": "anchor",
        "PRF_SPATIAL_ANCHORS_PER_AXIS": 1,
        "CATALOG_PRIOR_EQUIVALENT_EXPOSURES": 5.0,
    }):
        result = solver.run_gls_solve(
            [fit_frame],
            [source],
            [prior],
            {},
            (1.0, 1.0e-7),
            np.zeros((9, 9)),
            fit_wcs,
            1,
        )
    np.testing.assert_allclose(
        result["star_fluxes"],
        [truth],
        rtol=5.0e-3 if not noise else 0.0,
        atol=0.0 if not noise else 3.0e-5,
    )
    assert result["star_prior_equivalent_exposures"] == pytest.approx(5.0)


@pytest.mark.parametrize("mask_mode", ["full", "partial", "off_footprint"])
def test_catalog_skips_unusable_exposures_but_retains_supported_ones(
    monkeypatch,
    mask_mode,
):
    truth, source, row, frames, catalog_wcs = _catalog_frames(
        monkeypatch,
        sr=1,
        gradient=0.0,
        mask_mode=mask_mode,
        staggered_cr=True,
    )
    prior, fraction = pipeline_fit._catalog_prior_flux_jy(
        row,
        source,
        [],
        catalog_wcs,
        (49, 49),
        "ch2",
        catalog_frames=frames,
        catalog_wcs=catalog_wcs,
        catalog_shape=(49, 49),
    )
    assert fraction > 0.0
    np.testing.assert_allclose(prior, truth, rtol=5.0e-3)


def test_catalog_reports_all_unsupported_exposures(monkeypatch):
    _, source, row, frames, catalog_wcs = _catalog_frames(
        monkeypatch,
        sr=1,
        gradient=0.0,
    )
    for frame in frames:
        frame["native_valid_mask"][:] = False
        frame["projected_valid_mask"][:] = False
        frame["sigma"][:] = np.inf
    with pytest.raises(ValueError, match="supported"):
        pipeline_fit._catalog_prior_flux_jy(
            row,
            source,
            [],
            catalog_wcs,
            (49, 49),
            "ch2",
            catalog_frames=frames,
            catalog_wcs=catalog_wcs,
            catalog_shape=(49, 49),
        )


def test_native_pipeline_reaches_joint_fit_with_one_unusable_catalog_frame(
    monkeypatch,
    tmp_path,
):
    truth, source, row, processed, mosaic_wcs = _catalog_frames(
        monkeypatch,
        sr=1,
        gradient=0.0,
        mask_mode="full",
    )
    table = Table(rows=[row])
    fit_wcs = _wcs(9)
    response = solver.native_point_source_column(
        fit_wcs,
        fit_wcs,
        (9, 9),
        (9, 9),
        "ch2",
        source.ra.deg,
        source.dec.deg,
    ).reshape(9, 9)
    cutout = _frame(fit_wcs, 9, 0)
    cutout["data"] += truth * response
    files = [
        {"image": f"review26_{index}.fits", "filename": f"review26_{index}_ch2.fits", "mjd": index}
        for index in range(3)
    ]
    monkeypatch.setattr(config, "OUTPUT_DIR", str(tmp_path / "output"))
    monkeypatch.setattr(config, "DIAGNOSTIC_DIR", str(tmp_path / "diagnostics"))
    monkeypatch.setattr(config, "FIT_ON_NATIVE_PIXELS", True)
    monkeypatch.setattr(config, "NATIVE_CUTOUT_SIZE", 0)
    monkeypatch.setattr(config, "GP_OPTIMIZE_HYPERPARAMS", False)
    monkeypatch.setattr(config, "UNMASK_SIGMA_INF_RADIUS_PX", 0.0)
    monkeypatch.setattr(preprocessing, "find_spitzer_files", lambda *_args: files)
    monkeypatch.setattr(preprocessing, "get_or_create_source_catalog", lambda *_args: {})
    monkeypatch.setattr(preprocessing, "categorize_observations", lambda *_args: ([], files))
    monkeypatch.setattr(preprocessing, "define_mosaic_wcs", lambda *_args: (mosaic_wcs, (49, 49)))
    monkeypatch.setattr(preprocessing, "reproject_to_grid", lambda *_args: processed)
    monkeypatch.setattr(
        preprocessing,
        "create_median_stack",
        lambda cube: (np.nanmedian(cube, axis=0), np.zeros(cube.shape[1:])),
    )
    monkeypatch.setattr(preprocessing, "update_catalog_with_template", lambda *_args, **_kwargs: table)
    monkeypatch.setattr(preprocessing, "align_frames_to_template", lambda *_args: None)
    monkeypatch.setattr(preprocessing, "extract_native_analysis_cutouts", lambda *_args: ([cutout], fit_wcs))
    monkeypatch.setattr(native_fit_campaign, "_scene_wcs_from_bcd_footprints", lambda *_args: (fit_wcs, (9, 9)))
    monkeypatch.setattr(native_fit_campaign, "apply_consensus_cr_masks", lambda *_args: None)
    monkeypatch.setattr(native_fit_campaign, "unmask_sigma_inf_in_radius", lambda *_args: None)
    monkeypatch.setattr(pipeline_fit.fits, "writeto", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(diagnostics, "diagnostic_stretch_mask", lambda *_args: np.ones((9, 9), dtype=bool))
    monkeypatch.setattr(run_manifest, "write_run_manifest", lambda *_args, **_kwargs: None)
    captured = {}

    def solve(_cutouts, stars, initial_fluxes, *_args, **_kwargs):
        captured["stars"] = stars
        captured["initial_fluxes"] = initial_fluxes
        return {"star_fluxes": np.asarray(initial_fluxes, dtype=float)}

    monkeypatch.setattr(solver, "run_gls_solve", solve)
    result = pipeline_fit.run_pipeline_fit_core(skip_pre_analysis_check=True)
    assert result is not None and len(captured["stars"]) == 1
    np.testing.assert_allclose(captured["initial_fluxes"], [truth], rtol=5.0e-3)
