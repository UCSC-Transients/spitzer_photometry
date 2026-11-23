"""Behavioral regressions for the fourth independent review of PR #22."""
from __future__ import annotations

from unittest.mock import patch

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
    *,
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
        "filename": f"review24_{index}_ch2.fits",
        "mjd": index / 10.0,
    }


def _integrated_gaussian(shape, x0: float, y0: float, sigma: float = 0.9):
    yy, xx = np.indices(shape, dtype=float)
    scale = np.sqrt(2.0) * sigma
    x_integral = 0.5 * (
        erf((xx - x0 + 0.5) / scale) - erf((xx - x0 - 0.5) / scale)
    )
    y_integral = 0.5 * (
        erf((yy - y0 + 0.5) / scale) - erf((yy - y0 - 0.5) / scale)
    )
    return x_integral * y_integral


@pytest.mark.parametrize("amplitude_sigma", [20.0, 80.0, 200.0, 2000.0])
def test_consensus_cr_recovers_cr_on_positive_static_source(amplitude_sigma):
    size = 25
    frames = []
    for index in range(3):
        transform = _wcs(size)
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


def test_consensus_cr_handles_target_and_peer_annulus_contamination_symmetrically():
    frames = [_frame(_wcs(31), 31, index) for index in range(3)]
    frames[0]["data"][15, 15] += 200.0e-6
    frames[1]["data"][15, 23] += 1000.0e-6

    native_fit_campaign.apply_consensus_cr_masks(frames)

    assert frames[0]["cr_mask"][15, 15]
    assert frames[1]["cr_mask"][15, 23]
    assert [int(frame["cr_mask"].sum()) for frame in frames] == [1, 1, 0]


def test_consensus_cr_does_not_treat_negative_source_or_flat_plateau_as_positive():
    size = 25
    for source_flux in (-0.002, 0.0):
        frames = []
        for index, phase in enumerate((0.0, 0.5, 0.5)):
            transform = _wcs(size, phase_x=phase)
            x0, y0 = transform.world_to_pixel_values(
                config.TRANSIENT_RA, config.TRANSIENT_DEC,
            )
            entry = _frame(transform, size, index)
            entry["data"] += source_flux * _integrated_gaussian(
                (size, size), x0, y0,
            )
            frames.append(entry)
        frames[0]["data"][12, 12] += 2000.0e-6
        native_fit_campaign.apply_consensus_cr_masks(frames)
        assert frames[0]["cr_mask"][12, 12]


def test_consensus_cr_preserves_combined_gradient_noise_controls():
    size = 25
    yy, xx = np.indices((size, size), dtype=float)
    for gradient in (-3.0e-4, 3.0e-4, -1.0e-4, 1.0e-4):
        for seed in range(40):
            generator = np.random.default_rng(seed)
            frames = []
            for index, phase in enumerate((0.0, 0.5, 0.5)):
                transform = _wcs(size, phase_x=phase)
                x0, y0 = transform.world_to_pixel_values(
                    config.TRANSIENT_RA, config.TRANSIENT_DEC,
                )
                entry = _frame(transform, size, index)
                entry["data"] = (
                    0.01
                    + 0.002 * _integrated_gaussian((size, size), x0, y0)
                    + gradient * (xx - x0)
                    + generator.normal(0.0, 1.0e-6, (size, size))
                )
                frames.append(entry)
            native_fit_campaign.apply_consensus_cr_masks(frames)
            assert [int(frame["cr_mask"].sum()) for frame in frames] == [0, 0, 0]


def _catalog_row(image, support, transform, source):
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


@pytest.mark.parametrize("sr", [1, 2, 4])
def test_catalog_per_frame_calibration_handles_backgrounds_and_detector_prfs(
    monkeypatch,
    sr,
):
    truth = 0.002
    native_size = 256
    native_wcs = [
        _wcs(native_size, phase_x=offset) for offset in (-60.0, 0.0, 60.0)
    ]
    source = SkyCoord(config.TRANSIENT_RA, config.TRANSIENT_DEC, unit="deg")
    detector_x = np.asarray([
        transform.world_to_pixel_values(source.ra.deg, source.dec.deg)[0]
        for transform in native_wcs
    ], dtype=float)

    def supplied_prf(_channel, x_det=None, y_det=None):
        index = int(np.argmin(np.abs(detector_x - float(x_det))))
        yy, xx = np.mgrid[-100:101, -100:101] / 20.0
        width = (0.5, 0.9, 1.8)[index]
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
            is_full_array=True,
        ).reshape(native_size, native_size)
        for transform in native_wcs
    ]
    backgrounds = (0.001, 0.003, 0.002)
    raw_frames = [{
        "data": (backgrounds[index] + truth * responses[index]) / config.MJY_SR_TO_JY,
        "sigma": np.full((native_size, native_size), 1.0e-6) / config.MJY_SR_TO_JY,
        "wcs": transform,
        "filename": f"catalog_background_{index}_ch2.fits",
        "mjd": float(index),
        "is_full_array": True,
    } for index, transform in enumerate(native_wcs)]
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

    prior, fraction = pipeline_fit._catalog_prior_flux_jy(
        row,
        source,
        [],
        catalog_wcs,
        (catalog_size, catalog_size),
        "ch2",
        catalog_frames=processed,
        catalog_wcs=catalog_wcs,
        catalog_shape=(catalog_size, catalog_size),
    )
    assert fraction > 0.0
    np.testing.assert_allclose(prior, truth, rtol=2.0e-3, atol=0.0)


def _clear_projection_caches():
    solver._PROJECTION_CACHE.clear()
    solver._WCS_PROVENANCE_CACHE.clear()


def _clear_operator_caches():
    _clear_projection_caches()
    solver._PRF_OPERATOR_BUNDLE_CACHE.clear()
    solver._PRF_NATIVE_OPERATOR_BUNDLE_CACHE.clear()
    solver._PRF_EXACT_OPERATOR_CACHE.clear()
    solver._prf_cache_fingerprint_cached.cache_clear()


@pytest.mark.parametrize("mutated", ["scene", "native"])
@pytest.mark.parametrize("change", ["scale", "rotation"])
def test_projection_cache_cd_mutation_warm_equals_cold(mutated, change):
    size = 9
    scene = _wcs(size)
    native = _wcs(size)
    scene.wcs.cd = np.diag(scene.wcs.cdelt)
    native.wcs.cd = np.diag(native.wcs.cdelt)
    image = np.arange(size * size, dtype=float).reshape(size, size) * 1.0e-5
    _clear_projection_caches()
    before = solver._project_scene_to_native(image, scene, native, (size, size))
    transform = scene if mutated == "scene" else native
    cd = np.asarray(transform.wcs.cd, dtype=float).copy()
    if change == "scale":
        cd[0, 0] *= 1.1
    else:
        angle = np.deg2rad(7.0)
        rotation = np.asarray([
            [np.cos(angle), -np.sin(angle)],
            [np.sin(angle), np.cos(angle)],
        ])
        cd = rotation @ cd
    transform.wcs.cd = cd
    transform.wcs.set()
    warm = solver._project_scene_to_native(image, scene, native, (size, size))
    _clear_projection_caches()
    cold = solver._project_scene_to_native(image, scene, native, (size, size))
    assert not np.allclose(before, cold, rtol=0.0, atol=1.0e-12)
    np.testing.assert_allclose(warm, cold, rtol=0.0, atol=0.0)


@pytest.mark.parametrize(
    ("mode", "project_then_convolve"),
    [("anchor", False), ("exact", False), ("anchor", True)],
)
@pytest.mark.parametrize("change", ["scale", "rotation"])
def test_attached_detector_cd_mutation_warm_equals_cold(
    monkeypatch,
    mode,
    project_then_convolve,
    change,
):
    size = 7
    cutout = _wcs(size)
    detector = _wcs(256)
    detector.wcs.crval[0] += 0.02
    detector.wcs.cd = np.diag(detector.wcs.cdelt)
    cutout._detector_wcs = detector

    def position_prf(_channel, x_det=None, y_det=None):
        yy, xx = np.mgrid[-100:101, -100:101] / 20.0
        width = 0.5 + 0.001 * (float(x_det) + 2.0 * float(y_det))
        response = np.exp(-(xx**2 + yy**2) / (2.0 * width**2))
        return response / response.sum()

    monkeypatch.setattr(config, "PRF_OVERSAMPLE", 20)
    monkeypatch.setattr(config, "PRF_OPERATOR_MODE", mode)
    monkeypatch.setattr(config, "PRF_ORDER_PROJECT_THEN_CONVOLVE", project_then_convolve)
    monkeypatch.setattr(config, "PRF_SPATIAL_ANCHORS_PER_AXIS", 1)
    monkeypatch.setattr(solver, "load_prf", position_prf)
    scene = np.zeros((size, size))
    scene[size // 2, size // 2] = 0.002
    _clear_operator_caches()
    before = solver._apply_frame_forward_operator(
        scene, cutout, cutout, (size, size), (size, size), "ch2",
    )
    cd = np.asarray(detector.wcs.cd, dtype=float).copy()
    if change == "scale":
        cd[0, 0] *= 1.1
    else:
        angle = np.deg2rad(7.0)
        rotation = np.asarray([
            [np.cos(angle), -np.sin(angle)],
            [np.sin(angle), np.cos(angle)],
        ])
        cd = rotation @ cd
    detector.wcs.cd = cd
    detector.wcs.set()
    warm = solver._apply_frame_forward_operator(
        scene, cutout, cutout, (size, size), (size, size), "ch2",
    )
    _clear_operator_caches()
    cold = solver._apply_frame_forward_operator(
        scene, cutout, cutout, (size, size), (size, size), "ch2",
    )
    assert not np.allclose(before, cold, rtol=0.0, atol=1.0e-12)
    np.testing.assert_allclose(warm, cold, rtol=0.0, atol=0.0)


@pytest.mark.parametrize(
    ("native_shape", "scene_shape", "stars"),
    [((100, 100), (3, 3), 50), ((40, 60), (5, 7), 17), ((15, 21), (9, 11), 3)],
)
def test_memory_estimate_counts_live_native_star_designs(
    monkeypatch,
    native_shape,
    scene_shape,
    stars,
):
    transform = _wcs(max(native_shape)).slice((
        slice(0, native_shape[0]), slice(0, native_shape[1]),
    ))
    frame = {
        "data": np.full(native_shape, 1.0e-3),
        "sigma": np.full(native_shape, 1.0e-6),
        "wcs": transform,
        "raw_wcs": transform,
        "epoch_id": 0,
        "is_template": True,
        "filename": "memory_ratio_ch2.fits",
        "mjd": 0.0,
    }
    monkeypatch.setattr(config, "PRF_OPERATOR_MODE", "anchor")
    monkeypatch.setattr(config, "PRF_ORDER_PROJECT_THEN_CONVOLVE", True)
    monkeypatch.setattr(config, "PRF_SPATIAL_ANCHORS_PER_AXIS", 1)
    estimate = solver._estimate_solver_peak_memory_bytes(
        [frame], scene_shape, n_scene_blocks=1, n_stars=stars, n_science_epochs=0,
    )
    native_pixels = int(np.prod(native_shape))
    expected = 8 * native_pixels * stars
    assert estimate["native_source_columns_bytes"] == expected
    assert estimate["native_star_stack_bytes"] == expected
    assert estimate["native_weighted_star_stack_bytes"] == expected
    assert estimate["source_assembly_workspace_bytes"] >= 3 * expected


def test_memory_guard_rejects_native_star_design_before_column_allocation(monkeypatch):
    native_wcs = _wcs(100)
    scene_wcs = _wcs(3)
    frame = _frame(native_wcs, 100, 0)
    stars = [
        SkyCoord(
            *native_wcs.pixel_to_world_values(
                float(10 + 9 * (index % 10)),
                float(20 + 12 * (index // 10)),
            ),
            unit="deg",
        )
        for index in range(50)
    ]
    monkeypatch.setattr(config, "PRF_OPERATOR_MODE", "anchor")
    monkeypatch.setattr(config, "PRF_ORDER_PROJECT_THEN_CONVOLVE", True)
    monkeypatch.setattr(config, "PRF_SPATIAL_ANCHORS_PER_AXIS", 1)
    monkeypatch.setattr(config, "SOLVER_DENSE_MEMORY_LIMIT_BYTES", 2_193_769)
    monkeypatch.setattr(
        solver,
        "native_point_source_column",
        lambda *args, **kwargs: pytest.fail("guard ran after native star allocation"),
    )
    with pytest.raises(MemoryError, match="Dense fit peak"):
        solver.run_gls_solve(
            [frame],
            stars,
            np.zeros(50),
            {},
            (1.0, 1.0e-7),
            np.zeros((3, 3)),
            scene_wcs,
            1,
        )
