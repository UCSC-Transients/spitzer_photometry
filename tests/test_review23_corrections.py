"""Behavioral regressions for the third independent review of PR #22."""
from __future__ import annotations

import numpy as np
import pytest
import sep
from astropy.coordinates import SkyCoord
from astropy.table import Table
from astropy.wcs import DistortionLookupTable, WCS
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


def _wcs(n: int, *, scale_arcsec: float = 1.22, phase: float = 0.0) -> WCS:
    wcs = WCS(naxis=2)
    wcs.wcs.crpix = [(n + 1) / 2 + phase, (n + 1) / 2 + phase]
    wcs.wcs.crval = [config.TRANSIENT_RA, config.TRANSIENT_DEC]
    wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    wcs.wcs.cdelt = [-scale_arcsec / 3600.0, scale_arcsec / 3600.0]
    return wcs


def _gaussian_prf(oversample: int = 20) -> np.ndarray:
    yy, xx = np.mgrid[-100:101, -100:101] / float(oversample)
    prf = np.exp(-(xx**2 + yy**2) / (2.0 * 0.9**2))
    return prf / prf.sum()


def _static_cr_frames(*, gradient: float = 0.0, seed: int | None = None) -> list[dict]:
    frames = []
    size = 15
    yy, xx = np.indices((size, size), dtype=float)
    generator = np.random.default_rng(seed)
    for index, phase in enumerate((0.0, 0.5, 0.5)):
        wcs = _wcs(size, phase=phase)
        x0, y0 = wcs.world_to_pixel_values(config.TRANSIENT_RA, config.TRANSIENT_DEC)
        scale = np.sqrt(2.0) * 0.9
        x_integral = 0.5 * (
            erf((xx - x0 + 0.5) / scale) - erf((xx - x0 - 0.5) / scale)
        )
        y_integral = 0.5 * (
            erf((yy - y0 + 0.5) / scale) - erf((yy - y0 - 0.5) / scale)
        )
        data = 1.0e-3 + 2.0e-3 * x_integral * y_integral + gradient * (xx - x0)
        if seed is not None:
            data = data + generator.normal(0.0, 1.0e-6, data.shape)
        frames.append({
            "data": data,
            "sigma": np.full(data.shape, 1.0e-6),
            "wcs": wcs,
            "raw_wcs": wcs,
            "epoch_id": 0,
            "is_template": True,
            "filename": f"static_{index}_ch2.fits",
            "mjd": index / 10.0,
        })
    return frames


def _catalog_row(
    image: np.ndarray,
    valid: np.ndarray,
    wcs: WCS,
    source: SkyCoord,
):
    x, y = wcs.world_to_pixel_values(source.ra.deg, source.dec.deg)
    scale_arcsec = 3600.0 * np.sqrt(abs(np.linalg.det(wcs.pixel_scale_matrix)))
    radius = 4.88 / scale_arcsec
    flux, _, flags = sep.sum_circle(
        np.ascontiguousarray(image, dtype=np.float64),
        np.asarray([x]),
        np.asarray([y]),
        radius,
        mask=~np.asarray(valid, dtype=bool),
        bkgann=(1.5 * radius, 2.5 * radius),
        subpix=5,
    )
    assert np.isfinite(flux[0])
    area = preprocessing._scene_sample_to_integrated_jy_factor(wcs)
    aperture_flux = float(flux[0]) * area
    return Table({
        "x": [x],
        "y": [y],
        "ra": [source.ra.deg],
        "dec": [source.dec.deg],
        "flux": [aperture_flux],
        "catalog_aperture_flux_jy": [aperture_flux],
        "catalog_aperture_radius_arcsec": [4.88],
        "catalog_aperture_annulus_inner_arcsec": [7.32],
        "catalog_aperture_annulus_outer_arcsec": [12.2],
        "catalog_aperture_subpix": [5],
    })[0]


def _catalog_measurement(
    monkeypatch,
    *,
    sr: int,
    core_holes: tuple[bool, ...],
):
    truth_jy = 0.002
    native_n = 49
    native_wcs = _wcs(native_n, phase=0.23)
    source = SkyCoord(*native_wcs.pixel_to_world_values(12.0, 24.0), unit="deg")
    response = solver.native_point_source_column(
        native_wcs,
        native_wcs,
        (native_n, native_n),
        (native_n, native_n),
        "ch2",
        source.ra.deg,
        source.dec.deg,
    ).reshape(native_n, native_n)
    raw_frames = []
    for index, hole in enumerate(core_holes):
        sigma = np.full((native_n, native_n), 1.0e-6)
        if hole:
            sigma[24, 12] = np.inf
        raw_frames.append({
            "data": (1.0e-3 + truth_jy * response) / config.MJY_SR_TO_JY,
            "sigma": sigma / config.MJY_SR_TO_JY,
            "wcs": native_wcs,
            "filename": f"catalog_{index}_ch2.fits",
            "mjd": float(index),
            "is_full_array": False,
        })
    catalog_n = native_n * sr
    catalog_wcs = _wcs(catalog_n, scale_arcsec=1.22 / sr, phase=0.23 * sr)
    iterator = iter(raw_frames)
    monkeypatch.setattr(preprocessing, "load_data", lambda _entry: next(iterator))
    processed = preprocessing.reproject_to_grid(
        [{} for _ in raw_frames], catalog_wcs, (catalog_n, catalog_n),
    )
    median, _ = preprocessing.create_median_stack(
        np.asarray([entry["data"] for entry in processed]),
    )
    support = np.any(np.asarray([
        entry["projected_valid_mask"] for entry in processed
    ]), axis=0)
    row = _catalog_row(median, support, catalog_wcs, source)
    return truth_jy, source, row, processed, catalog_wcs


def test_catalog_masked_core_uses_identical_measurement_and_response_operator(monkeypatch):
    monkeypatch.setattr(config, "PRF_OVERSAMPLE", 20)
    monkeypatch.setattr(solver, "load_prf", lambda *args, **kwargs: _gaussian_prf())
    truth, source, row, frames, catalog_wcs = _catalog_measurement(
        monkeypatch, sr=1, core_holes=(True,),
    )
    prior, fraction = pipeline_fit._catalog_prior_flux_jy(
        row,
        source,
        [],
        catalog_wcs,
        catalog_wcs.array_shape or (49, 49),
        "ch2",
        catalog_frames=frames,
        catalog_wcs=catalog_wcs,
        catalog_shape=(49, 49),
    )
    assert fraction > 0.0
    np.testing.assert_allclose(prior, truth, rtol=2.0e-3, atol=0.0)

    fit_wcs = _wcs(9)
    fit_wcs.wcs.crval = [source.ra.deg, source.dec.deg]
    fit_response = solver.native_point_source_column(
        fit_wcs,
        fit_wcs,
        (9, 9),
        (9, 9),
        "ch2",
        source.ra.deg,
        source.dec.deg,
    ).reshape(9, 9)
    fit_frame = {
        "data": 1.0e-3 + truth * fit_response,
        "sigma": np.full((9, 9), 1.0e-6),
        "wcs": fit_wcs,
        "raw_wcs": fit_wcs,
        "epoch_id": 0,
        "is_template": True,
        "filename": "masked_core_fit_ch2.fits",
        "mjd": 0.0,
    }
    with native_fit_campaign._temporary_config({
        "USE_SCENE_GP_PRIOR": False,
        "SCENE_INDEPENDENT_RIDGE": 1.0e10,
        "ENFORCE_GP_CENTRAL_MONOTONICITY": False,
        "FLOAT_TRANSIENT_POSITION": False,
        "USE_HOST_GAUSSIAN_CORE": False,
        "USE_NUCLEAR_POINT_SOURCE": False,
        "PRF_ORDER_PROJECT_THEN_CONVOLVE": True,
        "PRF_OPERATOR_MODE": "anchor",
        "PRF_SPATIAL_ANCHORS_PER_AXIS": 1,
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
    np.testing.assert_allclose(result["star_fluxes"], [truth], rtol=2.0e-3)


@pytest.mark.parametrize("sr", [1, 2, 4])
def test_catalog_actual_native_reprojection_is_flux_invariant(monkeypatch, sr):
    monkeypatch.setattr(config, "PRF_OVERSAMPLE", 20)
    monkeypatch.setattr(solver, "load_prf", lambda *args, **kwargs: _gaussian_prf())
    truth, source, row, frames, catalog_wcs = _catalog_measurement(
        monkeypatch, sr=sr, core_holes=(False,),
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
    np.testing.assert_allclose(prior, truth, rtol=2.0e-3, atol=0.0)


def test_catalog_per_frame_hole_uses_same_median_stack_support(monkeypatch):
    monkeypatch.setattr(config, "PRF_OVERSAMPLE", 20)
    monkeypatch.setattr(solver, "load_prf", lambda *args, **kwargs: _gaussian_prf())
    truth, source, row, frames, catalog_wcs = _catalog_measurement(
        monkeypatch, sr=2, core_holes=(True, False, False),
    )
    prior, _ = pipeline_fit._catalog_prior_flux_jy(
        row,
        source,
        [],
        catalog_wcs,
        (98, 98),
        "ch2",
        catalog_frames=frames,
        catalog_wcs=catalog_wcs,
        catalog_shape=(98, 98),
    )
    np.testing.assert_allclose(prior, truth, rtol=2.0e-3, atol=0.0)


def test_nonnative_pipeline_uses_scene_catalog_grid_and_provenance(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "PRF_OVERSAMPLE", 20)
    monkeypatch.setattr(config, "OUTPUT_DIR", str(tmp_path / "output"))
    monkeypatch.setattr(config, "DIAGNOSTIC_DIR", str(tmp_path / "diagnostics"))
    monkeypatch.setattr(config, "FIT_ON_NATIVE_PIXELS", False)
    monkeypatch.setattr(config, "NATIVE_CUTOUT_SIZE", 0)
    monkeypatch.setattr(config, "GP_OPTIMIZE_HYPERPARAMS", False)
    monkeypatch.setattr(solver, "load_prf", lambda *args, **kwargs: _gaussian_prf())
    truth_jy = 0.002
    mosaic_wcs = _wcs(49)
    scene_wcs = _wcs(25, phase=0.31)
    source = SkyCoord(*scene_wcs.wcs.crval, unit="deg")
    response = solver.native_point_source_column(
        scene_wcs,
        scene_wcs,
        (25, 25),
        (25, 25),
        "ch2",
        source.ra.deg,
        source.dec.deg,
    ).reshape(25, 25)
    cutout = {
        "data": 1.0e-3 + truth_jy * response,
        "sigma": np.full((25, 25), 1.0e-6),
        "wcs": scene_wcs,
        "raw_wcs": scene_wcs,
        "epoch_id": 0,
        "is_template": True,
        "filename": "offset_ch2.fits",
        "mjd": 1.0,
    }
    row = _catalog_row(
        cutout["data"], np.ones((25, 25), dtype=bool), scene_wcs, source,
    )
    table = Table(rows=[row])
    files = [{"image": "synthetic.fits", "filename": "offset_ch2.fits", "mjd": 1.0}]
    processed = [{
        "data": np.full((49, 49), 1.0e-3),
        "sigma": np.full((49, 49), 1.0e-6),
        "raw_wcs": mosaic_wcs,
        "corrected_wcs": mosaic_wcs,
        "native_shape": (49, 49),
        "native_valid_mask": np.ones((49, 49), dtype=bool),
        "projected_valid_mask": np.ones((49, 49), dtype=bool),
    }]
    monkeypatch.setattr(preprocessing, "find_spitzer_files", lambda *args: files)
    monkeypatch.setattr(preprocessing, "get_or_create_source_catalog", lambda *args: {})
    monkeypatch.setattr(preprocessing, "categorize_observations", lambda *args: ([], files))
    monkeypatch.setattr(preprocessing, "define_mosaic_wcs", lambda *args: (mosaic_wcs, (49, 49)))
    monkeypatch.setattr(preprocessing, "reproject_to_grid", lambda *args: processed)
    monkeypatch.setattr(
        preprocessing,
        "create_median_stack",
        lambda cube: (np.nanmedian(cube, axis=0), np.zeros(cube.shape[1:])),
    )
    monkeypatch.setattr(preprocessing, "update_catalog_with_template", lambda *args, **kwargs: None)
    monkeypatch.setattr(preprocessing, "align_frames_to_template", lambda *args: None)
    monkeypatch.setattr(preprocessing, "extract_analysis_cutouts", lambda *args: ([cutout], scene_wcs))
    monkeypatch.setattr(preprocessing, "detect_sources", lambda *args, **kwargs: table)
    monkeypatch.setattr(
        native_fit_campaign,
        "_scene_wcs_from_bcd_footprints",
        lambda *args: (scene_wcs, (25, 25)),
    )
    monkeypatch.setattr(native_fit_campaign, "apply_consensus_cr_masks", lambda *args: None)
    monkeypatch.setattr(pipeline_fit.fits, "writeto", lambda *args, **kwargs: None)
    monkeypatch.setattr(diagnostics, "diagnostic_stretch_mask", lambda *args: np.ones((25, 25), bool))
    monkeypatch.setattr(run_manifest, "write_run_manifest", lambda *args, **kwargs: None)
    captured = {}

    def solve(_cutouts, stars, initial_fluxes, *_args, **_kwargs):
        captured["stars"] = stars
        captured["initial_fluxes"] = initial_fluxes
        return {"star_fluxes": np.asarray(initial_fluxes, dtype=float)}

    monkeypatch.setattr(solver, "run_gls_solve", solve)
    result = pipeline_fit.run_pipeline_fit_core(skip_pre_analysis_check=True)
    assert result is not None and len(captured["stars"]) == 1
    np.testing.assert_allclose(captured["initial_fluxes"], [truth_jy], rtol=2.0e-3)


def test_consensus_cr_preserves_static_source_on_gradients_and_independent_noise():
    cases = [
        _static_cr_frames(gradient=-1.0e-5),
        _static_cr_frames(gradient=-1.0e-4),
    ]
    cases.extend(_static_cr_frames(seed=seed) for seed in range(40))
    for frames in cases:
        native_fit_campaign.apply_consensus_cr_masks(frames)
        assert [int(frame["cr_mask"].sum()) for frame in frames] == [0, 0, 0]


def test_consensus_cr_batches_whole_frame_wcs_work(monkeypatch):
    size = 128
    wcs = _wcs(size)
    frames = [{
        "data": np.full((size, size), 1.0e-3),
        "sigma": np.full((size, size), 1.0e-6),
        "wcs": wcs.deepcopy(),
        "raw_wcs": wcs.deepcopy(),
        "epoch_id": 0,
        "is_template": True,
        "filename": f"scale_{index}_ch2.fits",
        "mjd": index / 10.0,
    } for index in range(3)]
    coordinates = np.linspace(10, size - 11, 8).astype(int)
    for y in coordinates:
        for x in coordinates:
            frames[0]["data"][y, x] += 2.0e-4
    counts = {"calls": 0, "pixels": 0}
    original = WCS.pixel_to_world_values

    def counted(self, x, y, *args, **kwargs):
        counts["calls"] += 1
        counts["pixels"] += int(np.size(x))
        return original(self, x, y, *args, **kwargs)

    monkeypatch.setattr(WCS, "pixel_to_world_values", counted)
    native_fit_campaign.apply_consensus_cr_masks(frames)
    assert [int(frame["cr_mask"].sum()) for frame in frames] == [64, 0, 0]
    assert counts["calls"] <= 30
    assert counts["pixels"] <= 400_000


def _clear_projection_caches():
    solver._PROJECTION_CACHE.clear()
    solver._WCS_PROVENANCE_CACHE.clear()


@pytest.mark.parametrize("attribute", ["cpdis1", "det2im1"])
def test_projection_cache_invalidates_mutated_lookup_table(attribute):
    n = 9
    scene_wcs = _wcs(n)
    native_wcs = _wcs(n, phase=0.27)
    table = DistortionLookupTable(
        np.zeros((2, 2), dtype=np.float32),
        (1.0, 1.0),
        (1.0, 1.0),
        (8.0, 8.0),
    )
    setattr(native_wcs, attribute, table)
    scene = np.arange(n * n, dtype=float).reshape(n, n) * 1.0e-5
    _clear_projection_caches()
    before = solver._project_scene_to_native(
        scene, scene_wcs, native_wcs, (n, n),
    )
    table.data[:] = 0.3
    warm = solver._project_scene_to_native(
        scene, scene_wcs, native_wcs, (n, n),
    )
    _clear_projection_caches()
    cold = solver._project_scene_to_native(
        scene, scene_wcs, native_wcs, (n, n),
    )
    assert not np.allclose(before, cold, rtol=0.0, atol=1.0e-12)
    np.testing.assert_allclose(warm, cold, rtol=0.0, atol=0.0)


def test_projection_cache_invalidates_mutated_header_parameter():
    n = 9
    scene_wcs = _wcs(n)
    native_wcs = _wcs(n, phase=0.27)
    native_wcs.wcs.lonpole = 180.0
    scene = np.arange(n * n, dtype=float).reshape(n, n) * 1.0e-5
    _clear_projection_caches()
    before = solver._project_scene_to_native(
        scene, scene_wcs, native_wcs, (n, n),
    )
    native_wcs.wcs.lonpole = 0.0
    warm = solver._project_scene_to_native(
        scene, scene_wcs, native_wcs, (n, n),
    )
    _clear_projection_caches()
    cold = solver._project_scene_to_native(
        scene, scene_wcs, native_wcs, (n, n),
    )
    assert not np.allclose(before, cold, rtol=0.0, atol=1.0e-12)
    np.testing.assert_allclose(warm, cold, rtol=0.0, atol=0.0)


def test_cache_identity_invalidates_mutated_attached_detector_lookup():
    cutout = _wcs(7)
    detector = _wcs(256)
    detector.det2im1 = DistortionLookupTable(
        np.zeros((2, 2), dtype=np.float32),
        (1.0, 1.0),
        (1.0, 1.0),
        (128.0, 128.0),
    )
    cutout._detector_wcs = detector
    before = solver._wcs_projection_cache_key(cutout)
    detector.det2im1.data[:] = 0.25
    after = solver._wcs_projection_cache_key(cutout)
    assert before != after


@pytest.mark.parametrize(
    ("mode", "project_then_convolve"),
    [("anchor", False), ("exact", False), ("anchor", True)],
)
def test_attached_detector_mutation_warm_equals_cold_for_all_operator_modes(
    monkeypatch,
    mode,
    project_then_convolve,
):
    monkeypatch.setattr(config, "PRF_OVERSAMPLE", 20)
    monkeypatch.setattr(config, "PRF_OPERATOR_MODE", mode)
    monkeypatch.setattr(config, "PRF_ORDER_PROJECT_THEN_CONVOLVE", project_then_convolve)
    monkeypatch.setattr(config, "PRF_SPATIAL_ANCHORS_PER_AXIS", 1)
    n = 7
    cutout = _wcs(n)
    cutout._detector_wcs = _wcs(256)
    x_before, _ = cutout._detector_wcs.world_to_pixel_values(
        config.TRANSIENT_RA, config.TRANSIENT_DEC,
    )
    midpoint = float(x_before) + 40.0

    def detector_prf(_channel, x_det=None, y_det=None):
        yy, xx = np.mgrid[-100:101, -100:101] / 20.0
        sigma = 0.7 if float(x_det) < midpoint else 1.5
        response = np.exp(-(xx**2 + yy**2) / (2.0 * sigma**2))
        return response / response.sum()

    monkeypatch.setattr(solver, "load_prf", detector_prf)
    scene = np.zeros((n, n))
    scene[n // 2, n // 2] = 0.002
    for cache in (
        solver._PROJECTION_CACHE,
        solver._PRF_OPERATOR_BUNDLE_CACHE,
        solver._PRF_NATIVE_OPERATOR_BUNDLE_CACHE,
        solver._PRF_EXACT_OPERATOR_CACHE,
        solver._WCS_PROVENANCE_CACHE,
    ):
        cache.clear()
    before = solver._apply_frame_forward_operator(
        scene, cutout, cutout, (n, n), (n, n), "ch2",
    )
    cutout._detector_wcs.wcs.crpix[0] += 80.0
    warm = solver._apply_frame_forward_operator(
        scene, cutout, cutout, (n, n), (n, n), "ch2",
    )
    for cache in (
        solver._PROJECTION_CACHE,
        solver._PRF_OPERATOR_BUNDLE_CACHE,
        solver._PRF_NATIVE_OPERATOR_BUNDLE_CACHE,
        solver._PRF_EXACT_OPERATOR_CACHE,
        solver._WCS_PROVENANCE_CACHE,
    ):
        cache.clear()
    cold = solver._apply_frame_forward_operator(
        scene, cutout, cutout, (n, n), (n, n), "ch2",
    )
    assert not np.allclose(before, cold, rtol=0.0, atol=1.0e-12)
    np.testing.assert_allclose(warm, cold, rtol=0.0, atol=0.0)


def _flat_frames(n: int, count: int) -> list[dict]:
    frames = []
    for index in range(count):
        wcs = _wcs(n, phase=0.001 * index)
        frames.append({
            "data": np.full((n, n), 1.0e-3),
            "sigma": np.full((n, n), 1.0e-5),
            "wcs": wcs,
            "raw_wcs": wcs,
            "epoch_id": index,
            "is_template": True,
            "filename": f"memory_{index}_ch2.fits",
            "mjd": index / 100.0,
        })
    return frames


def test_memory_guard_counts_covariance_normalization_and_eigensolver(monkeypatch):
    frames = _flat_frames(3, 200)
    monkeypatch.setattr(config, "PRF_OPERATOR_MODE", "anchor")
    monkeypatch.setattr(config, "PRF_ORDER_PROJECT_THEN_CONVOLVE", False)
    monkeypatch.setattr(config, "PRF_SPATIAL_ANCHORS_PER_AXIS", 1)
    estimate = solver._estimate_solver_peak_memory_bytes(
        frames,
        (3, 3),
        n_scene_blocks=1,
        n_stars=0,
        n_science_epochs=0,
    )
    assert estimate["covariance_normalization_bytes"] == 8 * (2 * 200**2 + 200)
    assert estimate["eigensolver_workspace_bytes"] == 8 * (4 * 200**2 + 8 * 200)
    assert estimate["peak_bytes"] > 2_227_889
    monkeypatch.setattr(config, "SOLVER_DENSE_MEMORY_LIMIT_BYTES", 2_227_889)
    monkeypatch.setattr(
        solver,
        "frame_design_matrix",
        lambda *args, **kwargs: pytest.fail("memory guard ran after design allocation"),
    )
    with pytest.raises(MemoryError, match="Dense fit peak"):
        solver.run_gls_solve(
            frames,
            [],
            [],
            {},
            (1.0, 1.0e-7),
            np.zeros((3, 3)),
            frames[0]["wcs"],
            200,
        )


def test_memory_estimate_covers_multiple_scene_blocks_and_allows_budget(monkeypatch):
    frames = _flat_frames(3, 8)
    monkeypatch.setattr(config, "PRF_OPERATOR_MODE", "anchor")
    monkeypatch.setattr(config, "PRF_ORDER_PROJECT_THEN_CONVOLVE", True)
    monkeypatch.setattr(config, "PRF_SPATIAL_ANCHORS_PER_AXIS", 2)
    one = solver._estimate_solver_peak_memory_bytes(
        frames, (3, 5), n_scene_blocks=1, n_stars=2, n_science_epochs=1,
    )
    two = solver._estimate_solver_peak_memory_bytes(
        frames, (3, 5), n_scene_blocks=2, n_stars=2, n_science_epochs=1,
    )
    assert two["n_params"] == one["n_params"] + 15
    assert two["covariance_base_workspace_bytes"] > one["covariance_base_workspace_bytes"]
    assert two["peak_bytes"] > one["peak_bytes"]
    monkeypatch.setattr(config, "SOLVER_DENSE_MEMORY_LIMIT_BYTES", two["peak_bytes"])
    assert solver._guard_solver_memory(
        frames,
        (3, 5),
        n_scene_blocks=2,
        n_stars=2,
        n_science_epochs=1,
    )["peak_bytes"] == two["peak_bytes"]
