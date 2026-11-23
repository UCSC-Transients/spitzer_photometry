"""Behavioral regressions for the independent review of PR #22."""
from __future__ import annotations

import numpy as np
import pytest
import sep
import json
from astropy.coordinates import SkyCoord
from astropy.table import Table
from astropy.wcs import WCS
from reproject import reproject_interp

from src import config, diagnostics, native_fit_campaign, pipeline_fit, postprocessing, preprocessing, run_manifest, solver
from src.prf_identity_context import identity_prf_operators_context
from scripts import validate_internal_consistency as validation


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


def _integrated_gaussian(shape: tuple[int, int], x0: float, y0: float) -> np.ndarray:
    from scipy.special import erf

    yy, xx = np.indices(shape, dtype=float)
    scale = np.sqrt(2.0) * 0.9
    px = 0.5 * (erf((xx - x0 + 0.5) / scale) - erf((xx - x0 - 0.5) / scale))
    py = 0.5 * (erf((yy - y0 + 0.5) / scale) - erf((yy - y0 - 0.5) / scale))
    return px * py


def _cr_frames(*, peer_hole: bool = False, amplitude: float = 0.0) -> list[dict]:
    frames = []
    n = 15
    for i, phase in enumerate((0.0, 0.5, 0.5)):
        wcs = _wcs(n, phase=phase)
        x0, y0 = wcs.world_to_pixel_values(config.TRANSIENT_RA, config.TRANSIENT_DEC)
        sigma = np.full((n, n), 1.0e-6)
        if peer_hole and i > 0:
            sigma[7, 10] = np.inf
        frames.append({
            "data": 1.0e-3 + 2.0e-3 * _integrated_gaussian((n, n), x0, y0),
            "sigma": sigma,
            "wcs": wcs,
            "raw_wcs": wcs,
            "epoch_id": 0,
            "is_template": True,
            "filename": f"cr_ch2_{i}.fits",
            "mjd": i / 10.0,
        })
    frames[0]["data"][7, 7] += amplitude
    return frames


def _clear_operator_caches() -> None:
    solver._PRF_OPERATOR_BUNDLE_CACHE.clear()
    solver._PRF_NATIVE_OPERATOR_BUNDLE_CACHE.clear()
    solver._PRF_EXACT_OPERATOR_CACHE.clear()
    solver._PROJECTION_CACHE.clear()
    solver._prf_cache_fingerprint_cached.cache_clear()


def _catalog_row(response: np.ndarray, wcs: WCS, source: SkyCoord, truth_jy: float):
    x, y = wcs.world_to_pixel_values(source.ra.deg, source.dec.deg)
    radius_arcsec = 4.88
    scale_arcsec = np.sqrt(abs(np.linalg.det(wcs.pixel_scale_matrix))) * 3600.0
    radius_px = radius_arcsec / scale_arcsec
    measured, _, flags = sep.sum_circle(
        np.ascontiguousarray(truth_jy * response, dtype=np.float64),
        np.asarray([x]),
        np.asarray([y]),
        radius_px,
        bkgann=(1.5 * radius_px, 2.5 * radius_px),
        subpix=5,
    )
    assert int(flags[0]) == 0
    return Table({
        "x": [x],
        "y": [y],
        "ra": [source.ra.deg],
        "dec": [source.dec.deg],
        "flux": [measured[0]],
        "catalog_aperture_flux_jy": [measured[0]],
        "catalog_aperture_radius_arcsec": [radius_arcsec],
        "catalog_aperture_annulus_inner_arcsec": [1.5 * radius_arcsec],
        "catalog_aperture_annulus_outer_arcsec": [2.5 * radius_arcsec],
        "catalog_aperture_subpix": [5],
    })[0]


def test_distant_catalog_source_has_no_native_fit_support(monkeypatch):
    monkeypatch.setattr(config, "PRF_OVERSAMPLE", 20)
    monkeypatch.setattr(solver, "load_prf", lambda *args, **kwargs: _gaussian_prf())
    wcs = _wcs(9)
    cutout = {
        "data": np.ones((9, 9)),
        "sigma": np.ones((9, 9)),
        "wcs": wcs,
        "raw_wcs": wcs,
    }
    far = SkyCoord(
        config.TRANSIENT_RA + 60.0 / 3600.0 / np.cos(np.deg2rad(config.TRANSIENT_DEC)),
        config.TRANSIENT_DEC,
        unit="deg",
    )
    assert not pipeline_fit._source_has_native_fit_support(far, [cutout], "ch2")


def test_full_field_distant_source_does_not_abort_native_pipeline(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "PRF_OVERSAMPLE", 20)
    monkeypatch.setattr(config, "OUTPUT_DIR", str(tmp_path / "output"))
    monkeypatch.setattr(config, "DIAGNOSTIC_DIR", str(tmp_path / "diagnostics"))
    monkeypatch.setattr(config, "FIT_ON_NATIVE_PIXELS", True)
    monkeypatch.setattr(config, "NATIVE_CUTOUT_SIZE", 0)
    monkeypatch.setattr(config, "GP_OPTIMIZE_HYPERPARAMS", False)
    monkeypatch.setattr(config, "UNMASK_SIGMA_INF_RADIUS_PX", 0.0)
    monkeypatch.setattr(solver, "load_prf", lambda *args, **kwargs: _gaussian_prf())
    wcs = _wcs(9)
    far = SkyCoord(
        config.TRANSIENT_RA + 60.0 / 3600.0 / np.cos(np.deg2rad(config.TRANSIENT_DEC)),
        config.TRANSIENT_DEC,
        unit="deg",
    )
    table = Table({
        "ra": [far.ra.deg],
        "dec": [far.dec.deg],
        "flux": [0.002],
        "catalog_aperture_flux_jy": [0.002],
        "catalog_aperture_radius_arcsec": [4.88],
    })
    cutout = {
        "data": np.full((9, 9), 1.0e-3),
        "sigma": np.full((9, 9), 1.0e-5),
        "wcs": wcs,
        "raw_wcs": wcs,
        "epoch_id": 0,
        "is_template": True,
        "filename": "synthetic_ch2.fits",
        "mjd": 1.0,
    }
    files = [{"image": "synthetic.fits", "filename": "synthetic_ch2.fits", "mjd": 1.0}]
    monkeypatch.setattr(preprocessing, "find_spitzer_files", lambda *args: files)
    monkeypatch.setattr(preprocessing, "get_or_create_source_catalog", lambda *args: {})
    monkeypatch.setattr(preprocessing, "categorize_observations", lambda *args: ([], files))
    monkeypatch.setattr(preprocessing, "define_mosaic_wcs", lambda *args: (wcs, (9, 9)))
    monkeypatch.setattr(preprocessing, "reproject_to_grid", lambda *args: [{"data": cutout["data"]}])
    monkeypatch.setattr(
        preprocessing,
        "create_median_stack",
        lambda *args: (cutout["data"], np.zeros((9, 9))),
    )
    monkeypatch.setattr(preprocessing, "update_catalog_with_template", lambda *args: table)
    monkeypatch.setattr(preprocessing, "align_frames_to_template", lambda *args: None)
    monkeypatch.setattr(preprocessing, "extract_native_analysis_cutouts", lambda *args: ([cutout], wcs))
    monkeypatch.setattr(
        native_fit_campaign,
        "_scene_wcs_from_bcd_footprints",
        lambda *args: (wcs, (9, 9)),
    )
    monkeypatch.setattr(native_fit_campaign, "apply_consensus_cr_masks", lambda *args: None)
    monkeypatch.setattr(native_fit_campaign, "unmask_sigma_inf_in_radius", lambda *args: None)
    reached = []

    def solve(_cutouts, stars, *_args, **_kwargs):
        reached.append(True)
        assert stars == []
        return {"star_fluxes": np.zeros(0)}

    monkeypatch.setattr(solver, "run_gls_solve", solve)
    monkeypatch.setattr(pipeline_fit.fits, "writeto", lambda *args, **kwargs: None)
    monkeypatch.setattr(diagnostics, "diagnostic_stretch_mask", lambda *args: np.ones((9, 9), dtype=bool))
    monkeypatch.setattr(run_manifest, "write_run_manifest", lambda *args, **kwargs: None)

    result = pipeline_fit.run_pipeline_fit_core(skip_pre_analysis_check=True)
    assert reached and result is not None


def test_consensus_cr_abstains_when_peers_have_different_valid_support():
    frames = _cr_frames(peer_hole=True)
    native_fit_campaign.apply_consensus_cr_masks(frames)
    assert [int(frame["cr_mask"].sum()) for frame in frames] == [0, 0, 0]


@pytest.mark.parametrize("amplitude_sigma", [20.0, 80.0, 200.0])
def test_consensus_cr_retains_pixel_threshold_sensitivity(amplitude_sigma):
    frames = _cr_frames()
    for frame in frames:
        frame["data"] = np.full((15, 15), 1.0e-3)
    frames[0]["data"][7, 7] += amplitude_sigma * 1.0e-6
    native_fit_campaign.apply_consensus_cr_masks(frames)
    assert frames[0]["cr_mask"][7, 7]


def test_consensus_cr_conservatively_abstains_at_clipped_boundary():
    frames = _cr_frames()
    frames[0]["data"][1, 1] += 2.0e-4
    native_fit_campaign.apply_consensus_cr_masks(frames)
    assert not frames[0]["cr_mask"][1, 1]


@pytest.mark.parametrize(
    ("mode", "project_then_convolve"),
    [("anchor", False), ("exact", False), ("anchor", True)],
)
def test_operator_cache_distinguishes_attached_detector_wcs(
    monkeypatch,
    mode,
    project_then_convolve,
):
    monkeypatch.setattr(config, "PRF_OVERSAMPLE", 20)
    monkeypatch.setattr(config, "PRF_OPERATOR_MODE", mode)
    monkeypatch.setattr(config, "PRF_ORDER_PROJECT_THEN_CONVOLVE", project_then_convolve)
    monkeypatch.setattr(config, "PRF_SPATIAL_ANCHORS_PER_AXIS", 1)
    n = 7
    cutout_a = _wcs(n)
    cutout_b = _wcs(n)
    cutout_a._detector_wcs = _wcs(256, phase=0.0)
    cutout_b._detector_wcs = _wcs(256, phase=80.0)

    def detector_prf(_channel, x_det=None, y_det=None):
        yy, xx = np.mgrid[-100:101, -100:101] / 20.0
        sigma = 0.7 if float(x_det) < 170.0 else 1.5
        prf = np.exp(-(xx**2 + yy**2) / (2.0 * sigma**2))
        return prf / prf.sum()

    monkeypatch.setattr(solver, "load_prf", detector_prf)
    scene = np.zeros((n, n))
    scene[n // 2, n // 2] = 0.002
    _clear_operator_caches()
    response_a = solver._apply_frame_forward_operator(
        scene, cutout_a, cutout_a, (n, n), (n, n), "ch2",
    )
    response_b_cached = solver._apply_frame_forward_operator(
        scene, cutout_b, cutout_b, (n, n), (n, n), "ch2",
    )
    _clear_operator_caches()
    response_b_cold = solver._apply_frame_forward_operator(
        scene, cutout_b, cutout_b, (n, n), (n, n), "ch2",
    )
    assert not np.allclose(response_a, response_b_cold, rtol=0.0, atol=1.0e-12)
    np.testing.assert_allclose(response_b_cached, response_b_cold, rtol=0.0, atol=0.0)


def test_wcs_fingerprint_invalidates_in_place_detector_change():
    cutout = _wcs(7)
    cutout._detector_wcs = _wcs(256)
    before = solver._wcs_projection_cache_key(cutout)
    cutout._detector_wcs.wcs.crpix[0] += 80.0
    after = solver._wcs_projection_cache_key(cutout)
    assert before != after


@pytest.mark.parametrize(
    ("mode", "project_then_convolve"),
    [("anchor", False), ("exact", False), ("anchor", True)],
)
def test_apodization_setting_is_in_every_operator_cache_identity(
    monkeypatch,
    mode,
    project_then_convolve,
):
    monkeypatch.setattr(config, "PRF_OVERSAMPLE", 20)
    monkeypatch.setattr(config, "PRF_OPERATOR_MODE", mode)
    monkeypatch.setattr(config, "PRF_ORDER_PROJECT_THEN_CONVOLVE", project_then_convolve)
    monkeypatch.setattr(config, "PRF_SPATIAL_ANCHORS_PER_AXIS", 1)
    n = 7
    wcs = _wcs(n)

    def configured_prf(_channel, x_det=None, y_det=None):
        yy, xx = np.mgrid[-100:101, -100:101] / 20.0
        sigma = 0.7 if not config.PRF_APPLY_APODIZATION else 1.5
        prf = np.exp(-(xx**2 + yy**2) / (2.0 * sigma**2))
        return prf / prf.sum()

    monkeypatch.setattr(solver, "load_prf", configured_prf)
    scene = np.zeros((n, n))
    scene[n // 2, n // 2] = 0.002
    _clear_operator_caches()
    monkeypatch.setattr(config, "PRF_APPLY_APODIZATION", False)
    response_a = solver._apply_frame_forward_operator(
        scene, wcs, wcs, (n, n), (n, n), "ch2",
    )
    monkeypatch.setattr(config, "PRF_APPLY_APODIZATION", True)
    response_b = solver._apply_frame_forward_operator(
        scene, wcs, wcs, (n, n), (n, n), "ch2",
    )
    monkeypatch.setattr(config, "PRF_APPLY_APODIZATION", False)
    response_a_warm = solver._apply_frame_forward_operator(
        scene, wcs, wcs, (n, n), (n, n), "ch2",
    )
    _clear_operator_caches()
    response_a_cold = solver._apply_frame_forward_operator(
        scene, wcs, wcs, (n, n), (n, n), "ch2",
    )
    assert not np.allclose(response_a, response_b, rtol=0.0, atol=1.0e-12)
    np.testing.assert_allclose(response_a_warm, response_a_cold, rtol=0.0, atol=0.0)


def _flat_frame(ny: int, nx: int, index: int) -> dict:
    wcs = _wcs(max(ny, nx), phase=0.01 * index)
    if (ny, nx) != (max(ny, nx), max(ny, nx)):
        wcs = wcs.slice((slice(0, ny), slice(0, nx)))
    return {
        "data": np.full((ny, nx), 1.0e-3),
        "sigma": np.full((ny, nx), 1.0e-5),
        "wcs": wcs,
        "raw_wcs": wcs,
        "epoch_id": 0,
        "is_template": True,
        "filename": f"memory_ch2_{index}.fits",
        "mjd": index / 100.0,
    }


def test_memory_guard_counts_exact_and_scene_anchor_caches_before_allocation(monkeypatch):
    frames = [_flat_frame(9, 9, index) for index in range(40)]
    monkeypatch.setattr(config, "PRF_OPERATOR_MODE", "exact")
    monkeypatch.setattr(config, "PRF_ORDER_PROJECT_THEN_CONVOLVE", False)
    monkeypatch.setattr(config, "PRF_SPATIAL_ANCHORS_PER_AXIS", 5)
    estimate = solver._estimate_solver_peak_memory_bytes(
        frames,
        (9, 9),
        n_scene_blocks=1,
        n_stars=0,
        n_science_epochs=0,
    )
    assert estimate["exact_operator_cache_bytes"] == 2_099_520
    assert estimate["scene_anchor_operator_cache_bytes"] == 1_321_920
    assert estimate["exact_operator_cache_bytes"] + estimate["anchor_operator_cache_bytes"] == 3_421_440
    monkeypatch.setattr(config, "SOLVER_DENSE_MEMORY_LIMIT_BYTES", 2_787_033)
    monkeypatch.setattr(
        solver,
        "frame_design_matrix",
        lambda *args, **kwargs: pytest.fail("guard ran after operator allocation"),
    )
    with pytest.raises(MemoryError, match="Dense fit peak"):
        solver.run_gls_solve(
            frames,
            [],
            [],
            {},
            (1.0, 1.0e-7),
            np.zeros((9, 9)),
            frames[0]["wcs"],
            1,
        )
    assert not solver._PRF_EXACT_OPERATOR_CACHE
    assert not solver._PRF_OPERATOR_BUNDLE_CACHE


def test_memory_estimate_covers_rectangular_design_and_multiple_scene_blocks(monkeypatch):
    frames = [_flat_frame(5, 7, index) for index in range(3)]
    monkeypatch.setattr(config, "PRF_OPERATOR_MODE", "anchor")
    monkeypatch.setattr(config, "PRF_ORDER_PROJECT_THEN_CONVOLVE", True)
    monkeypatch.setattr(config, "PRF_SPATIAL_ANCHORS_PER_AXIS", 2)
    one = solver._estimate_solver_peak_memory_bytes(
        frames,
        (7, 11),
        n_scene_blocks=1,
        n_stars=2,
        n_science_epochs=1,
    )
    two = solver._estimate_solver_peak_memory_bytes(
        frames,
        (7, 11),
        n_scene_blocks=2,
        n_stars=2,
        n_science_epochs=1,
    )
    assert one["rectangular_design_bytes"] == 8 * 2 * 35 * 77
    assert one["scene_anchor_operator_cache_bytes"] == 8 * 9 * 3 * 77
    assert one["native_anchor_operator_cache_bytes"] == 8 * 9 * 3 * 35
    assert two["prior_bytes"] == 2 * one["prior_bytes"]
    assert two["n_params"] == one["n_params"] + 77
    assert two["peak_bytes"] > one["peak_bytes"]


def test_memory_estimate_bounds_retained_operator_arrays(monkeypatch):
    frames = [_flat_frame(7, 7, index) for index in range(5)]
    settings = {
        "PRF_OPERATOR_MODE": "exact",
        "PRF_ORDER_PROJECT_THEN_CONVOLVE": False,
        "PRF_SPATIAL_ANCHORS_PER_AXIS": 3,
        "SOLVER_DENSE_MEMORY_LIMIT_BYTES": 100_000_000,
        "USE_SCENE_GP_PRIOR": False,
        "SCENE_INDEPENDENT_RIDGE": 1.0e10,
        "USE_HOST_GAUSSIAN_CORE": False,
        "USE_NUCLEAR_POINT_SOURCE": False,
        "FLOAT_TRANSIENT_POSITION": False,
    }
    for name, value in settings.items():
        monkeypatch.setattr(config, name, value)

    def delta(scene_wcs, native_wcs, shape, channel, ra, dec, **kwargs):
        x, y = scene_wcs.world_to_pixel_values(ra, dec)
        response = np.zeros(shape)
        ix = int(np.clip(np.round(x), 0, shape[1] - 1))
        iy = int(np.clip(np.round(y), 0, shape[0] - 1))
        response[iy, ix] = 1.0
        return response.ravel()

    monkeypatch.setattr(solver, "convolved_delta_column", delta)
    result = solver.run_gls_solve(
        frames,
        [],
        [],
        {},
        (1.0, 1.0e-7),
        np.zeros((7, 7)),
        frames[0]["wcs"],
        1,
    )

    def retained_bytes(cache) -> int:
        arrays = []

        def collect(value):
            if isinstance(value, np.ndarray):
                arrays.append(value)
            elif isinstance(value, dict):
                for nested in value.values():
                    collect(nested)
            elif isinstance(value, (tuple, list)):
                for nested in value:
                    collect(nested)

        collect(cache)
        return sum(array.nbytes for array in {id(array): array for array in arrays}.values())

    actual_exact = retained_bytes(solver._PRF_EXACT_OPERATOR_CACHE)
    actual_anchor = retained_bytes(solver._PRF_OPERATOR_BUNDLE_CACHE)
    estimate = result["memory_estimate"]
    assert 0 < actual_exact <= estimate["exact_operator_cache_bytes"]
    assert 0 < actual_anchor <= estimate["scene_anchor_operator_cache_bytes"]


@pytest.mark.parametrize("n", [5, 7, 9, 11])
@pytest.mark.parametrize("sigma", [1.0e-3, 1.0e-5, 1.0e-7])
def test_colocated_science_sources_are_scale_invariantly_nonidentifiable(
    monkeypatch,
    n,
    sigma,
):
    settings = {
        "USE_SCENE_GP_PRIOR": False,
        "SCENE_INDEPENDENT_RIDGE": 1.0e10,
        "USE_NUCLEAR_POINT_SOURCE": True,
        "NUCLEAR_POINT_RA": config.TRANSIENT_RA,
        "NUCLEAR_POINT_DEC": config.TRANSIENT_DEC,
        "NUCLEAR_POINT_NONNEGATIVE": True,
        "TRANSIENT_NONNEGATIVE": True,
        "FLOAT_TRANSIENT_POSITION": False,
        "PRF_ORDER_PROJECT_THEN_CONVOLVE": True,
        "PRF_OPERATOR_MODE": "anchor",
        "PRF_SPATIAL_ANCHORS_PER_AXIS": 1,
    }
    for name, value in settings.items():
        monkeypatch.setattr(config, name, value)
    wcs = _wcs(n)
    frame = _flat_frame(n, n, 0)
    frame["is_template"] = False
    frame["sigma"][:] = sigma
    with identity_prf_operators_context():
        response = solver.native_point_source_column(
            wcs,
            wcs,
            (n, n),
            (n, n),
            "ch2",
            config.TRANSIENT_RA,
            config.TRANSIENT_DEC,
        ).reshape(n, n)
        frame["data"] += 0.002 * response
        result = solver.run_gls_solve(
            [frame],
            [],
            [],
            {},
            (1.0, 1.0e-7),
            np.zeros((n, n)),
            wcs,
            1,
        )
    assert result["identifiability_status"] == "non_identifiable_or_singular_curvature"
    assert result["uncertainty_status"] == "unavailable"
    assert result["transient_epoch_errs"].shape == (1,)
    assert np.isnan(result["transient_epoch_errs"][0])
    assert np.isnan(result["nuclear_point_err"])


def test_near_colocated_sources_report_scale_aware_diagnostic(monkeypatch):
    n = 9
    settings = {
        "USE_SCENE_GP_PRIOR": False,
        "SCENE_INDEPENDENT_RIDGE": 1.0e10,
        "USE_NUCLEAR_POINT_SOURCE": True,
        "NUCLEAR_POINT_RA": config.TRANSIENT_RA + 1.0e-9,
        "NUCLEAR_POINT_DEC": config.TRANSIENT_DEC,
        "TRANSIENT_NONNEGATIVE": True,
        "FLOAT_TRANSIENT_POSITION": False,
        "PRF_ORDER_PROJECT_THEN_CONVOLVE": True,
        "PRF_OPERATOR_MODE": "anchor",
        "PRF_SPATIAL_ANCHORS_PER_AXIS": 1,
    }
    for name, value in settings.items():
        monkeypatch.setattr(config, name, value)
    wcs = _wcs(n)
    frame = _flat_frame(n, n, 0)
    frame["is_template"] = False
    with identity_prf_operators_context():
        response = solver.native_point_source_column(
            wcs,
            wcs,
            (n, n),
            (n, n),
            "ch2",
            config.TRANSIENT_RA,
            config.TRANSIENT_DEC,
        ).reshape(n, n)
        frame["data"] += 0.002 * response
        result = solver.run_gls_solve(
            [frame], [], [], {}, (1.0, 1.0e-7), np.zeros((n, n)), wcs, 1,
        )
    assert result["uncertainty_status"] == "unavailable"
    diagnostics_payload = result["identifiability_diagnostics"]
    assert diagnostics_payload["covariance_correlation_rcond"] <= diagnostics_payload["required_rcond"]
    _, summary = postprocessing.group_and_average([dict(frame, channel="ch2")], result)
    assert len(summary) == 1
    assert np.isnan(float(summary.iloc[0]["err"]))


def test_validation_threshold_status_precedes_api_and_json_success(monkeypatch, tmp_path):
    original = solver.run_gls_solve

    def high_pull(*args, **kwargs):
        result = original(*args, **kwargs)
        result["transient_fluxes"][:] += 0.01
        return result

    monkeypatch.setattr(solver, "run_gls_solve", high_pull)
    destination = tmp_path / "high-pull"
    report = validation.run_validation(destination, trials=1)
    artifact = json.loads((destination / "internal_consistency.json").read_text())
    assert report["status"] == "failed"
    assert report["failure_kind"] == "acceptance_threshold"
    assert report["max_abs_pull"] > 5.0
    assert artifact["status"] == report["status"]
    assert artifact["max_abs_pull"] == report["max_abs_pull"]
    assert validation.main(["--trials", "1", "--output-dir", str(destination)]) == 3
    assert json.loads((destination / "internal_consistency.json").read_text())["status"] == "failed"


def test_validation_invalid_output_overwrites_reused_success_artifact(monkeypatch, tmp_path):
    original = solver.run_gls_solve

    def invalid_error(*args, **kwargs):
        result = original(*args, **kwargs)
        result["transient_errs"][:] = np.inf
        return result

    monkeypatch.setattr(solver, "run_gls_solve", invalid_error)
    destination = tmp_path / "reused"
    destination.mkdir()
    artifact_path = destination / "internal_consistency.json"
    artifact_path.write_text('{"status":"passed","stale":true}\n')
    exit_code = validation.main(["--trials", "1", "--output-dir", str(destination)])
    artifact_text = artifact_path.read_text()
    artifact = json.loads(
        artifact_text,
        parse_constant=lambda value: pytest.fail(f"non-standard JSON constant: {value}"),
    )
    assert exit_code == 2
    assert artifact["status"] == "failed"
    assert artifact["failure_kind"] == "invalid_science_output"
    assert artifact["max_abs_pull"] is None
    assert "stale" not in artifact
    assert "Infinity" not in artifact_text and "NaN" not in artifact_text


@pytest.mark.parametrize("sr", [1, 2, 4])
def test_catalog_measurement_reprojection_is_crop_invariant(monkeypatch, sr):
    monkeypatch.setattr(config, "PRF_OVERSAMPLE", 20)
    monkeypatch.setattr(solver, "load_prf", lambda *args, **kwargs: _gaussian_prf())
    truth_jy = 0.002
    native_n = 25
    native_wcs = _wcs(native_n, phase=0.23)
    source = SkyCoord(*native_wcs.pixel_to_world_values(12.0, 12.0), unit="deg")
    catalog_n = native_n * sr
    catalog_wcs = _wcs(catalog_n, scale_arcsec=1.22 / sr)
    response_native = solver.native_point_source_column(
        native_wcs,
        native_wcs,
        (native_n, native_n),
        (native_n, native_n),
        "ch2",
        source.ra.deg,
        source.dec.deg,
    ).reshape(native_n, native_n)
    response_catalog, _ = reproject_interp(
        (response_native, native_wcs),
        catalog_wcs,
        shape_out=(catalog_n, catalog_n),
    )
    response_catalog = np.nan_to_num(response_catalog)
    area_factor = preprocessing._scene_sample_to_integrated_jy_factor(catalog_wcs)
    row = _catalog_row(response_catalog * area_factor, catalog_wcs, source, truth_jy)
    catalog_frames = [{
        "raw_wcs": native_wcs,
        "native_shape": (native_n, native_n),
        "native_valid_mask": np.ones((native_n, native_n), dtype=bool),
        "is_full_array": False,
    }]

    priors = []
    for width in (25, 14, 13):
        fit_wcs = native_wcs.slice((slice(None), slice(0, width)))
        cutout = {
            "data": np.ones((native_n, width)),
            "sigma": np.ones((native_n, width)),
            "wcs": fit_wcs,
            "raw_wcs": fit_wcs,
            "is_template": True,
        }
        prior, fraction = pipeline_fit._catalog_prior_flux_jy(
            row,
            source,
            [cutout],
            fit_wcs,
            (native_n, width),
            "ch2",
            catalog_frames=catalog_frames,
            catalog_wcs=catalog_wcs,
            catalog_shape=(catalog_n, catalog_n),
        )
        assert fraction > 0.0
        priors.append(prior)
    np.testing.assert_allclose(priors, truth_jy, rtol=2.0e-3, atol=0.0)
    np.testing.assert_allclose(priors, np.full(3, priors[0]), rtol=0.0, atol=0.0)
