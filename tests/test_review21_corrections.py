"""Behavioral regressions for the independent review of PR #21."""
from __future__ import annotations

from math import erf

import numpy as np
import pytest
from astropy.coordinates import SkyCoord
from astropy.wcs import WCS

from scripts import validate_internal_consistency as validation
from src import (
    config,
    native_fit_campaign as campaign,
    pipeline_fit,
    postprocessing,
    preprocessing,
    solver,
)
from src.prf_identity_context import identity_prf_operators_context


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


def _flat_cutout(n: int, index: int, *, template: bool = True) -> dict:
    wcs = _wcs(n, phase=0.01 * index)
    return {
        "data": np.full((n, n), 1.0e-3),
        "sigma": np.full((n, n), 1.0e-5),
        "wcs": wcs,
        "raw_wcs": wcs,
        "epoch_id": 0,
        "is_template": template,
        "filename": f"memory_ch2_{index}.fits",
        "mjd": index / 10.0,
    }


def test_memory_guard_rejects_9x9_40_frame_exact_cache_before_allocation(monkeypatch):
    cutouts = [_flat_cutout(9, i) for i in range(40)]
    monkeypatch.setattr(config, "PRF_OPERATOR_MODE", "exact")
    monkeypatch.setattr(config, "PRF_ORDER_PROJECT_THEN_CONVOLVE", False)
    monkeypatch.setattr(config, "SOLVER_DENSE_MEMORY_LIMIT_BYTES", 800_000)
    monkeypatch.setattr(
        solver,
        "frame_design_matrix",
        lambda *args, **kwargs: pytest.fail("preallocation guard was reached too late"),
    )
    with pytest.raises(MemoryError, match="exact cache"):
        solver.run_gls_solve(
            cutouts, [], [], {}, (1.0, 1.0e-7), np.zeros((9, 9)), cutouts[0]["wcs"], 1,
        )


def test_memory_guard_allows_under_budget_small_fit(monkeypatch):
    cutouts = [_flat_cutout(5, i) for i in range(2)]
    monkeypatch.setattr(config, "PRF_OPERATOR_MODE", "anchor")
    monkeypatch.setattr(config, "PRF_ORDER_PROJECT_THEN_CONVOLVE", True)
    estimate = solver._estimate_solver_peak_memory_bytes(
        cutouts, (5, 5), n_scene_blocks=1, n_stars=0, n_science_epochs=0,
    )
    monkeypatch.setattr(config, "SOLVER_DENSE_MEMORY_LIMIT_BYTES", estimate["peak_bytes"] + 1)
    monkeypatch.setattr(config, "USE_SCENE_GP_PRIOR", False)
    monkeypatch.setattr(config, "SCENE_INDEPENDENT_RIDGE", 1.0)
    with identity_prf_operators_context():
        result = solver.run_gls_solve(
            cutouts, [], [], {}, (1.0, 1.0e-7), np.zeros((5, 5)), cutouts[0]["wcs"], 1,
        )
    assert result["memory_estimate"]["peak_bytes"] == estimate["peak_bytes"]
    assert np.isfinite(result["model_scene"]).all()


def test_saved_fit_prediction_is_stable_across_blur_and_calibration_ab(monkeypatch):
    n = 7
    wcs = _wcs(n)
    prf_y, prf_x = np.mgrid[-200:201, -200:201]

    def calibration(*args, **kwargs):
        sigma = 50.0 if str(config.PRF_DIR).endswith("calibration-A") else 110.0
        array = np.exp(-(prf_x**2 + prf_y**2) / (2.0 * sigma**2))
        return array / array.sum()

    monkeypatch.setattr(solver, "load_prf", calibration)
    cutout = _flat_cutout(n, 0)
    image_y, image_x = np.mgrid[-3:4, -3:4]
    cutout["data"] = 1.0e-3 + 5.0e-4 * np.exp(-(image_x**2 + image_y**2) / 2.0)
    common = {
        "USE_SCENE_GP_PRIOR": False,
        "SCENE_INDEPENDENT_RIDGE": 1.0e10,
        "PRF_ORDER_PROJECT_THEN_CONVOLVE": False,
        "PRF_OPERATOR_MODE": "anchor",
        "PRF_SPATIAL_ANCHORS_PER_AXIS": 1,
        "FLOAT_TRANSIENT_POSITION": False,
    }
    with campaign._temporary_config({
        **common,
        "PRF_DIR": "/synthetic/calibration-A",
        "PRF_PREBLUR_IN_OVERSAMPLE_PIXELS": 0.0,
    }):
        result_a = solver.run_gls_solve(
            [cutout], [], [], {}, (1.0, 1.0e-7), np.zeros((n, n)), wcs, 1,
        )
        prediction_a_before = solver.predict_cutout_model(result_a, [cutout], [], [], 0)
    with campaign._temporary_config({
        **common,
        "PRF_DIR": "/synthetic/calibration-B",
        "PRF_PREBLUR_IN_OVERSAMPLE_PIXELS": 30.0,
    }):
        result_b = solver.run_gls_solve(
            [cutout], [], [], {}, (1.0, 1.0e-7), np.zeros((n, n)), wcs, 1,
        )
        prediction_b = solver.predict_cutout_model(result_b, [cutout], [], [], 0)
    prediction_a_after = solver.predict_cutout_model(result_a, [cutout], [], [], 0)

    np.testing.assert_allclose(prediction_a_after, prediction_a_before, rtol=0.0, atol=0.0)
    assert not np.allclose(prediction_a_before, prediction_b, rtol=0.0, atol=1.0e-10)


def test_singular_colocated_sources_preserve_uncertainty_shapes(tmp_path):
    n = 5
    wcs = _wcs(n)
    cutout = _flat_cutout(n, 0, template=False)
    response = solver.native_point_source_column(
        wcs, wcs, (n, n), (n, n), "ch2", config.TRANSIENT_RA, config.TRANSIENT_DEC,
    ).reshape(n, n)
    cutout["data"] += 2.0e-3 * response
    with campaign._temporary_config({
        "DIAGNOSTIC_DIR": str(tmp_path),
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
    }):
        result = solver.run_gls_solve(
            [cutout], [], [], {}, (1.0, 1.0e-7), np.zeros((n, n)), wcs, 1,
        )
        _, summary = postprocessing.group_and_average(
            [dict(cutout, channel="ch2")], result,
        )

    assert result["uncertainty_status"] == "unavailable"
    assert result["identifiability_status"] == "non_identifiable_or_singular_curvature"
    assert result["transient_errs"].shape == (1,)
    assert result["transient_epoch_errs"].shape == (1,)
    assert np.isnan(result["transient_errs"][0])
    assert np.isnan(result["transient_epoch_errs"][0])
    assert np.isnan(result["nuclear_point_err"])
    assert len(summary) == 1 and np.isnan(float(summary.iloc[0]["err"]))


@pytest.mark.parametrize(
    ("flux", "error", "prediction", "message"),
    [
        (np.nan, 1.0, np.ones((2, 2)), "recovered flux"),
        (1.0, np.inf, np.ones((2, 2)), "reported uncertainty"),
        (1.0, 0.0, np.ones((2, 2)), "must be positive"),
        (1.0, 1.0, np.array([[1.0, np.inf], [1.0, 1.0]]), "prediction"),
    ],
)
def test_validation_rejects_invalid_science_values(flux, error, prediction, message):
    with pytest.raises(RuntimeError, match=message):
        validation._validate_trial_values(
            flux_jy=flux,
            error_jy=error,
            truth_jy=1.0,
            prediction=prediction,
            residual_z=np.ones((2, 2)),
        )


def test_validation_cli_infinite_error_exits_nonzero_and_json_is_strict(tmp_path, monkeypatch, capsys):
    original_solve = solver.run_gls_solve

    def corrupt_error(*args, **kwargs):
        result = original_solve(*args, **kwargs)
        result["transient_errs"][:] = np.inf
        return result

    monkeypatch.setattr(solver, "run_gls_solve", corrupt_error)
    status = validation.main([
        "--trials", "1", "--output-dir", str(tmp_path / "invalid-validation"),
    ])
    output = capsys.readouterr().out
    assert status == 2
    assert '"status": "failed"' in output
    assert "Infinity" not in output and "NaN" not in output


def test_validation_report_is_standards_compliant_json(tmp_path):
    import json

    report = validation.run_validation(tmp_path / "valid-validation", trials=1)
    payload = (tmp_path / "valid-validation" / "internal_consistency.json").read_text()
    decoded = json.loads(
        payload,
        parse_constant=lambda value: pytest.fail(f"non-standard JSON constant: {value}"),
    )
    assert decoded["status"] == "passed"
    assert report["status"] == "passed"
