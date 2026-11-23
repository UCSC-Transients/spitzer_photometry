"""Behavioral regressions for the third independent review of PR #22."""
from __future__ import annotations

import numpy as np
import sep
from astropy.coordinates import SkyCoord
from astropy.table import Table
from astropy.wcs import WCS

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
