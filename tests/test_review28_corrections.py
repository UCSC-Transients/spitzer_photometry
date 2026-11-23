"""Behavioral regressions for the eighth independent review of PR #22."""
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
    phase_x: float = 0.0,
    phase_y: float | None = None,
    *,
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
    transform.wcs.cdelt = [
        -scale_arcsec / 3600.0,
        scale_arcsec / 3600.0,
    ]
    return transform


def _frame(transform: WCS, size: int, index: int) -> dict:
    return {
        "data": np.full((size, size), 1.0e-3),
        "sigma": np.full((size, size), 1.0e-6),
        "wcs": transform,
        "raw_wcs": transform,
        "epoch_id": 0,
        "is_template": True,
        "filename": f"review28_{index}_ch2.fits",
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


def _cr_frames(size: int, mode: str, *, source: bool = False) -> list[dict]:
    frames = []
    for index in range(3):
        phase = 0.25 if index and mode == "phase" else 0.0
        transform = _wcs(size, phase)
        if index and mode == "roll":
            angle = np.deg2rad(15.0)
            transform.wcs.pc = [
                [np.cos(angle), -np.sin(angle)],
                [np.sin(angle), np.cos(angle)],
            ]
        entry = _frame(transform, size, index)
        if source:
            x0, y0 = transform.world_to_pixel_values(
                config.TRANSIENT_RA, config.TRANSIENT_DEC,
            )
            entry["data"] += 0.002 * _integrated_gaussian(
                (size, size), x0, y0,
            )
        frames.append(entry)
    return frames


@pytest.mark.parametrize("mode", ["phase", "roll"])
@pytest.mark.parametrize("masked_peer_count", [1, 2])
@pytest.mark.parametrize("source", [False, True])
def test_mask_aware_bicubic_reference_preserves_distant_support(
    mode,
    masked_peer_count,
    source,
):
    frames = _cr_frames(49, mode, source=source)
    for index in range(1, 1 + masked_peer_count):
        frames[index]["sigma"][2, 2] = np.inf
    frames[0]["data"][24, 24] += 200.0e-6

    native_fit_campaign.apply_consensus_cr_masks(frames)

    assert frames[0]["cr_mask"][24, 24]
    assert [int(frame["cr_mask"].sum()) for frame in frames] == [1, 0, 0]
    assert frames[0]["cr_unsupported_pixel_count"] > 0


@pytest.mark.parametrize("mode", ["phase", "roll"])
@pytest.mark.parametrize("masked_peer_count", [1, 2])
def test_local_peer_hole_reports_partial_support_and_abstains(
    mode,
    masked_peer_count,
):
    frames = _cr_frames(49, mode)
    for index in range(1, 1 + masked_peer_count):
        x0, y0 = frames[index]["wcs"].world_to_pixel_values(
            config.TRANSIENT_RA, config.TRANSIENT_DEC,
        )
        frames[index]["sigma"][int(np.rint(y0)), int(np.rint(x0))] = np.inf
    frames[0]["data"][24, 24] += 200.0e-6

    native_fit_campaign.apply_consensus_cr_masks(frames)

    assert not frames[0]["cr_mask"][24, 24]
    assert frames[0]["cr_unsupported_pixel_count"] > 0
    assert frames[0]["cr_status"] == "consensus"


@pytest.mark.parametrize("noise_ratio", [10.0, 100.0])
@pytest.mark.parametrize("high_first", [False, True])
@pytest.mark.parametrize("remote_cr", [False, True])
def test_candidate_group_uncertainty_is_spatially_local(
    noise_ratio,
    high_first,
    remote_cr,
):
    size = 65
    low_y, high_y = (44, 20) if high_first else (20, 44)
    frames = [_frame(_wcs(size), size, index) for index in range(3)]
    for frame in frames:
        frame["sigma"][high_y - 11:high_y + 12, 21:44] = noise_ratio * 1.0e-6
    frames[0]["data"][low_y, 32] += 200.0e-6
    frames[0]["data"][low_y, 33] += 100.0e-6
    if remote_cr:
        frames[0]["data"][high_y, 32] += 200.0 * noise_ratio * 1.0e-6

    native_fit_campaign.apply_consensus_cr_masks(frames)

    expected = np.zeros((size, size), dtype=bool)
    expected[low_y, 32:34] = True
    if remote_cr:
        expected[high_y, 32] = True
    np.testing.assert_array_equal(frames[0]["cr_mask"], expected)


@pytest.mark.parametrize("sampling", ["phase", "roll"])
@pytest.mark.parametrize("gradient", [-1.0e-5, 0.0, 1.0e-5])
@pytest.mark.parametrize("offset", [0, 1])
@pytest.mark.parametrize("amplitude_sigma", [0.0, 20.0, 200.0])
def test_changed_sampling_localization_removes_affine_sky(
    sampling,
    gradient,
    offset,
    amplitude_sigma,
):
    size = 49
    yy, xx = np.indices((size, size), dtype=float)
    frames = []
    for index in range(3):
        phase = 0.5 if index and sampling == "phase" else 0.0
        transform = _wcs(size, phase)
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
        sky_ra, _ = transform.pixel_to_world_values(xx, yy)
        sky_x = (
            (sky_ra - config.TRANSIENT_RA)
            * np.cos(np.deg2rad(config.TRANSIENT_DEC))
            / (1.22 / 3600.0)
        )
        entry["data"] = (
            0.01
            + 0.002 * _integrated_gaussian((size, size), x0, y0)
            + gradient * sky_x
        )
        frames.append(entry)
    frames[0]["data"][24, 24 + offset] += amplitude_sigma * 1.0e-6

    native_fit_campaign.apply_consensus_cr_masks(frames)

    expected = np.zeros((size, size), dtype=bool)
    if amplitude_sigma > 0.0:
        expected[24, 24 + offset] = True
    np.testing.assert_array_equal(frames[0]["cr_mask"], expected)
    assert frames[0]["cr_localization_status"] == "supported"


def _catalog_row(image: np.ndarray, transform: WCS, source: SkyCoord):
    x, y = transform.world_to_pixel_values(source.ra.deg, source.dec.deg)
    pixel_scale = 3600.0 * np.sqrt(
        abs(np.linalg.det(transform.pixel_scale_matrix)),
    )
    radius = 4.88 / pixel_scale
    flux = sep.sum_circle(
        np.ascontiguousarray(image, dtype=np.float64),
        np.asarray([x]),
        np.asarray([y]),
        radius,
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
@pytest.mark.parametrize("amplitude_sigma", [200.0, 2000.0])
@pytest.mark.parametrize("contaminated_frames", [1, 2, 3])
def test_catalog_annulus_artifacts_are_rejected_before_affine_projection(
    monkeypatch,
    sr,
    amplitude_sigma,
    contaminated_frames,
):
    native_size = 49
    native_wcs = _wcs(native_size)
    catalog_size = native_size * sr
    catalog_wcs = _wcs(catalog_size, scale_arcsec=1.22 / sr)
    source = SkyCoord(
        config.TRANSIENT_RA, config.TRANSIENT_DEC, unit="deg",
    )
    oversample = 20
    grid_y, grid_x = np.mgrid[-100:101, -100:101] / oversample
    prf = np.exp(-(grid_x**2 + grid_y**2) / (2.0 * 0.9**2))
    prf /= prf.sum()
    monkeypatch.setattr(config, "PRF_OVERSAMPLE", oversample)
    monkeypatch.setattr(
        config, "PRF_PIXEL_SCALE_ARCSEC", 1.22 / oversample,
    )
    monkeypatch.setattr(config, "PRF_PREBLUR_IN_OVERSAMPLE_PIXELS", 0.0)
    monkeypatch.setattr(config, "PRF_APPLY_PMAP_GAIN", False)
    monkeypatch.setattr(solver, "load_prf", lambda *_args, **_kwargs: prf)
    response = solver.native_point_source_column(
        catalog_wcs,
        native_wcs,
        (catalog_size, catalog_size),
        (native_size, native_size),
        "ch2",
        source.ra.deg,
        source.dec.deg,
    ).reshape(native_size, native_size)
    x0, y0 = native_wcs.world_to_pixel_values(source.ra.deg, source.dec.deg)
    x0 = int(np.rint(x0))
    y0 = int(np.rint(y0))
    offsets = ((0, 8), (8, 0), (0, -8))
    raw_frames = []
    for index, (dy, dx) in enumerate(offsets):
        data = 0.001 + 0.002 * response
        if index < contaminated_frames:
            data[y0 + dy, x0 + dx] += amplitude_sigma * 1.0e-6
        raw_frames.append({
            "data": data / config.MJY_SR_TO_JY,
            "sigma": np.full(data.shape, 1.0e-6) / config.MJY_SR_TO_JY,
            "wcs": native_wcs,
            "filename": f"review28_annulus_{index}_ch2.fits",
            "mjd": index / 10.0,
            "is_full_array": False,
        })
    iterator = iter(raw_frames)
    monkeypatch.setattr(preprocessing, "load_data", lambda _entry: next(iterator))
    processed = preprocessing.reproject_to_grid(
        [{} for _ in raw_frames],
        catalog_wcs,
        (catalog_size, catalog_size),
    )
    median, _ = preprocessing.create_median_stack(
        np.asarray([entry["data"] for entry in processed]),
    )
    row = _catalog_row(median, catalog_wcs, source)

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
    np.testing.assert_allclose(prior, 0.002, rtol=5.0e-6)
