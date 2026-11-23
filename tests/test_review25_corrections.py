"""Behavioral regressions for the fifth independent review of PR #22."""
from __future__ import annotations

import inspect
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
        "filename": f"review25_{index}_ch2.fits",
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
    x_integral = 0.5 * (
        erf((xx - x0 + 0.5) / scale) - erf((xx - x0 - 0.5) / scale)
    )
    y_integral = 0.5 * (
        erf((yy - y0 + 0.5) / scale) - erf((yy - y0 - 0.5) / scale)
    )
    return x_integral * y_integral


@pytest.mark.parametrize("sampling", ["same", "quarter", "half", "roll"])
@pytest.mark.parametrize("amplitude_sigma", [0.0, 20.0, 80.0, 200.0, 2000.0])
def test_consensus_cr_phase_roll_controls_and_injections(sampling, amplitude_sigma):
    size = 25
    phase = {"same": 0.0, "quarter": 0.25, "half": 0.5, "roll": 0.0}[sampling]
    frames = []
    for index in range(3):
        transform = _wcs(size, phase if index else 0.0)
        if sampling == "roll" and index:
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

    expected = [0, 0, 0] if amplitude_sigma == 0.0 else [1, 0, 0]
    assert [int(frame["cr_mask"].sum()) for frame in frames] == expected
    assert bool(frames[0]["cr_mask"][12, 12]) == (amplitude_sigma > 0.0)


@pytest.mark.parametrize("peer_width", [1.2, 1.8])
def test_consensus_cr_preserves_clean_spatial_prf_changes(peer_width):
    size = 256
    frames = []
    for index, (shift, width) in enumerate(
        zip((-60.0, 0.0, 60.0), (0.5, peer_width, peer_width)),
    ):
        transform = _wcs(size, shift, 0.0)
        x0, y0 = transform.world_to_pixel_values(
            config.TRANSIENT_RA, config.TRANSIENT_DEC,
        )
        entry = _frame(transform, size, index)
        entry["data"] += 0.002 * _integrated_gaussian(
            (size, size), x0, y0, width,
        )
        frames.append(entry)

    native_fit_campaign.apply_consensus_cr_masks(frames)

    assert [int(frame["cr_mask"].sum()) for frame in frames] == [0, 0, 0]


def test_consensus_cr_extreme_peer_annulus_contamination_is_symmetric():
    frames = [_frame(_wcs(31), 31, index) for index in range(3)]
    frames[0]["data"][15, 15] += 200.0e-6
    frames[1]["data"][15, 23] += 100_000.0e-6

    native_fit_campaign.apply_consensus_cr_masks(frames)

    assert [int(frame["cr_mask"].sum()) for frame in frames] == [1, 1, 0]
    assert frames[0]["cr_mask"][15, 15]
    assert frames[1]["cr_mask"][15, 23]


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


@pytest.mark.parametrize("sr", [1, 2, 4])
def test_catalog_matched_median_rejects_crs_with_backgrounds_prfs_and_masks(
    monkeypatch,
    sr,
):
    truth = 0.002
    native_size = 96
    source = SkyCoord(config.TRANSIENT_RA, config.TRANSIENT_DEC, unit="deg")
    native_wcs = [_wcs(native_size, shift, 0.0) for shift in (-20.0, 0.0, 20.0)]
    detector_x = np.asarray([
        float(np.asarray(
            transform.world_to_pixel_values(source.ra.deg, source.dec.deg)[0],
        ).ravel()[0])
        for transform in native_wcs
    ])

    def supplied_prf(_channel, x_det=None, y_det=None):
        del y_det
        index = int(np.argmin(np.abs(detector_x - float(x_det))))
        yy, xx = np.mgrid[-100:101, -100:101] / 20.0
        width = (0.5, 0.9, 1.4)[index]
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
    offsets = ((0, 2), (2, 0), (0, -2))
    raw_frames = []
    for index, (transform, response, background, offset) in enumerate(
        zip(native_wcs, responses, backgrounds, offsets),
    ):
        data = background + truth * response
        x0, y0 = transform.world_to_pixel_values(source.ra.deg, source.dec.deg)
        x0 = float(np.asarray(x0).ravel()[0])
        y0 = float(np.asarray(y0).ravel()[0])
        data[int(round(y0)) + offset[0], int(round(x0)) + offset[1]] += 200.0e-6
        sigma = np.full((native_size, native_size), 1.0e-6)
        if index == 0:
            sigma[int(round(y0)), int(round(x0))] = np.inf
        raw_frames.append({
            "data": data / config.MJY_SR_TO_JY,
            "sigma": sigma / config.MJY_SR_TO_JY,
            "wcs": transform,
            "filename": f"review25_catalog_{index}_ch2.fits",
            "mjd": float(index),
            "is_full_array": True,
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
    np.testing.assert_allclose(prior, truth, rtol=3.0e-3)
    assert fraction > 0.0

    fit_wcs = _wcs(9)
    fit_response = solver.native_point_source_column(
        fit_wcs,
        fit_wcs,
        (9, 9),
        (9, 9),
        "ch2",
        source.ra.deg,
        source.dec.deg,
    ).reshape(9, 9)
    fit_frame = _frame(fit_wcs, 9, 0)
    fit_frame["data"] += truth * fit_response
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
    np.testing.assert_allclose(result["star_fluxes"], [truth], rtol=3.0e-3)
    assert result["star_prior_equivalent_exposures"] == pytest.approx(5.0)


def _memory_fixture():
    native_wcs = [_wcs(100, index * 0.1) for index in range(2)]
    frames = [_frame(transform, 100, index) for index, transform in enumerate(native_wcs)]
    scene_wcs = _wcs(15)
    stars = [
        SkyCoord(
            *native_wcs[0].pixel_to_world_values(
                float(10 + 9 * (index % 10)),
                float(20 + 12 * (index // 10)),
            ),
            unit="deg",
        )
        for index in range(50)
    ]
    return frames, scene_wcs, stars


def _memory_options() -> dict:
    return {
        "USE_SCENE_GP_PRIOR": False,
        "SCENE_INDEPENDENT_RIDGE": 1.0e10,
        "ENFORCE_GP_CENTRAL_MONOTONICITY": False,
        "FLOAT_TRANSIENT_POSITION": False,
        "TRANSIENT_NONNEGATIVE": False,
        "USE_HOST_GAUSSIAN_CORE": False,
        "USE_NUCLEAR_POINT_SOURCE": False,
        "PRF_ORDER_PROJECT_THEN_CONVOLVE": False,
        "PRF_OPERATOR_MODE": "anchor",
        "PRF_SPATIAL_ANCHORS_PER_AXIS": 1,
        "PRF_OVERSAMPLE": 20,
        "PRF_PIXEL_SCALE_ARCSEC": 1.22 / 20.0,
        "PRF_PREBLUR_IN_OVERSAMPLE_PIXELS": 0.0,
        "PRF_APPLY_PMAP_GAIN": False,
        "SUPERSAMPLE_FACTOR": 1,
    }


def _test_prf():
    yy, xx = np.mgrid[-100:101, -100:101] / 20.0
    response = np.exp(-(xx**2 + yy**2) / (2.0 * 0.9**2))
    return response / response.sum()


def test_interframe_star_arrays_are_released_below_accepted_cap():
    frames, scene_wcs, stars = _memory_fixture()
    input_ids = {id(frame[key]) for frame in frames for key in ("data", "sigma")}
    observations = []

    def observe(inputs, output):
        current = inspect.currentframe().f_back
        active_frames = []
        while current is not None:
            if current.f_code.co_name in ("build_system", "run_gls_solve"):
                active_frames.append(current)
            current = current.f_back
        if not active_frames:
            return
        roots = {}

        def collect(value):
            if isinstance(value, np.ndarray):
                while isinstance(value.base, np.ndarray):
                    value = value.base
                if id(value) not in input_ids:
                    roots[id(value)] = value.nbytes
            elif isinstance(value, dict):
                for item in value.values():
                    collect(item)
            elif isinstance(value, (tuple, list)):
                for item in value:
                    collect(item)

        for active in active_frames:
            collect(active.f_locals)
        collect(inputs)
        collect(output)
        build = next(
            active for active in active_frames if active.f_code.co_name == "build_system"
        )
        observations.append({
            "frame_index": int(build.f_locals["i"]),
            "live_bytes": int(sum(roots.values())),
            "previous_stack_present": "S" in build.f_locals,
        })

    class TraceDesign(np.ndarray):
        def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
            base = tuple(
                np.asarray(value) if isinstance(value, TraceDesign) else value
                for value in inputs
            )
            output = getattr(ufunc, method)(*base, **kwargs)
            if ufunc == np.multiply and getattr(output, "shape", None) == (10_000, 225):
                observe(base, output)
            return output

    original = solver.frame_design_matrix

    def traced(*args, **kwargs):
        return original(*args, **kwargs).view(TraceDesign)

    with native_fit_campaign._temporary_config(_memory_options()), patch.object(
        solver, "load_prf", return_value=_test_prf(),
    ):
        estimate = solver._estimate_solver_peak_memory_bytes(
            frames,
            (15, 15),
            n_scene_blocks=1,
            n_stars=50,
            n_science_epochs=0,
        )
        options = {"SOLVER_DENSE_MEMORY_LIMIT_BYTES": estimate["peak_bytes"] + 1}
        with native_fit_campaign._temporary_config(options), patch.object(
            solver, "frame_design_matrix", side_effect=traced,
        ):
            traced_result = solver.run_gls_solve(
                frames,
                stars,
                np.zeros(50),
                {},
                (1.0, 1.0e-7),
                np.zeros((15, 15)),
                scene_wcs,
                1,
            )
        reference_result = solver.run_gls_solve(
            frames,
            stars,
            np.zeros(50),
            {},
            (1.0, 1.0e-7),
            np.zeros((15, 15)),
            scene_wcs,
            1,
        )

    assert len(observations) == 2
    assert max(row["live_bytes"] for row in observations) <= estimate["peak_bytes"]
    assert not any(row["previous_stack_present"] for row in observations)
    np.testing.assert_allclose(
        traced_result["gp_scene"], reference_result["gp_scene"], rtol=0.0, atol=0.0,
    )
    np.testing.assert_allclose(
        traced_result["star_fluxes"], reference_result["star_fluxes"], rtol=0.0, atol=0.0,
    )
    np.testing.assert_allclose(
        traced_result["bcd_backgrounds"],
        reference_result["bcd_backgrounds"],
        rtol=0.0,
        atol=0.0,
    )


def test_multiframe_memory_guard_rejects_before_scene_allocation(monkeypatch):
    frames, scene_wcs, stars = _memory_fixture()
    with native_fit_campaign._temporary_config(_memory_options()), patch.object(
        solver, "load_prf", return_value=_test_prf(),
    ):
        estimate = solver._estimate_solver_peak_memory_bytes(
            frames,
            (15, 15),
            n_scene_blocks=1,
            n_stars=50,
            n_science_epochs=0,
        )
        monkeypatch.setattr(
            config, "SOLVER_DENSE_MEMORY_LIMIT_BYTES", estimate["peak_bytes"] - 1,
        )
        monkeypatch.setattr(
            solver,
            "frame_design_matrix",
            lambda *args, **kwargs: pytest.fail("guard ran after scene allocation"),
        )
        with pytest.raises(MemoryError, match="Dense fit peak"):
            solver.run_gls_solve(
                frames,
                stars,
                np.zeros(50),
                {},
                (1.0, 1.0e-7),
                np.zeros((15, 15)),
                scene_wcs,
                1,
            )


@pytest.mark.parametrize(
    ("native_shape", "scene_shape", "star_count"),
    [((40, 60), (5, 7), 17), ((15, 21), (9, 11), 3)],
)
def test_successor_scene_allocation_has_no_previous_star_arrays(
    native_shape,
    scene_shape,
    star_count,
):
    native_wcs = _wcs(max(native_shape)).slice((
        slice(0, native_shape[0]), slice(0, native_shape[1]),
    ))
    scene_wcs = _wcs(max(scene_shape)).slice((
        slice(0, scene_shape[0]), slice(0, scene_shape[1]),
    ))
    frames = []
    for index in range(2):
        entry = _frame(native_wcs.deepcopy(), max(native_shape), index)
        entry["data"] = np.full(native_shape, 1.0e-3)
        entry["sigma"] = np.full(native_shape, 1.0e-6)
        frames.append(entry)
    stars = [
        SkyCoord(
            *native_wcs.pixel_to_world_values(
                float(2 + (index * 3) % max(3, native_shape[1] - 4)),
                float(2 + (index * 5) % max(3, native_shape[0] - 4)),
            ),
            unit="deg",
        )
        for index in range(star_count)
    ]
    original = solver.frame_design_matrix
    retained_stack = []

    def traced(*args, **kwargs):
        current = inspect.currentframe().f_back
        while current is not None and current.f_code.co_name != "build_system":
            current = current.f_back
        if current is not None:
            retained_stack.append("S" in current.f_locals)
        return original(*args, **kwargs)

    with native_fit_campaign._temporary_config(_memory_options()), patch.object(
        solver, "load_prf", return_value=_test_prf(),
    ), patch.object(solver, "frame_design_matrix", side_effect=traced):
        result = solver.run_gls_solve(
            frames,
            stars,
            np.zeros(star_count),
            {},
            (1.0, 1.0e-7),
            np.zeros(scene_shape),
            scene_wcs,
            1,
        )

    assert retained_stack == [False, False]
    assert np.isfinite(result["gp_scene"]).all()
    assert np.isfinite(result["star_fluxes"]).all()
