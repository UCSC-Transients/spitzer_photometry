"""Required numerical regressions; all analytic tables are explicitly synthetic."""
import os

import numpy as np
import pytest
from astropy.io import fits

from src import config, solver
from src.isolated_astrometry import StarFrame, fit_isolated_stars, response_derivatives, star_response
from src.point_response import NativePRF, PRFGrid, file_identity, sample_native_response
from src.stage2_validation import validation_wcs as wcs, synthetic_integrated_prf as synthetic_table, isolated_star_fixture as recovery_case


@pytest.mark.parametrize("sr", [1, 2, 4])
@pytest.mark.parametrize("single", [False, True])
def test_independent_flux_centroid_registration(sr, single):
    frames, reference, initial, truth, flux, shifts = recovery_case(sr=sr, single=single)
    result = fit_isolated_stars(frames, initial, flux*.8, reference_wcs=reference, fit_registration=not single)
    np.testing.assert_allclose(result.flux, flux, rtol=8e-5)
    np.testing.assert_allclose(result.positions_xy/sr, truth/sr, atol=8e-5)
    np.testing.assert_allclose(result.translations_xy, shifts, atol=8e-5)
    assert np.all(result.flux >= 0)
    assert np.all(result.translations_xy[0] == 0)
    assert result.diagnostics["rank"] == result.diagnostics["n_parameters"]
    predictions = result.predict(frames)
    residual = np.concatenate([(pred[f.valid]-f.data[f.valid])/f.sigma[f.valid]
                               for pred, f in zip(predictions, frames)])
    assert residual @ residual == pytest.approx(result.diagnostics["chi2"], abs=1e-10)
    assert not result.diagnostics["absolute_calibration_verified"]


def test_derivatives_and_snapshot_immutability():
    frames, ref, initial, truth, flux, shifts = recovery_case()
    frame = frames[1]
    before = frame.data.copy()
    sky, det = response_derivatives(frame, ref, truth[0], shifts[1], step=1e-3)
    sky2, det2 = response_derivatives(frame, ref, truth[0], shifts[1], step=2e-4)
    np.testing.assert_allclose(sky, sky2, atol=2e-5)
    np.testing.assert_allclose(det, det2, atol=2e-5)
    for axis in range(2):
        step = np.eye(2)[axis]*1e-5
        numerical_sky = (star_response(frame, ref, truth[0]+step, shifts[1])
                         - star_response(frame, ref, truth[0]-step, shifts[1]))/2e-5
        numerical_detector = (star_response(frame, ref, truth[0], shifts[1]+step)
                              - star_response(frame, ref, truth[0], shifts[1]-step))/2e-5
        np.testing.assert_allclose(sky[axis], numerical_sky, atol=3e-6)
        np.testing.assert_allclose(det[axis], numerical_detector, atol=3e-6)
    assert not np.allclose(sky, det, atol=1e-3)  # roll separates sky and detector shifts
    with pytest.raises(ValueError):
        frame.response.values.setflags(write=True)
    source_wcs = wcs()
    frozen = StarFrame(np.ones((25, 25)), np.ones((25, 25)), source_wcs, synthetic_table())
    prior = star_response(frozen, ref, truth[0], shifts[0])
    source_wcs.wcs.crpix += 4
    np.testing.assert_array_equal(prior, star_response(frozen, ref, truth[0], shifts[0]))
    np.testing.assert_array_equal(before, frame.data)


def test_phase_and_crop_keep_response_not_unity(monkeypatch):
    yy, xx = np.mgrid[-400:401, -400:401]/50.
    # A detector-integrated synthetic table with a deliberate phase dependence.
    values = np.exp(-(xx**2+yy**2)/2)* (1+.12*np.cos(2*np.pi*xx))
    table = NativePRF(values, 50, synthetic=True)
    sums = [table.sample((21, 21), 10+phase, 10).sum() for phase in (0., .25, .5)]
    assert max(sums)-min(sums) > .2
    assert table.sample((5, 5), .1, 2).sum() < table.sample((21, 21), 10.1, 10).sum()*.9
    monkeypatch.setattr(solver, "load_prf", lambda *a: values)
    monkeypatch.setattr(config, "PRF_OVERSAMPLE", 50)
    native = wcs()
    ra, dec = native.pixel_to_world_values(10.2, 10.3)
    legacy = solver.native_point_source_column(native, native, (25, 25), (25, 25), "ch2", ra, dec)
    np.testing.assert_allclose(legacy.reshape(25, 25), table.sample((25, 25), 10.2, 10.3), atol=1e-10)


def test_explicit_aperture_convention_retains_phase():
    yy, xx = np.mgrid[-120:121, -120:121]/10.
    values = np.exp(-(xx**2+yy**2)/2)*(1+.12*np.cos(2*np.pi*xx))
    table = NativePRF(values, 10, "zero_phase_aperture", synthetic=True)
    yy, xx = np.indices((21, 21))
    aperture = (xx-10)**2+(yy-10)**2 <= 100
    assert table.sample((21, 21), 10, 10)[aperture].sum() == pytest.approx(1.)
    assert table.sample((21, 21), 10.5, 10)[aperture].sum() < .9


def test_calibration_cache_invalidates_restored_mtime(tmp_path, monkeypatch):
    path = tmp_path/"a.fits"
    fits.writeto(path, np.ones((5, 5)))
    monkeypatch.setattr(config, "PRF_DIR", str(tmp_path))
    before = solver._prf_cache_fingerprint("ch2")
    old = solver._read_fits_image_array(path).copy()
    stat = path.stat()
    fits.writeto(path, np.eye(5)+1, overwrite=True)
    os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns))
    assert solver._prf_cache_fingerprint("ch2") != before
    assert not np.array_equal(old, solver._read_fits_image_array(path))
    warm = solver._read_fits_image_array(path)
    solver._read_fits_image_cached.cache_clear()
    np.testing.assert_array_equal(warm, solver._read_fits_image_array(path))
    oldkey = file_identity(path)
    replacement = tmp_path/"b.fits"
    fits.writeto(replacement, np.ones((5, 5))*3)
    replacement.replace(path)
    assert oldkey != file_identity(path)


def test_missing_calibration_and_registration_gauge(tmp_path):
    grid = PRFGrid(str(tmp_path), "ch2")
    with pytest.raises(FileNotFoundError):
        grid.sample((5, 5), 2, 2, 128, 128)
    with pytest.raises(ValueError, match="provenance"):
        StarFrame(np.ones((5, 5)), np.ones((5, 5)), wcs(), grid)
    frames, ref, initial, truth, flux, shifts = recovery_case(single=True)
    with pytest.raises(ValueError, match="multiple stars"):
        fit_isolated_stars(frames*2, initial, flux)


def test_single_iteration_is_failure():
    frames, ref, initial, truth, flux, shifts = recovery_case()
    with pytest.raises(RuntimeError, match="converge"):
        fit_isolated_stars(frames, initial, flux*.8, max_nfev=1)


def test_rank_deficiency_does_not_report_precise_errors():
    table = NativePRF(np.ones((101, 101)), 1, "table", synthetic=True)
    frame = StarFrame(np.full((5, 5), .01), np.full((5, 5), 1e-5), wcs(), table)
    result = fit_isolated_stars((frame,), np.array([[2., 2.]]), np.array([.01]), fit_registration=False)
    assert result.diagnostics["rank"] < result.diagnostics["n_parameters"]
    assert np.isnan(result.covariance).all()


def test_reserved_seeds_are_not_generated():
    from src.validation_fixtures import ACCEPTANCE_SEEDS
    for seed in ACCEPTANCE_SEEDS:
        with pytest.raises(ValueError, match="development"):
            recovery_case(seed=seed)


def test_noisy_development_recovery_converges():
    # This noisy trial exposed finite differences crossing interpolation cells.
    frames, reference, initial, truth, flux, shifts = recovery_case(seed=1729)
    fit = fit_isolated_stars(frames, initial, flux*.8, reference_wcs=reference)
    assert fit.diagnostics["optimality_scaled"] <= 1e-3
    assert np.all(fit.flux >= 0)
    np.testing.assert_allclose(fit.translations_xy, shifts, atol=.03)


def test_fit_rejects_changed_prediction_geometry():
    frames, ref, initial, truth, flux, shifts = recovery_case(single=True)
    result = fit_isolated_stars(frames, initial, flux, fit_registration=False)
    frames[0].wcs.wcs.crpix += .1
    with pytest.raises(ValueError, match="changed"):
        result.predict(frames)


def test_full_detector_grid_coordinates_and_weights(tmp_path, monkeypatch):
    from src import point_response
    table = NativePRF(np.ones((5, 5)), 1, "table", synthetic=True)
    calls = []

    def read(identity, channel, mission, convention):
        calls.append(identity[0])
        return table

    monkeypatch.setattr(point_response, "_read_snapshot", read)
    monkeypatch.setattr(point_response, "file_identity", lambda p: (str(p),))
    grid = PRFGrid(str(tmp_path), "ch2")
    a = grid.sample((3, 3), 1, 1, 128, 128)
    assert len(calls) == 1 and "col129_row129" in calls[0]
    calls.clear()
    b = grid.sample((3, 3), 1, 1, 154, 154)
    assert len(calls) == 4
    np.testing.assert_array_equal(a, b)


@pytest.mark.real_calibration
def test_real_warm_grid_header_and_native_sampling():
    from pathlib import Path
    from src.point_response import _read_snapshot
    directory = Path(os.environ.get("SPITZER_STAGE2_PRF_DIR", config.PRF_DIR))
    path = directory/"apex_sh_IRACPC2_col129_row129_x100.fits"
    if not path.is_file():
        pytest.skip("Real warm calibration unavailable; scientific gate remains unmet")
    table = _read_snapshot(file_identity(path), "ch2", "warm", "legacy_integral")
    direct = table.values[79::100, 79::100]/table.normalization
    grid = PRFGrid(str(directory), "ch2")
    np.testing.assert_allclose(grid.sample((25, 25), 12, 12, 128, 128), direct, rtol=1e-14)
    with pytest.raises(ValueError, match="mission"):
        _read_snapshot(file_identity(path), "ch2", "cryogenic", "legacy_integral")


@pytest.mark.real_calibration
@pytest.mark.parametrize("channel", ["ch1", "ch2"])
def test_real_table_fit_prediction_closure_only(channel):
    """Same-renderer algebra with real tables, NOT independent flux calibration."""
    from pathlib import Path
    from src.point_response import _read_snapshot
    directory = Path(os.environ.get("SPITZER_STAGE2_PRF_DIR", config.PRF_DIR))
    path = directory/f"apex_sh_IRACPC{channel[-1]}_col129_row129_x100.fits"
    if not path.is_file():
        pytest.skip("Real warm calibration unavailable; scientific gate remains unmet")
    table = _read_snapshot(file_identity(path), channel, "warm", "legacy_integral")
    truth = np.array([[12.13, 11.73]])
    data = .01*table.sample((25, 25), *truth[0])+2e-5
    frame = StarFrame(data, np.full_like(data, 1e-5), wcs(), table)
    result = fit_isolated_stars((frame,), truth+np.array([.23, -.18]), np.array([.008]), fit_registration=False)
    np.testing.assert_allclose(result.flux, [.01], atol=1e-9)
    np.testing.assert_allclose(result.positions_xy, truth, atol=1e-6)
    np.testing.assert_allclose(result.predict((frame,))[0], data, atol=1e-9)
