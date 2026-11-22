"""Behavioral regressions identified in the November 2026 repository review."""
import numpy as np
import pytest
from astropy.io import fits
from astropy.wcs import WCS

from src import config, native_fit_campaign as campaign, preprocessing, residual_metrics, solver
from src.instrument import channel_from_filename, frame_channel
from src.prf_identity_context import identity_prf_operators_context


def make_wcs():
    w = WCS(naxis=2)
    w.wcs.crpix = [3, 3]
    w.wcs.crval = [config.TRANSIENT_RA, config.TRANSIENT_DEC]
    w.wcs.ctype = ['RA---TAN', 'DEC--TAN']
    w.wcs.cd = np.diag([-1.22, 1.22]) / 3600
    return w


@pytest.mark.parametrize('name,expected', [('SPITZER_I2_123_cbcd.fits', 'ch2'),
                                         ('SPITZER_I1_123_cbcd.fits', 'ch1'),
                                         ('synthetic_ch2_000.fits', 'ch2')])
def test_ingestion_and_model_channel_agree(name, expected):
    assert preprocessing.is_correct_channel(name, expected)
    assert frame_channel({'filename': name}) == expected


def test_channel_conflicts_raise():
    with pytest.raises(ValueError):
        frame_channel({'filename': 'SPITZER_I2_123.fits', 'channel': 'ch1'})
    with pytest.raises(ValueError):
        channel_from_filename('ch1_ch2.fits')
    assert channel_from_filename('notch2image.fits') is None


def test_cd_projection_cache_does_not_collide():
    w = make_wcs()
    v = w.deepcopy()
    v.wcs.cd *= 2
    assert solver._wcs_projection_cache_key(w) != solver._wcs_projection_cache_key(v)
    a = solver._projection_matrix(w, w, (5, 5), (5, 5))
    b = solver._projection_matrix(w, v, (5, 5), (5, 5))
    assert not np.allclose(a.toarray(), b.toarray())


def test_partial_prf_grid_uses_available_array(monkeypatch):
    prf = np.ones((3, 3))
    monkeypatch.setattr(solver, 'load_real_prf_file', lambda channel, row, col: prf.copy() if col == 1 else None)
    out = solver.load_interpolated_prf('ch2', 50, 50)
    np.testing.assert_allclose(out, np.full((3, 3), 1 / 9))


def test_cr_mask_survives_recovery(monkeypatch):
    w = make_wcs()
    c = {'data': np.ones((5, 5)), 'sigma': np.ones((5, 5)), 'wcs': w, 'raw_wcs': w}
    c['sigma'][0, 1] = np.inf
    cr = np.eye(5, dtype=bool)
    monkeypatch.setattr(campaign, '_cr_mask_local', lambda c: cr)
    campaign.apply_native_cutout_cr_mask(c)
    campaign.unmask_sigma_inf_in_radius(c, 20, 'transient')
    assert np.all(np.isinf(c['sigma'][cr]))
    assert c['sigma'][0, 1] == 1


def test_bad_fits_values_remain_masked(tmp_path):
    data = np.array([[np.nan, np.inf, 3.], [-np.inf, 2., 1.]])
    sigma = np.array([[1., 1., 0.], [1., -1., 2.]])
    image, unc = tmp_path / 'x.fits', tmp_path / 'u.fits'
    fits.writeto(image, data, make_wcs().to_header())
    fits.writeto(unc, sigma)
    loaded = preprocessing.load_data({'image': str(image), 'unc': str(unc)})
    assert np.isfinite(loaded['data']).all()
    assert solver.valid_fit_mask(loaded).sum() == 1


def test_constant_masked_residual_has_no_acf_structure():
    data = np.full((16, 16), 5.)
    mask = np.zeros_like(data, dtype=bool)
    mask[4:12, 4:12] = True
    assert np.count_nonzero(residual_metrics._acf2d_centered(data, mask)) == 0


def test_acf_half_height_finds_downward_crossing():
    yy, xx = np.mgrid[-20:21, -20:21]
    acf = np.exp(-(xx**2 + yy**2) / (2 * 3.**2))
    assert 3 < residual_metrics.acf_e90_scale_pix(acf) < 4.5


def test_epoch_total_span_and_warning():
    entries = [{'mjd': t, 'filename': str(t), 'is_template': False}
               for t in [0., .75, 1., 1.5, 2.5]]
    with pytest.warns(UserWarning, match='one-day'):
        count = preprocessing.assign_flux_epochs(entries)
    assert count == 2
    assert [c['epoch_id'] for c in entries] == [0, 0, 0, 1, 1]
    assert entries[3]['epoch_gap_warning']
    for eid in range(count):
        times = [c['mjd'] for c in entries if c['epoch_id'] == eid]
        assert max(times) - min(times) <= 1


def test_sparse_prior_solver_and_shared_epoch_flux(monkeypatch):
    w = make_wcs()
    monkeypatch.setattr(config, 'GP_KERNEL_TYPE', 'diagonal')
    monkeypatch.setattr(config, 'GP_DIAGONAL_EPS', 10.)
    monkeypatch.setattr(config, 'PRF_GLS_LTWL_FULL_MAX_PIXELS', 100)
    monkeypatch.setattr(config, 'ENFORCE_GP_CENTRAL_MONOTONICITY', False)
    monkeypatch.setattr(config, 'TRANSIENT_NONNEGATIVE', True)
    cutouts = []
    for i in range(3):
        data = np.ones((5, 5))
        if i < 2:
            data[2, 2] += 2 + i * .2
        cutouts.append({'data': data, 'sigma': np.full((5, 5), .1), 'wcs': w,
                        'raw_wcs': w, 'epoch_id': 0 if i < 2 else 1,
                        'mjd': i * .5, 'is_template': i == 2, 'filename': f'test_ch2_{i}.fits'})
    with identity_prf_operators_context():
        result = solver.run_gls_solve(cutouts, [], [], {'ell': 1., 'var': 1.},
                                      (1., 1.), np.zeros((5, 5)), w, 2)
    assert 'gp_scene' in result
    np.testing.assert_allclose(result['transient_fluxes'][:2], 2.1, atol=.03)
    assert result['transient_fluxes'][0] == result['transient_fluxes'][1]
    assert result['transient_fluxes'][2] == 0


def test_solver_failure_is_not_a_zero_scene(monkeypatch):
    w = make_wcs()
    monkeypatch.setattr(solver, '_build_central_annulus_terms', lambda *a: (_ for _ in ()).throw(ValueError('bad geometry')))
    c = {'data': np.ones((5, 5)), 'sigma': np.ones((5, 5)), 'wcs': w,
         'raw_wcs': w, 'filename': 'ch2.fits', 'epoch_id': 0, 'is_template': True}
    with pytest.raises(RuntimeError, match='no valid result'):
        solver.run_gls_solve([c], [], [], {}, (1., 1.), np.zeros((5, 5)), w, 1)


@pytest.mark.parametrize('phase', [0., .2, .5])
def test_native_point_flux_independent_of_scene_grid(monkeypatch, phase):
    native = make_wcs()
    native.wcs.crpix = [10. + phase, 10. + phase]
    prf = solver.get_gaussian_kernel_array(size=1001, fwhm=2.12, oversample=100)
    monkeypatch.setattr(solver, 'load_prf', lambda *a: prf)
    responses = []
    for sr in (1, 2, 4):
        scene = native.deepcopy()
        scene.wcs.cd /= sr
        responses.append(solver.native_point_source_column(scene, native, (19 * sr, 19 * sr),
                         (19, 19), 'ch2', config.TRANSIENT_RA, config.TRANSIENT_DEC))
    np.testing.assert_allclose(responses[0], responses[1], atol=0)
    np.testing.assert_allclose(responses[0], responses[2], atol=0)
    assert abs(responses[0].sum() - 1.) < 1e-4


def test_native_point_crop_keeps_aperture_loss(monkeypatch):
    prf = solver.get_gaussian_kernel_array(size=1001, fwhm=2.12, oversample=100)
    monkeypatch.setattr(solver, 'load_prf', lambda *a: prf)
    native = make_wcs()
    native.wcs.crpix = [1., 1.]
    col = solver.native_point_source_column(native, native, (5, 5), (5, 5), 'ch2',
                                           config.TRANSIENT_RA, config.TRANSIENT_DEC)
    assert .3 < col.sum() < .8


def test_consensus_cr_preserves_static_nucleus(monkeypatch):
    w = make_wcs()
    yy, xx = np.indices((9, 9))
    host = 1 + 100 * np.exp(-((xx - 4)**2 + (yy - 4)**2) / 2)
    cutouts = [{'data': host.copy() + i, 'sigma': np.ones((9, 9)), 'wcs': w,
                'raw_wcs': w, 'epoch_id': 0, 'mjd': i / 10, 'is_template': True}
               for i in range(4)]
    cutouts[0]['data'][2, 2] += 100
    campaign.apply_consensus_cr_masks(cutouts)
    assert cutouts[0]['cr_mask'][2, 2]
    assert all(not c['cr_mask'][4, 4] for c in cutouts)
    assert sum(c['cr_mask'].sum() for c in cutouts) == 1


def test_bounded_map_reoptimizes_coupled_parameters():
    h = np.array([[2., 1.], [1., 2.]])
    rhs = np.array([-2., 1.])
    result = solver._solve_map_bounds(h, rhs, np.array([0., -np.inf]), np.full(2, np.inf))
    np.testing.assert_allclose(result, [0., .5], atol=1e-6)


def test_floated_position_recovers_offset(monkeypatch):
    w = make_wcs()
    w.wcs.crpix = [5., 5.]
    monkeypatch.setattr(config, 'FLOAT_TRANSIENT_POSITION', True)
    monkeypatch.setattr(config, 'TRANSIENT_POS_RIDGE', 1e4)
    monkeypatch.setattr(config, 'TRANSIENT_NONNEGATIVE', True)
    monkeypatch.setattr(config, 'USE_SCENE_GP_PRIOR', True)
    monkeypatch.setattr(config, 'ENFORCE_GP_CENTRAL_MONOTONICITY', False)
    monkeypatch.setattr(config, 'USE_HOST_GAUSSIAN_CORE', False)
    monkeypatch.setattr(config, 'USE_NUCLEAR_POINT_SOURCE', False)
    ra, dec = w.pixel_to_world_values(4.2, 3.85)
    col = solver.native_point_source_column(w, w, (9, 9), (9, 9), 'ch2', ra, dec).reshape(9, 9)
    cutouts = [dict(data=.01 + (col if i < 2 else 0.) + np.zeros((9, 9)),
                    sigma=np.full((9, 9), .001), wcs=w, raw_wcs=w,
                    filename=f'position_ch2_{i}.fits', epoch_id=int(i >= 2),
                    mjd=i / 10, is_template=i >= 2) for i in range(4)]
    result = solver.run_gls_solve(cutouts, [], [], {'ell': 1., 'var': 1e-5},
                                  (1., 1e-5), np.zeros((9, 9)), w, 2)
    recovered = w.world_to_pixel_values(config.TRANSIENT_RA + result['transient_dra_deg'],
                                        config.TRANSIENT_DEC + result['transient_ddec_deg'])
    np.testing.assert_allclose(recovered, [4.2, 3.85], atol=.003)
    np.testing.assert_allclose(result['transient_fluxes'][:2], 1., atol=.003)


def test_metric_units_and_saved_noise(monkeypatch):
    from src import fit_metrics
    w = make_wcs()
    c = dict(data=np.full((5, 5), .004), sigma=np.full((5, 5), .001), wcs=w)
    monkeypatch.setattr(solver, 'predict_cutout_model', lambda *a, **kw: np.full((5, 5), .002))
    m = fit_metrics.compute_fit_metrics([c], {'effective_sigmas': [np.full((5, 5), .002)]}, [], [],
                                        center_ra_deg=config.TRANSIENT_RA, center_dec_deg=config.TRANSIENT_DEC)
    assert m['center_rmse'] == pytest.approx(.002)
    assert m['center_noise_ratio'] == pytest.approx(1.)
    assert np.isnan(m['center_poisson_proxy'])


def test_scene_shift_converts_to_native_pixels():
    from src.dipole_chi2_scan import shift_model_in_scene_pixels
    native = make_wcs()
    scene = native.deepcopy()
    scene.wcs.cd /= 2
    image = np.zeros((7, 7))
    image[3, 3] = 1.
    moved = shift_model_in_scene_pixels(image, native, scene, 2., 0.)
    assert np.unravel_index(moved.argmax(), moved.shape) == (3, 4)
    assert moved[3, 4] == pytest.approx(1., abs=1e-8)


def test_scene_mask_reprojects_native_coverage():
    from src.diagnostics import diagnostic_stretch_mask
    native = make_wcs()
    shifted = native.deepcopy()
    shifted.wcs.crpix[0] += 1
    c = dict(data=np.ones((5, 5)), sigma=np.ones((5, 5)), wcs=native)
    c['sigma'][2, 2] = np.inf
    mask = diagnostic_stretch_mask([c], (5, 5), shifted)
    assert not mask[2, 3]
    assert mask[2, 2]


def test_missing_calibration_fails_explicitly(monkeypatch):
    monkeypatch.setattr(solver, 'load_real_prf_file', lambda *a: None)
    with pytest.raises(FileNotFoundError, match='No PRF calibration'):
        solver.load_interpolated_prf('ch2', 128, 128)


def test_signed_dipole_detects_symmetric_positive_negative_lobes():
    a = np.zeros((7, 7))
    a[3, 2], a[3, 4] = -1., 1.
    dx, dy, amplitude = residual_metrics.dipole_moment_xy(a, np.ones_like(a, dtype=bool))
    assert dx == 1.
    assert dy == 0.
    assert amplitude == 1.


def test_legacy_postprocessing_preserves_shared_epoch_error(tmp_path, monkeypatch):
    from src.postprocessing import group_and_average
    monkeypatch.setattr(config, 'DIAGNOSTIC_DIR', str(tmp_path))
    files = [dict(mjd=t, filename=str(t), epoch_id=0, channel='ch2') for t in [0., .5]]
    results = dict(transient_fluxes=[1., 1.], transient_errs=[.2, .2],
                   transient_epoch_index_by_id={0: 0}, transient_epoch_fluxes=[1.],
                   transient_epoch_errs=[.2])
    _, summary = group_and_average(files, results)
    assert summary.iloc[0]['err'] == .2
    empty, empty_summary = group_and_average([], dict(transient_fluxes=[], transient_errs=[]))
    assert empty.empty and empty_summary.empty
