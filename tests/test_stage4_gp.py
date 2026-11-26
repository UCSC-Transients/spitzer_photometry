"""Required CI: exact algebra, positivity, leakage and failure-focused regressions."""
from dataclasses import replace

import numpy as np
import pytest
from scipy.stats import norm, truncnorm

from src.gp_identifiability import (
    StaticData, SamplingFailure, conditional_log_likelihood, dense_host_operator,
    fit_static, gaussian_update, input_digest, matern_covariance, mixture_moments,
    model_weights, template_design,
    mixture_conditional_likelihood,
)
from src.stage4_validation import gp_fixture, fixture_design
from src.validation_metrics import write_artifact_bundle


def small_data(y=1.3, partition='train', verified=True):
    return StaticData(np.array([y]), np.array([[.3**2]]), np.ones((1, 1)),
                      np.zeros((1, 2)), (), ('t' if partition == 'train' else partition,),
                      partition, verified, 'synthetic scalar Gaussian; no real calibration')


def test_exact_forward_noise_evidence_and_conditional_identity():
    rng = np.random.default_rng(1729)
    a = rng.uniform(size=(8, 4))
    root = rng.normal(size=(8, 8))
    noise = root@root.T+np.eye(8)
    kroot = rng.normal(size=(4, 4))
    k = kroot@kroot.T+np.eye(4)
    mean, y = rng.normal(size=4), rng.normal(size=8)
    xy = np.column_stack([np.arange(4), np.zeros(4)])
    data = StaticData(y, noise, a, xy, (), ('t',), 'train', True, 'algebra')
    pm, pc, logz, _ = gaussian_update(data, mean, k)
    c = noise+a@k@a.T
    r = y-a@mean
    expected = -.5*(r@np.linalg.solve(c, r)+np.linalg.slogdet(c)[1]+8*np.log(2*np.pi))
    assert logz == pytest.approx(expected, abs=1e-12)
    np.testing.assert_allclose(pm, mean+k@a.T@np.linalg.solve(c, r), atol=2e-14)
    np.testing.assert_allclose(pc, k-k@a.T@np.linalg.solve(c, a@k), atol=2e-14)
    # Sequential likelihood counts each independent block once.
    n = np.diag(np.linspace(.3, .8, 8))
    all_data = replace(data, noise=n)
    first = replace(data, y=y[:4], noise=n[:4, :4], design=a[:4])
    second = replace(data, y=y[4:], noise=n[4:, 4:], design=a[4:], exposure_ids=('new',))
    m, v, z1, _ = gaussian_update(first, mean, k)
    _, _, z2, _ = gaussian_update(second, m, v)
    assert gaussian_update(all_data, mean, k)[2] == pytest.approx(z1+z2, abs=2e-12)


def test_positive_posterior_matches_truncated_normal_and_evidence():
    data = small_data(-.2)
    fit = fit_static(data, scale_arcsec=2., amplitude=.7, host_mean=np.array([.2]),
                     source_prior_sigma=1., seed=2718, draws=131072)
    m, c, logz, _ = gaussian_update(data, np.array([.2]), np.array([[.49]]))
    sd = np.sqrt(c[0, 0])
    truth = truncnorm(-m[0]/sd, np.inf, loc=m[0], scale=sd)
    assert fit.mean[0] == pytest.approx(truth.mean(), abs=5*fit.diagnostics['mean_mc_se'][0])
    assert fit.covariance[0, 0] == pytest.approx(truth.var(), rel=.025)
    exact_z = logz+norm.logcdf(m[0]/sd)-norm.logcdf(.2/.7)
    lo, hi = fit.diagnostics['log_evidence_95']
    assert lo < exact_z < hi
    assert fit.samples.min() >= 0
    assert np.quantile(fit.samples, .025) > 0  # genuine posterior interval, not clipped Gaussian
    new = small_data(.1, partition='heldout')
    score = fit.predictive_evidence(new, seed=31415, draws=131072)
    joint = replace(data, y=np.array([-.2, .1]), noise=np.eye(2)*.09, design=np.ones((2, 1)))
    jm, jc, jz, _ = gaussian_update(joint, np.array([.2]), np.array([[.49]]))
    joint_positive_z = jz+norm.logcdf(jm[0]/np.sqrt(jc[0, 0]))-norm.logcdf(.2/.7)
    assert abs(score['log_score']-(joint_positive_z-exact_z)) < .03


def test_absent_unverified_and_science_templates_fail():
    with pytest.raises(ValueError, match='Nonempty'):
        replace(small_data(), y=np.array([]), noise=np.empty((0, 0)), design=np.empty((0, 1)))
    for data in (small_data(verified=False), small_data(partition='science')):
        with pytest.raises(ValueError, match='verified training'):
            fit_static(data, scale_arcsec=2., amplitude=.7, host_mean=np.array([.2]),
                       source_prior_sigma=1., seed=1)


def test_rejection_failure_retains_counts_and_never_fake_errors():
    with pytest.raises(SamplingFailure) as error:
        fit_static(small_data(-10), scale_arcsec=2., amplitude=.7, host_mean=np.array([.2]),
                   source_prior_sigma=1., seed=1, draws=1024)
    assert error.value.diagnostics['accepted'] == 0


def test_prediction_retains_host_uncertainty_and_rejects_template_reuse():
    fit = fit_static(small_data(), scale_arcsec=2., amplitude=.7, host_mean=np.array([.2]),
                     source_prior_sigma=1., seed=1)
    with pytest.raises(ValueError, match='twice'):
        fit.predict(small_data())
    new = small_data(partition='science')
    _, covariance = fit.predict(new)
    assert covariance[0, 0] > new.noise[0, 0]
    with pytest.raises(ValueError, match='calibration gauge'):
        fit.predict(replace(new, calibration_gauge=('changed-normalization',)))
    logl, ess = conditional_log_likelihood(fit, new, np.ones(1), .1)
    assert np.isfinite(logl) and ess > 100
    combined = mixture_conditional_likelihood((fit, fit), (new, new), np.array([.25, .75]), np.ones(1), .1)
    assert combined['log_likelihood'] == pytest.approx(logl)
    assert combined['status'] == 'evaluated'
    with pytest.raises(ValueError, match='nonnegative'):
        conditional_log_likelihood(fit, new, np.ones(1), -.1)
    with pytest.raises(ValueError, match='template partitions'):
        fit.predictive_evidence(new, seed=1)


def test_exact_coincident_components_report_nonidentifiability():
    d = StaticData(np.array([2.]), np.array([[.01]]), np.ones((1, 3)), np.zeros((1, 2)),
                   ('star_0', 'nucleus'), ('t',), 'train', True, 'coincident negative control')
    fit = fit_static(d, scale_arcsec=1., amplitude=.1, host_mean=np.array([.1]),
                     source_prior_sigma=2., seed=1729, draws=32768)
    assert fit.diagnostics['likelihood_rank_deficient']
    corr = fit.covariance[1, 2]/np.sqrt(fit.covariance[1, 1]*fit.covariance[2, 2])
    assert corr < -.9
    assert np.std(fit.samples[:, 1]) > .4
    # Constant contamination can be exactly a static source: do not assert detection.
    np.testing.assert_array_equal(d.design[:, 1], d.design[:, 2])


def test_physical_gp_units_and_sampling_budget():
    xy = np.array([[0., 0.], [1.22, 0.]])
    np.testing.assert_array_equal(matern_covariance(xy, 2.44, 1e-5),
                                  matern_covariance(xy/4, .61, 1e-5))
    with pytest.raises(ValueError, match='budget'):
        dense_host_operator(lambda x: x, (17, 17), (17, 17))


def test_evidence_mixture_carries_between_model_uncertainty():
    d = small_data()
    fits = tuple(fit_static(d, scale_arcsec=s, amplitude=a, host_mean=np.array([.2]),
                            source_prior_sigma=1., seed=1) for s, a in [(1., .1), (3., .7)])
    weight_info = model_weights(fits, np.array([.5, .5]))
    w = weight_info['weights']
    m = mixture_moments(fits, w)
    assert m['between_covariance'][0, 0] > 0
    np.testing.assert_allclose(m['covariance'], m['within_covariance']+m['between_covariance'])
    bounds = weight_info['orthant_mc_weight_bounds']
    assert np.all(bounds[:, 0] <= w) and np.all(bounds[:, 1] >= w)
    with pytest.raises(ValueError, match='identical template'):
        model_weights((fits[0], replace(fits[1], data=replace(d, y=d.y+1))), np.ones(2))


@pytest.mark.parametrize('sr', [1, 2, 4])
def test_native_design_units_masks_and_geometry(sr):
    fixture = gp_fixture(sr=sr, masked=True)
    data = fixture_design(fixture, 'gp_stars_nucleus', 'train')
    assert data.design.shape == (3*78, 83)
    assert data.source_names == ('star_0', 'nucleus')
    assert np.array_equal(data.host_xy_arcsec, fixture_design(gp_fixture(), 'gp', 'train').host_xy_arcsec)
    frames, matrices, reference, xy, star, nucleus, _ = fixture
    changed = (replace(frames[0], registration_covariance=np.eye(2)*.01),)
    with pytest.raises(ValueError, match='Conditional geometry'):
        template_design(changed, reference, xy, (matrices[0],), stars_xy=np.array([star]),
                        nucleus_xy=nucleus, operator_provenance='test')


@pytest.mark.parametrize('sr', [1, 2, 4])
@pytest.mark.parametrize('phase', [-.37, .17, .43])
def test_injected_host_is_fixed_on_sky_across_roll_phase_and_sr(sr: int, phase: float) -> None:
    """Pull detector quadrature points back to one analytic sky, independently.

    The fixture uses WCS covariance transport and its own pixel integrator. This
    oracle instead uses the declared rolls/dithers and evaluates fixed sky
    densities directly, without either renderer or a WCS-derived covariance.
    Widths are the original optical-convolved widths at 1.22 arcsec/pixel;
    the isotropic .78-pixel detector optics commute with these pure rotations.
    """
    fixture = gp_fixture(sr=sr, phase=phase, gradient=2., masked=True)
    truth = fixture[-1]
    nodes, weights = np.polynomial.legendre.leggauss(16)
    yy, xx = np.indices((9, 9))
    detector = np.stack(np.broadcast_arrays(xx[..., None, None]+nodes[None, None, :, None]/2,
                                           yy[..., None, None]+nodes[None, None, None, :]/2), axis=-1)

    def integrate(points: np.ndarray, center: np.ndarray, widths: np.ndarray, flux: float) -> np.ndarray:
        density = flux*np.exp(-.5*np.sum(((points-center)/widths)**2, axis=-1))/(2*np.pi*np.prod(widths))
        return np.einsum('ijab,a,b->ij', density, weights, weights)/4

    for i, roll in enumerate((0, 31, 79, 47, 137)):
        angle = np.deg2rad(roll)
        rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
        native_origin = np.array([4.+phase+.11*i, 4.-.13*i])
        # Same TAN tangent point and native angular scale: a rigid sky pullback.
        sky = (detector-native_origin)@rotation.T+4.
        expected = integrate(sky, np.array([3.6, 4.4]), np.array([5., 5.]), .025)
        expected += integrate(sky, np.array([5.7, 2.5]), np.array([1.8, 3.]), .006)
        np.testing.assert_allclose(truth['host'][i], expected, rtol=1e-7, atol=1e-12)
        # Optical point-source images stay native and unchanged by reference SR.
        star_center = native_origin+rotation.T@np.array([2., .2])
        np.testing.assert_allclose(truth['star'][i],
                                   integrate(detector, star_center, np.array([.78, .78]), .002),
                                   rtol=1e-7, atol=1e-12)
        np.testing.assert_allclose(truth['nucleus'][i],
                                   integrate(detector, native_origin, np.array([.78, .78]), .004),
                                   rtol=1e-7, atol=1e-12)


def test_small_independent_leakage_and_template_contamination(tmp_path):
    fits, records = [], []
    for flux in (.001, .003):
        fixture = gp_fixture(flux=flux)
        data = fixture_design(fixture, 'gp_stars_nucleus', 'train')
        fit = fit_static(data, scale_arcsec=3.66, amplitude=5e-5, host_mean=np.full(81, 1.2e-4),
                         source_prior_sigma=.01, seed=1729, draws=8192)
        fits.append(fit)
        records.append(dict(flux=flux, recovered=float(fit.mean[-2]), digest=input_digest(data),
                            diagnostics=fit.diagnostics))
    slope = (fits[1].mean[-2]-fits[0].mean[-2])/.002
    # Numerical smoke bound only; scientific tolerances remain unapproved.
    assert .8 < slope < 1.2
    assert fits[0].covariance[40, 40] > 0
    clean = fixture_design(gp_fixture(flux=.003), 'gp_stars_nucleus', 'heldout')
    contaminated = fixture_design(gp_fixture(flux=.003, contaminated=True), 'gp_stars_nucleus', 'train')
    bad = fit_static(contaminated, scale_arcsec=3.66, amplitude=5e-5, host_mean=np.full(81, 1.2e-4),
                     source_prior_sigma=.01, seed=1729, draws=8192)
    score = fits[1].predictive_evidence(clean, seed=2718, draws=4096)
    bad_score = bad.predictive_evidence(clean, seed=2718, draws=4096)
    assert score['log_score'] > bad_score['log_score']+10
    assert bad.diagnostics['static_mismatch']
    write_artifact_bundle(tmp_path/'stage4', metrics=dict(records=records, slope=slope,
                          clean_score=score, contamination_score=bad_score, science_promotion_approved=False),
                          arrays=dict(native_data=data.y[:81].reshape(9, 9), covariance=fits[0].covariance, samples=fits[0].samples,
                                      contaminated_covariance=bad.covariance),
                          labels={'truth': 'independent analytic synthetic instrument'})


@pytest.mark.real_calibration
def test_real_native_source_columns_unchanged():
    """Real-table algebra only: fixed F is supplied, no real host-fit claim."""
    import os
    from pathlib import Path
    from src import config
    from src.point_response import _read_snapshot, file_identity
    from src.isolated_astrometry import StarFrame, star_response
    from src.nucleus import TemplateFrame
    from src.stage2_validation import validation_wcs
    path = Path(os.environ.get('SPITZER_STAGE2_PRF_DIR', config.PRF_DIR))/'apex_sh_IRACPC2_col129_row129_x100.fits'
    if not path.is_file():
        pytest.skip('Real warm PRF unavailable; no synthetic substitution')
    table = _read_snapshot(file_identity(path), 'ch2', 'warm', 'legacy_integral')
    wcs = validation_wcs()
    image = StarFrame(np.zeros((7, 7)), np.full((7, 7), 1e-5), wcs, table)
    frame = TemplateFrame(image, 'real-table-algebra', 'train', True, provenance='calibration-table algebra only')
    star, nucleus = np.array([3.13, 3.24]), np.array([4.11, 2.7])
    host = np.ones((49, 1))
    data = template_design((frame,), wcs, np.zeros((1, 2)), (host,), stars_xy=np.array([star]),
                            nucleus_xy=nucleus, operator_provenance='fixed uniform F for column identity only')
    np.testing.assert_array_equal(data.design[:, 1], star_response(image, wcs, star, np.zeros(2)).ravel())
    np.testing.assert_array_equal(data.design[:, 2], star_response(image, wcs, nucleus, np.zeros(2)).ravel())
    np.testing.assert_array_equal(data.design[:, :1], host)


def test_uncertainty_and_evidence_sensitive_to_declared_priors():
    data = StaticData(np.array([2.]), np.array([[.01]]), np.ones((1, 3)), np.zeros((1, 2)),
                      ('star_0', 'nucleus'), ('t',), 'train', True, 'unresolved prior-sensitivity control')
    fits = [fit_static(data, scale_arcsec=1., amplitude=.1, host_mean=np.array([.1]),
                       source_prior_sigma=s, seed=2718, draws=16384) for s in (.3, 3.)]
    assert abs(fits[0].log_evidence-fits[1].log_evidence) > 1
    assert fits[1].covariance[1, 1] > 2*fits[0].covariance[1, 1]
    assert all(fit.diagnostics['likelihood_rank_deficient'] for fit in fits)


def test_established_host_operator_identity_and_input_immutability():
    from src import solver
    from src.stage4_validation import _geometry, synthetic_host_context
    from src.stage2_validation import validation_wcs
    reference, _, native, matrices = _geometry(1, .17, False)
    host_wcs = validation_wcs()
    host_wcs.wcs.crpix = [5., 5.]
    scene = np.random.default_rng(1729).uniform(size=(9, 9))
    before = scene.copy()
    with synthetic_host_context():
        direct = solver._apply_frame_forward_operator(scene, host_wcs, native[0], (9, 9), (9, 9), 'ch2', is_full_array=True)
    np.testing.assert_allclose(matrices[0]@scene.ravel(), direct.ravel(), atol=2e-15, rtol=2e-15)
    np.testing.assert_array_equal(scene, before)
    data = fixture_design(gp_fixture(), 'gp_stars_nucleus', 'train')
    with pytest.raises(ValueError):
        data.design[0, 0] = 0
