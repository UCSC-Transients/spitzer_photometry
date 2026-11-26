"""Independent dense-oracle, calibrated-operator and failure gates for Stage 5."""
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from scipy.linalg import block_diag
from scipy.optimize import lsq_linear
from scipy.stats import norm, truncnorm

from src import config, solver
from src.gp_identifiability import StaticData, fit_static, gaussian_update, matern_covariance
from src.scalable_operators import (Design, DenseNoise, BackgroundNoise, OperatorData,
                                    from_dense, frozen_host_operator, native_template_data)
from src.scalable_gp import (Budget, GridPrior, GaussianPosterior, NumericalFailure,
                            fit_positive, conditional_log_likelihood, model_weights,
                            mixture_moments, mixture_conditional_likelihood)
from src.stage4_validation import gp_fixture, fixture_design, synthetic_host_context
from src.stage2_validation import validation_wcs


def grid(shape=(2, 3), spacing=1.):
    y, x = np.indices(shape)
    return np.column_stack((x.ravel(), y.ravel()))*spacing


def algebra_fixture(shape=(2, 3), sources=2, n=12, *, negative=False):
    rng = np.random.default_rng(1729)
    xy = grid(shape)
    a = rng.uniform(.01, .3, (n, len(xy)+sources))
    noise_root = rng.normal(size=(n, n))*.03
    noise = noise_root@noise_root.T+np.eye(n)*.04
    truth = np.r_[np.full(len(xy), .8), np.full(sources, .6)]
    y = a@truth+rng.normal(size=n)*.05
    if negative:
        y -= 1.8*a[:, -1]
    dense = StaticData(y, noise, a, xy, tuple(f'star_{i}' for i in range(sources)),
                       ('train',), 'train', True, 'independent dense algebra fixture')
    return dense, from_dense(dense)


def prior_for(data, shape=(2, 3), *, budget=Budget(), amplitude=.3, mean=.8, source_sd=1.):
    return GridPrior(data.host_xy_arcsec, shape, scale_arcsec=1.7, amplitude=amplitude,
                     host_mean=np.full(len(data.host_xy_arcsec), mean), source_count=len(data.source_names),
                     source_prior_sigma=source_sd, budget=budget)


@pytest.mark.parametrize('shape', [(1, 1), (1, 5), (4, 1), (3, 4)])
def test_finite_fft_covariance_precision_and_sampling_are_exact(shape):
    xy = grid(shape, .7)
    rotation = np.array([[.6, -.8], [.8, .6]])
    xy = xy@rotation+np.array([1.1, -2.])
    prior = GridPrior(xy, shape, scale_arcsec=2.1, amplitude=.4,
                      host_mean=np.ones(len(xy)), source_count=1, source_prior_sigma=.9)
    dense = block_diag(matern_covariance(xy, 2.1, .4), [[.81]])
    np.testing.assert_allclose(prior.action(np.eye(prior.size)), dense, rtol=2e-13, atol=3e-16)
    vector = np.random.default_rng(2718).normal(size=prior.size)
    np.testing.assert_allclose(prior.precision(vector), np.linalg.solve(dense, vector), rtol=1e-8, atol=1e-8)
    rng = np.random.default_rng(31415)
    samples = np.column_stack([prior.sample(rng, 16) for _ in range(1024)])
    empirical = np.cov(samples)
    se = np.sqrt((np.diag(dense)[:, None]*np.diag(dense)[None, :]+dense**2)/(samples.shape[1]-1))
    assert np.max(np.abs(empirical-dense)/se) < 5
    assert np.min(prior.spectrum) > 0


def test_gaussian_objective_derivatives_uncertainty_and_evidence_match_dense():
    dense, data = algebra_fixture()
    prior = prior_for(data)
    fit = GaussianPosterior(data, prior)
    covariance = block_diag(matern_covariance(dense.host_xy_arcsec, 1.7, .3), np.eye(2))
    mean, cov, logz, _ = gaussian_update(dense, prior.mean, covariance)
    np.testing.assert_allclose(fit.mean, mean, rtol=1e-11, atol=1e-12)
    np.testing.assert_allclose(fit.selected_covariance(np.arange(8)), cov, rtol=1e-10, atol=1e-12)
    assert fit.log_evidence == pytest.approx(logz, abs=1e-11)
    parameters = np.linspace(.1, 1.3, 8)
    value, gradient = fit.objective_gradient(parameters)
    residual, delta = dense.design@parameters-dense.y, parameters-prior.mean
    expected = .5*(residual@np.linalg.solve(dense.noise, residual)+delta@np.linalg.solve(covariance, delta))
    assert value == pytest.approx(expected, rel=1e-10)
    step = 1e-5
    fd = np.array([(fit.objective_gradient(parameters+e*step)[0]-fit.objective_gradient(parameters-e*step)[0])/(2*step)
                   for e in np.eye(8)])
    np.testing.assert_allclose(gradient, fd, atol=1e-7, rtol=1e-7)
    assert fit.maximum_solve_residual <= 1e-10
    assert fit.adjoint_error < 1e-14


def test_fft_workspace_chunks_preserve_rng_draws_and_covariance_actions():
    _, data = algebra_fixture()
    full = prior_for(data, budget=Budget(batch_size=32, fft_work_bytes=64*2**20))
    chunked = prior_for(data, budget=Budget(batch_size=32, fft_work_bytes=1))
    assert full.fft_batch_size == 32 and chunked.fft_batch_size == 1
    np.testing.assert_array_equal(chunked.sample(np.random.default_rng(1729), 32),
                                  full.sample(np.random.default_rng(1729), 32))
    vectors = np.random.default_rng(2718).normal(size=(full.size, 32))
    np.testing.assert_array_equal(chunked.action(vectors), full.action(vectors))
    # This SR=4 angular grid previously refused its exact positive embedding at
    # batch 128. Its physical covariance and proposal batch remain unchanged.
    shape = (20, 20)
    prior = GridPrior(grid(shape, 1.22/4), shape, scale_arcsec=3.66, amplitude=5e-5,
                      host_mean=np.full(400, 1.2e-4), source_count=2, source_prior_sigma=.01,
                      budget=Budget(batch_size=128))
    assert prior.fft_batch_size < 128
    expected = matern_covariance(prior.xy, 3.66, 5e-5)
    np.testing.assert_allclose(prior.action(np.eye(402))[:400, :400], expected, rtol=1e-12, atol=1e-22)
    with pytest.raises(MemoryError, match='GP embedding'):
        GridPrior(grid(shape, 1.22/4), shape, scale_arcsec=3.66, amplitude=5e-5,
                  host_mean=np.full(400, 1.2e-4), source_count=2, source_prior_sigma=.01,
                  budget=Budget(memory_bytes=100, fft_work_bytes=1))


def test_active_bound_reoptimizes_nuisance_and_requires_kkt():
    dense, data = algebra_fixture(negative=True)
    prior = prior_for(data)
    fit = GaussianPosterior(data, prior)
    k = prior.action(np.eye(prior.size))
    hessian = dense.design.T@np.linalg.solve(dense.noise, dense.design)+np.linalg.inv(k)
    rhs = dense.design.T@np.linalg.solve(dense.noise, dense.y)+np.linalg.solve(k, prior.mean)
    root = np.linalg.cholesky(hessian)
    expected = lsq_linear(root.T, np.linalg.solve(root, rhs), bounds=(0, np.inf), tol=1e-12)
    result = fit.bounded_mode()
    np.testing.assert_allclose(result['parameters'], expected.x, atol=1e-6)
    assert result['active_indices'] and result['projected_gradient'] <= 1e-6
    assert np.linalg.norm(result['parameters']-np.maximum(fit.mean, 0)) > .05
    with pytest.raises(NumericalFailure, match='KKT'):
        fit.bounded_mode(maxiter=1, kkt_tolerance=1e-10)
    with pytest.raises(NumericalFailure, match='PCG'):
        prior.precision(np.arange(prior.size, dtype=float), maxiter=1)


def scalar_data(value=-.2, partition='train'):
    return OperatorData(np.array([value]), BackgroundNoise((np.array([.09]),), (0.,)),
                        Design.from_matrix(np.ones((1, 1))), np.zeros((1, 2)), (), (partition,),
                        partition, True, 'synthetic scalar independent analytic truth')


def test_positive_scalar_evidence_intervals_prediction_and_conditional(tmp_path):
    data = scalar_data()
    prior = GridPrior(data.host_xy_arcsec, (1, 1), scale_arcsec=2., amplitude=.7,
                      host_mean=np.array([.2]), source_count=0, source_prior_sigma=1., budget=Budget(batch_size=128))
    fit = fit_positive(data, prior, output_dir=tmp_path, seed=2718, draws=32768)
    variance = 1/(1/.49+1/.09)
    mean = variance*(.2/.49-.2/.09)
    truth = truncnorm(-mean/np.sqrt(variance), np.inf, loc=mean, scale=np.sqrt(variance))
    assert fit.mean[0] == pytest.approx(truth.mean(), abs=5*fit.diagnostics['mean_mc_se'][0])
    assert fit.variance[0] == pytest.approx(truth.var(), rel=.04)
    logz = norm.logpdf(-.2, .2, np.sqrt(.58))+norm.logcdf(mean/np.sqrt(variance))-norm.logcdf(.2/.7)
    assert fit.diagnostics['log_evidence_95'][0] < logz < fit.diagnostics['log_evidence_95'][1]
    samples = np.column_stack(tuple(fit.batches()))
    assert np.min(samples) >= 0 and np.quantile(samples, .025) > 0
    np.testing.assert_allclose(fit.moments(np.array([0]))[1], np.atleast_2d(np.cov(samples)), rtol=1e-13)
    new = scalar_data(.1, 'heldout')
    pm, pc = fit.predict(new, full_covariance=True)
    assert pm[0] == fit.mean[0]
    assert pc[0, 0] == pytest.approx(.09+fit.variance[0])
    score = fit.predictive_evidence(new, seed=31415, draws=32768)
    joint_v = 1/(1/.49+2/.09); joint_m = joint_v*(.2/.49+(-.2+.1)/.09)
    zjoint = norm.logpdf(.1, mean, np.sqrt(.09+variance))+norm.logcdf(joint_m/np.sqrt(joint_v))-norm.logcdf(mean/np.sqrt(variance))
    assert score['log_score_95'][0] < zjoint < score['log_score_95'][1]
    science = scalar_data(.1, 'science')
    val, ess = conditional_log_likelihood(fit, science, np.ones(1), .1)
    likelihood = norm.pdf(.1, samples[0]+.1, .3)
    assert val == pytest.approx(np.log(likelihood.mean()), abs=1e-12)
    assert ess == pytest.approx(likelihood.sum()**2/(likelihood@likelihood), rel=1e-12)
    with pytest.raises(ValueError, match='twice'):
        fit.predict(data)
    with pytest.raises(ValueError, match='calibration gauge'):
        fit.predict(replace(science, calibration_gauge=('different',)))
    with pytest.raises(ValueError, match='nonnegative'):
        conditional_log_likelihood(fit, science, np.ones(1), -.1)


def test_positive_joint_matches_dense_including_uncertainty_and_model_mixture(tmp_path):
    dense, data = algebra_fixture(shape=(2, 2), n=8)
    prior = prior_for(data, shape=(2, 2), budget=Budget(batch_size=128))
    fit = fit_positive(data, prior, output_dir=tmp_path, seed=1729, draws=32768)
    oracle = fit_static(dense, scale_arcsec=1.7, amplitude=.3, host_mean=np.full(4, .8),
                        source_prior_sigma=1., seed=2718, draws=32768)
    sigma = np.hypot(fit.diagnostics['mean_mc_se'], oracle.diagnostics['mean_mc_se'])
    assert np.max(np.abs(fit.mean-oracle.mean)/sigma) < 5
    _, covariance = fit.moments(np.arange(6))
    np.testing.assert_allclose(np.diag(covariance), np.diag(oracle.covariance), rtol=.05)
    np.testing.assert_allclose(covariance, oracle.covariance, atol=.003)
    assert abs(fit.log_evidence-oracle.log_evidence) < .03
    # Same accepted samples isolate conditional-likelihood algebra from MC noise.
    samples = np.column_stack(tuple(fit.batches())).T
    same_draw_oracle = replace(oracle, samples=samples, mean=fit.mean, covariance=covariance)
    future = replace(dense, exposure_ids=('science',), partition='science', y=dense.y+.01)
    new = from_dense(future)
    assert fit.log_predictive(new)[0] == pytest.approx(same_draw_oracle.log_predictive(future)[0], abs=1e-11)
    means, predictions = fit.predict(new, full_covariance=True)
    em, ec = same_draw_oracle.predict(future)
    np.testing.assert_allclose(means, em, atol=1e-13)
    np.testing.assert_allclose(predictions, ec, atol=1e-13)
    weights = model_weights((fit, fit), np.array([.25, .75]))
    np.testing.assert_allclose(weights['weights'], [.25, .75])
    mixture = mixture_moments((fit, fit), weights['weights'], (0, 'star_0', 'star_1'))
    np.testing.assert_allclose(mixture['covariance'], covariance[np.ix_([0, 4, 5], [0, 4, 5])])
    score = mixture_conditional_likelihood((fit, fit), (new, new), np.array([.25, .75]), np.ones(8)*.1, .2)
    assert score['log_likelihood'] == pytest.approx(conditional_log_likelihood(fit, new, np.ones(8)*.1, .2)[0])


def test_background_rank_one_terms_and_nuisance_marginalization(tmp_path):
    noise = BackgroundNoise((np.array([.1, .2]), np.array([.15, .13, .3])), (.4, .2))
    full = block_diag(np.diag([.1, .2])+.16, np.diag([.15, .13, .3])+.04)
    np.testing.assert_allclose(noise.action(np.eye(5)), full)
    np.testing.assert_allclose(noise.solve(np.eye(5)), np.linalg.inv(full), atol=1e-14)
    np.testing.assert_allclose(noise.action(np.ones(5, dtype=int)), full@np.ones(5))
    np.testing.assert_allclose(noise.solve(np.ones(5, dtype=int)), np.linalg.solve(full, np.ones(5)))
    assert noise.logdet == pytest.approx(np.linalg.slogdet(full)[1])
    a = np.array([[1., .1], [.2, 1.], [1., .5], [.5, 1.], [.6, .7]])
    y = a@np.array([2., 1.])
    data = OperatorData(y, noise, Design.from_matrix(a), np.zeros((1, 2)), ('star_0',),
                        ('a', 'b'), 'train', True, 'background nuisance algebra')
    prior = GridPrior(data.host_xy_arcsec, (1, 1), scale_arcsec=1., amplitude=.2,
                      host_mean=np.array([2.]), source_count=1, source_prior_sigma=1., budget=Budget(batch_size=128))
    fit = fit_positive(data, prior, output_dir=tmp_path, seed=1729, draws=16384)
    result = fit.nuisance_moments(np.array([0, 1]))
    samples = np.column_stack(tuple(fit.batches()))
    bg, variance = noise.conditional_background(y[:, None]-a@samples)
    all_samples = np.vstack([samples, bg])
    expected = np.cov(all_samples); expected[2:, 2:] += np.diag(variance)
    np.testing.assert_allclose(result['mean'], all_samples.mean(axis=1), atol=1e-13)
    np.testing.assert_allclose(result['covariance'], expected, atol=1e-13)
    # Independent augmented *Gaussian* model verifies nuisance elimination first.
    z = np.array([[1, 0], [1, 0], [0, 1], [0, 1], [0, 1]])
    augmented = np.column_stack([a, z])
    k = np.diag([.04, 1., .16, .04]); d = np.diag([.1, .2, .15, .13, .3])
    covariance = np.linalg.inv(np.linalg.inv(k)+augmented.T@np.linalg.solve(d, augmented))
    np.testing.assert_allclose(fit.gaussian.selected_covariance(np.array([0, 1])), covariance[:2, :2], atol=1e-13)
    assert abs(result['covariance'][1, 2]) > 1e-3


@pytest.mark.parametrize('order', [False, True])
@pytest.mark.parametrize('anchors', [1, 2])
@pytest.mark.parametrize('sr', [1, 2, 4])
def test_frozen_calibrated_host_forward_adjoint_sr_masks_and_config(order, anchors, sr, monkeypatch):
    native_shape, scene_shape = (5, 5), (5*sr, 5*sr)
    scene = validation_wcs(sr=sr); scene.wcs.crpix = (np.array([3., 3.])-.5)*sr+.5
    native = validation_wcs(31., (.17, -.13)); native.wcs.crpix = [3.17, 2.87]
    monkeypatch.setattr(config, 'PRF_OPERATOR_MODE', 'anchor')
    monkeypatch.setattr(config, 'PRF_ORDER_PROJECT_THEN_CONVOLVE', order)
    monkeypatch.setattr(config, 'PRF_SPATIAL_ANCHORS_PER_AXIS', anchors)
    solver._PRF_OPERATOR_BUNDLE_CACHE.clear(); solver._PRF_NATIVE_OPERATOR_BUNDLE_CACHE.clear()
    op = frozen_host_operator(scene, native, scene_shape, native_shape, 'ch2', is_full_array=True)
    x = np.random.default_rng(1729).normal(size=scene_shape)
    y = np.random.default_rng(2718).normal(size=native_shape)
    expected = solver._apply_frame_forward_operator(x, scene, native, scene_shape, native_shape, 'ch2', is_full_array=True)
    expected_adj = solver._apply_frame_adjoint_operator(y, scene, native, scene_shape, 'ch2', is_full_array=True)
    np.testing.assert_allclose(op.forward(x.ravel()), expected.ravel(), atol=1e-14)
    np.testing.assert_allclose(op.adjoint(y.ravel()), expected_adj.ravel(), atol=1e-14)
    assert op.check_adjoint() < 1e-13
    monkeypatch.setattr(config, 'PRF_ORDER_PROJECT_THEN_CONVOLVE', not order)
    native.wcs.crpix += 2
    np.testing.assert_array_equal(op.forward(x.ravel()), expected.ravel())


def test_native_stage4_fixture_sources_likelihood_and_position_derivative():
    fixture = gp_fixture(masked=True)
    frames, matrices, reference, xy, star, nucleus, _ = fixture
    selected = tuple(f for f in frames if f.split == 'train')
    operators = tuple(Design.from_matrix(a) for a in matrices[:3])
    dense = fixture_design(fixture, 'gp_stars_nucleus', 'train')
    def native(point):
        return native_template_data(selected, reference, xy, operators, stars_xy=np.array([point]), nucleus_xy=nucleus)
    data = native(star)
    np.testing.assert_allclose(data.design.forward(np.eye(83)), dense.design, rtol=1e-14, atol=1e-14)
    np.testing.assert_allclose(data.noise.action(np.eye(len(data.y))), dense.noise, atol=1e-24)
    np.testing.assert_array_equal(data.y, dense.y)
    step = 1e-3
    column = np.zeros(83); column[81] = 1
    derivative = (native(star+[step, 0]).design.forward(column)-native(star-[step, 0]).design.forward(column))/(2*step)
    from src.isolated_astrometry import response_derivatives
    expected = np.concatenate([response_derivatives(f.image, reference, star, f.registration_xy)[0][0].ravel()[f.image.valid.ravel()]
                               for f in selected])
    np.testing.assert_allclose(derivative, expected, atol=1e-6, rtol=5e-3)
    changed = (replace(selected[0], registration_covariance=np.eye(2)*.01),)
    with pytest.raises(ValueError, match='Conditional geometry'):
        native_template_data(changed, reference, xy, operators[:1], stars_xy=np.array([star]))


def test_coincident_sources_keep_broad_anticorrelated_uncertainty(tmp_path):
    data = OperatorData(np.array([2.]), BackgroundNoise((np.array([.01]),), (0.,)),
                        Design.from_matrix(np.ones((1, 3))), np.zeros((1, 2)), ('star', 'nucleus'),
                        ('t',), 'train', True, 'coincident non-identifiability control')
    prior = GridPrior(data.host_xy_arcsec, (1, 1), scale_arcsec=1., amplitude=.1, host_mean=np.array([.1]),
                      source_count=2, source_prior_sigma=2., budget=Budget(batch_size=128))
    fit = fit_positive(data, prior, output_dir=tmp_path, seed=1729, draws=32768)
    _, cov = fit.moments(np.array([1, 2]))
    assert fit.diagnostics['likelihood_rank_deficient'] is True
    assert cov[0, 1]/np.sqrt(cov[0, 0]*cov[1, 1]) < -.9
    assert np.sqrt(cov[0, 0]) > .4


def test_guards_fail_explicitly_and_rare_mass_retains_diagnostics(tmp_path):
    data = scalar_data(-10.)
    prior = prior_for(data, (1, 1))
    with pytest.raises(NumericalFailure, match='Positive mass') as failure:
        fit_positive(data, prior, output_dir=tmp_path, seed=1729, draws=512, minimum_accepted=32)
    assert failure.value.diagnostics['accepted'] == 0
    assert list(tmp_path.glob('positive-*/diagnostics.json'))
    dense, data = algebra_fixture()
    with pytest.raises(ValueError, match='regular orthogonal'):
        prior_for(replace(data, host_xy_arcsec=data.host_xy_arcsec+np.random.default_rng(1).normal(size=(6, 2))*.01))
    with pytest.raises(MemoryError):
        prior_for(data, budget=Budget(memory_bytes=100))
    # Enough to construct the exact embedding, insufficient for this larger S.
    prior = prior_for(data, budget=Budget(memory_bytes=1_000_000))
    bigger = replace(data, y=np.tile(data.y, 10), noise=BackgroundNoise((np.ones(120),), (0.,)),
                     design=Design.from_matrix(np.tile(dense.design, (10, 1))))
    with pytest.raises(MemoryError, match='conditioning'):
        GaussianPosterior(bigger, prior)
    broken = Design((2, 2), lambda x: x, lambda x: 2*x, provenance='deliberately wrong')
    with pytest.raises(ValueError, match='Adjoint'):
        broken.check_adjoint()
    with pytest.raises(ValueError, match='verified training'):
        fit_positive(replace(data, partition='science'), prior, output_dir=tmp_path, seed=1)


def test_host_scaling_uses_no_host_square_and_keeps_unobserved_uncertainty():
    shape = (64, 64); h = int(np.prod(shape)); n = 9
    indices = np.arange(n)*413
    def adjoint(x):
        out = np.zeros((h,)+x.shape[1:]); out[indices] = x
        return out
    design = Design((n, h), lambda x: x[indices], adjoint, provenance='explicit sparse identity observations')
    data = OperatorData(np.ones(n), BackgroundNoise((np.full(n, .01),), (.02,)), design, grid(shape), (),
                        ('t',), 'train', True, 'synthetic covariance-action scaling')
    prior = prior_for(data, shape, budget=Budget(memory_bytes=64*2**20, batch_size=4))
    fit = GaussianPosterior(data, prior)
    assert fit.memory['peak_bytes'] < 64*2**20
    assert fit.memory['no_host_square_allocation']
    assert fit.cross.shape == (4096, 9)
    covariance = fit.selected_covariance(np.array([indices[0], h-1]))
    assert covariance[1, 1] > covariance[0, 0] > 0


def test_exact_prf_mode_cannot_silently_change_to_anchors(monkeypatch):
    monkeypatch.setattr(config, 'PRF_OPERATOR_MODE', 'exact')
    monkeypatch.setattr(config, 'PRF_ORDER_PROJECT_THEN_CONVOLVE', False)
    wcs = validation_wcs()
    with pytest.raises(ValueError, match='no anchor substitution'):
        frozen_host_operator(wcs, wcs, (7, 7), (7, 7), 'ch2')


@pytest.mark.real_calibration
def test_real_native_prf_and_correlated_background_equal_dense_reference():
    """Real-table algebra only, not absolute calibration or real BCD acceptance."""
    import os
    from src.point_response import _read_snapshot, file_identity
    from src.isolated_astrometry import StarFrame
    from src.nucleus import TemplateFrame
    from src.gp_identifiability import template_design
    path = Path(os.environ.get('SPITZER_STAGE2_PRF_DIR', config.PRF_DIR))/'apex_sh_IRACPC2_col129_row129_x100.fits'
    if not path.is_file():
        pytest.skip('Real warm PRF unavailable; no synthetic substitution')
    response = _read_snapshot(file_identity(path), 'ch2', 'warm', 'legacy_integral')
    wcs = validation_wcs()
    image = StarFrame(np.full((7, 7), 1e-4), np.full((7, 7), 1e-5), wcs, response)
    frame = TemplateFrame(image, 'real-table-algebra', 'train', True, background_sigma=2e-6,
                          provenance='real-table algebra only; synthetic uniform host')
    host = np.ones((49, 1)); xy = np.zeros((1, 2))
    kwargs = dict(stars_xy=np.array([[3.13, 3.24]]), nucleus_xy=np.array([4.11, 2.7]))
    dense = template_design((frame,), wcs, xy, (host,), operator_provenance='fixed F', **kwargs)
    data = native_template_data((frame,), wcs, xy, (Design.from_matrix(host),), **kwargs)
    np.testing.assert_array_equal(data.design.forward(np.eye(3)), dense.design)
    np.testing.assert_allclose(data.noise.action(np.eye(49)), dense.noise, atol=1e-25)
    prior = GridPrior(xy, (1, 1), scale_arcsec=1., amplitude=5e-5, host_mean=np.array([1.2e-4]),
                      source_count=2, source_prior_sigma=.01)
    fit = GaussianPosterior(data, prior)
    mean, cov, logz, _ = gaussian_update(dense, prior.mean, np.diag([2.5e-9, 1e-4, 1e-4]))
    np.testing.assert_allclose(fit.mean, mean, atol=1e-12)
    np.testing.assert_allclose(fit.selected_covariance(np.arange(3)), cov, atol=1e-15)
    assert fit.log_evidence == pytest.approx(logz, abs=1e-8)
