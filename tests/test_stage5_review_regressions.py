"""RV27 behavioral regressions: independent arithmetic, uncertainty and gates."""
from dataclasses import replace
from decimal import Decimal, localcontext
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
from scipy.special import logsumexp
from scipy.stats import chi2

from src.gp_identifiability import matern_covariance
from src.scalable_gp import Budget, GridPrior, GaussianPosterior, NumericalFailure, fit_positive, mixture_moments
from src.scalable_operators import BackgroundNoise, NoiseSolveFailure, DenseNoise, Design, OperatorData


def problem(matrix, noise, *, value=2., shape=(1, 1), budget=Budget(batch_size=128)):
    yy, xx = np.indices(shape)
    xy = np.column_stack((xx.ravel(), yy.ravel()))
    prior = GridPrior(xy, shape, scale_arcsec=1.7, amplitude=1., host_mean=np.ones(len(xy)),
                      source_count=0, source_prior_sigma=1., budget=budget)
    data = OperatorData(np.full(len(matrix), value), noise, Design.from_matrix(matrix), xy, (),
                        ('train',), 'train', True, 'independent review regression algebra')
    return data, prior


def decimal_noise_solve(variance, tau, rhs):
    """80-digit independent full-rank-one inverse, only for tiny test fixtures."""
    with localcontext() as context:
        context.prec = 80
        v = [Decimal(float(x)) for x in variance]
        t2 = Decimal(float(tau))**2
        q = sum(1/x for x in v)
        columns = []
        for column in np.asarray(rhs).reshape(len(v), -1).T:
            z = [Decimal(float(b))/a for a, b in zip(v, column)]
            correction = t2*sum(z)/(1+t2*q)
            columns.append([float(b-correction/a) for a, b in zip(v, z)])
    return np.array(columns).T.reshape(rhs.shape)


@pytest.mark.parametrize('variance', [1e-4, 1e-8, 1e-12, 1e-16, 1e-20])
def test_background_scalar_and_unequal_variance_common_modes(variance):
    # The scalar is perfectly conditioned regardless of the variance ratio.
    scalar = BackgroundNoise((np.array([variance]),), (1.,))
    rhs = np.array([[1., -3., 1e-100, 1e100]])
    np.testing.assert_allclose(scalar.solve(rhs), rhs/(1+variance), rtol=2e-15, atol=0)
    assert scalar.maximum_solve_residual <= 1e-10
    # Unequal variances, multiple pixels/RHS, enormous common-mode dynamic range.
    v = variance*np.array([1., 2., 4.])
    noise = BackgroundNoise((v,), (1.,))
    rhs = np.tile(np.array([1., -3., 1e-40]), (3, 1))
    np.testing.assert_allclose(noise.solve(rhs), decimal_noise_solve(v, 1., rhs), rtol=1e-13, atol=0)
    np.testing.assert_allclose(noise.action(noise.solve(rhs)), rhs, rtol=1e-13, atol=0)


@pytest.mark.parametrize('variance', [1., 1e-4, 1e-8, 1e-12, 1e-16])
def test_background_differential_modes_are_accurate_or_explicitly_unresolved(variance):
    v = variance*np.array([1., 2., 5.])
    rhs = np.array([[1., 1., 2.], [1., -.7, 1.5], [1., .2, 3.]])
    noise = BackgroundNoise((v,), (.8,))
    reference = decimal_noise_solve(v, .8, rhs)
    try:
        actual = noise.solve(rhs)
    except NoiseSolveFailure as failure:
        assert variance <= 1e-8
        assert failure.diagnostics['relative_residual'] > 1e-10
        assert max(failure.diagnostics['per_rhs']) > 1e-10
    else:
        np.testing.assert_allclose(actual, reference, rtol=1e-10, atol=1e-12)
        assert noise.maximum_solve_residual <= 1e-10


def test_background_failure_cannot_hide_behind_large_rhs():
    noise = BackgroundNoise((np.array([1., 2., 5.])*1e-16,), (1.,))
    rhs = np.array([[1e100, 1.], [1e100, -.7], [1e100, .2]])
    with pytest.raises(NoiseSolveFailure) as failure:
        noise.solve(rhs)
    assert failure.value.diagnostics['per_rhs'][0] <= 1e-10
    assert failure.value.diagnostics['per_rhs'][1] > 1e-10
    blocks = BackgroundNoise((np.ones(1), np.array([1., 2., 5.])*1e-16), (0., 1.))
    with pytest.raises(NoiseSolveFailure) as failure:
        blocks.solve(np.array([1e100, 1., -.7, .2]))
    assert failure.value.diagnostics['per_exposure_rhs'][0][0] <= 1e-10
    assert failure.value.diagnostics['per_exposure_rhs'][1][0] > 1e-10


def test_zero_background_diagonal_limit_does_not_subtract_dynamic_rhs():
    v = np.array([1., 2., 3.])
    rhs = np.array([[1e100, 1.], [1., 1e-100], [1e-50, -1e50]])
    np.testing.assert_array_equal(BackgroundNoise((v,), (0.,)).solve(rhs), rhs/v[:, None])


def test_background_objective_mode_likelihood_ess_and_mismatch_diagnostic(tmp_path):
    noise = BackgroundNoise((np.array([1e-16]),), (1.,))
    data, prior = problem(np.ones((1, 1)), noise)
    gaussian = GaussianPosterior(data, prior)
    for x in (.2, 1., 1.5, 3.):
        value, gradient = gaussian.objective_gradient(np.array([x]))
        assert value == pytest.approx(.5*((x-1)**2+(x-2)**2), abs=1e-14)
        assert gradient[0] == pytest.approx(2*x-3, abs=1e-14)
    mode = gaussian.bounded_mode()
    np.testing.assert_allclose(mode['parameters'], [1.5], atol=1e-12)
    assert mode['projected_gradient'] < 1e-12
    fit = fit_positive(data, prior, output_dir=tmp_path, seed=1729, draws=4096)
    samples = np.column_stack(tuple(fit.batches()))[0]
    for y in (2., 20.):
        future = replace(data, y=np.array([y]), exposure_ids=('future',), partition='science')
        actual, ess = fit.log_predictive(future)
        score = -.5*((y-samples)**2+np.log(2*np.pi))
        expected = logsumexp(score)-np.log(len(score))
        assert actual == pytest.approx(expected, abs=1e-12)
        assert ess == pytest.approx(np.exp(2*logsumexp(score)-logsumexp(2*score)), rel=1e-12)
    tail = chi2.sf((2-samples[:512])**2, 1).mean()
    assert fit.diagnostics['training_posterior_predictive_tail'] == pytest.approx(tail, abs=1e-14)
    assert fit.diagnostics['static_mismatch'] == (tail < 1e-4)
    assert fit.diagnostics['maximum_noise_solve_residual'] <= 1e-10


def test_multi_pixel_background_objective_and_bounded_mode_against_dense_precision():
    v = np.array([.01, .02, .05])
    a = np.array([[1.], [.2], [.7]])
    noise = BackgroundNoise((v,), (2.,))
    data, prior = problem(a, noise, value=.7)
    fit = GaussianPosterior(data, prior)
    inverse = decimal_noise_solve(v, 2., np.eye(3))
    precision = 1+float((a.T@inverse@a)[0, 0])
    information = 1+float((a.T@inverse@data.y)[0])
    expected_mode = max(0., information/precision)
    for x in (.0, .2, 1., 2.):
        residual = a[:, 0]*x-data.y
        expected_value = .5*((x-1)**2+residual@inverse@residual)
        value, gradient = fit.objective_gradient(np.array([x]))
        assert value == pytest.approx(expected_value, rel=1e-12, abs=1e-14)
        assert gradient[0] == pytest.approx(precision*x-information, rel=1e-12, abs=1e-14)
    mode = fit.bounded_mode()
    np.testing.assert_allclose(mode['parameters'], [expected_mode], atol=1e-12)
    assert mode['projected_gradient'] < 1e-10


@pytest.mark.parametrize('variance', [1e-4, 1e-8, 1e-12, 1e-16, 1e-20])
def test_selected_scalar_covariance_retains_tiny_variances(variance):
    data, prior = problem(np.ones((1, 1)), DenseNoise(np.array([[variance]])))
    fit = GaussianPosterior(data, prior)
    expected = variance/(1+variance)  # Independent information-form scalar oracle.
    actual = fit.selected_covariance(np.array([0]))[0, 0]
    assert actual > 0
    assert actual == pytest.approx(expected, rel=1e-8, abs=0)
    if variance <= 1e-12:
        with pytest.raises(NumericalFailure, match='Covariance action precision'):
            fit.covariance_action(np.ones(1))


@pytest.mark.parametrize('variance', [1e-4, 1e-8, 1e-12, 1e-16])
def test_selected_correlated_dynamic_range_matches_information_oracle(variance):
    a = np.array([[1., .25], [.1, 1.]])
    n = np.array([[1., .3], [.3, 2.]])
    data, prior = problem(a, DenseNoise(variance*n), shape=(1, 2))
    # Dense oracle independently constructs the GP, then factors in information
    # form with the common noise scale removed. No scalable covariance helper.
    k = matern_covariance(data.host_xy_arcsec, 1.7, 1.)
    expected = variance*np.linalg.inv(variance*np.linalg.inv(k)+a.T@np.linalg.solve(n, a))
    actual = GaussianPosterior(data, prior).selected_covariance(np.arange(2))
    scale = np.sqrt(np.diag(expected))
    assert np.max(abs(actual-expected)/np.outer(scale, scale)) < 1e-8
    assert actual[0, 1] != 0


def test_selected_covariance_refuses_unresolved_precision_without_floor():
    data, prior = problem(np.ones((1, 1)), DenseNoise(np.array([[1e-40]])))
    with pytest.raises(NumericalFailure, match='Selected covariance precision') as failure:
        GaussianPosterior(data, prior).selected_covariance(np.array([0]))
    assert failure.value.diagnostics['relative_roundoff_bound'][0] > 1e-8


def test_prior_output_budget_precedes_allocation(monkeypatch):
    shape = (64, 64); yy, xx = np.indices(shape)
    prior = GridPrior(np.column_stack((xx.ravel(), yy.ravel())), shape, scale_arcsec=1.7,
                      amplitude=1., host_mean=np.ones(4096), source_count=0, source_prior_sigma=1.,
                      budget=Budget(memory_bytes=64*2**20, fft_work_bytes=1, batch_size=4096))
    empty = np.empty
    def spy(shape, *args, **kwargs):
        assert tuple(np.atleast_1d(shape)) != (4096, 4096), 'output allocated before guard'
        return empty(shape, *args, **kwargs)
    monkeypatch.setattr(np, 'empty', spy)
    rng = np.random.default_rng(1729); state = copy.deepcopy(rng.bit_generator.state)
    with pytest.raises(MemoryError, match='Prior sampling output'):
        prior.sample(rng, 4096)
    assert rng.bit_generator.state == state
    assert prior.sample(rng, 4).shape == (4096, 4)


def test_numpy_mixture_indices_preserve_all_covariance_terms(tmp_path):
    data, prior = problem(np.array([[1., .2], [.3, 1.]]), DenseNoise(np.eye(2)*.1), shape=(1, 2))
    fit = fit_positive(data, prior, output_dir=tmp_path, seed=1729, draws=2048)
    # Distinct model means ensure the between-model term is also nonzero.
    second = replace(fit, mean=fit.mean+.01)
    expected = mixture_moments((fit, second), np.array([.3, .7]), (0, 1, 'absent'))
    actual = mixture_moments((fit, second), np.array([.3, .7]), (*np.arange(2), 'absent'))
    for key in ('mean', 'covariance', 'within_covariance', 'between_covariance'):
        np.testing.assert_array_equal(actual[key], expected[key])
    assert actual['between_covariance'][0, 0] > 0
    for invalid in (0., True, None, np.array([0])):
        with pytest.raises(ValueError, match='Components must'):
            mixture_moments((fit,), np.ones(1), (invalid,))


def benchmark_module():
    scripts = Path(__file__).resolve().parents[1]/'scripts'
    sys.path.insert(0, str(scripts))
    spec = importlib.util.spec_from_file_location('review_stage5_benchmarks', scripts/'run_stage5_benchmarks.py')
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def records(tmp_path):
    np.savez(tmp_path/'numeric.npz', selected_covariance=np.eye(1))
    np.savez(tmp_path/'positive.npz', positive_mean=np.ones(1), mean_mc_se=np.ones(1)*.01,
             selected_covariance=np.eye(1), selected_covariance_mc_se=np.ones((1, 1))*.01)
    metadata = dict(file='numeric.npz', sha256=hashlib.sha256((tmp_path/'numeric.npz').read_bytes()).hexdigest(),
                    keys=['selected_covariance'])
    def record(seconds):
        return dict(status='completed', wall_seconds=seconds, peak_memory_bytes=200*2**20,
                    rss_budget_bytes=2**30, rss_within_budget=True, input_sha256='same',
                    numerical_arrays=metadata, positive_arrays='positive.npz', positive_log_evidence_95=[-1, 1])
    initial = {m: {s: [record(1 if s == 'base' else 2) for _ in range(5)]
                   for s in ('base', 'head')} for m in ('cold', 'warm')}
    confirmation = copy.deepcopy(initial)
    for mode in confirmation.values():
        for side in mode.values():
            for row in side: row['wall_seconds'] = 1.
    return initial, confirmation


@pytest.mark.parametrize('side', ['base', 'head'])
@pytest.mark.parametrize('failure', ['rss', 'failed', 'missing', 'empty'])
def test_confirmation_only_failures_are_rejected_before_arrays(tmp_path, side, failure):
    module = benchmark_module(); initial, confirmation = records(tmp_path)
    assert all(r['status'] == 'confirmation_required' for r in module.assess(initial, tmp_path).values())
    for mode in confirmation.values():
        if failure == 'rss':
            # Deliberately stale success flag and inflated declared cap cannot
            # conceal the actual 1 GiB hard-cap violation.
            mode[side][0].update(peak_memory_bytes=1100*2**20, rss_budget_bytes=2**31)
        elif failure == 'failed': mode[side][0] = dict(status='failed', error='controlled worker failure')
        elif failure == 'missing': mode[side].pop()
        else: mode[side] = []
    report = module.assess(initial, tmp_path, confirmation)
    assert all(r['status'].startswith('failed') for r in report.values())


def test_valid_and_noisy_confirmations_keep_original_gates(tmp_path):
    module = benchmark_module(); initial, confirmation = records(tmp_path)
    assert all(r['status'] == 'passed' for r in module.assess(initial, tmp_path, confirmation).values())
    for mode in confirmation.values():
        for row, seconds in zip(mode['head'], [1, 2, 3, 4, 5]): row['wall_seconds'] = seconds
    assert all(r['status'] == 'inconclusive' for r in module.assess(initial, tmp_path, confirmation).values())


def test_paired_but_incomplete_confirmation_is_not_accepted(tmp_path):
    module = benchmark_module(); initial, confirmation = records(tmp_path)
    for mode in confirmation.values():
        for side in mode.values(): side.pop()
    assert all(r['status'] == 'failed_completion' for r in module.assess(initial, tmp_path, confirmation).values())


def test_cli_propagates_confirmation_only_rss_failure(tmp_path):
    initial, confirmation = records(tmp_path)
    for mode in confirmation.values():
        mode['head'][0].update(peak_memory_bytes=1100*2**20, rss_within_budget=False)
    input_file = tmp_path/'records.json'; input_file.write_text(json.dumps([initial, confirmation]))
    scripts = Path(__file__).resolve().parents[1]/'scripts'
    # Real CLI main and assessment; only the expensive subprocess collector is
    # controlled. Numerical files/checksums are genuine, including confirmation.
    wrapper = tmp_path/'cli.py'
    wrapper.write_text(f'''import sys,json,shutil
from pathlib import Path
sys.path.insert(0,{str(scripts)!r})
import run_stage5_benchmarks as b
initial,confirmation=json.loads(Path({str(input_file)!r}).read_text())
def collect(**kwargs):
    for name in ('numeric.npz','positive.npz'):
        shutil.copyfile(Path({str(tmp_path)!r})/name,kwargs['output_dir']/name)
    return initial if kwargs['label']=='initial' else confirmation
b._collect=collect
raise SystemExit(b.main())
''')
    result = subprocess.run([sys.executable, str(wrapper), '--base-repo', '.', '--head-repo', '.',
                             '--output-dir', str(tmp_path/'cli-output'), '--cases', 'small'],
                            capture_output=True, text=True)
    assert result.returncode == 1, result.stdout+result.stderr
    summary = json.loads((tmp_path/'cli-output/summary.json').read_text())
    assert all(r['status'] == 'failed_rss_budget' for r in summary['cases']['small'].values())
