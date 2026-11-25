"""Small physics and failure regressions discovered by required numerical CI."""
from dataclasses import replace
import numpy as np
import pytest
from src.nucleus import (gaussian_response, fit_nucleus, predictive_score,
                         compare_models, validate_frames, residual_signatures)
from src.isolated_astrometry import star_response, StarFrame
from src.stage3_validation import nucleus_fixture, integrated_gaussian


def test_extended_point_limit_and_lost_wings():
    frames, scene, _ = nucleus_fixture(geometry='single')
    f = frames[0].image
    pos, shift = np.array([.23, .41]), np.zeros(2)
    point = star_response(f, scene.reference, pos, shift)
    zero = gaussian_response(f, scene.reference, pos, np.zeros(3), shift)
    tiny = gaussian_response(f, scene.reference, pos, np.array([1e-6, 2e-6, .3]), shift)
    np.testing.assert_array_equal(zero, point)
    np.testing.assert_allclose(tiny, point, atol=2e-10, rtol=0)
    assert zero.sum() < .9  # Never renormalize cropped flux.


def test_extended_independent_integral_and_quadrature():
    frames, scene, _ = nucleus_fixture(geometry='single')
    frame = frames[0].image
    profile = np.array([.48, .27, .3])
    pos = np.array([12.13, 11.79])
    cov = np.array([[profile[0]**2, np.prod(profile[:2])*.3], [np.prod(profile[:2])*.3, profile[1]**2]])
    truth = integrated_gaussian((25, 25), pos, cov+np.eye(2)*.78**2, 1.)
    fit = gaussian_response(frame, scene.reference, pos, profile, np.zeros(2), order=11)
    high = gaussian_response(frame, scene.reference, pos, profile, np.zeros(2), order=21)
    assert np.linalg.norm(fit-truth)/np.linalg.norm(truth) < 2e-4
    assert np.max(np.abs(high-fit)) < 1e-5
    assert abs(fit.sum()-1) < 1e-8


def test_full_predictive_likelihood_matches_dense():
    rng = np.random.default_rng(1729)
    r, s, u = rng.normal(size=13), rng.uniform(.1, 2., 13), rng.normal(size=(13, 5))
    score = predictive_score(r, s, u)
    c = np.diag(s*s)+u@u.T
    chi = r@np.linalg.solve(c, r)
    expected = .5*(chi+np.linalg.slogdet(c)[1]+len(r)*np.log(2*np.pi))
    np.testing.assert_allclose(score['chi2'], chi, rtol=1e-12)
    np.testing.assert_allclose(score['nll'], expected, rtol=1e-12)


@pytest.mark.parametrize('change', ['channel', 'convention'])
def test_incompatible_calibrations_rejected(change):
    frames, scene, _ = nucleus_fixture()
    f = frames[1]
    response = replace(f.image.response, **{change: 'ch1' if change == 'channel' else 'table'})
    bad = replace(f, image=StarFrame(f.image.data, f.image.sigma, f.image.wcs, response))
    with pytest.raises(ValueError, match='one channel'):
        validate_frames((frames[0], bad))


def test_template_partition_and_identity_guards():
    frames, scene, _ = nucleus_fixture()
    with pytest.raises(ValueError, match='verified template'):
        replace(frames[0], verified_template=False)
    with pytest.raises(ValueError, match='unique'):
        validate_frames((frames[0], frames[0]))
    with pytest.raises(ValueError, match='training-only'):
        fit_nucleus((frames[3],), scene)
    with pytest.raises(ValueError, match='Independent training'):
        compare_models(frames[:3], scene)
    with pytest.raises(ValueError, match='positive definite'):
        replace(frames[0], registration_covariance=np.diag([-1., 1.]))


def test_point_centroid_fit_and_uncertainty():
    frames, scene, truth = nucleus_fixture(geometry='dithers')
    fit = fit_nucleus(frames[:3], scene)
    assert fit.diagnostics['optimality'] < 1e-3
    assert fit.diagnostics['covariance_available']
    np.testing.assert_allclose(fit.center_xy, truth['centroid'], atol=2e-4)
    np.testing.assert_allclose(fit.flux, [.01, .004, .006, .02], rtol=5e-4)
    mean, factor = fit.predict(frames[3])
    assert factor.shape == (625, fit.problem.ng+3)
    assert np.any(np.sum(factor*factor, axis=1) > 0)
    assert np.isfinite(fit.score_frames(frames[4:])['nll'])
    before = frames[3].image.data.copy()
    changed = replace(frames[3], image=StarFrame(before+1., frames[3].image.sigma, frames[3].image.wcs, frames[3].image.response))
    np.testing.assert_array_equal(fit.predict(changed)[0], mean) # No held-out fit.
    assert fit.score_frames((changed,))['nll'] > fit.score_frames((frames[3],))['nll']


def test_resolved_fit_beats_point_on_independent_heldout():
    frames, scene, truth = nucleus_fixture('extent')
    point = fit_nucleus(frames[:3], scene)
    resolved = fit_nucleus(frames[:3], scene, 'gaussian')
    assert np.all(resolved.flux >= 0)
    np.testing.assert_allclose(resolved.center_xy, truth['centroid'], atol=5e-4)
    np.testing.assert_allclose(resolved.profile, truth['profile'], atol=1e-3)
    assert resolved.score_frames(frames[4:])['nll'] < point.score_frames(frames[4:])['nll']-10
    residual = residual_signatures(frames[4], scene, point.predict(frames[4])[0], probe_xy=truth['probe_xy'])
    assert abs(residual['source_bias']) > 1e-5


def test_no_uncertainty_inflation_or_false_convergence():
    frames, scene, _ = nucleus_fixture('extent', geometry='single')
    with pytest.raises(RuntimeError, match='did not converge'):
        fit_nucleus(frames, scene, 'gaussian', max_nfev=1)
    sigma = frames[0].image.sigma.copy()
    fit_nucleus(frames, scene)
    np.testing.assert_array_equal(frames[0].image.sigma, sigma)


def test_profile_derivatives_and_registration_jacobian():
    from src.nucleus import gaussian_derivatives, _Problem
    frames, scene, _ = nucleus_fixture('extent', 1729)
    frame = frames[1].image
    p = np.array([12.1234, 11.8142, .4123, .2511, .231, .1334, -.1197])
    analytic = gaussian_derivatives(frame, scene.reference, p[:2], p[2:5], p[5:], order=11)
    for j in range(7):
        a,b=p.copy(),p.copy();a[j]+=1e-6;b[j]-=1e-6
        fd=(gaussian_response(frame,scene.reference,a[:2],a[2:5],a[5:])-gaussian_response(frame,scene.reference,b[:2],b[2:5],b[5:]))/2e-6
        np.testing.assert_allclose(analytic[j],fd,atol=4e-6,rtol=2e-3)
    fit=fit_nucleus(frames[:3],scene,'gaussian')
    assert fit.diagnostics['optimality']<1e-3
    assert fit.diagnostics['covariance_available']
    assert np.any(np.abs(fit.covariance[fit.problem.na:fit.problem.ng, fit.problem.ng:])>0)


@pytest.mark.parametrize('sr',[1,2,4])
def test_fixed_footprint_sr_and_roll_recovery(sr):
    frames,scene,truth=nucleus_fixture('extent',sr=sr)
    fit=fit_nucleus(frames[:3],scene,'gaussian')
    np.testing.assert_allclose(fit.center_xy/sr,truth['centroid']/sr,atol=5e-4)
    np.testing.assert_allclose(fit.profile[:2]/sr,truth['profile'][:2]/sr,atol=1e-3)
    assert abs(fit.flux[0]-.01)<5e-6


def test_uncertainty_preserves_known_noise_scale():
    frames,scene,_=nucleus_fixture(geometry='single')
    f=frames[0]
    larger=replace(f,image=StarFrame(f.image.data,2*f.image.sigma,f.image.wcs,f.image.response,f.image.valid))
    a,b=fit_nucleus(frames,scene),fit_nucleus((larger,),scene)
    np.testing.assert_allclose(a.parameters,b.parameters,atol=1e-8)
    np.testing.assert_allclose(b.covariance,4*a.covariance,rtol=1e-5,atol=1e-12)


def test_reference_and_prediction_geometry_caches_invalidate():
    frames,scene,_=nucleus_fixture(geometry='single')
    fit=fit_nucleus(frames,scene)
    scene.reference.wcs.crpix += .1
    with pytest.raises(ValueError,match='Reference WCS changed'):
        fit.training_predictions()


def test_core_envelope_contains_point_and_gaussian_limits():
    from src.nucleus import _Problem
    frames,scene,_=nucleus_fixture(geometry='single')
    q=_Problem(frames,scene,'core_envelope')
    p=np.r_[[.4,.6,.2,.3,2.],scene.center_xy,[.41,.28,.3],0.]
    total=q.render(p,frames[0],np.zeros(2),0.)
    point=_Problem(frames,scene,'point')
    resolved=_Problem(frames,scene,'gaussian')
    pp=np.r_[[.4,.2,.3,2.],scene.center_xy,0.]
    pg=np.r_[[.6,0.,0.,0.],scene.center_xy,[.41,.28,.3],0.]
    np.testing.assert_allclose(total,point.render(pp,frames[0],np.zeros(2),0.)+resolved.render(pg,frames[0],np.zeros(2),0.),rtol=1e-14,atol=1e-18)
    p[q.na+2:q.ng]=[0.,0.,.2]
    pp[0]=1.
    np.testing.assert_allclose(q.render(p,frames[0],np.zeros(2),0.),point.render(pp,frames[0],np.zeros(2),0.),rtol=1e-14,atol=1e-18)


def test_nonconvergence_saves_rejected_attempt():
    from src.nucleus import NucleusConvergenceError
    frames,scene,_=nucleus_fixture('extent',geometry='single')
    with pytest.raises(NucleusConvergenceError) as info:
        fit_nucleus(frames,scene,'gaussian',max_nfev=1)
    assert info.value.arrays['training_model'].shape==(1,25,25)
    assert np.isfinite(info.value.arrays['parameters']).all()
    assert info.value.diagnostics['nfev']==1


def test_real_morphology_requires_explicit_experimental_opt_in():
    frames,scene,_=nucleus_fixture(geometry='single')
    f=frames[0]
    real=replace(f,image=StarFrame(f.image.data,f.image.sigma,f.image.wcs,replace(f.image.response,synthetic=False)))
    with pytest.raises(ValueError,match='quadrature is not validated'):
        fit_nucleus((real,),scene)


@pytest.mark.real_calibration
def test_real_native_profile_exact_point_limit():
    import os
    from pathlib import Path
    from src.point_response import _read_snapshot,file_identity
    from src import config
    directory=Path(os.environ.get('SPITZER_STAGE2_PRF_DIR',config.PRF_DIR))
    path=directory/'apex_sh_IRACPC2_col129_row129_x100.fits'
    if not path.exists():
        pytest.skip('Real warm PRF unavailable; not a synthetic substitute')
    response=_read_snapshot(file_identity(path),'ch2','warm','legacy_integral')
    frames,scene,_=nucleus_fixture(geometry='single')
    f=frames[0].image
    real=StarFrame(f.data,f.sigma,f.wcs,response)
    xy=np.array([12.17,11.81])
    for phase in (np.zeros(2),np.array([.21,-.17])):
        np.testing.assert_array_equal(gaussian_response(real,scene.reference,xy,np.zeros(3),phase),
                                      star_response(real,scene.reference,xy,phase))


def test_dense_reference_allocation_is_bounded():
    frames,scene,_=nucleus_fixture()
    crowded=replace(scene,star_positions=np.tile(scene.star_positions,(1200,1)))
    with pytest.raises(MemoryError,match='Jacobian exceeds'):
        fit_nucleus(frames[:3],crowded)
