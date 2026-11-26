"""Small, opt-in positive static-GP reference posterior for Stage 4.

The Gaussian linear problem is integrated exactly, then corrected for the
positive orthant by independent rejection draws. Evidence and moments therefore
have *reported Monte Carlo error*, not a Laplace/boundary covariance shortcut.
There is no transient fitting, science-based selection, or production default.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import time
from typing import Callable

import numpy as np
from astropy.wcs import WCS
from scipy.linalg import cho_factor, cho_solve, solve_triangular, block_diag
from scipy.special import logsumexp
from scipy.stats import norm, chi2

from .isolated_astrometry import _freeze, _frame_identity, star_response
from .nucleus import TemplateFrame, gaussian_response, validate_frames


class SamplingFailure(RuntimeError):
    """Too little positive Gaussian mass; diagnostics survive the failed fit."""

    def __init__(self, message: str, diagnostics: dict):
        super().__init__(message)
        self.diagnostics = diagnostics


@dataclass(frozen=True)
class StaticData:
    """Masked native observations and a fixed linear component design.

    Host coefficients are nonnegative surface brightness in Jy/native pixel.
    Source coefficients are nonnegative integrated amplitudes in the declared
    PRF convention. Background calibration is already subtracted; its covariance
    belongs in ``noise``. Arrays are copied; IDs refer to whole exposures.
    ``verified`` is a provenance assertion, never inferred from a date alone.
    """
    y: np.ndarray
    noise: np.ndarray
    design: np.ndarray
    host_xy_arcsec: np.ndarray
    source_names: tuple[str, ...]
    exposure_ids: tuple[str, ...]
    partition: str
    verified: bool
    provenance: str
    calibration_gauge: tuple = ('synthetic-linear-algebra',)

    def __post_init__(self) -> None:
        y, v, a, xy = map(np.asarray, (self.y, self.noise, self.design, self.host_xy_arcsec))
        n, h = y.size, len(xy)
        if y.ndim != 1 or not n or xy.shape != (h, 2) or not h:
            raise ValueError('Nonempty native data and angular host coordinates required')
        if n > 2500 or h > 256 or a.size > 1_000_000:
            raise ValueError('Stage 4 dense reference budget exceeded; use Stage 5')
        if a.shape != (n, h+len(self.source_names)) or v.shape != (n, n):
            raise ValueError('Design/noise dimensions do not match component layout')
        roundoff = 64*np.finfo(float).eps*max(float(np.max(np.abs(a))), 1.)
        if not all(np.isfinite(z).all() for z in (y, v, a, xy)) or np.any(a < -roundoff):
            raise ValueError('Finite data/noise and nonnegative brightness operator required')
        if not np.allclose(v, v.T, rtol=1e-12, atol=0):
            raise ValueError('Noise covariance must be symmetric')
        np.linalg.cholesky(v)
        if (not self.exposure_ids or len(set(self.exposure_ids)) != len(self.exposure_ids)
                or len(set(self.source_names)) != len(self.source_names) or not self.provenance
                or self.partition not in ('train', 'validation', 'heldout', 'science')):
            raise ValueError('Unique exposure/component IDs, provenance and partition required')
        for name, value in zip(('y', 'noise', 'design', 'host_xy_arcsec'), (y, v, a, xy)):
            object.__setattr__(self, name, _freeze(value))


def template_design(frames: tuple[TemplateFrame, ...], reference: WCS, host_xy_arcsec: np.ndarray,
                    host_operators: tuple[np.ndarray, ...], *, stars_xy: np.ndarray,
                    nucleus_xy: np.ndarray | None = None, nucleus_profile: np.ndarray | None = None,
                    core_envelope: bool = False, quadrature_order: int = 21,
                    experimental_real_quadrature: bool = False,
                    operator_provenance: str) -> StaticData:
    """Combine established F matrices with native stars and Stage 3 nucleus.

    Each F maps the caller's physical nonnegative host grid to all native pixels;
    no core masking, orthogonalization, crop renormalization or replacement F is
    applied here. The caller must establish units and footprint convergence.
    Geometry is conditional: nonzero registration covariance is rejected instead
    of silently omitted. Background calibration is independent of fitting pixels
    and is integrated with its full within-exposure rank-one covariance.
    """
    validate_frames(frames)
    if sum(int(f.image.valid.sum()) for f in frames) > 2500 or len(host_xy_arcsec) > 256:
        raise ValueError('Stage 4 dense reference budget exceeded before native allocation')
    if len(host_operators) != len(frames) or not operator_provenance:
        raise ValueError('One documented host operator per frame required')
    if len({f.split for f in frames}) != 1:
        raise ValueError('Do not combine template partitions during selection')
    if any(np.any(f.registration_covariance) for f in frames):
        raise ValueError('Conditional geometry only: supply separate registration alternatives')
    stars = np.asarray(stars_xy, dtype=float).reshape(-1, 2)
    names = tuple(f'star_{i}' for i in range(len(stars)))
    if nucleus_xy is not None:
        names += ('nucleus_core', 'nucleus_envelope') if core_envelope else ('nucleus',)
    elif core_envelope or nucleus_profile is not None:
        raise ValueError('Nuclear profile requires a nuclear centroid')
    designs, data, noises = [], [], []
    for frame, operator in zip(frames, host_operators):
        image = frame.image
        if np.shape(operator) != (image.data.size, len(host_xy_arcsec)):
            raise ValueError('Host operator dimensions do not match native frame')
        columns = [star_response(image, reference, p, frame.registration_xy).ravel() for p in stars]
        if nucleus_xy is not None:
            if nucleus_profile is None or core_envelope:
                columns.append(star_response(image, reference, nucleus_xy, frame.registration_xy).ravel())
            if nucleus_profile is not None:
                if not getattr(image.response, 'synthetic', False) and not experimental_real_quadrature:
                    raise ValueError('Real extended-profile quadrature requires explicit experimental opt-in')
                columns.append(gaussian_response(image, reference, nucleus_xy, nucleus_profile,
                                                frame.registration_xy, order=quadrature_order).ravel())
            elif core_envelope:
                raise ValueError('Core/envelope requires a resolved envelope profile')
        mask = image.valid.ravel()
        designs.append(np.column_stack([operator, *columns])[mask])
        data.append((image.data-frame.background).ravel()[mask])
        noises.append(np.diag(image.sigma.ravel()[mask]**2)+frame.background_sigma**2)
    return StaticData(np.concatenate(data), block_diag(*noises), np.vstack(designs), host_xy_arcsec,
                      names, tuple(f.exposure_id for f in frames), frames[0].split, True,
                      operator_provenance+'; '+ '; '.join(f.provenance for f in frames)
                      +'; calibration/geometry='+repr(tuple(_frame_identity(f.image) for f in frames)),
                      (frames[0].image.response.channel, frames[0].image.response.mission,
                       frames[0].image.response.convention, getattr(frames[0].image.response, 'synthetic', False)))


def dense_host_operator(forward: Callable[[np.ndarray], np.ndarray], shape: tuple[int, int],
                        native_shape: tuple[int, int]) -> np.ndarray:
    """Snapshot an established linear F without changing its objective or units.

    ``forward`` normally wraps solver._apply_frame_forward_operator with an
    explicit frozen configuration. This reference helper guards allocation before
    evaluating impulses. Tiny negative roundoff is rejected, never clipped.
    """
    h, n = int(np.prod(shape)), int(np.prod(native_shape))
    if h > 256 or h*n > 1_000_000:
        raise ValueError('Dense host operator exceeds reference budget')
    impulse = np.zeros(shape)
    columns = []
    for j in range(h):
        impulse.flat[j] = 1.
        columns.append(np.asarray(forward(impulse)).reshape(n).copy())
        impulse.flat[j] = 0.
    return np.column_stack(columns)


def matern_covariance(xy_arcsec: np.ndarray, scale_arcsec: float, amplitude: float) -> np.ndarray:
    """Matérn 3/2 covariance; amplitude is SD in Jy/native pixel, scale in arcsec."""
    if not np.isfinite([scale_arcsec, amplitude]).all() or min(scale_arcsec, amplitude) <= 0:
        raise ValueError('Positive finite physical GP scale and amplitude required')
    xy = np.asarray(xy_arcsec)
    r = np.linalg.norm(xy[:, None]-xy[None, :], axis=-1)*np.sqrt(3)/scale_arcsec
    return amplitude**2*(1+r)*np.exp(-r)


def gaussian_update(data: StaticData, mean: np.ndarray, covariance: np.ndarray) -> tuple:
    """Untruncated exact Gaussian posterior/evidence, using parameter-space solves.

    Return posterior mean/covariance, normalized log evidence and whitened
    prior-to-data design. No inverse pixel covariance or diagonal GP substitute.
    """
    root = np.linalg.cholesky(covariance)
    noise_root = np.linalg.cholesky(data.noise)
    w = solve_triangular(noise_root, data.design@root, lower=True)
    r = solve_triangular(noise_root, data.y-data.design@mean, lower=True)
    factor = cho_factor(np.eye(len(mean))+w.T@w, lower=True)
    q = cho_solve(factor, w.T@r)
    post_mean = mean+root@q
    post_cov = root@cho_solve(factor, root.T)
    # Use the augmented residual to avoid subtracting nearly equal large chi2s.
    chi2 = np.sum((r-w@q)**2)+q@q
    logdet = 2*np.log(np.diag(noise_root)).sum()+2*np.log(np.diag(factor[0])).sum()
    log_evidence = -.5*(chi2+logdet+len(r)*np.log(2*np.pi))
    return post_mean, post_cov, float(log_evidence), w


def _wilson(k: int, n: int) -> tuple[float, float]:
    """97.5% Wilson intervals; two mass estimates use Bonferroni for approx 95%."""
    z = norm.ppf(.9875)
    p = k/n
    center = (p+z*z/(2*n))/(1+z*z/n)
    radius = z*np.sqrt(p*(1-p)/n+z*z/(4*n*n))/(1+z*z/n)
    return float(max(0, center-radius)), float(min(1, center+radius))


@dataclass(frozen=True)
class StaticPosterior:
    """IID positive static draws conditional on each template exposure exactly once."""
    data: StaticData
    samples: np.ndarray
    mean: np.ndarray
    covariance: np.ndarray
    log_evidence: float
    diagnostics: dict
    gaussian_mean: np.ndarray
    gaussian_covariance: np.ndarray

    def check_prediction(self, data: StaticData) -> None:
        """Disallow likelihood reuse, component reordering and changed host grids."""
        if set(data.exposure_ids) & set(self.data.exposure_ids):
            raise ValueError('Template likelihood would be counted twice')
        if data.source_names != self.data.source_names or not np.array_equal(data.host_xy_arcsec, self.data.host_xy_arcsec):
            raise ValueError('Prediction must preserve the static component layout')
        if data.calibration_gauge != self.data.calibration_gauge:
            raise ValueError('Prediction calibration gauge differs from training templates')

    def predict(self, data: StaticData) -> tuple[np.ndarray, np.ndarray]:
        """Mean and full measurement-plus-static covariance, including source cores."""
        self.check_prediction(data)
        return data.design@self.mean, data.noise+data.design@self.covariance@data.design.T

    def log_predictive(self, data: StaticData, offset: np.ndarray | None = None) -> tuple[float, float]:
        """Integrate a new likelihood over positive draws; return log score and ESS.

        ESS diagnoses Monte Carlo concentration, not MCMC mixing. Small ESS is
        explicitly unsuitable for accurate predictive scoring or Stage 6 use.
        """
        self.check_prediction(data)
        shifted = data.y if offset is None else data.y-np.asarray(offset)
        root = np.linalg.cholesky(data.noise)
        chunks = []
        for start in range(0, len(self.samples), 256):
            residual = solve_triangular(root, shifted[:, None]-data.design@self.samples[start:start+256].T, lower=True)
            chunks.append(-.5*np.sum(residual**2, axis=0)-np.log(np.diag(root)).sum()-.5*len(shifted)*np.log(2*np.pi))
        logl = np.concatenate(chunks)
        logsum = logsumexp(logl)
        ess = float(np.exp(2*logsum-logsumexp(2*logl)))
        return float(logsum-np.log(len(logl))), ess

    def predictive_evidence(self, data: StaticData, *, seed: int, draws: int = 16384) -> dict:
        """New-template log evidence via a Gaussian update and orthant correction.

        Avoids importance collapse on many pixels. Predictive scoring conditions
        on the training posterior; it does not update stored selection weights.
        Monte Carlo bounds include both positive-mass estimates. This is only a
        template diagnostic; science cannot enter hyperparameter selection.
        """
        self.check_prediction(data)
        if data.partition not in ('validation', 'heldout') or data.verified is not True:
            raise ValueError('Predictive model comparison uses verified template partitions only')
        if not isinstance(draws, int) or not 512 <= draws <= 262144:
            raise ValueError('Invalid bounded predictive sampling budget')
        mean, cov, logz, _ = gaussian_update(data, self.gaussian_mean, self.gaussian_covariance)
        root = np.linalg.cholesky(cov)
        rng = np.random.default_rng(seed)
        count = 0
        for start in range(0, draws, 1024):
            samples = mean+rng.standard_normal((min(1024, draws-start), len(mean)))@root.T
            count += int(np.all(samples >= 0, axis=1).sum())
        if count < 32:
            raise SamplingFailure('Predictive orthant mass unresolved', dict(accepted=count, proposals=draws))
        old_mass = self.diagnostics['accepted']/self.diagnostics['proposals']
        old_bounds = self.diagnostics['posterior_mass_95']
        bounds = _wilson(count, draws)
        return dict(log_score=float(logz+np.log(count/draws/old_mass)),
                    log_score_95=[float(logz+np.log(bounds[0]/old_bounds[1])),
                                  float(logz+np.log(bounds[1]/old_bounds[0]))],
                    accepted=count, proposals=draws)


def fit_static(data: StaticData, *, scale_arcsec: float, amplitude: float,
               host_mean: np.ndarray, source_prior_sigma: float,
               seed: int, draws: int = 16384, minimum_accepted: int = 512) -> StaticPosterior:
    """Fit verified training templates with a proper positive prior.

    H=mean+signed GP perturbation is truncated at H>=0; sources have independent
    half-normal priors. Truncation is part of the model, including its evidence
    normalizer, rather than clipping a fit/error bar. No boundary Gaussian errors.
    Failed orthant estimates raise with counts; finite evidence uncertainty is
    saved. Source prior scale and host mean must be declared before seeing science.
    """
    start = time.perf_counter()
    if data.partition != 'train' or data.verified is not True:
        raise ValueError('Only independently verified training templates may select the static model')
    h = len(data.host_xy_arcsec)
    hm = np.asarray(host_mean)
    if hm.shape != (h,) or not np.isfinite(hm).all() or np.any(hm < 0):
        raise ValueError('Declare a finite nonnegative host mean on the host grid')
    if not np.isfinite(source_prior_sigma) or source_prior_sigma <= 0:
        raise ValueError('Declare a positive source prior SD in integrated flux units')
    if not isinstance(draws, int) or not 512 <= draws <= 262144 or not 32 <= minimum_accepted <= draws:
        raise ValueError('Invalid bounded sampling budget')
    s = len(data.source_names)
    if draws*(h+s) > 8_000_000:
        raise ValueError('Posterior draw allocation exceeds reference budget')
    mean = np.r_[hm, np.zeros(s)]
    k = matern_covariance(data.host_xy_arcsec, scale_arcsec, amplitude)
    prior_cov = block_diag(k, np.eye(s)*source_prior_sigma**2) if s else k
    pm, pc, logz, w = gaussian_update(data, mean, prior_cov)
    rng = np.random.default_rng(seed)
    prior_good = 0
    accepted = []
    post_root, host_root = np.linalg.cholesky(pc), np.linalg.cholesky(k)
    for begin in range(0, draws, 1024):
        n = min(1024, draws-begin)
        prior_good += int(np.all(hm+rng.standard_normal((n, h))@host_root.T >= 0, axis=1).sum())
        proposal = pm+rng.standard_normal((n, len(mean)))@post_root.T
        accepted.append(proposal[np.all(proposal >= 0, axis=1)])
    samples = np.concatenate(accepted)
    n_good = len(samples)
    pi, qi = _wilson(prior_good, draws), _wilson(n_good, draws)
    diagnostics = dict(seed=seed, proposals=draws, accepted=n_good, prior_host_accepted=prior_good,
                       scale_arcsec=scale_arcsec, amplitude_jy_per_native_pixel=amplitude,
                       source_prior_sigma=source_prior_sigma, prior_host_mass_95=pi, posterior_mass_95=qi,
                       mass_interval_confidence=.975,
                       host_mean=hm.tolist(), sampling='independent Gaussian rejection; no MCMC',
                       conditioning_exposures=data.exposure_ids, geometry_uncertainty='conditional; not integrated')
    if prior_good < minimum_accepted or n_good < minimum_accepted:
        raise SamplingFailure('Positive mass insufficient; increase declared budget or change model explicitly', diagnostics)
    logz += np.log(n_good/prior_good)+s*np.log(2)
    logz_interval = (logz+np.log(qi[0]/(n_good/draws))-np.log(pi[1]/(prior_good/draws)),
                     logz+np.log(qi[1]/(n_good/draws))-np.log(pi[0]/(prior_good/draws)))
    estimate, cov = samples.mean(axis=0), np.atleast_2d(np.cov(samples, rowvar=False))
    sd = np.sqrt(np.diag(cov))
    if np.any(sd == 0) or not np.isfinite(cov).all():
        raise SamplingFailure('Unresolved posterior variance; no zero-error estimate returned', diagnostics)
    correlation = cov/np.outer(sd, sd)
    # Conservative posterior-predictive discrepancy; fitting has already used y.
    # This is not a uniform frequentist p-value or a science-bias acceptance gate.
    noise_root = np.linalg.cholesky(data.noise)
    residual = solve_triangular(noise_root, data.y[:, None]-data.design@samples[:512].T, lower=True)
    discrepancy = np.sum(residual**2, axis=0)
    ppc = float(np.mean(chi2.sf(discrepancy, len(data.y))))
    prior_root = np.linalg.cholesky(prior_cov)
    q = solve_triangular(prior_root, pm-mean, lower=True)
    r = solve_triangular(noise_root, data.y-data.design@mean, lower=True)
    normal_error = w.T@(w@q-r)+q
    # The parameter Gram matrix avoids allocating a full pixel-space U.
    _, eigenvectors = np.linalg.eigh(w.T@w)
    singular = np.linalg.svd(w, compute_uv=False)
    vh = eigenvectors[:, ::-1].T
    rank = int(np.sum(singular > (singular[0] if len(singular) else 1)*1e-10))
    diagnostics.update(log_evidence_95=list(map(float, logz_interval)),
                       likelihood_rank=rank, parameters=len(mean), likelihood_rank_deficient=rank < len(mean),
                       singular_values=singular.tolist(), weakest_prior_scaled_mode=vh[-1].tolist(),
                       maximum_absolute_component_correlation=float(np.max(np.abs(correlation-np.eye(len(mean))))),
                       mean_mc_se=(sd/np.sqrt(n_good)).tolist(), seconds=time.perf_counter()-start,
                       gaussian_normal_residual_relative=float(np.linalg.norm(normal_error)/max(np.linalg.norm(w.T@r), 1.)),
                       training_posterior_predictive_tail=ppc,
                       static_mismatch=ppc < 1e-4,
                       mismatch_threshold=1e-4,
                       mismatch_note='conservative posterior-predictive diagnostic; not calibrated science acceptance',
                       science_promotion_approved=False)
    return StaticPosterior(data, _freeze(samples), _freeze(estimate), _freeze(cov), float(logz), diagnostics,
                           _freeze(pm), _freeze(pc))


def model_weights(posteriors: tuple[StaticPosterior, ...], prior_weights: np.ndarray) -> dict:
    """Template evidence weights with orthant-MC uncertainty bounds.

    Supply total prior mass per model/grid cell (refining a grid must split mass).
    These probabilities condition on the enumerated candidates, finite footprint,
    fixed geometry and declared priors. Failed candidates MUST NOT be dropped and
    the remainder renormalized; the caller must keep comparison incomplete.
    """
    p = np.asarray(prior_weights, dtype=float)
    if not posteriors or p.shape != (len(posteriors),) or not np.isfinite(p).all() or np.any(p <= 0):
        raise ValueError('Positive explicit prior mass for every completed candidate required')
    first = posteriors[0].data
    for fit in posteriors[1:]:
        d = fit.data
        if (d.exposure_ids != first.exposure_ids or not np.array_equal(d.y, first.y)
                or not np.array_equal(d.noise, first.noise) or d.calibration_gauge != first.calibration_gauge):
            raise ValueError('Evidence comparison requires identical template pixels and noise')
    logw = np.array([f.log_evidence for f in posteriors])+np.log(p/p.sum())
    bounds = np.array([f.diagnostics['log_evidence_95'] for f in posteriors])+np.log(p/p.sum())[:, None]
    intervals = []
    for j in range(len(p)):
        lower, upper = bounds[:, 1].copy(), bounds[:, 0].copy()
        lower[j], upper[j] = bounds[j, 0], bounds[j, 1]
        intervals.append([np.exp(lower[j]-logsumexp(lower)), np.exp(upper[j]-logsumexp(upper))])
    return dict(weights=np.exp(logw-logsumexp(logw)), orthant_mc_weight_bounds=np.array(intervals),
                status='all_candidates_misspecified' if all(f.diagnostics['static_mismatch'] for f in posteriors) else 'conditional_comparison',
                caveat='Bonferroni approximate 95% evidence bounds; not simultaneous across all models')


def conditional_log_likelihood(posterior: StaticPosterior, science: StaticData,
                               transient_column: np.ndarray, flux: float) -> tuple[float, float]:
    """Stage 6 interface: E[p(science|static,f)|templates], never p(T) twice.

    This is a likelihood evaluation, not recovery or epoch grouping. Stage 6 must
    supply one-day grouped nonnegative fluxes and integrate geometry/epoch priors.
    Low predictive ESS must trigger more sampling or a better integral there.
    """
    p = np.asarray(transient_column)
    if science.partition != 'science' or p.shape != science.y.shape or not np.isfinite(p).all() or np.any(p < 0):
        raise ValueError('Science partition and nonnegative native transient response required')
    if not np.isfinite(flux) or flux < 0:
        raise ValueError('Physical transient flux must be nonnegative')
    return posterior.log_predictive(science, p*flux)


def mixture_moments(posteriors: tuple[StaticPosterior, ...], weights: np.ndarray) -> dict:
    """Align absent components at zero; retain within- and between-model covariance.

    Hyperparameter, nuclear-profile and structural candidates are all discrete
    alternatives here. This does not integrate unrepresented geometry or kernels.
    """
    p = np.asarray(weights, dtype=float)
    if p.shape != (len(posteriors),) or not len(p) or np.any(p < 0) or not np.isfinite(p).all() or not np.isclose(p.sum(), 1):
        raise ValueError('Normalized nonnegative candidate weights required')
    h = len(posteriors[0].data.host_xy_arcsec)
    names = tuple(dict.fromkeys(name for fit in posteriors for name in fit.data.source_names))
    means, covariances = [], []
    for fit in posteriors:
        if not np.array_equal(fit.data.host_xy_arcsec, posteriors[0].data.host_xy_arcsec):
            raise ValueError('Mixture host grids must match')
        indices = np.r_[np.arange(h), [h+names.index(n) for n in fit.data.source_names]].astype(int)
        m, c = np.zeros(h+len(names)), np.zeros((h+len(names), h+len(names)))
        m[indices], c[np.ix_(indices, indices)] = fit.mean, fit.covariance
        means.append(m)
        covariances.append(c)
    means = np.array(means)
    mean = p@means
    within = np.einsum('i,ijk->jk', p, covariances)
    delta = means-mean
    between = np.einsum('i,ij,ik->jk', p, delta, delta)
    return dict(mean=mean, covariance=within+between, within_covariance=within,
                between_covariance=between, source_names=names)


def mixture_conditional_likelihood(posteriors: tuple[StaticPosterior, ...],
                                   science_by_model: tuple[StaticData, ...],
                                   weights: np.ndarray, transient_column: np.ndarray, flux: float) -> dict:
    """Stage 6 likelihood including template-derived model/hyperparameter weights.

    The same science pixels/noise and native transient response must be supplied
    for each structural model. A low importance ESS returns an explicit failed
    numerical diagnostic; it is not a precise flux inference. Selection weights
    remain template-only and no template likelihood is multiplied again.
    """
    p = np.asarray(weights, dtype=float)
    if (not posteriors or len(science_by_model) != len(posteriors) or p.shape != (len(posteriors),)
            or not np.isfinite(p).all() or np.any(p < 0) or not np.isclose(p.sum(), 1)):
        raise ValueError('Matched models and normalized nonnegative weights required')
    first = science_by_model[0]
    for data in science_by_model[1:]:
        if data.exposure_ids != first.exposure_ids or not np.array_equal(data.y, first.y) or not np.array_equal(data.noise, first.noise):
            raise ValueError('Every candidate must evaluate identical science data/noise')
    scores = [conditional_log_likelihood(fit, data, transient_column, flux)
              for fit, data in zip(posteriors, science_by_model)]
    use = p > 0
    value = logsumexp(np.log(p[use])+np.array([s[0] for s in scores])[use])
    ess = np.array([s[1] for s in scores])
    return dict(log_likelihood=float(value), component_importance_ess=ess.tolist(),
                status='insufficient_predictive_samples' if np.any(ess[use] < 32) else 'evaluated',
                interpretation='likelihood only; no confidence interval or science promotion')


def input_digest(data: StaticData) -> str:
    """Checksum numeric likelihood inputs for artifact comparison and provenance."""
    digest = hashlib.sha256()
    for value in (data.y, data.noise, data.design, data.host_xy_arcsec):
        digest.update(value.tobytes())
    digest.update(repr((data.source_names, data.exposure_ids, data.partition, data.provenance, data.calibration_gauge)).encode())
    return digest.hexdigest()
