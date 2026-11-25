"""Opt-in, template-only nucleus inference; no transient or flexible local GP.

Surface-brightness profiles are integrals over sky source positions of the
already detector-integrated native PRF. Physical amplitudes are nonnegative;
per-frame calibrated background corrections may be signed. Absolute flux units
inherit the supplied PRF convention (warm IRAC absolute calibration is open).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
import time
import resource
import sys
from typing import Any, Literal

import numpy as np
from astropy.wcs import WCS
from numpy.polynomial.hermite import hermgauss
from scipy.ndimage import map_coordinates
from scipy.optimize import least_squares
from scipy.linalg import cho_factor, cho_solve
from scipy.stats import chi2 as chi2_distribution

from .isolated_astrometry import StarFrame, _frame_identity, _freeze, star_response, response_derivatives
from .point_response import NativePRF

Model = Literal['point', 'gaussian', 'core_envelope']
MODELS: tuple[Model, ...] = ('point', 'gaussian', 'core_envelope')


@lru_cache(maxsize=16)
def _quadrature(order: int) -> tuple[np.ndarray, np.ndarray]:
    """Standard normal product quadrature, with positive unit-sum weights."""
    if not isinstance(order, int) or not 3 <= order <= 65:
        raise ValueError('Quadrature order must be an integer from 3 to 65')
    x, w = hermgauss(order)
    xx, yy = np.meshgrid(x*np.sqrt(2), x*np.sqrt(2))
    return _freeze(np.array([xx.ravel(), yy.ravel()])), _freeze(np.outer(w, w).ravel()/np.pi)


def gaussian_response(frame: StarFrame, reference: WCS, center_xy: np.ndarray,
                      profile: np.ndarray, translation_xy: np.ndarray, *, order: int = 11) -> np.ndarray:
    """Forward-integrate a unit-flux elliptical Gaussian in reference sky pixels.

    ``profile=(sigma_x,sigma_y,correlation)`` defines a positive-semidefinite
    intrinsic covariance. This integrates *source positions*, never detector
    pixels a second time. Width zero on both axes gives the point renderer
    exactly. No crop renormalization occurs. Reference WCS must be a small TAN
    sky patch; nonlinear WCS is evaluated separately at every quadrature node.
    Check order convergence for each supported width/instrument combination.
    """
    sx, sy, rho = np.asarray(profile, dtype=float)
    if not np.isfinite([sx, sy, rho]).all() or min(sx, sy) < 0 or abs(rho) >= 1:
        raise ValueError('Invalid intrinsic Gaussian covariance')
    if sx == sy == 0:
        return star_response(frame, reference, np.asarray(center_xy), np.asarray(translation_xy))
    nodes, weights = _quadrature(order)
    root = np.array([[sx, 0.], [rho*sy, sy*np.sqrt(1-rho*rho)]])
    points = np.asarray(center_xy)[:, None] + root @ nodes
    response = frame.response
    if isinstance(response, NativePRF):
        # Batched node sampling avoids Python calls and repeated WCS transforms.
        sky = reference.pixel_to_world_values(*points)
        xy = np.array(frame.wcs.world_to_pixel_values(*sky)) + np.asarray(translation_xy)[:, None]
        yy, xx = np.indices(frame.data.shape, dtype=float)
        # Chunk the positive integral to bound workspace for high quadrature order.
        result = np.zeros(frame.data.shape)
        chunk = max(1,min(64,65536//frame.data.size))
        for start in range(0, len(weights), chunk):
            stop = start+chunk
            tx = response.center_xy[0]+(xx[None]-xy[0, start:stop, None, None])*response.oversample
            ty = response.center_xy[1]+(yy[None]-xy[1, start:stop, None, None])*response.oversample
            values = map_coordinates(response.values, [ty, tx], order=1, mode='constant', cval=0., prefilter=False)
            result += np.einsum('q,qij->ij', weights[start:stop], values)/response.normalization
        return result
    # The strict grid renderer selects detector-position calibration for each node.
    return sum(weight*star_response(frame, reference, pos, translation_xy)
               for pos, weight in zip(points.T, weights))


def gaussian_derivatives(frame: StarFrame, reference: WCS, center_xy: np.ndarray,
                         profile: np.ndarray, shift: np.ndarray, *, order: int = 11) -> np.ndarray:
    """Derivatives for sky x/y, sigma x/y, correlation, detector x/y.

    For a native table these differentiate the same bilinear source integral
    analytically, including the covariance-root derivatives. Only the WCS local
    Jacobian uses finite differences. Grid-provider derivatives retain full
    spatial calibration variation and use finite differences instead.
    """
    response = frame.response
    if not isinstance(response, NativePRF):
        p = np.r_[center_xy, profile, shift]
        out = []
        for j in range(7):
            a, b = p.copy(), p.copy()
            a[j] += 1e-4
            b[j] -= 1e-4
            if j in (2, 3) and b[j] < 0:
                b[j] = p[j]
            out.append((gaussian_response(frame, reference, a[:2], a[2:5], a[5:], order=order)-
                        gaussian_response(frame, reference, b[:2], b[2:5], b[5:], order=order))/(a[j]-b[j]))
        return np.array(out)
    sx, sy, rho = profile
    z, weights = _quadrature(order)
    root = np.sqrt(1-rho*rho)
    points = np.asarray(center_xy)[:, None]+np.array([sx*z[0], sy*(rho*z[0]+root*z[1])])
    def project(points: np.ndarray) -> np.ndarray:
        return np.array(frame.wcs.world_to_pixel_values(*reference.pixel_to_world_values(*points)))
    native = project(points)+np.asarray(shift)[:, None]
    geometry = np.array([(project(points+d[:, None])-project(points-d[:, None]))/2e-3 for d in np.eye(2)*1e-3])
    yy, xx = np.indices(frame.data.shape, dtype=float)
    out = np.zeros((7, *frame.data.shape))
    chunk = max(1,min(64,65536//frame.data.size))
    for start in range(0, len(weights), chunk):
        sl = slice(start, start+chunk)
        tx = response.center_xy[0]+(xx[None]-native[0, sl, None, None])*response.oversample
        ty = response.center_xy[1]+(yy[None]-native[1, sl, None, None])*response.oversample
        valid = (tx>=0)&(ty>=0)&(tx<response.values.shape[1]-1)&(ty<response.values.shape[0]-1)
        ix=np.clip(np.floor(tx).astype(int),0,response.values.shape[1]-2)
        iy=np.clip(np.floor(ty).astype(int),0,response.values.shape[0]-2)
        u,v=tx-ix,ty-iy
        a,b=response.values[iy,ix],response.values[iy,ix+1]
        c,d=response.values[iy+1,ix],response.values[iy+1,ix+1]
        dx=-response.oversample/response.normalization*((1-v)*(b-a)+v*(d-c))*valid
        dy=-response.oversample/response.normalization*((1-u)*(c-a)+u*(d-b))*valid
        skyx=geometry[0,0,sl,None,None]*dx+geometry[0,1,sl,None,None]*dy
        skyy=geometry[1,0,sl,None,None]*dx+geometry[1,1,sl,None,None]*dy
        w=weights[sl]
        for j, derivative in enumerate((skyx,skyy,z[0,sl,None,None]*skyx,
                                        (rho*z[0,sl,None,None]+root*z[1,sl,None,None])*skyy,
                                        sy*(z[0,sl,None,None]-rho/root*z[1,sl,None,None])*skyy,dx,dy)):
            out[j] += np.einsum('q,qij->ij',w,derivative)
    return out


@dataclass(frozen=True)
class TemplateFrame:
    """Verified template plus independent astrometric/background calibration.

    Registration is in native detector pixels; covariance is 2x2 and either
    positive definite or exactly zero for a fixed gauge. Calibrations must come
    from external or disjoint field-star/sky pixels, never the nuclear residual.
    ``background_sigma`` is needed for predictions of unseen exposures; training
    backgrounds are fitted without this prior. Sigma data use the actual BCD
    uncertainty product, with the same unit conversion as science pixels.
    """
    image: StarFrame
    exposure_id: str
    split: Literal['train', 'validation', 'heldout']
    verified_template: bool
    registration_xy: np.ndarray = field(default_factory=lambda: np.zeros(2))
    registration_covariance: np.ndarray = field(default_factory=lambda: np.zeros((2, 2)))
    background: float = 0.
    background_sigma: float = 0.
    provenance: str = ''

    def __post_init__(self) -> None:
        if self.verified_template is not True or not self.exposure_id or self.split not in ('train', 'validation', 'heldout'):
            raise ValueError('Explicit verified template, unique ID, and valid split are required')
        r, c = np.asarray(self.registration_xy), np.asarray(self.registration_covariance)
        if r.shape != (2,) or c.shape != (2, 2) or not np.isfinite(r).all() or not np.isfinite(c).all():
            raise ValueError('Invalid registration calibration')
        if not np.allclose(c, c.T, rtol=0, atol=1e-15) or (np.any(c) and np.linalg.eigvalsh(c).min() <= 0):
            raise ValueError('Registration covariance must be positive definite or exactly zero')
        if not np.isfinite([self.background, self.background_sigma]).all() or self.background_sigma < 0:
            raise ValueError('Invalid background calibration')
        if not self.provenance:
            raise ValueError('Template/noise/registration provenance is required')
        object.__setattr__(self, 'registration_xy', _freeze(r))
        object.__setattr__(self, 'registration_covariance', _freeze(c))


def validate_frames(frames: tuple[TemplateFrame, ...]) -> None:
    """Reject duplicated exposures and incompatible shared-amplitude conventions."""
    if not frames or len({f.exposure_id for f in frames}) != len(frames):
        raise ValueError('Need nonempty frames with unique exposure IDs')
    conventions = {(f.image.response.channel, f.image.response.mission,
                    f.image.response.convention, getattr(f.image.response, 'synthetic', False)) for f in frames}
    if len(conventions) != 1:
        raise ValueError('Shared nucleus flux requires one channel, mission, and normalization convention')
    if frames[0].image.response.channel != 'ch2':
        raise ValueError('Stage 3 scientific fitting is restricted to ch2')


@dataclass(frozen=True)
class NucleusScene:
    """Fixed catalog stars and a controlled positive smooth host, in sky pixels.

    Star locations and host shape are supplied independently, not tuned against
    transient flux. Host width is deliberately fixed, with one nonnegative
    amplitude. Their structural uncertainty is an explicit limitation of this
    controlled Stage 3 experiment. Gaussian widths scale with reference-WCS SR.
    """
    reference: WCS
    center_xy: np.ndarray
    star_positions: np.ndarray
    host_center_xy: np.ndarray
    host_profile: np.ndarray
    flux_scale: float = .01
    centroid_radius: float = 1.
    maximum_width: float = 1.5
    quadrature_order: int = 11
    host_order: int = 35
    experimental_real_quadrature: bool = False

    def __post_init__(self) -> None:
        for name, shape in (('center_xy', (2,)), ('host_center_xy', (2,)), ('host_profile', (3,))):
            a = np.asarray(getattr(self, name), dtype=float)
            if a.shape != shape or not np.isfinite(a).all():
                raise ValueError(f'Invalid {name}')
            object.__setattr__(self, name, _freeze(a))
        stars = np.asarray(self.star_positions, dtype=float)
        if stars.ndim != 2 or stars.shape[1] != 2 or not len(stars) or not np.isfinite(stars).all():
            raise ValueError('Need explicit finite neighboring star positions')
        if not np.isfinite([self.flux_scale, self.centroid_radius, self.maximum_width]).all() or min(self.flux_scale, self.centroid_radius, self.maximum_width) <= 0:
            raise ValueError('Flux scale and search domains must be positive')
        if min(self.host_profile[:2]) <= self.maximum_width or abs(self.host_profile[2]) >= 1:
            raise ValueError('Controlled smooth host must be broader than nucleus domain')
        _quadrature(self.quadrature_order)
        _quadrature(self.host_order)
        object.__setattr__(self, 'star_positions', _freeze(stars))
        object.__setattr__(self, 'reference', self.reference.deepcopy())


class _Problem:
    """One explicit parameter layout and renderer shared by fit and prediction."""
    def __init__(self, frames: tuple[TemplateFrame, ...], scene: NucleusScene, model: Model):
        self.frames, self.scene, self.model = frames, scene, model
        self.nc = 2 if model == 'core_envelope' else 1
        self.na = self.nc + len(scene.star_positions) + 1
        self.ng = self.na + 2 + (3 if model != 'point' else 0)
        self.reg_indices: dict[int, int] = {}
        k = self.ng
        for i, frame in enumerate(frames):
            if np.any(frame.registration_covariance):
                self.reg_indices[i] = k
                k += 2
        self.bg0 = k
        self.size = k+len(frames)
        self.cache: dict[tuple[int, bytes], tuple[np.ndarray, ...]] = {}
        self.calls = 0
        self.derivative_cache: dict[tuple[int, bytes], tuple[np.ndarray, ...]] = {}

    def static(self, frame: TemplateFrame, shift: np.ndarray) -> tuple[np.ndarray, ...]:
        key = (id(frame), shift.tobytes())
        if key not in self.cache:
            s = self.scene
            # Bound geometry cache; never reuse across fits or changed calibrations.
            if len(self.cache) >= 48:
                self.cache.clear()
            self.cache[key] = tuple(star_response(frame.image, s.reference, p, shift) for p in s.star_positions) + (
                gaussian_response(frame.image, s.reference, s.host_center_xy, s.host_profile, shift, order=s.host_order),)
        return self.cache[key]

    def render(self, p: np.ndarray, frame: TemplateFrame, shift: np.ndarray, background: float) -> np.ndarray:
        s = self.scene
        center = p[self.na:self.na+2]
        point = star_response(frame.image, s.reference, center, shift)
        if self.model == 'point':
            nuclear = (point,)
        else:
            extended = gaussian_response(frame.image, s.reference, center, p[self.na+2:self.ng], shift, order=s.quadrature_order)
            nuclear = (point, extended) if self.nc == 2 else (extended,)
        columns = nuclear + self.static(frame, shift)
        self.calls += 1
        return background + s.flux_scale*np.einsum('k,kij->ij', p[:self.na], columns)

    def fitted_render(self, p: np.ndarray, i: int) -> np.ndarray:
        frame = self.frames[i]
        shift = frame.registration_xy
        if i in self.reg_indices:
            j = self.reg_indices[i]
            shift = shift+p[j:j+2]
        return self.render(p, frame, shift, p[self.bg0+i]*self.scene.flux_scale)

    def residual(self, p: np.ndarray) -> np.ndarray:
        blocks = []
        for i, frame in enumerate(self.frames):
            f = frame.image
            blocks.append(((self.fitted_render(p, i)-f.data)/f.sigma)[f.valid])
            if i in self.reg_indices:
                j = self.reg_indices[i]
                blocks.append(np.linalg.solve(np.linalg.cholesky(frame.registration_covariance), p[j:j+2]))
        return np.concatenate(blocks)

    def jacobian(self, p: np.ndarray) -> np.ndarray:
        blocks = []
        for i, frame in enumerate(self.frames):
            f, s = frame.image, self.scene
            shift = frame.registration_xy.copy()
            if i in self.reg_indices:
                j = self.reg_indices[i]
                shift += p[j:j+2]
            pos = p[self.na:self.na+2]
            point = star_response(f, s.reference, pos, shift)
            sky, detector = response_derivatives(f, s.reference, pos, shift)
            if self.model == 'point':
                nuclear = (point,)
                center_derivative, registration_derivative = p[0]*sky, p[0]*detector
                profile_derivative = None
            else:
                profile = p[self.na+2:self.ng]
                extended = gaussian_response(f, s.reference, pos, profile, shift, order=s.quadrature_order)
                derivatives = gaussian_derivatives(f, s.reference, pos, profile, shift, order=s.quadrature_order)
                amp = p[self.nc-1]
                center_derivative, registration_derivative = amp*derivatives[:2], amp*derivatives[5:]
                profile_derivative = amp*derivatives[2:5]
                nuclear = (point, extended) if self.nc == 2 else (extended,)
                if self.nc == 2:
                    center_derivative += p[0]*sky
                    registration_derivative += p[0]*detector
            response_columns = nuclear+self.static(frame, shift)
            sigma = f.sigma[f.valid]
            jac = np.zeros((int(f.valid.sum()), self.size))
            jac[:, :self.na] = s.flux_scale*np.column_stack([a[f.valid]/sigma for a in response_columns])
            jac[:, self.bg0+i] = s.flux_scale/sigma
            jac[:, self.na:self.na+2] = (s.flux_scale*center_derivative[:, f.valid]/sigma).T
            if profile_derivative is not None:
                jac[:, self.na+2:self.ng] = (s.flux_scale*profile_derivative[:, f.valid]/sigma).T
            if i in self.reg_indices:
                key=(id(frame),shift.tobytes())
                if key not in self.derivative_cache:
                    if len(self.derivative_cache)>=48:
                        self.derivative_cache.clear()
                    self.derivative_cache[key] = tuple(response_derivatives(f,s.reference,xy,shift)[1] for xy in s.star_positions)+(
                        gaussian_derivatives(f,s.reference,s.host_center_xy,s.host_profile,shift,order=s.host_order)[5:],)
                for amp, derivative in zip(p[self.nc:self.na], self.derivative_cache[key]):
                    registration_derivative += amp*derivative
                j=self.reg_indices[i]
                jac[:, j:j+2] = (s.flux_scale*registration_derivative[:, f.valid]/sigma).T
                prior=np.zeros((2,self.size))
                prior[:,j:j+2]=np.linalg.solve(np.linalg.cholesky(frame.registration_covariance),np.eye(2))
                blocks.extend((jac,prior))
            else:
                blocks.append(jac)
        return np.vstack(blocks)


def _covariance_factor(covariance: np.ndarray) -> np.ndarray:
    values, vectors = np.linalg.eigh(covariance)
    if values.min() < -1e-9*max(values.max(), 1e-30):
        raise ValueError('Predictive covariance is not positive semidefinite')
    return vectors*np.sqrt(np.maximum(values, 0.))


def predictive_score(residual: np.ndarray, sigma: np.ndarray, factor: np.ndarray) -> dict[str, float]:
    """Exact Gaussian score for diag(sigma²)+factor@factor.T using Woodbury.

    Retains off-diagonal predictive covariance. Pointwise standardized residuals
    are useful plots but are not a full whitening or this likelihood statistic.
    """
    r, s, u = np.asarray(residual), np.asarray(sigma), np.asarray(factor)
    if r.ndim != 1 or s.shape != r.shape or u.ndim != 2 or u.shape[0] != len(r):
        raise ValueError('Incompatible predictive arrays')
    if not np.isfinite(r).all() or not np.isfinite(s).all() or not np.isfinite(u).all() or np.any(s <= 0):
        raise ValueError('Prediction requires finite arrays and positive sigma')
    z, v = r/s, u/s[:, None]
    middle = np.eye(v.shape[1])+v.T@v
    chol = cho_factor(middle)
    projection = v.T@z
    chi2 = float(z@z-projection@cho_solve(chol, projection))
    logdet = float(2*np.log(s).sum()+2*np.log(np.diag(chol[0])).sum())
    return dict(chi2=chi2, logdet=logdet, nll=.5*(chi2+logdet+len(r)*np.log(2*np.pi)))


@dataclass
class NucleusFit:
    """Constrained template solution and full local nuisance covariance.

    Interior covariance is a local Gaussian approximation, not validated
    coverage. At active bounds/rank loss it is unavailable; predictions may be
    examined conditionally but must not be labeled uncertainty-propagated.
    """
    model: Model
    scene: NucleusScene
    parameters: np.ndarray
    covariance: np.ndarray
    diagnostics: dict[str, Any]
    problem: _Problem = field(repr=False)
    identities: tuple = field(repr=False)
    reference_identity: str = field(repr=False)

    def _check(self) -> None:
        if self.scene.reference.to_header(relax=True).tostring() != self.reference_identity:
            raise ValueError('Reference WCS changed; refit required')
        if tuple(_frame_identity(f.image) for f in self.problem.frames) != self.identities:
            raise ValueError('Training geometry/calibration changed; refit required')

    @property
    def center_xy(self) -> np.ndarray:
        return self.parameters[self.problem.na:self.problem.na+2].copy()

    @property
    def profile(self) -> np.ndarray:
        return np.zeros(3) if self.model == 'point' else self.parameters[self.problem.na+2:self.problem.ng].copy()

    @property
    def flux(self) -> np.ndarray:
        return self.parameters[:self.problem.na]*self.scene.flux_scale

    def training_predictions(self) -> tuple[np.ndarray, ...]:
        self._check()
        return tuple(self.problem.fitted_render(self.parameters, i) for i in range(len(self.problem.frames)))

    def predict(self, frame: TemplateFrame, *, propagate: bool = True) -> tuple[np.ndarray, np.ndarray]:
        """Predict unseen template using independent registration/sky calibration.

        Returned factor U retains full covariance through C=diag(sigma²)+UUᵀ.
        The training posterior is used once; unseen pixels are never refitted.
        Predictions across separate exposures also share global uncertainty;
        ``score_frames`` assembles those cross-exposure blocks correctly.
        """
        self._check()
        validate_frames((self.problem.frames[0], frame))
        if frame.exposure_id in {f.exposure_id for f in self.problem.frames}:
            raise ValueError('Use training_predictions for fitted exposures')
        # New prediction boundary: invalidate per-exposure static caches so mutated
        # unseen geometry/calibration cannot return stale host/star responses.
        self.problem.cache.clear()
        mean = self.problem.render(self.parameters, frame, frame.registration_xy, frame.background)
        if not propagate:
            return mean, np.empty((mean.size, 0))
        n = self.problem.ng
        if not np.isfinite(self.covariance).all():
            raise ValueError('Boundary/rank-limited fit has no local predictive covariance')
        prediction_problem = _Problem((frame,), self.scene, self.model)
        prediction_problem.cache = self.problem.cache
        vector = np.r_[self.parameters[:n], np.zeros(prediction_problem.bg0-n),
                       frame.background/self.scene.flux_scale]
        jac = prediction_problem.jacobian(vector)[:int(frame.image.valid.sum())]
        valid = frame.image.valid.ravel()
        sigma = frame.image.sigma[frame.image.valid]
        global_factor = np.zeros((mean.size,n))
        global_factor[valid] = (jac[:,:n]*sigma[:,None]) @ _covariance_factor(self.covariance[:n,:n])
        local_factor = np.zeros((mean.size,2))
        if 0 in prediction_problem.reg_indices:
            j=prediction_problem.reg_indices[0]
            local_factor[valid] = (jac[:,j:j+2]*sigma[:,None]) @ _covariance_factor(frame.registration_covariance)
        return mean, np.column_stack((global_factor, local_factor, np.full(mean.size, frame.background_sigma)))

    def score_frames(self, frames: tuple[TemplateFrame, ...], *, propagate: bool = True) -> dict[str, float]:
        """Joint predictive score including cross-exposure global covariance."""
        validate_frames(self.problem.frames+tuple(frames))
        predictions = [self.predict(f, propagate=propagate) for f in frames]
        residual = np.concatenate([(f.image.data-m)[f.image.valid] for f, (m, _) in zip(frames, predictions)])
        sigma = np.concatenate([f.image.sigma[f.image.valid] for f in frames])
        n = self.problem.ng
        if propagate:
            factor = np.zeros((len(residual), n+3*len(frames)))
            offset = 0
            for i, (frame, (_, u)) in enumerate(zip(frames, predictions)):
                v = u[frame.image.valid.ravel()]
                rows = slice(offset, offset+len(v))
                factor[rows, :n] = v[:, :n]
                factor[rows, n+3*i:n+3*i+3] = v[:, n:]
                offset += len(v)
        else:
            factor = np.empty((len(residual), 0))
        return predictive_score(residual, sigma, factor)


class NucleusConvergenceError(RuntimeError):
    """A rejected fit with numerical evidence; never a successful science result."""
    def __init__(self, message: str, diagnostics: dict[str, Any], arrays: dict[str, np.ndarray]):
        super().__init__(message)
        self.diagnostics = diagnostics
        self.arrays = arrays


def fit_nucleus(frames: tuple[TemplateFrame, ...], scene: NucleusScene, model: Model = 'point',
                *, max_nfev: int = 100, initial_width: float = .35) -> NucleusFit:
    """Joint nonnegative source fit on training templates only.

    Fitted centroid is common sky position, independent of native registration.
    Known catalog stars and registration priors set the reference gauge. All
    parameters (including backgrounds, registration, centroid, and profile) enter
    the same likelihood/Jacobian. No uncertainty rescaling by residual chi-square.
    Search boundaries are reported and disallow scientific promotion; model
    degeneracy never becomes a precise zero uncertainty.
    """
    setup = time.perf_counter()
    frames = tuple(frames)
    validate_frames(frames)
    if not getattr(frames[0].image.response, 'synthetic', False) and not scene.experimental_real_quadrature:
        raise ValueError('Real-profile quadrature is not validated; explicit experimental_real_quadrature opt-in required, including the smooth host')
    if model not in MODELS or any(f.split != 'train' for f in frames):
        raise ValueError('Known model and training-only templates required')
    identities = tuple(_frame_identity(f.image) for f in frames)
    q = _Problem(frames, scene, model)
    observations = sum(int(f.image.valid.sum()) for f in frames)+2*len(q.reg_indices)
    if observations*q.size > 2_000_000:
        raise MemoryError('Stage 3 dense reference Jacobian exceeds 2,000,000 elements; production scaling is unvalidated')
    p = np.r_[np.full(q.na, .5), scene.center_xy,
              [] if model == 'point' else [initial_width, initial_width*.8, .05],
              np.zeros(q.bg0-q.ng), [f.background/scene.flux_scale for f in frames]]
    p[q.na-1] = 2.
    lower = np.r_[np.zeros(q.na), scene.center_xy-scene.centroid_radius,
                  [] if model == 'point' else [0., 0., -.95],
                  np.full(q.bg0-q.ng, -1.), np.full(len(frames), -np.inf)]
    upper = np.r_[np.full(q.na, np.inf), scene.center_xy+scene.centroid_radius,
                  [] if model == 'point' else [scene.maximum_width, scene.maximum_width, .95],
                  np.full(q.bg0-q.ng, 1.), np.full(len(frames), np.inf)]
    history: list[float] = []
    def residual(v: np.ndarray) -> np.ndarray:
        r = q.residual(v)
        history.append(float(r@r))
        return r
    start = time.perf_counter()
    result = least_squares(residual, p, jac=q.jacobian, bounds=(lower, upper), x_scale='jac',
                           ftol=1e-11, xtol=1e-11, gtol=1e-8, max_nfev=max_nfev)
    attempts = [dict(message=str(result.message),optimality=float(result.optimality),cost=float(result.cost),nfev=int(result.nfev))]
    # A fresh trust region can recover from premature relative-cost termination
    # near small profile eigenvalues. Preserve the objective and stationarity
    # criterion; never promote the failed first iterate or relax its threshold.
    if max_nfev > 1:
        for _ in range(2):
            if result.success and result.optimality <= 1e-3:
                break
            candidate = least_squares(residual, result.x, jac=q.jacobian, bounds=(lower,upper),x_scale='jac',
                                      ftol=None,xtol=1e-14,gtol=1e-8,max_nfev=max_nfev)
            attempts.append(dict(message=str(candidate.message),optimality=float(candidate.optimality),cost=float(candidate.cost),nfev=int(candidate.nfev)))
            if candidate.cost <= result.cost:
                result = candidate
    elapsed = time.perf_counter()-start
    if not result.success or not np.isfinite(result.fun).all() or result.optimality > 1e-3:
        raise NucleusConvergenceError(
            f'Nucleus fit did not converge: {result.message}; optimality={result.optimality:g}',
            dict(attempts=attempts,optimality=float(result.optimality),message=str(result.message),nfev=int(result.nfev),
                 setup_seconds=start-setup,solve_seconds=elapsed,renderer_calls=q.calls),
            dict(parameters=result.x,objective=np.array(history),jacobian=result.jac,
                 training_model=np.array([q.fitted_render(result.x,i) for i in range(len(frames))])))
    if tuple(_frame_identity(f.image) for f in frames) != identities:
        raise RuntimeError('Calibration/geometry changed during fit')
    _, singular, vt = np.linalg.svd(result.jac, full_matrices=False)
    rank = int(np.sum(singular > singular[0]*max(result.jac.shape)*np.finfo(float).eps))
    near_boundary = (result.x-lower < 1e-5) | (upper-result.x < 1e-5)
    covariance = np.full((q.size, q.size), np.nan)
    if rank == q.size and not np.any(near_boundary):
        covariance = (vt.T/singular**2)@vt
    npixels = sum(int(f.image.valid.sum()) for f in frames)
    pixel_chi2 = float(sum(np.sum(((q.fitted_render(result.x, i)-f.image.data)[f.image.valid]/
                                    f.image.sigma[f.image.valid])**2) for i, f in enumerate(frames)))
    diagnostics = dict(training_pixel_chi2=pixel_chi2,
                       bic_regular_approximation=pixel_chi2+q.size*np.log(npixels),
                       complexity_note='BIC is descriptive only: boundaries/priors make regular asymptotics unreliable',
                       chi2_including_registration_prior=float(result.fun@result.fun),
                       parameter_count=q.size, pixel_count=npixels, rank=rank,
                       singular_values=singular.tolist(), condition_number=float(singular[0]/singular[-1]) if singular[-1]>0 else float('inf'), active_or_near_bounds=near_boundary.tolist(),
                       covariance_available=bool(np.isfinite(covariance).all()),
                       covariance_interpretation='local joint curvature; no boundary/coverage certification',
                       objective_history=history, attempts=attempts, optimality=float(result.optimality),
                       setup_seconds=start-setup, solve_seconds=elapsed, renderer_calls=q.calls,
                       peak_rss_bytes=int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)*(1 if sys.platform=='darwin' else 1024),
                       jacobian_elements=observations*q.size,
                       science_promotion_approved=False)
    return NucleusFit(model, scene, _freeze(result.x), _freeze(covariance), diagnostics, q, identities, scene.reference.to_header(relax=True).tostring())


def compare_models(frames: tuple[TemplateFrame, ...], scene: NucleusScene) -> dict[str, Any]:
    """Fit on train, compare validation, report untouched held-out scores.

    Default remains point. Alternatives and ambiguity are explicit: validation
    scores alone are evidence, not an automatic physical resolution decision.
    Boundary/rank-limited fits report conditional scores separately; they cannot
    beat propagated scores by silently pretending their uncertainty is zero.
    """
    frames = tuple(frames)
    validate_frames(frames)
    groups = {name: tuple(f for f in frames if f.split == name) for name in ('train', 'validation', 'heldout')}
    if any(not v for v in groups.values()):
        raise ValueError('Independent training, validation and held-out exposures required')
    fits, records, failed_arrays = {}, {}, {}
    for model in MODELS:
        try:
            fit = fit_nucleus(groups['train'], scene, model)
            fits[model] = fit
            record = dict(status='converged', **fit.diagnostics)
            for name in ('validation', 'heldout'):
                record[name+'_conditional'] = fit.score_frames(groups[name], propagate=False)
                record[name+'_predictive'] = (fit.score_frames(groups[name]) if fit.diagnostics['covariance_available'] else None)
            record['centroid_xy'] = fit.center_xy.tolist()
            record['profile'] = fit.profile.tolist()
            record['flux'] = fit.flux.tolist()
            records[model] = record
        except (ValueError, RuntimeError) as error:
            records[model] = dict(status='failed', error=str(error))
            if isinstance(error, NucleusConvergenceError):
                records[model]['diagnostics'] = error.diagnostics
                failed_arrays[model] = error.arrays
    return dict(default='point', selection='No automatic promotion; retain alternatives and model uncertainty',
                fits=fits, metrics=records, failed_arrays=failed_arrays, science_promotion_approved=False)


def residual_signatures(frame: TemplateFrame, scene: NucleusScene, model: np.ndarray,
                        *, probe_xy: np.ndarray, center_xy: np.ndarray | None = None,
                        predictive_factor: np.ndarray | None = None, multiplicity: int = 1) -> dict[str, Any]:
    """Signed sky derivative/dipole and nearby-source leakage diagnostics.

    Modes are jointly fitted with a constant nuisance; their full covariance is
    returned. ``source_bias`` is a signed projection of residual onto an injected
    source response after removing a constant, not a physical negative flux fit.
    The injected nonnegative flux recovery is max(0, truth+bias). This is a fixed
    static-model leakage test, not Stage 6 joint science inference.
    """
    f = frame.image
    center = scene.center_xy if center_xy is None else np.asarray(center_xy)
    if multiplicity < 1:
        raise ValueError('Multiplicity must be positive')
    modes = []
    for d in np.eye(2)*1e-3:
        modes.append((star_response(f, scene.reference, center+d, frame.registration_xy)-
                      star_response(f, scene.reference, center-d, frame.registration_xy))/2e-3)
    p = star_response(f, scene.reference, probe_xy, frame.registration_xy)
    valid, sigma = f.valid, f.sigma[f.valid]
    residual = (f.data-model)[valid]
    design = np.column_stack([a[valid]/sigma for a in modes]+[1/sigma])
    u = np.empty((len(sigma), 0)) if predictive_factor is None else predictive_factor[valid.ravel()]/sigma[:,None]
    middle = cho_factor(np.eye(u.shape[1])+u.T@u)
    def solve(values: np.ndarray) -> np.ndarray:
        return values-u@cho_solve(middle,u.T@values)
    information_modes = design.T@solve(design)
    rank = np.linalg.matrix_rank(information_modes)
    cov = np.linalg.pinv(information_modes)
    coef = cov@design.T@solve(residual/sigma)
    mode_chi2 = float(coef[:2]@np.linalg.pinv(cov[:2,:2])@coef[:2])
    pvalue = min(1.,float(chi2_distribution.sf(mode_chi2,2))*multiplicity)
    weight = 1/sigma**2
    centered = p[valid]-np.average(p[valid], weights=weight)
    information = float(centered@(centered*weight))
    bias = float(centered@(residual*weight)/information)
    structure = (np.column_stack([a[valid] for a in modes])@coef[:2])
    return dict(derivative_chi2=mode_chi2, bonferroni_pvalue=pvalue, multiplicity=multiplicity,
                pvalue_interpretation='approximate held-out Gaussian local prediction only; training values descriptive',
                derivative_coefficients=coef[:2].tolist(), derivative_covariance=cov[:2, :2].tolist(),
                mode_rank=int(rank), structured_rms=float(np.sqrt(np.mean(structure**2))),
                residual_rms=float(np.sqrt(np.mean(residual**2))), source_bias=bias,
                source_measurement_sigma=information**-.5,
                interpretation='signed diagnostic coefficients; fixed-host projection; multiplicity/coverage unapproved')
