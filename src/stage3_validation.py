"""Independent synthetic truth for template nucleus discrimination.

Truth integrates analytic bivariate Gaussian *optics* over detector pixels with
Legendre quadrature. Fits instead integrate source positions against tabulated,
detector-integrated PRFs. Synthetic response is never called IRAC calibration.
"""
from __future__ import annotations

from dataclasses import replace
import numpy as np
from numpy.polynomial.legendre import leggauss

from .isolated_astrometry import StarFrame
from .nucleus import NucleusScene, TemplateFrame
from .stage2_validation import validation_wcs, synthetic_integrated_prf
from .validation_fixtures import DEVELOPMENT_SEEDS

TRUTH_CASES = ('centroid', 'extent', 'prf_mismatch', 'asymmetric')


def integrated_gaussian(shape: tuple[int, int], center: np.ndarray, covariance: np.ndarray,
                        flux: float, *, order: int = 12) -> np.ndarray:
    """Independent pixel-area integral of a normalized analytic optical Gaussian."""
    nodes, weights = leggauss(order)
    yy, xx = np.indices(shape, dtype=float)
    inverse = np.linalg.inv(covariance)
    norm = flux/(2*np.pi*np.sqrt(np.linalg.det(covariance)))
    result = np.zeros(shape)
    for x, wx in zip(nodes/2, weights/2):
        for y, wy in zip(nodes/2, weights/2):
            dx, dy = xx+x-center[0], yy+y-center[1]
            result += wx*wy*norm*np.exp(-.5*(inverse[0, 0]*dx*dx+2*inverse[0, 1]*dx*dy+inverse[1, 1]*dy*dy))
    return result


def nucleus_fixture(case: str = 'centroid', seed: int | None = None, *, sr: int = 1,
                    geometry: str = 'rolls') -> tuple[tuple[TemplateFrame, ...], NucleusScene, dict]:
    """Development frames with independent truth, phases and declared partitions.

    ``single`` exposes one training BCD; ``dithers`` six same-roll phases;
    ``rolls`` six phases/rolls. Same sky footprint is used for SR=1/2/4. Noise is
    known Gaussian in flux units, not a sqrt(Jy) proxy. Calibration of backgrounds
    and registration is exact for noiseless algebra, independent with known
    uncertainty for noisy trials. This does not establish real BCD calibration.
    """
    if case not in TRUTH_CASES or (seed is not None and seed not in DEVELOPMENT_SEEDS):
        raise ValueError('Use declared truth cases and development seeds')
    if geometry not in ('single', 'dithers', 'rolls') or sr not in (1, 2, 4):
        raise ValueError('Unsupported geometry or SR')
    reference = validation_wcs(sr=sr)
    to_ref = lambda a: (np.asarray(a)+.5)*sr-.5
    center = np.array([12.18, 11.86])
    profile = np.array([.44, .26, .32]) if case == 'extent' else np.zeros(3)
    stars = np.array([[6.4, 8.2], [18.1, 16.4]])
    host_center = np.array([11.4, 12.7])
    host_profile = np.array([3.1, 2.6, .18])
    scene = NucleusScene(reference, to_ref([12., 12.]), to_ref(stars), to_ref(host_center),
                         host_profile*np.array([sr, sr, 1.]), centroid_radius=.8*sr,
                         maximum_width=1.2*sr, quadrature_order=11, host_order=65)
    truth_position, truth_profile = to_ref(center), profile*np.array([sr, sr, 1.])
    response = synthetic_integrated_prf()
    rng = np.random.default_rng(seed)
    frames = []
    rolls, phases = [], []
    for i in range(1 if geometry == 'single' else 6):
        roll = [0., 0., 31., 51., 79., 109.][i] if geometry == 'rolls' else 0.
        phase = (.13*i, -.17*i)
        native = validation_wcs(roll, phase)
        shift = np.array([.12*np.sin(i), -.14*np.sin(i)])
        offset = np.zeros(2) if seed is None or i == 0 else rng.normal(0, .003, 2)
        prior_shift = shift+offset
        def projection(pos: np.ndarray) -> np.ndarray:
            return np.array(native.world_to_pixel_values(*reference.pixel_to_world_values(*pos)))+shift
        step = 1e-3
        jac = np.column_stack([(projection(truth_position+d)-projection(truth_position-d))/(2*step)
                               for d in np.eye(2)*step])
        optical = np.diag(np.array([.86, .72])**2) if case == 'prf_mismatch' else np.eye(2)*.78**2
        def source(pos: np.ndarray, shape: np.ndarray, flux: float) -> np.ndarray:
            sx, sy, rho = shape
            intrinsic = np.array([[sx*sx, sx*sy*rho], [sx*sy*rho, sy*sy]])
            return integrated_gaussian((25, 25), projection(pos), optical+jac@intrinsic@jac.T, flux)
        background = (i-2)*2e-6
        image = np.full((25, 25), background)
        if case == 'asymmetric':
            image += source(truth_position, np.zeros(3), .0075)
            image += source(truth_position+np.array([.85, .3])*sr, np.zeros(3), .0025)
        else:
            image += source(truth_position, truth_profile, .01)
        for pos, flux in zip(scene.star_positions, [.004, .006]):
            image += source(pos, np.zeros(3), flux)
        image += source(scene.host_center_xy, scene.host_profile, .02)
        sigma = np.full(image.shape, 1e-5)
        if seed is not None:
            image += rng.normal(size=image.shape)*sigma
        mask = np.ones(image.shape, dtype=bool)
        mask[3, 4] = False
        f = StarFrame(image, sigma, native, response, mask)
        frames.append(TemplateFrame(f, f'development-{geometry}-{i}', ['train','train','train','validation','heldout','heldout'][i],
                                    True, prior_shift, np.zeros((2, 2)) if seed is None or i == 0 else np.eye(2)*.003**2,
                                    background+(rng.normal(0, 1e-7) if seed is not None else 0.), 1e-7 if seed is not None else 0.,
                                    'independent analytic synthetic truth; no real calibration; known noise/registration'))
        rolls.append(roll)
        phases.append(phase)
    truth = dict(case=case, seed=seed, geometry=geometry, sr=sr, centroid=truth_position+(np.array([.85,.3])*sr*.25 if case=='asymmetric' else 0),
                 profile=truth_profile, nuclear_flux=.01, rolls=np.array(rolls), phases=np.array(phases),
                 probe_xy=to_ref(center+[1.8, -.6]))
    return tuple(frames), scene, truth
