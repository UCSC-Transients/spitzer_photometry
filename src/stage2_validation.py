"""Stage 2 consumers of the independent Stage 1 native-injection interface.

Development only; no reserved acceptance seeds are used. These quadrature PRFs
are synthetic test instruments, never substitutes for real IRAC calibration.
"""
from __future__ import annotations

import numpy as np
from astropy.wcs import WCS, Sip
from numpy.polynomial.legendre import leggauss

from .isolated_astrometry import StarFrame
from .point_response import NativePRF
from .validation_fixtures import DEVELOPMENT_SEEDS, _integrated_axis_gaussian


def validation_wcs(roll: float = 0., phase: tuple[float, float] = (0., 0.), sr: int = 1, sip: bool = False) -> WCS:
    """Fixed-footprint TAN geometry; SR changes only reference pixel units."""
    w = WCS(naxis=2)
    w.wcs.crpix = (np.array([13., 13.])+phase-.5)*sr+.5
    w.wcs.crval = [45., 32.]
    w.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    angle = np.deg2rad(roll)
    w.wcs.cd = 1.22/3600/sr*np.array([[-np.cos(angle), np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    if sip:
        w.wcs.ctype = ["RA---TAN-SIP", "DEC--TAN-SIP"]
        a, b = np.zeros((3, 3)), np.zeros((3, 3))
        a[2, 0], b[0, 2] = 1e-4, -2e-4
        w.sip = Sip(a, b, None, None, w.wcs.crpix)
    return w


def synthetic_integrated_prf() -> NativePRF:
    """Pixel-integrate by quadrature; injections independently use analytic erf."""
    osamp = 50
    axis = np.arange(-350, 351)/osamp
    nodes, weights = leggauss(12)
    integrated = sum(weight/2*np.exp(-.5*((axis+node/2)/.78)**2)/(.78*np.sqrt(2*np.pi))
                     for node, weight in zip(nodes, weights))
    return NativePRF(np.outer(integrated, integrated), osamp, synthetic=True)


def isolated_star_fixture(seed: int | None = None, sr: int = 1, single: bool = False) -> tuple:
    """Return native frames, reference, initial/truth centroids, flux, and shifts."""
    if seed is not None and seed not in DEVELOPMENT_SEEDS:
        raise ValueError("Stage 2 routine recovery uses registered development seeds only")
    reference = validation_wcs(sr=sr)
    positions = np.array([[7.4, 8.2], [17.3, 9.1], [11.5, 17.2]])
    offsets = np.array([[.19, -.16], [-.22, .14], [.11, .23]])
    truth = (positions+offsets+.5)*sr-.5
    initial = (positions+.5)*sr-.5
    flux = np.array([.01, .006, .008])
    shifts = np.array([[0., 0.], [.27, -.19], [-.23, .31]])
    if single:
        truth, initial, flux, shifts = truth[:1], initial[:1], flux[:1], shifts[:1]
    table = synthetic_integrated_prf()
    rng = np.random.default_rng(seed)
    frames = []
    for i, shift in enumerate(shifts):
        native = validation_wcs(roll=[0, 27, 73][i], phase=(.13*i, -.21*i), sip=i > 0)
        data = np.full((25, 25), (i-1)*2e-5)
        for pos, amplitude in zip(truth, flux):
            ra, dec = reference.pixel_to_world_values(*pos)
            x, y = np.array(native.world_to_pixel_values(ra, dec))+shift
            data += _integrated_axis_gaussian(data.shape, x-12., y-12., .78, amplitude)
        sigma = np.full(data.shape, 1e-5)
        if seed is not None:
            data += rng.normal(size=data.shape)*sigma
        mask = np.ones(data.shape, dtype=bool)
        mask[3, 4] = False
        data[0, 0], sigma[0, 0] = np.nan, np.inf
        frames.append(StarFrame(data, sigma, native, table, mask))
    return tuple(frames), reference, initial, truth, flux, shifts
