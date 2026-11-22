"""Regression: celestial N/E derivatives must match the WCS tied to the plotted pixel grid."""
import numpy as np
from astropy.wcs import WCS

from src.native_fit_campaign import _compass_vectors


def test_compass_north_points_toward_increasing_dec():
    w = WCS(naxis=2)
    w.wcs.crval = [180.0, 0.0]
    w.wcs.crpix = [50.0, 50.0]
    w.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    w.wcs.cdelt = [-1.0 / 3600.0, 1.0 / 3600.0]
    w.wcs.pc = np.eye(2)
    vn, ve = _compass_vectors(w, 180.0, 0.0)
    assert vn[1] > 0, "North increment should move toward +pixel_y for this TAN frame"
    assert np.hypot(*vn) > 1e-6


def test_compass_rotated_cd_matrix():
    """45° rotation mixes x/y; N still orthogonal-ish to E in pixel plane."""
    w = WCS(naxis=2)
    w.wcs.crval = [180.0, 10.0]
    w.wcs.crpix = [32.0, 32.0]
    w.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    th = np.deg2rad(40.0)
    scale = 1.2 / 3600.0
    w.wcs.pc = [[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]]
    w.wcs.cdelt = [-scale, scale]
    vn, ve = _compass_vectors(w, 180.0, 10.0)
    dot = vn[0] * ve[0] + vn[1] * ve[1]
    nrm = np.hypot(*vn) * np.hypot(*ve)
    assert abs(dot / nrm) < 0.05, "N and E sky steps should be nearly perpendicular in pixels"

