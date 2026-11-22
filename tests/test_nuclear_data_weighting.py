"""Optional nuclear-disk σ scaling for native χ² weights (FR-2)."""
import numpy as np
import pytest
from astropy.wcs import WCS

from src import config, solver


def _tiny_cutout_wcs(n_pix: int = 16):
    w = WCS(naxis=2)
    w.wcs.crpix = [n_pix / 2, n_pix / 2]
    w.wcs.crval = [float(config.TRANSIENT_RA), float(config.TRANSIENT_DEC)]
    w.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    sc = float(config.PIXEL_SCALE) / 3600.0
    w.wcs.cdelt = [-sc, sc]
    w.wcs.pc = np.eye(2)
    return w


def test_native_sigma_weighting_disabled_by_default():
    sig = np.ones((8, 8), dtype=np.float64)
    entry = {"raw_wcs": _tiny_cutout_wcs(8), "wcs": None}
    out = solver.native_sigma_with_nuclear_weighting(sig, entry)
    assert np.allclose(out, sig)


def test_native_sigma_weighting_scales_core(monkeypatch):
    # Radius small enough that (0,0) lies outside the disk for n=16 (center ~(7,7) px).
    monkeypatch.setattr(config, "NUCLEAR_DATA_WEIGHT_RADIUS_NATIVE_PX", 5.0)
    monkeypatch.setattr(config, "NUCLEAR_DATA_WEIGHT_SIGMA_SCALE", 2.0)
    monkeypatch.setattr(config, "NUCLEAR_DATA_WEIGHT_CENTER", "transient")
    n = 16
    sig = np.ones((n, n), dtype=np.float64)
    entry = {"raw_wcs": _tiny_cutout_wcs(n), "filename": "x_ch2_cbcd.fits"}
    out = solver.native_sigma_with_nuclear_weighting(sig, entry)
    cy, cx = n // 2, n // 2
    assert out[cy, cx] == pytest.approx(2.0)
    assert out[0, 0] == pytest.approx(1.0)


@pytest.mark.slow
def test_fit_metrics_uses_same_sigma_scaling(monkeypatch, tiny_cutouts):
    """Reduced χ² metrics should use the same effective σ as the solver when enabled."""
    monkeypatch.setattr(config, "NUCLEAR_DATA_WEIGHT_RADIUS_NATIVE_PX", 50.0)
    monkeypatch.setattr(config, "NUCLEAR_DATA_WEIGHT_SIGMA_SCALE", 2.0)
    cutouts, _ = tiny_cutouts
    results = solver.run_gls_solve(
        cutouts,
        [],
        [],
        {"ell": 3.0, "var": 1e-8},
        (3.0, 1e-8),
        np.zeros(cutouts[0]["data"].shape, dtype=np.float64),
        cutouts[0]["wcs"],
        1,
    )
    assert results is not None
    from src.fit_metrics import compute_fit_metrics

    m = compute_fit_metrics(
        cutouts,
        results,
        [],
        np.zeros(0),
        center_ra_deg=float(config.TRANSIENT_RA),
        center_dec_deg=float(config.TRANSIENT_DEC),
        center_radius_px=3.0,
    )
    assert np.isfinite(m["center_reduced_chi2"])
