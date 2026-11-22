"""Global σ transforms (quadrature floor, flux exponent) for χ² weights."""
import numpy as np
import pytest

from src import config, solver


def test_quadrature_floor_uniform_sigma(monkeypatch):
    monkeypatch.setattr(config, "SIGMA_QUADRATURE_FLOOR_JY", 3e-5)
    monkeypatch.setattr(config, "SIGMA_RELATIVE_FLUX_EXPONENT", 0.0)
    sig = np.full((4, 4), 1e-6, dtype=np.float64)
    entry = {"raw_wcs": None, "data": np.ones((4, 4))}
    out = solver.effective_native_sigma_for_fit(sig, entry)
    expect = float(np.sqrt((1e-6) ** 2 + (3e-5) ** 2))
    assert np.allclose(out, expect)


def test_flux_exponent_scales_bright_pixels(monkeypatch):
    monkeypatch.setattr(config, "SIGMA_QUADRATURE_FLOOR_JY", 0.0)
    monkeypatch.setattr(config, "SIGMA_RELATIVE_FLUX_EXPONENT", -1.0 / 6.0)
    # median(|d|)=1 → bright pixel 4 → σ × 4^(-1/6)
    d = np.array([[1.0, 1.0], [1.0, 4.0]], dtype=np.float64)
    sig = np.ones_like(d)
    entry = {"data": d, "raw_wcs": None}
    out = solver.effective_native_sigma_for_fit(sig, entry)
    assert out[1, 1] == pytest.approx(4.0 ** (-1.0 / 6.0))


def test_chain_order_floor_then_exponent(monkeypatch):
    monkeypatch.setattr(config, "SIGMA_QUADRATURE_FLOOR_JY", 0.01)
    monkeypatch.setattr(config, "SIGMA_RELATIVE_FLUX_EXPONENT", 1.0)
    sig = np.full((2, 2), 0.005, dtype=np.float64)
    d = np.ones((2, 2), dtype=np.float64)
    entry = {"data": d, "raw_wcs": None}
    out = solver.effective_native_sigma_for_fit(sig, entry)
    after_floor = float(np.sqrt(0.005**2 + 0.01**2))
    assert out[0, 0] == pytest.approx(after_floor * 1.0 ** 1.0)


@pytest.mark.parametrize("name", ("effective_native_sigma_for_fit", "native_sigma_with_nuclear_weighting"))
def test_aliases_match(name):
    fn = getattr(solver, name)
    sig = np.ones((2, 2), dtype=np.float64)
    entry = {}
    assert np.allclose(fn(sig, entry), solver.effective_native_sigma_for_fit(sig, entry))
