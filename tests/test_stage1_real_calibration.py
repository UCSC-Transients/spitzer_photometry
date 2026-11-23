import os
from pathlib import Path

import numpy as np
import pytest

from src import config, solver


@pytest.mark.real_calibration
def test_real_irac_prfs_are_available_finite_and_normalized(monkeypatch):
    """Manual-only real-calibration smoke; absence is an explicit unmet gate."""

    calibration_dir = Path(os.environ.get("SPITZER_PRF_DIR", config.PRF_DIR))
    available = list(calibration_dir.glob("*.fits")) if calibration_dir.is_dir() else []
    if not available:
        pytest.skip(f"real PRF calibration unavailable at {calibration_dir}")
    monkeypatch.setattr(config, "PRF_DIR", str(calibration_dir))
    monkeypatch.setattr(config, "USE_SYNTHETIC_PRF", False)
    for channel in ("ch1", "ch2"):
        for x_detector, y_detector in ((32.0, 32.0), (128.0, 128.0), (224.0, 224.0)):
            response = np.asarray(solver.load_prf(channel, x_detector, y_detector), dtype=float)
            assert response.ndim == 2
            assert np.isfinite(response).all()
            assert np.all(response >= 0.0)
            assert np.sum(response) == pytest.approx(1.0, rel=1e-12, abs=1e-12)
