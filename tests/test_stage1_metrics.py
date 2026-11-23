import json

import numpy as np
import pytest

from src.validation_metrics import (
    component_flux_metrics,
    component_leakage_metrics,
    interval_coverage_metrics,
    residual_metrics,
    solver_residual_metrics,
    write_artifact_bundle,
)


def test_flux_metrics_do_not_create_fractional_bias_at_zero_boundary():
    metrics = component_flux_metrics(
        {"source": 2.0, "zero": 0.0}, {"source": 2.2, "zero": 0.1}
    )
    assert metrics["source"]["fractional_bias"] == pytest.approx(0.1)
    assert metrics["zero"]["fractional_bias"] is None
    assert metrics["zero"]["at_nonnegative_boundary"] is False


def test_component_leakage_recovers_cross_transfer_matrix():
    injected = np.asarray(
        [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [2.0, 0.5]]
    )
    transfer = np.asarray([[1.0, 0.2], [0.1, 0.9]])
    recovered = injected @ transfer + np.asarray([0.03, -0.02])
    metrics = component_leakage_metrics(injected, recovered, ("host", "transient"))
    assert np.asarray(metrics["transfer_matrix"]) == pytest.approx(transfer)
    assert metrics["intercept_jy"] == pytest.approx([0.03, -0.02])
    assert metrics["maximum_absolute_off_diagonal"] == pytest.approx(0.2)


def test_training_leverage_and_heldout_predictive_variance_are_distinct():
    shape = (2, 4, 4)
    sigma = np.ones(shape)
    residual = np.zeros(shape)
    pattern = np.asarray([-1.0, 1.0] * 8).reshape(4, 4)
    residual[0] = np.sqrt(0.75) * pattern
    residual[1] = np.sqrt(2.0) * pattern
    train = np.zeros(shape, dtype=bool)
    heldout = np.zeros(shape, dtype=bool)
    train[0] = True
    heldout[1] = True
    leverage = np.where(train, 0.25, 0.0)
    predictive = np.where(heldout, 1.0, 0.0)
    metrics = residual_metrics(
        residual,
        sigma,
        training_mask=train,
        heldout_mask=heldout,
        leverage=leverage,
        predictive_variance=predictive,
    )
    assert metrics["training"]["raw_normalized_std"] == pytest.approx(np.sqrt(0.75))
    assert metrics["training"]["leverage_adjusted_std"] == pytest.approx(1.0)
    assert metrics["heldout"]["predictive_normalized_std"] == pytest.approx(1.0)
    assert metrics["whiteness"]["lag_x_correlation"] == pytest.approx(-1.0)


def test_zero_flux_coverage_is_not_reported_as_a_symmetric_pull():
    metrics = interval_coverage_metrics(
        np.asarray([0.0, 0.0, 1.0, 1.0]),
        np.asarray([0.0, 0.1, 0.8, 1.1]),
        np.asarray([0.2, 0.3, 1.2, 1.4]),
    )
    assert metrics["zero_truth_upper_limit_coverage"] == 1.0
    assert metrics["zero_truth_interval_coverage"] == 0.5
    assert metrics["positive_truth_coverage"] == 0.5
    assert "upper limits" in metrics["boundary_note"]


def test_solver_residual_metrics_use_relative_rhs_scale():
    metrics = solver_residual_metrics(
        np.asarray([3.0, 4.0]), np.asarray([6.0, 8.0]), projected_kkt_inf=1e-9
    )
    assert metrics["normal_residual_l2"] == 5.0
    assert metrics["normal_residual_relative_l2"] == 0.5
    assert metrics["projected_kkt_inf"] == 1e-9


def test_artifact_bundle_is_versioned_strict_and_checksum_manifested(tmp_path):
    paths = write_artifact_bundle(
        tmp_path,
        metrics={"bias_jy": 1e-6},
        arrays={"residual": np.arange(9, dtype=float).reshape(3, 3)},
        labels={"plot_title": "Synthetic residual", "calibration": "not_real"},
    )
    payload = json.loads((tmp_path / "metrics.json").read_text())
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert payload["schema_version"] == "stage1-metrics-v1"
    assert set(manifest["files"]) == {"metrics.json", "arrays.npz", "diagnostics.png"}
    assert all((tmp_path / name).stat().st_size > 0 for name in manifest["files"])
    assert set(paths) == {"metrics", "arrays", "plot", "manifest"}
