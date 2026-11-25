"""Versioned numerical metrics and artifact interface for staged validation."""
from __future__ import annotations

from collections.abc import Mapping
import hashlib
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


METRICS_SCHEMA_VERSION = "stage1-metrics-v1"


def _finite_float(value: float, name: str) -> float:
    result = float(value)
    if not np.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def component_flux_metrics(
    truth_jy: Mapping[str, float], estimate_jy: Mapping[str, float]
) -> dict[str, dict[str, float | None]]:
    """Return per-component flux bias without dividing by a zero truth value."""

    if set(truth_jy) != set(estimate_jy):
        raise ValueError("truth and estimate component names must match")
    metrics: dict[str, dict[str, float | None]] = {}
    for name in sorted(truth_jy):
        truth = _finite_float(truth_jy[name], f"truth[{name}]")
        estimate = _finite_float(estimate_jy[name], f"estimate[{name}]")
        bias = estimate - truth
        metrics[name] = {
            "truth_jy": truth,
            "estimate_jy": estimate,
            "bias_jy": bias,
            "fractional_bias": None if truth == 0.0 else bias / truth,
            "at_nonnegative_boundary": bool(truth == 0.0 and estimate <= 1e-15),
        }
    return metrics


def component_leakage_metrics(
    injected_jy: np.ndarray, recovered_jy: np.ndarray, component_names: tuple[str, ...]
) -> dict[str, Any]:
    """Estimate recovery slopes and cross-component leakage across injections.

    Rows are trials and columns are components.  The returned matrix maps injected
    components to recovered components; off-diagonal values are leakage slopes.
    An intercept is included so a zero-flux boundary offset is not mistaken for a
    transfer slope.
    """

    injected = np.asarray(injected_jy, dtype=np.float64)
    recovered = np.asarray(recovered_jy, dtype=np.float64)
    if injected.ndim != 2 or recovered.shape != injected.shape:
        raise ValueError("injected and recovered arrays must have the same 2-D shape")
    if injected.shape[1] != len(component_names):
        raise ValueError("component_names length must match the matrix width")
    if not np.isfinite(injected).all() or not np.isfinite(recovered).all():
        raise ValueError("leakage inputs must be finite")
    design = np.column_stack([injected, np.ones(injected.shape[0], dtype=np.float64)])
    coefficients, _, rank, singular_values = np.linalg.lstsq(design, recovered, rcond=None)
    transfer = coefficients[:-1, :]
    intercept = coefficients[-1, :]
    return {
        "component_names": list(component_names),
        "transfer_matrix": transfer.tolist(),
        "intercept_jy": intercept.tolist(),
        "design_rank": int(rank),
        "singular_values": singular_values.tolist(),
        "maximum_absolute_off_diagonal": float(
            np.max(np.abs(transfer - np.diag(np.diag(transfer))))
        ),
    }


def centroid_profile_metrics(
    *,
    truth_centroid_xy: tuple[float, float],
    estimate_centroid_xy: tuple[float, float],
    truth_profile: tuple[float, float, float],
    estimate_profile: tuple[float, float, float],
) -> dict[str, Any]:
    """Measure centroid and elliptical-profile bias in native-pixel units."""

    truth_centroid = np.asarray(truth_centroid_xy, dtype=np.float64)
    estimate_centroid = np.asarray(estimate_centroid_xy, dtype=np.float64)
    truth_shape = np.asarray(truth_profile, dtype=np.float64)
    estimate_shape = np.asarray(estimate_profile, dtype=np.float64)
    if not all(np.isfinite(value).all() for value in (truth_centroid, estimate_centroid, truth_shape, estimate_shape)):
        raise ValueError("centroid/profile inputs must be finite")
    centroid_bias = estimate_centroid - truth_centroid
    angle_bias = (estimate_shape[2] - truth_shape[2] + 90.0) % 180.0 - 90.0
    return {
        "centroid_bias_xy_pix": centroid_bias.tolist(),
        "centroid_bias_radius_pix": float(np.hypot(*centroid_bias)),
        "sigma_major_bias_pix": float(estimate_shape[0] - truth_shape[0]),
        "sigma_minor_bias_pix": float(estimate_shape[1] - truth_shape[1]),
        "position_angle_bias_deg": float(angle_bias),
    }


def _lag_correlation(values: np.ndarray, mask: np.ndarray, dy: int, dx: int) -> float | None:
    y0 = max(0, -dy)
    y1 = min(values.shape[-2], values.shape[-2] - dy)
    x0 = max(0, -dx)
    x1 = min(values.shape[-1], values.shape[-1] - dx)
    left = values[..., y0:y1, x0:x1]
    right = values[..., y0 + dy:y1 + dy, x0 + dx:x1 + dx]
    pairs = mask[..., y0:y1, x0:x1] & mask[..., y0 + dy:y1 + dy, x0 + dx:x1 + dx]
    if np.count_nonzero(pairs) < 3:
        return None
    a, b = left[pairs], right[pairs]
    a, b = a - np.mean(a), b - np.mean(b)
    denominator = float(np.sqrt(np.dot(a, a) * np.dot(b, b)))
    return None if denominator == 0.0 else float(np.dot(a, b) / denominator)


def residual_metrics(
    residual: np.ndarray,
    measurement_sigma: np.ndarray,
    *,
    training_mask: np.ndarray,
    heldout_mask: np.ndarray,
    leverage: np.ndarray | None = None,
    predictive_variance: np.ndarray | None = None,
) -> dict[str, Any]:
    """Measure training and held-out residual behavior with correct variances.

    Training residuals are reported both raw and leverage-adjusted; no assertion
    that raw training-residual variance equals one is implied.  Held-out residuals
    include predictive variance in addition to measurement variance.
    """

    residual_array = np.asarray(residual, dtype=np.float64)
    sigma = np.asarray(measurement_sigma, dtype=np.float64)
    train = np.asarray(training_mask, dtype=bool).copy()
    heldout = np.asarray(heldout_mask, dtype=bool).copy()
    if not (residual_array.shape == sigma.shape == train.shape == heldout.shape):
        raise ValueError("residual, sigma, and masks must have identical shapes")
    if np.any(train & heldout):
        raise ValueError("training and held-out masks must not overlap")
    valid = np.isfinite(residual_array) & np.isfinite(sigma) & (sigma > 0.0)
    train &= valid
    heldout &= valid
    if not np.any(train) or not np.any(heldout):
        raise ValueError("both training and held-out masks require valid pixels")

    raw_z = residual_array / sigma
    if leverage is None:
        leverage_array = np.zeros_like(residual_array)
    else:
        leverage_array = np.asarray(leverage, dtype=np.float64)
        if leverage_array.shape != residual_array.shape or np.any((leverage_array < 0.0) | (leverage_array >= 1.0)):
            raise ValueError("leverage must have the residual shape and lie in [0, 1)")
    adjusted_z = raw_z / np.sqrt(np.maximum(1.0 - leverage_array, 1e-12))

    if predictive_variance is None:
        predictive = np.zeros_like(residual_array)
    else:
        predictive = np.asarray(predictive_variance, dtype=np.float64)
        if predictive.shape != residual_array.shape or np.any(~np.isfinite(predictive)) or np.any(predictive < 0.0):
            raise ValueError("predictive variance must be finite, nonnegative, and shape-matched")
    heldout_z = residual_array / np.sqrt(sigma**2 + predictive)
    whiteness_mask = heldout & np.isfinite(heldout_z)
    return {
        "training": {
            "count": int(np.count_nonzero(train)),
            "raw_normalized_mean": float(np.mean(raw_z[train])),
            "raw_normalized_std": float(np.std(raw_z[train])),
            "leverage_adjusted_std": float(np.std(adjusted_z[train])),
            "interpretation": "raw training width is reduced by fitting and is not required to equal one",
        },
        "heldout": {
            "count": int(np.count_nonzero(heldout)),
            "predictive_normalized_mean": float(np.mean(heldout_z[heldout])),
            "predictive_normalized_std": float(np.std(heldout_z[heldout])),
            "rmse_data_units": float(np.sqrt(np.mean(residual_array[heldout] ** 2))),
        },
        "whiteness": {
            "lag_x_correlation": _lag_correlation(heldout_z, whiteness_mask, 0, 1),
            "lag_y_correlation": _lag_correlation(heldout_z, whiteness_mask, 1, 0),
        },
    }


def interval_coverage_metrics(
    truth_jy: np.ndarray, lower_jy: np.ndarray, upper_jy: np.ndarray
) -> dict[str, Any]:
    """Report interval coverage while treating zero-flux boundaries separately."""

    truth = np.asarray(truth_jy, dtype=np.float64)
    lower = np.asarray(lower_jy, dtype=np.float64)
    upper = np.asarray(upper_jy, dtype=np.float64)
    if truth.shape != lower.shape or truth.shape != upper.shape:
        raise ValueError("truth and interval arrays must have identical shapes")
    if not np.isfinite(truth).all() or not np.isfinite(lower).all() or not np.isfinite(upper).all():
        raise ValueError("coverage inputs must be finite")
    if np.any(truth < 0.0) or np.any(lower > upper):
        raise ValueError("truth must be nonnegative and intervals ordered")
    covered = (lower <= truth) & (truth <= upper)
    at_boundary = truth == 0.0
    positive = ~at_boundary
    return {
        "count": int(truth.size),
        "overall_coverage": float(np.mean(covered)),
        "positive_truth_count": int(np.count_nonzero(positive)),
        "positive_truth_coverage": None if not np.any(positive) else float(np.mean(covered[positive])),
        "zero_truth_count": int(np.count_nonzero(at_boundary)),
        "zero_truth_upper_limit_coverage": None if not np.any(at_boundary) else float(np.mean(upper[at_boundary] >= 0.0)),
        "zero_truth_interval_coverage": None if not np.any(at_boundary) else float(np.mean(covered[at_boundary])),
        "boundary_note": "zero-flux trials are assessed as constrained intervals/upper limits, not symmetric Gaussian pulls",
    }


def solver_residual_metrics(
    normal_residual: np.ndarray, rhs: np.ndarray, *, projected_kkt_inf: float
) -> dict[str, float]:
    """Summarize linear-system and constrained KKT residuals."""

    residual_array = np.asarray(normal_residual, dtype=np.float64)
    rhs_array = np.asarray(rhs, dtype=np.float64)
    if residual_array.shape != rhs_array.shape or not np.isfinite(residual_array).all() or not np.isfinite(rhs_array).all():
        raise ValueError("solver residual and rhs must be finite and shape-matched")
    absolute = float(np.linalg.norm(residual_array))
    relative = absolute / max(float(np.linalg.norm(rhs_array)), np.finfo(float).tiny)
    return {
        "normal_residual_l2": absolute,
        "normal_residual_relative_l2": relative,
        "projected_kkt_inf": _finite_float(projected_kkt_inf, "projected_kkt_inf"),
    }


def write_artifact_bundle(
    output_dir: str | Path,
    *,
    metrics: Mapping[str, Any],
    arrays: Mapping[str, np.ndarray],
    labels: Mapping[str, str],
) -> dict[str, str]:
    """Write strict JSON, compressed arrays, checksums, and a labeled diagnostic plot."""

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    metrics_path = destination / "metrics.json"
    arrays_path = destination / "arrays.npz"
    plot_path = destination / "diagnostics.png"
    payload = {"schema_version": METRICS_SCHEMA_VERSION, "labels": dict(labels), "metrics": dict(metrics)}
    metrics_path.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n")
    np.savez_compressed(arrays_path, **{name: np.asarray(value) for name, value in arrays.items()})

    plot_arrays = [np.asarray(value) for value in arrays.values() if np.asarray(value).ndim >= 2]
    if plot_arrays:
        first = plot_arrays[0]
        image = first[0] if first.ndim > 2 else first
        figure, axis = plt.subplots(figsize=(4.8, 4.0))
        artist = axis.imshow(image, origin="lower", cmap="coolwarm")
        axis.set(title=labels.get("plot_title", "Stage 1 validation diagnostic"), xlabel="native x pixel", ylabel="native y pixel")
        figure.colorbar(artist, ax=axis, shrink=0.8)
        figure.tight_layout()
        figure.savefig(plot_path, dpi=140)
        plt.close(figure)
    else:
        raise ValueError("at least one two-dimensional array is required for the diagnostic plot")

    checksums = {}
    for path in (metrics_path, arrays_path, plot_path):
        checksums[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    manifest_path = destination / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": METRICS_SCHEMA_VERSION,
                "files": checksums,
                "limitations": [
                    "synthetic artifacts do not certify native calibrated-PRF normalization",
                    "science-bias promotion tolerance remains unset",
                ],
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    return {"metrics": str(metrics_path), "arrays": str(arrays_path), "plot": str(plot_path), "manifest": str(manifest_path)}
