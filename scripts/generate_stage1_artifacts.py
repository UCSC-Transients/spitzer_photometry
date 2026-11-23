#!/usr/bin/env python3
"""Generate small, labeled Stage 1 fixture/metric artifacts for required CI."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.validation_fixtures import DEVELOPMENT_SEEDS, make_validation_fixture
from src.validation_metrics import (
    component_flux_metrics,
    interval_coverage_metrics,
    residual_metrics,
    write_artifact_bundle,
)


def generate(output_dir: str | Path) -> dict[str, object]:
    """Generate positive/zero fixtures and a compact versioned diagnostic bundle."""

    positive = make_validation_fixture(seed=DEVELOPMENT_SEEDS[0], transient_flux_jy=2.0e-3)
    zero = make_validation_fixture(seed=DEVELOPMENT_SEEDS[1], transient_flux_jy=0.0)
    usable = positive.valid_mask & ~positive.cosmic_ray_mask
    residual = positive.data - positive.truth_components["total"]
    residual = np.where(usable, residual, 0.0)
    train = usable.copy()
    train[1::2] = False
    heldout = usable.copy()
    heldout[0::2] = False
    leverage = np.where(train, 0.12, 0.0)
    predictive_variance = np.where(heldout, (0.35 * positive.uncertainty) ** 2, 0.0)
    finite_sigma = np.where(np.isfinite(positive.uncertainty), positive.uncertainty, 1.0)
    metrics = {
        "fixture": {
            "positive_checksum": positive.checksum(),
            "zero_checksum": zero.checksum(),
            "positive_metadata": dict(positive.metadata),
            "zero_metadata": dict(zero.metadata),
            "exposure_ids": [item.exposure_id for item in positive.exposures],
        },
        "flux_boundary_example": component_flux_metrics(
            {"positive_transient": 2.0e-3, "zero_transient": 0.0},
            {"positive_transient": 2.01e-3, "zero_transient": 0.0},
        ),
        "residuals": residual_metrics(
            residual,
            finite_sigma,
            training_mask=train,
            heldout_mask=heldout,
            leverage=leverage,
            predictive_variance=predictive_variance,
        ),
        "coverage_example": interval_coverage_metrics(
            np.asarray([2.0e-3, 2.0e-3, 0.0, 0.0]),
            np.asarray([1.8e-3, 2.1e-3, 0.0, 0.0]),
            np.asarray([2.2e-3, 2.4e-3, 1.0e-4, 2.0e-4]),
        ),
        "promotion_status": "unapproved: no user-declared absolute science-bias tolerance",
    }
    paths = write_artifact_bundle(
        output_dir,
        metrics=metrics,
        arrays={
            "positive_truth_total": positive.truth_components["total"],
            "positive_residual": residual,
            "positive_valid_mask": usable,
            "zero_truth_transient": zero.truth_components["transient"],
        },
        labels={
            "kind": "synthetic_independent_truth_fixture",
            "calibration": "not_real_prf_calibration",
            "plot_title": "Stage 1 synthetic truth (Jy/native pixel)",
        },
    )
    result = {"status": "passed", "artifacts": paths, "metrics": metrics}
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return result


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="test-results/stage1-validation")
    arguments = parser.parse_args(argv)
    generate(arguments.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
