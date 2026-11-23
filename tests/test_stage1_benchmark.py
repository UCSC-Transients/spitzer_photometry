import numpy as np
import json
from pathlib import Path
import subprocess
import sys

from src.stage1_benchmark import (
    EngineeringBudget,
    compare_numerical_arrays,
    compare_performance,
)


def _record(times, memory_mib):
    return {
        "wall_seconds": list(times),
        "peak_memory_bytes": [value * 1024 * 1024 for value in memory_mib],
    }


def test_deliberate_numerical_regression_is_detected():
    result = compare_numerical_arrays(
        np.asarray([1.0, 2.0, 3.0]), np.asarray([1.0, 2.0, 3.001])
    )
    assert result["passed"] is False
    assert result["maximum_absolute_difference"] > 9e-4


def test_controlled_performance_regression_requires_and_passes_confirmation():
    base = _record([1.00, 1.01, 0.99], [100, 101, 100])
    slow = _record([1.40, 1.41, 1.39], [130, 131, 130])
    first = compare_performance(base, slow)
    assert first["status"] == "confirmation_required"
    confirmed = compare_performance(
        base,
        slow,
        confirmation_base=base,
        confirmation_head=slow,
    )
    assert confirmed["status"] == "failed"
    assert confirmed["time_suspect"] is True
    assert confirmed["memory_suspect"] is True


def test_subthreshold_relative_or_absolute_changes_do_not_fail():
    base = _record([0.10, 0.101, 0.099], [100, 100, 100])
    head = _record([0.20, 0.201, 0.199], [115, 115, 115])
    result = compare_performance(base, head)
    assert result["status"] == "passed"
    assert result["time_suspect"] is False  # absolute increase is below 0.25 s
    assert result["memory_suspect"] is False  # both memory clauses must cross


def test_noisy_result_is_inconclusive_not_a_pass():
    base = _record([1.0, 1.5, 0.5], [100, 100, 100])
    head = _record([1.0, 1.6, 0.4], [100, 100, 100])
    result = compare_performance(base, head, budget=EngineeringBudget(maximum_relative_mad=0.1))
    assert result["status"] == "inconclusive"
    assert "dispersion" in result["reason"]


def test_noisy_suspected_regression_still_requires_confirmation_then_is_inconclusive():
    base = _record([1.0, 1.5, 0.5], [100, 100, 100])
    head = _record([2.0, 3.0, 1.0], [130, 160, 100])
    first = compare_performance(base, head)
    assert first["status"] == "confirmation_required"
    confirmed = compare_performance(
        base,
        head,
        confirmation_base=base,
        confirmation_head=head,
    )
    assert confirmed["status"] == "inconclusive"
    assert "too noisy" in confirmed["reason"]


def test_worker_records_matrix_operations_reads_and_allocation_scaling(tmp_path):
    root = Path(__file__).resolve().parents[1]
    output = tmp_path / "worker.json"
    completed = subprocess.run(
        [
            sys.executable,
            str(root / "scripts" / "stage1_benchmark_worker.py"),
            "--repo",
            str(root),
            "--mode",
            "cold",
            "--iterations",
            "1",
            "--output",
            str(output),
        ],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    record = json.loads(output.read_text())
    expected_head = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    assert record["repo_head"] == expected_head
    assert record["workload"]["operator_calls"] == 2
    assert record["workload"]["calibration_file_reads"] == 0
    assert record["workload"]["scene_parameters"] == 23 * 23
    assert record["workload"]["native_measurements"] == 19 * 19
    assert record["allocation_scaling"]["large_estimated_peak_bytes"] > record["allocation_scaling"]["small_estimated_peak_bytes"]
    assert record["allocation_scaling"]["peak_growth_ratio"] > 1.0
