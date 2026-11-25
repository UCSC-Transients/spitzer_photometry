import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import numpy as np

from src.stage1_benchmark import (
    EngineeringBudget,
    compare_allocation_scaling,
    compare_numerical_arrays,
    compare_performance,
)


ROOT = Path(__file__).resolve().parents[1]
DRIVER_SPEC = importlib.util.spec_from_file_location(
    "stage1_benchmark_driver", ROOT / "scripts" / "run_stage1_benchmarks.py"
)
assert DRIVER_SPEC is not None and DRIVER_SPEC.loader is not None
DRIVER = importlib.util.module_from_spec(DRIVER_SPEC)
DRIVER_SPEC.loader.exec_module(DRIVER)


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


def test_new_confirmation_budget_crossing_remains_inconclusive():
    base = _record([1.0, 1.01, 0.99], [100, 100, 100])
    initial = _record([1.4, 1.41, 1.39], [100, 100, 100])
    confirmation_base = _record([1.0, 1.01, 0.99], [100, 100, 100])
    confirmation_head = _record([1.0, 1.01, 0.99], [130, 130, 130])
    result = compare_performance(
        base,
        initial,
        confirmation_base=confirmation_base,
        confirmation_head=confirmation_head,
    )
    assert result["status"] == "inconclusive"
    assert "new budget crossing" in result["reason"]


def _allocation_record(*, large: int = 950, ratio: float = 9.5):
    return {
        "small_scene_shape": [12, 12],
        "large_scene_shape": [24, 24],
        "small_estimated_peak_bytes": 100,
        "large_estimated_peak_bytes": large,
        "peak_growth_ratio": ratio,
        "small_parameter_count": 150,
        "large_parameter_count": 582,
    }


def test_controlled_allocation_scaling_violation_is_detected():
    passed = compare_allocation_scaling([_allocation_record()] * 3, [_allocation_record()] * 3)
    failed = compare_allocation_scaling(
        [_allocation_record()] * 3,
        [_allocation_record(large=95_000, ratio=950.0)] * 3,
    )
    assert passed["passed"] is True
    assert failed["passed"] is False
    assert failed["maximum_peak_growth_ratio"] > 12.0


def test_required_cli_returns_nonzero_for_inconclusive(monkeypatch, tmp_path):
    monkeypatch.setattr(DRIVER, "run_paired", lambda **_kwargs: {"status": "inconclusive"})
    code = DRIVER.main(
        [
            "--base-repo", str(tmp_path),
            "--head-repo", str(tmp_path),
            "--output-dir", str(tmp_path / "out"),
        ]
    )
    assert code != 0


def test_worker_records_complete_arrays_operations_reads_and_allocation_scaling(tmp_path):
    root = ROOT
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
    arrays_path = tmp_path / record["numerical_arrays"]["file"]
    with np.load(arrays_path, allow_pickle=False) as arrays:
        assert arrays["forward"].shape == (19, 19)
        assert arrays["adjoint"].shape == (23, 23)


def test_complete_outputs_detect_spatial_change_with_unchanged_fingerprint(tmp_path):
    def record(name: str, forward: np.ndarray):
        path = tmp_path / f"{name}.npz"
        np.savez_compressed(path, forward=forward, adjoint=np.arange(9.0).reshape(3, 3))
        import hashlib

        return {
            "numerical": [float(np.sum(forward)), float(np.linalg.norm(forward))],
            "numerical_arrays": {
                "file": path.name,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "keys": ["forward", "adjoint"],
            },
        }

    base = np.asarray([[1.0, 0.0], [0.0, 1.0]])
    changed = np.asarray([[0.0, 1.0], [1.0, 0.0]])
    base_record = record("base", base)
    head_record = record("head", changed)
    assert base_record["numerical"] == head_record["numerical"]
    result = DRIVER._numerical_check(
        {"base": [base_record], "head": [head_record]}, tmp_path
    )
    assert result["passed"] is False
    assert result["maximum_absolute_difference"] == 1.0


def test_confirmation_only_numerical_change_fails_driver(monkeypatch, tmp_path):
    import hashlib

    def sample(label: str, mode: str, side: str, index: int, value: float, wall: float):
        path = tmp_path / f"{label}-{mode}-{side}-{index}.npz"
        np.savez_compressed(path, forward=np.full((2, 2), value), adjoint=np.full((2, 2), 2.0))
        return {
            "wall_seconds": wall,
            "peak_memory_bytes": 100 * 1024**2,
            "numerical": [4.0 * value, 2.0 * abs(value), 8.0, 4.0, 0.0],
            "numerical_arrays": {
                "file": path.name,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "keys": ["forward", "adjoint"],
            },
            "allocation_scaling": _allocation_record(),
        }

    def collect(**kwargs):
        label = kwargs["label"]
        result = {}
        for mode in ("cold", "warm"):
            result[mode] = {"base": [], "head": []}
            for index in range(3):
                base_wall = 1.0
                head_wall = 1.4 if label == "initial" else 1.0
                result[mode]["base"].append(sample(label, mode, "base", index, 1.0, base_wall))
                head_value = 2.0 if label == "confirmation" else 1.0
                result[mode]["head"].append(sample(label, mode, "head", index, head_value, head_wall))
        return result

    monkeypatch.setattr(DRIVER, "_collect", collect)
    report = DRIVER.run_paired(
        base_repo=tmp_path,
        head_repo=tmp_path,
        output_dir=tmp_path,
        repeats=3,
        timeout_seconds=1.0,
    )
    assert report["status"] == "failed"
    assert any("confirmation numerical mismatch" in failure for failure in report["failures"])
