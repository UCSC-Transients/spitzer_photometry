#!/usr/bin/env python3
"""Run alternating isolated base/head Stage 1 benchmarks and enforce budgets."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.stage1_benchmark import (
    BENCHMARK_SCHEMA_VERSION,
    EngineeringBudget,
    compare_numerical_arrays,
    compare_performance,
)


def _run_worker(
    *,
    worker: Path,
    repo: Path,
    mode: str,
    output: Path,
    timeout_seconds: float,
) -> dict[str, Any]:
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f"runtime-{output.stem}-", dir=output.parent) as runtime_dir:
        environment = os.environ.copy()
        environment.update(
            OPENBLAS_NUM_THREADS="1",
            OMP_NUM_THREADS="1",
            MKL_NUM_THREADS="1",
            NUMEXPR_NUM_THREADS="1",
            PYTHONHASHSEED="0",
            MPLBACKEND="Agg",
            MPLCONFIGDIR=str(Path(runtime_dir) / "mpl-cache"),
            PYTHONPYCACHEPREFIX=str(Path(runtime_dir) / "pycache"),
        )
        command = [
            sys.executable,
            str(worker),
            "--repo",
            str(repo),
            "--mode",
            mode,
            "--output",
            str(output),
        ]
        completed = subprocess.run(
            command,
            cwd=repo,
            env=environment,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            check=False,
        )
    if completed.returncode != 0:
        raise RuntimeError(
            f"benchmark worker failed for {repo} ({mode}) with exit {completed.returncode}: "
            f"{completed.stderr[-2000:]}"
        )
    return json.loads(output.read_text())


def _collect(
    *,
    base_repo: Path,
    head_repo: Path,
    worker: Path,
    output_dir: Path,
    repeats: int,
    timeout_seconds: float,
    label: str,
) -> dict[str, dict[str, list[dict[str, Any]]]]:
    records: dict[str, dict[str, list[dict[str, Any]]]] = {
        mode: {"base": [], "head": []} for mode in ("cold", "warm")
    }
    for mode in ("cold", "warm"):
        for repeat in range(repeats):
            order = ("base", "head") if repeat % 2 == 0 else ("head", "base")
            for side in order:
                repo = base_repo if side == "base" else head_repo
                path = output_dir / f"{label}-{mode}-{side}-{repeat}.json"
                records[mode][side].append(
                    _run_worker(
                        worker=worker,
                        repo=repo,
                        mode=mode,
                        output=path,
                        timeout_seconds=timeout_seconds,
                    )
                )
    return records


def _performance_input(records: list[dict[str, Any]]) -> dict[str, list[float]]:
    return {
        "wall_seconds": [float(record["wall_seconds"]) for record in records],
        "peak_memory_bytes": [float(record["peak_memory_bytes"]) for record in records],
    }


def _numerical_check(records: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    reference = np.asarray(records["base"][0]["numerical"], dtype=np.float64)
    comparisons = [
        compare_numerical_arrays(reference, np.asarray(record["numerical"], dtype=np.float64))
        for side in ("base", "head")
        for record in records[side]
    ]
    return {
        "passed": all(bool(item["passed"]) for item in comparisons),
        "maximum_absolute_difference": max(float(item["maximum_absolute_difference"]) for item in comparisons),
        "maximum_relative_difference": max(float(item["maximum_relative_difference"]) for item in comparisons),
        "comparisons": len(comparisons),
    }


def _write_manifest(output_dir: Path) -> None:
    """Checksum every persisted benchmark artifact without self-hashing."""

    files = {}
    for path in sorted(output_dir.iterdir()):
        if path.is_file() and path.name != "manifest.json":
            files[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    (output_dir / "manifest.json").write_text(
        json.dumps(
            {
                "schema_version": BENCHMARK_SCHEMA_VERSION,
                "files": files,
                "artifact_scope": "synthetic paired numerical/performance evidence",
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )


def _write_plot_and_arrays(
    records: dict[str, dict[str, list[dict[str, Any]]]], output_dir: Path
) -> None:
    """Persist raw numerical samples and a paired diagnostic plot."""

    arrays: dict[str, np.ndarray] = {}
    figure, axes = plt.subplots(1, 2, figsize=(9.0, 3.8))
    for mode_index, mode in enumerate(("cold", "warm")):
        for side_index, side in enumerate(("base", "head")):
            samples = records[mode][side]
            times = np.asarray([record["wall_seconds"] for record in samples], dtype=np.float64)
            memory = np.asarray([record["peak_memory_bytes"] for record in samples], dtype=np.float64)
            numerical = np.asarray([record["numerical"] for record in samples], dtype=np.float64)
            arrays[f"{mode}_{side}_wall_seconds"] = times
            arrays[f"{mode}_{side}_peak_memory_bytes"] = memory
            arrays[f"{mode}_{side}_numerical"] = numerical
            x_values = np.arange(len(times), dtype=float) + 0.08 * side_index
            axes[0].plot(x_values, times, "o-", label=f"{mode} {side}")
            axes[1].plot(x_values, memory / (1024**2), "o-", label=f"{mode} {side}")
    axes[0].set(xlabel="alternating subprocess repeat", ylabel="wall time (s)", title="Paired timing")
    axes[1].set(xlabel="alternating subprocess repeat", ylabel="process peak RSS (MiB)", title="Paired memory")
    for axis in axes:
        axis.legend(fontsize=7)
        axis.grid(alpha=0.25)
    figure.suptitle("Stage 1 synthetic stable-API benchmark (one BLAS thread)")
    figure.tight_layout()
    figure.savefig(output_dir / "performance.png", dpi=150)
    plt.close(figure)
    np.savez_compressed(output_dir / "benchmark_arrays.npz", **arrays)


def run_paired(
    *,
    base_repo: Path,
    head_repo: Path,
    output_dir: Path,
    repeats: int,
    timeout_seconds: float,
) -> dict[str, Any]:
    """Run base/head subprocess pairs, confirming any suspected regression."""

    if repeats < 3:
        raise ValueError("at least three repeats are required")
    worker = Path(__file__).with_name("stage1_benchmark_worker.py").resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    initial = _collect(
        base_repo=base_repo,
        head_repo=head_repo,
        worker=worker,
        output_dir=output_dir,
        repeats=repeats,
        timeout_seconds=timeout_seconds,
        label="initial",
    )
    _write_plot_and_arrays(initial, output_dir)
    report: dict[str, Any] = {
        "schema_version": BENCHMARK_SCHEMA_VERSION,
        "base_repo": str(base_repo),
        "head_repo": str(head_repo),
        "repeats": repeats,
        "timeout_seconds": timeout_seconds,
        "thread_environment": {
            "OPENBLAS_NUM_THREADS": "1",
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "NUMEXPR_NUM_THREADS": "1",
        },
        "cases": {},
    }
    needs_confirmation = False
    for mode in ("cold", "warm"):
        numerical = _numerical_check(initial[mode])
        performance = compare_performance(
            _performance_input(initial[mode]["base"]),
            _performance_input(initial[mode]["head"]),
        )
        report["cases"][mode] = {"numerical": numerical, "performance": performance}
        needs_confirmation |= performance["status"] == "confirmation_required"

    confirmation = None
    if needs_confirmation:
        confirmation = _collect(
            base_repo=base_repo,
            head_repo=head_repo,
            worker=worker,
            output_dir=output_dir,
            repeats=repeats,
            timeout_seconds=timeout_seconds,
            label="confirmation",
        )
        for mode in ("cold", "warm"):
            report["cases"][mode]["performance"] = compare_performance(
                _performance_input(initial[mode]["base"]),
                _performance_input(initial[mode]["head"]),
                confirmation_base=_performance_input(confirmation[mode]["base"]),
                confirmation_head=_performance_input(confirmation[mode]["head"]),
            )

    failures = []
    inconclusive = []
    for mode, case in report["cases"].items():
        if not case["numerical"]["passed"]:
            failures.append(f"{mode}: numerical mismatch")
        status = case["performance"]["status"]
        if status == "failed":
            failures.append(f"{mode}: confirmed performance regression")
        elif status in {"inconclusive", "confirmation_required"}:
            inconclusive.append(f"{mode}: {status}")
    report["status"] = "failed" if failures else ("inconclusive" if inconclusive else "passed")
    report["failures"] = failures
    report["inconclusive"] = inconclusive
    report["limitations"] = [
        "synthetic stable-API workload does not certify native calibrated-PRF science",
        "hardware-absolute timing is reported but never gated",
        "science-bias promotion tolerance remains unset",
    ]
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n")
    _write_manifest(output_dir)
    return report


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-repo", required=True, type=Path)
    parser.add_argument("--head-repo", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--timeout", type=float, default=45.0)
    arguments = parser.parse_args(argv)
    try:
        report = run_paired(
            base_repo=arguments.base_repo.resolve(),
            head_repo=arguments.head_repo.resolve(),
            output_dir=arguments.output_dir.resolve(),
            repeats=arguments.repeats,
            timeout_seconds=arguments.timeout,
        )
    except (RuntimeError, ValueError, subprocess.TimeoutExpired) as error:
        arguments.output_dir.mkdir(parents=True, exist_ok=True)
        failure = {
            "schema_version": BENCHMARK_SCHEMA_VERSION,
            "status": "failed",
            "error": str(error),
        }
        (arguments.output_dir / "summary.json").write_text(
            json.dumps(failure, indent=2, sort_keys=True) + "\n"
        )
        _write_manifest(arguments.output_dir)
        print(json.dumps(failure, indent=2, sort_keys=True))
        return 2
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))
    return 0 if report["status"] in {"passed", "inconclusive"} else 3


if __name__ == "__main__":
    raise SystemExit(main())
