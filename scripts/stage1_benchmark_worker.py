#!/usr/bin/env python3
"""Run one isolated cold- or warm-cache stable numerical benchmark sample."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import resource
import sys
import time
import tracemalloc

import numpy as np
from astropy.wcs import WCS


def _peak_rss_bytes() -> int:
    """Normalize ``ru_maxrss`` to bytes on Linux and macOS."""

    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return value if sys.platform == "darwin" else value * 1024


def _wcs(n_pix: int, roll_deg: float, pixel_scale_arcsec: float) -> WCS:
    result = WCS(naxis=2)
    result.wcs.crpix = [n_pix / 2.0 + 0.17, n_pix / 2.0 - 0.23]
    result.wcs.crval = [197.450375, -23.381486]
    result.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    result.wcs.cdelt = [-pixel_scale_arcsec / 3600.0, pixel_scale_arcsec / 3600.0]
    theta = np.deg2rad(roll_deg)
    result.wcs.pc = np.asarray([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    return result


def run_sample(
    repo: Path,
    mode: str,
    iterations: int,
    seed: int,
    arrays_output: Path,
) -> dict[str, object]:
    """Benchmark the pre-Stage-1 frame operator shared by base and head trees."""

    sys.path.insert(0, str(repo))
    from src import config, solver  # pylint: disable=import-outside-toplevel

    required = ("_apply_frame_forward_operator", "_apply_frame_adjoint_operator")
    if not all(hasattr(solver, name) for name in required):
        raise RuntimeError("target lacks the stable Stage 0 frame-operator benchmark API")
    scene_shape = (23, 23)
    native_shape = (19, 19)
    scene_wcs = _wcs(scene_shape[0], 0.0, 0.61)
    native_wcs = _wcs(native_shape[0], 23.0, 1.22)
    rng = np.random.default_rng(seed)
    scene = rng.normal(0.0, 1.0, scene_shape)
    probe = rng.normal(0.0, 1.0, native_shape)
    yy, xx = np.mgrid[-80:81, -80:81]
    synthetic_prf = np.exp(-0.5 * ((xx / 18.0) ** 2 + (yy / 20.0) ** 2))
    synthetic_prf /= np.sum(synthetic_prf)
    calibration_reads = 0

    def load_synthetic_prf(*_args: object, **_kwargs: object) -> np.ndarray:
        nonlocal calibration_reads
        calibration_reads += 1
        return synthetic_prf

    solver.load_prf = load_synthetic_prf
    config.PRF_SPATIAL_ANCHORS_PER_AXIS = 1
    config.PRF_OPERATOR_MODE = "anchor"
    config.PRF_ORDER_PROJECT_THEN_CONVOLVE = True
    for name in (
        "_GEOMETRY_CACHE",
        "_PRF_OPERATOR_BUNDLE_CACHE",
        "_PRF_NATIVE_OPERATOR_BUNDLE_CACHE",
        "_PRF_EXACT_OPERATOR_CACHE",
        "_WCS_PROVENANCE_CACHE",
        "_PROJECTION_CACHE",
    ):
        cache = getattr(solver, name, None)
        if cache is not None:
            cache.clear()

    def evaluate() -> tuple[np.ndarray, np.ndarray, float]:
        forward = solver._apply_frame_forward_operator(
            scene,
            scene_wcs,
            native_wcs,
            scene_shape,
            native_shape,
            "ch2",
            is_full_array=True,
        )
        adjoint = solver._apply_frame_adjoint_operator(
            probe,
            scene_wcs,
            native_wcs,
            scene_shape,
            "ch2",
            is_full_array=True,
        )
        lhs = float(np.vdot(forward, probe))
        rhs = float(np.vdot(scene, adjoint))
        adjoint_relative = abs(lhs - rhs) / max(abs(lhs), abs(rhs), np.finfo(float).tiny)
        return forward, adjoint, adjoint_relative

    warmup_calls = 0
    if mode == "warm":
        evaluate()
        warmup_calls = 2
    measured_iterations = 1 if mode == "cold" else iterations
    rss_before = _peak_rss_bytes()
    tracemalloc.start()
    start = time.perf_counter()
    forward = adjoint = np.empty(0)
    adjoint_relative = float("nan")
    for _ in range(measured_iterations):
        forward, adjoint, adjoint_relative = evaluate()
    wall_seconds = time.perf_counter() - start
    _, traced_peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    rss_peak = _peak_rss_bytes()
    numerical = np.asarray(
        [
            np.sum(forward),
            np.linalg.norm(forward),
            np.sum(adjoint),
            np.linalg.norm(adjoint),
            adjoint_relative,
        ],
        dtype=np.float64,
    )
    numerical_hash = hashlib.sha256(numerical.tobytes()).hexdigest()
    arrays_output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        arrays_output,
        forward=np.asarray(forward, dtype=np.float64),
        adjoint=np.asarray(adjoint, dtype=np.float64),
    )
    arrays_hash = hashlib.sha256(arrays_output.read_bytes()).hexdigest()
    input_digest = hashlib.sha256()
    for value in (scene, probe, synthetic_prf):
        input_digest.update(np.ascontiguousarray(value).tobytes())
    input_digest.update(scene_wcs.to_header_string().encode("utf-8"))
    input_digest.update(native_wcs.to_header_string().encode("utf-8"))
    allocation_frames = [
        {"data": np.zeros(native_shape, dtype=np.float64)} for _ in range(3)
    ]
    allocation_small = solver._estimate_solver_peak_memory_bytes(
        allocation_frames,
        (12, 12),
        n_scene_blocks=1,
        n_stars=2,
        n_science_epochs=2,
    )
    allocation_large = solver._estimate_solver_peak_memory_bytes(
        allocation_frames,
        (24, 24),
        n_scene_blocks=1,
        n_stars=2,
        n_science_epochs=2,
    )
    return {
        "schema_version": "stage1-benchmark-v2",
        "mode": mode,
        "repo_head": _read_head(repo),
        "runner": {
            "python": sys.version.split()[0],
            "platform": sys.platform,
            "blas_threads": 1,
            "seed": seed,
            "iterations": measured_iterations,
        },
        "workload": {
            "kind": "synthetic_stable_stage0_frame_operator",
            "calibration_status": "in_memory_analytic_prf_not_real_calibration",
            "scene_shape": list(scene_shape),
            "native_shape": list(native_shape),
            "scene_parameters": int(np.prod(scene_shape)),
            "native_measurements": int(np.prod(native_shape)),
            "operator_calls": 2 * measured_iterations,
            "warmup_operator_calls": warmup_calls,
            "calibration_function_calls": calibration_reads,
            "calibration_file_reads": 0,
            "input_sha256": input_digest.hexdigest(),
        },
        "allocation_scaling": {
            "small_scene_shape": [12, 12],
            "large_scene_shape": [24, 24],
            "small_estimated_peak_bytes": allocation_small["peak_bytes"],
            "large_estimated_peak_bytes": allocation_large["peak_bytes"],
            "peak_growth_ratio": allocation_large["peak_bytes"] / allocation_small["peak_bytes"],
            "small_parameter_count": allocation_small["n_params"],
            "large_parameter_count": allocation_large["n_params"],
        },
        "wall_seconds": wall_seconds,
        "peak_memory_bytes": max(rss_before, rss_peak),
        "peak_memory_growth_bytes": max(0, rss_peak - rss_before),
        "tracemalloc_peak_bytes": traced_peak,
        "numerical": numerical.tolist(),
        "numerical_sha256": numerical_hash,
        "numerical_arrays": {
            "file": arrays_output.name,
            "sha256": arrays_hash,
            "keys": ["forward", "adjoint"],
            "forward_shape": list(forward.shape),
            "adjoint_shape": list(adjoint.shape),
            "dtype": "float64",
        },
    }


def _read_head(repo: Path) -> str:
    """Read a target checkout's exact HEAD without invoking Git."""

    git_path = repo / ".git"
    if git_path.is_file():
        text = git_path.read_text().strip()
        git_dir = (repo / text.removeprefix("gitdir: ").strip()).resolve()
    else:
        git_dir = git_path
    common_dir = git_dir
    common_dir_file = git_dir / "commondir"
    if common_dir_file.exists():
        common_dir = (git_dir / common_dir_file.read_text().strip()).resolve()
    head = (git_dir / "HEAD").read_text().strip()
    if head.startswith("ref: "):
        ref_path = common_dir / head[5:]
        if ref_path.exists():
            return ref_path.read_text().strip()
        packed = common_dir / "packed-refs"
        if packed.exists():
            for line in packed.read_text().splitlines():
                if line and not line.startswith(("#", "^")):
                    sha, ref = line.split(" ", 1)
                    if ref == head[5:]:
                        return sha
    return head


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", required=True, type=Path)
    parser.add_argument("--mode", required=True, choices=("cold", "warm"))
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--seed", type=int, default=81073)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--arrays-output", type=Path)
    arguments = parser.parse_args(argv)
    arrays_output = (
        arguments.arrays_output.resolve()
        if arguments.arrays_output is not None
        else arguments.output.resolve().with_suffix(".npz")
    )
    result = run_sample(
        arguments.repo.resolve(),
        arguments.mode,
        arguments.iterations,
        arguments.seed,
        arrays_output,
    )
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(json.dumps(result, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
