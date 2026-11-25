"""Comparison logic for paired Stage 1 numerical/performance benchmarks."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


BENCHMARK_SCHEMA_VERSION = "stage1-benchmark-v2"
MIB = 1024 * 1024


@dataclass(frozen=True)
class EngineeringBudget:
    """Two-part mechanical regression budgets from the staged science plan."""

    time_fraction: float = 0.25
    time_absolute_seconds: float = 0.25
    memory_fraction: float = 0.20
    memory_absolute_bytes: int = 20 * MIB
    maximum_relative_mad: float = 0.10
    allocation_fraction: float = 0.20
    maximum_allocation_growth_ratio: float = 12.0


def summarize_samples(samples: list[float]) -> dict[str, float | list[float]]:
    """Return median and robust dispersion for positive repeated measurements."""

    values = np.asarray(samples, dtype=np.float64)
    if values.ndim != 1 or values.size < 2 or not np.isfinite(values).all() or np.any(values < 0.0):
        raise ValueError("benchmark samples require at least two finite nonnegative values")
    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median)))
    return {
        "samples": values.tolist(),
        "median": median,
        "mad": mad,
        "relative_mad": mad / max(median, np.finfo(float).tiny),
        "minimum": float(np.min(values)),
        "maximum": float(np.max(values)),
    }


def _crosses_budget(base: float, head: float, fraction: float, absolute: float) -> bool:
    return (head - base) > absolute and (head / max(base, np.finfo(float).tiny) - 1.0) > fraction


def compare_performance(
    base: dict[str, Any],
    head: dict[str, Any],
    *,
    confirmation_base: dict[str, Any] | None = None,
    confirmation_head: dict[str, Any] | None = None,
    budget: EngineeringBudget = EngineeringBudget(),
) -> dict[str, Any]:
    """Evaluate paired time/memory samples with variability and confirmation rules.

    A first-run budget crossing is ``confirmation_required``.  Failure requires a
    second independent paired set to cross the same budget.  High-dispersion sets
    are ``inconclusive`` and never silently converted into a pass.
    """

    time_base = summarize_samples(list(base["wall_seconds"]))
    time_head = summarize_samples(list(head["wall_seconds"]))
    memory_base = summarize_samples(list(base["peak_memory_bytes"]))
    memory_head = summarize_samples(list(head["peak_memory_bytes"]))
    noisy = any(
        float(summary["relative_mad"]) > budget.maximum_relative_mad
        for summary in (time_base, time_head, memory_base, memory_head)
    )
    time_suspect = _crosses_budget(
        float(time_base["median"]),
        float(time_head["median"]),
        budget.time_fraction,
        budget.time_absolute_seconds,
    )
    memory_suspect = _crosses_budget(
        float(memory_base["median"]),
        float(memory_head["median"]),
        budget.memory_fraction,
        float(budget.memory_absolute_bytes),
    )
    result: dict[str, Any] = {
        "base": {"time": time_base, "memory": memory_base},
        "head": {"time": time_head, "memory": memory_head},
        "time_suspect": time_suspect,
        "memory_suspect": memory_suspect,
        "noisy": noisy,
        "budget": budget.__dict__,
    }
    if not (time_suspect or memory_suspect):
        if noisy:
            result.update(status="inconclusive", reason="measurement dispersion exceeds the predeclared bound")
            return result
        result.update(status="passed", reason="no paired engineering budget crossed")
        return result
    if confirmation_base is None or confirmation_head is None:
        reason = "suspected regression requires a confirming paired rerun"
        if noisy:
            reason += "; the initial samples are also noisy"
        result.update(status="confirmation_required", reason=reason)
        return result

    confirm = compare_performance(confirmation_base, confirmation_head, budget=budget)
    result["confirmation"] = confirm
    confirmed_time = time_suspect and bool(confirm["time_suspect"])
    confirmed_memory = memory_suspect and bool(confirm["memory_suspect"])
    new_time_suspect = not time_suspect and bool(confirm["time_suspect"])
    new_memory_suspect = not memory_suspect and bool(confirm["memory_suspect"])
    if noisy or bool(confirm["noisy"]):
        result.update(status="inconclusive", reason="initial or confirming paired samples are too noisy")
    elif confirmed_time or confirmed_memory:
        result.update(status="failed", reason="controlled regression crossed the budget on both paired runs")
    elif new_time_suspect or new_memory_suspect:
        result.update(
            status="inconclusive",
            reason="a new budget crossing appeared only in confirmation and remains unconfirmed",
        )
    else:
        result.update(status="passed", reason="suspected regression was not reproduced by the confirming rerun")
    return result


def compare_allocation_scaling(
    base: list[dict[str, Any]],
    head: list[dict[str, Any]],
    *,
    budget: EngineeringBudget = EngineeringBudget(),
) -> dict[str, Any]:
    """Enforce deterministic allocation-growth and base/head scaling bounds.

    Estimated allocation sizes are deterministic properties of the fixed workload,
    so they are gated directly rather than subjected to noisy timing confirmation.
    """

    if not base or not head:
        return {"passed": False, "reason": "allocation records are missing", "budget": budget.__dict__}
    required = {
        "small_scene_shape",
        "large_scene_shape",
        "small_estimated_peak_bytes",
        "large_estimated_peak_bytes",
        "peak_growth_ratio",
        "small_parameter_count",
        "large_parameter_count",
    }
    records = base + head
    if any(not required.issubset(record) for record in records):
        return {"passed": False, "reason": "allocation record is incomplete", "budget": budget.__dict__}

    reference = base[0]
    structural_keys = (
        "small_scene_shape",
        "large_scene_shape",
        "small_parameter_count",
        "large_parameter_count",
    )
    structure_matches = all(
        all(record[key] == reference[key] for key in structural_keys) for record in records
    )
    try:
        base_large = np.asarray(
            [record["large_estimated_peak_bytes"] for record in base], dtype=np.float64
        )
        head_large = np.asarray(
            [record["large_estimated_peak_bytes"] for record in head], dtype=np.float64
        )
        growth = np.asarray([record["peak_growth_ratio"] for record in records], dtype=np.float64)
    except (TypeError, ValueError):
        return {"passed": False, "reason": "allocation values are invalid", "budget": budget.__dict__}
    finite_positive = all(
        values.size > 0 and np.isfinite(values).all() and np.all(values > 0.0)
        for values in (base_large, head_large, growth)
    )
    base_median = float(np.median(base_large)) if finite_positive else None
    maximum_head = float(np.max(head_large)) if finite_positive else None
    maximum_growth = float(np.max(growth)) if finite_positive else None
    head_fraction = maximum_head / base_median - 1.0 if finite_positive else None
    passed = bool(
        structure_matches
        and finite_positive
        and head_fraction is not None
        and head_fraction <= budget.allocation_fraction
        and maximum_growth is not None
        and maximum_growth <= budget.maximum_allocation_growth_ratio
    )
    return {
        "passed": passed,
        "reason": (
            "deterministic allocation bounds satisfied"
            if passed
            else "deterministic allocation or workload-structure bound crossed"
        ),
        "structure_matches": structure_matches,
        "base_large_median_bytes": base_median,
        "head_large_maximum_bytes": maximum_head,
        "head_large_fractional_growth": head_fraction,
        "maximum_peak_growth_ratio": maximum_growth,
        "budget": budget.__dict__,
    }


def compare_numerical_arrays(
    base: np.ndarray, head: np.ndarray, *, rtol: float = 1e-10, atol: float = 1e-12
) -> dict[str, float | bool]:
    """Gate performance claims on finite, shape-matched numerical equivalence."""

    base_array = np.asarray(base, dtype=np.float64)
    head_array = np.asarray(head, dtype=np.float64)
    if base_array.shape != head_array.shape or not np.isfinite(base_array).all() or not np.isfinite(head_array).all():
        return {"passed": False, "maximum_absolute_difference": float("inf"), "maximum_relative_difference": float("inf")}
    absolute = np.abs(base_array - head_array)
    relative = absolute / np.maximum(np.abs(base_array), atol)
    return {
        "passed": bool(np.allclose(base_array, head_array, rtol=rtol, atol=atol)),
        "maximum_absolute_difference": float(np.max(absolute, initial=0.0)),
        "maximum_relative_difference": float(np.max(relative, initial=0.0)),
    }
