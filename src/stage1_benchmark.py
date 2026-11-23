"""Comparison logic for paired Stage 1 numerical/performance benchmarks."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


BENCHMARK_SCHEMA_VERSION = "stage1-benchmark-v1"
MIB = 1024 * 1024


@dataclass(frozen=True)
class EngineeringBudget:
    """Two-part mechanical regression budgets from the staged science plan."""

    time_fraction: float = 0.25
    time_absolute_seconds: float = 0.25
    memory_fraction: float = 0.20
    memory_absolute_bytes: int = 20 * MIB
    maximum_relative_mad: float = 0.10


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
    if noisy or bool(confirm["noisy"]):
        result.update(status="inconclusive", reason="initial or confirming paired samples are too noisy")
    elif confirmed_time or confirmed_memory:
        result.update(status="failed", reason="controlled regression crossed the budget on both paired runs")
    else:
        result.update(status="passed", reason="suspected regression was not reproduced by the confirming rerun")
    return result


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
