"""Aggregating per-case scores and comparing two runs case by case."""

from __future__ import annotations

from collections.abc import Mapping
import random
import statistics
from typing import Any

from ..errors import MimiRAGError


def macro(values: Mapping[str, float | None], groups: Mapping[str, str] | None = None) -> float | None:
    """Mean over cases, skipping None values.

    With `groups` (case to group), the mean of the group means.
    """
    defined = {case: value for case, value in values.items() if value is not None}
    if not groups:
        return statistics.fmean(defined.values()) if defined else None
    by_group: dict[str, list[float]] = {}
    for case, value in defined.items():
        by_group.setdefault(groups[case], []).append(value)
    return statistics.fmean(statistics.fmean(v) for v in by_group.values()) if by_group else None


def paired(
    a: Mapping[str, float | None],
    b: Mapping[str, float | None],
    *,
    seed: int,
    resamples: int = 2000,
    groups: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Paired comparison of two runs scored on the same cases (one value per case).

    Returns the mean difference b - a, a percentile bootstrap 95% interval, a
    two-sided sign-flip randomization p-value and, with `groups`, the sign of each
    group's mean difference. One seeded generator drives both resamplings.

    Cases are resampled independently even with `groups`, and the difference is a mean
    over cases, so it can differ from the difference of `macro` group means.
    """
    if resamples < 1:
        raise MimiRAGError('At least one resample is required')
    if set(a) != set(b):
        raise MimiRAGError('Paired runs must score the same cases')
    first = {case: value for case, value in a.items() if value is not None}
    second = {case: value for case, value in b.items() if value is not None}
    if len(first) != len(a) or len(second) != len(b):
        raise MimiRAGError('Paired values must be defined for every case; pool question variants first')
    cases = sorted(first)
    diffs = [second[c] - first[c] for c in cases]
    rng = random.Random(seed)
    boot = sorted(statistics.fmean(rng.choice(diffs) for _ in diffs) for _ in range(resamples))
    observed = statistics.fmean(diffs)
    extreme = sum(
        abs(statistics.fmean(d if rng.random() < 0.5 else -d for d in diffs)) >= abs(observed) - 1e-12
        for _ in range(resamples)
    )
    return dict(
        difference=observed,
        interval=[boot[int(0.025 * resamples)], boot[int(0.975 * resamples) - 1]],
        p_value=(extreme + 1) / (resamples + 1),
        cases=len(cases),
        resamples=resamples,
        seed=seed,
        group_signs=_group_signs(cases, diffs, groups) if groups else None,
    )


def _group_signs(cases: list[str], diffs: list[float], groups: Mapping[str, str]) -> dict[str, int]:
    """+1, 0 or -1 for each group's mean difference, by group name."""
    by_group: dict[str, list[float]] = {}
    for case, difference in zip(cases, diffs, strict=True):
        by_group.setdefault(groups[case], []).append(difference)
    means = {group: statistics.fmean(values) for group, values in sorted(by_group.items())}
    return {group: (mean > 0) - (mean < 0) for group, mean in means.items()}
