"""Pure fusion of caller-selected candidates; no retrieval or model runtime."""

from __future__ import annotations

from collections.abc import Sequence
import math

from ..errors import MimiRAGError


def rrf(rankings: Sequence[Sequence[str]], *, constant: float) -> list[tuple[str, float]]:
    """Sum 1/(constant + one-based rank), breaking ties by node ID.

    Missing candidates contribute zero. Duplicate IDs within a channel are invalid.
    Truncate each channel before calling; select the final evidence budget afterward.
    """
    if not math.isfinite(constant) or constant <= 0:
        raise MimiRAGError('RRF constant must be finite and positive')
    scores: dict[str, float] = {}
    for ranking in rankings:
        if len(set(ranking)) != len(ranking):
            raise MimiRAGError('Duplicate candidate ID')
        for rank, node in enumerate(ranking, 1):
            scores[node] = scores.get(node, 0.0) + 1 / (constant + rank)
    return sorted(scores.items(), key=lambda item: (-item[1], item[0]))


def _normalized(candidates: Sequence[tuple[str, float]]) -> dict[str, float]:
    scores = dict(candidates)
    if len(scores) != len(candidates):
        raise MimiRAGError('Duplicate candidate ID')
    if any(not math.isfinite(score) for score in scores.values()):
        raise MimiRAGError('Candidate scores must be finite')
    if not scores:
        return {}
    low, high = min(scores.values()), max(scores.values())
    if low == high:
        return dict.fromkeys(scores, 1.0)
    # Scale first to avoid overflow when finite scores straddle the float range.
    scale = max(abs(low), abs(high))
    lo, hi = low / scale, high / scale
    return {node: (score / scale - lo) / (hi - lo) for node, score in scores.items()}


def convex(
    lexical: Sequence[tuple[str, float]], dense: Sequence[tuple[str, float]], *, weight: float
) -> list[tuple[str, float]]:
    """(1-weight)*lexical + weight*dense after per-channel min-max normalization.

    Inputs must be higher-is-better: negate raw SQLite BM25 before calling.
    Normalize over the supplied candidates only. A constant nonempty channel maps
    to one; missing candidates map to zero. Keep the union, including zero scores,
    and break ties by node ID. Empty channels do not redistribute weight.
    """
    if not math.isfinite(weight) or not 0 <= weight <= 1:
        raise MimiRAGError('Dense weight must be finite and in [0, 1]')
    left, right = _normalized(lexical), _normalized(dense)
    scores = {
        node: (1 - weight) * left.get(node, 0.0) + weight * right.get(node, 0.0) for node in left.keys() | right.keys()
    }
    return sorted(scores.items(), key=lambda item: (-item[1], item[0]))
