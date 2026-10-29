"""Blinded judgment pools: several runs' results, merged, shuffled and without run labels."""

from __future__ import annotations

from collections.abc import Mapping
import random
from typing import Any

from .scoring import Nodes

POOL_DEPTH = 10
"""Results taken from the top of each ranked list."""


def blinded_pool(
    runs: Mapping[str, Mapping[str, Any]],
    nodes: Nodes,
    *,
    cases: Mapping[str, Mapping[str, Any]],
    rows: Mapping[str, Mapping[str, Any]],
    seed: int,
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    """Pool the top 10 of every run into one shuffled judgment packet without run labels.

    `runs` maps a run name to its `prefixes` ({case: {style: ranked node IDs}}).
    Each node needs `source` besides the `document_nodes` fields; each case needs
    `questions` (with `style` and `text`) and `nuggets` (optionally limited to some
    `styles`); `rows` gives each case's `context` for the judge. Returns (packet,
    provenance): provenance maps packet item IDs to the node and the runs and ranks
    that retrieved it, and is kept away from the judge.
    """
    pooled: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for arm, run in runs.items():
        for case_id, styles in run['prefixes'].items():
            for style, ranked in styles.items():
                for rank, n in enumerate(ranked[:POOL_DEPTH], 1):
                    pooled.setdefault((case_id, style, n), []).append(dict(arm=arm, rank=rank))
    entries = sorted(pooled)
    random.Random(seed).shuffle(entries)
    items, provenance = [], {}
    for number, (case_id, style, n) in enumerate(entries, 1):
        item_id = f'p{number:05d}'
        case = cases[case_id]
        items.append(
            dict(
                id=item_id,
                case=case_id,
                style=style,
                question=next(q['text'] for q in case['questions'] if q['style'] == style),
                context=rows[case_id]['context'],
                requirements=[r for r in case['nuggets'] if style in r.get('styles', ['blind', 'named'])],
                source=nodes[n]['source'],
                evidence={k: nodes[n][k] for k in ('doc', 'page', 'start', 'end', 'text')},
                judgment=None,
            )
        )
        provenance[item_id] = dict(node=n, origins=pooled[case_id, style, n])
    return dict(seed=seed, items=items), provenance
