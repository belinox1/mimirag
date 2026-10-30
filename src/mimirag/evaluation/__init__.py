"""Retrieval evaluation: quote alignment, nugget coverage, rank metrics, paired comparisons,
run manifests, packet cost, an in-memory mirror of the index and blinded judgment pools."""

from .alignment import ALIGNMENT, align, cited_page, cited_pages, document_key, line_window, normalize
from .comparison import macro, paired
from .mirror import TOKENIZERS, MirrorIndex, document_nodes, node_id
from .pooling import blinded_pool
from .runs import check_pair, run_manifest, run_nodes
from .scoring import (
    SCORER,
    TEXT_KINDS,
    condensed,
    evidence_status,
    judged_at,
    packet_cost,
    reciprocal_rank,
    score_case,
    unmapped_rate,
    within_budget,
)

__all__ = [
    'ALIGNMENT',
    'SCORER',
    'TEXT_KINDS',
    'TOKENIZERS',
    'MirrorIndex',
    'align',
    'blinded_pool',
    'check_pair',
    'cited_page',
    'cited_pages',
    'condensed',
    'document_key',
    'document_nodes',
    'evidence_status',
    'judged_at',
    'line_window',
    'macro',
    'node_id',
    'normalize',
    'packet_cost',
    'paired',
    'reciprocal_rank',
    'run_manifest',
    'run_nodes',
    'score_case',
    'unmapped_rate',
    'within_budget',
]
