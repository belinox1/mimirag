"""Run manifests: self-hashing records of a retrieval run, for re-scoring it later.

A manifest records what was ranked, not how it was scored: put the reference key's
hash and the scoring options in `system`. `check_pair` and `run_nodes` compare and
rebuild recorded fields; recompute `run_sha256` and `documents_sha256` with
`integrity.fingerprint` to verify a stored manifest before trusting it.
"""

from __future__ import annotations

from collections.abc import Mapping
import copy
from typing import Any

from ..errors import MimiRAGError
from ..integrity import fingerprint
from .alignment import ALIGNMENT, Documents
from .scoring import SCORER, Nodes

PAIRED_FIELDS = ('corpus_manifest_sha256', 'record_export_sha256', 'scorer', 'alignment', 'documents_sha256')


def run_manifest(
    *,
    system: Any,
    corpus_manifest_sha256: str,
    record_export_sha256: str | None,
    ranked: Mapping[str, Mapping[str, list[str]]],
    nodes: Nodes,
    documents: Documents,
    source_map: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Record of a run, hashed as `run_sha256`.

    Holds the ranked lists, the text of every ranked node, the full node inventory
    and the hash of the extracted text, so a later re-score sees the same nodes and text.
    The record is a copy: later changes to the arguments do not alter it.
    """
    used = {n for lists in ranked.values() for r in lists.values() for n in r}
    inventory = {n: {k: v[k] for k in ('doc', 'page', 'start', 'end')} for n, v in sorted(nodes.items())}
    text = {d: {str(p): t for p, t in pages.items()} for d, pages in documents.items()}
    value = dict(
        system=system,
        scorer=SCORER,
        alignment=ALIGNMENT,
        corpus_manifest_sha256=corpus_manifest_sha256,
        record_export_sha256=record_export_sha256,
        documents_sha256=fingerprint(text),
        source_map=source_map or {},
        ranked=ranked,
        inventory=inventory,
        nodes={n: nodes[n] for n in sorted(used)},
    )
    return copy.deepcopy(dict(value, run_sha256=fingerprint(value)))


def check_pair(first: Mapping[str, Any], second: Mapping[str, Any], *, same_units: bool = True) -> None:
    """Refuse a pair of run manifests that differ in corpus, records, scorer or extracted text.

    With `same_units`, the node inventories must match too.
    """
    for field in PAIRED_FIELDS + (('inventory',) if same_units else ()):
        if first[field] != second[field]:
            raise MimiRAGError(f'Paired runs differ in {field}')


def run_nodes(manifest: Mapping[str, Any], documents: Documents) -> dict[str, dict[str, Any]]:
    """Rebuild nodes with their text from a run manifest's inventory and the extracted documents."""
    nodes = {}
    for n, spec in manifest['inventory'].items():
        text = documents[spec['doc']][spec['page']]
        nodes[n] = dict(spec, text=text[spec['start'] : spec['end']])
    return nodes
