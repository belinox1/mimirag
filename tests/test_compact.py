import copy
import json

import pytest

from mimirag.errors import MimiRAGError
from mimirag.packets import compact_packet, evidence_packet


def expand(packet):
    return {
        **{k: v for k, v in packet.items() if k not in ('format', 'sources')},
        'evidence': [{**e, **packet['sources'][e['source']]} for e in packet['evidence']],
    }


@pytest.mark.parametrize('count', [0, 1, 4])
def test_lossless_local_metadata(count):
    item = dict(
        source=dict(id='manual', title='Lámpara ' * 100),
        locator=dict(source='manual', sha256='a' * 64),
        excerpt='Luz\n',
        qualification='Check the original.',
        freshness='Verified.',
        truncated=False,
        context=[dict(excerpt='Adjacent text')],
    )
    original = dict(evidence_packet('search', [copy.deepcopy(item) for _ in range(count)], notices=['Notice']))
    snapshot = copy.deepcopy(original)
    compact = compact_packet(original)
    assert expand(compact) == original == snapshot
    assert compact == compact_packet(original)
    assert len(compact['sources']) == bool(count)
    if count > 1:
        assert len(json.dumps(compact, ensure_ascii=False)) < len(json.dumps(original, ensure_ascii=False))
    if count:
        assert compact['evidence'][0]['locator'] == item['locator']


def test_metadata_conflict_refused():
    item = dict(source=dict(id='manual'), qualification='One', freshness='Checked')
    with pytest.raises(MimiRAGError, match='Conflicting'):
        compact_packet(dict(evidence_packet('search', [item, dict(item, qualification='Two')])))
