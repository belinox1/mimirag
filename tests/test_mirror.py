from helpers import source
import pytest

from mimirag.errors import MimiRAGError
from mimirag.evaluation import MirrorIndex, blinded_pool, document_nodes, node_id, run_nodes
from mimirag.extraction import extract_document
from mimirag.index import Index

LONG = ' filler' * 300  # over the 1800-character unit target, so each block is its own unit
DOCUMENTS = {
    'a': {None: 'Alpha one.\n\nAlpha two\ncontinues here.\n'},
    'b': {1: 'Beta page.\n', 2: 'Alpha on page two.\n'},
    'c': {1: 'Gamma first' + LONG + '\n\nGamma second\nline' + LONG + '\n'},
}


def test_document_nodes_keep_offsets_into_the_extracted_page():
    nodes = document_nodes(DOCUMENTS)
    assert len(nodes) == 5
    for key, node in nodes.items():
        assert key == node_id(node['doc'], node['page'], *node['lines'])
        page = DOCUMENTS[node['doc']][node['page']]
        assert page[node['start'] : node['end']] == node['text'] == node['ranking_text']
    assert [n['lines'] for n in nodes.values()] == [[1, 4], [1, 1], [1, 1], [1, 1], [3, 4]]


def test_mirror_ranks_like_the_index(tmp_path):
    # Blocks over the unit target give several units per file; c.txt repeats b.txt to force ties.
    texts = {
        'a.txt': f'river stone river{LONG}\n\nstone only{LONG}\n\nriver only here{LONG}\n',
        'b.txt': f'stone river path{LONG}\n\nnothing related{LONG}\n\nriver{LONG}\n',
    }
    texts['c.txt'] = texts['b.txt']
    for name, text in texts.items():
        (tmp_path / name).write_text(text)
    manifest = dict(documents=[source(tmp_path, name) for name in texts])
    index = Index(tmp_path / 'index.sqlite3', manifest, root=tmp_path)
    index.build()
    documents = {s['id']: dict(extract_document(tmp_path, s).pages) for s in manifest['documents']}
    with MirrorIndex(document_nodes(documents)) as mirror:
        for query in ['river', 'stone', 'river stone path', 'only']:
            served = index.search(query, limit=20)['evidence']
            locators = [item['locator'] for item in served]
            want = [node_id(x['source'], x['page'], *x['lines']) for x in locators]
            assert mirror.rank(query) == want
            if query == 'river':
                assert len(want) == 6  # two units per file, b and c tied


def test_headers_change_ranking_only():
    nodes = document_nodes(DOCUMENTS)
    beta = next(k for k, n in nodes.items() if n['text'] == 'Beta page.')
    with MirrorIndex(nodes) as plain, MirrorIndex(nodes, {beta: 'Alpha guide'}) as headed:
        assert beta not in plain.rank('alpha')
        assert beta in headed.rank('alpha')
    assert nodes[beta]['text'] == 'Beta page.'


def test_stemming_and_stopwords_are_ranking_variants():
    text = f'The lanterns burned{LONG}\n\nA lantern of the keeper{LONG}\n\nOf the night{LONG}\n'
    nodes = document_nodes({'d': {None: text}})
    ids = list(nodes)
    variant = dict(tokenizer='porter unicode61', stopwords={'THE', 'of'})
    with MirrorIndex(nodes) as plain, MirrorIndex(nodes, **variant) as stemmed:
        assert plain.rank('lantern') == [ids[1]]
        assert stemmed.rank('lantern') == ids[:2]
        assert ids[2] in plain.rank('of the lantern')
        assert ids[2] not in stemmed.rank('of the lantern')  # stopwords dropped from the question only
        with pytest.raises(MimiRAGError, match='at least one word'):
            stemmed.rank('Of the')
    with pytest.raises(MimiRAGError, match='Unknown tokenizer'):
        MirrorIndex(nodes, tokenizer='trigram')


def test_blinded_pool_hides_runs_and_is_seeded():
    nodes = {k: dict(v, source=dict(id=v['doc'])) for k, v in document_nodes(DOCUMENTS).items()}
    ids = list(nodes)
    runs = {'first': dict(prefixes={'c1': {'blind': ids[:2]}}), 'second': dict(prefixes={'c1': {'blind': ids[1:3]}})}
    cases = {
        'c1': dict(
            questions=[dict(style='blind', text='Which alpha?')],
            nuggets=[dict(id='n1'), dict(id='n2', styles=['named'])],
        )
    }
    rows = {'c1': dict(context='Testing context')}
    packet, provenance = blinded_pool(runs, nodes, cases=cases, rows=rows, seed=3)
    assert (packet, provenance) == blinded_pool(runs, nodes, cases=cases, rows=rows, seed=3)
    assert len(packet['items']) == 3
    assert packet['seed'] == 3
    assert 'first' not in str(packet)
    assert 'second' not in str(packet)
    item = packet['items'][0]
    assert item['question'] == 'Which alpha?'
    assert item['requirements'] == [dict(id='n1')]
    assert item['judgment'] is None
    shared = next(p for p in provenance.values() if p['node'] == ids[1])
    assert shared['origins'] == [dict(arm='first', rank=2), dict(arm='second', rank=1)]


def test_run_nodes_rebuild_text_from_a_manifest():
    nodes = document_nodes(DOCUMENTS)
    inventory = {k: {f: v[f] for f in ('doc', 'page', 'start', 'end')} for k, v in nodes.items()}
    rebuilt = run_nodes(dict(inventory=inventory), DOCUMENTS)
    assert {k: v['text'] for k, v in rebuilt.items()} == {k: v['text'] for k, v in nodes.items()}
