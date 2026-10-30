import pytest

from mimirag import evaluation as ev
from mimirag.errors import MimiRAGError

DOCUMENTS = {
    'birds': {
        1: 'Owls hunt at night.\n\nThey fly silently over fields.\n',
        2: 'Bats also hunt at night,\nusing echo-\nlocation.\n',
    },
    'repo:src/call.py': {None: 'def hoot():\n    return "hoo"\n\ndef quiet():\n    return "hoo"\n'},
}


def node(doc, page, text, occurrence=0):
    page_text = DOCUMENTS[doc][page]
    start = -1
    for _ in range(occurrence + 1):
        start = page_text.index(text, start + 1)
    return dict(doc=doc, page=page, start=start, end=start + len(text), text=text)


NODES = {
    'owls': node('birds', 1, 'Owls hunt at night.'),
    'fly': node('birds', 1, 'They fly silently over fields.'),
    'bats': node('birds', 2, 'Bats also hunt at night,\nusing echo-\nlocation.'),
    'hoot': node('repo:src/call.py', None, 'def hoot():\n    return "hoo"'),
    'quiet': node('repo:src/call.py', None, 'def quiet():\n    return "hoo"'),
}


def evidence(id, quote, kind='text-layer', source='birds', **fields):
    return dict(id=id, quote=quote, verified_against=kind, source=source, **fields)


CASE = dict(
    evidence=[
        evidence('fly', 'fly silently', locator='p1'),
        evidence('echo', 'echolocation', locator='p2'),
        evidence('code', 'return "hoo"', kind='git-blob', source='repo', path='src/call.py', lines=[5, 5]),
        evidence('image', 'Owls hunt at night', kind='page-image', locator='pdf p1'),
    ]
)
SUPPORT = {
    'flight': dict(styles=['blind', 'named'], coverage_gap=False, alternatives=[['fly']]),
    'sonar': dict(styles=['blind', 'named'], coverage_gap=False, alternatives=[['echo', 'code']]),
    'named-only': dict(styles=['named'], coverage_gap=False, alternatives=[['image']]),
    'gap': dict(styles=['blind'], coverage_gap=True, alternatives=[]),
}


def score(ranked, **kwargs):
    return ev.score_case(CASE, SUPPORT, DOCUMENTS, NODES, ranked, **dict(dict(style='blind', overlap=0.5), **kwargs))


def test_normalize_folds_text_and_maps_back_to_raw_offsets():
    raw = 'Café “quoted”  well-\n  known — text'
    text, starts, ends = ev.normalize(raw)
    assert text == 'Café "quoted" wellknown text'
    assert raw[starts[0] : ends[3]] == 'Café'
    assert raw[starts[text.index('w')] : ends[text.index('known') + 4]] == 'well-\n  known'
    assert ev.normalize('echo-\r\nlocation')[0] == 'echolocation'
    # Fullwidth A and typographic single quotes.
    assert ev.normalize('\uff21\u2018x\u2019')[0] == "A'x'"


def test_page_locators():
    assert ev.cited_page(dict(locator='pdf p12, table 3')) == 12
    assert ev.cited_page(dict(locator='Table 3 (2019)')) is None
    assert ev.cited_pages(dict(locator='p2-4 and p7')) == [2, 3, 4, 7]
    assert ev.cited_pages(dict(locator='p3\u2013p4')) == [3, 4]  # en dash
    assert ev.cited_pages(dict(locator='section 2')) == []
    with pytest.raises(MimiRAGError, match='Invalid PDF page range'):
        ev.cited_pages(dict(locator='p5-2'))


def test_alignment_prefers_the_cited_page_and_respects_line_windows():
    assert ev.align(evidence('x', 'hunt at night', locator='p2'), DOCUMENTS) == ('birds', 2, 10, 23)
    assert ev.align(evidence('x', 'hunt at night', locator='p1'), DOCUMENTS) == ('birds', 1, 5, 18)
    pages = {'birds': {1: 'Owls rest.\n', 2: 'Fields.\n', 3: 'Owls rest.\n'}}
    assert ev.align(evidence('x', 'Owls rest', locator='p2-3'), pages)[1] == 3  # every cited page before others
    short = {'birds': {1: 'Owls.\n', 2: 'a\nb\nOwls rest.\n'}}
    assert ev.align(evidence('x', 'Owls rest', locator='p1-2', lines=[3, 3]), short)[:2] == ('birds', 2)
    code = CASE['evidence'][2]
    key, page, start, end = ev.align(code, DOCUMENTS)
    assert DOCUMENTS[key][page][start:end] == 'return "hoo"'
    assert start > NODES['quiet']['start']
    assert ev.align(evidence('x', 'absent words'), DOCUMENTS) is None
    assert ev.align(evidence('x', 'Owls', source='old-id'), DOCUMENTS, {'old-id': 'birds'})[:2] == ('birds', 1)
    with pytest.raises(MimiRAGError, match='No extracted document'):
        ev.align(evidence('x', 'Owls', source='unknown'), DOCUMENTS)
    with pytest.raises(MimiRAGError, match='outside the document'):
        ev.align(dict(code, lines=[9, 9]), DOCUMENTS)


def test_budgets():
    ranked = ['owls', 'fly', 'bats']
    assert ev.within_budget(ranked, NODES, k=2) == ['owls', 'fly']
    assert ev.within_budget(ranked, NODES, characters=50) == ['owls', 'fly']
    assert ev.within_budget(ranked, NODES, characters=10) == []


def test_evidence_states():
    def status(item, retrieved, nodes=NODES, page_image='strict'):
        span = ev.align(item, DOCUMENTS)
        return ev.evidence_status(
            item, span, nodes, set(retrieved), overlap=0.5, page_image=page_image, documents=DOCUMENTS
        )

    across = evidence('x', 'night. They fly', locator='p1')
    assert status(across, ['owls', 'fly']) == 'retrieved'
    assert status(across, ['owls']) == 'chunk-boundary'
    assert status(across, ['bats']) == 'missed'
    assert status(evidence('x', 'not in the text'), ['owls']) == 'unmapped'
    without_page_two = {k: v for k, v in NODES.items() if k != 'bats'}
    assert status(CASE['evidence'][1], [], without_page_two) == 'uncovered'
    assert status(CASE['evidence'][2], [], {}) == 'missed'
    image = CASE['evidence'][3]
    assert status(image, ['owls']) == 'retrieved'
    assert status(image, ['fly']) == 'missed'
    assert status(image, ['fly'], page_image='lenient') == 'retrieved'
    with pytest.raises(MimiRAGError, match='cited extracted page 3'):
        status(dict(image, locator='p3'), ['owls'])


def test_case_coverage_by_style():
    result = score(['fly', 'bats', 'quiet'])
    assert result['requirements'] == ['flight', 'sonar']
    assert result['covered'] == {'flight': True, 'sonar': True}
    assert result['coverage'] == 1.0
    result = score(['fly', 'bats'], k=1)
    assert result['coverage'] == 0.5
    assert result['any_evidence'] == 0.5
    result = score(['fly', 'bats'])
    assert result['covered']['sonar'] is False
    assert result['any_evidence'] == 1.0
    assert result['states']['code'] == 'missed'
    assert score(['owls'], style='named')['covered'] == {'flight': False, 'sonar': False, 'named-only': True}


def test_added_alternatives_count_only_for_named_sources_or_explicit_blind_credit():
    support = dict(
        flight=dict(styles=['blind'], coverage_gap=False, alternatives=[dict(evidence=['echo'], added=True)])
    )

    def run(**options):
        return ev.score_case(CASE, support, DOCUMENTS, NODES, ['bats'], style='blind', overlap=0.5, **options)

    assert run()['coverage'] == 0.0
    assert run(named_sources=['birds'])['coverage'] == 1.0
    assert run(added_credit='unrestricted')['coverage'] == 1.0
    twice = dict(support, again=support['flight'])
    listed = ev.score_case(CASE, twice, DOCUMENTS, NODES, ['bats'], style='blind', overlap=0.5, named_sources=['birds'])
    assert listed['coverage'] == 1.0
    assert listed == ev.score_case(
        CASE, twice, DOCUMENTS, NODES, ['bats'], style='blind', overlap=0.5, named_sources=iter(['birds'])
    )
    with pytest.raises(MimiRAGError, match='only valid for blind'):
        ev.score_case(CASE, support, DOCUMENTS, NODES, [], style='named', overlap=0.5, added_credit='unrestricted')


def test_off_page_matches_are_reported_and_optionally_unmapped():
    case = dict(evidence=[evidence('x', 'Bats also', locator='p1')])
    support = dict(r=dict(styles=['blind'], coverage_gap=False, alternatives=[['x']]))
    result = ev.score_case(case, support, DOCUMENTS, NODES, ['bats'], style='blind', overlap=0.5)
    assert result['off_page'] == ['x']
    assert result['coverage'] == 1.0
    in_range = dict(evidence=[evidence('x', 'Bats also', locator='p1-2')])
    assert ev.score_case(in_range, support, DOCUMENTS, NODES, ['bats'], style='blind', overlap=0.5)['off_page'] == []
    result = ev.score_case(case, support, DOCUMENTS, NODES, ['bats'], style='blind', overlap=0.5, cited_page_only=True)
    assert result['states'] == {'x': 'unmapped'}
    assert result['coverage'] == 0.0


def test_unsupported_evidence_and_rules_are_refused():
    with pytest.raises(MimiRAGError, match='page-image rule'):
        score([], page_image='Lenient')
    with pytest.raises(MimiRAGError, match='Unsupported evidence kind'):
        ev.score_case(
            dict(evidence=[evidence('x', 'q', kind='record')]), {}, DOCUMENTS, NODES, [], style='blind', overlap=0.5
        )
    with pytest.raises(MimiRAGError, match='no parseable page'):
        ev.score_case(
            dict(evidence=[evidence('x', 'q', kind='page-image', locator='figure 2')]),
            {},
            DOCUMENTS,
            NODES,
            [],
            style='blind',
            overlap=0.5,
        )


def test_reciprocal_rank():
    assert ev.reciprocal_rank(CASE, DOCUMENTS, NODES, ['owls', 'bats', 'fly']) == 0.5
    assert ev.reciprocal_rank(CASE, DOCUMENTS, NODES, ['owls', 'bats', 'fly'], evidence_ids={'fly'}) == 1 / 3
    assert ev.reciprocal_rank(CASE, DOCUMENTS, NODES, ['owls']) == 0.0
    assert ev.reciprocal_rank(CASE, DOCUMENTS, NODES, ['owls'], evidence_ids={'image'}) is None


def test_diagnostics():
    assert ev.judged_at(['owls', 'fly', 'bats'], {'fly'}, 2) == 0.5
    assert ev.judged_at([], {'fly'}, 2) is None
    assert ev.condensed(['owls', 'fly', 'bats'], {'bats', 'owls'}) == ['owls', 'bats']
    cases = [CASE, dict(evidence=[evidence('x', 'missing text')])]
    assert ev.unmapped_rate(cases, DOCUMENTS) == 0.25
    assert ev.unmapped_rate([], DOCUMENTS) is None


def test_macro_means():
    values = dict(a=1.0, b=0.0, c=0.0, d=None)
    assert ev.macro(values) == pytest.approx(1 / 3)
    assert ev.macro(values, dict(a='x', b='y', c='y', d='y')) == 0.5
    assert ev.macro(dict(a=None)) is None


def test_paired_comparison_is_seeded_and_checks_its_inputs():
    a = dict(c1=0.2, c2=0.5, c3=0.1, c4=0.4)
    b = dict(c1=0.6, c2=0.7, c3=0.3, c4=0.4)
    result = ev.paired(a, b, seed=7, resamples=500, groups=dict(c1='x', c2='x', c3='y', c4='z'))
    assert result == ev.paired(a, b, seed=7, resamples=500, groups=dict(c1='x', c2='x', c3='y', c4='z'))
    assert result['difference'] == pytest.approx(0.2)
    assert result['interval'][0] <= 0.2 <= result['interval'][1]
    assert 0 < result['p_value'] <= 1
    assert result['group_signs'] == dict(x=1, y=1, z=0)
    with pytest.raises(MimiRAGError, match='same cases'):
        ev.paired(a, dict(c1=1.0), seed=1)
    with pytest.raises(MimiRAGError, match='defined for every case'):
        ev.paired(dict(a, c1=None), b, seed=1)
    with pytest.raises(MimiRAGError, match='resample'):
        ev.paired(a, b, seed=1, resamples=0)


def test_paired_comparison_has_the_expected_extremes():
    scores = {f'c{i}': i / 20 for i in range(12)}
    same = ev.paired(scores, scores, seed=1, resamples=500)
    assert same['difference'] == 0
    assert same['interval'] == [0, 0]
    assert same['p_value'] == 1
    shifted = ev.paired(scores, {c: v + 0.1 for c, v in scores.items()}, seed=1, resamples=500)
    assert shifted['difference'] == pytest.approx(0.1)
    # Only two of 4096 sign patterns are as extreme, so the p-value is near its floor of 1/501.
    assert shifted['p_value'] < 0.01


def test_run_manifests_carry_ranked_nodes_and_pairs_must_share_the_corpus():
    ranked = {'c1': {'blind': ['fly', 'owls']}}

    def make(**changes):
        fields = dict(
            system=dict(name='bm25'),
            corpus_manifest_sha256='a',
            record_export_sha256=None,
            ranked=ranked,
            nodes=NODES,
            documents=DOCUMENTS,
        )
        return ev.run_manifest(**dict(fields, **changes))

    first = make()
    assert first['scorer'] == ev.SCORER
    assert first['alignment'] == ev.ALIGNMENT
    assert set(first['nodes']) == {'fly', 'owls'}
    assert set(first['inventory']) == set(NODES)
    assert first['run_sha256'] == make()['run_sha256'] != make(system=dict(name='other'))['run_sha256']
    ev.check_pair(first, make(system=dict(name='other')))
    with pytest.raises(MimiRAGError, match='corpus_manifest_sha256'):
        ev.check_pair(first, make(corpus_manifest_sha256='b'))
    fewer = make(nodes={k: NODES[k] for k in ('fly', 'owls')})
    with pytest.raises(MimiRAGError, match='inventory'):
        ev.check_pair(first, fewer)
    ev.check_pair(first, fewer, same_units=False)
    ranked['c1']['blind'].clear()  # the manifest keeps its own copy
    assert first['ranked'] == {'c1': {'blind': ['fly', 'owls']}}


def test_packet_cost_per_useful_item():
    cost = ev.packet_cost(['owls', 'hoot'], NODES, useful=2)
    assert cost['characters'] == len('Owls hunt at night.') + len('def hoot():\n    return "hoo"')
    assert cost['tokens'] == 5 + 9  # 'Owls hunt at night .' and 'def hoot ( ) : return " hoo "'
    assert cost['characters_per_item'] == cost['characters'] / 2
    assert cost['tokens_per_item'] == 7
    empty = ev.packet_cost(['owls'], NODES, useful=0)
    assert empty['characters_per_item'] is None
    assert empty['tokens_per_item'] is None
