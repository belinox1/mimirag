import json
import os
import sqlite3

from helpers import source
import pytest

from mimirag import index as index_module
from mimirag.datatypes import Page
from mimirag.errors import IntegrityError, LocatorError, MimiRAGError, RebuildRequired
from mimirag.index import Index, document_title
from mimirag.packets import LOCATOR_FORMAT


@pytest.fixture
def corpus(tmp_path, make_pdf, git_repo):
    """Text, HTML, PDF and Git sources about a fictional lighthouse, with a manifest."""
    (tmp_path / 'guide.rst').write_text(
        'Lantern care\n============\n\nClean the lantern glass every morning.\n\n'
        'Trim the wick before dusk; a long wick smokes the glass.\n'
    )
    (tmp_path / 'log.html').write_text('<h1>Keeper log</h1><p>The fog bell rang twice at noon.</p>')
    make_pdf(
        'manual.pdf',
        [
            ['Keeper manual', 'The lens turns once a minute.', '', 'Page 1'],
            ['Keeper manual', 'Oil the clockwork gears weekly.', '', 'Page 2'],
            ['Keeper manual', 'Record each passing ship.', '', 'Page 3'],
        ],
    )
    repo, commit = git_repo({'notes/tides.md': 'Tides\n\nHigh tide floods the lower stair.\n'})
    manifest = dict(
        documents=[
            source(tmp_path, 'guide.rst', family='guide', topic='lantern', date='2026-01-10'),
            source(tmp_path, 'log.html', kind='html', family='log', topic='fog', date='2026-02-20'),
            source(tmp_path, 'manual.pdf', kind='pdf', pages=3, citation='Keeper manual (2026)', topic='lantern'),
            dict(id='tides', kind='git', repo=repo.name, path='notes/tides.md', version=commit, ref='main'),
        ]
    )
    return tmp_path, manifest


@pytest.fixture
def index(corpus):
    root, manifest = corpus
    built = Index(root / 'out/index.sqlite3', manifest, root=root)
    built.build()
    return built


def test_build_records_sources_and_search_returns_versioned_locators(index):
    result = index.search('glass wick')
    assert result['tool'] == 'search'
    assert result['trace']['extractor'] == 'layout-v1'
    item = result['evidence'][0]
    assert item['source']['id'] == 'guide'
    assert item['source']['title'] == 'Lantern care'
    locator = item['locator']
    assert locator['format'] == LOCATOR_FORMAT
    assert 'extractor' not in locator
    assert set(locator) == {'tool', 'format', 'source', 'version', 'sha256', 'page', 'lines', 'extraction', 'passage'}
    assert item['rank'] == 1
    assert item['excerpt'] in [c['excerpt'] for c in item['context']]
    assert result['qualification'].startswith('Retrieved sources are evidence')
    with sqlite3.connect(index.path) as db:
        ids = {row[0] for row in db.execute('SELECT id FROM source')}
    assert ids == {'guide', 'log', 'manual', 'tides'}
    assert not list(index.path.parent.glob('index-*'))


def test_pdf_passages_are_page_addressed_without_banners(index):
    item = index.search('clockwork')['evidence'][0]
    assert item['locator']['page'] == 2
    assert item['source']['title'] == 'Keeper manual (2026)'
    assert 'Keeper manual' not in index.search('gears manual')['evidence'][0]['excerpt']


def test_query_text_is_never_fts_syntax_and_limits_are_bounded(index):
    for query in ['"fog', 'fog NEAR(bell)', 'fog* OR -bell', 'fog AND (']:
        assert index.search(query)['evidence']
    with pytest.raises(MimiRAGError, match='at least one word'):
        index.search('!!')
    with pytest.raises(MimiRAGError, match='limit'):
        index.search('fog', limit=21)
    result = index.search('the', limit=1)
    assert len(result['evidence']) == 1
    assert result['trace']['more_results'] is True


def test_filters(index):
    def ids(**filters):
        return {i['source']['id'] for i in index.search('the', limit=20, **filters)['evidence']}

    assert ids(source='log') == {'log'}
    assert ids(family='guide') == {'guide'}
    assert ids(topic='lantern') == {'guide', 'manual'}
    assert ids(since='2026-02-01') == {'log'}
    assert ids(until='2026-01-31') == {'guide'}
    # Other ISO forms would compare wrongly as text, so only YYYY-MM-DD is accepted.
    for bad in ['February', '20260201', '2026-W05-1', '2026-02-30']:
        with pytest.raises(MimiRAGError, match='YYYY-MM-DD'):
            ids(since=bad)


def test_resolve_verifies_every_kind_of_source(index):
    for query in ['wick', 'fog bell', 'clockwork', 'stair']:
        item = index.search(query)['evidence'][0]
        resolved = index.resolve(item['locator'])
        assert resolved['tool'] == 'resolve'
        assert resolved['trace']['extractor'] == 'layout-v1'
        assert resolved['evidence'][0]['excerpt'] == item['excerpt']
        assert resolved['evidence'][0]['original']['version'] == item['locator']['version']


def test_resolve_refuses_other_formats_and_altered_locators(index):
    locator = index.search('wick')['evidence'][0]['locator']
    with pytest.raises(LocatorError, match='Unsupported locator format'):
        index.resolve({k: v for k, v in locator.items() if k != 'format'})
    with pytest.raises(LocatorError, match='passage mismatch'):
        index.resolve(dict(locator, passage='0' * 64))
    with pytest.raises(LocatorError, match='Unavailable'):
        index.resolve(dict(locator, lines=[1, 99]))
    for lines in ['1-2', [1], [1, '2']]:
        with pytest.raises(LocatorError, match='fields'):
            index.resolve(dict(locator, lines=lines))


def test_citations_survive_rebuilds_that_add_sources(index):
    locators = [index.search(query)['evidence'][0]['locator'] for query in ['wick', 'clockwork', 'stair']]
    (index.root / 'more.txt').write_text('A second keeper joined in spring.\n')
    manifest = dict(index.manifest, documents=[*index.manifest['documents'], source(index.root, 'more.txt')])
    rebuilt = Index(index.path, manifest, root=index.root)
    rebuilt.build()
    for locator in locators:
        assert rebuilt.resolve(locator)['evidence'][0]['locator'] == locator


def test_ranking_prefers_rarer_and_more_query_words(tmp_path):
    texts = {
        'common.txt': 'the keeper and the lamp\n',
        'rare.txt': 'the keeper polished the fresnel lens\n',
        'both.txt': 'the keeper polished the fresnel lens and the lamp\n',
    }
    for name, text in texts.items():
        (tmp_path / name).write_text(text)
    index = Index(tmp_path / 'index.sqlite3', dict(documents=[source(tmp_path, n) for n in texts]), root=tmp_path)
    index.build()

    def ranked(query):
        return [item['source']['id'] for item in index.search(query)['evidence']]

    assert ranked('fresnel keeper')[-1] == 'common'  # the rare word outweighs the common one
    assert ranked('fresnel lamp')[0] == 'both'  # matching more query words ranks higher
    assert ranked('fresnel unicorn') == ['rare', 'both']  # an unmatched word does not empty the result


def test_missing_index_sources_and_malformed_locators_are_refusals(index):
    locator = index.search('wick')['evidence'][0]['locator']
    with pytest.raises(LocatorError, match='fields'):
        index.resolve(dict(locator, extra=1))
    with pytest.raises(LocatorError, match='fields'):
        index.resolve({k: v for k, v in locator.items() if k != 'source'})
    too_large = 2**63  # beyond SQLite's integers
    for malformed in [
        dict(locator, source=[]),
        dict(locator, page={}),
        dict(locator, page=too_large),
        dict(locator, lines=[1, too_large]),
    ]:
        with pytest.raises(LocatorError, match='fields'):
            index.resolve(malformed)
    with pytest.raises(MimiRAGError, match='No index'):
        Index(index.path.with_name('missing.sqlite3'), index.manifest, root=index.root).search('wick')
    (index.root / 'guide.rst').unlink()
    with pytest.raises(IntegrityError, match='missing'):
        index.search('wick')


def test_changed_sources_fail_and_touched_sources_are_rehashed(index):
    guide = index.root / 'guide.rst'
    guide.write_bytes(guide.read_bytes())
    assert 'rehashed' in index.search('wick')['trace']['notices'][0]
    guide.write_text('changed')
    with pytest.raises(IntegrityError, match='hash mismatch'):
        index.search('wick')


def test_changed_manifest_and_moved_ref_fail(index, git_repo):
    fewer = dict(index.manifest, documents=index.manifest['documents'][:1])
    with pytest.raises(RebuildRequired, match='Manifest changed'):
        Index(index.path, fewer, root=index.root).search('wick')
    git_repo({'notes/tides.md': 'changed\n'}, index.root / 'repo')
    with pytest.raises(IntegrityError, match='no longer points'):
        index.search('wick')


def test_pdf_resolution_rechecks_the_cited_page_and_the_tools(index, monkeypatch):
    locator = index.search('clockwork')['evidence'][0]['locator']
    original = index_module.extractor_identity
    monkeypatch.setattr(index_module, 'extractor_identity', lambda ocr=None: dict(original(ocr), pdf_utility='other'))
    with pytest.raises(RebuildRequired, match='PDF utility or OCR command changed'):
        index.resolve(locator)
    monkeypatch.undo()
    monkeypatch.setattr(index_module, 'extract_page', lambda *args: 'different text')
    with pytest.raises(RebuildRequired, match='Extraction output changed'):
        index.resolve(locator)


def test_text_resolution_rechecks_the_whole_extraction(index, monkeypatch):
    locator = index.search('wick')['evidence'][0]['locator']
    monkeypatch.setattr(index_module, 'extract_document', lambda *args: ([Page(None, 'different text')], None))
    with pytest.raises(RebuildRequired, match='Extraction output changed'):
        index.resolve(locator)


def test_ocr_pages_are_indexed_and_resolved(tmp_path, make_pdf, ocr):
    make_pdf('scan.pdf', [['Printed page'], []])
    manifest = dict(documents=[source(tmp_path, 'scan.pdf', kind='pdf', ocr_pages=[2])])
    path = tmp_path / 'index.sqlite3'
    Index(path, manifest, root=tmp_path, ocr=ocr).build()
    item = Index(path, manifest, root=tmp_path).search('Recognized')['evidence'][0]
    assert item['locator']['page'] == 2
    assert Index(path, manifest, root=tmp_path, ocr=ocr).resolve(item['locator'])['evidence']
    with pytest.raises(MimiRAGError, match='built with OCR settings'):
        Index(path, manifest, root=tmp_path).resolve(item['locator'])
    other = dict(ocr, dpi=30)
    with pytest.raises(RebuildRequired, match='OCR command changed'):
        Index(path, manifest, root=tmp_path, ocr=other).resolve(item['locator'])


def test_oversized_blocks_are_qualified_and_truncated(tmp_path):
    (tmp_path / 'long.txt').write_text(' '.join(['word'] * 2000) + '\n\nshort tail\n')
    manifest = dict(documents=[source(tmp_path, 'long.txt', qualification='Draft text')])
    index = Index(tmp_path / 'index.sqlite3', manifest, root=tmp_path)
    index.build()
    item = index.search('word')['evidence'][0]
    assert item['truncated']
    assert len(item['excerpt']) == 8000
    expected = 'Draft text; oversized structural unit: inspect original; search excerpt truncated'
    assert item['qualification'] == expected
    assert index.resolve(item['locator'])['evidence']
    tail = index.search('tail')['evidence'][0]
    assert [c['lines'] for c in tail['context']] == [[1, 1], [3, 3]]
    assert [c['truncated'] for c in tail['context']] == [True, False]


def test_failed_build_keeps_the_previous_index(index):
    before = index.path.read_bytes()
    (index.root / 'guide.rst').write_text('changed')
    with pytest.raises(IntegrityError, match='hash mismatch'):
        index.build()
    assert index.path.read_bytes() == before
    assert not list(index.path.parent.glob('index-*'))


class Admission:
    """Deployment hooks that add one source, store metadata and exclude a superseded source."""

    def __init__(self, root):
        (root / 'extra.txt').write_text('The spare lamp sits in the cellar.\n')
        self.extra = source(root, 'extra.txt')
        self.calls, self.veto = [], False

    def prepare(self, manifest, previous):
        self.calls.append(('prepare', previous))
        return [self.extra], dict(round=(previous or {}).get('round', 0) + 1)

    def before_publish(self, manifest):
        if self.veto:
            raise MimiRAGError('vetoed')

    def state(self, manifest, metadata, indexed):
        assert 'extra' in indexed
        return {'guide': 'extra'}, [f'round {metadata["round"]}']


def test_admission_adds_sources_stores_metadata_and_excludes(corpus):
    root, manifest = corpus
    admission = Admission(root)
    index = Index(root / 'index.sqlite3', manifest, root=root, admission=admission)
    index.build()
    index.build()
    assert admission.calls == [('prepare', None), ('prepare', {'round': 1})]
    result = index.search('lamp glass')
    assert {i['source']['id'] for i in result['evidence']} == {'extra'}
    assert result['trace']['notices'] == ['round 2']
    unfiltered = Index(index.path, manifest, root=root)
    locator = unfiltered.search('wick')['evidence'][0]['locator']
    with pytest.raises(LocatorError, match='successor: extra'):
        index.resolve(locator)
    admission.veto = True
    before = index.path.read_bytes()
    with pytest.raises(MimiRAGError, match='vetoed'):
        index.build()
    assert index.path.read_bytes() == before
    admission.veto, admission.extra = False, dict(admission.extra, id='guide')
    with pytest.raises(MimiRAGError, match='Duplicate source IDs'):
        index.build()
    assert index.path.read_bytes() == before


def test_indexes_with_another_schema_are_refused(tmp_path):
    path = tmp_path / 'old.sqlite3'
    with sqlite3.connect(path) as db:
        db.executescript('CREATE TABLE source(id TEXT); CREATE TABLE metadata(key TEXT, value TEXT);')
    with pytest.raises(RebuildRequired, match='Unsupported index schema'):
        Index(path, dict(documents=[]), root=tmp_path).search('x')


def test_document_titles_are_declared_or_from_markup():
    assert document_title(dict(id='a', kind='pdf', title='Declared'), []) == 'Declared'
    assert document_title(dict(id='a', kind='pdf'), [Page(1, 'Big Heading\n')]) == 'a'
    html = [Page(None, '\nFirst line\nmore')]
    assert document_title(dict(id='a', kind='html'), html) == 'First line'
    assert document_title(dict(id='a', kind='text'), [Page(None, 'Intro\n-----\nbody')]) == 'Intro'
    assert document_title(dict(id='a', kind='text'), [Page(None, 'Intro\n--\nbody')]) == 'a'


def test_index_identity_depends_on_content_not_build_time(corpus):
    root, manifest = corpus
    first = Index(root / 'a.sqlite3', manifest, root=root).build()
    second = Index(root / 'b.sqlite3', manifest, root=root).build()
    assert first['identity'] == second['identity']
    (root / 'guide.rst').write_text('Other text\n')
    manifest['documents'][0] = source(root, 'guide.rst', family='guide', topic='lantern', date='2026-01-10')
    assert Index(root / 'c.sqlite3', manifest, root=root).build()['identity'] != first['identity']
    with sqlite3.connect(root / 'a.sqlite3') as db:
        row = db.execute("SELECT value FROM metadata WHERE key='extractor'").fetchone()
    extractor = json.loads(row[0])
    assert extractor['name'] == 'layout-v1'
    assert extractor['pdf_utility'].startswith('pdftotext')


def test_a_failing_pdf_reader_is_a_refusal(tmp_path, monkeypatch):
    tools = tmp_path / 'bin'
    tools.mkdir()
    (tools / 'pdftotext').write_text('#!/bin/sh\nexit 3\n')
    (tools / 'pdftotext').chmod(0o755)
    monkeypatch.setenv('PATH', f'{tools}{os.pathsep}{os.environ["PATH"]}')
    with pytest.raises(MimiRAGError, match='Poppler'):
        index_module.extractor_identity()
