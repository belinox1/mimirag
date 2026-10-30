from helpers import source
import pytest

from mimirag.candidates import CandidateIndex
from mimirag.errors import MimiRAGError
from mimirag.index import Index


@pytest.fixture
def many_candidates(tmp_path):
    documents = []
    for number in range(105):
        name = f'guide{number:03}.txt'
        (tmp_path / name).write_text(f'Lantern maintenance entry {number}.\n')
        documents.append(source(tmp_path, name))
    index = Index(tmp_path / 'index.sqlite', dict(documents=documents), root=tmp_path)
    index.build()
    return CandidateIndex(index)


def test_candidate_depth_scores_and_resolvable_full_text(many_candidates):
    candidate = many_candidates
    inventory = candidate.inventory()
    assert len(inventory['items']) == 105
    assert len(candidate.lexical('lantern', limit=25)['ranking']) == 25
    ranking = candidate.lexical('lantern', limit=100)
    assert len(ranking['ranking']) == 100
    assert all(score > 0 for _, score in ranking['ranking'])
    ids = [key for key, _ in ranking['ranking']]
    packet = candidate.render(ids, identity=ranking['identity'])
    assert len(packet['evidence']) == 100
    for item in packet['evidence']:
        resolved = candidate.index.resolve(item['locator'])
        assert resolved['evidence'][0]['excerpt'] == item['excerpt']
    assert candidate.lexical('missingword')['ranking'] == []
    assert candidate.render([], identity=inventory['identity'])['evidence'] == []
    for limit in (0, 101, True, 1.5):
        with pytest.raises(MimiRAGError):
            candidate.lexical('lantern', limit=limit)
    for selection in ([ids[0], ids[0]], ['unknown']):
        with pytest.raises(MimiRAGError):
            candidate.render(selection, identity=inventory['identity'])


def test_dense_invalid_vectors_and_stable_ties(many_candidates):
    candidate = many_candidates
    inventory = candidate.inventory()
    ids = list(inventory['items'])
    vectors = {key: [1.0, 0.0] for key in ids}
    result = candidate.dense([1.0, 0.0], vectors, identity=inventory['identity'])
    assert [key for key, _ in result['ranking']] == sorted(ids)[:100]
    for invalid in ([], [0, 0], [float('nan'), 0], [float('inf'), 0], [1], ['one', 0], [True, 0]):
        with pytest.raises(MimiRAGError):
            candidate.dense(
                [1.0, 0.0], dict(vectors, **{ids[0]: invalid}), identity=inventory['identity'], source='absent'
            )
    assert candidate.dense([1.0, 0.0], vectors, identity=inventory['identity'], source='absent')['ranking'] == []


@pytest.mark.parametrize('method', ['lexical', 'dense'])
@pytest.mark.parametrize('initially_excluded', [False, True])
def test_admission_transition_during_ranking_is_refused(tmp_path, monkeypatch, method, initially_excluded):
    (tmp_path / 'guide.txt').write_text('Lantern maintenance.\n')

    class Admission:
        excluded = False

        def prepare(self, manifest, previous):
            return [], None

        def before_publish(self, manifest):
            pass

        def state(self, manifest, metadata, indexed):
            return ({'guide': None} if self.excluded else {}), []

    admission = Admission()
    admission.excluded = initially_excluded
    index = Index(
        tmp_path / 'index.sqlite', dict(documents=[source(tmp_path, 'guide.txt')]), root=tmp_path, admission=admission
    )
    index.build()
    candidate = CandidateIndex(index)
    original = candidate.inventory
    inventory = original()
    vectors = {key: [1.0, 0.0] for key in inventory['items']}
    calls = 0

    def transition(**filters):
        nonlocal calls
        result = original(**filters)
        calls += 1
        if calls == 1:
            admission.excluded = not initially_excluded
        return result

    monkeypatch.setattr(candidate, 'inventory', transition)

    def rank():
        if method == 'lexical':
            return candidate.lexical('lantern')
        return candidate.dense([1.0, 0.0], vectors, identity=inventory['identity'])

    with pytest.raises(MimiRAGError, match='Inventory changed'):
        rank()
    with pytest.raises(MimiRAGError):
        candidate.render(list(vectors), identity=inventory['identity'])


def test_candidates_default_filters_and_full_packet(tmp_path):
    path = tmp_path / 'guide.txt'
    path.write_text('The lantern needs oil.\n')
    index = Index(tmp_path / 'index.sqlite', dict(documents=[source(tmp_path, 'guide.txt')]), root=tmp_path)
    index.build()
    candidates = CandidateIndex(index)
    inventory = candidates.inventory()
    ranking = candidates.lexical('lantern')
    ids = [key for key, _ in ranking['ranking']]
    assert ids == list(inventory['items'])
    packet = candidates.render(ids, identity=ranking['identity'])
    assert packet['evidence'][0]['excerpt'] == path.read_text().strip()
    assert not packet['evidence'][0]['truncated']
    assert candidates.lexical('lantern', source='absent')['ranking'] == []
    with pytest.raises(MimiRAGError):
        candidates.inventory(unknown='value')
    with pytest.raises(MimiRAGError):
        candidates.render(ids, identity='stale')


def test_dense_filters_before_selection_and_rejects_changed_source(tmp_path):
    for name in ('one.txt', 'two.txt'):
        (tmp_path / name).write_text('Lantern ' + name)
    manifest = dict(documents=[source(tmp_path, name) for name in ('one.txt', 'two.txt')])
    index = Index(tmp_path / 'index.sqlite', manifest, root=tmp_path)
    index.build()
    candidate = CandidateIndex(index)
    inventory = candidate.inventory()
    ids = list(inventory['items'])
    vectors = {ids[0]: [1.0, 0.0], ids[1]: [0.0, 1.0]}
    selected = candidate.dense([1.0, 0.0], vectors, identity=inventory['identity'], source='two', limit=1)
    assert selected['ranking'][0][0] == ids[1]
    with pytest.raises(MimiRAGError):
        candidate.dense([1.0, 0.0], {ids[0]: [1.0, 0.0]}, identity=inventory['identity'])
    (tmp_path / 'one.txt').write_text('Changed source evidence')
    with pytest.raises(MimiRAGError):
        candidate.render(ids, identity=inventory['identity'])
