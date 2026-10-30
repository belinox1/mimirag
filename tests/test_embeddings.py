from contextlib import nullcontext
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from mimirag.embeddings import OfflineEmbedder, load_vectors, normalize, save_vectors
from mimirag.errors import MimiRAGError
from mimirag.integrity import fingerprint


@pytest.fixture
def receipt():
    return dict(inventory='lamp', model='local', tokenizer='local', runtime='fixture', dimensions=2, synthetic=True)


@pytest.mark.parametrize(
    ('field', 'value'),
    [
        ('dimensions', True),
        ('dimensions', 1.0),
        ('dimensions', 0),
        ('synthetic', 'false'),
        ('synthetic', 1),
        ('inventory', None),
        ('model', {}),
        ('tokenizer', ''),
        ('runtime', None),
        ('runtime', {'value': float('nan')}),
    ],
)
def test_malformed_receipts_preserve_existing_artifact(tmp_path, receipt, field, value):
    path = tmp_path / 'vectors.json'
    vectors = {'one': [0.6, 0.8]}
    save_vectors(path, vectors, receipt)
    before = path.read_bytes()
    invalid = dict(receipt, **{field: value})
    with pytest.raises(MimiRAGError):
        save_vectors(path, vectors, invalid)
    assert path.read_bytes() == before
    payload = dict(format=1, receipt=invalid, vectors=vectors)
    path.write_text(json.dumps(dict(payload=payload, sha256=fingerprint(payload))))
    with pytest.raises(MimiRAGError):
        load_vectors(path, receipt, {'one'})


@pytest.mark.parametrize('content', ['{', '[]', '{}', '{"payload": []}', '{"payload": {}, "payload": {}}'])
def test_malformed_vector_envelopes(tmp_path, receipt, content):
    path = tmp_path / 'vectors.json'
    path.write_text(content)
    with pytest.raises(MimiRAGError):
        load_vectors(path, receipt, {'one'})


@pytest.mark.parametrize(
    'vectors',
    [
        [],
        {'one': None},
        {'one': ['x', 0]},
        {'one': [True, 0]},
        {'one': [1]},
        {'one': [float('inf'), 0]},
        {'one': [0, 0]},
    ],
)
def test_malformed_vector_payload(tmp_path, receipt, vectors):
    payload = dict(format=1, receipt=receipt, vectors=vectors)
    path = tmp_path / 'vectors.json'
    path.write_text(json.dumps(dict(payload=payload, sha256=fingerprint(payload))))
    with pytest.raises(MimiRAGError):
        load_vectors(path, receipt, {'one'})


def test_failed_atomic_replacement_keeps_old_artifact(tmp_path, receipt, monkeypatch):
    path = tmp_path / 'vectors.json'
    save_vectors(path, {'one': [1.0, 0.0]}, receipt)
    before = path.read_bytes()

    def fail_replace(self, target):
        raise OSError('replacement refused')

    monkeypatch.setattr(Path, 'replace', fail_replace)
    with pytest.raises(OSError, match='replacement refused'):
        save_vectors(path, {'one': [0.0, 1.0]}, receipt)
    assert path.read_bytes() == before
    assert list(tmp_path.iterdir()) == [path]


def test_empty_vector_inventory_and_wrong_ids(tmp_path, receipt):
    path = tmp_path / 'vectors.json'
    save_vectors(path, {}, receipt)
    assert load_vectors(path, receipt, set()) == {}
    with pytest.raises(MimiRAGError):
        load_vectors(path, receipt, {'missing'})


def test_receipt_json_identity_distinguishes_boolean_and_number(tmp_path, receipt):
    path = tmp_path / 'vectors.json'
    receipt = dict(receipt, runtime={'flag': True})
    save_vectors(path, {'one': [1.0, 0.0]}, receipt)
    with pytest.raises(MimiRAGError):
        load_vectors(path, dict(receipt, runtime={'flag': 1}), {'one'})


def test_receipt_json_roundtrip_accepts_tuple_as_array(tmp_path, receipt):
    path = tmp_path / 'vectors.json'
    receipt = dict(receipt, runtime={'versions': ('one', 'two')})
    vectors = {'one': [1.0, 0.0]}
    save_vectors(path, vectors, receipt)
    assert load_vectors(path, receipt, {'one'}) == vectors


def test_atomic_vectors_validate_identity_and_corruption(tmp_path):
    path = tmp_path / 'vectors.json'
    receipt = dict(inventory='lamp', model='local', tokenizer='local', runtime='fixture', dimensions=2, synthetic=True)
    vectors = {'one': normalize([3, 4])}
    save_vectors(path, vectors, receipt)
    assert load_vectors(path, receipt, {'one'}) == vectors
    before = path.read_bytes()
    with pytest.raises(MimiRAGError):
        save_vectors(path, {'one': [0, 0]}, receipt)
    assert path.read_bytes() == before
    with pytest.raises(MimiRAGError):
        load_vectors(path, dict(receipt, inventory='other'), {'one'})
    path.write_text(path.read_text().replace('0.6', '0.7'))
    with pytest.raises(MimiRAGError):
        load_vectors(path, receipt, {'one'})


class Positions:
    def __init__(self, positions=(0, 1, 2)):
        self.positions = positions

    def flatten(self):
        return self

    def numel(self):
        return len(self.positions)

    def __getitem__(self, index):
        return self.positions[index]


class Mask:
    def __init__(self, positions=(0, 1, 2)):
        self.positions = positions

    def __getitem__(self, index):
        return self

    def nonzero(self):
        return Positions(self.positions)


class Hidden:
    def __getitem__(self, position):
        assert position == (0, 2)
        return self

    def float(self):
        return self

    def tolist(self):
        return [3, 4]


@pytest.mark.parametrize('positions', [(0, 1, 2), (1, 2), (2,)])
def test_injected_embedding_pooling_prefix_and_token_cap(positions):
    texts = []

    def tokenizer(text, **kwargs):
        texts.append(text)
        assert kwargs['truncation'] is False
        return dict(input_ids=SimpleNamespace(shape=(1, len(text))), attention_mask=Mask(positions))

    model = SimpleNamespace(eval=lambda: None)

    class Model:
        def eval(self):
            model.eval()

        def __call__(self, **encoded):
            return SimpleNamespace(last_hidden_state=Hidden())

    backend = OfflineEmbedder(
        tokenizer,
        Model(),
        SimpleNamespace(inference_mode=nullcontext),
        dimensions=2,
        query_prefix='Find: ',
        max_tokens=20,
    )
    assert backend.embed('lamp', query=True) == [0.6, 0.8]
    assert texts == ['Find: lamp']
    with pytest.raises(MimiRAGError):
        backend.embed('x' * 21)


def test_local_assets_reject_before_optional_imports(tmp_path):
    with pytest.raises(MimiRAGError):
        OfflineEmbedder.from_local(tmp_path, files={}, dimensions=2, query_prefix='')


@pytest.mark.parametrize('failure', ['hash', 'extra', 'link', 'directory_link'])
def test_local_asset_integrity_before_imports(tmp_path, monkeypatch, failure):
    (tmp_path / 'model.bin').write_bytes(b'authored asset placeholder')
    files = {'model.bin': hashlib.sha256((tmp_path / 'model.bin').read_bytes()).hexdigest()}
    if failure == 'hash':
        files['model.bin'] = '0' * 64
    elif failure == 'extra':
        (tmp_path / 'extra.txt').write_text('unexpected')
    elif failure == 'link':
        (tmp_path / 'model.bin').rename(tmp_path / 'target.bin')
        (tmp_path / 'model.bin').symlink_to(tmp_path / 'target.bin')
    else:
        (tmp_path / 'empty').mkdir()
        (tmp_path / 'linked').symlink_to(tmp_path / 'empty', target_is_directory=True)

    def no_import(name):
        pytest.fail(f'Optional import before validation: {name}')

    monkeypatch.setattr('mimirag.embeddings.importlib.import_module', no_import)
    with pytest.raises(MimiRAGError):
        OfflineEmbedder.from_local(tmp_path, files=files, dimensions=2, query_prefix='')


def test_empty_attention_mask_refused():
    backend = OfflineEmbedder(
        lambda *a, **k: dict(input_ids=SimpleNamespace(shape=(1, 1)), attention_mask=Mask(())),
        SimpleNamespace(eval=lambda: None),
        SimpleNamespace(inference_mode=nullcontext),
        dimensions=2,
        query_prefix='',
    )
    backend.model = lambda **k: SimpleNamespace(last_hidden_state=Hidden())
    with pytest.raises(MimiRAGError, match='no tokens'):
        backend.embed('lantern')
