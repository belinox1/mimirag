"""The package-level API, as the README and the package docstring show it."""

import json
from pathlib import Path

from helpers import source
import pytest

import mimirag


def test_the_documented_workflow(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    Path('notes.md').write_text('Lantern\n\nClean the lantern glass at dawn.\n')
    Path('manifest.json').write_text(json.dumps(dict(documents=[source('.', 'notes.md')])))

    index = mimirag.Index('index.sqlite3', mimirag.load_manifest('manifest.json'), root='.')
    assert index.build()['sources'] == 1
    packet = index.search('lantern glass')
    resolved = index.resolve(packet['evidence'][0]['locator'])

    assert resolved['evidence'][0]['excerpt'] == packet['evidence'][0]['excerpt']


@pytest.mark.parametrize('error', [mimirag.IntegrityError, mimirag.RebuildRequired, mimirag.LocatorError])
def test_the_error_classes_are_value_errors(error):
    assert issubclass(error, mimirag.MimiRAGError)
    assert issubclass(error, ValueError)
