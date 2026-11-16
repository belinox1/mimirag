import asyncio
import json
import subprocess
import sys

from helpers import source
import pytest

pytest.importorskip('mcp')
from mcp import Client, StdioServerParameters

from mimirag.candidates import CandidateIndex
from mimirag.code import CodeRepository
from mimirag.errors import MimiRAGError
from mimirag.index import Index
from mimirag.server import create_server


class Keywords:
    def embed(self, text, *, query=False):
        return [float(text.lower().count(word)) for word in ('oil', 'glass')] + [0.1]


@pytest.fixture
def index(tmp_path):
    texts = {'oil.txt': 'Pour fresh oil into the font.', 'glass.txt': 'Polish the lantern glass weekly.'}
    for name, text in texts.items():
        (tmp_path / name).write_text(text + '\n')
    built = Index(tmp_path / 'index.sqlite', dict(documents=[source(tmp_path, n) for n in texts]), root=tmp_path)
    built.build()
    return built


def call(server, *steps):
    async def run():
        async with Client(server) as client:
            return [await client.call_tool(name, arguments) for name, arguments in steps]

    return asyncio.run(run())


def tools(server):
    async def run():
        async with Client(server) as client:
            return {tool.name: tool for tool in (await client.list_tools()).tools}

    return asyncio.run(run())


def test_search_and_resolve_round_trip(index):
    server = create_server(index)
    listed = tools(server)
    assert set(listed) == {'search', 'resolve'}
    assert all(tool.annotations.read_only_hint for tool in listed.values())
    (found,) = call(server, ('search', dict(query='lantern glass')))
    assert not found.is_error
    hit = found.structured_content['evidence'][0]
    assert hit['locator']['source'] == 'glass'
    (resolved,) = call(server, ('resolve', dict(locator=hit['locator'])))
    assert resolved.structured_content['evidence'][0]['excerpt'] == hit['excerpt']


def test_refusals_are_actionable_tool_errors(index, tmp_path):
    server = create_server(index)
    hybrid, stale = call(
        server, ('search', dict(query='oil', mode='hybrid')), ('resolve', dict(locator=dict(tool='search', format=0)))
    )
    assert hybrid.is_error
    assert 'mode="lexical"' in hybrid.content[0].text
    assert stale.is_error
    assert 'LocatorError' in stale.content[0].text
    (tmp_path / 'oil.txt').write_text('Changed.\n')
    (changed,) = call(server, ('search', dict(query='oil')))
    assert changed.is_error
    assert 'Error' in changed.content[0].text


def test_hybrid_mode_and_code_tools(index, tmp_path, git_repo):
    repo, commit = git_repo({'src/lamp.py': 'def trim_wick():\n    return 1\n'})
    code = CodeRepository(tmp_path, dict(repo=repo.name, version=commit, scope=['src/']), qualification='Fixture')
    vectors = CandidateIndex(index).embed(Keywords())
    server = create_server(index, code=code, embedder=Keywords(), vectors=vectors)
    assert set(tools(server)) == {'search', 'resolve', 'code_search', 'code_read'}
    hybrid, filtered, found, read = call(
        server,
        ('search', dict(query='oil', mode='hybrid', limit=2)),
        ('search', dict(query='oil', mode='hybrid', source='glass')),
        ('code_search', dict(query='TRIM_WICK')),
        ('code_read', dict(path='src/lamp.py', first=1, last=1)),
    )
    assert [e['locator']['source'] for e in hybrid.structured_content['evidence']] == ['oil', 'glass']
    assert [e['locator']['source'] for e in filtered.structured_content['evidence']] == ['glass']
    assert found.structured_content['evidence'][0]['locator']['path'] == 'src/lamp.py'
    assert 'trim_wick' in read.structured_content['evidence'][0]['excerpt']
    with pytest.raises(MimiRAGError):
        create_server(index, embedder=Keywords())


def test_command_line_serves_over_stdio(index, tmp_path):
    manifest = tmp_path / 'manifest.json'
    manifest.write_text(json.dumps(index.manifest))
    command = StdioServerParameters(
        command=sys.executable,
        args=['-m', 'mimirag.server', '--manifest', str(manifest), '--index', str(index.path), '--root', str(tmp_path)],
    )
    (found,) = call(command, ('search', dict(query='font')))
    assert found.structured_content['evidence'][0]['locator']['source'] == 'oil'
    help_text = subprocess.run(
        [sys.executable, '-m', 'mimirag.server', '--help'], capture_output=True, text=True, check=True
    ).stdout
    assert '--manifest' in help_text
