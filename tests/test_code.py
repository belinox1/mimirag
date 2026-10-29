from helpers import run_git
import pytest

from mimirag.code import CodeRepository, in_scope
from mimirag.errors import IntegrityError, MimiRAGError


@pytest.fixture
def repository(tmp_path, git_repo):
    long_file = ''.join(f'line {i}\n' for i in range(1, 201))
    path, commit = git_repo(
        {
            'src/pump.py': '"""Pump control."""\n\ndef start_pump(rate):\n    return rate * 2\n',
            'src/valve.py': 'def open_valve():\n    return start_pump(1)\n' + '\n' * 30 + 'start_pump(2)\n',
            'src/long.txt': long_file,
            'src/image.bin': b'\x00\x01start_pump',
            'src/latin.txt': 'start_pump caf\xe9'.encode('latin-1'),
            'src/start_pump_notes.md': 'Notes only.\n',
            'src/data.csv': 'start_pump\n',
            'private/secret.py': 'start_pump = "hidden"\n',
        }
    )
    run_git(path, 'tag', 'v1')
    spec = dict(repo=path.name, version=commit, ref='v1', scope=['src/'])
    return CodeRepository(tmp_path, spec, qualification='Fixture code')


def test_literal_search_returns_the_first_match_per_file_in_path_order(repository):
    result = repository.search('START_PUMP')
    paths = [item['source'] for item in result['evidence']]
    assert paths == ['src/pump.py', 'src/start_pump_notes.md', 'src/valve.py']
    first = result['evidence'][0]
    assert first['locator']['lines'] == [1, 4]
    assert first['locator']['path'] == 'src/pump.py'
    assert first['locator']['version'] == repository.spec['version']
    assert first['qualification'] == 'Fixture code'
    valve = result['evidence'][2]
    assert valve['locator']['lines'] == [1, 17]  # around line 2, not the later call on line 33
    assert result['trace']['order'].startswith('path order')
    assert result['trace']['scanned_files'] == 3
    assert result['qualification'].startswith('Retrieved sources are evidence')


def test_a_match_far_into_a_file_is_returned_with_context(repository, git_repo):
    later = 'a = 1\n' * 40 + 'target_call()\n' + 'b = 2\n' * 40
    git_repo({'src/later.py': later}, repository.repo)
    run_git(repository.repo, 'tag', '-f', 'v1')
    commit = run_git(repository.repo, 'rev-parse', 'HEAD')
    moved = CodeRepository(repository.repo.parent, dict(repository.spec, version=commit), qualification='Fixture code')
    item = moved.search('target_call')['evidence'][0]
    assert item['locator']['lines'] == [36, 56]
    assert item['excerpt'].splitlines()[5] == 'target_call()'


def test_files_that_cannot_be_read_are_skipped(repository, git_repo):
    (repository.repo / 'src/start_pump_link.py').symlink_to('pump.py')
    git_repo({}, repository.repo)
    run_git(repository.repo, 'tag', '-f', 'v1')
    commit = run_git(repository.repo, 'rev-parse', 'HEAD')
    moved = CodeRepository(repository.repo.parent, dict(repository.spec, version=commit), qualification='Fixture code')
    result = moved.search('start_pump')
    assert 'src/start_pump_link.py' not in [item['source'] for item in result['evidence']]
    assert result['trace']['scanned_files'] == 3


def test_queries_are_literal_and_bounded(repository):
    assert repository.search('rate * 2')['evidence'][0]['source'] == 'src/pump.py'
    assert repository.search('rate . 2')['evidence'] == []
    result = repository.search('start_pump', limit=1)
    assert len(result['evidence']) == 1
    assert result['trace']['more_results'] is True
    for bad in ['x' * 257, '']:
        with pytest.raises(MimiRAGError, match=r'[Qq]uery'):
            repository.search(bad)
    with pytest.raises(MimiRAGError, match='limit'):
        repository.search('pump', limit=0)


def test_reads_are_windowed_and_confined_to_the_scope(repository):
    item = repository.read('src/long.txt')['evidence'][0]
    assert item['locator']['lines'] == [1, 60]
    assert item['excerpt'].endswith('line 60')
    item = repository.read('src/long.txt', first=150, last=160)['evidence'][0]
    assert item['excerpt'].splitlines() == [f'line {i}' for i in range(150, 161)]
    for first, last in [(1, 121), (190, 210)]:
        with pytest.raises(MimiRAGError, match=r'1\.\.120'):
            repository.read('src/long.txt', first=first, last=last)
    for bad in ['private/secret.py', 'src/missing.py']:
        with pytest.raises(MimiRAGError, match='outside allowlisted'):
            repository.read(bad)
    with pytest.raises(MimiRAGError, match='Invalid source path'):
        repository.read('src/../private/secret.py')


def test_only_the_declared_commit_is_readable(repository, git_repo):
    with pytest.raises(MimiRAGError, match='declared code version'):
        repository.search('pump', version='0' * 40)
    git_repo({'src/pump.py': 'changed\n'}, repository.repo)
    run_git(repository.repo, 'tag', '-f', 'v1')
    with pytest.raises(IntegrityError, match='no longer points'):
        repository.search('pump')


def test_scope_matches_whole_path_components():
    assert in_scope('src/a.py', ['src/'])
    assert in_scope('src', ['src/'])
    assert not in_scope('srcx/a.py', ['src'])
    with pytest.raises(MimiRAGError):
        in_scope('/src/a.py', ['src'])


def test_search_never_leaves_the_scope(tmp_path, repository):
    unscoped = CodeRepository(tmp_path, dict(repository.spec, scope=[]), qualification='')
    assert unscoped.search('start_pump')['evidence'] == []


def test_unusual_file_names_are_found_and_read(tmp_path, git_repo):
    repo, commit = git_repo({'src/caf\u00e9 notes.py': 'brew = 1\n'})
    code = CodeRepository(tmp_path, dict(repo=repo.name, version=commit, scope=['src']), qualification='')
    assert [item['source'] for item in code.search('brew')['evidence']] == ['src/caf\u00e9 notes.py']
    assert code.read('src/caf\u00e9 notes.py')['evidence'][0]['excerpt'] == 'brew = 1'
