"""Fixtures: generated PDFs, temporary Git repositories and a stub OCR command."""

from __future__ import annotations

import sys

from helpers import pdf_bytes, run_git
import pytest


@pytest.fixture
def make_pdf(tmp_path):
    def make(name, pages):
        path = tmp_path / name
        path.write_bytes(pdf_bytes(pages))
        return path

    return make


@pytest.fixture
def git_repo(tmp_path):
    """Commit {path: text or bytes} to a new repository; returns (path, commit).

    Pass the returned path as `repo` to add another commit.
    """

    def make(files, repo=None, message='Fixture commit'):
        if repo is None:
            repo = tmp_path / 'repo'
            repo.mkdir()
            run_git(repo, 'init', '-q', '-b', 'main')
        for name, content in files.items():
            path = repo / name
            path.parent.mkdir(parents=True, exist_ok=True)
            if isinstance(content, bytes):
                path.write_bytes(content)
            else:
                path.write_text(content)
        run_git(repo, 'add', '-A')
        run_git(repo, 'commit', '-q', '-m', message)
        return repo, run_git(repo, 'rev-parse', 'HEAD')

    return make


STUB_OCR = """
import sys
from pathlib import Path
names = Path(sys.argv[1]).read_text().split()
pages = [f'Recognized text of {Path(n).stem}\\nsecond line' for n in names]
sys.stdout.write('\\f'.join(pages) + '\\f')
"""


@pytest.fixture
def ocr(tmp_path):
    """OCR settings for a stub command that prints a fixed text per listed page image."""
    script = tmp_path / 'stub_ocr.py'
    script.write_text(STUB_OCR)
    return dict(command=[sys.executable, str(script), '{images}/list.txt'], dpi=20)
