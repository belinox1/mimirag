"""Test helpers: a minimal PDF writer, Git with a fixed identity and manifest entries."""

from __future__ import annotations

import os
from pathlib import Path
import subprocess

from mimirag.integrity import digest


def pdf_bytes(pages):
    """A PDF with one Helvetica text line per string.

    An empty string starts a new paragraph; an empty page has no text.
    """

    def escape(text):
        return text.replace('\\', '\\\\').replace('(', '\\(').replace(')', '\\)')

    count = len(pages)
    objects = [
        b'<< /Type /Catalog /Pages 2 0 R >>',
        (
            '<< /Type /Pages /Kids [' + ' '.join(f'{4 + 2 * i} 0 R' for i in range(count)) + f'] /Count {count} >>'
        ).encode(),
        b'<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica /Encoding /WinAnsiEncoding >>',
    ]
    for i, lines in enumerate(pages):
        ops = ['BT', '/F1 11 Tf', '14 TL', '72 740 Td']
        ops += ['T* T*' if not line else f'({escape(line)}) Tj T*' for line in lines]
        stream = '\n'.join([*ops, 'ET']).encode('cp1252')
        objects.append(
            f'<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] '
            f'/Resources << /Font << /F1 3 0 R >> >> '
            f'/Contents {5 + 2 * i} 0 R >>'.encode()
        )
        objects.append(b'<< /Length %d >>\nstream\n' % len(stream) + stream + b'\nendstream')
    out = bytearray(b'%PDF-1.4\n')
    offsets = []
    for number, body in enumerate(objects, 1):
        offsets.append(len(out))
        out += b'%d 0 obj\n' % number + body + b'\nendobj\n'
    xref = len(out)
    out += b'xref\n0 %d\n0000000000 65535 f \n' % (len(objects) + 1)
    out += b''.join(b'%010d 00000 n \n' % o for o in offsets)
    out += b'trailer\n<< /Size %d /Root 1 0 R >>\nstartxref\n%d\n%%%%EOF\n' % (len(objects) + 1, xref)
    return bytes(out)


def run_git(repo, *args):
    env = {k: v for k, v in os.environ.items() if k not in ('GIT_DIR', 'GIT_INDEX_FILE', 'GIT_WORK_TREE')}
    env.update(
        GIT_CONFIG_GLOBAL=os.devnull,
        GIT_CONFIG_NOSYSTEM='1',
        GIT_AUTHOR_DATE='2026-01-01T00:00:00Z',
        GIT_COMMITTER_DATE='2026-01-01T00:00:00Z',
    )
    return subprocess.run(
        ['git', '-C', str(repo), '-c', 'user.name=Test', '-c', 'user.email=test@example.invalid', *args],
        check=True,
        capture_output=True,
        text=True,
        env=env,
    ).stdout.strip()


def source(root, name, kind='text', **fields):
    """Manifest entry for a file below `root`, with its hash filled in."""
    data = (Path(root) / name).read_bytes()
    return dict(
        id=fields.pop('id', Path(name).stem),
        kind=kind,
        path=name,
        version=digest(data)[:12],
        sha256=digest(data),
        **fields,
    )
