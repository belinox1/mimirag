# MimiRAG

[![CI](https://github.com/belinox1/mimirag/actions/workflows/ci.yml/badge.svg)](https://github.com/belinox1/mimirag/actions/workflows/ci.yml)
![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue)
[![License: Apache-2.0](https://img.shields.io/badge/license-Apache--2.0-blue)](LICENSE)

Retrieval with verifiable citations over documents and source code.

MimiRAG is a Python library for the evidence layer of retrieval-augmented
generation (RAG). It retrieves passages with content-addressed citations,
then verifies cited text against the original source bytes.

The document index is a single SQLite file. The library runs offline, with
no Python runtime dependencies beyond the standard library.

## What it includes

- **Document search:** BM25 over text, Markdown, HTML and PDFs, with metadata
  and date filters.
- **Citation verification:** source versions, page and line locations, and
  content hashes. Resolution re-extracts the source and rejects mismatches.
- **Code retrieval:** literal search and bounded file reads at a pinned Git
  commit, within declared path prefixes.
- **Retrieval evaluation:** gold-quote alignment, evidence coverage under
  rank or character budgets, and paired comparisons of ranking variants.

**Pre-alpha:** document retrieval, citation verification, pinned code reads
and retrieval evaluation work today. APIs may change. An MCP server, hybrid
retrieval and answer generation are on the [roadmap](docs/reference.md#roadmap).

## Quick start

You'll need Python 3.11+ with SQLite FTS5, Git and
[Poppler](https://poppler.freedesktop.org/) (`pdftotext`, `pdftoppm`).
Install Poppler with `brew install poppler` on macOS or
`sudo apt-get install poppler-utils` on Debian/Ubuntu. It's currently required
even for text-only index builds.

```sh
git clone https://github.com/belinox1/mimirag.git
cd mimirag
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .
```

Save this as `try_mimirag.py` and run `python try_mimirag.py`. It creates a tiny
lighthouse manual in a temporary directory, searches it, and verifies the citation:

```python
from hashlib import sha256
from pathlib import Path
from tempfile import TemporaryDirectory

from mimirag import Index

with TemporaryDirectory() as folder:
    root = Path(folder)
    manual = root / 'manual.txt'
    manual.write_text('If the lantern flickers, clean the glass before blaming ghosts.\n', encoding='utf-8')
    manifest = {
        'documents': [
            {
                'id': 'keeper-notes',
                'kind': 'text',
                'path': 'manual.txt',
                'version': '1',
                'sha256': sha256(manual.read_bytes()).hexdigest(),
            }
        ]
    }

    index = Index(root / 'index.sqlite3', manifest, root=root)
    index.build()
    hit = index.search('lantern glass')['evidence'][0]
    print(hit['excerpt'])

    verified = index.resolve(hit['locator'])
    print('Verified:', verified['evidence'][0]['locator']['source'])
```

```text
If the lantern flickers, clean the glass before blaming ghosts.
Verified: keeper-notes
```

**Next experiment:** add `manual.write_text('Actually, it was ghosts.')` just
before `index.resolve(...)` and run it again. Verification refuses the changed
source. To index an intentional edit, update its manifest hash and rebuild.

The [usage guide](docs/reference.md#usage) covers manifests for your own files,
metadata filters, PDFs, OCR and code search.

## Documentation

- [Retrieval internals](docs/reference.md#how-retrieval-works): extraction, chunks,
  ranking and known limits.
- [Evaluation](docs/reference.md#evaluation): gold quotes, evidence
  coverage and paired comparisons.
- [Development](docs/reference.md#development): local checks and tests.
- [Roadmap](docs/reference.md#roadmap): planned work.

The name comes from Mímir, the Norse guardian of the well of wisdom whom Odin
consulted for counsel.

Apache-2.0 · [Content and licensing](docs/reference.md#content-and-licensing)
