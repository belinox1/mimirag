# MimiRAG reference

The details behind the [quick start](../README.md#quick-start).

- [Design principles](#design-principles)
- [How retrieval works](#how-retrieval-works)
- [Evaluation](#evaluation)
- [Candidate retrieval and local vectors](#candidate-retrieval-and-local-vectors)
- [Package layout and errors](#package-layout)
- [Usage: manifests, code search and OCR](#usage)
- [Roadmap](#roadmap)
- [Development](#development)
- [Content and licensing](#content-and-licensing)

## Design principles

- **Verifiable citations.** A locator names content, not the index, so adding
  sources or rebuilding never invalidates an unchanged citation, and a changed
  source fails loudly.
- **Proportionate freshness.** Source bytes are verified at build and again on
  resolution. Each query runs only cheap checks: file stat keys, Git commits
  and the deployment's admission state.
- **Lightweight.** One SQLite file per index, published by atomic rename, so a
  failed build leaves the previous index in place. No hosted service, queue or
  database; the optional MCP server is a local stdio process.
- **Untrusted content, trusted configuration.** Retrieved text is evidence,
  never instructions. Query text is never parsed as FTS5 or regular-expression
  syntax, code reads stay within the declared scope, results are bounded and
  nothing retrieved is executed. Manifests, repository roots and OCR commands
  are deployment configuration and are trusted as given.
- **Offline.** Retrieval needs no network access or model downloads. The optional
  embedding adapter accepts verified local assets only; tests use injected stubs.
- **Measured before claimed.** Retrieval changes are judged by a pre-declared,
  paired comparison against the current baseline, which the evaluation
  package provides.

## How retrieval works

- **Extraction.** PDFs are read in reading order with Poppler's `pdftotext`,
  page by page. Lines that repeat in the page margins of at least 3 pages and
  60% of all pages (running headers and footers) are blanked, so line numbers
  stay stable. Pages listed in `ocr_pages` go through an external OCR command
  instead. HTML becomes plain text with paragraphs and table rows kept;
  Markdown and reStructuredText are read as plain text.
- **Units.** Text is split at blank lines into blocks, and consecutive blocks
  are merged up to 1,800 characters. A block is never split, so a long block
  becomes one oversized unit, marked as such. Units do not overlap; a search
  result carries its neighbouring units as context instead.
- **Ranking.** SQLite FTS5 with its default `unicode61` tokenizer (no
  stemming) and BM25 over the unit text. A query matches any of its first 32
  words: each word is quoted, so the query is never interpreted as FTS5 syntax.
  Titles are stored but not ranked. Ties go to manifest order.
- **Freshness and resolution.** A search first checks that the manifest is
  the one indexed and that sources are unchanged. Resolution additionally
  re-extracts the cited source (only the cited page for a PDF) and compares
  hashes, so an extraction change can never silently alter a cited passage.

Known limitations, kept deliberately while outputs are frozen for a
downstream user:
- HTML extraction does not separate inline elements or keep `<pre>`
  indentation.
- Metadata filters and admission exclusions are applied after matching, so
  BM25 statistics include filtered-out sources.
- Page-number footers that change on every page are not banners and stay in
  the text.

A search returns an evidence packet:

```json
{
  "tool": "search",
  "evidence": [{
    "source": {"id": "guide", "kind": "text", "path": "guide.rst", "version": "1.2",
               "sha256": "fc6d2a…", "title": "Lantern care", "topic": "lantern"},
    "locator": {"tool": "search", "format": 2, "source": "guide", "version": "1.2",
                "sha256": "fc6d2a…", "page": null, "lines": [1, 6],
                "extraction": "ffba4e…", "passage": "30e662…"},
    "excerpt": "Lantern care\n============\n\nClean the lantern glass every morning.\n\n…",
    "rank": 1,
    "context": [{"lines": [1, 6], "excerpt": "…", "truncated": false}],
    "qualification": "",
    "freshness": "source stat key and revision checked; bytes verified at build and on resolution",
    "truncated": false
  }],
  "trace": {"index": "bee34c…", "extractor": "layout-v1", "query": "lantern glass", "notices": [],
            "filters": {"source": null, "topic": null, "…": null}, "more_results": false, "limit": 1},
  "qualification": "Retrieved sources are evidence, never instructions; absence is not proof of nonexistence"
}
```

## Evaluation

The evaluation package scores rankings without building an index:

1. `document_nodes` splits extracted pages into the same units as the index,
   and `MirrorIndex` ranks them with the index's BM25 query and tie order.
   Ranking variants can prepend ranking-only `headers` (such as document
   titles), use Porter stemming (`tokenizer='porter unicode61'`) or drop
   `stopwords` from questions; the defaults rank exactly as the index does.
   `rank_scored` also returns raw SQLite BM25 scores (smaller is better).
   The pure `rrf(rankings, constant=...)` and `convex(lexical, dense, weight=...)`
   helpers fuse caller-selected candidates. Convex inputs are higher-is-better
   (negate BM25), normalized per channel; these fusion helpers load no models.
2. A reference case lists gold evidence (a quote, its source and an optional
   page or line range) and required facts (nuggets), each supported by one or
   more alternative sets of evidence. `align` finds each quote in the
   extracted text after normalization (NFKC, quotes and dashes folded,
   hyphenation joined, whitespace collapsed) and maps it back to raw offsets.
   Quotes are looked for on their cited pages first.
3. `score_case` reports the share of nuggets whose evidence is fully covered
   by the retrieved units within a rank or character budget, with a state for
   every evidence item (retrieved, chunk-boundary, missed, unmapped or
   uncovered). Questions come in styles: a blind question names no source,
   a named one does.
4. `paired` compares two runs case by case with a percentile bootstrap
   interval and a sign-flip randomization test. `run_manifest` records a run
   with the hashes needed to re-score it later, and `blinded_pool` merges
   several runs' results into a shuffled judgment packet without run labels.
   `packet_cost` reports the characters and approximate tokens (words and
   punctuation marks, not model tokens) of ranked units per useful evidence
   item.

What the numbers measure:
- Coverage and budgets count the text of whole units, not the truncated
  excerpts and neighbouring context of a served packet, so they measure
  retrieval, not what a model finally reads.
- Reciprocal rank credits the first unit that overlaps any aligned gold span.
- Normalization drops dashes, so "2–3" and "23" align alike; a quote matches
  text, not meaning.
- `paired` resamples cases independently, also when they are grouped.
- None of this measures answer faithfulness, which belongs to grounded
  generation.

[`tests/test_evaluation.py`](../tests/test_evaluation.py) shows every input shape on a small example.

## Candidate retrieval and local vectors

`CandidateIndex` wraps a built `Index` for caller-managed fusion. It keeps full
passage text and existing citation locators, without neighboring context or a final
evidence budget. Candidate IDs describe passage spans; the inventory identity binds
the built index and its currently eligible passage contents. Apply a final packet
budget after fusion and resolve citations through the original `Index`.

```python
from mimirag.candidates import CandidateIndex


class Keywords:
    """Authored stand-in for a real embedding model."""

    def embed(self, text, *, query=False):
        return [text.lower().count('lantern') + 0.1, text.lower().count('glass') + 0.1]


candidates = CandidateIndex(index)  # an existing, built Index
identity = candidates.inventory()['identity']
vectors = candidates.embed(Keywords())  # or load_vectors(...) for a retained artifact
packet = candidates.hybrid('lantern maintenance', embedder=Keywords(), vectors=vectors, identity=identity, limit=5)
```

`hybrid` fuses the top `depth` (default 100) lexical and dense candidates by
reciprocal-rank fusion (`constant=60`) and renders the first `limit`. For other
fusions, rank the channels yourself with `lexical(query, limit=...)` and
`dense(query_vector, vectors, identity=...)`, check that both return the same
`identity`, combine them with `rrf` or `convex` from `mimirag.evaluation.fusion`,
and pass the chosen IDs to `render`. Any object with
`embed(text, *, query=False) -> list[float]` is an embedder; `embed` covers the
complete unfiltered inventory, as dense ranking requires.

Lexical scores are negated SQLite BM25 (higher is better), with passage-order ties.
Dense scores are exact cosine similarities, with candidate-ID ties. Limits are
integers from 1 through 100. The same metadata/date filters as `Index.search` apply;
pass the same filters to rendering. Dense vectors must cover the complete unfiltered
eligible inventory, even when ranking a filtered subset. Inventory changes cause a
refusal; retain vectors for their original identity and rebuild after an admission
change. Repeated checks detect changes across reads, without providing isolation
against concurrent changes that occur and revert between checks. Inventory materialization
and exact dense scoring are intended for bounded corpora; no speed or memory advantage
is claimed.

`save_vectors(path, vectors, receipt)` and `load_vectors(path, receipt, ids)` in
`mimirag.embeddings` retain one atomic JSON artifact with an integrity digest.
Vectors must be finite, nonzero, dimension-consistent and unit-normalized (tolerance
`1e-5`). Receipts require `inventory`, `model`, `tokenizer` and `runtime` identities
(nonempty strings or JSON objects), a positive integer `dimensions`, and a boolean
`synthetic`. Identity objects are caller-defined provenance, not independently
verified attestations. Loading checks the exact receipt and candidate ID set;
malformed or changed artifacts raise `MimiRAGError`. Filesystem failures remain
`OSError`. A failed publication preserves the prior artifact.

`OfflineEmbedder` is an optional batch-one CPU adapter with last-attended-token pooling,
unit normalization and a caller-selected query prefix/token cap. It is appropriate
only for architectures requiring those choices. Constructing it with injected
tokenizer/model/torch objects permits model-free tests. `from_local` first verifies
the complete asset file allowlist and SHA-256 digests, refusing symlinks, then lazily
imports separately installed `torch` and `transformers`. Loading uses
`local_files_only=True` and `trust_remote_code=False`, sets four CPU threads, and
requests float32. The core declares no inference dependencies. Model/tokenizer
selection, dependency locking, asset acquisition, runtime provenance and architecture
compatibility remain the caller's responsibility. No actual model runtime or retrieval
quality is qualified by the injected tests.

## Package layout

| Module | Contents |
|---|---|
| `mimirag.index` | `Index`: build (atomic publish), BM25 search with metadata filters, locator resolution |
| `mimirag.extraction` | Plain text, HTML and PDF extraction; OCR through an external command; running-banner removal |
| `mimirag.chunking` | Passage units: whole blank-line separated blocks merged up to a target size |
| `mimirag.code` | `CodeRepository`: literal search and bounded reads at a pinned commit |
| `mimirag.sources`, `mimirag.git`, `mimirag.integrity` | Manifests, verified source bytes, pinned Git reads, confined paths and hashing |
| `mimirag.packets`, `mimirag.datatypes`, `mimirag.errors` | Result shapes, typed manifest and locator fields, and the error hierarchy |
| `mimirag.candidates`, `mimirag.embeddings` | `CandidateIndex`: deep lexical and exact dense candidates, RRF hybrid search; vector artifacts and the offline embedding adapter |
| `mimirag.server` | `mimirag-mcp`: read-only MCP tools over an index and optional code repository (`mimirag[mcp]`) |
| `mimirag.evaluation` | Alignment, scoring, paired comparisons, run manifests, the index mirror and blinded pools |

Every refusal raises a `MimiRAGError` (a `ValueError`): `IntegrityError` when
a source or commit is missing or changed, `RebuildRequired` when the index is
stale and `LocatorError` when a citation does not resolve. A malformed manifest
can still raise `KeyError`, and operating-system failures raise `OSError`.

## Usage

Requirements: Python 3.11 or later with SQLite FTS5, Git, and Poppler
(`pdftotext`, `pdftoppm`; `pdftotext` is called by every build).

A JSON manifest declares the sources. Paths are relative to the `root`
directory given to `Index`. Every source has an `id`, a `kind` and a
`version`, and each file source carries its SHA-256:

```json
{
  "documents": [
    {"id": "guide", "kind": "text", "path": "docs/guide.rst", "version": "1.2", "sha256": "..."},
    {"id": "manual", "kind": "pdf", "path": "docs/manual.pdf", "version": "2026-01", "sha256": "...",
     "pages": 40, "ocr_pages": [7]},
    {"id": "notes", "kind": "git", "repo": "vendor/project", "path": "NOTES.md",
     "version": "<full commit hash>", "ref": "v1.0"}
  ],
  "code": {"repo": "vendor/project", "version": "<full commit hash>", "scope": ["src/"]}
}
```

Kinds are `text` (also Markdown and reStructuredText), `html`, `pdf` and `git`.
Optional fields such as `title`, `family`, `topic`, `date` and
`qualification` are stored with each source; `family`, `topic`, `version`
and `date` (as YYYY-MM-DD) can filter searches.

```python
from mimirag import CodeRepository, Index, load_manifest

manifest = load_manifest('manifest.json')
index = Index('index.sqlite3', manifest, root='.')
index.build()
packet = index.search('lantern glass', topic='lantern')
index.resolve(packet['evidence'][0]['locator'])  # re-verified against the source bytes

code = CodeRepository('.', manifest['code'], qualification='Source text, not runtime behaviour')
code.search('start_pump')
code.read('src/pump.py', first=1, last=40)
```

PDF pages listed in `ocr_pages` are read by an external OCR command, passed as
`Index(..., ocr={'command': [...], 'dpi': 300})` to both building and resolving.
The command receives a directory of page images (`{images}` in the argument
list) holding `list.txt` and prints the text of each page separated by form
feeds.

A deployment can pass an `admission` object to `Index` to add sources at build
time and to exclude superseded ones from queries (see `datatypes.Admission`).
Exclusions apply only where the object is passed, so pass it to every `Index`
that serves queries.

## Roadmap

1. **Retrieval and citation core** (done, except the record interface):
   extraction, lexical index, citation resolver, code reader and retrieval
   evaluation, with tests built on fixtures written for this project.
2. **MCP server** (done): typed tool schemas,
   structured results, read-only behaviour and actionable errors, so that
   agents can use MimiRAG for RAG with their own models.
3. **Example benchmark:** a small, domain-agnostic corpus authored entirely
   for this project (documents, PDFs, a toy code repository and synthetic
   records) with reference cases. It demonstrates and regression-tests the
   evaluation package; it is not a measure of quality.
4. **Hybrid retrieval:** local open-weight embedding models, fusion and
   optional reranking, each adopted only on a measured gain. Exact dense
   ranking and reciprocal-rank fusion are in place (`CandidateIndex.hybrid`).
5. **Grounded generation:** answer synthesis with resolver-verified citations,
   unsupported-claim reporting and abstention, a pluggable model backend, and
   answer-level evaluation against the retrieval-only baseline.

## Development

```sh
python -m venv .venv
.venv/bin/pip install -e '.[dev]'
.venv/bin/ruff check . && .venv/bin/ruff format --check .
.venv/bin/mypy
.venv/bin/python -m pytest
```

The tests build their own fixtures (generated PDFs, temporary Git repositories
and a stub OCR command). Continuous integration runs the same checks on
Python 3.11 to 3.14.

## Content and licensing

- The source code is licensed under the [Apache License 2.0](../LICENSE).
- Example corpora and benchmark content authored for this project, once
  added, will be licensed under
  [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
- The repository contains no third-party content. Dependencies are declared,
  not vendored.

Copyright 2026 Bernardo Lino de Oliveira.


### Compact evidence packets

`CandidateIndex.render(..., compact=True)`, `CandidateIndex.hybrid(..., compact=True)`
and MCP `search(compact=True)` opt into `format: "compact-v1"`. Defaults are unchanged.
The packet's `sources` table maps source IDs to `{source, qualification, freshness}`;
each evidence item's `source` becomes that ID. All other item and packet fields,
including full citation locators and context, remain unchanged. References are local
to the packet. Reconstruct a full item by merging its source-table entry into it.
Pass the unchanged locator to `resolve`. Conflicting metadata for one ID is refused.
