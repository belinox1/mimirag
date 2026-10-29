"""Single-file SQLite FTS5 index with content-addressed passage locators.

A build extracts every source into a new file and publishes it by atomic rename,
so readers never see a partial index. Every query first checks that the index
still describes the manifest and the sources; resolving a locator re-extracts
the cited text from the original bytes and compares it with the index.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from contextlib import closing, contextmanager
from datetime import UTC, date, datetime
import hashlib
from itertools import islice, pairwise
import json
import os
from pathlib import Path
import platform
import re
import sqlite3
import subprocess
import tempfile
from typing import Any

from .chunking import units
from .datatypes import Admission, BuildReport, Manifest, OcrSettings, Packet, Page, PassageLocator, Source
from .errors import IntegrityError, LocatorError, MimiRAGError, RebuildRequired
from .extraction import extract_document, extract_page
from .git import verify_commit
from .integrity import canonical, digest, fingerprint
from .packets import LOCATOR_FORMAT, MAX_EXCERPT, evidence_packet, excerpt, passage_item
from .sources import source_bytes, stat_key

EXTRACTOR = 'layout-v1'
EXTRACTION_POLICY = 'reading-order-ocr-banners-v1'
MAX_RESULTS = 20
MAX_QUERY_TOKENS = 32
OVERSIZED = '; oversized structural unit: inspect original; search excerpt truncated'
# Modules whose code determines what an index stores. Their hash is recorded with the extractor, and so is
# part of the index identity: an index built by different code is never mistaken for this one's output.
IMPLEMENTATION = (
    'index.py',
    'packets.py',
    'integrity.py',
    'chunking.py',
    'extraction/document.py',
    'extraction/html.py',
    'extraction/pdf.py',
)

SCHEMA = """
CREATE TABLE metadata(key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE source(id TEXT PRIMARY KEY, description TEXT NOT NULL, sha256 TEXT NOT NULL, stat TEXT,
                    extraction TEXT NOT NULL, page_outputs TEXT NOT NULL, banners TEXT);
CREATE TABLE passage(id INTEGER PRIMARY KEY, source_id TEXT NOT NULL, page INTEGER,
                     first_line INTEGER, last_line INTEGER, text TEXT NOT NULL,
                     qualification TEXT NOT NULL);
CREATE VIRTUAL TABLE passage_fts USING fts5(text, content='passage', content_rowid='id');
"""

SEARCH = """
SELECT p.*, s.description, s.sha256, s.extraction, bm25(passage_fts) AS score
FROM passage_fts JOIN passage p ON p.id = passage_fts.rowid JOIN source s ON s.id = p.source_id
WHERE {conditions} ORDER BY score, p.id LIMIT ?
"""

CONTEXT = """
SELECT first_line, last_line, text FROM passage
WHERE source_id = ? AND page IS ? AND id BETWEEN ? AND ? ORDER BY id
"""

CITED = """
SELECT p.*, s.description, s.sha256, s.extraction, s.page_outputs, s.banners
FROM passage p JOIN source s ON s.id = p.source_id
WHERE source_id = ? AND page IS ? AND first_line = ? AND last_line = ?
"""


def extractor_identity(ocr: OcrSettings | None = None) -> dict[str, Any]:
    """The extraction tools, settings and implementation, recorded in every index."""
    try:
        version = subprocess.run(['pdftotext', '-v'], capture_output=True, check=True, timeout=10)
    except subprocess.SubprocessError as error:
        raise MimiRAGError('pdftotext -v failed; Poppler is required') from error
    package = Path(__file__).parent
    code = b''.join((package / name).read_bytes() for name in IMPLEMENTATION)
    return dict(
        name=EXTRACTOR,
        pdf_utility=version.stderr.decode().splitlines()[0],
        settings=['-enc', 'UTF-8'],
        policy=EXTRACTION_POLICY,
        ocr=ocr,
        python=platform.python_version(),
        implementation=digest(code),
    )


def extraction_identity(pages: Sequence[Page]) -> str:
    """Hash of extracted pages: it changes exactly when the text or its page mapping changes."""
    return fingerprint(pages)


def document_title(source: Source, pages: Sequence[Page]) -> str:
    """A declared title, else the first HTML line or a reStructuredText heading, else the ID.

    PDF titles are never guessed from the text: a PDF uses its declared citation, else its ID.
    """
    if source.get('title'):
        return source['title']
    if source['kind'] == 'pdf':
        return source.get('citation', source['id'])
    text = pages[0][1] if pages else ''
    if source['kind'] == 'html':
        return next((line.strip() for line in text.splitlines() if line.strip()), source['id'])
    lines = text.splitlines()
    for line, next_line in pairwise(lines):
        heading, underline = line.strip(), next_line.strip()
        if heading and len(underline) >= len(heading) and len(set(underline)) == 1 and underline[0] in '=-~^':
            return heading
    return source['id']


def fts_query(query: str) -> str:
    """FTS5 expression matching any word of `query`, for the index and the evaluation mirror."""
    tokens = [m[0] for m in islice(re.finditer(r'\w+', query), MAX_QUERY_TOKENS)]  # words past the cap are ignored
    if not tokens:
        raise MimiRAGError('Query needs at least one word or identifier')
    # Each word is quoted (a word never contains a quote), so query text is never parsed as FTS5 syntax.
    return ' OR '.join(f'"{token}"' for token in tokens)


class Index:
    """An index file built from a manifest whose relative paths are resolved against `root`.

    `ocr` configures the OCR command for PDF pages that need it. `admission` lets
    a deployment add sources and gate builds and queries (see `datatypes.Admission`).
    """

    def __init__(
        self,
        path: str | Path,
        manifest: Manifest,
        *,
        root: str | Path,
        ocr: OcrSettings | None = None,
        admission: Admission | None = None,
    ) -> None:
        self.path = Path(path).resolve()
        self.manifest = manifest
        self.root = Path(root).resolve()
        self.ocr = ocr
        self.admission = admission

    def build(self) -> BuildReport:
        """Extract every source and publish a new index; on any failure the old index stays."""
        previous = self._stored_metadata()
        extra, metadata = self.admission.prepare(self.manifest, previous) if self.admission else ([], None)
        sources = self.manifest['documents'] + extra
        if len({s['id'] for s in sources}) != len(sources):
            raise MimiRAGError('Duplicate source IDs')
        self.path.parent.mkdir(parents=True, exist_ok=True)
        descriptor, name = tempfile.mkstemp(prefix='index-', suffix='.sqlite3', dir=self.path.parent)
        os.close(descriptor)
        temporary = Path(name)
        try:
            identity = self._write(temporary, sources, extra, metadata)
            # Check every source once more and let the deployment veto before replacing the index.
            for source in sources:
                source_bytes(self.root, source)
            if self.admission:
                self.admission.before_publish(self.manifest)
            temporary.replace(self.path)
        finally:
            temporary.unlink(missing_ok=True)
        return BuildReport(
            index=str(self.path), identity=identity, sources=len(sources), bytes=self.path.stat().st_size
        )

    def search(
        self,
        query: str,
        *,
        limit: int = 10,
        source: str | None = None,
        family: str | None = None,
        topic: str | None = None,
        version: str | None = None,
        since: str | None = None,
        until: str | None = None,
    ) -> Packet:
        """BM25-ranked passages with locators, optionally filtered by source metadata and date.

        Ties are broken by passage order. Each item carries its neighbouring passages
        on the same page as context.
        """
        if not 1 <= limit <= MAX_RESULTS:
            raise MimiRAGError(f'limit must be 1..{MAX_RESULTS}')
        conditions, arguments = ['passage_fts MATCH ?'], [fts_query(query)]
        with self._open() as db:
            excluded, notices = self._check_freshness(db)
            identity, extractor = self._built_identity(db)
            if excluded:
                conditions.append('s.id NOT IN (' + ','.join('?' * len(excluded)) + ')')
                arguments.extend(sorted(excluded))
            filters = dict(source=source, family=family, topic=topic, version=version, since=since, until=until)
            _add_filters(conditions, arguments, filters)
            sql = SEARCH.format(conditions=' AND '.join(conditions))
            rows = db.execute(sql, [*arguments, limit + 1]).fetchall()
            items = []
            for rank, row in enumerate(rows[:limit], 1):
                item = passage_item(row)
                item['rank'] = rank
                item['context'] = _context(db, row)
                items.append(item)
        return evidence_packet(
            'search',
            items,
            index=identity,
            extractor=extractor['name'],
            query=query,
            notices=notices,
            filters=filters,
            more_results=len(rows) > limit,
            limit=limit,
        )

    def resolve(self, locator: Mapping[str, Any]) -> Packet:
        """The cited passage, after checking the locator against the index and the source."""
        if locator.get('format') != LOCATOR_FORMAT:
            raise LocatorError(f'Unsupported locator format; search again for a format {LOCATOR_FORMAT} locator')
        valid = (
            set(locator) == PassageLocator.__required_keys__
            and isinstance(locator['source'], str)
            and (locator['page'] is None or _is_number(locator['page']))
            and _is_span(locator['lines'])
        )
        if not valid:
            raise LocatorError('A passage locator has exactly the fields a search returns')
        with self._open() as db:
            excluded, notices = self._check_freshness(db)
            identity, extractor = self._built_identity(db)
            if locator['source'] in excluded:
                successor = excluded[locator['source']]
                raise LocatorError(
                    'Cited source is superseded or no longer eligible'
                    + (f'; successor: {successor}' if successor else '')
                )
            row = db.execute(CITED, (locator['source'], locator['page'], *locator['lines'])).fetchone()
        if row is None:
            raise LocatorError('Unavailable passage locator')
        item = passage_item(row)
        if item['locator'] != locator:
            raise LocatorError('Citation identity, version, extraction or passage mismatch')
        self._verify_extraction(row, item['source'], locator, extractor)
        item['original'] = dict(
            path=item['source'].get('path'),
            page=locator['page'],
            repo=item['source'].get('repo'),
            version=locator['version'],
        )
        return evidence_packet('resolve', [item], index=identity, extractor=extractor['name'], notices=notices)

    def _stored_metadata(self) -> Any:
        """The admission metadata of the index being replaced, or None."""
        if not self.path.exists():
            return None
        with closing(sqlite3.connect(self.path.as_uri() + '?mode=ro', uri=True)) as old:
            row = old.execute("SELECT value FROM metadata WHERE key='profile'").fetchone()
        return json.loads(row[0]) if row else None

    def _write(self, temporary: Path, sources: list[Source], extra: list[Source], metadata: Any) -> str:
        """Fill a new index file; return its identity, a hash of the inputs and extracted text."""
        extractor = extractor_identity(self.ocr)
        hasher = hashlib.sha256(canonical(dict(config=self.manifest, extra=extra, extractor=extractor)).encode())
        with closing(sqlite3.connect(temporary)) as db, db:
            db.executescript(SCHEMA)
            db.executemany(
                'INSERT INTO metadata VALUES (?, ?)',
                [
                    ('manifest', canonical(self.manifest)),
                    ('profile', canonical(metadata)),
                    ('created_at', datetime.now(UTC).isoformat()),
                    ('extractor', canonical(extractor)),
                ],
            )
            for source in sources:
                self._add_source(db, source, hasher)
            identity = hasher.hexdigest()
            db.execute("INSERT INTO metadata VALUES ('identity', ?)", (identity,))
            db.execute("INSERT INTO passage_fts(passage_fts) VALUES ('rebuild')")
            if db.execute('PRAGMA integrity_check').fetchone()[0] != 'ok':
                raise MimiRAGError('Index integrity failure')
        return identity

    def _add_source(self, db: sqlite3.Connection, source: Source, hasher: Any) -> None:
        """Insert a source with its passages and feed its bytes and text to the identity hash."""
        # The stat key is taken before reading, so a write during the build shows as a change.
        key = stat_key(self.root, source)
        data = source_bytes(self.root, source)
        sha256 = digest(data)
        pages, banners = extract_document(self.root, source, self.ocr)
        description = dict(source, title=document_title(source, pages))
        # A hash per page lets resolution re-extract only the cited page of a PDF.
        outputs = [[page, digest(text.encode())] for page, text in pages]
        db.execute(
            'INSERT INTO source VALUES (?, ?, ?, ?, ?, ?, ?)',
            (
                source['id'],
                canonical(description),
                sha256,
                canonical(key),
                extraction_identity(pages),
                canonical(outputs),
                canonical(banners),
            ),
        )
        hasher.update(canonical([source['id'], sha256]).encode())
        for page, text in pages:
            hasher.update(canonical([page, text]).encode())
            for first, last, body in units(text):
                qualification = source.get('qualification', '')
                if len(body) > MAX_EXCERPT:
                    qualification += OVERSIZED
                db.execute(
                    'INSERT INTO passage(source_id, page, first_line, last_line, text, '
                    'qualification) VALUES (?, ?, ?, ?, ?, ?)',
                    (source['id'], page, first, last, body, qualification),
                )

    @contextmanager
    def _open(self) -> Iterator[sqlite3.Connection]:
        if not self.path.is_file():
            raise MimiRAGError(f'No index at {self.path}; build it first')
        db = sqlite3.connect(self.path.as_uri() + '?mode=ro', uri=True)
        db.row_factory = sqlite3.Row
        try:
            yield db
        finally:
            db.close()

    def _check_freshness(self, db: sqlite3.Connection) -> tuple[dict[str, str | None], list[str]]:
        """Checks run before every query; returns (excluded sources, notices).

        The manifest must be the one indexed. File sources are compared by stat key,
        and rehashed only when that changed; each Git revision is checked once. Full
        byte checks happen at build and on resolution, not here.
        """
        # Indexes from before per-page output hashes and admission metadata cannot be verified.
        if 'page_outputs' not in {column[1] for column in db.execute('PRAGMA table_info(source)')}:
            raise RebuildRequired('Unsupported index schema; rebuild the index')
        stored = db.execute("SELECT value FROM metadata WHERE key='profile'").fetchone()
        if stored is None:
            raise RebuildRequired('Unsupported index schema; rebuild the index')
        indexed = json.loads(db.execute("SELECT value FROM metadata WHERE key='manifest'").fetchone()[0])
        if self.manifest != indexed:
            raise RebuildRequired('Manifest changed; rebuild the index')
        revisions: dict[tuple[str, str, str | None], None] = {}
        rehashed = 0
        for row in db.execute('SELECT description, sha256, stat FROM source'):
            source = json.loads(row['description'])
            if source['kind'] == 'git':
                revisions[source['repo'], source['version'], source.get('ref')] = None
            elif canonical(stat_key(self.root, source)) != row['stat']:
                if digest(source_bytes(self.root, source)) != row['sha256']:
                    raise IntegrityError('Indexed source changed; rebuild')
                rehashed += 1
        for repo, commit, ref in revisions:
            verify_commit((self.root / repo).resolve(), commit, ref)
        excluded: dict[str, str | None] = {}
        notices = []
        if rehashed:
            notices.append(f'{rehashed} unchanged file source(s) rehashed on every query; rebuild to refresh stat keys')
        if self.admission:
            ids = {row[0] for row in db.execute('SELECT id FROM source')}
            state, more = self.admission.state(self.manifest, json.loads(stored[0]), ids)
            excluded = dict(state)
            notices.extend(more)
        return excluded, notices

    @staticmethod
    def _built_identity(db: sqlite3.Connection) -> tuple[str, dict[str, Any]]:
        identity = db.execute("SELECT value FROM metadata WHERE key='identity'").fetchone()[0]
        extractor = json.loads(db.execute("SELECT value FROM metadata WHERE key='extractor'").fetchone()[0])
        return identity, extractor

    def _verify_extraction(
        self, row: sqlite3.Row, source: Source, locator: Mapping[str, Any], built: dict[str, Any]
    ) -> None:
        """Re-extract and compare the cited text: the cited page of a PDF, else the whole text."""
        if source['kind'] != 'pdf':
            pages, _ = extract_document(self.root, source, self.ocr)
            if extraction_identity(pages) != row['extraction']:
                raise RebuildRequired('Extraction output changed; rebuild with current extractor')
            return
        # Banners were chosen from all pages, which this check does not re-read, so a
        # different reader or OCR command requires a rebuild instead.
        if built['ocr'] is not None and self.ocr is None:
            raise MimiRAGError('The index was built with OCR settings; pass them to resolve PDF pages')
        current = extractor_identity(self.ocr)
        if built['pdf_utility'] != current['pdf_utility'] or built['ocr'] != current['ocr']:
            raise RebuildRequired('PDF utility or OCR command changed since build; rebuild with current extractor')
        text = extract_page(self.root, source, locator['page'], json.loads(row['banners']), self.ocr)
        first, last = locator['lines']
        page_hash = dict(json.loads(row['page_outputs'])).get(locator['page'])
        if digest(text.encode()) != page_hash or '\n'.join(text.splitlines()[first - 1 : last]) != row['text']:
            raise RebuildRequired('Extraction output changed; rebuild with current extractor')


def _add_filters(conditions: list[str], arguments: list[Any], filters: dict[str, str | None]) -> None:
    """SQL conditions for metadata filters (exact match) and an inclusive ISO date range."""
    for field in ('source', 'family', 'topic', 'version'):
        if filters[field]:
            column = 's.id' if field == 'source' else f"json_extract(s.description, '$.{field}')"
            conditions.append(f'{column} = ?')
            arguments.append(filters[field])
    for operator, field in (('>=', 'since'), ('<=', 'until')):
        value = filters[field]
        if value:
            # Dates compare as text, so only the YYYY-MM-DD form orders correctly.
            if not _is_date(value):
                raise MimiRAGError(f'Expected a date as YYYY-MM-DD, got {value!r}')
            conditions.append(f"json_extract(s.description, '$.date') {operator} ?")
            arguments.append(value)


def _is_span(lines: Any) -> bool:
    return isinstance(lines, list) and len(lines) == 2 and all(_is_number(n) for n in lines)


def _is_number(value: Any) -> bool:
    """Whether `value` is a page or line number that SQLite can compare: a positive 64-bit integer."""
    return type(value) is int and 1 <= value < 2**63


def _is_date(value: str) -> bool:
    """Whether `value` is a valid date written exactly as YYYY-MM-DD."""
    try:
        return date.fromisoformat(value).isoformat() == value
    except ValueError:
        return False


def _context(db: sqlite3.Connection, row: sqlite3.Row) -> list[dict[str, Any]]:
    """`row` and the passages next to it on the same page, as bounded surrounding context."""
    rows = db.execute(CONTEXT, (row['source_id'], row['page'], row['id'] - 1, row['id'] + 1))
    return [dict(lines=[first, last], **excerpt(text)) for first, last, text in rows]
