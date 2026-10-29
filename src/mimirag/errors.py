"""Exceptions raised by MimiRAG.

Every refusal is a `MimiRAGError`, which subclasses `ValueError`, so a caller can
treat all refusals alike or tell a bad request from a changed source or a stale index.
"""

from __future__ import annotations


class MimiRAGError(ValueError):
    """A request, source or setting that MimiRAG refuses."""


class IntegrityError(MimiRAGError):
    """A declared source or commit is missing, or its content no longer matches the index."""


class RebuildRequired(MimiRAGError):
    """The index was built from another manifest, schema or extractor; rebuild it."""


class LocatorError(MimiRAGError):
    """A locator does not name a passage that the current index and sources still hold."""
