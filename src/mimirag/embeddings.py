"""Optional local embeddings and atomic identity-bound vector artifacts.

Importing this module loads no model and requires no inference dependencies.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any

from .errors import MimiRAGError
from .integrity import confined, fingerprint


def normalize(vector: Sequence[float]) -> list[float]:
    if (
        not isinstance(vector, (list, tuple))
        or not vector
        or any(type(x) not in (int, float) or not math.isfinite(x) for x in vector)
    ):
        raise MimiRAGError('Empty or nonfinite vector')
    norm = math.hypot(*vector)
    if norm == 0 or not math.isfinite(norm):
        raise MimiRAGError('Invalid vector norm')
    return [x / norm for x in vector]


def validate_vectors(vectors: Mapping[str, Sequence[float]], dimensions: int) -> None:
    if type(dimensions) is not int or dimensions < 1:
        raise MimiRAGError('Invalid dimensions')
    if not isinstance(vectors, Mapping) or any(not isinstance(key, str) or not key for key in vectors):
        raise MimiRAGError('Invalid vector IDs')
    for vector in vectors.values():
        normalized = normalize(vector)
        if len(vector) != dimensions:
            raise MimiRAGError('Vector dimensions differ')
        if any(abs(a - b) > 1e-5 for a, b in zip(vector, normalized, strict=True)):
            raise MimiRAGError('Vectors must be normalized')


def _validate_receipt(receipt: Mapping[str, Any]) -> None:
    required = {'inventory', 'model', 'tokenizer', 'runtime', 'dimensions', 'synthetic'}
    if not isinstance(receipt, Mapping) or not required <= receipt.keys():
        raise MimiRAGError('Incomplete vector receipt')
    if type(receipt['dimensions']) is not int or receipt['dimensions'] < 1 or type(receipt['synthetic']) is not bool:
        raise MimiRAGError('Invalid vector receipt')
    if any(
        not isinstance(receipt[key], (str, dict)) or not receipt[key]
        for key in ('inventory', 'model', 'tokenizer', 'runtime')
    ):
        raise MimiRAGError('Missing vector identity')
    try:
        json.dumps(dict(receipt), sort_keys=True, allow_nan=False)
    except (TypeError, ValueError) as error:
        raise MimiRAGError('Invalid vector receipt JSON') from error


def save_vectors(path: Path, vectors: Mapping[str, Sequence[float]], receipt: Mapping[str, Any]) -> None:
    """Single atomic file: metadata and payload cannot be published independently."""
    _validate_receipt(receipt)
    validate_vectors(vectors, receipt['dimensions'])
    payload = dict(format=1, receipt=dict(receipt), vectors=dict(vectors))
    content = dict(payload=payload, sha256=fingerprint(payload))
    data = json.dumps(content, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(dir=path.parent, prefix='vectors-')
    temporary = Path(name)
    try:
        with os.fdopen(fd, 'wb') as output:
            output.write(data)
            output.flush()
            os.fsync(output.fileno())
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def load_vectors(path: Path, receipt: Mapping[str, Any], ids: set[str]) -> dict[str, list[float]]:
    _validate_receipt(receipt)

    def unique(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise MimiRAGError('Duplicate vector artifact key')
            result[key] = value
        return result

    try:
        content = json.loads(path.read_text(), object_pairs_hook=unique)
        if not isinstance(content, dict) or set(content) != {'payload', 'sha256'}:
            raise MimiRAGError('Invalid vector envelope')
        payload = content['payload']
        if not isinstance(payload, dict) or set(payload) != {'format', 'receipt', 'vectors'}:
            raise MimiRAGError('Invalid vector payload')
        if type(payload['format']) is not int or payload['format'] != 1 or content['sha256'] != fingerprint(payload):
            raise MimiRAGError('Vector artifact corrupted')
        _validate_receipt(payload['receipt'])
        vectors: dict[str, list[float]] = payload['vectors']
        validate_vectors(vectors, receipt['dimensions'])
        if fingerprint(payload['receipt']) != fingerprint(dict(receipt)) or set(vectors) != ids:
            raise MimiRAGError('Vector artifact identity mismatch')
        return vectors
    except (ValueError, TypeError, KeyError, OverflowError) as error:
        raise MimiRAGError('Invalid vector artifact') from error


class OfflineEmbedder:
    """Batch-one local adapter; injected backend permits model-free verification."""

    def __init__(
        self, tokenizer: Any, model: Any, torch: Any, *, dimensions: int, query_prefix: str, max_tokens: int = 8192
    ):
        if type(dimensions) is not int or dimensions < 1 or type(max_tokens) is not int or max_tokens < 1:
            raise MimiRAGError('Invalid embedding dimensions or token cap')
        if not isinstance(query_prefix, str):
            raise MimiRAGError('Invalid query prefix')
        self.tokenizer, self.model, self.torch = tokenizer, model, torch
        self.dimensions, self.query_prefix, self.max_tokens = dimensions, query_prefix, max_tokens
        self.model.eval()

    @classmethod
    def from_local(
        cls, directory: Path, *, files: Mapping[str, str], dimensions: int, query_prefix: str, max_tokens: int = 8192
    ) -> OfflineEmbedder:
        """Never downloads: verify the complete local file allowlist before imports."""
        directory = directory.resolve()
        entries = list(directory.rglob('*'))
        if any(p.is_symlink() for p in entries):
            raise MimiRAGError('Unsafe model asset path')
        actual = {p.relative_to(directory).as_posix() for p in entries if p.is_file()}
        if not files or actual != set(files):
            raise MimiRAGError('Model asset allowlist mismatch')
        for name, expected in files.items():
            path = confined(directory, name)
            if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
                raise MimiRAGError('Model asset digest mismatch')
        # Optional dependencies are deliberately lazy.
        torch = importlib.import_module('torch')
        transformers = importlib.import_module('transformers')
        AutoModel, AutoTokenizer = transformers.AutoModel, transformers.AutoTokenizer
        torch.set_num_threads(4)
        tokenizer = AutoTokenizer.from_pretrained(str(directory), local_files_only=True, trust_remote_code=False)
        model = AutoModel.from_pretrained(
            str(directory), local_files_only=True, trust_remote_code=False, torch_dtype=torch.float32
        ).to('cpu')
        return cls(tokenizer, model, torch, dimensions=dimensions, query_prefix=query_prefix, max_tokens=max_tokens)

    def embed(self, text: str, *, query: bool = False) -> list[float]:
        encoded = self.tokenizer(
            (self.query_prefix if query else '') + text, return_tensors='pt', truncation=False, padding=False
        )
        if encoded['input_ids'].shape[1] > self.max_tokens:
            raise MimiRAGError('Embedding input exceeds token cap')
        with self.torch.inference_mode():
            hidden = self.model(**encoded).last_hidden_state
            # Batch size is one; masks may be either left- or right-padded.
            positions = encoded['attention_mask'][0].nonzero().flatten()
            if positions.numel() == 0:
                raise MimiRAGError('Embedding input has no tokens')
            vector = hidden[0, positions[-1]].float().tolist()
        if len(vector) != self.dimensions:
            raise MimiRAGError('Embedding output dimensions differ')
        return normalize(vector)
