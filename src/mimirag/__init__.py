"""MimiRAG: retrieval with verifiable citations over documents and source code.

from mimirag import Index, load_manifest

index = Index('index.sqlite3', load_manifest('manifest.json'), root='.')
index.build()
packet = index.search('lantern glass')
index.resolve(packet['evidence'][0]['locator'])
"""

from .code import CodeRepository
from .errors import IntegrityError, LocatorError, MimiRAGError, RebuildRequired
from .index import Index
from .packets import LOCATOR_FORMAT, evidence_packet
from .sources import load_manifest

__version__ = '0.1.0.dev0'

__all__ = [
    'LOCATOR_FORMAT',
    'CodeRepository',
    'Index',
    'IntegrityError',
    'LocatorError',
    'MimiRAGError',
    'RebuildRequired',
    'evidence_packet',
    'load_manifest',
]
