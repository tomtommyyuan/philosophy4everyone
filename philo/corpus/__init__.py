"""Reading, cleaning and chunking source texts."""

from .chunker import chunk_work, split_blocks, stats_for
from .ingest import IngestReport, ingest, preview_chunks
from .loader import CorpusError, LoadedWork, load_file, load_library, parse_front_matter

__all__ = [
    "CorpusError", "IngestReport", "LoadedWork", "chunk_work", "ingest",
    "load_file", "load_library", "parse_front_matter", "preview_chunks",
    "split_blocks", "stats_for",
]
