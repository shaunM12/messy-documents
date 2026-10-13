import re
from dataclasses import dataclass, fields

from docsearch.config import CHUNK_ID_SLUG_MAX

DOC_TYPES = ("text", "table", "layout")


def slug(locator: str) -> str:
    s = re.sub(r"[^a-z0-9]+", "-", locator.lower()).strip("-")
    return s[:CHUNK_ID_SLUG_MAX].strip("-")


def make_chunk_id(doc_id: str, version: str, locator: str, n: int) -> str:
    return f"{doc_id}:{version}:{slug(locator)}:{n}"


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    doc_id: str
    source_file: str
    doc_type: str
    locator: str
    text: str
    visibility: list[str]
    version: str
    updated_at: str
    metadata: dict


@dataclass(frozen=True)
class ScoredChunk:
    chunk: Chunk
    score: float  # BM25 score, cosine, or RRF score depending on mode
    rank: int  # 1-based


def missing_fields(chunk: Chunk) -> list[str]:
    """Names of required fields that are empty or invalid."""
    bad = [f.name for f in fields(chunk) if not getattr(chunk, f.name)]
    if chunk.doc_type not in DOC_TYPES:
        bad.append("doc_type")
    return bad
