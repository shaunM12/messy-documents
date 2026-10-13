from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from docsearch.config import PROJECT_ROOT
from docsearch.models import Chunk, make_chunk_id, slug


@dataclass(frozen=True)
class Piece:
    """A chunker's output before document-level fields are attached."""

    locator: str
    text: str
    metadata: dict


DOC_TYPE_BY_SUFFIX = {".md": "text", ".txt": "text", ".csv": "table", ".pdf": "layout"}


def doc_type_for(path: Path) -> str:
    try:
        return DOC_TYPE_BY_SUFFIX[path.suffix.lower()]
    except KeyError:
        raise ValueError(f"Unsupported file type: {path}") from None


def relative_source(path: Path) -> str:
    path = path.resolve()
    try:
        return path.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def build_chunks(
    pieces: list[Piece],
    *,
    doc_id: str,
    version: str,
    source_file: str,
    doc_type: str,
    visibility: list[str],
    updated_at: str,
) -> list[Chunk]:
    # n counts per slug (not raw locator) so truncated or colliding slugs still yield unique IDs.
    seen: Counter[str] = Counter()
    chunks = []
    for piece in pieces:
        key = slug(piece.locator)
        chunks.append(
            Chunk(
                chunk_id=make_chunk_id(doc_id, version, piece.locator, seen[key]),
                doc_id=doc_id,
                source_file=source_file,
                doc_type=doc_type,
                locator=piece.locator,
                text=piece.text,
                visibility=list(visibility),
                version=version,
                updated_at=updated_at,
                metadata=piece.metadata,
            )
        )
        seen[key] += 1
    return chunks


def chunk_file(
    path: Path, *, doc_id: str, version: str, visibility: list[str], updated_at: str
) -> list[Chunk]:
    from docsearch.chunkers.layout_chunker import chunk_layout
    from docsearch.chunkers.table_chunker import chunk_table, table_title
    from docsearch.chunkers.text_chunker import chunk_text
    from docsearch.loaders.layout import load_layout
    from docsearch.loaders.table import load_table
    from docsearch.loaders.text import load_text

    doc_type = doc_type_for(path)
    source_file = relative_source(path)
    if doc_type == "text":
        pieces = chunk_text(load_text(path))
    elif doc_type == "table":
        pieces = chunk_table(load_table(path), title=table_title(source_file))
    else:
        pieces = chunk_layout(load_layout(path), source_file=source_file)
    return build_chunks(
        pieces,
        doc_id=doc_id,
        version=version,
        source_file=source_file,
        doc_type=doc_type,
        visibility=visibility,
        updated_at=updated_at,
    )
