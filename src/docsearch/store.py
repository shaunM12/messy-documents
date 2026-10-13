"""SQLite chunk store. Replacing a doc hard-deletes its old chunks in the same transaction (D1)."""

import json
import sqlite3
from pathlib import Path

from docsearch.chunkers import chunk_file
from docsearch.models import Chunk, missing_fields

SCHEMA = """
CREATE TABLE IF NOT EXISTS chunks (
    seq         INTEGER PRIMARY KEY AUTOINCREMENT,
    chunk_id    TEXT NOT NULL UNIQUE,
    doc_id      TEXT NOT NULL,
    source_file TEXT NOT NULL,
    doc_type    TEXT NOT NULL,
    locator     TEXT NOT NULL,
    text        TEXT NOT NULL,
    visibility  TEXT NOT NULL,
    version     TEXT NOT NULL,
    updated_at  TEXT NOT NULL,
    metadata    TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_chunks_doc_id ON chunks (doc_id);
"""

_COLUMNS = (
    "chunk_id, doc_id, source_file, doc_type, locator, text, visibility, version, updated_at, "
    "metadata"
)


def _row(chunk: Chunk) -> tuple:
    return (
        chunk.chunk_id,
        chunk.doc_id,
        chunk.source_file,
        chunk.doc_type,
        chunk.locator,
        chunk.text,
        json.dumps(chunk.visibility),
        chunk.version,
        chunk.updated_at,
        json.dumps(chunk.metadata, sort_keys=True),
    )


def _chunk(row: tuple) -> Chunk:
    *head, visibility, version, updated_at, metadata = row
    return Chunk(*head, json.loads(visibility), version, updated_at, json.loads(metadata))


class Store:
    def __init__(self, db_path: Path | str):
        if str(db_path) != ":memory:":
            Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(db_path))
        self.conn.executescript(SCHEMA)

    def close(self) -> None:
        self.conn.close()

    def replace(self, doc_id: str, chunks: list[Chunk]) -> None:
        """Delete every chunk of doc_id and insert the new chunks in one transaction."""
        for chunk in chunks:
            if chunk.doc_id != doc_id:
                raise ValueError(f"{chunk.chunk_id} does not belong to {doc_id}")
            if bad := missing_fields(chunk):
                raise ValueError(f"{chunk.chunk_id} has empty fields: {bad}")
        with self.conn:
            self.conn.execute("DELETE FROM chunks WHERE doc_id = ?", (doc_id,))
            self.conn.executemany(
                f"INSERT INTO chunks ({_COLUMNS}) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                [_row(c) for c in chunks],
            )

    def ingest(
        self, path: Path, *, doc_id: str, version: str, visibility: list[str], updated_at: str
    ) -> list[Chunk]:
        chunks = chunk_file(
            Path(path), doc_id=doc_id, version=version, visibility=visibility, updated_at=updated_at
        )
        self.replace(doc_id, chunks)
        return chunks

    def delete(self, doc_id: str) -> int:
        with self.conn:
            cur = self.conn.execute("DELETE FROM chunks WHERE doc_id = ?", (doc_id,))
        return cur.rowcount

    def all_chunks(self) -> list[Chunk]:
        rows = self.conn.execute(f"SELECT {_COLUMNS} FROM chunks ORDER BY seq").fetchall()
        return [_chunk(r) for r in rows]

    def doc_versions(self) -> dict[str, list[str]]:
        rows = self.conn.execute(
            "SELECT doc_id, version FROM chunks GROUP BY doc_id, version ORDER BY doc_id, version"
        ).fetchall()
        out: dict[str, list[str]] = {}
        for doc_id, version in rows:
            out.setdefault(doc_id, []).append(version)
        return out
