import hashlib
import sqlite3
from pathlib import Path
from typing import Protocol

import numpy as np

from docsearch.config import EMBEDDING_MODEL
from docsearch.models import Chunk


class Embedder(Protocol):
    name: str

    def encode(self, texts: list[str]) -> np.ndarray:
        """Unit-normalized float32 vectors, one row per text."""
        ...


class SentenceTransformerEmbedder:
    def __init__(self, model_name: str = EMBEDDING_MODEL):
        self.name = model_name
        self._model = None

    def encode(self, texts: list[str]) -> np.ndarray:
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.name)
        return self._model.encode(
            texts, normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False
        ).astype(np.float32)


class EmbeddingCache:
    """Chunk vectors keyed by a hash of (model, chunk_id, text)."""

    def __init__(self, path: Path | str):
        if str(path) != ":memory:":
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(path))
        self.conn.execute("CREATE TABLE IF NOT EXISTS vectors (key TEXT PRIMARY KEY, vec BLOB)")

    @staticmethod
    def key(model: str, chunk: Chunk) -> str:
        return hashlib.sha256(f"{model}\0{chunk.chunk_id}\0{chunk.text}".encode()).hexdigest()

    def embed_chunks(self, embedder: Embedder, chunks: list[Chunk]) -> np.ndarray:
        keys = [self.key(embedder.name, c) for c in chunks]
        found = {}
        for i in range(0, len(keys), 500):
            batch = keys[i : i + 500]
            marks = ",".join("?" * len(batch))
            rows = self.conn.execute(f"SELECT key, vec FROM vectors WHERE key IN ({marks})", batch)
            found.update({k: np.frombuffer(v, dtype=np.float32) for k, v in rows})
        missing = [i for i, k in enumerate(keys) if k not in found]
        if missing:
            vecs = embedder.encode([chunks[i].text for i in missing])
            with self.conn:
                self.conn.executemany(
                    "INSERT OR REPLACE INTO vectors (key, vec) VALUES (?, ?)",
                    [(keys[i], v.tobytes()) for i, v in zip(missing, vecs, strict=True)],
                )
            found.update({keys[i]: v for i, v in zip(missing, vecs, strict=True)})
        return np.vstack([found[k] for k in keys])


class DenseIndex:
    def __init__(self, chunks: list[Chunk], embedder: Embedder, cache: EmbeddingCache):
        self.chunks = chunks
        self.embedder = embedder
        self.vectors = (
            cache.embed_chunks(embedder, chunks) if chunks else np.zeros((0, 1), np.float32)
        )

    def search(self, query: str, visible: list[int], n: int) -> list[tuple[int, float]]:
        """(chunk index, cosine) of the top n visible chunks."""
        if not visible:
            return []
        q = self.embedder.encode([query])[0]
        sims = self.vectors[visible] @ q
        ranked = sorted(zip(visible, sims, strict=True), key=lambda p: (-p[1], p[0]))
        return [(i, float(s)) for i, s in ranked[:n]]
