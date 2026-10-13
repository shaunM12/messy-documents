from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TypeVar

from docsearch.config import CANDIDATES_N, DEFAULT_K, RRF_K, SEARCH_MODES
from docsearch.index import is_visible
from docsearch.index.bm25 import BM25Index
from docsearch.index.dense import DenseIndex, Embedder, EmbeddingCache
from docsearch.models import Chunk, ScoredChunk
from docsearch.store import Store

T = TypeVar("T")


def rrf(rankings: Iterable[Sequence[T]], k: int = RRF_K) -> list[tuple[T, float]]:
    """Reciprocal Rank Fusion with 1-based ranks; ties broken by best rank, then id."""
    scores: dict[T, float] = {}
    best: dict[T, int] = {}
    for ranking in rankings:
        for rank, item in enumerate(ranking, start=1):
            scores[item] = scores.get(item, 0.0) + 1.0 / (k + rank)
            best[item] = min(best.get(item, rank), rank)
    return sorted(scores.items(), key=lambda p: (-p[1], best[p[0]], p[0]))


@dataclass(frozen=True)
class Retrieval:
    results: list[ScoredChunk]
    top_bm25: float  # best BM25 score / max attainable for the query, in [0, 1] (0 if none)
    top_dense: float  # best cosine among visible chunks, whatever the mode (0 if none)


class SearchEngine:
    """Owns the store and in-memory indexes; indexes are rebuilt from SQLite (D2)."""

    def __init__(self, store: Store, embedder: Embedder, cache: EmbeddingCache | Path | str):
        self.store = store
        self.embedder = embedder
        self.cache = cache if isinstance(cache, EmbeddingCache) else EmbeddingCache(cache)
        self.rebuild()

    def rebuild(self) -> None:
        self.chunks: list[Chunk] = self.store.all_chunks()
        self.bm25 = BM25Index(self.chunks)
        self.dense = DenseIndex(self.chunks, self.embedder, self.cache)

    def ingest(
        self, path: Path, *, doc_id: str, version: str, visibility: list[str], updated_at: str
    ) -> list[Chunk]:
        chunks = self.store.ingest(
            path, doc_id=doc_id, version=version, visibility=visibility, updated_at=updated_at
        )
        self.rebuild()
        return chunks

    def delete(self, doc_id: str) -> int:
        removed = self.store.delete(doc_id)
        self.rebuild()
        return removed

    def retrieve(
        self, query: str, user_visibility: Iterable[str], k: int = DEFAULT_K, mode: str = "hybrid"
    ) -> Retrieval:
        if mode not in SEARCH_MODES:
            raise ValueError(f"mode must be one of {SEARCH_MODES}, got {mode!r}")
        user_visibility = list(user_visibility)
        visible = [i for i, c in enumerate(self.chunks) if is_visible(c, user_visibility)]

        bm25 = self.bm25.search(query, visible, CANDIDATES_N)
        dense = self.dense.search(query, visible, CANDIDATES_N)
        if mode == "hybrid":
            ranked = rrf([[i for i, _ in bm25], [i for i, _ in dense]])
        else:
            ranked = bm25 if mode == "bm25" else dense

        results = [
            ScoredChunk(chunk=self.chunks[i], score=s, rank=r)
            for r, (i, s) in enumerate(ranked[:k], start=1)
        ]
        bound = self.bm25.max_score(query, visible)
        return Retrieval(
            results=results,
            top_bm25=bm25[0][1] / bound if bm25 and bound else 0.0,
            top_dense=dense[0][1] if dense else 0.0,
        )

    def search(
        self, query: str, user_visibility: Iterable[str], k: int = DEFAULT_K, mode: str = "hybrid"
    ) -> list[ScoredChunk]:
        return self.retrieve(query, user_visibility, k, mode).results
