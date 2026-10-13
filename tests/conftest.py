import hashlib

import numpy as np
import pytest

from docsearch.config import PROJECT_ROOT
from docsearch.index.bm25 import tokenize
from docsearch.index.dense import EmbeddingCache
from docsearch.index.hybrid import SearchEngine
from docsearch.store import Store

RAW = PROJECT_ROOT / "data" / "raw"
STOPWORDS = {
    *("a", "an", "and", "are", "as", "at", "be", "by", "can", "do", "does", "for", "from"),
    *("how", "i", "in", "is", "it", "my", "of", "on", "or", "the", "to", "what", "when"),
    *("where", "which", "who", "will", "with", "you"),
}


class HashingEmbedder:
    """Offline stand-in for the sentence-transformer: hashed bag of words, unit-normalized."""

    name = "test-hashing-embedder"
    dims = 512

    def encode(self, texts: list[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dims), dtype=np.float32)
        for row, text in enumerate(texts):
            for tok in (t for t in tokenize(text) if t not in STOPWORDS):
                out[row, int(hashlib.sha1(tok.encode()).hexdigest(), 16) % self.dims] += 1.0
        norms = np.linalg.norm(out, axis=1, keepdims=True)
        return out / np.where(norms == 0, 1.0, norms)


@pytest.fixture
def engine():
    store = Store(":memory:")
    eng = SearchEngine(store, HashingEmbedder(), EmbeddingCache(":memory:"))
    yield eng
    store.close()


def ingest(engine, name, doc_id, version="v1", visibility=("public",), updated_at="2026-10-01"):
    return engine.ingest(
        RAW / name,
        doc_id=doc_id,
        version=version,
        visibility=list(visibility),
        updated_at=updated_at,
    )
