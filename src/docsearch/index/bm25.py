import re

from rank_bm25 import BM25Okapi

from docsearch.models import Chunk

# Words, numbers and SKU-like compounds ("nw-1042", "3.5", "usb-c").
_TOKEN = re.compile(r"[a-z0-9]+(?:[-.][a-z0-9]+)*")


def tokenize(text: str) -> list[str]:
    tokens = []
    for tok in _TOKEN.findall(text.lower()):
        tokens.append(tok)
        if "-" in tok:
            tokens.extend(part for part in tok.split("-") if part)
    return tokens


class BM25Index:
    """BM25 over the visible subset only, so filtering happens before ranking (D10)."""

    def __init__(self, chunks: list[Chunk]):
        self.chunks = chunks
        self.tokens = [tokenize(c.text) for c in chunks]
        self._cache: dict[tuple[int, ...], BM25Okapi] = {}

    def _model(self, visible: tuple[int, ...]) -> BM25Okapi:
        if visible not in self._cache:
            self._cache[visible] = BM25Okapi([self.tokens[i] for i in visible])
        return self._cache[visible]

    def search(self, query: str, visible: list[int], n: int) -> list[tuple[int, float]]:
        """(chunk index, score) of the top n visible chunks with a positive score."""
        terms = tokenize(query)
        if not visible or not terms:
            return []
        key = tuple(visible)
        scores = self._model(key).get_scores(terms)
        ranked = sorted(zip(key, scores, strict=True), key=lambda p: (-p[1], p[0]))
        return [(i, float(s)) for i, s in ranked[:n] if s > 0]

    def max_score(self, query: str, visible: list[int]) -> float:
        """Upper bound on any visible chunk's BM25 score for this query.

        Tokens absent from the visible chunks contribute nothing, matching get_scores; counting
        them made filler words like "what" dominate the bound.
        """
        terms = tokenize(query)
        if not visible or not terms:
            return 0.0
        model = self._model(tuple(visible))
        return sum(model.idf.get(t, 0.0) * (model.k1 + 1) for t in terms)
