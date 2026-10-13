"""Tune MIN_SIM / MIN_BM25 / MIN_SENT_SIM on the dev split only (D5).

Run: uv run python scripts/tune_thresholds.py
Copy the printed values into src/docsearch/config.py.
"""

from docsearch.answer import ExtractiveAnswerer
from docsearch.config import EMBEDDING_CACHE_PATH
from docsearch.eval import abstention_signals, load_answer_key, tune_thresholds
from docsearch.index.dense import SentenceTransformerEmbedder
from docsearch.index.hybrid import SearchEngine
from docsearch.manifest import ingest_manifest
from docsearch.store import Store


def main() -> None:
    embedder = SentenceTransformerEmbedder()
    engine = SearchEngine(Store(":memory:"), embedder, EMBEDDING_CACHE_PATH)
    ingest_manifest(engine)
    signals = abstention_signals(engine, ExtractiveAnswerer(embedder), load_answer_key("dev"))

    print(f"{'qid':<5} {'abstain?':<9} {'dense':>6} {'bm25':>7} {'sent':>6}")
    for s in signals:
        print(
            f"{s['qid']:<5} {str(s['should_abstain']):<9} {s['top_dense']:>6.3f} "
            f"{s['top_bm25']:>7.3f} {s['sent_sim']:>6.3f}"
        )
    best, score = tune_thresholds(signals)
    print(f"\nweighted accuracy on dev = {score:.3f}")
    print(f"MIN_SIM = {best.min_sim}")
    print(f"MIN_BM25 = {best.min_bm25}")
    print(f"MIN_SENT_SIM = {best.min_sent_sim}")


if __name__ == "__main__":
    main()
