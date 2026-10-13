"""Retrieval metrics against the answer key, with per-question run logs."""

import csv
import json
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from docsearch.answer import Answerer, ExtractiveAnswerer
from docsearch.config import (
    ANSWER_KEY_PATH,
    ANSWER_MODE,
    EVAL_KS,
    MRR_CUTOFF,
    RUN_TIMESTAMP_FORMAT,
    RUNS_DIR,
    TUNE_ANSWER_WEIGHT,
    TUNE_BM25_GRID,
    TUNE_SIM_GRID,
)
from docsearch.index.hybrid import SearchEngine

SUMMARY_FIELDS = ["run_id", "mode", "split", "k", "recall", "precision", "mrr"]


def recall_at_k(retrieved: Sequence[str], relevant: set[str], k: int) -> float:
    return len(relevant & set(retrieved[:k])) / len(relevant) if relevant else 0.0


def precision_at_k(retrieved: Sequence[str], relevant: set[str], k: int) -> float:
    return len(relevant & set(retrieved[:k])) / k


def reciprocal_rank(retrieved: Sequence[str], relevant: set[str], cutoff: int = MRR_CUTOFF):
    for rank, cid in enumerate(retrieved[:cutoff], start=1):
        if cid in relevant:
            return 1.0 / rank
    return 0.0


def load_answer_key(split: str = "all", path: Path = ANSWER_KEY_PATH) -> list[dict]:
    rows = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    return rows if split == "all" else [q for q in rows if q["split"] == split]


@dataclass
class RunResult:
    run_id: str
    mode: str
    split: str
    log_path: Path
    summary: list[dict]  # one row per k, SUMMARY_FIELDS
    per_question: list[dict]

    def recall(self, k: int, tag: str | None = None) -> float:
        rows = [
            q for q in self.per_question if q["answerable"] and (tag is None or tag in q["tags"])
        ]
        return sum(q["recall"][str(k)] for q in rows) / len(rows) if rows else 0.0

    @property
    def abstention_accuracy(self) -> float | None:
        return abstention_accuracy(self.per_question)

    @property
    def false_abstention_rate(self) -> float | None:
        return false_abstention_rate(self.per_question)


def abstention_accuracy(rows: list[dict]) -> float | None:
    """Share of unanswerable + permission questions where the system abstained."""
    should = [r for r in rows if not r["answerable"] and r["abstained"] is not None]
    return sum(r["abstained"] for r in should) / len(should) if should else None


def false_abstention_rate(rows: list[dict]) -> float | None:
    """Share of answerable questions where the system abstained."""
    can = [r for r in rows if r["answerable"] and r["abstained"] is not None]
    return sum(r["abstained"] for r in can) / len(can) if can else None


def evaluate(
    engine: SearchEngine,
    questions: list[dict],
    *,
    mode: str,
    split: str,
    ks: Sequence[int] = EVAL_KS,
    runs_dir: Path = RUNS_DIR,
    run_id: str | None = None,
    answerer: Answerer | None = None,
) -> RunResult:
    run_id = run_id or datetime.now().strftime(RUN_TIMESTAMP_FORMAT)
    depth = max(max(ks), MRR_CUTOFF)
    per_question = []
    for q in questions:
        retrieval = engine.retrieve(q["question"], q["user_visibility"], k=depth, mode=mode)
        results = retrieval.results
        answer = answerer.answer(q["question"], retrieval) if answerer else None
        ids = [r.chunk.chunk_id for r in results]
        relevant = set(q["relevant_chunk_ids"])
        per_question.append(
            {
                "qid": q["qid"],
                "split": q["split"],
                "mode": mode,
                "answerable": q["answerable"],
                "tags": q["tags"],
                "retrieved_ids": ids,
                "scores": [round(r.score, 6) for r in results],
                "ranks": [r.rank for r in results],
                "hits": [cid in relevant for cid in ids],
                "recall": {str(k): recall_at_k(ids, relevant, k) for k in ks},
                "precision": {str(k): precision_at_k(ids, relevant, k) for k in ks},
                "rr": reciprocal_rank(ids, relevant),
                "abstained": answer.abstained if answer else None,
                "answer": answer.answer if answer else None,
                "abstain_reason": answer.reason if answer else None,
            }
        )

    answerable = [q for q in per_question if q["answerable"]]
    n = len(answerable) or 1
    mrr = sum(q["rr"] for q in answerable) / n
    summary = [
        {
            "run_id": run_id,
            "mode": mode,
            "split": split,
            "k": k,
            "recall": round(sum(q["recall"][str(k)] for q in answerable) / n, 4),
            "precision": round(sum(q["precision"][str(k)] for q in answerable) / n, 4),
            "mrr": round(mrr, 4),
        }
        for k in ks
    ]

    runs_dir.mkdir(parents=True, exist_ok=True)
    log_path = runs_dir / f"{run_id}_{mode}_{split}.jsonl"
    log_path.write_text("".join(json.dumps(q) + "\n" for q in per_question))
    summary_path = runs_dir / "summary.csv"
    new_file = not summary_path.exists()
    with summary_path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_FIELDS)
        if new_file:
            writer.writeheader()
        writer.writerows(summary)

    return RunResult(run_id, mode, split, log_path, summary, per_question)


@dataclass(frozen=True)
class Thresholds:
    min_sim: float
    min_bm25: float
    min_sent_sim: float


def abstention_signals(
    engine: SearchEngine, answerer: ExtractiveAnswerer, questions: list[dict]
) -> list[dict]:
    """Per-question inputs to the abstention rules; independent of the thresholds."""
    rows = []
    for q in questions:
        retrieval = engine.retrieve(
            q["question"], q["user_visibility"], k=answerer.context_k, mode=ANSWER_MODE
        )
        best = answerer.best_sentence(q["question"], retrieval.results)
        rows.append(
            {
                "qid": q["qid"],
                "should_abstain": not q["answerable"],
                "has_results": bool(retrieval.results),
                "top_dense": retrieval.top_dense,
                "top_bm25": retrieval.top_bm25,
                "sent_sim": best[0] if best else 0.0,
            }
        )
    return rows


def predicts_abstain(signal: dict, t: Thresholds) -> bool:
    return (
        not signal["has_results"]
        or (signal["top_dense"] < t.min_sim and signal["top_bm25"] < t.min_bm25)
        or signal["sent_sim"] < t.min_sent_sim
    )


def tune_thresholds(
    signals: list[dict],
    sim_grid: Sequence[float] = TUNE_SIM_GRID,
    bm25_grid: Sequence[float] = TUNE_BM25_GRID,
    answer_weight: float = TUNE_ANSWER_WEIGHT,
) -> tuple[Thresholds, float]:
    """Grid search for the best weighted accuracy (abstain when needed, answer otherwise).

    answer_weight > 1 makes a false abstention cost more than a missed abstention.
    Ties go to the lowest thresholds, i.e. the least eager to abstain.
    """
    should = [s for s in signals if s["should_abstain"]]
    can = [s for s in signals if not s["should_abstain"]]

    def score(t: Thresholds) -> float:
        hit = sum(predicts_abstain(s, t) for s in should) / len(should) if should else 1.0
        ok = sum(not predicts_abstain(s, t) for s in can) / len(can) if can else 1.0
        return (hit + answer_weight * ok) / (1 + answer_weight)

    best, best_score = None, -1.0
    for sent in sim_grid:
        for sim in sim_grid:
            for bm25 in bm25_grid:
                t = Thresholds(sim, bm25, sent)
                s = score(t)
                if s > best_score:
                    best, best_score = t, s
    return best, best_score
