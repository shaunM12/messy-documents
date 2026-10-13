import csv
import json

import pytest
from conftest import ingest

from docsearch.eval import (
    Thresholds,
    abstention_accuracy,
    evaluate,
    false_abstention_rate,
    precision_at_k,
    predicts_abstain,
    recall_at_k,
    reciprocal_rank,
    tune_thresholds,
)

RANKING = ["a", "x", "b", "y", "z", "c"]
RELEVANT = {"a", "b", "c", "d"}


@pytest.mark.parametrize(
    ("k", "recall", "precision"),
    [(1, 1 / 4, 1 / 1), (3, 2 / 4, 2 / 3), (5, 2 / 4, 2 / 5), (10, 3 / 4, 3 / 10)],
)
def test_metrics(k, recall, precision):
    assert recall_at_k(RANKING, RELEVANT, k) == pytest.approx(recall)
    assert precision_at_k(RANKING, RELEVANT, k) == pytest.approx(precision)


def test_reciprocal_rank():
    assert reciprocal_rank(RANKING, RELEVANT) == 1.0
    assert reciprocal_rank(["x", "y", "b"], {"b"}) == pytest.approx(1 / 3)
    assert reciprocal_rank(["x"] * 10 + ["b"], {"b"}) == 0.0
    assert reciprocal_rank([], {"b"}) == 0.0


def test_abstention_metrics():
    rows = [
        {"answerable": False, "abstained": True},
        {"answerable": False, "abstained": False},
        {"answerable": True, "abstained": True},
        {"answerable": True, "abstained": False},
        {"answerable": True, "abstained": False},
        {"answerable": True, "abstained": False},
    ]
    assert abstention_accuracy(rows) == 0.5
    assert false_abstention_rate(rows) == 0.25
    assert abstention_accuracy([{"answerable": True, "abstained": False}]) is None


def _signal(abstain, dense, bm25, sent):
    return {"should_abstain": abstain, "has_results": True, "top_dense": dense,
            "top_bm25": bm25, "sent_sim": sent}  # fmt: skip


def test_tune_thresholds_separates_toy_signals():
    signals = [
        _signal(False, 0.7, 12.0, 0.6),
        _signal(False, 0.6, 3.0, 0.55),
        _signal(True, 0.3, 2.0, 0.6),  # caught by weak retrieval
        _signal(True, 0.8, 15.0, 0.3),  # caught by sentence similarity
    ]
    best, score = tune_thresholds(signals, sim_grid=[0.0, 0.5, 0.9], bm25_grid=[0.0, 5.0])
    assert score == 1.0
    assert [predicts_abstain(s, best) for s in signals] == [False, False, True, True]
    assert best == Thresholds(min_sim=0.5, min_bm25=5.0, min_sent_sim=0.5)


def test_answer_weight_prefers_answering_over_catching_overlapping_abstention():
    signals = [
        _signal(False, 0.9, 0.9, 0.6),
        _signal(False, 0.9, 0.9, 0.45),  # only catchable together with the abstention below
        _signal(True, 0.9, 0.9, 0.4),
    ]
    grids = {"sim_grid": [0.0, 0.5], "bm25_grid": [0.0]}
    balanced, _ = tune_thresholds(signals, answer_weight=1.0, **grids)
    favor_answer, _ = tune_thresholds(signals, answer_weight=3.0, **grids)
    assert balanced.min_sent_sim == 0.5
    assert favor_answer.min_sent_sim == 0.0


def test_evaluate_writes_logs_and_summary(engine, tmp_path):
    ingest(engine, "policy_handbook_v2.md", "policy_handbook", version="v2")
    questions = [
        {
            "qid": "t1",
            "split": "dev",
            "question": "refund window for digital goods",
            "relevant_chunk_ids": ["policy_handbook:v2:refunds-digital-goods:0"],
            "answerable": True,
            "user_visibility": ["public"],
            "tags": ["paraphrase"],
        },
        {
            "qid": "t2",
            "split": "dev",
            "question": "kayaks",
            "relevant_chunk_ids": [],
            "answerable": False,
            "user_visibility": ["public"],
            "tags": ["unanswerable"],
        },
    ]
    run = evaluate(engine, questions, mode="bm25", split="dev", runs_dir=tmp_path, run_id="r1")

    assert run.log_path == tmp_path / "r1_bm25_dev.jsonl"
    logged = [json.loads(line) for line in run.log_path.read_text().splitlines()]
    assert [q["qid"] for q in logged] == ["t1", "t2"]
    first = logged[0]
    for field in ("retrieved_ids", "scores", "ranks", "hits", "abstained"):
        assert field in first
    assert first["hits"][0] is True and first["rr"] == 1.0

    rows = list(csv.DictReader((tmp_path / "summary.csv").open()))
    assert [r["k"] for r in rows] == ["1", "3", "5", "10"]
    assert all(r["run_id"] == "r1" and r["mode"] == "bm25" for r in rows)
    assert float(rows[0]["recall"]) == 1.0  # only the answerable question counts
    assert run.recall(1) == 1.0 and run.recall(1, tag="keyword") == 0.0
