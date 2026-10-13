import pytest
from conftest import ingest

from docsearch.config import RRF_K, SEARCH_MODES
from docsearch.index import is_visible
from docsearch.index.bm25 import tokenize
from docsearch.index.hybrid import rrf
from docsearch.models import Chunk

DIGITAL_V1 = "policy_handbook:v1:refunds-digital-goods:0"
DIGITAL_V2 = "policy_handbook:v2:refunds-digital-goods:0"
REFUND_QUESTION = "How many days do I have to get a refund on digital goods?"


def _ids(results):
    return [r.chunk.chunk_id for r in results]


def test_rrf_fusion():
    fused = rrf([["a", "b", "c"], ["c", "a", "d"]])
    expected = {
        "a": 1 / (RRF_K + 1) + 1 / (RRF_K + 2),
        "b": 1 / (RRF_K + 2),
        "c": 1 / (RRF_K + 3) + 1 / (RRF_K + 1),
        "d": 1 / (RRF_K + 3),
    }
    assert [item for item, _ in fused] == ["a", "c", "b", "d"]
    for item, score in fused:
        assert score == pytest.approx(expected[item])


def test_rrf_ties_break_by_best_rank_then_id():
    fused = rrf([["x", "y"], ["y", "x"], ["z"]])
    assert [item for item, _ in fused] == ["x", "y", "z"]
    assert rrf([["b"], ["a"]]) == [("a", 1 / (RRF_K + 1)), ("b", 1 / (RRF_K + 1))]


def test_bm25_tokenizer_keeps_numbers_and_skus():
    toks = tokenize("Ridgeline (NW-1042) costs $329.00, boils in 3.5 min; USB-C!")
    for expected in ["nw-1042", "1042", "329.00", "3.5", "usb-c", "ridgeline"]:
        assert expected in toks


def _chunk(visibility):
    return Chunk("d:v1:x:0", "d", "f.md", "text", "x", "t", visibility, "v1", "2026", {"a": 1})


@pytest.mark.parametrize(
    ("chunk_vis", "user_vis", "visible"),
    [
        (["public"], ["public"], True),
        (["public"], [], True),
        (["role:internal"], ["public"], False),
        (["role:internal"], ["public", "role:internal"], True),
        (["customer:acme"], ["customer:globex"], False),
        (["customer:acme", "role:internal"], ["role:internal"], True),
    ],
)
def test_visibility_rule(chunk_vis, user_vis, visible):
    assert is_visible(_chunk(chunk_vis), user_vis) is visible


@pytest.mark.parametrize("mode", SEARCH_MODES)
def test_visibility_filter(engine, mode):
    ingest(engine, "policy_handbook_v2.md", "policy_handbook")
    ingest(engine, "internal_ops_notes.md", "internal_ops_notes", visibility=["role:internal"])
    question = "landed cost of the Ridgeline 2P tent supplier"

    public = engine.search(question, ["public"], k=50, mode=mode)
    assert public and all(r.chunk.doc_id == "policy_handbook" for r in public)

    staff = engine.search(question, ["public", "role:internal"], k=5, mode=mode)
    assert staff[0].chunk.chunk_id == "internal_ops_notes:v1:supplier-costs-ridgeline-tents:0"


@pytest.mark.parametrize("mode", SEARCH_MODES)
def test_replace_removes_old_version(engine, mode):
    ingest(engine, "policy_handbook_v1.md", "policy_handbook", version="v1")
    before = engine.search(REFUND_QUESTION, ["public"], k=3, mode=mode)
    assert DIGITAL_V1 in _ids(before)

    ingest(engine, "policy_handbook_v2.md", "policy_handbook", version="v2")
    for m in SEARCH_MODES:
        everything = engine.search(REFUND_QUESTION, ["public"], k=1000, mode=m)
        assert not any(":v1:" in cid for cid in _ids(everything)), m
    after = engine.search(REFUND_QUESTION, ["public"], k=3, mode=mode)
    assert DIGITAL_V2 in _ids(after)
    assert "within 30 days of purchase" not in " ".join(r.chunk.text for r in after)


def test_delete_removes_from_both_indexes(engine):
    ingest(engine, "policy_handbook_v2.md", "policy_handbook")
    ingest(engine, "pricing_table.csv", "pricing_table")
    assert engine.search("NW-1042 Enterprise price", ["public"], mode="bm25")

    engine.delete("pricing_table")
    assert all(c.doc_id != "pricing_table" for c in engine.bm25.chunks)
    assert all(c.doc_id != "pricing_table" for c in engine.dense.chunks)
    assert len(engine.dense.vectors) == len(engine.bm25.chunks)
    for mode in SEARCH_MODES:
        results = engine.search("NW-1042 Enterprise price", ["public"], k=1000, mode=mode)
        assert all(r.chunk.doc_id != "pricing_table" for r in results), mode


def test_normalized_bm25_is_bounded_and_stable_as_corpus_grows(engine):
    question = "What is the refund window for digital goods?"
    ingest(engine, "policy_handbook_v1.md", "policy_handbook", version="v1")
    alone = engine.retrieve(question, ["public"]).top_bm25
    ingest(engine, "pricing_table.csv", "pricing_table")
    ingest(engine, "product_flyer.pdf", "product_flyer")
    grown = engine.retrieve(question, ["public"]).top_bm25
    assert 0.0 < alone <= 1.0 and 0.0 < grown <= 1.0
    assert grown == pytest.approx(alone, abs=0.1)


def test_search_on_empty_index_returns_nothing(engine):
    for mode in SEARCH_MODES:
        assert engine.search("anything", ["public"], mode=mode) == []


def test_ranks_are_one_based_and_k_respected(engine):
    ingest(engine, "policy_handbook_v2.md", "policy_handbook")
    for mode in SEARCH_MODES:
        results = engine.search("shipping", ["public"], k=3, mode=mode)
        assert [r.rank for r in results] == [1, 2, 3]
