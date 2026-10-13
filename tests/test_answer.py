import json

import pytest
from conftest import ingest

from docsearch.answer import ExtractiveAnswerer, LLMAnswerer, ask
from docsearch.config import IDK
from docsearch.index.hybrid import Retrieval

# Thresholds for the offline hashing embedder; production values are tuned on dev in config.py.
THRESHOLDS = {"min_sim": 0.3, "min_bm25": 0.6}
SENT = {"min_sent_sim": 0.25}
INTERNAL = "internal_ops_notes:v1:supplier-costs-ridgeline-tents:0"


@pytest.fixture
def corpus(engine):
    ingest(engine, "policy_handbook_v2.md", "policy_handbook", version="v2")
    ingest(engine, "internal_ops_notes.md", "internal_ops_notes", visibility=["role:internal"])
    return engine


@pytest.fixture
def extractive(engine):
    return ExtractiveAnswerer(engine.embedder, **THRESHOLDS, **SENT)


def test_abstains_on_unanswerable_and_permission(corpus, extractive):
    landed = "What is the landed cost of the Ridgeline 2P Tent?"
    permission = ask(corpus, landed, ["public"], extractive)
    assert permission.to_dict() == {"answer": IDK, "citations": [], "abstained": True}

    unanswerable = ask(corpus, "What is the CEO phone number?", ["public"], extractive)
    assert unanswerable.abstained and unanswerable.answer == IDK

    staff = ask(corpus, landed, ["role:internal"], extractive)
    assert not staff.abstained
    assert "$118" in staff.answer
    assert [c["chunk_id"] for c in staff.citations] == [INTERNAL]


def test_answers_with_full_citation(corpus, extractive):
    result = ask(corpus, "How much does overnight shipping cost?", ["public"], extractive)
    assert not result.abstained
    assert "$29.95" in result.answer
    assert result.citations == [
        {
            "chunk_id": "policy_handbook:v2:shipping-delivery-times:0",
            "source_file": "data/raw/policy_handbook_v2.md",
            "locator": "Shipping > Delivery times",
            "version": "v2",
            "updated_at": "2026-10-01",
        }
    ]
    assert set(result.to_dict()) == {"answer", "citations", "abstained"}


def test_abstains_when_no_visible_chunks(engine, extractive):
    ingest(engine, "internal_ops_notes.md", "internal_ops_notes", visibility=["role:internal"])
    result = ask(engine, "landed cost of the Ridgeline tent", ["public"], extractive)
    assert result.abstained and result.reason == "no visible chunks"


def test_weak_retrieval_needs_both_signals_low(engine):
    answerer = ExtractiveAnswerer(engine.embedder, min_sim=0.5, min_bm25=0.5, min_sent_sim=0.0)
    ingest(engine, "policy_handbook_v2.md", "policy_handbook", version="v2")
    results = engine.search("overnight shipping cost", ["public"])
    weak = Retrieval(results=results, top_bm25=0.49, top_dense=0.49)
    assert answerer.answer("overnight shipping cost", weak).abstained
    for strong in (Retrieval(results, 0.5, 0.0), Retrieval(results, 0.0, 0.5)):
        assert not answerer.answer("overnight shipping cost", strong).abstained


def test_abstains_on_low_sentence_similarity(corpus):
    strict = ExtractiveAnswerer(corpus.embedder, **THRESHOLDS, min_sent_sim=0.99)
    result = ask(corpus, "How much does overnight shipping cost?", ["public"], strict)
    assert result.abstained and "sentence similarity" in result.reason


def test_answer_cites_new_version_after_replace(engine, extractive):
    question = "What is the refund window for digital goods?"
    ingest(engine, "policy_handbook_v1.md", "policy_handbook", version="v1")
    before = ask(engine, question, ["public"], extractive)
    assert "30 days" in before.answer and before.citations[0]["version"] == "v1"

    ingest(engine, "policy_handbook_v2.md", "policy_handbook", version="v2")
    after = ask(engine, question, ["public"], extractive)
    assert "14 days" in after.answer
    assert [c["version"] for c in after.citations] == ["v2"]
    assert after.citations[0]["chunk_id"] == "policy_handbook:v2:refunds-digital-goods:0"


def _llm(reply: dict | str, seen: list | None = None):
    def complete(prompt: str) -> str:
        if seen is not None:
            seen.append(prompt)
        return reply if isinstance(reply, str) else json.dumps(reply)

    return LLMAnswerer(complete, **THRESHOLDS)


def test_llm_answerer_uses_fixed_prompt_and_valid_citations(corpus):
    cid = "policy_handbook:v2:shipping-delivery-times:0"
    prompts: list[str] = []
    answerer = _llm({"answer": "$29.95", "citations": [cid]}, prompts)
    result = ask(corpus, "How much does overnight shipping cost?", ["public"], answerer)
    assert not result.abstained and result.answer == "$29.95"
    assert result.citations[0]["chunk_id"] == cid
    assert f"[{cid}]" in prompts[0] and "Question: How much does overnight" in prompts[0]


@pytest.mark.parametrize(
    "reply",
    [
        {"answer": "I don't know", "citations": []},
        {"answer": "$29.95", "citations": ["made:up:chunk:0"]},
        {"answer": "$29.95", "citations": []},
        "not json at all",
    ],
)
def test_llm_answerer_abstains(corpus, reply):
    result = ask(corpus, "How much does overnight shipping cost?", ["public"], _llm(reply))
    assert result.abstained and result.answer == IDK and result.citations == []


def test_llm_answerer_cannot_cite_hidden_chunk(corpus):
    answerer = _llm({"answer": "$118", "citations": [INTERNAL]})
    result = ask(corpus, "landed cost of the Ridgeline 2P Tent", ["public"], answerer)
    assert result.abstained
