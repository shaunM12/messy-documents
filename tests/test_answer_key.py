import json
from collections import Counter

import pytest

from docsearch.config import PROJECT_ROOT
from docsearch.index import is_visible
from docsearch.manifest import ingest_manifest
from docsearch.store import Store

ANSWER_KEY = PROJECT_ROOT / "data" / "eval" / "answer_key.jsonl"


@pytest.fixture(scope="module")
def key():
    return [json.loads(line) for line in ANSWER_KEY.read_text().splitlines() if line.strip()]


@pytest.fixture(scope="module")
def corpus():
    store = Store(":memory:")
    ingest_manifest(store)
    chunks = {c.chunk_id: c for c in store.all_chunks()}
    store.close()
    return chunks


def test_answer_key_ids_exist(key, corpus):
    for q in key:
        for cid in q["relevant_chunk_ids"] + q.get("hidden_chunk_ids", []):
            assert cid in corpus, f"{q['qid']}: {cid}"


def test_relevant_chunks_are_visible_and_hidden_chunks_are_not(key, corpus):
    for q in key:
        for cid in q["relevant_chunk_ids"]:
            assert is_visible(corpus[cid], q["user_visibility"]), q["qid"]
        for cid in q.get("hidden_chunk_ids", []):
            assert not is_visible(corpus[cid], q["user_visibility"]), q["qid"]


def test_answer_key_required_mix(key):
    assert len(key) >= 25
    assert len({q["qid"] for q in key}) == len(key)
    answerable = [q for q in key if q["answerable"]]
    types = Counter(q["doc_type"] for q in answerable)
    assert types["text"] >= 6 and types["table"] >= 6 and types["layout"] >= 6
    tags = Counter(t for q in key for t in q["tags"])
    assert tags["unanswerable"] >= 3
    assert tags["permission"] >= 2
    assert tags["version"] >= 2
    assert tags["keyword"] >= 1 and tags["paraphrase"] >= 1
    for q in key:
        assert q["split"] in {"dev", "test"}
        assert bool(q["relevant_chunk_ids"]) == q["answerable"], q["qid"]
    splits = Counter(q["split"] for q in key)
    assert 8 <= splits["dev"] < splits["test"]
    dev_tags = {t for q in key if q["split"] == "dev" for t in q["tags"]}
    assert {"unanswerable", "permission"} <= dev_tags, "thresholds need abstention cases on dev"


def test_version_questions_point_at_current_version(key, corpus):
    for q in key:
        if "version" in q["tags"]:
            assert all(corpus[cid].version == "v2" for cid in q["relevant_chunk_ids"]), q["qid"]
