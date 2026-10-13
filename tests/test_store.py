import sqlite3
from dataclasses import replace

import pytest

from docsearch.config import PROJECT_ROOT
from docsearch.store import Store

RAW = PROJECT_ROOT / "data" / "raw"


@pytest.fixture
def store():
    s = Store(":memory:")
    yield s
    s.close()


def _ingest(store, version, **kw):
    return store.ingest(
        RAW / f"policy_handbook_{version}.md",
        doc_id="policy_handbook",
        version=version,
        visibility=kw.get("visibility", ["public"]),
        updated_at=kw.get("updated_at", "2026-10-01"),
    )


def test_ingest_roundtrips_chunks(store):
    chunks = _ingest(store, "v1")
    assert store.all_chunks() == chunks


def test_replace_store_keeps_only_new_version(store):
    _ingest(store, "v1")
    v2 = _ingest(store, "v2")
    stored = store.all_chunks()
    assert stored == v2
    assert all(c.version == "v2" and ":v1:" not in c.chunk_id for c in stored)
    assert store.doc_versions() == {"policy_handbook": ["v2"]}


def test_replace_does_not_touch_other_docs(store):
    store.ingest(
        RAW / "pricing_table.csv",
        doc_id="pricing_table",
        version="v1",
        visibility=["public"],
        updated_at="2026-09-15",
    )
    _ingest(store, "v1")
    _ingest(store, "v2")
    assert store.doc_versions() == {"pricing_table": ["v1"], "policy_handbook": ["v2"]}


def test_failed_replace_rolls_back(store):
    v1 = _ingest(store, "v1")
    bad = [replace(v1[0], version="v2", chunk_id="x:v2:a:0"), replace(v1[1], text="")]
    with pytest.raises(ValueError):
        store.replace("policy_handbook", bad)
    duplicate = [replace(v1[0], version="v2"), replace(v1[0], version="v2")]
    with pytest.raises(sqlite3.IntegrityError):
        store.replace("policy_handbook", duplicate)
    assert store.all_chunks() == v1


def test_delete_removes_all_chunks_of_doc(store):
    _ingest(store, "v2")
    assert store.delete("policy_handbook") > 0
    assert store.all_chunks() == []
    assert store.delete("policy_handbook") == 0
