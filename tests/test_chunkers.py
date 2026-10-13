from pathlib import Path

import pandas as pd
import pytest

from docsearch.chunkers import chunk_file
from docsearch.chunkers.table_chunker import chunk_table
from docsearch.chunkers.text_chunker import chunk_text
from docsearch.config import PROJECT_ROOT, TABLE_ROWS_PER_CHUNK, TEXT_MAX_WORDS
from docsearch.models import missing_fields
from docsearch.text_utils import word_count

RAW = PROJECT_ROOT / "data" / "raw"
FILES = {
    "text": RAW / "policy_handbook_v2.md",
    "table": RAW / "pricing_table.csv",
    "layout": RAW / "product_flyer.pdf",
}


def _chunks(path: Path):
    return chunk_file(
        path, doc_id=path.stem, version="v1", visibility=["public"], updated_at="2026-10-01"
    )


@pytest.fixture(scope="module")
def corpus_chunks():
    return {doc_type: _chunks(path) for doc_type, path in FILES.items()}


@pytest.mark.parametrize("doc_type", FILES)
def test_chunk_schema(corpus_chunks, doc_type):
    chunks = corpus_chunks[doc_type]
    assert chunks
    for chunk in chunks:
        assert missing_fields(chunk) == [], chunk.chunk_id
        assert chunk.doc_type == doc_type
    assert len({c.chunk_id for c in chunks}) == len(chunks)


@pytest.mark.parametrize("doc_type", FILES)
def test_chunk_ids_deterministic(corpus_chunks, doc_type):
    again = _chunks(FILES[doc_type])
    assert [c.chunk_id for c in again] == [c.chunk_id for c in corpus_chunks[doc_type]]
    assert [c.text for c in again] == [c.text for c in corpus_chunks[doc_type]]


def test_text_chunker_respects_headings():
    md = (
        "Intro line.\n\n"
        "# Refunds\n\n"
        "## Physical goods\n\nSixty days.\n\n"
        "## Digital goods\n\nFourteen days.\n\nSecond paragraph.\n\n"
        "### Maps\n\nThree maps max.\n\n"
        "# Shipping\n\nShips fast.\n"
    )
    pieces = chunk_text(md)
    assert [p.locator for p in pieces] == [
        "Overview",
        "Refunds > Physical goods",
        "Refunds > Digital goods",
        "Refunds > Digital goods > Maps",
        "Shipping",
    ]
    digital = pieces[2]
    assert digital.text == "Refunds > Digital goods: Fourteen days.\n\nSecond paragraph."
    assert "Three maps" not in digital.text
    assert "Fourteen" not in pieces[3].text


def test_text_chunker_splits_oversize_section_on_paragraphs_with_overlap():
    paras = [" ".join([f"w{i}"] * 150) + f". Last sentence of para {i}." for i in range(4)]
    pieces = chunk_text("# Long\n\n" + "\n\n".join(paras))
    assert len(pieces) > 1
    assert {p.locator for p in pieces} == {"Long"}
    for p in pieces:
        assert word_count(p.text.removeprefix("Long: ")) <= TEXT_MAX_WORDS
    for prev, nxt in zip(pieces, pieces[1:], strict=False):
        last_sentence = prev.text.rsplit(". ", 1)[-1]
        assert nxt.text.removeprefix("Long: ").startswith(last_sentence)


def test_table_chunker_repeats_headers_and_no_split_rows():
    df = pd.DataFrame(
        {"sku": [f"NW-{i:04d}" for i in range(25)], "price": [str(i) for i in range(25)]}
    )
    pieces = chunk_table(df, title="Table: t")
    row_pieces = [p for p in pieces if not p.metadata.get("summary")]
    assert [p.locator for p in row_pieces] == ["rows 1-10", "rows 11-20", "rows 21-25"]
    seen = []
    for p in row_pieces:
        lines = p.text.splitlines()
        assert lines[:2] == ["Table: t", "Columns: sku, price"]
        assert len(lines) - 2 <= TABLE_ROWS_PER_CHUNK
        seen.extend(lines[2:])
    assert seen == [f"sku: NW-{i:04d}; price: {i}" for i in range(25)]

    summary = pieces[-1]
    assert summary.metadata["summary"] and summary.metadata["row_count"] == 25
    assert "Row count: 25" in summary.text


def test_table_chunker_groups_by_key_column(corpus_chunks):
    chunks = corpus_chunks["table"]
    df = pd.read_csv(FILES["table"], dtype=str)
    for chunk in chunks:
        assert chunk.text.splitlines()[1].startswith("Columns: plan, sku")
        if chunk.metadata.get("summary"):
            continue
        plan = chunk.locator.removeprefix("plan=")
        rows = chunk.text.splitlines()[2:]
        assert rows and all(r.startswith(f"plan: {plan};") for r in rows)
    serialized = [line for c in chunks for line in c.text.splitlines() if line.startswith("plan:")]
    assert len(serialized) == len(df)


def test_layout_chunker_strips_headers_footers_and_keeps_column_order(corpus_chunks):
    chunks = corpus_chunks["layout"]
    text = "\n".join(c.text for c in chunks)
    assert "Spring 2026 Product Flyer" not in text
    assert "Page 1 of 3" not in text and "northwindgear.example |" not in text

    page1 = [c for c in chunks if c.metadata["page"] == 1]
    starts = [c.text.split(" (")[0] for c in page1]
    assert starts == [
        "Ridgeline 2P Tent",
        "Ridgeline 3P Tent",
        "Alpine Down Sleeping Bag",
        "Alpine Synthetic Sleeping Bag",
        "Quick facts 68% of the nylon in our tents is recycled. Every tent is seam taped by hand.",
    ]
    assert [c.metadata["column"] for c in page1] == [1, 1, 2, 2, 3]
    for c in chunks:
        assert not ("Ridgeline" in c.text and "Alpine" in c.text), "block crossed columns"


def test_layout_chunker_routes_embedded_table(corpus_chunks):
    chunks = corpus_chunks["layout"]
    tables = [c for c in chunks if c.metadata.get("embedded_table")]
    assert tables
    assert all(c.locator.startswith("page 3 table 1") for c in tables)
    rows = next(c for c in tables if not c.metadata.get("summary"))
    assert "Columns: SKU, Product, Weight, Waterproof rating, List price" in rows.text
    assert "SKU: NW-5100; Product: Stormshell Jacket; Weight: 410 g" in rows.text
    blocks = [c for c in chunks if not c.metadata.get("embedded_table")]
    assert not any("Waterproof rating" in c.text for c in blocks)
