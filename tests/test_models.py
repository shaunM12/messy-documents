from dataclasses import fields

import pytest

from docsearch.models import Chunk, make_chunk_id, missing_fields, slug


def _chunk(**overrides) -> Chunk:
    base = dict(
        chunk_id="policy_handbook:v2:refunds-digital-goods:0",
        doc_id="policy_handbook",
        source_file="data/raw/policy_handbook_v2.md",
        doc_type="text",
        locator="Refunds > Digital goods",
        text="Refunds > Digital goods: 14 days.",
        visibility=["public"],
        version="v2",
        updated_at="2026-10-01",
        metadata={"heading_path": ["Refunds", "Digital goods"]},
    )
    base.update(overrides)
    return Chunk(**base)


def test_chunk_has_exactly_the_schema_fields():
    assert [f.name for f in fields(Chunk)] == [
        "chunk_id",
        "doc_id",
        "source_file",
        "doc_type",
        "locator",
        "text",
        "visibility",
        "version",
        "updated_at",
        "metadata",
    ]


def test_valid_chunk_has_no_missing_fields():
    assert missing_fields(_chunk()) == []


@pytest.mark.parametrize("field", [f.name for f in fields(Chunk)])
def test_empty_field_is_reported(field):
    empty = {"visibility": [], "metadata": {}}.get(field, "")
    assert field in missing_fields(_chunk(**{field: empty}))


def test_unknown_doc_type_is_reported():
    assert "doc_type" in missing_fields(_chunk(doc_type="image"))


@pytest.mark.parametrize(
    ("locator", "expected"),
    [
        ("Refunds > Digital goods", "refunds-digital-goods"),
        ("page 2 block 3", "page-2-block-3"),
        ("rows 10-19", "rows-10-19"),
        ("plan=Enterprise", "plan-enterprise"),
        ("  --Weird!!  ", "weird"),
        ("a" * 50, "a" * 40),
        ("x" * 39 + " yz", "x" * 39),
    ],
)
def test_slug(locator, expected):
    assert slug(locator) == expected


def test_make_chunk_id_format():
    assert (
        make_chunk_id("policy_handbook", "v2", "Refunds > Digital goods", 0)
        == "policy_handbook:v2:refunds-digital-goods:0"
    )
