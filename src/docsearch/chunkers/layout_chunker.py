"""Layout-aware PDF chunking: strip running headers/footers, read column by column,
split into blocks on vertical gaps, and route embedded tables to the table chunker."""

import math
import re
from dataclasses import dataclass

import pandas as pd

from docsearch.chunkers import Piece
from docsearch.chunkers.table_chunker import chunk_table
from docsearch.config import (
    LAYOUT_BLOCK_GAP,
    LAYOUT_COLUMN_GAP,
    LAYOUT_HEADER_FOOTER_MIN_PAGES,
    LAYOUT_HEADER_FOOTER_PAGE_FRACTION,
    LAYOUT_HEADER_FOOTER_Y_TOLERANCE,
    LAYOUT_LINE_TOLERANCE,
    LAYOUT_WORD_GAP,
)
from docsearch.loaders.layout import Page, Word


@dataclass(frozen=True)
class Fragment:
    """A run of words on one line with no wide horizontal gap."""

    text: str
    x0: float
    x1: float
    top: float
    bottom: float


def _fragments(words: list[Word]) -> list[Fragment]:
    lines: list[list[Word]] = []
    for w in sorted(words, key=lambda w: (w.top, w.x0)):
        if lines and abs(w.top - lines[-1][0].top) < LAYOUT_LINE_TOLERANCE:
            lines[-1].append(w)
        else:
            lines.append([w])

    frags = []
    for line in lines:
        run: list[Word] = []
        for w in sorted(line, key=lambda w: w.x0):
            if run and w.x0 - run[-1].x1 > LAYOUT_WORD_GAP:
                frags.append(_merge(run))
                run = []
            run.append(w)
        frags.append(_merge(run))
    return frags


def _merge(words: list[Word]) -> Fragment:
    return Fragment(
        text=" ".join(w.text for w in words),
        x0=min(w.x0 for w in words),
        x1=max(w.x1 for w in words),
        top=min(w.top for w in words),
        bottom=max(w.bottom for w in words),
    )


def _normalize(text: str) -> str:
    return re.sub(r"\d+", "#", text.lower()).strip()


def _running_lines(pages: list[list[Fragment]]) -> set[tuple[int, int]]:
    """(page index, fragment index) of lines repeated at a similar y on enough pages."""
    needed = max(
        LAYOUT_HEADER_FOOTER_MIN_PAGES, math.ceil(LAYOUT_HEADER_FOOTER_PAGE_FRACTION * len(pages))
    )
    found = set()
    for pi, frags in enumerate(pages):
        for fi, frag in enumerate(frags):
            norm = _normalize(frag.text)
            hits = sum(
                any(
                    _normalize(o.text) == norm
                    and abs(o.top - frag.top) <= LAYOUT_HEADER_FOOTER_Y_TOLERANCE
                    for o in other
                )
                for other in pages
            )
            if hits >= needed:
                found.add((pi, fi))
    return found


def _inside(frag: Fragment, bbox: tuple[float, float, float, float]) -> bool:
    cx, cy = (frag.x0 + frag.x1) / 2, (frag.top + frag.bottom) / 2
    x0, top, x1, bottom = bbox
    return x0 <= cx <= x1 and top <= cy <= bottom


def _columns(frags: list[Fragment]) -> list[list[Fragment]]:
    """Cluster fragments by left edge; columns ordered left to right."""
    clusters: list[list[Fragment]] = []
    for frag in sorted(frags, key=lambda f: f.x0):
        if clusters and frag.x0 - clusters[-1][-1].x0 > LAYOUT_COLUMN_GAP:
            clusters.append([])
        if not clusters:
            clusters.append([])
        clusters[-1].append(frag)
    return [sorted(c, key=lambda f: f.top) for c in clusters]


def _blocks(column: list[Fragment]) -> list[list[Fragment]]:
    blocks: list[list[Fragment]] = []
    for frag in column:
        if blocks and frag.top - blocks[-1][-1].bottom <= LAYOUT_BLOCK_GAP:
            blocks[-1].append(frag)
        else:
            blocks.append([frag])
    return blocks


def _table_frame(rows: list[list[str]]) -> pd.DataFrame | None:
    if len(rows) < 2:
        return None
    header = [h or f"column {i + 1}" for i, h in enumerate(rows[0])]
    return pd.DataFrame(rows[1:], columns=header)


def chunk_layout(pages: list[Page], *, source_file: str) -> list[Piece]:
    page_frags = [_fragments(p.words) for p in pages]
    running = _running_lines(page_frags)

    pieces = []
    for pi, (page, frags) in enumerate(zip(pages, page_frags, strict=True)):
        body = [
            f
            for fi, f in enumerate(frags)
            if (pi, fi) not in running and not any(_inside(f, t.bbox) for t in page.tables)
        ]
        block_no = 0
        for col_no, column in enumerate(_columns(body), start=1):
            for block in _blocks(column):
                block_no += 1
                bbox = [
                    round(min(f.x0 for f in block), 1),
                    round(min(f.top for f in block), 1),
                    round(max(f.x1 for f in block), 1),
                    round(max(f.bottom for f in block), 1),
                ]
                pieces.append(
                    Piece(
                        locator=f"page {page.number} block {block_no}",
                        text=" ".join(f.text for f in block),
                        metadata={
                            "page": page.number,
                            "block": block_no,
                            "column": col_no,
                            "bbox": bbox,
                        },
                    )
                )
        for ti, table in enumerate(page.tables, start=1):
            df = _table_frame(table.rows)
            if df is None:
                continue
            pieces.extend(
                chunk_table(
                    df,
                    title=f"Table {ti} on page {page.number} of {source_file}",
                    locator_prefix=f"page {page.number} table {ti}",
                    extra_metadata={
                        "page": page.number,
                        "bbox": [round(v, 1) for v in table.bbox],
                        "embedded_table": True,
                    },
                )
            )
    return pieces
