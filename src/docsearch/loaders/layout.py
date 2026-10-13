from dataclasses import dataclass, field
from pathlib import Path

import pdfplumber


@dataclass(frozen=True)
class Word:
    text: str
    x0: float
    x1: float
    top: float
    bottom: float


@dataclass(frozen=True)
class PageTable:
    bbox: tuple[float, float, float, float]  # x0, top, x1, bottom
    rows: list[list[str]]


@dataclass
class Page:
    number: int  # 1-based
    words: list[Word] = field(default_factory=list)
    tables: list[PageTable] = field(default_factory=list)


def load_layout(path: Path) -> list[Page]:
    pages = []
    with pdfplumber.open(path) as pdf:
        for p in pdf.pages:
            words = [
                Word(w["text"], w["x0"], w["x1"], w["top"], w["bottom"]) for w in p.extract_words()
            ]
            tables = [
                PageTable(
                    bbox=tuple(t.bbox),
                    rows=[[(cell or "").strip() for cell in row] for row in t.extract()],
                )
                for t in p.find_tables()
            ]
            pages.append(Page(number=p.page_number, words=words, tables=tables))
    return pages
