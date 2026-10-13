"""Structure-aware markdown chunking: one section per chunk, split oversize sections."""

import re

from docsearch.chunkers import Piece
from docsearch.config import (
    TEXT_MAX_HEADING_LEVEL,
    TEXT_MAX_WORDS,
    TEXT_OVERLAP_SENTENCES,
    TEXT_PREAMBLE_TITLE,
)
from docsearch.text_utils import split_sentences, word_count

_HEADING = re.compile(rf"^(#{{1,{TEXT_MAX_HEADING_LEVEL}}})\s+(.+?)\s*#*\s*$")


def _paragraphs(lines: list[str]) -> list[str]:
    paras, current = [], []
    for line in lines:
        if line.strip():
            current.append(line.strip())
        elif current:
            paras.append(" ".join(current))
            current = []
    if current:
        paras.append(" ".join(current))
    return paras


def _sections(markdown: str) -> list[tuple[list[str], list[str]]]:
    """(heading path, body paragraphs) for every section in document order."""
    sections = []
    path: list[str] = [TEXT_PREAMBLE_TITLE]
    stack: list[tuple[int, str]] = []
    body: list[str] = []
    for line in markdown.splitlines():
        m = _HEADING.match(line)
        if not m:
            body.append(line)
            continue
        sections.append((path, _paragraphs(body)))
        level, title = len(m.group(1)), m.group(2).strip()
        stack = [(lvl, t) for lvl, t in stack if lvl < level] + [(level, title)]
        path = [t for _, t in stack]
        body = []
    sections.append((path, _paragraphs(body)))
    return [(p, paras) for p, paras in sections if paras]


def _split_oversize(paragraphs: list[str]) -> list[list[str]]:
    """Pack paragraphs into groups <= TEXT_MAX_WORDS, overlapping by sentences."""
    units: list[str] = []
    for para in paragraphs:
        if word_count(para) <= TEXT_MAX_WORDS:
            units.append(para)
        else:
            units.extend(split_sentences(para))

    groups: list[list[str]] = []
    current: list[str] = []
    for unit in units:
        if current and word_count(" ".join(current + [unit])) > TEXT_MAX_WORDS:
            groups.append(current)
            overlap = split_sentences(" ".join(current))[-TEXT_OVERLAP_SENTENCES:]
            current = overlap if word_count(" ".join(overlap + [unit])) <= TEXT_MAX_WORDS else []
        current.append(unit)
    if current:
        groups.append(current)
    return groups


def chunk_text(markdown: str) -> list[Piece]:
    pieces = []
    for path, paragraphs in _sections(markdown):
        locator = " > ".join(path)
        if word_count(" ".join(paragraphs)) <= TEXT_MAX_WORDS:
            groups = [paragraphs]
        else:
            groups = _split_oversize(paragraphs)
        for part, group in enumerate(groups):
            pieces.append(
                Piece(
                    locator=locator,
                    text=f"{locator}: " + "\n\n".join(group),
                    metadata={"heading_path": path, "part": part, "parts": len(groups)},
                )
            )
    return pieces
