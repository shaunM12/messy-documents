"""Row-group chunking: never split a row, repeat headers in every chunk, add a summary."""

from pathlib import PurePosixPath

import pandas as pd

from docsearch.chunkers import Piece
from docsearch.config import TABLE_KEY_COLUMNS, TABLE_ROWS_PER_CHUNK, TABLE_SUMMARY_LOCATOR


def table_title(source_file: str) -> str:
    stem = PurePosixPath(source_file).stem.replace("_", " ")
    return f"Table: {stem} ({PurePosixPath(source_file).name})"


def _key_column(columns: list[str]) -> str | None:
    lowered = {c.lower(): c for c in columns}
    return next((lowered[k] for k in TABLE_KEY_COLUMNS if k in lowered), None)


def _serialize_row(row: pd.Series, columns: list[str]) -> str:
    return "; ".join(f"{c}: {row[c]}" for c in columns)


def _batches(df: pd.DataFrame) -> list[pd.DataFrame]:
    return [df.iloc[i : i + TABLE_ROWS_PER_CHUNK] for i in range(0, len(df), TABLE_ROWS_PER_CHUNK)]


def chunk_table(
    df: pd.DataFrame, *, title: str, locator_prefix: str = "", extra_metadata: dict | None = None
) -> list[Piece]:
    columns = [str(c) for c in df.columns]
    df = df.reset_index(drop=True)
    extra = extra_metadata or {}
    header = f"{title}\nColumns: {', '.join(columns)}"
    key = _key_column(columns)

    groups: list[tuple[str, pd.DataFrame]] = []
    if key:
        for value, group in df.groupby(key, sort=False):
            groups.extend((f"{key}={value}", batch) for batch in _batches(group))
    else:
        for batch in _batches(df):
            first, last = batch.index[0] + 1, batch.index[-1] + 1
            groups.append((f"rows {first}-{last}", batch))

    def locator(name: str) -> str:
        return f"{locator_prefix} {name}".strip()

    pieces = []
    for name, batch in groups:
        rows = "\n".join(_serialize_row(row, columns) for _, row in batch.iterrows())
        pieces.append(
            Piece(
                locator=locator(name),
                text=f"{header}\n{rows}",
                metadata={
                    **extra,
                    "columns": columns,
                    "row_start": int(batch.index[0]) + 1,
                    "row_end": int(batch.index[-1]) + 1,
                },
            )
        )

    summary = f"{header}\nRow count: {len(df)}"
    if key:
        summary += f"\n{key} values: {', '.join(map(str, df[key].unique()))}"
    pieces.append(
        Piece(
            locator=locator(TABLE_SUMMARY_LOCATOR),
            text=summary,
            metadata={**extra, "columns": columns, "row_count": len(df), "summary": True},
        )
    )
    return pieces
