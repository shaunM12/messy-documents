import json
from datetime import datetime
from pathlib import Path
from typing import Annotated

import typer

from docsearch.answer import ExtractiveAnswerer, LLMAnswerer
from docsearch.answer import ask as answer_question
from docsearch.config import (
    ANSWER_CONTEXT_K,
    DB_PATH,
    EMBEDDING_CACHE_PATH,
    EVAL_SPLITS,
    HEADLINE_K,
    RUN_TIMESTAMP_FORMAT,
    SEARCH_MODES,
)
from docsearch.eval import evaluate, load_answer_key
from docsearch.index.dense import SentenceTransformerEmbedder
from docsearch.index.hybrid import SearchEngine
from docsearch.manifest import ingest_manifest
from docsearch.store import Store

app = typer.Typer(help="Messy document search + evaluation.", no_args_is_help=True)


def _csv(value: str) -> list[str]:
    return [v.strip() for v in value.split(",") if v.strip()]


def _engine(db_path: Path | str = DB_PATH) -> SearchEngine:
    return SearchEngine(Store(db_path), SentenceTransformerEmbedder(), EMBEDDING_CACHE_PATH)


@app.command()
def ingest(
    path: Annotated[Path, typer.Argument(exists=True, dir_okay=False)],
    doc_id: Annotated[str, typer.Option("--doc-id")],
    version: Annotated[str, typer.Option("--version")],
    updated_at: Annotated[str, typer.Option("--updated-at", help="ISO date, e.g. 2026-10-01.")],
    visibility: Annotated[str, typer.Option("--visibility", help="Comma-separated tags.")] = (
        "public"
    ),
):
    """Ingest a document, replacing every older chunk of the same doc id."""
    chunks = _engine().ingest(
        path, doc_id=doc_id, version=version, visibility=_csv(visibility), updated_at=updated_at
    )
    typer.echo(f"Ingested {len(chunks)} chunks for {doc_id} {version}.")


@app.command()
def delete(doc_id: Annotated[str, typer.Option("--doc-id")]):
    """Delete every chunk of a document."""
    removed = _engine().delete(doc_id)
    typer.echo(f"Deleted {removed} chunks for {doc_id}.")


@app.command()
def ask(
    question: str,
    user: Annotated[str, typer.Option(help="Comma-separated visibility tags.")] = "public",
    k: Annotated[int, typer.Option("--k")] = ANSWER_CONTEXT_K,
    llm: Annotated[bool, typer.Option("--llm", help="Use the optional LLM answerer.")] = False,
):
    """Answer a question with citations, or "I don't know"."""
    engine = _engine()
    answerer = LLMAnswerer() if llm else ExtractiveAnswerer(engine.embedder)
    result = answer_question(engine, question, _csv(user), answerer, k=k)
    typer.echo(json.dumps(result.to_dict(), indent=2))


@app.command("eval")
def eval_cmd(
    mode: Annotated[str, typer.Option(help="bm25, dense, hybrid or all.")] = "all",
    split: Annotated[str, typer.Option(help="dev, test or all.")] = "test",
    k: Annotated[str, typer.Option("--k", help="Comma-separated cutoffs.")] = "1,3,5,10",
):
    """Evaluate retrieval on a fresh index built from data/manifest.json."""
    modes = list(SEARCH_MODES) if mode == "all" else [mode]
    if any(m not in SEARCH_MODES for m in modes) or split not in EVAL_SPLITS:
        raise typer.BadParameter(f"mode in {SEARCH_MODES + ('all',)}, split in {EVAL_SPLITS}")
    ks = [int(x) for x in _csv(k)]

    engine = _engine(":memory:")
    ingest_manifest(engine)
    questions = load_answer_key(split)
    run_id = datetime.now().strftime(RUN_TIMESTAMP_FORMAT)

    typer.echo(f"{len(questions)} questions, split={split}, run_id={run_id}")
    typer.echo(f"{'mode':<8} {'k':>3} {'recall':>8} {'precision':>10} {'mrr':>7}")
    answerer = ExtractiveAnswerer(engine.embedder)
    runs = []
    for m in modes:
        run = evaluate(
            engine, questions, mode=m, split=split, ks=ks, run_id=run_id, answerer=answerer
        )
        runs.append(run)
        for row in run.summary:
            typer.echo(
                f"{m:<8} {row['k']:>3} {row['recall']:>8.3f} {row['precision']:>10.3f} "
                f"{row['mrr']:>7.3f}"
            )
    if HEADLINE_K in ks:
        typer.echo(f"\nRecall@{HEADLINE_K} by question tag:")
        for run in runs:
            typer.echo(
                f"  {run.mode:<8} keyword={run.recall(HEADLINE_K, 'keyword'):.3f} "
                f"paraphrase={run.recall(HEADLINE_K, 'paraphrase'):.3f}"
            )
    typer.echo("\nAbstention (extractive answerer):")
    for run in runs:
        typer.echo(
            f"  {run.mode:<8} abstention_accuracy={_pct(run.abstention_accuracy)} "
            f"false_abstention_rate={_pct(run.false_abstention_rate)}"
        )
    typer.echo(f"\nLogs: {runs[0].log_path.parent}")


def _pct(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.3f}"


if __name__ == "__main__":
    app()
