"""Answer step: extractive by default, optional LLM (D4). Abstains with "I don't know"."""

import json
import os
import re
import urllib.request
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from typing import Protocol

import numpy as np

from docsearch.config import (
    ANSWER_CONTEXT_K,
    ANSWER_MODE,
    IDK,
    LLM_API_KEY_ENV,
    LLM_API_URL,
    LLM_MODEL,
    LLM_TEMPERATURE,
    LLM_TIMEOUT_SECONDS,
    MIN_BM25,
    MIN_SENT_SIM,
    MIN_SIM,
)
from docsearch.index.dense import Embedder
from docsearch.index.hybrid import Retrieval, SearchEngine
from docsearch.models import Chunk, ScoredChunk
from docsearch.text_utils import split_sentences

LLM_PROMPT = """You answer questions about Northwind Gear using ONLY the context chunks below.
Each chunk starts with its chunk_id in square brackets.
Reply with a single JSON object and nothing else:
{{"answer": "<short answer>", "citations": ["<chunk_id>", ...]}}
Cite only chunk_ids that appear in the context and that support the answer.
If the context does not contain the answer, reply exactly:
{{"answer": "I don't know", "citations": []}}

Context:
{context}

Question: {question}
"""


@dataclass(frozen=True)
class Answer:
    answer: str
    citations: list[dict]
    abstained: bool
    reason: str = field(default="", compare=False)

    def to_dict(self) -> dict:
        return {"answer": self.answer, "citations": self.citations, "abstained": self.abstained}


def citation(chunk: Chunk) -> dict:
    return {
        "chunk_id": chunk.chunk_id,
        "source_file": chunk.source_file,
        "locator": chunk.locator,
        "version": chunk.version,
        "updated_at": chunk.updated_at,
    }


def abstain(reason: str) -> Answer:
    return Answer(answer=IDK, citations=[], abstained=True, reason=reason)


def weak_evidence(retrieval: Retrieval, min_sim: float, min_bm25: float) -> str | None:
    """Rules 1 and 2: no visible chunks, or both retrievers below threshold."""
    if not retrieval.results:
        return "no visible chunks"
    if retrieval.top_dense < min_sim and retrieval.top_bm25 < min_bm25:
        return f"weak retrieval (dense {retrieval.top_dense:.3f}, bm25 {retrieval.top_bm25:.2f})"
    return None


def answer_units(chunk: Chunk) -> list[str]:
    """Candidate answer spans: table rows for tables, sentences otherwise."""
    if "columns" in chunk.metadata:
        return [line for line in chunk.text.splitlines()[1:] if line.strip()]
    text = chunk.text.removeprefix(f"{chunk.locator}: ") if chunk.doc_type == "text" else chunk.text
    return split_sentences(text)


class Answerer(Protocol):
    def answer(self, question: str, retrieval: Retrieval) -> Answer: ...


class ExtractiveAnswerer:
    def __init__(
        self,
        embedder: Embedder,
        *,
        min_sim: float = MIN_SIM,
        min_bm25: float = MIN_BM25,
        min_sent_sim: float = MIN_SENT_SIM,
        context_k: int = ANSWER_CONTEXT_K,
    ):
        self.embedder = embedder
        self.min_sim = min_sim
        self.min_bm25 = min_bm25
        self.min_sent_sim = min_sent_sim
        self.context_k = context_k

    def best_sentence(
        self, question: str, results: Iterable[ScoredChunk]
    ) -> tuple[float, str, Chunk] | None:
        context = list(results)[: self.context_k]
        candidates = [(unit, r.chunk) for r in context for unit in answer_units(r.chunk)]
        if not candidates:
            return None
        vecs = self.embedder.encode([question] + [unit for unit, _ in candidates])
        sims = vecs[1:] @ vecs[0]
        best = int(np.argmax(sims))
        return float(sims[best]), candidates[best][0], candidates[best][1]

    def answer(self, question: str, retrieval: Retrieval) -> Answer:
        if reason := weak_evidence(retrieval, self.min_sim, self.min_bm25):
            return abstain(reason)
        best = self.best_sentence(question, retrieval.results)
        if best is None:
            return abstain("no answer sentences")
        sim, sentence, chunk = best
        if sim < self.min_sent_sim:
            return abstain(f"best sentence similarity {sim:.3f} < {self.min_sent_sim}")
        return Answer(answer=sentence, citations=[citation(chunk)], abstained=False)


def openai_complete(prompt: str) -> str:
    """Chat completion against an OpenAI-compatible endpoint at temperature 0."""
    key = os.environ.get(LLM_API_KEY_ENV)
    if not key:
        raise RuntimeError(f"Set {LLM_API_KEY_ENV} to use --llm")
    body = {
        "model": LLM_MODEL,
        "temperature": LLM_TEMPERATURE,
        "messages": [{"role": "user", "content": prompt}],
    }
    req = urllib.request.Request(
        LLM_API_URL,
        data=json.dumps(body).encode(),
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=LLM_TIMEOUT_SECONDS) as resp:
        return json.load(resp)["choices"][0]["message"]["content"]


class LLMAnswerer:
    def __init__(
        self,
        complete: Callable[[str], str] = openai_complete,
        *,
        min_sim: float = MIN_SIM,
        min_bm25: float = MIN_BM25,
        context_k: int = ANSWER_CONTEXT_K,
    ):
        self.complete = complete
        self.min_sim = min_sim
        self.min_bm25 = min_bm25
        self.context_k = context_k

    def answer(self, question: str, retrieval: Retrieval) -> Answer:
        if reason := weak_evidence(retrieval, self.min_sim, self.min_bm25):
            return abstain(reason)
        context = {r.chunk.chunk_id: r.chunk for r in retrieval.results[: self.context_k]}
        prompt = LLM_PROMPT.format(
            context="\n\n".join(f"[{cid}]\n{c.text}" for cid, c in context.items()),
            question=question,
        )
        raw = self.complete(prompt)
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        try:
            parsed = json.loads(match.group(0)) if match else None
        except json.JSONDecodeError:
            parsed = None
        if not isinstance(parsed, dict):
            return abstain("unparseable LLM output")
        text = str(parsed.get("answer", "")).strip()
        cited = parsed.get("citations") or []
        if not text or text.rstrip(".") == IDK:
            return abstain("LLM said it does not know")
        if not cited or any(cid not in context for cid in cited):
            return abstain(f"LLM cited chunks outside the context: {cited}")
        return Answer(
            answer=text, citations=[citation(context[cid]) for cid in cited], abstained=False
        )


def ask(
    engine: SearchEngine,
    question: str,
    user_visibility: Iterable[str],
    answerer: Answerer,
    k: int = ANSWER_CONTEXT_K,
) -> Answer:
    retrieval = engine.retrieve(question, user_visibility, k=k, mode=ANSWER_MODE)
    return answerer.answer(question, retrieval)
