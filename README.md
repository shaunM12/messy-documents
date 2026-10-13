# docsearch: messy document search + evaluation

Small, testable retrieval over three kinds of messy documents for a fictional company,
Northwind Gear. It uses type-aware chunking, hybrid BM25 + embedding search fused with
Reciprocal Rank Fusion, extractive answers with citations, and "I don't know" when the
evidence is weak. `CONTEXT.md` is the spec.

## How each document type is split, and why

| Type | File | Strategy | Why |
|------|------|----------|-----|
| Plain text | `policy_handbook_v*.md`, `internal_ops_notes.md` | One chunk per `#`/`##`/`###` section, prefixed with its heading path (`Refunds > Digital goods: ...`). Sections over 400 words split on paragraphs with 1 sentence of overlap. | Policies are organised by heading, so a section is a self-contained answer. The heading prefix keeps context that the body text alone leaves out ("14 days" means nothing without "Digital goods"). |
| Table | `pricing_table.csv` | Rows grouped by the `plan` key column (10 rows max per chunk, rows never split). Every chunk repeats the column headers, rows serialized as `col: value; ...`. Plus one summary chunk (columns, row count, plan values). | A row without its headers is just numbers. Grouping by plan keeps "Enterprise price for NW-1042" in one chunk. The summary answers questions about the table itself. |
| Messy layout | `product_flyer.pdf` | pdfplumber word coordinates: drop header/footer lines that repeat on at least half the pages at a similar height, cluster lines by left edge into columns, read column by column, start a new block on a vertical gap, never cross pages. Embedded tables go to the table chunker. | Reading a 2-column page with a sidebar line by line interleaves unrelated products. Running headers and footers are noise in every chunk. Table cells read as plain text lose their headers. |

## Measured result

Test split (24 questions; 20 answerable), from `docsearch eval --mode all --split test`:

| Mode | Recall@1 | Recall@3 | **Recall@5** | Recall@10 | MRR |
|------|---------:|---------:|-------------:|----------:|----:|
| BM25 | 0.550 | 0.800 | 0.950 | 1.000 | 0.733 |
| Dense | 0.650 | 0.850 | 0.900 | 1.000 | 0.804 |
| **Hybrid (RRF)** | 0.700 | 0.950 | **0.975** | 1.000 | **0.854** |

**Hybrid Recall@5 = 0.975 vs BM25 0.950 and dense 0.900.** By tag, BM25 wins on `keyword`
questions (SKUs, prices, codes): Recall@5 1.000 against 0.889 for dense. On `paraphrase`
questions both single retrievers score 0.909 and hybrid gets 0.955. Hybrid also has the best
MRR, so the right chunk appears higher up in the results.

With hybrid retrieval, the extractive answerer wrongly abstained on 1 of 20 answerable test
questions (false abstention rate 0.050). It correctly abstained on 3 of the 4 unanswerable and
permission questions (abstention accuracy 0.750). The miss was a permission question about an
internal price increase: it was answered with an unrelated public pricing row that mentions
the same backpack.

The tuning deliberately favours answering: a false abstention costs twice as much as a missed
abstention (`TUNE_ANSWER_WEIGHT`). With that weighting, the dev split gives `MIN_SIM` and
`MIN_BM25` of 0, which turns the "weak retrieval" rule off, so "I don't know" comes from
sentence similarity (`MIN_SENT_SIM = 0.5`). On this corpus, the retrieval scores of
should-abstain questions overlap with those of answerable ones, so no cutoff on them separates
the two.

**Disclosure:** the abstention thresholds (`MIN_SIM`, `MIN_BM25`, `MIN_SENT_SIM` in
`src/docsearch/config.py`) were tuned on the dev split only
(`scripts/tune_thresholds.py`, 10 questions). Every number above is from the test split,
which was not used for tuning. The abstention rule was revised once after the first test run.
The BM25 signal is now normalized to 0–1, so its cutoff doesn't depend on how many documents
are loaded, and the tuning now favours answering. That first run (thresholds `0.65 / 8.0 raw /
0.5`) abstained on all 4 should-abstain test questions but also on 5 of 20 answerable ones.
Retrieval numbers were unaffected. The corpus and answer key are small and synthetic, so treat
these results as a sanity check, not a benchmark. Sample logs are in `runs/sample/`.

## How to run

Requires [uv](https://docs.astral.sh/uv/). Python 3.11 and pinned dependencies are installed
by `uv sync`. The embedding model (`all-MiniLM-L6-v2`) downloads on first use; tests never
need it or any network.

```bash
uv sync

# 1. Regenerate the synthetic corpus (output is committed and deterministic)
uv run python scripts/make_corpus.py

# 2. Ingest. Re-ingesting a doc id replaces all of its old chunks.
uv run docsearch ingest data/raw/policy_handbook_v1.md --doc-id policy_handbook --version v1 --visibility public --updated-at 2026-03-01
uv run docsearch ingest data/raw/policy_handbook_v2.md --doc-id policy_handbook --version v2 --visibility public --updated-at 2026-10-01
uv run docsearch ingest data/raw/pricing_table.csv --doc-id pricing_table --version v1 --updated-at 2026-09-15
uv run docsearch ingest data/raw/product_flyer.pdf --doc-id product_flyer --version v1 --updated-at 2026-04-01
uv run docsearch ingest data/raw/internal_ops_notes.md --doc-id internal_ops_notes --version v1 --visibility role:internal --updated-at 2026-09-30
uv run docsearch delete --doc-id product_flyer   # optional

# 3. Ask (JSON with citations, or "I don't know")
uv run docsearch ask "What is the refund window for digital goods?" --user public --k 5
uv run docsearch ask "What is the landed cost of the Ridgeline 2P Tent?" --user public,role:internal
uv run docsearch ask "..." --llm    # optional LLM answerer, needs OPENAI_API_KEY

# 4. Evaluate all three modes. Builds a fresh index from data/manifest.json.
uv run docsearch eval --mode all --split test --k 1,3,5,10

# 5. Re-tune abstention thresholds on dev, then copy the values into config.py
uv run python scripts/tune_thresholds.py

# Tests and lint
uv run pytest
uv run ruff check && uv run ruff format --check
```

Each eval run writes `runs/{timestamp}_{mode}_{split}.jsonl` (retrieved IDs, scores, ranks,
hits and the abstention decision for every question) and appends to `runs/summary.csv`.
