"""All tunables live here (D7). No magic numbers elsewhere."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
INDEX_DIR = PROJECT_ROOT / "data" / "index"
DB_PATH = INDEX_DIR / "docsearch.sqlite"
MANIFEST_PATH = PROJECT_ROOT / "data" / "manifest.json"
ANSWER_KEY_PATH = PROJECT_ROOT / "data" / "eval" / "answer_key.jsonl"
RUNS_DIR = PROJECT_ROOT / "runs"

# Chunk IDs (D9)
CHUNK_ID_SLUG_MAX = 40

# Text chunking
TEXT_MAX_WORDS = 400
TEXT_MAX_HEADING_LEVEL = 3
TEXT_PREAMBLE_TITLE = "Overview"
TEXT_OVERLAP_SENTENCES = 1

# Table chunking
TABLE_ROWS_PER_CHUNK = 10
TABLE_KEY_COLUMNS = ("plan",)
TABLE_SUMMARY_LOCATOR = "summary"

# Layout chunking (PDF units are points)
LAYOUT_LINE_TOLERANCE = 2.0  # words whose tops differ by less than this share a line
LAYOUT_WORD_GAP = 8.0  # a wider horizontal gap splits a line into separate fragments
LAYOUT_COLUMN_GAP = 30.0  # fragment x0 values further apart than this are separate columns
LAYOUT_BLOCK_GAP = 6.0  # a wider vertical gap starts a new block
LAYOUT_HEADER_FOOTER_PAGE_FRACTION = 0.5
LAYOUT_HEADER_FOOTER_MIN_PAGES = 2
LAYOUT_HEADER_FOOTER_Y_TOLERANCE = 5.0

# Visibility (D10)
PUBLIC = "public"
DEFAULT_USER_VISIBILITY = (PUBLIC,)

# Embeddings (D11)
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_CACHE_PATH = INDEX_DIR / "embeddings.sqlite"

# Retrieval
SEARCH_MODES = ("bm25", "dense", "hybrid")
DEFAULT_K = 5
CANDIDATES_N = 20
RRF_K = 60

# Evaluation
EVAL_KS = (1, 3, 5, 10)
MRR_CUTOFF = 10
HEADLINE_K = 5  # D12: hybrid Recall@5 on the test split
EVAL_SPLITS = ("dev", "test", "all")
RUN_TIMESTAMP_FORMAT = "%Y%m%dT%H%M%S"

# Answer step + abstention. Thresholds are tuned on the dev split only
# (scripts/tune_thresholds.py) and copied here.
IDK = "I don't know"
ANSWER_MODE = "hybrid"
ANSWER_CONTEXT_K = 5
MIN_SIM = 0.0
MIN_BM25 = 0.0
MIN_SENT_SIM = 0.5
TUNE_SIM_GRID = tuple(round(0.05 * i, 2) for i in range(21))  # 0.00 .. 1.00
TUNE_ANSWER_WEIGHT = 2.0  # answering an answerable question counts this much vs one abstention
TUNE_BM25_GRID = tuple(round(0.02 * i, 2) for i in range(51))  # 0.00 .. 1.00 (normalized)

# Optional LLM answerer (D4); never used by tests.
LLM_API_URL = "https://api.openai.com/v1/chat/completions"
LLM_API_KEY_ENV = "OPENAI_API_KEY"
LLM_MODEL = "gpt-4o-mini"
LLM_TEMPERATURE = 0
LLM_TIMEOUT_SECONDS = 60
