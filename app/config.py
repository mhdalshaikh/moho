from pathlib import Path

# --- Paths ---
BASE_DIR = Path(__file__).resolve().parent.parent
DOCS_DIR = BASE_DIR / "docs"
CACHE_DIR = BASE_DIR / "data_cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)
(CACHE_DIR / "tables").mkdir(parents=True, exist_ok=True)

# --- Models ---
# Ollama model name you have locally; change if you prefer a different one.
LLM_MODEL = "llama3.1:8b"
# SentenceTransformer embedding model (CPU friendly)
EMBEDDING_MODEL = "intfloat/multilingual-e5-base"
# Cross-encoder for reranking (optional; we fall back if not available)
RERANKER_MODEL = "BAAI/bge-reranker-base"

# --- Retrieval params ---
CHUNK_SIZE = 1200
CHUNK_OVERLAP = 200
TOPK_VECTOR = 20
TOPK_BM25 = 30
TOPK_RERANK = 7
MIN_SIM_THRESHOLD = 0.25

APP_TITLE = "🦙 Local RAG — LLaMA Chat"
# Limit rows we chart so Streamlit stays snappy
MAX_ROWS_PREVIEW = 2000
DF_REQUIRED = True
