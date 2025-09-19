from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DOCS_DIR = BASE_DIR / "docs"
CACHE_DIR = BASE_DIR / "data_cache"
CACHE_DIR.mkdir(exist_ok=True)

# Model choice: LLaMA
LLM_MODEL = "llama3.1:8b"   # you can pull this with: ollama pull llama3.1:8b
EMBEDDING_MODEL = "intfloat/multilingual-e5-base"
RERANKER_MODEL = "BAAI/bge-reranker-base"

# Chunking & retrieval
CHUNK_SIZE = 1200
CHUNK_OVERLAP = 200
TOPK_VECTOR = 20
TOPK_BM25 = 30
TOPK_RERANK = 7
MIN_SIM_THRESHOLD = 0.25

APP_TITLE = "🦙 Local RAG — LLaMA Chat"
MAX_ROWS_PREVIEW = 2000
