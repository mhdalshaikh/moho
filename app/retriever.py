# app/retriever.py

import nltk

# Ensure required tokenizers are present (one-time download if missing)
try:
    nltk.data.find("tokenizers/punkt_tab")
except LookupError:
    nltk.download("punkt_tab")

try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt")

from typing import List, Tuple, Dict
import numpy as np
from rank_bm25 import BM25Okapi
from nltk.tokenize import word_tokenize
from sentence_transformers import SentenceTransformer, CrossEncoder
import chromadb
from chromadb.config import Settings

from app.config import (
    CACHE_DIR, TOPK_VECTOR, TOPK_BM25, TOPK_RERANK,
    EMBEDDING_MODEL, RERANKER_MODEL, MIN_SIM_THRESHOLD
)

# --- Consistent Chroma client/collection (match ingest.py) ---
def _client():
    return chromadb.PersistentClient(
        path=str(CACHE_DIR),
        settings=Settings(allow_reset=True)
    )

def _collection():
    client = _client()
    try:
        return client.get_collection("local_rag")
    except Exception:
        return client.create_collection("local_rag")

_bm25 = None
_bm25_docs: List[str] = []
_bm25_meta: List[Dict] = []
_embedder = None
_reranker = None

def _embed():
    global _embedder
    if _embedder is None:
        _embedder = SentenceTransformer(EMBEDDING_MODEL)
    return _embedder

def _ranker():
    global _reranker
    if _reranker is None:
        _reranker = CrossEncoder(RERANKER_MODEL)
    return _reranker

def _ensure_bm25():
    """Build BM25 index if there are documents; otherwise leave _bm25 as None."""
    global _bm25, _bm25_docs, _bm25_meta
    if _bm25 is not None:
        return
    coll = _collection()
    got = coll.get()
    docs = got.get("documents", []) or []
    metas = got.get("metadatas", []) or []
    if not docs:
        _bm25 = None
        _bm25_docs = []
        _bm25_meta = []
        return
    _bm25_docs = docs
    _bm25_meta = metas
    tokens = [word_tokenize((t or "").lower()) for t in _bm25_docs]
    # If everything tokenizes to empty, skip BM25
    if not any(tokens) or all(len(tok) == 0 for tok in tokens):
        _bm25 = None
        return
    _bm25 = BM25Okapi(tokens)

def hybrid_retrieve(query: str) -> List[Tuple[str, Dict, float]]:
    coll = _collection()

    # Vector search
    qv = _embed().encode([query], normalize_embeddings=True)[0].tolist()  # Chroma wants list, not ndarray
    res = coll.query(query_embeddings=[qv], n_results=TOPK_VECTOR)
    vdocs = res.get("documents", [[]])
    vmeta = res.get("metadatas", [[]])
    vdocs = vdocs[0] if vdocs else []
    vmeta = vmeta[0] if vmeta else []

    # Optional BM25 branch
    _ensure_bm25()
    bdocs, bmeta = [], []
    if _bm25 is not None and len(_bm25_docs) > 0:
        toks = word_tokenize(query.lower())
        scores = _bm25.get_scores(toks)
        if isinstance(scores, np.ndarray) and scores.size > 0:
            idx = np.argsort(scores)[-TOPK_BM25:][::-1]
            bdocs = [_bm25_docs[i] for i in idx]
            bmeta = [_bm25_meta[i] for i in idx]

    # Merge & dedupe
    cand = list(zip(vdocs, vmeta)) + list(zip(bdocs, bmeta))
    seen, uniq = set(), []
    for t, m in cand:
        # guard against None
        m = m or {}
        key = (m.get("source"), m.get("chunk"), (t or "")[:48])
        if key in seen:
            continue
        seen.add(key)
        uniq.append((t or "", m))
    if not uniq:
        return []

    # Rerank with cross-encoder
    pairs = [[query, t] for t, _ in uniq]
    sc = _ranker().predict(pairs)  # np.ndarray of scores

    ranked = sorted(zip(uniq, sc), key=lambda x: x[1], reverse=True)[:TOPK_RERANK]

    # Robust normalization (fix the single-item -> 1.0 case)
    raw = np.array([r for _, r in ranked], dtype=float)
    if raw.size == 0:
        return []
    if raw.size == 1:
        norm = np.array([1.0], dtype=float)
    else:
        mn, mx = raw.min(), raw.max()
        norm = (raw - mn) / (mx - mn + 1e-6)

    out: List[Tuple[str, Dict, float]] = []
    for ((t, m), _), ns in zip(ranked, norm):
        if float(ns) >= float(MIN_SIM_THRESHOLD):
            out.append((t, m, float(ns)))
    return out

def citations(items):
    return " ".join(
        f"[{i}: {m.get('source','?')}#chunk{m.get('chunk','?')} · conf={s:.2f}]"
        for i, (_, m, s) in enumerate(items, 1)
    )
