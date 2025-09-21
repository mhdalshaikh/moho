from __future__ import annotations

import nltk

# Ensure punkt is available (silently skip if offline and already present)
for pkg in ["punkt", "punkt_tab"]:
    try:
        nltk.data.find(f"tokenizers/{pkg}")
    except LookupError:
        try:
            nltk.download(pkg, quiet=True)
        except Exception:
            pass

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

# --- Chroma ---
def _client():
    return chromadb.PersistentClient(
        path=str(CACHE_DIR),
        settings=Settings(allow_reset=True)
    )

def _collection():
    c = _client()
    try:
        return c.get_collection("local_rag")
    except Exception:
        return c.create_collection("local_rag")

# --- Globals ---
_bm25 = None
_bm25_docs: List[str] = []
_bm25_meta: List[Dict] = []
_embedder = None
_reranker = None
_reranker_ok = False

def _embed():
    global _embedder
    if _embedder is None:
        _embedder = SentenceTransformer(EMBEDDING_MODEL)
    return _embedder

def _ranker():
    global _reranker, _reranker_ok
    if _reranker is None and not _reranker_ok:
        try:
            _reranker = CrossEncoder(RERANKER_MODEL)
            _reranker_ok = True
        except Exception:
            _reranker = None
            _reranker_ok = False
    return _reranker

def _ensure_bm25():
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
    if not any(tokens) or all(len(tok) == 0 for tok in tokens):
        _bm25 = None
        return
    _bm25 = BM25Okapi(tokens)

def hybrid_retrieve(query: str) -> List[Tuple[str, Dict, float]]:
    coll = _collection()

    # Vector search
    qv = _embed().encode([query], normalize_embeddings=True)[0].tolist()
    res = coll.query(query_embeddings=[qv], n_results=TOPK_VECTOR)
    vdocs = (res.get("documents") or [[]])[0]
    vmeta = (res.get("metadatas") or [[]])[0]

    # Optional BM25
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
        m = m or {}
        key = (m.get("source"), m.get("chunk"), (t or "")[:64])
        if key in seen:
            continue
        seen.add(key)
        uniq.append((t or "", m))
    if not uniq:
        return []

    # Rerank if possible
    ce = _ranker()
    if ce is not None:
        try:
            pairs = [[query, t] for t, _ in uniq]
            sc = ce.predict(pairs)  # np.ndarray
        except Exception:
            sc = np.linspace(0.2, 0.8, num=len(uniq))
    else:
        # heuristic: prefer vector half, then bm25 half
        sc = np.linspace(0.51, 0.71, num=len(uniq))

    # Top K & normalize
    ranked = sorted(zip(uniq, sc), key=lambda x: x[1], reverse=True)[:TOPK_RERANK]
    raw = np.array([r for _, r in ranked], dtype=float)
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
