from __future__ import annotations

import re
from typing import Dict, Any, List, Tuple

from .retriever import hybrid_retrieve, citations
from .llm import chat_llm
from .tools import (
    aggregate,
    to_chart,
    load_full_tables_from_hits,
    lookup_net_sales,
)

def _collect_tables(retrieved: List[Tuple[str, dict, float]]):
    """
    Prefer loading full sheets (Parquet) via metadata pointers written at ingest time.
    Falls back to the 10-row preview text only if necessary.
    Returns list[(df, meta)].
    """
    return load_full_tables_from_hits(retrieved)

def answer_query(query: str) -> Dict[str, Any]:
    hits = hybrid_retrieve(query)
    if not hits:
        return {"text": "I don't know (no evidence).", "confidence": 0.0, "citations": ""}

    cits = citations(hits)
    conf = sum(s for *_, s in hits) / max(len(hits), 1)

    # Load full tables for numeric work
    tabs = _collect_tables(hits)

    # 0) Direct point/range lookups (e.g., 'net sales on Jan 15', 'sum Jan 1-30')
    looked = lookup_net_sales(query, tabs)
    if looked is not None:
        note, df = looked
        return {
            "text": f"{note}\n\n```\n{df.to_string(index=False)}\n```\n\nSources: {cits}",
            "confidence": conf,
            "citations": cits,
        }

    # 1) Aggregates (avg/mean)
    agg = aggregate(query, tabs)
    if agg is not None:
        note, df = agg
        return {
            "text": f"{note}\n\n```\n{df.to_string(index=False)}\n```\n\nSources: {cits}",
            "confidence": conf,
            "citations": cits,
        }

    # 2) Charts (optional)
    chart = to_chart(query, tabs)

    # 3) Fallback to LLM grounded in retrieved text
    ctx = []
    for i, (t, m, _) in enumerate(hits, 1):
        ctx.append(f"[Source {i}: {m.get('source','?')}]\n{t}")

    prompt = f"Question: {query}\n\nUse ONLY these sources.\n\n" + "\n\n".join(ctx)
    out = chat_llm([{"role": "user", "content": prompt}])
    out = re.sub(r"\n{3,}", "\n\n", out).strip()

    res = {
        "text": f"{out}\n\nSources: {cits}",
        "confidence": conf,
        "citations": cits,
    }
    if chart:
        res["chart_html"] = chart  # HTML string
    return res
