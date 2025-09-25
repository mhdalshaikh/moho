from __future__ import annotations

import re
from typing import Dict, Any, List, Tuple

from .retriever import hybrid_retrieve, citations
from .llm import chat_llm
from .tools import (

    to_chart,
    load_full_tables_from_hits,

)

def _collect_tables(retrieved: List[Tuple[str, dict, float]]):
    """
    Prefer loading full sheets (Parquet) via metadata pointers written at ingest time.
    Falls back to the 10-row preview text only if necessary.
    Returns list[(df, meta)].
    """
    return load_full_tables_from_hits(retrieved)

from .config import DF_REQUIRED
from . import tools as T  # for answer_with_df_or_explain and other helpers
# --- Lightweight intent router ---
_CHITCHAT = {
    "hi","hello","hey","salam","السلام","مرحبا","thanks","thank you",
    "how are you","كيف الحال","حرّك","شو الأخبار","good morning","good evening",
    "lol","haha","ok","okay"
}
_DATA_HINTS = {
    # EN
    "sum","total","average","avg","mean","median","max","min","count",
    "between","from","to","range","vs","compare","trend",
    "today","yesterday","this week","this month","last month","last year",
    "month","daily","weekly","monthly",
    "jan","feb","mar","apr","may","jun","jul","aug","sep","oct","nov","dec",
    "chart","plot","graph","line","bar","column","pie",
    "sales","revenue","net","gross","cash","card","visa","mada","online","koinz","jahez","%",
    # AR
    "إجمالي","المجموع","مجموع","صافي","صافي المبيعات","المبيعات الصافية",
    "إجمالي المبيعات","المبيعات الإجمالية","متوسط","المتوسط","أفضل","أسوأ",
    "مقارنة","مقارن","ترند","اتجاه",
    "اليوم","أمس","هذا الأسبوع","هذا الشهر","الشهر الماضي","السنة الماضية",
    "شهري","أسبوعي","يومي","بالـيوم","حسب اليوم",
    "يناير","فبراير","مارس","أبريل","ابريل","مايو","يونيو","يوليو",
    "أغسطس","اغسطس","سبتمبر","أكتوبر","اكتوبر","نوفمبر","ديسمبر",
    "مخطط","رسم","بياني","أعمدة","خط","دائري","باي","حصة","قنوات","قناة",
    "نقد","كاش","بطاقة","مدى","أونلاين","اونلاين","جاهز","كوينز"
}

def _looks_like_chitchat(q: str) -> bool:
    ql = (q or "").strip().lower()
    if not ql:
        return False
    # very short or pure greeting/smalltalk
    if len(ql.split()) <= 4 and any(tok in ql for tok in _CHITCHAT):
        return True
    # no digits/dates and no analysis verbs
    if not any(ch.isdigit() for ch in ql) and not any(k in ql for k in _DATA_HINTS):
        # still allow doc Q&A later — but this is likely chit-chat
        return True
    return False

def _looks_like_data(q: str) -> bool:
    ql = (q or "").lower()
    # digits or % are إشارات قوية
    if any(ch.isdigit() for ch in ql) or "%" in ql:
        return True
    return any(k in ql for k in _DATA_HINTS)

def answer_query(query: str) -> Dict[str, Any]:
    q = (query or "").strip()
    ql = q.lower()

    # 0) Chit-chat → general assistant
    if _looks_like_chitchat(q):
        out = chat_llm([
            {"role": "system", "content": "You are a friendly, concise assistant."},
            {"role": "user", "content": q},
        ])
        return {"text": out, "confidence": 1.0, "citations": ""}

    # 1) Retrieve context (tables/docs)
    hits = hybrid_retrieve(q)  # -> iterable of (text, meta, score)
    cits = citations(hits) if hits else ""
    conf = (sum(s for *_, s in (hits or [])) / max(len(hits or []), 1)) if hits else 0.0

    # If nothing retrieved and it doesn't look like a data task → answer normally
    if not hits and not _looks_like_data(q):
        out = chat_llm([
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": q},
        ])
        return {"text": out, "confidence": 0.7, "citations": ""}

    # 2) DataFrame-first path (text answer)
    df_ans = None
    if _looks_like_data(q) and hits:
        try:
            df_ans = T.answer_with_df(q, hits)  # -> (text, meta) | None
        except Exception:
            df_ans = None  # avoid breaking the UI

    # 3) Chart intent (independent; shown whenever requested)
    chart_html = None
    try:
        if T._wants_chart(q) and hits:
            # let to_chart handle table selection from hits internally
            chart_html = T.to_chart(q, hits=hits)
    except Exception:
        chart_html = None

    # 4) If DF path succeeded, prefer it; attach chart if any
    if df_ans:
        ans_text, telemetry = df_ans
        resp: Dict[str, Any] = {
            "text": f"{ans_text}\n\nSources: {cits}" if cits else ans_text,
            "confidence": conf,
            "citations": cits,
            "meta": {"source": "dataframe", **(telemetry or {})},
        }
        # allow DF path to pass a chart via telemetry, else use the one we built
        thtml = (telemetry or {}).get("chart_html") or chart_html
        if thtml:
            resp["chart_html"] = thtml
        return resp

    # 5) Enforce DF_REQUIRED only for data-intent questions with no usable tables
    if DF_REQUIRED and _looks_like_data(q):
        try:
            tabs = T.load_full_tables_from_hits(hits or [])
        except Exception:
            tabs = []
        if not tabs:
            resp = {
                "text": ("I didn’t find any tables to compute from.\n"
                         "Upload or reference a sheet, or ask a non-tabular question."),
                "confidence": conf,
                "citations": cits,
                "meta": {"source": "policy", "reason": "DF_REQUIRED_no_tables"},
            }
            if chart_html:
                resp["chart_html"] = chart_html
            return resp

    # 6) LLM fallback constrained to retrieved sources (doc-QA style)
    ctx_blocks = []
    for i, (t, m, _) in enumerate(hits or [], 1):
        ctx_blocks.append(f"[Source {i}: {m.get('source','?')}]\n{t}")
    prompt = f"Question: {q}\n\nUse ONLY these sources.\n\n" + "\n\n".join(ctx_blocks)

    out = chat_llm([{"role": "user", "content": prompt}]).strip()
    out = re.sub(r"\n{3,}", "\n\n", out)

    res: Dict[str, Any] = {
        "text": f"{out}\n\nSources: {cits}" if cits else out,
        "confidence": conf,
        "citations": cits,
    }
    if chart_html:
        res["chart_html"] = chart_html
    return res
