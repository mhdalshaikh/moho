import re, pandas as pd
from typing import Dict, Any
from .retriever import hybrid_retrieve, citations
from .llm import chat_llm
from .tools import aggregate, to_chart

def _collect_tables(retrieved):
    tabs=[]
    for text, meta, _ in retrieved:
        if meta.get("type")=="table" and "\\n" in text:
            lines=text.splitlines()
            for i,ln in enumerate(lines):
                if ln.startswith("TABLE "):
                    csv="\\n".join(lines[i+1:])
                    try: tabs.append(pd.read_csv(pd.io.common.StringIO(csv)))
                    except Exception: pass
                    break
    return tabs

def answer_query(query:str)->Dict[str,Any]:
    hits=hybrid_retrieve(query)
    if not hits:
        return {"text":"I don't know (no evidence).","confidence":0.0,"citations":""}
    cits=citations(hits)
    conf=sum(s for *_, s in hits)/max(len(hits),1)
    tabs=_collect_tables(hits)

    agg=aggregate(query,tabs)
    if agg is not None:
        note,df=agg
        return {"text":f"{note}\\n\\n``\\n{df.to_string(index=False)}\\n```\\n\\nSources: {cits}",
                "confidence":conf,"citations":cits}

    chart=to_chart(query,tabs)
    ctx=[]; 
    for i,(t,m,_) in enumerate(hits,1):
        ctx.append(f"[Source {i}: {m.get('source','?')}]\\n{t}")
        prompt = f"Question: {query}\n\nUse ONLY these sources.\n\n" + "\n\n".join(ctx)
        out = chat_llm([{"role": "user", "content": prompt}])

    out = re.sub(r"\n{3,}", "\n\n", out).strip()
    res = {
        "text": f"{out}\n\nSources: {cits}",
        "confidence": conf,
        "citations": cits
    }
    if chart:
        res["chart_html"] = chart
    return res
