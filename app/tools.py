import io, re
import pandas as pd
import plotly.express as px
from .config import MAX_ROWS_PREVIEW

AGG = {"average":"mean","avg":"mean","mean":"mean","sum":"sum","total":"sum",
       "count":"count","min":"min","max":"max"}
HINTS = ["amount","cost","price","value","total","qty","quantity","score"]

def _agg_word(q:str):
    q=q.lower()
    for k,v in AGG.items():
        if k in q: return v

def _col_hint(q:str, cols):
    q=q.lower()
    for c in cols:
        if c.lower() in q: return c
    for c in cols:
        if any(h in c.lower() for h in HINTS): return c

def _filter_year(df, q:str):
    m=re.search(r"(\\d{4})", q)
    if m and any('year' in c.lower() for c in df.columns):
        ycol=[c for c in df.columns if 'year' in c.lower()][0]
        return df[df[ycol].astype(str)==m.group(1)]
    return df

def aggregate(query, tables):
    kind=_agg_word(query)
    if not kind: return None
    for df in tables:
        if df.empty: continue
        df2=_filter_year(df,query)
        cols=list(df2.columns)
        col=_col_hint(query, cols)
        if not col:
            num=[c for c in cols if pd.api.types.is_numeric_dtype(df2[c])]
            if not num: continue
            col=num[0]
        try:
            if kind=="count":
                out=pd.DataFrame({"count":[len(df2)]})
            else:
                out=pd.DataFrame({kind:[getattr(pd.Series(df2[col].astype(float)), kind)()]})
            return f"Computed {kind} on '{col}'.", out
        except Exception: continue

def to_chart(query, tables):
    q=query.lower(); kind=None
    if "pie" in q: kind="pie"
    elif "line" in q: kind="line"
    elif "bar" in q or "column" in q: kind="bar"
    if not kind: return None
    for df in tables:
        if df.empty: continue
        d=df.head(MAX_ROWS_PREVIEW)
        x=d.columns[0]
        nums=[c for c in d.columns if pd.api.types.is_numeric_dtype(d[c])]
        if not nums: continue
        y=nums[0]
        try:
            fig=px.bar(d,x=x,y=y) if kind=='bar' else (px.line(d,x=x,y=y) if kind=='line' else px.pie(d,names=x,values=y))
            buf=io.BytesIO(); fig.write_html(buf); buf.seek(0); return buf
        except Exception: continue
