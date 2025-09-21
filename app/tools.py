# app/tools.py
from __future__ import annotations

import re
from io import StringIO
from pathlib import Path
from datetime import datetime
from typing import List, Tuple

import pandas as pd
import plotly.express as px
from rapidfuzz import fuzz

from .config import CACHE_DIR, MAX_ROWS_PREVIEW

# ---------- Paths ----------
TABLE_STORE = (CACHE_DIR / "tables")

# ---------- Loading full tables ----------
def _maybe_flatten_columns(df: pd.DataFrame) -> pd.DataFrame:
    if isinstance(df.columns, pd.MultiIndex):
        df = df.copy()
        df.columns = [
            " ".join([str(x) for x in tup if str(x) != "nan"]).strip()
            for tup in df.columns
        ]
    else:
        df = df.rename(columns=lambda c: str(c))
    return df

def load_table_from_meta(meta: dict) -> pd.DataFrame | None:
    p = meta.get("table_path")
    if not p:
        return None
    try:
        df = pd.read_parquet(p)
        return _maybe_flatten_columns(df)
    except Exception:
        return None

def _load_all_parquet_tables(filter_sources: list[str] | None) -> list[tuple[pd.DataFrame, dict]]:
    out: list[tuple[pd.DataFrame, dict]] = []
    if not TABLE_STORE.exists():
        return out
    for p in TABLE_STORE.glob("*.parquet"):
        if filter_sources:
            src_ok = any(Path(s).stem in p.name for s in filter_sources if s)
            if not src_ok:
                continue
        try:
            df = pd.read_parquet(p)
            df = _maybe_flatten_columns(df)
            meta = {
                "source": Path(p.name).name.split("__")[0] + ".xlsx",
                "sheet": p.stem.split("__")[1] if "__" in p.stem else None,
                "type": "table",
                "table_path": str(p),
            }
            out.append((df, meta))
        except Exception:
            continue
    return out

def load_full_tables_from_hits(hits) -> list[tuple[pd.DataFrame, dict]]:
    """
    hits: iterable of (text, meta, score).
    Returns list of (df, meta). Prefers Parquet pointers; falls back to TABLE_STORE.
    """
    tables: list[tuple[pd.DataFrame, dict]] = []
    hit_sources = []

    # 1) Follow pointers from retrieved items
    for text, meta, _ in hits:
        meta = meta or {}
        src = meta.get("source")
        if src: hit_sources.append(src)
        df = load_table_from_meta(meta)
        if df is not None and not df.empty:
            tables.append((df, meta))

    # 2) Fallback: load by matching sources (if we have any)
    if not tables and TABLE_STORE.exists() and hit_sources:
        matched_any = False
        for p in TABLE_STORE.glob("*.parquet"):
            if any(Path(s).stem in p.name for s in hit_sources):
                try:
                    df = pd.read_parquet(p)
                    df = _maybe_flatten_columns(df)
                    meta = {
                        "source": Path(p.name).name.split("__")[0] + ".xlsx",
                        "sheet": p.stem.split("__")[1] if "__" in p.stem else None,
                        "type": "table",
                        "table_path": str(p),
                    }
                    tables.append((df, meta))
                    matched_any = True
                except Exception:
                    pass
        # if nothing matched, fall through to “load all” below

    # 3) Last resort: load ALL parquet sheets (no filtering)
    if not tables and TABLE_STORE.exists():
        for p in TABLE_STORE.glob("*.parquet"):
            try:
                df = pd.read_parquet(p)
                df = _maybe_flatten_columns(df)
                meta = {
                    "source": Path(p.name).name.split("__")[0] + ".xlsx",
                    "sheet": p.stem.split("__")[1] if "__" in p.stem else None,
                    "type": "table",
                    "table_path": str(p),
                }
                tables.append((df, meta))
            except Exception:
                pass

    # (optional) parse preview CSV from text if still nothing — unchanged...
    if not tables:
        for text, meta, _ in hits:
            if isinstance(text, str) and "\n" in text:
                lines = text.splitlines()
                for i, ln in enumerate(lines):
                    if ln.startswith("TABLE "):
                        csv = "\n".join(lines[i+1:])
                        try:
                            df = pd.read_csv(StringIO(csv))
                            df = _maybe_flatten_columns(df)
                            tables.append((df, meta or {}))
                        except Exception:
                            pass
                        break

    # Rank most “sales-like” — unchanged...
    def _score(df, meta):
        cols = [str(c).lower() for c in df.columns]
        has_date = any("date" in c for c in cols)
        has_net  = any(("net" in c and "sale" in c) for c in cols)
        n_days = 0
        if has_date:
            d = pd.to_datetime(df[[c for c in df.columns if "date" in str(c).lower()][0]], errors="coerce").dt.date
            n_days = d.nunique()
        name = f"{meta.get('sheet','')} {meta.get('source','')}".lower()
        sales_hint = 1 if "sales" in name else 0
        return 10*sales_hint + 8*int(has_date and has_net) + min(n_days, 40)

    tables.sort(key=lambda pair: _score(*pair), reverse=True)
    return tables

# ---------- Flexible column detection ----------
_DATE_CANDIDATE_TOKENS = [
    "date", "day", "invoice date", "posting date", "transaction date",
    "التاريخ", "tarikh", "fecha", "data", "datum"
]

_NET_SALES_TOKENS = [
    "net sales", "net sale", "net revenue", "revenue net", "sales net",
    "sales (net)", "net amount", "amount net", "صافي المبيعات",
    "chiffre d’affaires net", "ingresos netos"
]

_CHANNEL_TOKENS = [
    "cash", "bank", "pos", "card", "visa", "mada", "online", "koinz",
    "jahez", "hunger", "al inma", "alinma", "fodics", "tap", "stripe"
]

_VAT_TOKENS = [
    "vat", "tax", "15%", "0.15", "ضريبة"
]

def _best_match(colnames, targets, min_score=65):
    best_col, best_score = None, -1
    for c in colnames:
        s = str(c)
        for t in targets:
            sc = fuzz.token_set_ratio(s.lower(), t.lower())
            if sc > best_score:
                best_col, best_score = c, sc
    return (best_col if best_score >= min_score else None, best_score)

def _find_numeric_cols(df: pd.DataFrame):
    return [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]

def _find_date_col(df: pd.DataFrame) -> str | None:
    cols = list(df.columns)
    c, _ = _best_match(cols, _DATE_CANDIDATE_TOKENS, min_score=60)
    if c is not None:
        return c
    # fallback: column with highest datetime parse rate
    def dt_score(series):
        s = pd.to_datetime(series, errors="coerce")
        return s.notna().mean()
    parse_rates = [(col, dt_score(df[col])) for col in cols]
    parse_rates.sort(key=lambda x: x[1], reverse=True)
    return parse_rates[0][0] if parse_rates and parse_rates[0][1] >= 0.5 else None

def _find_net_sales_col(df: pd.DataFrame) -> str | None:
    num_cols = _find_numeric_cols(df)
    best, best_score = None, -1
    for c in num_cols:
        s = str(c).lower()
        sc = max(fuzz.token_set_ratio(s, t.lower()) for t in _NET_SALES_TOKENS)
        if sc > best_score:
            best, best_score = c, sc
    return best if best_score >= 65 else None

def _derive_net_sales(df: pd.DataFrame) -> pd.Series | None:
    num_cols = _find_numeric_cols(df)
    if not num_cols:
        return None
    channel_cols, vat_cols = [], []
    for c in num_cols:
        name = str(c).lower()
        ch_sc = max(fuzz.token_set_ratio(name, t) for t in _CHANNEL_TOKENS)
        vat_sc = max(fuzz.token_set_ratio(name, t) for t in _VAT_TOKENS)
        if ch_sc >= 70 and vat_sc < 60:
            channel_cols.append(c)
        elif vat_sc >= 70:
            vat_cols.append(c)
    if not channel_cols:
        return None
    summed = pd.DataFrame(df[channel_cols]).apply(pd.to_numeric, errors="coerce").sum(axis=1)
    if vat_cols:
        summed = summed - pd.DataFrame(df[vat_cols]).apply(pd.to_numeric, errors="coerce").sum(axis=1)
    return summed

# ---------- Query date parsing ----------
_MONTH_MAP = {
    "jan": 1, "january": 1, "feb": 2, "february": 2, "mar": 3, "march": 3,
    "apr": 4, "april": 4, "may": 5, "jun": 6, "june": 6, "jul": 7, "july": 7,
    "aug": 8, "august": 8, "sep": 9, "sept": 9, "september": 9, "oct": 10,
    "october": 10, "nov": 11, "november": 11, "dec": 12, "december": 12
}

def _parse_any_range(q: str):
    ql = q.lower()
    # ISO-like range
    m = re.search(r'(\d{4}-\d{2}-\d{2})\s*(?:to|–|—|-|\.{2,})\s*(\d{4}-\d{2}-\d{2})', ql)
    if m:
        a = pd.to_datetime(m.group(1)).date()
        b = pd.to_datetime(m.group(2)).date()
        if a <= b: return a, b
    # Month with day range: Jan 1-30
    m = re.search(r'(' + '|'.join(_MONTH_MAP.keys()) + r')\s*(\d{1,2})\s*(?:to|–|—|-)\s*(\d{1,2})', ql)
    if m:
        mo = _MONTH_MAP[m.group(1)]
        d1, d2 = int(m.group(2)), int(m.group(3))
        return datetime(1900, mo, d1).date(), datetime(1900, mo, d2).date()
    # Specific day: Jan 15
    m = re.search(r'(' + '|'.join(_MONTH_MAP.keys()) + r')\s*(\d{1,2})\b', ql)
    if m:
        mo = _MONTH_MAP[m.group(1)]
        d = int(m.group(2))
        dt = datetime(1900, mo, d).date()
        return dt, dt
    # Whole month mention
    m = re.search(r'\b(' + '|'.join(_MONTH_MAP.keys()) + r')\b', ql)
    if m:
        mo = _MONTH_MAP[m.group(1)]
        return datetime(1900, mo, 1).date(), datetime(1900, mo, 31).date()
    return None

def _parse_single_or_range(q: str):
    ql = q.lower()
    # ISO single
    m = re.search(r'\b(\d{4}-\d{2}-\d{2})\b', ql)
    if m:
        d = pd.to_datetime(m.group(1)).date()
        return ('day', d)
    # Month + day single (Jan 15)
    m = re.search(r'(' + '|'.join(_MONTH_MAP.keys()) + r')\s*(\d{1,2})\b', ql)
    if m:
        mo = _MONTH_MAP[m.group(1)]
        d  = int(m.group(2))
        return ('day', datetime(1900, mo, d).date())
    # Ranges
    m = re.search(r'(\d{4}-\d{2}-\d{2})\s*(?:to|–|—|-|\.{2,})\s*(\d{4}-\d{2}-\d{2})', ql)
    if m:
        a = pd.to_datetime(m.group(1)).date()
        b = pd.to_datetime(m.group(2)).date()
        if a <= b: return ('range', (a,b))
    m = re.search(r'(' + '|'.join(_MONTH_MAP.keys()) + r')\s*(\d{1,2})\s*(?:to|–|—|-)\s*(\d{1,2})', ql)
    if m:
        mo = _MONTH_MAP[m.group(1)]
        a, b = int(m.group(2)), int(m.group(3))
        return ('range', (datetime(1900,mo,a).date(), datetime(1900,mo,b).date()))
    return None

# ---------- Analytics primitives ----------
def aggregate(query, tables):
    """
    Compute mean of daily totals if the user asks for average/mean.
    Works even if names vary: finds date col and net sales via fuzzy logic,
    or derives net from channels if necessary.
    """
    ql = query.lower()
    if not any(k in ql for k in ("avg", "average", "mean")):
        return None

    # normalize input: list of (df, meta) tuples
    norm_tables: list[pd.DataFrame] = []
    for item in tables:
        df = item[0] if isinstance(item, tuple) else item
        if df is None or df.empty:
            continue
        df = _maybe_flatten_columns(df)
        norm_tables.append(df)
    if not norm_tables:
        return None

    def table_score(df):
        dcol = _find_date_col(df)
        ncol = _find_net_sales_col(df)
        days = 0
        if dcol:
            d = pd.to_datetime(df[dcol], errors="coerce").dt.date
            days = d.nunique()
        return (10 if dcol else 0) + (10 if (ncol or _derive_net_sales(df) is not None) else 0) + min(days, 40)

    best = max(norm_tables, key=table_score)

    date_col = _find_date_col(best)
    net_col  = _find_net_sales_col(best)
    if date_col is None:
        return None

    dates = pd.to_datetime(best[date_col], errors="coerce")
    if net_col is not None:
        net_series = pd.to_numeric(best[net_col], errors="coerce")
    else:
        net_series = _derive_net_sales(best)
    if net_series is None:
        return None

    gkey = pd.to_datetime(dates, errors="coerce").dt.date
    daily = (
        pd.DataFrame({"date": gkey, "net": net_series})
        .dropna(subset=["date"])
        .groupby("date", as_index=False)["net"]
        .sum()
    )

    rng = _parse_any_range(ql)
    if rng:
        a, b = rng
        def key(d): return (d.month, d.day)
        daily = daily[(daily["date"].apply(key) >= key(a)) & (daily["date"].apply(key) <= key(b))]

    if daily.empty:
        return None

    mean_val = float(daily["net"].mean())
    label = f"{rng[0]}–{rng[1]}" if rng and rng[0] != rng[1] else (f"{rng[0]}" if rng else "selected period")
    note = f"Computed mean of NET SALES per day for {label}."
    return note, pd.DataFrame({"mean": [mean_val]})

def lookup_net_sales(query, tables):
    """
    Point/sum lookup for 'what is net sales on Jan 15' or 'sum net sales Jan 1-30'.
    Returns (note, df) or None.
    """
    ql = query.lower()
    if any(k in ql for k in ("avg","average","mean")):
        return None

    norm: list[pd.DataFrame] = []
    for item in tables:
        df = item[0] if isinstance(item, tuple) else item
        if df is None or df.empty:
            continue
        df = _maybe_flatten_columns(df)
        norm.append(df)
    if not norm:
        return None

    def table_score(df):
        dcol = _find_date_col(df)
        ncol = _find_net_sales_col(df)
        days = 0
        if dcol:
            d = pd.to_datetime(df[dcol], errors="coerce").dt.date
            days = d.nunique()
        return (10 if dcol else 0) + (10 if (ncol or _derive_net_sales(df) is not None) else 0) + min(days, 40)

    best = max(norm, key=table_score)

    dcol = _find_date_col(best)
    ncol = _find_net_sales_col(best)
    if dcol is None:
        return None

    dates = pd.to_datetime(best[dcol], errors="coerce")
    if ncol is not None:
        net = pd.to_numeric(best[ncol], errors="coerce")
    else:
        net = _derive_net_sales(best)
        if net is None:
            return None

    gkey = pd.to_datetime(dates, errors="coerce").dt.date
    daily = (
        pd.DataFrame({"date": gkey, "net": net})
        .dropna(subset=["date"])
        .groupby("date", as_index=False)["net"]
        .sum()
    )

    spec = _parse_single_or_range(ql)
    if not spec:
        return None

    if spec[0] == "day":
        d = spec[1]
        row = daily[(daily["date"].apply(lambda x: (x.month, x.day)) == (d.month, d.day))]
        if row.empty:
            return ("No NET SALES found for that day.", pd.DataFrame({"value":[float('nan')]}))
        val = float(row["net"].iloc[0])
        return (f"NET SALES on {d}:", pd.DataFrame({"value":[val]}))

    # range sum
    a, b = spec[1]
    def key(x): return (x.month, x.day)
    select = daily[(daily["date"].apply(key) >= key(a)) & (daily["date"].apply(key) <= key(b))]
    if select.empty:
        return ("No NET SALES found in that range.", pd.DataFrame({"sum":[float('nan')]}))
    return (f"Sum of NET SALES for {a}–{b}:", pd.DataFrame({"sum":[float(select['net'].sum())]}))

# ---------- Charting ----------
def to_chart(query, tables):
    """
    Build a quick chart from the best-matching table.
    Prefers DATE vs NET SALES; returns an HTML string ready for Streamlit.
    """
    q = (query or "").lower()
    if "pie" in q: kind = "pie"
    elif "bar" in q or "column" in q: kind = "bar"
    else: kind = "line"

    norm: list[pd.DataFrame] = []
    for item in tables:
        df = item[0] if isinstance(item, tuple) else item
        if df is None or df.empty:
            continue
        df = _maybe_flatten_columns(df)
        norm.append(df)
    if not norm:
        return None

    def table_score(df):
        dcol = _find_date_col(df)
        ncol = _find_net_sales_col(df)
        days = 0
        if dcol:
            d = pd.to_datetime(df[dcol], errors="coerce").dt.date
            days = d.nunique()
        return (10 if dcol else 0) + (10 if (ncol or _derive_net_sales(df) is not None) else 0) + min(days, 40)

    best = max(norm, key=table_score)

    dcol = _find_date_col(best)
    ncol = _find_net_sales_col(best)
    df_plot = best.copy()

    if dcol is not None:
        x = pd.to_datetime(df_plot[dcol], errors="coerce")
        if ncol is not None:
            y = pd.to_numeric(df_plot[ncol], errors="coerce")
        else:
            derived = _derive_net_sales(df_plot)
            if derived is None:
                num_cols = [c for c in df_plot.columns if pd.api.types.is_numeric_dtype(df_plot[c])]
                if not num_cols:
                    return None
                y = pd.to_numeric(df_plot[num_cols[0]], errors="coerce")
            else:
                y = pd.to_numeric(derived, errors="coerce")

        plot_df = pd.DataFrame({"x": x, "y": y}).dropna()
        if plot_df.empty:
            return None

        # Aggregate to daily totals if multiple rows per day
        if plot_df["x"].dt.date.duplicated().any():
            plot_df = plot_df.groupby(plot_df["x"].dt.date, as_index=False)["y"].sum()
            plot_df["x"] = pd.to_datetime(plot_df["x"])

        plot_df = plot_df.sort_values("x").head(MAX_ROWS_PREVIEW)

        fig = px.line(plot_df, x="x", y="y") if kind == "line" else \
              (px.bar(plot_df, x="x", y="y") if kind == "bar" else \
               px.pie(plot_df, names="x", values="y"))
    else:
        cols = list(df_plot.columns)
        if not cols:
            return None
        xcol = cols[0]
        num_cols = [c for c in df_plot.columns if pd.api.types.is_numeric_dtype(df_plot[c])]
        if not num_cols:
            return None
        ycol = num_cols[0]
        d = df_plot[[xcol, ycol]].dropna().head(MAX_ROWS_PREVIEW)
        if d.empty:
            return None
        fig = px.line(d, x=xcol, y=ycol) if kind == "line" else \
              (px.bar(d, x=xcol, y=ycol) if kind == "bar" else \
               px.pie(d, names=xcol, values=ycol))

    return fig.to_html(full_html=False)
