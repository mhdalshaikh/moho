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
# Map any month token → month number
_NAME2MON = {
    "jan":1,"january":1,
    "feb":2,"february":2,
    "mar":3,"march":3,
    "apr":4,"april":4,
    "may":5,
    "jun":6,"june":6,
    "jul":7,"july":7,
    "aug":8,"august":8,
    "sep":9,"sept":9,"september":9,
    "oct":10,"october":10,
    "nov":11,"november":11,
    "dec":12,"december":12,
}
def _merge_tables_dedup(primary: list[tuple[pd.DataFrame, dict]],
                        extra: list[tuple[pd.DataFrame, dict]] | None = None):
    """Merge and dedupe tables by meta['table_path']."""
    merged = []
    seen = set()
    for df, meta in (primary or []) + (extra or []):
        tp = (meta or {}).get("table_path")
        if tp in seen:
            continue
        seen.add(tp)
        merged.append((df, meta))
    return merged
def _compute_daily_metric_all(tables, query: str) -> tuple[pd.DataFrame, str] | None:
    """
    Build one combined daily df by applying the same metric selection/derivation
    to each table and concatenating.
    Returns (daily_df, phrase) with columns ['date','val'].
    """
    parts = []
    phrase_final = None
    for item in (tables or []):
        df = item[0] if isinstance(item, tuple) else item
        if df is None or df.empty:
            continue
        df = _maybe_flatten_columns(df)

        dcol = _find_date_col(df)
        if dcol is None:
            continue

        mcol, phrase = _best_metric_column(df, query)
        phrase_final = phrase if phrase_final is None else phrase_final
        if mcol is None:
            derived = _maybe_derive_metric(df, phrase)
            if derived is None:
                # last resort: pick first numeric
                num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
                if not num_cols:
                    continue
                vals = pd.to_numeric(df[num_cols[0]], errors="coerce")
            else:
                vals = pd.to_numeric(derived, errors="coerce")
        else:
            vals = pd.to_numeric(df[mcol], errors="coerce")

        dates = pd.to_datetime(df[dcol], errors="coerce")
        daily = (
            pd.DataFrame({"date": dates.dt.date, "val": vals})
            .dropna(subset=["date", "val"])
            .groupby("date", as_index=False)["val"].sum()
        )
        if not daily.empty:
            parts.append(daily)

    if not parts:
        return None

    combined = (
        pd.concat(parts, ignore_index=True)
        .groupby("date", as_index=False)["val"].sum()
    )
    return (combined, phrase_final or _extract_metric_phrase(query))



_VAT_TOKENS = [
    "vat", "tax", "15%", "0.15", "ضريبة"
]
# ---- Natural language month/year parsing ----
_MONTHS = {
    "january":1,"jan":1, "february":2,"feb":2, "march":3,"mar":3,
    "april":4,"apr":4, "may":5, "june":6,"jun":6, "july":7,"jul":7,
    "august":8,"aug":8, "september":9,"sep":9,"sept":9,
    "october":10,"oct":10, "november":11,"nov":11, "december":12,"dec":12
}
def _normalize(s: str) -> str:
    return re.sub(r"\s+", " ", str(s).strip().lower())
_MONTH_TOKEN = r"(jan|january|feb|february|mar|march|apr|april|may|jun|june|jul|july|aug|august|sep|sept|september|oct|october|nov|november|dec|december)"


def _parse_month_list(q: str) -> list[int]:
    """
    Return months mentioned in `q` in order of appearance, deduped.
    Works for: 'compare jan vs feb', 'compare jan,feb,mar', 'jan and feb and mar', etc.
    Requires at least 2 distinct months to be useful.
    """
    ql = (q or "").lower()
    # grab all month tokens in order
    toks = re.findall(rf"\b{_MONTH_TOKEN}\b", ql)
    months = []
    seen = set()
    for t in toks:
        m = _NAME2MON.get(t, None)
        if m and m not in seen:
            months.append(m)
            seen.add(m)
    return months

# Some extra aliases (mix English/Arabic/common typos); extend over time
_METRIC_SYNONYMS = {
    "net sales": ["net sales","net sale","net amount","net revenue","net",
                  "صافي","صافي المبيعات","المبيعات الصافية"],
    "gross sales": ["gross sales","gross","total gross",
                    "المبيعات الإجمالية","إجمالي المبيعات","إجمالي"],
    "vat amount": ["vat amount","vat","tax","0.15","15%","ضريبة","ضريبة القيمة المضافة"],
    "cash": ["cash","نقد","كاش"],
    "card": ["card","visa","pos","بطاقة"],
    "mada": ["mada","مدى"],
    "online": ["online","اونلاين","أونلاين","web","internet"],
    "jahez": ["jahez","جاهز"],
    "koinz": ["koinz","كوينز","كوينز"],
    "hunger": ["hunger"],
    "alinma": ["alinma","al inma","الإنماء","الانماء"],
    "gross": ["gross","إجمالي"],
    "net": ["net","صافي"]
}

_CHANNEL_SYNONYMS = {
    "cash": ["cash","كاش","نقد"],
    "card": ["card","visa","pos","بطاقة"],
    "mada": ["mada","مدى"],
    "online": ["online","اونلاين","web","internet"],
    "jahez": ["jahez","جاهز"],
    "koinz": ["koinz"],
    "hunger": ["hunger"],
    "alinma": ["alinma","al inma","الإنماء","الانماء"],
    # add more gateways if you use them (tap/stripe/etc.)
}
def _looks_like_channel(colname: str) -> tuple[str|None, int]:
    """
    Return (canonical_channel, score) if the column name matches a known channel, else (None,0).
    Uses fuzzy matching against _CHANNEL_SYNONYMS.
    """
    name = str(colname).lower().strip()
    best, best_sc = None, 0
    for canon, alts in _CHANNEL_SYNONYMS.items():
        sc = max(fuzz.token_set_ratio(name, a) for a in alts + [canon])
        if sc > best_sc:
            best, best_sc = canon, sc
    return (best if best_sc >= 70 else None, best_sc)

def _map_channel_columns(df: pd.DataFrame) -> dict[str, str]:
    """
    Scan df columns and return {canon_channel: column_name} dict for matched channels.
    If duplicates map to the same canon, keep the one with the highest score.
    """
    found: dict[str, tuple[str,int]] = {}
    for c in df.columns:
        if pd.api.types.is_numeric_dtype(df[c]):
            canon, sc = _looks_like_channel(c)
            if canon:
                if canon not in found or sc > found[canon][1]:
                    found[canon] = (c, sc)
    return {k:v for k,(v,_) in found.items()}

def _wants_channel_compare(q: str) -> bool:
    ql = (q or "").lower()
    # crude but effective: user names 2+ channels and says vs/and/',' or asks for pie of cash/card/online
    mentions = sum(1 for ch in _CHANNEL_SYNONYMS if re.search(rf"\b{ch}\b", ql))
    return ("vs" in ql or "," in ql or "and" in ql or "pie" in ql) and mentions >= 2
# ==== number word parsing (1..31) ====

_NUM_UNITS = {
    "one":1, "two":2, "three":3, "four":4, "five":5,
    "six":6, "seven":7, "eight":8, "nine":9,
}
_NUM_TEENS = {
    "ten":10, "eleven":11, "twelve":12, "thirteen":13, "fourteen":14,
    "fifteen":15, "sixteen":16, "seventeen":17, "eighteen":18, "nineteen":19,
}
_NUM_TENS = {"twenty":20, "thirty":30}

# Optional: a few common Arabic words (extend as you wish)
_NUM_AR = {
    "واحد":1, "اثنان":2, "اثنين":2, "ثلاثة":3, "اربعة":4, "أربعة":4, "خمسة":5,
    "ستة":6, "سبعة":7, "ثمانية":8, "تسعة":9, "عشرة":10, "احد عشر":11, "أحد عشر":11,
    "اثنا عشر":12, "إثنا عشر":12, "ثلاثة عشر":13, "اربعة عشر":14, "أربعة عشر":14,
    "خمسة عشر":15, "ستة عشر":16, "سبعة عشر":17, "ثمانية عشر":18, "تسعة عشر":19,
    "عشرون":20, "واحد وعشرون":21, "اثنان وعشرون":22, "اثنين وعشرون":22,
    "ثلاثة وعشرون":23, "اربعة وعشرون":24, "أربعة وعشرون":24,
    "خمسة وعشرون":25, "ستة وعشرون":26, "سبعة وعشرون":27,
    "ثمانية وعشرون":28, "تسعة وعشرون":29, "ثلاثون":30, "واحد وثلاثون":31,
}
# helpers (put near your other intent helpers)
_SINGLE_DAY_HINT = re.compile(
    r"\b(?:on\s+)?((?:19|20)\d{2}-\d{2}-\d{2}|"
    r"(jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)\s+\d{1,2})\b",
    re.IGNORECASE
)
def _is_single_day_intent(q: str) -> bool:
    ql = (q or "").lower()
    # explicitly exclude spans
    if re.search(r"\bdays?\b|\bweek\b", ql):
        return False
    return bool(_SINGLE_DAY_HINT.search(ql))
def _infer_compare_op(q: str) -> str:
    ql = (q or "").lower()
    if any(w in ql for w in ("avg", "average", "mean", "per day", "per-day", "perday")):
        return "avg"
    if any(w in ql for w in ("sum", "total", "overall", "cumulative")):
        return "sum"
    # default if unspecified
    return "sum"

def _mon_stat(daily: pd.DataFrame, month: int, op: str) -> float | None:
    """
    daily: columns ['date','val'] (already combined across files)
    month: 1..12
    op: 'sum' or 'avg'
    Returns None if no rows for that month.
    """
    sub = daily[daily["date"].map(lambda d: getattr(d, "month", None) == month)]
    if sub.empty:
        return None
    if op == "avg":
        # average per *day with data* in that month
        return float(sub["val"].mean())
    # 'sum'
    return float(sub["val"].sum())

def _strip_ordinal_suffix(tok: str) -> str:
    # 1st -> 1, 2nd -> 2, 3rd -> 3, 11th -> 11, etc.
    return re.sub(r"(?<=\d)(st|nd|rd|th)$", "", tok.lower())

def _word_to_int_1_31(s: str) -> int | None:
    """
    Convert number words/digits/ordinals (1..31) to int.
    Accepts: 'seven', 'eleven', 'twenty-one', 'twenty one', '7', '7th', Arabic samples above.
    """
    if not s:
        return None
    t = s.strip().lower()

    # digits / ordinals
    if re.fullmatch(r"\d{1,2}(st|nd|rd|th)?", t):
        n = int(_strip_ordinal_suffix(t))
        return n if 1 <= n <= 31 else None

    # quick Arabic lookup
    t_ar = t.replace("ـ", "").replace("–", "-").replace("—", "-")
    if t_ar in _NUM_AR:
        n = _NUM_AR[t_ar]
        return n if 1 <= n <= 31 else None

    # normalize hyphen to space (twenty-one -> twenty one)
    t = t.replace("-", " ")
    # simple words
    if t in _NUM_UNITS:
        return _NUM_UNITS[t]
    if t in _NUM_TEENS:
        return _NUM_TEENS[t]
    if t in _NUM_TENS:
        return _NUM_TENS[t]

    # compounds: twenty one .. twenty nine / thirty one
    parts = t.split()
    if len(parts) == 2 and parts[0] in _NUM_TENS and parts[1] in _NUM_UNITS:
        n = _NUM_TENS[parts[0]] + _NUM_UNITS[parts[1]]
        return n if 1 <= n <= 31 else None

    return None

def _num_from_phrase(s: str) -> int | None:
    """
    Try to find a number (1..31) in a small phrase like 'seven', '11th', 'twenty one'.
    """
    s = s.strip().lower()
    # single token try first
    n = _word_to_int_1_31(s)
    if n is not None:
        return n
    # fallback: scan tokens to catch 'day seven', etc.
    toks = re.split(r"[^\w\u0600-\u06FF]+", s)
    for i in range(len(toks)):
        # single token
        n = _word_to_int_1_31(toks[i])
        if n is not None:
            return n
        # two-token compound
        if i + 1 < len(toks):
            n = _word_to_int_1_31(toks[i] + " " + toks[i+1])
            if n is not None:
                return n
    return None

def _extract_metric_phrase(query: str) -> str:
    """
    Pull likely metric words from the question, e.g. 'sum of gross sales' -> 'gross sales'.
    Falls back to the whole query if nothing obvious.
    """
    q = _normalize(query)
    m = re.search(r"(sum|total|avg|average|mean|count)\s+(of\s+)?([a-z\u0600-\u06FF ]+)", q)
    if m:
        return _normalize(m.group(3))
    # try last noun-ish span
    m2 = re.search(r"(gross sales|net sales|gross|net|cash|card|mada|online|jahez|koinz|hunger|alinma|vat amount|vat|tax)", q)
    return _normalize(m2.group(0)) if m2 else q
def _maybe_derive_metric(df: pd.DataFrame, metric_phrase: str) -> pd.Series | None:
    """
    Try deriving common metrics if the exact column isn't present.
    - net ≈ sum(channels) − VAT Amount
    - gross ≈ Net + VAT Amount
    Returns a Series aligned to df or None.
    """
    cols = { _normalize(c): c for c in df.columns }
    # Channels present?
    channels = [c for c in df.columns if _normalize(c) in
                {_normalize(x) for x in ["Cash","Mada","Card","Online","Jahez","Koinz","Hunger","Alinma"]}]
    vat_col = next((cols.get(k) for k in cols if "vat amount" in k or k=="vat"), None)
    net_col = next((cols.get(k) for k in cols if "net sales" in k or k=="net"), None)
    gross_col = next((cols.get(k) for k in cols if "gross sales" in k or k=="gross"), None)

    want_net = any(w in metric_phrase for w in ["net","net sales"])
    want_gross = any(w in metric_phrase for w in ["gross","gross sales"])

    if want_net and channels:
        s = pd.DataFrame(df[channels]).apply(pd.to_numeric, errors="coerce").sum(axis=1)
        if vat_col:
            s = s - pd.to_numeric(df[vat_col], errors="coerce")
        return s

    if want_gross and (net_col or channels):
        if net_col:
            s = pd.to_numeric(df[net_col], errors="coerce")
        else:
            # derive net from channels as above
            if channels:
                s = pd.DataFrame(df[channels]).apply(pd.to_numeric, errors="coerce").sum(axis=1)
            else:
                return None
        if vat_col:
            s = s + pd.to_numeric(df[vat_col], errors="coerce")
        return s

    return None

def _best_metric_column(df: pd.DataFrame, query: str) -> tuple[str|None, str]:
    """
    Pick the best numeric column for the user's metric phrase using fuzzy matching and synonyms.
    Returns: (column_name or None, chosen_label_for_display)
    """
    phrase = _extract_metric_phrase(query)
    targets = {phrase}
    # expand using synonyms if we have an exact mapping
    for canon, alts in _METRIC_SYNONYMS.items():
        if any(alt in phrase for alt in alts):
            targets.update(alts)
    # score only numeric columns
    num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    best_col, best_sc = None, -1
    for c in num_cols:
        name = _normalize(c)
        # favor exact-ish containment, else fuzzy
        sc = 0
        if any(t in name or name in t for t in targets):
            sc = 100
        else:
            sc = max(fuzz.token_set_ratio(name, t) for t in targets)
        if sc > best_sc:
            best_sc, best_col = sc, c
    return (best_col if (best_sc >= 65) else None, phrase)

def _extract_month_year_from_text(q: str):
    ql = q.lower()
    y = m = None

    # ISO date like 2025-02-28
    m_iso = re.search(r'\b(19|20)\d{2}-(\d{2})-(\d{2})\b', ql)
    if m_iso:
        y = int(ql[m_iso.start():m_iso.end()].split('-')[0])
        m = int(ql[m_iso.start():m_iso.end()].split('-')[1])
        return m, y

    # existing year
    yobj = re.search(r"\b(20\d{2}|19\d{2})\b", ql)
    if yobj:
        y = int(yobj.group(1))

    # existing month tokens
    for tok, num in _MONTH_MAP.items():
        if re.search(rf"\b{tok}\b", ql):
            m = num
            break
    return m, y

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
    # English
    "jan": 1, "january": 1, "feb": 2, "february": 2, "mar": 3, "march": 3,
    "apr": 4, "april": 4, "may": 5, "jun": 6, "june": 6, "jul": 7, "july": 7,
    "aug": 8, "august": 8, "sep": 9, "sept": 9, "september": 9,
    "oct": 10, "october": 10, "nov": 11, "november": 11, "dec": 12, "december": 12,
    # Arabic
    "يناير": 1, "فبراير": 2, "مارس": 3, "أبريل": 4, "ابريل": 4, "مايو": 5,
    "يونيو": 6, "يوليو": 7, "أغسطس": 8, "اغسطس": 8, "سبتمبر": 9,
    "أكتوبر": 10, "اكتوبر": 10, "نوفمبر": 11, "ديسمبر": 12
}


def _parse_any_range(q: str):
    ql = q.lower()

    # ISO range
    m = re.search(r'\b(19|20)\d{2}-\d{2}-\d{2}\b', ql)
    if m:
        d = pd.to_datetime(m.group(0), errors="coerce")
        if pd.notna(d):
            a = d.date()
            return a, a
    m = re.search(r'(\d{4}-\d{2}-\d{2})\s*(?:to|–|—|-|\.{2,})\s*(\d{4}-\d{2}-\d{2})', ql)
    if m:
        a = pd.to_datetime(m.group(1), errors="coerce")
        b = pd.to_datetime(m.group(2), errors="coerce")
        if pd.notna(a) and pd.notna(b) and a <= b:
            return a.date(), b.date()

    # Month with day range: "Jan 1-30"
    m = re.search(r'(' + '|'.join(_MONTH_MAP.keys()) + r')\s*(\d{1,2})\s*(?:to|–|—|-)\s*(\d{1,2})', ql)
    if m:
        mo = _MONTH_MAP[m.group(1)]
        d1, d2 = int(m.group(2)), int(m.group(3))
        a = _safe_date(2000, mo, d1)
        b = _safe_date(2000, mo, d2)
        if a <= b:
            return a, b

    # Specific day (as a degenerate range): "Jan 15"
    m = re.search(r'(' + '|'.join(_MONTH_MAP.keys()) + r')\s*(\d{1,2})\b', ql)
    if m:
        mo = _MONTH_MAP[m.group(1)]
        d  = int(m.group(2))
        a = _safe_date(2000, mo, d)
        return a, a

    # Whole month mention: "February"
    m = re.search(r'\b(' + '|'.join(_MONTH_MAP.keys()) + r')\b', ql)
    if m:
        mo = _MONTH_MAP[m.group(1)]
        return _safe_month_span(mo, 2000)

    return None

def _parse_single_or_range(q: str):
    ql = q.lower()

    # ISO single day: 2025-02-14
    m = re.search(r'\b(\d{4}-\d{2}-\d{2})\b', ql)
    if m:
        d = pd.to_datetime(m.group(1), errors="coerce")
        if pd.notna(d):
            return ('day', d.date())

    # Month + day single: "Jan 15"
    m = re.search(r'(' + '|'.join(_MONTH_MAP.keys()) + r')\s*(\d{1,2})\b', ql)
    if m:
        mo = _MONTH_MAP[m.group(1)]
        d  = int(m.group(2))
        return ('day', _safe_date(2000, mo, d))

    # Month day–day range: "Jan 1-31", "Jan 5 to 12"
    m = re.search(r'(' + '|'.join(_MONTH_MAP.keys()) + r')\s*(\d{1,2})\s*(?:to|–|—|-)\s*(\d{1,2})', ql)
    if m:
        mo = _MONTH_MAP[m.group(1)]
        a, b = int(m.group(2)), int(m.group(3))
        a_dt = _safe_date(2000, mo, a)
        b_dt = _safe_date(2000, mo, b)
        if a_dt <= b_dt:
            return ('range', (a_dt, b_dt))

    # ISO range: "2025-02-01 to 2025-02-28"
    m = re.search(r'(\d{4}-\d{2}-\d{2})\s*(?:to|–|—|-|\.{2,})\s*(\d{4}-\d{2}-\d{2})', ql)
    if m:
        a = pd.to_datetime(m.group(1), errors="coerce")
        b = pd.to_datetime(m.group(2), errors="coerce")
        if pd.notna(a) and pd.notna(b) and a <= b:
            return ('range', (a.date(), b.date()))

    return None

def answer_sales_question(file_path: str, natural_question: str, sheet: str | None = None) -> str:
    """
    Loads the workbook (forgiving), aggregates full daily data, then answers.
    Never uses preview rows for calculations.
    """
    if SalesData is None:
        raise RuntimeError("SalesData module not available.")
    q = natural_question.strip()

    sd = SalesData(file_path, sheet=sheet)
    daily = sd.daily_net()   # columns: date, net (full month/day coverage)
    if daily.empty:
        return "I couldn't find any daily net sales in that file."

    # derive month/year from the question or fallback to file hints / data
    qm, qy = _extract_month_year_from_text(q)
    # if year not specified, prefer the file’s hint; else infer from daily dates
    file_m = sd.m_hint
    file_y = sd.y_hint
    if qm is None:
        qm = file_m or (daily["date"].iloc[0].month if not daily.empty else None)
    if qy is None:
        qy = file_y or (daily["date"].iloc[0].year if not daily.empty else None)

    # Filter by month/year if we have them; else sum everything we loaded
    df = daily.copy()
    if qm is not None:
        df = df[df["date"].apply(lambda d: d.month == qm)]
    if qy is not None:
        df = df[df["date"].apply(lambda d: d.year == qy)]

    if df.empty:
        # fallback: if question asked for a month we didn't find, say so gracefully
        if qm is not None:
            mon_name = [k for k,v in _MONTHS.items() if v == qm and len(k) > 3][0].capitalize()
            return f"I couldn't find any days for {mon_name}{(' ' + str(qy)) if qy else ''}."
        return "I couldn't find matching rows for that period."

    total = float(df["net"].sum())
    avg = float(df["net"].mean())
    days = int(len(df))
    best = df.loc[df["net"].idxmax()]
    worst = df.loc[df["net"].idxmin()]

    # If user asked explicitly for total net
    if "total" in q.lower() and "net" in q.lower():
        where = ""
        if qm is not None:
            mon_name = [k for k,v in _MONTHS.items() if v == qm and len(k) > 3][0].capitalize()
            where = f" for {mon_name}{(' ' + str(qy)) if qy else ''}"
        return f"Total net{where}: {total:,.2f} (across {days} day{'s' if days!=1 else ''})."

    # Otherwise handle a few common intents, then default preview
    if "average" in q.lower() or "avg" in q.lower():
        return f"Average net per day: {avg:,.2f} (based on {days} day{'s' if days!=1 else ''})."
    if "best" in q.lower():
        return f"Best day: {best['date']} — {best['net']:,.2f}"
    if "worst" in q.lower():
        return f"Worst day: {worst['date']} — {worst['net']:,.2f}"

    # default: summary
    return (
        f"Period days: {days}\n"
        f"Total net: {total:,.2f}\n"
        f"Average: {avg:,.2f}\n"
        f"Best: {best['date']} — {best['net']:,.2f}\n"
        f"Worst: {worst['date']} — {worst['net']:,.2f}"
    )


# ---------- Analytics primitives ----------

# === SalesData integration ===
try:
    from .data_loader import SalesData
except Exception as _e:
    SalesData = None

def load_sales_workbook(path: str, sheet: str = "Data"):
    """
    Unified loader the chatbot can call. Returns:
      (daily_df, totals_dict)

    Example:
        daily, totals = load_sales_workbook("/path/to/Sales_2025-01.xlsx")
    """
    if SalesData is None:
        raise RuntimeError("SalesData module not available.")
    sd = SalesData(path, sheet=sheet)
    return sd.daily_net(), sd.totals_dict()
# near other utils
def _ensure_tables(tables=None, hits=None) -> list[tuple[pd.DataFrame, dict]]:
    """
    Return a non-empty list of (df, meta) by:
      1) trusting `tables` if provided and non-empty,
      2) else loading from `hits`,
      3) then merging with all cached Parquet sheets.
    """
    out = tables or []
    if not out or all((df is None or df.empty) for df, _ in out):
        try:
            out = load_full_tables_from_hits(hits or []) or []
        except Exception:
            out = []
        if TABLE_STORE.exists():
            out = _merge_tables_dedup(out, _load_all_parquet_tables(filter_sources=None))
    return out

# --- totals per payment channel with sensible filtering ---
def _channel_totals(df: pd.DataFrame, query: str | None = None) -> dict[str, float]:
    """
    Return {"Cash": 123.0, "Card": ..., ...} using detected channel columns.
    - Honors channels explicitly mentioned in the query (cash/card/online/...), if any.
    - Drops obviously-empty channels (all NaN/0).
    """
    if df is None or df.empty:
        return {}

    ch_cols = _map_channel_columns(df)  # {canon -> actual column name}
    if not ch_cols:
        return {}

    ql = (query or "").lower()
    mentioned = {ch for ch in _CHANNEL_SYNONYMS if re.search(rf"\b{ch}\b", ql)}
    if mentioned:
        ch_cols = {k: v for k, v in ch_cols.items() if k in mentioned} or ch_cols

    totals: dict[str, float] = {}
    for canon, col in ch_cols.items():
        s = pd.to_numeric(df[col], errors="coerce")
        if s.notna().any() and s.abs().sum() > 0:
            totals[canon.capitalize()] = float(s.sum())

    # If everything summed to 0, still return zeros for at least one channel so callers can decide.
    if not totals and ch_cols:
        for canon, col in ch_cols.items():
            totals[canon.capitalize()] = float(pd.to_numeric(df[col], errors="coerce").sum())
    return totals

def _pretty_span(a, b):
    # a,b are date objects (year ignored by caller)
    return f"{a.month:02d}-{a.day:02d} to {b.month:02d}-{b.day:02d}"

def _safe_first_numeric(df: pd.DataFrame) -> str | None:
    for c in df.columns:
        if pd.api.types.is_numeric_dtype(df[c]):
            return c
    return None

def to_chart(query: str,
             tables: list[tuple[pd.DataFrame, dict]] | None = None,
             hits=None) -> str | None:
    """
    Render a Plotly chart that uses the SAME selection/aggregation as DF answers:
      • If user asks for channels compare -> pie/bar of channel totals for the span.
      • Otherwise -> bar/line of the chosen/derived metric by day for the span.

    IMPORTANT: We first select tables with select_tables_for_query, then aggregate
    with _compute_daily_metric(...). This keeps chart and text in sync.
    """
    ql = (query or "").lower()

    # Ensure we have tables
    tables = _ensure_tables(tables, hits)
    if not tables:
        return None

    # Use the same month/year-aware selector as the DF path
    picked = select_tables_for_query(query, tables) or tables
    if not picked:
        return None

    # ----- Channels branch (cash vs card ... ) -----
    if _wants_channel_compare(query):
        # Build one raw frame (only from picked tables) with a valid date column
        raws = []
        for df, _meta in picked:
            if df is None or df.empty:
                continue
            df2 = _maybe_flatten_columns(df)
            dcol = _find_date_col(df2)
            if not dcol:
                continue
            df2 = df2.copy()
            df2["_dt"] = pd.to_datetime(df2[dcol], errors="coerce")
            df2 = df2.dropna(subset=["_dt"])
            if not df2.empty:
                raws.append(df2)
        if not raws:
            return None

        raw = pd.concat(raws, ignore_index=True)

        # Apply the same span logic used elsewhere
        probe = (raw.assign(_d=raw["_dt"].dt.date)
                    .groupby("_d", as_index=False).size()
                    .rename(columns={"_d": "date", "size": "val"}))
        span = _resolve_month_day_range(query, probe)
        span_txt = "selected period"
        if span:
            a, b = span
            md = raw["_dt"].dt.month * 100 + raw["_dt"].dt.day
            raw = raw[md.between(a.month * 100 + a.day, b.month * 100 + b.day)]
            if raw.empty:
                return None
            span_txt = _pretty_span(a, b)

        totals = _channel_totals(raw, query)
        if not totals:
            return None

        dfp = pd.DataFrame({"Channel": list(totals.keys()), "Total": list(totals.values())})
        dfp = dfp.sort_values("Total", ascending=False)

        if (dfp["Total"] != 0).sum() >= MIN_NONZERO_FOR_PIE and "pie" in ql:
            fig = px.pie(dfp, names="Channel", values="Total",
                         title=f"{' / '.join(dfp['Channel'])} ({span_txt})")
            fig.update_traces(textposition="inside", textinfo="percent+label")
            return fig.to_html(full_html=False, include_plotlyjs=True, config={"responsive": True})
        else:
            fig = px.bar(dfp, x="Channel", y="Total",
                         title=f"Channel totals ({span_txt})",
                         labels={"Total": "Total", "Channel": "Channel"})
            return fig.to_html(full_html=False, include_plotlyjs=True, config={"responsive": True})

    # ----- Metric-over-time branch (e.g., "gross Feb 10–20") -----
    got = _compute_daily_metric(picked, query)
    if not got:
        return None
    daily, phrase = got
    if daily.empty:
        return None

    # Apply the exact same span filter as text answers
    span = _resolve_month_day_range(query, daily)
    span_txt = "selected period"
    if span:
        a, b = span
        lo = a.month * 100 + a.day
        hi = b.month * 100 + b.day
        md = daily["date"].map(lambda d: d.month * 100 + d.day)
        daily = daily[md.between(lo, hi)]
        if daily.empty:
            return None
        span_txt = _pretty_span(a, b)

    want_bar = ("bar" in ql) or ("column" in ql)
    title = f"{phrase.title()} ({span_txt})"
    fig = (px.bar(daily.sort_values("date"), x="date", y="val",
                  title=title, labels={"date": "Date", "val": phrase.title()})
           if want_bar else
           px.line(daily.sort_values("date"), x="date", y="val",
                   title=title, labels={"date": "Date", "val": phrase.title()}))

    # Inline plotly.js so it renders everywhere
    return fig.to_html(full_html=False, include_plotlyjs=True, config={"responsive": True})



# ===== Robust month + week/day-range parsing =====
from datetime import date

def _detect_month_in_query(q: str) -> int | None:
    ql = q.lower()
    for tok, num in _MONTH_MAP.items():
        if re.search(rf"\b{tok}\b", ql):
            return num
    return None
def _is_data_intent(query: str) -> bool:
    ql = (query or "").lower()
    ar_keys = [
        "إجمالي","المجموع","صافي","متوسط","مقارنة","ترند","اتجاه",
        "اليوم","أمس","هذا الأسبوع","هذا الشهر","الشهر الماضي","السنة الماضية",
        "شهري","أسبوعي","يومي","يناير","فبراير","مارس","أبريل","ابريل","مايو",
        "يونيو","يوليو","أغسطس","اغسطس","سبتمبر","أكتوبر","اكتوبر","نوفمبر","ديسمبر",
        "مخطط","رسم","بياني","أعمدة","خط","دائري","باي","نقد","كاش","بطاقة","مدى","أونلاين","اونلاين"
    ]
    if any(ch.isdigit() for ch in ql) or "%" in ql:
        return True
    return any(k in ql for k in ar_keys) or any(k in ql for k in [
        "sum","total","average","avg","compare","vs","trend","month","daily","weekly","chart","pie"
    ])
# ---------- Arabic helpers ----------

def is_ar_query(q: str) -> bool:
    ql = (q or "").strip().lower()
    ar_keys = [
        "إجمالي","المجموع","صافي","المبيعات","متوسط","أفضل","أسوأ",
        "يناير","فبراير","مارس","أبريل","ابريل","مايو","يونيو","يوليو",
        "أغسطس","اغسطس","سبتمبر","أكتوبر","اكتوبر","نوفمبر","ديسمبر",
        "مخطط","بياني","دائري","أعمدة","خط","قنوات","نقد","كاش","بطاقة","مدى","أونلاين","اونلاين"
    ]
    if any(k in ql for k in ar_keys):
        return True
    # وجود حروف عربية
    return any("\u0600" <= ch <= "\u06FF" for ch in ql)


_AR_MONTHS = {
    1:"يناير", 2:"فبراير", 3:"مارس", 4:"أبريل", 5:"مايو", 6:"يونيو",
    7:"يوليو", 8:"أغسطس", 9:"سبتمبر", 10:"أكتوبر", 11:"نوفمبر", 12:"ديسمبر"
}

def pretty_span_ar(a, b) -> str:
    """صياغة مدى الأيام مثل: ١–٧ يناير / ٣١ يناير–٢ فبراير (تبقى الأرقام هندية اختيارياً)."""
    import datetime as _dt
    def _d(n):
        # لو تبي أرقام عربية (١٢٣٤٥...) استبدل return str(n) بتحويل مخصص
        return str(n)
    if a.month == b.month:
        return f"{_d(a.day)}–{_d(b.day)} {_AR_MONTHS.get(a.month, a.strftime('%b'))}"
    return f"{_d(a.day)} {_AR_MONTHS.get(a.month, a.strftime('%b'))}–{_d(b.day)} {_AR_MONTHS.get(b.month, b.strftime('%b'))}"

def fmt_ar(n: float) -> str:
    """تنسيق أرقام عربية مع الفاصلة كفاصل آلاف (يمكن لاحقاً تحويل الأرقام لهندية)."""
    return f"{n:,.2f}".replace(",", "٬")  # فاصلة عربية رفيعة

def ar_metric_name(phrase_en: str) -> str:
    """تحويل اسم المقياس الظاهر إلى عربي مبسّط."""
    q = (phrase_en or "").strip().lower()
    if "net" in q or "صافي" in q: return "صافي المبيعات"
    if "gross" in q or "إجمالي" in q: return "إجمالي المبيعات"
    if "vat" in q or "ضريبة" in q: return "الضريبة"
    return "القيمة"  # اسم عام

def build_ar_single_period_text(daily_df, phrase_en: str, span_tuple=None) -> str:
    """نص عربي موجز للفترة المفصّلة (إجمالي/متوسط/أفضل/أسوأ)."""
    import pandas as pd
    if daily_df is None or daily_df.empty:
        return "لا توجد بيانات مطابقة للفترة المطلوبة."
    total = float(daily_df["val"].sum())
    avg   = float(daily_df["val"].mean())
    best  = daily_df.loc[daily_df["val"].idxmax()]
    worst = daily_df.loc[daily_df["val"].idxmin()]
    span_txt = pretty_span_ar(*span_tuple) if span_tuple else "الفترة المحددة"
    metric = ar_metric_name(phrase_en)

    # اليوم بصيغة YYYY-MM-DD كما في المصدر
    bdate = str(best["date"])
    wdate = str(worst["date"])
    return (
        f"{metric} — {span_txt}\n"
        f"الإجمالي: {fmt_ar(total)}   المتوسّط/اليوم: {fmt_ar(avg)}\n"
        f"أفضل يوم: {bdate} — {fmt_ar(float(best['val']))}   "
        f"أسوأ يوم: {wdate} — {fmt_ar(float(worst['val']))}"
    )

def build_ar_total_only_text(daily_df, phrase_en: str, span_tuple=None) -> str:
    """نص عربي مختصر يُرجع الإجمالي فقط (مفيد للسؤال من نوع: إجمالي صافي المبيعات...)."""
    if daily_df is None or daily_df.empty:
        return "لا توجد بيانات مطابقة للفترة المطلوبة."
    total = float(daily_df["val"].sum())
    metric = ar_metric_name(phrase_en)
    span_txt = pretty_span_ar(*span_tuple) if span_tuple else "الفترة المحددة"
    return f"إجمالي {metric} لـ{span_txt}: {fmt_ar(total)}"

def build_ar_month_compare_text(per_mon: dict[int, float|None], phrase_en: str, months_order: list[int]) -> str:
    """نص مقارنة شهور بالعربية (مجموع أو متوسط)."""
    def _v(v): return "لا بيانات" if v is None else fmt_ar(float(v))
    metric = ar_metric_name(phrase_en)
    parts = [f"{_AR_MONTHS.get(m, m)}: {_v(per_mon.get(m))}" for m in months_order]
    return f"مقارنة {metric} حسب الشهر: " + "  |  ".join(parts)

def build_ar_channel_totals_text(totals: dict[str,float], span_label: str) -> str:
    """نص حصص القنوات بالعربية مع النسب."""
    if not totals:
        return "لا توجد قنوات قابلة للعرض في هذه الفترة."
    grand = sum(abs(v) for v in totals.values()) or 1.0
    ordered = sorted(totals.items(), key=lambda kv: kv[1], reverse=True)
    items = [f"{k}: {fmt_ar(v)} ({(v/grand)*100:.1f}%)" for k, v in ordered]
    return f"إجماليات القنوات — {span_label}\n" + "  |  ".join(items)

def _dominant_month_from_daily(daily_df: pd.DataFrame) -> int | None:
    """Pick the most common month in already-aggregated daily data."""
    if daily_df is None or daily_df.empty or "date" not in daily_df.columns:
        return None
    months = daily_df["date"].map(lambda d: getattr(d, "month", None)).dropna().astype(int)
    return int(months.mode().iloc[0]) if not months.empty else None

_ORDINAL_WEEK_MAP = {
    "1": (1, 7), "first": (1, 7), "1st": (1, 7), "wk1": (1, 7), "week1": (1, 7),
    "2": (8, 14), "second": (8, 14), "2nd": (8, 14), "wk2": (8, 14), "week2": (8, 14),
    "3": (15, 21), "third": (15, 21), "3rd": (15, 21), "wk3": (15, 21), "week3": (15, 21),
    "4": (22, 28), "fourth": (22, 28), "4th": (22, 28), "wk4": (22, 28), "week4": (22, 28),
    # fifth week = remaining days
    "5": (29, 31), "fifth": (29, 31), "5th": (29, 31), "wk5": (29, 31), "week5": (29, 31),
    # common phrases
    "first week": (1, 7),
    "second week": (8, 14),
    "third week": (15, 21),
    "fourth week": (22, 28),
    "fifth week": (29, 31),
    "last week": (25, 31),
}

def _week_phrase_to_range(q: str) -> tuple[int, int] | None:
    """
    Return (start_day, end_day) for phrases like:
      'first week', 'first week in', 'first week of', 'first week days',
      'week 1', 'wk1', 'second week', 'week two', etc.
    """
    ql = re.sub(r"\s+", " ", q.lower())

    # 1) explicit numbers: week 1..5 / wk1..wk5 / week one..five
    #    (supports both digits and words)
    m = re.search(r"\b(?:week|wk)\s*([1-5]|one|two|three|four|five)\b", ql)
    if m:
        tok = m.group(1)
        word2idx = {"one":"1","two":"2","three":"3","four":"4","five":"5"}
        idx = word2idx.get(tok, tok)  # map words -> digit strings
        rng = _ORDINAL_WEEK_MAP.get(idx)
        if rng: return rng

    # 2) ordinal phrases (with optional 'in|of' and optional 'days')
    #    e.g., 'first week', 'first week in', 'first week of', 'first week days'
    for key, rng in _ORDINAL_WEEK_MAP.items():
        # keys include: 'first', 'second', 'third', 'fourth', 'fifth', 'first week', ...
        # Build tolerant patterns like: r'\bfirst(?:\s+week)?(?:\s+(?:in|of))?(?:\s+days)?\b'
        if key in ("1","2","3","4","5"):  # skip pure digits here; handled above
            continue
        core = re.escape(key)
        pat = rf"\b{core}(?:\s+week)?(?:\s+(?:in|of))?(?:\s+days)?\b"
        if re.search(pat, ql):
            return rng

    return None

def _first_or_last_n_days(q: str) -> tuple[int, int] | None:
    """
    'first 7 days', 'first ten days', 'last 5 days', 'last eleven days',
    'first half', 'second half'
    Returns (lo, hi) within month.
    """
    ql = q.lower()

    # First/last N days (numeric or words)
    m = re.search(r"\b(first|last)\s+([a-z\u0600-\u06FF\- ]+?)\s+days?\b", ql)
    if m:
        which, raw = m.group(1), m.group(2).strip()
        n = _num_from_phrase(raw)
        if n is not None:
            n = max(1, min(n, 31))
            return (1, n) if which == "first" else (32 - n, 31)

    # First/second half
    if re.search(r"\b(first\s+half)\b", ql):
        return (1, 15)
    if re.search(r"\b(second\s+half)\b", ql):
        return (16, 31)

    return None
# ===== Calendar-aware sheet selection =====

def _calendar_signature(df: pd.DataFrame) -> dict | None:
    """
    Inspect a table -> detect its date column -> build a compact calendar signature:
      - counts per (year, month)
      - dominant (year, month)
      - min/max real dates
    Returns None if no usable date column is found.
    """
    dcol = _find_date_col(df)
    if not dcol:
        return None
    ds = pd.to_datetime(df[dcol], errors="coerce")
    dvalid = ds.dropna()
    if dvalid.empty:
        return None
    ddates = dvalid.dt.date
    # counts per (Y,M)
    ym = dvalid.dt.to_period("M")
    counts = ym.value_counts().sort_index()
    # dominant month/year by count
    dom = counts.idxmax()
    dom_year, dom_month = dom.year, dom.month
    # min/max
    dmin, dmax = ddates.min(), ddates.max()
    # produce a dict mapping (year, month) -> count (json-serializable)
    ym_counts = {(int(p.year), int(p.month)): int(n) for p, n in counts.items()}
    return {
        "date_col": dcol,
        "counts_by_year_month": ym_counts,
        "dominant_year": int(dom_year),
        "dominant_month": int(dom_month),
        "min_date": dmin,
        "max_date": dmax,
    }

def _index_tables_by_month(tables: list[tuple[pd.DataFrame, dict]]) -> list[tuple[pd.DataFrame, dict]]:
    """
    For each (df, meta), attach a 'cal' signature to meta.
    """
    out = []
    for df, meta in tables:
        try:
            sig = _calendar_signature(df)
        except Exception:
            sig = None
        m2 = dict(meta or {})
        if sig:
            m2["cal"] = sig
        out.append((df, m2))
    return out

def _parse_month_year_from_query(q: str) -> tuple[int | None, int | None]:
    """
    Reuse your existing month parsing, but centralized for this selector.
    """
    return _extract_month_year_from_text(q)

def _score_table_for_month_year(meta: dict, want_m: int | None, want_y: int | None) -> tuple[int, int]:
    """
    Return a (primary, secondary) score tuple to rank tables for a requested (month, year).
    Higher is better.
      - primary: count of rows in that exact (y,m)
      - secondary: if year not specified, count of rows in any year with that month
                   (or the dominant month match as a fallback)
    """
    cal = (meta or {}).get("cal")
    if not cal:
        return (0, 0)
    counts = cal.get("counts_by_year_month", {})
    primary = 0
    secondary = 0
    if want_m is not None and want_y is not None:
        primary = counts.get((int(want_y), int(want_m)), 0)
    if want_m is not None:
        # sum counts across all years for that month
        secondary = sum(v for (y, m), v in counts.items() if m == int(want_m))
        # tiny boost if dominant month matches
        if cal.get("dominant_month") == int(want_m):
            secondary += 1
    return (primary, secondary)

def select_tables_for_query(query: str, tables: list[tuple[pd.DataFrame, dict]]) -> list[tuple[pd.DataFrame, dict]]:
    """
    Filter/rank tables by the month/year implied by the query.
    If a month is specified:
      - Prefer sheets whose dates actually contain that (year,month).
      - If no year in query, prefer any sheet whose *dominant* month equals it, or that has most rows for that month.
    If no month specified, return the tables unchanged (other logic will handle dominance).
    """
    if not tables:
        return tables

    # attach calendar signatures once
    itab = _index_tables_by_month(tables)

    want_m, want_y = _parse_month_year_from_query(query or "")
    if want_m is None and want_y is None:
        return itab  # nothing to filter by

    # score each table
    scored = []
    for df, meta in itab:
        p, s = _score_table_for_month_year(meta, want_m, want_y)
        scored.append((p, s, df, meta))

    # if nothing scores positively, keep original but with signatures (better than losing data)
    if not any(p > 0 or s > 0 for p, s, *_ in scored):
        # still, if want_m given, keep tables whose dominant_month equals it to reduce noise
        if want_m is not None:
            narrowed = [(df, meta) for _, _, df, meta in scored
                        if (meta.get("cal") or {}).get("dominant_month") == int(want_m)]
            if narrowed:
                return narrowed
        return itab

    # sort by primary then secondary descending
    scored.sort(key=lambda t: (t[0], t[1]), reverse=True)
    # keep all with same top primary/secondary to allow multi-sheet months
    top_p, top_s = scored[0][0], scored[0][1]
    picked = [(df, meta) for p, s, df, meta in scored if (p, s) == (top_p, top_s)]
    return picked

def _explicit_day_span(q: str) -> tuple[int, int] | None:
    ql = q.lower().replace("–", "-").replace("—", "-")
    m = re.search(
        r"\bdays?\s+(?:between\s+)?([a-z\u0600-\u06FF\-0-9 ]+?)\s*"
        r"(?:to|through|until|and|-)\s*([a-z\u0600-\u06FF\-0-9 ]+)\b",
        ql, flags=re.IGNORECASE
    )
    if m:
        a, b = _num_from_phrase(m.group(1)), _num_from_phrase(m.group(2))
        if a and b and 1 <= a <= 31 and 1 <= b <= 31 and a <= b:
            return a, b
    # ... keep your existing bare "7-11" branch ...
    return None

def _resolve_month_day_range(query: str, daily_df: pd.DataFrame) -> tuple[date, date] | None:
    """
    Priority:
      1) weeks & N-day spans (use month as context)
      2) explicit month-day ranges like 'Jan 1–31'
      3) whole-month fallback
    """
    qn = query.lower().replace('"', '').replace("’", "'")
    mo = _detect_month_in_query(qn) or _dominant_month_from_daily(daily_df)

    # ---- 1) Weeks & day-span phrases ----
    wk = _week_phrase_to_range(qn)
    if wk and mo:
        a_day, b_day = wk
        return date(1900, mo, a_day), date(1900, mo, b_day)

    span = _first_or_last_n_days(qn)
    if span and mo:
        a_day, b_day = span
        return date(1900, mo, a_day), date(1900, mo, b_day)

    dspan = _explicit_day_span(qn)
    if dspan and mo:
        a_day, b_day = dspan
        return date(1900, mo, a_day), date(1900, mo, b_day)

    # ---- 2) Explicit month-day range like 'Jan 1–31' ----
    rng = _parse_any_range(qn)
    if rng:
        a_dt, b_dt = rng
        # no clamping here; caller will overlap with available days
        return a_dt, b_dt

    # ---- 3) Whole-month fallback ----
    if mo and re.search(r'\b(' + '|'.join(_MONTH_MAP.keys()) + r')\b', qn):
        maxday = _max_day_in_month_from_daily(daily_df, mo) or _last_day_of_month(mo, 2000)
        return date(1900, mo, 1), date(1900, mo, maxday)

    return None
def _max_day_in_month_from_daily(daily_df: pd.DataFrame, month: int) -> int | None:
    """Return the max day available in `daily_df` for the given month (1..31), or None."""
    if daily_df is None or daily_df.empty or "date" not in daily_df.columns:
        return None
    days = (
        daily_df["date"]
        .dropna()
        .map(lambda d: (getattr(d, "month", None), getattr(d, "day", None)))
        .dropna()
    )
    days_in_month = [dy for mo, dy in days if mo == month and dy is not None]
    return max(days_in_month) if days_in_month else None
import calendar
from datetime import date  # you already import datetime; make sure date is imported too

def _last_day_of_month(mo: int, year: int = 2000) -> int:
    # 2000 is a leap year → Feb 29 is valid; year value won't matter later since we compare (month, day)
    return calendar.monthrange(year, mo)[1]

def _safe_date(year: int, mo: int, day: int) -> date:
    return date(year, mo, min(day, _last_day_of_month(mo, year)))

def _safe_month_span(mo: int, year: int = 2000) -> tuple[date, date]:
    return date(year, mo, 1), date(year, mo, _last_day_of_month(mo, year))

# --- Intent helpers (reuse your existing ones) ---
def _is_avg_intent(q: str) -> bool:
    ql = (q or "").lower()
    return ("avg" in ql or "average" in ql or "mean" in ql or
            re.search(r"\bper[-\s]?day\b", ql) is not None)

def _is_sum_intent(q: str) -> bool:
    ql = (q or "").lower()
    return any(w in ql for w in ("sum", "total", "overall"))

def _is_extrema_intent(q: str) -> tuple[bool,bool]:
    ql = (q or "").lower()
    return ("best" in ql or "max" in ql or "highest" in ql,
            "worst" in ql or "min" in ql or "lowest" in ql)
def _pretty_span(a, b):
    """Render date objects as 'Jan 1–7' or 'Jan 30–Feb 2'."""
    if a.month == b.month:
        return f"{a.strftime('%b')} {a.day}–{b.day}"
    return f"{a.strftime('%b')} {a.day}–{b.strftime('%b')} {b.day}"
# --- Add near your other constants/helpers ---
DEFAULT_EXCLUDE_CHANNELS = {"alinma", "bank", "tap", "stripe"}  # settlements/gateways you don't want in 'sales'
MIN_NONZERO_FOR_PIE = 3
TOP_K_CHANNELS = 5
SMALL_SLICE_FRACTION = 0.03  # <3% goes to Other

def _pretty_span(a, b):
    """Render date objects as 'Jan 1–7' or 'Jan 30–Feb 2'."""
    if a.month == b.month:
        return f"{a.strftime('%b')} {a.day}–{b.day}"
    return f"{a.strftime('%b')} {a.day}–{b.strftime('%b')} {b.day}"

def _mentioned(token: str, q: str) -> bool:
    return re.search(rf"\b{re.escape(token)}\b", (q or "").lower()) is not None

def _channel_totals(raw: pd.DataFrame, query: str) -> dict[str, float]:
    """
    Map channel-like columns -> totals with sensible pruning:
      - exclude settlements by default (Alinma/Bank/… unless explicitly mentioned in the query)
      - merge <3% slices into 'Other'
      - cap to Top-5 channels (rest -> 'Other')
    """
    ch_cols = _map_channel_columns(raw)
    if not ch_cols:
        return {}

    # dynamic exclude: only keep excluded channels if user asked for them
    dynamic_exclude = {c for c in DEFAULT_EXCLUDE_CHANNELS if not _mentioned(c, query)}
    totals = {
        canon.capitalize(): float(pd.to_numeric(raw[col], errors="coerce").sum())
        for canon, col in ch_cols.items()
        if canon not in dynamic_exclude
    }
    # drop zeros
    totals = {k: v for k, v in totals.items() if v != 0}
    if not totals:
        return {}

    # collapse tiny slices
    grand = sum(abs(v) for v in totals.values())
    if grand > 0:
        small_keys = [k for k, v in totals.items() if abs(v) < SMALL_SLICE_FRACTION * grand]
        if small_keys:
            other_sum = sum(totals.pop(k) for k in small_keys)
            if other_sum:
                totals["Other"] = totals.get("Other", 0.0) + other_sum

    # cap to Top-K
    if len(totals) > TOP_K_CHANNELS:
        items = sorted(totals.items(), key=lambda kv: kv[1], reverse=True)
        head, tail = items[:TOP_K_CHANNELS - 1], items[TOP_K_CHANNELS - 1:]
        totals = dict(head + [("Other", sum(v for _, v in tail))])

    return totals

def _wants_chart(q: str) -> bool:
    ql = (q or "").lower()
    return any(x in ql for x in ("chart","plot","graph","line","bar","column","pie","trend"))
def _compute_daily_metric(tables, query: str) -> tuple[pd.DataFrame, str] | None:
    """
    Pick the best table by date coverage, choose/derive the metric from the query,
    and return a daily aggregated df: columns ['date', 'val'] plus the chosen phrase.
    """
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
        days = 0
        if dcol:
            d = pd.to_datetime(df[dcol], errors="coerce").dt.date
            days = d.nunique()
        return (10 if dcol else 0) + min(days, 40)

    best = max(norm, key=table_score)
    dcol = _find_date_col(best)
    if dcol is None:
        return None

    # choose/derive metric
    mcol, phrase = _best_metric_column(best, query)
    if mcol is None:
        derived = _maybe_derive_metric(best, phrase)
        if derived is None:
            any_num = [c for c in best.columns if pd.api.types.is_numeric_dtype(best[c])]
            if not any_num:
                return None
            mcol = any_num[0]
            vals = pd.to_numeric(best[mcol], errors="coerce")
        else:
            vals = pd.to_numeric(derived, errors="coerce")
    else:
        vals = pd.to_numeric(best[mcol], errors="coerce")

    dates = pd.to_datetime(best[dcol], errors="coerce")
    daily = (
        pd.DataFrame({"date": dates.dt.date, "val": vals})
        .dropna(subset=["date", "val"])
        .groupby("date", as_index=False)["val"].sum()
    )
    return (daily, phrase)

# --- One unified DF path ---
def answer_with_df(query: str, hits):
    """
    DF-first answering with Arabic-aware phrasing.
      • 'Jan vs Feb' → month-compare summary.
      • Channels request ('cash vs card ...') → channels totals summary (if pie not rendered).
      • Otherwise → single-period metric summary.
    Returns (text, meta) or None.
    """
    ql = (query or "").lower()

    # ---- Arabic helpers (local to this function) ----------------------------
    AR_MONTHS = {
        1:"يناير", 2:"فبراير", 3:"مارس", 4:"أبريل", 5:"مايو", 6:"يونيو",
        7:"يوليو", 8:"أغسطس", 9:"سبتمبر", 10:"أكتوبر", 11:"نوفمبر", 12:"ديسمبر"
    }
    def _is_ar_query(q: str) -> bool:
        ar_keys = [
            "إجمالي","المجموع","صافي","المبيعات","متوسط","أفضل","أسوأ",
            "يناير","فبراير","مارس","أبريل","ابريل","مايو","يونيو","يوليو",
            "أغسطس","اغسطس","سبتمبر","أكتوبر","اكتوبر","نوفمبر","ديسمبر",
            "مخطط","بياني","دائري","أعمدة","خط","قنوات","نقد","كاش","بطاقة","مدى","أونلاين","اونلاين"
        ]
        ql_ = (q or "").strip().lower()
        return any(k in ql_ for k in ar_keys) or any("\u0600" <= ch <= "\u06FF" for ch in ql_)

    def _fmt_ar(n: float) -> str:
        return f"{n:,.2f}".replace(",", "٬")

    def _ar_metric_name(phrase_en: str) -> str:
        p = (phrase_en or "").strip().lower()
        if "net" in p or "صافي" in p: return "صافي المبيعات"
        if "gross" in p or "إجمالي" in p: return "إجمالي المبيعات"
        if "vat" in p or "ضريبة" in p: return "الضريبة"
        return "القيمة"

    def _pretty_span_ar(a, b) -> str:
        if a.month == b.month:
            return f"{a.day}–{b.day} {AR_MONTHS.get(a.month, a.strftime('%b'))}"
        return f"{a.day} {AR_MONTHS.get(a.month, a.strftime('%b'))}–{b.day} {AR_MONTHS.get(b.month, b.strftime('%b'))}"

    is_ar = _is_ar_query(query)

    # ---- 1) Multi-month compare --------------------------------------------
    months_req = _parse_month_list(query)
    if len(months_req) >= 2:
        tables = load_full_tables_from_hits(hits) or []
        if TABLE_STORE.exists():
            tables = _merge_tables_dedup(tables, _load_all_parquet_tables(filter_sources=None))
        if not tables:
            return None

        got = _compute_daily_metric_all(tables, query)
        if not got:
            return None
        daily, phrase = got
        if daily.empty:
            return (("لا توجد بيانات لهذه الفترة." if is_ar else f"No data found for {phrase} in the requested period."), {})

        years = [int(y) for y in re.findall(r"\b(19|20)\d{2}\b", query)]
        if years:
            daily = daily[daily["date"].map(lambda d: getattr(d, "year", None) in years)]

        op = _infer_compare_op(query)  # 'sum' or 'avg'
        per_mon = {m: _mon_stat(daily, m, op) for m in months_req}

        def mon_name_en(m): return datetime(2000, m, 1).strftime("%b")
        def mon_name_ar(m): return AR_MONTHS.get(m, str(m))
        mon_name = mon_name_ar if is_ar else mon_name_en

        a_m, b_m = months_req[0], months_req[1]
        a_v, b_v = per_mon[a_m], per_mon[b_m]

        if is_ar:
            # Arabic wording
            label = "الإجمالي" if op == "sum" else "المتوسط/اليوم"
            def vfmt(v): return _fmt_ar(v) if v is not None else "لا بيانات"
            head = f"مقارنة {_ar_metric_name(phrase)} حسب الشهر:"
            detail = f"{mon_name(a_m)}: {vfmt(a_v)}  |  {mon_name(b_m)}: {vfmt(b_v)}"
            if a_v is not None and b_v is not None:
                diff = b_v - a_v
                pct = (diff / a_v * 100.0) if a_v != 0 else None
                tail = f"الفرق ({mon_name(b_m)} − {mon_name(a_m)}): {_fmt_ar(diff)}" + (f" ({pct:+.2f}%)" if pct is not None else "")
            else:
                tail = "الفرق: غير متاح (أحد الشهور بلا بيانات)"
            more = ""
            if len(months_req) > 2:
                more = "  |  " + "  ".join(f"{mon_name(m)}: {vfmt(per_mon[m])}" for m in months_req[2:])
            year_hint = (f" (السنوات: {', '.join(map(str, years))})" if years else "")
            text = f"{head}\n{label}: {detail}  {tail}{more}{year_hint}"
        else:
            # English wording (original)
            op_label = "Total" if op == "sum" else "Average per day"
            def fmt(v): return f"{v:,.2f}" if v is not None else "no data"
            if a_v is not None and b_v is not None and op == "avg":
                higher = b_m if b_v > a_v else a_m
                lower  = a_m if b_v > a_v else b_m
                text_head = (f"{mon_name_en(higher)} has higher average {phrase} per day "
                             f"({mon_name_en(higher)} {max(a_v, b_v):,.2f} vs {mon_name_en(lower)} {min(a_v, b_v):,.2f}).")
            else:
                text_head = f"{phrase.title()} {mon_name_en(a_m)} Vs {mon_name_en(b_m)} — month compare:"
            detail = f"{mon_name_en(a_m)}: {fmt(a_v)}  {mon_name_en(b_m)}: {fmt(b_v)}"
            if a_v is not None and b_v is not None:
                diff = b_v - a_v
                pct = (diff / a_v * 100.0) if a_v != 0 else None
                pct_part = f" ({pct:+.2f}%)" if pct is not None else ""
                tail = f"Difference ({mon_name_en(b_m)} - {mon_name_en(a_m)}): {diff:,.2f}{pct_part}"
            else:
                tail = f"Difference ({mon_name_en(b_m)} - {mon_name_en(a_m)}): not available (one month has no data)"
            more = ""
            if len(months_req) > 2:
                more_list = [f"{mon_name_en(m)}: {fmt(per_mon[m])}" for m in months_req[2:]]
                more = "  |  " + "  ".join(more_list)
            year_hint = (" (years: " + ", ".join(map(str, years)) + ")") if years else ""
            text = f"{text_head}\n{op_label}: {detail}  {tail}{more}{year_hint}"

        srcs = []
        for _, m in tables:
            s = (m or {}).get("source")
            if s and s not in srcs:
                srcs.append(s)
        meta = {"mode": "month_compare", "sources": srcs[:8]}
        return (text, meta)

    # ---- 2) Channels text fallback -----------------------------------------
    if _wants_channel_compare(query):
        tables = load_full_tables_from_hits(hits) or []
        if TABLE_STORE.exists():
            tables = _merge_tables_dedup(tables, _load_all_parquet_tables(filter_sources=None))
        if not tables:
            return None

        norm = [_maybe_flatten_columns(df) for (df, _) in tables if df is not None and not df.empty]
        if not norm:
            return None

        def _score(df):
            dcol = _find_date_col(df)
            if not dcol: return 0
            return (10 if dcol else 0) + pd.to_datetime(df[dcol], errors="coerce").dt.date.nunique()

        best = max(norm, key=_score)
        dcol = _find_date_col(best)
        if not dcol:
            return None

        raw = best.copy()
        raw["_dt"] = pd.to_datetime(raw[dcol], errors="coerce")
        raw = raw.dropna(subset=["_dt"])
        if raw.empty:
            return None

        _probe = (raw.assign(_d=raw["_dt"].dt.date)
                  .groupby("_d", as_index=False).size()
                  .rename(columns={"_d": "date", "size": "val"}))
        span = _resolve_month_day_range(query, _probe)
        span_txt_en = "selected period"
        if span:
            a, b = span
            md = raw["_dt"].dt.month * 100 + raw["_dt"].dt.day
            raw = raw[md.between(a.month * 100 + a.day, b.month * 100 + b.day)]
            if raw.empty:
                return None
            span_txt_en = _pretty_span(a, b)
        span_txt = _pretty_span_ar(*span) if (is_ar and span) else (span_txt_en)

        totals = _channel_totals(raw, query)
        if not totals:
            return None

        if is_ar:
            # Arabic list with percentages
            grand = sum(abs(v) for v in totals.values()) or 1.0
            ordered = sorted(totals.items(), key=lambda kv: kv[1], reverse=True)
            parts = [f"{k}: {_fmt_ar(v)} ({(v / grand) * 100:.1f}%)" for k, v in ordered]
            text = f"إجماليات القنوات — {span_txt}\n" + "  |  ".join(parts)
        else:
            grand = sum(abs(v) for v in totals.values()) or 1.0
            ordered = sorted(totals.items(), key=lambda kv: kv[1], reverse=True)
            parts = [f"{k}: {v:,.2f} ({(v / grand) * 100:.1f}%)" for k, v in ordered]
            text = f"Channel totals — {span_txt}\n" + "  |  ".join(parts)

        meta = {"mode": "channel_totals", "channels": ordered}
        return (text, meta)

    # ---- 3) Default single-period metric summary ---------------------------
    tables = load_full_tables_from_hits(hits) or []
    if TABLE_STORE.exists():
        tables = _merge_tables_dedup(tables, _load_all_parquet_tables(filter_sources=None))
    if not tables:
        return None

    picked = select_tables_for_query(query, tables)
    got = _compute_daily_metric(picked, query)
    if not got:
        return None
    daily, phrase = got
    if daily.empty:
        return (("لا توجد بيانات لهذه الفترة." if is_ar else "No data found for the requested period."), {})

    span = _resolve_month_day_range(query, daily)
    if span:
        a, b = span
        lo = a.month * 100 + a.day
        hi = b.month * 100 + b.day
        md = daily["date"].map(lambda d: d.month * 100 + d.day)
        daily = daily[md.between(lo, hi)]

    if daily.empty:
        return (("لا توجد بيانات لهذه الفترة." if is_ar else "No data found for the requested period."), {})

    # English numbers (for backward-compat) if not Arabic
    total = float(daily["val"].sum())
    avg   = float(daily["val"].mean())
    best  = daily.loc[daily["val"].idxmax()]
    worst = daily.loc[daily["val"].idxmin()]

    if is_ar:
        # If the user asked for "إجمالي ..." specifically, return total-only Arabic
        total_only = any(k in ql for k in ["إجمالي","المجموع"]) and any(
            k in ql for k in ["صافي","صافي المبيعات","إجمالي المبيعات","المبيعات الإجمالية"]
        )
        span_txt = _pretty_span_ar(*span) if span else "الفترة المحددة"
        metric = _ar_metric_name(phrase)

        if total_only:
            text = f"إجمالي {metric} لـ{span_txt}: {_fmt_ar(total)}"
        else:
            bdate = str(best["date"]); wdate = str(worst["date"])
            text = (
                f"{metric} — {span_txt}\n"
                f"الإجمالي: {_fmt_ar(total)}   المتوسّط/اليوم: {_fmt_ar(avg)}\n"
                f"أفضل يوم: {bdate} — {_fmt_ar(float(best['val']))}   "
                f"أسوأ يوم: {wdate} — {_fmt_ar(float(worst['val']))}"
            )
    else:
        span_txt = _pretty_span(*span) if span else "selected period"
        text = (
            f"{phrase.title()} — {span_txt}\n"
            f"Total: {total:,.2f}   Average/day: {avg:,.2f}\n"
            f"Best: {best['date']} — {best['val']:,.2f}   "
            f"Worst: {worst['date']} — {worst['val']:,.2f}"
        )

    srcs = []
    for _, m in picked:
        s = (m or {}).get("source")
        if s and s not in srcs:
            srcs.append(s)
    meta = {"mode": "single_period", "sources": srcs[:8]}
    return (text, meta)

