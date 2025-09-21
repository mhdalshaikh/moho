# test.py
import re
from pathlib import Path
from datetime import datetime
import pandas as pd
from rapidfuzz import fuzz

# === CHANGE THIS if your path differs ===
PARQUET_PATH = r"data_cache\tables\JANUARY 2025 VAT, EXPENCES & SALES REPORT__SALES REPORT__1.parquet"

# ---- name hints pulled from file/sheet name ----
MONTH_MAP = {
    "jan": 1, "january": 1, "feb": 2, "february": 2, "mar": 3, "march": 3,
    "apr": 4, "april": 4, "may": 5, "jun": 6, "june": 6, "jul": 7, "july": 7,
    "aug": 8, "august": 8, "sep": 9, "sept": 9, "september": 9, "oct": 10,
    "october": 10, "nov": 11, "november": 11, "dec": 12, "december": 12
}
MONTH_WORDS = list(MONTH_MAP.keys())

HEADER_HINTS = [
    "date", "day", "invoice", "posting", "transaction", "التاريخ",
    "net sales", "net amount", "net revenue", "vat", "cash", "bank", "remarks", "notes"
]

NET_TOKENS = [
    "net sales", "net sale", "net amount", "net revenue",
    "sales net", "sales (net)", "صافي المبيعات"
]
CHANNEL_TOKENS = [
    "cash", "bank", "pos", "card", "visa", "mada", "online",
    "koinz", "jahez", "hunger", "alinma", "al inma", "tap", "stripe"
]
VAT_TOKENS = ["vat", "tax", "15%", "0.15", "ضريبة"]


def month_year_hint_from_path(p: Path):
    name = p.name.lower()
    m = None
    for key in MONTH_WORDS:
        if key in name:
            m = MONTH_MAP[key]
            break
    y = None
    mobj = re.search(r"(20\d{2}|19\d{2})", name)
    if mobj:
        y = int(mobj.group(1))
    return m, y


def excel_serial_to_datetime_safe(s: pd.Series) -> pd.Series:
    """
    Convert only plausible Excel serials to datetime.
    Windows base is 1899-12-30 (Excel's bug-adjusted epoch).
    Only accept [20000..60000] (~1954-11-27 to ~2064-05-05) to avoid amounts.
    """
    ser = pd.to_numeric(pd.Series(s), errors="coerce")
    mask = ser.between(20000, 60000)
    out = pd.Series(pd.NaT, index=ser.index, dtype="datetime64[ns]")
    if mask.any():
        base = pd.Timestamp("1899-12-30")
        safe_days = ser[mask].astype("Int64")  # keep NA-aware int
        out.loc[mask] = base + pd.to_timedelta(safe_days.astype("float"), unit="D")
    return out


def parse_day_mon_tokens(s: pd.Series, year_hint: int | None) -> pd.Series:
    """
    Parse strings like '1-JAN', '01 JAN', '1 / Jan' (case-insensitive).
    Accept only if >50% rows match; returns datetime Series (NaT elsewhere).
    """
    s = pd.Series(s).astype("string")
    m = s.str.extract(r'^\s*(\d{1,2})\s*[-/ ]\s*([A-Za-z]{3,})\s*$', expand=True)
    out = pd.Series(pd.NaT, index=s.index, dtype="datetime64[ns]")
    if m.empty:
        return out

    day = pd.to_numeric(m[0], errors="coerce")
    mon = (
        m[1].str.lower().str.slice(0, 4).str.replace(r'[^a-z]', '', regex=True)
        .map({
            'jan': 1, 'feb': 2, 'mar': 3, 'apr': 4, 'may': 5, 'jun': 6,
            'jul': 7, 'aug': 8, 'sep': 9, 'sept': 9, 'oct': 10, 'nov': 11, 'dec': 12
        })
    )
    ok = day.between(1, 31) & mon.notna()
    if ok.mean() <= 0.5:
        return out

    y = year_hint or 1900  # year-less math; filter later by month/day
    out.loc[ok] = pd.to_datetime(
        {"year": y, "month": mon[ok].astype("Int64"), "day": day[ok].astype("Int64")},
        errors="coerce"
    )
    return out


def promote_header(df: pd.DataFrame) -> pd.DataFrame:
    head = df.head(12)

    def header_score(vals) -> int:
        s = 0
        for v in vals:
            t = str(v).strip().lower()
            if not t or t.startswith("unnamed"):
                continue
            if len(t) <= 40 and not re.fullmatch(r"[-+]?\d*\.?\d+", t):
                s += 1
            if any(h in t for h in HEADER_HINTS):
                s += 2
        return s

    best_i, best_s = 0, -1
    for i in range(len(head)):
        sc = header_score(list(head.iloc[i].values))
        if sc > best_s:
            best_i, best_s = i, sc

    new_cols = [str(x).strip() if str(x).strip() else f"col_{j}" for j, x in enumerate(df.iloc[best_i])]
    out = df.iloc[best_i + 1:].copy()
    out.columns = new_cols
    return out


def clean_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.dropna(how="all").dropna(axis=1, how="all")

    # unique, non-empty headers
    cols, seen = [], {}
    for c in df.columns:
        name = (str(c) or "").strip() or "col"
        if name in seen:
            seen[name] += 1
            name = f"{name}_{seen[name]}"
        else:
            seen[name] = 0
        cols.append(name)
    df.columns = cols

    # strip object/string columns by position
    for i in range(df.shape[1]):
        col = df.iloc[:, i]
        if col.dtype == "object":
            df.iloc[:, i] = col.astype("string").str.strip()

    return df


def normalize(df: pd.DataFrame) -> pd.DataFrame:
    if all(str(c).lower().startswith("unnamed") for c in df.columns):
        df = promote_header(df)
    return clean_df(df)


def parse_dates_smart(df: pd.DataFrame, month_hint: int | None, year_hint: int | None) -> pd.Series | None:
    """
    Try multiple strategies per column; return the best date series (max non-na rate).
    Strategies:
      - direct to_datetime (dayfirst True)
      - excel serial (range-limited)
      - ISO 'YYYY-MM-DD' inside text
      - 'Jan 5' / '5 Jan' (month names)
      - numbers 1..31 + (month/year) hints
    """
    best, best_rate = None, -1.0

    for col in df.columns:
        s_raw = df[col]
        s_str = s_raw.astype("string")

        candidates: list[pd.Series] = []

        # A) direct parse
        cand_A = pd.to_datetime(s_str, errors="coerce", dayfirst=True)
        if cand_A.notna().mean() > 0.5:
            candidates.append(cand_A)

        # B) excel serials (safe)
        cand_B = excel_serial_to_datetime_safe(s_str)
        if cand_B.notna().mean() > 0.5:
            candidates.append(cand_B)

        # C) ISO in text
        iso = s_str.str.extract(r"(?P<iso>\d{4}-\d{2}-\d{2})")["iso"]
        cand_C = pd.to_datetime(iso, errors="coerce")
        if cand_C.notna().mean() > 0.5:
            candidates.append(cand_C)

        # D) Month name + day
        md = s_str.str.extract(
            r"(?P<m>(?i:" + "|".join(MONTH_WORDS) + r"))\s*[-/., ]*\s*(?P<d>\d{1,2})"
        )
        if not md.empty:
            mo = md["m"].str.lower().map(MONTH_MAP)
            dy = pd.to_numeric(md["d"], errors="coerce")
            y = year_hint if year_hint else 1900
            cand_D = pd.to_datetime(dict(year=y, month=mo, day=dy), errors="coerce")
            if cand_D.notna().mean() > 0.5:
                candidates.append(cand_D)

        # E) pure day numbers 1..31 with hints (NA-safe)
        if month_hint:
            nums = pd.to_numeric(s_str.str.extract(r"\b(\d{1,2})\b")[0], errors="coerce")
            mask_valid = nums.between(1, 31).fillna(False)
            if mask_valid.mean() > 0.5:
                base = pd.Timestamp(year=(year_hint or 1900), month=month_hint, day=1)
                cand_E = pd.Series(pd.NaT, index=df.index, dtype="datetime64[ns]")
                cand_E.loc[mask_valid] = base + pd.to_timedelta(nums[mask_valid] - 1, unit="D")
                candidates.append(cand_E)

        # F) tokens like "1-JAN", "01 JAN"
        cand_F = parse_day_mon_tokens(s_str, year_hint=year_hint)
        if cand_F.notna().mean() > 0.5:
            candidates.append(cand_F)

        # pick best candidate for this column
        for cnd in candidates:
            rate = cnd.notna().mean()
            if rate > best_rate:
                best_rate = rate
                best = cnd

    # accept even somewhat sparse dates; we’ll group & ffill
    return best.ffill() if (best is not None and best_rate >= 0.2) else None


def find_net_series(df: pd.DataFrame) -> pd.Series | None:
    # 1) try a fuzzy-named numeric column
    best_name, best_sc = None, -1
    for c in df.columns:
        ser = pd.to_numeric(df[c], errors="coerce")
        if ser.notna().mean() < 0.4:
            continue
        name = str(c).lower()
        sc = max(fuzz.token_set_ratio(name, t) for t in NET_TOKENS)
        if sc > best_sc:
            best_sc, best_name = sc, c
    if best_name and best_sc >= 65:
        return pd.to_numeric(df[best_name], errors="coerce")

    # 2) derive from channels - VAT
    channels, vats = [], []
    for c in df.columns:
        ser = pd.to_numeric(df[c], errors="coerce")
        if ser.notna().mean() < 0.4:
            continue
        name = str(c).lower()
        ch = max(fuzz.token_set_ratio(name, t) for t in CHANNEL_TOKENS)
        vt = max(fuzz.token_set_ratio(name, t) for t in VAT_TOKENS)
        if ch >= 70 and vt < 60:
            channels.append(c)
        if vt >= 70:
            vats.append(c)
    if channels:
        net = df[channels].apply(pd.to_numeric, errors="coerce").sum(axis=1)
        if vats:
            net = net - df[vats].apply(pd.to_numeric, errors="coerce").sum(axis=1)
        return net

    return None

DAY_COL_RE = re.compile(r'^(?:0?[1-9]|[12]\d|3[01])(?:st|nd|rd|th)?$', re.I)
_ORD_RE = re.compile(r'^\s*(\d{1,2})(?:st|nd|rd|th)?\s*$', re.I)
PERIOD_TOKENS = {"period", "shift", "session", "am/pm", "am pm"}

def _coerce_day_ordinal(series: pd.Series) -> pd.Series:
    s = series.astype("string")
    m = s.str.extract(_ORD_RE)[0]
    d = pd.to_numeric(m, errors="coerce")
    return d.where(d.between(1, 31))
def extract_daily_cut_total(df: pd.DataFrame, month_hint: int | None, year_hint: int | None):
    df = normalize(df).copy()

    # 0) Cut off at TOTAL: if any row has "TOTAL" anywhere, stop above it.
    mask_total_any = df.apply(lambda col: col.astype("string").str.strip().str.upper().eq("TOTAL"), axis=1).any(axis=1)
    if mask_total_any.any():
        df = df.iloc[:mask_total_any.idxmax()]  # everything above the first TOTAL row

    # 1) Date column
    date_col = None
    for c in df.columns:
        if str(c).strip().lower() in {"date", "day", "التاريخ"}:
            date_col = c
            break
    if date_col is None:
        return None

    # 2) Period column (AM/PM) if it exists
    period_col = None
    # prefer explicit AM/PM content
    for c in df.columns:
        vals = df[c].astype("string").str.upper()
        if vals.isin(["AM", "PM"]).mean() >= 0.5:
            period_col = c
            break
    if period_col is None:
        # fallback by name
        for c in df.columns:
            if any(tok in str(c).strip().lower() for tok in PERIOD_TOKENS):
                period_col = c
                break

    # 3) NET SALES column
    net_col = find_fuzzy_col(df, NET_TOKENS, min_score=65)
    if net_col is None:
        # try exact contains
        for c in df.columns:
            if "net" in str(c).lower() and "sale" in str(c).lower():
                net_col = c
                break
    if net_col is None:
        return None

    # 4) Keep just data rows (optionally filter to AM/PM if present)
    df2 = df.copy()
    if period_col:
        per = df2[period_col].astype("string").str.upper()
        df2 = df2[per.isin(["AM", "PM"])]

    # 5) Build robust dates:
    dates = coerce_any_date(df2[date_col], month_hint, year_hint).ffill()

    # 6) Aggregate
    net_vals = pd.to_numeric(df2[net_col], errors="coerce")
    daily = (
        pd.DataFrame({"date": dates, "net": net_vals})
        .dropna(subset=["date"])
        .groupby(pd.to_datetime(dates, errors="coerce").dt.date, as_index=False)["net"]
        .sum()
    )
    return daily

def find_fuzzy_col(df: pd.DataFrame, tokens: list[str], min_score: int = 70) -> str | None:
    best, sc = None, -1
    for c in df.columns:
        name = str(c).lower()
        s = max(fuzz.token_set_ratio(name, t) for t in tokens)
        if s > sc:
            best, sc = c, s
    return best if sc >= min_score else None

def try_wide_day_matrix(df: pd.DataFrame, month_hint: int | None, year_hint: int | None):
    # 1) find the label column (mostly non-numeric strings)
    label_col = None
    for c in df.columns:
        s = df[c].astype("string")
        nonnum_ratio = pd.to_numeric(s, errors="coerce").isna().mean()
        if nonnum_ratio >= 0.7:
            label_col = c
            break
    if label_col is None:
        return None

    # 2) collect day columns like '1st','2nd',...,'31st'
    day_cols = [c for c in df.columns if isinstance(c, str) and DAY_COL_RE.match(c.strip())]
    if not day_cols:
        return None

    # 3) melt to long
    day_map = {c: int(re.match(r'(\d{1,2})', c).group(1)) for c in day_cols}
    wide = df[[label_col] + day_cols].copy()
    long = wide.melt(id_vars=[label_col], var_name="day", value_name="value")
    long["day"] = long["day"].map(day_map)

    y = year_hint or datetime.now().year
    m = month_hint or 1
    long["date"] = pd.to_datetime(dict(year=y, month=m, day=long["day"]), errors="coerce")

    # 4) pick the NET SALES row (fuzzy)
    def best_label(labels):
        uniq = pd.Index(labels.astype("string").str.lower().unique())
        def score(t): return max(fuzz.token_set_ratio(t, tok) for tok in NET_TOKENS)
        sc = {t: score(t) for t in uniq}
        best = max(sc, key=sc.get)
        return best if sc[best] >= 65 else None

    target = best_label(long[label_col])
    if not target:
        return None

    net_daily = (
        long[long[label_col].astype("string").str.lower() == target]
        .dropna(subset=["date"])
        .groupby("date", as_index=False)["value"].sum()
        .rename(columns={"value": "net"})
    )
    return net_daily
def coerce_any_date(series: pd.Series, month_hint: int | None, year_hint: int | None) -> pd.Series:
    """Return a datetime64[ns] Series from various date-like contents."""
    sr = pd.Series(series)

    # A) direct parse (dd/mm or iso, mixed text)
    d = pd.to_datetime(sr, errors="coerce", dayfirst=True)
    if d.notna().mean() >= 0.5:
        return d

    # B) strict common formats
    for fmt in ("%d/%m/%Y", "%Y-%m-%d", "%d-%m-%Y", "%d.%m.%Y"):
        d2 = pd.to_datetime(sr, format=fmt, errors="coerce")
        if d2.notna().mean() >= 0.5:
            return d2

    # C) Excel serials
    d3 = excel_serial_to_datetime_safe(sr)
    if d3.notna().mean() >= 0.5:
        return d3

    # D) ordinals like '1st'
    day_ord = _coerce_day_ordinal(sr)
    if day_ord.notna().mean() >= 0.5 and month_hint is not None:
        y = year_hint or datetime.now().year
        m = month_hint
        return pd.to_datetime(
            {"year": y, "month": m, "day": day_ord.astype("Int64")},
            errors="coerce"
        )

    # E) pure integers 1..31  (FIX: fillna(False) before mean)
    dn = pd.to_numeric(sr, errors="coerce")
    valid = dn.between(1, 31)
    if valid.fillna(False).mean() >= 0.5 and month_hint is not None:
        y = year_hint or datetime.now().year
        m = month_hint
        dn = dn.clip(1, 31).astype("Int64")
        return pd.to_datetime({"year": y, "month": m, "day": dn}, errors="coerce")

    # F) tokens like '5 Jan' or 'Jan 5'
    cand_F = parse_day_mon_tokens(sr.astype("string"), year_hint=year_hint)
    if cand_F.notna().mean() >= 0.5:
        return cand_F

    return pd.Series(pd.NaT, index=sr.index, dtype="datetime64[ns]")

def main():
    p = Path(PARQUET_PATH)
    if not p.exists():
        raise FileNotFoundError(f"Parquet not found: {p}")

    df0 = pd.read_parquet(p)
    df = normalize(df0)
    print("Columns:", list(df.columns)[:40])

    m_hint, y_hint = month_year_hint_from_path(p)

    # NEW: try the “long” table with TOTAL cutoff first
    daily = extract_daily_cut_total(df, m_hint, y_hint)

    if daily is None or daily.empty:
        # fallbacks you already have:
        dates = parse_dates_smart(df, m_hint, y_hint)
        if dates is None:
            net_daily = try_wide_day_matrix(df, m_hint, y_hint)
            if net_daily is None or net_daily.empty:
                raise RuntimeError("No date-like signal found (long+TOTAL, column dates, or wide day matrix).")
            daily = net_daily[["date", "net"]].copy()
        else:
            net = find_net_series(df)
            if net is None:
                raise RuntimeError("Could not find or derive a NET SALES series.")
            daily = (
                pd.DataFrame({"date": dates, "net": pd.to_numeric(net, errors="coerce")})
                .dropna(subset=["date"])
                .groupby(pd.to_datetime(dates, errors="coerce").dt.date, as_index=False)["net"]
                .sum()
            )

    # Sanity prints
    print("Date range:", daily["date"].min(), "→", daily["date"].max(), " | rows:", len(daily))

    # Example lookup (Jan 5)
    row = daily[daily["date"].apply(lambda x: (x.month, x.day)) == (1, 5)]
    if row.empty:
        print("No NET SALES found for Jan 5.")
    else:
        print("NET SALES on Jan 5:", float(row["net"].iloc[0]))


if __name__ == "__main__":
    main()
