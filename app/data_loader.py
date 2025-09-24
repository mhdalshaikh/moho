# app/data_loader.py
from __future__ import annotations

import re
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional

import pandas as pd
from rapidfuzz import fuzz

# -----------------------
# Flexible, fuzzy loader
# -----------------------

MONTH_MAP = {
    "jan":1,"january":1,"feb":2,"february":2,"mar":3,"march":3,"apr":4,"april":4,
    "may":5,"jun":6,"june":6,"jul":7,"july":7,"aug":8,"august":8,"sep":9,"sept":9,
    "september":9,"oct":10,"october":10,"nov":11,"november":11,"dec":12,"december":12
}

# Canonical target names we expose to the rest of the app
CANONICAL_COLS = [
    "Date","Period","Net Sales","Cash","Mada","Card","Online",
    "Jahez","Koinz","Hunger","Alinma","VAT Amount","VAT Rate","Gross Sales","Notes"
]

# Fuzzy name tokens → canonical column
FUZZY_MAP = {
    "date": "Date",
    "day": "Date",
    "period": "Period",
    "shift": "Period",
    "session": "Period",
    "net sales": "Net Sales",
    "net amount": "Net Sales",
    "net revenue": "Net Sales",
    "gross sales": "Gross Sales",
    "gross": "Gross Sales",
    "cash": "Cash",
    "mada": "Mada",
    "card": "Card",
    "visa": "Card",
    "pos": "Card",
    "online": "Online",
    "jahez": "Jahez",
    "koinz": "Koinz",
    "hunger": "Hunger",
    "alinma": "Alinma",
    "vat amount": "VAT Amount",
    "vat": "VAT Amount",
    "0.15": "VAT Rate",
    "vat rate": "VAT Rate",
    "notes": "Notes",
    "remark": "Notes",
    "remarks": "Notes",
}

TOTAL_PREFIXES = ["total", "grand total", "overall"]

def _infer_month_year_from_name(name: str) -> Tuple[Optional[int], Optional[int]]:
    s = name.lower()
    m1 = re.search(r"(20\d{2})[-_ ]?(0[1-9]|1[0-2])", s)
    if m1:
        return int(m1.group(2)), int(m1.group(1))
    m2 = re.search(r"(" + "|".join(MONTH_MAP.keys()) + r")[a-z]*[ -_]*(20\d{2})", s)
    if m2:
        return MONTH_MAP[m2.group(1)], int(m2.group(2))
    return None, None

def _best_match(name: str, candidates: Dict[str, str], min_score=70) -> Optional[str]:
    name_l = name.strip().lower()
    best_key, best_sc = None, -1
    for k in candidates.keys():
        sc = fuzz.token_set_ratio(name_l, k)
        if sc > best_sc:
            best_sc, best_key = sc, k
    return candidates.get(best_key) if best_sc >= min_score else None

def _coerce_date_flexible(series: pd.Series, month_hint: Optional[int], year_hint: Optional[int]) -> pd.Series:
    s = pd.Series(series)

    # Try generic parse
    d = pd.to_datetime(s, errors="coerce", dayfirst=True)
    if d.notna().mean() >= 0.5:
        return d

    # Try common strict formats
    for fmt in ("%Y-%m-%d","%d/%m/%Y","%d-%m-%Y","%d.%m.%Y","%m/%d/%Y"):
        d2 = pd.to_datetime(s, format=fmt, errors="coerce")
        if d2.notna().mean() >= 0.5:
            return d2

    # Excel serials (safe window)
    ser = pd.to_numeric(s, errors="coerce")
    mask = ser.between(20000, 60000)
    if mask.any():
        base = pd.Timestamp("1899-12-30")
        d3 = pd.Series(pd.NaT, index=s.index, dtype="datetime64[ns]")
        d3.loc[mask] = base + pd.to_timedelta(ser[mask].astype(float), unit="D")
        if d3.notna().mean() >= 0.5:
            return d3

    # Ordinals like "1st", "2nd"
    ord_day = s.astype("string").str.extract(r'^\s*(\d{1,2})(?:st|nd|rd|th)?\s*$', expand=False)
    day_num = pd.to_numeric(ord_day, errors="coerce")
    if day_num.between(1,31).fillna(False).mean() >= 0.5 and month_hint is not None:
        y = year_hint or datetime.now().year
        return pd.to_datetime({"year": y, "month": month_hint, "day": day_num.clip(1,31).astype("Int64")},
                              errors="coerce")

    # Pure integers 1..31
    day_num2 = pd.to_numeric(s, errors="coerce")
    if day_num2.between(1,31).fillna(False).mean() >= 0.5 and month_hint is not None:
        y = year_hint or datetime.now().year
        return pd.to_datetime({"year": y, "month": month_hint, "day": day_num2.clip(1,31).astype("Int64")},
                              errors="coerce")

    # Month + day tokens e.g., "5 Jan" / "Jan 5"
    md = s.astype("string").str.extract(r'(?i)\b(' + "|".join(MONTH_MAP.keys()) + r')\b\D{0,3}(\d{1,2})')
    if not md.empty and md.notna().all(axis=None):
        mo = md[0].str.lower().map(MONTH_MAP)
        dy = pd.to_numeric(md[1], errors="coerce")
        if (dy.between(1,31).fillna(False).mean() >= 0.5) and mo.notna().mean() >= 0.5:
            y = year_hint or datetime.now().year
            return pd.to_datetime({"year": y, "month": mo, "day": dy}, errors="coerce")

    # Give up
    return pd.Series(pd.NaT, index=s.index, dtype="datetime64[ns]")

def _detect_data_sheet(xls: pd.ExcelFile) -> str:
    # Prefer 'Data' if present
    for nm in xls.sheet_names:
        if nm.strip().lower() == "data":
            return nm
    # Else, pick the first sheet that contains a 'date'-ish column in row 1
    for nm in xls.sheet_names:
        head = pd.read_excel(xls, sheet_name=nm, nrows=1)
        cols = [str(c).strip().lower() for c in head.columns]
        if any("date" in c or "day" in c or "التاريخ" in c for c in cols):
            return nm
    # Fallback to first sheet
    return xls.sheet_names[0]

def _normalize_columns(cols: List[str]) -> Dict[str, str]:
    """
    Map original columns to canonical names (when possible).
    Returns: {original_name: canonical_or_original}
    """
    mapping = {}
    for c in cols:
        c_str = str(c).strip()
        if not c_str:
            continue
        # Direct keep for obvious canonical
        if c_str in CANONICAL_COLS:
            mapping[c_str] = c_str
            continue
        # Try fuzzy
        canon = _best_match(c_str, FUZZY_MAP, min_score=72)
        mapping[c_str] = canon or c_str
    # Ensure 'Date' exists if there is any likely candidate
    if "Date" not in mapping.values():
        # pick the best column by fuzzy against "date"
        best_cand, best_sc = None, -1
        for c in cols:
            sc = fuzz.token_set_ratio(str(c).lower(), "date")
            if sc > best_sc:
                best_sc, best_cand = sc, c
        if best_sc >= 72 and best_cand is not None:
            mapping[str(best_cand)] = "Date"
    return mapping

def _extract_totals_columns(df_raw: pd.DataFrame) -> List[str]:
    # Any column whose header starts with "total" (case-insensitive) is a totals column
    totals = []
    for c in df_raw.columns:
        name = str(c).strip().lower()
        if any(name.startswith(pfx) for pfx in TOTAL_PREFIXES):
            totals.append(str(c))
    return totals

class SalesData:
    """
    Robust monthly loader:
    - Auto-detects the sheet (prefers 'Data').
    - Fuzzy-maps headers to canonical names; missing channels are allowed (filled with 0).
    - Date parsing handles true dates, excel serials, ordinals, pure 1..31 with filename hints, etc.
    - Accepts any number of daily rows (≤31 is fine).
    - 'Net Sales' can be missing: we derive it by summing channels minus VAT if possible.
    - Totals columns are detected automatically by prefix (e.g., 'Total Net', 'Total Cash', ...).
    """
    def __init__(self, path: str | Path, sheet: Optional[str] = None):
        self.path = Path(path)
        if not self.path.exists():
            raise FileNotFoundError(self.path)
        self.sheet = sheet  # may be None; will auto-detect
        self.df: pd.DataFrame = pd.DataFrame()
        self.totals: Dict[str, float] = {}
        self.m_hint: Optional[int] = None
        self.y_hint: Optional[int] = None
        self._load()

    # -------- public API --------
    def daily_net(self) -> pd.DataFrame:
        """Return DataFrame with columns: date, net"""
        if "Net Sales" not in self.df.columns:
            # derive if needed
            self._derive_net_sales()
        out = (self.df[self.df["Date"].notna()]
               .assign(date=self.df["Date"].dt.date)
               .groupby("date", as_index=False)["Net Sales"].sum()
               .rename(columns={"Net Sales": "net"}))
        return out

    def channels_by_day(self) -> pd.DataFrame:
        chans = [c for c in ["Cash","Mada","Card","Online","Jahez","Koinz","Hunger","Alinma"] if c in self.df.columns]
        if not chans:
            return pd.DataFrame(columns=["date"])
        out = (self.df[self.df["Date"].notna()][["Date"]+chans]
               .assign(date=self.df["Date"].dt.date)
               .groupby("date", as_index=False).sum())
        return out

    def totals_dict(self) -> Dict[str, float]:
        return dict(self.totals)

    def raw(self) -> pd.DataFrame:
        return self.df.copy()

    # -------- internals --------
    def _load(self):
        xls = pd.ExcelFile(self.path)
        sheet_to_use = self.sheet or _detect_data_sheet(xls)
        raw = pd.read_excel(xls, sheet_name=sheet_to_use)
        raw = raw.rename(columns=lambda c: str(c).strip())

        # Hints from filename
        self.m_hint, self.y_hint = _infer_month_year_from_name(self.path.name)

        # Column normalization
        col_map = _normalize_columns(list(raw.columns))
        df = raw.rename(columns=col_map).copy()

        # Build Date column (if missing or partially valid, try to coerce)
        if "Date" in df.columns:
            df["Date"] = _coerce_date_flexible(df["Date"], self.m_hint, self.y_hint)
        else:
            # Try to create a Date from any column that looks like day numbers
            best_alt, best_rate = None, -1.0
            for c in df.columns:
                dn = pd.to_numeric(df[c], errors="coerce")
                rate = dn.between(1,31).fillna(False).mean()
                if rate > best_rate and rate >= 0.4:
                    best_alt, best_rate = c, rate
            if best_alt is not None and self.m_hint:
                dn = pd.to_numeric(df[best_alt], errors="coerce").clip(1,31).astype("Int64")
                y = self.y_hint or datetime.now().year
                df["Date"] = pd.to_datetime({"year": y, "month": self.m_hint, "day": dn}, errors="coerce")
            else:
                df["Date"] = pd.NaT

        # Coerce numerics for everything except non-numerical canonical columns
        for c in df.columns:
            if c in ("Date","Period","Notes"):
                continue
            df[c] = pd.to_numeric(df[c], errors="coerce")

        # If "Net Sales" is missing, we'll derive later when daily_net() is called
        # Fill missing optional channels with 0 to keep math consistent
        for c in ["Cash","Mada","Card","Online","Jahez","Koinz","Hunger","Alinma","VAT Amount","VAT Rate","Gross Sales"]:
            if c not in df.columns:
                df[c] = 0.0

        # Pick totals columns by prefix
        totals_cols = _extract_totals_columns(raw)
        self.totals = {}
        if totals_cols:
            trow = raw[totals_cols].apply(lambda s: pd.to_numeric(s, errors="coerce"))
            trow = trow.dropna(how="all")
            if not trow.empty:
                first = trow.iloc[0]
                self.totals = {str(k): float(v) for k, v in first.to_dict().items() if pd.notna(v)}

        self.df = df.reset_index(drop=True)

    def _derive_net_sales(self):
        # If we don't have Net Sales, try to compute it:
        #   sum(channels) - VAT Amount (if present), else just sum(channels).
        if "Net Sales" in self.df.columns:
            return
        chans = [c for c in ["Cash","Mada","Card","Online","Jahez","Koinz","Hunger","Alinma"] if c in self.df.columns]
        if chans:
            provisional = self.df[chans].sum(axis=1)
            if "VAT Amount" in self.df.columns and self.df["VAT Amount"].notna().any():
                provisional = provisional - self.df["VAT Amount"].fillna(0)
            self.df["Net Sales"] = provisional
        else:
            # Nothing to derive from; create zero column to keep interface stable
            self.df["Net Sales"] = 0.0
def metrics(self) -> dict:
    """
    Lightweight computed stats from whatever is available.
    Returns keys only when they can be computed (no hard failures).
    """
    out = {}
    df = self.df.copy()

    # Ensure Net Sales exists
    if "Net Sales" not in df.columns:
        self._derive_net_sales()
    if "Net Sales" in df.columns:
        daily = (df[df["Date"].notna()]
                 .assign(date=df["Date"].dt.date)
                 .groupby("date", as_index=False)["Net Sales"].sum()
                 .rename(columns={"Net Sales":"net"}))
        if not daily.empty:
            out["days_count"]      = int(len(daily))
            out["net_total"]       = float(daily["net"].sum())
            out["net_avg_per_day"] = float(daily["net"].mean())
            best = daily.loc[daily["net"].idxmax()]
            worst= daily.loc[daily["net"].idxmin()]
            out["best_day"]  = {"date": str(best["date"]),  "net": float(best["net"])}
            out["worst_day"] = {"date": str(worst["date"]), "net": float(worst["net"])}

    # Channel shares if available
    chan_cols = [c for c in ["Cash","Mada","Card","Online","Jahez","Koinz","Hunger","Alinma"] if c in df.columns]
    if chan_cols:
        chan_totals = df[chan_cols].sum(numeric_only=True)
        chan_sum = float(chan_totals.sum())
        out["channels_total"] = float(chan_sum)
        out["channel_shares"] = {c: (float(v) / chan_sum if chan_sum else 0.0)
                                 for c, v in chan_totals.items()}

    # VAT / Gross if present
    if "VAT Amount" in df.columns:
        out["vat_total"] = float(df["VAT Amount"].sum(skipna=True))
    if "Gross Sales" in df.columns:
        out["gross_total"] = float(df["Gross Sales"].sum(skipna=True))

    # Declared totals block (if detected)
    if getattr(self, "totals", None):
        out["declared_totals"] = dict(self.totals)

    return out
