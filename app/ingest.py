from __future__ import annotations

import re, uuid
from pathlib import Path
from typing import Iterable, Tuple

import pandas as pd
from pypdf import PdfReader
import pdfplumber
import docx
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings

from app.config import (
    DOCS_DIR, CACHE_DIR, CHUNK_SIZE, CHUNK_OVERLAP, EMBEDDING_MODEL
)

SUPPORTED_EXT = {".txt", ".pdf", ".docx", ".csv", ".xlsx", ".xls"}
TABLE_STORE = CACHE_DIR / "tables"

# ---------- Parsers ----------
def read_txt(p: Path) -> str:
    return p.read_text(encoding="utf-8", errors="ignore")

def read_pdf_text(p: Path) -> str:
    out = []
    with open(p, "rb") as f:
        r = PdfReader(f)
        for pg in r.pages:
            try:
                out.append(pg.extract_text() or "")
            except Exception:
                out.append("")
    return "\n".join(out)

def read_pdf_tables(p: Path) -> list[Tuple[pd.DataFrame, str | None]]:
    tabs: list[Tuple[pd.DataFrame, str | None]] = []
    try:
        with pdfplumber.open(str(p)) as pdf:
            for page_i, page in enumerate(pdf.pages):
                for t in (page.extract_tables() or []):
                    df = pd.DataFrame(t)
                    if df.shape[0] > 1:
                        df.columns = df.iloc[0]
                        df = df[1:]
                    tabs.append((df, f"page_{page_i+1}"))
    except Exception:
        pass
    return tabs

def read_excel_sheets(p: Path) -> Iterable[Tuple[str, pd.DataFrame]]:
    try:
        xls = pd.ExcelFile(p)
        for name in xls.sheet_names:
            try:
                df = pd.read_excel(xls, sheet_name=name)
                yield name, df
            except Exception:
                continue
    except Exception:
        return []

def read_docx(p: Path) -> str:
    d = docx.Document(str(p))
    return "\n".join(par.text for par in d.paragraphs)

# ---------- Chunking ----------
def chunk_text(text: str, size=CHUNK_SIZE, overlap=CHUNK_OVERLAP) -> list[str]:
    text = re.sub(r"\s+", " ", text).strip()
    res, i = [], 0
    while i < len(text):
        j = min(i + size, len(text))
        res.append(text[i:j])
        if j >= len(text):
            break
        i = max(0, j - overlap)
    return [c for c in res if c]

# ---------- Chroma helpers ----------
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

# ---------- Normalization helpers ----------
def _flatten_cols(df: pd.DataFrame) -> pd.DataFrame:
    if isinstance(df.columns, pd.MultiIndex):
        df = df.copy()
        df.columns = [
            " ".join([str(x) for x in tup if str(x) != "nan"]).strip()
            for tup in df.columns
        ]
    else:
        df = df.rename(columns=lambda c: str(c))
    return df

def _normalize_for_parquet(df: pd.DataFrame) -> pd.DataFrame:
    df = _flatten_cols(df.copy())
    # Convert bytes→str then cast object cols to pandas StringDtype, keep numerics as-is
    for c in df.columns:
        s = df[c]
        if pd.api.types.is_object_dtype(s):
            s = s.apply(lambda v: v.decode("utf-8", "ignore")
                        if isinstance(v, (bytes, bytearray)) else v)
            # Casting to "string" avoids Arrow mixed-type issues
            try:
                s = s.astype("string")
            except Exception:
                s = s.astype(str)
        df[c] = s
    # Prefer nullable dtypes where possible
    try:
        df = df.convert_dtypes()
    except Exception:
        pass
    return df

# ---------- Ingest ----------
def ingest_docs():
    DOCS_DIR.mkdir(exist_ok=True)
    coll = _collection()
    embedder = SentenceTransformer(EMBEDDING_MODEL)

    existing = set(coll.get()["ids"]) if coll.count() else set()

    for p in sorted(DOCS_DIR.rglob("*")):
        if not p.is_file() or p.suffix.lower() not in SUPPORTED_EXT:
            continue

        text = ""
        tables: list[Tuple[pd.DataFrame, str | None]] = []

        if p.suffix.lower() == ".txt":
            text = read_txt(p)

        elif p.suffix.lower() == ".pdf":
            text = read_pdf_text(p)
            tables = read_pdf_tables(p)

        elif p.suffix.lower() == ".docx":
            text = read_docx(p)

        elif p.suffix.lower() == ".csv":
            df = pd.read_csv(p)
            tables = [(df, None)]
            text = f"CSV {p.name} columns: {', '.join(map(str, df.columns))} (rows={len(df)})"

        elif p.suffix.lower() in {".xlsx", ".xls"}:
            sheets = list(read_excel_sheets(p))
            tables = [(df, sheet) for sheet, df in sheets]
            parts = []
            for sheet, df in sheets:
                cols = ", ".join(map(str, df.columns))
                parts.append(f"EXCEL {p.name} [sheet={sheet}] columns: {cols} (rows={len(df)})")
            text = "\n".join(parts)

        # --- index text chunks ---
        if text:
            chunks = chunk_text(text)
            ids, docs, metas = [], [], []
            for i, ch in enumerate(chunks):
                cid = f"{p.name}:{i}:{uuid.uuid5(uuid.NAMESPACE_DNS, ch)}"
                if cid in existing:
                    continue
                ids.append(cid)
                docs.append(ch)
                metas.append({"source": p.name, "chunk": i, "type": "text"})
            if docs:
                embs = embedder.encode(docs, normalize_embeddings=True).tolist()
                coll.add(ids=ids, documents=docs, embeddings=embs, metadatas=metas)

        # --- index table previews & persist full sheets ---
        for t_i, item in enumerate(tables):
            df, sheet = item
            # Clean the dataframe for Parquet
            df_norm = _normalize_for_parquet(df)

            # Persist full sheet
            safe_sheet = (sheet or "sheet").replace("/", "_")
            table_fname = f"{p.stem}__{safe_sheet}__{t_i}.parquet"
            table_path = TABLE_STORE / table_fname
            df_norm.to_parquet(table_path, engine="pyarrow", index=False)

            # Build small preview text
            preview_csv = df_norm.head(10).to_csv(index=False)
            cols = ", ".join(map(str, df_norm.columns))
            rows = int(len(df_norm))
            doc_text = (
                f"TABLE {p.name}"
                f"{f' [sheet={sheet}]' if sheet else ''}; "
                f"rows={rows}; cols={cols}\n"
                f"PREVIEW (first 10 rows):\n{preview_csv}"
            )

            cid = f"{p.name}{f':{sheet}' if sheet else ''}:table:{t_i}:{uuid.uuid5(uuid.NAMESPACE_DNS, preview_csv)}"
            if cid in existing:
                continue

            emb = embedder.encode([doc_text], normalize_embeddings=True).tolist()
            coll.add(
                ids=[cid],
                documents=[doc_text],
                embeddings=emb,
                metadatas=[{
                    "source": p.name,
                    "type": "table",
                    "chunk": int(t_i),
                    **({"sheet": str(sheet)} if sheet else {}),
                    "table_path": str(table_path),  # pointer to full data
                    "table_rows": rows,
                    "table_cols": ", ".join(map(str, df_norm.columns)),  # primitives only
                }],
            )

    print("Ingestion complete.")

if __name__ == "__main__":
    ingest_docs()
