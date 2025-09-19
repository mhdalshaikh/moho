import re, uuid
from pathlib import Path
import pandas as pd
from pypdf import PdfReader
import pdfplumber
import docx
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings

# If you use absolute imports elsewhere, keep this:
from app.config import (
    DOCS_DIR, CACHE_DIR, CHUNK_SIZE, CHUNK_OVERLAP, EMBEDDING_MODEL
)

SUPPORTED_EXT = {".txt", ".pdf", ".docx", ".csv"}

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

def read_pdf_tables(p: Path):
    tabs = []
    try:
        with pdfplumber.open(str(p)) as pdf:
            for page in pdf.pages:
                for t in (page.extract_tables() or []):
                    df = pd.DataFrame(t)
                    if df.shape[0] > 1:
                        df.columns = df.iloc[0]
                        df = df[1:]
                    tabs.append(df)
    except Exception:
        pass
    return tabs

def read_docx(p: Path) -> str:
    d = docx.Document(str(p))
    return "\n".join(par.text for par in d.paragraphs)

# ---------- Chunking ----------

def chunk_text(text: str, size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    text = re.sub(r"\s+", " ", text).strip()
    res = []
    i = 0
    while i < len(text):
        j = min(i + size, len(text))
        res.append(text[i:j])
        if j >= len(text):
            break
        i = max(0, j - overlap)
    return [c for c in res if c]

# ---------- Chroma helpers ----------

def get_client():
    # keep allow_reset=True consistent with retriever
    return chromadb.PersistentClient(
        path=str(CACHE_DIR),
        settings=Settings(allow_reset=True)
    )

def build_or_get_collection(client, name="local_rag"):
    try:
        return client.get_collection(name)
    except Exception:
        return client.create_collection(name)

# ---------- Ingest ----------

def ingest_docs():
    DOCS_DIR.mkdir(exist_ok=True)
    client = get_client()
    coll = build_or_get_collection(client)
    embedder = SentenceTransformer(EMBEDDING_MODEL)

    existing = set(coll.get()["ids"]) if coll.count() else set()

    for p in sorted(DOCS_DIR.rglob("*")):
        if p.suffix.lower() not in SUPPORTED_EXT:
            continue

        text = ""
        tables = []

        if p.suffix.lower() == ".txt":
            text = read_txt(p)

        elif p.suffix.lower() == ".pdf":
            text = read_pdf_text(p)
            tables = read_pdf_tables(p)

        elif p.suffix.lower() == ".docx":
            text = read_docx(p)

        elif p.suffix.lower() == ".csv":
            # add encoding if your CSVs are not UTF-8: encoding="utf-8-sig", errors="ignore"
            df = pd.read_csv(p)
            tables = [df]
            # ✅ fixed closing braces/parens
            text = f"CSV {p.name} columns: {', '.join(map(str, df.columns))}"

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
                # Chroma expects lists, not ndarrays
                embs = embedder.encode(docs, normalize_embeddings=True).tolist()
                coll.add(ids=ids, documents=docs, embeddings=embs, metadatas=metas)

        # --- index table previews as separate chunks ---
        for t_i, df in enumerate(tables):
            try:
                df = df.dropna(how="all").dropna(axis=1, how="all")
            except Exception:
                pass
            preview = df.head(10).to_csv(index=False)
            cid = f"{p.name}:table:{t_i}:{uuid.uuid5(uuid.NAMESPACE_DNS, preview)}"
            if cid in existing:
                continue
            embs = embedder.encode([preview], normalize_embeddings=True).tolist()
            coll.add(
                ids=[cid],
                documents=[f"TABLE {p.name}\n{preview}"],
                embeddings=embs,
                metadatas=[{"source": p.name, "chunk": t_i, "type": "table"}],
            )

    print("Ingestion complete.")

if __name__ == "__main__":
    ingest_docs()
