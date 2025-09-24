import os, sys, io, zipfile, shutil
from pathlib import Path

# Ensure we can import the package as "app.*"
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import streamlit as st

from app.config import APP_TITLE
from app.ingest import ingest_docs
from app.chat import answer_query

# ------------------------------------------------------------------------------
# Paths
# ------------------------------------------------------------------------------
PKG_ROOT = Path(ROOT)               # .../app
PROJECT_ROOT = PKG_ROOT.parent      # repo root
DOCS_DIR = PROJECT_ROOT / "docs"    # shared docs/ folder at repo root
DOCS_DIR.mkdir(parents=True, exist_ok=True)


# ------------------------------------------------------------------------------
# Upload helpers
# ------------------------------------------------------------------------------
ALLOWED_TYPES = ["pdf", "docx", "txt", "csv", "xlsx", "xls", "zip"]

def _safe_name(name: str) -> str:
    # simple filename hardening (strip path separators)
    return os.path.basename(name).replace("\\", "_").replace("/", "_")

def _save_single_file(file) -> Path:
    """
    Save a single uploaded file-like object to DOCS_DIR.
    Returns the saved Path.
    """
    fname = _safe_name(file.name)
    dest = DOCS_DIR / fname
    # if same name exists, version it
    base = dest.stem
    ext  = dest.suffix
    i = 1
    while dest.exists():
        dest = DOCS_DIR / f"{base} ({i}){ext}"
        i += 1
    # write bytes
    with open(dest, "wb") as f:
        f.write(file.getbuffer())
    return dest

def _extract_zip(zip_path: Path) -> list[Path]:
    """
    Extract zip into a subfolder under docs/.
    Returns list of extracted file paths (only allowed types).
    """
    extracted: list[Path] = []
    target_dir = DOCS_DIR / (zip_path.stem + "_unzipped")
    target_dir.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(zip_path, "r") as zf:
        for info in zf.infolist():
            # skip folders
            if info.is_dir():
                continue
            # filter by extension (case-insensitive)
            ext = Path(info.filename).suffix.lower().lstrip(".")
            if ext not in ALLOWED_TYPES or ext == "zip":
                # skip disallowed or nested zips
                continue
            # safe target path
            out_name = _safe_name(Path(info.filename).name)
            out_path = target_dir / out_name
            # ensure parent exists
            out_path.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(info, "r") as src, open(out_path, "wb") as dst:
                shutil.copyfileobj(src, dst)
            extracted.append(out_path)
    return extracted

def handle_uploads(uploaded_files: list) -> tuple[list[Path], list[Path]]:
    """
    Save all uploaded files to docs/.
    Also extracts zip archives and returns (saved_files, extracted_files).
    """
    saved: list[Path] = []
    extracted: list[Path] = []
    for file in uploaded_files or []:
        try:
            p = _save_single_file(file)
            saved.append(p)
            if p.suffix.lower() == ".zip":
                extracted += _extract_zip(p)
        except Exception as e:
            st.error(f"تعذّر حفظ الملف: {getattr(file, 'name', 'unknown')} — {e}")
    return saved, extracted


# ------------------------------------------------------------------------------
# UI
# ------------------------------------------------------------------------------
st.set_page_config(page_title=APP_TITLE, page_icon="🦙", layout="wide")
st.title(APP_TITLE)

with st.sidebar:
    st.header("⚙️ الإعدادات")
    st.caption("اسحب وأسقط ملفاتك هنا، ثم أعد الفهرسة ليقرأها البوت.")

    # --- Drag & Drop Uploader ---
    uploads = st.file_uploader(
        "اسحب وأسقط الملفات أو اختر من جهازك",
        type=ALLOWED_TYPES,
        accept_multiple_files=True,
        help="يدعم: PDF, DOCX, TXT, CSV, XLSX/XLS, و ZIP (يتم فكّه تلقائيًا)."
    )

    auto_index = st.checkbox("أعد الفهرسة تلقائيًا بعد الرفع", value=True)

    if uploads:
        with st.spinner("جارٍ حفظ الملفات..."):
            saved, extracted = handle_uploads(uploads)
        if saved:
            st.success(f"تم حفظ {len(saved)} ملف/ملفات في مجلد docs/")
            with st.expander("الملفات المحفوظة"):
                for p in saved:
                    st.write(f"• {p.name}")
        if extracted:
            st.info(f"تم فك ضغط {len(extracted)} ملف من الأرشيفات")
            with st.expander("الملفات المُستخرجة من ZIP"):
                for p in extracted:
                    st.write(f"• {p.relative_to(DOCS_DIR)}")

        if auto_index:
            with st.spinner("جارٍ إعادة الفهرسة..."):
                ingest_docs()
            st.success("تمت إعادة الفهرسة.")
        else:
            if st.button("🔁 إعادة الفهرسة الآن", use_container_width=True):
                with st.spinner("جارٍ إعادة الفهرسة..."):
                    ingest_docs()
                st.success("تمت إعادة الفهرسة.")

    # Manual re-index button stays available
    if st.button("🔁 إعادة فهرسة المستندات", use_container_width=True):
        with st.spinner("جارٍ إعادة الفهرسة..."):
            ingest_docs()
        st.success("تمت إعادة الفهرسة.")
    st.caption("ضع ملفات PDF/DOCX/TXT/CSV/XLSX داخل مجلد docs/ ثم أعد الفهرسة.")

# ------------------------------------------------------------------------------
# Chat history
# ------------------------------------------------------------------------------
if "history" not in st.session_state:
    st.session_state.history = []

# Render history
for role, content in st.session_state.history:
    with st.chat_message(role):
        if isinstance(content, dict) and content.get("chart_html"):
            st.markdown(content.get("text", ""))
            html = content["chart_html"]
            if not isinstance(html, str) and hasattr(html, "getvalue"):
                try:
                    html = html.getvalue().decode("utf-8", "ignore")
                except Exception:
                    html = str(html)
            st.components.v1.html(html, height=520, scrolling=True)
        else:
            st.markdown(content if isinstance(content, str) else content.get("text", ""))

# ------------------------------------------------------------------------------
# Chat input
# ------------------------------------------------------------------------------
q = st.chat_input("اسأل عن مستنداتك… ")
if q:
    with st.chat_message("user"):
        st.markdown(q)
    with st.chat_message("assistant"):
        with st.spinner("يفكّر…"):
            out = answer_query(q)
        st.session_state.history.append(("user", q))
        st.session_state.history.append(("assistant", out))
        if out.get("chart_html"):
            st.markdown(out["text"])
            html = out["chart_html"]
            if not isinstance(html, str) and hasattr(html, "getvalue"):
                try:
                    html = html.getvalue().decode("utf-8", "ignore")
                except Exception:
                    html = str(html)
            st.components.v1.html(html, height=520, scrolling=True)
        else:
            st.markdown(out["text"])
        st.caption(f"Confidence ≈ {out.get('confidence', 0.0):.2f}")
