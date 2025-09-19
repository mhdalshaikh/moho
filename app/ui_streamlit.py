import os, sys
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
import streamlit as st
from app.config import APP_TITLE
from app.ingest import ingest_docs
from app.chat import answer_query

st.set_page_config(page_title=APP_TITLE, page_icon="🦙", layout="wide")
st.title(APP_TITLE)

with st.sidebar:
    st.header("⚙️ Controls")
    if st.button("Re-index documents", use_container_width=True):
        with st.spinner("Indexing..."):
            ingest_docs(); st.success("Done.")
    st.caption("Put PDFs/DOCX/TXT/CSV into the `docs/` folder.")

if "history" not in st.session_state: st.session_state.history=[]
for role, content in st.session_state.history:
    with st.chat_message(role):
        if isinstance(content, dict) and content.get("chart_html"):
            st.markdown(content.get("text",""))
            st.components.v1.html(content["chart_html"].getvalue().decode("utf-8"), height=520)
        else:
            st.markdown(content if isinstance(content,str) else content.get("text",""))

q=st.chat_input("Ask about your documents… (e.g. 'average amount 2024', 'bar chart of sales')")
if q:
    with st.chat_message("user"): st.markdown(q)
    with st.chat_message("assistant"):
        with st.spinner("Thinking with LLaMA…"):
            out=answer_query(q)
            st.session_state.history.append(("user", q))
            st.session_state.history.append(("assistant", out))
            if out.get("chart_html"):
                st.markdown(out["text"])
                st.components.v1.html(out["chart_html"].getvalue().decode("utf-8"), height=520)
            else:
                st.markdown(out["text"])
            st.caption(f"Confidence ≈ {out.get('confidence',0.0):.2f}")
