# 1) Install Ollama
#    Download & install from https://ollama.com/
ollama pull llama3.1:8b            # one-time model download
# (optional) sanity check:
curl http://localhost:11434/api/tags

# 2) Create & activate a virtual env
python -m venv .venv
# Windows PowerShell:
# .\.venv\Scripts\Activate.ps1

# 3) Install Python deps
pip install -r requirements.txt

# 4) Run the app (from repo root)
python -m streamlit run app/ui_streamlit.py

EXCEL FILES FORMAT ARE TO BE AS THE ONE ATTACHED IN THIS REPO in /DOCS