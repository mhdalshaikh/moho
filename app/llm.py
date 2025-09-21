from __future__ import annotations

import json
from typing import List, Dict

from app.config import LLM_MODEL

def chat_llm(messages: List[Dict[str, str]], model: str = LLM_MODEL) -> str:
    """
    Minimal wrapper around Ollama's /chat. Returns assistant text.
    Gracefully degrades if Ollama or the model are missing.
    """
    try:
        import ollama  # type: ignore
    except Exception:
        return ("[LLM unavailable] Ollama is not installed or not importable. "
                "Install it from https://ollama.com and run `ollama serve`.")

    try:
        r = ollama.chat(model=model, messages=messages)  # type: ignore
        # Newer ollama python returns dict with key 'message' {'role','content'}
        msg = r.get("message") or {}
        return msg.get("content", "").strip() or "(no content)"
    except Exception as e:
        # Most common: model not pulled yet
        return (f"[LLM error] {e}\n"
                f"If this says the model is not found, run:  `ollama pull {model}` "
                "then restart the app.")
