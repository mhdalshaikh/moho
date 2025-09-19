import ollama
from typing import List, Dict
from .config import LLM_MODEL

SYSTEM = (
    "You are a careful RAG assistant. Use ONLY retrieved sources. "
    "Cite as [#]. If unsure, say 'I don't know'."
)

def chat_llm(messages: List[Dict[str,str]], model:str=LLM_MODEL)->str:
    msgs=[{"role":"system","content":SYSTEM}]+messages
    r=ollama.chat(model=model, messages=msgs)
    return r["message"]["content"]
