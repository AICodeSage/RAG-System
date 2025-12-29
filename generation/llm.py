"""OpenAI-backed LLM wrapper used by the RAG system (OpenAI Python >= 1.x)."""

import os
from pathlib import Path
from typing import Dict

try:
    from openai import OpenAI
except ImportError:  # pragma: no cover
    OpenAI = None

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover
    load_dotenv = None


_ENV_FILE = Path(".env")
if load_dotenv and _ENV_FILE.exists():
    load_dotenv(_ENV_FILE)


def is_llm_available() -> bool:
    """Return True when the OpenAI client and API key are configured."""
    return OpenAI is not None and bool(os.environ.get("OPENAI_API_KEY"))


def query_llm(context: str, query: str) -> str:
    """Query OpenAI's chat completion API for a concise response."""
    if OpenAI is None:
        raise RuntimeError("install the openai>=1.0.0 package to use the LLM backend")
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("set OPENAI_API_KEY to call the LLM backend")

    client = OpenAI(api_key=api_key)
    model = os.environ.get("OPENAI_MODEL", "gpt-3.5-turbo")
    temperature = float(os.environ.get("OPENAI_TEMPERATURE", "0.0"))
    max_tokens = int(os.environ.get("OPENAI_MAX_TOKENS", "300"))

    prompt = f"Context:\n{context}\n\nQuestion: {query}\nAnswer with a brief, factual response."
    payload: Dict[str, object] = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": "You answer questions concisely using only the provided context.",
            },
            {"role": "user", "content": prompt},
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }

    response = client.chat.completions.create(**payload)
    return response.choices[0].message.content.strip()
