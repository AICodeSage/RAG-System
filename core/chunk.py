"""Chunk representation used within the RAG pipeline."""

from dataclasses import dataclass, field
from typing import Dict


@dataclass
class Chunk:
    id: str
    text: str
    metadata: Dict[str, str] = field(default_factory=dict)

    def summary(self, length: int = 120) -> str:
        """Return a short representation of the chunk for logging or prompts."""
        snippet = self.text.strip()
        return snippet[:length] + ("..." if len(snippet) > length else "")

