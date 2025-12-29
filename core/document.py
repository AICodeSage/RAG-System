"""Document abstractions used by the RAG pipeline."""

from dataclasses import dataclass, field
from typing import Dict


@dataclass
class Document:
    """Represents a source document with optional metadata."""

    id: str
    text: str
    metadata: Dict[str, str] = field(default_factory=dict)

    def with_metadata(self, **values: str) -> "Document":
        """Return a shallow copy enriched with the provided metadata."""
        combined = dict(self.metadata)
        combined.update(values)
        return Document(id=self.id, text=self.text, metadata=combined)

