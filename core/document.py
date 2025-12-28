"""Core document abstractions for RAG system."""


class Document:
    """Represents a source document in the RAG pipeline."""

    def __init__(self, id: str, text: str):
        self.id = id
        self.text = text
        self.metadata = {}


