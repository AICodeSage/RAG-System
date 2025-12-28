"""Chunk representation used by the retriever."""


class Chunk:
    """A piece of text derived from a document."""

    def __init__(self, id: str, text: str, metadata: dict):
        self.id = id
        self.text = text
        self.metadata = metadata or {}


