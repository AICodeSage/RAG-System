"""Embedding management helpers."""


class EmbeddingStore:
    """Wraps embedding lookup functionality."""

    def __init__(self):
        self.vectors = {}

    def add(self, key: str, vector: list[float]):
        self.vectors[key] = vector


