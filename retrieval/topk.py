"""Thin wrapper for retrieving the top-k entries."""

from core.retriever import Retriever


class TopKRetriever:
    def __init__(self, retriever: Retriever):
        self._retriever = retriever

    def retrieve(self, query: str, k: int = 5):
        """Return the highest-scoring `k` chunks for the query."""
        return self._retriever.retrieve(query, top_k=k)
