"""Retriever that uses the vector index for similarity search."""

import logging
from typing import List, Tuple

from core.index import IndexEntry, VectorIndex

LOGGER = logging.getLogger(__name__)


class Retriever:
    """Retrieves relevant chunks from a vector index."""

    def __init__(self, index: VectorIndex):
        self.index = index

    def retrieve(self, query: str, top_k: int = 5) -> List[Tuple[IndexEntry, float]]:
        """
        Retrieve the top_k most relevant chunks for a query.

        Returns:
            List of (IndexEntry, similarity_score) tuples, sorted by relevance.
        """
        if not self.index.entries:
            LOGGER.warning("Index is empty")
            return []

        results = self.index.search(query, top_k=top_k)

        for i, (entry, score) in enumerate(results):
            LOGGER.debug(
                "[%d] score=%.4f chunk=%s",
                i + 1,
                score,
                entry.chunk.id,
            )

        return results
