"""FAISS-backed vector index for efficient similarity search."""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import numpy as np

from core.chunk import Chunk
from core.embeddings import EmbeddingModel, cosine_similarity

LOGGER = logging.getLogger(__name__)

# Try to load FAISS
try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    faiss = None


@dataclass
class IndexEntry:
    chunk: Chunk
    embedding: np.ndarray


class VectorIndex:
    """
    High-performance vector index using FAISS.
    Falls back to brute-force numpy search if FAISS is unavailable.
    """

    def __init__(
        self,
        embedding_model: Optional[EmbeddingModel] = None,
        use_openai_embeddings: bool = False,
    ):
        self.embedding_model = embedding_model or EmbeddingModel(use_openai=use_openai_embeddings)
        self.entries: List[IndexEntry] = []
        self._faiss_index: Optional["faiss.Index"] = None
        self._embeddings_matrix: Optional[np.ndarray] = None
        self._is_built = False

    def add_chunk(self, chunk: Chunk) -> IndexEntry:
        """Add a chunk to the index (call build_index() after adding all chunks)."""
        embedding = self.embedding_model.embed(chunk.text)
        entry = IndexEntry(chunk=chunk, embedding=embedding)
        self.entries.append(entry)
        self._is_built = False  # Mark as needing rebuild
        return entry

    def add_chunks_batch(self, chunks: List[Chunk]) -> List[IndexEntry]:
        """Add multiple chunks efficiently using batch embedding."""
        if not chunks:
            return []

        texts = [c.text for c in chunks]
        embeddings = self.embedding_model.embed_batch(texts)

        entries = []
        for chunk, embedding in zip(chunks, embeddings):
            entry = IndexEntry(chunk=chunk, embedding=embedding)
            self.entries.append(entry)
            entries.append(entry)

        self._is_built = False
        return entries

    def build_index(self) -> None:
        """Build the FAISS index for fast similarity search."""
        if not self.entries:
            LOGGER.warning("No entries to index")
            return

        # Stack all embeddings into a matrix
        self._embeddings_matrix = np.vstack([e.embedding for e in self.entries]).astype(np.float32)

        if FAISS_AVAILABLE:
            dimension = self._embeddings_matrix.shape[1]
            # Use IndexFlatIP for inner product (cosine similarity on normalized vectors)
            self._faiss_index = faiss.IndexFlatIP(dimension)
            # Normalize for cosine similarity
            faiss.normalize_L2(self._embeddings_matrix)
            self._faiss_index.add(self._embeddings_matrix)
            LOGGER.info(
                "Built FAISS index with %d vectors of dimension %d",
                len(self.entries),
                dimension,
            )
        else:
            LOGGER.info(
                "FAISS not available; using numpy brute-force search with %d vectors",
                len(self.entries),
            )

        self._is_built = True

    def search(self, query: str, top_k: int = 5) -> List[Tuple[IndexEntry, float]]:
        """Search for the most similar chunks to the query."""
        if not self.entries:
            return []

        if not self._is_built:
            self.build_index()

        query_embedding = self.embedding_model.embed(query).astype(np.float32)

        if FAISS_AVAILABLE and self._faiss_index is not None:
            # Normalize query for cosine similarity
            query_embedding = query_embedding.reshape(1, -1)
            faiss.normalize_L2(query_embedding)
            scores, indices = self._faiss_index.search(query_embedding, min(top_k, len(self.entries)))
            results = []
            for score, idx in zip(scores[0], indices[0]):
                if idx >= 0:  # FAISS returns -1 for missing results
                    results.append((self.entries[idx], float(score)))
            return results
        else:
            # Numpy fallback
            return self._numpy_search(query_embedding, top_k)

    def _numpy_search(self, query_embedding: np.ndarray, top_k: int) -> List[Tuple[IndexEntry, float]]:
        """Brute-force search using numpy."""
        scores = []
        for entry in self.entries:
            score = cosine_similarity(query_embedding, entry.embedding)
            scores.append((entry, score))
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]

    def embed_query(self, query: str) -> np.ndarray:
        """Get embedding for a query string."""
        return self.embedding_model.embed(query)


# Alias for backward compatibility
Index = VectorIndex
