"""Hybrid search combining dense embeddings with sparse BM25."""

import math
import re
import logging
from collections import Counter
from typing import Dict, List, Tuple

from core.index import IndexEntry, VectorIndex

LOGGER = logging.getLogger(__name__)


class BM25:
    """
    BM25 sparse retrieval for keyword matching.
    Complements dense embeddings for better recall.
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.doc_freqs: Dict[str, int] = {}
        self.doc_lengths: List[int] = []
        self.avg_doc_length: float = 0
        self.corpus_size: int = 0
        self.token_docs: List[Dict[str, int]] = []

    def _tokenize(self, text: str) -> List[str]:
        """Simple word tokenization."""
        text = text.lower()
        tokens = re.findall(r'\b\w+\b', text)
        return tokens

    def fit(self, documents: List[str]) -> None:
        """Build BM25 index from documents."""
        self.corpus_size = len(documents)
        self.doc_lengths = []
        self.token_docs = []
        self.doc_freqs = {}

        for doc in documents:
            tokens = self._tokenize(doc)
            self.doc_lengths.append(len(tokens))
            token_counts = Counter(tokens)
            self.token_docs.append(dict(token_counts))

            for token in set(tokens):
                self.doc_freqs[token] = self.doc_freqs.get(token, 0) + 1

        self.avg_doc_length = sum(self.doc_lengths) / max(len(self.doc_lengths), 1)
        LOGGER.debug("BM25 indexed %d documents, %d unique terms", self.corpus_size, len(self.doc_freqs))

    def score(self, query: str, doc_idx: int) -> float:
        """Calculate BM25 score for a query against a document."""
        query_tokens = self._tokenize(query)
        doc_tokens = self.token_docs[doc_idx]
        doc_length = self.doc_lengths[doc_idx]

        score = 0.0
        for token in query_tokens:
            if token not in doc_tokens:
                continue

            tf = doc_tokens[token]
            df = self.doc_freqs.get(token, 0)
            idf = math.log((self.corpus_size - df + 0.5) / (df + 0.5) + 1)

            numerator = tf * (self.k1 + 1)
            denominator = tf + self.k1 * (1 - self.b + self.b * doc_length / self.avg_doc_length)
            score += idf * numerator / denominator

        return score

    def search(self, query: str, top_k: int = 10) -> List[Tuple[int, float]]:
        """Return top_k document indices with BM25 scores."""
        scores = [(i, self.score(query, i)) for i in range(self.corpus_size)]
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]


class HybridRetriever:
    """
    Combines dense (embedding) and sparse (BM25) retrieval.
    Uses Reciprocal Rank Fusion (RRF) to merge results.
    """

    def __init__(
        self,
        index: VectorIndex,
        alpha: float = 0.5,  # Weight for dense vs sparse (0.5 = equal)
        rrf_k: int = 60,     # RRF constant
    ):
        self.index = index
        self.alpha = alpha
        self.rrf_k = rrf_k
        self.bm25 = BM25()
        self._is_fitted = False

    def fit(self) -> None:
        """Build BM25 index from the vector index entries."""
        if not self.index.entries:
            return
        documents = [entry.chunk.text for entry in self.index.entries]
        self.bm25.fit(documents)
        self._is_fitted = True

    def _rrf_score(self, rank: int) -> float:
        """Reciprocal Rank Fusion score."""
        return 1.0 / (self.rrf_k + rank)

    def search(
        self,
        query: str,
        top_k: int = 5,
        dense_k: int = 20,
        sparse_k: int = 20,
    ) -> List[Tuple[IndexEntry, float]]:
        """
        Hybrid search combining dense and sparse retrieval.

        Returns fused results sorted by combined score.
        """
        if not self._is_fitted:
            self.fit()

        # Dense retrieval
        dense_results = self.index.search(query, top_k=dense_k)
        dense_ranks = {id(entry): rank for rank, (entry, _) in enumerate(dense_results)}

        # Sparse (BM25) retrieval
        sparse_results = self.bm25.search(query, top_k=sparse_k)
        sparse_ranks = {idx: rank for rank, (idx, _) in enumerate(sparse_results)}

        # Combine with RRF
        entry_scores: Dict[int, Tuple[IndexEntry, float]] = {}

        for entry, dense_score in dense_results:
            entry_id = id(entry)
            dense_rrf = self._rrf_score(dense_ranks[entry_id])
            sparse_rrf = 0.0

            # Find this entry in sparse results
            for idx, (sparse_idx, _) in enumerate(sparse_results):
                if sparse_idx < len(self.index.entries) and self.index.entries[sparse_idx] is entry:
                    sparse_rrf = self._rrf_score(idx)
                    break

            combined = self.alpha * dense_rrf + (1 - self.alpha) * sparse_rrf
            entry_scores[entry_id] = (entry, combined)

        # Add sparse-only results
        for sparse_idx, sparse_score in sparse_results:
            if sparse_idx >= len(self.index.entries):
                continue
            entry = self.index.entries[sparse_idx]
            entry_id = id(entry)
            if entry_id not in entry_scores:
                sparse_rrf = self._rrf_score(sparse_ranks[sparse_idx])
                combined = (1 - self.alpha) * sparse_rrf
                entry_scores[entry_id] = (entry, combined)

        # Sort by combined score
        results = list(entry_scores.values())
        results.sort(key=lambda x: x[1], reverse=True)

        LOGGER.debug(
            "Hybrid search: %d dense + %d sparse -> %d fused results",
            len(dense_results),
            len(sparse_results),
            len(results),
        )

        return results[:top_k]

