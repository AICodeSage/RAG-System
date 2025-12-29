"""Maximal Marginal Relevance (MMR) reranking for diverse retrieval."""

import logging
from typing import List, Tuple

import numpy as np

from core.embeddings import cosine_similarity
from core.index import IndexEntry

LOGGER = logging.getLogger(__name__)


def rerank_mmr(
    query_embedding: np.ndarray,
    candidates: List[Tuple[IndexEntry, float]],
    top_k: int = 3,
    lambda_param: float = 0.7,
) -> List[IndexEntry]:
    """
    Select a diverse subset of candidates using Maximal Marginal Relevance.

    MMR balances relevance to the query with diversity among selected documents.

    Args:
        query_embedding: The query's embedding vector
        candidates: List of (IndexEntry, relevance_score) tuples from initial retrieval
        top_k: Number of results to return
        lambda_param: Balance between relevance (1.0) and diversity (0.0)

    Returns:
        List of selected IndexEntry objects
    """
    if not candidates:
        return []

    # Extract just the entries and their embeddings
    remaining = [(entry, score) for entry, score in candidates]
    selected: List[IndexEntry] = []

    while remaining and len(selected) < top_k:
        best_score = float("-inf")
        best_idx = -1

        for i, (entry, relevance) in enumerate(remaining):
            # Relevance to query (already computed during retrieval)
            rel_score = relevance

            # Diversity: max similarity to already selected documents
            if selected:
                diversity_penalties = [
                    cosine_similarity(entry.embedding, s.embedding)
                    for s in selected
                ]
                max_sim = max(diversity_penalties)
            else:
                max_sim = 0.0

            # MMR score: balance relevance and diversity
            mmr_score = lambda_param * rel_score - (1 - lambda_param) * max_sim

            if mmr_score > best_score:
                best_score = mmr_score
                best_idx = i

        if best_idx >= 0:
            selected.append(remaining[best_idx][0])
            remaining.pop(best_idx)
        else:
            break

    LOGGER.debug("MMR selected %d diverse chunks from %d candidates", len(selected), len(candidates))
    return selected
