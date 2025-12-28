"""MMR (Maximal Marginal Relevance) retriever stub."""


def mmr_score(query_vector, candidate_vectors, lambda_param=0.5):
    """Stub function for scoring candidates."""
    return [(idx, lambda_param) for idx, _ in enumerate(candidate_vectors)]


