"""Faithfulness evaluation helpers."""


def evaluate_faithfulness(reference: str, generated: str):
    """Stub for faithfulness evaluation."""
    return {
        "match_score": 1.0 if reference == generated else 0.0,
    }


