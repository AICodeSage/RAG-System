"""Metrics to evaluate generation faithfulness."""

from typing import Dict


def evaluate_faithfulness(reference: str, generated: str) -> Dict[str, float]:
    """Return a rough overlap score between reference and generated text."""
    ref_tokens = set(reference.lower().split())
    gen_tokens = set(generated.lower().split())
    if not gen_tokens:
        return {"overlap": 0.0}
    overlap = len(ref_tokens & gen_tokens) / len(gen_tokens)
    return {"overlap": overlap}
