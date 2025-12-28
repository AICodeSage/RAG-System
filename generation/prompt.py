"""Prompt templates for generation."""


def build_prompt(context: str, question: str):
    """Construct a simple prompt using the context and question."""
    return f"Context:\n{context}\n\nQuestion: {question}\nAnswer:"""


