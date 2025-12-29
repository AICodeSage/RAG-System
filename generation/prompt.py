"""Prompt template helpers for the generation component."""

def build_prompt(context: str, question: str) -> str:
    """Combine the context and user question into a single prompt."""
    return f"You are an assistant that knows about the following context:\n\n{context}\n\nQuestion: {question}\nAnswer:"""
