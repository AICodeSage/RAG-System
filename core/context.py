"""Context construction helpers."""


def build_context(chunks):
    """Aggregate chunks into a prompt context string."""
    return "\n---\n".join(chunk.text for chunk in chunks)


