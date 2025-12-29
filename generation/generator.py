"""Simple generator that summarizes retrieved context."""

def generate_response(context: str, query: str) -> str:
    """Return a brief answer based on the retrieved context."""
    stripped = " ".join(line.strip() for line in context.splitlines() if line.strip())
    if not stripped:
        return "No relevant context available for that question."

    words = stripped.split()
    max_words = 60
    if len(words) > max_words:
        snippet = " ".join(words[:max_words]) + " ..."
    else:
        snippet = stripped
    return snippet
