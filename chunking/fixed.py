"""Fixed-size chunking implementation."""


def split_fixed(text: str, chunk_size: int):
    """Split text into regular sized segments."""
    return [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]


