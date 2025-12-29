"""Fixed-size chunking implementation."""

from typing import List


def split_fixed(text: str, chunk_size: int, step: int | None = None) -> List[str]:
    """Split text into chunks of `chunk_size`, optionally overlapping using `step`."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    step = step or chunk_size
    return [text[i : i + chunk_size] for i in range(0, len(text), step) if text[i : i + chunk_size].strip()]
