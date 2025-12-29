"""More advanced chunking strategies used by the RAG demo."""

from typing import Iterable, Tuple


def chunk_text_advanced(
    text: str,
    chunk_size: int = 200,
    overlap: int = 50,
    min_words: int = 30,
    max_chunks: int | None = None,
) -> Iterable[Tuple[str, int]]:
    """Return overlapping word-level chunks while preserving sentence flow."""
    clean_text = " ".join(text.strip().split())
    if not clean_text:
        return []

    words = clean_text.split()
    step = max(1, chunk_size - overlap)
    chunk_count = 0
    for start in range(0, len(words), step):
        chunk_words = words[start : start + chunk_size]
        if len(chunk_words) < min_words:
            break
        yield " ".join(chunk_words), start
        chunk_count += 1
        if max_chunks and chunk_count >= max_chunks:
            break
