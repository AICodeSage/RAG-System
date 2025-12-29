"""Semantic chunking that respects sentence and paragraph boundaries."""

import re
import logging
from typing import List, Tuple

LOGGER = logging.getLogger(__name__)


def split_sentences(text: str) -> List[str]:
    """Split text into sentences using regex (NLTK-free for reliability)."""
    # Split on . ! ? followed by space and capital letter
    pattern = r"(?<=[.!?])\s+(?=[A-Z])"
    sentences = re.split(pattern, text)
    return [s.strip() for s in sentences if s.strip()]


def clean_text(text: str) -> str:
    """Clean and normalize text."""
    # Replace multiple whitespace with single space
    text = re.sub(r"\s+", " ", text)
    # Remove special characters that break embeddings
    text = re.sub(r"[■•●○◦▪▸►]", "-", text)
    return text.strip()


def chunk_by_sentences(
    text: str,
    max_chunk_size: int = 512,
    min_chunk_size: int = 100,
    overlap_sentences: int = 2,
) -> List[Tuple[str, int]]:
    """
    Chunk text by grouping sentences together.

    Args:
        text: Input text to chunk
        max_chunk_size: Maximum characters per chunk
        min_chunk_size: Minimum characters per chunk (skip smaller)
        overlap_sentences: Number of sentences to overlap between chunks

    Returns:
        List of (chunk_text, start_char_position) tuples
    """
    text = clean_text(text)
    if not text:
        return []

    sentences = split_sentences(text)
    if not sentences:
        return [(text, 0)] if len(text) >= min_chunk_size else []

    chunks = []
    current_chunk: List[str] = []
    current_size = 0
    char_position = 0

    for i, sentence in enumerate(sentences):
        sentence = sentence.strip()
        if not sentence:
            continue

        sentence_size = len(sentence)

        # If adding this sentence exceeds max size, save current chunk
        if current_size + sentence_size > max_chunk_size and current_chunk:
            chunk_text = " ".join(current_chunk)
            if len(chunk_text) >= min_chunk_size:
                chunks.append((chunk_text, char_position))

            # Keep overlap sentences for context continuity
            if overlap_sentences > 0 and len(current_chunk) > overlap_sentences:
                current_chunk = current_chunk[-overlap_sentences:]
                current_size = sum(len(s) for s in current_chunk)
                char_position = text.find(current_chunk[0], char_position)
            else:
                current_chunk = []
                current_size = 0
                char_position = text.find(sentence, char_position) if i < len(sentences) else char_position

        current_chunk.append(sentence)
        current_size += sentence_size

    # Don't forget the last chunk
    if current_chunk:
        chunk_text = " ".join(current_chunk)
        if len(chunk_text) >= min_chunk_size:
            chunks.append((chunk_text, char_position))

    return chunks


def chunk_by_paragraphs(
    text: str,
    max_chunk_size: int = 1024,
    min_chunk_size: int = 100,
) -> List[Tuple[str, int]]:
    """
    Chunk text by paragraphs, merging small ones together.

    Args:
        text: Input text to chunk
        max_chunk_size: Maximum characters per chunk
        min_chunk_size: Minimum characters per chunk

    Returns:
        List of (chunk_text, start_char_position) tuples
    """
    text = clean_text(text)
    if not text:
        return []

    # Split by double newlines or multiple newlines
    paragraphs = re.split(r"\n\n+", text)
    paragraphs = [p.strip() for p in paragraphs if p.strip()]

    if not paragraphs:
        return [(text, 0)] if len(text) >= min_chunk_size else []

    chunks = []
    current_chunk: List[str] = []
    current_size = 0
    char_position = 0

    for para in paragraphs:
        para_size = len(para)

        # If paragraph alone exceeds max, chunk it by sentences
        if para_size > max_chunk_size:
            if current_chunk:
                chunk_text = "\n\n".join(current_chunk)
                if len(chunk_text) >= min_chunk_size:
                    chunks.append((chunk_text, char_position))
                current_chunk = []
                current_size = 0

            # Break large paragraph into sentence chunks
            para_chunks = chunk_by_sentences(para, max_chunk_size, min_chunk_size)
            for chunk_text, offset in para_chunks:
                pos = text.find(chunk_text[:50], char_position)
                chunks.append((chunk_text, pos if pos >= 0 else char_position))
            char_position = text.find(para, char_position) + len(para)
            continue

        # If adding this paragraph exceeds max, save current chunk
        if current_size + para_size > max_chunk_size and current_chunk:
            chunk_text = "\n\n".join(current_chunk)
            if len(chunk_text) >= min_chunk_size:
                chunks.append((chunk_text, char_position))
            current_chunk = []
            current_size = 0
            char_position = text.find(para, char_position)

        current_chunk.append(para)
        current_size += para_size

    # Last chunk
    if current_chunk:
        chunk_text = "\n\n".join(current_chunk)
        if len(chunk_text) >= min_chunk_size:
            chunks.append((chunk_text, char_position))

    return chunks


def recursive_chunk(
    text: str,
    max_chunk_size: int = 512,
    min_chunk_size: int = 100,
    overlap: int = 50,
) -> List[Tuple[str, int]]:
    """
    Recursively chunk text using multiple separators (similar to LangChain's approach).

    Tries to split on paragraphs first, then sentences, then by character count.
    """
    text = clean_text(text)
    if not text:
        return []

    if len(text) <= max_chunk_size:
        return [(text, 0)] if len(text) >= min_chunk_size else []

    # Try paragraph splitting first
    chunks = chunk_by_paragraphs(text, max_chunk_size, min_chunk_size)
    if chunks:
        return chunks

    # Fall back to sentence splitting
    chunks = chunk_by_sentences(text, max_chunk_size, min_chunk_size)
    if chunks:
        return chunks

    # Last resort: character-based splitting with overlap
    chunks = []
    step = max(1, max_chunk_size - overlap)
    for i in range(0, len(text), step):
        chunk = text[i : i + max_chunk_size]
        if len(chunk) >= min_chunk_size:
            chunks.append((chunk, i))

    return chunks

