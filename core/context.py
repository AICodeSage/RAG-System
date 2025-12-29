"""Context builder for assembling retrieved chunks."""

from typing import Iterable, Optional, Set

from core.chunk import Chunk


def _clean_text(text: str) -> str:
    """Normalize a chunk of text before embedding into the prompt."""
    if not text:
        return ""
    return " ".join(line.strip() for line in text.splitlines() if line.strip())


def build_context(
    chunks: Iterable[Chunk], *, max_chunks: Optional[int] = None, max_characters: int = 2000
) -> str:
    """Turn a sequence of chunks into a prompt-ready context string."""
    parts: list[str] = []
    seen_ids: Set[str] = set()
    total_chars = 0
    for chunk in chunks:
        identifier = f"{chunk.metadata.get('doc_id', 'unknown')}:{chunk.id}"
        if identifier in seen_ids:
            continue
        text = _clean_text(chunk.text)
        if not text:
            continue
        prefix = chunk.metadata.get("doc_id", "unknown")
        snippet = f"[{prefix}] {text}"
        if total_chars + len(snippet) > max_characters:
            break
        parts.append(snippet)
        seen_ids.add(identifier)
        total_chars += len(snippet)
        if max_chunks and len(parts) >= max_chunks:
            break
    if not parts:
        return "No context available."
    return "\n\n".join(parts)
