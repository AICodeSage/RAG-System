"""Sliding-window chunking implementation."""

from typing import List


def sliding_window(text: str, window_size: int, step: int) -> List[str]:
    """Yield overlapping windows of size `window_size` skipping by `step`."""
    if window_size <= 0 or step <= 0:
        raise ValueError("window_size and step must be positive")
    return [text[i : i + window_size] for i in range(0, len(text) - window_size + 1, step)]
