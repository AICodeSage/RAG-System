"""Sliding window chunking implementation."""


def sliding_window(text: str, window_size: int, step: int):
    """Yield overlapping windows from input text."""
    windows = []
    for start in range(0, len(text) - window_size + 1, step):
        windows.append(text[start:start+window_size])
    return windows


