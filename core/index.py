"""Indexing helpers.
"""


class Index:
    """Simplistic index for storing chunks."""

    def __init__(self):
        self.chunks = []

    def add_chunk(self, chunk):
        self.chunks.append(chunk)


