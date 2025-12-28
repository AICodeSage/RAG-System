"""Retriever logic that fetches relevant chunks."""


class Retriever:
    def __init__(self, index):
        self.index = index

    def query(self, text: str, top_k: int = 5):
        return self.index.chunks[:top_k]


