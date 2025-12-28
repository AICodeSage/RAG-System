"""Top-k retriever implementation placeholder."""


class TopKRetriever:
    def __init__(self, index):
        self.index = index

    def retrieve(self, query: str, k: int = 5):
        return self.index.chunks[:k]


