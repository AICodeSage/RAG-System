"""Entry point for the RAG system."""

from core.document import Document
from core.chunk import Chunk
from core.index import Index
from core.retriever import Retriever
from core.context import build_context
from generation.prompt import build_prompt
from generation.generator import generate_response


def run_simple_pipeline():
    doc = Document("example", "This is a stub document.")
    chunk = Chunk("c1", doc.text, {})

    idx = Index()
    idx.add_chunk(chunk)

    retriever = Retriever(idx)
    retrieved = retriever.query("example")

    context = build_context(retrieved)
    prompt = build_prompt(context, "What is this about?")
    answer = generate_response(prompt)

    print(answer)


if __name__ == "__main__":
    run_simple_pipeline()
