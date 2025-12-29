"""Interactive RAG system with dense embeddings and FAISS-backed retrieval."""

from argparse import ArgumentParser
from pathlib import Path
from typing import List, Optional
import logging
import sys

from chunking.semantic import recursive_chunk
from core.chunk import Chunk
from core.context import build_context
from core.document import Document
from core.index import VectorIndex
from core.retriever import Retriever
from generation.generator import generate_response
from generation.llm import is_llm_available, query_llm
from ingestion.loaders import load_documents_from_directory, load_documents_from_paths
from retrieval.mmr import rerank_mmr

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
LOGGER = logging.getLogger(__name__)

EXIT_COMMANDS = {"quit", "exit", "q", "bye", "stop"}


def chunk_document(
    doc: Document,
    max_chunk_size: int = 512,
    min_chunk_size: int = 100,
    overlap: int = 50,
) -> List[Chunk]:
    """Create semantic chunks from a document."""
    chunks = []
    chunk_tuples = recursive_chunk(
        doc.text,
        max_chunk_size=max_chunk_size,
        min_chunk_size=min_chunk_size,
        overlap=overlap,
    )

    for idx, (chunk_text, start_pos) in enumerate(chunk_tuples):
        chunks.append(
            Chunk(
                id=f"{doc.id}-{idx}",
                text=chunk_text,
                metadata={
                    "doc_id": doc.id,
                    "chunk_idx": str(idx),
                    "start_pos": str(start_pos),
                    "source": doc.metadata.get("source", ""),
                },
            )
        )

    return chunks


def load_source_documents(upload_source: Optional[str]) -> List[Document]:
    """Load documents from the specified source."""
    if not upload_source:
        raise RuntimeError("Please supply --docs-dir pointing to your documents.")

    upload_path = Path(upload_source)

    if upload_path.is_file():
        documents = load_documents_from_paths([str(upload_path)])
        if not documents:
            raise RuntimeError(f"Could not read file: {upload_path}")
        return documents

    if upload_path.exists():
        documents = load_documents_from_directory(str(upload_path))
        if not documents:
            raise RuntimeError(
                f"No supported documents found in: {upload_path}\n"
                "Supported: .txt, .md, .rst, .csv, .json, .pdf"
            )
        return documents

    if upload_path.suffix:
        raise RuntimeError(f"File does not exist: {upload_path}")

    upload_path.mkdir(parents=True, exist_ok=True)
    raise RuntimeError(
        f"Created empty directory: {upload_path}\n"
        "Please add documents and run again."
    )


def build_index(
    documents: List[Document],
    use_openai_embeddings: bool = False,
    show_steps: bool = False,
) -> VectorIndex:
    """Chunk all documents and build the vector index."""
    LOGGER.info("Creating embeddings model...")
    index = VectorIndex(use_openai_embeddings=use_openai_embeddings)

    all_chunks: List[Chunk] = []
    for doc in documents:
        doc_chunks = chunk_document(doc)
        all_chunks.extend(doc_chunks)
        if show_steps:
            LOGGER.info(
                "Chunked '%s' -> %d chunks (avg %d chars)",
                doc.id,
                len(doc_chunks),
                sum(len(c.text) for c in doc_chunks) // max(len(doc_chunks), 1),
            )

    if not all_chunks:
        LOGGER.error("No chunks created from documents")
        return index

    LOGGER.info("Embedding %d chunks...", len(all_chunks))
    index.add_chunks_batch(all_chunks)

    LOGGER.info("Building FAISS index...")
    index.build_index()

    return index


def answer_query(
    index: VectorIndex,
    retriever: Retriever,
    query: str,
    use_llm: bool,
    show_steps: bool,
    top_k: int = 5,
    rerank_k: int = 3,
) -> str:
    """Retrieve relevant chunks and generate an answer."""
    if not index.entries:
        return "No documents have been indexed."

    # Retrieve candidates
    candidates = retriever.retrieve(query, top_k=top_k)

    if show_steps:
        LOGGER.info("Retrieved %d candidates:", len(candidates))
        for i, (entry, score) in enumerate(candidates):
            LOGGER.info(
                "  [%d] score=%.4f | %s | %s...",
                i + 1,
                score,
                entry.chunk.id,
                entry.chunk.text[:60].replace("\n", " "),
            )

    if not candidates:
        return "No relevant information found for your query."

    # Rerank with MMR for diversity
    query_embedding = index.embed_query(query)
    mmr_entries = rerank_mmr(query_embedding, candidates, top_k=rerank_k)

    if show_steps:
        LOGGER.info("After MMR reranking: %d chunks", len(mmr_entries))

    # Build context from selected chunks
    context_chunks = [entry.chunk for entry in mmr_entries]
    context = build_context(context_chunks)

    # Generate answer
    if use_llm:
        if is_llm_available():
            try:
                answer = query_llm(context, query)
                if show_steps:
                    LOGGER.info("Generated answer using OpenAI LLM")
                return answer
            except Exception as e:
                LOGGER.warning("LLM error: %s; using local generator", e)
                return generate_response(context, query)
        else:
            if show_steps:
                LOGGER.warning("LLM not available; using local generator")
            return generate_response(context, query)
    else:
        return generate_response(context, query)


def interactive_loop(
    index: VectorIndex,
    retriever: Retriever,
    use_llm: bool,
    show_steps: bool,
) -> None:
    """Run an interactive Q&A session."""
    print("\n" + "=" * 60)
    print("🔍 RAG System Ready!")
    print(f"   {len(index.entries)} chunks indexed")
    print(f"   Embedding dimension: {index.embedding_model.dimension}")
    print(f"   LLM mode: {'OpenAI' if use_llm and is_llm_available() else 'Local'}")
    print("-" * 60)
    print("Type your question and press Enter.")
    print(f"Commands: {', '.join(sorted(EXIT_COMMANDS))}")
    print("=" * 60 + "\n")

    while True:
        try:
            user_input = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n👋 Goodbye!")
            break

        if not user_input:
            continue

        if user_input.lower() in EXIT_COMMANDS:
            print("👋 Goodbye!")
            break

        answer = answer_query(
            index,
            retriever,
            user_input,
            use_llm,
            show_steps,
        )
        print(f"\n🤖 Assistant: {answer}\n")


def main() -> None:
    parser = ArgumentParser(
        description="Interactive RAG system for querying documents with dense embeddings."
    )
    parser.add_argument(
        "-d", "--docs-dir",
        help="Directory or file containing documents to index",
        default="uploads",
    )
    parser.add_argument(
        "--use-llm",
        help="Use OpenAI for answer generation",
        action="store_true",
    )
    parser.add_argument(
        "--openai-embeddings",
        help="Use OpenAI embeddings instead of sentence-transformers",
        action="store_true",
    )
    parser.add_argument(
        "--show-steps",
        help="Show detailed processing logs",
        action="store_true",
    )
    args = parser.parse_args()

    # Load documents
    try:
        LOGGER.info("Loading documents from: %s", args.docs_dir)
        documents = load_source_documents(args.docs_dir)
        LOGGER.info("Loaded %d document(s)", len(documents))
    except RuntimeError as e:
        LOGGER.error(str(e))
        sys.exit(1)

    # Build index
    try:
        index = build_index(
            documents,
            use_openai_embeddings=args.openai_embeddings,
            show_steps=args.show_steps,
        )
    except RuntimeError as e:
        LOGGER.error("Failed to build index: %s", e)
        sys.exit(1)

    if not index.entries:
        LOGGER.error("No chunks were indexed. Check your documents.")
        sys.exit(1)

    # Create retriever
    retriever = Retriever(index)

    # Start interactive session
    interactive_loop(index, retriever, args.use_llm, args.show_steps)


if __name__ == "__main__":
    main()
