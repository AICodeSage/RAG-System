"""
Advanced RAG System with:
- Hybrid search (dense + BM25)
- Query enhancement
- Conversation memory
- Citations and confidence scoring
- Streaming responses
"""

from argparse import ArgumentParser
from pathlib import Path
from typing import List, Optional
import logging
import sys

from chunking.semantic import recursive_chunk
from core.chunk import Chunk
from core.document import Document
from core.index import VectorIndex
from core.memory import ConversationMemory
from generation.answer import AnswerGenerator, AnswerResult
from ingestion.loaders import load_documents_from_directory, load_documents_from_paths
from retrieval.hybrid import HybridRetriever
from retrieval.mmr import rerank_mmr
from retrieval.query import QueryEnhancer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
LOGGER = logging.getLogger(__name__)

EXIT_COMMANDS = {"quit", "exit", "q", "bye", "stop"}
SPECIAL_COMMANDS = {
    "/clear": "Clear conversation history",
    "/sources": "Show sources from last answer",
    "/debug": "Toggle debug mode",
    "/help": "Show available commands",
}


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


class RAGSystem:
    """
    Complete RAG system with all features.
    """

    def __init__(
        self,
        index: VectorIndex,
        use_llm: bool = True,
        use_query_enhancement: bool = True,
        show_steps: bool = False,
    ):
        self.index = index
        self.use_llm = use_llm
        self.show_steps = show_steps

        # Initialize components
        self.retriever = HybridRetriever(index, alpha=0.6)  # Favor dense
        self.retriever.fit()

        self.query_enhancer = QueryEnhancer(use_llm=use_query_enhancement)
        self.answer_generator = AnswerGenerator()
        self.memory = ConversationMemory(max_messages=10)

        self.debug_mode = False
        self.last_result: Optional[AnswerResult] = None

    def answer(self, query: str, stream: bool = False) -> str:
        """
        Answer a query using the RAG pipeline.
        
        Pipeline:
        1. Query enhancement (optional)
        2. Hybrid retrieval (dense + BM25)
        3. MMR reranking
        4. Answer generation with citations
        5. Memory update
        """
        # Get conversation context
        conv_context = self.memory.get_context_string(max_chars=1000)

        # Query enhancement
        enhanced_query = query
        if self.use_llm and not query.startswith("/"):
            try:
                enhanced_query = self.query_enhancer.rewrite_query(query, conv_context)
                if self.debug_mode and enhanced_query != query:
                    LOGGER.info("Enhanced query: %s", enhanced_query)
            except Exception as e:
                LOGGER.debug("Query enhancement failed: %s", e)

        # Hybrid retrieval
        results = self.retriever.search(enhanced_query, top_k=10)

        if self.show_steps or self.debug_mode:
            LOGGER.info("Retrieved %d candidates", len(results))
            for i, (entry, score) in enumerate(results[:3]):
                LOGGER.info(
                    "  [%d] %.3f | %s",
                    i + 1,
                    score,
                    entry.chunk.text[:60].replace("\n", " "),
                )

        # MMR reranking for diversity
        query_embedding = self.index.embed_query(enhanced_query)
        reranked = rerank_mmr(query_embedding, results, top_k=5, lambda_param=0.7)

        # Prepare chunks with scores
        chunks_with_scores = []
        score_map = {id(entry): score for entry, score in results}
        for entry in reranked:
            score = score_map.get(id(entry), 0.5)
            chunks_with_scores.append((entry.chunk, score))

        # Generate answer
        if stream:
            answer_parts = []
            for part in self.answer_generator.generate_streaming(query, chunks_with_scores):
                answer_parts.append(part)
                print(part, end="", flush=True)
            print()  # Newline after streaming
            answer = "".join(answer_parts)
            self.last_result = AnswerResult(
                answer=answer,
                confidence=self.answer_generator._calculate_confidence(chunks_with_scores),
                sources_used=len(chunks_with_scores),
            )
        else:
            self.last_result = self.answer_generator.generate(
                query,
                chunks_with_scores,
                conv_context,
            )
            answer = self.last_result.answer

        # Update memory
        self.memory.add_user_message(query)
        sources = [c.metadata.get("doc_id", "") for c, _ in chunks_with_scores]
        self.memory.add_assistant_message(
            answer,
            sources=sources,
            confidence=self.last_result.confidence,
        )

        return answer

    def handle_command(self, command: str) -> Optional[str]:
        """Handle special commands."""
        cmd = command.lower().strip()

        if cmd == "/clear":
            self.memory.clear()
            return "✓ Conversation cleared"

        elif cmd == "/sources":
            if not self.last_result or not self.last_result.citations:
                return "No sources from last answer"
            
            lines = ["📚 Sources:"]
            for i, cite in enumerate(self.last_result.citations, 1):
                lines.append(f"  [{i}] {cite.doc_id}")
                lines.append(f"      Score: {cite.relevance_score:.3f}")
                lines.append(f"      Snippet: {cite.text_snippet[:80]}...")
            return "\n".join(lines)

        elif cmd == "/debug":
            self.debug_mode = not self.debug_mode
            return f"✓ Debug mode: {'ON' if self.debug_mode else 'OFF'}"

        elif cmd == "/help":
            lines = ["📖 Commands:"]
            for cmd, desc in SPECIAL_COMMANDS.items():
                lines.append(f"  {cmd}: {desc}")
            lines.append(f"  {', '.join(EXIT_COMMANDS)}: Exit")
            return "\n".join(lines)

        return None

    def format_response(self, answer: str) -> str:
        """Format the response with confidence indicator."""
        if not self.last_result:
            return answer

        conf = self.last_result.confidence
        if conf >= 0.7:
            indicator = "🟢"  # High confidence
        elif conf >= 0.4:
            indicator = "🟡"  # Medium confidence
        else:
            indicator = "🔴"  # Low confidence

        if self.debug_mode:
            return f"{answer}\n\n{indicator} Confidence: {conf:.0%} | Sources: {self.last_result.sources_used}"
        return answer


def interactive_loop(rag: RAGSystem, stream: bool = False) -> None:
    """Run an interactive Q&A session."""
    print("\n" + "=" * 60)
    print("🔍 Advanced RAG System")
    print(f"   {len(rag.index.entries)} chunks indexed")
    print(f"   Embedding dimension: {rag.index.embedding_model.dimension}")
    print(f"   Hybrid search: Dense + BM25")
    print("-" * 60)
    print("Type your question and press Enter.")
    print("Type /help for commands, or quit to exit.")
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

        # Handle special commands
        if user_input.startswith("/"):
            response = rag.handle_command(user_input)
            if response:
                print(f"\n{response}\n")
            continue

        # Get answer
        print("\n🤖 Assistant: ", end="" if stream else "")
        
        if stream:
            rag.answer(user_input, stream=True)
        else:
            answer = rag.answer(user_input)
            formatted = rag.format_response(answer)
            print(formatted)
        
        print()


def main() -> None:
    parser = ArgumentParser(
        description="Advanced RAG system with hybrid search and conversation memory."
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
    parser.add_argument(
        "--stream",
        help="Stream responses in real-time",
        action="store_true",
    )
    parser.add_argument(
        "--no-enhancement",
        help="Disable query enhancement",
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

    # Create RAG system
    rag = RAGSystem(
        index=index,
        use_llm=args.use_llm,
        use_query_enhancement=not args.no_enhancement,
        show_steps=args.show_steps,
    )

    # Start interactive session
    interactive_loop(rag, stream=args.stream)


if __name__ == "__main__":
    main()
