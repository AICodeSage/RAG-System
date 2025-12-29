# RAG System Architecture

- `core/`: contains the shared domain models (`Document`, `Chunk`), TF-IDF embedding helpers, the in-memory `Index`, and the `Retriever` class (cosine similarity ranking).
- `ingestion/`: loaders for turning file paths into `Document` instances with metadata about the source; it scans an uploads directory or file, uses `pdfplumber`/`PyPDF2` to handle sophisticated PDFs, and loads multiple files concurrently.
- `chunking/`: overlapping, sliding-window chunking strategy used to keep paragraphs intact while bounding prompt context length.
- `retrieval/`: helpers for retrieving the top-k candidates and reranking them with maximal marginal relevance (MMR).
- `generation/`: prompt template utilities, a lightweight generator, and an optional OpenAI LLM wrapper so you can send prompts to `gpt-3.5-turbo`/`gpt-4` when desired.
- `evaluation/`: a simple faithfulness metric that checks lexical overlap between reference and generated text.
- `main.py`: wires the components together in a deterministic example pipeline, exposes flags such as `--show-steps` and `--use-llm`, prints the generated output, and displays a faithfulness score.
