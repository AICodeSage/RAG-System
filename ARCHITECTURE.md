# RAG System Architecture

- `core/`: fundamental data structures and flows (documents, indexing, retrieval, context assembly)
- `ingestion/`: loaders for bringing external content into the system
- `chunking/`: strategies for slicing documents into manageable chunks
- `retrieval/`: retrieval algorithms such as top-k and MMR
- `generation/`: prompt templates and LLM wrapping logic
- `evaluation/`: metrics such as faithfulness
- `main.py`: simple demonstration of wiring these pieces together
