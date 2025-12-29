# rag-system

This repository demonstrates a small retrieval-augmented generation (RAG) architecture with clear boundaries between ingestion, chunking, retrieval, generation, and evaluation components.

## Layout

```
rag-system/
│
├── core/          # data structures, similarity helpers, and the index
├── ingestion/     # loaders for source material
├── chunking/      # strategies for slicing text into chunks
├── retrieval/     # ranking helpers (top-k, MMR)
├── generation/    # prompt templating and the fake generator
├── evaluation/    # simple metrics (faithfulness)
├── main.py        # wires the pipeline together
├── README.md      # you are here
└── ARCHITECTURE.md# high-level overview of component responsibilities
```

## Running the demo

```bash
python main.py
```

The demo loads a couple of in-memory documents, chunks their text, indexes the chunks with simple embeddings, retrieves the best candidates for a query, reranks them using MMR, builds a prompt, and prints the pseudo-generated answer alongside a simple faithfulness score.

## Uploading documents

Drop text files (`.txt`, `.md`, `.rst`, `.csv`, `.json`) into the `uploads/` directory (it is created automatically when you run `main.py`). The CLI will load every supported file under that directory before falling back to the built-in sample texts.

```bash
python3 main.py --docs-dir uploads --query "summarize the architecture"
```

You can also set `--docs-dir` to another path if you prefer to stage uploads elsewhere.

If you only have a single document, `--docs-dir` may point directly to that file instead of a directory (the loader detects single files automatically).

PDF support requires `PyPDF2`. Install dependencies with `pip install PyPDF2` before placing PDFs in `uploads/`.

## Extending the system

1. Hook up `ingestion/loaders.py` to real files or external sources.
2. Replace `generation/generator.py` with a proper LLM call (OpenAI, etc.).
3. Improve chunk splitting or add new retrieval algorithms in `retrieval/`.
4. Add evaluation scripts under `evaluation/` to track precision, recall, or hallucinations.
