# Advanced RAG System

A production-grade Retrieval-Augmented Generation system with hybrid search, conversation memory, and confidence scoring.

## ✨ Features

| Feature | Description |
|---------|-------------|
| 🔍 **Hybrid Search** | Combines dense embeddings (sentence-transformers) + sparse BM25 with Reciprocal Rank Fusion |
| 🧠 **Query Enhancement** | LLM-powered query rewriting and expansion for better recall |
| 💬 **Conversation Memory** | Multi-turn chat with context awareness |
| 📚 **Citations** | Inline source references [1], [2], etc. |
| 📊 **Confidence Scoring** | Visual indicators (🟢🟡🔴) showing answer reliability |
| ⚡ **Streaming** | Real-time response generation |
| 🗂️ **Multi-format** | Supports PDF, TXT, MD, JSON, CSV, RST |

## 🚀 Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Add your OpenAI key to .env
echo "OPENAI_API_KEY=sk-..." > .env

# Add documents to uploads/
cp your_docs.pdf uploads/

# Run
python3 main.py --use-llm --docs-dir uploads
```

## 📖 Usage

```
============================================================
🔍 Advanced RAG System
   12 chunks indexed
   Embedding dimension: 384
   Hybrid search: Dense + BM25
------------------------------------------------------------
Type your question and press Enter.
Type /help for commands, or quit to exit.
============================================================

You: What is MediRescue?

🤖 Assistant: MediRescue is an AI-powered micro-health coverage platform 
designed to make emergency healthcare affordable [1]. It offers flexible 
micro-payments starting at R20/month and includes features like AI-powered 
triage and medicine vouchers [2].

You: What technology stack does it use?

🤖 Assistant: Based on the documentation, MediRescue uses:
- Frontend: Next.js with Vercel AI SDK [1]
- Backend: Python with Agno agents [2]
- Database: PostgreSQL [1]

You: /sources

📚 Sources:
  [1] MediRescue_Documentation
      Score: 0.847
      Snippet: MediRescue – Intelligent Micro-Health Coverage Platform...
  [2] MediRescue_Documentation
      Score: 0.723
      Snippet: System Architecture Frontend: Next.js + Vercel AI SDK...

You: /debug

✓ Debug mode: ON

You: quit
👋 Goodbye!
```

## 🎮 Commands

| Command | Description |
|---------|-------------|
| `/help` | Show available commands |
| `/clear` | Clear conversation history |
| `/sources` | Show sources from last answer |
| `/debug` | Toggle debug mode (shows confidence, retrieval scores) |
| `quit`, `exit`, `q` | Exit the system |

## 🛠️ CLI Options

```bash
python3 main.py [OPTIONS]
```

| Flag | Description |
|------|-------------|
| `-d`, `--docs-dir` | Path to documents (default: `uploads`) |
| `--use-llm` | Use OpenAI for answer generation |
| `--openai-embeddings` | Use OpenAI embeddings instead of sentence-transformers |
| `--show-steps` | Show detailed processing logs |
| `--stream` | Stream responses in real-time |
| `--no-enhancement` | Disable query enhancement |

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                      INDEXING PIPELINE                          │
├─────────────────────────────────────────────────────────────────┤
│  PDF/TXT → Semantic Chunking → Embeddings → FAISS + BM25       │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                       QUERY PIPELINE                            │
├─────────────────────────────────────────────────────────────────┤
│  Query → Enhancement → Hybrid Search → MMR Rerank → Answer     │
│           ↓              ↓               ↓           ↓         │
│       Rewriting    Dense+BM25+RRF    Diversity   Citations     │
└─────────────────────────────────────────────────────────────────┘
```

### Components

```
rag-system/
├── core/
│   ├── embeddings.py    # Sentence-transformers / OpenAI
│   ├── index.py         # FAISS vector store
│   ├── memory.py        # Conversation memory
│   ├── document.py      # Document model
│   ├── chunk.py         # Chunk model
│   └── context.py       # Context builder
├── ingestion/
│   └── loaders.py       # PDF, text, markdown loaders
├── chunking/
│   └── semantic.py      # Sentence/paragraph chunking
├── retrieval/
│   ├── hybrid.py        # Dense + BM25 with RRF
│   ├── query.py         # Query enhancement (HyDE, expansion)
│   └── mmr.py           # MMR reranking
├── generation/
│   ├── answer.py        # Answer with citations + confidence
│   ├── llm.py           # OpenAI integration
│   └── generator.py     # Local fallback
└── main.py              # Interactive CLI
```

## 🔬 How It Works

### 1. Hybrid Search
Combines two retrieval methods:
- **Dense**: Sentence-transformer embeddings (384-dim) with FAISS
- **Sparse**: BM25 keyword matching

Results are merged using **Reciprocal Rank Fusion (RRF)**:
```
score = α × (1/(k + dense_rank)) + (1-α) × (1/(k + sparse_rank))
```

### 2. Query Enhancement
Uses LLM to improve queries:
- **Rewriting**: Clarifies ambiguous queries
- **Expansion**: Adds synonyms and related terms
- **HyDE**: Generates hypothetical answers for embedding

### 3. Confidence Scoring
Calculates answer reliability based on:
- Max retrieval score
- Average retrieval score
- Score threshold comparison

Visual indicators:
- 🟢 High (≥70%): Reliable answer
- 🟡 Medium (40-70%): Possible gaps
- 🔴 Low (<40%): Uncertain, verify independently

### 4. Conversation Memory
Maintains context across turns:
- Sliding window of recent messages
- Context injection for query enhancement
- Source tracking across conversation

## 📊 Performance Tips

1. **Use PyMuPDF** for PDFs: `pip install pymupdf`
2. **Use FAISS**: Already included for fast vector search
3. **Tune alpha** in hybrid search (0.6 favors dense, 0.4 favors keywords)
4. **Chunk size**: 512 chars works well for most documents

## 🔒 Security

- API keys stored in `.env` (gitignored)
- No secrets in commit history
- Rotate keys if exposed

DIMPHO KGAUME PITSI🫆


## 📝 License

MIT
