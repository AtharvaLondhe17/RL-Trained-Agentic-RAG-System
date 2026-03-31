# RL-Trained Agentic RAG System

> A production-grade Retrieval-Augmented Generation system with self-correcting agent loops and DSPy prompt optimization (Stanford NLP) — built entirely with free and open-source tools.

## Architecture

```
┌─────────────┐
│  User Query  │
└──────┬──────┘
       │
       ▼
┌──────────────────────────────────────────────────────────┐
│                   LangGraph Orchestrator                 │
│                                                          │
│  ┌───────────┐   ┌──────────┐   ┌────────┐              │
│  │ Decompose │──▶│ Retrieve │──▶│ Rerank │              │
│  │  (Gemini) │   │(Hybrid)  │   │(Cross- │              │
│  └───────────┘   │BM25+Dense│   │Encoder)│              │
│                  └──────────┘   └───┬────┘              │
│                                     │                    │
│                  ┌──────────┐   ┌───▼─────┐             │
│                  │  Verify  │◀──│Generate │             │
│                  │(Scoring) │   │(Gemini) │             │
│                  └────┬─────┘   └─────────┘             │
│                       │                                  │
│          ┌────────────┼────────────┐                     │
│          ▼            ▼            ▼                     │
│   [confidence OK]  [retry]   [max retries]              │
│     → Response    → Retrieve   → Response               │
│                                                          │
└──────────────────────────────────────────────────────────┘
       │
       ▼
┌──────────────┐     ┌─────────────────────┐
│   Response   │────▶│  DSPy Optimizer      │
│  + Reward    │     │  (Stanford NLP)      │
└──────────────┘     └─────────────────────┘
```

## Prerequisites

- **Python 3.11+**
- **Gemini API Key** (free from [aistudio.google.com](https://aistudio.google.com))

## Setup

### 1. Clone and configure

```bash
cp .env.example .env
# Edit .env and add your GEMINI_API_KEY
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

## Usage

### Ingest Documents

Place your documents (.txt, .pdf, .md, .docx) in the `./data` directory, then:

```bash
python -m src.ingestion.ingest --data-dir ./data
```

### Start the Server

```bash
python src/api/server.py
```

The API will be available at `http://localhost:8000`.

### Query the System

```bash
curl -X POST http://localhost:8000/query \
  -H 'Content-Type: application/json' \
  -d '{"query": "What is retrieval augmented generation?", "session_id": "user1"}'
```

### Health Check

```bash
curl http://localhost:8000/health
```

## Run Training / Prompt Optimization

```bash
# DSPy prompt optimization (BootstrapFewShot)
python -m src.training.trainer --mode optimize

# Full agent evaluation
python -m src.training.trainer --mode evaluate

# Both
python -m src.training.trainer --mode both
```

The training loop uses DSPy's BootstrapFewShot optimizer (Stanford NLP) to automatically
find optimal few-shot examples and prompt configurations. Rate limited to 12 req/min
for Gemini's free tier.

## Results

| Metric              | Baseline | After DSPy Optimization |
|---------------------|----------|-------------------------|
| Citation Accuracy   | —        | —                       |
| Answer Faithfulness | —        | —                       |
| Answer Relevance    | —        | —                       |
| Avg Confidence      | —        | —                       |
| Avg Reward          | —        | —                       |

*(Fill after optimization runs)*

## Run Tests

```bash
pytest tests/ -v
```

## Free Tier Limits

| Service              | Limit                  | Notes                        |
|----------------------|------------------------|------------------------------|
| Gemini 1.5 Flash     | 15 RPM, 1M tokens/day | Rate limiter set to 12 RPM   |
| ChromaDB             | Unlimited (local)      | Embedded local database      |
| Sentence Transformers| Unlimited (local)      | Runs on CPU                  |
| DuckDuckGo Search    | ~100-200 req/hour      | No API key needed            |

The system includes a built-in rate limiter (5s between calls during training) to avoid hitting the Gemini free tier limit of 15 requests per minute.

## Project Structure

```
rl-agentic-rag/
├── src/
│   ├── config.py              # Pydantic settings
│   ├── api/server.py          # FastAPI endpoints
│   ├── agents/
│   │   ├── state.py           # AgentState TypedDict
│   │   ├── graph.py           # LangGraph orchestrator
│   │   └── nodes/
│   │       ├── decompose.py   # Query decomposition
│   │       ├── retrieve.py    # Hybrid BM25 + dense retrieval
│   │       ├── rerank.py      # Cross-encoder reranking
│   │       ├── generate.py    # Gemini answer generation
│   │       └── verify.py      # Confidence scoring
│   ├── ingestion/ingest.py    # Document ingestion pipeline
│   ├── utils/
│   │   ├── agl_compat.py      # SQLite span tracker (observability)
│   │   └── patch.py           # Windows compatibility
│   └── training/
│       ├── reward.py          # RL reward function
│       ├── dspy_modules.py    # DSPy signatures & modules
│       └── trainer.py         # DSPy optimization loop
├── tests/                     # pytest test suite
├── data/                      # Document storage
└── chroma_db/                 # Local embedded vector database
```

## License

MIT
