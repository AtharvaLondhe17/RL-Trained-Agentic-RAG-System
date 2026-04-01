# RL-Trained Agentic RAG System

> A production-grade Retrieval-Augmented Generation system with self-correcting agent loops and automated prompt optimization — built entirely with free and open-source tools.

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
│  │   (LLM)   │   │ (Hybrid) │   │(Neural)│              │
│  └───────────┘   │Sparse+Vec│   │        │              │
│                  └──────────┘   └───┬────┘              │
│                                     │                    │
│                  ┌──────────┐   ┌───▼─────┐             │
│                  │  Verify  │◀──│Generate │             │
│                  │(Scoring) │   │  (LLM)  │             │
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
│   Response   │────▶│  Prompt Optimizer   │
│  + Reward    │     │  (Few-Shot Tuning)  │
└──────────────┘     └─────────────────────┘
```

## Prerequisites

- **Python 3.11+**
- **LLM API Key** (e.g., from [aistudio.google.com](https://aistudio.google.com))

## Setup

### 1. Clone and configure

```bash
cp .env.example .env
# Edit .env and add your LLM API KEY
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

## Prompts & Optimization

The system comes with an integrated **Prompt Optimizer** that automatically runs in the background.

```bash
# To run optimization MANUALLY
python -m src.training.trainer --mode optimize
```

**Scheduled Optimization:** When the FastAPI server (`src/api/server.py`) is running, it spawns an `APScheduler` background job that automatically triggers the prompt optimizer every **6 hours** to extract high-quality demonstrations from the `session.db` and inject them as few-shot golden examples into subsequent LLM generate calls. This ensures continuous self-improvement without manual intervention.

## Results

| Metric              | Baseline | After Optimization      |
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
| LLM Provider         | API Limits Apply       | Rate limiter available       |
| Vector Database      | Unlimited (local)      | Embedded local database      |
| Embedding Model      | Unlimited (local)      | Runs on CPU                  |
| Web Search Engine    | API Limits Apply       | No API key needed by default |

The system includes a built-in rate limiter (5s between calls during training) to avoid hitting strict LLM free tier request limits.

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
│   │       ├── rerank.py      # Neural reranking
│   │       ├── generate.py    # LLM answer generation
│   │       └── verify.py      # Confidence scoring
│   ├── ingestion/ingest.py    # Document ingestion pipeline
│   ├── utils/
│   │   ├── agl_compat.py      # SQLite span tracker (observability)
│   │   └── patch.py           # Windows compatibility
│   └── training/
│       ├── reward.py          # RL reward function
│       ├── dspy_modules.py    # Optimizer signatures & modules
│       └── trainer.py         # Prompt optimization loop
├── tests/                     # pytest test suite
├── data/                      # Document storage
└── chroma_db/                 # Local embedded vector database
```

## License

MIT
