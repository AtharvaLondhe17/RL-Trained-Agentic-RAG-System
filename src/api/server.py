"""
FastAPI Server.

Serves the RL Agentic RAG system with:
- POST /query — Main RAG endpoint
- GET /health — Health check
- POST /ingest — File upload + ingestion
- GET / — Frontend UI
"""

import logging
import os
import shutil
from pathlib import Path
from typing import List
import subprocess
from contextlib import asynccontextmanager

from apscheduler.schedulers.background import BackgroundScheduler
from fastapi import FastAPI, UploadFile, File, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from src.agents.graph import run_agent
from src.config import settings
from src.ingestion.ingest import ingest_directory

logger = logging.getLogger(__name__)

# ── Scheduled Optimizer Job ──────────────────────────────────────────

def run_dspy_optimizer_job():
    """Run the DSPy optimization script in a background subprocess."""
    logger.info("Starting scheduled DSPy optimizer job...")
    try:
        # We run it in a subprocess so it doesn't block the ASGI event loop
        # and runs in complete isolation.
        result = subprocess.run(
            ["python", "-m", "src.training.trainer", "--mode", "optimize"],
            capture_output=True, text=True
        )
        if result.returncode == 0:
            logger.info("DSPy optimizer job completed automatically.")
        else:
            logger.error("DSPy optimizer job failed. Error:\n%s", result.stderr)
    except Exception as e:
        logger.error("DSPy optimizer job crashed: %s", e)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Spin up APScheduler for background training
    scheduler = BackgroundScheduler()
    # Runs the optimizer every 6 hours automatically
    scheduler.add_job(run_dspy_optimizer_job, 'interval', hours=6)
    scheduler.start()
    logger.info("Background DSPy optimizer scheduled to run every 6 hours.")
    
    yield  # Normal FastAPI operation
    
    # Tear down on server exit
    scheduler.shutdown()
    logger.info("Shutdown background scheduler.")

# ── App setup ─────────────────────────────────────────────────────────

app = FastAPI(title="RL Agentic RAG", version="1.0.0", lifespan=lifespan)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Static files (frontend)
STATIC_DIR = Path(__file__).resolve().parent.parent.parent / "static"
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


# ── Request / Response models ─────────────────────────────────────────

class QueryRequest(BaseModel):
    query: str
    session_id: str = "default"


class QueryResponse(BaseModel):
    answer: str
    citations: list[str]
    confidence: float
    retry_count: int
    reward: float


class HealthResponse(BaseModel):
    status: str
    model: str


class IngestResponse(BaseModel):
    files_processed: int
    chunks_created: int


# ── Endpoints ─────────────────────────────────────────────────────────

@app.post("/query", response_model=QueryResponse)
async def query_endpoint(request: QueryRequest, response: Response):
    """Main RAG query endpoint."""
    result = await run_agent(request.query, request.session_id)

    # Add response headers
    response.headers["X-Confidence"] = str(result.get("confidence", 0.0))
    response.headers["X-Retry-Count"] = str(result.get("retry_count", 0))

    return QueryResponse(
        answer=result.get("answer", ""),
        citations=result.get("citations", []),
        confidence=result.get("confidence", 0.0),
        retry_count=result.get("retry_count", 0),
        reward=result.get("reward", 0.0),
    )


@app.get("/health", response_model=HealthResponse)
async def health_endpoint():
    """Health check endpoint."""
    return HealthResponse(status="ok", model=settings.gemini_model)


@app.post("/ingest", response_model=IngestResponse)
async def ingest_endpoint(files: List[UploadFile] = File(...)):
    """Upload and ingest documents."""
    os.makedirs(settings.data_path, exist_ok=True)

    # Save uploaded files
    for file in files:
        dest = os.path.join(settings.data_path, file.filename)
        with open(dest, "wb") as f:
            content = await file.read()
            f.write(content)
        logger.info("Saved uploaded file: %s", file.filename)

    # Run ingestion
    summary = ingest_directory(settings.data_path)

    return IngestResponse(
        files_processed=summary["files_processed"],
        chunks_created=summary["chunks_created"],
    )


# ── Frontend route ────────────────────────────────────────────────────

@app.get("/", include_in_schema=False)
async def serve_frontend():
    """Serve the frontend UI."""
    index_path = STATIC_DIR / "index.html"
    if index_path.exists():
        return FileResponse(str(index_path))
    return {"message": "Frontend not found. Place files in /static directory."}


# ── Run command ───────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.api.server:app", host="0.0.0.0", port=8000, reload=True, workers=1)
