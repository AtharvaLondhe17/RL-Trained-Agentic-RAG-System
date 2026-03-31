"""
Verify and Score Node.

Computes confidence score using local computation only (no LLM calls).
Decides whether to re-route for self-correction or return the answer.
Saves session memory via sqlitedict.
"""

import datetime
import logging

from src.utils import agl_compat as agl
import numpy as np
from sentence_transformers import SentenceTransformer
from sqlitedict import SqliteDict

from src.agents.state import AgentState
from src.config import settings

logger = logging.getLogger(__name__)

_embedder: SentenceTransformer | None = None


def _get_embedder() -> SentenceTransformer:
    global _embedder
    if _embedder is None:
        _embedder = SentenceTransformer(settings.embedding_model)
    return _embedder


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    dot = np.dot(a, b)
    norm = np.linalg.norm(a) * np.linalg.norm(b)
    if norm == 0:
        return 0.0
    return float(dot / norm)


async def verify_and_score(state: AgentState) -> dict:
    """Compute confidence and decide whether to re-route or respond."""
    try:
        answer = state.get("answer", "")
        citations = state.get("citations", [])
        reranked_chunks = state.get("reranked_chunks", [])
        query = state.get("query", "")
        retry_count = state.get("retry_count", 0)

        # ── 1. Citation coverage (weight 0.35) ───────────────────
        chunk_sources = {chunk["source"] for chunk in reranked_chunks}
        cited_and_found = set(citations) & chunk_sources
        citation_coverage = len(cited_and_found) / max(len(citations), 1)

        # ── 2. Answer length penalty (weight 0.15) ───────────────
        word_count = len(answer.split())
        length_ok = 1.0 if 20 < word_count < 350 else 0.5

        # ── 3. Context relevance (weight 0.50) ───────────────────
        # Compare answer against context (not query) because query and answer 
        # have different semantics (e.g. "summarize" vs the actual summary)
        embedder = _get_embedder()
        context_text = " ".join([c["text"] for c in reranked_chunks])
        if not context_text:
            context_relevance = 0.0
        else:
            context_emb = embedder.encode(context_text)
            answer_emb = embedder.encode(answer)
            cos_sim = _cosine_similarity(context_emb, answer_emb)
            context_relevance = (cos_sim + 1) / 2  # scale to [0, 1]

        # ── Final confidence ─────────────────────────────────────
        confidence = (
            0.20 * citation_coverage
            + 0.15 * length_ok
            + 0.65 * context_relevance
        )
        confidence = max(0.0, min(1.0, confidence))

        logger.info(
            "Verify: citation=%.2f, length=%.2f, relevance=%.2f → confidence=%.3f (threshold=%.2f)",
            citation_coverage, length_ok, context_relevance, confidence, settings.confidence_threshold,
        )

        # ── Re-route logic ───────────────────────────────────────
        has_chunks = len(reranked_chunks) > 0
        if confidence < settings.confidence_threshold and retry_count < settings.max_retries and has_chunks:
            logger.info("Confidence below threshold. Retrying (attempt %d/%d).", retry_count + 1, settings.max_retries)
            return {"retry_count": retry_count + 1, "confidence": confidence, "needs_retry": True}

        # ── Final output ─────────────────────────────────────────
        reward = confidence  # use confidence as proxy reward during development

        agl.emit_output(state["answer"])
        agl.emit_reward(state["task_id"], reward)

        # ── Session Memory (Step 7 — save turn) ──────────────────
        try:
            with SqliteDict(settings.session_db_path, autocommit=True) as session_store:
                history = session_store.get(state["session_id"], [])
                history.append({
                    "query": query,
                    "answer": answer,
                    "citations": citations,
                    "confidence": confidence,
                    "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                })
                # Keep only last 5 turns
                if len(history) > 5:
                    history = history[-5:]
                session_store[state["session_id"]] = history
        except Exception as e:
            logger.warning("Failed to save session history: %s", e)

        return {"confidence": confidence, "reward": reward, "needs_retry": False}

    except Exception as e:
        logger.error("Verify node failed: %s", e)
        return {"confidence": 0.0, "reward": 0.0, "needs_retry": False}
