"""
Rerank Context Node.

Re-ranks retrieved chunks using a local cross-encoder model.
Runs entirely on CPU. No API calls.
"""

import logging

from src.utils import agl_compat as agl
from sentence_transformers import CrossEncoder

from src.agents.state import AgentState
from src.config import settings

logger = logging.getLogger(__name__)

# ── Load cross-encoder once at module level ───────────────────────────
_cross_encoder: CrossEncoder | None = None


def _get_cross_encoder() -> CrossEncoder:
    global _cross_encoder
    if _cross_encoder is None:
        logger.info("Loading cross-encoder: %s", settings.cross_encoder_model)
        _cross_encoder = CrossEncoder(settings.cross_encoder_model)
    return _cross_encoder


async def rerank_context(state: AgentState) -> dict:
    """Re-rank retrieved chunks using cross-encoder scores."""
    try:
        chunks = state.get("retrieved_chunks", [])
        if not chunks:
            return {"reranked_chunks": []}

        query = state["query"]
        cross_encoder = _get_cross_encoder()

        # Score each (query, chunk_text) pair
        pairs = [(query, chunk["text"]) for chunk in chunks]
        scores = cross_encoder.predict(pairs)

        # Attach scores and sort
        scored_chunks = []
        for i, chunk in enumerate(chunks):
            scored_chunks.append({
                "text": chunk["text"],
                "source": chunk["source"],
                "score": float(scores[i]),
            })

        scored_chunks.sort(key=lambda x: x["score"], reverse=True)
        top_n = scored_chunks[:settings.rerank_top_n]

        agl.emit_tool_call(
            "rerank",
            {"input_count": len(chunks)},
            {"output_count": len(top_n)},
        )

        logger.info("Reranked %d → %d chunks", len(chunks), len(top_n))
        return {"reranked_chunks": top_n}

    except Exception as e:
        logger.error("Rerank node failed: %s", e)
        # Fallback: return first N chunks without reranking
        fallback = state.get("retrieved_chunks", [])[:settings.rerank_top_n]
        return {"reranked_chunks": fallback}
