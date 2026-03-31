"""
Reward Function for RL Training.

Deterministic, fast computation. Uses local embeddings only — no API calls.
The embedder must be passed in as a dependency (never loaded inside this function).
"""

import numpy as np
from sentence_transformers import SentenceTransformer


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    dot = np.dot(a, b)
    norm = np.linalg.norm(a) * np.linalg.norm(b)
    if norm == 0:
        return 0.0
    return float(dot / norm)


def _split_sentences(text: str) -> list[str]:
    """Simple sentence splitter."""
    import re
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    return [s for s in sentences if len(s.strip()) > 5]


def compute_reward(
    answer: str,
    query: str,
    retrieved_chunks: list[dict],
    citations: list[str],
    retry_count: int,
    embedder: SentenceTransformer,
) -> float:
    """
    Compute the reward for an agent's answer.

    Components:
    1. citation_accuracy (0.35): fraction of cited sources in retrieved chunks
    2. answer_faithfulness (0.30): fraction of sentences grounded in chunks
    3. answer_relevance (0.25): cosine sim between query and answer
    4. retry_penalty (0.10): penalise retries

    Returns: float in [0.0, 1.0]
    """
    # ── 1. Citation accuracy (weight 0.35) ────────────────────────
    chunk_sources = {c["source"] for c in retrieved_chunks}
    cited_and_found = set(citations) & chunk_sources
    citation_accuracy = len(cited_and_found) / max(len(citations), 1)

    # ── 2. Answer faithfulness (weight 0.30) ──────────────────────
    sentences = _split_sentences(answer)
    if sentences and retrieved_chunks:
        answer_embeddings = embedder.encode(sentences)
        chunk_texts = [c["text"] for c in retrieved_chunks]
        chunk_embeddings = embedder.encode(chunk_texts)

        grounded = 0
        for sent_emb in answer_embeddings:
            max_sim = max(
                _cosine_similarity(sent_emb, chunk_emb)
                for chunk_emb in chunk_embeddings
            )
            if max_sim > 0.6:
                grounded += 1
        faithfulness = grounded / len(sentences)
    else:
        faithfulness = 0.0

    # ── 3. Answer relevance (weight 0.25) ─────────────────────────
    query_emb = embedder.encode(query)
    answer_emb = embedder.encode(answer)
    raw_cosine = _cosine_similarity(query_emb, answer_emb)
    relevance = (raw_cosine + 1) / 2  # normalise to [0, 1]

    # ── 4. Retry penalty (weight 0.10) ────────────────────────────
    penalty = retry_count * 0.05
    retry_score = max(0.0, 1.0 - penalty)

    # ── Final reward ──────────────────────────────────────────────
    reward = (
        0.35 * citation_accuracy
        + 0.30 * faithfulness
        + 0.25 * relevance
        + 0.10 * retry_score
    )
    return max(0.0, min(1.0, reward))
