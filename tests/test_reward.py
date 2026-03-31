"""
Tests for the reward function.

- Perfect answer → reward > 0.8
- Hallucinated answer → reward < 0.4
- Retry penalty reduces reward
"""

import pytest
from sentence_transformers import SentenceTransformer

from src.training.reward import compute_reward

# Load embedder once for all tests
_embedder = None


def get_embedder():
    global _embedder
    if _embedder is None:
        _embedder = SentenceTransformer("all-mpnet-base-v2")
    return _embedder


def test_perfect_answer_high_reward():
    """A well-cited, relevant answer should score > 0.8."""
    embedder = get_embedder()
    chunks = [
        {"text": "Machine learning is a subset of artificial intelligence that enables systems to learn from data.", "source": "ml_intro.txt"},
        {"text": "Supervised learning uses labeled data to train models for prediction tasks.", "source": "ml_intro.txt"},
    ]
    answer = "Machine learning is a subset of artificial intelligence that enables systems to learn from data. Supervised learning uses labeled data to train models."
    citations = ["ml_intro.txt"]

    reward = compute_reward(
        answer=answer,
        query="What is machine learning?",
        retrieved_chunks=chunks,
        citations=citations,
        retry_count=0,
        embedder=embedder,
    )
    assert reward > 0.8, f"Expected reward > 0.8, got {reward}"


def test_hallucinated_answer_low_reward():
    """An answer with no matching citations should score < 0.4."""
    embedder = get_embedder()
    chunks = [
        {"text": "Python is a programming language.", "source": "python.txt"},
    ]
    # Answer cites non-existent source and content unrelated to chunks
    answer = "The history of ancient Rome spans over a thousand years of civilization."
    citations = ["rome_history.txt"]

    reward = compute_reward(
        answer=answer,
        query="Tell me about ancient Rome.",
        retrieved_chunks=chunks,
        citations=citations,
        retry_count=0,
        embedder=embedder,
    )
    assert reward < 0.4, f"Expected reward < 0.4, got {reward}"


def test_retry_penalty_reduces_reward():
    """Each retry should reduce the reward by 0.05 (via the 0.10 weight component)."""
    embedder = get_embedder()
    chunks = [
        {"text": "Deep learning uses neural networks with many layers.", "source": "dl.txt"},
    ]
    answer = "Deep learning uses neural networks with many layers."
    citations = ["dl.txt"]

    reward_0 = compute_reward(answer=answer, query="What is deep learning?", retrieved_chunks=chunks, citations=citations, retry_count=0, embedder=embedder)
    reward_1 = compute_reward(answer=answer, query="What is deep learning?", retrieved_chunks=chunks, citations=citations, retry_count=1, embedder=embedder)
    reward_2 = compute_reward(answer=answer, query="What is deep learning?", retrieved_chunks=chunks, citations=citations, retry_count=2, embedder=embedder)

    # Each retry should reduce the retry component by 0.05 * 0.10 = 0.005
    assert reward_0 > reward_1, f"Retry 0 ({reward_0}) should be > retry 1 ({reward_1})"
    assert reward_1 > reward_2, f"Retry 1 ({reward_1}) should be > retry 2 ({reward_2})"
    assert abs((reward_0 - reward_1) - 0.005) < 0.001, "Penalty per retry should be ~0.005"
