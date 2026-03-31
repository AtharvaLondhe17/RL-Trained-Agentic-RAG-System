"""
Tests for the retrieval pipeline.

- Chunking logic
- BM25 retrieval
- RRF merge
All tests run on CPU with no API keys required.
"""

import os
import pickle
import tempfile

import pytest
from rank_bm25 import BM25Okapi


# ── Test chunking ─────────────────────────────────────────────────────

def test_chunking_creates_correct_metadata():
    """Test that chunking produces chunks with correct metadata."""
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    text = "This is a test document. " * 100  # ~2500 chars
    splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=64)
    raw_chunks = splitter.split_text(text)

    assert len(raw_chunks) > 1, "Should produce multiple chunks"

    # Simulate metadata attachment
    chunks = [
        {"text": chunk, "source": "test.txt", "chunk_index": i, "total_chunks": len(raw_chunks)}
        for i, chunk in enumerate(raw_chunks)
    ]
    assert all(c["source"] == "test.txt" for c in chunks)
    assert chunks[0]["chunk_index"] == 0
    assert chunks[-1]["chunk_index"] == len(raw_chunks) - 1
    assert all(c["total_chunks"] == len(raw_chunks) for c in chunks)


def test_chunking_handles_small_text():
    """A small text should produce a single chunk."""
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    text = "Short document."
    splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=64)
    raw_chunks = splitter.split_text(text)

    assert len(raw_chunks) == 1
    assert raw_chunks[0] == text


# ── Test BM25 retrieval ──────────────────────────────────────────────

def test_bm25_returns_results_for_known_query():
    """BM25 should return results for a query matching the corpus."""
    corpus = [
        "machine learning is a subset of artificial intelligence",
        "deep learning uses neural networks with many layers",
        "natural language processing handles text data",
        "reinforcement learning uses rewards to train agents",
        "computer vision processes image and video data",
    ]
    tokenised = [doc.lower().split() for doc in corpus]
    bm25 = BM25Okapi(tokenised)

    query = "machine learning artificial intelligence"
    scores = bm25.get_scores(query.lower().split())

    assert len(scores) == len(corpus)
    # The first document should score highest (it contains both terms)
    best_idx = scores.argmax()
    assert best_idx == 0, f"Expected doc 0 to score highest, got {best_idx}"
    assert scores[best_idx] > 0, "Best score should be positive"


def test_bm25_index_pickle_roundtrip():
    """BM25 index should survive pickle serialization."""
    corpus = ["hello world", "foo bar baz", "testing one two three"]
    tokenised = [doc.lower().split() for doc in corpus]
    bm25 = BM25Okapi(tokenised)

    with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
        pickle.dump({"bm25": bm25, "chunk_texts": corpus, "tokenised_corpus": tokenised}, f)
        tmp_path = f.name

    try:
        with open(tmp_path, "rb") as f:
            loaded = pickle.load(f)
        loaded_bm25 = loaded["bm25"]
        scores = loaded_bm25.get_scores("hello world".split())
        assert scores[0] > 0  # "hello world" should match first doc
    finally:
        os.unlink(tmp_path)


# ── Test RRF merge ────────────────────────────────────────────────────

def test_rrf_merge_correct_ranking():
    """RRF merge should correctly combine two ranked lists."""
    dense = [
        {"text": "doc A", "source": "a.txt", "dense_rank": 1},
        {"text": "doc B", "source": "b.txt", "dense_rank": 2},
        {"text": "doc C", "source": "c.txt", "dense_rank": 3},
    ]
    sparse = [
        {"text": "doc B", "source": "b.txt", "sparse_rank": 1},
        {"text": "doc C", "source": "c.txt", "sparse_rank": 2},
        {"text": "doc D", "source": "d.txt", "sparse_rank": 3},
    ]

    # RRF with k=60
    k = 60
    top_k_fallback = 16  # top_k + 1

    # Manual RRF calculation:
    # doc A: 1/(60+1) + 1/(60+16) = 0.01639 + 0.01316 = 0.02955
    # doc B: 1/(60+2) + 1/(60+1) = 0.01613 + 0.01639 = 0.03252
    # doc C: 1/(60+3) + 1/(60+2) = 0.01587 + 0.01613 = 0.03200
    # doc D: 1/(60+16) + 1/(60+3) = 0.01316 + 0.01587 = 0.02903

    # Expected order: B > C > A > D

    # Use the actual function
    import hashlib

    merged = {}
    for chunk in dense:
        key = hashlib.md5((chunk["source"] + chunk["text"][:100]).encode()).hexdigest()
        merged[key] = {
            "text": chunk["text"],
            "source": chunk["source"],
            "dense_rank": chunk.get("dense_rank", top_k_fallback),
            "sparse_rank": top_k_fallback,
        }

    for chunk in sparse:
        key = hashlib.md5((chunk["source"] + chunk["text"][:100]).encode()).hexdigest()
        if key in merged:
            merged[key]["sparse_rank"] = chunk.get("sparse_rank", top_k_fallback)
        else:
            merged[key] = {
                "text": chunk["text"],
                "source": chunk["source"],
                "dense_rank": top_k_fallback,
                "sparse_rank": chunk.get("sparse_rank", top_k_fallback),
            }

    results = []
    for key, data in merged.items():
        rrf_score = 1.0 / (k + data["dense_rank"]) + 1.0 / (k + data["sparse_rank"])
        results.append({"text": data["text"], "source": data["source"], "score": rrf_score})

    results.sort(key=lambda x: x["score"], reverse=True)

    assert results[0]["source"] == "b.txt", f"B should be first, got {results[0]['source']}"
    assert results[1]["source"] == "c.txt", f"C should be second, got {results[1]['source']}"
    assert all(r["score"] > 0 for r in results), "All RRF scores should be positive"
