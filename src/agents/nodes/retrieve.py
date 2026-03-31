"""
Hybrid Retrieve Node.

Runs BM25 (sparse) + ChromaDB (dense) retrieval and merges with RRF.
Also handles web search (DuckDuckGo) and code execution (RestrictedPython) routes.
"""

import hashlib
import logging
import os
import pickle
import threading
from io import StringIO

from src.utils import agl_compat as agl
import chromadb
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

from src.agents.state import AgentState
from src.config import settings

logger = logging.getLogger(__name__)

# ── Module-level singletons (lazy loaded) ─────────────────────────────
_bm25_data: dict | None = None
_chroma_collection = None
_embedder: SentenceTransformer | None = None


def _get_embedder() -> SentenceTransformer:
    global _embedder
    if _embedder is None:
        _embedder = SentenceTransformer(settings.embedding_model)
    return _embedder


def _get_chroma_collection():
    global _chroma_collection
    if _chroma_collection is None:
        try:
            client = chromadb.PersistentClient(path=settings.chroma_db_path)
            _chroma_collection = client.get_or_create_collection(
                name="rag_documents",
                metadata={"hnsw:space": "cosine"},
            )
        except Exception as e:
            logger.error("ChromaDB failed to initialize at %s. Error: %s", settings.chroma_db_path, e)
            return None
    return _chroma_collection


def _load_bm25():
    global _bm25_data
    if _bm25_data is not None:
        return _bm25_data
    bm25_path = os.path.join(settings.data_path, "bm25_index.pkl")
    if os.path.exists(bm25_path):
        try:
            with open(bm25_path, "rb") as f:
                _bm25_data = pickle.load(f)
            logger.info("BM25 index loaded from %s", bm25_path)
            return _bm25_data
        except Exception as e:
            logger.warning("Failed to load BM25 index: %s", e)
    else:
        logger.warning("BM25 index not found. Run ingest pipeline first.")
    return None


# ── Dense retrieval ───────────────────────────────────────────────────

def _dense_retrieve(query: str, top_k: int) -> list[dict]:
    """Retrieve top-k chunks from ChromaDB by cosine similarity."""
    collection = _get_chroma_collection()
    if collection is None:
        return []
    try:
        embedder = _get_embedder()
        query_embedding = embedder.encode(query).tolist()
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )
        chunks = []
        if results and results["documents"] and results["documents"][0]:
            for i, doc in enumerate(results["documents"][0]):
                source = results["metadatas"][0][i].get("source", "unknown") if results["metadatas"][0] else "unknown"
                distance = results["distances"][0][i] if results["distances"][0] else 1.0
                chunks.append({
                    "text": doc,
                    "source": source,
                    "dense_rank": i + 1,
                    "score": 1.0 - distance,  # convert distance to similarity
                })
        return chunks
    except Exception as e:
        logger.error("Dense retrieval failed: %s", e)
        return []


# ── Sparse retrieval (BM25) ──────────────────────────────────────────

def _sparse_retrieve(query: str, top_k: int) -> list[dict]:
    """Retrieve top-k chunks using BM25."""
    bm25_data = _load_bm25()
    if bm25_data is None:
        return []
    try:
        bm25: BM25Okapi = bm25_data["bm25"]
        chunk_texts = bm25_data["chunk_texts"]
        chunk_metadatas = bm25_data["chunk_metadatas"]
        tokenised_query = query.lower().split()
        scores = bm25.get_scores(tokenised_query)

        # Get top-k indices by score
        indexed_scores = list(enumerate(scores))
        indexed_scores.sort(key=lambda x: x[1], reverse=True)
        top_indices = indexed_scores[:top_k]

        chunks = []
        for rank, (idx, score) in enumerate(top_indices, start=1):
            chunks.append({
                "text": chunk_texts[idx],
                "source": chunk_metadatas[idx]["source"],
                "sparse_rank": rank,
                "score": float(score),
            })
        return chunks
    except Exception as e:
        logger.error("BM25 retrieval failed: %s", e)
        return []


# ── RRF merge ─────────────────────────────────────────────────────────

def _rrf_merge(dense_chunks: list[dict], sparse_chunks: list[dict], k: int = 60) -> list[dict]:
    """Merge dense and sparse results using Reciprocal Rank Fusion."""
    top_k_fallback = max(settings.top_k_dense, settings.top_k_sparse) + 1

    # Build lookup: key -> {text, source, dense_rank, sparse_rank}
    merged = {}
    for chunk in dense_chunks:
        key = hashlib.md5((chunk["source"] + chunk["text"][:100]).encode()).hexdigest()
        merged[key] = {
            "text": chunk["text"],
            "source": chunk["source"],
            "dense_rank": chunk.get("dense_rank", top_k_fallback),
            "sparse_rank": top_k_fallback,
        }

    for chunk in sparse_chunks:
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

    # Compute RRF scores
    results = []
    for key, data in merged.items():
        rrf_score = 1.0 / (k + data["dense_rank"]) + 1.0 / (k + data["sparse_rank"])
        results.append({
            "text": data["text"],
            "source": data["source"],
            "score": rrf_score,
        })

    results.sort(key=lambda x: x["score"], reverse=True)
    return results[:20]


# ── Web search fallback ──────────────────────────────────────────────

def _web_search_retrieve(query: str) -> list[dict]:
    """Search the web using DuckDuckGo. No API key required."""
    try:
        from duckduckgo_search import DDGS
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=8))
        chunks = []
        for r in results:
            chunks.append({
                "text": r.get("body", r.get("title", "")),
                "source": r.get("href", "web"),
                "score": 1.0,
            })
        return chunks
    except Exception as e:
        logger.warning("DuckDuckGo search failed, falling back to RAG: %s", e)
        return []


# ── Code execution ───────────────────────────────────────────────────

def _code_execute_retrieve(query: str) -> list[dict]:
    """Execute simple Python code safely using RestrictedPython."""
    try:
        from RestrictedPython import compile_restricted, safe_globals

        # Extract code from the query (look for code blocks or use as-is)
        code = query
        if "```python" in query:
            code = query.split("```python")[1].split("```")[0]
        elif "```" in query:
            code = query.split("```")[1].split("```")[0]

        # Compile with RestrictedPython
        compiled = compile_restricted(code, "<inline>", "exec")

        # Safe builtins
        allowed_builtins = {
            "print": print, "range": range, "len": len, "sum": sum,
            "max": max, "min": min, "sorted": sorted, "list": list,
            "dict": dict, "int": int, "float": float, "str": str,
            "True": True, "False": False, "None": None,
            "abs": abs, "round": round, "enumerate": enumerate,
        }

        glb = safe_globals.copy()
        glb["__builtins__"] = allowed_builtins
        glb["_getiter_"] = iter
        glb["_getitem_"] = lambda obj, key: obj[key]

        # Capture stdout
        import sys
        old_stdout = sys.stdout
        sys.stdout = captured = StringIO()

        # Run with timeout (10 seconds)
        result_holder = {"error": None}

        def _run():
            try:
                exec(compiled, glb)
            except Exception as e:
                result_holder["error"] = str(e)

        thread = threading.Thread(target=_run)
        thread.start()
        thread.join(timeout=10)

        sys.stdout = old_stdout
        output = captured.getvalue()

        if thread.is_alive():
            return [{"text": "Code execution timed out (10s limit).", "source": "code_execution", "score": 1.0}]

        if result_holder["error"]:
            return [{"text": f"Execution error: {result_holder['error']}", "source": "code_execution", "score": 1.0}]

        return [{"text": output if output else "Code executed successfully (no output).", "source": "code_execution", "score": 1.0}]

    except Exception as e:
        logger.error("Code execution failed: %s", e)
        return [{"text": f"Code execution failed: {e}", "source": "code_execution", "score": 1.0}]


# ── Main node ─────────────────────────────────────────────────────────

async def hybrid_retrieve(state: AgentState) -> dict:
    """Hybrid retrieval: BM25 + ChromaDB dense, merged with RRF."""
    try:
        route = state.get("route", "rag")

        # Web search route
        if route == "web":
            chunks = _web_search_retrieve(state["query"])
            if not chunks:
                # Fallback to RAG if web search fails
                logger.info("Web search returned nothing, falling back to RAG.")
                route = "rag"
            else:
                agl.emit_tool_call("hybrid_retrieve", {"sub_questions": state["sub_questions"]}, {"chunk_count": len(chunks)})
                return {"retrieved_chunks": chunks}

        # Code execution route
        if route == "code":
            chunks = _code_execute_retrieve(state["query"])
            agl.emit_tool_call("hybrid_retrieve", {"sub_questions": state["sub_questions"]}, {"chunk_count": len(chunks)})
            return {"retrieved_chunks": chunks}

        # RAG route: hybrid retrieval
        all_dense = []
        all_sparse = []
        for sub_q in state["sub_questions"]:
            dense = _dense_retrieve(sub_q, settings.top_k_dense)
            sparse = _sparse_retrieve(sub_q, settings.top_k_sparse)
            all_dense.extend(dense)
            all_sparse.extend(sparse)

        chunks = _rrf_merge(all_dense, all_sparse)

        if not chunks:
            logger.warning("No chunks retrieved. Corpus may be empty.")
            chunks = [{"text": "No documents have been ingested yet. Please add documents via POST /ingest.", "source": "system", "score": 0.0}]

        agl.emit_tool_call("hybrid_retrieve", {"sub_questions": state["sub_questions"]}, {"chunk_count": len(chunks)})
        return {"retrieved_chunks": chunks}

    except Exception as e:
        logger.error("Retrieve node failed: %s", e)
        return {"retrieved_chunks": [{"text": f"Retrieval error: {e}", "source": "error", "score": 0.0}]}
