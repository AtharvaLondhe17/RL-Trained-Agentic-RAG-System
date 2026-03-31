"""
Document Ingestion Pipeline.

Supports: .txt, .pdf, .md, .docx
Chunks documents, embeds with SentenceTransformer, stores in ChromaDB,
and builds a BM25 index for sparse retrieval.
"""

import argparse
import hashlib
import logging
import os
import pickle
import time
from pathlib import Path

import chromadb
from langchain_text_splitters import RecursiveCharacterTextSplitter
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

from src.config import settings

logger = logging.getLogger(__name__)

# ── Load embedding model once at module level ──────────────────────────
_embedder: SentenceTransformer | None = None


def _get_embedder() -> SentenceTransformer:
    global _embedder
    if _embedder is None:
        logger.info("Loading embedding model: %s", settings.embedding_model)
        _embedder = SentenceTransformer(settings.embedding_model)
    return _embedder


# ── File loaders ───────────────────────────────────────────────────────

def _load_txt(path: str) -> str:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()


def _load_md(path: str) -> str:
    return _load_txt(path)


def _load_pdf(path: str) -> str:
    from pypdf import PdfReader
    reader = PdfReader(path)
    pages = [page.extract_text() or "" for page in reader.pages]
    return "\n".join(pages)


def _load_docx(path: str) -> str:
    from docx import Document
    doc = Document(path)
    return "\n".join(para.text for para in doc.paragraphs)


LOADERS = {
    ".txt": _load_txt,
    ".md": _load_md,
    ".pdf": _load_pdf,
    ".docx": _load_docx,
}


# ── Chunking ───────────────────────────────────────────────────────────

def _chunk_text(text: str, source: str) -> list[dict]:
    """Split text into chunks with metadata."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        length_function=len,
    )
    raw_chunks = splitter.split_text(text)
    total = len(raw_chunks)
    return [
        {
            "text": chunk,
            "source": source,
            "chunk_index": i,
            "total_chunks": total,
        }
        for i, chunk in enumerate(raw_chunks)
    ]


# ── ChromaDB helpers ──────────────────────────────────────────────────

def _get_chroma_collection():
    """Connect to local ChromaDB and return the rag_documents collection."""
    try:
        client = chromadb.PersistentClient(path=settings.chroma_db_path)
        collection = client.get_or_create_collection(
            name="rag_documents",
            metadata={"hnsw:space": "cosine"},
        )
        return collection
    except Exception as e:
        logger.error(
            "ChromaDB failed to initialize at %s. Error: %s",
            settings.chroma_db_path, e
        )
        raise


def _chunk_id(source: str, chunk_index: int) -> str:
    """Generate a deterministic ID for a chunk."""
    raw = f"{source}::{chunk_index}"
    return hashlib.md5(raw.encode()).hexdigest()


# ── Main ingestion logic ──────────────────────────────────────────────

def ingest_directory(data_dir: str) -> dict:
    """
    Ingest all supported files from data_dir.
    Returns summary dict: {files_processed, chunks_created, time_taken}.
    """
    start = time.time()
    data_path = Path(data_dir)

    if not data_path.exists():
        logger.error("Data directory does not exist: %s", data_dir)
        return {"files_processed": 0, "chunks_created": 0, "time_taken": 0.0}

    # Gather supported files
    files = [
        f for f in data_path.iterdir()
        if f.is_file() and f.suffix.lower() in LOADERS
    ]

    if not files:
        logger.warning("No supported files found in %s", data_dir)
        return {"files_processed": 0, "chunks_created": 0, "time_taken": 0.0}

    # Chunk all files
    all_chunks: list[dict] = []
    for filepath in files:
        try:
            loader = LOADERS[filepath.suffix.lower()]
            text = loader(str(filepath))
            if not text.strip():
                logger.warning("Empty file skipped: %s", filepath.name)
                continue
            chunks = _chunk_text(text, filepath.name)
            all_chunks.extend(chunks)
            logger.info("Chunked %s → %d chunks", filepath.name, len(chunks))
        except Exception as e:
            logger.error("Failed to process %s: %s", filepath.name, e)

    if not all_chunks:
        logger.warning("No chunks created from any file.")
        return {"files_processed": len(files), "chunks_created": 0, "time_taken": time.time() - start}

    # Embed all chunks
    embedder = _get_embedder()
    texts = [c["text"] for c in all_chunks]
    logger.info("Embedding %d chunks in batches of 64...", len(texts))
    embeddings = embedder.encode(texts, batch_size=64, show_progress_bar=True)

    # Upsert into ChromaDB
    collection = _get_chroma_collection()
    batch_size = 100
    for i in range(0, len(all_chunks), batch_size):
        batch_end = min(i + batch_size, len(all_chunks))
        batch_chunks = all_chunks[i:batch_end]
        batch_embeddings = embeddings[i:batch_end].tolist()
        batch_ids = [_chunk_id(c["source"], c["chunk_index"]) for c in batch_chunks]
        batch_documents = [c["text"] for c in batch_chunks]
        batch_metadatas = [
            {"source": c["source"], "chunk_index": c["chunk_index"], "total_chunks": c["total_chunks"]}
            for c in batch_chunks
        ]
        collection.upsert(
            ids=batch_ids,
            documents=batch_documents,
            embeddings=batch_embeddings,
            metadatas=batch_metadatas,
        )
    logger.info("Upserted %d chunks into ChromaDB.", len(all_chunks))

    # Build BM25 index
    tokenised_corpus = [text.lower().split() for text in texts]
    bm25_index = BM25Okapi(tokenised_corpus)
    bm25_data = {
        "bm25": bm25_index,
        "tokenised_corpus": tokenised_corpus,
        "chunk_texts": texts,
        "chunk_metadatas": [
            {"source": c["source"], "chunk_index": c["chunk_index"]}
            for c in all_chunks
        ],
    }
    bm25_path = os.path.join(data_dir, "bm25_index.pkl")
    with open(bm25_path, "wb") as f:
        pickle.dump(bm25_data, f)
    logger.info("BM25 index saved to %s", bm25_path)

    elapsed = time.time() - start
    summary = {
        "files_processed": len(files),
        "chunks_created": len(all_chunks),
        "time_taken": round(elapsed, 2),
    }
    logger.info(
        "Ingestion complete: %d files, %d chunks, %.2fs",
        summary["files_processed"], summary["chunks_created"], summary["time_taken"],
    )
    return summary


# ── CLI entry point ───────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Ingest documents into the RAG system.")
    parser.add_argument("--data-dir", type=str, default=settings.data_path, help="Directory containing documents.")
    args = parser.parse_args()

    summary = ingest_directory(args.data_dir)
    print(f"\n{'='*50}")
    print(f"  Ingestion Summary")
    print(f"{'='*50}")
    print(f"  Files processed : {summary['files_processed']}")
    print(f"  Chunks created  : {summary['chunks_created']}")
    print(f"  Time taken      : {summary['time_taken']}s")
    print(f"{'='*50}\n")


if __name__ == "__main__":
    main()
