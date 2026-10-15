"""Retrieval module: BM25, FAISS and hybrid (RRF) search over RAG chunks.

Typical use from the repository root::

    from src.retrieval import HybridRetriever
    retriever = HybridRetriever.from_jsonl()          # uses the default chunk file
    results = retriever.search("Why does bamboo age matter?", top_k=5)
"""
from .bm25_retriever import BM25Retriever, tokenize
from .data_loader import (
    DEFAULT_CHUNKS_PATH,
    Chunk,
    ChunkLoadError,
    RetrievedChunk,
    load_chunks,
    resolve_chunks_path,
)
from .faiss_retriever import Embedder, FaissRetriever, SentenceTransformerEmbedder
from .hybrid_retriever import HybridRetriever

__all__ = [
    "BM25Retriever",
    "Chunk",
    "ChunkLoadError",
    "DEFAULT_CHUNKS_PATH",
    "Embedder",
    "FaissRetriever",
    "HybridRetriever",
    "RetrievedChunk",
    "SentenceTransformerEmbedder",
    "load_chunks",
    "resolve_chunks_path",
    "tokenize",
]
