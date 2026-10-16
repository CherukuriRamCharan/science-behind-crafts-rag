"""Hybrid retrieval: BM25 + FAISS merged with Reciprocal Rank Fusion (RRF)."""
from __future__ import annotations

from pathlib import Path
from typing import Sequence, Union

from .bm25_retriever import BM25Retriever
from .data_loader import Chunk, RetrievedChunk, load_chunks
from .faiss_retriever import DEFAULT_MODEL_NAME, Embedder, FaissRetriever

RRF_K = 60  # standard RRF constant (Cormack et al., 2009)


class HybridRetriever:
    """Combine two ranked lists by rank, not by raw score.

    RRF score of a chunk = sum over retrievers of ``1 / (rrf_k + rank)``.
    BM25 scores and cosine scores live on different scales, so only ranks are fused.
    """

    def __init__(self, bm25: BM25Retriever, faiss_retriever: FaissRetriever,
                 rrf_k: int = RRF_K, candidate_k: int = 50) -> None:
        ids_bm25 = [c.chunk_id for c in bm25.chunks]
        ids_faiss = [c.chunk_id for c in faiss_retriever.chunks]
        if ids_bm25 != ids_faiss:
            raise ValueError("BM25 and FAISS retrievers must be built from the same chunks in the same order")
        self.bm25 = bm25
        self.faiss = faiss_retriever
        self.rrf_k = rrf_k
        self.candidate_k = candidate_k

    @classmethod
    def from_chunks(cls, chunks: Sequence[Chunk], embedder: Embedder | None = None,
                    model_name: str = DEFAULT_MODEL_NAME, **kwargs) -> "HybridRetriever":
        return cls(BM25Retriever(chunks), FaissRetriever(chunks, embedder, model_name), **kwargs)

    @classmethod
    def from_jsonl(cls, path: Union[str, Path, None] = None, embedder: Embedder | None = None,
                   model_name: str = DEFAULT_MODEL_NAME, **kwargs) -> "HybridRetriever":
        """Load chunks (default path / ``RETRIEVAL_CHUNKS_PATH`` / explicit) and build both indexes."""
        return cls.from_chunks(load_chunks(path), embedder, model_name, **kwargs)

    def search(self, query: str, top_k: int = 5) -> list[RetrievedChunk]:
        """Return the top ``top_k`` unique chunks with BM25, FAISS and hybrid info attached."""
        if top_k <= 0:
            raise ValueError("top_k must be a positive integer")
        depth = max(self.candidate_k, top_k)
        merged: dict[str, RetrievedChunk] = {}

        for result in self.bm25.search(query, depth):
            merged[result.chunk_id] = result
            result.hybrid_score = 1.0 / (self.rrf_k + result.bm25_rank)
        for result in self.faiss.search(query, depth):
            existing = merged.get(result.chunk_id)
            if existing is None:
                merged[result.chunk_id] = result
                result.hybrid_score = 1.0 / (self.rrf_k + result.faiss_rank)
            else:  # same chunk found by both: one record, scores added
                existing.faiss_rank = result.faiss_rank
                existing.faiss_score = result.faiss_score
                existing.hybrid_score += 1.0 / (self.rrf_k + result.faiss_rank)

        ranked = sorted(merged.values(), key=lambda r: (-r.hybrid_score, r.chunk_id))
        return ranked[:top_k]
