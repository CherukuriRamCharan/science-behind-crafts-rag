"""Run a sample search: ``python -m src.retrieval.demo "Why does bamboo age matter?"``."""
from __future__ import annotations

import sys

from .bm25_retriever import BM25Retriever
from .data_loader import load_chunks
from .hybrid_retriever import HybridRetriever


def main() -> None:
    query = " ".join(sys.argv[1:]) or "Why does bamboo age matter for material selection?"
    chunks = load_chunks()
    try:
        retriever = HybridRetriever.from_chunks(chunks)
        mode = "hybrid (BM25 + FAISS)"
    except ImportError as exc:
        print(f"[semantic search unavailable: {exc}]\n[falling back to BM25 only]\n")
        retriever, mode = BM25Retriever(chunks), "BM25 only"
    print(f"Mode: {mode} | Query: {query}\n")
    for result in retriever.search(query, top_k=5):
        print(f"{result.chunk_id} | source={result.source_id} | page={result.page} | "
              f"bm25={result.bm25_rank} faiss={result.faiss_rank} hybrid={result.hybrid_score}")
        print(f"  {result.title}\n  {result.text[:200]}...\n")


if __name__ == "__main__":
    main()
