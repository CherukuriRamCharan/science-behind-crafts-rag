"""FAISS semantic retrieval over sentence embeddings (cosine similarity)."""
from __future__ import annotations

from typing import Protocol, Sequence

import faiss
import numpy as np

from .data_loader import Chunk, RetrievedChunk

DEFAULT_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"  # ~22M params, CPU-friendly


class Embedder(Protocol):
    """Anything with ``encode(list_of_texts) -> 2-D float array`` works (also used for test fakes)."""

    def encode(self, texts: Sequence[str]) -> np.ndarray: ...


class SentenceTransformerEmbedder:
    """Wrapper around ``sentence-transformers``; the model downloads on first use."""

    def __init__(self, model_name: str = DEFAULT_MODEL_NAME) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:  # keep a clear message instead of a bare ImportError
            raise ImportError(
                "sentence-transformers is required for semantic search. "
                "Install it with: pip install -r requirements-retrieval.txt"
            ) from exc
        self.model_name = model_name
        self._model = SentenceTransformer(model_name)

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        return np.asarray(self._model.encode(list(texts), show_progress_bar=False), dtype=np.float32)


def _normalize(vectors: np.ndarray) -> np.ndarray:
    """L2-normalise rows so inner product == cosine similarity (same for index and query)."""
    vectors = np.ascontiguousarray(vectors, dtype=np.float32)
    if vectors.ndim != 2:
        raise ValueError(f"Embeddings must be 2-D, got shape {vectors.shape}")
    if not np.isfinite(vectors).all():
        raise ValueError("Embeddings contain NaN or infinite values")
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / np.where(norms == 0, 1.0, norms)


class FaissRetriever:
    """Embed all chunks once, keep a FAISS inner-product index, and search it.

    FAISS row ``i`` always corresponds to ``chunks[i]`` (same order as given).
    """

    def __init__(self, chunks: Sequence[Chunk], embedder: Embedder | None = None,
                 model_name: str = DEFAULT_MODEL_NAME) -> None:
        self._chunks = list(chunks)
        self._index = None
        if not self._chunks:
            self._embedder = embedder  # nothing to index; avoid loading a model
            return
        self._embedder = embedder or SentenceTransformerEmbedder(model_name)
        vectors = _normalize(self._embedder.encode([c.text for c in self._chunks]))
        if vectors.shape[0] != len(self._chunks):
            raise ValueError(
                f"Embedder returned {vectors.shape[0]} vectors for {len(self._chunks)} chunks"
            )
        self._index = faiss.IndexFlatIP(vectors.shape[1])  # exact search; fine for small corpora
        self._index.add(vectors)

    def __len__(self) -> int:
        return len(self._chunks)

    @property
    def chunks(self) -> list[Chunk]:
        return list(self._chunks)

    def search(self, query: str, top_k: int = 5) -> list[RetrievedChunk]:
        """Return the ``top_k`` most similar chunks (cosine similarity, higher is better)."""
        if top_k <= 0:
            raise ValueError("top_k must be a positive integer")
        if not query.strip() or self._index is None:
            return []
        query_vec = _normalize(self._embedder.encode([query]))
        if query_vec.shape != (1, self._index.d):
            raise ValueError(
                f"Query embedding shape {query_vec.shape} does not match index dimension {self._index.d}"
            )
        scores, positions = self._index.search(query_vec, min(top_k, len(self._chunks)))
        results: list[RetrievedChunk] = []
        for rank, (score, pos) in enumerate(zip(scores[0], positions[0]), start=1):
            if pos < 0:  # FAISS pads with -1 when fewer results exist
                continue
            results.append(
                RetrievedChunk.from_chunk(self._chunks[pos], faiss_rank=rank, faiss_score=float(score))
            )
        return results
