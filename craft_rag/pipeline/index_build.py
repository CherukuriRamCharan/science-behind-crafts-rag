"""Dense/sparse index and Chroma metadata store adapters."""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
from rank_bm25 import BM25Okapi

from .common import is_evidence_eligible, tokens, write_json
from .chunk import chunk_records
from .embed_text import TextEmbedder


def _searchable_text(row):
    metadata = row.get("metadata", {})
    values = [
        row.get("text_chunk", ""),
        metadata.get("title", row.get("title", "")),
        metadata.get("relevance_note", ""),
        metadata.get("questions_answered", []),
        row.get("why_theme", ""),
        row.get("craft_cluster", ""),
        row.get("process_stage", ""),
        row.get("material", ""),
    ]
    return " ".join(
        str(value)
        for value in values
        if value and not isinstance(value, dict)
    )


class VectorStore:
    """Small local Chroma adapter; callers depend on this boundary."""

    def __init__(self, path, collection):
        self.client = None
        self.collection = None
        try:
            import chromadb

            self.client = chromadb.PersistentClient(path=str(path))
            self.collection = self.client.get_or_create_collection(
                collection,
                metadata={"hnsw:space": "cosine"},
            )
        except ImportError as exc:
            logging.warning("ChromaDB is not installed; metadata remains available in the JSON index.")

    def upsert(self, chunks, vectors):
        if self.collection is None:
            return
        for start in range(0, len(chunks), 256):
            rows = chunks[start:start + 256]
            vector_batch = vectors[start:start + 256]
            metadata_batch = []
            for r in rows:
                merged = {
                    key: value for key, value in r.items()
                    if key not in {"text_chunk", "metadata"} and value is not None
                }
                merged.update(r.get("metadata", {}))
                metadata_batch.append({
                    key: (
                        ", ".join(str(item) for item in value)
                        if isinstance(value, list)
                        else value if isinstance(value, (str, int, float, bool))
                        else str(value)
                    )
                    for key, value in merged.items()
                    if value is not None
                })
            self.collection.upsert(
                ids=[r["chunk_id"] for r in rows],
                documents=[r["text_chunk"] for r in rows],
                embeddings=vector_batch.tolist(),
                metadatas=metadata_batch,
            )

    def replace(self, chunks, vectors):
        """Replace the active corpus so deleted/renamed records cannot stay searchable."""
        if self.collection is not None:
            while True:
                ids = self.collection.get(limit=100000, include=[]).get("ids", [])
                if not ids:
                    break
                self.collection.delete(ids=ids)
        self.upsert(chunks, vectors)

    def query(self, query_vector, top_k=5, where=None):
        """Return Chroma documents and metadata using the shared text/image schema."""
        if self.collection is None:
            return {"ids": [], "documents": [], "metadatas": []}
        return self.collection.query(
            query_embeddings=[query_vector.tolist()],
            n_results=top_k,
            where=where,
            include=["documents", "metadatas", "distances"],
        )

# Pinecone adapter stub: implement VectorStore.upsert/query with the same chunk_id,
# text_chunk, vector, and metadata fields. Keep credentials in environment variables.
# class PineconeVectorStore(VectorStore): ...

def build_indices(records, config):
    chunk_config = config["chunking"]
    chunks = chunk_records(
        records,
        chunk_config["words"],
        chunk_config["overlap"],
        chunk_config.get("max_chars", 1200),
    )
    if not chunks:
        raise ValueError("No non-empty passages to index.")
    all_chunks = chunks
    if not config["retrieval"].get("include_non_evidence", False):
        chunks = [row for row in all_chunks if is_evidence_eligible(row)]
    if not chunks:
        raise ValueError("No evidence-eligible passages are available for retrieval.")
    embedder = TextEmbedder(config)
    search_texts = [_searchable_text(row) for row in all_chunks]
    all_vectors = embedder.encode(search_texts, input_type="document")
    if all_vectors.ndim != 2 or all_vectors.shape[0] != len(all_chunks):
        raise ValueError(f"Embedding backend returned an invalid matrix shape: {all_vectors.shape}")
    if not np.isfinite(all_vectors).all():
        raise ValueError("Embedding backend returned NaN or infinite values.")
    searchable_ids = {row["chunk_id"] for row in chunks}
    searchable_positions = [
        position for position, row in enumerate(all_chunks)
        if row["chunk_id"] in searchable_ids
    ]
    vectors = all_vectors[searchable_positions]
    index_config = config["index"]
    artifacts = Path(config["data"].get("artifact_dir", "artifacts"))
    artifacts.mkdir(parents=True, exist_ok=True)
    faiss_index = None
    try:
        import faiss
    except ImportError as exc:
        logging.warning("FAISS is not installed; using exact NumPy vector ranking: %s", exc)
    else:
        faiss_index = faiss.IndexFlatIP(vectors.shape[1])
        faiss_index.add(vectors)

    tokenized = [tokens(_searchable_text(row)) for row in chunks]
    bm25 = BM25Okapi(tokenized) if tokenized and any(tokenized) else None
    metadata_path = Path(index_config.get("metadata_path", artifacts / "chunks.json"))
    faiss_path = Path(index_config.get("faiss_path", artifacts / "dense.faiss"))
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    write_json(metadata_path, all_chunks)
    if faiss_index is not None:
        faiss_path.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(faiss_index, str(faiss_path))
    store = VectorStore(
        index_config.get("chroma_path", artifacts / "chroma"),
        index_config.get("chroma_collection", "sbc_text"),
    )
    store.replace(all_chunks, all_vectors)
    return {
        "chunks": chunks,
        "vectors": vectors,
        "all_chunks": all_chunks,
        "all_vectors": all_vectors,
        "faiss": faiss_index,
        "bm25": bm25,
        "tokenized": tokenized,
        "embedder": embedder,
        "store": store,
        "backend_info": {
            "dense": "faiss" if faiss_index is not None else "numpy exact fallback",
            "sparse": "rank_bm25" if bm25 is not None else "python overlap fallback",
            "vector_store": "chroma" if store.collection is not None else "JSON metadata only",
            "embedding": embedder.backend,
        },
    }
