"""Hybrid dense/BM25 retrieval with Reciprocal Rank Fusion.

Behaviour (findings H3 and M35):
* filters are applied *before* ranking (mask, then rank), so a filter never starves the result list;
* filters compare normalised facets (see ``facets.py``), not free text;
* de-duplication is per ``chunk_id``; ``max_per_source`` caps how many chunks one source may add;
* ranking is deterministic: every tie is broken by chunk position / ``chunk_id``, so results do not
  depend on FAISS vs NumPy or on the platform.
"""
from __future__ import annotations

from collections import Counter

import numpy as np

from .common import tokens
from .facets import facet_value, normalize_filter_value

DEFAULT_MAX_PER_SOURCE = 2
_UNSET = object()  # lets callers pass max_per_source=None to mean "no cap"


def rrf_fuse(dense_ranked, sparse_ranked, rrf_k=60, dense_weight=1.0, sparse_weight=1.0):
    """Fuse two ranked lists by rank. Returns ``[(chunk_id, score)]`` sorted by (-score, chunk_id)."""
    scores = {}
    for weight, ranked in ((dense_weight, dense_ranked), (sparse_weight, sparse_ranked)):
        for rank, item in enumerate(ranked, 1):
            key = item["chunk_id"]
            scores[key] = scores.get(key, 0.0) + weight / (rrf_k + rank)
    return sorted(scores.items(), key=lambda pair: (-pair[1], pair[0]))


def _ranked_positions(scores: np.ndarray, tie_break: np.ndarray, depth: int) -> np.ndarray:
    """Positions of the best ``depth`` scores; exact ties are broken by ``chunk_id`` order (M35)."""
    order = np.lexsort((tie_break, -scores))  # last key is primary
    return order[:depth]


class HybridRetriever:
    def __init__(self, index, config):
        self.index = index
        self.cfg = config["retrieval"]

    def _allowed_mask(self, chunks, filters):
        allowed = np.ones(len(chunks), dtype=bool)
        for key, value in (filters or {}).items():
            if value in (None, ""):
                continue
            wanted = normalize_filter_value(key, value)
            allowed &= np.fromiter(
                (facet_value(row, key) == wanted for row in chunks), dtype=bool, count=len(chunks)
            )
        return allowed

    def retrieve(self, query, top_k=None, filters=None, max_per_source=_UNSET):
        if top_k is not None and top_k <= 0:
            raise ValueError("top_k must be a positive integer.")
        if not query.strip():
            return []
        chunks = self.index["chunks"]
        if not chunks:
            return []
        limit = top_k if top_k is not None else self.cfg.get("top_k", 5)
        if limit <= 0:
            raise ValueError("Configured retrieval top_k must be a positive integer.")
        if max_per_source is _UNSET:
            max_per_source = self.cfg.get("max_per_source", DEFAULT_MAX_PER_SOURCE)
        if max_per_source is not None and max_per_source <= 0:
            raise ValueError("max_per_source must be a positive integer or None.")

        allowed = self._allowed_mask(chunks, filters)
        pool = int(allowed.sum())
        if pool == 0:
            return []
        depth = min(pool, max(limit * 10, self.cfg.get("candidate_k", 20)))

        # Exact scoring over every chunk, then mask. (FAISS cannot mask, and exact scoring keeps
        # tie order independent of the vector-search backend.)
        query_vector = self.index["embedder"].encode([query])[0]
        dense_scores = np.asarray(self.index["vectors"] @ query_vector, dtype=np.float64)
        query_tokens = tokens(query)
        if self.index["bm25"] is not None:
            sparse_scores = np.asarray(self.index["bm25"].get_scores(query_tokens), dtype=np.float64)
        else:
            term_set = set(query_tokens)
            sparse_scores = np.asarray(
                [len(term_set & set(document)) for document in self.index["tokenized"]], dtype=np.float64
            )
        dense_scores[~allowed] = -np.inf
        sparse_scores[~allowed] = -np.inf
        # rank of each chunk's id in sorted-id order: a tie-break that ignores corpus order
        id_order = np.argsort(np.argsort(np.asarray([row["chunk_id"] for row in chunks], dtype=object), kind="stable"), kind="stable")
        dense = [chunks[int(i)] for i in _ranked_positions(dense_scores, id_order, depth)]
        sparse = [chunks[int(i)] for i in _ranked_positions(sparse_scores, id_order, depth)]
        dense_rank = {row["chunk_id"]: rank for rank, row in enumerate(dense, 1)}
        sparse_rank = {row["chunk_id"]: rank for rank, row in enumerate(sparse, 1)}

        fused = rrf_fuse(
            dense,
            sparse,
            self.cfg.get("rrf_k", 60),
            self.cfg.get("dense_weight", 1.0),
            self.cfg.get("sparse_weight", 1.0),
        )
        by_id = {row["chunk_id"]: row for row in chunks}
        evidence_boost = self.cfg.get("evidence_boost", {})
        grade_boost = self.cfg.get("grade_boost", {})
        scored = []
        for chunk_id, score in fused:
            row = dict(by_id[chunk_id])
            adjustment = (
                evidence_boost.get(row.get("evidence_level"), 0)
                + grade_boost.get(row.get("grade"), 0)
            )
            row["retrieval_score"] = score + adjustment
            row["rrf_score"] = score
            row["dense_rank"] = dense_rank.get(chunk_id)
            row["sparse_rank"] = sparse_rank.get(chunk_id)
            scored.append(row)
        # Reranker hook: load a cross-encoder here when enabled; left inert to avoid checkpoint downloads.
        scored.sort(key=lambda row: (-row["retrieval_score"], row["chunk_id"]))

        floor = self.cfg.get("minimum_score", 0.0)
        selected, per_source = [], Counter()
        for row in scored:
            if row["retrieval_score"] < floor:
                continue
            source = str(row["source_id"]).casefold()
            if max_per_source is not None and per_source[source] >= max_per_source:
                continue
            per_source[source] += 1
            selected.append(row)
            if len(selected) >= limit:
                break
        return selected
