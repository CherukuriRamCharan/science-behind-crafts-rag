"""Hybrid FAISS/BM25 retrieval with Reciprocal Rank Fusion."""
from __future__ import annotations

from urllib.parse import urlsplit, urlunsplit

import numpy as np

from .common import tokens


def rrf_fuse(dense_ranked, sparse_ranked, rrf_k=60, dense_weight=1.0, sparse_weight=1.0):
    scores = {}
    for weight, ranked in ((dense_weight, dense_ranked), (sparse_weight, sparse_ranked)):
        for rank, item in enumerate(ranked, 1):
            key = item["chunk_id"]
            scores[key] = scores.get(key, 0.0) + weight / (rrf_k + rank)
    return sorted(scores.items(), key=lambda pair: (-pair[1], pair[0]))


class HybridRetriever:
    def __init__(self, index, config):
        self.index = index
        self.cfg = config["retrieval"]

    def retrieve(
        self,
        query: str,
        top_k: int | None = None,
        filters: dict | None = None,
        max_per_source: int | None = None,
    ) -> list[dict]:
        if top_k is not None and top_k <= 0:
            raise ValueError("top_k must be a positive integer.")
        if not query.strip():
            return []
        chunks = self.index["chunks"]
        if not chunks:
            return []
        vectors = self.index["vectors"]
        embedder = self.index["embedder"]
        limit = top_k if top_k is not None else self.cfg.get("top_k", 5)
        if limit <= 0:
            raise ValueError("Configured retrieval top_k must be a positive integer.")
        max_source_passages = (
            max_per_source
            if max_per_source is not None
            else self.cfg.get("max_per_source", 2)
        )
        candidate_count = min(
            len(chunks),
            max(limit * 4, self.cfg.get("candidate_k", 10)),
        )
        query_vector = embedder.encode([query], input_type="query")[0]
        similarities = vectors @ query_vector
        if self.index["faiss"] is not None:
            _, ids = self.index["faiss"].search(
                query_vector.reshape(1, -1),
                candidate_count,
            )
            dense_positions = [int(index) for index in ids[0] if index >= 0]
        else:
            dense_positions = np.argsort(-similarities, kind="stable")[:candidate_count].tolist()
        query_tokens = tokens(query)
        if self.index["bm25"] is not None:
            scores = self.index["bm25"].get_scores(query_tokens)
            positive_positions = np.flatnonzero(scores > 0)
            positions = positive_positions[
                np.argsort(-scores[positive_positions], kind="stable")[:candidate_count]
            ]
            sparse = [chunks[int(index)] for index in positions]
        else:
            term_set = set(query_tokens)
            overlaps = [
                sum(term in set(document) for term in term_set)
                for document in self.index["tokenized"]
            ]
            overlap_array = np.asarray(overlaps)
            positive_positions = np.flatnonzero(overlap_array > 0)
            positions = positive_positions[
                np.argsort(-overlap_array[positive_positions], kind="stable")[:candidate_count]
            ]
            sparse = [chunks[int(index)] for index in positions]
            scores = overlap_array
        semantic_backend = self.index["backend_info"]["embedding"] != "hashing"
        dense_threshold = self.cfg.get("minimum_dense_similarity", 0.25)
        dense = [
            chunks[index]
            for index in dense_positions
            if scores[index] > 0
            or (semantic_backend and similarities[index] >= dense_threshold)
        ]
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
        annotation_match_boost = self.cfg.get("annotation_match_boost", 0.0)
        query_terms = set(tokens(query))
        scored = []
        for chunk_id, score in fused:
            row = dict(by_id[chunk_id])
            metadata = row.get("metadata", {})
            annotation_text = " ".join([
                str(metadata.get("relevance_note", "")),
                " ".join(str(item) for item in metadata.get("questions_answered", [])),
            ])
            annotation_terms = set(tokens(annotation_text))
            annotation_coverage = (
                len(query_terms & annotation_terms) / len(query_terms)
                if query_terms else 0.0
            )
            annotation_match = (
                annotation_coverage * annotation_match_boost
                if annotation_coverage >= self.cfg.get("annotation_match_min_coverage", 0.5)
                else 0.0
            )
            adjustment = (
                evidence_boost.get(row.get("evidence_level"), 0)
                + grade_boost.get(row.get("grade"), 0)
                + annotation_match
            )
            row["retrieval_score"] = score + adjustment
            row["rrf_score"] = score
            row["annotation_match_score"] = annotation_match
            row["annotation_relevance"] = annotation_coverage
            scored.append(row)
        # Reranker hook: load a cross-encoder here when enabled; left inert to avoid checkpoint downloads.
        scored.sort(key=lambda x: x["retrieval_score"], reverse=True)
        floor = self.cfg.get("minimum_score", 0.0)
        selected: list[dict] = []
        source_counts: dict[tuple, int] = {}
        seen_chunks: set[str] = set()
        seen_texts: set[str] = set()

        for row in scored:
            if row["retrieval_score"] < floor:
                continue
            metadata = row.get("metadata", {})
            if filters and any(row.get(key, metadata.get(key)) != value for key, value in filters.items()):
                continue

            # Guarantee core identification and content fields
            row["source_id"] = str(row.get("source_id") or "UNKNOWN")
            row["chunk_id"] = str(row.get("chunk_id") or row.get("record_id") or "")
            row["record_id"] = str(row.get("record_id") or row.get("chunk_id") or "")
            row["text_chunk"] = str(row.get("text_chunk") or "")

            chunk_id = row["chunk_id"]
            if chunk_id in seen_chunks:
                continue

            # Filter exact duplicate passage text
            norm_text = " ".join(row["text_chunk"].split()).casefold()
            if norm_text and norm_text in seen_texts:
                continue

            title = " ".join(str(metadata.get("title", "")).split()).casefold()
            doi = str(row.get("doi") or metadata.get("doi") or "").strip().casefold()
            link = str(
                row.get("link")
                or row.get("source_url")
                or metadata.get("source_url")
                or ""
            ).strip()
            record_type = metadata.get("record_type", row.get("record_type", ""))
            if title and record_type in {"source_document", "full_paper_chunk"}:
                source_key = ("publication", title)
            elif doi:
                source_key = ("doi", doi.removeprefix("https://doi.org/"))
            elif link:
                parsed = urlsplit(link)
                source_key = (
                    "url",
                    urlunsplit((parsed.scheme.casefold(), parsed.netloc.casefold(), parsed.path.rstrip("/"), parsed.query, "")),
                )
            else:
                source_key = ("record", str(row["source_id"]).casefold(), title or row.get("record_id", ""))

            current_count = source_counts.get(source_key, 0)
            if max_source_passages is not None and max_source_passages > 0 and current_count >= max_source_passages:
                continue

            selected.append(row)
            seen_chunks.add(chunk_id)
            if norm_text:
                seen_texts.add(norm_text)
            source_counts[source_key] = current_count + 1

            if len(selected) >= limit:
                break
        return selected
