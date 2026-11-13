"""Regression tests for Stage 4A retrieval improvements.

Covers:
- Configurable per-source passage limits (max_per_source: 1, 2, 3)
- Multiple relevant chunks from one paper
- Multiple distinct relevant sources
- Fewer than five eligible sources in the corpus or query
- Duplicate passage deduplication (identical chunk_id or text)
- Preservation of required fields (source_id, chunk_id, record_id, text_chunk)
- Ranking score preservation
"""
from __future__ import annotations

import sys
from pathlib import Path
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from craft_rag.pipeline.common import load_config
from craft_rag.pipeline.prepared_data import load_prepared_corpus
from craft_rag.pipeline.index_build import build_indices
from craft_rag.pipeline.retrieve import HybridRetriever


@pytest.fixture(scope="module")
def prepared_engine():
    """Build index once for the test module."""
    config = load_config()
    records, manifest = load_prepared_corpus(config)
    index = build_indices(records, config)
    retriever = HybridRetriever(index, config)
    return config, index, retriever


def test_required_fields_present_in_all_retrieved_passages(prepared_engine):
    _, _, retriever = prepared_engine
    results = retriever.retrieve("hand tool design grip strength", top_k=5)

    assert len(results) > 0
    for r in results:
        assert "source_id" in r and isinstance(r["source_id"], str) and r["source_id"]
        assert "chunk_id" in r and isinstance(r["chunk_id"], str) and r["chunk_id"]
        assert "record_id" in r and isinstance(r["record_id"], str) and r["record_id"]
        assert "text_chunk" in r and isinstance(r["text_chunk"], str) and r["text_chunk"]
        assert "retrieval_score" in r and isinstance(r["retrieval_score"], (int, float))
        assert "metadata" in r and isinstance(r["metadata"], dict)


def test_retrieval_ranking_is_strictly_non_increasing(prepared_engine):
    _, _, retriever = prepared_engine
    results = retriever.retrieve("cutting mechanics rake angle wood grain", top_k=5)

    assert len(results) > 1
    scores = [r["retrieval_score"] for r in results]
    for i in range(len(scores) - 1):
        assert scores[i] >= scores[i + 1]


def test_multiple_relevant_chunks_from_one_paper_with_default_max_per_source(prepared_engine):
    _, _, retriever = prepared_engine
    # WBC-0051 has 97 chunks all discussing hand tool design, grip strength, and handle diameter
    results = retriever.retrieve("hand tool handle diameter grip strength span", top_k=5, max_per_source=2)

    wbc51_chunks = [r for r in results if r["source_id"] == "WBC-0051"]
    # With max_per_source=2, WBC-0051 can now return 2 chunks instead of being locked to 1
    assert len(wbc51_chunks) == 2
    # Ensure they are distinct chunks from that paper
    assert wbc51_chunks[0]["chunk_id"] != wbc51_chunks[1]["chunk_id"]
    assert wbc51_chunks[0]["text_chunk"] != wbc51_chunks[1]["text_chunk"]


def test_max_per_source_limits_respected_across_settings(prepared_engine):
    _, _, retriever = prepared_engine
    query = "hand tool handle diameter grip strength span"

    # Setting 1: max_per_source = 1 (legacy behavior)
    results_1 = retriever.retrieve(query, top_k=5, max_per_source=1)
    source_counts_1 = {}
    for r in results_1:
        source_counts_1[r["source_id"]] = source_counts_1.get(r["source_id"], 0) + 1
    assert all(count <= 1 for count in source_counts_1.values())

    # Setting 2: max_per_source = 2 (default)
    results_2 = retriever.retrieve(query, top_k=5, max_per_source=2)
    source_counts_2 = {}
    for r in results_2:
        source_counts_2[r["source_id"]] = source_counts_2.get(r["source_id"], 0) + 1
    assert all(count <= 2 for count in source_counts_2.values())

    # Setting 3: max_per_source = 3
    results_3 = retriever.retrieve(query, top_k=5, max_per_source=3)
    source_counts_3 = {}
    for r in results_3:
        source_counts_3[r["source_id"]] = source_counts_3.get(r["source_id"], 0) + 1
    assert all(count <= 3 for count in source_counts_3.values())


def test_multiple_relevant_sources_retrieved(prepared_engine):
    _, _, retriever = prepared_engine
    # Query matching multiple distinct research cards (WBC-2001, WBC-2003, WBC-0009)
    results = retriever.retrieve("orthogonal cutting mechanics rake angle grain direction", top_k=5)

    distinct_sources = {r["source_id"] for r in results}
    # Must retrieve multiple distinct papers/cards
    assert len(distinct_sources) >= 2


def test_fewer_than_five_eligible_sources_does_not_invent_sources(prepared_engine):
    _, _, retriever = prepared_engine
    # Highly specific query matching only 1 or 2 sources
    results = retriever.retrieve("SPSS Sawmill Workers Occupational Health Kerala", top_k=5)

    # Should return only the genuine matching passages, without fabricating to meet top_k=5
    assert 0 < len(results) <= 5
    # Distinct sources must not be artificially padded
    source_ids = {r["source_id"] for r in results}
    assert "WBC-0013" in source_ids


def test_synthetic_corpus_fewer_than_five_sources_and_duplicate_handling(tmp_path):
    """Test with a tiny synthetic index to strictly verify duplicates and < 5 sources."""
    from craft_rag.pipeline.retrieve import HybridRetriever

    # 3 chunks total: 2 from S1 (one exact text duplicate), 1 from S2
    sample_chunks = [
        {
            "chunk_id": "S1::doc0001::0000",
            "record_id": "S1::doc0001",
            "source_id": "S1",
            "title": "Paper 1",
            "text_chunk": "Unique passage about bamboo fibers",
            "metadata": {"title": "Paper 1", "record_type": "full_paper_chunk"},
            "retrieval_score": 0.9,
            "evidence_level": "E3",
        },
        {
            "chunk_id": "S1::doc0001::0001",
            "record_id": "S1::doc0001",
            "source_id": "S1",
            "title": "Paper 1",
            "text_chunk": "Unique passage about bamboo fibers",  # DUPLICATE TEXT
            "metadata": {"title": "Paper 1", "record_type": "full_paper_chunk"},
            "retrieval_score": 0.85,
            "evidence_level": "E3",
        },
        {
            "chunk_id": "S1::doc0001::0002",
            "record_id": "S1::doc0001",
            "source_id": "S1",
            "title": "Paper 1",
            "text_chunk": "Second distinct passage from Paper 1 about bamboo tensile strength",
            "metadata": {"title": "Paper 1", "record_type": "full_paper_chunk"},
            "retrieval_score": 0.80,
            "evidence_level": "E3",
        },
        {
            "chunk_id": "S2::doc0001::0000",
            "record_id": "S2::doc0001",
            "source_id": "S2",
            "title": "Paper 2",
            "text_chunk": "Distinct passage from Paper 2 about bamboo wood grain",
            "metadata": {"title": "Paper 2", "record_type": "full_paper_chunk"},
            "retrieval_score": 0.70,
            "evidence_level": "E3",
        },
    ]

    import numpy as np

    class FakeEmbedder:
        def encode(self, texts, input_type="document"):
            return np.ones((len(texts), 4), dtype="float32")

    fake_index = {
        "chunks": sample_chunks,
        "vectors": np.ones((len(sample_chunks), 4), dtype="float32"),
        "faiss": None,
        "bm25": None,
        "tokenized": [c["text_chunk"].split() for c in sample_chunks],
        "embedder": FakeEmbedder(),
        "backend_info": {"embedding": "hashing"},
    }

    config = {
        "retrieval": {
            "top_k": 5,
            "candidate_k": 10,
            "max_per_source": 2,
            "rrf_k": 60,
            "dense_weight": 1.0,
            "sparse_weight": 1.0,
            "minimum_score": 0.0,
        }
    }

    retriever = HybridRetriever(fake_index, config)
    results = retriever.retrieve("bamboo", top_k=5, max_per_source=2)

    # 1. Total results must be 3, because the duplicate passage was filtered out!
    # Even though top_k=5 was requested, it only returns the 3 genuine distinct passages.
    assert len(results) == 3
    # 2. Duplicate text was discarded
    texts = [r["text_chunk"] for r in results]
    assert len(texts) == len(set(texts))
    # 3. S1 has at most 2 passages (max_per_source=2)
    s1_results = [r for r in results if r["source_id"] == "S1"]
    assert len(s1_results) == 2
    # 4. S2 has 1 passage
    s2_results = [r for r in results if r["source_id"] == "S2"]
    assert len(s2_results) == 1
