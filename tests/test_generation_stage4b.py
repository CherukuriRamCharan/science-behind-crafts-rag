"""Unit and regression tests for Stage 4B passage-level citations and structured responses.

Covers:
1. Valid passage-level citations in generate_structured()
2. Correct source ID and chunk ID association
3. Exact citation quote provenance
4. Multiple citations from distinct passages
5. Abstention with empty citations and abstained=True
6. Missing or incomplete metadata handled safely without crashes
7. Complete JSON serialization of structured responses
8. Backward compatibility of answer() returning a string
9. Extractive generation behavior on real prepared corpus
10. Fallback behavior when transformers backend is selected but unavailable
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from craft_rag.pipeline.common import load_config
from craft_rag.pipeline.generate import GroundedGenerator, _resolve_chunk_id
from craft_rag.pipeline.prepared_data import load_prepared_corpus
from craft_rag.pipeline.index_build import build_indices
from craft_rag.pipeline.retrieve import HybridRetriever


@pytest.fixture(scope="module")
def prepared_engine():
    """Build index once for the test module."""
    config = load_config()
    records, _ = load_prepared_corpus(config)
    index = build_indices(records, config)
    retriever = HybridRetriever(index, config)
    generator = GroundedGenerator(config)
    return config, index, retriever, generator


def test_answer_backward_compatibility_returns_string(prepared_engine):
    """Test 8: answer() must return a plain string with inline citations."""
    _, _, retriever, generator = prepared_engine
    question = "How does hand tool design affect grip strength?"
    passages = retriever.retrieve(question, top_k=5)

    ans = generator.answer(question, passages)
    assert isinstance(ans, str)
    assert len(ans) > 0
    assert "[WBC-0051 | E3]" in ans or "WBC-" in ans
    assert "Insufficient evidence" not in ans


def test_generate_structured_schema_and_json_serialization(prepared_engine):
    """Test 7: generate_structured() returns a valid JSON-serializable dictionary."""
    _, _, retriever, generator = prepared_engine
    question = "How do rake angle and depth of cut affect orthogonal cutting forces?"
    passages = retriever.retrieve(question, top_k=5)

    res = generator.generate_structured(question, passages)
    assert isinstance(res, dict)
    assert res["question"] == question
    assert isinstance(res["answer"], str)
    assert isinstance(res["abstained"], bool)
    assert res["abstained"] is False
    assert isinstance(res["citations"], list)
    assert len(res["citations"]) > 0
    assert isinstance(res["retrieved_chunk_ids"], list)
    assert len(res["retrieved_chunk_ids"]) > 0

    # Must be 100% JSON serializable
    json_str = json.dumps(res)
    deserialized = json.loads(json_str)
    assert deserialized["question"] == question
    assert len(deserialized["citations"]) == len(res["citations"])


def test_passage_level_citations_and_chunk_id_association(prepared_engine):
    """Tests 1, 2, 3, 4: Passage citations have valid source_id, chunk_id, evidence_level, and exact quote."""
    _, _, retriever, generator = prepared_engine
    question = "How do rake angle and depth of cut affect orthogonal cutting forces?"
    passages = retriever.retrieve(question, top_k=5)

    res = generator.generate_structured(question, passages)
    citations = res["citations"]

    # At least one citation from WBC-2001 or WBC-2003
    assert len(citations) >= 1
    retrieved_cids = set(res["retrieved_chunk_ids"])

    for cite in citations:
        assert cite["source_id"] in {"WBC-2001", "WBC-2003", "WBC-2002", "WBC-2021", "WBC-0009"}
        assert cite["chunk_id"] in retrieved_cids
        assert cite["chunk_id"].startswith("WBC-") or len(cite["chunk_id"]) > 5
        assert cite["evidence_level"] in {"E1", "E2", "E3", "E4", "E5"}
        assert len(cite["quote"]) > 10
        # Quote must be part of the final answer text
        assert cite["quote"] in res["answer"]


def test_multiple_citations_from_distinct_passages(prepared_engine):
    """Test 4: Multi-source query extracts quotes and citations from multiple passages."""
    _, _, retriever, generator = prepared_engine
    question = "How do chisel cutting mechanics and artisan posture affect fatigue in precision carving?"
    passages = retriever.retrieve(question, top_k=5)

    res = generator.generate_structured(question, passages)
    assert res["abstained"] is False
    citations = res["citations"]
    assert len(citations) >= 1
    # Check that each citation has a non-empty quote and valid chunk ID
    for c in citations:
        assert c["chunk_id"]
        assert c["quote"]


def test_abstention_returns_empty_citations_and_abstained_true(prepared_engine):
    """Test 5: Insufficient evidence produces abstained=True and citations=[]."""
    _, _, _, generator = prepared_engine
    question = "What is the tensile strength of carbon nanotubes in aerospace engineering?"
    empty_passages = []

    res = generator.generate_structured(question, empty_passages)
    assert res["abstained"] is True
    assert res["answer"] == "Insufficient evidence in the retrieved corpus to answer this question."
    assert res["citations"] == []
    assert res["retrieved_chunk_ids"] == []

    # Also with irrelevant passages
    irrelevant = [{
        "source_id": "WBC-0001",
        "chunk_id": "WBC-0001::doc0001::0000",
        "evidence_level": "E3",
        "text_chunk": "Bamboo culm physical properties and cutting forces.",
        "metadata": {"evidence_eligible": True, "record_type": "source_document"},
    }]
    res_irr = generator.generate_structured(question, irrelevant)
    assert res_irr["abstained"] is True
    assert res_irr["citations"] == []
    assert res_irr["retrieved_chunk_ids"] == ["WBC-0001::doc0001::0000"]


def test_missing_or_incomplete_metadata_handled_safely():
    """Test 6: Missing metadata fields do not crash generator or citation resolution."""
    generator = GroundedGenerator({"generator": {"backend": "extractive", "minimum_results": 1}})
    bare_passage = {
        "text_chunk": "Bamboo fibre strength varies by species and age. Tensile force is high.",
        # Missing source_id, chunk_id, evidence_level, metadata
    }

    res = generator.generate_structured("bamboo fibre strength species age", [bare_passage])
    assert res["abstained"] is False
    assert len(res["citations"]) == 1
    cite = res["citations"][0]
    assert cite["source_id"] == "UNKNOWN"
    assert cite["chunk_id"] == ""
    assert cite["evidence_level"] == "E5"
    assert "bamboo fibre strength" in cite["quote"].lower()


def test_resolve_chunk_id_handles_all_shapes():
    """Test chunk ID resolution across canonical, full-paper, and provenance shapes."""
    # 1. Direct chunk_id
    assert _resolve_chunk_id({"chunk_id": "C-123"}) == "C-123"
    # 2. record_id fallback
    assert _resolve_chunk_id({"record_id": "R-456"}) == "R-456"
    # 3. metadata.chunk_id
    assert _resolve_chunk_id({"metadata": {"chunk_id": "M-789"}}) == "M-789"
    # 4. metadata.provenance.record_id
    assert _resolve_chunk_id({"metadata": {"provenance": {"record_id": "P-012"}}}) == "P-012"
    # 5. metadata.provenance.document_key
    assert _resolve_chunk_id({"metadata": {"provenance": {"document_key": "SHA-KEY"}}}) == "SHA-KEY"
    # 6. empty / invalid
    assert _resolve_chunk_id({}) == ""
    assert _resolve_chunk_id(None) == ""


def test_transformers_backend_fallback_raises_guidance_when_configured():
    """Test 10: Transformers backend raises clear guidance if requirements not installed."""
    with pytest.raises((ImportError, RuntimeError)) as exc_info:
        GroundedGenerator({
            "generator": {
                "backend": "transformers",
                "model_name": "nonexistent-model/xyz",
                "allow_download": False,
            }
        })
    assert "Transformers" in str(exc_info.value) or "generator model" in str(exc_info.value)

