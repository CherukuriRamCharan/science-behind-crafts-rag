"""Tests for Stage 4D Evaluation Implementation & Metric Reporting.

Covers:
- Metric calculations with known synthetic examples
- Empty retrieval handling
- Fewer than K retrieved passages handling
- Duplicate retrieved chunk ID handling
- Missing ground-truth target handling
- Citation precision and recall calculations
- Abstention scoring
- End-to-end evaluation using the 28-query ground-truth benchmark
- Backward-compatible functions (retrieval_metrics, format_table)
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import pytest

from craft_rag.pipeline.common import load_config
from craft_rag.pipeline.evaluate import (
    compute_abstention_accuracy,
    compute_citation_metrics,
    compute_retrieval_metrics,
    evaluate_benchmark,
    format_table,
    retrieval_metrics,
)
from craft_rag.pipeline.generate import GroundedGenerator
from craft_rag.pipeline.index_build import build_indices
from craft_rag.pipeline.prepared_data import load_prepared_corpus
from craft_rag.pipeline.retrieve import HybridRetriever


def test_metric_calculations_known_synthetic_example() -> None:
    """Verify exact mathematical computations of P@K, R@K, MRR, and nDCG@K on known synthetic input."""
    # Given:
    # 4 targets: {T1, T2, T3, T4}
    # Retrieved ranking: [T1, NonRel, T2, NonRel, T3]
    # At K=1: [T1] -> 1 hit. P@1 = 1/1 = 1.0. R@1 = 1/4 = 0.25.
    # At K=3: [T1, NonRel, T2] -> 2 hits. P@3 = 2/3 = 0.6667. R@3 = 2/4 = 0.5.
    # At K=5: [T1, NonRel, T2, NonRel, T3] -> 3 hits. P@5 = 3/5 = 0.6. R@5 = 3/4 = 0.75.
    # MRR: first hit at rank 1 -> 1/1 = 1.0.
    targets = ["T1", "T2", "T3", "T4"]
    retrieved = ["T1", "N1", "T2", "N2", "T3"]

    m = compute_retrieval_metrics(retrieved, targets, ks=(1, 3, 5))

    assert m["precision@1"] == 1.0
    assert m["recall@1"] == 0.25
    assert m["precision@3"] == pytest.approx(2 / 3, abs=1e-4)
    assert m["recall@3"] == 0.5
    assert m["precision@5"] == 0.6
    assert m["recall@5"] == 0.75
    assert m["mrr"] == 1.0

    # Manual nDCG calculations:
    # At K=1: DCG@1 = 1.0 / log2(2) = 1.0. IDCG@1 = 1.0 / log2(2) = 1.0. nDCG@1 = 1.0.
    assert m["ndcg@1"] == 1.0

    # At K=3:
    # DCG@3 = 1/log2(2) + 0 + 1/log2(4) = 1.0 + 0.5 = 1.5.
    # IDCG@3 = 1/log2(2) + 1/log2(3) + 1/log2(4) = 1.0 + 0.63093 + 0.5 = 2.13093.
    # nDCG@3 = 1.5 / 2.13093 = 0.7039.
    expected_ndcg3 = (1.0 + 0.5) / (1.0 + (1.0 / math.log2(3)) + 0.5)
    assert m["ndcg@3"] == pytest.approx(expected_ndcg3, abs=1e-4)


def test_empty_retrieval_results() -> None:
    """Empty retrieval must return 0.0 for all metrics without raising errors."""
    m = compute_retrieval_metrics([], ["T1", "T2"], ks=(1, 3, 5))
    for k in (1, 3, 5):
        assert m[f"precision@{k}"] == 0.0
        assert m[f"recall@{k}"] == 0.0
        assert m[f"ndcg@{k}"] == 0.0
    assert m["mrr"] == 0.0


def test_fewer_than_k_retrieved_passages() -> None:
    """When fewer than K passages are retrieved, precision denominator remains K and no slots are fabricated."""
    # Retrieved 2 passages, K=5
    retrieved = ["T1", "N1"]
    targets = ["T1", "T2"]

    m = compute_retrieval_metrics(retrieved, targets, ks=(1, 3, 5))

    # K=1: [T1] -> 1/1 = 1.0
    assert m["precision@1"] == 1.0
    assert m["recall@1"] == 0.5

    # K=3: [T1, N1] -> 1/3 = 0.3333
    assert m["precision@3"] == pytest.approx(1 / 3, abs=1e-4)
    assert m["recall@3"] == 0.5

    # K=5: [T1, N1] -> 1/5 = 0.2000
    assert m["precision@5"] == 0.2
    assert m["recall@5"] == 0.5
    assert m["mrr"] == 1.0


def test_duplicate_retrieved_chunk_ids() -> None:
    """Duplicate retrieved chunk IDs must only count once toward relevance."""
    # Retrieved: ["T1", "T1", "T2"] at ranks 1, 2, 3
    # Targets: ["T1", "T2"]
    retrieved = ["T1", "T1", "T2"]
    targets = ["T1", "T2"]

    m = compute_retrieval_metrics(retrieved, targets, ks=(1, 2, 3))

    # At K=2: [T1, T1] -> only 1 unique hit!
    # Precision@2 = 1/2 = 0.5 (NOT 2/2)
    # Recall@2 = 1/2 = 0.5 (NOT 2/2)
    assert m["precision@2"] == 0.5
    assert m["recall@2"] == 0.5

    # At K=3: [T1, T1, T2] -> 2 unique hits!
    # Precision@3 = 2/3 = 0.6667
    # Recall@3 = 2/2 = 1.0
    assert m["precision@3"] == pytest.approx(2 / 3, abs=1e-4)
    assert m["recall@3"] == 1.0


def test_missing_ground_truth_targets() -> None:
    """Empty or None targets must return 0.0 without divide-by-zero exceptions."""
    m1 = compute_retrieval_metrics(["T1", "T2"], [], ks=(1, 3, 5))
    assert m1["precision@1"] == 0.0
    assert m1["recall@1"] == 0.0
    assert m1["mrr"] == 0.0

    m2 = compute_retrieval_metrics(["T1", "T2"], None, ks=(1, 3, 5))  # type: ignore[arg-type]
    assert m2["precision@1"] == 0.0
    assert m2["recall@1"] == 0.0
    assert m2["mrr"] == 0.0


def test_citation_metrics_calculations() -> None:
    """Verify citation precision and recall across dictionary citations and raw string citations."""
    targets = ["C1", "C2", "C3"]
    target_sources = ["S1", "S2"]

    # Case 1: Partial match
    citations = [
        {"chunk_id": "C1", "source_id": "S1"},
        {"chunk_id": "C4", "source_id": "S3"},  # irrelevant
    ]
    cm = compute_citation_metrics(citations, targets, target_sources)
    # 1 of 2 cited chunks is relevant -> Precision = 0.5
    # 1 of 3 target chunks is cited -> Recall = 1/3 = 0.3333
    assert cm["citation_precision"] == 0.5
    assert cm["citation_recall"] == pytest.approx(1 / 3, abs=1e-4)
    assert cm["citation_source_precision"] == 0.5
    assert cm["citation_source_recall"] == 0.5

    # Case 2: Empty citations
    cm_empty = compute_citation_metrics([], targets, target_sources)
    assert cm_empty["citation_precision"] == 0.0
    assert cm_empty["citation_recall"] == 0.0

    # Case 3: Empty targets
    cm_no_target = compute_citation_metrics(citations, [])
    assert cm_no_target["citation_precision"] == 0.0
    assert cm_no_target["citation_recall"] == 0.0


def test_abstention_scoring() -> None:
    """Verify abstention accuracy calculation on expected abstention queries."""
    # 3 correct, 1 failed (cited something), 1 failed (abstained=False)
    results = [
        {"abstained": True, "citations": []},
        {"abstained": True, "citations": []},
        {"abstained": True, "citations": []},
        {"abstained": True, "citations": [{"chunk_id": "C1"}]},  # hallucinated citation
        {"abstained": False, "citations": []},  # failed to abstain
    ]
    acc = compute_abstention_accuracy(results)
    assert acc == 0.6  # 3 / 5 = 0.6

    # All correct
    assert compute_abstention_accuracy([{"abstained": True, "citations": []}] * 4) == 1.0

    # Empty list
    assert compute_abstention_accuracy([]) == 1.0


def test_backward_compatible_interfaces() -> None:
    """Verify legacy retrieval_metrics and format_table interfaces."""
    legacy = retrieval_metrics(["A", "B"], ["A"], 5)
    assert "precision_at_k" in legacy
    assert "recall_at_k" in legacy
    assert "mrr" in legacy
    assert "ndcg_at_k" in legacy
    assert legacy["precision_at_k"] == 0.2
    assert legacy["recall_at_k"] == 1.0
    assert legacy["mrr"] == 1.0

    mock_report = {
        "overall_metrics": {
            "retrieval": {
                "precision@1": 0.8,
                "precision@3": 0.5,
                "precision@5": 0.3,
                "recall@1": 0.2,
                "recall@3": 0.35,
                "recall@5": 0.4,
                "mrr": 0.85,
                "ndcg@1": 0.8,
                "ndcg@3": 0.6,
                "ndcg@5": 0.55,
                "citation_precision": 0.65,
                "citation_recall": 0.3,
            },
            "abstention_accuracy": 1.0,
        }
    }
    table_str = format_table(mock_report)
    assert "Precision@1" in table_str
    assert "Abstention Accuracy         100.0%" in table_str


@pytest.fixture(scope="module")
def prepared_rag_engine() -> tuple[dict[str, Any], HybridRetriever, GroundedGenerator]:
    """Initialize hybrid retriever and grounded generator once for end-to-end tests."""
    cfg = load_config()
    records, _ = load_prepared_corpus(cfg)
    index = build_indices(records, cfg)
    retriever = HybridRetriever(index, cfg)
    generator = GroundedGenerator(cfg)
    return cfg, retriever, generator


def test_end_to_end_benchmark_evaluation(
    prepared_rag_engine: tuple[dict[str, Any], HybridRetriever, GroundedGenerator],
    tmp_path: Path,
) -> None:
    """Run full benchmark evaluation on all 28 queries and assert report generation."""
    cfg, retriever, generator = prepared_rag_engine

    result = evaluate_benchmark(
        config=cfg,
        retriever=retriever,
        generator=generator,
        output_dir=tmp_path,
    )

    meta = result["metadata"]
    assert meta["total_queries"] == 28
    assert meta["answerable_queries"] == 22
    assert meta["abstention_queries"] == 6

    overall = result["overall_metrics"]
    ret = overall["retrieval"]

    # Verify retrieval metrics are reasonable and non-trivial
    assert ret["precision@1"] > 0.5
    assert ret["recall@5"] >= ret["recall@1"]
    assert ret["mrr"] > 0.7
    assert ret["ndcg@1"] > 0.5
    assert ret["ndcg@5"] > 0.4

    # Verify 100% abstention on out-of-domain probes
    assert overall["abstention_accuracy"] == 1.0

    # Verify report artifacts exist
    report_file = tmp_path / "benchmark_report.md"
    results_json = tmp_path / "benchmark_results.json"

    assert report_file.exists()
    assert results_json.exists()

    report_text = report_file.read_text(encoding="utf-8")
    assert "# Science Behind Crafts RAG" in report_text
    assert "Abstention Accuracy" in report_text
    assert "Q28" in report_text

    parsed_json = json.loads(results_json.read_text(encoding="utf-8"))
    assert len(parsed_json["query_details"]) == 28

    # Also verify that canonical craft_rag/artifacts/benchmark_report.md exists
    pkg_report = Path("craft_rag/artifacts/benchmark_report.md")
    assert pkg_report.exists()
    assert pkg_report.stat().st_size > 1000

