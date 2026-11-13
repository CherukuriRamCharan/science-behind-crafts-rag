"""Stage 4C Benchmark Ground-Truth Validation Tests.

Validates the 28-query benchmark dataset:
- Schema and structural integrity
- Target source and chunk ID resolution against the prepared corpus
- Multi-source coverage
- Abstention / out-of-domain query assertions
- Negative validation checks for missing, invalid, or mismatched targets
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from craft_rag.pipeline.chunk import chunk_records
from craft_rag.pipeline.common import load_config
from craft_rag.pipeline.generate import GroundedGenerator
from craft_rag.pipeline.prepared_data import load_prepared_corpus

BENCHMARK_PATH = Path("craft_rag/data/eval/craft_ground_truth_eval.json")
EXPECTED_CATEGORIES = {
    "Wood Carving Mechanics & Species Properties",
    "Bamboo Structural & Splitting Mechanics",
    "Tool Ergonomics, Posture & Body Mechanics",
    "Adhesives, Finishing & Preservation",
    "Multi-Source Craft Process Synthesis",
    "Insufficient Evidence & Out-of-Domain Probes",
    "Hallucination & Adversarial Probes",
}
REQUIRED_FIELDS = {
    "id",
    "category",
    "query",
    "target_source_ids",
    "expected_abstention",
    "multi_source",
    "target_chunk_ids",
}


def validate_benchmark_entry(
    entry: dict[str, Any],
    valid_source_ids: set[str],
    chunk_map: dict[str, dict[str, Any]],
) -> list[str]:
    """Validate a single benchmark query entry, returning a list of error strings."""
    errors: list[str] = []
    missing_fields = REQUIRED_FIELDS - set(entry.keys())
    if missing_fields:
        errors.append(f"Missing required fields: {sorted(missing_fields)}")
        return errors  # Cannot proceed further without base fields

    qid = entry.get("id", "UNKNOWN")
    category = entry.get("category", "")
    query = entry.get("query", "")
    target_sources = entry.get("target_source_ids")
    target_chunks = entry.get("target_chunk_ids")
    is_abstention = entry.get("expected_abstention")
    is_multi = entry.get("multi_source")

    if not isinstance(qid, str) or not qid.strip():
        errors.append("Invalid or empty 'id'")
    if not isinstance(category, str) or category not in EXPECTED_CATEGORIES:
        errors.append(f"Unknown or invalid category: '{category}'")
    if not isinstance(query, str) or not query.strip():
        errors.append("Invalid or empty 'query'")
    if not isinstance(is_abstention, bool):
        errors.append("'expected_abstention' must be a boolean")
    if not isinstance(is_multi, bool):
        errors.append("'multi_source' must be a boolean")
    if not isinstance(target_sources, list):
        errors.append("'target_source_ids' must be a list")
    if not isinstance(target_chunks, list):
        errors.append("'target_chunk_ids' must be a list")

    if errors:
        return errors

    if is_abstention:
        if target_sources:
            errors.append(f"Abstention query {qid} must have empty target_source_ids, got {target_sources}")
        if target_chunks:
            errors.append(f"Abstention query {qid} must have empty target_chunk_ids, got {target_chunks}")
        if is_multi:
            errors.append(f"Abstention query {qid} cannot have multi_source=True")
    else:
        if not target_sources:
            errors.append(f"Answerable query {qid} has empty target_source_ids")
        if not target_chunks:
            errors.append(f"Answerable query {qid} has empty target_chunk_ids")

        for sid in target_sources:
            if sid not in valid_source_ids:
                errors.append(f"Query {qid} references unknown source_id '{sid}' not in corpus")

        for cid in target_chunks:
            if cid not in chunk_map:
                errors.append(f"Query {qid} references unknown chunk_id '{cid}' not in corpus")
            else:
                chunk = chunk_map[cid]
                chunk_sid = chunk.get("source_id")
                if chunk_sid not in target_sources:
                    errors.append(
                        f"Query {qid} target chunk '{cid}' belongs to '{chunk_sid}', "
                        f"which is not in declared target_source_ids {target_sources}"
                    )

        if is_multi:
            if len(target_sources) < 2:
                errors.append(f"Multi-source query {qid} has fewer than 2 target sources: {target_sources}")
            chunk_sources = {chunk_map[cid].get("source_id") for cid in target_chunks if cid in chunk_map}
            if len(chunk_sources) < 2:
                errors.append(f"Multi-source query {qid} target chunks cover fewer than 2 distinct sources: {chunk_sources}")
        else:
            if len(target_sources) != 1:
                errors.append(f"Single-source query {qid} should have exactly 1 target source, got {len(target_sources)}")

    return errors


@pytest.fixture(scope="module")
def benchmark_data() -> list[dict[str, Any]]:
    assert BENCHMARK_PATH.exists(), f"Benchmark file not found: {BENCHMARK_PATH}"
    with BENCHMARK_PATH.open(encoding="utf-8") as handle:
        data = json.load(handle)
    assert isinstance(data, list), "Benchmark file must contain a JSON list"
    return data


@pytest.fixture(scope="module")
def corpus_index() -> tuple[set[str], dict[str, dict[str, Any]]]:
    cfg = load_config()
    records, _manifest = load_prepared_corpus(cfg)
    chunk_cfg = cfg.get("chunking", {})
    chunks = chunk_records(
        records,
        words=chunk_cfg.get("words", 260),
        overlap=chunk_cfg.get("overlap", 45),
        max_chars=chunk_cfg.get("max_chars", 1200),
    )
    chunk_map = {c["chunk_id"]: c for c in chunks if "chunk_id" in c}
    valid_source_ids = {c["source_id"] for c in chunks if "source_id" in c}
    return valid_source_ids, chunk_map


def test_benchmark_file_exists_and_count(benchmark_data: list[dict[str, Any]]) -> None:
    """Benchmark ground truth must contain exactly 28 queries numbered Q01 to Q28."""
    assert len(benchmark_data) == 28, f"Expected 28 queries, got {len(benchmark_data)}"
    expected_ids = [f"Q{i:02d}" for i in range(1, 29)]
    actual_ids = [q["id"] for q in benchmark_data]
    assert actual_ids == expected_ids, f"Mismatched query IDs: {actual_ids}"


def test_benchmark_schema_and_categories(benchmark_data: list[dict[str, Any]]) -> None:
    """Each benchmark query must possess all required schema keys and a valid category."""
    categories_found = set()
    for q in benchmark_data:
        assert REQUIRED_FIELDS.issubset(set(q.keys())), f"Missing keys in {q.get('id')}"
        assert q["category"] in EXPECTED_CATEGORIES, f"Unexpected category {q['category']} in {q.get('id')}"
        categories_found.add(q["category"])
        assert len(q["query"].strip()) > 10, f"Query text too short in {q.get('id')}"

    assert categories_found == EXPECTED_CATEGORIES, "Not all expected categories are represented"


def test_target_resolution_in_corpus(
    benchmark_data: list[dict[str, Any]],
    corpus_index: tuple[set[str], dict[str, dict[str, Any]]],
) -> None:
    """All answerable queries (Q01-Q22) must have valid targets resolving to indexed corpus chunks."""
    valid_source_ids, chunk_map = corpus_index
    all_errors = []
    for q in benchmark_data:
        errors = validate_benchmark_entry(q, valid_source_ids, chunk_map)
        all_errors.extend(errors)

    assert not all_errors, f"Benchmark target resolution errors:\n" + "\n".join(all_errors)


def test_multi_source_queries_coverage(benchmark_data: list[dict[str, Any]]) -> None:
    """Multi-source queries must target at least 2 distinct sources."""
    multi_queries = [q for q in benchmark_data if q.get("multi_source")]
    assert len(multi_queries) == 16, f"Expected 16 multi-source queries, got {len(multi_queries)}"
    for q in multi_queries:
        assert len(q["target_source_ids"]) >= 2, f"{q['id']} has < 2 target sources"
        assert not q["expected_abstention"], f"{q['id']} cannot be multi-source and abstention"


def test_abstention_queries_properties(benchmark_data: list[dict[str, Any]]) -> None:
    """Q23-Q28 must be abstentions with empty target sets."""
    abstention_queries = [q for q in benchmark_data if q.get("expected_abstention")]
    assert len(abstention_queries) == 6, f"Expected 6 abstention queries, got {len(abstention_queries)}"
    abstention_ids = {q["id"] for q in abstention_queries}
    assert abstention_ids == {f"Q{i:02d}" for i in range(23, 29)}

    for q in abstention_queries:
        assert q["target_source_ids"] == [], f"{q['id']} must have empty target_source_ids"
        assert q["target_chunk_ids"] == [], f"{q['id']} must have empty target_chunk_ids"
        assert not q["multi_source"], f"{q['id']} must have multi_source=False"
        assert q["category"] in {"Insufficient Evidence & Out-of-Domain Probes", "Hallucination & Adversarial Probes"}


def test_grounded_generator_abstention_behavior(benchmark_data: list[dict[str, Any]]) -> None:
    """GroundedGenerator must abstain with citations=[] on out-of-domain queries with no evidence."""
    abstention_queries = [q for q in benchmark_data if q.get("expected_abstention")]
    cfg = load_config()
    generator = GroundedGenerator(cfg)

    for q in abstention_queries:
        result = generator.generate_structured(q["query"], passages=[])
        assert result["abstained"] is True, f"Expected generator to abstain on {q['id']}"
        assert result["citations"] == [], f"Expected no citations on abstained query {q['id']}"
        assert any(
            phrase in result["answer"].lower()
            for phrase in ("insufficient evidence", "not discussed", "no reliable")
        )


def test_validator_negative_cases(
    corpus_index: tuple[set[str], dict[str, dict[str, Any]]]
) -> None:
    """Validation helper must reject invalid, corrupted, or mismatched benchmark entries."""
    valid_source_ids, chunk_map = corpus_index
    sample_cid = next(iter(chunk_map.keys()))
    sample_sid = chunk_map[sample_cid]["source_id"]

    # 1. Missing field
    bad_entry1 = {"id": "QX", "query": "test"}
    errs1 = validate_benchmark_entry(bad_entry1, valid_source_ids, chunk_map)
    assert any("Missing required fields" in e for e in errs1)

    # 2. Nonexistent source_id
    bad_entry2 = {
        "id": "QX",
        "category": "Wood Carving Mechanics & Species Properties",
        "query": "Valid query test?",
        "target_source_ids": ["NONEXISTENT-9999"],
        "expected_abstention": False,
        "multi_source": False,
        "target_chunk_ids": [sample_cid],
    }
    errs2 = validate_benchmark_entry(bad_entry2, valid_source_ids, chunk_map)
    assert any("unknown source_id 'NONEXISTENT-9999'" in e for e in errs2)

    # 3. Nonexistent chunk_id
    bad_entry3 = {
        "id": "QX",
        "category": "Wood Carving Mechanics & Species Properties",
        "query": "Valid query test?",
        "target_source_ids": [sample_sid],
        "expected_abstention": False,
        "multi_source": False,
        "target_chunk_ids": ["NONEXISTENT-CHUNK::0000"],
    }
    errs3 = validate_benchmark_entry(bad_entry3, valid_source_ids, chunk_map)
    assert any("unknown chunk_id 'NONEXISTENT-CHUNK::0000'" in e for e in errs3)

    # 4. Chunk belonging to a different source than target_source_ids
    # Find another source
    other_cid = None
    for cid, c in chunk_map.items():
        if c.get("source_id") != sample_sid:
            other_cid = cid
            break
    assert other_cid is not None

    bad_entry4 = {
        "id": "QX",
        "category": "Wood Carving Mechanics & Species Properties",
        "query": "Valid query test?",
        "target_source_ids": [sample_sid],
        "expected_abstention": False,
        "multi_source": False,
        "target_chunk_ids": [other_cid],
    }
    errs4 = validate_benchmark_entry(bad_entry4, valid_source_ids, chunk_map)
    assert any("is not in declared target_source_ids" in e for e in errs4)

    # 5. Abstention query with non-empty targets
    bad_entry5 = {
        "id": "QX",
        "category": "Insufficient Evidence & Out-of-Domain Probes",
        "query": "Valid query test?",
        "target_source_ids": [sample_sid],
        "expected_abstention": True,
        "multi_source": False,
        "target_chunk_ids": [sample_cid],
    }
    errs5 = validate_benchmark_entry(bad_entry5, valid_source_ids, chunk_map)
    assert any("must have empty target_source_ids" in e for e in errs5)
    assert any("must have empty target_chunk_ids" in e for e in errs5)

    # 6. Multi-source query with only 1 source
    bad_entry6 = {
        "id": "QX",
        "category": "Wood Carving Mechanics & Species Properties",
        "query": "Valid query test?",
        "target_source_ids": [sample_sid],
        "expected_abstention": False,
        "multi_source": True,
        "target_chunk_ids": [sample_cid],
    }
    errs6 = validate_benchmark_entry(bad_entry6, valid_source_ids, chunk_map)
    assert any("fewer than 2 target sources" in e for e in errs6)
