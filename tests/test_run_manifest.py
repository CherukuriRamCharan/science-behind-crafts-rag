"""Regression tests for evaluation run manifests."""
from __future__ import annotations

import json

from src.run_manifest import write_run_manifest


def test_run_manifest_records_git_config_corpus_versions_and_outputs(tmp_path):
    result_file = tmp_path / "retrieval_summary.csv"
    result_file.write_text("retriever,MRR\nbm25,0.5\n", encoding="utf-8")
    chunks = [
        {"chunk_id": "b", "source_id": "S2", "record_id": "r2", "text_chunk": "second"},
        {"chunk_id": "a", "source_id": "S1", "record_id": "r1", "text_chunk": "first"},
    ]

    output = write_run_manifest(tmp_path, config={"retrieval": {"top_k": 5}}, corpus_chunks=chunks)
    manifest = json.loads(output.read_text(encoding="utf-8"))

    assert manifest["git_sha"]
    assert manifest["config_snapshot"] == {"retrieval": {"top_k": 5}}
    assert manifest["config_sha256"]
    assert manifest["corpus_sha256"]
    assert manifest["corpus_chunk_count"] == 2
    assert "numpy" in manifest["package_versions"]
    assert manifest["output_sha256"]["retrieval_summary.csv"]
    assert "run_manifest.json" not in manifest["output_sha256"]


def test_corpus_hash_is_independent_of_chunk_order(tmp_path):
    chunks = [
        {"chunk_id": "b", "source_id": "S2", "record_id": "r2", "text_chunk": "second"},
        {"chunk_id": "a", "source_id": "S1", "record_id": "r1", "text_chunk": "first"},
    ]
    first = json.loads(write_run_manifest(tmp_path, config={}, corpus_chunks=chunks).read_text())
    second = json.loads(write_run_manifest(tmp_path, config={}, corpus_chunks=chunks[::-1]).read_text())
    assert first["corpus_sha256"] == second["corpus_sha256"]
