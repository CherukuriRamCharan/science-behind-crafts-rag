"""Tests for deterministic chunk-reference generation."""
import json

from src.build_searchable_chunk_reference import build_reference


def test_reference_contains_exact_chunks_and_corpus_hash(tmp_path):
    chunks = [{
        "chunk_id": "S1::0000", "record_id": "S1", "source_id": "S1",
        "text_chunk": "Grounded text", "record_type": "source_document",
        "evidence_level": "E3", "craft_cluster": "Wood",
        "metadata": {"title": "Title", "document_type": "source_document",
                     "provenance": {"record_id": "record-1"}},
    }]
    path, manifest = build_reference(chunks, tmp_path / "searchable_chunks_reference.json")
    rows = json.loads(path.read_text(encoding="utf-8"))
    row = rows[0]
    assert row["chunk_id"] == "S1::0000"
    assert row["text"] == "Grounded text"
    assert row["corpus_sha256"] == manifest["corpus_sha256"]
    assert len(manifest["corpus_sha256"]) == 64
    sidecar = json.loads(path.with_name("searchable_chunks_reference.manifest.json").read_text())
    assert sidecar == manifest

