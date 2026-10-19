"""Tests for prepared-corpus ingestion and the integrated RAG application."""
from __future__ import annotations

import builtins
import csv
import json
import threading
from pathlib import Path
from urllib.request import Request, urlopen

import pytest

from craft_rag import server
from craft_rag.pipeline.chunk import chunk_records
from craft_rag.pipeline.common import load_config
from craft_rag.pipeline.embed_text import TextEmbedder
from craft_rag.pipeline.generate import GroundedGenerator
from craft_rag.pipeline.index_build import build_indices
from craft_rag.pipeline.ingest import ingest
from craft_rag.pipeline.prepared_data import load_prepared_corpus
from craft_rag.pipeline.retrieve import HybridRetriever
from craft_rag.pipeline.web_fetch import AllowListedFetcher

ROOT = Path(__file__).resolve().parents[1]


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def _write_minimal_corpus(root: Path, **datasets: list[dict]) -> dict:
    paths = {
        "documents": "data/extracted/documents.jsonl",
        "gap_analysis_sources": "data/processed/gap_analysis_sources.jsonl",
        "gap_analysis_questions": "data/processed/gap_analysis_questions.jsonl",
        "process_observations": "data/processed/process_observations.jsonl",
        "full_paper_chunks": "data/processed/full_paper_chunks.jsonl",
    }
    for name, relative_path in paths.items():
        _write_jsonl(root / relative_path, datasets.get(name, []))
    catalog = root / "data" / "metadata" / "source_catalog.csv"
    catalog.parent.mkdir(parents=True, exist_ok=True)
    with catalog.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["source_id", "title", "doi", "source_url", "craft", "review_status"],
        )
        writer.writeheader()
        writer.writerow({
            "source_id": "WBC-0001",
            "title": "Bamboo material science",
            "doi": "10.1234/example",
            "source_url": "https://doi.org/10.1234/example",
            "craft": "Bamboo",
            "review_status": "catalog verified",
        })
    return {"data": {"prepared_root": str(root)}}


def test_actual_prepared_datasets_map_records_and_keep_provenance():
    records, report = load_prepared_corpus(load_config())

    assert report["input_record_counts"] == {
        "documents": 66,
        "gap_analysis_sources": 55,
        "gap_analysis_questions": 95,
        "process_observations": 140,
        "full_paper_chunks": 194,
    }
    assert report["indexed_record_count"] == len(records)
    assert report["quality_issue_count"] > 0
    assert any(record["record_type"] == "research_question" for record in records)
    clusters = {record["craft_cluster"] for record in records if record["craft_cluster"]}
    assert {"Wood carving", "Bamboo carving"} <= clusters
    assert any(
        record["craft_cluster"] == "Wood carving"
        and record["craft"] != "Wood carving"
        for record in records
    )
    paper_chunk = next(record for record in records if record["record_type"] == "full_paper_chunk")
    assert paper_chunk["source_id"] in {"WBC-0051", "WBC-2016"}
    assert paper_chunk["metadata"]["provenance"]["dataset"] == "full_paper_chunks"
    assert paper_chunk["metadata"]["provenance"]["chunk_index"] == 1
    hand_tool_paper = next(
        record for record in records
        if record["record_type"] == "full_paper_chunk" and record["source_id"] == "WBC-0051"
    )
    assert hand_tool_paper["metadata"]["relevance_note"]

    source_card = next(
        record for record in records
        if record["record_type"] == "source_document" and record["source_id"] == "WBC-0005"
    )
    assert source_card["title"] == "Generation Rate and Particle Size Distribution of Wood Dust by Handheld Sanding Operation"
    assert source_card["evidence_level"] == "E3"
    assert source_card["doi"] == "10.1539/joh.16-0136-BR"
    assert source_card["link"] == "https://doi.org/10.1539/joh.16-0136-BR"
    assert source_card["metadata"]["provenance"]["relative_path"].startswith("data\\raw\\")
    assert source_card["text_chunk"].startswith(source_card["title"])
    assert len(source_card["text_chunk"]) < 1000
    assert "Field Entry Source ID" not in source_card["text_chunk"]
    assert "relevance_note" in source_card["metadata"]
    assert any(
        record["metadata"].get("relevance_note")
        for record in records
        if record["record_type"] == "source_document"
    )


def test_config_and_actual_corpus_paths_do_not_depend_on_working_directory(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    config = load_config()
    records, report = load_prepared_corpus(config)

    assert Path(config["data"]["prepared_root"]) == ROOT
    assert report["input_record_counts"]["full_paper_chunks"] == 194
    assert records
    assert len(ingest(config)) == len(records)
    with pytest.raises(ValueError, match="Synthetic seed records"):
        ingest(config, include_seed=True)
    assert AllowListedFetcher().tier("https://doi.org/10.1234/example") == "scientific"


def test_duplicate_catalog_entries_merge_without_collapsing_distinct_record_types(tmp_path):
    config = _write_minimal_corpus(
        tmp_path,
        documents=[{
            "document_key": "doc-1",
            "file_name": "WBC-0001_card.pdf",
            "relative_path": "data/raw/WBC-0001_card.pdf",
            "text": (
                "Field Entry\nSource ID WBC-0001\nTitle Bamboo material science\n"
                "Source type Paper\nEvidence level E3\nLink / File https://example.org/source"
            ),
        }],
        gap_analysis_sources=[{
            "record_id": "gap-source-1",
            "source_id": "WBC-0001",
            "title": "Bamboo material science",
            "source_type": "Paper",
            "evidence": "E3",
            "source_url": "https://example.org/source",
        }],
        gap_analysis_questions=[{
            "record_id": "question-1",
            "question": "Why does bamboo age affect strength?",
            "sources_currently_answering": "WBC-0001",
        }],
        process_observations=[{
            "record_id": "observation-1",
            "source_id": "WBC-0001",
            "observation_or_question": "Artisans select mature bamboo culms.",
        }],
        full_paper_chunks=[{
            "record_id": "WBC-0001-0001",
            "source_id": "WBC-0001",
            "title": "Bamboo material science",
            "text": "Bamboo age affects fibre properties.",
        }],
    )
    records, report = load_prepared_corpus(config)

    matching_records = [
        record for record in records
        if record["source_id"] == "WBC-0001" and record["title"] == "Bamboo material science"
    ]
    assert sum(record["record_type"] == "source_document" for record in matching_records) == 1
    assert sum(record["record_type"] == "full_paper_chunk" for record in matching_records) == 1
    assert sum(
        record["record_type"] == "process_observation"
        for record in records
        if record["source_id"] == "WBC-0001"
    ) == 1
    assert report["duplicate_source_records_merged"] == 1
    card = next(record for record in matching_records if record["record_type"] == "source_document")
    assert card["metadata"]["provenance"]["merged_records"][0]["record_id"] == "gap-source-1"


def test_missing_fields_are_reported_and_records_are_retained_or_explicitly_skipped(tmp_path):
    config = _write_minimal_corpus(
        tmp_path,
        documents=[{
            "document_key": "document-without-id",
            "file_name": "untitled.pdf",
            "text": "This extracted record has no source identifier or title.",
        }],
        gap_analysis_sources=[{
            "record_id": "catalog-without-id",
            "title": "Missing source identifier",
        }],
        gap_analysis_questions=[{
            "record_id": "question-without-text",
            "question": "",
        }],
        process_observations=[
            {"record_id": "observation-without-text", "source_id": "WBC-0001"},
            {
                "record_id": "observation-without-id",
                "observation_or_question": "A recorded craft process note.",
            },
        ],
        full_paper_chunks=[{
            "record_id": "paper-without-source",
            "text": "A retained paper passage with missing source metadata.",
        }],
    )
    records, report = load_prepared_corpus(config)
    by_type = {record["record_type"] for record in records}

    assert {"source_document", "catalog_entry", "process_observation", "full_paper_chunk"} <= by_type
    assert any(issue["record_id"] == "question-without-text" for issue in report["quality_issues"])
    assert any(issue["record_id"] == "observation-without-text" for issue in report["quality_issues"])
    assert any(issue["record_id"] == "document-without-id" for issue in report["quality_issues"])
    assert any(record["source_id"] == "PAPER:paper-without-source" for record in records)
    assert any(record["source_id"] == "OBS:observation-without-id" for record in records)


def test_chunk_ids_are_unique_and_text_respects_word_and_character_limits():
    records = [
        {
            "record_id": "record-1",
            "source_id": "SAME-SOURCE",
            "title": "Long record",
            "text_chunk": "carving " * 140,
            "metadata": {"source_url": "https://example.org/1"},
        },
        {
            "record_id": "record-2",
            "source_id": "SAME-SOURCE",
            "title": "Second record",
            "text_chunk": "x" * 100,
        },
    ]

    chunks = chunk_records(records, words=20, overlap=3, max_chars=40)

    assert chunks
    assert len({chunk["chunk_id"] for chunk in chunks}) == len(chunks)
    assert all(len(chunk["text_chunk"]) <= 40 for chunk in chunks)
    assert all(len(chunk["text_chunk"].split()) <= 20 for chunk in chunks)
    assert all(chunk["chunk_id"].startswith("SAME-SOURCE::doc") for chunk in chunks)
    assert chunks[-1]["text_chunk"] == "x" * 40 or chunks[-1]["text_chunk"] == "x" * 20


def test_index_build_and_hybrid_retrieval_use_prepared_style_records(tmp_path):
    config = load_config()
    config["chunking"] = {"words": 40, "overlap": 5, "max_chars": 300}
    config["embedding"] = {
        "backend": "hashing",
        "cache_dir": str(tmp_path / "cache"),
        "model_name": "offline-test",
    }
    config["data"]["artifact_dir"] = str(tmp_path / "artifacts")
    config["index"] = {
        "metadata_path": str(tmp_path / "artifacts" / "chunks.json"),
        "faiss_path": str(tmp_path / "artifacts" / "dense.faiss"),
        "chroma_path": str(tmp_path / "artifacts" / "chroma"),
        "chroma_collection": "test",
    }
    records = [
        {
            "source_id": "WBC-1001",
            "title": "Bamboo maturity",
            "record_id": "r1",
            "record_type": "full_paper_chunk",
            "text_chunk": "Bamboo age and maturity affect culm fibre strength, flexibility, and material selection.",
            "evidence_level": "E3",
            "metadata": {"source_url": "https://example.org/bamboo", "evidence_eligible": True},
        },
        {
            "source_id": "WBC-1002",
            "title": "Metal finishing",
            "record_id": "r2",
            "record_type": "full_paper_chunk",
            "text_chunk": "A metal surface treatment changes corrosion resistance and polishing.",
            "evidence_level": "E3",
        },
        {
            "source_id": "WBC-1001",
            "title": "Bamboo maturity",
            "record_id": "r1-copy",
            "record_type": "source_document",
            "text_chunk": "Bamboo age and maturity affect culm fibre strength and material selection.",
            "evidence_level": "E3",
        },
        {
            "source_id": "GAP:question-1",
            "title": "Research question: bamboo maturity",
            "record_id": "q1",
            "record_type": "research_question",
            "text_chunk": "Why does bamboo age affect fibre strength?",
            "evidence_eligible": False,
        },
        {
            "source_id": "WBC-1003",
            "title": "Hand tool ergonomics",
            "record_id": "r3",
            "record_type": "source_document",
            "text_chunk": "The paper discusses ergonomic factors.",
            "evidence_level": "E3",
            "metadata": {
                "record_type": "source_document",
                "questions_answered": ["How does handle diameter affect grip strength?"],
                "relevance_note": "Handle diameter affects grip strength.",
            },
        },
    ]

    index = build_indices(records, config)
    found = HybridRetriever(index, config).retrieve("How does bamboo age affect fibre strength?", top_k=2)

    assert index["chunks"]
    assert all(
        chunk["metadata"].get("record_type") != "research_question"
        for chunk in index["chunks"]
    )
    assert any(
        chunk["metadata"].get("record_type") == "research_question"
        for chunk in index["all_chunks"]
    )
    assert Path(config["index"]["metadata_path"]).is_file()
    assert found[0]["source_id"] == "WBC-1001"
    assert found[0]["metadata"]["source_url"] == "https://example.org/bamboo"
    assert sum(row["source_id"] == "WBC-1001" for row in found) == 1
    assert index["backend_info"]["embedding"] == "hashing"
    assert not HybridRetriever(index, config).retrieve("Why does bamboo age affect fibre strength?", top_k=2)[0]["source_id"].startswith("GAP:")
    metadata_hit = HybridRetriever(index, config).retrieve(
        "How does handle diameter affect grip strength?",
        top_k=1,
    )
    assert metadata_hit[0]["source_id"] == "WBC-1003"


def test_hashing_retrieval_returns_no_random_dense_collisions():
    config = load_config()
    records, _ = load_prepared_corpus(config)
    index = build_indices(records, config)

    assert HybridRetriever(index, config).retrieve("quantum entanglement") == []


def test_real_prepared_corpus_retrieves_and_cites_relevant_sources(tmp_path):
    config = load_config()
    config["data"]["artifact_dir"] = str(tmp_path / "artifacts")
    config["embedding"]["backend"] = "hashing"
    config["embedding"]["cache_dir"] = str(tmp_path / "cache")
    config["index"]["metadata_path"] = str(tmp_path / "artifacts" / "chunks.json")
    config["index"]["faiss_path"] = str(tmp_path / "artifacts" / "dense.faiss")
    config["index"]["chroma_path"] = str(tmp_path / "artifacts" / "chroma")
    records, _ = load_prepared_corpus(config)
    index = build_indices(records, config)
    retriever = HybridRetriever(index, config)
    generator = GroundedGenerator(config)
    samples = {
        "How does hand tool design affect grip strength?": "WBC-0051",
        "Why does grain angle affect cutting across the wood grain?": "WBC-0009",
        "Why does mature bamboo density affect carving force?": "WBC-0059",
    }

    for question, expected_source in samples.items():
        passages = retriever.retrieve(question)
        assert expected_source in {passage["source_id"] for passage in passages}
        answer = generator.answer(question, passages)
        assert f"[{expected_source} | E3]" in answer
        assert "Field Entry Source ID" not in answer
        assert len(answer) < 1000
    grip_answer = generator.answer(
        "How does hand tool design affect grip strength?",
        retriever.retrieve("How does hand tool design affect grip strength?"),
    )
    assert "handle diameter" in grip_answer.casefold()
    assert "repository annotation (not source text)" in grip_answer.casefold()


def test_generator_cites_retrieved_evidence_and_rejects_question_records():
    generator = GroundedGenerator({"generator": {"backend": "extractive", "minimum_results": 1}})
    evidence = {
        "source_id": "WBC-0001",
        "evidence_level": "E3",
        "text_chunk": "Bamboo age affects fibre strength and material selection.",
        "metadata": {"record_type": "full_paper_chunk", "evidence_eligible": True},
    }
    question_record = {
        "source_id": "GAP:question-1",
        "evidence_level": "E5",
        "text_chunk": "Why does bamboo age affect fibre strength?",
        "metadata": {"record_type": "research_question", "evidence_eligible": False},
    }

    answer = generator.answer("Why does bamboo age affect fibre strength?", [question_record, evidence])

    assert "[WBC-0001 | E3]" in answer
    assert generator.answer("Why does bamboo age affect fibre strength?", [question_record]).startswith(
        "Insufficient evidence"
    )
    weakly_related = {
        "source_id": "WBC-0002",
        "evidence_level": "E3",
        "text_chunk": "Bamboo fibre strength varies by species.",
        "metadata": {"record_type": "source_document", "evidence_eligible": True},
    }
    assert generator.answer(
        "What causes bamboo to split along its fibres?",
        [weakly_related],
    ).startswith("Insufficient evidence")


def test_optional_model_backends_fail_with_setup_guidance(monkeypatch):
    original_import = builtins.__import__

    def missing_optional_module(name, *args, **kwargs):
        if name in {"sentence_transformers", "transformers"}:
            raise ImportError(f"No module named {name}")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", missing_optional_module)
    with pytest.raises(ImportError, match="requirements-llm.txt"):
        TextEmbedder({"embedding": {"backend": "sentence-transformers"}})
    with pytest.raises(ImportError, match="requirements-llm.txt"):
        GroundedGenerator({"generator": {"backend": "transformers", "model_name": "local-model"}})


def test_e5_embeddings_use_query_and_passage_prefixes(tmp_path):
    config = {
        "embedding": {
            "backend": "sentence-transformers",
            "model_name": "intfloat/multilingual-e5-small",
            "cache_dir": str(tmp_path),
        }
    }
    embedder = object.__new__(TextEmbedder)
    embedder.backend = "sentence-transformers"
    embedder.model_name = config["embedding"]["model_name"]
    embedder.batch_size = 32
    embedder.cache_dir = Path(config["embedding"]["cache_dir"])
    embedder.model = None

    class FakeModel:
        def __init__(self):
            self.inputs = []

        def encode(self, texts, **kwargs):
            self.inputs.append(texts)
            return [[1.0, 0.0] for _ in texts]

    model = FakeModel()
    embedder.model = model
    embedder.encode(["paper excerpt"], input_type="document")
    embedder.encode(["research question"], input_type="query")

    assert model.inputs == [["passage: paper excerpt"], ["query: research question"]]


def test_local_server_status_and_query_api(monkeypatch):
    passage = {
        "source_id": "WBC-0001",
        "title": "Bamboo material science",
        "evidence_level": "E3",
        "grade": "A",
        "craft_cluster": "Bamboo",
        "why_theme": "Material selection",
        "process_stage": "Material",
        "text_chunk": "Bamboo maturity affects fibre strength.",
        "link": "https://example.org/bamboo",
        "metadata": {"title": "Bamboo material science", "source_url": "https://example.org/bamboo"},
        "retrieval_score": 0.5,
    }

    class FakeRetriever:
        def retrieve(self, question, filters=None):
            return [passage]

    class FakeGenerator:
        def answer(self, question, passages):
            if "capital of france" in question.casefold():
                return "Insufficient evidence in the retrieved corpus to answer this question."
            return "Retrieved evidence: Bamboo maturity [WBC-0001 | E3]"

    app = {
        "record_count": 2,
        "quality_issue_count": 0,
        "index": {
            "backend_info": {"embedding": "hashing"},
            "chunks": [
                {"craft_cluster": "Wood carving"},
                {"craft_cluster": "Bamboo carving"},
            ],
        },
        "retriever": FakeRetriever(),
        "generator": FakeGenerator(),
    }
    monkeypatch.setattr(server, "get_engine", lambda: app)
    http_server = server.LocalServer(("127.0.0.1", 0), server.Handler)
    thread = threading.Thread(target=http_server.serve_forever, daemon=True)
    thread.start()
    try:
        base = f"http://127.0.0.1:{http_server.server_port}"
        with urlopen(f"{base}/api/status") as response:
            status = json.load(response)
        with urlopen(f"{base}/api/filters") as response:
            filters = json.load(response)
        request = Request(
            f"{base}/api/query",
            data=json.dumps({"question": "How does bamboo maturity affect fibres?"}).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urlopen(request) as response:
            body = json.load(response)
        unsupported_request = Request(
            f"{base}/api/query",
            data=json.dumps({"question": "What is the capital of France?"}).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urlopen(unsupported_request) as response:
            unsupported_body = json.load(response)
        assert status["status"] == "ready"
        assert {"Wood carving", "Bamboo carving"} <= set(filters["craft_clusters"])
        assert body["sources"][0]["source_id"] == "WBC-0001"
        assert body["sources"][0]["source_url"] == "https://example.org/bamboo"
        assert "[WBC-0001 | E3]" in body["answer"]
        assert unsupported_body["sources"] == []
        assert server.DEFAULT_HOST == "127.0.0.1"
    finally:
        http_server.shutdown()
        http_server.server_close()
        thread.join(timeout=5)
