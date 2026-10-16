"""Retrieval tests: tiny synthetic data + fake embeddings (no model download, no internet)."""
import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.retrieval import (  # noqa: E402
    BM25Retriever,
    ChunkLoadError,
    FaissRetriever,
    HybridRetriever,
    load_chunks,
    resolve_chunks_path,
)

RECORDS = [
    {"record_id": "T-0001", "source_id": "S1", "title": "Bamboo paper", "text": "Bamboo age affects fibre strength and material selection.",
     "craft": "Bamboo", "evidence_level": "E3", "chunk_index": 1},
    {"record_id": "T-0002", "source_id": "S1", "title": "Bamboo paper", "text": "Chisel handle design influences grip strength and posture.",
     "craft": "Bamboo", "chunk_index": 2},
    {"record_id": "T-0003", "source_id": "S2", "title": "Robot paper", "text": "Robotic carving learns tool angle from demonstration.",
     "craft": "Wood", "page": 7, "chunk_index": 1},
    {"record_id": "T-0004", "source_id": "S2", "title": "Robot paper", "text": "Wood grain direction changes cutting force.",
     "craft": "Wood", "chunk_index": 2},
]

# Fake embeddings: fixed 4-D vectors per text, so semantic ranking is fully controlled.
VECTORS = {
    RECORDS[0]["text"]: [1.0, 0.0, 0.0, 0.0],
    RECORDS[1]["text"]: [0.0, 1.0, 0.0, 0.0],
    RECORDS[2]["text"]: [0.0, 0.0, 1.0, 0.0],
    RECORDS[3]["text"]: [0.0, 0.0, 0.0, 1.0],
    "how long does a culm need to mature": [0.9, 0.1, 0.0, 0.0],   # semantically ~ T-0001, no shared words
    "hand posture": [0.0, 0.9, 0.1, 0.0],                          # ~ T-0002
    "bamboo grain": [0.5, 0.0, 0.0, 0.8],                          # FAISS: T-0004 first, T-0001 second
}


class FakeEmbedder:
    def encode(self, texts):
        return np.array([VECTORS[t] for t in texts], dtype=np.float32)


def write_jsonl(path, records):
    path.write_text("\n".join(json.dumps(r) for r in records) + "\n", encoding="utf-8")
    return path


@pytest.fixture()
def chunks(tmp_path):
    return load_chunks(write_jsonl(tmp_path / "chunks.jsonl", RECORDS))


# 1. loading valid records
def test_load_valid_chunks(chunks):
    assert [c.chunk_id for c in chunks] == ["T-0001", "T-0002", "T-0003", "T-0004"]
    assert chunks[0].source_id == "S1" and chunks[0].title == "Bamboo paper"
    assert chunks[0].text == RECORDS[0]["text"]


def test_path_is_configurable(tmp_path, monkeypatch):
    path = write_jsonl(tmp_path / "x.jsonl", RECORDS)
    monkeypatch.setenv("RETRIEVAL_CHUNKS_PATH", str(path))
    assert resolve_chunks_path() == path
    assert len(load_chunks()) == 4
    assert resolve_chunks_path("other.jsonl") == Path("other.jsonl")  # explicit argument wins


# 2. clear failures
def test_malformed_json_fails_clearly(tmp_path):
    path = tmp_path / "bad.jsonl"
    path.write_text(json.dumps(RECORDS[0]) + "\n{not json}\n", encoding="utf-8")
    with pytest.raises(ChunkLoadError, match="line 2"):
        load_chunks(path)


@pytest.mark.parametrize("missing", ["record_id", "text", "source_id", "title"])
def test_missing_required_field_fails_clearly(tmp_path, missing):
    record = {k: v for k, v in RECORDS[0].items() if k != missing}
    with pytest.raises(ChunkLoadError, match=missing):
        load_chunks(write_jsonl(tmp_path / "m.jsonl", [record]))


def test_empty_text_duplicate_id_and_missing_file(tmp_path):
    with pytest.raises(ChunkLoadError, match="text"):
        load_chunks(write_jsonl(tmp_path / "a.jsonl", [{**RECORDS[0], "text": "   "}]))
    with pytest.raises(ChunkLoadError, match="Duplicate"):
        load_chunks(write_jsonl(tmp_path / "b.jsonl", [RECORDS[0], RECORDS[0]]))
    with pytest.raises(ChunkLoadError, match="not found"):
        load_chunks(tmp_path / "nope.jsonl")


# 3. BM25
def test_bm25_returns_keyword_matches(chunks):
    results = BM25Retriever(chunks).search("grip posture", top_k=3)
    assert results[0].chunk_id == "T-0002"
    assert results[0].bm25_rank == 1 and results[0].bm25_score > 0
    assert all(r.faiss_rank is None and r.hybrid_score is None for r in results)


def test_bm25_no_match_returns_empty(chunks):
    assert BM25Retriever(chunks).search("zzzunknownzzz") == []


# 4. FAISS maps vectors to the right chunk IDs
def test_faiss_maps_ids_correctly(chunks):
    retriever = FaissRetriever(chunks, FakeEmbedder())
    top = retriever.search("how long does a culm need to mature", top_k=2)
    assert [r.chunk_id for r in top] == ["T-0001", "T-0002"]
    assert top[0].faiss_rank == 1 and top[0].faiss_score > top[1].faiss_score
    assert retriever.search("hand posture", top_k=1)[0].chunk_id == "T-0002"


# 5. hybrid merges and deduplicates
def test_hybrid_merges_and_deduplicates(chunks):
    hybrid = HybridRetriever.from_chunks(chunks, FakeEmbedder(), rrf_k=60)
    results = hybrid.search("bamboo grain", top_k=4)
    ids = [r.chunk_id for r in results]
    assert len(ids) == len(set(ids))                       # no duplicates
    both = {r.chunk_id: r for r in results}
    # Both T-0001 ("bamboo") and T-0004 ("grain") are found by BM25 AND FAISS -> merged into one record each.
    for cid in ("T-0001", "T-0004"):
        r = both[cid]
        assert r.bm25_rank is not None and r.faiss_rank is not None
        assert r.hybrid_score == pytest.approx(1 / (60 + r.bm25_rank) + 1 / (60 + r.faiss_rank))
    assert both["T-0004"].faiss_rank == 1 and both["T-0001"].faiss_rank == 2
    assert {ids[0], ids[1]} == {"T-0001", "T-0004"}  # items found by both lists outrank single-list items
    assert results == sorted(results, key=lambda r: (-r.hybrid_score, r.chunk_id))


def test_hybrid_semantic_only_hit_is_included(chunks):
    hybrid = HybridRetriever.from_chunks(chunks, FakeEmbedder())
    top = hybrid.search("how long does a culm need to mature", top_k=1)[0]
    assert top.chunk_id == "T-0001" and top.bm25_rank is None and top.faiss_rank == 1


def test_hybrid_rejects_mismatched_retrievers(chunks):
    with pytest.raises(ValueError):
        HybridRetriever(BM25Retriever(chunks), FaissRetriever(chunks[:2], FakeEmbedder()))


# 6. metadata preserved
def test_original_metadata_preserved(chunks):
    results = HybridRetriever.from_chunks(chunks, FakeEmbedder()).search("bamboo grain", top_k=4)
    by_id = {r.chunk_id: r for r in results}
    for record in RECORDS:
        r = by_id.get(record["record_id"])
        if r is None:
            continue
        assert r.text == record["text"] and r.source_id == record["source_id"] and r.title == record["title"]
        extra = {k: v for k, v in record.items() if k not in ("record_id", "text", "source_id", "title")}
        assert r.metadata == extra
        assert r.to_dict()["metadata"] == extra


# 7. empty query / empty corpus / invalid top_k
def test_empty_query(chunks):
    hybrid = HybridRetriever.from_chunks(chunks, FakeEmbedder())
    for query in ["", "   ", "the of a"]:  # last one is all stop-words for BM25
        assert BM25Retriever(chunks).search(query) == []
    assert hybrid.search("") == []
    assert FaissRetriever(chunks, FakeEmbedder()).search("  ") == []


def test_empty_corpus():
    hybrid = HybridRetriever.from_chunks([], FakeEmbedder())  # no model is loaded for an empty corpus
    assert hybrid.search("anything") == []
    assert BM25Retriever([]).search("anything") == []


def test_invalid_top_k(chunks):
    with pytest.raises(ValueError):
        BM25Retriever(chunks).search("bamboo", top_k=0)
    with pytest.raises(ValueError):
        HybridRetriever.from_chunks(chunks, FakeEmbedder()).search("bamboo", top_k=-1)


# 8. top_k larger than corpus
def test_top_k_larger_than_corpus(chunks):
    assert len(FaissRetriever(chunks, FakeEmbedder()).search("hand posture", top_k=100)) == 4
    assert len(BM25Retriever(chunks).search("strength grip wood bamboo", top_k=100)) <= 4
    assert len(HybridRetriever.from_chunks(chunks, FakeEmbedder()).search("bamboo grain", top_k=100)) == 4


# 9. page handling
def test_missing_page_is_not_fabricated(chunks):
    results = HybridRetriever.from_chunks(chunks, FakeEmbedder()).search("bamboo grain", top_k=4)
    by_id = {r.chunk_id: r for r in results}
    assert by_id["T-0001"].page is None
    assert "page" not in by_id["T-0001"].metadata and "page_number" not in by_id["T-0001"].metadata
    assert by_id["T-0001"].to_dict()["page"] is None


def test_existing_page_is_kept(chunks):
    r = FaissRetriever(chunks, FakeEmbedder()).search("how long does a culm need to mature", top_k=4)
    assert {x.chunk_id: x.page for x in r}["T-0003"] == 7


# determinism + real repository file (BM25 only, no model needed)
def test_repeated_searches_are_identical(chunks):
    hybrid = HybridRetriever.from_chunks(chunks, FakeEmbedder())
    assert hybrid.search("bamboo grain", 3) == hybrid.search("bamboo grain", 3)


def test_real_repository_chunks_load_and_bm25_search():
    real_path = ROOT / "data" / "processed" / "full_paper_chunks.jsonl"
    if not real_path.exists():
        pytest.skip("repository chunk file not present")
    real = load_chunks(real_path)
    assert len(real) == 194
    results = BM25Retriever(real).search("grip strength hand tool handle design", top_k=5)
    assert results and all(r.source_id in {"WBC-0051", "WBC-2016"} for r in results)
    assert all(r.page is None for r in results)  # real data has no page field
