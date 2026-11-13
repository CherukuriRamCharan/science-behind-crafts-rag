"""Tests for findings H3 (filters, results count, per-source cap) and M35 (deterministic ties)."""
from __future__ import annotations

import numpy as np
import pytest
from rank_bm25 import BM25Okapi

from craft_rag.pipeline.common import tokens
from craft_rag.pipeline.facets import (
    facet_options, facet_value, normalize_craft, normalize_filter_value, normalize_theme,
)
from craft_rag.pipeline.retrieve import HybridRetriever

CONFIG = {"retrieval": {"top_k": 5, "candidate_k": 20, "rrf_k": 60, "minimum_score": 0.0}}


class FakeEmbedder:
    def __init__(self, vector):
        self.vector = np.asarray(vector, dtype=np.float32)

    def encode(self, texts):
        return np.tile(self.vector, (len(texts), 1))


def make_index(rows, vectors=None, query_vector=(1.0, 0.0)):
    """rows: dicts with chunk_id, source_id, text_chunk and optional facet fields."""
    chunks = [{"evidence_level": "E3", "grade": "B", "metadata": {}, **row} for row in rows]
    if vectors is None:
        vectors = np.ones((len(chunks), 2), dtype=np.float32)  # every dense score identical
    tokenized = [tokens(row["text_chunk"]) for row in chunks]
    return {
        "chunks": chunks,
        "vectors": np.asarray(vectors, dtype=np.float32),
        "embedder": FakeEmbedder(query_vector),
        "faiss": None,
        "bm25": BM25Okapi(tokenized),
        "tokenized": tokenized,
    }


def corpus(n_sources=12, per_source=3, **extra):
    rows = []
    for source in range(n_sources):
        for part in range(per_source):
            rows.append({
                "chunk_id": f"S{source:02d}-{part}",
                "source_id": f"S{source:02d}",
                "text_chunk": f"bamboo grip strength handle {source} part {part}",
                "craft_cluster": "Bamboo" if source % 2 == 0 else "Wood (walnut)",
                **extra,
            })
    return rows


def ids(results):
    return [row["chunk_id"] for row in results]


# ---- H3: result count -----------------------------------------------------------------------

def test_top_k_10_returns_10_when_enough_chunks_match():
    retriever = HybridRetriever(make_index(corpus()), CONFIG)
    assert len(retriever.retrieve("bamboo grip strength handle", top_k=10)) == 10


def test_unfiltered_results_are_unique_chunks():
    retriever = HybridRetriever(make_index(corpus()), CONFIG)
    found = ids(retriever.retrieve("bamboo grip", top_k=10))
    assert len(found) == len(set(found))


# ---- H3: filters ----------------------------------------------------------------------------

def test_filter_returns_every_matching_chunk_available():
    index = make_index(corpus(n_sources=6, per_source=3))
    retriever = HybridRetriever(index, CONFIG)
    # bamboo sources are 0, 2, 4 -> 9 chunks; ask for more than exist, cap lifted
    found = retriever.retrieve("grip", top_k=50, filters={"craft_cluster": "bamboo"}, max_per_source=None)
    assert sorted(ids(found)) == sorted(r["chunk_id"] for r in index["chunks"] if r["source_id"] in {"S00", "S02", "S04"})


def test_filter_is_applied_before_ranking_so_it_does_not_starve():
    rows = corpus(n_sources=40, per_source=1)  # 20 bamboo, 20 wood
    # make every wood chunk the best lexical match so a post-filter would drop all top candidates
    for row in rows:
        if row["craft_cluster"].startswith("Wood"):
            row["text_chunk"] = "grip " * 10 + row["text_chunk"]
    retriever = HybridRetriever(make_index(rows), CONFIG)
    found = retriever.retrieve("grip", top_k=10, filters={"craft_cluster": "Bamboo"})
    assert len(found) == 10
    assert all(normalize_craft(row["craft_cluster"]) == "bamboo" for row in found)


def test_filter_with_no_match_returns_empty():
    retriever = HybridRetriever(make_index(corpus()), CONFIG)
    assert retriever.retrieve("grip", filters={"why_theme": "11"}) == []


def test_legacy_free_text_filter_values_still_work():
    rows = corpus(n_sources=4, per_source=1)
    rows[0]["craft_cluster"] = "Bamboo (Battal chisel handles used in root carving)"
    retriever = HybridRetriever(make_index(rows), CONFIG)
    found = retriever.retrieve("grip", filters={"craft_cluster": "Bamboo (something else entirely)"}, top_k=10)
    assert "S00-0" in ids(found)
    assert all(facet_value(row, "craft_cluster") == "bamboo" for row in found)


def test_theme_filter_matches_numbered_labels():
    rows = corpus(n_sources=4, per_source=1)
    rows[0]["why_theme"] = "7. Posture, Grip and Body Mechanics"
    rows[1]["why_theme"] = "7"
    rows[2]["why_theme"] = "8. Noise"
    retriever = HybridRetriever(make_index(rows), CONFIG)
    found = retriever.retrieve("grip", top_k=10, filters={"why_theme": "7. Posture, Grip and Body Mechanics"})
    assert sorted(ids(found)) == ["S00-0", "S01-0"]


# ---- H3: per-source cap ---------------------------------------------------------------------

def test_max_per_source_is_respected():
    retriever = HybridRetriever(make_index(corpus(n_sources=5, per_source=6)), CONFIG)
    found = retriever.retrieve("bamboo grip", top_k=10, max_per_source=2)
    counts = {}
    for row in found:
        counts[row["source_id"]] = counts.get(row["source_id"], 0) + 1
    assert max(counts.values()) <= 2
    assert len(found) == 10


def test_one_long_source_can_contribute_several_chunks():
    """Old behaviour kept 1 chunk per source/title; the right chunk could be lost."""
    rows = [{"chunk_id": f"P-{i}", "source_id": "PAPER", "text_chunk": f"bamboo splitting note {i}"}
            for i in range(10)]
    retriever = HybridRetriever(make_index(rows), CONFIG)
    assert len(retriever.retrieve("bamboo splitting", top_k=5, max_per_source=3)) == 3


def test_default_cap_comes_from_config():
    config = {"retrieval": {**CONFIG["retrieval"], "max_per_source": 1}}
    retriever = HybridRetriever(make_index(corpus(n_sources=5, per_source=4)), config)
    found = retriever.retrieve("bamboo grip", top_k=5)
    assert len({row["source_id"] for row in found}) == len(found) == 5


@pytest.mark.parametrize("bad", [0, -1])
def test_invalid_arguments_raise(bad):
    retriever = HybridRetriever(make_index(corpus()), CONFIG)
    with pytest.raises(ValueError):
        retriever.retrieve("grip", top_k=bad)
    with pytest.raises(ValueError):
        retriever.retrieve("grip", max_per_source=bad)


# ---- M35: deterministic ties ----------------------------------------------------------------

def test_exact_ties_are_broken_by_chunk_id_not_input_order():
    rows = [{"chunk_id": cid, "source_id": cid, "text_chunk": "identical text for every chunk " + "!" * i}
            for i, cid in enumerate(["c", "a", "d", "b", "e"])]  # deliberately unsorted
    retriever = HybridRetriever(make_index(rows), CONFIG)
    found = ids(retriever.retrieve("identical text", top_k=5, max_per_source=None))
    assert found == ["a", "b", "c", "d", "e"]


def test_results_identical_when_corpus_order_is_shuffled():
    rows = [{"chunk_id": f"c{i:02d}", "source_id": f"s{i:02d}", "text_chunk": "same words here " + "!" * i}
            for i in range(12)]
    forward = ids(HybridRetriever(make_index(rows), CONFIG).retrieve("same words", top_k=6))
    backward = ids(HybridRetriever(make_index(rows[::-1]), CONFIG).retrieve("same words", top_k=6))
    assert forward == backward == [f"c{i:02d}" for i in range(6)]


def test_repeated_calls_give_identical_results():
    retriever = HybridRetriever(make_index(corpus()), CONFIG)
    first = ids(retriever.retrieve("bamboo grip strength", top_k=8))
    assert all(ids(retriever.retrieve("bamboo grip strength", top_k=8)) == first for _ in range(5))


def test_results_expose_rank_provenance():
    found = HybridRetriever(make_index(corpus()), CONFIG).retrieve("bamboo grip", top_k=3)
    assert all(row["dense_rank"] is not None and row["sparse_rank"] is not None for row in found)


# ---- facets ---------------------------------------------------------------------------------

@pytest.mark.parametrize("raw, expected", [
    ("Wood", "wood"), ("Bamboo", "bamboo"), ("Both", "both"),
    ("Bamboo (Battal chisel handles used in root carving)", "bamboo"),
    ("Wood (tool ergonomics — applies to both wood and bamboo carving tools)", "wood"),
    ("Wood (likely comparable for bamboo dust, though not directly tested)", "wood"),
    ("Bamboo/Wood", "both"), ("Wood/Bamboo", "both"),
    ("Robotic wood carving and digital fabrication; indirect relevance", "wood"),
    ("Metalwork", "other"), ("", ""), (None, ""),
])
def test_normalize_craft(raw, expected):
    assert normalize_craft(raw) == expected


@pytest.mark.parametrize("raw, expected", [
    ("7. Posture, Grip and Body Mechanics", "7"), ("12", "12"), ("9", "9"),
    ("10. Failure, Repair and Quality (with a secondary link to 7. Tools)", "10"),
    ("Digital fabrication and tool action", "other"), ("13", "other"), ("", ""),
])
def test_normalize_theme(raw, expected):
    assert normalize_theme(raw) == expected


def test_facet_options_are_a_controlled_vocabulary():
    rows = corpus(n_sources=4, per_source=1)
    rows[0]["why_theme"] = "7. Posture"
    rows[1]["why_theme"] = "12"
    options = facet_options(rows)
    assert options["craft_clusters"] == ["wood", "bamboo"] or options["craft_clusters"] == ["bamboo", "wood"][::-1]
    assert options["why_themes"] == ["7", "12"]
    assert normalize_filter_value("craft", "Bamboo (x)") == "bamboo"
