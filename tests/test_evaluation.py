"""Unit tests for evaluation metrics with hand-computed reference values.

Every test includes step-by-step arithmetic in comments verifying that the
functions in src/eval_metrics.py produce exact, mathematically sound outputs.
"""
from __future__ import annotations

import importlib
import ast
import inspect
import json
from pathlib import Path
import math
import pytest

from src.eval_metrics import (
    collapse_ranked_list,
    hit_rate_at_k,
    recall_at_k,
    precision_at_k,
    mrr,
    ndcg_at_k,
    map_score,
    extract_citations,
    citation_precision,
    citation_recall,
    citation_validity,
    is_abstention,
    token_f1,
    rouge_l,
    detect_evidence_status_leakage,
)


def test_collapse_ranked_list_with_aliases():
    """Verify that multiple aliases of the same gold record are collapsed into one.

    Hand calculation:
    Gold Groups:
      Group 0: ["WBC-0005::doc0001::0000", "WBC-0005::doc0001", "hash_0005"]
      Group 1: ["WBC-0004::doc0001::0000", "WBC-0004::doc0001", "hash_0004"]

    Retrieved items:
      1. "WBC-0005::doc0001::0000" -> Matches Group 0. Seen: {0}. Retain: ("WBC-0005::doc0001::0000", 0)
      2. "WBC-0005::doc0001"        -> Matches Group 0. Already in seen {0} -> SKIP duplicate alias!
      3. "WBC-9999::doc0001::0000" -> Irrelevant. Seen: {"WBC-9999..."}. Retain: ("WBC-9999...", None)
      4. "hash_0004"               -> Matches Group 1. Seen: {0, 1}. Retain: ("hash_0004", 1)
      5. "WBC-9999::doc0001::0000" -> Irrelevant duplicate -> SKIP!

    Expected collapsed list length: 3 distinct records
      Rank 1: ("WBC-0005::doc0001::0000", 0) -> relevant
      Rank 2: ("WBC-9999::doc0001::0000", None) -> non-relevant
      Rank 3: ("hash_0004", 1) -> relevant
    """
    gold_groups = [
        ["WBC-0005::doc0001::0000", "WBC-0005::doc0001", "hash_0005"],
        ["WBC-0004::doc0001::0000", "WBC-0004::doc0001", "hash_0004"],
    ]
    retrieved = [
        "WBC-0005::doc0001::0000",
        "WBC-0005::doc0001",
        "WBC-9999::doc0001::0000",
        "hash_0004",
        "WBC-9999::doc0001::0000",
    ]

    collapsed = collapse_ranked_list(retrieved, gold_groups)
    assert len(collapsed) == 3
    assert collapsed[0] == ("WBC-0005::doc0001::0000", 0)
    assert collapsed[1] == ("WBC-9999::doc0001::0000", None)
    assert collapsed[2] == ("hash_0004", 1)


def test_hand_computed_retrieval_metrics():
    """Verify HitRate, Recall, Precision, MRR, nDCG, MAP against manual arithmetic.

    Setup:
      Total relevant items (n_relevant) = 2
      Collapsed ranked list (length 4):
        Rank 1: irrelevant  (rel = 0)
        Rank 2: Group 0     (rel = 1)
        Rank 3: irrelevant  (rel = 0)
        Rank 4: Group 1     (rel = 1)

    Manual Calculations:
      k = 1:
        Hits in top-1 = 0
        HitRate@1  = 0.0
        Recall@1   = 0 / 2 = 0.0
        Precision@1= 0 / 1 = 0.0

      k = 2:
        Hits in top-2 = {Group 0} = 1 hit
        HitRate@2  = 1.0
        Recall@2   = 1 / 2 = 0.5
        Precision@2= 1 / 2 = 0.5

      k = 3:
        Hits in top-3 = {Group 0} = 1 hit
        HitRate@3  = 1.0
        Recall@3   = 1 / 2 = 0.5
        Precision@3= 1 / 3 = 0.3333333333...

      k = 4:
        Hits in top-4 = {Group 0, Group 1} = 2 hits
        HitRate@4  = 1.0
        Recall@4   = 2 / 2 = 1.0
        Precision@4= 2 / 4 = 0.5

      MRR:
        First hit is at Rank 2.
        MRR = 1 / 2 = 0.5

      nDCG@4:
        DCG@4 = (0 / log2(2)) + (1 / log2(3)) + (0 / log2(4)) + (1 / log2(5))
              = 0 + (1 / 1.5849625) + 0 + (1 / 2.321928)
              = 0.63092975 + 0.43067656
              = 1.0616063
        IDCG@4: 2 relevant items ideally placed at ranks 1 and 2:
        IDCG@4= (1 / log2(2)) + (1 / log2(3))
              = 1.0 + 0.63092975
              = 1.63092975
        nDCG@4= 1.0616063 / 1.63092975 = 0.6509209...

      MAP:
        Relevant hit 1 at rank 2: Precision@2 = 1 / 2 = 0.5
        Relevant hit 2 at rank 4: Precision@4 = 2 / 4 = 0.5
        AP = (0.5 + 0.5) / 2 = 0.5
    """
    collapsed = [
        ("doc_irr_1", None),
        ("doc_rel_0", 0),
        ("doc_irr_2", None),
        ("doc_rel_1", 1),
    ]
    n_rel = 2

    # k = 1
    assert hit_rate_at_k(collapsed, k=1) == 0.0
    assert recall_at_k(collapsed, k=1, n_relevant=n_rel) == 0.0
    assert precision_at_k(collapsed, k=1) == 0.0

    # k = 2
    assert hit_rate_at_k(collapsed, k=2) == 1.0
    assert recall_at_k(collapsed, k=2, n_relevant=n_rel) == 0.5
    assert precision_at_k(collapsed, k=2) == 0.5

    # k = 3
    assert hit_rate_at_k(collapsed, k=3) == 1.0
    assert recall_at_k(collapsed, k=3, n_relevant=n_rel) == 0.5
    assert pytest.approx(precision_at_k(collapsed, k=3), rel=1e-4) == 1.0 / 3.0

    # k = 4
    assert hit_rate_at_k(collapsed, k=4) == 1.0
    assert recall_at_k(collapsed, k=4, n_relevant=n_rel) == 1.0
    assert precision_at_k(collapsed, k=4) == 0.5

    # MRR
    assert mrr(collapsed) == 0.5

    # nDCG@4
    expected_dcg = (1.0 / math.log2(3)) + (1.0 / math.log2(5))
    expected_idcg = (1.0 / math.log2(2)) + (1.0 / math.log2(3))
    expected_ndcg = expected_dcg / expected_idcg
    assert pytest.approx(ndcg_at_k(collapsed, k=4, n_relevant=n_rel), rel=1e-5) == expected_ndcg

    # MAP
    assert map_score(collapsed, n_relevant=n_rel) == 0.5


def test_citation_precision_hand_computed():
    """Verify Citation Precision against hand-calculated fractions.

    Arithmetic:
    Gold Records: {"WBC-0051::doc0001", "WBC-0051"} -> Source key: "WBC-0051"
    
    Case 1:
      Cited: ["WBC-0051", "WBC-0010"] (2 citations)
      Hits: "WBC-0051" is in gold (1 hit), "WBC-0010" is not.
      Precision = 1 / 2 = 0.50

    Case 2:
      Cited: ["WBC-0051", "WBC-0010", "WBC-0020"] (3 citations)
      Hits: "WBC-0051" (1 hit).
      Precision = 1 / 3 = 0.333333...

    Case 3:
      Cited: ["WBC-0051"] (1 citation)
      Hits: 1.
      Precision = 1 / 1 = 1.00
    """
    gold = ["WBC-0051::doc0001", "WBC-0051"]

    assert citation_precision(["WBC-0051", "WBC-0010"], gold) == 0.5
    assert pytest.approx(citation_precision(["WBC-0051", "WBC-0010", "WBC-0020"], gold), rel=1e-4) == 1.0 / 3.0
    assert citation_precision(["WBC-0051"], gold) == 1.0


def test_citation_recall_hand_computed():
    """Verify Citation Recall against hand-calculated fractions.

    Arithmetic:
    Gold Records: {"WBC-0005::doc0001", "WBC-0009::doc0001"} (2 distinct gold sources)
    
    Case 1:
      Cited: ["WBC-0005", "WBC-0010"]
      Hits: "WBC-0005" matches 1 gold source. "WBC-0009" not cited.
      Recall = 1 / 2 = 0.50

    Case 2:
      Cited: ["WBC-0005", "WBC-0009", "WBC-0020"]
      Hits: both "WBC-0005" and "WBC-0009" cited (2 hits).
      Recall = 2 / 2 = 1.00

    Case 3:
      Cited: ["WBC-9999"]
      Hits: 0.
      Recall = 0 / 2 = 0.00
    """
    gold = ["WBC-0005::doc0001", "WBC-0009::doc0001"]

    assert citation_recall(["WBC-0005", "WBC-0010"], gold) == 0.5
    assert citation_recall(["WBC-0005", "WBC-0009", "WBC-0020"], gold) == 1.0
    assert citation_recall(["WBC-9999"], gold) == 0.0


def test_citation_validity_hand_computed():
    """Verify Citation Validity: detection of hallucinated and unretrieved IDs.

    Corpus contains: {"WBC-0001", "WBC-0002", "WBC-0003"}
    Retrieved context contains: {"WBC-0001", "WBC-0002"}

    Case 1: Valid citations only
      Cited: ["WBC-0001", "WBC-0002"]
      Hallucinated = []
      Unretrieved = []
      all_valid = True, has_hallucinations = False

    Case 2: Cited ID in corpus but not in retrieved context (Unretrieved citation)
      Cited: ["WBC-0001", "WBC-0003"]
      Hallucinated = []
      Unretrieved = ["WBC-0003"]
      all_valid = False, has_hallucinations = False

    Case 3: Cited ID does not exist anywhere in corpus (Fabricated / Hallucinated citation)
      Cited: ["WBC-0001", "WBC-9999-FAKE"]
      Hallucinated = ["WBC-9999-FAKE"]
      all_valid = False, has_hallucinations = True
    """
    corpus = {"WBC-0001", "WBC-0002", "WBC-0003"}
    retrieved = {"WBC-0001", "WBC-0002"}

    val1 = citation_validity(["WBC-0001", "WBC-0002"], corpus, retrieved)
    assert val1["all_valid"] is True
    assert val1["has_hallucinations"] is False
    assert len(val1["hallucinated_citations"]) == 0

    val2 = citation_validity(["WBC-0001", "WBC-0003"], corpus, retrieved)
    assert val2["all_valid"] is False
    assert val2["has_hallucinations"] is False
    assert val2["unretrieved_citations"] == ["WBC-0003"]

    val3 = citation_validity(["WBC-0001", "WBC-9999-FAKE"], corpus, retrieved)
    assert val3["all_valid"] is False
    assert val3["has_hallucinations"] is True
    assert val3["hallucinated_citations"] == ["WBC-9999-FAKE"]


def test_abstention_detection_hand_computed():
    """Verify Abstention Detection across standard phrases, casing, and negative cases.

    Tested Phrases:
      - "Insufficient evidence in the retrieved corpus to answer this question." -> True
      - "This question cannot be answered based on the provided documents." -> True
      - "Not enough evidence is present." -> True
      - "The corpus does not contain information on treadle lathes." -> True
      - "" (empty string) -> True (safely treated as abstention)
      - "Wood grain orientation affects cleavage strain under shear." -> False
    """
    assert is_abstention("Insufficient evidence in the retrieved corpus to answer this question.") is True
    assert is_abstention("This question cannot be answered based on the provided documents.") is True
    assert is_abstention("Not enough evidence is present.") is True
    assert is_abstention("The corpus does not contain information on treadle lathes.") is True
    assert is_abstention("") is True
    assert is_abstention("Wood grain orientation affects cleavage strain under shear.") is False


def test_false_abstention_rate_hand_computed():
    """Verify False-Abstention Rate calculation against manual fractions.

    Arithmetic:
      Total Answerable Questions = 4
      Answers:
        q1: "Retrieved evidence: Wood density..." -> abstained = False
        q2: "Retrieved evidence: Tool socket..."  -> abstained = False
        q3: "Insufficient evidence..."           -> abstained = True (False Abstention!)
        q4: "Retrieved evidence: Bamboo fiber..." -> abstained = False

      Total false abstentions = 1
      False Abstention Rate = 1 / 4 = 0.25 (25.0%)
    """
    answers = [
        "Retrieved evidence: Wood density governs impact splitting resistance.",
        "Retrieved evidence: Tool socket collars reduce cleavage failure.",
        "Insufficient evidence in the retrieved corpus to answer this question.",
        "Retrieved evidence: Bamboo fiber tensile strength increases with age.",
    ]
    abstained_flags = [is_abstention(a) for a in answers]
    assert abstained_flags == [False, False, True, False]
    false_abstention_rate = sum(abstained_flags) / len(abstained_flags)
    assert false_abstention_rate == 0.25


def test_evidence_status_leakage_hand_computed():
    """Verify Evidence-Status Leakage: detecting artisan observations presented as proven science.

    Criteria:
      - expected_status == 'artisan observation'
      - Contains unhedged scientific claims like 'scientifically proven', 'proven scientific fact',
        'conclusively proven' without qualifiers.

    Case 1: Leakage present
      Answer: "It is a scientifically proven fact that boiling wood in neem oil prevents borer attack."
      Expected Status: "artisan observation"
      Result: True (leakage detected)

    Case 2: No leakage (Properly hedged observation)
      Answer: "Artisans observe that seasoning timber prevents fungal borer attack."
      Expected Status: "artisan observation"
      Result: False (no leakage)

    Case 3: Abstention never counts as leakage
      Answer: "Insufficient evidence in the retrieved corpus to answer this question."
      Expected Status: "artisan observation"
      Result: False (no leakage)
    """
    leaked = "It is a scientifically proven fact that boiling wood in neem oil prevents borer attack."
    assert detect_evidence_status_leakage(leaked, [], expected_status="artisan observation") is True

    hedged = "Artisans observe that seasoning timber prevents fungal borer attack."
    assert detect_evidence_status_leakage(hedged, [], expected_status="artisan observation") is False

    abstained = "Insufficient evidence in the retrieved corpus to answer this question."
    assert detect_evidence_status_leakage(abstained, [], expected_status="artisan observation") is False


def test_edge_case_empty_citations():
    """Verify edge cases when citations list is empty.

    Arithmetic:
      Cited: []
      Gold: ["WBC-0005"]
      Precision = 0.0 (no citations made)
      Recall = 0 / 1 = 0.0
      Validity: all_valid = True (no invalid or hallucinated citations present)
    """
    assert citation_precision([], ["WBC-0005"]) == 0.0
    assert citation_recall([], ["WBC-0005"]) == 0.0

    val = citation_validity([], {"WBC-0005"}, {"WBC-0005"})
    assert val["all_valid"] is True
    assert val["has_hallucinations"] is False
    assert val["total_citations"] == 0


def test_edge_case_no_gold_items():
    """Verify edge cases when gold item list is empty (e.g. unanswerable query).

    Arithmetic:
      Cited: ["WBC-0005"]
      Gold: [] (no gold items exist)
      Precision = 0.0
      Recall = 0.0
    """
    assert citation_precision(["WBC-0005"], []) == 0.0
    assert citation_recall(["WBC-0005"], []) == 0.0

    # Retrieval metrics with n_relevant = 0
    collapsed = [("doc1", None), ("doc2", None)]
    assert hit_rate_at_k(collapsed, k=5) == 0.0
    assert recall_at_k(collapsed, k=5, n_relevant=0) == 0.0
    assert ndcg_at_k(collapsed, k=5, n_relevant=0) == 0.0
    assert map_score(collapsed, n_relevant=0) == 0.0


def test_edge_case_duplicate_aliases():
    """Verify edge case where retrieved list contains multiple duplicate aliases of gold items.

    Arithmetic:
      Gold group 0 has 3 aliases: {"WBC-0001::doc0001::0000", "WBC-0001::doc0001", "WBC-0001"}
      Retrieved list contains all 3 aliases consecutively:
        Rank 1: "WBC-0001::doc0001::0000" -> Hit group 0
        Rank 2: "WBC-0001::doc0001"        -> Duplicate of group 0 -> DROPPED
        Rank 3: "WBC-0001"                 -> Duplicate of group 0 -> DROPPED
        Rank 4: "WBC-0002::doc0001::0000" -> Non-relevant item

      Collapsed List should have length 2 (1 relevant group 0, 1 irrelevant item).
      HitRate@1 = 1.0, Precision@1 = 1.0, Recall@1 = 1.0
      Precision@2 = 1 / 2 = 0.50
    """
    gold_groups = [
        ["WBC-0001::doc0001::0000", "WBC-0001::doc0001", "WBC-0001"]
    ]
    retrieved = [
        "WBC-0001::doc0001::0000",
        "WBC-0001::doc0001",
        "WBC-0001",
        "WBC-0002::doc0001::0000",
    ]
    collapsed = collapse_ranked_list(retrieved, gold_groups)
    assert len(collapsed) == 2
    assert collapsed[0] == ("WBC-0001::doc0001::0000", 0)
    assert collapsed[1] == ("WBC-0002::doc0001::0000", None)

    assert hit_rate_at_k(collapsed, k=1) == 1.0
    assert precision_at_k(collapsed, k=1) == 1.0
    assert recall_at_k(collapsed, k=1, n_relevant=1) == 1.0
    assert precision_at_k(collapsed, k=2) == 0.5


def test_token_f1_hand_computed():
    """Verify Token F1 calculation against manual token arithmetic.

    Prediction:   "handle diameter dictates grip strength" (5 tokens)
    Ground Truth: "handle diameter and material affect grip strength" (7 tokens)

    Tokens:
      Pred: {"handle", "diameter", "dictates", "grip", "strength"} (5)
      Gold: {"handle", "diameter", "and", "material", "affect", "grip", "strength"} (7)

    Overlap = {"handle", "diameter", "grip", "strength"} -> 4 tokens
    Precision = 4 / 5 = 0.8
    Recall    = 4 / 7 = 0.57142857...
    F1        = 2 * (0.8 * (4/7)) / (0.8 + (4/7))
              = 2 * (3.2 / 7) / (5.6 / 7 + 4 / 7)
              = 2 * 3.2 / 9.6 = 6.4 / 9.6 = 2 / 3 = 0.6666666...
    """
    pred = "handle diameter dictates grip strength"
    gold = "handle diameter and material affect grip strength"
    expected_f1 = 2.0 / 3.0
    assert pytest.approx(token_f1(pred, gold), rel=1e-5) == expected_f1


def test_rouge_l_hand_computed():
    """Verify ROUGE-L calculation against manual Longest Common Subsequence.

    Prediction:   "wood handles split at the socket under impact" (8 tokens)
    Ground Truth: "wooden handles split under impact" (5 tokens)

    Tokens:
      Pred: ["wood", "handles", "split", "at", "the", "socket", "under", "impact"] (m = 8)
      Gold: ["wooden", "handles", "split", "under", "impact"] (n = 5)

    Longest Common Subsequence (order preserved):
      ["handles", "split", "under", "impact"] (length = 4)

    Precision = 4 / 8 = 0.5
    Recall    = 4 / 5 = 0.8
    ROUGE-L   = 2 * (0.5 * 0.8) / (0.5 + 0.8) = 0.8 / 1.3 = 8 / 13 = 0.6153846...
    """
    pred = "wood handles split at the socket under impact"
    gold = "wooden handles split under impact"
    expected_rouge = (2.0 * 0.5 * 0.8) / (0.5 + 0.8)
    assert pytest.approx(rouge_l(pred, gold), rel=1e-5) == expected_rouge


# Branch 1 — M26/M27/M28/M29/M35 (already fixed, kept as regression guards)
# ---------------------------------------------------------------------------

def test_m26_citation_precision_no_prefix_truncation():
    gold = ["WBC-0051::doc0005::0000"]
    assert citation_precision(["WBC-0051"], gold) == 1.0
    assert citation_precision(["WBC-0099"], gold) == 0.0


def test_m26_citation_precision_obs_prefix_id():
    gold = ["OBS:process-observation-4bf00725dc74b4::0000"]
    assert citation_precision(["OBS:process-observation-4bf00725dc74b4"], gold) == 1.0
    assert citation_precision(["OBS:process-observation-deadbeef000000"], gold) == 0.0


def test_m27_citation_validity_no_substring_match():
    corpus = {"WBC-0051", "WBC-0099"}
    retrieved = {"WBC-0051"}
    result = citation_validity(["WBC-005"], corpus, retrieved)
    assert result["has_hallucinations"] is True
    assert "WBC-005" in result["hallucinated_citations"]


def test_m27_citation_validity_exact_match_still_works():
    corpus = {"WBC-0051", "OBS:process-observation-4bf00725dc74b4"}
    retrieved = {"WBC-0051", "OBS:process-observation-4bf00725dc74b4"}
    result = citation_validity(
        ["WBC-0051", "OBS:process-observation-4bf00725dc74b4"], corpus, retrieved
    )
    assert result["all_valid"] is True
    assert result["has_hallucinations"] is False


def test_m28_citation_re_exported():
    m = importlib.import_module("src.eval_metrics")
    assert hasattr(m, "CITATION_RE"), "CITATION_RE must be exported from eval_metrics"
    assert hasattr(m.CITATION_RE, "findall"), "CITATION_RE must be a compiled regex"


def test_m28_extract_citations_wbc_with_level():
    assert extract_citations("[WBC-0051 | E3] bamboo fibre") == ["WBC-0051"]


def test_m28_extract_citations_obs_colon_id():
    assert extract_citations("[OBS:process-observation-4bf00725dc74b4 | E5] artisan note") == [
        "OBS:process-observation-4bf00725dc74b4"
    ]


def test_m28_extract_citations_gap_colon_id():
    assert extract_citations("[GAP:gap-question-00587206f73d8a | E5] research gap") == [
        "GAP:gap-question-00587206f73d8a"
    ]


def test_m28_extract_citations_catalog_colon_id():
    assert extract_citations("[CATALOG:gap-source-50a580f7ddc066 | E5] catalog entry") == [
        "CATALOG:gap-source-50a580f7ddc066"
    ]


def test_m28_extract_citations_process_observation_bare_id():
    assert extract_citations("[process-observation-4bf00725dc74b4 | E5] craft note") == [
        "process-observation-4bf00725dc74b4"
    ]


def test_m28_extract_citations_wbc_chunk_id_with_colons():
    assert extract_citations("[WBC-0051::doc0005::0000 | E3] grip strength") == [
        "WBC-0051::doc0005::0000"
    ]


def test_m28_extract_citations_multiple_mixed():
    text = (
        "Bamboo fibre [WBC-0051 | E3] is strong. "
        "Artisan note [OBS:process-observation-4bf00725dc74b4 | E5]. "
        "See also [WBC-2016::doc0004::0000 | E3]."
    )
    assert extract_citations(text) == [
        "WBC-0051",
        "OBS:process-observation-4bf00725dc74b4",
        "WBC-2016::doc0004::0000",
    ]


def test_m28_extract_citations_no_level_suffix():
    assert extract_citations("[WBC-0051] bamboo") == ["WBC-0051"]


def test_m29_leakage_fires_on_extractive_unhedged_artisan_observation():
    answer = (
        "Retrieved evidence: Boiling wood in neem oil prevents borer attack. "
        "[WBC-0001 | E5]"
    )
    assert detect_evidence_status_leakage(answer, [], expected_status="artisan observation") is True


def test_m29_leakage_does_not_fire_on_hedged_artisan_observation():
    hedged_answers = [
        "Artisans observe that boiling wood in neem oil prevents borer attack. [WBC-0001 | E5]",
        "Traditionally, craftspeople soak bamboo in salt water. [WBC-0002 | E5]",
        "Reportedly, this technique reduces cracking. [WBC-0003 | E5]",
        "According to artisans, seasoning timber prevents fungal attack. [WBC-0004 | E5]",
        "This is a craft practice passed down through generations. [WBC-0005 | E5]",
        "Anecdotally, artisans prefer aged bamboo. [WBC-0006 | E5]",
    ]
    for answer in hedged_answers:
        assert detect_evidence_status_leakage(
            answer, [], expected_status="artisan observation"
        ) is False, f"Should NOT flag hedged answer: {answer!r}"


def test_m29_leakage_does_not_fire_on_verified_science_status():
    answer = "Retrieved evidence: Bamboo fibre strength increases with age. [WBC-0051 | E3]"
    assert detect_evidence_status_leakage(answer, [], expected_status="verified science") is False


def test_m29_leakage_does_not_fire_on_abstention():
    answer = "Insufficient evidence in the retrieved corpus to answer this question."
    assert detect_evidence_status_leakage(answer, [], expected_status="artisan observation") is False

# Branch 2 — M30/M31/M32/M33  (FAILING before fixes)
# ===========================================================================

# ---------------------------------------------------------------------------
# M30 — _build_parser must exist and --skip-llm must default to False
# ---------------------------------------------------------------------------

def test_m30_build_parser_is_exported():
    """M30: run_evaluation must expose _build_parser() so the flag can be
    tested without running the full pipeline.
    Currently FAILS because _build_parser does not exist.
    """
    import src.run_evaluation as runner
    assert hasattr(runner, "_build_parser"), (
        "run_evaluation must expose _build_parser() (M30)"
    )


def test_m30_skip_llm_defaults_to_false():
    """M30: parsing [] must give skip_llm=False; parsing ['--skip-llm'] must
    give skip_llm=True.  With default=True both states are unreachable.
    Currently FAILS because default=True.
    """
    import src.run_evaluation as runner
    args_default = runner._build_parser().parse_args([])
    assert args_default.skip_llm is False, (
        "--skip-llm must default to False (M30)"
    )
    args_set = runner._build_parser().parse_args(["--skip-llm"])
    assert args_set.skip_llm is True


@pytest.mark.parametrize(
    ("skip_flag", "expected_backend"),
    [("--skip-llm", "extractive"), ("--no-skip-llm", "transformers")],
)
def test_m30_runner_routes_skip_llm_mode_to_generator(
    monkeypatch, tmp_path, skip_flag, expected_backend
):
    import sys
    from src import run_evaluation as runner
    from src import validate_eval_dataset, evaluate_retrieval, evaluate_generation
    from src import analyze_errors, generate_charts

    results_dir = tmp_path / "results"
    monkeypatch.setattr(
        sys, "argv",
        ["run_evaluation.py", skip_flag, "--retriever", "hybrid", "--results_dir", str(results_dir)],
    )
    monkeypatch.setattr(validate_eval_dataset, "validate", lambda _path: True)
    calls = {}
    monkeypatch.setattr(
        evaluate_retrieval,
        "evaluate_retrieval",
        lambda **kwargs: calls.setdefault("retrieval", kwargs) or [],
    )

    def run_generation_evaluation(**kwargs):
        calls["generation"] = kwargs
        return {"retriever": "hybrid", "total_questions": 1}, [{"qid": "q001"}]

    monkeypatch.setattr(evaluate_generation, "run_generation_evaluation", run_generation_evaluation)
    monkeypatch.setattr(analyze_errors, "run_error_analysis", lambda **kwargs: [])
    monkeypatch.setattr(generate_charts, "generate_all_charts", lambda **kwargs: [])
    monkeypatch.setattr(runner, "compute_file_md5", lambda _path: "test-md5")

    runner.main()

    assert calls["generation"]["generator_backend"] == expected_backend
    assert calls["generation"]["top_k"] == 5
    assert calls["retrieval"]["retriever"] == "hybrid"
    assert calls["retrieval"]["top_k"] == 5
    assert results_dir.joinpath("generation_per_question.csv").is_file()


@pytest.mark.parametrize("backend", ["extractive", "transformers"])
def test_m30_generation_evaluation_selects_requested_backend(tmp_path, monkeypatch, backend):
    from src import evaluate_generation as module

    class Generator:
        def __init__(self, config):
            self.backend = config["generator"]["backend"]

        def answer(self, question, passages):
            return "Insufficient evidence in the retrieved corpus to answer this question."

    class Retriever:
        def __init__(self, index_data, config):
            pass

        def retrieve(self, query, top_k):
            return []

    dataset = tmp_path / "questions.jsonl"
    dataset.write_text(json.dumps({
        "qid": "u1", "question": "Outside topic?", "question_type": "unanswerable",
        "document_type": "unanswerable",
    }) + "\n", encoding="utf-8")
    monkeypatch.setattr(module, "load_config", lambda: {"generator": {}})
    monkeypatch.setattr(module, "GroundedGenerator", Generator)
    monkeypatch.setattr(module, "load_prepared_corpus", lambda config: ([], None))
    monkeypatch.setattr(module, "build_indices", lambda corpus, config: {})
    monkeypatch.setattr(module, "HybridRetriever", Retriever)

    summary, _ = module.run_generation_evaluation(
        eval_file=str(dataset), results_dir=str(tmp_path / backend),
        retriever_name="hybrid", generator_backend=backend,
    )
    assert summary["backend"] == backend


# ---------------------------------------------------------------------------
# M31 — evaluate_retrieval must accept retriever and top_k
# ---------------------------------------------------------------------------

def test_m31_evaluate_retrieval_accepts_retriever_and_top_k():
    """M31: evaluate_retrieval() must have 'retriever' and 'top_k' parameters.
    Currently FAILS because neither parameter exists.
    """
    from src.evaluate_retrieval import evaluate_retrieval
    sig = inspect.signature(evaluate_retrieval)
    params = set(sig.parameters.keys())
    assert "retriever" in params, "evaluate_retrieval must accept 'retriever' (M31)"
    assert "top_k" in params, "evaluate_retrieval must accept 'top_k' (M31)"


def test_m31_selected_retriever_receives_requested_cutoff(tmp_path, monkeypatch):
    from src import evaluate_retrieval as module

    dataset = tmp_path / "questions.jsonl"
    dataset.write_text(json.dumps({"qid": "u1", "question": "q", "question_type": "unanswerable"}) + "\n")
    calls = []
    monkeypatch.setattr(module, "setup_retrievers", lambda cfg: ({}, object()))
    monkeypatch.setattr(module, "retrieve_bm25", lambda query, data, candidate_k: calls.append(candidate_k) or [])

    result = module.evaluate_retrieval(
        dataset_path=str(dataset), output_dir=str(tmp_path / "out"), retriever="bm25", top_k=7
    )

    assert calls == [7]
    assert result["n_questions"] == 1
    assert {row["retriever"] for row in result["summary"]} == set()


# ---------------------------------------------------------------------------
# M32 — run_error_analysis must accept results_dir
# ---------------------------------------------------------------------------

def test_m32_run_error_analysis_accepts_results_dir():
    """M32: run_error_analysis must accept a results_dir argument.
    Currently FAILS because the parameter does not exist.
    """
    from src.analyze_errors import run_error_analysis
    sig = inspect.signature(run_error_analysis)
    assert "results_dir" in sig.parameters, (
        "run_error_analysis must accept 'results_dir' (M32)"
    )


def test_m32_error_analysis_reads_and_writes_selected_directory(tmp_path):
    from src.analyze_errors import run_error_analysis

    dataset = tmp_path / "questions.jsonl"
    dataset.write_text("", encoding="utf-8")
    results_dir = tmp_path / "custom-results"
    results_dir.mkdir()
    (results_dir / "retrieval_per_question.csv").write_text("qid,retriever,HitRate@5\n", encoding="utf-8")
    (results_dir / "generation_per_question.csv").write_text("qid,retriever,is_answerable,abstained\n", encoding="utf-8")

    assert run_error_analysis(results_dir=results_dir, dataset_path=dataset) == []
    assert (results_dir / "error_analysis.json").is_file()
    assert (results_dir / "error_summary.csv").is_file()


# ---------------------------------------------------------------------------
# M33 — validator must enforce total count == 45
# ---------------------------------------------------------------------------

def test_m33_validator_enforces_total_count(tmp_path):
    """M33: validate() must return False when total != 45.
    Currently FAILS because it prints the count but always returns True.
    """
    from src.validate_eval_dataset import validate

    ds = tmp_path / "eval_questions.jsonl"
    q = {
        "qid": "q001",
        "question": "How does bamboo age affect fibre strength?",
        "gold_record_ids": ["WBC-0051::doc0004::0000"],
        "gold_record_groups": [["WBC-0051::doc0004::0000"]],
        "gold_answer": "Bamboo age affects fibre strength.",
        "gold_evidence_quote": "Strength can be defined as the capacity",
        "question_type": "factual",
        "expected_evidence_level": "E3",
        "document_type": "full_paper_chunk",
        "craft": "Bamboo",
        "paraphrase_group": None,
        "origin": "auto_drafted",
        "needs_human_review": True,
        "expected_evidence_status": "verified science",
    }
    ds.write_text(json.dumps(q) + "\n", encoding="utf-8")

    result = validate(str(ds), reference_path=str(tmp_path / "no_ref.json"))
    assert result is False, "validate() must return False when total questions != 45 (M33)"


def test_m33_validator_enforces_card_grounding(tmp_path):
    """M33: validate() must return False when card grounding < 50%.
    Currently FAILS because the constraint is printed but not enforced.
    """
    from src.validate_eval_dataset import validate

    ds = tmp_path / "eval_questions.jsonl"
    lines = []

    # 33 answerable full_paper_chunk questions — 0% source_document grounding
    for i in range(1, 34):
        q = {
            "qid": f"q{i:03d}",
            "question": f"Question {i} about bamboo fibre?",
            "gold_record_ids": [f"WBC-0051::doc{i:04d}::0000"],
            "gold_record_groups": [[f"WBC-0051::doc{i:04d}::0000"]],
            "gold_answer": f"Answer {i}.",
            "gold_evidence_quote": f"quote {i}",
            "question_type": "factual",
            "expected_evidence_level": "E3",
            "document_type": "full_paper_chunk",
            "craft": "Bamboo",
            "paraphrase_group": None,
            "origin": "auto_drafted",
            "needs_human_review": True,
            "expected_evidence_status": "verified science",
        }
        lines.append(json.dumps(q))

    # 12 unanswerable questions
    for i in range(34, 46):
        q = {
            "qid": f"q{i:03d}",
            "question": f"Unanswerable question {i}?",
            "gold_record_ids": [],
            "gold_record_groups": [],
            "gold_answer": "Insufficient evidence in the retrieved corpus to answer this question.",
            "gold_evidence_quote": "",
            "question_type": "unanswerable",
            "expected_evidence_level": "None",
            "document_type": "unanswerable",
            "craft": "None",
            "paraphrase_group": None,
            "origin": "auto_drafted",
            "needs_human_review": True,
            "expected_evidence_status": "unanswerable",
        }
        lines.append(json.dumps(q))

    ds.write_text("\n".join(lines) + "\n", encoding="utf-8")

    result = validate(str(ds), reference_path=str(tmp_path / "no_ref.json"))
    assert result is False, "validate() must return False when card grounding < 50% (M33)"


def test_m33_validator_accepts_grounded_dataset_within_constraints(tmp_path):
    from src.validate_eval_dataset import validate

    dataset = tmp_path / "eval_questions.jsonl"
    reference = tmp_path / "chunks.json"
    questions = []
    chunks = []
    for i in range(1, 34):
        record_id = f"record-{i}"
        doc_type = "source_document" if i <= 17 else "full_paper_chunk"
        questions.append({
            "qid": f"q{i:03d}", "question": "How does bamboo fibre strength respond?",
            "gold_record_ids": [record_id], "gold_record_groups": [[record_id]],
            "gold_answer": "Thermal cycling changes material dimensions.",
            "gold_evidence_quote": "Thermal cycling changes material dimensions.",
            "question_type": "factual", "expected_evidence_level": "E3",
            "document_type": doc_type, "craft": "Bamboo", "paraphrase_group": None,
            "origin": "reviewed", "needs_human_review": False,
        })
        chunks.append({
            "chunk_id": record_id, "record_id": record_id, "prov_record_id": record_id,
            "document_type": doc_type, "text": "Thermal cycling changes material dimensions.",
        })
    for i in range(34, 46):
        questions.append({
            "qid": f"q{i:03d}", "question": f"Unanswerable question {i}?",
            "gold_record_ids": [], "gold_record_groups": [],
            "gold_answer": "Insufficient evidence in the retrieved corpus to answer this question.",
            "gold_evidence_quote": "", "question_type": "unanswerable",
            "expected_evidence_level": "None", "document_type": "unanswerable",
            "craft": "None", "paraphrase_group": None, "origin": "reviewed",
            "needs_human_review": False,
        })
    dataset.write_text("\n".join(json.dumps(q) for q in questions) + "\n", encoding="utf-8")
    reference.write_text(json.dumps(chunks), encoding="utf-8")

    assert validate(dataset, reference_path=reference) is True


def test_m33_overlap_constraint_is_strictly_below_half():
    from src.validate_eval_dataset import overlap_is_valid

    assert overlap_is_valid(0.4999) is True
    assert overlap_is_valid(0.50) is False
    assert overlap_is_valid(0.5001) is False


# ===========================================================================

def test_h5_categories_follow_q023_q025_q026_q031_measured_ranks(tmp_path):
    import csv
    from src.analyze_errors import run_error_analysis

    dataset = tmp_path / "questions.jsonl"
    source_questions = {
        q["qid"]: q for q in (
            json.loads(line)
            for line in Path("data/eval/eval_questions.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
    }
    qids = ("q023", "q025", "q026", "q031")
    dataset.write_text("\n".join(json.dumps(source_questions[qid]) for qid in qids) + "\n", encoding="utf-8")
    results = tmp_path / "results"
    results.mkdir()
    reciprocal_ranks = {
        "q023": {"bm25": 1 / 9, "dense (hashing baseline)": 1 / 21, "hybrid (no dedup)": 1 / 11, "hybrid (pipeline with dedup)": 0},
        "q025": {"bm25": 1, "dense (hashing baseline)": 1 / 4, "hybrid (no dedup)": 1 / 2, "hybrid (pipeline with dedup)": 0},
        "q026": {"bm25": 1 / 2, "dense (hashing baseline)": 0, "hybrid (no dedup)": 1 / 17, "hybrid (pipeline with dedup)": 0},
        "q031": {"bm25": 0, "dense (hashing baseline)": 0, "hybrid (no dedup)": 0, "hybrid (pipeline with dedup)": 0},
    }
    with (results / "retrieval_per_question.csv").open("w", newline="", encoding="utf-8") as stream:
        fields = ["qid", "retriever", "MRR"]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for qid, systems in reciprocal_ranks.items():
            for retriever, reciprocal_rank in systems.items():
                writer.writerow({"qid": qid, "retriever": retriever, "MRR": f"{reciprocal_rank:.4f}"})
    (results / "generation_per_question.csv").write_text(
        "qid,retriever,is_answerable,abstained\n", encoding="utf-8"
    )

    failures = run_error_analysis(results, dataset)
    categories = {failure["qid"]: failure["category"] for failure in failures}
    assert categories == {
        "q023": "retrieval_miss_all_systems",
        "q025": "retrieval_deduplication_drop",
        "q026": "dense_embedding_miss",
        "q031": "retrieval_miss_all_systems",
    }
    assert failures[0]["measured_ranks"]["bm25"] == 9

def test_h5_no_128dim_in_error_descriptions():
    """H5: '128-dim' must not appear in analyze_errors.py.
    Currently FAILS because the string is present in a hard-coded description.
    """
    src = Path("src/analyze_errors.py").read_text(encoding="utf-8")
    assert "128-dim" not in src, "'128-dim' must be removed from error descriptions (H5)"


def test_h5_label_fixes_not_loaded_unused():
    """H5: label_fixes is loaded but never used — remove the load or use it.
    Currently FAILS because the variable is assigned but has zero Load uses.
    """
    src = Path("src/analyze_errors.py").read_text(encoding="utf-8")
    tree = ast.parse(src)

    assigned = any(
        isinstance(t, ast.Name) and t.id == "label_fixes"
        for node in ast.walk(tree)
        if isinstance(node, ast.Assign)
        for t in node.targets
    )
    if assigned:
        load_uses = sum(
            1 for node in ast.walk(tree)
            if isinstance(node, ast.Name)
            and node.id == "label_fixes"
            and isinstance(node.ctx, ast.Load)
        )
        assert load_uses > 0, (
            "'label_fixes' is assigned but never used — remove the load or use it (H5)"
        )


def test_h5_dedup_drop_derived_from_no_dedup_comparison():
    """H5: retrieval_deduplication_drop must be detected by comparing
    no-dedup vs pipeline ranks, not by checking pipeline HitRate@5 alone.
    Currently FAILS because the source only checks the pipeline retriever.
    """
    src = Path("src/analyze_errors.py").read_text(encoding="utf-8")
    assert "no dedup" in src.lower() or "no_dedup" in src.lower(), (
        "dedup drop detection must compare no-dedup vs pipeline ranks (H5)"
    )



# Branch 3 — M34/M35/L41 regression tests
def test_m35_retrieve_hybrid_no_dedup_tie_breaking_is_by_chunk_id():
    from src.evaluate_retrieval import retrieve_hybrid_no_dedup
    import numpy as np

    class Embedder:
        def encode(self, texts):
            return np.asarray([[1.0, 0.0] for _ in texts])

    class EqualBM25:
        def get_scores(self, tokens):
            return np.asarray([1.0, 1.0])

    chunk_a = {
        "chunk_id": "AAA-chunk", "source_id": "AAA", "title": "AAA title",
        "record_id": "AAA-chunk", "record_type": "source_document",
        "text_chunk": "bamboo fibre strength material selection craft",
        "evidence_level": "E3",
        "metadata": {"evidence_eligible": True, "record_type": "source_document"},
    }
    chunk_b = {
        "chunk_id": "ZZZ-chunk", "source_id": "ZZZ", "title": "ZZZ title",
        "record_id": "ZZZ-chunk", "record_type": "source_document",
        "text_chunk": "bamboo fibre strength material selection craft",
        "evidence_level": "E3",
        "metadata": {"evidence_eligible": True, "record_type": "source_document"},
    }

    idx_az = {
        "chunks": [chunk_a, chunk_b], "embedder": Embedder(),
        "vectors": np.asarray([[1.0, 0.0], [1.0, 0.0]]),
        "faiss": None, "bm25": EqualBM25(), "tokenized": [["bamboo"], ["bamboo"]],
    }
    result_az = retrieve_hybrid_no_dedup("bamboo fibre strength", idx_az, candidate_k=2)
    idx_za = dict(idx_az, chunks=[chunk_b, chunk_a], vectors=np.asarray([[1.0, 0.0], [1.0, 0.0]]))
    result_za = retrieve_hybrid_no_dedup("bamboo fibre strength", idx_za, candidate_k=2)

    ids_az = [r["chunk_id"] for r in result_az]
    ids_za = [r["chunk_id"] for r in result_za]

    assert ids_az == ids_za, f"Ranking changed with corpus order: az={ids_az}, za={ids_za}"
    assert ids_az[0] < ids_az[1], f"Lexicographically smaller chunk_id must rank first, got {ids_az}"

@pytest.mark.parametrize(
    ("module_name", "retriever_name"),
    [
        ("src.evaluate_retrieval", "retrieve_dense"),
        ("src.evaluate_retrieval", "retrieve_bm25"),
        ("src.evaluate_generation", "retrieve_dense"),
        ("src.evaluate_generation", "retrieve_bm25"),
    ],
)
def test_m35_primitive_retriever_ties_ignore_corpus_order(module_name, retriever_name):
    evaluation = importlib.import_module(module_name)
    import numpy as np

    class Embedder:
        def encode(self, texts):
            return np.asarray([[1.0, 0.0] for _ in texts])

    class EqualBM25:
        def get_scores(self, tokens):
            return np.asarray([1.0, 1.0])

    chunks = []
    for chunk_id in ("AAA-chunk", "ZZZ-chunk"):
        chunks.append({
            "chunk_id": chunk_id, "source_id": chunk_id[:3], "title": chunk_id,
            "record_id": chunk_id, "record_type": "source_document",
            "text_chunk": "bamboo fibre strength material selection craft",
            "evidence_level": "E3",
            "metadata": {"evidence_eligible": True, "record_type": "source_document"},
        })
    def make_index(items):
        return {
            "chunks": items, "embedder": Embedder(),
            "vectors": np.asarray([[1.0, 0.0], [1.0, 0.0]]),
            "faiss": None, "bm25": EqualBM25(), "tokenized": [["bamboo"], ["bamboo"]],
        }
    fn = getattr(evaluation, retriever_name)
    first = fn("bamboo fibre strength", make_index(chunks), candidate_k=2)
    second = fn("bamboo fibre strength", make_index(list(reversed(chunks))), candidate_k=2)
    assert [r["chunk_id"] for r in first] == [r["chunk_id"] for r in second]
    assert [r["chunk_id"] for r in first] == ["AAA-chunk", "ZZZ-chunk"]

def test_m34_chart_sample_labels_follow_summary_data(tmp_path, monkeypatch):
    import csv
    import matplotlib.pyplot as plt
    from src.generate_charts import generate_all_charts

    charts = []
    monkeypatch.setattr(
        plt, "savefig",
        lambda path: charts.append((plt.gcf().axes[0].get_title(), plt.gcf().axes[0].get_ylim()[0])),
    )
    summary = tmp_path / "retrieval_summary.csv"
    fields = ["retriever", "slice", "n_questions", "MRR", "HitRate@5", "Recall@1", "Recall@3", "Recall@5", "Recall@10"]
    rows = [
        {"retriever": "bm25", "slice": "Overall", "n_questions": "8", "MRR": ".5", "HitRate@5": ".5", "Recall@1": ".2", "Recall@3": ".3", "Recall@5": ".4", "Recall@10": ".5"},
        {"retriever": "bm25", "slice": "question_type:factual", "n_questions": "8", "MRR": ".5", "HitRate@5": ".5", "Recall@1": ".2", "Recall@3": ".3", "Recall@5": ".4", "Recall@10": ".5"},
    ]
    for count in (8, 9):
        rows[0]["n_questions"] = str(count)
        with summary.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
        generate_all_charts(str(tmp_path))
    assert "N=8" in charts[0][0]
    assert "N=9" in charts[3][0]
    assert all(bottom == 0 for _, bottom in charts)

def test_m34_n_answerable_not_hardcoded():
    """L41: evaluate_retrieval must not hard-code n_answerable as len(questions) - 12.
    Currently FAILS because the source contains exactly that expression.
    """
    src = Path("src/evaluate_retrieval.py").read_text(encoding="utf-8")
    assert "len(questions) - 12" not in src, (
        "n_answerable must not be hard-coded as len(questions) - 12 (L41)"
    )


def test_m34_chart_recall_yaxis_starts_at_zero():
    """M34: the Recall@k chart y-axis must start at 0."""
    src = Path("src/generate_charts.py").read_text(encoding="utf-8")
    assert "ylim(0.5" not in src, (
        "Recall@k chart y-axis must start at 0, not 0.5 (M34)"
    )


def test_m34_chart_n_labels_not_hardcoded():
    """M34: chart labels must derive sample counts from results data."""
    src = Path("src/generate_charts.py").read_text(encoding="utf-8")
    for hard_coded in ["N=33", "N=18", "N=10", "N=5)", "N=14", "N=8)", "N=6)", "N=4)", "N=1)"]:
        assert hard_coded not in src, (
            f"Hard-coded label '{hard_coded}' must be replaced with data-derived count (M34)"
        )
