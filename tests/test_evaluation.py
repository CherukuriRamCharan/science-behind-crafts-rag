"""Unit tests for evaluation metrics with hand-computed reference values.

Every test includes step-by-step arithmetic in comments verifying that the
functions in src/eval_metrics.py produce exact, mathematically sound outputs.
"""
from __future__ import annotations

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
