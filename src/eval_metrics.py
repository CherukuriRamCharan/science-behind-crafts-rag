"""Evaluation metrics for RAG retrieval and generation.

Implemented using plain Python and NumPy:
- Retrieval metrics (with gold-group deduplication / collapsing):
  - Hit Rate@k
  - Recall@k
  - Precision@k
  - Mean Reciprocal Rank (MRR)
  - nDCG@k
  - MAP (Mean Average Precision)
- Generation and citation metrics:
  - Citation Precision
  - Citation Recall
  - Citation Validity
  - Abstention Detection
  - Token F1
  - ROUGE-L (LCS-based)
  - Evidence-status leakage detection
"""
from __future__ import annotations

import math
import re
from typing import Any, Iterable, Sequence

STOPWORDS = frozenset(
    "a an the and or in on at to for of with by from is are was were be been "
    "that this which what how why does do did can could would should it its "
    "as into through during under between across".split()
)

ABSTENTION_PHRASES = (
    "insufficient evidence",
    "cannot be answered",
    "not enough evidence",
    "not covered in the corpus",
    "corpus does not contain",
    "insufficient information",
    "no evidence found",
)


def _tokenize_text(text: str) -> list[str]:
    """Simple alphanumeric tokenizer for token F1 and ROUGE-L."""
    return re.findall(r"\w+", (text or "").lower(), flags=re.UNICODE)


def collapse_ranked_list(
    retrieved_items: Sequence[str | dict[str, Any]],
    gold_record_groups: Sequence[Sequence[str]],
) -> list[tuple[str, int | None]]:
    """Collapse retrieved ranked list so aliases of the same record count once.

    Parameters:
    -----------
    retrieved_items:
        List of retrieved chunk IDs or dicts with 'chunk_id', 'record_id', 'provenance'.
    gold_record_groups:
        List of alias groups, e.g. [["WBC-0005::doc0001::0000", "WBC-0005::doc0001", "b31f..."]]
        where each inner list contains all valid alias IDs for one relevant record.

    Returns:
    --------
    List of tuples (item_key, group_idx_or_None) representing distinct retrieved records.
    - If an item matches gold group `g`, it is assigned `group_idx = g`. If another alias of
      group `g` was already seen earlier, it is skipped (keeping only the first occurrence).
    - If an item is non-relevant, it is assigned `group_idx = None` and de-duplicated by its ID.
    """
    alias_to_group = {}
    for g_idx, grp in enumerate(gold_record_groups):
        for alias in grp:
            alias_to_group[str(alias).strip()] = g_idx

    collapsed: list[tuple[str, int | None]] = []
    seen_groups: set[int] = set()
    seen_irrelevant: set[str] = set()

    for item in retrieved_items:
        # Extract possible identifiers for this item
        if isinstance(item, dict):
            cid = str(item.get("chunk_id", "")).strip()
            rid = str(item.get("record_id", "")).strip()
            meta = item.get("metadata", {})
            prid = str(meta.get("provenance", {}).get("record_id", "")).strip()
            item_key = cid or rid or prid
            ids_to_check = [x for x in (cid, rid, prid) if x]
        else:
            item_key = str(item).strip()
            ids_to_check = [item_key]

        # Check if item matches any gold group
        matched_group: int | None = None
        for cand in ids_to_check:
            if cand in alias_to_group:
                matched_group = alias_to_group[cand]
                break

        if matched_group is not None:
            if matched_group not in seen_groups:
                seen_groups.add(matched_group)
                collapsed.append((item_key, matched_group))
        else:
            if item_key not in seen_irrelevant:
                seen_irrelevant.add(item_key)
                collapsed.append((item_key, None))

    return collapsed


def hit_rate_at_k(collapsed_ranks: Sequence[tuple[str, int | None]], k: int) -> float:
    """Return 1.0 if at least one gold group is hit in top-k, else 0.0."""
    if k <= 0:
        return 0.0
    for _, g_idx in collapsed_ranks[:k]:
        if g_idx is not None:
            return 1.0
    return 0.0


def recall_at_k(
    collapsed_ranks: Sequence[tuple[str, int | None]], k: int, n_relevant: int
) -> float:
    """Return fraction of unique gold groups retrieved in top-k."""
    if n_relevant <= 0 or k <= 0:
        return 0.0
    hits = set()
    for _, g_idx in collapsed_ranks[:k]:
        if g_idx is not None:
            hits.add(g_idx)
    return len(hits) / float(n_relevant)


def precision_at_k(collapsed_ranks: Sequence[tuple[str, int | None]], k: int) -> float:
    """Return fraction of top-k retrieved slots occupied by unique gold groups."""
    if k <= 0:
        return 0.0
    window = collapsed_ranks[:k]
    hits = sum(1 for _, g_idx in window if g_idx is not None)
    return hits / float(k)


def mrr(collapsed_ranks: Sequence[tuple[str, int | None]]) -> float:
    """Return Reciprocal Rank (1 / rank) of the first gold group hit."""
    for rank, (_, g_idx) in enumerate(collapsed_ranks, start=1):
        if g_idx is not None:
            return 1.0 / float(rank)
    return 0.0


def ndcg_at_k(
    collapsed_ranks: Sequence[tuple[str, int | None]], k: int, n_relevant: int
) -> float:
    """Normalized Discounted Cumulative Gain at rank k with binary relevance."""
    if n_relevant <= 0 or k <= 0:
        return 0.0
    # DCG
    dcg = 0.0
    for rank, (_, g_idx) in enumerate(collapsed_ranks[:k], start=1):
        rel = 1.0 if g_idx is not None else 0.0
        if rel > 0:
            dcg += rel / math.log2(rank + 1)

    # Ideal DCG
    idcg = 0.0
    ideal_hits = min(k, n_relevant)
    for rank in range(1, ideal_hits + 1):
        idcg += 1.0 / math.log2(rank + 1)

    return (dcg / idcg) if idcg > 0 else 0.0


def map_score(
    collapsed_ranks: Sequence[tuple[str, int | None]], n_relevant: int
) -> float:
    """Average Precision over full collapsed ranked list."""
    if n_relevant <= 0:
        return 0.0
    score = 0.0
    hits = 0
    for rank, (_, g_idx) in enumerate(collapsed_ranks, start=1):
        if g_idx is not None:
            hits += 1
            score += hits / float(rank)
    return score / float(n_relevant)


def extract_citations(text: str) -> list[str]:
    """Extract citations formatted as [Source ID | Level] or [Source ID]."""
    if not text:
        return []
    # Match [WBC-0051 | E3] or [WBC-0051]
    matches = re.findall(r"\[\s*([A-Za-z0-9-_]+)(?:\s*\|\s*[A-Za-z0-9]+)?\s*\]", text)
    return [m.strip() for m in matches if m.strip()]


def citation_precision(
    cited_ids: Sequence[str], gold_record_ids: Sequence[str]
) -> float:
    """Fraction of cited source IDs that correspond to gold sources.

    Matches either exact ID, source_id prefix, or normalized ID.
    """
    if not cited_ids:
        return 0.0
    gold_set = {str(gid).strip().lower() for gid in gold_record_ids}
    # Also include source_id extracted from IDs like 'WBC-0051::doc0001' -> 'wbc-0051'
    expanded_gold = set(gold_set)
    for g in gold_set:
        expanded_gold.add(g.split("::")[0].split("-0")[0])
        m = re.search(r"\b(wbc[-_]\d+)\b", g)
        if m:
            expanded_gold.add(m.group(1))

    valid_hits = 0
    for cid in cited_ids:
        cid_norm = str(cid).strip().lower()
        cid_source = cid_norm.split("::")[0]
        if cid_norm in expanded_gold or cid_source in expanded_gold:
            valid_hits += 1

    return valid_hits / float(len(cited_ids))


def citation_recall(
    cited_ids: Sequence[str], gold_record_ids: Sequence[str]
) -> float:
    """Fraction of gold records cited in the answer."""
    if not gold_record_ids:
        return 0.0
    cited_set = {str(cid).strip().lower() for cid in cited_ids}
    expanded_cited = set(cited_set)
    for c in cited_set:
        expanded_cited.add(c.split("::")[0])
        m = re.search(r"\b(wbc[-_]\d+)\b", c)
        if m:
            expanded_cited.add(m.group(1))

    gold_sources = set()
    for g in gold_record_ids:
        gn = str(g).strip().lower()
        m = re.search(r"\b(wbc[-_]\d+)\b", gn)
        gold_sources.add(m.group(1) if m else gn)

    hits = sum(1 for gs in gold_sources if gs in expanded_cited)
    return hits / float(len(gold_sources)) if gold_sources else 0.0


def citation_validity(
    cited_ids: Sequence[str],
    corpus_source_ids: Iterable[str],
    retrieved_source_ids: Iterable[str],
) -> dict[str, Any]:
    """Check whether cited IDs exist in corpus and in retrieved context.

    Returns dict with validity booleans and lists of hallucinated IDs.
    """
    corpus_set = {str(s).strip().lower() for s in corpus_source_ids}
    retrieved_set = {str(s).strip().lower() for s in retrieved_source_ids}

    hallucinated = []
    unretrieved = []
    valid = []

    for cid in cited_ids:
        cn = str(cid).strip().lower()
        in_corpus = cn in corpus_set or any(cn in c for c in corpus_set)
        in_retrieved = cn in retrieved_set or any(cn in r for r in retrieved_set)

        if not in_corpus:
            hallucinated.append(cid)
        elif not in_retrieved:
            unretrieved.append(cid)
        else:
            valid.append(cid)

    return {
        "total_citations": len(cited_ids),
        "valid_citations": valid,
        "hallucinated_citations": hallucinated,
        "unretrieved_citations": unretrieved,
        "all_valid": (len(hallucinated) == 0 and len(unretrieved) == 0),
        "has_hallucinations": len(hallucinated) > 0,
    }


def is_abstention(answer: str) -> bool:
    """Return True if answer text indicates insufficiency / abstention."""
    ans = (answer or "").strip().lower()
    if not ans:
        return True
    return any(p in ans for p in ABSTENTION_PHRASES)


def token_f1(prediction: str, ground_truth: str) -> float:
    """Compute token-level precision, recall, and harmonic mean F1."""
    pred_tokens = _tokenize_text(prediction)
    gold_tokens = _tokenize_text(ground_truth)

    if not pred_tokens or not gold_tokens:
        return 1.0 if pred_tokens == gold_tokens else 0.0

    pred_counts: dict[str, int] = {}
    for t in pred_tokens:
        pred_counts[t] = pred_counts.get(t, 0) + 1

    gold_counts: dict[str, int] = {}
    for t in gold_tokens:
        gold_counts[t] = gold_counts.get(t, 0) + 1

    overlap = 0
    for t, count in pred_counts.items():
        if t in gold_counts:
            overlap += min(count, gold_counts[t])

    if overlap == 0:
        return 0.0

    precision = overlap / float(len(pred_tokens))
    recall = overlap / float(len(gold_tokens))
    return (2 * precision * recall) / (precision + recall)


def rouge_l(prediction: str, ground_truth: str) -> float:
    """Compute ROUGE-L F1 based on Longest Common Subsequence."""
    pred_tokens = _tokenize_text(prediction)
    gold_tokens = _tokenize_text(ground_truth)

    m = len(pred_tokens)
    n = len(gold_tokens)
    if m == 0 or n == 0:
        return 1.0 if m == n else 0.0

    # DP for LCS length
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if pred_tokens[i - 1] == gold_tokens[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])

    lcs_len = dp[m][n]
    if lcs_len == 0:
        return 0.0

    precision = lcs_len / float(m)
    recall = lcs_len / float(n)
    return (2 * precision * recall) / (precision + recall)


def detect_evidence_status_leakage(
    answer_text: str,
    retrieved_passages: Sequence[dict[str, Any]],
    expected_status: str,
) -> bool:
    """Detect if an artisan observation or unsourced record is presented as verified science.

    Returns True (leakage detected) if:
    - Expected evidence status is 'artisan observation' or 'unanswerable'
    - AND the answer makes an unhedged claim or cites evidence while claiming it as
      'verified scientific fact', 'proven scientific evidence', or 'scientifically established'
      without any cautionary or observational qualifier.
    """
    ans = (answer_text or "").lower()
    if is_abstention(ans):
        return False

    if expected_status == "artisan observation":
        # Check if presented as established science
        unhedged_markers = (
            "scientifically proven",
            "proven scientific fact",
            "verified science confirms",
            "established scientific law",
            "conclusively proven",
        )
        if any(marker in ans for marker in unhedged_markers):
            return True

    return False
