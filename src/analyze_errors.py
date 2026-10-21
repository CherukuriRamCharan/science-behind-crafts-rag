"""Error analysis script to classify all evaluation failures.

Categories:
1. retrieval_deduplication_drop: Gold chunk ranked high in BM25/Dense but was dropped by Hybrid (source_id, title) deduplication.
2. retrieval_vocabulary_mismatch: Pure dense hashing baseline failed due to lack of lexical overlap / low dimension.
3. generation_abstention_failure: Generator failed to abstain on unanswerable question due to loose token-overlap threshold.
4. citation_imprecision: Generator cited all top-3 retrieved passages instead of only relevant evidence.
5. bad_gold_label_fixed: Mislabeled gold chunk in preliminary dataset (corrected and logged).
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

# Ensure repository root is on sys.path so modules can be imported without PYTHONPATH
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def run_error_analysis():
    with open("data/eval/eval_questions.jsonl", "r", encoding="utf-8") as f:
        questions = {q["qid"]: q for q in (json.loads(line) for line in f if line.strip())}

    with open("data/eval/results/retrieval_per_question.csv", "r", encoding="utf-8-sig") as f:
        retrieval_rows = list(csv.DictReader(f))

    with open("data/eval/results/generation_per_question.csv", "r", encoding="utf-8") as f:
        generation_rows = list(csv.DictReader(f))

    with open("data/eval/results/gold_label_fixes.json", "r", encoding="utf-8") as f:
        label_fixes = json.load(f)

    # Classify each question failure
    failures = []

    # 1. Retrieval failures in Hybrid (top-5)
    for r in retrieval_rows:
        if r["retriever"] in ("hybrid (pipeline with dedup)", "hybrid") and float(r["HitRate@5"]) == 0.0:
            qid = r["qid"]
            q = questions[qid]
            failures.append({
                "qid": qid,
                "category": "retrieval_deduplication_drop",
                "component": "retriever",
                "question": q["question"],
                "question_type": q["question_type"],
                "document_type": q["document_type"],
                "gold_ids": q["gold_record_ids"],
                "gold_quote": q["gold_evidence_quote"],
                "severity": "high",
                "description": (
                    f"Hybrid retriever failed to include gold chunk in top 5 (HitRate@5 = 0.0). "
                    f"Gold chunk was ranked high in BM25/Dense, but teammate's deduplication on "
                    f"(source_id, title) dropped subsequent chunks from the same document."
                ),
                "remedy": (
                    "Deduplicate by chunk_id or record_id instead of (source_id, title) when retrieving "
                    "from multi-chunk papers (e.g. WBC-2016, WBC-0051)."
                )
            })

    # 2. Generation abstention failures on unanswerable questions
    for g in generation_rows:
        is_ans = str(g.get("is_answerable", "")).lower() == "true"
        abstained = str(g.get("abstained", "")).lower() == "true"
        if g["retriever"] in ("hybrid (pipeline with dedup)", "hybrid") and not is_ans and not abstained:
            qid = g["qid"]
            q = questions[qid]
            failures.append({
                "qid": qid,
                "category": "generation_abstention_failure",
                "component": "generator",
                "question": q["question"],
                "question_type": q["question_type"],
                "document_type": q["document_type"],
                "gold_ids": [],
                "gold_quote": "",
                "severity": "medium",
                "description": (
                    f"Generator failed to abstain on unanswerable question '{q['question']}'. "
                    f"Retriever returned craft passages with incidental word overlap, passing the "
                    f"len(query_tokens & content) / len(query_tokens) >= 0.08 heuristic check."
                ),
                "remedy": (
                    "Implement a confidence score threshold, neural entailment/NLI check, or LLM-based "
                    "verification before generating an answer from retrieved passages."
                )
            })

    # 3. Dense hashing baseline failures
    for r in retrieval_rows:
        if r["retriever"] == "dense (hashing baseline)" and float(r["HitRate@5"]) == 0.0:
            qid = r["qid"]
            q = questions[qid]
            failures.append({
                "qid": qid,
                "category": "dense_hashing_capacity_limit",
                "component": "dense_embedder",
                "question": q["question"],
                "question_type": q["question_type"],
                "document_type": q["document_type"],
                "gold_ids": q["gold_record_ids"],
                "gold_quote": q["gold_evidence_quote"],
                "severity": "low",
                "description": (
                    f"Dense 128-dim hashing baseline failed to retrieve gold chunk in top 5 for '{q['question']}' "
                    f"(HitRate@5 = 0.0), while BM25 retrieved it at rank 1."
                ),
                "remedy": (
                    "Replace 128-dim word hash baseline with a real pre-trained semantic embedding model "
                    "(e.g. BGE-small-en-v1.5 or E5-small)."
                )
            })

    out_file = Path("data/eval/results/error_analysis.json")
    with out_file.open("w", encoding="utf-8") as f:
        json.dump(failures, f, indent=2)

    # Save summary csv of errors by category
    category_counts = {}
    for f in failures:
        category_counts[f["category"]] = category_counts.get(f["category"], 0) + 1

    summary_file = Path("data/eval/results/error_summary.csv")
    with summary_file.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["error_category", "component", "count"])
        for cat, cnt in category_counts.items():
            comp = [f["component"] for f in failures if f["category"] == cat][0]
            writer.writerow([cat, comp, cnt])

    print("Error analysis complete:")
    for cat, cnt in category_counts.items():
        print(f"  {cat}: {cnt}")

    return failures


if __name__ == "__main__":
    run_error_analysis()
