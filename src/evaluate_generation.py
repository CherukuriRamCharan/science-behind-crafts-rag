"""Evaluate generation and citation quality in offline (--skip-llm) mode.

Metrics evaluated:
- Citation Precision
- Citation Recall
- Citation Validity (identifying fabricated / hallucinated citations)
- Abstention Accuracy on unanswerable questions
- False-Abstention Rate on answerable questions
- Evidence-status leakage (artisan observations presented as verified science)
- Token F1 and ROUGE-L against gold reference answers
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path
from typing import Any

# Ensure repository root is on sys.path so modules can be imported without PYTHONPATH
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np

from craft_rag.pipeline.common import load_config, tokens
from craft_rag.pipeline.generate import GroundedGenerator
from craft_rag.pipeline.index_build import build_indices
from craft_rag.pipeline.prepared_data import load_prepared_corpus
from craft_rag.pipeline.retrieve import HybridRetriever
from src.eval_metrics import (
    citation_precision,
    citation_recall,
    citation_validity,
    detect_evidence_status_leakage,
    is_abstention,
    rouge_l,
    token_f1,
)


def extract_cited_sources(answer: str) -> list[str]:
    """Extract source identifiers from [Source ID | Level] markers in answer."""
    matches = re.findall(r"\[([^\]|]+)\s*\|\s*E[1-5]\]", answer or "")
    return [m.strip() for m in matches if m.strip()]


def retrieve_dense(query: str, index_data: dict[str, Any], candidate_k: int = 5) -> list[dict[str, Any]]:
    if not query.strip():
        return []
    chunks = index_data["chunks"]
    embedder = index_data["embedder"]
    q_vec = embedder.encode([query])[0]
    faiss_index = index_data.get("faiss")

    if faiss_index is not None:
        scores, ids = faiss_index.search(q_vec.reshape(1, -1), min(candidate_k, len(chunks)))
        results = []
        for score, idx in zip(scores[0], ids[0]):
            if idx >= 0:
                item = dict(chunks[int(idx)])
                item["score"] = float(score)
                results.append(item)
        return results
    else:
        vectors = index_data["vectors"]
        sims = vectors @ q_vec
        pos = np.argsort(-sims, kind="stable")[:candidate_k]
        results = []
        for p in pos:
            item = dict(chunks[int(p)])
            item["score"] = float(sims[p])
            results.append(item)
        return results


def retrieve_bm25(query: str, index_data: dict[str, Any], candidate_k: int = 5) -> list[dict[str, Any]]:
    if not query.strip():
        return []
    chunks = index_data["chunks"]
    bm25 = index_data.get("bm25")
    q_tokens = tokens(query)

    if bm25 is not None:
        scores = bm25.get_scores(q_tokens)
        positions = np.argsort(-scores, kind="stable")[:candidate_k]
        results = []
        for p in positions:
            item = dict(chunks[int(p)])
            item["score"] = float(scores[p])
            results.append(item)
        return results
    else:
        term_set = set(q_tokens)
        overlaps = [sum(t in term_set for t in doc) for doc in index_data["tokenized"]]
        positions = np.argsort(-np.asarray(overlaps), kind="stable")[:candidate_k]
        results = []
        for p in positions:
            item = dict(chunks[int(p)])
            item["score"] = float(overlaps[p])
            results.append(item)
        return results


def run_generation_evaluation(
    eval_file: str = "data/eval/eval_questions.jsonl",
    results_dir: str = "data/eval/results",
    top_k: int = 5,
    retriever_name: str = "hybrid",
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    config = load_config()
    config["generator"]["backend"] = "extractive"
    generator = GroundedGenerator(config)

    corpus, _ = load_prepared_corpus(config)
    index_data = build_indices(corpus, config)
    retriever = HybridRetriever(index_data, config)

    # Collect all valid corpus source IDs for citation validity check
    corpus_source_ids = set()
    for row in corpus:
        if row.get("source_id"):
            corpus_source_ids.add(str(row["source_id"]).strip())
        meta = row.get("metadata", {})
        if meta.get("provenance", {}).get("source_id"):
            corpus_source_ids.add(str(meta["provenance"]["source_id"]).strip())

    eval_path = Path(eval_file)
    with eval_path.open(encoding="utf-8") as f:
        questions = [json.loads(line) for line in f if line.strip()]

    per_question_rows = []

    answerable_count = 0
    unanswerable_count = 0
    correct_abstentions = 0
    false_abstentions = 0
    leakage_count = 0

    citation_precisions = []
    citation_recalls = []
    token_f1s = []
    rouge_ls = []
    all_hallucinations = []

    for q in questions:
        qid = q["qid"]
        q_text = q["question"]
        q_type = q["question_type"]
        doc_type = q["document_type"]
        is_ans = (q_type != "unanswerable")
        gold_ids = q.get("gold_record_ids", [])
        gold_ans = q.get("gold_answer", "")
        expected_status = q.get("expected_evidence_status", "")

        # Retrieve passages
        if retriever_name == "hybrid":
            passages = retriever.retrieve(q_text, top_k=top_k)
        elif retriever_name == "bm25":
            passages = retrieve_bm25(q_text, index_data, candidate_k=top_k)
        elif retriever_name == "dense":
            passages = retrieve_dense(q_text, index_data, candidate_k=top_k)
        else:
            raise ValueError(f"Unknown retriever: {retriever_name}")

        retrieved_source_ids = {str(p.get("source_id", "")).strip() for p in passages}

        # Generate answer
        answer = generator.answer(q_text, passages)
        abstained = is_abstention(answer)
        cited = extract_cited_sources(answer)

        # Validity check
        validity = citation_validity(cited, corpus_source_ids, retrieved_source_ids)
        if validity["hallucinated_citations"]:
            all_hallucinations.extend(validity["hallucinated_citations"])

        # Leakage check
        leakage = detect_evidence_status_leakage(answer, passages, expected_status)
        if leakage:
            leakage_count += 1

        row: dict[str, Any] = {
            "qid": qid,
            "retriever": retriever_name,
            "question_type": q_type,
            "document_type": doc_type,
            "is_answerable": is_ans,
            "abstained": abstained,
            "generated_answer": answer,
            "cited_sources": ";".join(cited),
            "citation_valid": validity["all_valid"],
            "hallucinated_citations": ";".join(validity["hallucinated_citations"]),
            "unretrieved_citations": ";".join(validity["unretrieved_citations"]),
            "evidence_status_leakage": leakage,
        }

        if is_ans:
            answerable_count += 1
            if abstained:
                false_abstentions += 1
                c_prec = 0.0
                c_rec = 0.0
                f1 = 0.0
                rouge = 0.0
            else:
                c_prec = citation_precision(cited, gold_ids)
                c_rec = citation_recall(cited, gold_ids)
                f1 = token_f1(answer, gold_ans)
                rouge = rouge_l(answer, gold_ans)

            citation_precisions.append(c_prec)
            citation_recalls.append(c_rec)
            token_f1s.append(f1)
            rouge_ls.append(rouge)

            row["citation_precision"] = round(c_prec, 4)
            row["citation_recall"] = round(c_rec, 4)
            row["token_f1"] = round(f1, 4)
            row["rouge_l"] = round(rouge, 4)
        else:
            unanswerable_count += 1
            if abstained:
                correct_abstentions += 1
            row["citation_precision"] = ""
            row["citation_recall"] = ""
            row["token_f1"] = ""
            row["rouge_l"] = ""

        per_question_rows.append(row)

    abstention_accuracy = (correct_abstentions / unanswerable_count) if unanswerable_count else 0.0
    false_abstention_rate = (false_abstentions / answerable_count) if answerable_count else 0.0
    mean_cit_prec = float(np.mean(citation_precisions)) if citation_precisions else 0.0
    mean_cit_rec = float(np.mean(citation_recalls)) if citation_recalls else 0.0
    mean_token_f1 = float(np.mean(token_f1s)) if token_f1s else 0.0
    mean_rouge_l = float(np.mean(rouge_ls)) if rouge_ls else 0.0

    retriever_label = "dense (hashing baseline)" if retriever_name == "dense" else retriever_name

    summary = {
        "retriever": retriever_label,
        "backend": "extractive (--skip-llm)",
        "total_questions": len(questions),
        "answerable_questions": answerable_count,
        "unanswerable_questions": unanswerable_count,
        "abstention_accuracy": round(abstention_accuracy, 4),
        "false_abstention_rate": round(false_abstention_rate, 4),
        "mean_citation_precision": round(mean_cit_prec, 4),
        "mean_citation_recall": round(mean_cit_rec, 4),
        "mean_token_f1": round(mean_token_f1, 4),
        "mean_rouge_l": round(mean_rouge_l, 4),
        "fabricated_citations_count": len(all_hallucinations),
        "evidence_status_leakage_count": leakage_count,
    }

    # Save outputs
    out_dir = Path(results_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    per_q_file = out_dir / f"generation_per_question_{retriever_name}.csv"
    with per_q_file.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(per_question_rows[0].keys()))
        writer.writeheader()
        writer.writerows(per_question_rows)

    return summary, per_question_rows


def evaluate_all_generation(top_k: int = 5, results_dir: str = "data/eval/results"):
    summaries = []
    all_per_q = []
    for ret in ["bm25", "dense", "hybrid"]:
        s, rows = run_generation_evaluation(results_dir=results_dir, top_k=top_k, retriever_name=ret)
        summaries.append(s)
        all_per_q.extend(rows)

    out_dir = Path(results_dir)
    summary_file = out_dir / "generation_summary.csv"
    with summary_file.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(summaries[0].keys()))
        writer.writeheader()
        writer.writerows(summaries)

    per_q_all_file = out_dir / "generation_per_question.csv"
    with per_q_all_file.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(all_per_q[0].keys()))
        writer.writeheader()
        writer.writerows(all_per_q)

    return summaries


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate generation and citation quality.")
    parser.add_argument("--retriever", choices=["all", "hybrid", "bm25", "dense"], default="all")
    parser.add_argument("--top_k", type=int, default=5)
    args = parser.parse_args()

    if args.retriever == "all":
        summaries = evaluate_all_generation(top_k=args.top_k)
        print("\n" + "=" * 60)
        print("ALL GENERATION EVALUATIONS COMPLETED (--skip-llm mode)")
        print("=" * 60)
        for s in summaries:
            print(f"[{s['retriever']}] CitPrec: {s['mean_citation_precision']} | CitRec: {s['mean_citation_recall']} | AbstAcc: {s['abstention_accuracy']} | FalseAbst: {s['false_abstention_rate']}")
    else:
        summary, rows = run_generation_evaluation(top_k=args.top_k, retriever_name=args.retriever)
        print("\n" + "=" * 60)
        print(f"GENERATION EVALUATION COMPLETED (--skip-llm mode, retriever={args.retriever})")
        print("=" * 60)
        for k, v in summary.items():
            print(f"  {k}: {v}")
