"""Evaluate BM25, Dense, and Hybrid retrievers on the unified corpus.

Metrics evaluated at k in {1, 3, 5, 10}:
- Hit Rate@k
- Recall@k
- MRR
- nDCG@k

Slices:
- Overall (33 answerable questions)
- By question_type (factual, process, scientific_explanation, multi_hop)
- By document_type (source_document, full_paper_chunk, process_observation)
- Unanswerable questions: recorded separately with top retrieval scores.

Collapsing rule:
Before computing any @k metric, collapse the ranked list by gold group / record so
aliases of the same record count once (keep first occurrence) and k means k distinct records.
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from typing import Any

# Ensure repository root is on sys.path so modules can be imported without PYTHONPATH
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np

from craft_rag.pipeline.common import load_config, tokens
from craft_rag.pipeline.prepared_data import load_prepared_corpus
from craft_rag.pipeline.index_build import build_indices
from craft_rag.pipeline.retrieve import HybridRetriever as AppHybridRetriever
from src.eval_metrics import (
    collapse_ranked_list,
    hit_rate_at_k,
    recall_at_k,
    precision_at_k,
    mrr,
    ndcg_at_k,
)

K_VALUES = (1, 3, 5, 10)


def load_eval_dataset(eval_path: str = "data/eval/eval_questions.jsonl") -> list[dict[str, Any]]:
    questions = []
    with open(eval_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                questions.append(json.loads(line))
    return questions


def setup_retrievers(cfg: dict[str, Any] | None = None) -> tuple[dict[str, Any], Any]:
    if cfg is None:
        cfg = load_config()
    corpus, _ = load_prepared_corpus(cfg)
    index_data = build_indices(corpus, cfg)
    app_hybrid = AppHybridRetriever(index_data, cfg)
    return index_data, app_hybrid


def retrieve_dense(query: str, index_data: dict[str, Any], candidate_k: int = 40) -> list[dict[str, Any]]:
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


def retrieve_bm25(query: str, index_data: dict[str, Any], candidate_k: int = 40) -> list[dict[str, Any]]:
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
        return [dict(chunks[int(p)]) for p in positions]


def retrieve_hybrid(query: str, app_hybrid: Any, candidate_k: int = 40) -> list[dict[str, Any]]:
    return app_hybrid.retrieve(query, top_k=candidate_k)


def retrieve_hybrid_no_dedup(
    query: str, index_data: dict[str, Any], candidate_k: int = 40, rrf_k: int = 60
) -> list[dict[str, Any]]:
    dense_results = retrieve_dense(query, index_data, candidate_k=candidate_k)
    sparse_results = retrieve_bm25(query, index_data, candidate_k=candidate_k)
    rrf_scores: dict[str, float] = {}
    chunk_map: dict[str, dict[str, Any]] = {}
    for rank, item in enumerate(dense_results, 1):
        cid = item["chunk_id"]
        chunk_map[cid] = item
        rrf_scores[cid] = rrf_scores.get(cid, 0.0) + (1.0 / (rrf_k + rank))
    for rank, item in enumerate(sparse_results, 1):
        cid = item["chunk_id"]
        chunk_map[cid] = item
        rrf_scores[cid] = rrf_scores.get(cid, 0.0) + (1.0 / (rrf_k + rank))
    sorted_cids = sorted(rrf_scores.keys(), key=lambda c: rrf_scores[c], reverse=True)
    return [dict(chunk_map[cid], score=rrf_scores[cid]) for cid in sorted_cids[:candidate_k]]


def evaluate_retrieval(
    dataset_path: str = "data/eval/eval_questions.jsonl",
    output_dir: str = "data/eval/results",
) -> dict[str, Any]:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    questions = load_eval_dataset(dataset_path)
    cfg = load_config()
    index_data, app_hybrid = setup_retrievers(cfg)

    retrievers = {
        "bm25": lambda q: retrieve_bm25(q, index_data, candidate_k=40),
        "dense (hashing baseline)": lambda q: retrieve_dense(q, index_data, candidate_k=40),
        "hybrid (no dedup)": lambda q: retrieve_hybrid_no_dedup(q, index_data, candidate_k=40),
        "hybrid (pipeline with dedup)": lambda q: retrieve_hybrid(q, app_hybrid, candidate_k=40),
    }

    per_question_rows = []
    unanswerable_rows = []
    zero_hits = {name: [] for name in retrievers}

    # Storage for summary aggregation: mode -> slice_key -> metric_name -> list of values
    aggregated: dict[str, dict[str, dict[str, list[float]]]] = {
        mode: {} for mode in retrievers
    }

    for q in questions:
        qid = q["qid"]
        q_text = q["question"]
        q_type = q["question_type"]
        doc_type = q.get("document_type", "unknown")
        is_ans = (q_type != "unanswerable")
        gold_groups = q.get("gold_record_groups", [])
        n_rel = len(gold_groups)

        for mode_name, retrieve_fn in retrievers.items():
            retrieved = retrieve_fn(q_text)

            if is_ans:
                # Apply collapsing rule by gold group / record
                collapsed = collapse_ranked_list(retrieved, gold_groups)

                m_hit = {k: hit_rate_at_k(collapsed, k) for k in K_VALUES}
                m_rec = {k: recall_at_k(collapsed, k, n_relevant=n_rel) for k in K_VALUES}
                m_prec = {k: precision_at_k(collapsed, k) for k in K_VALUES}
                m_ndcg = {k: ndcg_at_k(collapsed, k, n_relevant=n_rel) for k in K_VALUES}
                q_mrr = mrr(collapsed)

                # Check for zero hits at k=10
                if m_hit[10] == 0.0:
                    zero_hits[mode_name].append({
                        "qid": qid,
                        "question": q_text,
                        "document_type": doc_type,
                        "gold_groups": gold_groups,
                        "top_retrieved": [c[0] for c in collapsed[:3]],
                    })

                # Record per-question row
                row = {
                    "qid": qid,
                    "retriever": mode_name,
                    "question_type": q_type,
                    "document_type": doc_type,
                    "MRR": round(q_mrr, 4),
                    "HitRate@1": round(m_hit[1], 4),
                    "HitRate@3": round(m_hit[3], 4),
                    "HitRate@5": round(m_hit[5], 4),
                    "HitRate@10": round(m_hit[10], 4),
                    "Recall@1": round(m_rec[1], 4),
                    "Recall@3": round(m_rec[3], 4),
                    "Recall@5": round(m_rec[5], 4),
                    "Recall@10": round(m_rec[10], 4),
                    "Precision@5": round(m_prec[5], 4),
                    "nDCG@5": round(m_ndcg[5], 4),
                    "nDCG@10": round(m_ndcg[10], 4),
                }
                per_question_rows.append(row)

                # Slices to aggregate: Overall, by question_type, by document_type
                slices = [
                    f"Overall",
                    f"question_type:{q_type}",
                    f"document_type:{doc_type}",
                ]
                for sl in slices:
                    s_dict = aggregated[mode_name].setdefault(sl, {})
                    s_dict.setdefault("MRR", []).append(q_mrr)
                    for k in K_VALUES:
                        s_dict.setdefault(f"HitRate@{k}", []).append(m_hit[k])
                        s_dict.setdefault(f"Recall@{k}", []).append(m_rec[k])
                        s_dict.setdefault(f"Precision@{k}", []).append(m_prec[k])
                        s_dict.setdefault(f"nDCG@{k}", []).append(m_ndcg[k])

            else:
                # Unanswerable question: record top retrieved item and score
                top_item = retrieved[0] if retrieved else {}
                top_id = top_item.get("chunk_id", "None")
                top_score = top_item.get("retrieval_score", top_item.get("score", 0.0))
                unanswerable_rows.append({
                    "qid": qid,
                    "retriever": mode_name,
                    "question": q_text,
                    "top_retrieved_id": top_id,
                    "top_score": round(float(top_score), 5),
                })

    # Save per-question CSV
    per_q_path = out_dir / "retrieval_per_question.csv"
    if per_question_rows:
        with per_q_path.open("w", encoding="utf-8-sig", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(per_question_rows[0].keys()))
            writer.writeheader()
            writer.writerows(per_question_rows)

    # Save unanswerable retrieval logs
    unans_path = out_dir / "retrieval_unanswerable_scores.csv"
    if unanswerable_rows:
        with unans_path.open("w", encoding="utf-8-sig", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(unanswerable_rows[0].keys()))
            writer.writeheader()
            writer.writerows(unanswerable_rows)

    # Build summary rows
    summary_rows = []
    for mode_name, slice_data in aggregated.items():
        for slice_name, metrics in sorted(slice_data.items()):
            n_samples = len(metrics["MRR"])
            s_row = {
                "retriever": mode_name,
                "slice": slice_name,
                "n_questions": n_samples,
                "MRR": round(float(np.mean(metrics["MRR"])), 4),
                "HitRate@1": round(float(np.mean(metrics["HitRate@1"])), 4),
                "HitRate@3": round(float(np.mean(metrics["HitRate@3"])), 4),
                "HitRate@5": round(float(np.mean(metrics["HitRate@5"])), 4),
                "HitRate@10": round(float(np.mean(metrics["HitRate@10"])), 4),
                "Recall@1": round(float(np.mean(metrics["Recall@1"])), 4),
                "Recall@3": round(float(np.mean(metrics["Recall@3"])), 4),
                "Recall@5": round(float(np.mean(metrics["Recall@5"])), 4),
                "Recall@10": round(float(np.mean(metrics["Recall@10"])), 4),
                "nDCG@1": round(float(np.mean(metrics["nDCG@1"])), 4),
                "nDCG@3": round(float(np.mean(metrics["nDCG@3"])), 4),
                "nDCG@5": round(float(np.mean(metrics["nDCG@5"])), 4),
                "nDCG@10": round(float(np.mean(metrics["nDCG@10"])), 4),
            }
            summary_rows.append(s_row)

    summary_path = out_dir / "retrieval_summary.csv"
    if summary_rows:
        with summary_path.open("w", encoding="utf-8-sig", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
            writer.writeheader()
            writer.writerows(summary_rows)

    print(f"Saved retrieval results to {per_q_path} and {summary_path}")
    return {
        "summary": summary_rows,
        "zero_hits": zero_hits,
        "n_questions": len(questions),
        "n_answerable": len(questions) - 12,
    }


if __name__ == "__main__":
    res = evaluate_retrieval()
    print("\nRETRIEVAL EVALUATION SUMMARY:")
    for r in res["summary"]:
        if r["slice"] == "Overall":
            print(f"[{r['retriever'].upper()}] MRR: {r['MRR']} | Hit@5: {r['HitRate@5']} | Rec@5: {r['Recall@5']} | nDCG@5: {r['nDCG@5']}")
