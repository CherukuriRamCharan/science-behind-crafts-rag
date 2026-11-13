"""Classify evaluation failures from measured retrieval ranks and answers."""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

RETRIEVER_ALIASES = {
    "bm25": "bm25",
    "dense (hashing baseline)": "dense",
    "dense": "dense",
    "hybrid (no dedup)": "hybrid_no_dedup",
    "hybrid (pipeline with dedup)": "hybrid_pipeline",
    "hybrid": "hybrid_pipeline",
}
TOP_K = 5


def _measured_rank(row: dict[str, Any]) -> int | None:
    """Read the explicit rank, with MRR fallback for legacy result CSVs."""
    raw_rank = row.get("gold_rank")
    if raw_rank not in (None, "", "None"):
        return int(raw_rank)
    mrr = float(row.get("MRR") or 0)
    return round(1 / mrr) if mrr > 0 else None


def _embedding_dimension() -> tuple[int | None, str]:
    """Measure output width from the configured embedder used by the pipeline."""
    try:
        from craft_rag.pipeline.common import load_config
        from craft_rag.pipeline.embed_text import TextEmbedder

        embedder = TextEmbedder(load_config())
        vectors = embedder.encode(["dimension probe"])
        return int(vectors.shape[1]), embedder.backend
    except (ImportError, OSError, RuntimeError, ValueError, KeyError, TypeError):
        return None, "configured embedding backend"


def run_error_analysis(results_dir: str | Path = "data/eval/results",
                       dataset_path: str | Path = "data/eval/eval_questions_v2.jsonl"):
    results_dir = Path(results_dir)
    with Path(dataset_path).open("r", encoding="utf-8") as f:
        questions = {q["qid"]: q for q in (json.loads(line) for line in f if line.strip())}

    with (results_dir / "retrieval_per_question.csv").open("r", encoding="utf-8-sig") as f:
        retrieval_rows = list(csv.DictReader(f))

    with (results_dir / "generation_per_question.csv").open("r", encoding="utf-8-sig") as f:
        generation_rows = list(csv.DictReader(f))

    ranks: dict[str, dict[str, int | None]] = {}
    for row in retrieval_rows:
        system = RETRIEVER_ALIASES.get(row.get("retriever", ""))
        if system is not None:
            ranks.setdefault(row["qid"], {})[system] = _measured_rank(row)

    failures = []
    dimension, backend = _embedding_dimension()
    dimension_label = f"{dimension}-dim {backend}" if dimension is not None else backend

    for qid, systems in ranks.items():
        q = questions.get(qid)
        if q is None:
            continue
        if q.get("question_type") == "unanswerable":
            continue
        ordered_systems = ("bm25", "dense", "hybrid_no_dedup", "hybrid_pipeline")
        measured = {name: systems.get(name) for name in ordered_systems}
        rank_text = ", ".join(
            f"{name}={'miss' if rank is None else rank}"
            for name, rank in measured.items()
        )
        category = None
        description = ""
        remedy = ""

        top_hits = {
            name: rank if rank is not None and rank <= TOP_K else None
            for name, rank in measured.items()
        }
        if all(rank is None for rank in top_hits.values()):
            category = "retrieval_miss_all_systems"
            description = f"No gold record was retrieved by any evaluated system (ranks: {rank_text})."
            remedy = "Review corpus coverage, chunking, and the question's gold evidence mapping."
        elif (top_hits["hybrid_no_dedup"] is not None
              and top_hits["hybrid_pipeline"] is None):
            category = "retrieval_deduplication_drop"
            description = (
                f"Gold ranked within top {TOP_K} before deduplication but not in the pipeline result "
                f"(ranks: {rank_text})."
            )
            remedy = "Inspect the deduplication decision for the measured gold record and its competing records."
        elif top_hits["bm25"] is not None and top_hits["dense"] is None:
            category = "dense_embedding_miss"
            description = (
                f"BM25 retrieved gold at rank {measured['bm25']} while dense retrieval missed it "
                f"using the measured {dimension_label} configuration (ranks: {rank_text})."
            )
            remedy = "Review the embedding model, corpus representation, and query-document similarity."

        if category:
            failures.append({
                "qid": qid,
                "category": category,
                "component": "retriever",
                "question": q.get("question", ""),
                "question_type": q.get("question_type", ""),
                "document_type": q.get("document_type", ""),
                "gold_ids": q.get("gold_record_ids", []),
                "gold_quote": q.get("gold_evidence_quote", ""),
                "severity": "high" if category != "dense_embedding_miss" else "medium",
                "description": description,
                "remedy": remedy,
                "measured_ranks": measured,
            })

    for row in generation_rows:
        is_answerable = str(row.get("is_answerable", "")).lower() == "true"
        abstained = str(row.get("abstained", "")).lower() == "true"
        if (row.get("retriever") in ("hybrid (pipeline with dedup)", "hybrid")
                and not is_answerable and not abstained):
            qid = row.get("qid")
            q = questions.get(qid, {})
            failures.append({
                "qid": qid,
                "category": "generation_abstention_failure",
                "component": "generator",
                "question": q.get("question", ""),
                "question_type": q.get("question_type", ""),
                "document_type": q.get("document_type", ""),
                "gold_ids": [],
                "gold_quote": "",
                "severity": "medium",
                "description": f"Generator did not abstain for unanswerable question {qid}.",
                "remedy": "Review evidence sufficiency and abstention threshold calibration.",
            })

    out_file = results_dir / "error_analysis.json"
    with out_file.open("w", encoding="utf-8") as f:
        json.dump(failures, f, indent=2)

    category_counts: dict[str, int] = {}
    components: dict[str, str] = {}
    for failure in failures:
        category = failure["category"]
        category_counts[category] = category_counts.get(category, 0) + 1
        components[category] = failure["component"]

    summary_file = results_dir / "error_summary.csv"
    with summary_file.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["error_category", "component", "count"])
        for category, count in category_counts.items():
            writer.writerow([category, components[category], count])

    print("Error analysis complete:")
    for category, count in category_counts.items():
        print(f"  {category}: {count}")
    return failures


if __name__ == "__main__":
    run_error_analysis()
