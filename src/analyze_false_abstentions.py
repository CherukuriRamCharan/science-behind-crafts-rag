"""Separate v2 hybrid false abstentions into top-5 hits and retrieval misses."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def split_false_abstentions(
    questions: list[dict[str, Any]],
    false_abstention_qids: list[str],
    retrieved_ids_by_qid: dict[str, list[str]],
) -> dict[str, dict[str, Any]]:
    """Classify false abstentions by exact gold-ID membership in retrieved top-k IDs."""
    by_qid = {str(question["qid"]): question for question in questions}
    retrieved_but_rejected: list[str] = []
    retrieval_misses: list[str] = []
    for qid in sorted(set(map(str, false_abstention_qids))):
        question = by_qid.get(qid)
        if question is None:
            raise ValueError(f"False-abstention question {qid!r} is absent from the dataset")
        gold_ids = set(map(str, question.get("gold_record_ids", [])))
        retrieved_ids = set(map(str, retrieved_ids_by_qid.get(qid, [])))
        if gold_ids & retrieved_ids:
            retrieved_but_rejected.append(qid)
        else:
            retrieval_misses.append(qid)
    return {
        "retrieved_but_rejected": {
            "count": len(retrieved_but_rejected),
            "question_ids": retrieved_but_rejected,
        },
        "retrieval_miss": {
            "count": len(retrieval_misses),
            "question_ids": retrieval_misses,
        },
    }


def analyze(
    dataset_path: str | Path = "data/eval/eval_questions_v2.jsonl",
    generation_path: str | Path = "data/eval/results_v2_final/generation_per_question_hybrid.csv",
    output_path: str | Path = "data/eval/results_v2_final/false_abstention_cause.json",
    top_k: int = 5,
) -> dict[str, Any]:
    """Re-run shared hybrid retrieval for v2 false abstentions and save a deterministic report."""
    if top_k < 1:
        raise ValueError("top_k must be positive")

    from craft_rag.pipeline.common import load_config
    from craft_rag.pipeline.index_build import build_indices
    from craft_rag.pipeline.prepared_data import load_prepared_corpus
    from craft_rag.pipeline.retrieve import HybridRetriever

    questions = [
        json.loads(line)
        for line in Path(dataset_path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    with Path(generation_path).open(encoding="utf-8-sig", newline="") as stream:
        generation_rows = list(csv.DictReader(stream))
    false_qids = [
        row["qid"] for row in generation_rows
        if row.get("retriever") == "hybrid"
        and row.get("is_answerable", "").casefold() == "true"
        and row.get("abstained", "").casefold() == "true"
    ]

    config = load_config()
    records, _ = load_prepared_corpus(config)
    retriever = HybridRetriever(build_indices(records, config), config)
    question_by_id = {str(question["qid"]): question for question in questions}
    retrieved_ids_by_qid: dict[str, list[str]] = {}
    for qid in sorted(set(false_qids)):
        if qid not in question_by_id:
            raise ValueError(f"False-abstention question {qid!r} is absent from the dataset")
        retrieved = retriever.retrieve(question_by_id[qid]["question"], top_k=top_k)
        ids: set[str] = set()
        for row in retrieved:
            for key in ("chunk_id", "record_id"):
                value = row.get(key)
                if value:
                    ids.add(str(value))
        retrieved_ids_by_qid[qid] = sorted(ids)

    result: dict[str, Any] = {
        "retriever": "hybrid pipeline",
        "top_k": top_k,
        **split_false_abstentions(questions, false_qids, retrieved_ids_by_qid),
    }
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="data/eval/eval_questions_v2.jsonl")
    parser.add_argument(
        "--generation-results",
        default="data/eval/results_v2_final/generation_per_question_hybrid.csv",
    )
    parser.add_argument(
        "--output",
        default="data/eval/results_v2_final/false_abstention_cause.json",
    )
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args(argv)
    print(json.dumps(analyze(args.dataset, args.generation_results, args.output, args.top_k), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
