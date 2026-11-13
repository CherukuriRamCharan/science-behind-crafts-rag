"""Main evaluation runner for science-behind-crafts-rag.

Unified entry point supporting:
- Retrieval evaluation across BM25, Dense (hashing baseline), and Hybrid
- Offline Generation evaluation in --skip-llm extractive mode
- Comprehensive error analysis and failure classification
- Automated figure and report table generation

Usage:
  python src/run_evaluation.py --skip-llm
  python src/run_evaluation.py --skip-llm --retriever hybrid --top_k 5
"""
from __future__ import annotations

import argparse
import hashlib
import os
import sys
import time
from pathlib import Path

# Ensure repository root is on sys.path so modules can be imported without PYTHONPATH
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def compute_file_md5(filepath: Path) -> str:
    """Compute MD5 checksum for determinism validation."""
    hasher = hashlib.md5()
    with filepath.open("rb") as f:
        while chunk := f.read(8192):
            hasher.update(chunk)
    return hasher.hexdigest()


def _build_parser() -> argparse.ArgumentParser:
    """Return the argument parser (exposed for testing)."""
    parser = argparse.ArgumentParser(
        description="Run complete evaluation suite for craft RAG system."
    )
    parser.add_argument(
        "--skip-llm",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Run offline extractive evaluation without calling external LLM APIs.",
    )
    parser.add_argument(
        "--retriever",
        choices=["all", "hybrid", "bm25", "dense"],
        default="all",
        help="Retriever backend to evaluate (default: all).",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=5,
        help="Top-k evaluation cutoff (default: 5).",
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        default="data/eval/results",
        help="Directory to save evaluation artifacts.",
    )
    return parser


def main():
    args = _build_parser().parse_args()

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("SCIENCE-BEHIND-CRAFTS-RAG EVALUATION SUITE")
    print(f"Mode: {'OFFLINE EXTRACTIVE (--skip-llm)' if args.skip_llm else 'ONLINE LLM'}")
    print(f"Retriever target: {args.retriever} | Top-k: {args.top_k}")
    print("=" * 70)

    # 1. Dataset Validation
    print("\n[Step 1/5] Validating evaluation dataset schema and integrity...")
    from src.validate_eval_dataset import validate
    validation_ok = validate("data/eval/eval_questions.jsonl")
    if not validation_ok:
        print("ERROR: Evaluation dataset validation failed. Aborting.")
        sys.exit(1)

    # 2. Retrieval Evaluation
    print("\n[Step 2/5] Running retrieval evaluation...")
    from src.evaluate_retrieval import evaluate_retrieval
    retrieval_summaries = evaluate_retrieval(
        output_dir=str(results_dir),
        retriever=args.retriever,
        top_k=args.top_k,
    )

    # 3. Generation Evaluation
    generator_backend = "extractive" if args.skip_llm else "transformers"
    print(f"\n[Step 3/5] Running generation & citation evaluation ({generator_backend})...")
    from src.evaluate_generation import evaluate_all_generation, run_generation_evaluation
    if args.retriever == "all":
        gen_summaries = evaluate_all_generation(
            top_k=args.top_k, results_dir=str(results_dir),
            generator_backend=generator_backend,
        )
    else:
        ret_arg = "dense" if args.retriever == "dense" else args.retriever
        s, rows = run_generation_evaluation(
            results_dir=str(results_dir), top_k=args.top_k, retriever_name=ret_arg,
            generator_backend=generator_backend,
        )
        gen_summaries = [s]
        import csv
        with (results_dir / "generation_per_question.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        with (results_dir / "generation_summary.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(s.keys()))
            writer.writeheader()
            writer.writerow(s)

    # 4. Error Analysis
    print("\n[Step 4/5] Running error analysis & failure classification...")
    from src.analyze_errors import run_error_analysis
    failures = run_error_analysis(results_dir=str(results_dir))

    # 5. Chart Generation
    print("\n[Step 5/5] Generating visual performance figures...")
    from src.generate_charts import generate_all_charts
    chart_paths = generate_all_charts(results_dir=str(results_dir))
    print(f"Generated {len(chart_paths)} figures in {results_dir / 'figures'}")

    from craft_rag.pipeline.common import load_config
    from craft_rag.pipeline.prepared_data import load_prepared_corpus
    from craft_rag.pipeline.chunk import chunk_records
    from src.run_manifest import write_run_manifest
    config = load_config()
    config["generator"]["backend"] = generator_backend
    records, _ = load_prepared_corpus(config)
    chunks = chunk_records(records, **config.get("chunking", {}))
    write_run_manifest(results_dir, config=config, corpus_chunks=chunks)

    print("\n" + "=" * 70)
    print("EVALUATION RUN COMPLETE")
    print("=" * 70)
    print("\nSummary of Generated Artifacts:")
    for f in sorted(results_dir.glob("*.csv")):
        md5 = compute_file_md5(f)
        print(f"  - {f.name:35s} (MD5: {md5})")
    for f in sorted(results_dir.glob("*.json")):
        md5 = compute_file_md5(f)
        print(f"  - {f.name:35s} (MD5: {md5})")


if __name__ == "__main__":
    main()
