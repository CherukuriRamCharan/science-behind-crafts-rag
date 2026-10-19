"""Build the prepared-corpus index and run grounded retrieval examples."""
from __future__ import annotations

import argparse

from .pipeline.common import load_config, write_json
from .pipeline.evaluate import evaluate, format_table
from .pipeline.generate import GroundedGenerator
from .pipeline.index_build import build_indices
from .pipeline.prepared_data import load_prepared_corpus
from .pipeline.retrieve import HybridRetriever


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", help="Configuration YAML path (defaults to craft_rag/config.yaml)")
    parser.add_argument("--no-evaluate", action="store_true", help="Skip evaluation on prepared gap questions")
    args = parser.parse_args()

    config = load_config(args.config)
    records, manifest = load_prepared_corpus(config)
    write_json(config["data"]["ingest_report"], manifest)
    index = build_indices(records, config)
    retriever = HybridRetriever(index, config)
    generator = GroundedGenerator(config)
    questions = (
        "How does hand tool design affect grip strength?",
        "Why does grain angle affect cutting across the wood grain?",
        "Why does mature bamboo density affect carving force?",
    )
    print("Backends: " + ", ".join(f"{key}={value}" for key, value in index["backend_info"].items()))
    for question in questions:
        passages = retriever.retrieve(question)
        print(f"\nQUESTION: {question}\nRETRIEVED PASSAGES:")
        for passage in passages:
            metadata = passage.get("metadata", {})
            print(
                f"  {passage['source_id']} | {passage.get('evidence_level', 'E5')} | "
                f"{metadata.get('title', 'Untitled source')} | "
                f"{passage['text_chunk'][:220]}"
            )
        print("ANSWER: " + generator.answer(question, passages))

    if not args.no_evaluate:
        evalset = []
        for record in records:
            metadata = record.get("metadata", {})
            provenance = metadata.get("provenance", {})
            associated_ids = provenance.get("associated_source_ids", [])
            if record.get("record_type") == "research_question" and associated_ids:
                evalset.append({
                    "query": record["text_chunk"],
                    "relevant_source_ids": sorted(set(associated_ids)),
                })
        if evalset:
            report = evaluate(
                config,
                index,
                generator=generator,
                evalset=evalset,
                eval_source="Prepared gap-analysis questions",
            )
            print("\nRETRIEVAL + GENERATION METRICS")
            print(format_table(report))
        else:
            print("\nNo linked prepared questions were available for evaluation.")
    print(
        f"\nIndexed {len(index['chunks'])} evidence passages and retained "
        f"{len(index['all_chunks']) - len(index['chunks'])} non-evidence passages from "
        f"{len(records)} prepared records. "
        f"Corpus report: {config['data']['ingest_report']}"
    )


if __name__ == "__main__":
    main()
