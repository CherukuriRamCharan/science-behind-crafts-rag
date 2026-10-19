"""Ask a grounded question against the repository's prepared craft corpus."""
from __future__ import annotations

import argparse

from .pipeline.common import load_config, write_json
from .pipeline.generate import GroundedGenerator
from .pipeline.index_build import build_indices
from .pipeline.prepared_data import load_prepared_corpus
from .pipeline.retrieve import HybridRetriever


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("question", nargs="+", help="Question to ask")
    parser.add_argument("--config", help="Configuration YAML path (defaults to craft_rag/config.yaml)")
    parser.add_argument("--craft-cluster", help="Filter to an exact craft cluster")
    parser.add_argument("--why-theme", help="Filter to an exact Why theme")
    args = parser.parse_args()

    config = load_config(args.config)
    records, report = load_prepared_corpus(config)
    write_json(config["data"]["ingest_report"], report)
    index = build_indices(records, config)
    question = " ".join(args.question)
    filters = {
        key: value
        for key, value in (
            ("craft_cluster", args.craft_cluster),
            ("why_theme", args.why_theme),
        )
        if value
    }
    passages = HybridRetriever(index, config).retrieve(question, filters=filters)
    print("Backends: " + ", ".join(f"{key}={value}" for key, value in index["backend_info"].items()))
    print(GroundedGenerator(config).answer(question, passages))
    print("\nRetrieved sources:")
    for passage in passages:
        metadata = passage.get("metadata", {})
        title = metadata.get("title", "Untitled source")
        link = passage.get("link") or metadata.get("source_url") or ""
        print(
            f"- {passage['source_id']} ({passage.get('evidence_level', 'E5')}, "
            f"{passage.get('grade', 'C')}) {title}\n  {passage['text_chunk'][:320]}"
        )
        if link:
            print(f"  Source: {link}")


if __name__ == "__main__":
    main()
