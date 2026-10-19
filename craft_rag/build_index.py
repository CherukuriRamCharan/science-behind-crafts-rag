"""Build the local hybrid index from the prepared project datasets."""
from __future__ import annotations

import argparse

from .pipeline.common import load_config, write_json
from .pipeline.index_build import build_indices
from .pipeline.prepared_data import load_prepared_corpus


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", help="Configuration YAML path (defaults to craft_rag/config.yaml)")
    args = parser.parse_args()

    config = load_config(args.config)
    records, manifest = load_prepared_corpus(config)
    write_json(config["data"]["ingest_report"], manifest)
    index = build_indices(records, config)
    print("Backends: " + ", ".join(f"{key}={value}" for key, value in index["backend_info"].items()))
    print(
        f"Built {len(index['chunks'])} evidence-eligible passages; retained "
        f"{len(index['all_chunks'])} total passages in the metadata file."
    )
    print(f"Corpus quality report: {config['data']['ingest_report']}")


if __name__ == "__main__":
    main()
