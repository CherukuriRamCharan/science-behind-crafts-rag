"""Rebuild the validator's searchable chunk reference and corpus hash."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from craft_rag.pipeline.chunk import chunk_records
from craft_rag.pipeline.common import load_config
from craft_rag.pipeline.prepared_data import load_prepared_corpus


def build_reference(
    chunks: list[dict[str, Any]], output_path: str | Path,
) -> tuple[Path, dict[str, Any]]:
    output_path = Path(output_path)
    reference = []
    for chunk in chunks:
        metadata = chunk.get("metadata", {})
        provenance = metadata.get("provenance", {})
        reference.append({
            "chunk_id": chunk.get("chunk_id"),
            "record_id": chunk.get("record_id"),
            "prov_record_id": provenance.get("record_id"),
            "source_id": chunk.get("source_id"),
            "title": metadata.get("title", ""),
            "document_type": metadata.get("document_type", chunk.get("record_type")),
            "craft": chunk.get("craft_cluster", ""),
            "evidence_level": chunk.get("evidence_level", ""),
            "text": chunk.get("text_chunk", ""),
        })
    canonical = json.dumps(reference, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    corpus_hash = hashlib.sha256(canonical).hexdigest()
    for row in reference:
        row["corpus_sha256"] = corpus_hash
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(reference, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    manifest = {
        "format_version": 1,
        "corpus_sha256": corpus_hash,
        "chunk_count": len(reference),
        "source_count": len({row["source_id"] for row in reference}),
        "builder": "src.build_searchable_chunk_reference.build_reference",
    }
    output_path.with_name("searchable_chunks_reference.manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return output_path, manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="data/eval/searchable_chunks_reference.json")
    args = parser.parse_args()
    cfg = load_config()
    records, _ = load_prepared_corpus(cfg)
    chunks = chunk_records(records, **cfg.get("chunking", {}))
    path, manifest = build_reference(chunks, args.output)
    print(f"Wrote {len(chunks)} chunks to {path}; corpus SHA-256 {manifest['corpus_sha256']}")


if __name__ == "__main__":
    main()
