from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Iterable

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
REGISTRY_PATH = ROOT / "data" / "metadata" / "full_paper_registry.csv"
OUTPUT_PATH = ROOT / "data" / "processed" / "full_paper_chunks.jsonl"
TEXT_DIR = ROOT / "data" / "extracted" / "full_papers"


def _normalize_text(value: str) -> str:
    return re.sub(r"\s+", " ", value or "").strip()


def _chunk_paragraphs(text: str, max_chars: int = 900, overlap: int = 120) -> list[str]:
    text = _normalize_text(text)
    if not text:
        return []

    chunks = []
    start = 0

    while start < len(text):
        end = min(start + max_chars, len(text))

        if end < len(text):
            boundary = text.rfind(" ", start + max_chars // 2, end)
            if boundary > start:
                end = boundary

        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)

        if end >= len(text):
            break

        start = max(start + 1, end - overlap)

    return chunks


def _paper_metadata_by_source_id() -> dict[str, dict[str, str]]:
    registry = pd.read_csv(REGISTRY_PATH, dtype=object)
    metadata: dict[str, dict[str, str]] = {}
    for _, row in registry.iterrows():
        source_id = str(row.get("source_id", "")).strip()
        if not source_id:
            continue
        metadata[source_id] = {
            "title": str(row.get("paper_title", "")).strip(),
            "source_url": str(row.get("source_url", "")).strip(),
            "doi": str(row.get("doi", "")).strip(),
            "local_pdf_path": str(row.get("local_pdf_path", "")).strip(),
            "document_type": "Full research paper PDF",
            "craft": str(row.get("craft_applicability", "")).strip(),
            "process": str(row.get("process_stage", "")).strip(),
            "extraction_status": str(row.get("extraction_status", "")).strip(),
        }
    return metadata


def build_chunks() -> list[dict[str, object]]:
    metadata_by_id = _paper_metadata_by_source_id()
    chunks: list[dict[str, object]] = []

    if not TEXT_DIR.exists():
        return chunks

    for text_file in sorted(TEXT_DIR.glob("*.txt")):
        source_id = text_file.stem.replace("_full_text", "").replace("_robotic_wood_carving", "")
        source_id = source_id.replace("WBC-0051", "WBC-0051")
        if source_id.startswith("WBC-"):
            source_id = source_id
        else:
            # Fallback to registry match by filename suffix.
            source_id = ""
            for row_id, row in metadata_by_id.items():
                if row["title"] and row["title"].lower() in text_file.name.lower():
                    source_id = row_id
                    break
        metadata = metadata_by_id.get(source_id, {})
        text = text_file.read_text(encoding="utf-8", errors="replace")
        paragraphs = _chunk_paragraphs(text)

        for chunk_index, chunk in enumerate(paragraphs, start=1):
            record = {
                "record_id": f"{source_id or text_file.stem}-{chunk_index:04d}",
                "source_id": source_id or "",
                "title": metadata.get("title") or text_file.stem,
                "text": chunk,
                "document_type": metadata.get("document_type") or "Full research paper text",
                "source_url": metadata.get("source_url") or "",
                "local_path": metadata.get("local_pdf_path") or str(text_file),
                "source_collection": "full_papers",
                "craft": metadata.get("craft") or "",
                "region": "",
                "material": "",
                "process": metadata.get("process") or "",
                "primary_theme": "",
                "evidence_level": "",
                "evidence_status": metadata.get("extraction_status") or "",
                "review_status": "Full-paper text chunk generated from extracted text; validate before use as evidence",
                "chunk_index": chunk_index,
                "chunk_count": len(paragraphs),
            }
            chunks.append(record)

    return chunks


def main() -> None:
    records = build_chunks()
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"Wrote {len(records)} chunk records to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
