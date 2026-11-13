"""Word-window passage splitting with complete metadata propagation."""
from __future__ import annotations

import hashlib
from typing import Any

from .common import normalize_record

SHEET_PRIORITY: dict[str, int] = {
    "sheet1": 1,
    "sheet2": 2,
    "master source log": 1,
    "summary": 2,
    "gap analysis": 3,
    "primary workshop evidence index": 4,
}

DATASET_PRIORITY: dict[str, int] = {
    "documents": 1,
    "gap_analysis_sources": 2,
    "gap_analysis_questions": 3,
    "process_observations": 4,
    "full_paper_chunks": 5,
}


def _safe_int(val: Any) -> int:
    try:
        return int(val)
    except (ValueError, TypeError):
        return 0


def canonical_record_sort_key(record: dict[str, Any]) -> tuple:
    """Compute a deterministic, stable sort key for corpus records.

    Ensures chunking yields identical chunk IDs, passages, and mappings
    regardless of input record permutation.
    """
    base = normalize_record(record)
    source_id = str(record.get("source_id") or base.get("source_id") or "UNKNOWN")
    prov = base.get("metadata", {}).get("provenance", {}) if isinstance(base.get("metadata"), dict) else {}
    if not prov and isinstance(record.get("metadata"), dict) and isinstance(record["metadata"].get("provenance"), dict):
        prov = record["metadata"]["provenance"]
    dataset = str(prov.get("dataset") or record.get("source_collection") or base.get("source_collection") or "")
    d_prio = DATASET_PRIORITY.get(dataset, 9)
    sheet_name = str(prov.get("sheet_name") or record.get("sheet_name") or "").lower()
    sheet_rank = SHEET_PRIORITY.get(sheet_name, 50)
    excel_row = _safe_int(prov.get("excel_row") or record.get("excel_row"))
    chunk_index = _safe_int(prov.get("chunk_index") or record.get("chunk_index"))
    rec_id = str(record.get("record_id") or (base.get("metadata", {}).get("record_id") if isinstance(base.get("metadata"), dict) else "") or base.get("record_id") or "")
    text = str(base.get("text_chunk") or record.get("text_chunk") or record.get("text") or "")
    text_sha256 = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return (
        source_id,
        d_prio,
        sheet_rank,
        excel_row,
        chunk_index,
        rec_id,
        text_sha256,
    )


def chunk_records(records, words: int = 260, overlap: int = 45, max_chars: int = 1200):
    if words < 1 or overlap < 0 or overlap >= words or max_chars < 1:
        raise ValueError("Require words > 0, max_chars > 0, and 0 <= overlap < words")

    # Sort records deterministically to guarantee order-invariance under shuffling
    ordered_records = sorted(records, key=canonical_record_sort_key)

    result = []
    counts = {}
    for row in ordered_records:
        source_id = str(row.get("source_id") or "UNKNOWN")
        counts[source_id] = counts.get(source_id, 0) + 1
    seen = {}
    for source in ordered_records:
        base = normalize_record(source)
        source_id = base["source_id"]
        text = [
            fragment
            for word in base["text_chunk"].split()
            for fragment in (word[i:i + max_chars] for i in range(0, len(word), max_chars))
        ]
        if not text:
            continue
        seen[source_id] = seen.get(source_id, 0) + 1
        unique_source = source_id if counts[source_id] == 1 else f"{source_id}::doc{seen[source_id]:04d}"
        part = 0
        start = 0
        while start < len(text):
            end = start
            char_count = 0
            while end < len(text) and end - start < words:
                addition = len(text[end]) + (1 if end > start else 0)
                if char_count + addition > max_chars:
                    break
                char_count += addition
                end += 1
            if end == start:
                end += 1
            passage = " ".join(text[start:end])
            item = dict(base)
            item["text_chunk"] = passage
            item["chunk_id"] = f"{unique_source}::{part:04d}"
            item["record_id"] = unique_source
            item["chunk_index"] = part
            result.append(item)
            if end >= len(text):
                break
            start = max(start + 1, end - overlap)
            part += 1

    chunk_ids = [row["chunk_id"] for row in result]
    if len(chunk_ids) != len(set(chunk_ids)):
        raise ValueError("Chunk ID generation produced duplicates; source IDs or record ordering are invalid.")

    shared_fields = ("source_url", "doi", "craft", "region", "material", "process_stage", "why_theme")
    shared_metadata = {}
    for row in result:
        metadata = row["metadata"]
        title = " ".join(str(metadata.get("title", "")).split()).casefold()
        identity = (row["source_id"].casefold(), title)
        shared = shared_metadata.setdefault(identity, {})
        for key in shared_fields:
            value = metadata.get(key) or (row.get("link") if key == "source_url" else "")
            if value and key not in shared:
                shared[key] = value
    for row in result:
        metadata = row["metadata"]
        title = " ".join(str(metadata.get("title", "")).split()).casefold()
        shared = shared_metadata[(row["source_id"].casefold(), title)]
        for key, value in shared.items():
            if not metadata.get(key):
                metadata[key] = value
    return result
