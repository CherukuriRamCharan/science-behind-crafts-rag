"""Word-window passage splitting with complete metadata propagation."""
from __future__ import annotations

from .common import normalize_record


def chunk_records(records, words: int = 260, overlap: int = 45, max_chars: int = 1200):
    if words < 1 or overlap < 0 or overlap >= words or max_chars < 1:
        raise ValueError("Require words > 0, max_chars > 0, and 0 <= overlap < words")
    result = []
    counts = {}
    for row in records:
        source_id = str(row.get("source_id") or "UNKNOWN")
        counts[source_id] = counts.get(source_id, 0) + 1
    seen = {}
    for source in records:
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
