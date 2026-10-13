"""Validate full-paper JSONL chunks in the Science Behind Crafts RAG project.
Run from the repository root: python src/validate_rag_chunks.py
Reads generated data only; writes a validation CSV and Markdown summary under data/metadata.
"""
from __future__ import annotations
import csv
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CHUNKS = ROOT / "data/processed/full_paper_chunks.jsonl"
REGISTRY = ROOT / "data/metadata/full_paper_registry.csv"
REPORT_CSV = ROOT / "data/metadata/chunk_validation.csv"
REPORT_MD = ROOT / "data/metadata/chunk_validation_report.md"

ID_KEYS = ("record_id", "chunk_id", "id")
TEXT_KEYS = ("text", "chunk_text", "content", "page_content")
SOURCE_KEYS = ("source_id", "sourceId", "source")
TITLE_KEYS = ("title", "paper_title", "source_title")
URL_KEYS = ("source_url", "url", "doi")


def first_present(row, keys):
    for key in keys:
        value = row.get(key)
        if value is not None and str(value).strip():
            return str(value).strip(), key
    return "", ""


def read_jsonl(path: Path):
    rows, errors = [], []
    if not path.exists():
        return rows, [f"Missing file: {path}"]
    with path.open(encoding="utf-8-sig") as fh:
        for line_no, line in enumerate(fh, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                if not isinstance(row, dict):
                    errors.append(f"Line {line_no}: JSON value is not an object")
                else:
                    row["__line_no"] = line_no
                    rows.append(row)
            except Exception as exc:
                errors.append(f"Line {line_no}: {type(exc).__name__}: {exc}")
    return rows, errors


def main():
    rows, parse_errors = read_jsonl(CHUNKS)
    counts = Counter()
    issues = []
    ids = defaultdict(list)
    lengths = []
    source_counts = Counter()
    per_record = []

    if not rows:
        issues.append("No chunk records found; confirm the JSONL path and file content.")

    for row in rows:
        line_no = row.get("__line_no", "")
        rid, rid_key = first_present(row, ID_KEYS)
        text, text_key = first_present(row, TEXT_KEYS)
        sid, sid_key = first_present(row, SOURCE_KEYS)
        title, title_key = first_present(row, TITLE_KEYS)
        url, url_key = first_present(row, URL_KEYS)
        counts["records"] += 1
        if not rid:
            counts["missing_chunk_id"] += 1
            issues.append(f"Line {line_no}: no chunk identifier found (checked {', '.join(ID_KEYS)})")
        else:
            ids[rid].append(line_no)
        if not text:
            counts["missing_text"] += 1
            issues.append(f"Line {line_no}: missing/empty text")
        else:
            lengths.append(len(text))
        if not sid:
            counts["missing_source_id"] += 1
            issues.append(f"Line {line_no}: no source_id-like field found")
        else:
            source_counts[sid] += 1
        if not title:
            counts["missing_title"] += 1
        if not url:
            counts["missing_url_or_doi"] += 1
        if "page_number" not in row and "page" not in row:
            counts["no_page_field"] += 1
        per_record.append({
            "jsonl_line": line_no,
            "chunk_id": rid,
            "chunk_id_field": rid_key,
            "source_id": sid,
            "title": title,
            "text_field": text_key,
            "character_count": len(text),
            "has_text": bool(text),
            "has_source_id": bool(sid),
            "has_title": bool(title),
            "has_source_url_or_doi": bool(url),
            "source_url_or_doi_field": url_key,
            "page_field_present": ("page_number" in row or "page" in row),
            "source_fields": ",".join(sorted(k for k in row if not k.startswith("__"))),
        })

    duplicates = {key: lines for key, lines in ids.items() if len(lines) > 1}
    for key, lines in duplicates.items():
        counts["duplicate_chunk_ids"] += len(lines) - 1
        issues.append(f"Duplicate chunk ID {key!r} on JSONL lines {lines}")

    # Compare source IDs with the full-paper registry where available.
    expected_ids = set()
    if REGISTRY.exists():
        try:
            import pandas as pd
            reg = pd.read_csv(REGISTRY, dtype=str).fillna("")
            if "source_id" in reg:
                expected_ids = {x.strip() for x in reg["source_id"] if x.strip()}
        except Exception as exc:
            issues.append(f"Could not read paper registry: {type(exc).__name__}: {exc}")
    seen_sources = set(source_counts)
    if expected_ids:
        for missing in sorted(expected_ids - seen_sources):
            issues.append(f"Registry source ID has no chunks: {missing}")
        for extra in sorted(seen_sources - expected_ids):
            issues.append(f"Chunk source ID not in full-paper registry: {extra}")

    for err in parse_errors:
        issues.append("JSONL parse error: " + err)

    REPORT_CSV.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(per_record[0].keys()) if per_record else ["jsonl_line", "chunk_id", "source_id", "title", "character_count"]
    with REPORT_CSV.open("w", newline="", encoding="utf-8-sig") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(per_record)

    counts["unique_chunk_ids"] = len(ids)
    counts["source_ids_present"] = len(seen_sources)
    counts["source_ids_expected_from_registry"] = len(expected_ids)
    min_len = min(lengths) if lengths else 0
    max_len = max(lengths) if lengths else 0
    avg_len = round(statistics.mean(lengths), 1) if lengths else 0
    median_len = round(statistics.median(lengths), 1) if lengths else 0

    lines = [
        "# Full-paper chunk validation report", "",
        f"- Chunk file: `{CHUNKS.relative_to(ROOT)}`",
        f"- Total parsed chunk records: **{len(rows)}**",
        f"- JSONL parse errors: **{len(parse_errors)}**",
        f"- Unique chunk IDs: **{len(ids)}**",
        f"- Duplicate chunk IDs: **{len(duplicates)}**",
        f"- Missing chunk IDs: **{counts['missing_chunk_id']}**",
        f"- Missing/empty text: **{counts['missing_text']}**",
        f"- Missing source ID: **{counts['missing_source_id']}**",
        f"- Missing title: **{counts['missing_title']}**",
        f"- Missing source URL/DOI field: **{counts['missing_url_or_doi']}**",
        f"- Records without a page/page_number field: **{counts['no_page_field']}** (not necessarily an error if page coordinates were not extracted)",
        f"- Sources represented in chunks: **{len(seen_sources)}**",
        f"- Source IDs in registry: **{len(expected_ids)}**",
        f"- Text lengths in characters: min **{min_len}**, median **{median_len}**, mean **{avg_len}**, max **{max_len}**",
        "", "## Chunk counts by source ID", "",
    ]
    for source, n in sorted(source_counts.items()):
        lines.append(f"- `{source}`: {n} chunks")
    lines += ["", "## Issues requiring attention", ""]
    if issues:
        lines.extend(f"- {issue}" for issue in issues)
    else:
        lines.append("- No structural issues detected by these checks. Manual content-quality review is still required.")
    lines += ["", f"Detailed per-chunk table: `{REPORT_CSV.relative_to(ROOT)}`", ""]
    REPORT_MD.write_text("\n".join(lines), encoding="utf-8")

    print("FULL-PAPER CHUNK VALIDATION")
    print(f"Chunks parsed: {len(rows)}")
    print(f"JSONL parse errors: {len(parse_errors)}")
    print(f"Unique chunk IDs: {len(ids)}; duplicate IDs: {len(duplicates)}")
    print(f"Missing IDs: {counts['missing_chunk_id']}; missing text: {counts['missing_text']}; missing source IDs: {counts['missing_source_id']}")
    print(f"Missing titles: {counts['missing_title']}; missing URL/DOI: {counts['missing_url_or_doi']}")
    print(f"Chunk character lengths: min={min_len}, median={median_len}, mean={avg_len}, max={max_len}")
    print("Chunks per source:", dict(sorted(source_counts.items())))
    print("Issues:", len(issues))
    for item in issues[:30]:
        print(" -", item)
    if len(issues) > 30:
        print(f" - ... and {len(issues) - 30} more; see report")
    print("Detailed report:", REPORT_CSV.relative_to(ROOT))
    print("Summary report:", REPORT_MD.relative_to(ROOT))
    print("Note: This checks structure/metadata only, not scientific correctness or chunk semantics.")

if __name__ == "__main__":
    main()
