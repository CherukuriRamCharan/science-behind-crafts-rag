"""Export process/observation JSONL records with unsourced/unverified markers.

Run from the repository root:
    python src/export_flagged_observations.py

This script is read-only with respect to original inputs. It creates a review CSV
and report under data/metadata; it never changes or upgrades evidence status.
"""
from __future__ import annotations

import csv
import json
import re
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "data/processed/process_observations.jsonl"
OUTPUT = ROOT / "data/metadata/flagged_observations_review.csv"
REPORT = ROOT / "data/metadata/flagged_observations_review.md"

MARKERS = [
    ("not yet sourced", re.compile(r"not\s+yet\s+sourced", re.I)),
    ("unsourced", re.compile(r"\bunsourced\b", re.I)),
    ("unverified", re.compile(r"\bunverified\b|not\s+independently\s+verified", re.I)),
    ("source needed", re.compile(r"source\s+(?:is\s+)?needed|needs?\s+(?:a\s+)?source", re.I)),
    ("not supported", re.compile(r"not\s+supported|unsupported\s+(?:claim|evidence|explanation)", re.I)),
    ("no source", re.compile(r"no\s+(?:external\s+)?source(?:\s+(?:found|identified|provided|available))?", re.I)),
    ("needs verification", re.compile(r"needs?\s+(?:manual\s+)?verification|requires?\s+(?:manual\s+)?review", re.I)),
]


def walk(value: Any, prefix: str = ""):
    """Yield (field path, string value) for nested dict/list data."""
    if isinstance(value, dict):
        for key, child in value.items():
            path = f"{prefix}.{key}" if prefix else str(key)
            yield from walk(child, path)
    elif isinstance(value, list):
        for index, child in enumerate(value):
            yield from walk(child, f"{prefix}[{index}]")
    elif value is not None:
        yield prefix, str(value)


def first(record: dict, *keys: str) -> str:
    for key in keys:
        value = record.get(key)
        if value not in (None, ""):
            return str(value)
    return ""


def main() -> int:
    if not INPUT.exists():
        print(f"ERROR: input file not found: {INPUT}")
        return 1

    flagged = []
    total = 0
    parse_errors = []
    with INPUT.open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            if not line.strip():
                continue
            total += 1
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                parse_errors.append(f"line {line_no}: {exc}")
                continue
            if not isinstance(record, dict):
                continue

            all_fields = list(walk(record))
            hits = []
            for field, value in all_fields:
                for label, pattern in MARKERS:
                    if pattern.search(value):
                        excerpt = re.sub(r"\s+", " ", value).strip()
                        if len(excerpt) > 500:
                            excerpt = excerpt[:497] + "..."
                        hits.append((field, label, excerpt))
            if not hits:
                continue

            hits_unique = []
            seen = set()
            for hit in hits:
                if hit not in seen:
                    seen.add(hit)
                    hits_unique.append(hit)

            # Support several possible schemas without rewriting source data.
            combined = "\n".join(f"{k}: {v}" for k, v in all_fields)
            sid = first(record, "source_id", "sourceId", "record_id", "recordId", "id")
            if not sid:
                m = re.search(r"\b(WBC-(?:EV-)?[A-Za-z0-9-]+)\b", combined)
                sid = m.group(1) if m else ""
            sheet = first(record, "sheet_name", "sheet", "worksheet")
            excel_row = first(record, "excel_row", "source_excel_row", "row_number", "source_row")
            status = first(record, "status", "evidence_status", "review_status", "coverage_status")
            title = first(record, "title", "paper_title", "question", "practice_observed", "Practice Observed (HOW)")
            reason_text = "; ".join(f"{label} in {field}" for field, label, _ in hits_unique)
            hit_text = " || ".join(f"{field} [{label}]: {excerpt}" for field, label, excerpt in hits_unique)

            flagged.append({
                "jsonl_line": line_no,
                "record_id": first(record, "record_id", "recordId", "id"),
                "source_id": sid,
                "title_or_observation": title,
                "source_workbook": first(record, "source_workbook", "workbook", "workbook_name"),
                "sheet_name": sheet,
                "excel_row": excel_row,
                "current_status": status,
                "review_reason": reason_text,
                "marker_context": hit_text,
                "raw_record_json": json.dumps(record, ensure_ascii=False, sort_keys=True),
            })

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    columns = ["jsonl_line", "record_id", "source_id", "title_or_observation", "source_workbook", "sheet_name", "excel_row", "current_status", "review_reason", "marker_context", "raw_record_json"]
    with OUTPUT.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(flagged)

    report_lines = [
        "# Flagged Process/Observation Review",
        "",
        f"- Input records read: **{total}**",
        f"- Records flagged for human review: **{len(flagged)}**",
        f"- JSONL parse errors: **{len(parse_errors)}**",
        f"- Review CSV: `{OUTPUT.relative_to(ROOT)}`",
        "",
        "These are review flags, not a conclusion that the record is false. Do not upgrade evidence status automatically.",
        "Review each marker in its original workbook context, check linked sources, and retain the original wording/provenance.",
    ]
    if parse_errors:
        report_lines += ["", "## JSONL parse errors", *[f"- {e}" for e in parse_errors]]
    REPORT.write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    print("FLAGGED OBSERVATION REVIEW EXPORT")
    print(f"Input records read: {total}")
    print(f"Records flagged for human review: {len(flagged)}")
    print(f"JSONL parse errors: {len(parse_errors)}")
    print(f"Review CSV: {OUTPUT.relative_to(ROOT)}")
    print(f"Summary: {REPORT.relative_to(ROOT)}")
    print("Original workbook and JSONL inputs were not modified.")
    return 0 if not parse_errors else 2


if __name__ == "__main__":
    raise SystemExit(main())
