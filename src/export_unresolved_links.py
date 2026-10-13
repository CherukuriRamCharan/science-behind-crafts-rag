"""Export unresolved link-check results without modifying the source reports.
Run from the repository root: python src/export_unresolved_links.py
"""
from __future__ import annotations

import csv
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPORT_CANDIDATES = [
    ROOT / "data/metadata/tool_data_link_check_live.csv",
    ROOT / "data/metadata/tool_data_link_check.csv",
]
OUT_CSV = ROOT / "data/metadata/unresolved_links_review.csv"
OUT_MD = ROOT / "data/metadata/unresolved_links_review.md"

STATUS_KEYS = ("status", "outcome", "result", "http_status", "check_status", "link_status", "response")
URL_KEYS = ("url", "link", "source_url", "original_url", "checked_url", "link_url", "uri")
ALT_KEYS = ("alternative_url", "alt_url", "alternative", "replacement_url", "verified_alternative_url")


def choose_column(headers: list[str], candidates: tuple[str, ...]) -> str | None:
    by_lower = {h.strip().lower(): h for h in headers}
    for candidate in candidates:
        if candidate in by_lower:
            return by_lower[candidate]
    for h in headers:
        lowered = h.strip().lower()
        if any(candidate in lowered for candidate in candidates):
            return h
    return None


def classify(status: str) -> str:
    s = (status or "").strip().lower()
    if not s:
        return "status_missing_manual_review"
    if any(x in s for x in ("reachable", "success", "200 ok", "status 200", "http 200")):
        # Do not treat contradictory access errors as reachable.
        if any(x in s for x in ("403", "404", "406", "429", "timeout", "failed", "restricted", "not_found")):
            return "manual_review"
        return "reachable"
    if any(x in s for x in ("403", "406", "429", "restricted", "rate", "denied")):
        return "access_restricted_or_rate_limited"
    if any(x in s for x in ("404", "not found", "not_found", "gone")):
        return "not_found_verify_alternative"
    if any(x in s for x in ("failed", "failure", "timeout", "error", "exception", "request_failed")):
        return "request_failed_manual_review"
    return "manual_review"


def main() -> int:
    source = next((p for p in REPORT_CANDIDATES if p.exists()), None)
    if source is None:
        print("ERROR: Could not find tool_data_link_check_live.csv or tool_data_link_check.csv")
        return 1

    with source.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        headers = reader.fieldnames or []
        rows = list(reader)

    status_col = choose_column(headers, STATUS_KEYS)
    url_col = choose_column(headers, URL_KEYS)
    alt_col = choose_column(headers, ALT_KEYS)

    # If no obvious status column exists, derive one from cells with known outcome terms.
    if status_col is None:
        status_col = next((h for h in headers if any(
            token in h.lower() for token in ("http", "outcome", "check", "note", "error")
        )), None)

    counts: Counter[str] = Counter()
    unresolved: list[dict[str, str]] = []
    for row in rows:
        status = (row.get(status_col, "") if status_col else "") or ""
        outcome = classify(status)
        counts[outcome] += 1
        if outcome != "reachable":
            new_row = dict(row)
            new_row["review_outcome"] = outcome
            new_row["needs_manual_review"] = "yes"
            new_row["review_url"] = (row.get(url_col, "") if url_col else "") or ""
            new_row["review_alternative_url"] = (row.get(alt_col, "") if alt_col else "") or ""
            new_row["review_status_column_used"] = status_col or "not_detected"
            unresolved.append(new_row)

    extra = ["review_outcome", "needs_manual_review", "review_url", "review_alternative_url", "review_status_column_used"]
    out_headers = list(headers) + [x for x in extra if x not in headers]
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with OUT_CSV.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=out_headers, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(unresolved)

    lines = [
        "# Unresolved Link Review Queue", "",
        f"- Source report: `{source.relative_to(ROOT).as_posix()}`",
        f"- Rows checked: **{len(rows)}**",
        f"- Rows queued for manual review: **{len(unresolved)}**",
        f"- Status column used: `{status_col or 'not detected'}`",
        f"- URL column used: `{url_col or 'not detected'}`",
        f"- Alternative URL column used: `{alt_col or 'not detected'}`", "",
        "## Outcomes detected", "",
    ]
    for key, value in sorted(counts.items()):
        lines.append(f"- `{key}`: {value}")
    lines += ["", "This is a triage queue, not proof that a URL is permanently broken. Access restrictions and request failures require manual or alternative-source review.", ""]
    OUT_MD.write_text("\n".join(lines), encoding="utf-8")

    print("UNRESOLVED LINK REVIEW QUEUE")
    print(f"Input report: {source}")
    print(f"Rows checked: {len(rows)}")
    print(f"Status column used: {status_col or 'not detected'}")
    print(f"URL column used: {url_col or 'not detected'}")
    print(f"Alternative URL column used: {alt_col or 'not detected'}")
    print("Outcomes:", dict(counts))
    print(f"Rows queued for manual review: {len(unresolved)}")
    print(f"CSV: {OUT_CSV.relative_to(ROOT)}")
    print(f"Summary: {OUT_MD.relative_to(ROOT)}")
    if not status_col:
        print("WARNING: No status column was detected. Review the CSV output before relying on classifications.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
