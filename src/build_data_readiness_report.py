from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CATALOG_PATH = ROOT / "data" / "metadata" / "source_catalog.csv"
TOOL_CHECK_PATH = ROOT / "data" / "metadata" / "tool_data_link_check.csv"
WORKBOOK_INVENTORY = ROOT / "data" / "metadata" / "workbook_inventory.csv"
REPORT_PATH = ROOT / "data" / "metadata" / "data_readiness_report.md"


def safe_read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path, dtype=object)


def build_report() -> str:
    catalog = safe_read_csv(CATALOG_PATH)
    tool_checks = safe_read_csv(TOOL_CHECK_PATH)
    workbook_inventory = safe_read_csv(WORKBOOK_INVENTORY)

    if catalog.empty:
        total_records = 0
        pdf_cards = 0
        docx_cards = 0
        full_papers = 0
    else:
        total_records = int(len(catalog))
        pdf_cards = int((catalog["document_type"] == "PDF metadata card").sum())
        docx_cards = int((catalog["document_type"] == "DOCX metadata card").sum())
        full_papers = int((catalog["document_type"] == "Full research paper PDF").sum())

    missing_urls = 0
    if not catalog.empty:
        missing_urls = int(catalog["source_url"].fillna("").astype(str).str.strip().eq("").sum())

    missing_titles = 0
    if not catalog.empty:
        missing_titles = int(catalog["title"].fillna("").astype(str).str.strip().eq("").sum())

    link_review_counts = {}
    if not tool_checks.empty:
        for status in tool_checks.get("status", pd.Series(dtype=object)).fillna("").astype(str):
            status = status.strip()
            link_review_counts[status] = link_review_counts.get(status, 0) + 1

    workbook_count = len(workbook_inventory) if not workbook_inventory.empty else 0
    workbook_note = (
        "Workbook inventory is present and tracks worksheet-level metadata. "
        "The workbook contents themselves still require manual review when they hold open questions, not-yet-sourced entries, or unsupported claims."
        if workbook_count
        else "Workbook inventory file is not yet populated; run `python src/workbook_analysis.py` to generate it."
    )

    lines = [
        "# Data Readiness Report",
        "",
        "## Summary",
        "",
        f"- Total catalog records: {total_records}",
        f"- PDF metadata cards: {pdf_cards}",
        f"- DOCX metadata cards: {docx_cards}",
        f"- Full research papers: {full_papers}",
        f"- Missing source URLs: {missing_urls}",
        f"- Missing titles: {missing_titles}",
        f"- Workbook inventory rows: {workbook_count}",
        "",
        "## Coverage status",
        "",
        "- The six primary craft collections are represented in the source catalog.",
        "- The Tool Data DOCX metadata cards are represented as separate source records and should remain separate from full paper records.",
        "- Full research papers are kept as distinct records with explicit document type labels; they are not treated as metadata cards.",
        "- `WBC-0010` occurs twice in the original PDF metadata set and both records remain present to preserve provenance and duplicate legitimate evidence items.",
        "",
        "## Link review",
        "",
        "The link-check table is retained for manual review and should be treated as a review signal, not as proof that a source is permanently broken.",
    ]

    if link_review_counts:
        lines.append("")
        for status, count in sorted(link_review_counts.items()):
            lines.append(f"- {status or 'Unspecified status'}: {count}")
    else:
        lines.append("- No link-review status rows were present in the current metadata export.")

    lines.extend(
        [
            "",
            "## Major gaps and actions",
            "",
            workbook_note,
            "- Keep unanswered questions labeled as `Not yet sourced` rather than turning them into scientific facts.",
            "- Treat metadata-card and paper-level evidence as separate record classes; do not conflate them in retrieval inputs.",
            "- Prioritize direct verification for more uncertain links and for any source where the automated URL check returned an HTTP error or timeout.",
            "- Preserve the original URL alongside any alternative or redirected URL so downstream reviewers can trace the provenance.",
            "",
            "## Recommended next actions",
            "",
            "1. Re-run the workbook normalization script after opening the workbook in a fully functional Python environment to confirm sheet names, row counts, and coverage gaps.",
            "2. Complete manual verification for any source marked as blocked, redirected, or requiring review.",
            "3. Keep a clear separation between source documents, observations, research questions, and evidence-backed scientific claims in downstream retrieval and evaluation outputs.",
            "4. Use the generated chunk files for full-paper ingestion only after confirming the paragraph-aware split preserves heading context and reference sections.",
        ]
    )

    return "\n".join(lines) + "\n"


def main() -> None:
    report = build_report()
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(report, encoding="utf-8")
    print(f"Report written to {REPORT_PATH}")


if __name__ == "__main__":
    main()
