"""Complete pending Science Behind Crafts data preparation locally.

Run from the repository root with:
    python src/complete_pending_data_work.py

This script only reads data/raw and current metadata, and writes derived outputs
under data/metadata and data/processed. It does not modify original workbooks,
PDFs, DOCX files, or extracted source-card JSONL.
"""
from __future__ import annotations

import concurrent.futures
import hashlib
import json
import re
import sys
from datetime import date, datetime
from pathlib import Path
from typing import Any

import pandas as pd

try:
    import requests
except Exception:
    requests = None

ROOT = Path(__file__).resolve().parents[1]
RAW_TEAM = ROOT / "data" / "raw" / "teammate_data"
GAP_FILE = RAW_TEAM / "gap ananlysis.xlsx"
LAKSHMI_FILE = RAW_TEAM / "LakshmiPrasanna_Data" / "LakshmiPrasanna_CollectedData.xlsx"
METADATA = ROOT / "data" / "metadata"
PROCESSED = ROOT / "data" / "processed"
CATALOG = METADATA / "source_catalog.csv"
SOURCE_INV = METADATA / "source_inventory.csv"
TOOL_INV = METADATA / "tool_data_inventory.csv"
LINK_REPORT = METADATA / "tool_data_link_check.csv"
PAPER_REGISTRY = METADATA / "full_paper_registry.csv"
EXTRACTED_DOCS = ROOT / "data" / "extracted" / "documents.jsonl"

for directory in (METADATA, PROCESSED):
    directory.mkdir(parents=True, exist_ok=True)


def clean_col(x: Any, i: int) -> str:
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return f"column_{i+1}"
    s = str(x).strip()
    if not s or s.lower().startswith("unnamed:"):
        return f"column_{i+1}"
    return s


def json_safe(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (pd.Timestamp, datetime, date)):
        return value.isoformat()
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    if hasattr(value, "item"):
        try:
            return json_safe(value.item())
        except Exception:
            pass
    if isinstance(value, (str, int, float, bool)):
        return value
    return str(value).strip()


def row_fields(row: pd.Series) -> dict[str, Any]:
    return {str(k): json_safe(v) for k, v in row.items()}


def normalized_key(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value).lower())


def find_col(df: pd.DataFrame, *needles: str) -> str | None:
    """Find a column by normalized substring, prioritizing first matching needle."""
    pairs = [(c, normalized_key(c)) for c in df.columns]
    for needle in needles:
        n = normalized_key(needle)
        for col, norm in pairs:
            if n and n in norm:
                return col
    return None


def value_for(row: pd.Series, df: pd.DataFrame, *needles: str) -> Any:
    col = find_col(df, *needles)
    return json_safe(row.get(col)) if col else None


def as_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")


def load_sheets(path: Path) -> dict[str, pd.DataFrame]:
    if not path.exists():
        raise FileNotFoundError(f"Required workbook not found: {path}")
    # openpyxl is pandas' normal .xlsx engine in this environment.
    raw_sheets = pd.read_excel(path, sheet_name=None)
    result: dict[str, pd.DataFrame] = {}
    for name, df in raw_sheets.items():
        df = df.copy()
        df.columns = [clean_col(c, i) for i, c in enumerate(df.columns)]
        # Retain empty rows in source? We report physical dataframe row counts and
        # emit only meaningful nonempty records into normalized JSONL.
        result[str(name)] = df
    return result


def workbook_inventory(path: Path, sheets: dict[str, pd.DataFrame]) -> list[dict[str, Any]]:
    out = []
    for sheet, df in sheets.items():
        missing = {str(c): int(df[c].isna().sum()) for c in df.columns}
        out.append({
            "workbook": str(path.relative_to(ROOT)).replace("\\", "/"),
            "sheet_name": sheet,
            "dataframe_rows": int(len(df)),
            "column_count": int(len(df.columns)),
            "columns_json": json.dumps([str(c) for c in df.columns], ensure_ascii=False),
            "missing_counts_json": json.dumps(missing, ensure_ascii=False),
            "nonempty_rows": int(df.notna().any(axis=1).sum()),
            "duplicate_full_rows": int(df.duplicated().sum()) if len(df) else 0,
        })
    return out


def meaningful_rows(df: pd.DataFrame):
    for idx, row in df.iterrows():
        if row.notna().any():
            yield int(idx) + 2, row  # Excel row number assuming row 1 is the header.


def workbook_processing(gap_sheets: dict[str, pd.DataFrame], lakshmi_sheets: dict[str, pd.DataFrame]) -> dict[str, Any]:
    inventory_rows = workbook_inventory(GAP_FILE, gap_sheets) + workbook_inventory(LAKSHMI_FILE, lakshmi_sheets)
    pd.DataFrame(inventory_rows).to_csv(METADATA / "workbook_inventory.csv", index=False, encoding="utf-8")

    # Gap workbook: identify the source-index sheet and question/coverage sheet
    # by column semantics, with a safe fallback to the first two sheets.
    source_sheet_name, source_df = next(iter(gap_sheets.items()))
    question_sheet_name, question_df = (list(gap_sheets.items())[1] if len(gap_sheets) > 1 else (source_sheet_name, source_df))
    for sheet_name, df in gap_sheets.items():
        if find_col(df, "source title", "source / title", "source title", "source type") and find_col(df, "url", "source url"):
            source_sheet_name, source_df = sheet_name, df
        if find_col(df, "why question", "question") and find_col(df, "coverage status", "recommended next action"):
            question_sheet_name, question_df = sheet_name, df

    source_records = []
    for excel_row, row in meaningful_rows(source_df):
        fields = row_fields(row)
        title = value_for(row, source_df, "source / title", "source title", "title")
        sid = value_for(row, source_df, "source id", "id", "source key")
        url = value_for(row, source_df, "url", "source url", "link")
        if not any(as_text(v) for v in fields.values()):
            continue
        source_records.append({
            "record_id": "gap-source-" + hashlib.sha1(f"{source_sheet_name}:{excel_row}:{sid}:{title}".encode("utf-8")).hexdigest()[:14],
            "record_type": "gap_analysis_source_record",
            "source_id": sid,
            "title": title,
            "source_type": value_for(row, source_df, "source type", "type"),
            "craft_application": value_for(row, source_df, "craft application", "craft"),
            "primary_theme": value_for(row, source_df, "primary theme", "theme"),
            "evidence": value_for(row, source_df, "evidence", "evidence level"),
            "source_url": url,
            "workbook": str(GAP_FILE.relative_to(ROOT)).replace("\\", "/"),
            "sheet_name": source_sheet_name,
            "excel_row": excel_row,
            "raw_fields": fields,
        })
    write_jsonl(PROCESSED / "gap_analysis_sources.jsonl", source_records)

    question_records = []
    for excel_row, row in meaningful_rows(question_df):
        fields = row_fields(row)
        question = value_for(row, question_df, "why question", "question", "research question")
        if not any(as_text(v) for v in fields.values()):
            continue
        if question is None and len(fields) == 0:
            continue
        question_records.append({
            "record_id": "gap-question-" + hashlib.sha1(f"{question_sheet_name}:{excel_row}:{question}".encode("utf-8")).hexdigest()[:14],
            "record_type": "research_question_or_coverage_gap",
            "theme": value_for(row, question_df, "theme", "primary theme"),
            "question": question,
            "sources_currently_answering": value_for(row, question_df, "sources currently answering", "currently answering sources", "sources"),
            "coverage_status": value_for(row, question_df, "coverage status", "status"),
            "recommended_next_action": value_for(row, question_df, "recommended next action", "next action"),
            "workbook": str(GAP_FILE.relative_to(ROOT)).replace("\\", "/"),
            "sheet_name": question_sheet_name,
            "excel_row": excel_row,
            "raw_fields": fields,
        })
    write_jsonl(PROCESSED / "gap_analysis_questions.jsonl", question_records)

    # Preserve each nonempty original row from Lakshmi workbook, plus conservative
    # standardized aliases inferred from column names. Never rewrite source status.
    observation_records = []
    for sheet_name, df in lakshmi_sheets.items():
        for excel_row, row in meaningful_rows(df):
            fields = row_fields(row)
            if not any(as_text(v) for v in fields.values()):
                continue
            source_id = value_for(row, df, "source id", "observation id", "record id", "id")
            status = value_for(row, df, "status", "coverage status", "source status", "verification status")
            source_text = value_for(row, df, "source", "source document", "source attribution", "reference")
            note = value_for(row, df, "observation", "process observation", "process description", "description", "note", "question")
            # The workbook sometimes stores phrases such as "Not yet sourced" in
            # explanation/note fields rather than a dedicated Status column. Scan
            # every original cell, and retain the exact field names that triggered review.
            unsourced_pattern = re.compile(
                r"not yet sourced|unsourced|needs? source|source needed|unverified|not verified|not covered|uncovered",
                re.IGNORECASE,
            )
            review_marker_fields = [
                str(key) for key, value in fields.items()
                if value is not None and unsourced_pattern.search(as_text(value))
            ]
            observation_records.append({
                "record_id": "process-observation-" + hashlib.sha1(f"{sheet_name}:{excel_row}:{source_id}".encode("utf-8")).hexdigest()[:14],
                "record_type": "craft_process_observation_or_question",
                "source_id": source_id,
                "craft": value_for(row, df, "craft", "tradition", "craft name"),
                "region": value_for(row, df, "region", "state", "location"),
                "material": value_for(row, df, "material", "wood or bamboo"),
                "theme": value_for(row, df, "theme", "why theme", "primary theme"),
                "process_stage": value_for(row, df, "process stage", "stage", "step"),
                "observation_or_question": note,
                "source_attribution": source_text,
                "evidence_status": status,
                "requires_source_review": bool(review_marker_fields),
                "source_review_marker_fields": review_marker_fields,
                "workbook": str(LAKSHMI_FILE.relative_to(ROOT)).replace("\\", "/"),
                "sheet_name": sheet_name,
                "excel_row": excel_row,
                "raw_fields": fields,
            })
    write_jsonl(PROCESSED / "process_observations.jsonl", observation_records)

    return {
        "inventory_rows": inventory_rows,
        "gap_source_sheet": source_sheet_name,
        "gap_question_sheet": question_sheet_name,
        "gap_source_records": source_records,
        "gap_question_records": question_records,
        "process_observations": observation_records,
    }


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    result = []
    with path.open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            if line.strip():
                try:
                    result.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid JSONL at {path}:{line_no}: {exc}") from exc
    return result


def chunk_text(text: str, target: int = 1600, overlap: int = 180) -> list[str]:
    """Paragraph-aware chunks; long paragraphs are split by sentence/character."""
    paragraphs = [re.sub(r"\s+", " ", p).strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    atomic: list[str] = []
    for p in paragraphs:
        if len(p) <= target:
            atomic.append(p)
            continue
        sentences = re.split(r"(?<=[.!?])\s+", p)
        buf = ""
        for sentence in sentences:
            if len(sentence) > target:
                if buf:
                    atomic.append(buf.strip())
                    buf = ""
                for start in range(0, len(sentence), target):
                    atomic.append(sentence[start:start + target].strip())
            elif not buf:
                buf = sentence
            elif len(buf) + 1 + len(sentence) <= target:
                buf += " " + sentence
            else:
                atomic.append(buf.strip())
                buf = sentence
        if buf:
            atomic.append(buf.strip())

    chunks: list[str] = []
    buf = ""
    for p in atomic:
        if not buf:
            buf = p
        elif len(buf) + 2 + len(p) <= target:
            buf += "\n\n" + p
        else:
            chunks.append(buf.strip())
            prefix = buf[-overlap:].strip() if overlap and len(buf) > overlap else ""
            buf = (prefix + "\n\n" + p).strip() if prefix else p
            if len(buf) > target:
                chunks.append(buf[:target].strip())
                buf = buf[target-overlap:].strip() if overlap else buf[target:].strip()
    if buf.strip():
        chunks.append(buf.strip())
    return [c for c in chunks if c]


def paper_quality_checks() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if not PAPER_REGISTRY.exists():
        return [], []
    registry = pd.read_csv(PAPER_REGISTRY).fillna("")
    checks, chunk_rows = [], []
    for _, item in registry.iterrows():
        source_id = as_text(item.get("source_id", ""))
        title = as_text(item.get("paper_title", ""))
        text_rel = as_text(item.get("extracted_text_path", ""))
        local_text_path = ROOT / Path(text_rel)
        record = {
            "source_id": source_id, "title": title,
            "extracted_text_path": text_rel,
            "file_exists": local_text_path.exists(),
            "character_count": 0, "word_count": 0,
            "has_intro_like_heading": False, "has_methods_like_heading": False,
            "has_results_like_heading": False, "has_discussion_like_heading": False,
            "has_references_like_heading": False,
            "replacement_character_count": 0,
            "first_500_chars_nonempty": False, "last_500_chars_nonempty": False,
            "title_terms_found": None,
            "completeness_assessment": "not assessed",
        }
        if local_text_path.exists():
            text = local_text_path.read_text(encoding="utf-8", errors="replace")
            lower = text.lower()
            record["character_count"] = len(text)
            record["word_count"] = len(text.split())
            record["has_intro_like_heading"] = bool(re.search(r"\b(introduction|background)\b", lower))
            record["has_methods_like_heading"] = bool(re.search(r"\b(methods?|materials and methods|methodology)\b", lower))
            record["has_results_like_heading"] = bool(re.search(r"\b(results|findings|experiments?)\b", lower))
            record["has_discussion_like_heading"] = bool(re.search(r"\b(discussion|conclusion)\b", lower))
            record["has_references_like_heading"] = bool(re.search(r"\b(references|bibliography)\b", lower))
            record["replacement_character_count"] = text.count("\ufffd")
            record["first_500_chars_nonempty"] = bool(text[:500].strip())
            record["last_500_chars_nonempty"] = bool(text[-500:].strip())
            title_terms = [w.lower() for w in re.findall(r"[A-Za-z]{4,}", title) if w.lower() not in {"from", "with", "the", "and", "case", "study", "review"}]
            record["title_terms_found"] = sum(1 for w in set(title_terms) if w in lower)
            present = sum(record[k] for k in ["has_intro_like_heading", "has_methods_like_heading", "has_results_like_heading", "has_discussion_like_heading", "has_references_like_heading"])
            if len(text) > 10000 and record["first_500_chars_nonempty"] and record["last_500_chars_nonempty"] and record["has_references_like_heading"]:
                record["completeness_assessment"] = "substantial extracted text; references and both ends present; manual PDF-text comparison still needed"
            else:
                record["completeness_assessment"] = f"needs review; {present}/5 section-marker groups found"
            chunks = chunk_text(text)
            for idx, chunk in enumerate(chunks, 1):
                chunk_id = "chunk-" + hashlib.sha1(f"{source_id}:{idx}:{chunk}".encode("utf-8")).hexdigest()[:16]
                chunk_rows.append({
                    "record_id": chunk_id,
                    "source_id": source_id,
                    "title": title,
                    "text": chunk,
                    "document_type": "full_paper_chunk",
                    "source_url": as_text(item.get("source_url", "")),
                    "doi": as_text(item.get("doi", "")),
                    "local_path": text_rel,
                    "chunk_index": idx,
                    "chunk_count": len(chunks),
                    "page_number": None,
                    "evidence_level": as_text(item.get("evidence_level", "")),
                    "craft_applicability": as_text(item.get("craft_applicability", "")),
                    "primary_why_theme": as_text(item.get("primary_why_theme", "")),
                    "process_stage": as_text(item.get("process_stage", "")),
                    "review_status": "Machine-chunked; paper completeness requires manual review",
                })
        checks.append(record)
    write_jsonl(PROCESSED / "full_paper_chunks.jsonl", chunk_rows)
    return checks, chunk_rows


def live_check_links() -> dict[str, Any]:
    if not LINK_REPORT.exists():
        return {"status": "skipped", "reason": "tool_data_link_check.csv not found", "checked_urls": 0}
    df = pd.read_csv(LINK_REPORT).fillna("")
    url_cols = [c for c in df.columns if any(k in normalized_key(c) for k in ("url", "link"))]
    if not url_cols:
        return {"status": "skipped", "reason": "No URL/link columns found", "checked_urls": 0}
    if requests is None:
        return {"status": "skipped", "reason": "Python requests package is unavailable", "checked_urls": 0}

    url_work: dict[str, str] = {}
    for col in url_cols:
        for v in df[col].astype(str):
            url = v.strip()
            if url.startswith("http://") or url.startswith("https://"):
                url_work[url] = url

    def check(url: str) -> tuple[str, dict[str, Any]]:
        try:
            response = requests.get(
                url, timeout=(4, 7), allow_redirects=True,
                headers={"User-Agent": "Mozilla/5.0 (compatible; CraftDataLinkAudit/1.0)"},
                stream=True,
            )
            status_code = int(response.status_code)
            final_url = response.url
            response.close()
            if 200 <= status_code < 400:
                label = "reachable"
            elif status_code in (401, 403, 406, 429):
                label = "access_restricted_or_rate_limited; manual review"
            elif status_code == 404 or status_code == 410:
                label = "not_found_response; verify alternative"
            else:
                label = "http_error; manual review"
            return url, {"status": label, "http_status": status_code, "final_url": final_url, "error": ""}
        except Exception as exc:
            return url, {"status": "request_failed; manual review", "http_status": "", "final_url": "", "error": str(exc)[:240]}

    checks: dict[str, dict[str, Any]] = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        for url, result in pool.map(check, list(url_work)):
            checks[url] = result

    # Keep every current report column and add a compact result for each URL-like column.
    for col in url_cols:
        suffix = re.sub(r"[^a-z0-9]+", "_", str(col).lower()).strip("_")[:32]
        for metric in ("status", "http_status", "final_url", "error"):
            df[f"live_{suffix}_{metric}"] = df[col].astype(str).map(lambda u: checks.get(u.strip(), {}).get(metric, ""))
    out_path = METADATA / "tool_data_link_check_live.csv"
    df.to_csv(out_path, index=False, encoding="utf-8")
    return {
        "status": "completed",
        "checked_urls": len(checks),
        "report": str(out_path.relative_to(ROOT)).replace("\\", "/"),
        "outcomes": pd.Series([x["status"] for x in checks.values()]).value_counts().to_dict(),
    }


def build_readiness_report(wb_result: dict[str, Any], paper_checks: list[dict[str, Any]], chunks: list[dict[str, Any]], link_result: dict[str, Any]) -> str:
    lines = [
        "# Data Readiness Report — Science Behind Crafts RAG",
        "",
        "Generated by `src/complete_pending_data_work.py`. This report describes the files present when the script ran.",
        "",
        "## Workbook inspection",
        f"- Gap-analysis source records normalized: **{len(wb_result['gap_source_records'])}** (sheet: `{wb_result['gap_source_sheet']}`; workbook inventory records total and nonempty rows).",
        f"- Research questions / coverage rows normalized: **{len(wb_result['gap_question_records'])}** (sheet: `{wb_result['gap_question_sheet']}`; fully blank rows are not emitted as records).",
        f"- LakshmiPrasanna nonempty process/observation rows preserved: **{len(wb_result['process_observations'])}**.",
        f"- Workbook sheet inventory: `data/metadata/workbook_inventory.csv`.",
        "- All nonempty source row fields are preserved in `raw_fields`; normalized aliases are convenience fields and should be reviewed when headers are ambiguous.",
        "",
        "## Coverage status summary",
    ]
    qrows = wb_result["gap_question_records"]
    status_counts: dict[str, int] = {}
    theme_counts: dict[str, int] = {}
    unsourced = []
    for record in qrows:
        status = as_text(record.get("coverage_status")) or "(status missing)"
        theme = as_text(record.get("theme")) or "(theme missing)"
        status_counts[status] = status_counts.get(status, 0) + 1
        theme_counts[theme] = theme_counts.get(theme, 0) + 1
        # Inspect status and other text fields. A coverage gap may be described
        # in the next action or source notes rather than in Coverage Status alone.
        review_text = " ".join(as_text(record.get(k)) for k in (
            "coverage_status", "recommended_next_action", "sources_currently_answering", "question"
        ))
        if re.search(r"not yet sourced|unsourced|no source|source needed|not covered|uncovered|needs? source", review_text, re.I):
            unsourced.append(record)
    if status_counts:
        lines.extend(["", "### Coverage statuses (exact workbook values)"])
        for status, count in sorted(status_counts.items(), key=lambda p: (-p[1], p[0])):
            lines.append(f"- `{status}`: {count}")
    if theme_counts:
        lines.extend(["", "### Questions by theme"])
        for theme, count in sorted(theme_counts.items(), key=lambda p: (-p[1], p[0])):
            lines.append(f"- {theme}: {count}")
    lines.extend(["", f"### Rows mentioning an unsourced / uncovered status or action: **{len(unsourced)}**"])
    for record in unsourced[:30]:
        q = as_text(record.get("question")) or "(question text missing)"
        lines.append(f"- Row {record.get('excel_row')} | Theme: {as_text(record.get('theme')) or 'unknown'} | Status: {as_text(record.get('coverage_status')) or 'missing'} | Question: {q[:280]}")
    if len(unsourced) > 30:
        lines.append(f"- ... plus {len(unsourced) - 30} additional rows preserved in `gap_analysis_questions.jsonl`.")

    lines.extend(["", "## Process observations requiring source review"])
    observations = wb_result["process_observations"]
    needs_source = [r for r in observations if r.get("requires_source_review")]
    lines.append(f"- Rows with an unsourced/unverified marker in any original cell: **{len(needs_source)} of {len(observations)}**.")
    for r in needs_source[:25]:
        raw = r.get("raw_fields", {})
        marker_fields = ", ".join(r.get("source_review_marker_fields", [])) or "field not identified"
        lines.append(f"- Sheet `{r['sheet_name']}`, Excel row {r['excel_row']}; status field: `{as_text(r.get('evidence_status')) or '(blank)'}`; marker fields: `{marker_fields}`; fields: `{json.dumps(raw, ensure_ascii=False)[:260]}`")
    if len(needs_source) > 25:
        lines.append(f"- ... plus {len(needs_source) - 25} additional rows in `process_observations.jsonl`.")

    lines.extend(["", "## Full-paper extraction checks"])
    if not paper_checks:
        lines.append("- No paper registry or extracted text paths were available; checks skipped.")
    for p in paper_checks:
        lines.append(
            f"- **{p['source_id']} — {p['title']}**: exists={p['file_exists']}; chars={p['character_count']}; words={p['word_count']}; "
            f"intro={p['has_intro_like_heading']}; methods={p['has_methods_like_heading']}; results={p['has_results_like_heading']}; "
            f"discussion/conclusion={p['has_discussion_like_heading']}; references={p['has_references_like_heading']}; "
            f"replacement_chars={p['replacement_character_count']}. Assessment: {p['completeness_assessment']}."
        )
    lines.append(f"- Generated full-paper chunks: **{len(chunks)}** in `data/processed/full_paper_chunks.jsonl`. Page numbers are null unless independently available; this script does not fabricate page coordinates.")

    lines.extend(["", "## Live source-link check"])
    lines.append(f"- {json.dumps(link_result, ensure_ascii=False, default=str)}")
    lines.append("- Live HTTP results are an access-time snapshot. 403/406/429 and timeout outcomes require manual review; they do not prove a source is broken.")

    lines.extend(["", "## Existing catalog / inventory checks"])
    for file in (CATALOG, SOURCE_INV, TOOL_INV, PAPER_REGISTRY):
        if file.exists():
            try:
                df = pd.read_csv(file)
                lines.append(f"- `{file.relative_to(ROOT).as_posix()}`: {len(df)} rows; columns={', '.join(map(str, df.columns))}.")
            except Exception as exc:
                lines.append(f"- `{file.relative_to(ROOT).as_posix()}`: read failed: {exc}.")
        else:
            lines.append(f"- `{file.relative_to(ROOT).as_posix()}`: missing.")
    lines.extend([
        "", "## Interpretation and limitations",
        "- Source-card text records what the card claims; it does not independently verify the scientific claim.",
        "- Research questions, artisan observations, source records, and evidence-backed explanations remain separate record types.",
        "- A source listed beside a question is not automatically proof that the question is fully answered.",
        "- Section-marker detection is heuristic. Final paper completeness needs visual/manual comparison against the source PDF.",
        "- Review blank source URLs, unmatched workbook columns, ambiguous status labels, and all restricted/failed HTTP checks before production use.",
    ])
    return "\n".join(lines) + "\n"


def main() -> int:
    print("Science Behind Crafts: completing pending data work")
    print(f"Project root: {ROOT}")
    try:
        gap_sheets = load_sheets(GAP_FILE)
        lakshmi_sheets = load_sheets(LAKSHMI_FILE)
        print("Gap workbook sheets:", [(name, df.shape[0], df.shape[1]) for name, df in gap_sheets.items()])
        print("LakshmiPrasanna workbook sheets:", [(name, df.shape[0], df.shape[1]) for name, df in lakshmi_sheets.items()])
        wb_result = workbook_processing(gap_sheets, lakshmi_sheets)
        print("Normalized gap-analysis source records:", len(wb_result["gap_source_records"]))
        print("Normalized gap-analysis questions:", len(wb_result["gap_question_records"]))
        print("Preserved process/observation rows:", len(wb_result["process_observations"]))
    except Exception as exc:
        print(f"ERROR processing workbooks: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2

    try:
        paper_checks, chunks = paper_quality_checks()
        pd.DataFrame(paper_checks).to_csv(METADATA / "full_paper_text_checks.csv", index=False, encoding="utf-8")
        print("Full paper text checks:", len(paper_checks))
        print("Generated full paper chunks:", len(chunks))
    except Exception as exc:
        print(f"WARNING paper text/chunk step failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        paper_checks, chunks = [], []

    try:
        link_result = live_check_links()
        print("Link review:", json.dumps(link_result, ensure_ascii=False, default=str))
    except Exception as exc:
        link_result = {"status": "failed", "error": f"{type(exc).__name__}: {exc}"}
        print("WARNING live link review failed:", link_result["error"], file=sys.stderr)

    report = build_readiness_report(wb_result, paper_checks, chunks, link_result)
    report_path = METADATA / "data_readiness_report.md"
    report_path.write_text(report, encoding="utf-8")

    # Final mechanical checks on outputs.
    expected = [
        PROCESSED / "gap_analysis_questions.jsonl",
        PROCESSED / "gap_analysis_sources.jsonl",
        PROCESSED / "process_observations.jsonl",
        PROCESSED / "full_paper_chunks.jsonl",
    ]
    jsonl_validation = {}
    for path in expected:
        records = load_jsonl(path)
        jsonl_validation[path.name] = len(records)
    catalog_check = {"exists": CATALOG.exists(), "rows": None, "missing_ids": None, "missing_titles": None, "missing_urls": None}
    if CATALOG.exists():
        cat = pd.read_csv(CATALOG).fillna("")
        catalog_check["rows"] = len(cat)
        for field, target in [("source_id", "missing_ids"), ("title", "missing_titles"), ("source_url", "missing_urls")]:
            catalog_check[target] = int(cat[field].astype(str).str.strip().eq("").sum()) if field in cat.columns else "column missing"
    result = {
        "workbook_inventory_sheets": len(wb_result["inventory_rows"]),
        "jsonl_records": jsonl_validation,
        "catalog": catalog_check,
        "readiness_report": str(report_path.relative_to(ROOT)).replace("\\", "/"),
        "full_paper_text_checks": str((METADATA / "full_paper_text_checks.csv").relative_to(ROOT)).replace("\\", "/") if (METADATA / "full_paper_text_checks.csv").exists() else "not produced",
    }
    print("\nFINAL VALIDATION SUMMARY")
    print(json.dumps(result, indent=2, ensure_ascii=False, default=str))
    print("\nNo files under data/raw were modified.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
