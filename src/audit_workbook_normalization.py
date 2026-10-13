"""Audit every data row in the two source workbooks against normalized JSONL.

Save this file as src/audit_workbook_normalization.py and run from the project root:
    python src/audit_workbook_normalization.py

Read-only with respect to original workbooks and normalized JSONL. Writes only:
    data/metadata/workbook_normalization_audit.csv
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
GAP_FILE = ROOT / "data/raw/teammate_data/gap ananlysis.xlsx"
LAKSHMI_FILE = ROOT / "data/raw/teammate_data/LakshmiPrasanna_Data/LakshmiPrasanna_CollectedData.xlsx"
SOURCES_JSONL = ROOT / "data/processed/gap_analysis_sources.jsonl"
QUESTIONS_JSONL = ROOT / "data/processed/gap_analysis_questions.jsonl"
OBSERVATIONS_JSONL = ROOT / "data/processed/process_observations.jsonl"
OUTPUT = ROOT / "data/metadata/workbook_normalization_audit.csv"


def json_safe(value: Any) -> Any:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    if isinstance(value, (str, int, float, bool)):
        return value
    if hasattr(value, "isoformat"):
        try:
            return value.isoformat()
        except Exception:
            pass
    if hasattr(value, "item"):
        try:
            return json_safe(value.item())
        except Exception:
            pass
    return str(value)


def has_nonblank_value(row: pd.Series) -> bool:
    for value in row.tolist():
        value = json_safe(value)
        if value is None:
            continue
        if str(value).strip():
            return True
    return False


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        print(f"WARNING: expected output not found: {path.relative_to(ROOT)}")
        return []
    records = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON in {path}:{line_number}: {exc}") from exc
    return records


def normalize_excel_row(value: Any) -> int | None:
    if value is None:
        return None
    try:
        number = int(value)
        if float(value) == number:
            return number
    except (TypeError, ValueError, OverflowError):
        return None
    return None


def workbook_audit_rows(
    workbook_path: Path,
    sheet_frames: dict[str, pd.DataFrame],
    output_groups: list[tuple[str, list[dict[str, Any]]]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    workbook_rel = workbook_path.relative_to(ROOT).as_posix()
    out: list[dict[str, Any]] = []
    summary: dict[str, Any] = {"workbook": workbook_rel, "sheets": {}}

    normalized_by_sheet_row: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for output_name, records in output_groups:
        for record in records:
            sheet = str(record.get("sheet_name", ""))
            excel_row = normalize_excel_row(record.get("excel_row"))
            if sheet and excel_row is not None:
                normalized_by_sheet_row.setdefault((sheet, excel_row), []).append(
                    {**record, "_output_name": output_name}
                )

    matched_keys: set[tuple[str, int]] = set()
    for sheet_name, frame in sheet_frames.items():
        sheet_records = [r for r in output_groups if any(
            str(item.get("sheet_name", "")) == sheet_name for item in r[1]
        )]
        # Only compare sheets whose rows are represented by the relevant output type.
        if not sheet_records:
            # Still audit every sheet; classify it as lacking a mapped output rather than hiding it.
            output_names = []
        else:
            output_names = sorted({name for name, _ in sheet_records})

        nonblank_rows = 0
        matched_nonblank = 0
        omitted_nonblank = 0
        blank_rows = 0
        for index, row in frame.iterrows():
            excel_row = int(index) + 2  # row 1 is assumed to be the header row
            fields = {str(col): json_safe(row[col]) for col in frame.columns}
            is_nonblank = has_nonblank_value(row)
            if is_nonblank:
                nonblank_rows += 1
            else:
                blank_rows += 1

            key = (sheet_name, excel_row)
            matches = normalized_by_sheet_row.get(key, [])
            if matches:
                matched_keys.add(key)

            # There may be different output types for the source and question sheets.
            expected_output_names = set(output_names)
            relevant_matches = [m for m in matches if m.get("_output_name") in expected_output_names]
            if is_nonblank and relevant_matches:
                inclusion = "included"
                reason = "Original row is nonblank and has a normalized record."
                matched_nonblank += 1
            elif is_nonblank:
                inclusion = "missing_from_normalized_output"
                reason = (
                    "Original row has content but no normalized record matched by sheet name and Excel row; investigate."
                )
                omitted_nonblank += 1
            else:
                inclusion = "blank_row_not_normalized"
                reason = "Original row has no nonblank cell values."

            primary = relevant_matches[0] if relevant_matches else (matches[0] if matches else {})
            record_id = primary.get("record_id", "")
            source_id = primary.get("source_id", primary.get("id", ""))
            title_or_question = primary.get("title", primary.get("question", ""))
            if not title_or_question:
                # Best-effort identification using the first nonblank text field.
                for value in fields.values():
                    if value is not None and str(value).strip():
                        title_or_question = str(value).strip()
                        break

            out.append({
                "workbook": workbook_rel,
                "sheet_name": sheet_name,
                "excel_row": excel_row,
                "source_or_question_id": source_id,
                "title_or_question": title_or_question,
                "normalized_record_id": record_id,
                "normalized_output": ";".join(output_names),
                "row_nonblank": is_nonblank,
                "inclusion_status": inclusion,
                "reason": reason,
                "original_fields_json": json.dumps(fields, ensure_ascii=False),
            })

        summary["sheets"][sheet_name] = {
            "data_rows_excluding_header": int(len(frame)),
            "nonblank_rows": nonblank_rows,
            "blank_rows": blank_rows,
            "normalized_rows_matched": matched_nonblank,
            "nonblank_rows_missing_normalized_record": omitted_nonblank,
            "columns": [str(c) for c in frame.columns],
        }

    # Find normalized records pointing to sheet rows not present in the workbook frames.
    frame_keys = {
        (sheet_name, int(index) + 2)
        for sheet_name, frame in sheet_frames.items()
        for index in frame.index
    }
    for (sheet_name, excel_row), records in normalized_by_sheet_row.items():
        if (sheet_name, excel_row) not in frame_keys:
            for record in records:
                out.append({
                    "workbook": workbook_rel,
                    "sheet_name": sheet_name,
                    "excel_row": excel_row,
                    "source_or_question_id": record.get("source_id", ""),
                    "title_or_question": record.get("title", record.get("question", "")),
                    "normalized_record_id": record.get("record_id", ""),
                    "normalized_output": record.get("_output_name", ""),
                    "row_nonblank": "unknown",
                    "inclusion_status": "orphan_normalized_record",
                    "reason": "Normalized record points to a sheet/row not found in the workbook's data rows.",
                    "original_fields_json": "{}",
                })
    return out, summary


def main() -> int:
    missing = [p for p in (GAP_FILE, LAKSHMI_FILE) if not p.exists()]
    if missing:
        for path in missing:
            print(f"ERROR: workbook missing: {path}")
        return 2

    gap_sheets = pd.read_excel(GAP_FILE, sheet_name=None, dtype=object)
    lakshmi_sheets = pd.read_excel(LAKSHMI_FILE, sheet_name=None, dtype=object)
    source_records = load_jsonl(SOURCES_JSONL)
    question_records = load_jsonl(QUESTIONS_JSONL)
    observation_records = load_jsonl(OBSERVATIONS_JSONL)

    audit_rows: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []

    rows, summary = workbook_audit_rows(
        GAP_FILE,
        gap_sheets,
        [
            ("gap_analysis_sources.jsonl", source_records),
            ("gap_analysis_questions.jsonl", question_records),
        ],
    )
    audit_rows.extend(rows)
    summaries.append(summary)

    rows, summary = workbook_audit_rows(
        LAKSHMI_FILE,
        lakshmi_sheets,
        [("process_observations.jsonl", observation_records)],
    )
    audit_rows.extend(rows)
    summaries.append(summary)

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(audit_rows).to_csv(OUTPUT, index=False, encoding="utf-8-sig")

    print("WORKBOOK NORMALIZATION AUDIT")
    for summary in summaries:
        print(f"\nWorkbook: {summary['workbook']}")
        for sheet_name, item in summary["sheets"].items():
            print(
                f"  {sheet_name}: {item['data_rows_excluding_header']} data rows; "
                f"{item['nonblank_rows']} nonblank; {item['blank_rows']} blank; "
                f"{item['normalized_rows_matched']} normalized matches; "
                f"{item['nonblank_rows_missing_normalized_record']} nonblank rows missing output"
            )

    audit_df = pd.DataFrame(audit_rows)
    if not audit_df.empty:
        print("\nAudit row statuses:")
        print(audit_df["inclusion_status"].value_counts(dropna=False).to_string())
        missing_rows = audit_df[audit_df["inclusion_status"] == "missing_from_normalized_output"]
        if not missing_rows.empty:
            print("\nNONBLANK ROWS TO INVESTIGATE:")
            for _, row in missing_rows.iterrows():
                print(
                    f"- {row['workbook']} | {row['sheet_name']} | Excel row {row['excel_row']} | "
                    f"{str(row['title_or_question'])[:180]}"
                )
        orphan_rows = audit_df[audit_df["inclusion_status"] == "orphan_normalized_record"]
        if not orphan_rows.empty:
            print(f"\nWARNING: {len(orphan_rows)} normalized records point to missing workbook rows.")
    print(f"\nSaved row-by-row audit: {OUTPUT}")
    print("Original workbooks and normalized JSONL files were not modified.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
