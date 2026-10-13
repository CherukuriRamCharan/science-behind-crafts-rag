from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
WORKBOOK_PATHS = [
    ROOT / "data" / "raw" / "teammate_data" / "gap ananlysis.xlsx",
    ROOT / "data" / "raw" / "teammate_data" / "LakshmiPrasanna_Data" / "LakshmiPrasanna_CollectedData.xlsx",
]
INVENTORY_PATH = ROOT / "data" / "metadata" / "workbook_inventory.csv"
PROCESSED_DIR = ROOT / "data" / "processed"
GAP_QUESTIONS_PATH = PROCESSED_DIR / "gap_analysis_questions.jsonl"
GAP_SOURCES_PATH = PROCESSED_DIR / "gap_analysis_sources.jsonl"
PROCESS_OBSERVATIONS_PATH = PROCESSED_DIR / "process_observations.jsonl"


def workbook_summary(path: Path) -> list[dict[str, Any]]:
    """Return a per-sheet inventory for a workbook."""
    if not path.exists():
        return []

    summaries: list[dict[str, Any]] = []
    with pd.ExcelFile(path) as workbook:
        for sheet_name in workbook.sheet_names:
            frame = pd.read_excel(workbook, sheet_name=sheet_name, dtype=object)
            columns = [str(column) for column in frame.columns]
            missing_values = int(frame.isna().sum().sum())
            summaries.append(
                {
                    "workbook": path.name,
                    "sheet_name": sheet_name,
                    "row_count": int(len(frame)),
                    "column_count": len(columns),
                    "columns": columns,
                    "missing_value_count": missing_values,
                    "source_path": str(path.relative_to(ROOT)).replace("\\", "/"),
                }
            )
    return summaries


def write_inventory_summary(summaries: Iterable[dict[str, Any]], output_path: Path) -> None:
    """Write per-sheet metadata to CSV."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(list(summaries))
    if frame.empty:
        frame = pd.DataFrame(
            columns=[
                "workbook",
                "sheet_name",
                "row_count",
                "column_count",
                "columns",
                "missing_value_count",
                "source_path",
            ]
        )
    frame.to_csv(output_path, index=False, encoding="utf-8")


def infer_record_type(sheet_name: str, columns: Iterable[str]) -> str:
    text = " ".join([sheet_name, *columns]).lower()
    if "question" in text or "gap" in text or "coverage" in text:
        return "question"
    if "source" in text or "citation" in text or "reference" in text or "title" in text:
        return "source"
    return "observation"


def rows_to_jsonl(frame: pd.DataFrame, sheet_name: str, source_path: str) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for row_index, row in enumerate(frame.to_dict(orient="records"), start=1):
        record = {
            "row_index": row_index,
            "sheet_name": sheet_name,
            "source_path": source_path,
            "record_type": infer_record_type(sheet_name, row.keys()),
            "raw_values": {
                str(key): "" if pd.isna(value) else str(value) for key, value in row.items()
            },
            "missing_columns": [
                str(column) for column, value in row.items() if pd.isna(value)
            ],
        }
        records.append(record)
    return records


def export_workbook_records() -> dict[str, Path]:
    """Export workbook rows to JSONL files in a reproducible format."""
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    questions: list[dict[str, Any]] = []
    sources: list[dict[str, Any]] = []
    observations: list[dict[str, Any]] = []

    for workbook_path in WORKBOOK_PATHS:
        if not workbook_path.exists():
            continue

        with pd.ExcelFile(workbook_path) as workbook:
            for sheet_name in workbook.sheet_names:
                frame = pd.read_excel(workbook, sheet_name=sheet_name, dtype=object)
                dataset = rows_to_jsonl(
                    frame,
                    sheet_name,
                    str(workbook_path.relative_to(ROOT)).replace("\\", "/"),
                )
                record_type = infer_record_type(sheet_name, frame.columns)
                if record_type == "question":
                    questions.extend(dataset)
                elif record_type == "source":
                    sources.extend(dataset)
                else:
                    observations.extend(dataset)

    for output_path, records in [
        (GAP_QUESTIONS_PATH, questions),
        (GAP_SOURCES_PATH, sources),
        (PROCESS_OBSERVATIONS_PATH, observations),
    ]:
        with output_path.open("w", encoding="utf-8") as handle:
            for item in records:
                handle.write(json.dumps(item, ensure_ascii=False) + "\n")

    return {
        "questions": GAP_QUESTIONS_PATH,
        "sources": GAP_SOURCES_PATH,
        "observations": PROCESS_OBSERVATIONS_PATH,
    }


def main() -> None:
    summaries: list[dict[str, Any]] = []
    for path in WORKBOOK_PATHS:
        if path.exists():
            summaries.extend(workbook_summary(path))
    write_inventory_summary(summaries, INVENTORY_PATH)
    export_workbook_records()

    print(f"Workbook inventory written to {INVENTORY_PATH}")
    print(f"Questions exported: {GAP_QUESTIONS_PATH}")
    print(f"Sources exported: {GAP_SOURCES_PATH}")
    print(f"Process observations exported: {PROCESS_OBSERVATIONS_PATH}")


if __name__ == "__main__":
    main()
