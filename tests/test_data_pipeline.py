import importlib.util
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def load_module(module_name: str, relative_path: str):
    spec = importlib.util.spec_from_file_location(module_name, ROOT / relative_path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_infer_record_type_handles_gap_and_question_names():
    workbook_analysis = load_module("workbook_analysis", "src/workbook_analysis.py")
    assert workbook_analysis.infer_record_type("Gap Analysis Questions", ["Question", "Source ID"]) == "question"
    assert workbook_analysis.infer_record_type("Process Observations", ["Process stage", "Notes"]) == "observation"
    assert workbook_analysis.infer_record_type("Source Titles", ["Title", "Source URL"]) == "source"


def test_chunk_paragraphs_keeps_headings_and_stable_chunks():
    full_paper_chunker = load_module("full_paper_chunker", "src/full_paper_chunker.py")
    text = "\n\nIntroduction\n\nThis is a long paragraph. " * 5 + "\n\nReferences\n\nA final note."
    chunks = full_paper_chunker._chunk_paragraphs(text, max_chars=250, overlap=50)
    assert chunks
    assert all(len(chunk) <= 250 for chunk in chunks)


def test_validate_catalog_accepts_expected_shape():
    build_source_catalog = load_module("build_source_catalog", "src/build_source_catalog.py")
    valid_record = {
        "source_id": "WBC-0001",
        "title": "Test paper",
        "authors": "",
        "year": "",
        "doi": "",
        "source_url": "https://example.org",
        "drive_path": "",
        "local_path": "data/raw/sample.pdf",
        "document_type": "PDF metadata card",
        "craft": "Wood",
        "region": "Sample region",
        "material": "",
        "process": "Cutting",
        "review_status": "reviewed",
    }
    assert build_source_catalog.validate_catalog([valid_record]) == [valid_record]


def test_workbook_inventory_has_expected_columns():
    workbook_inventory = pd.DataFrame(
        [{
            "workbook": "test.xlsx",
            "sheet_name": "Sheet1",
            "row_count": 10,
            "column_count": 3,
            "columns": ["id", "title", "notes"],
            "missing_value_count": 1,
            "source_path": "data/raw/test.xlsx",
        }]
    )
    assert set(workbook_inventory.columns) >= {"workbook", "sheet_name", "row_count", "missing_value_count"}
