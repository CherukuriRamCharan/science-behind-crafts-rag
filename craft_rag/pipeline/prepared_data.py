"""Adapters for the repository's prepared JSONL datasets."""
from __future__ import annotations

import csv
import json
import logging
import re
from pathlib import Path
from typing import Any

from .common import clean_text
from .ingest import _parse_card

LOGGER = logging.getLogger(__name__)
SOURCE_ID_RE = re.compile(r"\bWBC[-_]\d+\b", re.IGNORECASE)
EMPTY_VALUES = {"", "—", "-", "nan", "none", "null"}


def _value(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    return "" if text.casefold() in EMPTY_VALUES else text


def _title_key(value: Any) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", _value(value).casefold()))


def _load_jsonl(path: Path, dataset: str, issues: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"Prepared {dataset} dataset not found: {path}")

    records: list[dict[str, Any]] = []
    with path.open(encoding="utf-8-sig") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                issues.append({
                    "dataset": dataset,
                    "line": line_number,
                    "issue": f"Malformed JSON: {exc}",
                    "action": "Skipped the malformed line.",
                })
                continue
            if not isinstance(value, dict):
                issues.append({
                    "dataset": dataset,
                    "line": line_number,
                    "issue": "Record is not a JSON object.",
                    "action": "Skipped the non-object record.",
                })
                continue
            records.append(value)
    return records


def _load_catalogs(prepared_root: Path) -> dict[tuple[str, str], dict[str, str]]:
    catalogs: dict[tuple[str, str], dict[str, str]] = {}
    catalog_paths = (
        prepared_root / "data" / "metadata" / "source_catalog.csv",
        prepared_root / "data" / "metadata" / "full_paper_registry.csv",
    )
    for path in catalog_paths:
        if not path.is_file():
            LOGGER.warning("Optional source metadata catalog not found: %s", path)
            continue
        with path.open(encoding="utf-8-sig", newline="") as handle:
            for row in csv.DictReader(handle):
                source_id = _value(row.get("source_id"))
                title = _value(row.get("title") or row.get("paper_title"))
                if not source_id or not title:
                    continue
                key = (source_id.casefold(), _title_key(title))
                catalogs.setdefault(key, {}).update({
                    name: _value(value)
                    for name, value in row.items()
                    if _value(value)
                })
    return catalogs


def _catalog_fields(
    source_id: str,
    title: str,
    catalogs: dict[tuple[str, str], dict[str, str]],
) -> dict[str, str]:
    return catalogs.get((source_id.casefold(), _title_key(title)), {})


def _quality_issue(
    issues: list[dict[str, Any]],
    dataset: str,
    record_id: str,
    issue: str,
    action: str,
) -> None:
    issues.append({
        "dataset": dataset,
        "record_id": record_id,
        "issue": issue,
        "action": action,
    })


def _make_record(
    *,
    source_id: str,
    title: str,
    text: str,
    record_id: str,
    record_type: str,
    dataset: str,
    metadata: dict[str, Any],
    source_url: str = "",
    doi: str = "",
    evidence_level: str = "E5",
    grade: str = "C",
    craft: str = "",
    region: str = "",
    material: str = "",
    process: str = "",
    primary_theme: str = "",
    evidence_status: str = "",
    review_status: str = "",
    caution: str = "",
    evidence_eligible: bool = True,
) -> dict[str, Any]:
    provenance = {
        "dataset": dataset,
        "record_id": record_id,
        **metadata,
    }
    return {
        "record_id": record_id,
        "source_id": source_id,
        "title": title,
        "text_chunk": clean_text(text),
        "source_type": record_type,
        "record_type": record_type,
        "source_collection": dataset,
        "document_type": record_type,
        "source_url": source_url,
        "doi": doi,
        "link": source_url or (f"https://doi.org/{doi}" if doi else ""),
        "craft_cluster": craft,
        "craft": craft,
        "region": region,
        "material": material,
        "process_stage": process,
        "why_theme": primary_theme,
        "evidence_level": evidence_level,
        "grade": grade,
        "evidence_status": evidence_status,
        "review_status": review_status,
        "caution": caution,
        "evidence_eligible": evidence_eligible,
        "metadata": {"provenance": provenance},
    }


def _merge_record_metadata(target: dict[str, Any], addition: dict[str, Any]) -> None:
    for key in (
        "source_url", "doi", "craft_cluster", "craft", "region", "material",
        "process_stage", "why_theme", "evidence_status", "review_status",
    ):
        if not target.get(key) and addition.get(key):
            target[key] = addition[key]
    if not target.get("link") and addition.get("link"):
        target["link"] = addition["link"]
    target_meta = target.setdefault("metadata", {})
    addition_meta = addition.get("metadata", {})
    target_provenance = target_meta.setdefault("provenance", {})
    target_provenance.setdefault("merged_records", []).append(addition_meta.get("provenance", {}))
    if not target.get("text_chunk") and addition.get("text_chunk"):
        target["text_chunk"] = addition["text_chunk"]
    if target.get("record_type") == "catalog_entry" and addition.get("evidence_eligible"):
        for key in (
            "record_type", "source_type", "document_type", "evidence_eligible",
            "evidence_level", "grade", "caution", "review_status",
        ):
            target[key] = addition[key]
        if addition.get("text_chunk"):
            target["text_chunk"] = addition["text_chunk"]


def load_prepared_corpus(config: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Load, map, enrich, and de-duplicate the prepared corpus without changing source files."""
    prepared_root = Path(config["data"]["prepared_root"]).expanduser().resolve()
    dataset_paths = {
        "documents": prepared_root / "data" / "extracted" / "documents.jsonl",
        "gap_analysis_sources": prepared_root / "data" / "processed" / "gap_analysis_sources.jsonl",
        "gap_analysis_questions": prepared_root / "data" / "processed" / "gap_analysis_questions.jsonl",
        "process_observations": prepared_root / "data" / "processed" / "process_observations.jsonl",
        "full_paper_chunks": prepared_root / "data" / "processed" / "full_paper_chunks.jsonl",
    }
    issues: list[dict[str, Any]] = []
    raw_data = {
        name: _load_jsonl(path, name, issues)
        for name, path in dataset_paths.items()
    }
    catalogs = _load_catalogs(prepared_root)
    records: list[dict[str, Any]] = []
    source_records: dict[tuple[str, str], dict[str, Any]] = {}
    duplicate_count = 0

    for row in raw_data["documents"]:
        record_id = _value(row.get("document_key")) or f"document-line-{len(records) + 1}"
        raw_text = _value(row.get("text"))
        if not raw_text:
            _quality_issue(issues, "documents", record_id, "Extracted text is missing.", "Skipped this record.")
            continue
        parsed = _parse_card(raw_text, Path(_value(row.get("file_name")) or f"{record_id}.txt"))
        parsed_metadata = parsed.get("metadata", {})
        explicit_source_id = re.search(
            r"\bSource\s+ID\b\s*:?\s*([^\s]+)",
            raw_text,
            re.IGNORECASE,
        )
        source_id = _value(parsed.get("source_id")) if explicit_source_id else ""
        title_match = re.search(r"\bTitle\b\s*:?\s*(.+)", raw_text, re.IGNORECASE)
        title = _value(parsed_metadata.get("title")) if title_match else ""
        title = title or _value(row.get("file_name")) or record_id
        if not source_id:
            source_id = f"DOC:{record_id}"
            _quality_issue(
                issues, "documents", record_id, "Source ID was not present as a parsed card field.",
                f"Assigned fallback source ID {source_id}.",
            )
        catalog = _catalog_fields(source_id, title, catalogs)
        if not title_match:
            _quality_issue(issues, "documents", record_id, "Title is missing.", "Used the input file name or record ID.")
        card_link = _value(parsed_metadata.get("link"))
        doi_match = re.search(r"\b10\.\d{4,9}/[-._;()/:A-Z0-9]+\b", card_link, re.IGNORECASE)
        doi = _value(catalog.get("doi")) or (doi_match.group(0).rstrip(".,)") if doi_match else "")
        source_url = card_link if card_link.casefold().startswith(("http://", "https://")) else _value(catalog.get("source_url"))
        if not doi:
            doi_match = re.search(r"\b10\.\d{4,9}/[-._;()/:A-Z0-9]+\b", source_url, re.IGNORECASE)
            doi = doi_match.group(0).rstrip(".,)") if doi_match else ""
        craft = _value(parsed.get("craft_applicability")) or _value(catalog.get("craft"))
        provenance = {
            "document_key": record_id,
            "file_name": _value(row.get("file_name")),
            "relative_path": _value(row.get("relative_path")),
            "file_type": _value(row.get("file_type")),
            "page_count": row.get("page_count"),
            "extraction_status": _value(row.get("extraction_status")),
            "catalog_match": bool(catalog),
        }
        record = _make_record(
            source_id=source_id,
            title=title,
            text=raw_text,
            record_id=record_id,
            record_type="source_document",
            dataset="documents",
            metadata=provenance,
            source_url=source_url,
            doi=doi,
            evidence_level=_value(parsed.get("evidence_level")) or _value(catalog.get("evidence_level")) or "E5",
            craft=craft or _value(catalog.get("craft")),
            region=_value(catalog.get("region")),
            material=_value(catalog.get("material")),
            process=_value(parsed.get("process_stage")) or _value(catalog.get("process")),
            primary_theme=_value(parsed.get("why_theme")) or _value(catalog.get("primary_why_theme")),
            evidence_status=_value(row.get("extraction_status")),
            review_status=_value(catalog.get("review_status")),
            caution=_value(parsed.get("caution")) or "Extracted metadata-card text; verify claims against the linked source.",
        )
        identity = (source_id.casefold(), _title_key(title))
        if identity in source_records:
            _merge_record_metadata(source_records[identity], record)
            duplicate_count += 1
            continue
        source_records[identity] = record
        records.append(record)

    for row in raw_data["gap_analysis_sources"]:
        record_id = _value(row.get("record_id")) or f"gap-source-line-{len(records) + 1}"
        source_id = _value(row.get("source_id"))
        title = _value(row.get("title"))
        if not source_id:
            source_id = f"CATALOG:{record_id}"
            _quality_issue(
                issues, "gap_analysis_sources", record_id, "Source ID is missing.",
                f"Assigned fallback source ID {source_id}.",
            )
        if not title:
            title = f"Untitled catalog entry ({record_id})"
            _quality_issue(issues, "gap_analysis_sources", record_id, "Title is missing.", "Used a labeled fallback title.")
        catalog = _catalog_fields(source_id, title, catalogs)
        raw_fields = row.get("raw_fields") if isinstance(row.get("raw_fields"), dict) else {}
        source_url = _value(row.get("source_url")) or _value(raw_fields.get("URL")) or _value(catalog.get("source_url"))
        doi = _value(catalog.get("doi"))
        craft = _value(row.get("craft_application")) or _value(catalog.get("craft"))
        theme = _value(row.get("primary_theme"))
        text = (
            f"Catalog entry for {title}. Source type: {_value(row.get('source_type')) or 'not supplied'}. "
            f"Craft application: {craft or 'not supplied'}. Primary theme: {theme or 'not supplied'}. "
            f"Evidence metadata: {_value(row.get('evidence')) or _value(catalog.get('evidence_level')) or 'not supplied'}. "
            "This is source-catalog metadata, not the source document text."
        )
        record = _make_record(
            source_id=source_id,
            title=title,
            text=text,
            record_id=record_id,
            record_type="catalog_entry",
            dataset="gap_analysis_sources",
            metadata={
                "workbook": _value(row.get("workbook")),
                "sheet_name": _value(row.get("sheet_name")),
                "excel_row": row.get("excel_row"),
                "raw_fields": raw_fields,
                "catalog_match": bool(catalog),
            },
            source_url=source_url,
            doi=doi,
            evidence_level=_value(row.get("evidence")) or _value(catalog.get("evidence_level")) or "E5",
            craft=craft,
            process=_value(catalog.get("process")),
            primary_theme=theme,
            review_status="Catalog metadata only; source text was not present in this dataset.",
            caution="Catalog metadata is not evidence from the cited source.",
            evidence_eligible=False,
        )
        identity = (source_id.casefold(), _title_key(title))
        if identity in source_records:
            _merge_record_metadata(source_records[identity], record)
            duplicate_count += 1
        else:
            source_records[identity] = record
            records.append(record)

    for row in raw_data["gap_analysis_questions"]:
        record_id = _value(row.get("record_id")) or f"gap-question-line-{len(records) + 1}"
        question = _value(row.get("question"))
        if not question:
            _quality_issue(issues, "gap_analysis_questions", record_id, "Question text is missing.", "Skipped this record.")
            continue
        linked_ids = [item.replace("_", "-").upper() for item in SOURCE_ID_RE.findall(_value(row.get("sources_currently_answering")))]
        source_id = f"GAP:{record_id}"
        if not linked_ids:
            _quality_issue(
                issues, "gap_analysis_questions", record_id, "No source IDs were linked to this question.",
                "Retained it as an unattributed research question.",
            )
        records.append(_make_record(
            source_id=source_id,
            title=f"Research question: {question}",
            text=question,
            record_id=record_id,
            record_type="research_question",
            dataset="gap_analysis_questions",
            metadata={
                "theme": row.get("theme"),
                "coverage_status": _value(row.get("coverage_status")),
                "recommended_next_action": _value(row.get("recommended_next_action")),
                "associated_source_ids": linked_ids,
                "workbook": _value(row.get("workbook")),
                "sheet_name": _value(row.get("sheet_name")),
                "excel_row": row.get("excel_row"),
            },
            primary_theme=_value(row.get("theme")),
            evidence_status=_value(row.get("coverage_status")),
            review_status="Coverage-gap question; not a factual claim.",
            caution="A research question is not evidence and must not be cited as a factual source.",
            evidence_eligible=False,
        ))

    for row in raw_data["process_observations"]:
        record_id = _value(row.get("record_id")) or f"observation-line-{len(records) + 1}"
        observation = _value(row.get("observation_or_question"))
        raw_fields = row.get("raw_fields") if isinstance(row.get("raw_fields"), dict) else {}
        if not observation:
            _quality_issue(
                issues, "process_observations", record_id, "Observation text is missing.",
                "Skipped this record; it has no indexable narrative.",
            )
            continue
        question_like = bool(
            re.match(r"^(why|what|how|when|where|which|who)\b", observation, re.IGNORECASE)
            or observation.rstrip().endswith("?")
        )
        source_id = _value(row.get("source_id"))
        attribution = _value(row.get("source_attribution"))
        if not source_id:
            attribution_match = SOURCE_ID_RE.search(attribution)
            source_id = attribution_match.group(0).replace("_", "-").upper() if attribution_match else f"OBS:{record_id}"
            _quality_issue(
                issues, "process_observations", record_id, "Source ID is missing.",
                f"Used source attribution or fallback ID {source_id}.",
            )
        catalog = _catalog_fields(source_id, _value(raw_fields.get("Title")), catalogs)
        evidence = _value(raw_fields.get("Evidence Level (E1-E5)")) or _value(raw_fields.get("Evidence Level"))
        records.append(_make_record(
            source_id=source_id,
            title=_value(raw_fields.get("Title")) or f"Process observation: {source_id}",
            text=observation,
            record_id=record_id,
            record_type="process_observation",
            dataset="process_observations",
            metadata={
                "theme": row.get("theme"),
                "source_attribution": attribution,
                "requires_source_review": row.get("requires_source_review"),
                "source_review_marker_fields": row.get("source_review_marker_fields"),
                "raw_fields": raw_fields,
                "workbook": _value(row.get("workbook")),
                "sheet_name": _value(row.get("sheet_name")),
                "excel_row": row.get("excel_row"),
            },
            source_url=_value(raw_fields.get("Link/URL")) or _value(raw_fields.get("Link / File")) or _value(catalog.get("source_url")),
            doi=_value(catalog.get("doi")),
            evidence_level=evidence or "E5",
            grade=_value(raw_fields.get("Grade (A/B/C)")) or _value(raw_fields.get("Grade")) or "C",
            craft=_value(row.get("craft")) or _value(catalog.get("craft")),
            region=_value(row.get("region")) or _value(catalog.get("region")),
            material=_value(row.get("material")) or _value(catalog.get("material")),
            process=_value(row.get("process_stage")) or _value(catalog.get("process")),
            primary_theme=_value(row.get("theme")),
            evidence_status=_value(row.get("evidence_status")),
            review_status="Review required." if row.get("requires_source_review") else "",
            caution=(
                "This record contains a question, not an observation; it is excluded from evidence retrieval."
                if question_like
                else "Informal craft-process record; do not treat as independently verified scientific evidence."
            ),
            evidence_eligible=not question_like,
        ))

    for row in raw_data["full_paper_chunks"]:
        record_id = _value(row.get("record_id"))
        source_id = _value(row.get("source_id"))
        title = _value(row.get("title"))
        text = _value(row.get("text"))
        if not record_id:
            record_id = f"full-paper-line-{len(records) + 1}"
            _quality_issue(issues, "full_paper_chunks", record_id, "record_id is missing.", "Assigned a line-order fallback ID.")
        if not source_id:
            source_id = f"PAPER:{record_id}"
            _quality_issue(issues, "full_paper_chunks", record_id, "source_id is missing.", f"Assigned fallback ID {source_id}.")
        if not title:
            title = f"Untitled full paper ({source_id})"
            _quality_issue(issues, "full_paper_chunks", record_id, "Title is missing.", "Used a labeled fallback title.")
        if not text:
            _quality_issue(issues, "full_paper_chunks", record_id, "Chunk text is missing.", "Skipped this record.")
            continue
        catalog = _catalog_fields(source_id, title, catalogs)
        records.append(_make_record(
            source_id=source_id,
            title=title,
            text=text,
            record_id=record_id,
            record_type="full_paper_chunk",
            dataset="full_paper_chunks",
            metadata={
                key: value for key, value in row.items()
                if key not in {"record_id", "source_id", "title", "text"}
            } | {"catalog_match": bool(catalog)},
            source_url=_value(row.get("source_url")) or _value(catalog.get("source_url")),
            doi=_value(catalog.get("doi")),
            evidence_level=_value(row.get("evidence_level")) or _value(catalog.get("evidence_level")) or "E5",
            craft=_value(row.get("craft")) or _value(catalog.get("craft")),
            region=_value(row.get("region")) or _value(catalog.get("region")),
            material=_value(row.get("material")) or _value(catalog.get("material")),
            process=_value(row.get("process")) or _value(catalog.get("process")),
            primary_theme=_value(row.get("primary_theme")) or _value(catalog.get("primary_why_theme")),
            evidence_status=_value(row.get("evidence_status")),
            review_status=_value(row.get("review_status")),
            caution="Full-paper text chunk; validate scientific claims against the source.",
        ))

    report = {
        "prepared_root": str(prepared_root),
        "input_record_counts": {name: len(rows) for name, rows in raw_data.items()},
        "indexed_record_count": len(records),
        "duplicate_source_records_merged": duplicate_count,
        "quality_issue_count": len(issues),
        "quality_issues": issues,
    }
    if issues:
        LOGGER.warning(
            "Prepared corpus has %d data-quality issue(s); details are included in the corpus manifest.",
            len(issues),
        )
    return records, report
