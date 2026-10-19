"""Shared schema, repository-relative configuration, and text helpers."""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import yaml

SCHEMA = ("source_id", "craft_cluster", "why_theme", "secondary_theme", "process_stage",
          "evidence_level", "grade", "questions_answered", "text_chunk", "species_material",
          "tool_treatment", "link", "caution")
PACKAGE_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = PACKAGE_ROOT.parent


def resolve_path(path: str | Path, base: str | Path = PACKAGE_ROOT) -> Path:
    resolved = Path(path).expanduser()
    if not resolved.is_absolute():
        resolved = Path(base) / resolved
    return resolved.resolve()


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    config_path = resolve_path(path or "config.yaml")
    with config_path.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"Configuration must contain a YAML mapping: {config_path}")

    config_dir = config_path.parent
    data = config.setdefault("data", {})
    if data.get("prepared_root"):
        data["prepared_root"] = str(resolve_path(data["prepared_root"], config_dir))
    if data.get("artifact_dir"):
        data["artifact_dir"] = str(resolve_path(data["artifact_dir"], config_dir))
    for key in ("eval_file", "real_eval_file", "seed_file", "ingest_report"):
        if data.get(key):
            data[key] = str(resolve_path(data[key], config_dir))

    embedding = config.setdefault("embedding", {})
    if embedding.get("cache_dir"):
        embedding["cache_dir"] = str(resolve_path(embedding["cache_dir"], config_dir))
    index = config.setdefault("index", {})
    for key in ("faiss_path", "metadata_path", "chroma_path"):
        if index.get(key):
            index[key] = str(resolve_path(index[key], config_dir))
    web = config.setdefault("web", {})
    if web.get("allowlist_file"):
        web["allowlist_file"] = str(resolve_path(web["allowlist_file"], config_dir))
    return config

def read_json(path: str | Path) -> Any:
    with Path(path).open(encoding="utf-8") as handle:
        return json.load(handle)

def write_json(path: str | Path, value: Any) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")

def clean_text(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()

def tokens(text: str) -> list[str]:
    return re.findall(r"[\w'-]+", text.casefold(), flags=re.UNICODE)


def is_evidence_eligible(record: dict[str, Any]) -> bool:
    metadata = record.get("metadata", {})
    record_type = metadata.get("record_type", record.get("record_type", ""))
    eligible = metadata.get("evidence_eligible", record.get("evidence_eligible", True))
    return eligible is not False and record_type not in {"research_question", "catalog_entry"}


def normalize_record(record: dict[str, Any]) -> dict[str, Any]:
    out = {key: record.get(key, "") for key in SCHEMA}
    out["source_id"] = str(out["source_id"] or "UNKNOWN")
    out["questions_answered"] = out["questions_answered"] if isinstance(out["questions_answered"], list) else ([out["questions_answered"]] if out["questions_answered"] else [])
    out["text_chunk"] = clean_text(str(out["text_chunk"]))
    evidence = str(out["evidence_level"] or "E5").upper()
    match = re.search(r"E[1-5]", evidence)
    out["evidence_level"] = match.group(0) if match else "E5"
    grade = str(out["grade"] or "C").split()[0].upper()
    out["grade"] = grade if grade in {"A", "B", "C"} else "C"
    metadata = {k: v for k, v in record.items() if k not in SCHEMA and k != "metadata"}
    supplied_metadata = record.get("metadata")
    if isinstance(supplied_metadata, dict):
        for key, value in supplied_metadata.items():
            metadata.setdefault(key, value)
    out["metadata"] = metadata
    return out
