"""Build a deterministic JSONL evaluation dataset from a human-editable YAML spec."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml

BUILDER_VERSION = "1.0.0"


def build_dataset(spec_path: str | Path, output_path: str | Path) -> list[dict[str, Any]]:
    spec_path = Path(spec_path)
    output_path = Path(output_path)
    spec = yaml.safe_load(spec_path.read_text(encoding="utf-8"))
    if not isinstance(spec, dict) or not isinstance(spec.get("questions"), list):
        raise ValueError("Dataset spec must contain a questions list.")
    metadata = spec.get("metadata", {})
    corpus_hash = metadata.get("corpus_sha256")
    if not isinstance(corpus_hash, str) or len(corpus_hash) != 64:
        raise ValueError("Dataset spec metadata must include a SHA-256 corpus_sha256.")
    questions = spec["questions"]
    qids = [str(row.get("qid", "")) for row in questions if isinstance(row, dict)]
    if len(qids) != len(questions) or not all(qids) or len(qids) != len(set(qids)):
        raise ValueError("Every question must be an object with a unique non-empty qid.")

    built = []
    for question in questions:
        row = dict(question)
        row["dataset_builder_version"] = BUILDER_VERSION
        row["corpus_sha256"] = corpus_hash
        built.append(row)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in built:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    return built


if __name__ == "__main__":
    repository = Path(__file__).resolve().parents[1]
    build_dataset(
        repository / "data/eval/eval_questions_v1_spec.yaml",
        repository / "data/eval/eval_questions_v1_rebuilt.jsonl",
    )
