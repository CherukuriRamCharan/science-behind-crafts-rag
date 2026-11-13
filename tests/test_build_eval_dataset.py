"""Tests for the deterministic human-editable evaluation dataset builder."""
from __future__ import annotations

import json

import yaml
import pytest

from src.build_eval_dataset import BUILDER_VERSION, build_dataset


def test_builder_preserves_questions_and_records_version_and_corpus_hash(tmp_path):
    corpus_hash = "a" * 64
    spec = {
        "metadata": {"corpus_sha256": corpus_hash},
        "questions": [
            {"qid": "q001", "question": "Question one?", "gold_answer": "Answer."},
            {"qid": "q002", "question": "Question two?", "gold_answer": "Answer two."},
        ],
    }
    spec_path = tmp_path / "spec.yaml"
    output_path = tmp_path / "questions.jsonl"
    spec_path.write_text(yaml.safe_dump(spec, sort_keys=False), encoding="utf-8")

    first = build_dataset(spec_path, output_path)
    first_bytes = output_path.read_bytes()
    second = build_dataset(spec_path, output_path)

    assert first == second
    assert output_path.read_bytes() == first_bytes
    rows = [json.loads(line) for line in first_bytes.decode().splitlines()]
    assert [row["qid"] for row in rows] == ["q001", "q002"]
    assert all(row["dataset_builder_version"] == BUILDER_VERSION for row in rows)
    assert all(row["corpus_sha256"] == corpus_hash for row in rows)


def test_builder_rejects_duplicate_question_ids(tmp_path):
    spec = {"metadata": {"corpus_sha256": "b" * 64}, "questions": [
        {"qid": "q001", "question": "first"}, {"qid": "q001", "question": "duplicate"},
    ]}
    spec_path = tmp_path / "spec.yaml"
    spec_path.write_text(yaml.safe_dump(spec), encoding="utf-8")
    with pytest.raises(ValueError, match="unique"):
        build_dataset(spec_path, tmp_path / "out.jsonl")
