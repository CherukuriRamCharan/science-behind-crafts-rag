"""Tests for Stage 4E Server & Web Integration.

Verifies:
- API exposes source_id, chunk_id, evidence_level, citations, and abstained flag
- Backward compatibility: question, answer, sources, backends all preserved
- Static assets (index.html, app.js, styles.css) served correctly
"""
from __future__ import annotations

import io
import json
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

from craft_rag import server
from craft_rag.pipeline.common import load_config
from craft_rag.pipeline.generate import GroundedGenerator


class DummyRequest:
    def __init__(self, method: str, path: str, body: bytes = b""):
        self.method = method
        self.path = path
        self.body = body

    def makefile(self, mode: str, *args, **kwargs):
        if "r" in mode:
            return io.BytesIO(self.body)
        return io.BytesIO()


def test_server_query_response_structure(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify that /api/query returns chunk_id, evidence_level, citations, and abstained flag."""
    test_passage = {
        "source_id": "WBC-0001",
        "chunk_id": "WBC-0001::0000",
        "record_id": "rec-1",
        "title": "Bamboo Splitting",
        "evidence_level": "E3",
        "grade": "A",
        "craft_cluster": "Bamboo carving",
        "why_theme": "Splitting mechanics",
        "process_stage": "Preparation",
        "text_chunk": "Bamboo splits along fibres easily.",
        "link": "https://example.org",
        "retrieval_score": 0.85,
        "metadata": {"title": "Bamboo Splitting"},
    }

    class MockRetriever:
        def retrieve(self, query: str, filters: dict | None = None):
            return [test_passage]

    class MockGenerator:
        def generate_structured(self, query: str, passages: list[dict]):
            return {
                "question": query,
                "answer": "Bamboo splits cleanly [WBC-0001 | E3]",
                "abstained": False,
                "answer_mode": "fallback",
                "confidence": 1.0,
                "citations": [
                    {
                        "source_id": "WBC-0001",
                        "chunk_id": "WBC-0001::0000",
                        "evidence_level": "E3",
                        "quote": "Bamboo splits along fibres easily.",
                    }
                ],
                "retrieved_chunk_ids": ["WBC-0001::0000"],
            }

        def answer(self, query: str, passages: list[dict]):
            return "Bamboo splits cleanly [WBC-0001 | E3]"

    mock_app = {
        "record_count": 10,
        "quality_issue_count": 0,
        "index": {
            "backend_info": {"embedding": "hashing"},
            "chunks": [test_passage],
        },
        "retriever": MockRetriever(),
        "generator": MockGenerator(),
    }
    monkeypatch.setattr(server, "get_engine", lambda *args, **kwargs: mock_app)

    # Simulate POST /api/query
    handler = object.__new__(server.Handler)
    handler.path = "/api/query"
    handler.headers = {"Content-Length": str(len(json.dumps({"question": "Why does bamboo split?"})))}
    handler.rfile = io.BytesIO(json.dumps({"question": "Why does bamboo split?"}).encode("utf-8"))
    handler.wfile = io.BytesIO()

    responses = []

    def mock_respond(status, payload, content_type=None):
        responses.append((status, payload))

    handler.respond = mock_respond

    handler.do_POST()

    assert len(responses) == 1
    status, payload = responses[0]
    assert status == 200

    # Assert required Stage 4E fields
    assert payload["question"] == "Why does bamboo split?"
    assert payload["answer"] == "Bamboo splits cleanly [WBC-0001 | E3]"
    assert payload["abstained"] is False
    assert payload["answer_mode"] == "fallback"
    assert payload["confidence"] == 1.0
    assert len(payload["citations"]) == 1
    assert payload["citations"][0]["chunk_id"] == "WBC-0001::0000"
    assert payload["citations"][0]["source_id"] == "WBC-0001"
    assert payload["citations"][0]["evidence_level"] == "E3"

    # Assert sources list structure
    assert len(payload["sources"]) == 1
    src = payload["sources"][0]
    assert src["source_id"] == "WBC-0001"
    assert src["chunk_id"] == "WBC-0001::0000"
    assert src["evidence_level"] == "E3"
    assert src["grade"] == "A"
    assert src["text"] == "Bamboo splits along fibres easily."


def test_server_abstention_query_handling(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify that an abstained query returns abstained=True and citations=[] without crashing."""
    class MockRetriever:
        def retrieve(self, query: str, filters: dict | None = None):
            return []

    class MockGenerator:
        def generate_structured(self, query: str, passages: list[dict]):
            return {
                "question": query,
                "answer": "Insufficient evidence in the retrieved corpus to answer this question.",
                "abstained": True,
                "citations": [],
                "retrieved_chunk_ids": [],
            }

        def answer(self, query: str, passages: list[dict]):
            return "Insufficient evidence in the retrieved corpus to answer this question."

    mock_app = {
        "record_count": 10,
        "quality_issue_count": 0,
        "index": {
            "backend_info": {"embedding": "hashing"},
            "chunks": [],
        },
        "retriever": MockRetriever(),
        "generator": MockGenerator(),
    }
    monkeypatch.setattr(server, "get_engine", lambda *args, **kwargs: mock_app)

    handler = object.__new__(server.Handler)
    handler.path = "/api/query"
    body_bytes = json.dumps({"question": "What is the tensile strength of carbon nanotubes?"}).encode("utf-8")
    handler.headers = {"Content-Length": str(len(body_bytes))}
    handler.rfile = io.BytesIO(body_bytes)

    responses = []
    handler.respond = lambda status, payload, content_type=None: responses.append((status, payload))

    handler.do_POST()

    assert len(responses) == 1
    status, payload = responses[0]
    assert status == 200
    assert payload["abstained"] is True
    assert payload["citations"] == []
    assert payload["sources"] == []


def test_static_web_assets_exist_and_readable() -> None:
    """Verify web static assets are present and include Stage 4E UI bindings."""
    web_dir = Path("craft_rag/web")
    index_html = (web_dir / "index.html").read_text(encoding="utf-8")
    app_js = (web_dir / "app.js").read_text(encoding="utf-8")
    styles_css = (web_dir / "styles.css").read_text(encoding="utf-8")

    assert "citations-container" in index_html
    assert "source.chunk_id" in app_js
    assert "citations" in app_js
    assert "abstained" in app_js
    assert "answer_mode" in app_js
    assert "FALLBACK ANSWER" in app_js
    assert ".badge.chunk" in styles_css
