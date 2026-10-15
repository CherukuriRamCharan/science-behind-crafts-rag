"""Load RAG chunks from JSONL and define the shared result types.

The chunk schema follows ``docs/data_contract.md``. Required fields are
``record_id``, ``text``, ``source_id`` and ``title``; every other field is kept
untouched in ``Chunk.metadata``. Nothing is invented: if a chunk has no page
field, ``page`` stays ``None``.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Union

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CHUNKS_PATH = ROOT / "data" / "processed" / "full_paper_chunks.jsonl"
CHUNKS_PATH_ENV_VAR = "RETRIEVAL_CHUNKS_PATH"

REQUIRED_FIELDS = ("record_id", "text", "source_id", "title")
PAGE_FIELDS = ("page", "page_number")  # only used if a record really has one


class ChunkLoadError(ValueError):
    """Raised when a chunk file is missing or contains a malformed record."""


@dataclass(frozen=True)
class Chunk:
    """One evidence chunk exactly as stored in the JSONL file."""

    chunk_id: str
    text: str
    source_id: str
    title: str
    metadata: dict[str, Any] = field(default_factory=dict)  # all other fields, unchanged

    @property
    def page(self) -> Optional[Any]:
        """Page value from the data, or ``None`` if the record has no page field."""
        for key in PAGE_FIELDS:
            value = self.metadata.get(key)
            if value not in (None, ""):
                return value
        return None


@dataclass
class RetrievedChunk:
    """A ranked search result. Fields for a retriever that was not used stay ``None``."""

    chunk_id: str
    text: str
    source_id: str
    title: str
    metadata: dict[str, Any]
    page: Optional[Any] = None
    bm25_rank: Optional[int] = None
    bm25_score: Optional[float] = None
    faiss_rank: Optional[int] = None
    faiss_score: Optional[float] = None
    hybrid_score: Optional[float] = None

    @classmethod
    def from_chunk(cls, chunk: Chunk, **scores: Any) -> "RetrievedChunk":
        return cls(
            chunk_id=chunk.chunk_id,
            text=chunk.text,
            source_id=chunk.source_id,
            title=chunk.title,
            metadata=dict(chunk.metadata),
            page=chunk.page,
            **scores,
        )

    def to_dict(self) -> dict[str, Any]:
        """Plain dict (JSON-serialisable) for the Generation module."""
        return {
            "chunk_id": self.chunk_id,
            "text": self.text,
            "source_id": self.source_id,
            "title": self.title,
            "page": self.page,
            "metadata": dict(self.metadata),
            "bm25_rank": self.bm25_rank,
            "bm25_score": self.bm25_score,
            "faiss_rank": self.faiss_rank,
            "faiss_score": self.faiss_score,
            "hybrid_score": self.hybrid_score,
        }


def resolve_chunks_path(path: Union[str, Path, None] = None) -> Path:
    """Pick the chunk file: explicit argument > ``RETRIEVAL_CHUNKS_PATH`` > repo default."""
    if path is not None:
        return Path(path)
    env_value = os.environ.get(CHUNKS_PATH_ENV_VAR)
    return Path(env_value) if env_value else DEFAULT_CHUNKS_PATH


def load_chunks(path: Union[str, Path, None] = None) -> list[Chunk]:
    """Read a chunk JSONL file into ``Chunk`` objects, validating every record."""
    file_path = resolve_chunks_path(path)
    if not file_path.is_file():
        raise ChunkLoadError(f"Chunk file not found: {file_path}")

    chunks: list[Chunk] = []
    seen_ids: dict[str, int] = {}
    with file_path.open(encoding="utf-8-sig") as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            where = f"{file_path.name}, line {line_no}"
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ChunkLoadError(f"Malformed JSON ({where}): {exc}") from exc
            if not isinstance(record, dict):
                raise ChunkLoadError(f"Record is not a JSON object ({where})")

            missing = [
                name for name in REQUIRED_FIELDS
                if not isinstance(record.get(name), str) or not record[name].strip()
            ]
            if missing:
                raise ChunkLoadError(
                    f"Missing or empty required field(s) {missing} ({where})"
                )

            chunk_id = record["record_id"]
            if chunk_id in seen_ids:
                raise ChunkLoadError(
                    f"Duplicate record_id '{chunk_id}' ({where}; first seen on line {seen_ids[chunk_id]})"
                )
            seen_ids[chunk_id] = line_no

            metadata = {k: v for k, v in record.items() if k not in ("record_id", "text", "source_id", "title")}
            chunks.append(
                Chunk(
                    chunk_id=chunk_id,
                    text=record["text"],
                    source_id=record["source_id"],
                    title=record["title"],
                    metadata=metadata,
                )
            )
    return chunks
