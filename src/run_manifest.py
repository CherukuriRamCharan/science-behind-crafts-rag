"""Write reproducibility metadata alongside evaluation outputs."""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import subprocess
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_NAMES = (
    "numpy", "rank-bm25", "faiss-cpu", "sentence-transformers", "torch",
    "transformers", "accelerate", "scikit-learn", "pytest",
)


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _package_versions() -> dict[str, str | None]:
    versions = {}
    for name in PACKAGE_NAMES:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def write_run_manifest(
    results_dir: str | Path,
    *,
    config: dict[str, Any],
    corpus_chunks: list[dict[str, Any]] | None = None,
) -> Path:
    """Write a manifest with Git SHA, config, corpus hash, versions, and output hashes."""
    destination = Path(results_dir).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    try:
        git_sha = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        git_sha = None

    config_bytes = json.dumps(config, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    corpus_sha = None
    corpus_count = None
    if corpus_chunks is not None:
        canonical_chunks = [
            {key: chunk.get(key) for key in ("chunk_id", "source_id", "record_id", "text_chunk")}
            for chunk in sorted(corpus_chunks, key=lambda row: str(row.get("chunk_id", "")))
        ]
        corpus_sha = _sha256(json.dumps(
            canonical_chunks, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode("utf-8"))
        corpus_count = len(corpus_chunks)
    else:
        existing_manifest = destination / "run_manifest.json"
        if existing_manifest.is_file():
            try:
                previous = json.loads(existing_manifest.read_text(encoding="utf-8"))
                corpus_sha = previous.get("corpus_sha256")
                corpus_count = previous.get("corpus_chunk_count")
            except (OSError, json.JSONDecodeError):
                pass

    output_hashes = {}
    for path in sorted(destination.rglob("*")):
        if path.is_file() and path.name != "run_manifest.json":
            output_hashes[path.relative_to(destination).as_posix()] = _sha256(path.read_bytes())

    manifest = {
        "manifest_version": 1,
        "git_sha": git_sha,
        "config_sha256": _sha256(config_bytes),
        "config_snapshot": config,
        "corpus_sha256": corpus_sha,
        "corpus_chunk_count": corpus_count,
        "package_versions": _package_versions(),
        "output_sha256": output_hashes,
    }
    output = destination / "run_manifest.json"
    output.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return output
