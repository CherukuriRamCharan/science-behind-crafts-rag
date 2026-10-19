"""Swappable text embedding backend with deterministic offline fallback."""
from __future__ import annotations

import hashlib
import logging
from pathlib import Path

import numpy as np

from .common import tokens


class TextEmbedder:
    def __init__(self, config: dict):
        cfg = config["embedding"]
        self.backend = cfg.get("backend", "hashing")
        self.model_name = cfg.get("model_name", "intfloat/multilingual-e5-small")
        self.batch_size = int(cfg.get("batch_size", 32))
        self.cache_dir = Path(cfg.get("cache_dir", "artifacts/embedding_cache"))
        self.model = None
        if self.backend not in {"auto", "sentence-transformers", "hashing"}:
            raise ValueError(f"Unsupported embedding backend: {self.backend}")
        if self.backend in {"auto", "sentence-transformers"}:
            try:
                from sentence_transformers import SentenceTransformer

                self.model = SentenceTransformer(
                    self.model_name,
                    local_files_only=not cfg.get("allow_download", False),
                )
                self.backend = "sentence-transformers"
            except ImportError as exc:
                if self.backend == "sentence-transformers":
                    raise ImportError(
                        "The sentence-transformers embedding backend is unavailable. "
                        "Install craft_rag/requirements-llm.txt or select embedding.backend: hashing."
                    ) from exc
                logging.warning("sentence-transformers is not installed; using deterministic hashing embeddings.")
                self.backend = "hashing"
            except (OSError, RuntimeError) as exc:
                if self.backend == "sentence-transformers":
                    raise RuntimeError(
                        f"Could not load embedding model {self.model_name!r}. "
                        "Check the local model cache or explicitly set embedding.allow_download: true."
                    ) from exc
                logging.warning(
                    "Embedding model %s is unavailable locally (%s); using deterministic hashing embeddings.",
                    self.model_name,
                    exc,
                )
                self.backend = "hashing"

    def encode(self, texts):
        texts = list(texts)
        cache_path = None
        if len(texts) > 1:
            cache_key = hashlib.sha256(
                (
                    self.backend
                    + ("\0hashing-v2" if self.backend == "hashing" else "")
                    + "\0"
                    + self.model_name
                    + "\0"
                    + "\0".join(texts)
                ).encode("utf-8")
            ).hexdigest()
            cache_path = self.cache_dir / f"{cache_key}.npz"
            if cache_path.exists():
                try:
                    with np.load(cache_path) as cached:
                        return cached["vectors"].astype("float32")
                except (OSError, ValueError, KeyError) as exc:
                    logging.warning("Ignoring invalid embedding cache %s: %s", cache_path, exc)
                    cache_path.unlink(missing_ok=True)
        if self.model is not None:
            matrix = np.asarray(
                self.model.encode(
                    texts,
                    normalize_embeddings=True,
                    batch_size=self.batch_size,
                    show_progress_bar=False,
                ),
                dtype="float32",
            )
        else:
            dim = 384
            matrix = np.zeros((len(texts), dim), dtype="float32")
            for i, text in enumerate(texts):
                for token in tokens(text):
                    digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
                    value = int.from_bytes(digest[:4], "little") % dim
                    sign = 1.0 if digest[4] & 1 else -1.0
                    matrix[i, value] += sign
                norm = np.linalg.norm(matrix[i])
                if norm:
                    matrix[i] /= norm
        if cache_path is not None:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(cache_path, vectors=matrix)
        return matrix
