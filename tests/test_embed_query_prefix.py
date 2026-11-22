"""H10: E5 models must get 'query: ' for questions and 'passage: ' for documents."""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from craft_rag.pipeline.embed_text import TextEmbedder  # noqa: E402


class FakeModel:
    """Stands in for a SentenceTransformer and records the texts it receives."""

    def __init__(self):
        self.seen = []

    def encode(self, texts, **kwargs):
        self.seen.append(list(texts))
        return np.ones((len(texts), 4), dtype="float32")


def make_embedder(tmp_path, model_name):
    embedder = TextEmbedder({"embedding": {"backend": "hashing", "cache_dir": str(tmp_path)}})
    embedder.model = FakeModel()
    embedder.model_name = model_name
    return embedder


def test_encode_query_adds_query_prefix_for_e5(tmp_path):
    embedder = make_embedder(tmp_path, "intfloat/multilingual-e5-small")
    embedder.encode_query(["why does bamboo split?"])
    assert embedder.model.seen[-1] == ["query: why does bamboo split?"]


def test_encode_documents_adds_passage_prefix_for_e5(tmp_path):
    embedder = make_embedder(tmp_path, "intfloat/multilingual-e5-small")
    embedder.encode_documents(["Bamboo fibres run lengthwise."])
    assert embedder.model.seen[-1] == ["passage: Bamboo fibres run lengthwise."]


def test_non_e5_model_gets_no_prefix(tmp_path):
    embedder = make_embedder(tmp_path, "sentence-transformers/all-MiniLM-L6-v2")
    embedder.encode_query(["hello"])
    assert embedder.model.seen[-1] == ["hello"]


def test_no_query_is_encoded_as_a_document():
    for rel in ["craft_rag/pipeline/retrieve.py", "src/evaluate_retrieval.py", "src/evaluate_generation.py"]:
        text = (ROOT / rel).read_text(encoding="utf-8")
        assert ".encode([query])" not in text, f"{rel} still encodes the query as a document"