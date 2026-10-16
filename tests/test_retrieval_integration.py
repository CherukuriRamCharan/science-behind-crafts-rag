"""Optional: real embedding model. Skipped unless sentence-transformers is installed AND the model is reachable.

Run explicitly:  RUN_EMBEDDING_INTEGRATION=1 python -m pytest -q tests/test_retrieval_integration.py
"""
import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

pytestmark = pytest.mark.skipif(
    os.environ.get("RUN_EMBEDDING_INTEGRATION") != "1",
    reason="set RUN_EMBEDDING_INTEGRATION=1 to download and use the real embedding model",
)


def test_real_model_hybrid_search_on_repository_chunks():
    pytest.importorskip("sentence_transformers")
    from src.retrieval import HybridRetriever

    retriever = HybridRetriever.from_jsonl()
    results = retriever.search("How does handle design affect grip strength?", top_k=5)
    assert 0 < len(results) <= 5
    assert all(r.hybrid_score > 0 and r.source_id and r.text for r in results)
    assert any(r.faiss_rank is not None for r in results)
