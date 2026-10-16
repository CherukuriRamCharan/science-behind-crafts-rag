"""BM25 keyword retrieval (uses the ``rank-bm25`` library)."""
from __future__ import annotations

import re
from typing import Sequence

from rank_bm25 import BM25Okapi

from .data_loader import Chunk, RetrievedChunk

_TOKEN_RE = re.compile(r"[a-z0-9]+")
# Small, fixed stop-word list: keeps WHY-question filler out of the query.
STOPWORDS = frozenset(
    "a an and are as at be by does do did for from how in is it of on or that the "
    "this to was were what when where which who why with".split()
)


def tokenize(text: str) -> list[str]:
    """Lowercase, split on non-alphanumerics, drop stop-words. Used for index AND query."""
    return [t for t in _TOKEN_RE.findall(text.lower()) if t not in STOPWORDS]


class BM25Retriever:
    """Index chunk text once, then answer many queries."""

    def __init__(self, chunks: Sequence[Chunk]) -> None:
        self._chunks = list(chunks)
        self._tokens = [tokenize(c.text) for c in self._chunks]  # original text is untouched
        self._bm25 = BM25Okapi(self._tokens) if self._chunks else None

    def __len__(self) -> int:
        return len(self._chunks)

    @property
    def chunks(self) -> list[Chunk]:
        return list(self._chunks)

    def search(self, query: str, top_k: int = 5) -> list[RetrievedChunk]:
        """Return up to ``top_k`` chunks that share at least one term with the query."""
        if top_k <= 0:
            raise ValueError("top_k must be a positive integer")
        query_tokens = tokenize(query)
        if not query_tokens or self._bm25 is None:
            return []

        scores = self._bm25.get_scores(query_tokens)
        query_terms = set(query_tokens)
        # Keep only chunks containing a query term (BM25 IDF can be <= 0 on tiny corpora).
        matches = [i for i, toks in enumerate(self._tokens) if query_terms.intersection(toks)]
        matches.sort(key=lambda i: (-scores[i], i))  # ties -> corpus order (deterministic)

        return [
            RetrievedChunk.from_chunk(
                self._chunks[i], bm25_rank=rank, bm25_score=float(scores[i])
            )
            for rank, i in enumerate(matches[:top_k], start=1)
        ]
