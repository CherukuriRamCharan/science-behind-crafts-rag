# Retrieval module (Vasanthi)

Finds evidence chunks for a question using **BM25** (keywords), **FAISS** (meaning) and **hybrid** (both, merged with RRF).
It returns chunks + metadata only. It does **not** write answers (that is the Generation module).

## Install
```bash
pip install -r requirements-retrieval.txt     # rank-bm25, faiss-cpu, numpy, sentence-transformers, pytest
```
BM25 needs only `rank-bm25`. FAISS/hybrid also need `sentence-transformers` (pulls PyTorch; the default model
`sentence-transformers/all-MiniLM-L6-v2` is ~90 MB, CPU is fine, downloaded on first use and **not** stored in the repo).

## Chunk file path
Default: `data/processed/full_paper_chunks.jsonl` (relative to the repo, no absolute paths). Override with an argument or an env var:
```bash
export RETRIEVAL_CHUNKS_PATH=/path/to/other_chunks.jsonl     # Windows PowerShell: $env:RETRIEVAL_CHUNKS_PATH="C:\path\chunks.jsonl"
```
Required fields per record: `record_id`, `text`, `source_id`, `title`. All other fields are kept unchanged in `metadata`.
Bad files raise `ChunkLoadError` with the line number.

## Usage (run from the repository root)
```python
from src.retrieval import load_chunks, BM25Retriever, FaissRetriever, HybridRetriever

chunks = load_chunks()                                   # or load_chunks("my.jsonl")

bm25 = BM25Retriever(chunks);  bm25.search("grip strength handle", top_k=5)
faiss = FaissRetriever(chunks);  faiss.search("how does handle shape affect the hand", top_k=5)   # model name configurable: FaissRetriever(chunks, model_name="...")

retriever = HybridRetriever.from_jsonl()                 # build once, reuse for every query
results = retriever.search("Why does bamboo age matter for material selection?", top_k=5)
```
Quick demo: `python -m src.retrieval.demo "your question"` (falls back to BM25-only if `sentence-transformers` is missing).

## How it works
- **BM25**: lowercases text, splits into words, removes common stop-words, scores chunks with `BM25Okapi`. Original text is never changed; only the index uses normalized words.
- **FAISS**: each chunk is embedded, vectors are L2-normalized (same for queries), stored in an exact `IndexFlatIP` index, so the score is cosine similarity. FAISS row *i* = `chunks[i]`, so IDs always map back correctly. Index is built once in memory (194 chunks takes seconds, so no index file is saved).
- **Hybrid**: takes the top 50 from each, then RRF: `score = 1/(60+bm25_rank) + 1/(60+faiss_rank)`. Raw BM25 and cosine scores are never added together. Chunks are de-duplicated by `chunk_id`; ties break by `chunk_id`.

## Result format (for Generation)
`search()` returns `RetrievedChunk` objects; call `.to_dict()` to get plain dicts:
```python
{"chunk_id": "WBC-0051-0049", "text": "...original text...", "source_id": "WBC-0051", "title": "...",
 "page": None, "metadata": {"source_url": "...", "craft": "...", "chunk_index": 49, ...},
 "bm25_rank": 1, "bm25_score": 5.4, "faiss_rank": 3, "faiss_score": 0.41, "hybrid_score": 0.0323}
```
Generation can do `[r.to_dict() for r in retriever.search(question, 5)]` and cite `source_id`, `title` and `metadata["source_url"]`.
`bm25_*` / `faiss_*` are `None` if that retriever did not return the chunk.

## Known limitations
- The chunk file currently covers **only 2 full papers** (WBC-0051 hand-tool grip review, WBC-2016 robotic wood carving). The 98 metadata cards, observations and gap questions are not chunked, so questions such as bamboo age may find no truly relevant evidence.
- **No `page` field exists** in the chunks, so `page` is `None`. About 19 chunk texts contain extraction markers like `--- PAGE 1 ---`, but chunks can span pages, so page citations are not derived automatically.
- `evidence_level` and `primary_theme` are empty for these chunks; they are passed through as-is and nothing is invented. `review_status` says the chunks must be validated before use as evidence.
- BM25 has no stemming ("age" vs "ages" are different words).
- Semantic search was not run in the authoring sandbox (no model download); core tests use fake embeddings. Optional real-model test: `RUN_EMBEDDING_INTEGRATION=1 python -m pytest -q tests/test_retrieval_integration.py`.

## Tests
```bash
python -m pytest -q tests/test_retrieval.py     # offline, no model needed
python -m pytest -q tests/                      # whole repo
```
