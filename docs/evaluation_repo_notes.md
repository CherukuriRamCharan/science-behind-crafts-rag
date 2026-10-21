# Evaluation Repository Notes: System Architecture and Retrieval/Generation Analysis

## 1. Overview and Architecture

The `science-behind-crafts-rag` repository contains two main subsystems for retrieval and question answering over craft-science evidence:

1. **`src/retrieval/`**: A standalone modular retrieval layer providing explicit implementations of:
   - Sparse keyword search (`BM25Retriever` using `rank-bm25`)
   - Dense semantic search (`FaissRetriever` using `faiss.IndexFlatIP` on cosine-normalized embeddings)
   - Hybrid retrieval (`HybridRetriever` using Reciprocal Rank Fusion / RRF)
   - Pure retrieval: returns `RetrievedChunk` instances containing provenance and scoring metadata without text generation.
   - Default corpus: `data/processed/full_paper_chunks.jsonl` (194 chunks from 2 full research papers: WBC-0051 and WBC-2016), configurable via parameter or `RETRIEVAL_CHUNKS_PATH`.

2. **`craft_rag/`**: An end-to-end integrated application and pipeline providing:
   - Multi-source corpus preparation (`pipeline/prepared_data.py`), adapting 5 datasets (`documents.jsonl`, `gap_analysis_sources.jsonl`, `gap_analysis_questions.jsonl`, `process_observations.jsonl`, `full_paper_chunks.jsonl`) and metadata catalogs.
   - Searchable evidence filtering (`is_evidence_eligible`): excludes unanswered research questions and catalog metadata from the evidence search space by default.
   - Unified chunking and passage indexing (`pipeline/chunk.py`, `pipeline/index_build.py`).
   - Retrieval (`pipeline/retrieve.py`): hybrid FAISS + BM25 with RRF, evidence level boosts (`E1` to `E5`), grade boosts (`A`, `B`, `C`), score thresholds, and title/source deduplication.
   - Generation (`pipeline/generate.py`): extractive answer synthesis (default) or optional local HuggingFace CausalLM (`Qwen/Qwen2.5-3B-Instruct`).
   - HTTP server (`server.py`) and CLI (`cli.py`).

---

## 2. Retriever Types Available

| Subsystem | Retriever Type | Mechanism / Details |
| :--- | :--- | :--- |
| `src.retrieval` | **BM25** (`BM25Retriever`) | Uses `rank_bm25.BM25Okapi`. Tokenizes via alphanumeric extraction and a custom 25-stopword set. Retains chunks sharing >= 1 query term, sorting by negative BM25 score with deterministic tie-breaking on index. |
| `src.retrieval` | **Dense (FAISS)** (`FaissRetriever`) | Cosine similarity via L2-normalized vectors in `faiss.IndexFlatIP`. Deterministic mapping of FAISS row index `i` to `chunks[i]`. |
| `src.retrieval` | **Hybrid** (`HybridRetriever`) | Reciprocal Rank Fusion (RRF): $\text{score} = \sum \frac{1}{k_{\text{rrf}} + \text{rank}}$, where $k_{\text{rrf}} = 60$. BM25 and cosine scores are not summed directly. Deduplication by `chunk_id`. Ties broken deterministically by `chunk_id`. |
| `src.retrieval` | **Reranker** | Not present in `src/retrieval`. |
| `craft_rag` | **Dense (FAISS / Matrix)** | `faiss.IndexFlatIP` or cosine dot-product fallback (`vectors @ query_vector`). |
| `craft_rag` | **Sparse (BM25 / Overlap)** | `BM25Okapi` or exact word overlap fallback. |
| `craft_rag` | **Hybrid (RRF)** | RRF with weights: $\text{score} = \frac{w_{\text{dense}}}{60 + \text{rank}_{\text{dense}}} + \frac{w_{\text{sparse}}}{60 + \text{rank}_{\text{sparse}}}$. Plus explicit metadata boosts: `E1: +0.003`, `E2: +0.002`, `E3: +0.001`, `E4: +0.0005`, `E5: 0.0`; `Grade A: +0.001`, `Grade B: +0.0005`, `Grade C: 0.0`. Minimum score threshold: `0.0001`. Deduplication by normalized `(source_id, title)`. |
| `craft_rag` | **Reranker** | Config option `retrieval.reranker_enabled: false`. Reranker hook is stubbed in `craft_rag/pipeline/retrieve.py` line 84: cross-encoder hook is left inert to avoid unapproved checkpoint downloads. |

---

## 3. Embedding Models

1. **`src/retrieval`**:
   - Default model: `sentence-transformers/all-MiniLM-L6-v2` (384-dimensional dense vectors, ~22M parameters, CPU-friendly).
   - Wrapper: `SentenceTransformerEmbedder` in `faiss_retriever.py`.
   - Embedder protocol: any object implementing `encode(texts: Sequence[str]) -> np.ndarray`. Fake embedders are used for unit tests.
   - Download behavior: lazy download on first real use.

2. **`craft_rag`**:
   - Configured backend (`config.yaml`): `embedding.backend: hashing` (default).
   - Hashing backend: deterministic offline 128-dimensional projection using word token hashes (`TextEmbedder` in `embed_text.py`). Requires **no network access or model downloads**.
   - Optional backend: `sentence-transformers` with `intfloat/multilingual-e5-small` (384-dimensional). Model downloads require explicit configuration `embedding.allow_download: true`.

---

## 4. Top-K Defaults and Retrieval Parameters

| Parameter | `src/retrieval` | `craft_rag` |
| :--- | :--- | :--- |
| Default `top_k` | `5` | `5` (`retrieval.top_k`) |
| Candidate retrieval pool | `50` (`candidate_k`) | `20` (`retrieval.candidate_k`) |
| RRF constant ($k$) | `60` (`rrf_k`) | `60` (`retrieval.rrf_k`) |
| Minimum score cutoff | None | `0.0001` (`retrieval.minimum_score`) |
| Non-evidence inclusion | N/A (indexes supplied file) | `False` (`retrieval.include_non_evidence`) |
| Score boosting | None | Evidence level (`+0.0005` to `+0.003`) & Grade (`+0.0005` to `+0.001`) |

---

## 5. Calling the Retriever and Full Pipeline from Python

### A. Calling `src/retrieval`

```python
from src.retrieval import load_chunks, BM25Retriever, FaissRetriever, HybridRetriever

# 1. Load chunks (defaults to data/processed/full_paper_chunks.jsonl)
chunks = load_chunks()  # Or load_chunks("path/to/custom_chunks.jsonl")

# 2. BM25 search
bm25 = BM25Retriever(chunks)
bm25_results = bm25.search("grip strength handle design", top_k=5)

# 3. Dense FAISS search (using default sentence-transformers/all-MiniLM-L6-v2 or a fake embedder)
faiss = FaissRetriever(chunks)
faiss_results = faiss.search("how handle shape affects grip comfort", top_k=5)

# 4. Hybrid search
hybrid = HybridRetriever(bm25, faiss, rrf_k=60, candidate_k=50)
# Or one-liner:
hybrid = HybridRetriever.from_jsonl()
results = hybrid.search("Why does bamboo age matter for material selection?", top_k=5)

# Each result is a RetrievedChunk:
# chunk.chunk_id, chunk.source_id, chunk.title, chunk.text, chunk.bm25_score, chunk.faiss_score, chunk.hybrid_score, chunk.metadata
```

### B. Calling `craft_rag` Full Pipeline

```python
from craft_rag.pipeline.common import load_config
from craft_rag.pipeline.prepared_data import load_prepared_corpus
from craft_rag.pipeline.index_build import build_indices
from craft_rag.pipeline.retrieve import HybridRetriever
from craft_rag.pipeline.generate import GroundedGenerator

# 1. Load configuration and prepared corpus
cfg = load_config()
corpus, manifest = load_prepared_corpus(cfg)

# 2. Build or load index
index_data = build_indices(corpus, cfg)

# 3. Retrieve
retriever = HybridRetriever(index_data, cfg)
passages = retriever.retrieve("How does hand tool design affect grip strength?", top_k=5)

# 4. Generate answer
generator = GroundedGenerator(cfg)
answer = generator.answer("How does hand tool design affect grip strength?", passages)
print(answer)
```

---

## 6. LLM Details and API Keys

- **Default Backend**: **`extractive`** (`craft_rag/pipeline/generate.py`).
  - No LLM is invoked.
  - Requires **no API keys** (offline, deterministic, fast).
  - Selects up to 3 evidence-eligible passages and concatenates them with formatted source citations:
    `"Retrieved evidence: {passage_text} [{source_id} | {evidence_level}] ..."`
- **Optional LLM Backend**: **`transformers`** with `Qwen/Qwen2.5-3B-Instruct`.
  - Runs locally via PyTorch / HuggingFace `transformers`.
  - Requires **no commercial API keys** (e.g. OpenAI / Anthropic / Google Gemini).
  - Disabled by default (`backend: extractive`). Requires downloading a ~6 GB checkpoint if opted in.
- **External API Keys**: None configured or required across the repository.

---

## 7. Prompt Template and Generation Logic

When the `transformers` backend is active in `craft_rag/pipeline/generate.py`:
- **System Prompt**:
  > `"Answer only from the supplied source passages. Cite factual statements inline as [Source ID | Evidence Level]. If sources do not support an answer, say 'Insufficient evidence.' Preserve cautions and never present placeholder data as fact."`
- **User Prompt**:
  > `"Source passages:\n{context}\n\nQuestion: {question}"`
  Where `{context}` concatenates eligible passages:
  `"[{source_id} | {evidence_level}] {text_chunk} Caution: {caution}\n\n..."`
- **Abstention Response**:
  If evidence count is below `minimum_results` (1), or token overlap between question and evidence is below threshold ($< 8\%$), the generator outputs:
  > `"Insufficient evidence in the retrieved corpus to answer this question."`

---

## 8. Citation Output and Formatting

- **Inline Citation Syntax**: `[{source_id} | {evidence_level}]`
  - Example: `[WBC-0051 | E3]` or `[WBC-0001 | E5]`.
- **Validation in Generation**:
  - The generator validates that cited source IDs in the generated text form a non-empty subset of the retrieved eligible passages:
    `cited.issubset(valid)`. If invalid or missing, it falls back to the extractive response.
- **Retrieved Chunk Metadata**:
  - In `src/retrieval`, `RetrievedChunk` carries:
    - `chunk_id`: unique chunk record identifier
    - `source_id`: source identifier (e.g., `WBC-0051`)
    - `title`: document title
    - `metadata["source_url"]`, `metadata["craft"]`, `metadata["evidence_level"]`, etc.

---

## 9. Chunk Identification and Critical Provenance Rules

> [!IMPORTANT]
> **`source_id` IS NOT UNIQUE.**
> A single `source_id` (e.g. `WBC-0051`) can correspond to:
> 1. A PDF or DOCX metadata card (1 record).
> 2. A full research paper partitioned into multiple chunks (e.g. 97 chunks `WBC-0051-0001` through `WBC-0051-0097`).
> 3. A workbook process observation row (`process-observation-4bf...`).
> 4. A gap analysis source entry (`gap-source-3c9...`).

### Chunk Identifier Fields:
- In `data/processed/full_paper_chunks.jsonl`: **`record_id`** (e.g., `"WBC-0051-0001"`).
- In `src/retrieval`: **`chunk.chunk_id`** (mapped directly from `record_id`).
- In `craft_rag`: **`chunk_id`** formatted as `{unique_source}::{part:04d}` where `unique_source` is `{source_id}` (or `{source_id}::doc{N:04d}` if multiple documents share the ID).
- In evaluation datasets and gold standards: **`gold_record_ids` MUST use specific record IDs (`record_id` or chunk `chunk_id`), NEVER `source_id` alone.**
