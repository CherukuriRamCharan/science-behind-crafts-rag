# Prepared-corpus RAG integration

The integrated application lives in [`craft_rag/`](../craft_rag/). It adapts the existing prepared files in `data/`; it does not replace their generation scripts or alter their schemas. `Ishan_BTP/sbc-rag-prototype/` remains an untouched local reference. The integration does not copy its nested Git repository, environment, source documents, seed data, prebuilt indexes, or caches.

## Data and retrieval flow

```text
prepared JSONL + metadata CSVs
        |
        v
pipeline/prepared_data.py (schema adapter, provenance and quality report)
        |
        v
pipeline/chunk.py (unique IDs, word and character limits)
        |
        +--> sentence-transformer embeddings or deterministic offline hashing
        +--> FAISS cosine-style dense ranking
        +--> BM25 sparse ranking
                   |
                   v
           reciprocal-rank fusion
                   |
                   v
       evidence-only extractive answer
```

The adapter consumes:

- `data/extracted/documents.jsonl`
- `data/processed/gap_analysis_sources.jsonl`
- `data/processed/gap_analysis_questions.jsonl`
- `data/processed/process_observations.jsonl`
- `data/processed/full_paper_chunks.jsonl`
- `data/metadata/source_catalog.csv` and `data/metadata/full_paper_registry.csv` for exact source-ID/title metadata matches.

The common retriever record carries source ID, title, text, URL/DOI, craft/process/evidence fields, and provenance. Source-card text is parsed into its concise annotation summary instead of indexing the full extracted layout; its original relevance note and detailed craft applicability remain metadata. Frontend filters normalize these details into `Wood carving`, `Bamboo carving`, and `Wood & bamboo carving` where applicable. Full-paper chunks, source cards, process observations, gap questions, and source-catalog entries remain distinct record types. Exact duplicate source-card/catalog records are merged only on normalized source ID and title; different titles sharing an ID are retained. Chunk IDs are unique even when multiple documents have the same source ID.

Every input record that cannot be parsed or has missing metadata is reported in `craft_rag/artifacts/corpus_manifest.json`. Records with missing source IDs receive an explicit fallback ID and remain traceable; records without any indexable text are skipped with a report entry. On the current prepared files, the adapter reports 62 issues: five gap-analysis source records without source IDs, 15 gap questions without linked source IDs, and 42 observation issues (30 missing source IDs and 12 empty narratives).

Gap questions and source-catalog summaries are retained in the local metadata JSON but excluded from the evidence retrieval index by default. Observation rows that contain questions rather than observations are likewise excluded from evidence search. This prevents unanswered questions and bibliographic metadata from being cited as factual evidence. Set `retrieval.include_non_evidence: true` in a custom config only when you intentionally need to search those records.

## Python environment

Use Python 3.11 (the supplied prototype supported Python 3.10–3.12). The integrated environment is separate from the main project's `.venv` and the prototype's broken `.venv`:

```powershell
py -3.11 -m venv craft_rag\.venv
.\craft_rag\.venv\Scripts\python.exe -m pip install --upgrade pip
.\craft_rag\.venv\Scripts\python.exe -m pip install -r craft_rag\requirements.txt
```

To activate it in PowerShell, run `.\craft_rag\.venv\Scripts\Activate.ps1`; in Command Prompt, run `craft_rag\.venv\Scripts\activate.bat`. The commands below use the environment's Python executable directly, so activation is optional.

The default embedding backend is deterministic hashing, which needs no model download. To use the configured multilingual E5 embedding model, install `craft_rag\requirements-llm.txt` and set `embedding.backend: sentence-transformers`. Model downloads are disabled unless `embedding.allow_download: true` is explicitly set. The default generator is extractive and quotes retrieved evidence with `[Source ID | Evidence Level]` citations. To use the optional local Transformers generator, install `craft_rag\requirements-llm.txt`, set `generator.backend: transformers`, and opt into `generator.allow_download: true` if the model is not already cached. The Qwen checkpoint and PyTorch can require substantial storage and memory; they are not installed or downloaded by default.

ChromaDB is optional. JSON metadata and FAISS/BM25 retrieval work without it. To enable the prototype's Chroma metadata store, run `.\craft_rag\.venv\Scripts\python.exe -m pip install -r craft_rag\requirements-vector-store.txt`.

## Build, demo, and web application

Run commands from the repository root. Paths in the configuration are resolved relative to `craft_rag/config.yaml`, not the terminal's current directory.

Build or rebuild the prepared-corpus index:

```powershell
.\craft_rag\.venv\Scripts\python.exe -m craft_rag.build_index
```

Run the sample questions and optional evaluation against linked prepared gap questions:

```powershell
.\craft_rag\.venv\Scripts\python.exe -m craft_rag.run_demo
```

Ask one question:

```powershell
.\craft_rag\.venv\Scripts\python.exe -m craft_rag.cli "How does hand tool design affect grip strength?"
```

Start the local web interface (loopback only by default):

```powershell
.\craft_rag\.venv\Scripts\python.exe -m craft_rag.server
```

Open <http://127.0.0.1:8000>. Use `--port 8001` for another port. The index is built on server startup. Errors appear in the API status response and process log; the interface does not silently switch to a synthetic corpus.

The built index, embedding cache, evaluation results, and corpus manifest are written under `craft_rag/artifacts/`. These files can contain full extracted research-paper text and remain local; they are ignored by Git. Do not stage or publish them. No prepared datasets or original papers are copied into `craft_rag/`.

## Current retrieval baseline

The offline hashing smoke checks retrieve and cite WBC-0051 for hand-tool grip strength, WBC-0009 for grain-angle cutting, and WBC-0059 for mature bamboo density. Answers now select short query-matched sentences from paper text or source-card relevance notes; when the corpus does not support a question, the response says so. The latest run over 80 linked prepared questions measured precision@5 0.028, recall@5 0.091, MRR 0.057, and nDCG@5 0.057. These aggregate scores remain weak and do not establish broad retrieval quality. The reported faithfulness and relevancy values are simple deterministic citation/overlap proxies, not expert or LLM judgments. Tune and evaluate the optional multilingual embedding backend with expert-labeled queries before relying on broader retrieval.

## Tests

Run the integrated adapter, retrieval, generation, and web API tests:

```powershell
.\craft_rag\.venv\Scripts\python.exe -m pytest -q tests/test_craft_rag.py
```

To run the complete repository suite, including the existing pandas- and python-docx-based workbook tests, install the test-only dependencies in the isolated environment and run from the repository root:

```powershell
.\craft_rag\.venv\Scripts\python.exe -m pip install -r craft_rag\requirements-test.txt
.\craft_rag\.venv\Scripts\python.exe -m pytest -q tests/
```

The integrated tests use the real prepared files for schema/provenance validation and use small synthetic records only for isolated index, error-path, and API checks.
