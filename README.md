# Science Behind Crafts RAG

This repository prepares and organizes evidence for a craft-science retrieval pipeline. The work focuses on provenance, source quality, and clear separation between verified evidence, artisan observations, unanswered questions, and supporting metadata.

## Repository scope

- `data/raw/`: original PDFs, DOCX metadata cards, and teammate workbooks. Do not modify these files.
- `data/extracted/`: extracted text for the source cards and full research papers.
- `data/metadata/`: source catalog, registries, inventory tables, and readiness reports.
- `data/processed/`: reproducible derived data files used downstream by retrieval and evaluation teams.
- `src/`: reusable Python scripts for extraction, catalog generation, workbook normalization, chunking, and reporting.

## Data contract

The data pipeline keeps a strict distinction between the following record types:

1. Verified or published source
2. Artisan observation or process description
3. Research question or coverage gap
4. Scientific explanation supported by evidence
5. A record marked `Not yet sourced` or requiring manual review

The same source ID may legitimately appear in more than one record when those records are distinct documents (for example, a metadata card and a full paper that share a common source ID). Do not silently deduplicate them.

## Generated outputs

- `data/metadata/source_catalog.csv`: unified catalog for the 66 original PDF cards, 32 DOCX tool-data cards, and 2 full papers.
- `data/metadata/workbook_inventory.csv`: inventory of workbook sheets, row counts, missing values, and column names.
- `data/processed/gap_analysis_questions.jsonl`: normalized question and coverage-gap entries.
- `data/processed/gap_analysis_sources.jsonl`: normalized source records linked to the workbook gap analysis.
- `data/processed/process_observations.jsonl`: normalized artisan observations and process notes.
- `data/processed/full_paper_chunks.jsonl`: paragraph-aware chunks for full research papers.
- `data/metadata/data_readiness_report.md`: summary of readiness, gaps, and recommended next actions.

## Fields and statuses

The pipelines prefer real data and explicit status labels over guessed values. Common fields include:

- `record_id`: unique identifier for downstream consumption
- `source_id`: original dataset/source identifier, not always unique across document types
- `title`: document or paper title
- `text`: extracted content or normalized question text
- `document_type`: PDF metadata card, DOCX metadata card, full research paper, or workbook-derived record
- `source_url`: source URL or DOI resolver URL when available
- `local_path`: repository-local path to the original file when available
- `craft`, `region`, `material`, `process`: available provenance and process metadata
- `primary_theme`, `evidence_level`, `evidence_status`: evidence and relevance metadata when known
- `review_status`: review or verification status for the record
- `chunk_index`, `chunk_count`: chunk metadata for full papers

Unsupported or uncertain observations should remain clearly labeled and should not be mixed into a verified-science dataset without a status flag.

## Regeneration

From the repository root, run the **master pipeline entry point** for the full end-to-end pipeline with built-in authoritative generation, schema validation, and reproducibility checking:

```powershell
# Full pipeline (runs document extraction, workbook processing, chunking, and catalog generation)
python run_pipeline.py

# Re-run pipeline twice and verify bit-for-bit identical SHA-256 hashes across all 6 canonical outputs
python run_pipeline.py --verify

# Skip generation stages and validate/hash existing canonical outputs
python run_pipeline.py --skip-generation
```

Or run individual authoritative generator scripts:

```powershell
python -c "from src.ingestion.extract_documents import extract_all; extract_all()"
python src/build_source_catalog.py
python src/full_paper_chunker.py
python src/build_data_readiness_report.py
```

These scripts are idempotent and 100% reproducible. Two consecutive runs of `run_pipeline.py --verify` produce identical SHA-256 hashes across all six canonical outputs and the unified corpus digest.

## Canonical outputs and Authoritative Writers (C1)

The pipeline produces six canonical outputs, each governed by a single authoritative writer:

| Dataset | Path | Writer | Records | Description |
|---|---|---|---|---|
| `documents.jsonl` | `data/extracted/` | `src.ingestion.extract_documents` | 98 | 66 PDF metadata cards + 32 DOCX tool-data cards |
| `gap_analysis_sources.jsonl` | `data/processed/` | `src.complete_pending_data_work` | 55 | Workbook gap-analysis source records |
| `gap_analysis_questions.jsonl` | `data/processed/` | `src.complete_pending_data_work` | 95 | Research questions and coverage gaps |
| `process_observations.jsonl` | `data/processed/` | `src.complete_pending_data_work` | 140 | Artisan process observations |
| `full_paper_chunks.jsonl` | `data/processed/` | `src.full_paper_chunker` | 194 | Full-paper text chunks (`WBC-XXXX-NNNN` IDs) |
| `source_catalog.csv` | `data/metadata/` | `src.build_source_catalog` | 100 | Unified source catalog across PDF cards, DOCX cards & full papers |

> **Note on C1 Protection:** Raw workbook analysis outputs from `src/workbook_analysis.py` are isolated under `data/metadata/workbook_analysis/raw_*.jsonl` to ensure running analysis never erases or replaces the normalized canonical datasets.

Corpus reproducibility hash (SHA-256): `e218fefb4d854476b82a443df6b91f5c979d83f7fa2486776789987197a40608`

## DOCX tool-data cards (H4)

All 32 DOCX tool-data cards (`WBC-2001` – `WBC-2032`) are extracted from
`data/raw/teammate_data/Tool Data/` and included in `documents.jsonl`.  Each
card is parsed from its 2-column metadata table format.  Source IDs, titles,
evidence levels and links are verified against `data/metadata/tool_data_inventory.csv`.
No unsupported evidence levels or invented metadata are assigned.

## Deterministic Chunk IDs & Input-Order Invariance (M2)

Chunk generation in `craft_rag.pipeline.chunk` implements a deterministic canonical sort key (`canonical_record_sort_key`) based on:
1. Source ID (`source_id`)
2. Dataset priority (`documents` → `gap_analysis_sources` → `gap_analysis_questions` → `process_observations` → `full_paper_chunks`)
3. Workbook sheet priority (`Master Source Log` / `Sheet1` → `Summary` / `Sheet2` → `Gap Analysis` → `Primary Workshop Evidence Index`)
4. Excel row number (`excel_row`)
5. Chunk index (`chunk_index`)
6. Upstream record identifier (`record_id`)
7. SHA-256 of the passage text (`text_sha256`)

This guarantees that:
- Chunk IDs, windowed passage texts, and metadata mappings are **100% identical** regardless of input record ordering (canonical, reversed, or randomly shuffled).
- Upstream full-paper chunks preserve stable `WBC-XXXX-NNNN` format.
- Benchmark targets (`WBC-0051-0003`, `WBC-0051-0085`, `WBC-0051::doc0085::0000`, `WBC-2016-0001`, and `WBC-0001` through `WBC-0005`) resolve to their exact expected passages across runs.

## Prepared-corpus RAG application

The integrated hybrid RAG application uses the prepared JSONL datasets directly.

### Stage 4 Integration & Evaluation

The RAG application under `craft_rag/` integrates:
- **Retrieval (`craft_rag/pipeline/retrieve.py`)**: Hybrid BM25 and vector search with Reciprocal Rank Fusion (RRF), per-source passage limits (`max_per_source: 2`), and passage-level metadata propagation.
- **Structured Generation (`craft_rag/pipeline/generate.py`)**: Verified provenance generation (`generate_structured`) providing `source_id`, `chunk_id`, `evidence_level`, and quoted passage spans, with explicit abstention on insufficient evidence.
- **Evaluation Benchmark (`craft_rag/data/eval/craft_ground_truth_eval.json`)**: 28-query ground-truth benchmark (22 answerable domain queries across 5 craft categories + 6 out-of-domain and adversarial abstention probes).
- **Automated Metric Suite (`craft_rag/pipeline/evaluate.py`)**: Computes Precision@1/3/5, Recall@1/3/5, MRR, nDCG@1/3/5 under binary relevance, citation precision/recall, and abstention accuracy. Benchmark report generated at `craft_rag/artifacts/benchmark_report.md`.
- **Web API & Interface (`craft_rag/server.py`, `craft_rag/web/`)**: Local REST API and responsive interface displaying passage-level chunk IDs, evidence levels, and verified provenance citations. See the [RAG integration guide](docs/rag_integration.md) for environment setup, index building, demos, web use, and tests.

## Test discovery

Run the repository's test suite with:

```powershell
python -m pytest --basetemp=temp_pytest tests/test_data_pipeline.py tests/test_craft_rag.py -v
```

> **Note on Environment & Test Execution:**
> - `tests/test_retrieval.py` directly imports `faiss`. On Windows under Python 3.14 (pre-release), `faiss-cpu` C-extension DLL initialization fails (`code 0xc0000008` / invalid handle) because binary wheels for Windows Python 3.14 are not yet available; run under Python 3.11 for native faiss testing.
> - The local HTTP server test `test_local_server_status_and_query_api` requires socket binding permissions (pass `-k "not test_local_server_status_and_query_api"` in sandbox environments where loopback ports are restricted).
> - On Windows sandbox environments, pytest requires `--basetemp=temp_pytest` because sandbox AppContainer restrictions prevent symlink/numbered-dir creation in the default system `%TEMP%`.

The duplicate test file under `pending_data_review/tests/` is intentionally excluded from the project-level pytest discovery to avoid an import-path mismatch when running from the repo root.
