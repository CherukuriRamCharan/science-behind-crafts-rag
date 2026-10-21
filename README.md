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

From the repository root, run:

python src/build_source_catalog.py
python src/workbook_analysis.py
python src/full_paper_chunker.py
python src/build_data_readiness_report.py

These scripts are designed to be idempotent and reproducible.

## Prepared-corpus RAG application

The integrated hybrid RAG application uses the prepared JSONL datasets directly. See the [RAG integration guide](docs/rag_integration.md) for environment setup, index building, demos, web use, and tests.

## Test discovery

Use the repository's root test suite with:

python -m pytest -q tests/

The duplicate test file under `pending_data_review/tests/` is intentionally excluded from the project-level pytest discovery to avoid an import-path mismatch when running from the repo root.


## Evaluation Suite

The repository includes a reproducible evaluation benchmark for retrieval and offline extractive generation:

- **Benchmark Dataset**: 45 auto-drafted questions (`data/eval/eval_questions.jsonl`): 33 answerable (54.5% grounded in PDF/DOCX metadata cards) + 12 unanswerable (hard near-domain, teammate unsourced, and out-of-domain baseline).
  - *Data Quality Warning*: Teammate corpus builder outputs: `Prepared corpus has 62 data-quality issue(s); details are included in the corpus manifest (artifacts/corpus_manifest.json)`.
  - *Question-to-Quote Overlap*: Average 58.84% across answerable questions (metric definition: $|T_{\text{question}} \cap T_{\text{quote}}| / |T_{\text{question}}|$ with stopwords removed, len > 2), with 22 questions > 50% due to strict single-sentence QA grounding.
  - *Limitation*: High lexical overlap between questions and gold quotes likely favors BM25; retrieval scores may be optimistic.
- **Collapsing Rule**: Evaluates retrieval after deduplicating ranked lists by gold record groups (aliases count once; slot $k$ represents $k$ distinct records).
- **Retrieval Performance (N=33 Answerable)**:
  - BM25 (Sparse, direct): MRR = 0.8657, HitRate@5 = 0.9091, Recall@5 = 0.9091, nDCG@5 = 0.8716
  - Dense (Hashing baseline): MRR = 0.6753, HitRate@5 = 0.7273, Recall@5 = 0.7273, nDCG@5 = 0.6765
  - Hybrid (No dedup): MRR = 0.7996, HitRate@5 = 0.8485, Recall@5 = 0.8485, nDCG@5 = 0.8037
  - Hybrid (Pipeline with dedup): MRR = 0.8081, HitRate@5 = 0.8788, Recall@5 = 0.8788, nDCG@5 = 0.8261
  *(Note: N=33 differences between retrievers are within sampling noise; no winner claimed. Teammate title deduplication in pipeline drops relevant chunks for multi-chunk papers).*
- **Generation & Citation Quality (`--skip-llm` Extractive Mode)**:
  - Evaluates offline extractive text concatenation; does not generalize to neural LLM generation.
  - Citation Validity: 100% valid by construction (only cites retrieved chunks).
  - Evidence Leakage: 0/5 instances (artisan observations not synthesized as proven law).
  - False Abstention: 0.0% (trivial, as extractive generator never abstains under BM25).
- **Run Evaluation**:
  ```bash
  python src/run_evaluation.py --skip-llm
  ```
- **Evaluation Report**: See [`data/metadata/evaluation_report.md`](data/metadata/evaluation_report.md) for full breakdown, case studies, label fix before/after comparisons, and figures.
