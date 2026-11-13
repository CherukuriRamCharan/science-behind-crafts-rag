# Evaluation and Generation Closure Report

**Date:** 2026-11-13
**Branch:** `codex/m30-m33-runner-validator`
**Current committed implementation:** `f6aea4e`
**Dataset:** approved v2, 45 questions (33 answerable, 12 unanswerable); original v1 retained.

## Finding closure table

| Finding | Status | Rerunnable proof | Commit(s) | Remaining work |
|---|---|---|---|---|
| M26 citation precision | DONE | `python -m pytest -q tests/test_evaluation.py -k 'm26 or m27 or m28 or m29 or m35'` | `ce5ec3f` | None identified. |
| M27 citation validity | DONE | Same Branch 1 regression command | `ce5ec3f` | None identified. |
| M28 citation parsing | DONE | Same Branch 1 regression command | `ce5ec3f` | None identified. |
| M29 evidence-status leakage | DONE | Same Branch 1 regression command | `ce5ec3f` | None identified. |
| M30 runner / Transformers | PARTLY | `python -m pytest -q tests/test_evaluation.py` (59 passed); `python src/run_evaluation.py --skip-llm` completed on v2 | `a303129`, `dcda078` | Flag routing is covered, but real Transformers model loading and generation remain unverified: Python 3.13.7 environment has no torch, transformers, sentence-transformers, or accelerate; Hugging Face DNS lookup failed. |
| M31 retriever and `top_k` wiring | DONE | `python -m pytest -q tests/test_evaluation.py` (59 passed) | `a303129` | None identified. |
| M32 results directory wiring | DONE | `python -m pytest -q tests/test_evaluation.py` (59 passed) | `a303129` | None identified. |
| M33 dataset validation constraints | DONE | `python -m src.validate_eval_dataset` (45/45 passed, 18/33 card-grounded, overlap mean 32.75%, 0 at/above 50%); evaluation tests pass | `a303129` | None identified for approved v2. |
| M34 chart labels and axes | DONE | `python -m pytest -q tests/test_evaluation.py` (59 passed) | `f2f64b3` | None identified. |
| M35 deterministic tie-breaking | DONE | Branch 1 regression command above; `tests/test_retriever_h3_m35.py` included (58 passed total) | `f2f64b3` | None identified. |
| L41 data-derived answerable count | DONE | `python -m pytest -q tests/test_evaluation.py` (59 passed) | `f2f64b3` | None identified. |
| H5 measured error classification | DONE | `python -m pytest -q tests/test_evaluation.py` (59 passed), including measured-rank regression cases | `09ab3da` | None identified. |
| C2 dataset builder, manifests, entry points | DONE | `python -m src.validate_eval_dataset`; `python src/run_evaluation.py --skip-llm`; `tests/test_run_manifest.py`, `tests/test_build_eval_dataset.py` are included in the filtered suite | `da5a55b`, `dcda078` | v1 remains available at `data/eval/eval_questions.jsonl`; default is v2. |
| H2 grounded generation structure | PARTLY | `python -m pytest -q tests/test_craft_rag.py::test_transformers_backend_generates_and_rejects_invalid_citations tests/test_craft_rag.py::test_h2_transformers_json_grounding_and_fallback tests/test_craft_rag.py::test_h2_extractive_strips_page_prefix_and_returns_confidence tests/test_craft_rag.py::test_h2_confidence_is_present_for_abstention tests/test_server_stage4e.py tests/test_generation_stage4b.py` (19 passed) | `f916de9`, `eaccf8e`, `f6aea4e` | Strict JSON and passage-span enforcement are tested with a stub model, not a real Transformers model. Real-model behavior is blocked by the M30 environment issue. |
| H1 real embeddings / E5 | BLOCKED | `python -c "import importlib.util; print({m: bool(importlib.util.find_spec(m)) for m in ['torch','transformers','sentence_transformers','accelerate']})"`; Hugging Face config HEAD request failed with “No such host is known.” | No change; teammate-owned | Ram/Vasanthi: install the embedding stack in a network-enabled environment and verify the actual E5 backend. Config remains hashing. |
| H3 merged retrieval behavior | DONE | `python -m pytest -q tests/test_retriever_h3_m35.py` (covered in the 58-test regression run); `tests/test_craft_rag.py::test_local_server_status_and_query_api` passes with loopback enabled | merged before this work; label compatibility updated in this commit | The API maps normalized H3 craft IDs to display labels while request normalization remains unchanged. |
| H6 shared retriever | DONE | `craft_rag/server.py`, `craft_rag/cli.py`, `src/evaluate_generation.py`, and `src/evaluate_retrieval.py` use the shared `HybridRetriever` path; retrieval/evaluation tests pass | merged before this work | Ram/Vasanthi own future changes. |
| H7 cached index | PARTLY | Embedding cache is configured; `build_indices` still constructs an index during evaluation runs | merged before this work | Ram/Vasanthi: verify and implement persistent index loading if required by the review. |
| C1 single artifact writer | PARTLY | Current v2 evaluation produced the manifest, CSVs, and charts in `data/eval/results/`; no single README regeneration writer was verified | merged before this work | Ram/Vasanthi: establish and verify a single writer and consistent README regeneration. |

## Generation and abstention measurements

The latest `python src/run_evaluation.py --skip-llm` run used v2 and the configured hashing backend, with runner default `top_k=5`. Hybrid false abstention remained **25/33 (75.8%)** and unanswerable abstention accuracy was **12/12 (100%)**. BM25 was 27/33 (81.8%); dense hashing was 26/33 (78.8%).

| Hybrid v2 false-abstention cause | Count |
|---|---:|
| Generator sentence relevance gate | 25 |
| `minimum_dense_similarity` | 0 (setting is not applied by the shared retriever) |
| `minimum_score` | 0 |
| Other | 0 |

The 2022 World Cup winner, France capital, glass annealing, and sourdough prompts each returned `answer_mode=abstained`, `abstained=true`, and zero citations/sources. No threshold was changed, and no LLM metrics are reported. The v1-to-v2 retrieval MRR values retained in the detailed evaluation report are BM25 0.7863→0.6059, dense hashing 0.7640→0.4919, hybrid no-dedup 0.7965→0.6361, and hybrid pipeline 0.7680→0.6165 (each version N=33). Paired tests and confidence intervals are in the report; no retriever winner is claimed.

## Verification

- `python -m src.validate_eval_dataset` — passed, 0 validation errors.
- `python src/run_evaluation.py --skip-llm` — passed; generated v2 retrieval/generation tables, four charts, and a manifest for 569 corpus chunks (SHA-256 `aaa1d0ff746d99a35d8bdadd68784a4ddb0a59a1570353d27da3610bd0821489`). Manifest code SHA: `f6aea4e6db5db30821ff792ddd7452fd65fd28e6`.
- `python -m pytest -q tests/test_evaluation.py` — 59 passed.
- H2 targeted command in the finding table — 19 passed.
- `python -m pytest -q tests/test_evaluation.py -k 'm26 or m27 or m28 or m29 or m35' tests/test_retriever_h3_m35.py` — 58 passed, 37 deselected.
- Full suite: 184 passed on Linux Python 3.12 (183 plus this fix); FAISS DLL failure is Windows-only. (Linux result supplied for this closure.)
- Local Windows Python 3.13.7: `python -m pytest -q tests/` is blocked during collection of `tests/test_retrieval.py` because FAISS `_swigfaiss` fails to load (`DLL load failed while importing _swigfaiss`).
- Local Windows `python -m pytest -q tests/ --ignore=tests/test_retrieval.py` — 161 passed, 1 skipped; loopback was enabled for this run.
- `py -0p` — no Python versions registered; WSL is not installed, so the provided Linux Python 3.12 full-suite result could not be independently rerun in this workspace.
- `python -m compileall -q craft_rag src tests` and `git diff --check` — passed.

## Mentor handover

M30–M33 runner/validator work and the dataset entry points are committed. H2 now verifies strict JSON, requires a citation for every answer sentence, checks passage support, retains matched source sentences as citation quotes, strips real PDF page labels from displayed extractive text, exposes `answer_mode`, `fallback_reason`, and confidence, and falls back explicitly on malformed or unsupported output. Ishan's remaining generation work was completed by me because he was unwell.

M30 and H2 remain partial until a real Transformers model can load and generate. H1 remains blocked on E5 dependencies and network. H7 and C1 remain with Ram/Vasanthi; H6 is present and shared. The measured v2 hashing false-abstention rate is high (75.8%) and must not be tuned against hashing. Remaining risks are small N, lexical overlap, and single-annotator labels.
