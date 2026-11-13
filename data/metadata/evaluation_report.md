# SBC-RAG Evaluation Report - v1/v2 comparison

**Run date:** 2026-11-13
**Corpus:** 569 searchable chunks; SHA-256 `aaa1d0ff746d99a35d8bdadd68784a4ddb0a59a1570353d27da3610bd0821489`
**Embedding:** `embedding.backend: hashing`; deterministic 384-dimensional token hashing. The configured model name is not loaded by this backend.
**Dataset:** 45 questions per version (33 answerable, 12 unanswerable); extractive generation; retrieval top-k=40, generation top-k=5. v1 is unchanged; v2 was built from the YAML spec.

## Benchmark wording and overlap

Overlap is the intersection of unique question and gold-quote content tokens divided by unique question content-token count (lowercase alphanumeric, stopwords removed, token length greater than two). v1 mean overlap was 58.84%; 22/33 answerable questions were above 50%, and 2/33 exactly 50%. v2 mean overlap is 32.75%; 0/33 are at or above 50%. **v2 reduces content overlap but adds source-name matching.** Retrieval differences therefore reflect changed question wording, not changed retrieval code.

v2 retains all 45 questions, gold answers, gold record IDs, and evidence quotes. Only the 24 approved answerable question wordings changed. Validation passed with 0 errors: 45 total, 33 answerable, 12 unanswerable, with 18/33 answerable items grounded in source cards (54.5%).

## Retrieval results

Metrics use the 33 answerable questions. Confidence intervals are percentile bootstrap 95% intervals with 10,000 resamples and seed 20261113. v1 and v2 use the same 569-chunk corpus and retrievers.

| Retriever | Version | MRR (95% CI) | Hit@5 | Recall@5 | nDCG@5 |
|---|---|---:|---:|---:|---:|
| BM25 | v1 | 0.7863 (0.6562-0.9015) | 0.8788 | 0.8788 | 0.8068 |
| BM25 | v2 | 0.6059 (0.4626-0.7453) | 0.7273 | 0.7273 | 0.6257 |
| Dense hashing | v1 | 0.7640 (0.6364-0.8864) | 0.7879 | 0.7879 | 0.7543 |
| Dense hashing | v2 | 0.4919 (0.3621-0.6245) | 0.6667 | 0.6667 | 0.5162 |
| Hybrid no-dedup | v1 | 0.7965 (0.6732-0.9105) | 0.9091 | 0.9091 | 0.8227 |
| Hybrid no-dedup | v2 | 0.6361 (0.5006-0.7707) | 0.7879 | 0.7879 | 0.6619 |
| Hybrid pipeline | v1 | 0.7680 (0.6450-0.8831) | 0.8788 | 0.8788 | 0.7926 |
| Hybrid pipeline | v2 | 0.6165 (0.4771-0.7525) | 0.7879 | 0.7879 | 0.6546 |

Paired bootstrap/sign-flip comparisons on matched qids (v1 minus v2 MRR, N=33): BM25 +0.1804 (95% CI 0.0842-0.2868, p=0.0013); dense hashing +0.2720 (0.1412-0.4120, p=0.0005); hybrid no-dedup +0.1604 (0.0370-0.2918, p=0.0197); hybrid pipeline +0.1515 (0.0043-0.3054, p=0.0645). These results quantify prompt-wording sensitivity. No retriever winner is claimed.

## Generation and abstention

| Dataset | Retriever | Abstention accuracy | False-abstention rate (answerable) |
|---|---|---:|---:|
| v1 | BM25 | 12/12 (100%) | 5/33 (15.2%) |
| v1 | Dense hashing | 12/12 (100%) | 6/33 (18.2%) |
| v1 | Hybrid | 12/12 (100%) | 5/33 (15.2%) |
| v2 | BM25 | 12/12 (100%) | 27/33 (81.8%) |
| v2 | Dense hashing | 12/12 (100%) | 26/33 (78.8%) |
| v2 | Hybrid | 12/12 (100%) | 25/33 (75.8%) |

Live API checks for the 2022 World Cup winner, the capital of France, glass annealing, and sourdough all returned `answer_mode=abstained`, `abstained=true`, and no sources. This was with the extractive backend; it does not verify Transformers generation. The high v2 false-abstention rate is a measured limitation after wording changes.

On the rerun after H2 grounding changes, hybrid v2 remained at 25/33 false abstentions (75.8%) and 12/12 correct unanswerable abstentions (100%). Inspecting those 25 questions against the shared retriever and generator attributed all 25 to the generator sentence relevance gate; minimum-score caused 0, minimum-dense-similarity caused 0, and other causes accounted for 0. The configured `minimum_dense_similarity: 0.25` is not applied by the active shared retriever, so it did not cause these outcomes. No threshold was changed. The four live abstention prompts remained abstained after the rerun.

| Measured cause | False abstentions |
|---|---:|
| Generator sentence relevance gate | 25 |
| `minimum_dense_similarity` | 0 (not applied in this path) |
| `minimum_score` | 0 |
| Other | 0 |

H2 now requires strict JSON from the Transformers path (`answer`, sentence-linked `citations`, and numeric `confidence`), verifies each cited answer sentence against its retrieved passage (at least 50% content-token overlap or a contiguous six-token match), and falls back to extractive output with a reason if parsing, citation, or support validation fails. Extractive, fallback, and abstained responses also carry confidence values. Real Transformers loading and generation remain unverified: PyTorch/Transformers are absent and Hugging Face is unreachable in the current environment. No LLM-based benchmark metrics are reported.

## Validation and reproducibility

- `python -c "from src.validate_eval_dataset import validate; raise SystemExit(0 if validate('data/eval/eval_questions_v2.jsonl') else 1)"` - passed with 0 errors.
- `python -m src.build_searchable_chunk_reference` - 569 chunks; SHA-256 above. Manifest: `data/eval/searchable_chunks_reference.manifest.json`.
- `data/eval/results_v1_final/` and `data/eval/results_v2_final/` each contain a `run_manifest.json`, retrieval and generation per-question files, summary tables, bootstrap CIs, paired tests, and figures.
- The prepared corpus reports 62 data-quality issues; details are in the generated corpus manifest.

## Limitations and remaining work

The sample is small (N=33 answerable, N=12 unanswerable), lexical matching influences retrieval, and annotations have a single annotator. The configured dense backend is hashing, not a real sentence-embedding model, so embedding-threshold calibration was not done. Current verification used Python 3.13.7. The selected evaluation, generator, and server regression tests passed; the full-suite result is reported in the closure report. Real Transformers and E5 model execution could not be verified because the required dependencies are absent and Hugging Face did not resolve from this environment. Regenerate these results after human review and when real embeddings are available.
