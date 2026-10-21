# Science Behind Crafts RAG: Comprehensive Evaluation Report
**Evaluation module**: RAG retrieval and answer evaluation for `science-behind-crafts-rag`
**Mode**: Offline extractive generator (`--skip-llm`); no LLM-based metrics
**Corpus**: Unified multi-source index, 332 passages (194 full-paper chunks, 66 metadata cards, 72 process observations)
**Dataset**: 45 auto-drafted questions (33 answerable, 12 unanswerable); not fully human-verified
**Date**: October 2026
**Reproducibility**: Deterministic across repeated runs (checked by file checksums)

---

## 1. Executive Summary & Assessment Caveats

This report benchmarks the retrieval and generation components of the `science-behind-crafts-rag` hybrid RAG system over craft-science evidence.

### Dataset Status
The 45 benchmark questions were auto-drafted from extracted corpus cards and observations. Gold labels for `q022` and `q023` were adjusted after the first retrieval results revealed chunk alignment errors. The dataset has not been fully human-verified by external domain scientists or craftspersons.

### Essential Evaluation Caveats:
1. **Sample Size & Statistical Significance ($N=33$)**: With $N=33$ answerable questions, differences between retrievers may be within noise (no significance test was run, N=33). BM25 achieved Hit@5 = 90.91% (30/33) and Hybrid Pipeline achieved Hit@5 = 87.88% (29/33), a margin of only a single question. **No retriever is claimed as an absolute winner.**
2. **Dense Retriever is a Hashing Baseline**: In `craft_rag/config.yaml`, `embedding.backend: hashing` denotes an offline 128-dimensional random hashing baseline, not a pre-trained semantic neural model. It provides a low-dimensional offline baseline; no general conclusions about neural dense retrieval are drawn.
3. **High Lexical Overlap**: High lexical overlap between questions and gold quotes (average 58.84%, with 22 of 33 answerable questions exceeding 50%) likely favors BM25; retrieval scores may be optimistic.
4. **Offline Extractive Generator Limitations**: Generation metrics evaluate only the fallback extractive summarizer (`--skip-llm`), which concatenates raw text chunks. These metrics cannot be generalized to an LLM-based generator.
5. **Teammate Deduplication Effect**: The teammate retriever in `craft_rag/pipeline/retrieve.py` enforces deduplication on `(source_id, title)`. BM25 and Dense do not use this filter. To enable a fair comparison, both `hybrid (no dedup)` and `hybrid (pipeline with dedup)` are evaluated.
6. **Corpus Data-Quality Warning**: The teammate corpus builder outputs: `Prepared corpus has 62 data-quality issue(s); details are included in the corpus manifest (artifacts/corpus_manifest.json)`.
7. **Future Work Deferred (Scope Reductions)**: Per time-constrained project instructions, the following analyses were excluded and deferred to future work:
   - Ablations beyond retriever mode (e.g. chunk size, overlap, dense/sparse weights)
   - Bootstrap confidence intervals
   - LLM-as-a-judge qualitative evaluation
   - Paraphrase robustness testing
   - Latency and throughput benchmarking
   - Secondary 194-chunk full-paper comparison corpus

---

## 2. Dataset Design & Label-Fix Transparency

### 2.1 Question Dataset Distribution

The evaluation benchmark contains **45 test questions**: **33 answerable questions** (including 1 card pointer) and **12 unanswerable questions** ($33 + 12 = 45$).

| Question Type | Answerable | Unanswerable | Total Count | % of Benchmark |
| :--- | :---: | :---: | :---: | :---: |
| **Factual** | 14 | 0 | 14 | 31.1% |
| **Process** | 8 | 0 | 8 | 17.8% |
| **Scientific Explanation** | 6 | 0 | 6 | 13.3% |
| **Multi-hop** | 4 | 0 | 4 | 8.9% |
| **Card Pointer (`q031`)** | 1 | 0 | 1 | 2.2% |
| **Unanswerable (Near-domain & OOD)** | 0 | 12 | 12 | 26.7% |
| **Total** | **33** | **12** | **45** | **100.0%** |

- **Card Grounding Ratio**: **18 / 33 answerable questions (54.5%)** are grounded directly in PDF/DOCX metadata cards ($\ge 50\%$ required).
- **Full Research Paper Chunks**: **10 / 33 answerable questions (30.3%)** are grounded in segmented text chunks from primary papers (`WBC-0051`, `WBC-2016`).
- **Process Observations**: **5 / 33 answerable questions (15.2%)** are grounded in artisan workbook observations (`process-observation-*`).

### 2.2 Gold Record Group Handling & Deduplication Rule

Because `source_id` is shared across distinct document representations (e.g. metadata cards vs full papers) and a single chunk has multiple equivalent IDs across ingestion stages (`chunk_id`, `record_id`, `provenance_id`), each question specifies `gold_record_groups`.

**Collapsing Rule**: Before computing any $@k$ retrieval metric, the ranked candidate list is collapsed by gold record group. If an alias of an already-seen gold group appears, it is skipped; if an irrelevant duplicate appears, it is skipped. Thus, rank index $k$ strictly denotes **$k$ distinct document records**.

### 2.3 Gold Label Adjustments & Post-Hoc Transparency

During preliminary evaluation, four questions were diagnosed and corrected after inspecting retrieval failures. The team must note that the dataset was adjusted after inspecting initial results:

| QID | Field Modified | Old Value | New Value | Reason for Change |
| :---: | :--- | :--- | :--- | :--- |
| **`q002`** | `question_type` | `scientific_explanation` | `process` | Restored original distribution (process: 8, sci: 7/6) as tool socket impact mechanics describes craft tool maintenance/use. |
| **`q022`** | `gold_record_ids` & `gold_evidence_quote` | `WBC-0051::doc0006::0000` (Hand anatomy text) | `WBC-0051::doc0005::0000` (Grip strength predicting weakness) | Initial label was an off-by-one chunk error; `doc0005` contains the true finding that grip strength predicts muscle weakness. |
| **`q023`** | `gold_record_ids` & `gold_evidence_quote` | `WBC-0051::doc0007::0000` (Handle dimensions text) | `WBC-0051::doc0006::0000` (Hand anatomy text: bones, tendons...) | Initial label pointed to handle dimensions; `doc0006` contains the actual anatomical structures. |
| **`q031`** | `question_type` | `scientific_explanation` | `card_pointer` | Reclassified to `card_pointer` with natural craft question. Chunk `WBC-0003::doc0003::0000` is merely a literature pointer that summarizes a review rather than stating why nodes are weaker. |

### 2.4 Retrieval Performance Before vs After Label Changes

To measure the exact numerical impact of post-hoc adjustments on the 33 answerable questions:

- **State A (Initial Dataset Before Any Label Fixes)**: `q022` and `q023` mislabeled; `q031` natural.
- **State B (Final Evaluated Dataset)**: `q022` and `q023` fixed to true text; `q031` natural (`card_pointer`).
- **State C (Post-Hoc Aligned Dataset)**: `q022` and `q023` fixed; `q031` artificially aligned to card wording.

| Dataset State | Retriever | MRR | HitRate@1 | HitRate@5 | Recall@5 | nDCG@5 |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **State A (Before Fixes)** | BM25 | 0.8344 | 0.7879 | 0.8788 | 0.8788 | 0.8413 |
| | Dense (Hashing) | 0.6626 | 0.6061 | 0.6970 | 0.6970 | 0.6573 |
| | Hybrid (Pipeline) | 0.7778 | 0.7273 | 0.8485 | 0.8485 | 0.7958 |
| **State B (Final Evaluated)** | **BM25** | **0.8657** | **0.8182** | **0.9091** | **0.9091** | **0.8716** |
| | **Dense (Hashing)** | **0.6753** | **0.6061** | **0.7273** | **0.7273** | **0.6765** |
| | **Hybrid (Pipeline)** | **0.8081** | **0.7576** | **0.8788** | **0.8788** | **0.8261** |
| **State C (Post-Hoc Aligned q031)** | BM25 | 0.8960 | 0.8485 | 0.9394 | 0.9394 | 0.9019 |
| | Dense (Hashing) | 0.7056 | 0.6364 | 0.7576 | 0.7576 | 0.7068 |
| | Hybrid (Pipeline) | 0.8384 | 0.7879 | 0.9091 | 0.9091 | 0.8564 |

> [!IMPORTANT]
> Correcting `q022` and `q023` improved BM25 Hit@5 from 87.88% to 90.91% (+3.03%). Artificially phrasing `q031` with card keywords further inflated it to 93.94% (+3.03%). State B reflects the honest evaluation benchmark.

### 2.5 Question-to-Quote Overlap & Lexical Bias

The token overlap between each question and its gold evidence quote is computed as:
$$\text{Overlap} = \frac{|T_{\text{question}} \cap T_{\text{quote}}|}{|T_{\text{question}}|}$$
where $T = \{w \in \text{tokens} : w \notin \text{STOPWORDS} \land \text{len}(w) > 2\}$.

- **Average Overlap**: **58.84%** across all 33 answerable questions.
- **Questions > 50% Overlap**: **22 questions** (e.g. `q015` at 90.0%, `q003` at 87.5%, `q019` at 85.7%, `q011` at 83.3%, `q001` at 80.0%, `q029` at 80.0%, `q024` at 77.8%, `q005` at 75.0%, `q014` at 75.0%, `q020` at 75.0%).
- **Cause of Elevated Overlap**: In Phase 1, broad paraphrasing achieved 21.76% overlap (0 questions > 50%). However, enforcing strict Question-Answer alignment (ensuring questions are answerable *strictly and only* from the 1–2 sentence gold quote without external factual claims) required reusing the quotes' exact technical domain vocabulary (e.g., "socket", "collars", "flanges", "grip strength", "muscle fatigue", "veneer layers", "strain", "dimensional stability").
- **Limitation**: *High lexical overlap between questions and gold quotes likely favors BM25; retrieval scores may be optimistic.*

---

## 3. Retrieval Evaluation & Code-Path Analysis

### 3.1 Code Paths Disclosed

- **`bm25`**: Direct BM25 candidate scoring without title deduplication (`retrieve_bm25`).
- **`dense (hashing baseline)`**: Direct 128-dim hash vector scoring without title deduplication (`retrieve_dense`).
- **`hybrid (pipeline with dedup)`**: Calls teammate `craft_rag.pipeline.retrieve.HybridRetriever.retrieve()`, which drops all subsequent chunks sharing `(source_id, title)`.
- **`hybrid (no dedup)`**: Direct Reciprocal Rank Fusion on BM25 + Dense without title suppression, establishing a fair like-for-like comparison.

### 3.2 Overall Retrieval Performance (State B Final, N=33 Answerable)

All metrics evaluated at $k \in \{1, 3, 5, 10\}$ using alias group collapsing:

| Retriever Backend | Code Path Setting | MRR | HitRate@1 | HitRate@5 | HitRate@10 | Recall@5 | nDCG@5 |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **BM25 (Sparse)** | Direct (No Dedup) | **0.8657** | **0.8182** | **0.9091** | **0.9697** | **0.9091** | **0.8716** |
| **Dense (hashing baseline)** | Direct (No Dedup) | 0.6753 | 0.6061 | 0.7273 | 0.8182 | 0.7273 | 0.6765 |
| **Hybrid (no dedup)** | Fair Like-for-Like (No Dedup) | 0.7996 | 0.7273 | 0.8485 | 0.8788 | 0.8485 | 0.8037 |
| **Hybrid (pipeline with dedup)** | Teammate Pipeline (With Dedup) | 0.8081 | 0.7576 | 0.8788 | 0.8788 | 0.8788 | 0.8261 |

### 3.3 Slices by Document Type and Question Type

#### Document Type Slices:
| Document Type Slice | Count | Retriever | MRR | HitRate@1 | HitRate@5 | Recall@5 | nDCG@5 |
| :--- | :---: | :--- | :---: | :---: | :---: | :---: | :---: |
| **Source Document Cards** | 18 | BM25 | **1.0000** | **1.0000** | **1.0000** | **1.0000** | **1.0000** |
| | | Dense (Hashing) | 0.7201 | 0.6667 | 0.7222 | 0.7222 | 0.7017 |
| | | Hybrid (No Dedup) | 0.8929 | 0.8333 | 0.9444 | 0.9444 | 0.9034 |
| | | Hybrid (Pipeline) | 0.9444 | 0.8889 | **1.0000** | **1.0000** | 0.9590 |
| **Full Paper Chunks** | 10 | BM25 | 0.7444 | 0.6000 | **0.9000** | **0.9000** | **0.7762** |
| | | Dense (Hashing) | 0.6798 | 0.6000 | 0.8000 | 0.8000 | 0.7062 |
| | | Hybrid (No Dedup) | **0.7650** | **0.7000** | 0.8000 | 0.8000 | 0.7631 |
| | | Hybrid (Pipeline) | 0.7000 | **0.7000** | 0.7000 | 0.7000 | 0.7000 |
| **Process Observations** | 5 | BM25 | **0.6250** | **0.6000** | 0.6000 | 0.6000 | **0.6000** |
| | | Dense (Hashing) | 0.5050 | 0.4000 | 0.6000 | 0.6000 | 0.5262 |
| | | Hybrid (No Dedup) | 0.5333 | 0.4000 | 0.6000 | 0.6000 | 0.5262 |
| | | Hybrid (Pipeline) | 0.5333 | 0.4000 | **0.8000** | **0.8000** | **0.6000** |

#### Question Type Slices:
| Question Type Slice | Count | Retriever | MRR | HitRate@1 | HitRate@5 | Recall@5 | nDCG@5 |
| :--- | :---: | :--- | :---: | :---: | :---: | :---: | :---: |
| **Factual** | 14 | BM25 | **0.9365** | **0.9286** | **0.9286** | **0.9286** | **0.9286** |
| | | Dense (Hashing) | 0.8040 | 0.7857 | 0.7857 | 0.7857 | 0.7857 |
| | | Hybrid (No Dedup) | 0.8994 | 0.8571 | **0.9286** | **0.9286** | 0.9022 |
| | | Hybrid (Pipeline) | 0.8929 | 0.8571 | **0.9286** | **0.9286** | 0.9022 |
| **Process** | 8 | BM25 | **0.9375** | **0.8750** | **1.0000** | **1.0000** | **0.9539** |
| | | Dense (Hashing) | 0.6531 | 0.6250 | 0.6250 | 0.6250 | 0.6250 |
| | | Hybrid (No Dedup) | 0.8824 | 0.8750 | 0.8750 | 0.8750 | 0.8750 |
| | | Hybrid (Pipeline) | 0.8750 | **0.8750** | 0.8750 | 0.8750 | 0.8750 |
| **Scientific Explanation** | 6 | BM25 | **1.0000** | **1.0000** | **1.0000** | **1.0000** | **1.0000** |
| | | Dense (Hashing) | 0.5463 | 0.3333 | 0.8333 | 0.8333 | 0.6154 |
| | | Hybrid (No Dedup) | 0.6786 | 0.5000 | 0.8333 | 0.8333 | 0.7103 |
| | | Hybrid (Pipeline) | 0.7500 | 0.6667 | 0.8333 | 0.8333 | 0.7718 |
| **Multi-hop** | 4 | BM25 | 0.4896 | 0.2500 | 0.7500 | 0.7500 | 0.5327 |
| | | Dense (Hashing) | 0.6312 | 0.5000 | 0.7500 | 0.7500 | 0.6577 |
| | | Hybrid (No Dedup) | **0.6667** | **0.5000** | 0.7500 | 0.7500 | 0.6577 |
| | | Hybrid (Pipeline) | **0.6667** | **0.5000** | **1.0000** | **1.0000** | **0.7500** |
| **Card Pointer (`q031`)** | 1 | All Retrievers | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |

### 3.4 Balanced Findings on Teammate Title Deduplication

The evaluation reveals two distinct effects of `(source_id, title)` deduplication:
1. **Negative Effect on Multi-Chunk Papers**: For long papers with dozens of chunks sharing the same title (e.g. `WBC-2016` with 97 chunks, `WBC-0051` with 10 chunks), selecting an irrelevant chunk at rank 1 permanently drops all subsequent chunks from that paper. This directly caused retrieval failures for `q023`, `q025`, and `q026`.
2. **Positive Effect on Overall Metrics**: For single-chunk metadata cards, suppressing duplicate card entries prevents identical cards from crowding the top 5 slots. Consequently, the net effect of title deduplication on overall Hybrid metrics is slightly positive: **MRR 0.8081 with dedup vs 0.7996 without dedup**, and **HitRate@5 0.8788 with dedup vs 0.8485 without dedup**. The deduplication is not simply harmful; it represents a design tradeoff that favors diverse source cards over multi-chunk paper depth.

---

## 4. Generation & Citation Evaluation (Offline Extractive Mode)

Generation was evaluated using `craft_rag.pipeline.generate.GroundedGenerator` in extractive mode (`--skip-llm`).

| Metric | BM25 Backend | Dense (hashing baseline) | Hybrid Backend |
| :--- | :---: | :---: | :---: |
| **Citation Precision** | 0.4646 | 0.4343 | 0.3283 |
| **Citation Recall** | 0.5909 | 0.5455 | **0.6364** |
| **Citation Validity (Hallucinations)** | 0 / 45 (100% Valid) | 0 / 45 (100% Valid) | 0 / 45 (100% Valid) |
| **Abstention Accuracy (12 Unanswerables)** | 0.0% (0 / 12) | 50.0% (6 / 12) | 16.67% (2 / 12) |
| **False Abstention Rate (33 Answerables)** | 0.0% (0 / 33) | 0.0% (0 / 33) | 0.0% (0 / 33) |
| **Evidence-Status Leakage** | 0 / 5 (0.0%) | 0 / 5 (0.0%) | 0 / 5 (0.0%) |
| **Token F1 vs Gold Reference** | 0.0792 | 0.0793 | 0.0885 |
| **ROUGE-L vs Gold Reference** | 0.0717 | 0.0681 | 0.0772 |

### Explicit Limitations of Offline Extractive Generation:
1. **0% False-Abstention Rate is Trivial**: Under BM25, the extractive generator never abstained on any query (0/12 on unanswerables). Because it never abstains at all, its 0.0% false-abstention rate on answerables is a trivial consequence of refusing to reject queries.
2. **Citation Validity is 100% by Construction**: The extractive generator constructs answers strictly by formatting strings from retrieved passage objects: `f"{text} [{source_id} | {level}]"`. It has no autoregressive decoder, so hallucinating a non-existent citation ID is structurally impossible.
3. **Evidence-Status Leakage (0/5) is Uninformative**: With $N=5$ process observation questions and an extractive concatenator, the system does not generate free-form text.
4. **Low Token F1 Reflects Chunk Length, Not Answer Quality**: Token F1 (~0.08) is low because the extractive summarizer pastes up to 3 long raw passages (200–500 words), while reference gold answers are concise 1–2 sentence factual summaries.
5. **Abstention Cannot Be Compared Across Retrievers**: Dense achieved 50% abstention on unanswerables solely because its 128-dim hash vectors retrieved poor chunks that failed the 8% query token overlap check. It abstains more because it retrieves worse, not because it possesses superior epistemic confidence.
6. **Zero Generalization to LLMs**: None of these generation numbers can be extrapolated to a generative LLM backend (e.g. Qwen or Llama).

---

## 5. Chunk Verifications & Error Analysis

### 5.1 Verification of `q031` (Card Pointer)
- **Question**: *"What explains why bamboo node regions are mechanically weaker than internodes?"*
- **Gold Chunk Text (`WBC-0003::doc0003::0000`)**:
  > `'ScienceDirect review with quantified MOE/MOR/tensile data and node-weakness explanation; strong evidence for engineering-based sustainability claims.'`
- **Assessment**: The text does **NOT** state why bamboo nodes are mechanically weaker than internodes. It merely notes that a ScienceDirect review contains an explanation. Reclassifying it as `card_pointer` isolates this corpus limitation: natural queries fail to retrieve this card because the card lacks the biological mechanism.

### 5.2 Verification of `q022` and `q023` (Exact Substring Confirmation)
- **`q022`**: *"Why are data on hand grip strength meaningful beyond ergonomic tool design?"*
  - **Gold Chunk Text (`WBC-0051::doc0005::0000`)**:
    > `'], [2], [3]. Grip strength is a magnitude of muscle strength generated by the hand and forearm muscles while gripping an object. Hand dynamometer and force gauge are the most common devices used to measure the hand grip strength. In industrial ergonomics, information on physical strength is useful for three purposes: 1) hand tool design  as guidelines and references for creating consumer products and industrial hand tools [3], [4]. 2) Worker selection and placement programs - to ensure the jobs involving heavy physical demands are not performed by those who lack the necessary strength capabilities [5]. 3) Job design  to match the worker strength and the job demands [6]. In addition to that, data on hand grip strength are also meaningful for predicting muscle weakness [7], determining the functional capabilities of Abstract: Hand is made up of bones, tendons, ligaments, nerves and'`
  - **Quote**: `'data on hand grip strength are also meaningful for predicting muscle weakness [7], determining the functional capabilities of'` $\rightarrow$ **Confirmed Exact Substring**.
  - **Alternative Chunks**: Adjacent chunk `WBC-0051::doc0006::0000` has the tail of this sentence due to sliding-window overlap (`kness [7], determining the functional capabilities of...`), but `doc0005` contains the complete proposition.
- **`q023`**: *"Which anatomical structures make up the human hand?"*
  - **Gold Chunk Text (`WBC-0051::doc0006::0000`)**:
    > `'kness [7], determining the functional capabilities of Abstract: Hand is made up of bones, tendons, ligaments, nerves and blood vessels that can be easily de bilitated and injured if the hand tool is not design ergonomically. Recently researchers have examined the effec ts of individual, environmental and occupational factors on hand grip strength. However, information on the influence of hand tool design on hand grip strength is still lacking. The aim of this paper is to provide a comprehensive review of factors influencing hand grip strength focusing more towards hand tool design factors. The authors searched the journal articles, book and guidelines from the online databases of Google Scholar, ScienceDirect and Pubmed. Fourteen factors includes handle diameter, handle l ength, handle orientation, handle shape, handle material, handle flange, handle inter distance/grip span, grip'`
  - **Quote**: `'Hand is made up of bones, tendons, ligaments, nerves and'` $\rightarrow$ **Confirmed Exact Substring**.
  - **Alternative Chunks**: `WBC-0051::doc0005::0000` ends with the beginning of the sentence (`Hand is made up of bones, tendons, ligaments, nerves and`), and `WBC-0051::doc0034::0000` mentions tendons/ligaments in wrist postures.
  - **BM25 Rank (9th) and Vagueness**: BM25 ranks `doc0006` at 9th because *"Which anatomical structures make up the human hand?"* is too generic and lacks craft-specific terminology. Segments of paper `WBC-2016` (robot carving with hand tools) repeat "human", "hand", and "structures" with higher frequency, pushing the gold chunk down.
  

### 5.3 Error Classification

All 23 failure instances across the pipeline were mapped and classified into three primary root causes (matching `data/eval/results/error_summary.csv` and `data/eval/results/error_analysis.json`):

1. **`retrieval_deduplication_drop`** (**4 questions**: `q023`, `q025`, `q026`, `q031`):
   - **Root Cause**: The teammate pipeline retriever (`craft_rag/pipeline/retrieve.py`) deduplicates candidate chunks on `(source_id, title)`. For multi-chunk research papers (`WBC-0051`, `WBC-2016`), only the first chunk of a paper is retained; subsequent relevant chunks are dropped even when they score highly in individual BM25 or Dense stages.

2. **`generation_abstention_failure`** (**10 questions**: `q034`, `q035`, `q036`, `q037`, `q038`, `q039`, `q041`, `q042`, `q043`, `q045`):
   - **Root Cause**: On near-domain unanswerable queries (e.g., questions asking about Udayagiri wood soaking duration or teak steam bending radii not present in the corpus), retrieved craft chunks contained incidental word matches (e.g., "wood", "carving", "bamboo") that exceeded the simple heuristic overlap threshold ($|T_{\text{query}} \cap T_{\text{passage}}| / |T_{\text{query}}| \ge 0.08$). Consequently, the extractive generator pasted irrelevant passages rather than abstaining with *"Insufficient evidence"*.

3. **`dense_hashing_capacity_limit`** (**9 questions**: `q004`, `q006`, `q008`, `q010`, `q018`, `q023`, `q026`, `q031`, `q032`):
   - **Root Cause**: The 128-dimensional offline token-hashing baseline (`TextEmbedder`) lacks the semantic representational capacity to bridge query terminology and passage phrasing without exact lexical matches or within crowded hash projections, causing the gold chunk to fall outside the top 5 candidates.

---

## 6. Reproducibility & Determinism Verification

The benchmark was executed twice consecutively via `python src/run_evaluation.py --skip-llm`. Both runs generated identical MD5 checksums:

- `retrieval_summary.csv` (`MD5: 59053b903c9c2b49f34c46456a2fbba7`)
- `retrieval_per_question.csv` (`MD5: 7aae4636c12489978ea07567718c9586`)
- `generation_summary.csv` (`MD5: 541a33f9825b31d050aee3a61d08620d`)
- `generation_per_question.csv` (`MD5: 0ed622b21d5895ca46579e52c104cc9b`)
- `error_summary.csv` (`MD5: c5f5131c62374b4b6c712073cd86e5b9`)
- `error_analysis.json` (`MD5: 82735848035cd4b65556551a016d4350`)
- `gold_label_fixes.json` (`MD5: c2edb7948ae1c4da95feedaf06418ded`)

Unit tests: `python -m pytest -q tests/` $\rightarrow$ **50 passed, 1 skipped**.
