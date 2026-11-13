# Science Behind Crafts RAG — Benchmark Evaluation Report

## 1. Dataset & Query Overview

- **Total Benchmark Queries**: 28
- **Answerable Domain Queries**: 22 (evaluating retrieval, ranking, and citations)
- **Out-of-Domain & Adversarial Queries**: 6 (evaluating abstention accuracy)
- **Retrieval Cutoff (Top-K)**: 5 passages
- **Per-Source Diversification Limit**: 2 passages per source
- **Benchmark Source**: `C:\Users\SAMA\Projects\NPTEL Project\science-behind-crafts-rag\craft_rag\data\eval\craft_ground_truth_eval.json`

## 2. Metric Definitions & Evaluation Methodology

- **Precision@K**: $\frac{|\text{Unique Relevant Chunks in Top-K}|}{K}$. Unretrieved slots when $|R| < K$ count as non-relevant.
- **Recall@K**: $\frac{|\text{Unique Relevant Chunks in Top-K}|}{|\text{Target Chunks}|}$. Measures target passage coverage.
- **MRR (Mean Reciprocal Rank)**: $\frac{1}{\text{rank of first relevant chunk}}$, or $0.0$ if no target chunk is retrieved.
- **nDCG@K**: Normalized Discounted Cumulative Gain under binary relevance ($rel \in \{0, 1\}$), with logarithmic discounting $\log_2(\text{rank}+1)$ and ideal ranking normalizer.
- **Citation Precision / Recall**: Fraction of passages cited in structured answers that match ground-truth target chunks / fraction of ground-truth target chunks cited.
- **Abstention Accuracy**: Percentage of out-of-domain and adversarial queries where the system explicitly abstained (`abstained=True`) with zero fabricated citations (`citations=[]`).
> [!NOTE]
> Automated metrics evaluate deterministic passage identifiers and provenance citations against verified ground-truth targets. They reflect empirical retrieval and provenance grounding, without claiming unverified automated semantic entailment.

## 3. Overall Benchmark Performance

### A. Retrieval & Ranking Metrics (Answerable Queries, N=22)

| Metric | Cutoff @1 | Cutoff @3 | Cutoff @5 |
| :--- | :---: | :---: | :---: |
| **Precision** | 0.7727 | 0.4848 | 0.3364 |
| **Recall** | 0.2097 | 0.3815 | 0.4421 |
| **nDCG** | 0.7727 | 0.5919 | 0.5553 |

- **Mean Reciprocal Rank (MRR)**: **`0.8788`**
- **Citation Passage Precision**: **`0.4318`**
- **Citation Passage Recall**: **`0.1903`**
- **Citation Source Precision**: **`0.4318`**
- **Citation Source Recall**: **`0.3864`**

### B. Abstention Accuracy (Out-of-Domain & Adversarial Probes, N=6)

- **Abstention Accuracy**: **`100.0%`** (6/6 correct)
- All out-of-domain queries and ungrounded adversarial assertions safely returned `abstained=True` with `citations=[]`.

## 4. Per-Category Results Breakdown

| Category | Queries | P@5 | R@5 | MRR | nDCG@5 | Cit. Prec | Cit. Rec | Abstain Acc |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Adhesives, Finishing & Preservation | 4 | 0.2500 | 0.5000 | 0.5833 | 0.4333 | 0.2500 | 0.1250 | — |
| Bamboo Structural & Splitting Mechanics | 5 | 0.2800 | 0.5667 | 0.9000 | 0.5864 | 0.5000 | 0.2167 | — |
| Hallucination & Adversarial Probes | 3 | — | — | — | — | — | — | 100.0% |
| Insufficient Evidence & Out-of-Domain Probes | 3 | — | — | — | — | — | — | 100.0% |
| Multi-Source Craft Process Synthesis | 3 | 0.4000 | 0.3398 | 1.0000 | 0.6012 | 0.0000 | 0.0000 | — |
| Tool Ergonomics, Posture & Body Mechanics | 5 | 0.4000 | 0.2579 | 1.0000 | 0.5665 | 0.6000 | 0.2206 | — |
| Wood Carving Mechanics & Species Properties | 5 | 0.3600 | 0.5167 | 0.9000 | 0.5829 | 0.6000 | 0.3000 | — |

## 5. Detailed Per-Query Results

| ID | Type | Category | Target Chunks | Retrieved Chunks | P@5 | R@5 | MRR | nDCG@5 | Status |
| :--- | :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| Q01 | Multi | Wood Carving Mechanics & Species Properties | 4 | 5 | 0.60 | 0.75 | 1.00 | 0.83 | Answered (3 cited) |
| Q02 | Multi | Wood Carving Mechanics & Species Properties | 4 | 5 | 0.40 | 0.50 | 0.50 | 0.41 | Answered (3 cited) |
| Q03 | Multi | Wood Carving Mechanics & Species Properties | 4 | 5 | 0.40 | 0.50 | 1.00 | 0.59 | Answered (1 cited) |
| Q04 | Multi | Wood Carving Mechanics & Species Properties | 3 | 5 | 0.20 | 0.33 | 1.00 | 0.47 | Abstained |
| Q05 | Single | Wood Carving Mechanics & Species Properties | 2 | 5 | 0.20 | 0.50 | 1.00 | 0.61 | Answered (1 cited) |
| Q06 | Multi | Bamboo Structural & Splitting Mechanics | 3 | 5 | 0.20 | 0.33 | 1.00 | 0.47 | Answered (1 cited) |
| Q07 | Multi | Bamboo Structural & Splitting Mechanics | 2 | 5 | 0.40 | 1.00 | 0.50 | 0.65 | Abstained |
| Q08 | Single | Bamboo Structural & Splitting Mechanics | 2 | 5 | 0.20 | 0.50 | 1.00 | 0.61 | Answered (2 cited) |
| Q09 | Single | Bamboo Structural & Splitting Mechanics | 2 | 5 | 0.20 | 0.50 | 1.00 | 0.61 | Abstained |
| Q10 | Multi | Bamboo Structural & Splitting Mechanics | 4 | 5 | 0.40 | 0.50 | 1.00 | 0.59 | Answered (2 cited) |
| Q11 | Multi | Tool Ergonomics, Posture & Body Mechanics | 101 | 5 | 0.40 | 0.02 | 1.00 | 0.55 | Answered (3 cited) |
| Q12 | Multi | Tool Ergonomics, Posture & Body Mechanics | 6 | 5 | 0.60 | 0.50 | 1.00 | 0.70 | Answered (2 cited) |
| Q13 | Multi | Tool Ergonomics, Posture & Body Mechanics | 4 | 5 | 0.20 | 0.25 | 1.00 | 0.39 | Answered (3 cited) |
| Q14 | Multi | Tool Ergonomics, Posture & Body Mechanics | 101 | 5 | 0.40 | 0.02 | 1.00 | 0.55 | Answered (2 cited) |
| Q15 | Multi | Tool Ergonomics, Posture & Body Mechanics | 4 | 5 | 0.40 | 0.50 | 1.00 | 0.64 | Answered (3 cited) |
| Q16 | Single | Adhesives, Finishing & Preservation | 2 | 5 | 0.20 | 0.50 | 0.50 | 0.39 | Abstained |
| Q17 | Multi | Adhesives, Finishing & Preservation | 4 | 5 | 0.40 | 0.50 | 0.33 | 0.35 | Answered (1 cited) |
| Q18 | Single | Adhesives, Finishing & Preservation | 2 | 5 | 0.20 | 0.50 | 1.00 | 0.61 | Answered (2 cited) |
| Q19 | Single | Adhesives, Finishing & Preservation | 2 | 5 | 0.20 | 0.50 | 0.50 | 0.39 | Abstained |
| Q20 | Multi | Multi-Source Craft Process Synthesis | 3 | 5 | 0.40 | 0.67 | 1.00 | 0.77 | Abstained |
| Q21 | Multi | Multi-Source Craft Process Synthesis | 103 | 5 | 0.40 | 0.02 | 1.00 | 0.55 | Answered (1 cited) |
| Q22 | Multi | Multi-Source Craft Process Synthesis | 6 | 5 | 0.40 | 0.33 | 1.00 | 0.49 | Abstained |
| Q23 | Abstain | Insufficient Evidence & Out-of-Domain Probes | 0 | 5 | — | — | — | — | Correct Abstention |
| Q24 | Abstain | Insufficient Evidence & Out-of-Domain Probes | 0 | 5 | — | — | — | — | Correct Abstention |
| Q25 | Abstain | Insufficient Evidence & Out-of-Domain Probes | 0 | 5 | — | — | — | — | Correct Abstention |
| Q26 | Abstain | Hallucination & Adversarial Probes | 0 | 5 | — | — | — | — | Correct Abstention |
| Q27 | Abstain | Hallucination & Adversarial Probes | 0 | 5 | — | — | — | — | Correct Abstention |
| Q28 | Abstain | Hallucination & Adversarial Probes | 0 | 5 | — | — | — | — | Correct Abstention |

## 6. Abstention Analysis (Q23–Q28)

| ID | Query | Expected | Actual Abstained | Citations | Assessment |
| :--- | :--- | :---: | :---: | :---: | :--- |
| Q23 | What is the tensile strength of carbon nanotubes in aerospace fuselages? | True | True | 0 | PASS: Refused ungrounded claim |
| Q24 | How does cryogenic tempering affect titanium alloy cutting inserts in CNC milling? | True | True | 0 | PASS: Refused ungrounded claim |
| Q25 | What are the polymerization curing kinetics of UV-activated dental composite resins? | True | True | 0 | PASS: Refused ungrounded claim |
| Q26 | Which synthetic polymer epoxy adhesive was documented in 12th-century Chola bronze casting? | True | True | 0 | PASS: Refused ungrounded claim |
| Q27 | How does neodymium magnet levitation stabilize hand chisels in Saharanpur wood carving? | True | True | 0 | PASS: Refused ungrounded claim |
| Q28 | Why do Udayagiri artisans boil cutlery in concentrated hydrochloric acid before food application? | True | True | 0 | PASS: Refused ungrounded claim |

## 7. Limitations & Engineering Recommendations

1. **Lexical vs Dense Embeddings**: Current baseline relies on BM25 combined with deterministic Murmur-hashing dense embeddings. Queries with significant vocabulary mismatch (e.g. specialized terminology vs craft card descriptions) retrieve fewer relevant passages than semantic dense embeddings (e.g., `intfloat/multilingual-e5-small`).
2. **Multi-Source Passage Fusion**: For complex queries spanning $\ge 3$ distinct scientific documents, top-5 retrieval with `max_per_source=2` limits representation to at most 2 sources. Expanding `top_k=8` or implementing iterative sub-query expansion would increase multi-source recall.
3. **Cross-Encoder Reranking**: Incorporating a lightweight cross-encoder reranker over top-40 candidate passages will significantly boost MRR and nDCG by re-ordering true relevant chunks to ranks 1–2.
4. **Full-Sentence Citation Provenance**: Structured responses now capture exact chunk IDs and evidence levels. Next-generation generation prompts can incorporate character-level offset span citations.
