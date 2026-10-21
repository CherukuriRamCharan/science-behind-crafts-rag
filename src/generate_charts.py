"""Generate evaluation charts and figures using matplotlib.

Figures produced in data/eval/results/figures/:
1. recall_at_k.png: Recall@k across retrievers (BM25, Dense hashing baseline, Hybrid no-dedup, Hybrid pipeline).
2. mrr_comparison.png: MRR comparison across retrievers overall and sliced by document type.
3. metrics_by_question_type.png: HitRate@5 and Recall@5 by question type.
4. error_category_breakdown.png: Breakdown of failures across pipeline stages.
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

# Ensure repository root is on sys.path so modules can be imported without PYTHONPATH
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import matplotlib.pyplot as plt
import numpy as np


def generate_all_charts(results_dir: str = "data/eval/results") -> list[str]:
    out_dir = Path(results_dir) / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    generated = []

    summary_file = Path(results_dir) / "retrieval_summary.csv"
    if not summary_file.exists():
        print(f"Warning: {summary_file} not found. Skipping retrieval charts.")
        return generated

    with summary_file.open("r", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))

    # 1. Recall@k vs k across retrievers (Overall)
    overall_rows = {r["retriever"]: r for r in rows if r["slice"] == "Overall"}
    if overall_rows:
        ks = [1, 3, 5, 10]
        plt.figure(figsize=(7.5, 4.8), dpi=300)

        styles = {
            "bm25": {"color": "#1f77b4", "marker": "o", "label": "BM25 (Sparse)"},
            "dense (hashing baseline)": {"color": "#ff7f0e", "marker": "s", "label": "Dense (Hashing baseline)"},
            "hybrid (no dedup)": {"color": "#2ca02c", "marker": "^", "label": "Hybrid (No Dedup)"},
            "hybrid (pipeline with dedup)": {"color": "#d62728", "marker": "d", "label": "Hybrid (Pipeline with Dedup)"},
        }

        for ret, style in styles.items():
            if ret in overall_rows:
                y = [float(overall_rows[ret][f"Recall@{k}"]) for k in ks]
                plt.plot(ks, y, label=style["label"], color=style["color"], marker=style["marker"], linewidth=2, markersize=7)

        plt.title("Recall@k Comparison on Unified Corpus (N=33)", fontsize=12, fontweight="bold", pad=12)
        plt.xlabel("k (Distinct Records Retrieved)", fontsize=10)
        plt.ylabel("Recall@k", fontsize=10)
        plt.xticks(ks)
        plt.ylim(0.5, 1.05)
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.legend(frameon=True, facecolor="white", edgecolor="none", fontsize=9)
        plt.tight_layout()

        p1 = out_dir / "recall_at_k.png"
        plt.savefig(p1)
        plt.close()
        generated.append(str(p1))

    # 2. MRR by Document Type
    doc_slices = ["Overall", "document_type:source_document", "document_type:full_paper_chunk", "document_type:process_observation"]
    slice_labels = ["Overall\n(N=33)", "Source Doc Card\n(N=18)", "Full Paper Chunk\n(N=10)", "Process Obs\n(N=5)"]

    retrievers_to_plot = ["bm25", "dense (hashing baseline)", "hybrid (no dedup)", "hybrid (pipeline with dedup)"]
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
    display_names = ["BM25", "Dense (Hashing)", "Hybrid (No Dedup)", "Hybrid (Pipeline)"]

    mrr_data = {ret: [] for ret in retrievers_to_plot}
    for s in doc_slices:
        for ret in retrievers_to_plot:
            matching = [r for r in rows if r["slice"] == s and r["retriever"] == ret]
            mrr_val = float(matching[0]["MRR"]) if matching else 0.0
            mrr_data[ret].append(mrr_val)

    plt.figure(figsize=(9.5, 5.0), dpi=300)
    x = np.arange(len(slice_labels))
    width = 0.20

    for i, (ret, col, disp) in enumerate(zip(retrievers_to_plot, colors, display_names)):
        offset = (i - 1.5) * width
        plt.bar(x + offset, mrr_data[ret], width, label=disp, color=col)

    plt.title("Mean Reciprocal Rank (MRR) by Document Type", fontsize=12, fontweight="bold", pad=12)
    plt.xlabel("Corpus Document Type Slice", fontsize=10)
    plt.ylabel("MRR", fontsize=10)
    plt.xticks(x, slice_labels)
    plt.ylim(0, 1.15)
    plt.grid(axis="y", linestyle="--", alpha=0.5)
    plt.legend(frameon=True, facecolor="white", fontsize=8.5)
    plt.tight_layout()

    p2 = out_dir / "mrr_comparison.png"
    plt.savefig(p2)
    plt.close()
    generated.append(str(p2))

    # 3. Metrics by Question Type (HitRate@5)
    qtype_slices = [
        "question_type:factual",
        "question_type:process",
        "question_type:scientific_explanation",
        "question_type:multi_hop",
        "question_type:card_pointer",
    ]
    qtype_labels = ["Factual\n(N=14)", "Process\n(N=8)", "Scientific Expl\n(N=6)", "Multi-hop\n(N=4)", "Card Pointer\n(N=1)"]

    hit5_data = {ret: [] for ret in retrievers_to_plot}
    for s in qtype_slices:
        for ret in retrievers_to_plot:
            matching = [r for r in rows if r["slice"] == s and r["retriever"] == ret]
            hit5_data[ret].append(float(matching[0]["HitRate@5"]) if matching else 0.0)

    plt.figure(figsize=(9.5, 5.0), dpi=300)
    x = np.arange(len(qtype_labels))

    for i, (ret, col, disp) in enumerate(zip(retrievers_to_plot, colors, display_names)):
        offset = (i - 1.5) * width
        plt.bar(x + offset, hit5_data[ret], width, label=disp, color=col)

    plt.title("Hit Rate@5 by Question Type", fontsize=12, fontweight="bold", pad=12)
    plt.xlabel("Question Type", fontsize=10)
    plt.ylabel("Hit Rate@5", fontsize=10)
    plt.xticks(x, qtype_labels)
    plt.ylim(0, 1.2)
    plt.grid(axis="y", linestyle="--", alpha=0.5)
    plt.legend(frameon=True, facecolor="white", fontsize=8.5)
    plt.tight_layout()

    p3 = out_dir / "metrics_by_question_type.png"
    plt.savefig(p3)
    plt.close()
    generated.append(str(p3))

    # 4. Error Category Breakdown
    error_summary_file = Path(results_dir) / "error_summary.csv"
    if error_summary_file.exists():
        with error_summary_file.open("r", encoding="utf-8-sig") as f:
            e_rows = list(csv.DictReader(f))

        cats = [r["error_category"].replace("_", " ").title() for r in e_rows]
        counts = [int(r["count"]) for r in e_rows]

        plt.figure(figsize=(8.0, 4.4), dpi=300)
        err_colors = ["#d62728", "#9467bd", "#8c564b"]
        bars = plt.bar(cats, counts, color=err_colors[:len(cats)], width=0.45)
        for bar in bars:
            height = bar.get_height()
            plt.annotate(f"{height}",
                         xy=(bar.get_x() + bar.get_width() / 2, height),
                         xytext=(0, 3), textcoords="offset points",
                         ha="center", va="bottom", fontsize=10, fontweight="bold")

        plt.title("Failure Breakdown by Category", fontsize=12, fontweight="bold", pad=12)
        plt.xlabel("Failure Mode Category", fontsize=10)
        plt.ylabel("Number of Failures Observed", fontsize=10)
        plt.ylim(0, max(counts) + 3)
        plt.grid(axis="y", linestyle="--", alpha=0.5)
        plt.tight_layout()

        p4 = out_dir / "error_category_breakdown.png"
        plt.savefig(p4)
        plt.close()
        generated.append(str(p4))

    return generated


if __name__ == "__main__":
    paths = generate_all_charts()
    print(f"Generated {len(paths)} charts:")
    for p in paths:
        print(f"  {p}")
