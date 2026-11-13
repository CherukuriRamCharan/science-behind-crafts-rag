"""Deterministic bootstrap intervals and paired randomisation tests."""
from __future__ import annotations

from collections import defaultdict
from itertools import combinations
from typing import Any

import numpy as np

METRICS = ("MRR", "HitRate@5", "Recall@5", "nDCG@5")


def bootstrap_and_paired_tests(
    rows: list[dict[str, Any]], *, repetitions: int = 10_000, seed: int = 20261113,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Return per-retriever percentile CIs and paired two-sided sign-flip tests."""
    if repetitions < 1:
        raise ValueError("repetitions must be positive")
    by_retriever: dict[str, dict[str, dict[str, float]]] = defaultdict(dict)
    for row in rows:
        by_retriever[row["retriever"]][row["qid"]] = {
            metric: float(row[metric]) for metric in METRICS
        }
    confidence_rows = []
    for retriever in sorted(by_retriever):
        qids = sorted(by_retriever[retriever])
        for metric_index, metric in enumerate(METRICS):
            values = np.asarray([by_retriever[retriever][qid][metric] for qid in qids], dtype=float)
            rng = np.random.default_rng(seed + metric_index)
            draws = values[rng.integers(0, len(values), size=(repetitions, len(values)))].mean(axis=1)
            confidence_rows.append({
                "retriever": retriever, "metric": metric, "n_questions": len(values),
                "mean": float(values.mean()), "ci95_low": float(np.quantile(draws, 0.025)),
                "ci95_high": float(np.quantile(draws, 0.975)),
            })

    paired_rows = []
    for left, right in combinations(sorted(by_retriever), 2):
        shared = sorted(set(by_retriever[left]) & set(by_retriever[right]))
        if not shared:
            continue
        for metric_index, metric in enumerate(METRICS):
            differences = np.asarray([
                by_retriever[left][qid][metric] - by_retriever[right][qid][metric]
                for qid in shared
            ], dtype=float)
            rng = np.random.default_rng(seed + 100 + metric_index)
            boot = differences[rng.integers(0, len(differences), size=(repetitions, len(differences)))].mean(axis=1)
            observed = abs(float(differences.mean()))
            signs = rng.choice(np.asarray([-1.0, 1.0]), size=(repetitions, len(differences)))
            null = np.abs((signs * differences).mean(axis=1))
            p_value = (1 + int(np.count_nonzero(null >= observed))) / (repetitions + 1)
            paired_rows.append({
                "left": left, "right": right, "metric": metric, "n_pairs": len(shared),
                "mean_difference": float(differences.mean()),
                "difference_ci95_low": float(np.quantile(boot, 0.025)),
                "difference_ci95_high": float(np.quantile(boot, 0.975)),
                "sign_flip_p_value": p_value,
            })
    return confidence_rows, paired_rows
