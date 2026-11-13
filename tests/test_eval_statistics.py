"""Tests for deterministic evaluation uncertainty calculations."""
from src.eval_statistics import bootstrap_and_paired_tests


def test_bootstrap_intervals_and_paired_test_are_deterministic():
    rows = []
    for qid, bm25, dense in (("q1", .9, .5), ("q2", .8, .6), ("q3", .7, .7)):
        for retriever, score in (("bm25", bm25), ("dense", dense)):
            rows.append({"qid": qid, "retriever": retriever, "MRR": score,
                         "HitRate@5": score, "Recall@5": score, "nDCG@5": score})
    first_ci, first_paired = bootstrap_and_paired_tests(rows, repetitions=500)
    second_ci, second_paired = bootstrap_and_paired_tests(rows, repetitions=500)
    assert first_ci == second_ci
    assert first_paired == second_paired
    assert len(first_ci) == 8
    assert len(first_paired) == 4
    assert first_paired[0]["n_pairs"] == 3
    assert first_paired[0]["difference_ci95_low"] <= first_paired[0]["mean_difference"]
    assert first_paired[0]["mean_difference"] <= first_paired[0]["difference_ci95_high"]
