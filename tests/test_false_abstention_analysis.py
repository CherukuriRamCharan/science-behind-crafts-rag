from src.analyze_false_abstentions import split_false_abstentions


def test_split_false_abstentions_counts_gold_hits_and_retrieval_misses():
    questions = [
        {"qid": "q1", "gold_record_ids": ["gold-1"]},
        {"qid": "q2", "gold_record_ids": ["gold-2", "gold-2b"]},
        {"qid": "q3", "gold_record_ids": ["gold-3"]},
    ]
    retrieved_top5 = {
        "q1": ["other", "gold-1"],
        "q2": ["other-a", "other-b"],
        "q3": ["gold-3"],
    }

    result = split_false_abstentions(questions, ["q3", "q1", "q2"], retrieved_top5)

    assert result == {
        "retrieved_but_rejected": {"count": 2, "question_ids": ["q1", "q3"]},
        "retrieval_miss": {"count": 1, "question_ids": ["q2"]},
    }
