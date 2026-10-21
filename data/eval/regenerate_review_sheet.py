"""Regenerate review_sheet.csv from eval_questions.jsonl."""
import csv
import json

def regenerate():
    with open("data/eval/eval_questions.jsonl", "r", encoding="utf-8") as f:
        questions = [json.loads(line) for line in f if line.strip()]

    rows = []
    for q in questions:
        rows.append({
            "qid": q["qid"],
            "question": q["question"],
            "question_with_id": q.get("question_with_id", ""),
            "question_type": q["question_type"],
            "document_type": q["document_type"],
            "craft": q["craft"],
            "expected_evidence_level": q["expected_evidence_level"],
            "gold_record_ids": "; ".join(q["gold_record_ids"]),
            "gold_record_groups": json.dumps(q.get("gold_record_groups", [])),
            "gold_evidence_quote": q["gold_evidence_quote"],
            "gold_answer": q["gold_answer"],
            "origin": q["origin"],
            "needs_human_review": "TRUE"
        })

    with open("data/eval/review_sheet.csv", "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "qid", "question", "question_with_id", "question_type", "document_type", "craft",
            "expected_evidence_level", "gold_record_ids", "gold_record_groups", "gold_evidence_quote",
            "gold_answer", "origin", "needs_human_review"
        ])
        writer.writeheader()
        writer.writerows(rows)
    print(f"Regenerated review_sheet.csv with {len(rows)} rows.")

if __name__ == "__main__":
    regenerate()
