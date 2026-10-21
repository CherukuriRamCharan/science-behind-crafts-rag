"""Apply verified gold label fixes and log changes."""
import json
from pathlib import Path

def apply_fixes():
    with open("data/eval/eval_questions.jsonl", "r", encoding="utf-8") as f:
        questions = [json.loads(line) for line in f if line.strip()]

    changes_log = []

    for q in questions:
        qid = q["qid"]

        # Fix q002 question_type back to process
        if qid == "q002":
            old_type = q["question_type"]
            q["question_type"] = "process"
            changes_log.append({
                "qid": "q002",
                "field": "question_type",
                "old": old_type,
                "new": "process",
                "reason": "Restored original target distribution (process: 8, scientific_explanation: 7) as tool socket impact mechanics describes craft tool maintenance/use process."
            })

        # Fix q022 gold chunk
        if qid == "q022":
            old_ids = list(q["gold_record_ids"])
            old_quote = q["gold_evidence_quote"]
            new_cid = "WBC-0051::doc0005::0000"
            new_rid = "WBC-0051::doc0005"
            new_prid = "WBC-0051-0003"
            new_quote = "data on hand grip strength are also meaningful for predicting muscle weakness [7], determining the functional capabilities of"
            q["gold_record_ids"] = [new_cid, new_rid, new_prid]
            q["gold_record_groups"] = [[new_cid, new_rid, new_prid]]
            q["gold_evidence_quote"] = new_quote
            changes_log.append({
                "qid": "q022",
                "field": "gold_record_ids & gold_evidence_quote",
                "old": f"IDs: {old_ids}, Quote: {repr(old_quote)}",
                "new": f"IDs: {[new_cid, new_rid, new_prid]}, Quote: {repr(new_quote)}",
                "reason": "Corrected mislabeled gold chunk: doc0005 is the actual text explaining that grip strength predicts muscle weakness and functional capabilities (doc0006 was hand anatomy)."
            })

        # Fix q023 gold chunk
        if qid == "q023":
            old_ids = list(q["gold_record_ids"])
            old_quote = q["gold_evidence_quote"]
            new_cid = "WBC-0051::doc0006::0000"
            new_rid = "WBC-0051::doc0006"
            new_prid = "WBC-0051-0004"
            new_quote = "Hand is made up of bones, tendons, ligaments, nerves and"
            q["gold_record_ids"] = [new_cid, new_rid, new_prid]
            q["gold_record_groups"] = [[new_cid, new_rid, new_prid]]
            q["gold_evidence_quote"] = new_quote
            changes_log.append({
                "qid": "q023",
                "field": "gold_record_ids & gold_evidence_quote",
                "old": f"IDs: {old_ids}, Quote: {repr(old_quote)}",
                "new": f"IDs: {[new_cid, new_rid, new_prid]}, Quote: {repr(new_quote)}",
                "reason": "Corrected mislabeled gold chunk: doc0006 contains the actual anatomy text ('Hand is made up of bones, tendons, ligaments, nerves and') whereas doc0007 described handle dimensions."
            })

        # Align q031 question to card text
        if qid == "q031":
            old_q = q["question"]
            new_q = "What quantified mechanical data and node-weakness explanation does the bamboo ScienceDirect review summarize?"
            new_ans = "The ScienceDirect review summarizes quantified MOE, MOR, and tensile data along with a node-weakness explanation to support engineering sustainability claims."
            q["question"] = new_q
            q["gold_answer"] = new_ans
            changes_log.append({
                "qid": "q031",
                "field": "question & gold_answer",
                "old": old_q,
                "new": new_q,
                "reason": "Aligned question to the actual text of card WBC-0003::doc0003::0000, which states that a ScienceDirect review summarizes quantified data and node-weakness, but does not contain the biological explanation itself."
            })

    # Save updated jsonl
    with open("data/eval/eval_questions.jsonl", "w", encoding="utf-8") as f:
        for q in questions:
            f.write(json.dumps(q, ensure_ascii=False) + "\n")

    # Save log of changes
    log_path = Path("data/eval/results/gold_label_fixes.json")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as f:
        json.dump(changes_log, f, indent=2)

    print(f"Applied {len(changes_log)} fixes. Logged to {log_path}")

if __name__ == "__main__":
    apply_fixes()
