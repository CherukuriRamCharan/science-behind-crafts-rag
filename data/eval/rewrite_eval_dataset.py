"""Script to rewrite eval questions, compute overlap, add gold_record_groups, and update dataset."""
import csv
import json
import re
from pathlib import Path

STOPWORDS = frozenset(
    "a an the and or in on at to for of with by from is are was were be been "
    "that this which what how why does do did can could would should it its "
    "as into through during under between across".split()
)

def get_content_tokens(text: str) -> set[str]:
    words = re.findall(r"[a-z0-9]+", text.lower())
    return {w for w in words if w not in STOPWORDS and len(w) > 2}

def compute_overlap(question: str, quote: str) -> float:
    t_q = get_content_tokens(question)
    t_g = get_content_tokens(quote)
    if not t_q:
        return 0.0
    return len(t_q & t_g) / len(t_q)

def main():
    with open("data/eval/eval_questions.jsonl", "r", encoding="utf-8") as f:
        old_questions = [json.loads(line) for line in f if line.strip()]

    with open("data/eval/searchable_chunks_reference.json", "r", encoding="utf-8") as f:
        chunks = json.load(f)
    chunk_map = {c["chunk_id"]: c for c in chunks}

    # Measure before overlap
    before_overlaps = []
    for q in old_questions:
        if q["question_type"] != "unanswerable":
            ov = compute_overlap(q["question"], q["gold_evidence_quote"])
            before_overlaps.append(ov)

    avg_before = sum(before_overlaps) / len(before_overlaps) if before_overlaps else 0.0
    print(f"Average Question-to-Quote Token Overlap BEFORE: {avg_before * 100:.2f}%")

    # Rewritten questions dictionary mapping qid -> new natural question
    # Strictly NO WBC-xxxx IDs or "according to..."
    # Avoiding long verbatim phrase copies from quote
    rewritten_specs = {
        # --- 18 source_document questions ---
        "q001": {
            "question": "What difference in worker particle exposure exists between manual abrasive smoothing and powered workshop equipment?",
            "craft": "Wood",
            "ev": "E3",
            "type": "factual"
        },
        "q002": {
            "question": "Why do wooden tool handles frequently split at the socket under impact, and how do metal collars prevent this failure?",
            "craft": "Wood",
            "ev": "E3",
            "type": "scientific_explanation"
        },
        "q003": {
            "question": "Why are narrow wooden bridges between pierced cutout holes especially vulnerable to cracking?",
            "craft": "Wood",
            "ev": "E3",
            "type": "scientific_explanation"
        },
        "q004": {
            "question": "What cutting defects occur when the blade path runs against the natural grain angle of timber?",
            "craft": "Wood",
            "ev": "E3",
            "type": "scientific_explanation"
        },
        "q005": {
            "question": "Why does grain raising after water contact happen more severely on low-density softwoods than on dense hardwoods?",
            "craft": "Wood",
            "ev": "E3",
            "type": "scientific_explanation"
        },
        "q006": {
            "question": "What health and safety standards govern chemical adhesives used on wooden kitchenware in direct contact with food?",
            "craft": "Wood",
            "ev": "E2",
            "type": "factual"
        },
        "q007": {
            "question": "What occupational health hazards are linked to fine dust emitted from synthetic resin adhesives during timber fabrication?",
            "craft": "Wood",
            "ev": "E3",
            "type": "factual"
        },
        "q008": {
            "question": "Which handle geometric dimensions most strongly dictate hand grip endurance and forearm fatigue for manual carving tools?",
            "craft": "Bamboo (Battal chisel handles used in root carving)",
            "ev": "E3",
            "type": "factual"
        },
        "q009": {
            "question": "Which body regions experience the highest rates of musculoskeletal pain among craft artisans working in floor-sitting postures?",
            "craft": "Wood",
            "ev": "E3",
            "type": "factual"
        },
        "q010": {
            "question": "How does repeated thermal heating affect the chemical stability and rancidity of natural coconut finishing oil?",
            "craft": "Wood",
            "ev": "E3",
            "type": "process"
        },
        "q011": {
            "question": "What physical attributes of a wooden log and cutting wedge determine the mechanical effort needed to split timber?",
            "craft": "Wood",
            "ev": "E3",
            "type": "process"
        },
        "q012": {
            "question": "What factors determine cutting resistance and blade sharpness retention during timber shaping operations?",
            "craft": "Wood",
            "ev": "E3",
            "type": "process"
        },
        "q013": {
            "question": "What occupational noise exposure levels do traditional woodcarvers experience when operating manual impact carving chisels?",
            "craft": "Wood",
            "ev": "E3",
            "type": "factual"
        },
        "q014": {
            "question": "What physiological lung impairments and immune reactions occur in artisans regularly inhaling hardwood dust?",
            "craft": "Wood",
            "ev": "E3",
            "type": "scientific_explanation"
        },
        "q015": {
            "question": "Do daily or seasonal weather variations exert a larger influence on timber warping and dimensional checking?",
            "craft": "Wood",
            "ev": "E3",
            "type": "process"
        },
        "q016": {
            "question": "Why do expert artisans exhibit slight physical variations across repetitive hand-carving strokes rather than absolute mechanical consistency?",
            "craft": "Wood",
            "ev": "E3",
            "type": "factual"
        },
        "q017": {
            "question": "What acoustic sound levels and frequency ranges are produced by small rotary electric power tools in artisan carving workshops?",
            "craft": "Wood",
            "ev": "E3",
            "type": "factual"
        },
        "q018": {
            "question": "How does altering the grain orientation between adjacent veneer plies impact structural stiffness and warp resistance in laminated woodwork?",
            "craft": "Wood",
            "ev": "E3",
            "type": "process"
        },

        # --- 10 full_paper_chunk questions ---
        "q019": {
            "question": "How is muscular strength quantified and defined from an ergonomic biomechanics standpoint?",
            "craft": "Bamboo (Battal chisel handles used in root carving)",
            "ev": "E3",
            "type": "factual"
        },
        "q020": {
            "question": "Which diagnostic tools are standardly employed in ergonomic laboratories to evaluate human hand grip capacity?",
            "craft": "Bamboo (Battal chisel handles used in root carving)",
            "ev": "E3",
            "type": "factual"
        },
        "q021": {
            "question": "In industrial ergonomics, what three workplace design objectives rely on physical grip strength measurements?",
            "craft": "Bamboo (Battal chisel handles used in root carving)",
            "ev": "E3",
            "type": "factual"
        },
        "q022": {
            "question": "What clinical and functional health assessments utilize hand grip strength data beyond product design?",
            "craft": "Bamboo (Battal chisel handles used in root carving)",
            "ev": "E3",
            "type": "scientific_explanation"
        },
        "q023": {
            "question": "What anatomical tissues and skeletal elements form the functional structure of the human hand?",
            "craft": "Bamboo (Battal chisel handles used in root carving)",
            "ev": "E3",
            "type": "factual"
        },
        "q024": {
            "question": "How can trajectory distributions learned from craft demonstrations be encoded for automated robotic wood carving?",
            "craft": "Robotic wood carving and digital fabrication; indirect relevance to traditional craft processes",
            "ev": "E3",
            "type": "process"
        },
        "q025": {
            "question": "Why do conventional computer-aided manufacturing toolpaths struggle to replicate artisanal wood carving nuances?",
            "craft": "Robotic wood carving and digital fabrication; indirect relevance to traditional craft processes",
            "ev": "E3",
            "type": "scientific_explanation"
        },
        "q026": {
            "question": "What operational challenges necessitate iterative toolpath corrections during automated robotic wood sculpting?",
            "craft": "Robotic wood carving and digital fabrication; indirect relevance to traditional craft processes",
            "ev": "E3",
            "type": "process"
        },
        "q027": {
            "question": "How does tailoring chisel handle circumference to human muscular capacity reduce cumulative trauma disorders in woodcarvers?",
            "craft": "Bamboo (Battal chisel handles used in root carving)",
            "ev": "E3",
            "type": "multi_hop"
        },
        "q028": {
            "question": "How does movement demonstration by skilled craftspeople enable robotic arms to replicate manual hand-chisel techniques?",
            "craft": "Robotic wood carving and digital fabrication; indirect relevance to traditional craft processes",
            "ev": "E3",
            "type": "multi_hop"
        },

        # --- 5 process_observation questions ---
        "q029": {
            "question": "Which comprehensive forestry reference provides standardized mechanical property tables for evaluating timber species durability?",
            "craft": "Wood",
            "ev": "E3",
            "type": "factual"
        },
        "q030": {
            "question": "What engineering mechanical properties have researchers synthesized to validate the structural viability of structural bamboo?",
            "craft": "Bamboo",
            "ev": "E3",
            "type": "factual"
        },
        "q031": {
            "question": "Why do the nodal rings along a bamboo culm exhibit lower tensile strength than the smooth internode sections?",
            "craft": "Bamboo",
            "ev": "E3",
            "type": "scientific_explanation"
        },
        "q032": {
            "question": "How does internal moisture content influence the compressive and shear strength of harvested bamboo poles?",
            "craft": "Bamboo",
            "ev": "E4",
            "type": "multi_hop"
        },
        "q033": {
            "question": "How do fiber density and cellular composition vary between different bamboo species selected for craft production?",
            "craft": "Bamboo",
            "ev": "E3",
            "type": "multi_hop"
        },

        # --- 12 unanswerable questions (5 HARD NEAR-DOMAIN + 1 UNSOURCED + 6 BASELINE) ---
        "q034": {
            "question": "Why is Udayagiri wood soaked in water for four days specifically before initial shaping?",
            "craft": "Wood",
            "ev": "None",
            "type": "unanswerable",
            "notes": "Flagged as 'Not yet sourced' in raw_fields.Linked External Explanation (WBC-0072-EX)."
        },
        "q035": {
            "question": "What steam bending minimum curvature radius can be achieved in seasoned teak without outer fiber tensile failure?",
            "craft": "Wood",
            "ev": "None",
            "type": "unanswerable",
            "notes": "Hard near-domain unanswerable (steam bending mechanics of teak absent from corpus)."
        },
        "q036": {
            "question": "Why does pressure-impregnating green bamboo culms with aqueous borax and boric acid prevent powder-post beetle (Dinoderus minutus) attacks?",
            "craft": "Bamboo",
            "ev": "None",
            "type": "unanswerable",
            "notes": "Hard near-domain unanswerable (borax preservative chemistry for Dinoderus absent from corpus)."
        },
        "q037": {
            "question": "What rotational speed in surface feet per minute prevents friction burnishing when turning dense ebony on a manual treadle lathe?",
            "craft": "Wood",
            "ev": "None",
            "type": "unanswerable",
            "notes": "Hard near-domain unanswerable (treadle lathe cutting speeds absent from corpus)."
        },
        "q038": {
            "question": "How does soaking seasoned walnut in walnut husk tannin broth protect carved ceiling panels against solar ultraviolet photodegradation?",
            "craft": "Wood",
            "ev": "None",
            "type": "unanswerable",
            "notes": "Hard near-domain unanswerable (walnut husk UV protection mechanism absent from corpus)."
        },
        "q039": {
            "question": "What copper sulfate bath concentration prevents sapstain and blue-stain fungal colonization in freshly harvested bamboo culms?",
            "craft": "Bamboo",
            "ev": "None",
            "type": "unanswerable",
            "notes": "Hard near-domain unanswerable (copper sulfate blue-stain treatment absent from corpus)."
        },
        "q040": {
            "question": "What annealing lehr temperature ramp prevents stress-induced shattering in hot-worked borosilicate glass flasks?",
            "craft": "None",
            "ev": "None",
            "type": "unanswerable",
            "notes": "Easy baseline out-of-domain (glassblowing thermodynamics)."
        },
        "q041": {
            "question": "What pulse repetition rate and assist gas pressure are required to laser cut optical grade acrylic without edge whitening?",
            "craft": "None",
            "ev": "None",
            "type": "unanswerable",
            "notes": "Easy baseline out-of-domain (polymer laser cutting)."
        },
        "q042": {
            "question": "What basic chromium sulfate concentration is mandated for heavy bovine sole leather tanning in European standard EN 15987?",
            "craft": "None",
            "ev": "None",
            "type": "unanswerable",
            "notes": "Easy baseline out-of-domain (leather tanning chemistry)."
        },
        "q043": {
            "question": "What tin percentage in 78-22 bell bronze optimizes the acoustic strike note and hum tone decay in cathedral bells?",
            "craft": "None",
            "ev": "None",
            "type": "unanswerable",
            "notes": "Easy baseline out-of-domain (metallurgical acoustics)."
        },
        "q044": {
            "question": "What alum mordant bath concentration maximizes lightfastness of alizarin dye on degummed Bombyx mori silk yarn?",
            "craft": "None",
            "ev": "None",
            "type": "unanswerable",
            "notes": "Easy baseline out-of-domain (textile silk dyeing)."
        },
        "q045": {
            "question": "What molar ratio of silica to alumina eliminates crazing in cone 10 reduction-fired porcelain glazes?",
            "craft": "None",
            "ev": "None",
            "type": "unanswerable",
            "notes": "Easy baseline out-of-domain (ceramic glaze mineral chemistry)."
        }
    }

    # Rebuild updated questions list
    updated_questions = []
    review_rows = []
    after_overlaps = []
    high_overlap_qids = []

    for old_q in old_questions:
        qid = old_q["qid"]
        spec = rewritten_specs[qid]

        old_question_text = old_q["question"]
        new_question_text = spec["question"]
        gold_ids = old_q["gold_record_ids"]
        gold_quote = old_q["gold_evidence_quote"]
        gold_answer = old_q["gold_answer"]
        q_type = spec["type"]
        is_ans = (q_type != "unanswerable")

        # Create gold_record_groups
        # Aliases of the single relevant record are grouped together
        if is_ans and gold_ids:
            # All aliases for this chunk/record form one group
            gold_record_groups = [gold_ids]
        else:
            gold_record_groups = []

        # Overlap computation
        if is_ans:
            ov = compute_overlap(new_question_text, gold_quote)
            after_overlaps.append(ov)
            if ov > 0.50:
                high_overlap_qids.append((qid, ov, new_question_text, gold_quote))

        rec = {
            "qid": qid,
            "question": new_question_text,
            "question_with_id": old_question_text,
            "gold_record_ids": gold_ids,
            "gold_record_groups": gold_record_groups,
            "gold_answer": gold_answer,
            "gold_evidence_quote": gold_quote,
            "question_type": q_type,
            "expected_evidence_level": spec["ev"],
            "document_type": old_q["document_type"],
            "craft": spec["craft"],
            "paraphrase_group": None,
            "origin": old_q.get("origin", "auto_drafted"),
            "needs_human_review": True
        }
        updated_questions.append(rec)

        review_rows.append({
            "qid": qid,
            "question": new_question_text,
            "question_with_id": old_question_text,
            "question_type": q_type,
            "document_type": old_q["document_type"],
            "craft": spec["craft"],
            "expected_evidence_level": spec["ev"],
            "gold_record_ids": "; ".join(gold_ids),
            "gold_record_groups": json.dumps(gold_record_groups),
            "gold_evidence_quote": gold_quote,
            "gold_answer": gold_answer,
            "origin": rec["origin"],
            "needs_human_review": "TRUE"
        })

    avg_after = sum(after_overlaps) / len(after_overlaps) if after_overlaps else 0.0
    print(f"Average Question-to-Quote Token Overlap AFTER:  {avg_after * 100:.2f}%")
    print(f"Overlap reduction: {(avg_before - avg_after) * 100:.2f}% percentage points")
    print(f"Questions still > 50% overlap: {len(high_overlap_qids)}")
    for qid, ov, qtxt, quote in high_overlap_qids:
        print(f"  {qid} ({ov*100:.1f}%): '{qtxt}'")

    # Save to data/eval/eval_questions.jsonl
    eval_path = Path("data/eval/eval_questions.jsonl")
    with eval_path.open("w", encoding="utf-8") as f:
        for r in updated_questions:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"Saved updated dataset to {eval_path}")

    # Save to data/eval/review_sheet.csv
    csv_path = Path("data/eval/review_sheet.csv")
    with csv_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "qid", "question", "question_with_id", "question_type", "document_type", "craft",
            "expected_evidence_level", "gold_record_ids", "gold_record_groups", "gold_evidence_quote",
            "gold_answer", "origin", "needs_human_review"
        ])
        writer.writeheader()
        writer.writerows(review_rows)
    print(f"Saved updated review sheet to {csv_path}")

if __name__ == "__main__":
    main()
