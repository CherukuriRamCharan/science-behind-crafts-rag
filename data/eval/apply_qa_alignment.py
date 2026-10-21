"""Apply all approved question-answer alignment rewrites and evidence status flags."""
import json
from pathlib import Path

def update_dataset():
    with open("data/eval/eval_questions.jsonl", "r", encoding="utf-8") as f:
        questions = [json.loads(line) for line in f if line.strip()]

    # Explicit question, quote, and answer updates ensuring 100% grounding ONLY in quote
    updates = {
        "q001": {
            "question": "How does dust exposure from powered handheld sanding relate to manual hand-sanding in workshop finishing?",
            "gold_answer": "Powered handheld sanding serves as an indirect proxy for manual hand-sanding, where dust generation is expected to be lower but has not been separately measured.",
            "expected_evidence_status": "verified science"
        },
        "q002": {
            "question": "Why do wooden tool handles tend to split at the socket under impact, and how do reinforcing collars or flanges help?",
            "gold_answer": "Wooden tool handles tend to split radially at the handle/socket junction under impact, and reinforcing collars or flanges at the socket reduce this failure.",
            "expected_evidence_status": "verified science"
        },
        "q003": {
            "question": "Why are narrow bridges between pierced openings in wood susceptible to stress concentration?",
            "gold_answer": "Wood's natural anisotropy affects stress concentration around circular and elliptic holes, providing a mechanical basis for stress buildup in narrow bridges between pierced openings.",
            "expected_evidence_status": "verified science"
        },
        "q004": {
            "question": "How does the angle of the wood grain relative to a cutting tool path explain torn-grain defects?",
            "gold_answer": "Grain angle relative to the tool path governs strain and cleavage ahead of the cutting edge, mechanically explaining torn-grain and fuzzy-grain defects.",
            "expected_evidence_status": "verified science"
        },
        "q005": {
            "question": "What causes grain raising after wood sanding, and how does wood density affect it?",
            "gold_answer": "Sanding tears cell walls creating loosely bonded fiber slivers that project after wetting, and grain raising is inversely related to wood density.",
            "expected_evidence_status": "verified science"
        },
        "q006": {
            "question": "What conditions govern the permissible use of adhesives and coatings on food-contact articles?",
            "gold_answer": "Specific legal and regulatory conditions establish when and how adhesives and coatings may be used on food-contact articles.",
            "expected_evidence_status": "verified science"
        },
        "q007": {
            "question": "What evidence exists regarding health risks from adhesive chemistry such as formaldehyde-based resins?",
            "gold_answer": "Experimental toxicity evidence confirms that adhesive chemistry like formaldehyde-based resins carries health risks via dust and emissions.",
            "expected_evidence_status": "verified science"
        },
        "q008": {
            "question": "Which handle design features most strongly affect grip strength and muscle fatigue in hand tools?",
            "gold_answer": "Handle diameter and handle material are the most significant hand-tool design factors affecting grip strength and muscle fatigue.",
            "expected_evidence_status": "verified science"
        },
        "q009": {
            "question": "What musculoskeletal disorder prevalence rates were reported for floor-sitting handicraft workers?",
            "gold_answer": "High MSD prevalence was confirmed among floor-sitting precision handicraft workers, specifically 61.6% for neck, 74.8% for lower back, and 54.7% for knee.",
            "expected_evidence_status": "verified science"
        },
        "q010": {
            "question": "How do prolonged heat and time exposure affect the chemical properties of virgin coconut oil?",
            "gold_answer": "Prolonged heat and time exposure significantly increase free fatty acid content and peroxide value in virgin coconut oil.",
            "expected_evidence_status": "verified science"
        },
        "q011": {
            "question": "Which factors affect the force required in wood splitting?",
            "gold_answer": "Wood type, log diameter, and splitting-wedge design affect the force required to split wood.",
            "expected_evidence_status": "verified science"
        },
        "q012": {
            "question": "What relationships govern cutting forces during wood machining?",
            "gold_answer": "Wood-cutting forces depend on the relationship between wood properties, cutting geometry, and tool characteristics.",
            "expected_evidence_status": "verified science"
        },
        "q013": {
            "question": "What average 8-hour noise exposure was measured among wood carvers using hand tools?",
            "gold_answer": "A mean 8-hour noise exposure (Lex,8h) of 93.17 dB(A), exceeding safe limits, was measured among wood carvers using hand carving tools.",
            "expected_evidence_status": "verified science"
        },
        "q014": {
            "question": "What specific lung function and immune changes were measured in workers exposed to mixed tropical hardwood dust?",
            "gold_answer": "Workers showed significant reductions in lung function (VC, FEV1, PEFR) and elevated IgE and IgG levels.",
            "expected_evidence_status": "verified science"
        },
        "q015": {
            "question": "Are daily or seasonal humidity swings the primary driver of wood dimensional change and cracking?",
            "gold_answer": "Seasonal (not daily) humidity swings are the main driver of dimensional change and cracking, as wood continuously exchanges moisture with its environment.",
            "expected_evidence_status": "verified science"
        },
        "q016": {
            "question": "Is some variation between repeated precision hand movements a normal human trait?",
            "gold_answer": "Yes, motor variability is an inherent, measurable feature of human repetitive manual precision tasks.",
            "expected_evidence_status": "verified science"
        },
        "q017": {
            "question": "What personal noise levels were measured from power tools including drill machines?",
            "gold_answer": "Personal noise exposure was measured around 80 dBA, with frequency peaks near 1 to 4 kHz from powered tools including drill machines.",
            "expected_evidence_status": "verified science"
        },
        "q018": {
            "question": "How does the angle at which veneer layers are stacked affect strain and dimensional stability in plywood under load?",
            "gold_answer": "Veneer layup angle and layer arrangement directly govern strain distribution and dimensional stability under load.",
            "expected_evidence_status": "verified science"
        },
        "q019": {
            "question": "How is strength defined in terms of body parts and muscle contraction?",
            "gold_answer": "Strength is defined as the capacity of body parts to produce force or torque with voluntary muscle contraction.",
            "expected_evidence_status": "verified science"
        },
        "q020": {
            "question": "Which instruments are most commonly used to measure hand grip strength?",
            "gold_answer": "Hand dynamometers and force gauges are the most common devices used to measure hand grip strength.",
            "expected_evidence_status": "verified science"
        },
        "q021": {
            "question": "What three purposes in industrial ergonomics make physical strength information useful?",
            "gold_answer": "Physical strength information is useful for 1) hand tool design, 2) worker selection and placement programs, and 3) job design to match worker strength with job demands.",
            "expected_evidence_status": "verified science"
        },
        "q022": {
            "question": "Why are data on hand grip strength meaningful beyond ergonomic tool design?",
            "gold_answer": "Data on hand grip strength are meaningful for predicting muscle weakness and determining functional capabilities.",
            "expected_evidence_status": "verified science"
        },
        "q023": {
            "question": "Which anatomical structures make up the human hand?",
            "gold_answer": "The hand is made up of bones, tendons, ligaments, nerves, and associated tissues.",
            "expected_evidence_status": "verified science"
        },
        "q024": {
            "question": "What framework was integrated into the case study on robot wood carving with hand tools?",
            "gold_answer": "Probabilistic movement primitives (ProMPs) were integrated into the case study on robot wood carving with hand tools.",
            "expected_evidence_status": "verified science"
        },
        "q025": {
            "question": "What limitation exists regarding conventional CAM tools in non-standard manufacturing?",
            "gold_answer": "Conventional CAM tools are tailored strictly for standard manufacturing processes rather than non-standard craft tasks.",
            "expected_evidence_status": "verified science"
        },
        "q026": {
            "question": "What limitation was discussed regarding the robotic wood carving pipeline across multiple applications?",
            "gold_answer": "A noted limitation is the need for iterative toolpath adjustments to address inaccuracies during wood carving.",
            "expected_evidence_status": "verified science"
        },
        "q027": {
            "question": "What physiological muscles generate grip strength when holding an object?",
            "gold_answer": "Grip strength is generated by the hand and forearm muscles while gripping an object.",
            "expected_evidence_status": "verified science"
        },
        "q028": {
            "question": "How were human demonstrations of wood carving cuts recorded for the robot case study?",
            "gold_answer": "Human demonstrations of different wood carving cuts were recorded via kinesthetic guidance/teaching.",
            "quote_override": "Human demonstrations of different wood carving cuts are recorded via kinesthetic",
            "expected_evidence_status": "verified science"
        },
        "q029": {
            "question": "What reference handbook is noted for covering the mechanical and physical properties of wood in product design?",
            "gold_answer": "The authoritative USDA Forest Products Laboratory handbook covers mechanical and physical properties of wood for product shape, thickness, and durability.",
            "expected_evidence_status": "artisan observation"
        },
        "q030": {
            "question": "What mechanical property data across bamboo studies are summarized to support performance claims?",
            "gold_answer": "Shear, compressive, tensile, bending, and MOE data across multiple bamboo studies are reviewed to support performance claims.",
            "expected_evidence_status": "artisan observation"
        },
        "q031": {
            "question": "What evidence explains why bamboo node regions are mechanically weaker than internodes?",
            "gold_answer": "A review with quantified MOE, MOR, and tensile data provides an explanation for mechanical weakness at bamboo node regions.",
            "expected_evidence_status": "artisan observation"
        },
        "q032": {
            "question": "What key relationship concerning bamboo mechanical properties is compiled in the literature bibliography?",
            "gold_answer": "The annotated bibliography aggregates multiple mechanical property studies including the moisture-strength relationship in bamboo.",
            "expected_evidence_status": "artisan observation"
        },
        "q033": {
            "question": "What aspects of bamboo fibers are compared across species to assist in material selection?",
            "gold_answer": "Fiber strength and chemical/anatomical composition are compared across bamboo species to assist in species selection.",
            "expected_evidence_status": "artisan observation"
        }
    }

    # Apply updates
    for q in questions:
        qid = q["qid"]
        if qid in updates:
            u = updates[qid]
            q["question"] = u["question"]
            q["gold_answer"] = u["gold_answer"]
            q["expected_evidence_status"] = u["expected_evidence_status"]
            if "quote_override" in u:
                q["gold_evidence_quote"] = u["quote_override"]
        else:
            q["expected_evidence_status"] = "unanswerable"

    # Save updated jsonl
    out_path = Path("data/eval/eval_questions.jsonl")
    with out_path.open("w", encoding="utf-8") as f:
        for q in questions:
            f.write(json.dumps(q, ensure_ascii=False) + "\n")
    print(f"Updated {len(questions)} questions in {out_path}")

if __name__ == "__main__":
    update_dataset()
