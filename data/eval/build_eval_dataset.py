"""Generate precisely validated 45-question evaluation dataset."""
import csv
import json
from pathlib import Path

def main():
    with open("data/eval/searchable_chunks_reference.json", "r", encoding="utf-8") as f:
        chunks = json.load(f)

    # Map chunk_id to chunk
    chunk_map = {c["chunk_id"]: c for c in chunks}

    # Helper to find exact chunk containing substring
    def get_chunk(cid, expected_sub):
        if cid not in chunk_map:
            raise KeyError(f"Chunk {cid} not found in searchable index")
        c = chunk_map[cid]
        if expected_sub not in c["text"]:
            raise ValueError(f"Substring '{expected_sub}' not found in {cid}\nChunk text preview: {c['text'][:200]}")
        return c

    # 45 Questions Specification:
    # 33 Answerable (18 source_document, 10 full_paper_chunk, 5 process_observation)
    # 12 Unanswerable (Abstain)
    specs = [
        # --- 18 source_document questions ---
        {
            "qid": "q001",
            "cid": "WBC-0005::doc0001::0000",
            "question": "What woodworking stage was directly measured for wood dust generation in study WBC-0005?",
            "quote": "This study directly measures exposure from powered handheld sanding, closely matching the machine-finishing stage; it serves as a reasonable but indirect proxy for the subsequent manual hand-sanding stage, where dust generation is expected to be lower but has not been separately measured.",
            "gold_answer": "The study directly measures dust exposure from powered handheld sanding matching the machine-finishing stage, serving as an indirect proxy for manual hand-sanding.",
            "question_type": "factual",
            "craft": "Wood",
            "expected_evidence_level": "E3",
        },
        {
            "qid": "q002",
            "cid": "WBC-0004::doc0001::0000",
            "question": "Why do reinforcing collars or flanges at the tool socket reduce failure under impact according to WBC-0004?",
            "quote": "Explains the mechanics of radial wood splitting at handle/socket junctions under impact and shows why reinforcing collars/flanges at the socket reduce failure",
            "gold_answer": "Reinforcing collars or flanges at the socket reduce radial wood splitting at the handle/socket junction under mechanical impact.",
            "question_type": "scientific_explanation",
            "craft": "Wood",
            "expected_evidence_level": "E3",
        },
        {
            "qid": "q003",
            "cid": "WBC-0008::doc0001::0000",
            "question": "How does wood anisotropy influence stress concentration around cut openings according to WBC-0008?",
            "quote": "Analyzes how wood's natural anisotropy affects stress concentration around circular and elliptic holes, providing a mechanical basis for why narrow bridges between pierced openings",
            "gold_answer": "Wood's natural anisotropy concentrates mechanical stresses around circular and elliptic holes, making narrow bridges between pierced openings susceptible to cracking.",
            "question_type": "scientific_explanation",
            "craft": "Wood",
            "expected_evidence_level": "E3",
        },
        {
            "qid": "q004",
            "cid": "WBC-0009::doc0001::0000",
            "question": "What cutting defect does grain angle relative to tool path explain in WBC-0009?",
            "quote": "Directly measures how grain angle relative to the tool path governs strain and cleavage ahead of the cutting edge, providing a mechanical explanation for the torn-grain and fuzzy-g",
            "gold_answer": "Grain angle relative to tool path governs strain and cleavage ahead of the cutting edge, mechanically explaining torn-grain and fuzzy-grain cutting defects.",
            "question_type": "scientific_explanation",
            "craft": "Wood",
            "expected_evidence_level": "E3",
        },
        {
            "qid": "q005",
            "cid": "WBC-0010::doc0001::0000",
            "question": "How is wood grain raising related to wood density according to WBC-0010?",
            "quote": "Shows that sanding tears cell walls to create loosely bonded fibre slivers that project after wetting, and that grain raising is inversely related to wood density",
            "gold_answer": "Grain raising is inversely related to wood density, caused when sanding tears cell walls to form loose fiber slivers that project upwards upon wetting.",
            "question_type": "scientific_explanation",
            "craft": "Wood",
            "expected_evidence_level": "E3",
        },
        {
            "qid": "q006",
            "cid": "WBC-0011::doc0001::0000",
            "question": "What legal/regulatory conditions are addressed by 21 CFR Part 175 in WBC-0011?",
            "quote": "Establishes the legal/regulatory conditions under which adhesives and coatings may be used on food-contact articles",
            "gold_answer": "It establishes FDA regulatory conditions governing the safe use of adhesives and coating components on food-contact craft articles.",
            "question_type": "factual",
            "craft": "Wood",
            "expected_evidence_level": "E2",
        },
        {
            "qid": "q007",
            "cid": "WBC-0012::doc0001::0000",
            "question": "What health risk from formaldehyde-based resin adhesives is identified in WBC-0012?",
            "quote": "Provides experimental toxicity evidence that adhesive chemistry (e.g., formaldehyde-based resins) carries real health risk via dust/emissions",
            "gold_answer": "Formaldehyde-based resin adhesives carry significant health risks via hazardous airborne dust and chemical emissions during woodworking.",
            "question_type": "factual",
            "craft": "Wood",
            "expected_evidence_level": "E3",
        },
        {
            "qid": "q008",
            "cid": "WBC-0003::doc0001::0000",
            "question": "Which hand-tool design factors most significantly affect grip strength and muscle fatigue in WBC-0003?",
            "quote": "The review synthesizes multiple studies showing handle diameter and material as the most significant hand-tool design factors affecting grip strength and fatigue",
            "gold_answer": "Handle diameter and handle material are the most significant hand-tool design factors affecting operator grip strength and muscle fatigue.",
            "question_type": "factual",
            "craft": "Bamboo (Battal chisel handles used in root carving)",
            "expected_evidence_level": "E3",
        },
        {
            "qid": "q009",
            "cid": "WBC-0007::doc0001::0000",
            "question": "What prevalence of lower back musculoskeletal disorders was confirmed among floor-sitting handicraft workers in WBC-0007?",
            "quote": "Confirms high MSD prevalence (neck 61.6%, lower back 74.8%, knee 54.7%) among floor-sitting Indian precision handicraft workers",
            "gold_answer": "Study WBC-0007 confirmed a 74.8% prevalence of lower back musculoskeletal disorders among floor-sitting Indian precision handicraft workers.",
            "question_type": "factual",
            "craft": "Wood",
            "expected_evidence_level": "E3",
        },
        {
            "qid": "q010",
            "cid": "WBC-0014::doc0001::0000",
            "question": "How do prolonged heating and frying affect virgin coconut oil according to WBC-0014?",
            "quote": "Directly measures that virgin coconut oil's free fatty acid content and peroxide value increase significantly with prolonged heat/time exposure",
            "gold_answer": "Prolonged heat and frying time significantly increase free fatty acid content and peroxide value, causing progressive oxidative degradation of virgin coconut oil.",
            "question_type": "process",
            "craft": "Wood",
            "expected_evidence_level": "E3",
        },
        {
            "qid": "q011",
            "cid": "WBC-0001::doc0001::0000",
            "question": "Which design parameters affect the required wood-splitting force according to study WBC-0001?",
            "quote": "The study experimentally investigates wood-splitting force and shows that wood type, log diameter and splitting-wedge design affect the force required.",
            "gold_answer": "Wood type, log diameter, and splitting-wedge design directly determine the mechanical force required for splitting wood.",
            "question_type": "process",
            "craft": "Wood",
            "expected_evidence_level": "E3",
        },
        {
            "qid": "q012",
            "cid": "WBC-0002::doc0001::0000",
            "question": "What key factors govern wood-cutting forces in industrial machining according to review WBC-0002?",
            "quote": "The review explains the relationship between wood-cutting forces, wood properties, cutting geometry and tool characteristics",
            "gold_answer": "Wood-cutting forces are governed by the interaction between wood mechanical properties, cutting geometry, and tool characteristics.",
            "question_type": "process",
            "craft": "Wood",
            "expected_evidence_level": "E3",
        },
        {
            "qid": "q013",
            "cid": "WBC-0006::doc0001::0000",
            "question": "What mean 8-hour noise exposure was measured among Indian wood carvers using hand carving tools in WBC-0006?",
            "quote": "Directly measures HAV and noise exposure (mean Lex,8h of 93.17 dB(A), exceeding safe limits) among Indian wood carvers using hand carving tools",
            "gold_answer": "A mean 8-hour equivalent continuous noise exposure (Lex,8h) of 93.17 dB(A), which exceeds safe occupational limits, was measured among Indian wood carvers.",
            "question_type": "factual",
            "craft": "Wood",
            "expected_evidence_level": "E3",
        },
        {
            "qid": "q014",
            "cid": "WBC-0017::doc0001::0000",
            "question": "What respiratory effects were documented in workers exposed to mixed hardwood dust in WBC-0017?",
            "quote": "Measured, significant reduction in lung function (VC, FEV1, PEFR) and elevated IgE/IgG in workers exposed to mixed tropical hardwood dust",
            "gold_answer": "Workers exposed to mixed tropical hardwood dust showed significant reductions in lung function parameters (VC, FEV1, PEFR) and elevated blood immunoglobulins IgE and IgG.",
            "question_type": "scientific_explanation",
            "craft": "Wood",
            "expected_evidence_level": "E3",
        },
        {
            "qid": "q015",
            "cid": "WBC-0018::doc0001::0000",
            "question": "What is the primary driver of dimensional change and cracking in seasoned wood according to WBC-0018?",
            "quote": "Explains that wood continuously exchanges moisture with its environment and that seasonal (not daily) humidity swings are the main driver of dimensional change and cracking",
            "gold_answer": "Seasonal (rather than daily) ambient humidity swings are the primary driver of dimensional change, moisture exchange, and cracking in seasoned wood.",
            "question_type": "process",
            "craft": "Wood",
            "expected_evidence_level": "E3",
        },
        {
            "qid": "q016",
            "cid": "WBC-0022::doc0001::0000",
            "question": "What does study WBC-0022 establish regarding motor variability in repetitive precision manual tasks?",
            "quote": "Confirms motor variability is an inherent, measurable feature of human repetitive manual precision tasks",
            "gold_answer": "Motor variability is confirmed to be an inherent, measurable characteristic of human performance across repetitive manual precision tasks.",
            "question_type": "factual",
            "craft": "Wood",
            "expected_evidence_level": "E3",
        },
        {
            "qid": "q017",
            "cid": "WBC-0023::doc0001::0000",
            "question": "What personal noise exposure levels were measured for woodworking power tools like drills in WBC-0023?",
            "quote": "Measures personal noise exposure (~80 dBA, peaks near 1",
            "gold_answer": "Personal noise exposures around 80 dBA with frequency peaks concentrated in the 1–4 kHz range were measured from powered woodworking tools.",
            "question_type": "factual",
            "craft": "Wood",
            "expected_evidence_level": "E3",
        },
        {
            "qid": "q018",
            "cid": "WBC-0020::doc0001::0000",
            "question": "How do veneer layup angle and arrangement govern plywood performance under load in WBC-0020?",
            "quote": "Shows that veneer layup angle and layer arrangement directly govern strain distribution and dimensional stability under load",
            "gold_answer": "Veneer layup angle and layer arrangement directly control the internal strain distribution and dimensional stability of plywood under mechanical load.",
            "question_type": "process",
            "craft": "Wood",
            "expected_evidence_level": "E3",
        },

        # --- 10 full_paper_chunk questions ---
        {
            "qid": "q019",
            "cid": "WBC-0051::doc0004::0000",
            "question": "How is body strength defined in ergonomics review WBC-0051?",
            "quote": "Strength can be defined as the capacity of body parts to produce force or torque with voluntary muscle contraction",
            "gold_answer": "Strength is defined as the capacity of body parts to produce force or torque through voluntary muscle contraction.",
            "question_type": "factual",
            "craft": "Bamboo (Battal chisel handles used in root carving)",
            "expected_evidence_level": "E3",
        },
        {
            "qid": "q020",
            "cid": "WBC-0051::doc0005::0000",
            "question": "What common instruments measure hand grip strength according to WBC-0051?",
            "quote": "Hand dynamometer and force gauge are the most common devices used to measure the hand grip strength.",
            "gold_answer": "Hand dynamometers and force gauges are the most common measurement devices used to quantify hand grip strength.",
            "question_type": "factual",
            "craft": "Bamboo (Battal chisel handles used in root carving)",
            "expected_evidence_level": "E3",
        },
        {
            "qid": "q021",
            "cid": "WBC-0051::doc0005::0000",
            "question": "For what three ergonomics applications is hand grip physical strength data utilized according to WBC-0051?",
            "quote": "1) hand tool design – as guidelines and references for creating consumer products and industrial hand tools [3], [4]. 2) Worker selection and placement programs - to ensure the jobs involving heavy physical demands are not performed by those who lack the necessary strength capabilities [5]. 3) Job design – to match the worker strength and the job demands",
            "gold_answer": "It is utilized for 1) hand tool design guidelines, 2) worker selection and placement programs, and 3) job design matching worker capabilities to job demands.",
            "question_type": "factual",
            "craft": "Bamboo (Battal chisel handles used in root carving)",
            "expected_evidence_level": "E3",
        },
        {
            "qid": "q022",
            "cid": "WBC-0051::doc0006::0000",
            "question": "Which anatomical structures comprise the human hand according to the review WBC-0051?",
            "quote": "Hand is made up of bones, tendons, ligaments, nerves and",
            "gold_answer": "The hand is anatomically composed of bones, tendons, ligaments, nerves, and musculature.",
            "question_type": "factual",
            "craft": "Bamboo (Battal chisel handles used in root carving)",
            "expected_evidence_level": "E3",
        },
        {
            "qid": "q023",
            "cid": "WBC-0051::doc0007::0000",
            "question": "Which handle geometric and physical parameters influence grip performance in WBC-0051?",
            "quote": "handle shape, handle material, handle flange, handle inter distance",
            "gold_answer": "Handle shape, handle material, presence of a handle flange, and handle inter-distance are key parameters influencing grip performance.",
            "question_type": "process",
            "craft": "Bamboo (Battal chisel handles used in root carving)",
            "expected_evidence_level": "E3",
        },
        {
            "qid": "q024",
            "cid": "WBC-2016::doc0004::0000",
            "question": "What algorithmic framework is integrated into robot wood carving with hand tools in WBC-2016?",
            "quote": "we perform a case study on robot wood carving with hand tools, in which we integrate probabilistic movement primitives (ProMPs)",
            "gold_answer": "Probabilistic movement primitives (ProMPs) are integrated into robot wood carving with hand tools to model tool trajectories.",
            "question_type": "process",
            "craft": "Robotic wood carving and digital fabrication; indirect relevance to traditional craft processes",
            "expected_evidence_level": "E3",
        },
        {
            "qid": "q025",
            "cid": "WBC-2016::doc0003::0000",
            "question": "What limitation of conventional CAM tools is highlighted in research paper WBC-2016?",
            "quote": "However, conventional CAM tools are tailored for standard manufacturing processes",
            "gold_answer": "Conventional CAM tools are tailored strictly for standard rigid manufacturing processes rather than nuanced craft techniques like wood carving.",
            "question_type": "scientific_explanation",
            "craft": "Robotic wood carving and digital fabrication; indirect relevance to traditional craft processes",
            "expected_evidence_level": "E3",
        },
        {
            "qid": "q026",
            "cid": "WBC-2016::doc0005::0000",
            "question": "What limitation of robotic carving demonstration is evaluated across multiple wood carving applications in WBC-2016?",
            "quote": "multiple wood carving applications and discuss its limitations, including the need for iterative toolpath adjustments to address inaccuracies.",
            "gold_answer": "The study demonstrates the robotic carving pipeline across multiple tasks and notes limitations including the need for iterative toolpath adjustments to address inaccuracies.",
            "question_type": "process",
            "craft": "Robotic wood carving and digital fabrication; indirect relevance to traditional craft processes",
            "expected_evidence_level": "E3",
        },
        {
            "qid": "q027",
            "cid": "WBC-0051::doc0005::0000",
            "question": "How do tool design guidelines derived from grip strength measurements connect to worker safety in manual crafts?",
            "quote": "Grip strength is a magnitude of muscle strength generated by the hand and forearm muscles while gripping an object.",
            "gold_answer": "Optimizing tool handle dimensions to match hand grip strength distributions prevents forearm muscle overexertion and reduces repetitive strain injuries.",
            "question_type": "multi_hop",
            "craft": "Bamboo (Battal chisel handles used in root carving)",
            "expected_evidence_level": "E3",
        },
        {
            "qid": "q028",
            "cid": "WBC-2016::doc0004::0000",
            "question": "How does learning from human demonstration bridge traditional carving techniques with automated computer-aided manufacturing?",
            "quote": "we perform a case study on robot wood carving with hand tools",
            "gold_answer": "By recording human carving demonstrations into probabilistic movement primitives, the robot learns force and trajectory variations characteristic of artisan carving.",
            "question_type": "multi_hop",
            "craft": "Robotic wood carving and digital fabrication; indirect relevance to traditional craft processes",
            "expected_evidence_level": "E3",
        },

        # --- 5 process_observation questions ---
        {
            "qid": "q029",
            "cid": "WBC-0001::doc0003::0000",
            "question": "What authoritative handbook serves as the core reference for wood physical and mechanical properties in process observation WBC-0001?",
            "quote": "Authoritative USDA Forest Products Laboratory handbook covering mechanical/physical properties of wood — core reference for product shape, thickness and durability Why-questions.",
            "gold_answer": "The authoritative USDA Forest Products Laboratory handbook covers wood mechanical and physical properties governing product shape, thickness, and durability.",
            "question_type": "factual",
            "craft": "Wood",
            "expected_evidence_level": "E3",
        },
        {
            "qid": "q030",
            "cid": "WBC-0002::doc0003::0000",
            "question": "What comprehensive mechanical property data across bamboo studies are summarized in observation WBC-0002?",
            "quote": "Peer-reviewed thesis reviewing shear/compressive/tensile/bending/MOE data across many bamboo studies; directly supports sustainability and performance claims.",
            "gold_answer": "It synthesizes experimental data for shear, compressive, tensile, bending strength, and modulus of elasticity (MOE) across numerous bamboo studies.",
            "question_type": "factual",
            "craft": "Bamboo",
            "expected_evidence_level": "E3",
        },
        {
            "qid": "q031",
            "cid": "WBC-0003::doc0003::0000",
            "question": "What mechanical weakness in bamboo culms is documented in observation WBC-0003?",
            "quote": "ScienceDirect review with quantified MOE/MOR/tensile data and node-weakness explanation; strong evidence for engineering-based sustainability claims.",
            "gold_answer": "It provides quantified MOE, MOR, and tensile data explaining why node regions represent mechanical points of weakness in bamboo culms.",
            "question_type": "scientific_explanation",
            "craft": "Bamboo",
            "expected_evidence_level": "E3",
        },
        {
            "qid": "q032",
            "cid": "WBC-0005::doc0003::0000",
            "question": "What physical property relationship in bamboo is aggregated in observation WBC-0005?",
            "quote": "Annotated bibliography aggregating multiple bamboo mechanical-property studies incl. the moisture-strength relationship",
            "gold_answer": "It aggregates studies examining the fundamental relationship between culm moisture content and mechanical strength in bamboo.",
            "question_type": "multi_hop",
            "craft": "Bamboo",
            "expected_evidence_level": "E4",
        },
        {
            "qid": "q033",
            "cid": "WBC-0006::doc0003::0000",
            "question": "What comparison across bamboo species is provided in observation WBC-0006?",
            "quote": "Peer-reviewed species comparison of bamboo fibre strength/composition; useful for species-selection Why-questions.",
            "gold_answer": "It compares fiber tensile strength and anatomical chemical composition across different bamboo species for material selection.",
            "question_type": "multi_hop",
            "craft": "Bamboo",
            "expected_evidence_level": "E3",
        },

        # --- 12 unanswerable questions (Abstain) ---
        {
            "qid": "q034",
            "cid": None,
            "question": "Why is Udayagiri wood specifically soaked in water for four days before initial shaping?",
            "quote": "",
            "gold_answer": "Insufficient evidence in the retrieved corpus to answer this question.",
            "question_type": "unanswerable",
            "craft": "Wood",
            "expected_evidence_level": "None",
        },
        {
            "qid": "q035",
            "cid": None,
            "question": "What is the optimal laser power and powder bed preheating temperature for additive manufacturing of Ti-6Al-4V titanium alloy turbine blades?",
            "quote": "",
            "gold_answer": "Insufficient evidence in the retrieved corpus to answer this question.",
            "question_type": "unanswerable",
            "craft": "None",
            "expected_evidence_level": "None",
        },
        {
            "qid": "q036",
            "cid": None,
            "question": "What annealing lehr temperature ramp prevents stress-induced shattering in hot-worked borosilicate glass flasks?",
            "quote": "",
            "gold_answer": "Insufficient evidence in the retrieved corpus to answer this question.",
            "question_type": "unanswerable",
            "craft": "None",
            "expected_evidence_level": "None",
        },
        {
            "qid": "q037",
            "cid": None,
            "question": "Which organometallic tin catalyst optimizes reaction kinetics during industrial reaction injection molding of polyurethane?",
            "quote": "",
            "gold_answer": "Insufficient evidence in the retrieved corpus to answer this question.",
            "question_type": "unanswerable",
            "craft": "None",
            "expected_evidence_level": "None",
        },
        {
            "qid": "q038",
            "cid": None,
            "question": "What whorl diameter ratio produces 28 twists per inch when spinning Egyptian Giza cotton on a traditional flyer wheel?",
            "quote": "",
            "gold_answer": "Insufficient evidence in the retrieved corpus to answer this question.",
            "question_type": "unanswerable",
            "craft": "None",
            "expected_evidence_level": "None",
        },
        {
            "qid": "q039",
            "cid": None,
            "question": "What potassium gold cyanide bath concentration produces a mirror-bright 24-karat gold electroplate on brass?",
            "quote": "",
            "gold_answer": "Insufficient evidence in the retrieved corpus to answer this question.",
            "question_type": "unanswerable",
            "craft": "None",
            "expected_evidence_level": "None",
        },
        {
            "qid": "q040",
            "cid": None,
            "question": "What molar ratio of silica to alumina eliminates crazing in cone 10 reduction-fired porcelain glazes?",
            "quote": "",
            "gold_answer": "Insufficient evidence in the retrieved corpus to answer this question.",
            "question_type": "unanswerable",
            "craft": "None",
            "expected_evidence_level": "None",
        },
        {
            "qid": "q041",
            "cid": None,
            "question": "What pulse repetition rate and assist gas pressure are required to laser cut optical grade acrylic without edge whitening?",
            "quote": "",
            "gold_answer": "Insufficient evidence in the retrieved corpus to answer this question.",
            "question_type": "unanswerable",
            "craft": "None",
            "expected_evidence_level": "None",
        },
        {
            "qid": "q042",
            "cid": None,
            "question": "What basic chromium sulfate concentration is mandated for heavy bovine sole leather tanning in European standard EN 15987?",
            "quote": "",
            "gold_answer": "Insufficient evidence in the retrieved corpus to answer this question.",
            "question_type": "unanswerable",
            "craft": "None",
            "expected_evidence_level": "None",
        },
        {
            "qid": "q043",
            "cid": None,
            "question": "What tin percentage in 78-22 bell bronze optimizes the acoustic strike note and hum tone decay in cathedral bells?",
            "quote": "",
            "gold_answer": "Insufficient evidence in the retrieved corpus to answer this question.",
            "question_type": "unanswerable",
            "craft": "None",
            "expected_evidence_level": "None",
        },
        {
            "qid": "q044",
            "cid": None,
            "question": "What alum mordant bath concentration maximizes lightfastness of alizarin dye on degummed Bombyx mori silk yarn?",
            "quote": "",
            "gold_answer": "Insufficient evidence in the retrieved corpus to answer this question.",
            "question_type": "unanswerable",
            "craft": "None",
            "expected_evidence_level": "None",
        },
        {
            "qid": "q045",
            "cid": None,
            "question": "What exact polymerization initiator mechanism drives cyanoacrylate curing when exposed to heartwood tannin extracts in artisan wood repairs?",
            "quote": "",
            "gold_answer": "Insufficient evidence in the retrieved corpus to answer this question.",
            "question_type": "unanswerable",
            "craft": "Wood",
            "expected_evidence_level": "None",
        }
    ]

    eval_questions = []
    review_rows = []

    for s in specs:
        qid = s["qid"]
        q_type = s["question_type"]
        is_answerable = (q_type != "unanswerable")

        if is_answerable:
            chunk = get_chunk(s["cid"], s["quote"])
            c_id = chunk["chunk_id"]
            r_id = chunk["record_id"]
            p_id = chunk.get("prov_record_id")
            gold_ids = [c_id]
            if r_id and r_id not in gold_ids:
                gold_ids.append(r_id)
            if p_id and p_id not in gold_ids:
                gold_ids.append(p_id)
            doc_type = chunk["document_type"]
            craft = chunk["craft"] or s["craft"]
            ev_level = chunk["evidence_level"] or s["expected_evidence_level"]
            quote = s["quote"]
        else:
            gold_ids = []
            doc_type = "unanswerable"
            craft = s["craft"]
            ev_level = "None"
            quote = ""

        rec = {
            "qid": qid,
            "question": s["question"],
            "gold_record_ids": gold_ids,
            "gold_answer": s["gold_answer"],
            "gold_evidence_quote": quote,
            "question_type": q_type,
            "expected_evidence_level": ev_level,
            "document_type": doc_type,
            "craft": craft,
            "paraphrase_group": None,
            "origin": "manual" if qid in ["q034", "q045"] else "auto_drafted",
            "needs_human_review": True
        }
        eval_questions.append(rec)

        review_rows.append({
            "qid": qid,
            "question": s["question"],
            "question_type": q_type,
            "document_type": doc_type,
            "craft": craft,
            "expected_evidence_level": ev_level,
            "gold_record_ids": "; ".join(gold_ids),
            "gold_evidence_quote": quote,
            "gold_answer": s["gold_answer"],
            "origin": rec["origin"],
            "needs_human_review": "TRUE"
        })

    # Save data/eval/eval_questions.jsonl
    jsonl_path = Path("data/eval/eval_questions.jsonl")
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    with jsonl_path.open("w", encoding="utf-8") as f:
        for q in eval_questions:
            f.write(json.dumps(q, ensure_ascii=False) + "\n")
    print(f"Successfully generated {len(eval_questions)} questions at {jsonl_path}")

    # Save data/eval/review_sheet.csv
    csv_path = Path("data/eval/review_sheet.csv")
    with csv_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "qid", "question", "question_type", "document_type", "craft",
            "expected_evidence_level", "gold_record_ids", "gold_evidence_quote",
            "gold_answer", "origin", "needs_human_review"
        ])
        writer.writeheader()
        writer.writerows(review_rows)
    print(f"Successfully generated review sheet at {csv_path}")

if __name__ == "__main__":
    main()
