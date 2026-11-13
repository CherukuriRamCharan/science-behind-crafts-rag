# Benchmark overlap audit

This audit applies the validator’s documented question-to-gold-quote metric: unique lowercased alphanumeric content tokens (stopwords and tokens of length ≤2 removed), with `|question tokens ∩ quote tokens| / |question tokens|`. The review requires overlap **below 50%**. Equality is therefore invalid. This is a diagnostic only; `data/eval/eval_questions.jsonl` was not changed.

The review reported **22** questions above 50%; this audit confirms those 22 and finds **two additional questions exactly at 50%** (q012 and q027). Since the stated rule is strictly below 50%, **24 answerable questions fail** in total. Each listed quote was checked as a verbatim substring of at least one indexed chunk identified by the question’s `gold_record_ids`. The primary cause is benchmark wording that repeats the same subject and relation terms used in its gold evidence or annotation; the validator’s token calculation and gold chunk mapping agree with the records.

| QID | Overlap | Shared / question tokens | Shared tokens | Gold chunk(s), type, quote found |
|---|---:|---:|---|---|
| q001 | 80.0% | 8 / 10 | dust, exposure, finishing, hand, handheld, manual, powered, sanding | `WBC-0005::doc0001::0000` (source_document; exact quote yes) |
|  |  |  | **Question:** How does dust exposure from powered handheld sanding relate to manual hand-sanding in workshop finishing? | **Gold evidence:** This study directly measures exposure from powered handheld sanding, closely matching the machine-finishing stage; it serves as a reasonable but indirect proxy for the subsequent manual hand-sanding stage, where dust generation is expected to be lower but has not been separately measured. |
| q003 | 87.5% | 7 / 8 | bridges, concentration, narrow, openings, pierced, stress, wood | `WBC-0008::doc0001::0000` (source_document; exact quote yes) |
|  |  |  | **Question:** Why are narrow bridges between pierced openings in wood susceptible to stress concentration? | **Gold evidence:** Analyzes how wood's natural anisotropy affects stress concentration around circular and elliptic holes, providing a mechanical basis for why narrow bridges between pierced openings |
| q004 | 70.0% | 7 / 10 | angle, cutting, grain, path, relative, tool, torn | `WBC-0009::doc0001::0000` (source_document; exact quote yes) |
|  |  |  | **Question:** How does the angle of the wood grain relative to a cutting tool path explain torn-grain defects? | **Gold evidence:** Directly measures how grain angle relative to the tool path governs strain and cleavage ahead of the cutting edge, providing a mechanical explanation for the torn-grain and fuzzy-g |
| q005 | 75.0% | 6 / 8 | after, density, grain, raising, sanding, wood | `WBC-0010::doc0001::0000` (source_document; exact quote yes) |
|  |  |  | **Question:** What causes grain raising after wood sanding, and how does wood density affect it? | **Gold evidence:** Shows that sanding tears cell walls to create loosely bonded fibre slivers that project after wetting, and that grain raising is inversely related to wood density |
| q006 | 66.7% | 6 / 9 | adhesives, articles, coatings, conditions, contact, food | `WBC-0011::doc0001::0000` (source_document; exact quote yes) |
|  |  |  | **Question:** What conditions govern the permissible use of adhesives and coatings on food-contact articles? | **Gold evidence:** Establishes the legal/regulatory conditions under which adhesives and coatings may be used on food-contact articles |
| q007 | 63.6% | 7 / 11 | adhesive, based, chemistry, evidence, formaldehyde, health, resins | `WBC-0012::doc0001::0000` (source_document; exact quote yes) |
|  |  |  | **Question:** What evidence exists regarding health risks from adhesive chemistry such as formaldehyde-based resins? | **Gold evidence:** Provides experimental toxicity evidence that adhesive chemistry (e.g., formaldehyde-based resins) carries real health risk via dust/emissions |
| q008 | 58.3% | 7 / 12 | design, fatigue, grip, hand, handle, most, strength | `WBC-0003::doc0001::0000` (source_document; exact quote yes) |
|  |  |  | **Question:** Which handle design features most strongly affect grip strength and muscle fatigue in hand tools? | **Gold evidence:** The review synthesizes multiple studies showing handle diameter and material as the most significant hand-tool design factors affecting grip strength and fatigue |
| q009 | 55.6% | 5 / 9 | floor, handicraft, prevalence, sitting, workers | `WBC-0007::doc0001::0000` (source_document; exact quote yes) |
|  |  |  | **Question:** What musculoskeletal disorder prevalence rates were reported for floor-sitting handicraft workers? | **Gold evidence:** Confirms high MSD prevalence (neck 61.6%, lower back 74.8%, knee 54.7%) among floor-sitting Indian precision handicraft workers |
| q010 | 70.0% | 7 / 10 | coconut, exposure, heat, oil, prolonged, time, virgin | `WBC-0014::doc0001::0000` (source_document; exact quote yes) |
|  |  |  | **Question:** How do prolonged heat and time exposure affect the chemical properties of virgin coconut oil? | **Gold evidence:** Directly measures that virgin coconut oil's free fatty acid content and peroxide value increase significantly with prolonged heat/time exposure |
| q011 | 83.3% | 5 / 6 | affect, force, required, splitting, wood | `WBC-0001::doc0001::0000` (source_document; exact quote yes) |
|  |  |  | **Question:** Which factors affect the force required in wood splitting? | **Gold evidence:** The study experimentally investigates wood-splitting force and shows that wood type, log diameter and splitting-wedge design affect the force required. |
| q012 | 50.0% | 3 / 6 | cutting, forces, wood | `WBC-0002::doc0001::0000` (source_document; exact quote yes) |
|  |  |  | **Question:** What relationships govern cutting forces during wood machining? | **Gold evidence:** The review explains the relationship between wood-cutting forces, wood properties, cutting geometry and tool characteristics |
| q013 | 72.7% | 8 / 11 | among, carvers, exposure, hand, noise, tools, using, wood | `WBC-0006::doc0001::0000` (source_document; exact quote yes) |
|  |  |  | **Question:** What average 8-hour noise exposure was measured among wood carvers using hand tools? | **Gold evidence:** Directly measures HAV and noise exposure (mean Lex,8h of 93.17 dB(A), exceeding safe limits) among Indian wood carvers using hand carving tools |
| q014 | 75.0% | 9 / 12 | dust, exposed, function, hardwood, lung, measured, mixed, tropical, workers | `WBC-0017::doc0001::0000` (source_document; exact quote yes) |
|  |  |  | **Question:** What specific lung function and immune changes were measured in workers exposed to mixed tropical hardwood dust? | **Gold evidence:** Measured, significant reduction in lung function (VC, FEV1, PEFR) and elevated IgE/IgG in workers exposed to mixed tropical hardwood dust |
| q015 | 90.0% | 9 / 10 | change, cracking, daily, dimensional, driver, humidity, seasonal, swings, wood | `WBC-0018::doc0001::0000` (source_document; exact quote yes) |
|  |  |  | **Question:** Are daily or seasonal humidity swings the primary driver of wood dimensional change and cracking? | **Gold evidence:** Explains that wood continuously exchanges moisture with its environment and that seasonal (not daily) humidity swings are the main driver of dimensional change and cracking |
| q018 | 60.0% | 6 / 10 | angle, dimensional, load, stability, strain, veneer | `WBC-0020::doc0001::0000` (source_document; exact quote yes) |
|  |  |  | **Question:** How does the angle at which veneer layers are stacked affect strain and dimensional stability in plywood under load? | **Gold evidence:** Shows that veneer layup angle and layer arrangement directly govern strain distribution and dimensional stability under load |
| q019 | 85.7% | 6 / 7 | body, contraction, defined, muscle, parts, strength | `WBC-0051::doc0004::0000` (full_paper_chunk; exact quote yes) |
|  |  |  | **Question:** How is strength defined in terms of body parts and muscle contraction? | **Gold evidence:** Strength can be defined as the capacity of body parts to produce force or torque with voluntary muscle contraction |
| q020 | 75.0% | 6 / 8 | grip, hand, measure, most, strength, used | `WBC-0051::doc0005::0000` (full_paper_chunk; exact quote yes) |
|  |  |  | **Question:** Which instruments are most commonly used to measure hand grip strength? | **Gold evidence:** Hand dynamometer and force gauge are the most common devices used to measure the hand grip strength. |
| q022 | 55.6% | 5 / 9 | data, grip, hand, meaningful, strength | `WBC-0051::doc0005::0000` (full_paper_chunk; exact quote yes) |
|  |  |  | **Question:** Why are data on hand grip strength meaningful beyond ergonomic tool design? | **Gold evidence:** data on hand grip strength are also meaningful for predicting muscle weakness [7], determining the functional capabilities of |
| q024 | 77.8% | 7 / 9 | carving, case, hand, robot, study, tools, wood | `WBC-2016::doc0004::0000` (full_paper_chunk; exact quote yes) |
|  |  |  | **Question:** What framework was integrated into the case study on robot wood carving with hand tools? | **Gold evidence:** we perform a case study on robot wood carving with hand tools, in which we integrate probabilistic movement primitives (ProMPs) |
| q025 | 55.6% | 5 / 9 | cam, conventional, manufacturing, standard, tools | `WBC-2016::doc0003::0000` (full_paper_chunk; exact quote yes) |
|  |  |  | **Question:** What limitation exists regarding conventional CAM tools in non-standard manufacturing? | **Gold evidence:** However, conventional CAM tools are tailored for standard manufacturing processes |
| q027 | 50.0% | 4 / 8 | grip, muscles, object, strength | `WBC-0051::doc0005::0000` (full_paper_chunk; exact quote yes) |
|  |  |  | **Question:** What physiological muscles generate grip strength when holding an object? | **Gold evidence:** Grip strength is a magnitude of muscle strength generated by the hand and forearm muscles while gripping an object. |
| q028 | 66.7% | 6 / 9 | carving, cuts, demonstrations, human, recorded, wood | `WBC-2016::doc0004::0000` (full_paper_chunk; exact quote yes) |
|  |  |  | **Question:** How were human demonstrations of wood carving cuts recorded for the robot case study? | **Gold evidence:** Human demonstrations of different wood carving cuts are recorded via kinesthetic |
| q029 | 80.0% | 8 / 10 | covering, handbook, mechanical, physical, product, properties, reference, wood | `WBC-0001::doc0003::0000` (process_observation; exact quote yes) |
|  |  |  | **Question:** What reference handbook is noted for covering the mechanical and physical properties of wood in product design? | **Gold evidence:** Authoritative USDA Forest Products Laboratory handbook covering mechanical/physical properties of wood — core reference for product shape, thickness and durability Why-questions. |
| q030 | 55.6% | 5 / 9 | bamboo, claims, data, performance, studies | `WBC-0002::doc0003::0000` (process_observation; exact quote yes) |
|  |  |  | **Question:** What mechanical property data across bamboo studies are summarized to support performance claims? | **Gold evidence:** Peer-reviewed thesis reviewing shear/compressive/tensile/bending/MOE data across many bamboo studies; directly supports sustainability and performance claims. |

## Interpretation and proposal

The overlap rule in the original review is explicit: the validator must enforce the documented constraint “overlap below 50%”. No denominator, tokenization, or gold-chunk lookup defect was found for these cases. They are benchmark-data violations under the stated rule. The review itself notes that the benchmark is auto-drafted, not human-validated, and recommends annotator review; it does not authorize rewriting question wording automatically.

Keep all 24 affected records and their gold evidence unchanged pending human review. A reviewer can propose revised question wording for each affected QID, preserve the existing evidence link and answer intent, then rerun the validator. No questions were deleted, and no gold evidence was fabricated or rewritten in this audit.

Validation after the enforcement change: the committed dataset remains invalid because the original 22 questions above 50% plus q012 and q027 exactly at 50% fail the strict below-50% rule.
