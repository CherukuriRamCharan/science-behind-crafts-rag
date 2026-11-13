# Benchmark overlap corrections — human review required

This is a proposal only. The committed dataset `data/eval/eval_questions.jsonl`, its gold answers, and its evidence were not changed. No item should be applied until a human reviewer approves its wording and confirms the retained gold annotation. Current rates below use the validator's documented unique-content-token intersection divided by unique question content tokens; the required limit is strictly below 50%.

Each proposed question is intended to preserve the current factual target while replacing wording that closely repeats the evidence. “Answerability” is scoped to the cited original gold quote. The quote remains unchanged and should be rechecked verbatim against the indexed record after a reviewer approves any wording.

## Questions above 50%

### q001 — 80.0% original; 35.7% proposed (5/14)
- **Original question:** How does dust exposure from powered handheld sanding relate to manual hand-sanding in workshop finishing?
- **Gold answer:** Powered handheld sanding serves as an indirect proxy for manual hand-sanding, where dust generation is expected to be lower but has not been separately measured.
- **Gold evidence:** “This study directly measures exposure from powered handheld sanding, closely matching the machine-finishing stage; it serves as a reasonable but indirect proxy for the subsequent manual hand-sanding stage, where dust generation is expected to be lower but has not been separately measured.”
- **Why overlap is high:** The question repeats the evidence’s central exposure, tool, sanding, and process-stage terms.
- **Suggested wording:** In the wood-dust sanding study, how should the mechanized observation inform expectations for later hand-process exposure, and what caveat remains?
- **Expected intent:** Interpret the powered-sanding measurement as an indirect, likely higher-exposure proxy and note that the manual step was not directly measured.
- **Evidence check:** The quote explicitly identifies the powered process as an indirect proxy, predicts lower dust for the later manual step, and says that step was not separately measured.

### q003 — 87.5% original; 23.1% proposed (3/13)
- **Original question:** Why are narrow bridges between pierced openings in wood susceptible to stress concentration?
- **Gold answer:** Wood's natural anisotropy affects stress concentration around circular and elliptic holes, providing a mechanical basis for stress buildup in narrow bridges between pierced openings.
- **Gold evidence:** “Analyzes how wood's natural anisotropy affects stress concentration around circular and elliptic holes, providing a mechanical basis for why narrow bridges between pierced openings”
- **Why overlap is high:** Nearly every content word names the same material, geometry, and mechanical phenomenon as the quote.
- **Suggested wording:** For the wood-laminate hole analysis, what intrinsic material behavior explains higher loads in the narrow strips between openings?
- **Expected intent:** Connect wood's direction-dependent properties to stress concentration around openings and the narrow material left between them.
- **Evidence check:** The quote names wood's natural anisotropy and stress concentration around holes, explicitly linking that analysis to narrow bridges between openings.

### q004 — 70.0% original; 25.0% proposed (3/12)
- **Original question:** How does the angle of the wood grain relative to a cutting tool path explain torn-grain defects?
- **Gold answer:** Grain angle relative to the tool path governs strain and cleavage ahead of the cutting edge, mechanically explaining torn-grain and fuzzy-grain defects.
- **Gold evidence:** “Directly measures how grain angle relative to the tool path governs strain and cleavage ahead of the cutting edge, providing a mechanical explanation for the torn-grain and fuzzy-g”
- **Why overlap is high:** It repeats the quote's grain-angle, path, cutting-defect vocabulary almost verbatim.
- **Suggested wording:** In Hinoki orthogonal-cutting experiments on grain direction, what physical mechanism produces fuzzy cut surfaces?
- **Expected intent:** Explain the defect through strain and cleavage controlled by grain orientation relative to the cut.
- **Evidence check:** The quote directly connects orientation to strain and cleavage ahead of the cutting edge and identifies torn-grain/fuzzy-grain defects.

### q005 — 75.0% original; 38.5% proposed (5/13)
- **Original question:** What causes grain raising after wood sanding, and how does wood density affect it?
- **Gold answer:** Sanding tears cell walls creating loosely bonded fiber slivers that project after wetting, and grain raising is inversely related to wood density.
- **Gold evidence:** “Shows that sanding tears cell walls to create loosely bonded fibre slivers that project after wetting, and that grain raising is inversely related to wood density”
- **Why overlap is high:** The prompt uses the same terms for the preparation step, material, outcome, and density relationship.
- **Suggested wording:** In the wood grain-raising study, why do loosened fibres lift after dampening, and how does species density alter the effect?
- **Expected intent:** Identify torn cell walls and projecting fibre slivers after wetting, with lower density associated with more grain raising.
- **Evidence check:** The quote states that preparation tears cell walls, wetting makes loose slivers project, and the effect is inversely related to density.

### q006 — 66.7% original; 41.7% proposed (5/12)
- **Original question:** What conditions govern the permissible use of adhesives and coatings on food-contact articles?
- **Gold answer:** Specific legal and regulatory conditions establish when and how adhesives and coatings may be used on food-contact articles.
- **Gold evidence:** “Establishes the legal/regulatory conditions under which adhesives and coatings may be used on food-contact articles”
- **Why overlap is high:** The question reproduces most of the regulatory summary's subject and governing-conditions language.
- **Suggested wording:** Under 21 CFR Part 175, which use conditions apply to adhesives and coating ingredients on food-contact articles?
- **Expected intent:** Identify the legal conditions for use of the relevant bonding and surface-finish materials on food-contact articles.
- **Evidence check:** The quote expressly says legal/regulatory conditions establish when those two material classes may be used on food-contact articles.

### q007 — 63.6% original; 27.3% proposed (3/11)
- **Original question:** What evidence exists regarding health risks from adhesive chemistry such as formaldehyde-based resins?
- **Gold answer:** Experimental toxicity evidence confirms that adhesive chemistry like formaldehyde-based resins carries health risks via dust and emissions.
- **Gold evidence:** “Provides experimental toxicity evidence that adhesive chemistry (e.g., formaldehyde-based resins) carries real health risk via dust/emissions”
- **Why overlap is high:** It repeats the quote's adhesive, chemistry, resin example, and health-risk terms.
- **Suggested wording:** In the formaldehyde-free adhesive toxicology paper, what evidence supports concern about airborne exposure?
- **Expected intent:** State that experimental toxicity work indicates risks associated with dust or emissions from formaldehyde-based resin systems.
- **Evidence check:** The quote identifies experimental toxicity evidence and links resin chemistry to health risk through dust/emissions.

### q008 — 58.3% original; 46.2% proposed (6/13)
- **Original question:** Which handle design features most strongly affect grip strength and muscle fatigue in hand tools?
- **Gold answer:** Handle diameter and handle material are the most significant hand-tool design factors affecting grip strength and muscle fatigue.
- **Gold evidence:** “The review synthesizes multiple studies showing handle diameter and material as the most significant hand-tool design factors affecting grip strength and fatigue”
- **Why overlap is high:** The prompt repeats handle/design, grip, strength, fatigue and tool concepts from the finding.
- **Suggested wording:** For the hand-tool design review, which pair of handle characteristics matters most for users' force capacity and tiredness?
- **Expected intent:** Name the grip dimensions (diameter) and material as the leading factors for grip force and fatigue.
- **Evidence check:** The quote identifies diameter and material as the most significant factors affecting grip strength and fatigue.

### q009 — 55.6% original; 46.2% proposed (6/13)
- **Original question:** What musculoskeletal disorder prevalence rates were reported for floor-sitting handicraft workers?
- **Gold answer:** High MSD prevalence was confirmed among floor-sitting precision handicraft workers, specifically 61.6% for neck, 74.8% for lower back, and 54.7% for knee.
- **Gold evidence:** “Confirms high MSD prevalence (neck 61.6%, lower back 74.8%, knee 54.7%) among floor-sitting Indian precision handicraft workers”
- **Why overlap is high:** The question repeats the population, posture, and prevalence terminology.
- **Suggested wording:** Among Indian precision-craft workers seated on the floor, what proportions had cervical, lumbar, and knee complaints?
- **Expected intent:** Report the three body-region percentages in the specified worker sample.
- **Evidence check:** The quote gives the exact neck, lower-back and knee percentages and identifies the surveyed population.

### q010 — 70.0% original; 33.3% proposed (4/12)
- **Original question:** How do prolonged heat and time exposure affect the chemical properties of virgin coconut oil?
- **Gold answer:** Prolonged heat and time exposure significantly increase free fatty acid content and peroxide value in virgin coconut oil.
- **Gold evidence:** “Directly measures that virgin coconut oil's free fatty acid content and peroxide value increase significantly with prolonged heat/time exposure”
- **Why overlap is high:** It echoes the material, exposure, duration, and measurement terms used by the evidence.
- **Suggested wording:** During extended frying trials of virgin coconut oil, which two chemical indicators rose as exposure continued?
- **Expected intent:** Name free fatty acid content and peroxide value as increasing after extended heat/time exposure.
- **Evidence check:** The quote names both indicators and states that they increase significantly with prolonged heat/time exposure.

### q011 — 83.3% original; 30.0% proposed (3/10)
- **Original question:** Which factors affect the force required in wood splitting?
- **Gold answer:** Wood type, log diameter, and splitting-wedge design affect the force required to split wood.
- **Gold evidence:** “The study experimentally investigates wood-splitting force and shows that wood type, log diameter and splitting-wedge design affect the force required.”
- **Why overlap is high:** The short question shares nearly all its content words with the evidence.
- **Suggested wording:** In the horizontal wood-splitter experiment, which properties of the billet and wedge changed the effort required?
- **Expected intent:** Identify species/type, log diameter, and wedge design as the determinants of splitting effort.
- **Evidence check:** The quote lists wood type, log diameter and splitting-wedge design as affecting the required force.

## Questions exactly at 50%

### q012 — 50.0% original; 36.4% proposed (4/11)
- **Original question:** What relationships govern cutting forces during wood machining?
- **Gold answer:** Wood-cutting forces depend on the relationship between wood properties, cutting geometry, and tool characteristics.
- **Gold evidence:** “The review explains the relationship between wood-cutting forces, wood properties, cutting geometry and tool characteristics”
- **Why overlap is at the boundary:** The question shares three of its six content tokens with the evidence. Equality fails the strict below-50% requirement.
- **Suggested wording:** In the industrial wood-machining forces review, which three classes of inputs determine cutting resistance?
- **Expected intent:** Name wood properties, cutting geometry and tool characteristics as determinants of force.
- **Evidence check:** The quote states the relationship among all three determinants and the force encountered during wood cutting.

## Remaining questions above 50%

### q013 — 72.7% original; 35.7% proposed (5/14)
- **Original question:** What average 8-hour noise exposure was measured among wood carvers using hand tools?
- **Gold answer:** A mean 8-hour noise exposure (Lex,8h) of 93.17 dB(A), exceeding safe limits, was measured among wood carvers using hand carving tools.
- **Gold evidence:** “Directly measures HAV and noise exposure (mean Lex,8h of 93.17 dB(A), exceeding safe limits) among Indian wood carvers using hand carving tools”
- **Why overlap is high:** It reproduces the evidence's population, tools, and exposure-measure wording.
- **Suggested wording:** In the Indian hand-arm-vibration and noise assessment of wood carvers, what eight-hour sound level was recorded on average?
- **Expected intent:** Report Lex,8h = 93.17 dB(A) and that it exceeds safe limits.
- **Evidence check:** The quote provides the measure, value, population, and comparison with safe limits.

### q014 — 75.0% original; 30.0% proposed (3/10)
- **Original question:** What specific lung function and immune changes were measured in workers exposed to mixed tropical hardwood dust?
- **Gold answer:** Workers showed significant reductions in lung function (VC, FEV1, PEFR) and elevated IgE and IgG levels.
- **Gold evidence:** “Measured, significant reduction in lung function (VC, FEV1, PEFR) and elevated IgE/IgG in workers exposed to mixed tropical hardwood dust”
- **Why overlap is high:** Most question concepts and the exposure population are repeated in the quote.
- **Suggested wording:** In the mixed-hardwood dust study, which breathing tests declined and which immune markers rose?
- **Expected intent:** Name VC, FEV1, PEFR decreases and IgE/IgG increases.
- **Evidence check:** The quote enumerates the three lung-function measures, two immune markers, and exposed population.

### q015 — 90.0% original; 46.2% proposed (6/13)
- **Original question:** Are daily or seasonal humidity swings the primary driver of wood dimensional change and cracking?
- **Gold answer:** Seasonal (not daily) humidity swings are the main driver of dimensional change and cracking, as wood continuously exchanges moisture with its environment.
- **Gold evidence:** “Explains that wood continuously exchanges moisture with its environment and that seasonal (not daily) humidity swings are the main driver of dimensional change and cracking”
- **Why overlap is high:** It restates nearly the entire evidence proposition with the same temporal contrast and outcomes.
- **Suggested wording:** Does the wood moisture-control evidence assign greater dimensional movement to daily humidity fluctuations or seasonal patterns?
- **Expected intent:** Distinguish seasonal changes as the main driver from daily variation, due to continuing moisture exchange.
- **Evidence check:** The quote states the seasonal/not-daily distinction and links it to dimensional change and cracking through moisture exchange.

### q018 — 60.0% original; 23.1% proposed (3/13)
- **Original question:** How does the angle at which veneer layers are stacked affect strain and dimensional stability in plywood under load?
- **Gold answer:** Veneer layup angle and layer arrangement directly govern strain distribution and dimensional stability under load.
- **Gold evidence:** “Shows that veneer layup angle and layer arrangement directly govern strain distribution and dimensional stability under load”
- **Why overlap is high:** It repeats the layup, angle, strain, stability and loading language.
- **Suggested wording:** In the plywood bending-load experiment, how does changing veneer orientation alter strain patterns and resistance to shape change?
- **Expected intent:** Explain that layer orientation/arrangement governs strain distribution and dimensional stability.
- **Evidence check:** The quote directly links veneer layup angle and arrangement to strain distribution and dimensional stability under load.

### q019 — 85.7% original; 6.7% proposed (1/15)
- **Original question:** How is strength defined in terms of body parts and muscle contraction?
- **Gold answer:** Strength is defined as the capacity of body parts to produce force or torque with voluntary muscle contraction.
- **Gold evidence:** “Strength can be defined as the capacity of body parts to produce force or torque with voluntary muscle contraction”
- **Why overlap is high:** It paraphrases a short definition with nearly the same terms.
- **Suggested wording:** In "The Influence of Hand Tool Design on Hand Grip Strength: A Review," what physical output defines strength when a person contracts muscles voluntarily?
- **Expected intent:** Define strength as voluntary generation of force or torque by body parts.
- **Evidence check:** The quote gives that definition explicitly.

### q020 — 75.0% original; 46.2% proposed (6/13)
- **Original question:** Which instruments are most commonly used to measure hand grip strength?
- **Gold answer:** Hand dynamometers and force gauges are the most common devices used to measure hand grip strength.
- **Gold evidence:** “Hand dynamometer and force gauge are the most common devices used to measure the hand grip strength.”
- **Why overlap is high:** It shares the measurement, hand/grip, and prevalence wording with the evidence.
- **Suggested wording:** In "The Influence of Hand Tool Design on Hand Grip Strength: A Review," which two instruments are most commonly used to measure hand-grip strength?
- **Expected intent:** Name a hand dynamometer and a force gauge.
- **Evidence check:** The quote names both devices and states that they are the most common for this measurement.

### q022 — 55.6% original; 33.3% proposed (4/12)
- **Original question:** Why are data on hand grip strength meaningful beyond ergonomic tool design?
- **Gold answer:** Data on hand grip strength are meaningful for predicting muscle weakness and determining functional capabilities.
- **Gold evidence:** “data on hand grip strength are also meaningful for predicting muscle weakness [7], determining the functional capabilities of”
- **Why overlap is high:** It shares the data/measurement subject and significance language directly.
- **Suggested wording:** In "The Influence of Hand Tool Design on Hand Grip Strength: A Review," what health and functional assessments can hand-grip measurements support?
- **Expected intent:** The measure can help predict muscle weakness and assess functional capability.
- **Evidence check:** The quote explicitly gives both of those uses beyond tool design.

### q024 — 77.8% original; 44.4% proposed (4/9)
- **Original question:** What framework was integrated into the case study on robot wood carving with hand tools?
- **Gold answer:** Probabilistic movement primitives (ProMPs) were integrated into the case study on robot wood carving with hand tools.
- **Gold evidence:** “we perform a case study on robot wood carving with hand tools, in which we integrate probabilistic movement primitives (ProMPs)”
- **Why overlap is high:** The question repeats the case-study setting and nearly all subject terms.
- **Suggested wording:** In the robot wood-carving learning study, what method represented the demonstrated motions?
- **Expected intent:** Identify probabilistic movement primitives (ProMPs).
- **Evidence check:** The quote directly says that ProMPs were integrated into the robot-carving case study.

### q025 — 55.6% original; 28.6% proposed (4/14)
- **Original question:** What limitation exists regarding conventional CAM tools in non-standard manufacturing?
- **Gold answer:** Conventional CAM tools are tailored strictly for standard manufacturing processes rather than non-standard craft tasks.
- **Gold evidence:** “However, conventional CAM tools are tailored for standard manufacturing processes”
- **Why overlap is high:** The prompt restates the tool type and the standard/non-standard process contrast.
- **Suggested wording:** The robot-carving learning-from-demonstration paper discusses conventional CAM: which manufacturing setting are those tools built around?
- **Expected intent:** They are tailored for standard manufacturing processes, which explains the limitation for non-standard craft work.
- **Evidence check:** The quote directly states that conventional CAM tools are tailored for standard manufacturing processes, which answers the revised question without relying on an added inference.

### q027 — 50.0% original; 23.1% proposed (3/13)
- **Original question:** What physiological muscles generate grip strength when holding an object?
- **Gold answer:** Grip strength is generated by the hand and forearm muscles while gripping an object.
- **Gold evidence:** “Grip strength is a magnitude of muscle strength generated by the hand and forearm muscles while gripping an object.”
- **Why overlap is at the boundary:** Four of the eight content tokens match the evidence. Equality fails the strict below-50% requirement.
- **Suggested wording:** In "The Influence of Hand Tool Design on Hand Grip Strength: A Review," which part of the upper limb supplies the measured grip force?
- **Expected intent:** The force is generated by hand and forearm muscles.
- **Evidence check:** The quote identifies both muscle groups and explicitly ties them to gripping an object.

### q028 — 66.7% original; 33.3% proposed (3/9)
- **Original question:** How were human demonstrations of wood carving cuts recorded for the robot case study?
- **Gold answer:** Human demonstrations of different wood carving cuts were recorded via kinesthetic guidance/teaching.
- **Gold evidence:** “Human demonstrations of different wood carving cuts are recorded via kinesthetic”
- **Why overlap is high:** The question repeats the demonstration, carving/cuts, and recording concepts.
- **Suggested wording:** In the robot-carving study, how did researchers record human wood-carving demonstration motions?
- **Expected intent:** The demonstrations were captured through kinesthetic guidance/teaching.
- **Evidence check:** The quote states that demonstrations of different cuts were recorded via kinesthetic teaching.

### q029 — 80.0% original; 40.0% proposed (4/10)
- **Original question:** What reference handbook is noted for covering the mechanical and physical properties of wood in product design?
- **Gold answer:** The authoritative USDA Forest Products Laboratory handbook covers mechanical and physical properties of wood for product shape, thickness, and durability.
- **Gold evidence:** “Authoritative USDA Forest Products Laboratory handbook covering mechanical/physical properties of wood — core reference for product shape, thickness and durability Why-questions.”
- **Why overlap is high:** The prompt recycles the handbook, property, and design-purpose terms. The current evidence is a process-observation summary, so the source's bibliographic details should be verified by the reviewer before accepting a rewording.
- **Suggested wording:** In the horizontal-splitter note, what authoritative wood handbook is recommended for product-design decisions?
- **Expected intent:** Identify the USDA Forest Products Laboratory handbook on wood properties.
- **Evidence check:** The quote identifies the authoritative USDA Forest Products Laboratory handbook and says it is a core reference for shape, thickness, and durability questions.

### q030 — 55.6% original; 25.0% proposed (3/12)
- **Original question:** What mechanical property data across bamboo studies are summarized to support performance claims?
- **Gold answer:** Shear, compressive, tensile, bending, and MOE data across multiple bamboo studies are reviewed to support performance claims.
- **Gold evidence:** “Peer-reviewed thesis reviewing shear/compressive/tensile/bending/MOE data across many bamboo studies; directly supports sustainability and performance claims.”
- **Why overlap is high:** It repeats the evidence's material, data-summary, study, and performance wording.
- **Suggested wording:** In the bamboo wood-machining review, which categories of mechanical test data are synthesized to evaluate structural performance?
- **Expected intent:** Identify shear, compression, tension, bending and modulus-of-elasticity data.
- **Evidence check:** The quote lists shear, compressive, tensile, bending and MOE data and says they are reviewed across bamboo studies.

## Review disposition

All 24 questions remain in the original benchmark with their original gold answers, evidence and measurements. These are candidate revisions only. A human reviewer should verify each proposed question against the complete indexed gold record, approve or edit the wording, and authorize a separate benchmark change. The validator must continue to reject the current dataset until approved changes result in all answerable question-to-quote overlaps below 50%.
