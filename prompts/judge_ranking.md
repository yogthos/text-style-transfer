### TASK: Rank Translations
You are an Expert Editor. You have draft translations of the same source text.

**Source Text:** "{source_text}"

**Candidates:**
A: "{candidate_a}"
B: "{candidate_b}"
C: "{candidate_c}"

**Note:** Some candidates may be empty. Only rank the non-empty candidates.

**Target Style:** {style_dna}
**Rhetorical Type:** {rhetorical_type}

**Evaluation Criteria (in order of importance):**
1. **Accuracy:** Must preserve the core meaning of Source Text. All key concepts, subjects, objects, and nuances must be present.
2. **Fluency:** Must sound like natural, sophisticated English. Reject stilted phrasing (e.g., "must, in its turn, submit" → prefer "eventually succumbs"). Avoid awkward constructions, interrupted verb phrases, or unnatural word order.
3. **Style:** Must match the requested tone and rhetorical type. The phrasing should sound authoritative, philosophical, revolutionary, etc. as appropriate.

**Guidelines:**
- Prefer complete, fluent sentences over fragments or awkward constructions.
- Prefer natural phrasing over literal translations that sound foreign.
- Prefer concise, impactful language over verbose or stilted constructions.
- If a candidate is missing key concepts from the Source Text, reject it.
- If a candidate sounds like a bad translation, reject it in favor of natural English.

**Output:**
Return ONLY the letter of the Best Candidate (A, B, or C).
If all candidates are unacceptable (missing meaning or severely flawed), return "NONE".

