You are a master literary mimic and ghostwriter. Your goal is to generate a high-volume batch of sentences that fit a specific Structural Skeleton while conveying a specific Meaning.

### INPUTS
1. **Meaning (Blueprint):**
   - Subjects: {subjects}
   - Verbs: {verbs}
   - Objects: {objects}
   - Core Message: "{original_text}"

2. **Structure (Skeleton):**
   - {skeleton}
   - *Constraint:* You must match this grammatical structure (clauses, connectors, rhythm).

3. **Style Palette:**
   - Vocabulary to Integrate: {style_lexicon}

### MANDATORY KEYWORD CHECKLIST
**CRITICAL:** You MUST include the following words (or direct synonyms) in every single variation:
{keywords}

**Failure Condition:** If a variation does not contain the core Subject and Verb of the input, it is a FAILURE. Do not output it.

### CONCEPT MAPPING STEP
**Before writing, mentally map:**
* Input Subject "{subject}" -> Skeleton Slot [NP]
* Input Verb "{verb}" -> Skeleton Slot [VP]
* Input Objects "{objects}" -> Skeleton Slots [NP]

**CRITICAL:** You MUST use these mapped concepts. Do NOT replace "Human experience" with "Theory of knowledge" or "reinforces" with "establishes" unless the meaning is preserved.

### TASK
Generate **20 distinct variations** of the sentence. You must vary your approach to find the best fit:

- **Candidates 1-5 (Strict Adherence):** Map the meaning directly into the skeleton slots. Minimal embellishment. **MUST use the mapped Subject and Verb.**
- **Candidates 6-10 (Style Heavy):** Aggressively use the 'Style Palette'. Replace neutral words with author-specific jargon (e.g., change "breaks" to "succumbs to internal contradictions"). **BUT preserve the mapped Subject and Verb.**
- **Candidates 11-15 (Logical Flow):** Prioritize the logic. Ensure the causal relationship (e.g., "Because X, Y") makes perfect sense. Fix any logic gaps in the skeleton. **MUST maintain the mapped concepts.**
- **Candidates 16-20 (Expansive):** If the input is short and the skeleton is long, add meaningful adjectives and adverbs to fill the rhythm naturally. Do not leave the sentence feeling "thin." **MUST start with the mapped Subject.**

### CRITICAL RULES
1. **NO Placeholders:** Never output brackets like `[NP]` or `[VP]`. Fill them with real words.
2. **NO Logic Inversion:** Do not say "Stars erode the sky" if the input is "Stars succumb to erosion."
3. **NO Content Drift:** Do NOT replace the input Subject/Verb with unrelated concepts. "Human experience" must remain "Human experience" (or direct synonym like "Mankind's experience").
4. **JSON Format:** Output PURE JSON. A single list of strings.

### OUTPUT FORMAT
[
  "Variation 1 text...",
  "Variation 2 text...",
  ...
  "Variation 20 text..."
]

