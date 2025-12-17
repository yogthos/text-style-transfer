### STRUCTURAL REFERENCE (MATCH THIS RHYTHM)

Reference Text:
"{structure_match}"

CRITICAL: You must match these structural features:
{structure_instructions}

- Average sentence length: ~{avg_sentence_len} words
- Match the EXACT punctuation pattern (commas, dashes, semicolons, etc.)
- **PREFERENCE: Minimize em-dashes (—). Use commas, parentheses, or restructure sentences instead. Only use em-dashes if the Structural Reference explicitly requires them.**
- Match the clause structure and complexity
- Match the voice (active vs passive)
- Match the rhythm and pacing of the reference

CRITICAL PRIORITY ORDER:
1. PRESERVE ALL CONTENT: Every fact, concept, detail, and piece of information from the Input MUST appear in the output
2. MATCH STRUCTURE RHYTHM/STYLE: Match the rhythm, punctuation, syntax, and voice of the Structural Reference
3. MATCH LENGTH: Only if the Structural Reference is similar in length to the Input (within ~50% tolerance)

ELASTIC CONSTRAINT: You are allowed to contract or expand the Structural Reference to fit ALL Input Content. Match the RHYTHM, not the exact word count.
- CRITICAL: Structure must adapt to accommodate ALL content from the input
- NEVER omit content to fit structure - always expand structure to preserve all facts, concepts, and details
- If the input contains more content than the reference, you MUST expand the structure to include all content
- If the input is short and the template is complex, you may pad using vocabulary from the Situational Reference, but maintain the structural rhythm (punctuation, clauses, voice) of the reference
- The structure is flexible - the content is NOT
- If the Structural Reference is very different in length from the Input, EXPAND or CONTRACT the structure to fit all content. Do NOT omit content.

DO NOT simplify the structure. If the reference has multiple clauses, dashes, or complex punctuation, your output must too.
**PREFERENCE: Prefer commas, parentheses, or restructuring over em-dashes (—) whenever possible. Keep em-dash usage to a minimum.**

---

### SITUATIONAL REFERENCE {situation_match_label}

{situation_match_content}

---

{vocab_block}

---

### LENGTH CONSTRAINT (CRITICAL)
- Input Word Count: {input_word_count} words
- Target Output Count: ~{target_word_count} words
- You are strictly FORBIDDEN from expanding a single sentence into a full paragraph.
- Maintain a 1:1 sentence mapping where possible.
- Do NOT add unnecessary elaboration or repetition.

---

### INPUT TEXT (RAW MEANING)
"{input_text}"

### TASK
Rewrite the 'Input Text' using the EXACT rhythm, structure, and punctuation pattern of the 'Structural Reference' and the vocabulary tone of the 'Situational Reference'.

STRUCTURE MATCHING REQUIREMENTS (MOST CRITICAL):
- Match the exact punctuation pattern from the Structural Reference (commas, dashes, semicolons, parentheses, asterisks)
- Match the clause count and complexity (simple, compound, complex)
- Match the voice (active vs passive)
- Match the rhythm and pacing (within ~20% word count tolerance is acceptable - focus on rhythm, not exact count)
- **PREFERENCE: Avoid em-dashes (—) whenever possible. Use commas, parentheses, or restructure sentences instead. Only use em-dashes if the Structural Reference explicitly uses them and matching the exact punctuation pattern is critical.**
- If the Structural Reference uses dashes, you MAY use dashes to match the pattern, but prefer alternative punctuation (commas, parentheses) when possible
- If the Structural Reference uses semicolons, you MUST use semicolons
- If the Structural Reference has parenthetical elements, you MUST include similar structure
- DO NOT simplify: if the reference is complex, your output must be complex too
- ACCORDION RULE: If input is short and template is long, you may expand using Situational Reference vocabulary while maintaining the structural rhythm

CRITICAL CONSTRAINTS (NON-NEGOTIABLE):
- PRESERVE ALL CONTENT: Every fact, concept, detail, and piece of information from the input MUST appear in the output
- DO NOT omit facts, concepts, or details to match structure - expand the structure to fit all content instead
- If the input contains multiple facts (e.g., "biological cycle", "stars", "logical trap"), ALL must appear in output
- If the input explains a concept (e.g., "container problem", "fractal model"), the explanation must be preserved
- If the input mentions entities, relationships, or explanations, ALL must be preserved
- DO NOT simplify or summarize - preserve the full content
- DO NOT truncate or shorten content to match structure length
- Structure should adapt to fit ALL content, never omit content to fit structure
- DO NOT add any new entities, locations, facts, people, or information not in the original
- DO NOT invent names, places, dates, or events
- Only use words and concepts that exist in the original text
- Preserve the EXACT meaning from the original
- **CRITICAL: Do NOT copy proper nouns, names, or entities (like 'August', 'Schneider', 'Tony Febbo') from the Structural Reference**
- **Copy ONLY the rhythm, punctuation, and syntax from the Structural Reference, NOT the content words or names**
- **If the Structural Reference starts with a name followed by a colon (e.g. 'Person: ...'), IGNORE the name and colon - they are dialogue tags, not content**

ABSOLUTE PRESERVATION REQUIREMENTS (MANDATORY):
- ALL citation references in the format [^number] (e.g., [^155], [^25]) MUST be preserved EXACTLY as they appear in the original text
- ALL direct quotations (text enclosed in quotation marks) MUST be preserved EXACTLY as they appear in the original text
- DO NOT modify, remove, or relocate any [^number] style references
- DO NOT modify, paraphrase, or change any quoted text
- These elements are non-negotiable and must appear in your output unchanged

OUTPUT:

