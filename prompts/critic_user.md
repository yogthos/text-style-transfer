Compare the generated text against these references using the HIERARCHY OF RULES:

{structure_section}{situation_section}{original_section}

GENERATED TEXT (to evaluate):
"{generated_text}"

--- TASK ---
Evaluate the GENERATED TEXT against the HIERARCHY:
1. GRAMMAR AND READABILITY: Is the text grammatically correct and readable? (NON-NEGOTIABLE)
   - Check for: proper sentence structure, complete sentences, readable phrasing
   - **CRITICAL**: The text must make semantic sense, not just be grammatically correct
   - **CRITICAL**: Title Case sentences (e.g., "The Human View Of Discrete Levels Scale...") are FORBIDDEN - they are word salad, not sentences
   - Check for incomplete thoughts, dangling clauses, and nonsensical constructions
   - Examples of semantic failures: "The code, even though it is embedded in every particle and field." (incomplete - missing main clause), "limits, even though they are only implied by an exterior." (incomplete fragment), "The Human View of Discrete Levels Scale as a Local Perspective Artifact Observe the Mandelbrot set." (word salad - jumble of keywords)
   - **CRITICAL: Logical Contradictions:** Check if adjectives/adverbs contradict their nouns (e.g., "infinite boundary", "ceaseless limit", "ceaseless finitude").
     - If found, this is a CRITICAL FAILURE - mark "pass": false, "score": 0.0, "primary_failure_type": "grammar"
     - Examples to flag: "ceaseless finitude" (ceaseless = unending, finitude = having limits), "infinite boundary" (infinite = unlimited, boundary = limit), "eternal decay" (eternal = permanent, decay = deterioration)
   - **CRITICAL**: If the Structural Reference uses em-dashes (—), colons (:), or other punctuation, and the Generated Text matches it, this is VALID STYLE, NOT a grammar error.
   - **NEVER flag punctuation as grammar error if it matches the Structural Reference.**
   - If grammar is broken, text is unreadable, OR text is semantically incomplete/nonsensical (AND it doesn't match the Structural Reference), this is a CRITICAL FAILURE - mark "pass": false, "score": 0.0, "primary_failure_type": "grammar"
2. SEMANTIC SAFETY: Does it preserve the original meaning? (Highest Priority - NON-NEGOTIABLE)
   - **CRITICAL**: Meaning preservation is NON-NEGOTIABLE. The generated text MUST preserve the original meaning completely
   - **CRITICAL**: You MUST NOT accept text that is grammatically correct but semantically nonsensical
   - **CRITICAL**: You MUST iterate until BOTH conditions are met: (1) Meaning is preserved, (2) Grammar is logical
   - **CRITICAL**: Do NOT move on to style matching until meaning is preserved
   - CRITICAL: Check that ALL facts, concepts, details, and information from the Original Text are present in the Generated Text
   - CRITICAL: Check that NO proper nouns, names, or specific entities appear in Generated Text that do NOT appear in Original Text
   - IMPORTANT: Allow synonyms and paraphrases (e.g., 'essential' for 'important', 'reinforces' for 'confirms', 'confirms' for 'validates')
   - CRITICAL RULE: ONLY flag words that are CAPITALIZED in the MIDDLE of a sentence (not at sentence start)
   - DO NOT flag lowercase words like "essential", "land", "necessary", "ingredient" - these are NEVER proper nouns
   - DO NOT flag phrases like "essential ingredient" - these are NEVER entities
   - Extract only CAPITALIZED words from the MIDDLE of sentences (not sentence starts) from Generated Text
   - For each capitalized word (mid-sentence) in Generated Text, verify it appears in Original Text (case-insensitive)
   - Examples of what to FLAG: "Schneider" (capitalized, mid-sentence, not in original), "Einstein" (capitalized, mid-sentence, not in original)
   - Examples of what NOT to flag: "essential" (lowercase), "land" (lowercase), "essential ingredient" (phrase with lowercase words), "Human" (at sentence start)
   - If any proper nouns, names, or entities in Generated Text are NOT in Original Text, this is a CRITICAL FAILURE - mark "pass": false, "score": 0.0, "primary_failure_type": "meaning"
   - If the Original Text contains multiple facts/concepts, verify ALL are present
   - If any facts, concepts, or details are missing, this is a CRITICAL FAILURE - mark "pass": false, "score": 0.0, "primary_failure_type": "meaning"
   - Examples: If original mentions "biological cycle", "stars", "logical trap", "container problem", "fractal model", "Mandelbrot set" - ALL must appear in output
   - **CRITICAL - LIST PRESERVATION:** If the original contains lists (e.g., "birth, life, and decay"), ALL items in the list must appear in the output. Do NOT accept output that only includes some items (e.g., "birth" alone or "birth and decay" without "life"). Every item in every list must be present. Missing any item from a list is a CRITICAL FAILURE.
3. STRUCTURAL RIGIDITY: Does it match the syntax/length/punctuation of the STRUCTURAL REFERENCE? (Second Priority)
4. VOCABULARY FLAVOR: Does it use the word choices/tone of the SITUATIONAL REFERENCE? (Third Priority)

If it fails, provide ONE single, specific instruction to fix the biggest error.
Do not list multiple conflicting errors. Pick the one that violates the highest priority rule.

Format your feedback as a direct editing instruction with specific metrics when relevant.
Example: "Current text has 25 words; Target has 12. Delete adjectives and split the relative clause."

For grammar failures: "CRITICAL: Text contains grammatical errors. [specific error]. Rewrite with proper grammar."
For logical contradictions: "CRITICAL: Logical contradiction found: [Adjective] contradicts [Noun]. Use a modifier that aligns with the noun's definition."
For hallucinated proper nouns/entities: "CRITICAL: Text contains proper noun/entity '[word]' that does not appear in original. Remove all proper nouns and entities not present in original text."
For meaning loss: "CRITICAL: Text omits [specific concept/fact] from original. Include all concepts from original text."

{preservation_checks}

OUTPUT JSON with "pass", "feedback", "score", and "primary_failure_type" fields.
{preservation_instruction}
