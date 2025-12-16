**EXAMPLE PARAGRAPHS FROM TARGET STYLE (for rhythm and structure reference only):**
{examples}

**MARKOV TEMPLATE - SENTENCE LENGTH PATTERN TO FOLLOW:**
{markov_template}

**INPUT PARAGRAPH TO RESTRUCTURE:**
{input_content}

**COHERENCE REQUIREMENTS (NON-NEGOTIABLE):**
Narrative flow and logical relationships MUST be preserved. These rules OVERRIDE sentence length targets.

1. **NEVER use "which" clauses:** This is the #1 AI detection signal. Keep sentences separate.
2. **Subject-predicate integrity:** Each sentence's subject MUST stay with its predicate. "That country is a ghost" CANNOT become a modifier attached to "ruins" - they have different subjects.
3. **Preserve short sentences:** Punchy rhythm is HUMAN. Don't merge everything.
4. **Narrative flow trumps metrics:** If varying sentence length would break coherence, DON'T vary it
5. **Context before action:** Time/place/age sentences MUST precede the actions they describe
6. **Cause before effect:** Keep logical order intact
7. **Clear references:** Pronouns must follow their referents

**FORBIDDEN PATTERNS (examples of what NOT to do):**

BAD: "Every morning, which required a pilgrimage, began when I was thirteen"
GOOD: "When I was thirteen, every morning required a pilgrimage"
(Time context must come first)

BAD: "One memory defines that era. The silence is that of a paralyzed economy."
GOOD: "One memory defines that era: the silence of a paralyzed economy."
(The two ideas must connect)

BAD: "The system, which crashed, was built by engineers; they worked hard"
GOOD: "Engineers built the system. They worked hard, but it crashed."
(Don't force unrelated clauses together)

BAD: "From the ruins of the Soviet Union, a ghost now that haunts history, I spent my childhood"
GOOD: "I spent my childhood scavenging in the ruins of the Soviet Union. That country is a ghost now."
(CRITICAL: "country" is the subject of "is a ghost" - you cannot move that predicate to modify "ruins")

BAD: "Human experience, which is defined by the biological cycle, reinforces the rule of finitude"
GOOD: "Human experience reinforces the rule of finitude. The biological cycle defines our reality."
(CRITICAL: "which" clause insertion is the #1 AI detection signal - NEVER use it)

**TASK:** Restructure the input paragraph to vary sentence lengths as specified in the Markov template above. Your ONLY job is to change sentence boundaries and punctuation - NOT to change words. If meeting length targets would break narrative coherence, preserve coherence instead.

**CRITICAL: PRESERVE THE ORIGINAL VOICE**
- This is a personal narrative. Keep it personal.
- Do NOT inject academic vocabulary, philosophical terms, or formal language.
- The author's voice and word choices are deliberate - preserve them exactly.
- If the input says "ghost" keep "ghost", not "specter" or "apparition"
- If the input says "ruins" keep "ruins", not "remnants" or "detritus"

**WHAT YOU MUST DO:**
1. VARY SENTENCE LENGTHS: Mix short (5-10 words), medium (15-25 words), and long (30+ words) sentences
2. VARY SENTENCE STARTERS: Don't start multiple sentences with the same word
3. USE ONLY THE ORIGINAL WORDS: 95%+ of input words must appear unchanged
4. PRESERVE ALL: Citations [^N], proper nouns, numbers, technical terms

**WHAT YOU MUST NOT DO:**
- DO NOT add new vocabulary or academic terms
- DO NOT paraphrase or use synonyms
- DO NOT change word forms ("defines" cannot become "defining")
- DO NOT use em-dashes (—) or double hyphens (--)
- DO NOT add philosophical jargon (process, development, contradiction, principal, aspect, concrete, revolutionary)
- DO NOT use "which" or "that" clauses to merge sentences - THIS IS THE #1 AI DETECTION SIGNAL
- DO NOT over-merge short sentences - punchy rhythm is human
- DO NOT output any analysis, headers, or explanations

**HOW TO RESTRUCTURE:**
- Keep short sentences SHORT: "Every object breaks. Every star dies." is GOOD human rhythm
- Combine ONLY related ideas with semicolons: "The sky was dark; rain fell." (same moment)
- Split long sentences: Use periods or semicolons to create natural breaks
- Reorder clauses: Move phrases to different positions within the sentence
- NEVER use "which" to merge: Keep sentences separate instead

**OUTPUT:** The restructured paragraph only. No commentary.
