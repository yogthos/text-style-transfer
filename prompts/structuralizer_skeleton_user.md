Extract the structural skeleton by replacing ALL specific nouns, verbs, and adjectives with generic placeholders.

**CRITICAL RULES (GHOST WORD BAN):**
- You MUST replace ALL specific nouns, verbs, and adjectives with placeholders ([NP], [VP], [ADJ])
- Do NOT leave any specific content words like 'theory', 'knowledge', 'standpoint', 'practice', 'ideas', 'skies', 'mind', etc.
- Replace EVERY content word, no exceptions
- If you see words like 'ideas', 'skies', 'mind', 'theory', 'knowledge', 'correct', 'innate', they MUST become [NP] or [ADJ] or [VP]
- Only keep strictly rhetorical connectors (but, however, thus, therefore, yet, etc.) and functional grammar words

**KEEP (Functional Grammar Only):**
- Prepositions: of, in, to, for, with, by, from, at, on, etc.
- Conjunctions: and, but, or, if, when, while, etc.
- Determiners: the, a, an, this, that, these, those
- Auxiliary verbs: is, are, was, were, has, have, had, will, would, could, should, may, might, must, can
- Rhetorical connectors: but, however, thus, therefore, yet, conversely, nevertheless, nonetheless, moreover, furthermore, consequently, hence, accordingly, indeed, in fact, specifically, notably, significantly, importantly, crucially, meanwhile, alternatively, additionally, similarly, likewise, instead

**REPLACE (ALL Content Words - NO EXCEPTIONS):**
- ALL Nouns → [NP] (e.g., 'theory', 'knowledge', 'standpoint', 'practice', 'ideas', 'skies', 'mind' → [NP])
- ALL Verbs → [VP] (e.g., 'reinforce', 'affirm', 'serve', 'come', 'drop' → [VP])
- ALL Adjectives → [ADJ] (e.g., 'primary', 'objective', 'dialectical', 'correct', 'innate' → [ADJ])

**Example:**
Input: "The standpoint of practice is the primary standpoint."
Output: "The [NP] of [NP] is the [ADJ] [NP]."
(NOT "The [NP] of practice..." - you must replace ALL content words)

Input: "Where do correct ideas come from? Do they drop from the skies? Are they innate in the mind?"
Output: "Where [VP] [ADJ] [NP] [VP] from? [VP] [NP] [VP] from the [NP]? [VP] [NP] [ADJ] in the [NP]?"
(NOT "Where do correct ideas come from? Do they drop from the skies?" - replace ALL content words including 'ideas', 'skies', 'mind', 'correct', 'innate')

Input: "{text}"

Output ONLY the skeleton with placeholders, no explanations:

