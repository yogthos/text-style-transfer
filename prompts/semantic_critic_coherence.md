You are checking for **Factual Contradictions**, not just differences.

**Rules:**
1. If the Output *contradicts* the Input → FAIL (e.g., "finite" vs "infinite", "exists" vs "does not exist")
2. If the Output adds philosophical framing that doesn't contradict → PASS
3. Style expansion is allowed if it doesn't change core facts

**Example:**
- Input: 'Social structures are vapor.'
- Output: 'Social structures are vapor before the material forces of history.'
- Evaluation: Does Output contradict Input? NO. Does it add framing? YES.
- Verdict: PASS (Style expansion is allowed).

**Additional Checks:**
1. Is the text grammatically fluent?
2. Does it make logical sense, or is it 'word salad'?
3. Does it contain random technical jargon (e.g., 'turing complete', 'namespace', 'dependency injection') that doesn't fit the narrative?
4. **CRITICAL:** Does the text insert abstract academic jargon (e.g., 'emergent complexity', 'systems theory', 'dialectical materialism') into a simple narrative? If yes, mark as Incoherent.
5. **CRITICAL - Logical Category Errors:** Check for Logical Category Errors.
   - Example: 'The silence is a practice.' → FAIL (Silence is a state, not an action).
   - Example: 'The economy is a ghost.' → PASS (Metaphor is okay).
   - **IMPORTANT:** Be lenient with Metaphorical Language. If the author's style is abstract/philosophical (e.g., Mao, Hegel), allow abstract subjects to take abstract actions. For example, "Silence defines the era" or "Memory forces an action" are acceptable in philosophical contexts. Only fail if the text is completely unintelligible or nonsensical.
   - Rule: If the text forces a Subject into a Definition that is logically impossible (not just metaphorical), mark it as Incoherent. However, distinguish between "stylistically complex" (acceptable for philosophical authors) and "genuinely incoherent" (reject).

**Output JSON:** {'is_coherent': bool, 'score': float (0.0-1.0), 'reason': '...'}

