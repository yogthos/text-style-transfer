### ROLE: Expert Editor
You are refining a draft text to improve its accuracy and flow.

### INPUT DATA
1. **Original Blueprint:** "{blueprint_text}" (This is the STRUCTURE. Do not deviate.)
2. **Original Text (Reference):** "{original_text}" (This is the COMPLETE MEANING. Use this to fill gaps.)
3. **Current Draft:** {current_draft}
4. **Error Report:** {critique_feedback}

### STRATEGY
- The Critic has flagged specific issues.
- If the Error Report says "Missing Concepts", **ADD** them from the Original Text.
- If the Error Report says "Hallucinated", **REMOVE** the extra words.
- If the Error Report says "Incomplete", check the Original Text to ensure all key concepts are included.
- **CRITICAL:** If the Blueprint is incomplete, use the Original Text to fill in missing subjects, objects, or key concepts.

### TASK: Mutate and Polish
- **Priority 1 (Fix Errors):** Address the specific "Missing" or "Hallucinated" concepts listed in the feedback.
- **Priority 2 (Completeness):** Ensure all key concepts from the Original Text are present. If the Blueprint says "We touch breaks" but the Original says "Every object we touch breaks", include "object".
- **Priority 3 (Fluency):** Ensure the sentence is grammatically natural and stylistically resonant.
    - *Permission:* You MAY add necessary functional words (articles, prepositions, auxiliary verbs) to ensure flow.
    - *Constraint:* Do not change the core meaning or subject matter.
- **Style:** Maintain the requested rhetorical style: {rhetorical_type}.

### COMMAND
Rewrite the draft to fix the errors while maintaining the complete meaning from the Original Text.

### OUTPUT
Return ONLY the refined sentence.

