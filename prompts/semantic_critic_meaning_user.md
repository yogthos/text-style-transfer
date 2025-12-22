Task 1: Verify factual recall.
Compare these two sentences and determine if the generated text preserves the core meaning of the original.

Original: "{original_text}"

Generated: "{generated_text}"

Does the generated text preserve the core meaning of the original? Pay attention to:
- Logical relationships: Does it add conditions (e.g., "only when", "if... then") that weren't in the original?
- Meaning shifts: Does it change a universal statement into a conditional, or vice versa?
- Contradictions: Does it imply something that contradicts the original meaning?

{global_context_section}

Task 2: Verify argumentative stance consistency.
Does the Generated text maintain the same argumentative stance (Agree/Disagree) as the Source regarding the main topic?
- If the Source criticizes a position or figure, does the Generated text also criticize it (or at least not endorse it)?
- If the Source supports a position, does the Generated text also support it?
- **CRITICAL:** If the Source criticizes X but the Generated text endorses X, this is a STANCE INVERSION (stance_score < 0.5)

{intent_task}

Respond with JSON:
{{
    "meaning_preserved": true/false,
    "confidence": 0.0-1.0,
    "stance_score": 0.0-1.0,
    {intent_score_field}
    "explanation": "brief reason (mention if conditional relationship, logic mismatch, meaning shift, stance inversion, or intent mismatch detected)"
}}

**Note:** stance_score < 0.5 indicates stance inversion (criticism became endorsement or vice versa) - this is a CRITICAL FAILURE.

