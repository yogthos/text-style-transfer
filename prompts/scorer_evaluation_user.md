Evaluate the following text transformation:

ORIGINAL TEXT:
{original_text}

GENERATED TEXT:
{generated_text}

TARGET STYLE EXAMPLES:
{style_examples_str}

Evaluate on two dimensions:

1. MEANING PRESERVATION (0.0 to 1.0): Does the generated text preserve the exact meaning of the original?
   - 1.0 = Perfect meaning preservation
   - 0.5 = Some meaning lost or changed
   - 0.0 = Completely different meaning

2. STYLE MATCH (0.0 to 1.0): Does the generated text match the writing style of the examples?
   - 1.0 = Perfect style match (reads like same author)
   - 0.5 = Partial style match
   - 0.0 = No style match (completely different style)

Provide your response in this exact format:
MEANING_SCORE: [number between 0.0 and 1.0]
STYLE_SCORE: [number between 0.0 and 1.0]
FEEDBACK: [specific feedback on what's good/bad and how to improve]

Be specific in your feedback. If style doesn't match, explain what's wrong (e.g., "too verbose", "not direct enough", "sentence structure too complex").

