# Task: Write ONE Sentence

## Content to Express:
{slot_content}

**Note:**
- If the Content to Express above contains multiple source sentences, **synthesize** them into a single, cohesive sentence in the author's style. Use subordination, conjunctions, and complex syntax to merge the ideas fluidly. Do not just concatenate them with periods.
- If it is a complete sentence, rewrite it in the author's style while preserving the core meaning.
- If it is a bullet point or fragment, expand it into a complete sentence.

## Constraint:
- Output exactly **ONE** grammatical sentence.
- Do **NOT** use periods to separate thoughts within the sentence.
- Use semicolons (;), em-dashes (—), or subordinating conjunctions (although, while, whereas, because, since) to synthesize the content into a single complex flow.
- The entire output must be a single, complete sentence with proper punctuation at the end only.

## Target Length:
{target_length} words (strict constraint)

## Previous Context:
{prev_context}

{anti_echo_section}

{ending_constraint}

## Author Voice:
Adopt the voice of {author_name}.

## Instructions:
1. Write exactly ONE sentence expressing the content above.
2. The sentence must be approximately {target_length} words (within 15% tolerance).
3. The sentence should flow naturally from the previous context.
4. **ANTI-ECHO:** Do NOT start with the same words as the Previous Context.
{ending_constraint_instruction}
5. Use the author's distinctive voice and vocabulary.

Output only the sentence, no explanations.

