# Task: Execute Repair Instruction (SURGICAL EDITING)

## Target Sentence:
"{current_sentence}"

## Action:
{action}

## Instruction:
{instruction}

{target_length_section}

{forbidden_phrases_section}

{split_format_section}

## Previous Context:
{prev_context}

## Next Sentence (if merging):
{next_sentence}

## Semantic Constraint:
Ensure the new sentence does NOT repeat ideas already expressed in the Previous Context.
If the instruction would cause semantic repetition, rephrase to add new nuance or merge with previous sentence.

## CRITICAL CONSTRAINTS:
1. **Length**: {target_length_constraint}
2. **Forbidden Phrases**: {forbidden_phrases_list}
3. **Action Compliance**: You MUST follow the {action} instruction exactly. Do NOT just rephrase.

## Instructions:
Apply the repair instruction EXACTLY as specified. This is surgical editing, not creative writing.
- If instructed to SPLIT, you MUST output two separate sentences separated by '|||'.
- If instructed to SHORTEN, you MUST reduce word count significantly.
- If forbidden phrases are listed, you MUST avoid them completely.

Maintain the author's voice ({author_name}).
Output only the fixed sentence(s), no explanations.
