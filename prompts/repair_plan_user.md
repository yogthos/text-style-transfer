# Task: Generate Repair Plan

**CRITICAL: YOU MUST OUTPUT VALID JSON ONLY. NO MARKDOWN, NO EXPLANATIONS, NO TEXT BEFORE OR AFTER THE JSON.**

## Draft Paragraph:
{draft}

## Original Blueprint (Structure Map):
{structure_map_info}

## Feedback:
{feedback}

## Instructions:
The original plan required: {structure_map_info}
The current draft has errors: {feedback}
Create a repair plan to match the original structure map.

Analyze the draft for flow, coherence, and natural transitions. Compare the draft against the blueprint targets. Generate a JSON repair plan with specific instructions for each sentence that needs fixing.

**OUTPUT FORMAT REQUIREMENTS:**
1. **MUST be valid JSON** - Use double quotes for ALL keys and string values
2. **MUST be a JSON array** - Start with `[` and end with `]`
3. **NO single quotes** - Use `"key"` not `'key'`
4. **NO markdown code blocks** - Do NOT wrap in ```json or ```
5. **NO explanatory text** - Output ONLY the JSON array, nothing else

Each instruction in the array MUST have these exact fields:
- **"sent_index"**: Integer (1-based: 1 = first sentence, 2 = second sentence, etc.)
- **"action"**: String, one of: "shorten", "lengthen", "merge_with_next", "split", "rewrite_connector", "add_transition", "simplify", "expand"
- **"instruction"**: String, specific instruction for the LLM (e.g., "Shorten to 25 words by removing adjectives", "Merge with next sentence using 'however'")
- **"target_len"**: Integer (optional), target word count if action is "shorten" or "lengthen"

Focus on:
- Sentence length compliance (if blueprint targets are specified)
- Smooth transitions between sentences
- Natural flow and coherence
- Maintaining the author's voice ({author_name})
- Fixing any choppy or disjointed sections

**CORRECT OUTPUT FORMAT (copy this structure exactly):**
[
  {{"sent_index": 1, "action": "add_transition", "instruction": "Add a transition word at the start to connect with previous context"}},
  {{"sent_index": 2, "action": "shorten", "target_len": 25, "instruction": "Shorten to 25 words by removing unnecessary adjectives"}},
  {{"sent_index": 3, "action": "merge_with_next", "instruction": "Merge with sentence 4 using 'however' to create smoother flow"}}
]

**REMEMBER: Output ONLY the JSON array. No markdown, no explanations, no other text.**
