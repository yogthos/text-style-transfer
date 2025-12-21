# Task: Content Distribution

## Neutral Content:
{neutral_text}

## Structure Slots:
{slot_descriptions}

## Instructions:
Distribute the content above into {num_slots} slots. Each slot has a target word count.

**CRITICAL CONSTRAINTS:**
1. **Distinct Information Only**: Each slot must contain DISTINCT information. Do NOT repeat the same fact in multiple slots.
2. **No Content Stretching**: If you run out of distinct facts, mark remaining slots as `EMPTY` (one per line).
3. **Content Density**: Prefer fewer, information-rich sentences over many repetitive sentences.

Output only the content for each slot, one per line. Use `EMPTY` for slots without distinct content.
Do not include slot numbers or labels.
