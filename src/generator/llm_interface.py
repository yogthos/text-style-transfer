"""LLM interface for constrained sentence generation.

This module handles communication with LLM APIs (DeepSeek) to generate
sentences that preserve semantic meaning while following specific
structural templates and vocabulary requirements.
"""

import re
from typing import List, Optional

from src.models import ContentUnit
from src.generator.prompt_builder import PromptAssembler
from src.generator.llm_provider import LLMProvider
from src.analyzer.style_metrics import get_style_vector


def clean_generated_text(text: str) -> str:
    """Deterministic cleanup of common LLM artifacts.

    Protects citations [^number] and quotes from being modified by cleanup operations.
    Also removes prompt leakage, instruction lines, and hallucinated citation formats.
    """
    if not text: return ""

    # NEW: Extract text after "REWRITE:" marker (if present)
    if "REWRITE:" in text.upper() or "REWRITE" in text.upper():
        # Try to find text after REWRITE: marker
        parts = re.split(r'REWRITE:?\s*', text, flags=re.IGNORECASE)
        if len(parts) > 1:
            # Take the last part (after the last REWRITE marker)
            text = parts[-1].strip()
        # Also handle "Rewrite:" with lowercase
        parts = re.split(r'Rewrite:?\s*', text, flags=re.IGNORECASE)
        if len(parts) > 1:
            text = parts[-1].strip()

    # NEW: Remove prompt instruction lines
    lines = text.split('\n')
    filtered_lines = []
    skip_keywords = [
        "CRITICAL:", "REWRITE:", "INPUT TEXT:", "TASK:", "INSTRUCTIONS:",
        "Subjects:", "Actions:", "Objects:", "Entities:", "Keywords:",
        "Citations:", "Quotes:", "Citations to include:", "Quotes to preserve:",
        "Citations to preserve:", "Direct quotes to preserve exactly:"
    ]
    for line in lines:
        line_stripped = line.strip()
        # Skip lines that start with any of the instruction keywords
        if any(line_stripped.startswith(kw) for kw in skip_keywords):
            continue
        # Skip lines that are just separators or empty
        if line_stripped in ["===", "---", ""]:
            continue
        filtered_lines.append(line)
    text = '\n'.join(filtered_lines)

    # NEW: Remove academic citation patterns BEFORE protecting valid citations
    # Remove (Author, Year, p. #) format
    text = re.sub(r'\([A-Z][a-z]+,?\s+\d{4}(?:,\s*p\.\s*#?)?\)', '', text)
    # Remove (Author, Year, p. #) template pattern
    text = re.sub(r'\(Author,?\s+Year,?\s+p\.\s*#\)', '', text, flags=re.IGNORECASE)
    # Remove (Smith 42) format
    text = re.sub(r'\([A-Z][a-z]+\s+\d+\)', '', text)

    # Protect citations and quotes by temporarily replacing them with placeholders
    citation_pattern = r'\[\^\d+\]'
    quote_pattern = r'["\'](?:[^"\']|(?<=\\)["\'])*["\']'

    # Store protected content
    protected_items = []
    placeholder_map = {}

    # Replace citations with placeholders
    def replace_citation(match):
        placeholder = f"__CITATION_{len(protected_items)}__"
        protected_items.append(match.group(0))
        placeholder_map[placeholder] = match.group(0)
        return placeholder

    text = re.sub(citation_pattern, replace_citation, text)

    # Replace quotes with placeholders
    def replace_quote(match):
        quote_text = match.group(0)
        # Only protect substantial quotes
        if len(quote_text.strip('"\'')) > 2:
            placeholder = f"__QUOTE_{len(protected_items)}__"
            protected_items.append(quote_text)
            placeholder_map[placeholder] = quote_text
            return placeholder
        return quote_text

    text = re.sub(quote_pattern, replace_quote, text)

    # 1. Fix Punctuation Spacing (" . " -> ". ")
    # But avoid affecting protected placeholders
    text = re.sub(r'\s+([.,!?;:])', r'\1', text)

    # 2. Fix Multiple Periods (".." -> ".") excluding ellipses
    text = re.sub(r'\.{2,}', '...', text)
    text = re.sub(r'\.\.\.(?!\.)', '...', text) # Ensure consistent ellipses

    # 3. Capitalization (Start of sentence)
    # But avoid capitalizing inside protected placeholders
    def capitalize_match(m):
        return f"{m.group(1)}{m.group(2).upper()}"
    text = re.sub(r'(^|[.!?]\s+)([a-z])', capitalize_match, text)

    # 4. Strip Metadata Artifacts (e.g., "Chapter 1")
    text = re.sub(r'(?i)\b(chapter|section|part)\s+([0-9]+|[ivx]+)\b', '', text)

    # 5. Remove "Output:" prefixes
    text = re.sub(r'^(output|response|rewritten):\s*', '', text, flags=re.IGNORECASE)

    # Restore protected citations and quotes
    for placeholder, original in placeholder_map.items():
        text = text.replace(placeholder, original)

    return text.strip()


def generate_sentence(
    content_unit: ContentUnit,
    structure_match: str,
    situation_match: Optional[str] = None,
    config_path: str = "config.json",
    hint: Optional[str] = None,
    target_author_name: str = "Target Author",
    global_vocab_list: Optional[List[str]] = None,
    author_names: Optional[List[str]] = None,
    blend_ratio: Optional[float] = None,
    use_fallback_structure: bool = False,
    constraint_mode: str = "STRICT",
    style_dna_dict: Optional[dict] = None
) -> str:
    """Generate a sentence using LLM with dual RAG references.

    Generates a sentence that:
    - Preserves the EXACT semantic meaning from content_unit
    - Matches the sentence structure and rhythm of structure_match
    - Uses vocabulary tone from situation_match (if available)

    Args:
        content_unit: ContentUnit containing SVO triples, entities, and original text.
        structure_match: Reference paragraph for rhythm/structure (required).
        situation_match: Reference paragraph for vocabulary grounding (optional).
        config_path: Path to configuration file.
        hint: Optional hint/feedback from previous attempt to improve generation.
        target_author_name: Name of target author for persona (default: "Target Author").
        global_vocab_list: Optional list of global vocabulary words to inject for variety.

    Returns:
        Generated sentence string.
    """
    # Initialize LLM provider
    llm = LLMProvider(config_path=config_path)

    # Extract Style DNA for the author(s)
    style_dna = None
    style_dna_dict_for_assembler = None

    if style_dna_dict:
        if author_names and len(author_names) >= 2 and blend_ratio is not None:
            # Blend mode: pass dict with both authors
            style_dna_dict_for_assembler = {}
            for author in author_names:
                if author in style_dna_dict:
                    style_dna_dict_for_assembler[author] = style_dna_dict[author]
        else:
            # Single-author mode: extract DNA for target author
            if target_author_name in style_dna_dict:
                style_dna = style_dna_dict[target_author_name]
            # Also check author_names if provided (for single-author mode)
            elif author_names and len(author_names) > 0 and author_names[0] in style_dna_dict:
                style_dna = style_dna_dict[author_names[0]]

    # Initialize prompt assembler with Style DNA
    assembler = PromptAssembler(
        target_author_name=target_author_name,
        style_dna=style_dna,
        style_dna_dict=style_dna_dict_for_assembler
    )

    # Build system prompt using PromptAssembler
    system_prompt = assembler.build_system_message()

    # Add examples to system prompt
    from pathlib import Path
    prompts_dir = Path(__file__).parent.parent.parent / "prompts"
    examples_path = prompts_dir / "generator_examples.md"
    if examples_path.exists():
        examples = examples_path.read_text().strip()
        system_prompt += "\n\n" + examples

    # Extract style metrics from structure_match
    style_vec = get_style_vector(structure_match)
    words = structure_match.split()
    sentences = structure_match.count('.') + structure_match.count('!') + structure_match.count('?')
    if sentences > 0:
        avg_sentence_len = len(words) / sentences
    else:
        avg_sentence_len = len(words)

    style_metrics = {'avg_sentence_len': avg_sentence_len}

    # Build user prompt using PromptAssembler (blend mode or single-author mode)
    if author_names and len(author_names) >= 2 and blend_ratio is not None:
        # Blend mode: use blended prompt
        user_prompt = assembler.build_blended_prompt(
            input_text=content_unit.original_text,
            bridge_template=structure_match,
            hybrid_vocab=global_vocab_list or [],
            author_a=author_names[0],
            author_b=author_names[1],
            blend_ratio=blend_ratio,
            constraint_mode=constraint_mode
        )
    else:
        # Single-author mode: use regular prompt
        user_prompt = assembler.build_generation_prompt(
            input_text=content_unit.original_text,
            situation_match=situation_match,
            structure_match=structure_match,
            style_metrics=style_metrics,
            global_vocab_list=global_vocab_list,
            use_fallback_structure=use_fallback_structure,
            constraint_mode=constraint_mode
        )

    # Add entity preservation if needed
    if content_unit.entities:
        user_prompt += "\n\n"
        user_prompt += f"IMPORTANT: Preserve these entities exactly (DO NOT add others): {', '.join(content_unit.entities)}"

    # Add content words hint if needed
    if content_unit.content_words:
        important_words = content_unit.content_words[:20]
        user_prompt += "\n"
        user_prompt += f"Key concepts to include: {', '.join(important_words)}"

    # Add hint from previous attempt if provided (for retries)
    # Use Chain-of-Thought format for retries
    if hint:
        from pathlib import Path
        prompts_dir = Path(__file__).parent.parent.parent / "prompts"
        retry_template_path = prompts_dir / "llm_interface_retry.md"
        if retry_template_path.exists():
            retry_template = retry_template_path.read_text().strip()
            length_constraint = ""
            if "words" in hint.lower() and ("delete" in hint.lower() or "expand" in hint.lower() or "add" in hint.lower()):
                length_constraint = "\n\nCRITICAL: The length constraint above is a hard requirement. You MUST follow the exact word count instruction."
            retry_prompt = retry_template.format(
                hint=hint,
                length_constraint=length_constraint
            )
            user_prompt += "\n\n" + retry_prompt
        else:
            # Fallback to old hardcoded version if template missing
            user_prompt += "\n\n"
            user_prompt += "--- RETRY MODE: CHAIN-OF-THOUGHT CORRECTION ---\n\n"
            user_prompt += f"CRITIC FEEDBACK: {hint}\n\n"
            user_prompt += "TASK:\n"
            user_prompt += "1. Analyze WHY the previous attempt failed the critic's specific rule.\n"
            user_prompt += "2. Write a 'Plan of Correction' (1 sentence).\n"
            user_prompt += "3. Generate the final corrected text.\n\n"
            user_prompt += "Output format:\n"
            user_prompt += "PLAN: [Your reasoning]\n"
            user_prompt += "TEXT: [The corrected text]"
            if "words" in hint.lower() and ("delete" in hint.lower() or "expand" in hint.lower() or "add" in hint.lower()):
                user_prompt += "\n\nCRITICAL: The length constraint above is a hard requirement. You MUST follow the exact word count instruction."

    # Calculate adaptive max_tokens based on input length
    # Allow 3-4x input length for style transfer (structure adaptation needs space)
    input_word_count = len(content_unit.original_text.split())
    adaptive_max_tokens = max(300, input_word_count * 4)  # Minimum 300, scale with input

    # Call LLM API
    generated_text = llm.call(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        model_type="editor",
        require_json=False,
        temperature=0.6,  # Increased from 0.3 to allow creative adaptation
        max_tokens=adaptive_max_tokens,  # Use adaptive value instead of fixed 200
        top_p=0.9  # Add top_p parameter for tighter grammar control
    )

    # Parse CoT response if hint was provided (retry mode)
    if hint:
        # Try to extract text after "TEXT:" marker
        text_match = re.search(r'TEXT:\s*(.+?)(?:\n\n|$)', generated_text, re.DOTALL | re.IGNORECASE)
        if text_match:
            generated_text = text_match.group(1).strip()
        # Fallback: if no TEXT marker, try to find the last paragraph or sentence
        elif "PLAN:" in generated_text.upper():
            # Split by PLAN: and take everything after the last occurrence
            parts = re.split(r'PLAN:', generated_text, flags=re.IGNORECASE)
            if len(parts) > 1:
                # Take the last part and try to extract text
                last_part = parts[-1]
                # Remove "TEXT:" if present and take the rest
                text_cleaned = re.sub(r'^TEXT:\s*', '', last_part, flags=re.IGNORECASE).strip()
                if text_cleaned:
                    generated_text = text_cleaned

    # Clean generated text to fix common LLM artifacts
    generated_text = clean_generated_text(generated_text)

    return generated_text

