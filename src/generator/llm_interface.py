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
        # If hint contains length information, emphasize it
        if "words" in hint.lower() and ("delete" in hint.lower() or "expand" in hint.lower() or "add" in hint.lower()):
            user_prompt += "\n\nCRITICAL: The length constraint above is a hard requirement. You MUST follow the exact word count instruction."

    # Call LLM API
    generated_text = llm.call(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        model_type="editor",
        require_json=False,
        temperature=0.1,  # Very low temperature for precise structure matching
        max_tokens=200
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

    return generated_text

