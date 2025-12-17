"""LLM interface for constrained sentence generation.

This module handles communication with LLM APIs (DeepSeek) to generate
sentences that preserve semantic meaning while following specific
structural templates and vocabulary requirements.
"""

import json
import os
from typing import Dict, List, Optional
import requests

from src.models import ContentUnit


def _load_config(config_path: str = "config.json") -> Dict:
    """Load configuration from config.json.

    Args:
        config_path: Path to the config file.

    Returns:
        Configuration dictionary.
    """
    with open(config_path, 'r') as f:
        return json.load(f)


def _call_deepseek_api(
    system_prompt: str,
    user_prompt: str,
    api_key: str,
    api_url: str,
    model: str = "deepseek-chat"
) -> str:
    """Call DeepSeek API to generate text.

    Args:
        system_prompt: System prompt for the LLM.
        user_prompt: User prompt with the request.
        api_key: DeepSeek API key.
        api_url: DeepSeek API URL.
        model: Model name to use.

    Returns:
        Generated text response.
    """
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }

    data = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        "temperature": 0.3,  # Lower temperature for more deterministic, less creative output
        "max_tokens": 200
    }

    try:
        response = requests.post(api_url, headers=headers, json=data, timeout=30)
        response.raise_for_status()
        result = response.json()

        if "choices" in result and len(result["choices"]) > 0:
            return result["choices"][0]["message"]["content"].strip()
        else:
            raise ValueError(f"Unexpected API response: {result}")
    except requests.exceptions.RequestException as e:
        raise RuntimeError(f"API request failed: {e}")


def generate_sentence(
    content_unit: ContentUnit,
    style_reference: Optional[str] = None,
    config_path: str = "config.json",
    hint: Optional[str] = None
) -> str:
    """Generate a sentence using LLM with style reference.

    Generates a sentence that:
    - Preserves the EXACT semantic meaning from content_unit
    - Matches the sentence structure, rhythm, and vocabulary complexity of style_reference

    Args:
        content_unit: ContentUnit containing SVO triples, entities, and original text.
        style_reference: Reference paragraph to mimic in style (structure, rhythm, vocabulary).
        config_path: Path to configuration file.
        hint: Optional hint/feedback from previous attempt to improve generation.

    Returns:
        Generated sentence string.
    """
    # Load configuration
    config = _load_config(config_path)
    provider = config.get("provider", "deepseek")

    if provider == "deepseek":
        deepseek_config = config.get("deepseek", {})
        api_key = deepseek_config.get("api_key")
        api_url = deepseek_config.get("api_url")
        model = deepseek_config.get("editor_model", "deepseek-chat")

        if not api_key or not api_url:
            raise ValueError("DeepSeek API key or URL not found in config")
    else:
        raise ValueError(f"Unsupported provider: {provider}")

    # Build system prompt with strong anti-hallucination constraints and style matching
    system_prompt_parts = [
        "You are a constrained text re-writer. CRITICAL RULES:",
        "1. Preserve the EXACT meaning from the original sentence",
        "2. DO NOT add any new entities, locations, facts, people, or information not in the original",
        "3. DO NOT invent names, places, dates, or events",
        "4. Only use words and concepts that exist in the original text",
        "5. Match the sentence structure, rhythm, and vocabulary complexity of the reference paragraph EXACTLY",
        "6. Generate ONLY the rewritten sentence, nothing else (no explanations, no notes)",
        "",
        "BAD EXAMPLE: Original 'The cat sat' → Generated 'Einstein's cat sat in New York' (WRONG - added Einstein and New York)",
        "GOOD EXAMPLE: Original 'The cat sat' → Generated 'The feline rested' (CORRECT - preserved meaning, changed style)"
    ]

    if style_reference:
        system_prompt_parts.append("")
        system_prompt_parts.append("STYLE REFERENCE (match this paragraph's structure, rhythm, and vocabulary complexity):")
        system_prompt_parts.append(f'"{style_reference}"')
        system_prompt_parts.append("")
        system_prompt_parts.append("Your output should:")
        system_prompt_parts.append("- Match the average sentence length of the reference")
        system_prompt_parts.append("- Match the punctuation style and density")
        system_prompt_parts.append("- Match the vocabulary complexity and word choice")
        system_prompt_parts.append("- Match the sentence structure and rhythm")

    system_prompt = "\n".join(system_prompt_parts)

    # Extract meaning from content unit
    svo_triples = content_unit.svo_triples
    entities = content_unit.entities
    content_words = content_unit.content_words or []

    # Format meaning with full phrases
    meaning_parts = []
    if svo_triples:
        for svo in svo_triples:
            if svo:
                subject, verb, obj = svo
                meaning_parts.append(f"Subject: {subject}, Verb: {verb}, Object: {obj}")

    meaning_str = "\n".join(meaning_parts) if meaning_parts else "No specific SVO structure provided."

    # Build user prompt with all constraints
    user_prompt_parts = [
        "Rewrite the following sentence while preserving its EXACT meaning:",
        f"Original: {content_unit.original_text}",
        "",
        "CRITICAL: DO NOT add any new information, entities, locations, or facts not in the original.",
        "",
        "Requirements:",
        f"1. Meaning to preserve: {meaning_str}",
    ]

    # Add content words list to help preserve all concepts
    req_num = 2
    if content_words:
        # Limit to most important words (avoid too long lists)
        important_words = content_words[:20]  # First 20 content words
        user_prompt_parts.append(f"{req_num}. Key concepts to include: {', '.join(important_words)}")
        req_num += 1

    if entities:
        user_prompt_parts.append(f"{req_num}. Preserve these entities exactly (DO NOT add others): {', '.join(entities)}")
        req_num += 1

    # Add style reference requirement
    if style_reference:
        user_prompt_parts.append("")
        user_prompt_parts.append("STYLE REQUIREMENT: Match the sentence structure, rhythm, and vocabulary complexity of this reference:")
        user_prompt_parts.append(f'"{style_reference}"')
        user_prompt_parts.append("")
        user_prompt_parts.append("Specifically match:")
        user_prompt_parts.append("- Average sentence length")
        user_prompt_parts.append("- Punctuation style and density")
        user_prompt_parts.append("- Vocabulary complexity and word choice")
        user_prompt_parts.append("- Sentence structure and rhythm")

    user_prompt_parts.append("")
    user_prompt_parts.append("Remember: Only use words and concepts from the original. Do not invent anything new.")

    # Add hint from previous attempt if provided (for retries)
    if hint:
        user_prompt_parts.append("")
        user_prompt_parts.append("IMPORTANT FEEDBACK FROM PREVIOUS ATTEMPT:")
        user_prompt_parts.append(hint)
        user_prompt_parts.append("")
        user_prompt_parts.append("Please address the feedback above in your rewrite.")

    user_prompt = "\n".join(user_prompt_parts)

    # Call API
    generated_text = _call_deepseek_api(system_prompt, user_prompt, api_key, api_url, model)

    return generated_text

