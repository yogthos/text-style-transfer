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
from src.generator.prompt_builder import PromptAssembler
from src.analyzer.style_metrics import get_style_vector


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
    structure_match: str,
    situation_match: Optional[str] = None,
    config_path: str = "config.json",
    hint: Optional[str] = None,
    target_author_name: str = "Target Author",
    global_vocab_list: Optional[List[str]] = None
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

    # Initialize prompt assembler
    assembler = PromptAssembler(target_author_name=target_author_name)

    # Build system prompt using PromptAssembler
    system_prompt = assembler.build_system_message()

    # Add examples to system prompt
    system_prompt += "\n\n"
    system_prompt += "BAD EXAMPLE: Original 'The cat sat' → Generated 'Einstein's cat sat in New York' (WRONG - added Einstein and New York)\n"
    system_prompt += "GOOD EXAMPLE: Original 'The cat sat' → Generated 'The feline rested' (CORRECT - preserved meaning, changed style)"

    # Extract style metrics from structure_match
    style_vec = get_style_vector(structure_match)
    words = structure_match.split()
    sentences = structure_match.count('.') + structure_match.count('!') + structure_match.count('?')
    if sentences > 0:
        avg_sentence_len = len(words) / sentences
    else:
        avg_sentence_len = len(words)

    style_metrics = {'avg_sentence_len': avg_sentence_len}

    # Build user prompt using PromptAssembler
    user_prompt = assembler.build_generation_prompt(
        input_text=content_unit.original_text,
        situation_match=situation_match,
        structure_match=structure_match,
        style_metrics=style_metrics,
        global_vocab_list=global_vocab_list
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
    if hint:
        user_prompt += "\n\n"
        user_prompt += "IMPORTANT FEEDBACK FROM PREVIOUS ATTEMPT:\n"
        user_prompt += hint
        user_prompt += "\n\n"
        user_prompt += "Please address the feedback above in your rewrite."

    # Call API
    generated_text = _call_deepseek_api(system_prompt, user_prompt, api_key, api_url, model)

    return generated_text

