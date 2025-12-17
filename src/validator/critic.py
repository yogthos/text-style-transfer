"""Adversarial Critic for style transfer quality control.

This module provides a critic LLM that evaluates generated text against
a reference style paragraph to detect style mismatches and "AI slop".
"""

import json
import requests
from typing import Dict, Optional, Tuple


def _load_config(config_path: str = "config.json") -> Dict:
    """Load configuration from config.json."""
    with open(config_path, 'r') as f:
        return json.load(f)


def _call_deepseek_api(system_prompt: str, user_prompt: str, api_key: str, api_url: str, model: str) -> str:
    """Call DeepSeek API for critic evaluation."""
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        "temperature": 0.3,
        "response_format": {"type": "json_object"}
    }

    response = requests.post(api_url, headers=headers, json=payload, timeout=30)
    response.raise_for_status()

    result = response.json()
    return result.get("choices", [{}])[0].get("message", {}).get("content", "")


def critic_evaluate(
    generated_text: str,
    structure_match: str,
    situation_match: Optional[str] = None,
    original_text: Optional[str] = None,
    config_path: str = "config.json"
) -> Dict[str, any]:
    """Evaluate generated text against dual RAG references.

    Uses an LLM to compare the generated text with structure_match (for rhythm)
    and situation_match (for vocabulary) to check for style mismatches.

    Args:
        generated_text: The generated text to evaluate.
        structure_match: Reference paragraph for rhythm/structure evaluation.
        situation_match: Optional reference paragraph for vocabulary evaluation.
        original_text: Original input text (for checking reference/quotation preservation).
        config_path: Path to configuration file.

    Returns:
        Dictionary with:
        - "pass": bool - Whether the generated text passes style check
        - "feedback": str - Specific feedback on what to improve
        - "score": float - Style match score (0-1)
    """
    config = _load_config(config_path)
    provider = config.get("provider", "deepseek")

    if provider == "deepseek":
        deepseek_config = config.get("deepseek", {})
        api_key = deepseek_config.get("api_key")
        api_url = deepseek_config.get("api_url")
        model = deepseek_config.get("critic_model", deepseek_config.get("editor_model", "deepseek-chat"))

        if not api_key or not api_url:
            raise ValueError("DeepSeek API key or URL not found in config")
    else:
        raise ValueError(f"Unsupported provider: {provider}")

    # Build system prompt
    system_prompt = """You are a style critic evaluating text style transfer quality.

Your task is to compare generated text against dual references:
1. STRUCTURAL REFERENCE: For evaluating sentence structure, rhythm, and pacing
2. SITUATIONAL REFERENCE: For evaluating vocabulary choices and word tone (if provided)

Determine:
1. Does the generated text match the structural reference's sentence structure and rhythm?
2. Does it match the vocabulary complexity and word choice (from situational reference if available)?
3. Does it avoid "AI words" (delve, underscore, testament, etc.)?
4. Does it match the average sentence length?
5. Does it match the punctuation style and density?
6. CRITICAL: Are ALL [^number] style citation references from the original text preserved exactly?
7. CRITICAL: Are ALL direct quotations from the original text preserved exactly?

Output your evaluation as JSON with:
- "pass": boolean (true if style matches well, false if needs improvement)
- "feedback": string (specific, actionable feedback on what to improve, mentioning which aspect failed)
- "score": float (0.0 to 1.0, where 1.0 is perfect style match)

Be strict but fair. Focus on structural and stylistic elements, not just meaning."""

    # Build user prompt
    structure_section = f"""STRUCTURAL REFERENCE (for rhythm/structure):
"{structure_match}"
"""

    situation_section = ""
    if situation_match:
        situation_section = f"""
SITUATIONAL REFERENCE (for vocabulary):
"{situation_match}"
"""
    else:
        situation_section = """
SITUATIONAL REFERENCE: Not provided (no similar topic found in corpus).
"""

    # Build original text section for reference preservation check
    original_section = ""
    if original_text:
        original_section = f"""
ORIGINAL TEXT (for reference preservation check):
"{original_text}"
"""

    user_prompt = f"""Compare the generated text against these references:

{structure_section}{situation_section}{original_section}
GENERATED TEXT (to evaluate):
"{generated_text}"

Evaluate whether the generated text matches:
- STRUCTURE: Sentence structure and rhythm from Structural Reference
- VOCABULARY: Word choice and tone from Situational Reference (if provided)
- Average sentence length
- Punctuation style and density
- Absence of "AI words" (delve, underscore, testament, etc.)"""

    if original_text:
        user_prompt += """
- CRITICAL: All [^number] style citation references from Original Text must be preserved exactly
- CRITICAL: All direct quotations (text in quotes) from Original Text must be preserved exactly"""

    user_prompt += """

Output JSON with "pass", "feedback", and "score" fields.
If vocabulary doesn't match, mention it. If structure doesn't match, mention it."""

    if original_text:
        user_prompt += """
If any references or quotations are missing, this is a CRITICAL FAILURE and "pass" must be false."""

    try:
        # Call API
        response_text = _call_deepseek_api(system_prompt, user_prompt, api_key, api_url, model)

        # Parse JSON response
        try:
            result = json.loads(response_text)
        except json.JSONDecodeError:
            # If JSON parsing fails, try to extract JSON from response
            import re
            json_match = re.search(r'\{[^}]+\}', response_text, re.DOTALL)
            if json_match:
                result = json.loads(json_match.group())
            else:
                # Fallback: create result from text
                result = {
                    "pass": False,
                    "feedback": "Could not parse critic response. Please retry.",
                    "score": 0.5
                }

        # Ensure required fields
        if "pass" not in result:
            result["pass"] = result.get("score", 0.5) >= 0.75
        if "feedback" not in result:
            result["feedback"] = "No specific feedback provided."
        if "score" not in result:
            result["score"] = 0.5

        # Ensure types
        result["pass"] = bool(result["pass"])
        result["score"] = float(result["score"])
        result["feedback"] = str(result["feedback"])

        return result

    except Exception as e:
        # Fallback on error
        return {
            "pass": True,  # Don't block generation on critic failure
            "feedback": f"Critic evaluation failed: {str(e)}",
            "score": 0.5
        }


def generate_with_critic(
    generate_fn,
    content_unit,
    structure_match: str,
    situation_match: Optional[str] = None,
    config_path: str = "config.json",
    max_retries: int = 3,
    min_score: float = 0.75
) -> Tuple[str, Dict[str, any]]:
    """Generate text with adversarial critic loop.

    Generates text, evaluates it with critic, and retries with feedback
    if the style doesn't match well enough.

    Args:
        generate_fn: Function to generate text (takes content_unit, structure_match, situation_match, hint).
        content_unit: ContentUnit to generate from.
        structure_match: Reference paragraph for rhythm/structure (required).
        situation_match: Optional reference paragraph for vocabulary.
        config_path: Path to configuration file.
        max_retries: Maximum number of retry attempts (default: 3).
        min_score: Minimum critic score to accept (default: 0.75).

    Returns:
        Tuple of:
        - generated_text: Best generated text
        - critic_result: Final critic evaluation result
    """
    if not structure_match:
        # No structure match, cannot proceed
        generated = content_unit.original_text
        return generated, {"pass": False, "feedback": "No structure match provided", "score": 0.0}

    best_text = None
    best_score = 0.0
    best_result = None
    hint = None

    for attempt in range(max_retries):
        # Generate with hint from previous attempt
        generated = generate_fn(content_unit, structure_match, situation_match, config_path, hint=hint)

        # Evaluate with critic
        critic_result = critic_evaluate(
            generated,
            structure_match,
            situation_match,
            original_text=content_unit.original_text,
            config_path=config_path
        )
        score = critic_result.get("score", 0.0)

        # Track best result
        if score > best_score:
            best_text = generated
            best_score = score
            best_result = critic_result

        # Check if we should accept
        if critic_result.get("pass", False) and score >= min_score:
            return generated, critic_result

        # Build hint for next attempt
        feedback = critic_result.get("feedback", "")
        if feedback:
            hint = f"Previous attempt feedback: {feedback}. Please address these style issues."

    # Return best result even if not perfect
    return best_text or generated, best_result or {"pass": False, "feedback": "Max retries reached", "score": best_score}

