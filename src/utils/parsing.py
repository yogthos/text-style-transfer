"""Utility functions for parsing LLM output, especially JSON extraction.

This module provides robust parsing utilities that handle common LLM formatting
errors like single quotes, Markdown code blocks, and extra text.
"""

import json
import re
import ast
from typing import Optional, Union, List, Dict


def extract_json_from_text(text: str, verbose: bool = False) -> Optional[Union[List, Dict]]:
    """Robustly extract JSON from text, handling single-quotes and Markdown.

    Handles common LLM output issues:
    - Single quotes instead of double quotes (Python dict format)
    - Markdown code blocks (```json ... ```)
    - Extra text before/after JSON
    - Multiple JSON objects

    Args:
        text: Text that may contain JSON
        verbose: If True, print debug information on failure

    Returns:
        Parsed JSON object (list or dict), or None if parsing fails
    """
    if not text or not text.strip():
        if verbose:
            print(f"  ⚠ extract_json_from_text: Empty input")
        return None

    original_text = text  # Keep for error reporting

    # 1. Clean Markdown
    text = re.sub(r'```(\w+)?', '', text).strip()

    # 2. Extract content between first [/{ and last ]/}
    match = re.search(r'(\[.*\]|\{.*\})', text, re.DOTALL)
    if not match:
        if verbose:
            print(f"  ⚠ extract_json_from_text: No JSON structure found in text")
            print(f"  ⚠ Original response (first 200 chars): {original_text[:200]}")
        return None
    candidate = match.group(1)

    # 3. Try Standard JSON
    try:
        result = json.loads(candidate)
        if verbose:
            print(f"  ✓ JSON parsed successfully (standard JSON)")
        return result
    except (json.JSONDecodeError, ValueError) as e:
        if verbose:
            print(f"  ⚠ Standard JSON parse failed: {e}")
        pass

    # 4. Try Python Literal Eval (Handles {'key': 'val'}) - THIS FIXES THE BUG
    try:
        result = ast.literal_eval(candidate)
        if verbose:
            print(f"  ✓ JSON parsed successfully (ast.literal_eval)")
        return result
    except (ValueError, SyntaxError) as e:
        if verbose:
            print(f"  ⚠ ast.literal_eval failed: {e}")
        pass

    # 5. Fallback: Regex replace single quotes
    try:
        # Replace 'key': with "key":
        candidate_fixed = re.sub(r"'([\w_]+)':", r'"\1":', candidate)
        # Replace 'value' with "value"
        candidate_fixed = re.sub(r":\s*'([^']*)'", r': "\1"', candidate_fixed)
        result = json.loads(candidate_fixed)
        if verbose:
            print(f"  ✓ JSON parsed successfully (regex fix)")
        return result
    except (json.JSONDecodeError, ValueError) as e:
        if verbose:
            print(f"  ⚠ Regex fix failed: {e}")
        pass

    # All parsing attempts failed - log the problematic text
    if verbose:
        print(f"  ⚠ CRITICAL: All JSON parsing attempts failed")
        print(f"  ⚠ Original response (first 500 chars): {original_text[:500]}")
        print(f"  ⚠ Extracted candidate (first 500 chars): {candidate[:500]}")
        print(f"  ⚠ This suggests the LLM did not return valid JSON format")

    return None

