"""Prompt loading utilities.

Loads prompt templates from the prompts/ directory, allowing prompts to be
edited without modifying code.
"""

from pathlib import Path
from typing import Dict, Optional
from functools import lru_cache

from .logging import get_logger

logger = get_logger(__name__)

# Default prompts directory
PROMPTS_DIR = Path(__file__).parent.parent.parent / "prompts"


@lru_cache(maxsize=32)
def load_prompt(name: str, prompts_dir: Optional[Path] = None) -> str:
    """Load a prompt template from file.

    Args:
        name: Prompt name (without .txt extension).
        prompts_dir: Optional custom prompts directory.

    Returns:
        Prompt template string.

    Raises:
        FileNotFoundError: If prompt file doesn't exist.
    """
    directory = prompts_dir or PROMPTS_DIR
    prompt_path = directory / f"{name}.txt"

    if not prompt_path.exists():
        raise FileNotFoundError(f"Prompt not found: {prompt_path}")

    with open(prompt_path, 'r', encoding='utf-8') as f:
        content = f.read()

    logger.debug(f"Loaded prompt: {name}")
    return content


def format_prompt(name: str, **kwargs) -> str:
    """Load and format a prompt template with variables.

    Uses Python's str.format() for variable substitution.

    Args:
        name: Prompt name (without .txt extension).
        **kwargs: Variables to substitute in the template.

    Returns:
        Formatted prompt string.

    Example:
        # prompts/style_transfer.txt contains:
        # "You are {author}. Rewrite the following text..."

        prompt = format_prompt("style_transfer", author="Carl Sagan")
    """
    template = load_prompt(name)
    return template.format(**kwargs)


def get_prompt_with_fallback(name: str, fallback: str, **kwargs) -> str:
    """Load a prompt with fallback if file doesn't exist.

    Useful for graceful degradation when prompt files are missing.

    Args:
        name: Prompt name (without .txt extension).
        fallback: Fallback prompt template if file not found.
        **kwargs: Variables to substitute in the template.

    Returns:
        Formatted prompt string.
    """
    try:
        template = load_prompt(name)
    except FileNotFoundError:
        logger.warning(f"Prompt file not found: {name}.txt, using fallback")
        template = fallback

    if kwargs:
        return template.format(**kwargs)
    return template


def list_prompts(prompts_dir: Optional[Path] = None) -> Dict[str, Path]:
    """List all available prompts.

    Args:
        prompts_dir: Optional custom prompts directory.

    Returns:
        Dictionary of prompt name -> file path.
    """
    directory = prompts_dir or PROMPTS_DIR

    if not directory.exists():
        return {}

    prompts = {}
    for path in directory.glob("*.txt"):
        prompts[path.stem] = path

    return prompts


def clear_prompt_cache():
    """Clear the prompt cache.

    Call this if prompts are modified at runtime.
    """
    load_prompt.cache_clear()
    logger.debug("Prompt cache cleared")
