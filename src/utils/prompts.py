"""Prompt loading utilities."""

from pathlib import Path
from functools import lru_cache

from .logging import get_logger

logger = get_logger(__name__)

# Prompts directory relative to project root
PROMPTS_DIR = Path(__file__).parent.parent.parent / "prompts"


@lru_cache(maxsize=50)
def load_prompt(name: str) -> str:
    """Load a prompt from the prompts directory.

    Args:
        name: Prompt filename without extension (e.g., "proposition_mapper_system")

    Returns:
        Prompt content as string.

    Raises:
        FileNotFoundError: If prompt file doesn't exist.
    """
    path = PROMPTS_DIR / f"{name}.md"

    if not path.exists():
        raise FileNotFoundError(f"Prompt not found: {path}")

    content = path.read_text(encoding="utf-8").strip()
    logger.debug(f"Loaded prompt: {name} ({len(content)} chars)")
    return content


def clear_prompt_cache() -> None:
    """Clear the prompt cache (useful for testing)."""
    load_prompt.cache_clear()
