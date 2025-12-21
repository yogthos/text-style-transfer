"""Linguistic testing helper utilities."""

import sys
from pathlib import Path
from typing import Tuple, List, Dict, Set
from spacy.tokens import Doc

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.validator.semantic_critic import SemanticCritic
from src.utils.nlp_manager import NLPManager


def calculate_semantic_drift(
    old_output: str,
    new_output: str,
    expected_similarity: float = 0.95,
    threshold: float = 0.15
) -> Tuple[float, float, bool]:
    """Calculate semantic drift between old and new outputs.

    Args:
        old_output: Original golden output
        new_output: Newly generated output
        expected_similarity: Expected similarity score (0.0-1.0)
        threshold: Maximum allowed relative drift (default 0.15 = 15%)

    Returns:
        Tuple of (actual_similarity, drift_percentage, passes_threshold)
        - actual_similarity: Cosine similarity between outputs (0.0-1.0)
        - drift_percentage: Relative drift as percentage (0.0-1.0)
        - passes_threshold: True if drift <= threshold
    """
    critic = SemanticCritic()

    # Calculate actual similarity
    actual_similarity = critic._calculate_semantic_similarity(old_output, new_output)

    # Calculate relative drift: (expected - actual) / expected
    if expected_similarity > 0:
        drift = (expected_similarity - actual_similarity) / expected_similarity
    else:
        drift = 1.0 if actual_similarity < 0.5 else 0.0

    # Pass if drift is within threshold
    passes = drift <= threshold

    return actual_similarity, drift, passes


def extract_perspective_pronouns(doc: Doc) -> Dict[str, List[str]]:
    """Extract pronouns with Person/Number from spaCy Doc.

    Args:
        doc: spaCy Doc object

    Returns:
        Dictionary with keys:
        - first_singular: List of first person singular pronouns (I, me, my, mine, myself)
        - first_plural: List of first person plural pronouns (we, us, our, ours, ourselves)
        - third_singular: List of third person singular pronouns (he, she, it, him, her, his, hers, its, himself, herself, itself)
        - third_plural: List of third person plural pronouns (they, them, their, theirs, themselves)
    """
    pronouns = {
        "first_singular": [],
        "first_plural": [],
        "third_singular": [],
        "third_plural": []
    }

    for token in doc:
        if token.pos_ == "PRON":
            person = token.morph.get("Person")
            number = token.morph.get("Number")

            # First person
            if person == ["1"]:
                if number == ["Sing"]:
                    pronouns["first_singular"].append(token.text)
                elif number == ["Plur"]:
                    pronouns["first_plural"].append(token.text)

            # Third person
            elif person == ["3"]:
                if number == ["Sing"]:
                    pronouns["third_singular"].append(token.text)
                elif number == ["Plur"]:
                    pronouns["third_plural"].append(token.text)

            # Fallback: Check by pronoun type
            elif token.text.lower() in ["i", "me", "my", "mine", "myself"]:
                pronouns["first_singular"].append(token.text)
            elif token.text.lower() in ["we", "us", "our", "ours", "ourselves"]:
                pronouns["first_plural"].append(token.text)
            elif token.text.lower() in ["he", "she", "it", "him", "her", "his", "hers", "its", "himself", "herself", "itself"]:
                pronouns["third_singular"].append(token.text)
            elif token.text.lower() in ["they", "them", "their", "theirs", "themselves"]:
                pronouns["third_plural"].append(token.text)

    return pronouns


def count_llm_calls(func):
    """Decorator to count LLM invocations.

    Works with both real and mocked LLM providers.
    Stores count in function attribute 'llm_call_count'.
    """
    def wrapper(*args, **kwargs):
        # Initialize counter if not exists
        if not hasattr(func, 'llm_call_count'):
            func.llm_call_count = 0

        # Call original function
        result = func(*args, **kwargs)

        # Increment counter (assumes LLM call happened)
        func.llm_call_count += 1

        return result

    return wrapper


def get_mock_llm_provider():
    """Factory function to create a mocked LLM provider for tests.

    Returns:
        MockLLMProvider instance
    """
    from tests.mocks.mock_llm_provider import get_mock_llm_provider as _get_mock
    return _get_mock()

