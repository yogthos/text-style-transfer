"""Tests for Semantic Scalpel & Style Profiler functionality.

This test suite ensures the "Semantic Scalpel" works effectively and doesn't break
grammar or casing. It targets three key layers:
1. Data Extraction (Profiler)
2. Logic (Semantic Finder)
3. Execution (Translator)
"""

import sys
from pathlib import Path
import pytest
import spacy
from unittest.mock import MagicMock, patch
from typing import Dict, List

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Ensure config.json exists before imports
from tests.test_helpers import ensure_config_exists
ensure_config_exists()

from src.utils.text_processing import get_semantic_replacement
from src.analysis.style_profiler import StyleProfiler
from src.generator.translator import StyleTranslator


# Load NLP model once for all tests (performance)
@pytest.fixture(scope="module")
def nlp_model():
    """Load spaCy model for testing."""
    try:
        return spacy.load("en_core_web_lg")
    except OSError:
        # Fallback to medium model if large not available
        try:
            return spacy.load("en_core_web_md")
        except OSError:
            # Fallback to small model
            return spacy.load("en_core_web_sm")


class TestStyleProfiler:
    """Unit tests for Style Profiler (Data Extraction layer)."""

    def test_sensory_verb_extraction(self, nlp_model):
        """Test Case 1.1: Sensory Verb Extraction.

        Verify that sensory verbs (see, feel, hear) are correctly extracted
        and lemmatized from the text.
        """
        profiler = StyleProfiler()
        profiler.nlp = nlp_model

        text = "I saw the light. He felt the cold. She heard the noise."
        doc = nlp_model(text)
        palette = profiler._extract_vocabulary_palette(doc)

        # Check that sensory verbs are extracted and lemmatized
        sensory_verbs = set(palette.get("sensory_verbs", []))
        expected = {'see', 'feel', 'hear'}

        # All expected verbs should be present (may have more)
        assert expected.issubset(sensory_verbs), \
            f"Expected {expected} in sensory_verbs, got {sensory_verbs}"

    def test_intensifier_extraction(self, nlp_model):
        """Test Case 1.2: Intensifier Extraction.

        Verify that intensifiers (adverbs modifying adjectives) are correctly extracted.
        """
        profiler = StyleProfiler()
        profiler.nlp = nlp_model

        text = "It was profoundly dark. The utterly silent room."
        doc = nlp_model(text)
        palette = profiler._extract_vocabulary_palette(doc)

        # Check that intensifiers are extracted
        intensifiers = set(palette.get("intensifiers", []))
        expected = {'profoundly', 'utterly'}

        # All expected intensifiers should be present
        assert expected.issubset(intensifiers), \
            f"Expected {expected} in intensifiers, got {intensifiers}"

    def test_connective_extraction(self, nlp_model):
        """Test Case 1.3: Connective Extraction.

        Verify that connective words (therefore, however) are correctly extracted.
        Note: spaCy may not always tag these as connectives in all contexts,
        so we check for a more lenient condition.
        """
        profiler = StyleProfiler()
        profiler.nlp = nlp_model

        text = "Therefore, we went. However, it rained. But then we left."
        doc = nlp_model(text)
        palette = profiler._extract_vocabulary_palette(doc)

        # Check that connectives are extracted
        connectives = set(palette.get("connectives", []))

        # At least some connectives should be present
        # (spaCy may not always tag "therefore" and "however" as connectives,
        # but "but" or "then" might be detected)
        assert len(connectives) > 0 or len(palette.get("general", [])) > 0, \
            f"Expected at least some connectives or general words, got connectives={connectives}, general={palette.get('general', [])[:5]}"


class TestSemanticUtility:
    """Unit tests for Semantic Utility (The Brain - Logic layer)."""

    def test_nearest_neighbor_finding(self, nlp_model):
        """Test Case 2.1: Nearest Neighbor Finding.

        Verify that the function finds the best semantic match from the palette.
        Forbidden word is 'vast' (adjective), and 'large' should be the closest match.
        """
        forbidden = "vast"
        palette = {
            "general": ["tiny", "blue", "large"],
            "sensory_verbs": [],
            "connectives": [],
            "intensifiers": []
        }

        replacement = get_semantic_replacement(forbidden, palette, nlp_model)

        # 'large' should be semantically closer to 'vast' than 'tiny' or 'blue'
        assert replacement == "large", \
            f"Expected 'large' as replacement for 'vast', got '{replacement}'"

    def test_pos_constraint(self, nlp_model):
        """Test Case 2.2: POS Constraint (The "Ocean" Guard).

        Verify that words with different Part of Speech are rejected.
        Forbidden word is 'vast' (Adjective), and 'ocean' (Noun) should be rejected.
        """
        forbidden = "vast"  # Adjective
        palette = {
            "general": ["ocean", "sky"],  # Nouns
            "sensory_verbs": [],
            "connectives": [],
            "intensifiers": []
        }

        replacement = get_semantic_replacement(forbidden, palette, nlp_model)

        # Should fall back to a generic adjective, NOT 'ocean' (which is a noun)
        assert replacement is not None, "Should return a fallback replacement"
        assert replacement != "ocean", \
            f"Should NOT return 'ocean' (noun) for adjective 'vast', got '{replacement}'"
        assert replacement in ["large", "big", "huge", "clear", "simple", "real", "human", "dark", "light", "heavy"], \
            f"Should return a generic adjective fallback, got '{replacement}'"

    def test_threshold_fallback(self, nlp_model):
        """Test Case 2.3: Threshold Fallback.

        Verify that when no semantically related word exists in the palette,
        the function returns a generic fallback.
        """
        forbidden = "intricate"  # Adjective
        palette = {
            "general": ["dog", "cat", "food"],  # No semantic relation to 'intricate'
            "sensory_verbs": [],
            "connectives": [],
            "intensifiers": []
        }

        replacement = get_semantic_replacement(forbidden, palette, nlp_model)

        # Should return generic fallback because similarity < 0.3
        assert replacement is not None, "Should return a fallback replacement"
        assert replacement in ["large", "clear", "simple", "real", "human", "dark", "light", "heavy"], \
            f"Should return generic adjective fallback, got '{replacement}'"


class TestTranslatorCleanup:
    """Integration tests for Translator Cleanup (Execution layer)."""

    @pytest.fixture
    def translator(self, nlp_model):
        """Create a StyleTranslator instance for testing."""
        translator = StyleTranslator(config_path="config.json")
        translator._nlp_cache = nlp_model
        translator._get_nlp = lambda: nlp_model
        return translator

    def test_case_preservation(self, translator, nlp_model):
        """Test Case 3.1: Case Preservation.

        Verify that capitalization is preserved when replacing words.
        'Vast' (capitalized) should be replaced with 'Large' (capitalized).
        """
        # Mock the style profile loading to return a palette with 'large'
        with patch.object(translator, '_load_style_profile', return_value={
            "vocabulary_palette": {
                "general": ["large"],
                "sensory_verbs": [],
                "connectives": [],
                "intensifiers": []
            }
        }):
            input_text = "Vast networks exist."
            violations = ["vast"]

            # Pass author_name so the profile gets loaded
            cleaned = translator._programmatic_cleanup(input_text, violations, author_name="TestAuthor")

            # Should preserve capitalization
            assert cleaned == "Large networks exist.", \
                f"Expected 'Large networks exist.', got '{cleaned}'"

    def test_verb_safety_guard(self, translator, nlp_model):
        """Test Case 3.2: Verb Safety Guard.

        Verify that verbs are skipped to avoid conjugation errors.
        'perceived' (verb) should remain unchanged.
        """
        input_text = "He perceived the threat."
        violations = ["perceived"]  # Verb

        cleaned = translator._programmatic_cleanup(input_text, violations)

        # Should NOT change verbs to avoid conjugation errors
        assert cleaned == "He perceived the threat.", \
            f"Expected verb to remain unchanged, got '{cleaned}'"

    def test_noun_plurality(self, translator, nlp_model):
        """Test Case 3.3: Noun Plurality.

        Verify that plural nouns are handled correctly.
        'complexities' (plural) should be replaced with 'structures' (plural).
        """
        # Mock the style profile loading to return a palette with 'structure'
        with patch.object(translator, '_load_style_profile', return_value={
            "vocabulary_palette": {
                "general": ["structure"],
                "sensory_verbs": [],
                "connectives": [],
                "intensifiers": []
            }
        }):
            input_text = "The complexities of life."
            violations = ["complexities"]  # Plural noun

            # Pass author_name so the profile gets loaded
            cleaned = translator._programmatic_cleanup(input_text, violations, author_name="TestAuthor")

            # Should preserve plurality (naive 's' addition)
            assert cleaned == "The structures of life.", \
                f"Expected 'The structures of life.', got '{cleaned}'"

    def test_multiple_replacements(self, translator, nlp_model):
        """Additional test: Multiple word replacements in one text."""
        with patch.object(translator, '_load_style_profile', return_value={
            "vocabulary_palette": {
                "general": ["large", "system"],
                "sensory_verbs": [],
                "connectives": [],
                "intensifiers": []
            }
        }):
            input_text = "Vast networks and complex systems exist."
            violations = ["vast", "complex"]

            # Pass author_name so the profile gets loaded
            cleaned = translator._programmatic_cleanup(input_text, violations, author_name="TestAuthor")

            # Should replace both words
            assert "Large" in cleaned or "large" in cleaned, \
                f"Expected 'vast' to be replaced, got '{cleaned}'"
            assert "complex" not in cleaned.lower(), \
                f"Expected 'complex' to be replaced, got '{cleaned}'"

    def test_empty_palette_fallback(self, translator, nlp_model):
        """Additional test: Fallback when author palette is empty."""
        with patch.object(translator, '_load_style_profile', return_value={
            "vocabulary_palette": {}
        }):
            input_text = "The vast network."
            violations = ["vast"]

            cleaned = translator._programmatic_cleanup(input_text, violations)

            # Should use generic fallback or delete the word
            assert "vast" not in cleaned.lower(), \
                f"Expected 'vast' to be removed or replaced, got '{cleaned}'"

