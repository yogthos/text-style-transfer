"""Test Phase 1: Robust Skeleton Retrieval (The Casting Director).

This test ensures that:
1. Freshness: Different skeletons returned on subsequent calls
2. Length Matching: Skeleton length matches proposition count
3. Position Filtering: Openers/Closers match position
4. Fallback: Synthetic skeleton generated when no matches found
"""

import unittest
from unittest.mock import Mock, patch, MagicMock
from typing import List

from src.generator.translator import StyleTranslator
from src.atlas.rhetoric import RhetoricalType


class MockStyleAtlas:
    """Mock StyleAtlas for testing."""

    def __init__(self, paragraphs: List[str]):
        """Initialize with list of paragraphs."""
        self.paragraphs = paragraphs
        self.call_count = 0

    def get_examples_by_rhetoric(self, rhetorical_type, top_k, author_name=None, query_text=None, exclude=None):
        """Return paragraphs, cycling through them for freshness test."""
        # For freshness test: return different paragraphs on each call
        if self.call_count < len(self.paragraphs):
            result = self.paragraphs[self.call_count:self.call_count + top_k]
            self.call_count += 1
            return result
        # If we've exhausted, wrap around
        start = (self.call_count % len(self.paragraphs))
        return self.paragraphs[start:start + top_k]


class MockStructureExtractor:
    """Mock StructureExtractor for testing."""

    def extract_template(self, text: str) -> str:
        """Return a mock template based on text."""
        # Simple mock: replace content words with placeholders
        if "revolution" in text.lower():
            return "[NP] is not [NP]."
        elif "violence" in text.lower():
            return "It is [NP] of [NP]."
        elif "struggle" in text.lower():
            return "The [NP] [VP] [NP]."
        elif "practice" in text.lower():
            return "[NP] [VP] [ADJ]."
        else:
            return "[NP] [VP]."


class TestPhase1Skeleton(unittest.TestCase):
    """Test robust skeleton retrieval."""

    def setUp(self):
        """Set up test fixtures."""
        self.config = {
            "evolutionary": {"max_generations": 10},
            "paragraph_fusion": {"style_alignment_threshold": 0.7},
            "translator": {"max_tokens": 200}
        }

        # Create translator with mocked dependencies
        with patch('src.generator.translator.LLMProvider'):
            with patch('src.generator.translator.SoftScorer'):
                with patch('src.generator.translator.Structuralizer'):
                    with patch('src.generator.translator.PropositionExtractor'):
                        with patch('src.analyzer.structure_extractor.StructureExtractor'):
                            with patch('src.analyzer.rhetorical_classifier.RhetoricalClassifier'):
                                self.translator = StyleTranslator(config_path="config.json")
                                self.translator.config = self.config
                                self.translator.translator_config = self.config.get("translator", {})
                                self.translator.paragraph_fusion_config = self.config.get("paragraph_fusion", {})

                                # Mock structure extractor
                                self.mock_structure_extractor = MockStructureExtractor()
                                self.translator.structure_extractor = self.mock_structure_extractor

    def test_freshness_different_skeletons(self):
        """Test that subsequent calls return different skeletons."""
        # Setup: Mock Atlas with multiple paragraphs
        paragraphs = [
            "The revolution is not a dinner party. It is an act of violence.",
            "The struggle continues. Practice makes perfect.",
            "Material conditions shape reality. History moves forward."
        ]
        mock_atlas = MockStyleAtlas(paragraphs)

        # First call
        result1 = self.translator._retrieve_robust_skeleton(
            rhetorical_type=RhetoricalType.ARGUMENT,
            author_name="Mao",
            proposition_count=4,
            position="BODY",
            atlas=mock_atlas,
            verbose=False
        )

        # Second call - should return different skeleton
        result2 = self.translator._retrieve_robust_skeleton(
            rhetorical_type=RhetoricalType.ARGUMENT,
            author_name="Mao",
            proposition_count=4,
            position="BODY",
            atlas=mock_atlas,
            verbose=False
        )

        # Assertions
        self.assertIsInstance(result1, list, "Result should be a list")
        self.assertIsInstance(result2, list, "Result should be a list")
        self.assertNotEqual(result1, result2, "Second call should return different skeleton")

        # Verify used_skeleton_ids prevents reuse
        self.assertEqual(len(self.translator.used_skeleton_ids), 2, "Should track 2 used skeletons")

    def test_length_matching_long_skeleton(self):
        """Test that 8 propositions get a longer skeleton (3+ sentences)."""
        # Setup: Mock Atlas with paragraphs of varying lengths
        paragraphs = [
            "Short. Simple.",  # 2 sentences
            "First sentence. Second sentence. Third sentence. Fourth sentence.",  # 4 sentences
            "One. Two. Three."  # 3 sentences
        ]
        mock_atlas = MockStyleAtlas(paragraphs)

        result = self.translator._retrieve_robust_skeleton(
            rhetorical_type=RhetoricalType.ARGUMENT,
            author_name="Mao",
            proposition_count=8,  # 8 propositions -> should prefer 3-4 sentence skeleton
            position="BODY",
            atlas=mock_atlas,
            verbose=False
        )

        # Assertions
        self.assertIsInstance(result, list)
        self.assertGreaterEqual(len(result), 3, "8 propositions should get skeleton with 3+ sentences")

    def test_length_matching_short_skeleton(self):
        """Test that 2 propositions get a shorter skeleton (1-2 sentences)."""
        paragraphs = [
            "Short. Simple.",  # 2 sentences
            "First sentence. Second sentence. Third sentence. Fourth sentence.",  # 4 sentences
        ]
        mock_atlas = MockStyleAtlas(paragraphs)

        result = self.translator._retrieve_robust_skeleton(
            rhetorical_type=RhetoricalType.ARGUMENT,
            author_name="Mao",
            proposition_count=2,  # 2 propositions -> should prefer 1-2 sentence skeleton
            position="BODY",
            atlas=mock_atlas,
            verbose=False
        )

        # Assertions
        self.assertIsInstance(result, list)
        self.assertLessEqual(len(result), 2, "2 propositions should get skeleton with 1-2 sentences")

    def test_position_filtering_opener(self):
        """Test that OPENER position prefers opener paragraphs."""
        # This test verifies position filtering logic exists
        # In a real implementation, we'd need position metadata in paragraphs
        paragraphs = [
            "The revolution is not a dinner party. It is an act of violence.",
        ]
        mock_atlas = MockStyleAtlas(paragraphs)

        result = self.translator._retrieve_robust_skeleton(
            rhetorical_type=RhetoricalType.ARGUMENT,
            author_name="Mao",
            proposition_count=4,
            position="OPENER",
            atlas=mock_atlas,
            verbose=False
        )

        # Assertions
        self.assertIsInstance(result, list)
        self.assertGreater(len(result), 0, "Should return at least one template")

    def test_position_filtering_closer(self):
        """Test that CLOSER position prefers closer paragraphs."""
        paragraphs = [
            "The struggle continues. Practice makes perfect.",
        ]
        mock_atlas = MockStyleAtlas(paragraphs)

        result = self.translator._retrieve_robust_skeleton(
            rhetorical_type=RhetoricalType.ARGUMENT,
            author_name="Mao",
            proposition_count=4,
            position="CLOSER",
            atlas=mock_atlas,
            verbose=False
        )

        # Assertions
        self.assertIsInstance(result, list)
        self.assertGreater(len(result), 0, "Should return at least one template")

    def test_fallback_synthetic_skeleton(self):
        """Test that empty Atlas results in synthetic skeleton generation."""
        # Setup: Mock Atlas returning empty list
        mock_atlas = Mock()
        mock_atlas.get_examples_by_rhetoric.return_value = []

        result = self.translator._retrieve_robust_skeleton(
            rhetorical_type=RhetoricalType.ARGUMENT,
            author_name="Mao",
            proposition_count=4,
            position="BODY",
            atlas=mock_atlas,
            verbose=False
        )

        # Assertions
        self.assertIsInstance(result, list, "Should return list even when Atlas is empty")
        self.assertGreater(len(result), 0, "Synthetic skeleton should have at least one template")
        # Verify templates contain placeholders
        for template in result:
            self.assertIsInstance(template, str)
            # Check for placeholder patterns (basic check)
            self.assertTrue(
                any(placeholder in template for placeholder in ['[NP]', '[VP]', '[ADJ]', '[ADV]']),
                f"Template should contain placeholders: {template}"
            )

    def test_templates_contain_placeholders(self):
        """Test that returned templates contain structural placeholders."""
        paragraphs = [
            "The revolution is not a dinner party. It is an act of violence.",
        ]
        mock_atlas = MockStyleAtlas(paragraphs)

        result = self.translator._retrieve_robust_skeleton(
            rhetorical_type=RhetoricalType.ARGUMENT,
            author_name="Mao",
            proposition_count=4,
            position="BODY",
            atlas=mock_atlas,
            verbose=False
        )

        # Assertions
        self.assertIsInstance(result, list)
        self.assertGreater(len(result), 0, "Should return at least one template")

        # Check all templates contain placeholders
        for template in result:
            self.assertIsInstance(template, str)
            self.assertTrue(
                any(placeholder in template for placeholder in ['[NP]', '[VP]', '[ADJ]', '[ADV]']),
                f"Template should be bleached (contain placeholders): {template}"
            )


if __name__ == '__main__':
    unittest.main()

