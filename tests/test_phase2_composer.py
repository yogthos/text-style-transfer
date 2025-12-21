"""Test Phase 2: Fusion Composer (The Architect).

This test ensures that:
1. Exact Fit: 3 props into 3 slots works correctly
2. Compression: 5 props into 2 slots (LLM combines)
3. Expansion: 2 props into 4 slots (LLM splits/elaborates)
4. Mismatch Handling: Wrong number of sentences handled gracefully
"""

import unittest
from unittest.mock import Mock, patch, MagicMock
import json

from src.generator.translator import StyleTranslator


class TestPhase2Composer(unittest.TestCase):
    """Test fusion composer implementation."""

    def setUp(self):
        """Set up test fixtures."""
        self.config = {
            "evolutionary": {"max_generations": 10},
            "paragraph_fusion": {"style_alignment_threshold": 0.7},
            "translator": {"max_tokens": 400}
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

    def test_exact_fit(self):
        """Test that 3 propositions fit exactly into 3 skeleton slots."""
        propositions = ["I was 12.", "I scavenged for food.", "It was cold."]
        skeleton = ["[NP] was [ADJ].", "[NP] [VP] [NP].", "It was [ADJ]."]

        # Mock LLM returning exactly 3 sentences
        mock_response = json.dumps([
            "The age of twelve was cold.",
            "I scavenged for food daily.",
            "It was cold outside."
        ])

        self.translator.llm_provider = Mock()
        self.translator.llm_provider.call.return_value = mock_response

        result = self.translator._compose_initial_draft(
            propositions=propositions,
            skeleton_templates=skeleton,
            author_name="Mao",
            verbose=False
        )

        # Assertions
        self.assertIsInstance(result, list)
        self.assertEqual(len(result), len(skeleton), "Output length must match skeleton length")
        self.assertEqual(len(result), 3, "Should have exactly 3 sentences")
        for sentence in result:
            self.assertIsInstance(sentence, str)
            self.assertGreater(len(sentence), 0)

    def test_compression(self):
        """Test that 5 propositions can be compressed into 2 skeleton slots."""
        propositions = [
            "I was 12.",
            "I scavenged for food.",
            "It was cold.",
            "The Soviet Union was a ghost.",
            "Life was hard."
        ]
        skeleton = ["[NP] was [ADJ].", "[NP] [VP] [NP]."]

        # Mock LLM combining propositions into 2 sentences
        mock_response = json.dumps([
            "The age of twelve was cold, and life was hard.",
            "I scavenged for food while the Soviet Union was a ghost."
        ])

        self.translator.llm_provider = Mock()
        self.translator.llm_provider.call.return_value = mock_response

        result = self.translator._compose_initial_draft(
            propositions=propositions,
            skeleton_templates=skeleton,
            author_name="Mao",
            verbose=False
        )

        # Assertions
        self.assertIsInstance(result, list)
        self.assertEqual(len(result), len(skeleton), "Output length must match skeleton length")
        self.assertEqual(len(result), 2, "Should have exactly 2 sentences")
        # Verify all propositions are represented (basic check - at least some keywords present)
        result_text = " ".join(result).lower()
        keywords = ["12", "scavenged", "cold", "soviet", "hard"]
        found_keywords = sum(1 for kw in keywords if kw in result_text)
        self.assertGreater(found_keywords, 2, "Most propositions should be represented")

    def test_expansion(self):
        """Test that 2 propositions can be expanded into 4 skeleton slots."""
        propositions = ["I was 12.", "I scavenged for food."]
        skeleton = ["[NP] was [ADJ].", "[NP] [VP] [NP].", "The [NP] [VP].", "It [VP] [ADJ]."]

        # Mock LLM splitting/elaborating propositions into 4 sentences
        mock_response = json.dumps([
            "The age of twelve was young.",
            "I scavenged for food daily.",
            "The practice of scavenging was necessary.",
            "It was a daily struggle."
        ])

        self.translator.llm_provider = Mock()
        self.translator.llm_provider.call.return_value = mock_response

        result = self.translator._compose_initial_draft(
            propositions=propositions,
            skeleton_templates=skeleton,
            author_name="Mao",
            verbose=False
        )

        # Assertions
        self.assertIsInstance(result, list)
        self.assertEqual(len(result), len(skeleton), "Output length must match skeleton length")
        self.assertEqual(len(result), 4, "Should have exactly 4 sentences")
        # Verify both propositions are represented
        result_text = " ".join(result).lower()
        self.assertTrue("12" in result_text or "twelve" in result_text, "First proposition should be present")
        self.assertTrue("scaveng" in result_text, "Second proposition should be present")

    def test_mismatch_handling_too_many(self):
        """Test handling when LLM returns too many sentences."""
        propositions = ["I was 12.", "I scavenged for food."]
        skeleton = ["[NP] was [ADJ].", "[NP] [VP] [NP]."]  # Expect 2 sentences

        # Mock LLM returning 3 sentences (mismatch)
        mock_response = json.dumps([
            "The age of twelve was cold.",
            "I scavenged for food daily.",
            "It was a difficult time."  # Extra sentence
        ])

        self.translator.llm_provider = Mock()
        self.translator.llm_provider.call.return_value = mock_response

        result = self.translator._compose_initial_draft(
            propositions=propositions,
            skeleton_templates=skeleton,
            author_name="Mao",
            verbose=False
        )

        # Assertions
        self.assertIsInstance(result, list)
        self.assertEqual(len(result), len(skeleton), "Should truncate to match skeleton length")
        self.assertEqual(len(result), 2, "Should have exactly 2 sentences after truncation")

    def test_mismatch_handling_too_few(self):
        """Test handling when LLM returns too few sentences."""
        propositions = ["I was 12.", "I scavenged for food.", "It was cold."]
        skeleton = ["[NP] was [ADJ].", "[NP] [VP] [NP].", "It was [ADJ]."]  # Expect 3 sentences

        # Mock LLM returning only 1 sentence (mismatch)
        mock_response = json.dumps([
            "The age of twelve was cold and I scavenged for food."
        ])

        self.translator.llm_provider = Mock()
        self.translator.llm_provider.call.return_value = mock_response

        result = self.translator._compose_initial_draft(
            propositions=propositions,
            skeleton_templates=skeleton,
            author_name="Mao",
            verbose=False
        )

        # Assertions
        self.assertIsInstance(result, list)
        # Should either retry or pad - for now, just verify it handles gracefully
        # In a full implementation, this might retry with stronger constraints
        self.assertGreater(len(result), 0, "Should return at least one sentence")
        # If padding is implemented, length should match; if retry, might be different
        # For now, just verify it doesn't crash


if __name__ == '__main__':
    unittest.main()

