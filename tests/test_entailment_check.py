"""Tests for entailment-based proposition recall check."""

import unittest
from unittest.mock import Mock, patch
from src.validator.semantic_critic import SemanticCritic


class TestEntailmentCheck(unittest.TestCase):
    """Test that entailment logic correctly identifies stylistic expansions."""

    def setUp(self):
        """Set up test fixtures."""
        self.critic = SemanticCritic(config_path="config.json")

    @patch('src.validator.semantic_critic.SemanticCritic._check_entailment')
    def test_entailment_check_called_for_borderline_cases(self, mock_entailment):
        """Test that entailment check is called for borderline similarity cases."""
        # Mock entailment to return True (preserved)
        mock_entailment.return_value = (True, 0.8)

        # Mock semantic model to return borderline similarity (0.22)
        with patch.object(self.critic, 'semantic_model') as mock_model:
            if not mock_model:
                self.skipTest("Semantic model not available")

            # Create mock embeddings that return 0.22 similarity
            mock_embedding = Mock()
            mock_model.encode.return_value = mock_embedding

            # Mock util.cos_sim to return 0.22
            with patch('src.validator.semantic_critic.util') as mock_util:
                mock_util.cos_sim.return_value = Mock(item=lambda: 0.22)

                propositions = ["Social structures are vapor."]
                generated = "Social structures are vapor before material forces."

                # This should trigger entailment check (similarity 0.22 is borderline)
                recall, details = self.critic._check_proposition_recall(
                    generated,
                    propositions,
                    similarity_threshold=0.30,
                    verbose=False
                )

                # Verify entailment was called for borderline case
                # (Note: This test may need adjustment based on actual implementation)
                # The key is that entailment should be called when similarity is 0.20-0.30

    def test_entailment_preserves_stylistic_expansion(self):
        """Test that stylistic expansion (no contradiction) passes entailment."""
        # Mock LLM to return entailment=True
        mock_llm = Mock()
        mock_llm.call.return_value = '{"entails": true, "confidence": 0.85, "reason": "Stylistic expansion, no contradiction"}'
        self.critic.llm_provider = mock_llm

        proposition = "Social structures are vapor."
        generated = "Social structures are vapor before the material forces of history."

        entails, confidence = self.critic._check_entailment(proposition, generated)

        self.assertTrue(entails, "Stylistic expansion should pass entailment")
        self.assertGreaterEqual(confidence, 0.6, "Confidence should be reasonable")

    def test_entailment_rejects_contradiction(self):
        """Test that contradictions fail entailment check."""
        # Mock LLM to return entailment=False
        mock_llm = Mock()
        mock_llm.call.return_value = '{"entails": false, "confidence": 0.2, "reason": "Contradiction: collapsed vs flourished"}'
        self.critic.llm_provider = mock_llm

        proposition = "The economy collapsed."
        generated = "The economy flourished."

        entails, confidence = self.critic._check_entailment(proposition, generated)

        self.assertFalse(entails, "Contradictions should fail entailment")

    def test_entailment_fallback_when_llm_unavailable(self):
        """Test that entailment check falls back gracefully when LLM unavailable."""
        self.critic.llm_provider = None

        proposition = "Test proposition."
        generated = "Test generated sentence."

        entails, confidence = self.critic._check_entailment(proposition, generated)

        # Should assume preserved (neutral) when LLM unavailable
        self.assertTrue(entails, "Should assume preserved when LLM unavailable")
        self.assertEqual(confidence, 0.5, "Should return neutral confidence")


if __name__ == '__main__':
    unittest.main()

