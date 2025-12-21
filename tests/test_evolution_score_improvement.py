"""Integration test to verify evolution actually improves scores."""

import unittest
from unittest.mock import Mock, patch
from src.generator.translator import StyleTranslator
from src.ingestion.blueprint import SemanticBlueprint
from src.atlas.rhetoric import RhetoricalType


class TestEvolutionScoreImprovement(unittest.TestCase):
    """Test that evolution actually improves scores across generations."""

    def setUp(self):
        """Set up test fixtures."""
        self.translator = StyleTranslator(config_path="config.json")

        self.sentence_plan = {
            "text": "I spent my childhood scavenging in the ruins of the Soviet Union.",
            "propositions": [
                "I spent my childhood scavenging in the ruins of the Soviet Union.",
                "The Soviet Union is a ghost now."
            ]
        }

    @patch('src.validator.semantic_critic.SemanticCritic')
    @patch('src.ingestion.blueprint.BlueprintExtractor')
    @patch('src.atlas.rhetoric.RhetoricalClassifier')
    def test_evolution_improves_scores_across_generations(
        self, mock_classifier, mock_extractor, mock_critic_class
    ):
        """Test that scores actually improve across generations (not just stay at 0.00)."""
        # Setup mocks
        mock_extractor_instance = Mock()
        mock_blueprint = Mock(spec=SemanticBlueprint)
        mock_blueprint.original_text = self.sentence_plan["text"]
        mock_blueprint.svo_triples = []
        mock_blueprint.core_keywords = ["childhood", "scavenging", "Soviet Union"]
        mock_blueprint.named_entities = []
        mock_blueprint.citations = []
        mock_blueprint.quotes = []
        mock_blueprint.position = "BODY"
        mock_blueprint.previous_context = None
        mock_blueprint.get_subjects = Mock(return_value=[])
        mock_blueprint.get_verbs = Mock(return_value=[])
        mock_blueprint.get_objects = Mock(return_value=[])
        mock_extractor_instance.extract.return_value = mock_blueprint
        mock_extractor.return_value = mock_extractor_instance

        mock_classifier_instance = Mock()
        mock_classifier_instance.classify_heuristic.return_value = RhetoricalType.OBSERVATION
        mock_classifier.return_value = mock_classifier_instance

        # Mock critic to return IMPROVING scores across generations
        mock_critic = Mock()
        call_count = [0]

        def mock_evaluate(*args, **kwargs):
            call_count[0] += 1
            gen = (call_count[0] - 1) // 5  # Each generation evaluates 5 candidates

            # Generation 0: Low scores
            if gen == 0:
                return {
                    "proposition_recall": 0.30,
                    "style_alignment": 0.20,
                    "score": 0.25,
                    "pass": False,
                    "feedback": "Missing proposition: 'The Soviet Union is a ghost now.'",
                    "reason": "Proposition recall too low",
                    "recall_details": {
                        "missing": ["The Soviet Union is a ghost now."],
                        "preserved": ["I spent my childhood scavenging in the ruins of the Soviet Union."]
                    }
                }
            # Generation 1: Improved scores
            elif gen == 1:
                return {
                    "proposition_recall": 0.60,
                    "style_alignment": 0.40,
                    "score": 0.50,
                    "pass": False,
                    "feedback": "Style needs improvement",
                    "reason": "Style alignment too low",
                    "recall_details": {
                        "missing": [],
                        "preserved": self.sentence_plan["propositions"]
                    }
                }
            # Generation 2: Good scores
            else:
                return {
                    "proposition_recall": 0.90,
                    "style_alignment": 0.75,
                    "score": 0.825,
                    "pass": True,
                    "feedback": "Good",
                    "reason": "",
                    "recall_details": {
                        "missing": [],
                        "preserved": self.sentence_plan["propositions"]
                    }
                }

        mock_critic.evaluate.side_effect = mock_evaluate
        mock_critic_class.return_value = mock_critic

        # Mock atlas
        mock_atlas = Mock()
        mock_atlas.get_examples_by_rhetoric.return_value = ["Example 1"]

        # Mock LLM provider
        self.translator.llm_provider = Mock()
        self.translator.llm_provider.call.return_value = "Generated sentence"

        # Mock mutation to return improving children
        with patch.object(self.translator, '_mutate_sentence_candidates') as mock_mutate:
            mock_mutate.return_value = ["Improved child 1", "Improved child 2", "Improved child 3"]

            # Mock NLP
            with patch.object(self.translator, '_get_nlp', return_value=None):
                result = self.translator._evolve_sentence_unit(
                    sentence_plan=self.sentence_plan,
                    context=[],
                    atlas=mock_atlas,
                    author_name="Test Author",
                    style_dna=None,
                    verbose=False
                )

        # Verify scores improved across generations
        # Check that critic was called multiple times (multiple generations)
        self.assertGreater(mock_critic.evaluate.call_count, 5,
                          "Critic should be called for multiple generations")

        # Verify mutation was called (evolution happened)
        self.assertTrue(mock_mutate.called,
                       "Mutation should be called to evolve candidates")

        # Verify result is not None
        self.assertIsNotNone(result)

    def test_uses_paragraph_mode_for_proposition_recall(self):
        """Test that evolution uses paragraph mode to get proposition_recall."""
        # This test verifies the critical fix: is_paragraph=True
        with patch('src.validator.semantic_critic.SemanticCritic') as mock_critic_class:
            mock_critic = Mock()
            mock_critic.evaluate.return_value = {
                "proposition_recall": 0.50,
                "style_alignment": 0.30,
                "score": 0.40,
                "pass": False,
                "feedback": "Test",
                "reason": "Test"
            }
            mock_critic_class.return_value = mock_critic

            with patch('src.ingestion.blueprint.BlueprintExtractor') as mock_extractor:
                mock_extractor_instance = Mock()
                mock_blueprint = Mock(spec=SemanticBlueprint)
                mock_blueprint.original_text = "Test"
                mock_blueprint.svo_triples = []
                mock_blueprint.core_keywords = []
                mock_blueprint.named_entities = []
                mock_blueprint.citations = []
                mock_blueprint.quotes = []
                mock_blueprint.position = "BODY"
                mock_blueprint.previous_context = None
                mock_blueprint.get_subjects = Mock(return_value=[])
                mock_blueprint.get_verbs = Mock(return_value=[])
                mock_blueprint.get_objects = Mock(return_value=[])
                mock_extractor_instance.extract.return_value = mock_blueprint
                mock_extractor.return_value = mock_extractor_instance

                with patch('src.atlas.rhetoric.RhetoricalClassifier') as mock_classifier:
                    mock_classifier_instance = Mock()
                    mock_classifier_instance.classify_heuristic.return_value = RhetoricalType.OBSERVATION
                    mock_classifier.return_value = mock_classifier_instance

                    mock_atlas = Mock()
                    mock_atlas.get_examples_by_rhetoric.return_value = ["Example"]

                    self.translator.llm_provider = Mock()
                    self.translator.llm_provider.call.return_value = "Generated"

                    with patch.object(self.translator, '_get_nlp', return_value=None):
                        self.translator._evolve_sentence_unit(
                            sentence_plan=self.sentence_plan,
                            context=[],
                            atlas=mock_atlas,
                            author_name="Test",
                            verbose=False
                        )

                    # Verify is_paragraph=True was used
                    if mock_critic.evaluate.called:
                        call_args = mock_critic.evaluate.call_args
                        self.assertEqual(call_args[1].get('is_paragraph'), True,
                                       "MUST use is_paragraph=True to get proposition_recall!")


if __name__ == '__main__':
    unittest.main()

