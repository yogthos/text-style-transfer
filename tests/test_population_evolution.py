"""Unit tests for population-based sentence evolution."""

import unittest
from unittest.mock import Mock, patch, MagicMock
from src.generator.translator import StyleTranslator
from src.ingestion.blueprint import SemanticBlueprint
from src.atlas.rhetoric import RhetoricalType


class TestPopulationEvolution(unittest.TestCase):
    """Test population-based evolutionary loop."""

    def setUp(self):
        """Set up test fixtures."""
        self.translator = StyleTranslator(config_path="config.json")

        # Mock sentence plan
        self.sentence_plan = {
            "text": "I spent my childhood scavenging in the ruins of the Soviet Union.",
            "propositions": [
                "I spent my childhood scavenging in the ruins of the Soviet Union.",
                "The Soviet Union is a ghost now."
            ]
        }

        # Mock blueprint
        self.blueprint = Mock(spec=SemanticBlueprint)
        self.blueprint.original_text = self.sentence_plan["text"]
        self.blueprint.svo_triples = []
        self.blueprint.core_keywords = ["childhood", "scavenging", "Soviet Union"]
        self.blueprint.named_entities = []
        self.blueprint.citations = []
        self.blueprint.quotes = []
        self.blueprint.position = "BODY"
        self.blueprint.previous_context = None
        self.blueprint.get_subjects = Mock(return_value=[])
        self.blueprint.get_verbs = Mock(return_value=[])
        self.blueprint.get_objects = Mock(return_value=[])

    def test_parse_variations_from_response_json(self):
        """Test parsing variations from JSON response."""
        # Test JSON array format
        response = '["Sentence 1", "Sentence 2", "Sentence 3"]'
        variations = self.translator._parse_variations_from_response(response, 3)
        self.assertEqual(len(variations), 3)
        self.assertEqual(variations[0], "Sentence 1")
        self.assertEqual(variations[1], "Sentence 2")
        self.assertEqual(variations[2], "Sentence 3")

    def test_parse_variations_from_response_numbered(self):
        """Test parsing variations from numbered list."""
        response = """Variation 1: First sentence here.
Variation 2: Second sentence here.
Variation 3: Third sentence here."""
        variations = self.translator._parse_variations_from_response(response, 3)
        self.assertEqual(len(variations), 3)
        self.assertIn("First sentence", variations[0])
        self.assertIn("Second sentence", variations[1])
        self.assertIn("Third sentence", variations[2])

    def test_parse_variations_from_response_list_format(self):
        """Test parsing variations from numbered list format."""
        response = """1. First sentence
2. Second sentence
3. Third sentence"""
        variations = self.translator._parse_variations_from_response(response, 3)
        self.assertEqual(len(variations), 3)
        self.assertIn("First sentence", variations[0])
        self.assertIn("Second sentence", variations[1])

    def test_parse_variations_from_response_fallback(self):
        """Test parsing variations fallback to line splitting."""
        response = """Line one
Line two
Line three"""
        variations = self.translator._parse_variations_from_response(response, 2)
        self.assertEqual(len(variations), 2)
        self.assertEqual(variations[0], "Line one")
        self.assertEqual(variations[1], "Line two")

    @patch.object(StyleTranslator, '_parse_variations_from_response')
    def test_generate_emergency_repair_candidates(self, mock_parse):
        """Test emergency repair candidate generation."""
        self.translator.llm_provider = Mock()
        self.translator.llm_provider.call.return_value = "Sentence 1: Test\nSentence 2: Test2"
        mock_parse.return_value = ["Test sentence 1", "Test sentence 2"]

        propositions = ["Fact 1", "Fact 2"]
        candidates = self.translator._generate_emergency_repair_candidates(
            propositions=propositions,
            count=2,
            verbose=False
        )

        self.assertEqual(len(candidates), 2)
        self.assertEqual(candidates[0], "Test sentence 1")
        self.assertEqual(candidates[1], "Test sentence 2")

        # Verify LLM was called with emergency repair prompt
        call_args = self.translator.llm_provider.call.call_args
        self.assertIn("EMERGENCY REPAIR", call_args[1]["user_prompt"])
        self.assertIn("Fact 1", call_args[1]["user_prompt"])
        self.assertIn("Fact 2", call_args[1]["user_prompt"])

    @patch.object(StyleTranslator, '_parse_variations_from_response')
    @patch.object(StyleTranslator, '_build_prompt')
    def test_mutate_sentence_candidates_emergency_mode(self, mock_build, mock_parse):
        """Test mutation with emergency repair mode (recall=0.00)."""
        self.translator.llm_provider = Mock()
        mock_build.return_value = "Base prompt"
        self.translator.llm_provider.call.return_value = "Variation 1: Test"
        mock_parse.return_value = ["Test variation 1", "Test variation 2"]

        parents = [{"text": "Bad sentence", "recall": 0.0, "style_score": 0.0}]

        candidates = self.translator._mutate_sentence_candidates(
            parents=parents,
            feedback="Recall too low",
            propositions=["Fact 1"],
            blueprint=self.blueprint,
            author_name="Test Author",
            style_dna=None,
            rhetorical_type=RhetoricalType.OBSERVATION,
            examples=[],
            count=2,
            verbose=False
        )

        self.assertEqual(len(candidates), 2)
        # Verify emergency repair mode was triggered
        call_args = self.translator.llm_provider.call.call_args
        self.assertIn("EMERGENCY REPAIR MODE", call_args[1]["user_prompt"])

    @patch.object(StyleTranslator, '_parse_variations_from_response')
    @patch.object(StyleTranslator, '_build_prompt')
    def test_mutate_sentence_candidates_normal_mode(self, mock_build, mock_parse):
        """Test mutation in normal mode (recall > 0.00)."""
        self.translator.llm_provider = Mock()
        mock_build.return_value = "Base prompt"
        self.translator.llm_provider.call.return_value = "Variation 1: Test"
        mock_parse.return_value = ["Test variation 1", "Test variation 2"]

        parents = [{"text": "Good sentence", "recall": 0.5, "style_score": 0.3}]

        candidates = self.translator._mutate_sentence_candidates(
            parents=parents,
            feedback="Style too low",
            propositions=["Fact 1"],
            blueprint=self.blueprint,
            author_name="Test Author",
            style_dna=None,
            rhetorical_type=RhetoricalType.OBSERVATION,
            examples=[],
            count=2,
            verbose=False
        )

        self.assertEqual(len(candidates), 2)
        # Verify normal mutation mode was used
        call_args = self.translator.llm_provider.call.call_args
        self.assertIn("You are refining a sentence", call_args[1]["user_prompt"])
        self.assertIn("Good sentence", call_args[1]["user_prompt"])
        self.assertIn("Style too low", call_args[1]["user_prompt"])

    @patch.object(StyleTranslator, '_mutate_sentence_candidates')
    @patch.object(StyleTranslator, '_generate_emergency_repair_candidates')
    @patch('src.validator.semantic_critic.SemanticCritic')
    @patch('src.ingestion.blueprint.BlueprintExtractor')
    @patch('src.atlas.rhetoric.RhetoricalClassifier')
    def test_evolution_loop_terminates_on_success(
        self, mock_classifier, mock_extractor, mock_critic_class,
        mock_emergency, mock_mutate
    ):
        """Test that evolution loop terminates when thresholds are met."""
        # Setup mocks
        mock_extractor_instance = Mock()
        mock_extractor_instance.extract.return_value = self.blueprint
        mock_extractor.return_value = mock_extractor_instance

        mock_classifier_instance = Mock()
        mock_classifier_instance.classify_heuristic.return_value = RhetoricalType.OBSERVATION
        mock_classifier.return_value = mock_classifier_instance

        # Mock critic to return passing scores
        mock_critic = Mock()
        mock_critic.evaluate.return_value = {
            "proposition_recall": 0.90,
            "style_alignment": 0.80,
            "score": 0.85,
            "pass": True,
            "feedback": "",
            "reason": ""
        }
        mock_critic_class.return_value = mock_critic

        # Mock atlas
        mock_atlas = Mock()
        mock_atlas.get_examples_by_rhetoric.return_value = ["Example 1", "Example 2"]

        # Mock LLM provider
        self.translator.llm_provider = Mock()
        self.translator.llm_provider.call.return_value = "Generated sentence"

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

        # Should return the generated sentence (meets thresholds)
        self.assertIsNotNone(result)
        # Verify critic was instantiated and evaluate was called
        # The critic is instantiated inside _evolve_sentence_unit, so we check the class was called
        self.assertTrue(mock_critic_class.called)
        # If critic was instantiated, evaluate should have been called
        if mock_critic_class.called:
            instantiated_critic = mock_critic_class.return_value
            self.assertTrue(instantiated_critic.evaluate.called)

    @patch.object(StyleTranslator, '_mutate_sentence_candidates')
    @patch.object(StyleTranslator, '_generate_emergency_repair_candidates')
    @patch('src.generator.translator.SemanticCritic')
    @patch('src.ingestion.blueprint.BlueprintExtractor')
    @patch('src.atlas.rhetoric.RhetoricalClassifier')
    def test_evolution_loop_triggers_emergency_repair(
        self, mock_classifier, mock_extractor, mock_critic_class,
        mock_emergency, mock_mutate
    ):
        """Test that emergency repair is triggered when recall=0.00."""
        # Setup mocks
        mock_extractor_instance = Mock()
        mock_extractor_instance.extract.return_value = self.blueprint
        mock_extractor.return_value = mock_extractor_instance

        mock_classifier_instance = Mock()
        mock_classifier_instance.classify_heuristic.return_value = RhetoricalType.OBSERVATION
        mock_classifier.return_value = mock_classifier_instance

        # Mock critic to return recall=0.00 initially, then better after emergency repair
        mock_critic = Mock()
        call_count = [0]
        def mock_evaluate(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] <= 5:  # Initial population
                return {
                    "proposition_recall": 0.00,
                    "style_alignment": 0.00,
                    "score": 0.00,
                    "pass": False,
                    "feedback": "Recall too low",
                    "reason": "Missing propositions"
                }
            else:  # After emergency repair
                return {
                    "proposition_recall": 0.50,
                    "style_alignment": 0.30,
                    "score": 0.40,
                    "pass": False,
                    "feedback": "Improving",
                    "reason": ""
                }
        mock_critic.evaluate.side_effect = mock_evaluate
        mock_critic_class.return_value = mock_critic

        # Mock emergency repair
        mock_emergency.return_value = ["Emergency sentence 1", "Emergency sentence 2"]

        # Mock atlas
        mock_atlas = Mock()
        mock_atlas.get_examples_by_rhetoric.return_value = ["Example 1"]

        # Mock LLM provider
        self.translator.llm_provider = Mock()
        self.translator.llm_provider.call.return_value = "Generated sentence"

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

        # Verify emergency repair was called
        mock_emergency.assert_called_once()
        self.assertIsNotNone(result)

    @patch.object(StyleTranslator, '_mutate_sentence_candidates')
    @patch('src.generator.translator.SemanticCritic')
    @patch('src.ingestion.blueprint.BlueprintExtractor')
    @patch('src.atlas.rhetoric.RhetoricalClassifier')
    def test_evolution_loop_breeds_children(
        self, mock_classifier, mock_extractor, mock_critic_class, mock_mutate
    ):
        """Test that evolution loop breeds new children from elites."""
        # Setup mocks
        mock_extractor_instance = Mock()
        mock_extractor_instance.extract.return_value = self.blueprint
        mock_extractor.return_value = mock_extractor_instance

        mock_classifier_instance = Mock()
        mock_classifier_instance.classify_heuristic.return_value = RhetoricalType.OBSERVATION
        mock_classifier.return_value = mock_classifier_instance

        # Mock critic to return improving scores
        mock_critic = Mock()
        call_count = [0]
        def mock_evaluate(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] <= 5:  # Initial population
                return {
                    "proposition_recall": 0.50,
                    "style_alignment": 0.30,
                    "score": 0.40,
                    "pass": False,
                    "feedback": "Needs improvement",
                    "reason": "Recall too low"
                }
            else:  # After breeding
                return {
                    "proposition_recall": 0.75,
                    "style_alignment": 0.50,
                    "score": 0.625,
                    "pass": False,
                    "feedback": "Better",
                    "reason": ""
                }
        mock_critic.evaluate.side_effect = mock_evaluate
        mock_critic_class.return_value = mock_critic

        # Mock mutation
        mock_mutate.return_value = ["Child 1", "Child 2", "Child 3"]

        # Mock atlas
        mock_atlas = Mock()
        mock_atlas.get_examples_by_rhetoric.return_value = ["Example 1"]

        # Mock LLM provider
        self.translator.llm_provider = Mock()
        self.translator.llm_provider.call.return_value = "Generated sentence"

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

        # Verify mutation was called (breeding happened)
        self.assertTrue(mock_mutate.called)
        self.assertIsNotNone(result)


if __name__ == '__main__':
    unittest.main()

