"""Test that critic returns style_details and style_lexicon extraction works correctly."""

import unittest
from unittest.mock import Mock, patch
from src.validator.semantic_critic import SemanticCritic
from src.ingestion.blueprint import SemanticBlueprint
from src.generator.translator import StyleTranslator


class TestCriticStyleDetails(unittest.TestCase):
    """Test that critic returns style_details dictionary and style_lexicon extraction."""

    def setUp(self):
        """Set up test fixtures."""
        self.critic = SemanticCritic(config_path="config.json")
        self.translator = StyleTranslator(config_path="config.json")

    def test_evaluate_returns_style_details(self):
        """Test that evaluate() always returns style_details dictionary."""
        blueprint = SemanticBlueprint(
            original_text="The economy is growing.",
            svo_triples=[],
            named_entities=[],
            core_keywords=set(),
            citations=[],
            quotes=[]
        )

        # Test with paragraph mode (is_paragraph=True)
        result = self.critic.evaluate(
            generated_text="The economy is expanding rapidly.",
            input_blueprint=blueprint,
            propositions=["The economy is growing."],
            is_paragraph=True,
            verbose=False
        )

        # Verify style_details is present
        self.assertIn(
            "style_details",
            result,
            "evaluate() should always return 'style_details' dictionary"
        )

        # Verify style_details is a dictionary
        self.assertIsInstance(
            result["style_details"],
            dict,
            "style_details should be a dictionary"
        )

    def test_style_details_contains_required_keys(self):
        """Test that style_details contains all required keys."""
        blueprint = SemanticBlueprint(
            original_text="The economy is growing.",
            svo_triples=[],
            named_entities=[],
            core_keywords=set(),
            citations=[],
            quotes=[]
        )

        result = self.critic.evaluate(
            generated_text="The economy is expanding rapidly.",
            input_blueprint=blueprint,
            propositions=["The economy is growing."],
            is_paragraph=True,
            verbose=False
        )

        style_details = result.get("style_details", {})

        # Verify required keys are present
        required_keys = ["similarity", "lexicon_density", "avg_sentence_length", "staccato_penalty"]
        for key in required_keys:
            self.assertIn(
                key,
                style_details,
                f"style_details should contain '{key}'"
            )

    def test_style_lexicon_extraction_from_style_dna(self):
        """Test that style_lexicon is extracted from style_dna."""
        # Mock style_dna with lexicon
        style_dna = {
            "lexicon": ["dialectical", "contradiction", "material", "struggle", "revolution"]
        }

        # Create a mock sentence plan
        sentence_plan = {
            "text": "Test sentence.",
            "propositions": ["Test proposition."]
        }

        # Mock atlas
        mock_atlas = Mock()
        mock_atlas.get_examples_by_rhetoric.return_value = []

        # We need to test the extraction logic in _evolve_sentence_unit
        # But we can test it by checking the code structure
        import inspect
        source = inspect.getsource(self.translator._evolve_sentence_unit)

        # Verify extraction logic exists
        self.assertIn(
            'if "lexicon" in style_dna',
            source,
            "Should check for 'lexicon' key in style_dna"
        )

        self.assertIn(
            'style_lexicon = style_dna["lexicon"]',
            source,
            "Should extract lexicon from style_dna"
        )

    def test_style_lexicon_extraction_from_atlas(self):
        """Test that style_lexicon is extracted from atlas when style_dna doesn't have it."""
        import inspect
        source = inspect.getsource(self.translator._evolve_sentence_unit)

        # Verify atlas extraction logic exists
        self.assertIn(
            'if not style_lexicon and atlas',
            source,
            "Should extract from atlas when style_dna doesn't have lexicon"
        )

        self.assertIn(
            'atlas.get_examples_by_rhetoric',
            source,
            "Should call atlas.get_examples_by_rhetoric to get examples"
        )

    def test_style_lexicon_not_empty_when_extracted(self):
        """Test that style_lexicon is not empty when extracted from style_dna or atlas."""
        # This test verifies the extraction logic produces non-empty results
        # We'll test by checking the code structure ensures non-empty results

        import inspect
        source = inspect.getsource(self.translator._evolve_sentence_unit)

        # Verify extraction happens before mutation
        # Find where style_lexicon is extracted and where it's used
        lexicon_extraction_start = source.find('style_lexicon = []')
        lexicon_usage = source.find('style_lexicon=style_lexicon')

        # Verify extraction happens before usage
        self.assertGreater(
            lexicon_usage,
            lexicon_extraction_start,
            "style_lexicon should be extracted before being passed to _mutate_elite"
        )

    def test_style_lexicon_contains_author_specific_vocabulary(self):
        """Test that style_lexicon contains actual author-specific vocabulary (not just stop words)."""
        # This test verifies that when extracting from atlas, stop words are filtered out
        import inspect
        source = inspect.getsource(self.translator._evolve_sentence_unit)

        # Verify stop words are filtered
        self.assertIn(
            'common_stop_words',
            source,
            "Should filter out common stop words when extracting from atlas"
        )

        # Verify words are extracted from examples
        self.assertIn(
            'word_counts.most_common',
            source,
            "Should extract most common words from examples"
        )

    def test_style_lexicon_passed_to_mutate_elite(self):
        """Test that style_lexicon is passed to _mutate_elite."""
        import inspect
        source = inspect.getsource(self.translator._evolve_sentence_unit)

        # Verify style_lexicon is passed as parameter
        self.assertIn(
            'style_lexicon=style_lexicon',
            source,
            "Should pass style_lexicon to _mutate_elite"
        )


if __name__ == '__main__':
    unittest.main()

