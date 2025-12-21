"""Test Shoehorn prompt format in _mutate_elite with author-specific style_lexicon."""

import unittest
from unittest.mock import Mock, patch
from src.generator.translator import StyleTranslator
from src.ingestion.blueprint import SemanticBlueprint
from src.atlas.rhetoric import RhetoricalType


class TestShoehornPrompt(unittest.TestCase):
    """Test that _mutate_elite uses Shoehorn prompt format with author-specific style_lexicon."""

    def setUp(self):
        """Set up test fixtures."""
        self.translator = StyleTranslator(config_path="config.json")
        self.mock_llm = Mock()
        self.translator.llm_provider = self.mock_llm

    def test_shoehorn_prompt_includes_critical_directive(self):
        """Test that mutation prompt includes 'CRITICAL: You MUST inject' when lexicon_score < 0.3."""
        # Create candidate with low lexicon density
        candidate = {
            "text": "The economy is growing.",
            "feedback": "Style alignment low.",
            "style_details": {
                "lexicon_density": 0.15,  # Below 0.3 threshold
                "similarity": 0.5,
                "avg_sentence_length": 10.0,
                "staccato_penalty": 0.0
            },
            "recall": 0.9
        }

        # Author-specific style lexicon
        style_lexicon = ["dialectical", "contradiction", "material", "struggle", "revolution"]

        blueprint = SemanticBlueprint(
            original_text="Test",
            svo_triples=[],
            named_entities=[],
            core_keywords=set(),
            citations=[],
            quotes=[]
        )

        # Mock LLM response
        self.mock_llm.call.return_value = "Variation 1: Test sentence 1\nVariation 2: Test sentence 2"

        with patch('src.generator.translator._load_prompt_template', return_value="System: {author_name}"), \
             patch.object(self.translator, '_build_prompt', return_value="Base prompt"):

            self.translator._mutate_elite(
                candidate=candidate,
                propositions=["Test proposition."],
                blueprint=blueprint,
                author_name="Mao",
                style_dna=None,
                rhetorical_type=RhetoricalType.ARGUMENT,
                examples=["Example sentence."],
                style_lexicon=style_lexicon,
                num_variations=2,
                verbose=False
            )

        # Verify LLM was called
        self.assertTrue(self.mock_llm.call.called, "LLM should be called")

        # Extract user prompt from call
        call_args = self.mock_llm.call.call_args
        user_prompt = call_args[1]['user_prompt'] if len(call_args) > 1 and 'user_prompt' in call_args[1] else call_args[0][1]

        # Verify prompt includes CRITICAL directive
        self.assertIn(
            "CRITICAL: The vocabulary is too simple. You MUST inject",
            user_prompt,
            "Prompt should include CRITICAL directive when lexicon_score < 0.3"
        )

    def test_shoehorn_prompt_uses_author_specific_lexicon(self):
        """Test that injected words are from style_lexicon (author-specific), not random dictionary words."""
        # Create candidate with low lexicon density
        candidate = {
            "text": "The economy is growing.",
            "feedback": "Style alignment low.",
            "style_details": {
                "lexicon_density": 0.15,  # Below 0.3 threshold
                "similarity": 0.5,
                "avg_sentence_length": 10.0,
                "staccato_penalty": 0.0
            },
            "recall": 0.9
        }

        # Author-specific style lexicon (Mao's vocabulary)
        style_lexicon = ["dialectical", "contradiction", "material", "struggle", "revolution", "class", "bourgeoisie"]

        blueprint = SemanticBlueprint(
            original_text="Test",
            svo_triples=[],
            named_entities=[],
            core_keywords=set(),
            citations=[],
            quotes=[]
        )

        # Mock LLM response
        self.mock_llm.call.return_value = "Variation 1: Test\nVariation 2: Test"

        with patch('src.generator.translator._load_prompt_template', return_value="System"), \
             patch.object(self.translator, '_build_prompt', return_value="Base prompt"):

            self.translator._mutate_elite(
                candidate=candidate,
                propositions=["Test proposition."],
                blueprint=blueprint,
                author_name="Mao",
                style_dna=None,
                rhetorical_type=RhetoricalType.ARGUMENT,
                examples=["Example."],
                style_lexicon=style_lexicon,
                num_variations=2,
                verbose=False
            )

        # Extract user prompt
        call_args = self.mock_llm.call.call_args
        user_prompt = call_args[1]['user_prompt'] if len(call_args) > 1 and 'user_prompt' in call_args[1] else call_args[0][1]

        # Verify prompt includes author-specific words from style_lexicon
        # Should include at least some of the first 15 words from style_lexicon
        top_words = style_lexicon[:15]
        found_words = [word for word in top_words if word in user_prompt]

        self.assertGreater(
            len(found_words),
            0,
            f"Prompt should include author-specific words from style_lexicon. Expected some of {top_words}, found: {found_words}"
        )

        # Verify it's not using random dictionary words (check for common words that shouldn't be in style_lexicon)
        random_words = ["apple", "banana", "car", "dog", "house"]
        found_random = [word for word in random_words if word in user_prompt and word not in style_lexicon]
        self.assertEqual(
            len(found_random),
            0,
            f"Prompt should not include random dictionary words. Found: {found_random}"
        )

    def test_shoehorn_prompt_expand_sentence_length(self):
        """Test that prompt includes 'Expand the sentence structure' when feedback mentions sentence length."""
        candidate = {
            "text": "Short sentence.",
            "feedback": "Style alignment low. Average sentence length: 5.0 words.",
            "style_details": {
                "lexicon_density": 0.4,  # Above threshold
                "similarity": 0.6,
                "avg_sentence_length": 5.0,
                "staccato_penalty": 0.2
            },
            "recall": 0.9
        }

        style_lexicon = ["word1", "word2"]

        blueprint = SemanticBlueprint(
            original_text="Test",
            svo_triples=[],
            named_entities=[],
            core_keywords=set(),
            citations=[],
            quotes=[]
        )

        self.mock_llm.call.return_value = "Variation 1: Test\nVariation 2: Test"

        with patch('src.generator.translator._load_prompt_template', return_value="System"), \
             patch.object(self.translator, '_build_prompt', return_value="Base prompt"):

            self.translator._mutate_elite(
                candidate=candidate,
                propositions=["Test proposition."],
                blueprint=blueprint,
                author_name="Mao",
                style_dna=None,
                rhetorical_type=RhetoricalType.ARGUMENT,
                examples=["Example."],
                style_lexicon=style_lexicon,
                num_variations=2,
                verbose=False
            )

        # Extract user prompt
        call_args = self.mock_llm.call.call_args
        user_prompt = call_args[1]['user_prompt'] if len(call_args) > 1 and 'user_prompt' in call_args[1] else call_args[0][1]

        # Verify prompt includes sentence length expansion directive
        self.assertIn(
            "Expand the sentence structure",
            user_prompt,
            "Prompt should include 'Expand the sentence structure' when feedback mentions sentence length"
        )

    def test_style_lexicon_not_empty_when_passed(self):
        """Test that style_lexicon is not empty when passed to _mutate_elite."""
        candidate = {
            "text": "Test sentence.",
            "feedback": "Style alignment low.",
            "style_details": {
                "lexicon_density": 0.15,
                "similarity": 0.5,
                "avg_sentence_length": 10.0,
                "staccato_penalty": 0.0
            },
            "recall": 0.9
        }

        # Non-empty style lexicon
        style_lexicon = ["dialectical", "contradiction", "material"]

        blueprint = SemanticBlueprint(
            original_text="Test",
            svo_triples=[],
            named_entities=[],
            core_keywords=set(),
            citations=[],
            quotes=[]
        )

        self.mock_llm.call.return_value = "Variation 1: Test\nVariation 2: Test"

        with patch('src.generator.translator._load_prompt_template', return_value="System"), \
             patch.object(self.translator, '_build_prompt', return_value="Base prompt"):

            # Should not raise error when style_lexicon is provided
            result = self.translator._mutate_elite(
                candidate=candidate,
                propositions=["Test proposition."],
                blueprint=blueprint,
                author_name="Mao",
                style_dna=None,
                rhetorical_type=RhetoricalType.ARGUMENT,
                examples=["Example."],
                style_lexicon=style_lexicon,
                num_variations=2,
                verbose=False
            )

            # Verify it returns variations
            self.assertIsInstance(result, list, "Should return list of variations")

        # Extract user prompt
        call_args = self.mock_llm.call.call_args
        user_prompt = call_args[1]['user_prompt'] if len(call_args) > 1 and 'user_prompt' in call_args[1] else call_args[0][1]

        # Verify style_lexicon words appear in prompt
        found_words = [word for word in style_lexicon if word in user_prompt]
        self.assertGreater(
            len(found_words),
            0,
            f"Prompt should include words from style_lexicon. Expected some of {style_lexicon}, found: {found_words}"
        )


if __name__ == '__main__':
    unittest.main()

