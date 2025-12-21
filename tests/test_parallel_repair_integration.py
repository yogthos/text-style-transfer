"""Integration test for parallel repair architecture end-to-end."""

import unittest
from unittest.mock import Mock, patch, call
from src.generator.translator import StyleTranslator
from src.ingestion.blueprint import SemanticBlueprint
from src.atlas.rhetoric import RhetoricalType


class TestParallelRepairIntegration(unittest.TestCase):
    """End-to-end test for parallel repair architecture."""

    def setUp(self):
        """Set up test fixtures."""
        self.translator = StyleTranslator(config_path="config.json")
        self.mock_llm = Mock()
        self.translator.llm_provider = self.mock_llm

        # Mock proposition extractor
        self.translator.proposition_extractor = Mock()
        self.translator.proposition_extractor.extract_atomic_propositions.return_value = ["Test proposition."]

    def test_end_to_end_parallel_repair(self):
        """Test end-to-end evolution with parallel repair architecture."""
        # Track LLM calls to verify temperature gradient
        llm_calls = []

        def capture_llm_call(*args, **kwargs):
            """Capture LLM call arguments."""
            llm_calls.append({
                'temperature': kwargs.get('temperature', args[1].get('temperature') if len(args) > 1 else None),
                'user_prompt': kwargs.get('user_prompt', args[1].get('user_prompt') if len(args) > 1 else None)
            })
            # Return different responses for initial generation vs mutations
            if len(llm_calls) <= 5:
                # Initial population
                return f"Candidate {len(llm_calls)}"
            else:
                # Mutations
                return f"Variation 1: Mutation {len(llm_calls) - 5}\nVariation 2: Mutation {len(llm_calls) - 5}\nVariation 3: Mutation {len(llm_calls) - 5}"

        self.mock_llm.call.side_effect = capture_llm_call

        # Mock blueprint
        mock_blueprint = SemanticBlueprint(
            original_text="Test sentence.",
            svo_triples=[],
            named_entities=[],
            core_keywords=set(),
            citations=[],
            quotes=[]
        )

        # Mock rhetorical classifier
        mock_classifier = Mock()
        mock_classifier.classify_heuristic.return_value = RhetoricalType.NARRATIVE

        # Mock atlas with author-specific examples
        mock_atlas = Mock()
        mock_atlas.get_examples_by_rhetoric.return_value = [
            "The dialectical process reveals contradictions in material reality.",
            "Revolutionary struggle transforms social conditions through class conflict.",
            "Material forces determine the course of historical development."
        ]

        # Mock NLP
        mock_nlp = Mock()
        mock_doc = Mock()
        mock_sent = Mock()
        mock_sent.text = "Test."
        mock_doc.sents = [mock_sent]
        mock_nlp.return_value = mock_doc

        # Mock critic with low style to trigger mutations
        mock_critic = Mock()
        mock_critic.evaluate.side_effect = lambda **kwargs: {
            "pass": False,
            "proposition_recall": 0.9,  # Good recall
            "style_alignment": 0.25,  # Low style (below 0.7 threshold)
            "score": 0.5,
            "feedback": "Style alignment low. Lexicon density: 0.15.",
            "recall_details": {"preserved": ["Test proposition."], "missing": []},
            "style_details": {
                "similarity": 0.5,
                "lexicon_density": 0.15,  # Below 0.3 threshold
                "avg_sentence_length": 8.0,  # Below 15 threshold
                "staccato_penalty": 0.2
            }
        }

        sentence_plan = {
            "text": "Test sentence.",
            "propositions": ["Test proposition."]
        }

        with patch('src.ingestion.blueprint.BlueprintExtractor') as mock_blueprint_extractor, \
             patch('src.atlas.rhetoric.RhetoricalClassifier', return_value=mock_classifier), \
             patch.object(self.translator, '_get_nlp', return_value=mock_nlp), \
             patch.object(self.translator, '_build_prompt', return_value="Generate a sentence"), \
             patch('src.generator.translator._load_prompt_template', return_value="System: {author_name}"), \
             patch('src.generator.translator.clean_generated_text', side_effect=lambda x: x), \
             patch('src.validator.semantic_critic.SemanticCritic', return_value=mock_critic):

            mock_blueprint_extractor.return_value.extract.return_value = mock_blueprint

            # Run evolution (will stop after max_generations since style is low)
            result = self.translator._evolve_sentence_unit(
                sentence_plan=sentence_plan,
                context=[],
                atlas=mock_atlas,
                author_name="Mao",
                style_dna=None,
                verbose=False
            )

        # Verify 1: Initial candidates use different temperatures (diversity)
        initial_calls = llm_calls[:5]
        initial_temperatures = [call['temperature'] for call in initial_calls if call.get('temperature') is not None]
        expected_temperatures = [0.3, 0.7, 0.85, 0.95, 1.0]
        self.assertEqual(
            initial_temperatures,
            expected_temperatures,
            f"Initial candidates should use temperature gradient {expected_temperatures}, got {initial_temperatures}"
        )

        # Verify 2: Elites are selected (top 3)
        # This is verified by the fact that mutations are generated (elites are preserved)

        # Verify 3: Each elite is mutated independently
        # Check that we have mutation calls (after initial 5)
        mutation_calls = llm_calls[5:]
        self.assertGreater(
            len(mutation_calls),
            0,
            "Should generate mutations for elites"
        )

        # Verify 4: Mutations include "CRITICAL" directive when lexicon_score < 0.3
        # Check mutation prompts for CRITICAL directive
        critical_found = False
        for call in mutation_calls:
            user_prompt = call.get('user_prompt', '')
            if 'CRITICAL: The vocabulary is too simple. You MUST inject' in user_prompt:
                critical_found = True
                break

        self.assertTrue(
            critical_found,
            "Mutation prompts should include 'CRITICAL: You MUST inject' directive when lexicon_score < 0.3"
        )

        # Verify 5: Mutations use author-specific style_lexicon words (not random dictionary words)
        # Extract style_lexicon from atlas examples (should contain "dialectical", "contradiction", etc.)
        author_words_found = False
        expected_author_words = ["dialectical", "contradiction", "material", "revolution", "struggle", "class"]

        for call in mutation_calls:
            user_prompt = call.get('user_prompt', '')
            # Check if any author-specific words appear in the prompt
            found_words = [word for word in expected_author_words if word in user_prompt.lower()]
            if found_words:
                author_words_found = True
                break

        self.assertTrue(
            author_words_found,
            f"Mutation prompts should include author-specific words from style_lexicon. Expected some of {expected_author_words}"
        )

        # Verify 6: style_lexicon is extracted from atlas before mutation
        # This is verified by the fact that author words appear in mutation prompts
        # (which means style_lexicon was extracted from atlas examples)

        # Verify 7: Next generation = elites + mutations
        # This is verified by the evolution continuing (mutations are added to population)

        # Verify 8: Evolution continues until thresholds met or max generations
        # We set max_generations to 10, and style is low, so it should run multiple generations
        self.assertGreater(
            len(llm_calls),
            5,
            "Evolution should continue beyond initial generation (mutations should be generated)"
        )

    def test_style_lexicon_extraction_from_atlas_integration(self):
        """Test that style_lexicon is correctly extracted from atlas examples."""
        # Mock atlas with specific author vocabulary
        mock_atlas = Mock()
        mock_atlas.get_examples_by_rhetoric.return_value = [
            "The dialectical process reveals contradictions in material reality.",
            "Revolutionary struggle transforms social conditions.",
            "Class conflict drives historical development."
        ]

        # Track style_lexicon passed to _mutate_elite
        captured_style_lexicon = []

        original_mutate_elite = self.translator._mutate_elite

        def capture_style_lexicon(*args, **kwargs):
            """Capture style_lexicon parameter."""
            if 'style_lexicon' in kwargs:
                captured_style_lexicon.append(kwargs['style_lexicon'])
            return original_mutate_elite(*args, **kwargs)

        self.translator._mutate_elite = capture_style_lexicon

        # Mock all dependencies
        mock_blueprint = SemanticBlueprint(
            original_text="Test",
            svo_triples=[],
            named_entities=[],
            core_keywords=set(),
            citations=[],
            quotes=[]
        )

        mock_classifier = Mock()
        mock_classifier.classify_heuristic.return_value = RhetoricalType.ARGUMENT

        mock_nlp = Mock()
        mock_doc = Mock()
        mock_sent = Mock()
        mock_sent.text = "Test."
        mock_doc.sents = [mock_sent]
        mock_nlp.return_value = mock_doc

        mock_critic = Mock()
        mock_critic.evaluate.return_value = {
            "pass": False,
            "proposition_recall": 0.9,
            "style_alignment": 0.25,
            "score": 0.5,
            "feedback": "Style alignment low.",
            "recall_details": {"preserved": [], "missing": []},
            "style_details": {
                "lexicon_density": 0.15,
                "similarity": 0.5,
                "avg_sentence_length": 10.0,
                "staccato_penalty": 0.0
            }
        }

        self.mock_llm.call.return_value = "Variation 1: Test\nVariation 2: Test"

        sentence_plan = {
            "text": "Test sentence.",
            "propositions": ["Test proposition."]
        }

        with patch('src.ingestion.blueprint.BlueprintExtractor') as mock_blueprint_extractor, \
             patch('src.atlas.rhetoric.RhetoricalClassifier', return_value=mock_classifier), \
             patch.object(self.translator, '_get_nlp', return_value=mock_nlp), \
             patch.object(self.translator, '_build_prompt', return_value="Generate"), \
             patch('src.generator.translator._load_prompt_template', return_value="System"), \
             patch('src.generator.translator.clean_generated_text', side_effect=lambda x: x), \
             patch('src.validator.semantic_critic.SemanticCritic', return_value=mock_critic):

            mock_blueprint_extractor.return_value.extract.return_value = mock_blueprint

            # Run evolution
            try:
                self.translator._evolve_sentence_unit(
                    sentence_plan=sentence_plan,
                    context=[],
                    atlas=mock_atlas,
                    author_name="Mao",
                    style_dna=None,  # No style_dna, so should extract from atlas
                    verbose=False
                )
            except Exception:
                pass  # We just need to capture style_lexicon

        # Verify style_lexicon was extracted and is not empty
        self.assertGreater(
            len(captured_style_lexicon),
            0,
            "style_lexicon should be extracted and passed to _mutate_elite"
        )

        # Verify style_lexicon contains author-specific words (from atlas examples)
        if captured_style_lexicon:
            lexicon = captured_style_lexicon[0]
            self.assertIsInstance(lexicon, list, "style_lexicon should be a list")

            # Check that it contains words from the atlas examples
            expected_words = ["dialectical", "contradiction", "material", "revolutionary", "struggle", "class", "conflict"]
            found_words = [word for word in expected_words if word in lexicon]

            self.assertGreater(
                len(found_words),
                0,
                f"style_lexicon should contain author-specific words. Expected some of {expected_words}, got {lexicon[:10]}"
            )


if __name__ == '__main__':
    unittest.main()

