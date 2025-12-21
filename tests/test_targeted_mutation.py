"""Test targeted mutation (replacement for breeding logic).

This test ensures that:
1. _breed_children is removed
2. _mutate_elite exists and works correctly
3. Style lexicon is injected into prompts
4. Each elite is mutated independently
5. Evolution loop uses targeted mutation, not breeding
"""

import unittest
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, List
import inspect

from src.generator.translator import StyleTranslator
from src.ingestion.blueprint import SemanticBlueprint
from src.atlas.rhetoric import RhetoricalType


class TestTargetedMutation(unittest.TestCase):
    """Test targeted mutation implementation."""

    def setUp(self):
        """Set up test fixtures."""
        self.config = {
            "evolutionary": {
                "max_generations": 10
            },
            "semantic_critic": {
                "recall_threshold": 0.90
            },
            "paragraph_fusion": {
                "style_alignment_threshold": 0.7
            },
            "translator": {
                "max_tokens": 200
            }
        }

        # Mock LLM provider
        self.mock_llm = Mock()
        self.mock_llm.call.return_value = "Variation 1: Test sentence 1\nVariation 2: Test sentence 2\nVariation 3: Test sentence 3"

        # Create translator with mocked dependencies
        with patch('src.generator.translator.LLMProvider') as mock_llm_provider:
            mock_llm_provider.return_value = self.mock_llm
            self.translator = StyleTranslator(config_path="config.json")
            self.translator.config = self.config
            self.translator.translator_config = self.config.get("translator", {})
            self.translator.paragraph_fusion_config = self.config.get("paragraph_fusion", {})
            self.translator.llm_provider = self.mock_llm

    def test_breed_children_removed(self):
        """Test that _breed_children method is removed."""
        # Check that _breed_children does not exist
        self.assertFalse(
            hasattr(self.translator, '_breed_children'),
            "_breed_children method should be removed"
        )

        # Check source code doesn't contain the method definition
        source = inspect.getsource(self.translator.__class__)
        self.assertNotIn(
            "def _breed_children",
            source,
            "Source code should not contain _breed_children method definition"
        )

    def test_mutate_elite_exists(self):
        """Test that _mutate_elite method exists."""
        self.assertTrue(
            hasattr(self.translator, '_mutate_elite'),
            "_mutate_elite method should exist"
        )

        # Verify it's callable
        self.assertTrue(
            callable(getattr(self.translator, '_mutate_elite')),
            "_mutate_elite should be callable"
        )

    def test_mutate_elite_injects_style_lexicon(self):
        """Test that _mutate_elite injects style lexicon into prompts."""
        style_lexicon = ["struggle", "conditions", "material", "dialectic", "contradiction"]
        captured_prompts = []

        def mock_llm_call(system_prompt, user_prompt, **kwargs):
            """Capture prompts to verify style lexicon injection."""
            captured_prompts.append(user_prompt)
            return "Variation 1: Test\nVariation 2: Test\nVariation 3: Test"

        self.mock_llm.call.side_effect = mock_llm_call

        candidate = {
            "text": "Test sentence.",
            "recall": 1.0,
            "style_score": 0.35,
            "feedback": "Style alignment low",
            "style_details": {
                "lexicon_density": 0.15,
                "similarity": 0.40,
                "avg_sentence_length": 12.0,
                "staccato_penalty": 0.1
            }
        }

        mock_blueprint = SemanticBlueprint(
            original_text="Test",
            svo_triples=[],
            named_entities=[],
            core_keywords=set(),
            citations=[],
            quotes=[]
        )

        with patch('src.generator.translator._load_prompt_template', return_value="System: {author_name}"), \
             patch.object(self.translator, '_build_prompt', return_value="Base prompt"):

            result = self.translator._mutate_elite(
                candidate=candidate,
                propositions=["Test proposition."],
                blueprint=mock_blueprint,
                author_name="Mao",
                style_dna=None,
                rhetorical_type=RhetoricalType.NARRATIVE,
                examples=["Example sentence."],
                style_lexicon=style_lexicon,
                num_variations=3,
                verbose=False
            )

            # Verify style lexicon was injected
            self.assertGreater(len(captured_prompts), 0, "LLM should be called")
            if captured_prompts:
                last_prompt = captured_prompts[-1]
                # Check for style lexicon injection
                self.assertIn(
                    "Use these Author Words",
                    last_prompt,
                    "Prompt should contain 'Use these Author Words'"
                )
                # Check that lexicon words appear
                for word in style_lexicon[:3]:  # Check first 3 words
                    self.assertIn(
                        word,
                        last_prompt,
                        f"Style lexicon word '{word}' should be in prompt"
                    )

    def test_mutate_elite_targets_specific_issues(self):
        """Test that _mutate_elite builds targeted instructions based on style_details."""
        captured_prompts = []

        def mock_llm_call(system_prompt, user_prompt, **kwargs):
            """Capture prompts."""
            captured_prompts.append(user_prompt)
            return "Variation 1: Test\nVariation 2: Test"

        self.mock_llm.call.side_effect = mock_llm_call

        # Candidate with low lexicon density
        candidate = {
            "text": "Test sentence.",
            "recall": 1.0,
            "style_score": 0.35,
            "feedback": "Lexicon density too low",
            "style_details": {
                "lexicon_density": 0.15,  # Below 0.3 threshold
                "similarity": 0.50,  # Below 0.7 threshold
                "avg_sentence_length": 10.0,  # Below 15
                "staccato_penalty": 0.2
            }
        }

        mock_blueprint = SemanticBlueprint(
            original_text="Test",
            svo_triples=[],
            named_entities=[],
            core_keywords=set(),
            citations=[],
            quotes=[]
        )

        with patch('src.generator.translator._load_prompt_template', return_value="System: {author_name}"), \
             patch.object(self.translator, '_build_prompt', return_value="Base prompt"):

            result = self.translator._mutate_elite(
                candidate=candidate,
                propositions=["Test proposition."],
                blueprint=mock_blueprint,
                author_name="Mao",
                style_dna=None,
                rhetorical_type=RhetoricalType.NARRATIVE,
                examples=["Example sentence."],
                style_lexicon=["word1", "word2"],
                num_variations=2,
                verbose=False
            )

            # Verify targeted instructions are present
            self.assertGreater(len(captured_prompts), 0)
            if captured_prompts:
                last_prompt = captured_prompts[-1]
                # Should mention lexicon density issue
                self.assertIn(
                    "Lexicon Density",
                    last_prompt,
                    "Prompt should mention Lexicon Density"
                )
                # Should mention similarity issue
                self.assertIn(
                    "Similarity",
                    last_prompt,
                    "Prompt should mention Similarity"
                )
                # Should mention sentence length issue
                self.assertIn(
                    "Sentence Length",
                    last_prompt,
                    "Prompt should mention Sentence Length"
                )
                # Should have targeted instructions
                self.assertIn(
                    "INJECT AUTHOR VOCABULARY",
                    last_prompt,
                    "Prompt should have instruction to inject vocabulary"
                )

    def test_evolution_loop_uses_targeted_mutation(self):
        """Test that evolution loop structure uses _mutate_elite (not _breed_children)."""
        import inspect

        # Check source code to verify the loop structure uses _mutate_elite
        source = inspect.getsource(self.translator._evolve_sentence_unit)

        # Verify _mutate_elite is called in the loop
        self.assertIn(
            "_mutate_elite",
            source,
            "Evolution loop should call _mutate_elite"
        )

        # Verify _breed_children is NOT called
        self.assertNotIn(
            "_breed_children",
            source,
            "Evolution loop should NOT call _breed_children (removed)"
        )

        # Verify elites are selected (top 3)
        self.assertIn(
            "elites = scored_candidates[:3]",
            source,
            "Should select top 3 elites"
        )

        # Verify loop mutates each elite independently
        self.assertIn(
            "for",
            source.split("elites = scored_candidates[:3]")[1].split("generation += 1")[0],
            "Should loop through elites to mutate each one"
        )

        # Verify next_gen_candidates is used (not new_children)
        self.assertIn(
            "next_gen_candidates",
            source,
            "Should use next_gen_candidates variable"
        )

        self.assertNotIn(
            "new_children",
            source,
            "Should NOT use new_children variable (fixed)"
        )

    def test_evolution_loop_selects_top_3_elites(self):
        """Test that evolution loop selects top 3 elites (not just 2)."""
        # Check source code for elite selection
        source = inspect.getsource(self.translator._evolve_sentence_unit)

        # Should select top 3, not top 2
        self.assertIn(
            "elites = scored_candidates[:3]",
            source,
            "Should select top 3 elites, not top 2"
        )

        # Should NOT have old "top 2" logic
        self.assertNotIn(
            "elites = scored_candidates[:2]",
            source,
            "Should not use old top 2 elite selection"
        )

    def test_mutate_elite_handles_emergency_repair(self):
        """Test that _mutate_elite handles recall=0.00 emergency repair."""
        captured_prompts = []

        def mock_llm_call(system_prompt, user_prompt, **kwargs):
            """Capture prompts."""
            captured_prompts.append(user_prompt)
            return "Variation 1: Emergency repair\nVariation 2: Emergency repair"

        self.mock_llm.call.side_effect = mock_llm_call

        # Candidate with recall=0.00
        candidate = {
            "text": "Wrong sentence.",
            "recall": 0.0,
            "style_score": 0.0,
            "feedback": "Failed to preserve meaning",
            "style_details": {}
        }

        mock_blueprint = SemanticBlueprint(
            original_text="Test",
            svo_triples=[],
            named_entities=[],
            core_keywords=set(),
            citations=[],
            quotes=[]
        )

        with patch('src.generator.translator._load_prompt_template', return_value="System: {author_name}"), \
             patch.object(self.translator, '_build_prompt', return_value="Base prompt"):

            result = self.translator._mutate_elite(
                candidate=candidate,
                propositions=["Must include this proposition."],
                blueprint=mock_blueprint,
                author_name="Mao",
                style_dna=None,
                rhetorical_type=RhetoricalType.NARRATIVE,
                examples=[],
                style_lexicon=[],
                num_variations=2,
                verbose=False
            )

            # Verify emergency repair mode was triggered
            self.assertGreater(len(captured_prompts), 0)
            if captured_prompts:
                last_prompt = captured_prompts[-1]
                self.assertIn(
                    "EMERGENCY REPAIR MODE",
                    last_prompt,
                    "Should trigger emergency repair for recall=0.00"
                )
                self.assertIn(
                    "Must include this proposition",
                    last_prompt,
                    "Should include propositions in emergency repair"
                )


if __name__ == "__main__":
    unittest.main()

