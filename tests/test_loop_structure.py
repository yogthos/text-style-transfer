"""Test that loop structure matches plan (next_generation, elites, mutations)."""

import unittest
import inspect
from src.generator.translator import StyleTranslator


class TestLoopStructure(unittest.TestCase):
    """Test that evolutionary loop structure matches plan exactly."""

    def setUp(self):
        """Set up test fixtures."""
        self.translator = StyleTranslator(config_path="config.json")

    def test_next_generation_variable_used(self):
        """Test that loop uses 'next_generation' variable (not 'next_gen_candidates')."""
        # Get source code of _evolve_sentence_unit
        source = inspect.getsource(self.translator._evolve_sentence_unit)

        # Verify 'next_generation' is used
        self.assertIn(
            "next_generation",
            source,
            "Loop should use 'next_generation' variable (matches plan)"
        )

        # Verify old variable name is not used
        self.assertNotIn(
            "next_gen_candidates",
            source,
            "Loop should NOT use 'next_gen_candidates' (replaced with 'next_generation')"
        )

    def test_elites_preserved(self):
        """Test that elites are preserved in next generation (elitism)."""
        source = inspect.getsource(self.translator._evolve_sentence_unit)

        # Verify elites are selected
        self.assertIn(
            "elites = scored_candidates[:3]",
            source,
            "Should select top 3 elites"
        )

        # Verify elites are extended to next_generation
        self.assertIn(
            "next_generation.extend([e",
            source,
            "Should extend next_generation with elites (elitism)"
        )

    def test_mutations_added_to_next_generation(self):
        """Test that mutations are added to next_generation."""
        source = inspect.getsource(self.translator._evolve_sentence_unit)

        # Verify mutations are extended to next_generation
        self.assertIn(
            "next_generation.extend(mutations)",
            source,
            "Should extend next_generation with mutations"
        )

        # Verify _mutate_elite is called
        self.assertIn(
            "_mutate_elite",
            source,
            "Should call _mutate_elite to generate mutations"
        )

    def test_feedback_and_style_gaps_extracted(self):
        """Test that feedback and style_gaps are extracted explicitly before mutation."""
        source = inspect.getsource(self.translator._evolve_sentence_unit)

        # Verify feedback is extracted
        self.assertIn(
            "feedback = elite.get",
            source,
            "Should extract feedback explicitly before mutation"
        )

        # Verify style_gaps is extracted
        self.assertIn(
            "style_gaps = elite.get",
            source,
            "Should extract style_gaps explicitly before mutation"
        )

    def test_candidates_assigned_from_next_generation(self):
        """Test that candidates is assigned from next_generation."""
        source = inspect.getsource(self.translator._evolve_sentence_unit)

        # Verify candidates is assigned from next_generation
        self.assertIn(
            "candidates = next_generation",
            source,
            "Should assign candidates from next_generation"
        )


if __name__ == '__main__':
    unittest.main()

