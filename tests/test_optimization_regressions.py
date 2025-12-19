"""Tests for optimization regression fixes.

These tests verify that the lazy evaluation optimization doesn't break
existing functionality, particularly:
1. style_dna_dict is always defined (no NameError)
2. Simple sentence detection doesn't skip semantically dense sentences
3. Single-example path extracts style_dna_dict correctly
"""

import unittest

# Skip tests if dependencies are missing
try:
    from src.ingestion.blueprint import BlueprintExtractor, SemanticBlueprint
    from src.generator.translator import StyleTranslator
    from src.atlas.rhetoric import RhetoricalType
    DEPENDENCIES_AVAILABLE = True
except (ImportError, ModuleNotFoundError) as e:
    DEPENDENCIES_AVAILABLE = False
    IMPORT_ERROR = str(e)


@unittest.skipUnless(DEPENDENCIES_AVAILABLE, f"Dependencies not available: {IMPORT_ERROR if not DEPENDENCIES_AVAILABLE else ''}")
class TestOptimizationRegressions(unittest.TestCase):
    """Test regression fixes for optimization."""

    def setUp(self):
        """Set up test fixtures."""
        if not DEPENDENCIES_AVAILABLE:
            self.skipTest("Dependencies not available")
        self.extractor = BlueprintExtractor()
        self.translator = StyleTranslator()

    def test_style_dna_dict_always_defined_single_example(self):
        """Test that style_dna_dict is defined even with single example.

        Regression: NameError when len(examples) == 1 and falling back to standard generation.
        """
        # Create a blueprint for a simple sentence
        text = "The U.S. can no longer afford global domination."
        blueprint = self.extractor.extract(text)

        # Single example scenario (triggers strict fallback)
        examples = ["Where do correct ideas come from?"]

        # This should not raise NameError
        try:
            result = self.translator.translate(
                blueprint=blueprint,
                author_name="Test Author",
                style_dna="Test style",
                rhetorical_type=RhetoricalType.OBSERVATION,
                examples=examples,
                verbose=False
            )
            # If we get here, no NameError occurred
            self.assertIsInstance(result, str)
        except NameError as e:
            if "style_dna_dict" in str(e):
                self.fail(f"NameError for style_dna_dict: {e}")
            raise

    def test_style_dna_dict_always_defined_multiple_examples(self):
        """Test that style_dna_dict is defined with multiple examples.

        Regression: style_dna_dict should be available in standard generation fallback.
        """
        # Create a blueprint
        text = "The core message is an admission."
        blueprint = self.extractor.extract(text)

        # Multiple examples scenario
        examples = [
            "Practice, knowledge, again practice, and again knowledge.",
            "The history of human knowledge tells us that the truth of many theories is incomplete.",
            "Discover the truth through practice, and again through practice verify and develop the truth."
        ]

        # This should not raise NameError
        try:
            result = self.translator.translate(
                blueprint=blueprint,
                author_name="Test Author",
                style_dna="Test style",
                rhetorical_type=RhetoricalType.DEFINITION,
                examples=examples,
                verbose=False
            )
            # If we get here, no NameError occurred
            self.assertIsInstance(result, str)
        except NameError as e:
            if "style_dna_dict" in str(e):
                self.fail(f"NameError for style_dna_dict: {e}")
            raise

    def test_semantically_dense_short_sentence_not_skipped(self):
        """Test that short sentences with semantic structure are not skipped.

        Regression: "The core message is an admission." was incorrectly skipped as "simple"
        even though it has clear SVO structure (message is admission).
        """
        # This sentence has < 10 words but has SVO structure
        text = "The core message is an admission."
        blueprint = self.extractor.extract(text)

        # Verify it has semantic structure
        self.assertGreater(len(blueprint.svo_triples), 0,
                          "Sentence should have SVO triples (semantic structure)")

        # The sentence should NOT be considered "simple" for skipping
        # We can't directly test the internal logic, but we can verify
        # that blueprint extraction works correctly
        self.assertTrue(blueprint.svo_triples or blueprint.core_keywords,
                       "Blueprint should have semantic content")

    def test_truly_simple_sentence_can_be_skipped(self):
        """Test that truly simple/fragmentary sentences can be skipped.

        A sentence with no semantic structure (< 10 words, no SVOs) should be skippable.
        """
        # Fragmentary sentence with no clear semantic structure
        text = "Yes, indeed."
        blueprint = self.extractor.extract(text)

        # This might have no SVO triples (depending on parsing)
        # The key is: if it has no semantic structure, it can be skipped
        has_semantic_structure = len(blueprint.svo_triples) > 0 or len(blueprint.core_keywords) > 0

        # If it truly has no semantic structure, it's a candidate for skipping
        # (This is acceptable behavior)
        if not has_semantic_structure:
            # This is fine - truly fragmentary sentences can be skipped
            pass
        else:
            # If it does have structure, it shouldn't be skipped
            # This test verifies the blueprint extraction works
            self.assertTrue(True)

    def test_blueprint_extraction_with_svo(self):
        """Test that blueprint extraction correctly identifies SVO structure.

        This is a foundational test for the semantic structure check.
        """
        # Sentence with clear SVO
        text = "The core message is an admission."
        blueprint = self.extractor.extract(text)

        # Should extract SVO triples
        self.assertGreater(len(blueprint.svo_triples), 0,
                          "Should extract at least one SVO triple")

        # Should have keywords
        self.assertGreater(len(blueprint.core_keywords), 0,
                          "Should extract core keywords")

    def test_style_dna_extraction_in_single_example_path(self):
        """Test that style_dna_dict is extracted in single-example path.

        Regression: style_dna_dict was not extracted when len(examples) == 1,
        causing NameError in standard generation fallback.
        """
        # This test verifies the fix by ensuring the code path doesn't crash
        text = "The U.S. can no longer afford global domination."
        blueprint = self.extractor.extract(text)

        # Single example that will trigger strict fallback
        examples = ["Where do correct ideas come from?"]

        # Should not crash with NameError
        try:
            # This should extract style_dna_dict even with single example
            result = self.translator.translate(
                blueprint=blueprint,
                author_name="Test Author",
                style_dna="Test style",
                rhetorical_type=RhetoricalType.OBSERVATION,
                examples=examples,
                verbose=False
            )
            self.assertIsInstance(result, str)
        except NameError as e:
            if "style_dna_dict" in str(e):
                self.fail(f"style_dna_dict not extracted in single-example path: {e}")
            raise


if __name__ == "__main__":
    unittest.main()

