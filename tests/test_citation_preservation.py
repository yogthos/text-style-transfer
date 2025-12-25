"""Tests for citation and quote preservation in the pipeline."""

import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import unittest
from src.ingestion.blueprint import BlueprintExtractor, SemanticBlueprint
from src.generator.translator import StyleTranslator
from src.validator.semantic_critic import SemanticCritic
from src.pipeline import _split_into_sentences_safe
from src.generator.llm_interface import clean_generated_text


class TestCitationExtraction(unittest.TestCase):
    """Test citation extraction from blueprint."""

    def setUp(self):
        self.extractor = BlueprintExtractor()

    def test_extract_single_citation(self):
        """Test extraction of a single citation."""
        text = "Tom Stonier proposed that information is interconvertible with energy[^155]."
        blueprint = self.extractor.extract(text)

        self.assertEqual(len(blueprint.citations), 1)
        self.assertEqual(blueprint.citations[0][0], "[^155]")
        self.assertGreater(blueprint.citations[0][1], 0)  # Position should be set

    def test_extract_multiple_citations(self):
        """Test extraction of multiple citations."""
        text = "A rigid pattern repeats itself across all scales[^25]. Another pattern exists[^155]."
        blueprint = self.extractor.extract(text)

        self.assertEqual(len(blueprint.citations), 2)
        citation_texts = [cit[0] for cit in blueprint.citations]
        self.assertIn("[^25]", citation_texts)
        self.assertIn("[^155]", citation_texts)

    def test_extract_no_citations(self):
        """Test extraction when no citations are present."""
        text = "This is a simple sentence without citations."
        blueprint = self.extractor.extract(text)

        self.assertEqual(len(blueprint.citations), 0)


class TestQuoteExtraction(unittest.TestCase):
    """Test quote extraction from blueprint."""

    def setUp(self):
        self.extractor = BlueprintExtractor()

    def test_extract_double_quotes(self):
        """Test extraction of double-quoted text."""
        text = 'He said "This is important" and continued.'
        blueprint = self.extractor.extract(text)

        self.assertEqual(len(blueprint.quotes), 1)
        self.assertEqual(blueprint.quotes[0][0], '"This is important"')

    def test_extract_single_quotes(self):
        """Test extraction of single-quoted text."""
        text = "He said 'This is important' and continued."
        blueprint = self.extractor.extract(text)

        self.assertEqual(len(blueprint.quotes), 1)
        self.assertEqual(blueprint.quotes[0][0], "'This is important'")

    def test_extract_multiple_quotes(self):
        """Test extraction of multiple quotes."""
        text = 'First he said "Hello" and then "Goodbye".'
        blueprint = self.extractor.extract(text)

        self.assertEqual(len(blueprint.quotes), 2)
        quote_texts = [q[0] for q in blueprint.quotes]
        self.assertIn('"Hello"', quote_texts)
        self.assertIn('"Goodbye"', quote_texts)

    def test_ignore_short_quotes(self):
        """Test that very short quotes are ignored."""
        text = 'He said "Hi" and left.'
        blueprint = self.extractor.extract(text)

        # "Hi" is only 2 chars after stripping quotes, should be ignored
        self.assertEqual(len(blueprint.quotes), 0)

    def test_extract_no_quotes(self):
        """Test extraction when no quotes are present."""
        text = "This is a simple sentence without quotes."
        blueprint = self.extractor.extract(text)

        self.assertEqual(len(blueprint.quotes), 0)


class TestSentenceSplitting(unittest.TestCase):
    """Test sentence splitting with citations."""

    def test_split_with_citation_at_end(self):
        """Test splitting when citation is at end of sentence."""
        paragraph = "First sentence[^155]. Second sentence."
        sentences = _split_into_sentences_safe(paragraph)

        self.assertEqual(len(sentences), 2)
        self.assertIn("[^155]", sentences[0])
        self.assertNotIn("[^155]", sentences[1])

    def test_split_with_citation_in_middle(self):
        """Test splitting when citation is in middle of sentence."""
        paragraph = "First sentence[^155] continues. Second sentence."
        sentences = _split_into_sentences_safe(paragraph)

        self.assertEqual(len(sentences), 2)
        self.assertIn("[^155]", sentences[0])

    def test_split_multiple_citations(self):
        """Test splitting with multiple citations."""
        paragraph = "First[^25]. Second[^155]. Third."
        sentences = _split_into_sentences_safe(paragraph)

        self.assertEqual(len(sentences), 3)
        self.assertIn("[^25]", sentences[0])
        self.assertIn("[^155]", sentences[1])

    def test_split_normal_sentences(self):
        """Test splitting normal sentences without citations."""
        paragraph = "First sentence. Second sentence. Third sentence."
        sentences = _split_into_sentences_safe(paragraph)

        self.assertEqual(len(sentences), 3)


class TestTranslatorPreservation(unittest.TestCase):
    """Test that translator preserves citations and quotes."""

    def setUp(self):
        self.translator = StyleTranslator(config_path="config.json")

    def test_restore_missing_citations(self):
        """Test that missing citations are restored."""
        blueprint = SemanticBlueprint(
            original_text="Test sentence[^155].",
            svo_triples=[],
            named_entities=[],
            core_keywords=set(),
            citations=[("[^155]", 0)],
            quotes=[]
        )

        generated = "Test sentence."
        restored = self.translator._restore_citations_and_quotes(generated, blueprint)

        self.assertIn("[^155]", restored)

    def test_preserve_existing_citations(self):
        """Test that existing citations are preserved."""
        blueprint = SemanticBlueprint(
            original_text="Test sentence[^155].",
            svo_triples=[],
            named_entities=[],
            core_keywords=set(),
            citations=[("[^155]", 0)],
            quotes=[]
        )

        generated = "Test sentence[^155]."
        restored = self.translator._restore_citations_and_quotes(generated, blueprint)

        # Should not duplicate
        self.assertEqual(restored.count("[^155]"), 1)

    def test_restore_multiple_citations(self):
        """Test restoration of multiple citations."""
        blueprint = SemanticBlueprint(
            original_text="Test[^25] sentence[^155].",
            svo_triples=[],
            named_entities=[],
            core_keywords=set(),
            citations=[("[^25]", 0), ("[^155]", 0)],
            quotes=[]
        )

        generated = "Test sentence."
        restored = self.translator._restore_citations_and_quotes(generated, blueprint)

        self.assertIn("[^25]", restored)
        self.assertIn("[^155]", restored)


class TestCriticValidation(unittest.TestCase):
    """Test that critic validates citations and quotes."""

    def setUp(self):
        self.critic = SemanticCritic(config_path="config.json")

    def test_critic_fails_missing_citation(self):
        """Test that critic fails when citation is missing."""
        input_blueprint = SemanticBlueprint(
            original_text="Test sentence[^155].",
            svo_triples=[("test", "sentence", "")],
            named_entities=[],
            core_keywords={"test", "sentence"},
            citations=[("[^155]", 0)],
            quotes=[]
        )

        generated = "Test sentence."  # Missing citation
        result = self.critic.evaluate(generated, input_blueprint)

        # The critic should fail when citations are missing (checked early in evaluate method)
        self.assertFalse(result.get("pass", True),
                        f"Critic should fail when citations are missing. Result: {result}")
        self.assertIn("Missing citations", result.get("feedback", ""),
                     f"Feedback should mention missing citations. Feedback: {result.get('feedback', '')}")

    def test_critic_passes_with_citation(self):
        """Test that critic passes when citation is present."""
        input_blueprint = SemanticBlueprint(
            original_text="Test sentence[^155].",
            svo_triples=[("test", "sentence", "")],
            named_entities=[],
            core_keywords={"test", "sentence"},
            citations=[("[^155]", 0)],
            quotes=[]
        )

        generated = "Test sentence[^155]."  # Has citation
        result = self.critic.evaluate(generated, input_blueprint)

        # Should not fail due to missing citations
        # (may still fail semantic checks, but citation check should pass)
        self.assertNotIn("Missing citations", result["feedback"])

    def test_critic_fails_missing_quote(self):
        """Test that critic fails when quote is missing."""
        input_blueprint = SemanticBlueprint(
            original_text='He said "This is important".',
            svo_triples=[("he", "said", "")],
            named_entities=[],
            core_keywords={"he", "said", "important"},
            citations=[],
            quotes=[('"This is important"', 0)]
        )

        generated = "He said something important."  # Missing quote content
        result = self.critic.evaluate(generated, input_blueprint)

        self.assertFalse(result["pass"])
        self.assertIn("Missing quote content", result["feedback"])

    def test_critic_fails_modified_quote(self):
        """Test that critic fails when quote content is modified (word changed)."""
        input_blueprint = SemanticBlueprint(
            original_text='He said "This is important".',
            svo_triples=[("he", "said", "")],
            named_entities=[],
            core_keywords={"he", "said", "important"},
            citations=[],
            quotes=[('"This is important"', 0)]
        )

        generated = 'He said "This was important".'  # Modified quote (is -> was)
        result = self.critic.evaluate(generated, input_blueprint)

        self.assertFalse(result["pass"])
        self.assertIn("Missing quote content", result["feedback"])

    def test_critic_passes_with_exact_quote(self):
        """Test that critic passes when quote is exact."""
        input_blueprint = SemanticBlueprint(
            original_text='He said "This is important".',
            svo_triples=[("he", "said", "")],
            named_entities=[],
            core_keywords={"he", "said", "important"},
            citations=[],
            quotes=[('"This is important"', 0)]
        )

        generated = 'He said "This is important".'  # Exact quote
        result = self.critic.evaluate(generated, input_blueprint)

        # Should not fail due to missing/modified quotes
        self.assertNotIn("Missing quote content", result["feedback"])

    def test_critic_passes_with_quote_content_no_quotes(self):
        """Test that critic passes when quote content exists without quotes (relaxed validation)."""
        input_blueprint = SemanticBlueprint(
            original_text='He said "toolset" is useful.',
            svo_triples=[("he", "said", "")],
            named_entities=[],
            core_keywords={"he", "said", "toolset", "useful"},
            citations=[],
            quotes=[('"toolset"', 0)]
        )

        generated = 'He said toolset is useful.'  # Content present but no quotes
        result = self.critic.evaluate(generated, input_blueprint)

        # Should pass because the word "toolset" is present (relaxed validation)
        self.assertNotIn("Missing quote content", result["feedback"])


class TestPostProcessingProtection(unittest.TestCase):
    """Test that post-processing protects citations and quotes."""

    def test_protect_citation_from_punctuation_cleanup(self):
        """Test that citations are protected from punctuation cleanup."""
        text = "Sentence with citation[^155] ."
        cleaned = clean_generated_text(text)

        self.assertIn("[^155]", cleaned)
        # Punctuation should be cleaned but citation preserved
        self.assertNotIn("citation[^155] .", cleaned)  # Space before period should be removed

    def test_protect_quote_from_capitalization(self):
        """Test that quotes are protected from capitalization fixes."""
        text = 'he said "this is important".'
        cleaned = clean_generated_text(text)

        # Quote should be preserved exactly
        self.assertIn('"this is important"', cleaned)

    def test_protect_multiple_citations(self):
        """Test that multiple citations are protected."""
        text = "First[^25] . Second[^155] ."
        cleaned = clean_generated_text(text)

        self.assertIn("[^25]", cleaned)
        self.assertIn("[^155]", cleaned)

    def test_protect_citation_and_quote_together(self):
        """Test protection when both citation and quote are present."""
        text = 'He said "important"[^155] .'
        cleaned = clean_generated_text(text)

        self.assertIn("[^155]", cleaned)
        self.assertIn('"important"', cleaned)


class TestEndToEndPreservation(unittest.TestCase):
    """End-to-end tests for citation and quote preservation in the pipeline."""

    def setUp(self):
        self.extractor = BlueprintExtractor()
        self.critic = SemanticCritic(config_path="config.json")

    def test_citation_preserved_through_blueprint(self):
        """Test that citations are preserved when extracting blueprint."""
        text = "Tom Stonier proposed that information is interconvertible with energy[^155]."
        blueprint = self.extractor.extract(text)

        # Citation should be extracted
        self.assertEqual(len(blueprint.citations), 1)
        self.assertEqual(blueprint.citations[0][0], "[^155]")

        # Original text should still contain citation
        self.assertIn("[^155]", blueprint.original_text)

    def test_quote_preserved_through_blueprint(self):
        """Test that quotes are preserved when extracting blueprint."""
        text = 'He said "This is important" and continued.'
        blueprint = self.extractor.extract(text)

        # Quote should be extracted
        self.assertEqual(len(blueprint.quotes), 1)
        self.assertEqual(blueprint.quotes[0][0], '"This is important"')

        # Original text should still contain quote
        self.assertIn('"This is important"', blueprint.original_text)

    def test_citation_and_quote_together(self):
        """Test that both citations and quotes are preserved together."""
        text = 'Tom Stonier said "Information is energy"[^155].'
        blueprint = self.extractor.extract(text)

        # Both should be extracted
        self.assertEqual(len(blueprint.citations), 1)
        self.assertEqual(len(blueprint.quotes), 1)
        self.assertEqual(blueprint.citations[0][0], "[^155]")
        self.assertEqual(blueprint.quotes[0][0], '"Information is energy"')

    def test_critic_rejects_missing_citation_in_realistic_scenario(self):
        """Test that critic correctly rejects text missing citations in a realistic scenario."""
        # Simulate a real sentence with citation
        input_text = "The biological cycle of birth, life, and decay defines our reality[^155]."
        input_blueprint = self.extractor.extract(input_text)

        # Generated text missing the citation
        generated = "The biological cycle of birth, life, and decay defines our reality."

        result = self.critic.evaluate(generated, input_blueprint)

        # Should fail because citation is missing
        self.assertFalse(result["pass"])
        self.assertIn("Missing citations", result["feedback"])
        self.assertEqual(result["score"], 0.0)  # Critical failure

    def test_critic_rejects_missing_quote_in_realistic_scenario(self):
        """Test that critic correctly rejects text missing quote content in a realistic scenario."""
        # Simulate a real sentence with quote
        input_text = 'He declared "A revolution is not a dinner party".'
        input_blueprint = self.extractor.extract(input_text)

        # Generated text missing the quote content entirely
        generated = "He declared something else entirely."

        result = self.critic.evaluate(generated, input_blueprint)

        # Should fail because quote content is missing
        self.assertFalse(result["pass"])
        self.assertIn("Missing quote content", result["feedback"])
        self.assertEqual(result["score"], 0.0)  # Critical failure

    def test_critic_passes_quote_content_without_quotes_in_realistic_scenario(self):
        """Test that critic passes when quote content exists without quotes (relaxed validation)."""
        # Simulate a real sentence with quote
        input_text = 'He declared "A revolution is not a dinner party".'
        input_blueprint = self.extractor.extract(input_text)

        # Generated text has the quote content but without quotes
        generated = "He declared that a revolution is not a dinner party."

        result = self.critic.evaluate(generated, input_blueprint)

        # Should pass because quote content is present (relaxed validation)
        self.assertNotIn("Missing quote content", result["feedback"])

    def test_sentence_splitting_preserves_citations_in_real_text(self):
        """Test sentence splitting with real text containing citations."""
        # Real paragraph from input file
        paragraph = "Tom Stonier proposed that information is interconvertible with energy and conserved alongside energy[^155]. The informational architecture of the cosmos provides the intrinsic scaffolding for the structure."

        sentences = _split_into_sentences_safe(paragraph)

        # Should split into 2 sentences
        self.assertEqual(len(sentences), 2)
        # First sentence should contain the citation
        self.assertIn("[^155]", sentences[0])
        # Second sentence should not contain the citation
        self.assertNotIn("[^155]", sentences[1])


if __name__ == '__main__':
    unittest.main()

