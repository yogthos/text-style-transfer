"""Tests for sentence function classification."""

import pytest
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.rhetorical.function_classifier import (
    SentenceFunction,
    SentenceFunctionClassifier,
    ClassifiedSentence,
)


@pytest.fixture
def classifier():
    """Create a classifier instance."""
    return SentenceFunctionClassifier()


class TestQuestionDetection:
    """Test QUESTION function detection."""

    def test_direct_question(self, classifier):
        result = classifier.classify("What is the meaning of life?")
        assert result.function == SentenceFunction.QUESTION
        assert result.confidence >= 0.9

    def test_rhetorical_question(self, classifier):
        result = classifier.classify("How could anyone believe such nonsense?")
        assert result.function == SentenceFunction.QUESTION

    def test_wh_question_without_mark(self, classifier):
        # Some questions may lack question mark
        result = classifier.classify("Why this happens.")
        # May or may not be detected depending on structure
        # Just ensure it doesn't crash

    def test_non_question_with_question_word(self, classifier):
        result = classifier.classify("I wonder what happened.")
        # Should NOT be classified as question (declarative with embedded question)
        assert result.function != SentenceFunction.QUESTION or result.confidence < 0.9


class TestContrastDetection:
    """Test CONTRAST function detection."""

    def test_but_starter(self, classifier):
        result = classifier.classify("But this view is fundamentally mistaken.")
        assert result.function == SentenceFunction.CONTRAST
        assert "but" in result.markers_found

    def test_however_starter(self, classifier):
        result = classifier.classify("However, the evidence suggests otherwise.")
        assert result.function == SentenceFunction.CONTRAST
        assert "however" in result.markers_found

    def test_yet_starter(self, classifier):
        result = classifier.classify("Yet we must consider the alternative.")
        assert result.function == SentenceFunction.CONTRAST

    def test_nevertheless(self, classifier):
        result = classifier.classify("Nevertheless, the argument holds.")
        assert result.function == SentenceFunction.CONTRAST

    def test_contrast_mid_sentence(self, classifier):
        # Contrast marker not at start - should NOT be contrast
        result = classifier.classify("The theory is elegant but wrong.")
        # This might be claim, not contrast
        assert result.function != SentenceFunction.CONTRAST or result.confidence < 0.8


class TestConcessionDetection:
    """Test CONCESSION function detection."""

    def test_admittedly(self, classifier):
        result = classifier.classify("Admittedly, this approach has limitations.")
        assert result.function == SentenceFunction.CONCESSION

    def test_granted(self, classifier):
        result = classifier.classify("Granted, there are some valid objections.")
        assert result.function == SentenceFunction.CONCESSION

    def test_of_course(self, classifier):
        result = classifier.classify("Of course, one might argue differently.")
        assert result.function == SentenceFunction.CONCESSION

    def test_it_is_true(self, classifier):
        result = classifier.classify("It is true that mistakes were made.")
        assert result.function == SentenceFunction.CONCESSION


class TestResolutionDetection:
    """Test RESOLUTION function detection."""

    def test_therefore(self, classifier):
        result = classifier.classify("Therefore, we must conclude that the hypothesis is correct.")
        assert result.function == SentenceFunction.RESOLUTION

    def test_thus(self, classifier):
        result = classifier.classify("Thus, the paradox is resolved.")
        assert result.function == SentenceFunction.RESOLUTION

    def test_consequently(self, classifier):
        result = classifier.classify("Consequently, we can accept this view.")
        assert result.function == SentenceFunction.RESOLUTION

    def test_in_conclusion(self, classifier):
        result = classifier.classify("In conclusion, the evidence supports our theory.")
        assert result.function == SentenceFunction.RESOLUTION


class TestEvidenceDetection:
    """Test EVIDENCE function detection."""

    def test_for_example(self, classifier):
        result = classifier.classify("For example, consider the case of Einstein.")
        assert result.function == SentenceFunction.EVIDENCE

    def test_studies_show(self, classifier):
        result = classifier.classify("Studies show that this effect is real.")
        assert result.function == SentenceFunction.EVIDENCE

    def test_according_to(self, classifier):
        result = classifier.classify("According to recent research, the theory holds.")
        assert result.function == SentenceFunction.EVIDENCE


class TestElaborationDetection:
    """Test ELABORATION function detection."""

    def test_that_is(self, classifier):
        result = classifier.classify("That is, the process works differently.")
        assert result.function == SentenceFunction.ELABORATION

    def test_in_other_words(self, classifier):
        result = classifier.classify("In other words, we must reconsider.")
        assert result.function == SentenceFunction.ELABORATION


class TestSetupDetection:
    """Test SETUP function detection."""

    def test_consider(self, classifier):
        result = classifier.classify("Consider the following scenario.")
        assert result.function == SentenceFunction.SETUP

    def test_imagine(self, classifier):
        result = classifier.classify("Imagine a world without gravity.")
        assert result.function == SentenceFunction.SETUP

    def test_suppose(self, classifier):
        result = classifier.classify("Suppose we accept this premise.")
        assert result.function == SentenceFunction.SETUP


class TestClaimDetection:
    """Test CLAIM function detection."""

    def test_strong_assertion(self, classifier):
        result = classifier.classify("The universe is infinite.")
        assert result.function == SentenceFunction.CLAIM

    def test_must_statement(self, classifier):
        result = classifier.classify("We must acknowledge this truth.")
        assert result.function == SentenceFunction.CLAIM


class TestContinuationDefault:
    """Test CONTINUATION as default."""

    def test_simple_statement(self, classifier):
        result = classifier.classify("The cat sat on the mat.")
        # Simple statement without strong markers
        assert result.function in {SentenceFunction.CLAIM, SentenceFunction.CONTINUATION}


class TestParagraphClassification:
    """Test paragraph-level classification."""

    def test_classify_paragraph(self, classifier):
        paragraph = """
        Consider the nature of the universe.
        Some believe it is finite, bounded by edges.
        But this view faces a logical problem.
        What exists outside those edges?
        Therefore, the universe must be infinite.
        """
        results = classifier.classify_paragraph(paragraph)

        assert len(results) >= 4
        # Check we get variety of functions
        functions = {r.function for r in results}
        assert len(functions) >= 2  # At least some variety


class TestFunctionProfileExtraction:
    """Test extraction of function profile from corpus."""

    def test_extract_profile(self, classifier):
        paragraphs = [
            "Consider the problem of consciousness. What makes us aware? But this question has no easy answer. Therefore, we must approach it carefully.",
            "The evidence suggests otherwise. However, some disagree. Admittedly, the data is limited. Thus, we need more research.",
            "Imagine a different scenario. The rules would change. But would the outcome differ? In conclusion, context matters.",
        ]

        profile = classifier.extract_function_profile(paragraphs)

        assert "function_distribution" in profile
        assert "function_transitions" in profile
        assert "initial_function_probs" in profile
        assert "function_samples" in profile

        # Should have detected multiple function types
        assert len(profile["function_distribution"]) >= 3

        # Should have some transitions
        assert len(profile["function_transitions"]) >= 1

        # Should have initial probabilities
        assert len(profile["initial_function_probs"]) >= 1


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_sentence(self, classifier):
        result = classifier.classify("")
        assert result.function == SentenceFunction.CONTINUATION

    def test_single_word(self, classifier):
        result = classifier.classify("Indeed.")
        # Should handle gracefully
        assert result is not None

    def test_very_long_sentence(self, classifier):
        long_sentence = "The " + "very " * 100 + "long sentence continues."
        result = classifier.classify(long_sentence)
        assert result is not None

    def test_multiple_markers(self, classifier):
        # Sentence with multiple function markers
        result = classifier.classify("However, it is true that, therefore, we conclude.")
        # Should pick highest priority marker
        assert result is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
