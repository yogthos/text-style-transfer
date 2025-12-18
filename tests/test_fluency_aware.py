"""Tests for fluency-aware semantic critic and refinement."""

import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pytest
from src.validator.semantic_critic import SemanticCritic
from src.ingestion.blueprint import SemanticBlueprint


class TestFluencyDetection:
    """Test that fluency check detects awkward sentences."""

    def test_fluency_detects_awkward_sentences(self):
        """Test that fluency check penalizes 'We touch breaks' type sentences."""
        critic = SemanticCritic(config_path="config.json")

        # Awkward sentence (verb + plural noun without article)
        awkward = "We touch breaks."
        fluency_score, feedback = critic._check_fluency(awkward)
        assert fluency_score < 0.8, f"Should detect awkward phrasing (score: {fluency_score})"
        assert len(feedback) > 0, "Should provide feedback"

        # Natural sentence
        natural = "Every object we touch eventually breaks."
        fluency_score, feedback = critic._check_fluency(natural)
        assert fluency_score >= 0.8, f"Should accept natural phrasing (score: {fluency_score})"

    def test_fluency_detects_incomplete_sentences(self):
        """Test that incomplete sentences get low fluency scores."""
        critic = SemanticCritic(config_path="config.json")

        # Fragment (no verb)
        fragment = "The object."
        fluency_score, feedback = critic._check_fluency(fragment)
        assert fluency_score < 0.8, "Should detect incomplete sentence"

        # Complete sentence
        complete = "The object breaks."
        fluency_score, feedback = critic._check_fluency(complete)
        assert fluency_score >= 0.7, "Should accept complete sentence"

    def test_fluency_detects_missing_punctuation(self):
        """Test that missing punctuation reduces fluency score."""
        critic = SemanticCritic(config_path="config.json")

        # Missing punctuation
        no_punct = "Every object we touch eventually breaks"
        fluency_score1, _ = critic._check_fluency(no_punct)

        # With punctuation
        with_punct = "Every object we touch eventually breaks."
        fluency_score2, _ = critic._check_fluency(with_punct)

        assert fluency_score2 >= fluency_score1, "Punctuation should improve fluency"

    def test_fluency_handles_short_sentences(self):
        """Test that very short sentences are flagged."""
        critic = SemanticCritic(config_path="config.json")

        # Too short
        short = "It breaks."
        fluency_score, feedback = critic._check_fluency(short)
        # Should still be acceptable but might be flagged
        assert fluency_score >= 0.0, "Should not crash on short sentences"

        # Normal length
        normal = "Every object we touch eventually breaks down."
        fluency_score, feedback = critic._check_fluency(normal)
        assert fluency_score >= 0.8, "Normal length should be fine"


class TestWeightedScoring:
    """Test that weighted scoring balances accuracy and fluency."""

    def test_weighted_scoring_penalizes_awkward(self):
        """Test that awkward but accurate sentences get lower scores."""
        critic = SemanticCritic(config_path="config.json")

        # Create a blueprint with keywords
        blueprint = SemanticBlueprint(
            original_text="Every object we touch eventually breaks.",
            svo_triples=[("we", "touch", "object"), ("object", "breaks", None)],
            named_entities=[],
            core_keywords={"object", "touch", "break", "eventually"},
            citations=[],
            quotes=[]
        )

        # High accuracy, low fluency (awkward phrasing)
        awkward = "We touch breaks."
        result1 = critic.evaluate(awkward, blueprint)

        # Moderate accuracy, high fluency (natural phrasing)
        natural = "Every object we touch eventually breaks."
        result2 = critic.evaluate(natural, blueprint)

        # Natural should score higher despite potentially lower keyword precision
        assert result2["score"] > result1["score"], \
            f"Fluency should boost natural sentences (awkward: {result1['score']:.2f}, natural: {result2['score']:.2f})"
        assert "fluency_score" in result1, "Should include fluency_score in result"
        assert "fluency_score" in result2, "Should include fluency_score in result"

    def test_weighted_scoring_formula(self):
        """Test that weighted score uses correct formula."""
        critic = SemanticCritic(config_path="config.json")

        blueprint = SemanticBlueprint(
            original_text="Test sentence.",
            svo_triples=[("test", "sentence", None)],
            named_entities=[],
            core_keywords={"test", "sentence"},
            citations=[],
            quotes=[]
        )

        result = critic.evaluate("This is a test sentence.", blueprint)

        # Verify score is weighted: accuracy * 0.7 + fluency * 0.3
        accuracy = (result["recall_score"] + result["precision_score"]) / 2.0
        expected_score = accuracy * 0.7 + result["fluency_score"] * 0.3

        assert abs(result["score"] - expected_score) < 0.01, \
            f"Score should be weighted (got {result['score']:.3f}, expected {expected_score:.3f})"

    def test_fluency_affects_pass_fail(self):
        """Test that low fluency can cause evaluation to fail."""
        critic = SemanticCritic(config_path="config.json")

        blueprint = SemanticBlueprint(
            original_text="Every object we touch eventually breaks.",
            svo_triples=[("we", "touch", "object")],
            named_entities=[],
            core_keywords={"object", "touch", "break"},
            citations=[],
            quotes=[]
        )

        # Awkward sentence that might pass accuracy but fail fluency
        awkward = "We touch breaks."
        result = critic.evaluate(awkward, blueprint)

        # If fluency is below threshold, should fail
        if result["fluency_score"] < 0.8:
            assert not result["pass"], "Low fluency should cause failure"
            assert "fluency" in result["feedback"].lower() or "grammatical" in result["feedback"].lower(), \
                "Feedback should mention fluency issues"


class TestFunctionalWords:
    """Test that functional words are not flagged as hallucinations."""

    def test_functional_words_not_hallucinations(self):
        """Test that adding 'the', 'eventually', etc. doesn't trigger precision failure."""
        critic = SemanticCritic(config_path="config.json")

        input_blueprint = SemanticBlueprint(
            original_text="Every object we touch breaks.",
            svo_triples=[("we", "touch", "object"), ("object", "breaks", None)],
            named_entities=[],
            core_keywords={"object", "touch", "break"},
            citations=[],
            quotes=[]
        )

        # Generated text with functional words added
        generated = "Every object that we touch eventually breaks."

        result = critic.evaluate(generated, input_blueprint)

        # Should not fail precision due to "that" and "eventually"
        assert result["precision_score"] >= 0.5, \
            f"Functional words should not trigger precision failure (score: {result['precision_score']:.2f})"

    def test_common_words_filtered(self):
        """Test that common words are filtered from precision check."""
        critic = SemanticCritic(config_path="config.json")

        blueprint = SemanticBlueprint(
            original_text="The cat sat on the mat.",
            svo_triples=[("cat", "sat", "mat")],
            named_entities=[],
            core_keywords={"cat", "sat", "mat"},
            citations=[],
            quotes=[]
        )

        # Generated with common words
        generated = "The cat eventually sat on the mat."

        result = critic.evaluate(generated, blueprint)

        # "eventually" should not be flagged as hallucination
        assert result["precision_score"] >= 0.5, "Common words should be filtered"

    def test_prepositions_allowed(self):
        """Test that prepositions are not flagged as hallucinations."""
        critic = SemanticCritic(config_path="config.json")

        blueprint = SemanticBlueprint(
            original_text="The book is on the table.",
            svo_triples=[("book", "is", "table")],
            named_entities=[],
            core_keywords={"book", "table"},
            citations=[],
            quotes=[]
        )

        # Generated with different prepositions
        generated = "The book sits upon the table."

        result = critic.evaluate(generated, blueprint)

        # "upon" should not cause precision failure (it's a preposition)
        assert result["precision_score"] >= 0.4, "Prepositions should be allowed"


class TestEvolutionFluency:
    """Test end-to-end that evolution improves fluency."""

    def test_evolution_improves_fluency(self):
        """Test that evolution improves awkward sentences to natural ones."""
        # This is a simplified test - full integration test would require mocking LLM
        critic = SemanticCritic(config_path="config.json")

        blueprint = SemanticBlueprint(
            original_text="Every object we touch eventually breaks.",
            svo_triples=[("we", "touch", "object"), ("object", "breaks", None)],
            named_entities=[],
            core_keywords={"object", "touch", "break", "eventually"},
            citations=[],
            quotes=[]
        )

        # Simulate initial awkward draft
        awkward = "We touch breaks."
        result1 = critic.evaluate(awkward, blueprint)

        # Simulate improved natural draft
        natural = "Every object we touch eventually breaks."
        result2 = critic.evaluate(natural, blueprint)

        # Natural should have higher overall score
        assert result2["score"] > result1["score"], \
            "Evolution should improve overall score by improving fluency"
        assert result2["fluency_score"] > result1["fluency_score"], \
            "Evolution should improve fluency score"


def test_spacy_filtering_handles_plurals():
    """Test that spaCy filtering handles plurals (processes -> process) correctly."""
    critic = SemanticCritic(config_path="config.json")

    input_blueprint = SemanticBlueprint(
        original_text="The biological cycle defines reality.",
        svo_triples=[("cycle", "define", "reality")],
        named_entities=[],
        core_keywords={"biological", "cycle", "define", "reality"},
        citations=[],
        quotes=[]
    )

    # Generated text with plural "processes" (should be lemmatized to "process")
    generated = "The biological processes of birth define reality."

    result = critic.evaluate(generated, input_blueprint)

    # "processes" should be lemmatized to "process" and not cause precision failure
    assert result["precision_score"] >= 0.80, \
        f"Plurals should be handled correctly (score: {result['precision_score']:.2f})"

    # Also test singular "process"
    generated2 = "The biological cycle is the very process which defines reality."
    result2 = critic.evaluate(generated2, input_blueprint)
    assert result2["precision_score"] >= 0.80, \
        f"Singular structural nouns should not trigger precision failure (score: {result2['precision_score']:.2f})"

