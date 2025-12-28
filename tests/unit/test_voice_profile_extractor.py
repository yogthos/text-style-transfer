"""Unit tests for voice profile extraction.

Tests AssertivenessProfile, RhetoricalProfile, and PhrasePatterns extraction.
"""

import pytest
from src.style.extractor import StyleProfileExtractor
from src.style.profile import (
    AssertivenessProfile,
    RhetoricalProfile,
    PhrasePatterns,
)


class TestAssertivenessProfileExtraction:
    """Test AssertivenessProfile extraction."""

    @pytest.fixture
    def extractor(self):
        return StyleProfileExtractor()

    def test_empty_sentences(self, extractor):
        """Test handling of empty input."""
        result = extractor._extract_assertiveness_profile([])
        assert isinstance(result, AssertivenessProfile)
        assert result.hedge_ratio == 0.0
        assert result.booster_ratio == 0.0

    def test_hedging_detection(self, extractor):
        """Test detection of hedging words."""
        sentences = [
            "Perhaps this is the case.",
            "The results are possibly incorrect.",
            "It seems that the data suggests otherwise.",
            "This might be true.",
        ]
        result = extractor._extract_assertiveness_profile(sentences)

        # Should detect hedging in most sentences
        assert result.hedge_ratio > 0.5
        assert "perhaps" in result.author_hedges or "possibly" in result.author_hedges

    def test_booster_detection(self, extractor):
        """Test detection of booster words."""
        sentences = [
            "This is certainly the correct answer.",
            "We must understand the fundamental truth.",
            "Clearly, this is essential.",
            "The evidence definitely supports this.",
        ]
        result = extractor._extract_assertiveness_profile(sentences)

        # Should detect boosters in most sentences
        assert result.booster_ratio > 0.5
        assert "certainly" in result.author_boosters or "clearly" in result.author_boosters

    def test_commitment_score_positive(self, extractor):
        """Test positive commitment score (assertive author)."""
        # All booster sentences, no hedging
        sentences = [
            "This is absolutely true.",
            "Certainly, we must act.",
            "The evidence clearly shows this.",
            "Indeed, this is fundamental.",
        ]
        result = extractor._extract_assertiveness_profile(sentences)

        # Positive commitment = more assertive than hedged
        assert result.average_commitment > 0

    def test_commitment_score_negative(self, extractor):
        """Test negative commitment score (hedged author)."""
        # All hedging sentences, no boosters
        sentences = [
            "Perhaps this is true.",
            "The data possibly suggests this.",
            "It seems to indicate something.",
            "Maybe we should consider this.",
        ]
        result = extractor._extract_assertiveness_profile(sentences)

        # Negative commitment = more hedged than assertive
        assert result.average_commitment < 0

    def test_factual_stance_detection(self, extractor):
        """Test detection of factual statements."""
        sentences = [
            "The Earth orbits the Sun.",
            "Water boils at 100 degrees Celsius.",
            "Gravity pulls objects downward.",
        ]
        result = extractor._extract_assertiveness_profile(sentences)

        # Most should be factual
        assert result.factual_ratio > 0.5

    def test_conditional_stance_detection(self, extractor):
        """Test detection of conditional statements."""
        sentences = [
            "If the temperature rises, the ice will melt.",
            "Unless we act, the situation will worsen.",
            "Assuming the data is correct, we can proceed.",
        ]
        result = extractor._extract_assertiveness_profile(sentences)

        # Most should be conditional
        assert result.conditional_ratio > 0.5

    def test_hypothetical_stance_detection(self, extractor):
        """Test detection of hypothetical statements."""
        sentences = [
            "This could be the solution.",
            "The experiment might fail.",
            "We would see different results.",
            "The theory may be incorrect.",
        ]
        result = extractor._extract_assertiveness_profile(sentences)

        # Most should be hypothetical
        assert result.hypothetical_ratio > 0.5

    def test_imperative_detection(self, extractor):
        """Test detection of imperative sentences."""
        sentences = [
            "Consider the following example.",
            "Look at the data carefully.",
            "Remember this important point.",
        ]
        result = extractor._extract_assertiveness_profile(sentences)

        # Should detect imperatives
        assert result.imperative_ratio > 0

    def test_question_pattern_extraction(self, extractor):
        """Test extraction of question patterns."""
        sentences = [
            "What does this mean for us?",
            "How can we solve this problem?",
            "Why did this happen?",
        ]
        result = extractor._extract_assertiveness_profile(sentences)

        # Should extract question patterns
        assert len(result.question_patterns) > 0

    def test_assertion_pattern_extraction(self, extractor):
        """Test extraction of assertion patterns."""
        sentences = [
            "It is clear that we must act.",
            "The fact is that nothing changes.",
            "This is certainly the case.",
        ]
        result = extractor._extract_assertiveness_profile(sentences)

        # Should extract assertion patterns
        assert len(result.assertion_patterns) > 0


class TestRhetoricalProfileExtraction:
    """Test RhetoricalProfile extraction."""

    @pytest.fixture
    def extractor(self):
        return StyleProfileExtractor()

    def test_empty_sentences(self, extractor):
        """Test handling of empty input."""
        result = extractor._extract_rhetorical_profile([], [])
        assert isinstance(result, RhetoricalProfile)
        assert result.contrast_frequency == 0.0

    def test_contrast_detection(self, extractor):
        """Test detection of contrast markers."""
        sentences = [
            "But the evidence suggests otherwise.",
            "However, we must consider alternatives.",
            "Yet this is not the whole story.",
            "Nevertheless, progress was made.",
        ]
        result = extractor._extract_rhetorical_profile(sentences, [])

        # All sentences have contrast markers
        assert result.contrast_frequency >= 0.5
        assert len(result.contrast_markers) > 0

    def test_resolution_detection(self, extractor):
        """Test detection of resolution markers."""
        sentences = [
            "Thus, we can conclude that this is true.",
            "Therefore, action must be taken.",
            "In truth, the situation is complex.",
            "Indeed, this confirms our hypothesis.",
        ]
        result = extractor._extract_rhetorical_profile(sentences, [])

        # All sentences have resolution markers
        assert result.resolution_frequency >= 0.5
        assert len(result.resolution_markers) > 0

    def test_question_frequency(self, extractor):
        """Test question frequency calculation."""
        sentences = [
            "What does this mean?",
            "How can we proceed?",
            "This is a statement.",
            "Another statement here.",
        ]
        result = extractor._extract_rhetorical_profile(sentences, [])

        # 2 of 4 sentences are questions
        assert result.question_frequency == 0.5

    def test_negation_affirmation_pattern(self, extractor):
        """Test detection of 'not X but Y' patterns."""
        sentences = [
            "The question is not whether but how.",
            "It is not about speed but about precision.",
            "We see not failure but opportunity.",
        ]
        result = extractor._extract_rhetorical_profile(sentences, [])

        # All have negation-affirmation pattern
        assert result.negation_affirmation_ratio >= 0.5
        assert len(result.pattern_samples.get("negation_affirmation", [])) > 0

    def test_opposition_extraction(self, extractor):
        """Test extraction of opposition pairs."""
        sentences = [
            "It is not darkness but light.",
            "We see not weakness but strength.",
        ]
        result = extractor._extract_rhetorical_profile(sentences, [])

        # Should extract opposition pairs
        assert len(result.detected_oppositions) > 0
        # Check that some oppositions were found
        oppositions_flat = [item for pair in result.detected_oppositions for item in pair]
        assert any(word in oppositions_flat for word in ["darkness", "light", "weakness", "strength"])

    def test_direct_address_detection(self, extractor):
        """Test detection of direct address patterns."""
        sentences = [
            "You must understand this principle.",
            "We should consider the alternatives.",
            "We need to take action now.",
        ]
        result = extractor._extract_rhetorical_profile(sentences, [])

        # All have direct address
        assert result.direct_address_ratio >= 0.5

    def test_appearance_reality_pattern(self, extractor):
        """Test detection of appearance vs reality patterns."""
        sentences = [
            "It seems simple, but actually it is complex.",
            "The data appears clear, however the reality differs.",
        ]
        result = extractor._extract_rhetorical_profile(sentences, [])

        # Both have appearance-reality pattern
        assert result.appearance_reality_ratio >= 0.5

    def test_pattern_samples_collected(self, extractor):
        """Test that pattern samples are collected."""
        sentences = [
            "But the truth is different.",
            "What does this mean for us?",
            "You must consider this carefully.",
        ]
        result = extractor._extract_rhetorical_profile(sentences, [])

        # Should have pattern samples
        assert len(result.pattern_samples) > 0


class TestPhrasePatternExtraction:
    """Test PhrasePatterns extraction."""

    @pytest.fixture
    def extractor(self):
        return StyleProfileExtractor()

    def test_paragraph_opener_extraction(self, extractor):
        """Test extraction of paragraph openers."""
        paragraphs = [
            "In the beginning, there was nothing.",
            "The fundamental truth is this.",
            "But we must consider the alternatives.",
        ]
        sentences = []
        for p in paragraphs:
            sentences.append(p)

        result = extractor._extract_phrase_patterns(sentences, paragraphs)

        # Should extract paragraph openers
        assert len(result.paragraph_openers) > 0
        # Openers should start with characteristic words
        assert any(o.startswith("In") or o.startswith("The") or o.startswith("But")
                  for o in result.paragraph_openers)

    def test_sentence_connector_extraction(self, extractor):
        """Test extraction of sentence connectors."""
        sentences = [
            "But this changes everything.",
            "However, we must proceed.",
            "Thus, the conclusion is clear.",
            "Therefore, action is needed.",
        ]
        result = extractor._extract_phrase_patterns(sentences, [])

        # Should extract connectors
        assert len(result.sentence_connectors) > 0
        # Should find transition-based connectors
        connector_words = [c.split()[0].lower() for c in result.sentence_connectors]
        assert any(w in connector_words for w in ["but", "however", "thus", "therefore"])

    def test_characteristic_construction_extraction(self, extractor):
        """Test extraction of characteristic constructions."""
        sentences = [
            "It is clear that we must act.",
            "What we call progress is really regression.",
            "The question is not whether but when.",
        ]
        result = extractor._extract_phrase_patterns(sentences, [])

        # Should extract characteristic constructions
        assert len(result.characteristic_constructions) > 0

    def test_emphasis_pattern_extraction(self, extractor):
        """Test extraction of emphasis patterns."""
        sentences = [
            "This point—and this is crucial—must be understood.",
            "The key insight -- often overlooked -- is this.",
        ]
        result = extractor._extract_phrase_patterns(sentences, [])

        # Should extract emphasis patterns with dashes
        assert len(result.emphasis_patterns) > 0

    def test_parallelism_extraction(self, extractor):
        """Test extraction of parallelism examples."""
        sentences = [
            "Not this, not that, but the other.",
            "First we learn, then we grow, finally we succeed.",
        ]
        result = extractor._extract_phrase_patterns(sentences, [])

        # Should find parallelism
        assert len(result.parallelism_examples) > 0

    def test_antithesis_extraction(self, extractor):
        """Test extraction of antithesis examples."""
        sentences = [
            "On one hand we have progress, on the other hand we have stability.",
            "While some see failure, others see opportunity.",
        ]
        result = extractor._extract_phrase_patterns(sentences, [])

        # Should find antithesis
        assert len(result.antithesis_examples) > 0

    def test_opener_frequency_calculation(self, extractor):
        """Test opener frequency calculation."""
        paragraphs = [
            "The key point is this.",
            "In essence, we must understand.",
            "Random start here.",
        ]
        sentences = paragraphs[:]
        result = extractor._extract_phrase_patterns(sentences, paragraphs)

        # Should have opener frequency > 0
        assert result.opener_frequency > 0


class TestIntegratedVoiceProfile:
    """Test integrated voice profile extraction in full profile."""

    @pytest.fixture
    def extractor(self):
        return StyleProfileExtractor()

    def test_full_profile_includes_voice_components(self, extractor):
        """Test that full profile extraction includes voice components."""
        paragraphs = [
            "The fundamental truth is this: we must act decisively. "
            "But the question is not whether to act, but how. "
            "Perhaps some hesitation is warranted. "
            "However, in truth, delay is not an option.",
            "What we call progress is really motion without direction. "
            "You must understand this clearly. "
            "Indeed, the facts are undeniable.",
        ]

        profile = extractor.extract(paragraphs, "TestAuthor")

        # Check voice profile components are populated
        assert profile.assertiveness_profile is not None
        assert profile.rhetorical_profile is not None
        assert profile.phrase_patterns is not None

        # Check assertiveness profile has data
        assert profile.assertiveness_profile.factual_ratio > 0 or profile.assertiveness_profile.hedge_ratio > 0

        # Check rhetorical profile has data
        assert len(profile.rhetorical_profile.contrast_markers) > 0 or profile.rhetorical_profile.contrast_frequency >= 0

        # Check phrase patterns has data
        assert len(profile.phrase_patterns.paragraph_openers) > 0

    def test_voice_profile_serialization(self, extractor):
        """Test that voice profiles serialize and deserialize correctly."""
        paragraphs = [
            "Certainly, this is the truth. But we must consider alternatives. "
            "Perhaps there is another way. Indeed, the evidence suggests so.",
        ]

        profile = extractor.extract(paragraphs, "TestAuthor")

        # Serialize to dict
        profile_dict = profile.to_dict()

        # Check voice components are in dict
        assert "assertiveness_profile" in profile_dict
        assert "rhetorical_profile" in profile_dict
        assert "phrase_patterns" in profile_dict

        # Check assertiveness fields
        assert "hedge_ratio" in profile_dict["assertiveness_profile"]
        assert "booster_ratio" in profile_dict["assertiveness_profile"]
        assert "average_commitment" in profile_dict["assertiveness_profile"]

        # Check rhetorical fields
        assert "contrast_frequency" in profile_dict["rhetorical_profile"]
        assert "detected_oppositions" in profile_dict["rhetorical_profile"]

        # Deserialize back
        from src.style.profile import AuthorStyleProfile
        restored = AuthorStyleProfile.from_dict(profile_dict)

        # Check restoration
        assert restored.assertiveness_profile.hedge_ratio == profile.assertiveness_profile.hedge_ratio
        assert restored.rhetorical_profile.contrast_frequency == profile.rhetorical_profile.contrast_frequency


class TestMixedStyleCorpus:
    """Test voice extraction on realistic mixed-style corpus."""

    @pytest.fixture
    def extractor(self):
        return StyleProfileExtractor()

    def test_dialectical_author_profile(self, extractor):
        """Test extraction from dialectical/confrontational style corpus."""
        paragraphs = [
            "The metaphysicians see things as static, fixed, eternal. "
            "But dialectics reveals the opposite: everything is in motion, "
            "everything transforms, everything develops. Not stasis but process.",
            "It seems the world is stable. However, this is mere appearance. "
            "In truth, beneath the surface, contradictions multiply. "
            "The question is not whether change comes, but how we direct it.",
            "You must understand: every thesis generates its antithesis. "
            "We cannot avoid this fundamental law. Indeed, it is the motor of history.",
        ]

        profile = extractor.extract(paragraphs, "DialecticalAuthor")

        # Dialectical authors should have high contrast usage
        assert profile.rhetorical_profile.contrast_frequency > 0.2

        # Should have negation-affirmation patterns
        assert profile.rhetorical_profile.negation_affirmation_ratio > 0

        # Should have direct address
        assert profile.rhetorical_profile.direct_address_ratio > 0

        # Should have boosters (assertive style)
        assert profile.assertiveness_profile.booster_ratio > 0

    def test_academic_hedged_profile(self, extractor):
        """Test extraction from academic/hedged style corpus."""
        paragraphs = [
            "The data suggests that perhaps the hypothesis is partially correct. "
            "It seems that the relationship may be somewhat weaker than expected. "
            "One might argue that additional research is possibly needed.",
            "The results appear to indicate a potential trend, although "
            "the evidence remains relatively inconclusive at this stage. "
            "Further investigation would likely be beneficial.",
        ]

        profile = extractor.extract(paragraphs, "AcademicAuthor")

        # Academic authors should have high hedging
        assert profile.assertiveness_profile.hedge_ratio > 0.3

        # Negative commitment (more hedged than assertive)
        assert profile.assertiveness_profile.average_commitment < 0

        # Lower direct address
        assert profile.rhetorical_profile.direct_address_ratio < 0.2
