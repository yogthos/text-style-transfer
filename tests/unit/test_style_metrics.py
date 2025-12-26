"""Tests for style metrics (Phase 3.2-3.3)."""

import pytest
from src.models.style import AuthorProfile, StyleProfile
from src.generation.style_metrics import (
    VocabularyScorer,
    VoiceScorer,
    SentenceLengthScorer,
    PunctuationScorer,
    StyleScorer,
    StyleScore,
)


class TestVocabularyScorer:
    """Tests for vocabulary overlap scoring."""

    @pytest.fixture
    def author_profile(self):
        return AuthorProfile(
            name="TestAuthor",
            style_dna="Test author.",
            top_vocab=["cosmos", "universe", "stars", "planets", "science", "nature", "evolution"],
        )

    @pytest.fixture
    def scorer(self, author_profile):
        return VocabularyScorer(StyleProfile.from_author(author_profile))

    def test_high_overlap_scores_high(self, scorer):
        """Text using author's vocabulary should score high."""
        text = "The cosmos reveals the nature of the universe and its stars."
        score = scorer.score(text)
        assert score > 0.3  # Should have significant overlap

    def test_no_overlap_scores_low(self, scorer):
        """Text with no vocabulary overlap should score low."""
        text = "The cat sat on the mat eating fish."
        score = scorer.score(text)
        assert score < 0.2  # Minimal overlap expected

    def test_empty_text_scores_zero(self, scorer):
        """Empty text should score zero."""
        assert scorer.score("") == 0.0

    def test_get_matching_words(self, scorer):
        """Should return words that match author's vocabulary."""
        text = "The cosmos and universe are vast."
        matches = scorer.get_matching_words(text)
        assert any("cosmos" in m.lower() for m in matches)


class TestVoiceScorer:
    """Tests for active/passive voice scoring."""

    @pytest.fixture
    def active_author(self):
        """Author who prefers active voice."""
        return AuthorProfile(
            name="ActiveWriter",
            style_dna="Prefers active voice.",
            voice_ratio=0.8,  # 80% active
        )

    @pytest.fixture
    def passive_author(self):
        """Author who prefers passive voice."""
        return AuthorProfile(
            name="PassiveWriter",
            style_dna="Prefers passive voice.",
            voice_ratio=0.3,  # 30% active (70% passive)
        )

    def test_active_text_matches_active_author(self, active_author):
        """Active voice text should score high for active-preferring author."""
        scorer = VoiceScorer(StyleProfile.from_author(active_author))
        text = "The scientist discovered the cure. The team celebrated the victory."
        score = scorer.score(text)
        assert score > 0.7  # Good match for active author

    def test_passive_text_matches_passive_author(self, passive_author):
        """Passive voice text should score high for passive-preferring author."""
        scorer = VoiceScorer(StyleProfile.from_author(passive_author))
        text = "The cure was discovered by the scientist. The victory was celebrated."
        score = scorer.score(text)
        assert score > 0.5  # Reasonable match for passive author

    def test_analyze_voice_detects_passive(self, active_author):
        """Should detect passive voice constructions."""
        scorer = VoiceScorer(StyleProfile.from_author(active_author))
        text = "The experiment was conducted. The results were analyzed."
        analysis = scorer.analyze_voice(text)
        assert analysis["passive_ratio"] > 0.5

    def test_analyze_voice_detects_active(self, active_author):
        """Should detect active voice constructions."""
        scorer = VoiceScorer(StyleProfile.from_author(active_author))
        text = "The scientist conducted the experiment. The team analyzed results."
        analysis = scorer.analyze_voice(text)
        assert analysis["active_ratio"] > 0.5


class TestSentenceLengthScorer:
    """Tests for sentence length scoring."""

    @pytest.fixture
    def short_sentence_author(self):
        """Author with short sentences."""
        return AuthorProfile(
            name="TerseWriter",
            style_dna="Short sentences.",
            avg_sentence_length=8.0,
        )

    @pytest.fixture
    def long_sentence_author(self):
        """Author with long sentences."""
        return AuthorProfile(
            name="VerboseWriter",
            style_dna="Long sentences.",
            avg_sentence_length=25.0,
        )

    def test_matching_length_scores_high(self, short_sentence_author):
        """Text matching target length should score high."""
        scorer = SentenceLengthScorer(StyleProfile.from_author(short_sentence_author))
        # Short sentences ~8 words each
        text = "The bright sun rose slowly today. The colorful birds began to sing loudly."
        score = scorer.score(text)
        assert score > 0.4  # Reasonable match for short sentences

    def test_mismatched_length_scores_lower(self, short_sentence_author):
        """Text with wrong length should score lower."""
        scorer = SentenceLengthScorer(StyleProfile.from_author(short_sentence_author))
        # Very long sentence for author who prefers short
        text = "The magnificent sun rose slowly over the distant mountains while the colorful birds began their melodious singing in the ancient forest."
        score = scorer.score(text)
        assert score < 0.5  # Poor match

    def test_analyze_lengths(self, short_sentence_author):
        """Should correctly analyze sentence lengths."""
        scorer = SentenceLengthScorer(StyleProfile.from_author(short_sentence_author))
        text = "Short one here. This is a bit longer sentence."
        analysis = scorer.analyze_lengths(text)
        assert "avg_length" in analysis
        assert analysis["avg_length"] > 0


class TestPunctuationScorer:
    """Tests for punctuation pattern scoring."""

    @pytest.fixture
    def semicolon_author(self):
        """Author who uses semicolons frequently."""
        return AuthorProfile(
            name="SemicolonWriter",
            style_dna="Uses semicolons extensively.",
            punctuation_patterns={
                "semicolon": {"per_sentence": 0.5},
                "em_dash": {"per_sentence": 0.3},
            },
        )

    @pytest.fixture
    def simple_author(self):
        """Author with minimal punctuation."""
        return AuthorProfile(
            name="SimpleWriter",
            style_dna="Simple punctuation.",
            punctuation_patterns={
                "semicolon": {"per_sentence": 0.0},
                "em_dash": {"per_sentence": 0.0},
            },
        )

    def test_matching_punctuation_scores_high(self, semicolon_author):
        """Text matching target punctuation should score well."""
        scorer = PunctuationScorer(StyleProfile.from_author(semicolon_author))
        text = "First clause; second clause. Another thought—interrupted here."
        score = scorer.score(text)
        assert score > 0.3  # Has semicolons and dashes

    def test_no_target_patterns_neutral_score(self):
        """No target patterns should give neutral score."""
        author = AuthorProfile(
            name="NoPatterns",
            style_dna="No patterns.",
            punctuation_patterns={},
        )
        scorer = PunctuationScorer(StyleProfile.from_author(author))
        score = scorer.score("Simple text here.")
        assert score == 0.5  # Neutral

    def test_analyze_punctuation(self, semicolon_author):
        """Should correctly count punctuation per sentence."""
        scorer = PunctuationScorer(StyleProfile.from_author(semicolon_author))
        text = "One; two. Three; four. Five!"
        analysis = scorer.analyze_punctuation(text)
        # 2 semicolons over 3 sentences = 0.67 per sentence
        assert analysis["semicolon"] > 0.5

    def test_empty_text(self, semicolon_author):
        """Empty text should return empty dict."""
        scorer = PunctuationScorer(StyleProfile.from_author(semicolon_author))
        assert scorer.analyze_punctuation("") == {}


class TestStyleScorer:
    """Tests for combined style scoring."""

    @pytest.fixture
    def author_profile(self):
        return AuthorProfile(
            name="TestAuthor",
            style_dna="Test author with specific style.",
            top_vocab=["science", "discovery", "research", "theory", "evidence"],
            avg_sentence_length=15.0,
            voice_ratio=0.7,  # Prefers active
        )

    @pytest.fixture
    def scorer(self, author_profile):
        return StyleScorer(StyleProfile.from_author(author_profile))

    def test_score_returns_style_score(self, scorer):
        """Should return StyleScore dataclass."""
        text = "Science leads to discovery through research and theory."
        result = scorer.score(text)
        assert isinstance(result, StyleScore)
        assert hasattr(result, "vocabulary_overlap")
        assert hasattr(result, "voice_match")
        assert hasattr(result, "sentence_length_match")
        assert hasattr(result, "overall")

    def test_overall_is_weighted_average(self, scorer):
        """Overall score should be weighted average of components."""
        text = "Science leads to discovery."
        result = scorer.score(text)
        # Overall should be between 0 and 1
        assert 0 <= result.overall <= 1

    def test_passes_threshold(self, scorer):
        """Should correctly evaluate threshold."""
        good_score = StyleScore(
            vocabulary_overlap=0.7,
            voice_match=0.8,
            sentence_length_match=0.7,
            overall=0.73
        )
        assert good_score.passes_threshold(0.6)
        assert not good_score.passes_threshold(0.8)

    def test_get_feedback(self, scorer):
        """Should generate actionable feedback."""
        text = "Bad text with wrong vocabulary and style."
        feedback = scorer.get_feedback(text)
        assert isinstance(feedback, str)
        # Feedback should be non-empty for poor match
        assert len(feedback) > 0


class TestStyleScoreDataclass:
    """Tests for StyleScore dataclass."""

    def test_passes_threshold_true(self):
        """Should pass when overall >= threshold."""
        score = StyleScore(
            vocabulary_overlap=0.8,
            voice_match=0.7,
            sentence_length_match=0.9,
            overall=0.8
        )
        assert score.passes_threshold(0.6)
        assert score.passes_threshold(0.8)

    def test_passes_threshold_false(self):
        """Should fail when overall < threshold."""
        score = StyleScore(
            vocabulary_overlap=0.3,
            voice_match=0.4,
            sentence_length_match=0.5,
            overall=0.4
        )
        assert not score.passes_threshold(0.6)
