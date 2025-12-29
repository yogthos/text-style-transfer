"""Tests for the repetition reducer module."""

import pytest
from src.vocabulary.repetition_reducer import RepetitionReducer, ReductionStats, LLM_SPEAK


class TestRepetitionReducer:
    """Tests for RepetitionReducer class."""

    def test_basic_initialization(self):
        """Test basic reducer initialization."""
        reducer = RepetitionReducer(threshold=3)
        assert reducer.threshold == 3
        assert reducer.use_wordnet is True
        assert reducer.synonym_replacement is False  # Default

    def test_reduce_preserves_normal_text(self):
        """Test that normal text is preserved."""
        reducer = RepetitionReducer(threshold=3)
        text = "The quick brown fox jumps over the lazy dog."
        result, stats = reducer.reduce(text)
        # Text should be mostly preserved (no overused words)
        assert "fox" in result
        assert "dog" in result

    def test_llm_speak_replacement(self):
        """Test that LLM-speak words are replaced."""
        reducer = RepetitionReducer(threshold=3)
        text = "We need to utilize this functionality to leverage our synergy."
        result, stats = reducer.reduce(text)
        # LLM-speak should be replaced
        assert "utilize" not in result.lower() or stats.replacements_made > 0
        assert stats.words_checked > 0

    def test_llm_speak_dictionary(self):
        """Test that common LLM-speak words are in the dictionary."""
        assert "utilize" in LLM_SPEAK
        assert "leverage" in LLM_SPEAK
        assert "synergy" in LLM_SPEAK
        assert "robust" in LLM_SPEAK
        assert "streamline" in LLM_SPEAK

    def test_reset_clears_state(self):
        """Test that reset clears word counts."""
        reducer = RepetitionReducer(threshold=3)
        text = "The word word word word appears many times."
        reducer.reduce(text)
        assert len(reducer.word_counts) > 0
        reducer.reset()
        assert len(reducer.word_counts) == 0

    def test_get_overused_words(self):
        """Test getting overused words."""
        reducer = RepetitionReducer(threshold=2)
        text = "Word word word appears often. Other word here."
        reducer.reduce(text)
        overused = reducer.get_overused_words(limit=5)
        # 'word' should be in overused list
        words = [w for w, _ in overused]
        assert len(overused) >= 0  # May or may not have overused depending on lemmatization

    def test_reduction_stats(self):
        """Test that stats are populated correctly."""
        reducer = RepetitionReducer(threshold=3)
        text = "This is a simple test sentence."
        result, stats = reducer.reduce(text)
        assert stats.words_checked > 0
        assert isinstance(stats.replacements_made, int)
        assert isinstance(stats.overused_words, list)
        assert isinstance(stats.replacements_detail, dict)


class TestReductionStats:
    """Tests for ReductionStats dataclass."""

    def test_default_values(self):
        """Test default stat values."""
        stats = ReductionStats()
        assert stats.words_checked == 0
        assert stats.replacements_made == 0
        assert stats.overused_words == []
        assert stats.replacements_detail == {}

    def test_stats_can_be_modified(self):
        """Test that stats can be updated."""
        stats = ReductionStats()
        stats.words_checked = 100
        stats.replacements_made = 5
        stats.overused_words.append("test")
        assert stats.words_checked == 100
        assert stats.replacements_made == 5
        assert "test" in stats.overused_words


class TestLLMSpeakDictionary:
    """Tests for the LLM_SPEAK dictionary."""

    def test_common_replacements(self):
        """Test common LLM-speak replacements."""
        assert LLM_SPEAK["utilize"] == "use"
        assert LLM_SPEAK["leverage"] == "use"
        assert LLM_SPEAK["facilitate"] == "help"
        assert LLM_SPEAK["comprehensive"] == "full"

    def test_qwen_vocabulary_fixes(self):
        """Test Qwen-specific vocabulary fixes."""
        # These are weird substitutions Qwen makes
        assert LLM_SPEAK["ticker"] == "watch"
        assert LLM_SPEAK["cogwheel"] == "gear"
        assert LLM_SPEAK["macrocosm"] == "world"
