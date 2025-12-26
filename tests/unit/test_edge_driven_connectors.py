"""Tests for edge-driven connector selection (Phase 2.1)."""

import pytest
from unittest.mock import MagicMock

from src.models.plan import TransitionType
from src.models.style import AuthorProfile, StyleProfile
from src.ingestion.context_analyzer import GlobalContext
from src.generation.prompt_builder import PromptBuilder, DEFAULT_TRANSITION_WORDS


class TestEdgeDrivenConnectors:
    """Tests for selecting connectors based on graph edges and author vocabulary."""

    @pytest.fixture
    def author_with_transitions(self):
        """Author profile with custom transition words."""
        return AuthorProfile(
            name="TestAuthor",
            style_dna="Test author with distinctive transitions.",
            transitions={
                "causal": [("hence", 5), ("thus", 3), ("so", 2)],
                "adversative": [("yet", 4), ("but", 3), ("however", 2)],
                "additive": [("moreover", 3), ("also", 2), ("besides", 1)],
                "temporal": [("then", 2), ("afterwards", 1)],
            },
            voice_ratio=0.7,
        )

    @pytest.fixture
    def author_without_transitions(self):
        """Author profile without custom transition words."""
        return AuthorProfile(
            name="MinimalAuthor",
            style_dna="Author with no extracted transitions.",
            transitions={},
            voice_ratio=0.5,
        )

    @pytest.fixture
    def mock_global_context(self):
        """Mock global context for testing."""
        context = MagicMock(spec=GlobalContext)
        context.to_system_prompt.return_value = "You are a helpful assistant."
        return context

    def test_uses_author_causal_connectors(
        self, author_with_transitions, mock_global_context
    ):
        """Should use author's causal connectors when available."""
        style_profile = StyleProfile.from_author(author_with_transitions)
        builder = PromptBuilder(mock_global_context, style_profile)

        words = builder.get_transition_words(TransitionType.CAUSAL)

        # Should return author's words, not defaults
        assert "hence" in words
        assert "thus" in words
        # Default words should not be present
        assert "therefore" not in words

    def test_uses_author_adversative_connectors(
        self, author_with_transitions, mock_global_context
    ):
        """Should use author's adversative connectors when available."""
        style_profile = StyleProfile.from_author(author_with_transitions)
        builder = PromptBuilder(mock_global_context, style_profile)

        words = builder.get_transition_words(TransitionType.ADVERSATIVE)

        assert "yet" in words
        assert "but" in words
        # Default "nevertheless" should not be present
        assert "nevertheless" not in words

    def test_uses_author_additive_connectors(
        self, author_with_transitions, mock_global_context
    ):
        """Should use author's additive connectors when available."""
        style_profile = StyleProfile.from_author(author_with_transitions)
        builder = PromptBuilder(mock_global_context, style_profile)

        words = builder.get_transition_words(TransitionType.ADDITIVE)

        assert "moreover" in words
        assert "also" in words
        # Default "furthermore" should not be present
        assert "furthermore" not in words

    def test_falls_back_to_defaults_when_no_author_transitions(
        self, author_without_transitions, mock_global_context
    ):
        """Should fall back to defaults when author has no transitions."""
        style_profile = StyleProfile.from_author(author_without_transitions)
        builder = PromptBuilder(mock_global_context, style_profile)

        words = builder.get_transition_words(TransitionType.CAUSAL)

        # Should use default words
        assert "therefore" in words or "thus" in words

    def test_falls_back_to_defaults_when_no_style_profile(self, mock_global_context):
        """Should fall back to defaults when no style profile provided."""
        builder = PromptBuilder(mock_global_context, style_profile=None)

        words = builder.get_transition_words(TransitionType.CAUSAL)

        # Should use default words
        expected = DEFAULT_TRANSITION_WORDS[TransitionType.CAUSAL]
        assert words == expected

    def test_returns_empty_for_none_transition(
        self, author_with_transitions, mock_global_context
    ):
        """Should return empty list for NONE transition type."""
        style_profile = StyleProfile.from_author(author_with_transitions)
        builder = PromptBuilder(mock_global_context, style_profile)

        words = builder.get_transition_words(TransitionType.NONE)

        assert words == []

    def test_respects_author_transition_ordering(
        self, author_with_transitions, mock_global_context
    ):
        """Should maintain author's preference ordering (most frequent first)."""
        style_profile = StyleProfile.from_author(author_with_transitions)
        builder = PromptBuilder(mock_global_context, style_profile)

        words = builder.get_transition_words(TransitionType.CAUSAL)

        # "hence" has highest frequency (5), should be first
        assert words[0] == "hence"

    def test_limits_transition_words_to_five(
        self, mock_global_context
    ):
        """Should limit returned transition words to 5."""
        author = AuthorProfile(
            name="VerboseAuthor",
            style_dna="Author with many transitions.",
            transitions={
                "causal": [
                    ("hence", 10), ("thus", 9), ("so", 8), ("therefore", 7),
                    ("consequently", 6), ("accordingly", 5), ("ergo", 4),
                ],
            },
        )
        style_profile = StyleProfile.from_author(author)
        builder = PromptBuilder(mock_global_context, style_profile)

        words = builder.get_transition_words(TransitionType.CAUSAL)

        assert len(words) <= 5


class TestTransitionTypeMapping:
    """Tests for mapping relationship types to transition categories."""

    def test_relationship_to_transition_mapping(self):
        """Verify the transition type to category mapping is correct."""
        from src.generation.prompt_builder import TRANSITION_TYPE_TO_CATEGORY

        assert TRANSITION_TYPE_TO_CATEGORY[TransitionType.CAUSAL] == "causal"
        assert TRANSITION_TYPE_TO_CATEGORY[TransitionType.ADVERSATIVE] == "adversative"
        assert TRANSITION_TYPE_TO_CATEGORY[TransitionType.ADDITIVE] == "additive"
        assert TRANSITION_TYPE_TO_CATEGORY[TransitionType.TEMPORAL] == "temporal"
        assert TRANSITION_TYPE_TO_CATEGORY[TransitionType.NONE] is None
