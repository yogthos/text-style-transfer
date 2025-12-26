"""Tests for cross-reference handling (Phase 2.2)."""

import pytest
from unittest.mock import MagicMock

from src.models.graph import PropositionNode
from src.models.plan import SentenceNode, SentenceRole, TransitionType
from src.ingestion.context_analyzer import GlobalContext
from src.generation.prompt_builder import PromptBuilder


class TestEntityTracking:
    """Tests for entity introduction tracking."""

    @pytest.fixture
    def mock_global_context(self):
        """Mock global context for testing."""
        context = MagicMock(spec=GlobalContext)
        context.to_system_prompt.return_value = "You are a helpful assistant."
        return context

    @pytest.fixture
    def builder(self, mock_global_context):
        """Create a PromptBuilder instance."""
        return PromptBuilder(mock_global_context, style_profile=None)

    def test_initial_state_has_no_entities(self, builder):
        """Should start with no introduced entities."""
        assert builder._introduced_entities == set()

    def test_register_entities_adds_to_tracking(self, builder):
        """Should track registered entities."""
        builder.register_introduced_entities(["The Theory", "Einstein"])

        assert "the theory" in builder._introduced_entities
        assert "einstein" in builder._introduced_entities

    def test_register_normalizes_case(self, builder):
        """Should normalize entity names to lowercase."""
        builder.register_introduced_entities(["QUANTUM Physics", "EINSTEIN"])

        assert "quantum physics" in builder._introduced_entities
        assert "einstein" in builder._introduced_entities

    def test_register_strips_whitespace(self, builder):
        """Should strip whitespace from entity names."""
        builder.register_introduced_entities(["  The Theory  ", "  Einstein  "])

        assert "the theory" in builder._introduced_entities
        assert "einstein" in builder._introduced_entities

    def test_reset_clears_entities(self, builder):
        """Should clear all tracked entities on reset."""
        builder.register_introduced_entities(["Theory", "Einstein"])
        builder.reset_entity_tracking()

        assert builder._introduced_entities == set()


class TestReferenceGuidance:
    """Tests for generating entity reference guidance."""

    @pytest.fixture
    def mock_global_context(self):
        """Mock global context for testing."""
        context = MagicMock(spec=GlobalContext)
        context.to_system_prompt.return_value = "You are a helpful assistant."
        return context

    @pytest.fixture
    def builder(self, mock_global_context):
        """Create a PromptBuilder instance."""
        return PromptBuilder(mock_global_context, style_profile=None)

    def test_no_guidance_when_no_prior_entities(self, builder):
        """Should return empty string when no entities previously introduced."""
        prop = PropositionNode(
            id="p1",
            text="Einstein developed the theory.",
            subject="Einstein",
            verb="developed",
            object="theory",
            entities=["Einstein", "theory"],
        )

        guidance = builder.get_reference_guidance([prop])

        assert guidance == ""

    def test_no_guidance_when_no_overlap(self, builder):
        """Should return empty when current entities don't overlap with prior."""
        builder.register_introduced_entities(["Newton", "gravity"])

        prop = PropositionNode(
            id="p1",
            text="Einstein developed the theory.",
            subject="Einstein",
            verb="developed",
            object="theory",
            entities=["Einstein", "theory"],
        )

        guidance = builder.get_reference_guidance([prop])

        assert guidance == ""

    def test_guidance_when_entity_repeated(self, builder):
        """Should provide guidance when entity was previously introduced."""
        builder.register_introduced_entities(["Einstein", "relativity"])

        prop = PropositionNode(
            id="p1",
            text="Einstein proved the theory.",
            subject="Einstein",
            verb="proved",
            object="theory",
            entities=["Einstein"],
        )

        guidance = builder.get_reference_guidance([prop])

        assert "einstein" in guidance.lower()
        assert "pronoun" in guidance.lower()

    def test_guidance_case_insensitive_matching(self, builder):
        """Should match entities case-insensitively."""
        builder.register_introduced_entities(["einstein"])

        prop = PropositionNode(
            id="p1",
            text="EINSTEIN proved the theory.",
            subject="EINSTEIN",
            verb="proved",
            entities=["EINSTEIN"],
        )

        guidance = builder.get_reference_guidance([prop])

        assert "einstein" in guidance.lower()

    def test_guidance_limits_to_three_entities(self, builder):
        """Should limit guidance to 3 entities for readability."""
        builder.register_introduced_entities(["entity1", "entity2", "entity3", "entity4", "entity5"])

        prop = PropositionNode(
            id="p1",
            text="All entities mentioned.",
            subject="All",
            verb="mentioned",
            entities=["entity1", "entity2", "entity3", "entity4", "entity5"],
        )

        guidance = builder.get_reference_guidance([prop])

        # Should have at most 3 entities mentioned
        entity_count = sum(1 for e in ["entity1", "entity2", "entity3", "entity4", "entity5"]
                          if e in guidance.lower())
        assert entity_count <= 3

    def test_guidance_includes_multiple_overlapping_entities(self, builder):
        """Should mention multiple overlapping entities."""
        builder.register_introduced_entities(["Einstein", "relativity"])

        prop = PropositionNode(
            id="p1",
            text="Einstein refined relativity.",
            subject="Einstein",
            verb="refined",
            object="relativity",
            entities=["Einstein", "relativity"],
        )

        guidance = builder.get_reference_guidance([prop])

        assert "einstein" in guidance.lower()
        assert "relativity" in guidance.lower()


class TestPromptIntegration:
    """Tests for cross-reference guidance integration into prompts."""

    @pytest.fixture
    def mock_global_context(self):
        """Mock global context for testing."""
        context = MagicMock(spec=GlobalContext)
        context.to_system_prompt.return_value = "You are a helpful assistant."
        return context

    @pytest.fixture
    def builder(self, mock_global_context):
        """Create a PromptBuilder instance."""
        return PromptBuilder(mock_global_context, style_profile=None)

    def test_prompt_includes_reference_guidance_when_applicable(self, builder):
        """Sentence prompt should include entity reference guidance."""
        # Introduce an entity first
        builder.register_introduced_entities(["quantum mechanics"])

        # Create proposition that references the entity
        prop = PropositionNode(
            id="p1",
            text="Quantum mechanics revolutionized physics.",
            subject="Quantum mechanics",
            verb="revolutionized",
            object="physics",
            entities=["quantum mechanics"],
        )

        node = SentenceNode(
            id="s1",
            propositions=[prop],
            role=SentenceRole.ELABORATION,
            transition=TransitionType.NONE,
            target_length=10,
        )

        # Create a minimal mock plan
        from src.models.plan import SentencePlan
        plan = MagicMock(spec=SentencePlan)
        plan.paragraph_role = "BODY"

        prompt = builder.build_sentence_prompt(node, plan, previous_sentence=None)

        # The user prompt should contain reference guidance
        assert "ENTITY REFERENCE NOTE" in prompt.user_prompt
        assert "quantum mechanics" in prompt.user_prompt.lower()

    def test_prompt_omits_reference_guidance_when_not_applicable(self, builder):
        """Sentence prompt should not include guidance when no overlap."""
        # No prior entities registered

        prop = PropositionNode(
            id="p1",
            text="Quantum mechanics revolutionized physics.",
            subject="Quantum mechanics",
            verb="revolutionized",
            object="physics",
            entities=["quantum mechanics"],
        )

        node = SentenceNode(
            id="s1",
            propositions=[prop],
            role=SentenceRole.THESIS,
            transition=TransitionType.NONE,
            target_length=10,
        )

        from src.models.plan import SentencePlan
        plan = MagicMock(spec=SentencePlan)
        plan.paragraph_role = "INTRO"

        prompt = builder.build_sentence_prompt(node, plan, previous_sentence=None)

        # Should not contain entity reference note
        assert "ENTITY REFERENCE NOTE" not in prompt.user_prompt
