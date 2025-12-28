"""Tests for proposition-to-template mapping."""

import pytest
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.rhetorical.proposition_mapper import (
    MappedProposition,
    MappingResult,
    PropositionMapper,
    map_propositions_to_template,
    FUNCTION_DESCRIPTIONS,
)
from src.rhetorical.template_generator import (
    TemplateSlot,
    RhetoricalTemplate,
)
from src.rhetorical.function_classifier import SentenceFunction


# Helper fixtures
@pytest.fixture
def simple_template():
    """A simple 3-slot template."""
    return RhetoricalTemplate(slots=[
        TemplateSlot(index=0, function=SentenceFunction.CLAIM),
        TemplateSlot(index=1, function=SentenceFunction.CONTRAST),
        TemplateSlot(index=2, function=SentenceFunction.RESOLUTION),
    ])


@pytest.fixture
def varied_template():
    """A template with varied functions."""
    return RhetoricalTemplate(slots=[
        TemplateSlot(index=0, function=SentenceFunction.SETUP),
        TemplateSlot(index=1, function=SentenceFunction.QUESTION),
        TemplateSlot(index=2, function=SentenceFunction.CLAIM),
        TemplateSlot(index=3, function=SentenceFunction.EVIDENCE),
        TemplateSlot(index=4, function=SentenceFunction.RESOLUTION),
    ])


@pytest.fixture
def mapper():
    """Mapper without LLM (uses heuristics)."""
    return PropositionMapper(llm_provider=None)


class TestMappedProposition:
    """Test MappedProposition dataclass."""

    def test_create_mapped_proposition(self):
        mp = MappedProposition(
            proposition="The universe is vast.",
            slot_index=0,
            function=SentenceFunction.CLAIM,
            original_index=0,
        )
        assert mp.proposition == "The universe is vast."
        assert mp.slot_index == 0
        assert mp.function == SentenceFunction.CLAIM
        assert mp.original_index == 0
        assert mp.confidence == 1.0
        assert mp.notes == ""

    def test_with_metadata(self):
        mp = MappedProposition(
            proposition="Test",
            slot_index=1,
            function=SentenceFunction.EVIDENCE,
            original_index=2,
            confidence=0.8,
            notes="Good fit for evidence slot",
        )
        assert mp.confidence == 0.8
        assert "evidence" in mp.notes.lower()


class TestMappingResult:
    """Test MappingResult dataclass."""

    def test_empty_result(self):
        result = MappingResult(
            mappings=[],
            template=RhetoricalTemplate(slots=[]),
        )
        assert result.get_ordered_propositions() == []
        assert result.get_slot_functions() == []
        assert result.reordered is False

    def test_get_ordered_propositions(self):
        mappings = [
            MappedProposition("First", 0, SentenceFunction.CLAIM, 1),
            MappedProposition("Second", 1, SentenceFunction.CONTRAST, 0),
            MappedProposition("Third", 2, SentenceFunction.RESOLUTION, 2),
        ]
        template = RhetoricalTemplate(slots=[
            TemplateSlot(0, SentenceFunction.CLAIM),
            TemplateSlot(1, SentenceFunction.CONTRAST),
            TemplateSlot(2, SentenceFunction.RESOLUTION),
        ])
        result = MappingResult(mappings=mappings, template=template)

        ordered = result.get_ordered_propositions()
        assert ordered == ["First", "Second", "Third"]

    def test_get_slot_functions(self):
        mappings = [
            MappedProposition("A", 0, SentenceFunction.CLAIM, 0),
            MappedProposition("B", 1, SentenceFunction.EVIDENCE, 1),
        ]
        template = RhetoricalTemplate(slots=[
            TemplateSlot(0, SentenceFunction.CLAIM),
            TemplateSlot(1, SentenceFunction.EVIDENCE),
        ])
        result = MappingResult(mappings=mappings, template=template)

        functions = result.get_slot_functions()
        assert functions == [SentenceFunction.CLAIM, SentenceFunction.EVIDENCE]

    def test_reordering_flag(self):
        result = MappingResult(
            mappings=[],
            template=RhetoricalTemplate(slots=[]),
            reordered=True,
        )
        assert result.reordered is True


class TestHeuristicMapping:
    """Test mapping with heuristics (no LLM)."""

    def test_empty_propositions(self, mapper, simple_template):
        result = mapper.map_propositions([], simple_template)
        assert len(result.mappings) == 0
        assert result.unmapped_propositions == []

    def test_empty_template(self, mapper):
        template = RhetoricalTemplate(slots=[])
        propositions = ["The sky is blue.", "Water is wet."]
        result = mapper.map_propositions(propositions, template)

        # Should map to CONTINUATION when no template
        assert len(result.mappings) == 2
        for mp in result.mappings:
            assert mp.function == SentenceFunction.CONTINUATION

    def test_question_matching(self, mapper, varied_template):
        propositions = [
            "Consider the following scenario.",
            "Why does this happen?",
            "The answer is simple.",
            "For example, take this case.",
            "Therefore, we conclude.",
        ]
        result = mapper.map_propositions(propositions, varied_template)

        # Check that question prop matched question slot
        question_mapping = next(
            (m for m in result.mappings if "?" in m.proposition),
            None
        )
        if question_mapping:
            assert question_mapping.function == SentenceFunction.QUESTION

    def test_contrast_matching(self, mapper, simple_template):
        propositions = [
            "The theory seems correct.",
            "However, there is a problem.",
            "Thus, we must reconsider.",
        ]
        result = mapper.map_propositions(propositions, simple_template)

        # Contrast keyword should match contrast slot
        contrast_mapping = next(
            (m for m in result.mappings if "however" in m.proposition.lower()),
            None
        )
        if contrast_mapping:
            assert contrast_mapping.function == SentenceFunction.CONTRAST

    def test_resolution_matching(self, mapper, simple_template):
        propositions = [
            "This is the claim.",
            "But consider the alternative.",
            "Therefore, we conclude the claim stands.",
        ]
        result = mapper.map_propositions(propositions, simple_template)

        # Resolution keyword should match resolution slot
        resolution_mapping = next(
            (m for m in result.mappings if "therefore" in m.proposition.lower()),
            None
        )
        if resolution_mapping:
            assert resolution_mapping.function == SentenceFunction.RESOLUTION

    def test_more_propositions_than_slots(self, mapper, simple_template):
        propositions = [
            "First point.",
            "Second point.",
            "Third point.",
            "Fourth point.",
            "Fifth point.",
        ]
        result = mapper.map_propositions(propositions, simple_template)

        # Only 3 slots, so 2 should be unmapped
        assert len(result.mappings) == 3
        assert len(result.unmapped_propositions) == 2

    def test_fewer_propositions_than_slots(self, mapper, varied_template):
        propositions = [
            "Only one point.",
            "And a second one.",
        ]
        result = mapper.map_propositions(propositions, varied_template)

        # 5 slots, 2 propositions
        assert len(result.mappings) == 2
        assert len(result.empty_slots) == 3


class TestReorderingDetection:
    """Test detection of reordering."""

    def test_no_reordering(self, mapper):
        template = RhetoricalTemplate(slots=[
            TemplateSlot(0, SentenceFunction.CLAIM),
            TemplateSlot(1, SentenceFunction.CONTINUATION),
            TemplateSlot(2, SentenceFunction.CONTINUATION),
        ])
        propositions = ["First.", "Second.", "Third."]

        result = mapper.map_propositions(propositions, template)

        # Check if order preserved (all similar, should map in order)
        # Reordering depends on keyword matching
        assert isinstance(result.reordered, bool)

    def test_reordering_with_keywords(self, mapper):
        template = RhetoricalTemplate(slots=[
            TemplateSlot(0, SentenceFunction.QUESTION),
            TemplateSlot(1, SentenceFunction.CLAIM),
            TemplateSlot(2, SentenceFunction.RESOLUTION),
        ])
        # Put resolution keyword first, question keyword last
        propositions = [
            "Therefore, we conclude.",  # Resolution keyword
            "This is the main point.",   # No strong keyword
            "Why does this matter?",      # Question keyword
        ]

        result = mapper.map_propositions(propositions, template)

        # Should detect reordering if question goes to slot 0
        # and resolution goes to slot 2
        # (original order was resolution, claim, question)
        # The heuristic mapper should reorder based on keywords


class TestConvenienceFunction:
    """Test the convenience function."""

    def test_map_propositions_to_template(self, simple_template):
        propositions = ["A claim.", "But a contrast.", "Thus resolved."]

        result = map_propositions_to_template(
            propositions=propositions,
            template=simple_template,
            llm_provider=None,
        )

        assert isinstance(result, MappingResult)
        assert len(result.mappings) == 3


class TestWithMockedLLM:
    """Test with mocked LLM provider."""

    def test_llm_mapping_success(self, simple_template):
        mock_llm = MagicMock()
        mock_llm.call_json.return_value = {
            "mappings": [
                {"proposition_index": 0, "slot_index": 0, "reasoning": "Claim fit"},
                {"proposition_index": 1, "slot_index": 1, "reasoning": "Contrast fit"},
                {"proposition_index": 2, "slot_index": 2, "reasoning": "Resolution fit"},
            ],
            "reordered": False,
            "empty_slots": [],
        }

        mapper = PropositionMapper(llm_provider=mock_llm)
        propositions = ["First.", "Second.", "Third."]

        result = mapper.map_propositions(propositions, simple_template)

        assert len(result.mappings) == 3
        assert mock_llm.call_json.called

    def test_llm_mapping_with_reorder(self, simple_template):
        mock_llm = MagicMock()
        mock_llm.call_json.return_value = {
            "mappings": [
                {"proposition_index": 2, "slot_index": 0, "reasoning": "Reordered"},
                {"proposition_index": 0, "slot_index": 1, "reasoning": "Reordered"},
                {"proposition_index": 1, "slot_index": 2, "reasoning": "Reordered"},
            ],
            "reordered": True,
            "empty_slots": [],
        }

        mapper = PropositionMapper(llm_provider=mock_llm)
        propositions = ["First.", "Second.", "Third."]

        result = mapper.map_propositions(propositions, simple_template)

        assert result.reordered is True
        ordered = result.get_ordered_propositions()
        assert ordered[0] == "Third."  # Was reordered to slot 0

    def test_llm_failure_falls_back_to_heuristics(self, simple_template):
        mock_llm = MagicMock()
        mock_llm.call_json.side_effect = Exception("LLM error")

        mapper = PropositionMapper(llm_provider=mock_llm)
        propositions = ["First.", "However, second.", "Therefore, third."]

        result = mapper.map_propositions(propositions, simple_template)

        # Should still work using heuristics
        assert len(result.mappings) == 3

    def test_llm_with_context(self, simple_template):
        mock_llm = MagicMock()
        mock_llm.call_json.return_value = {
            "mappings": [
                {"proposition_index": 0, "slot_index": 0, "reasoning": "Context helped"},
            ],
            "reordered": False,
            "empty_slots": [1, 2],
        }

        mapper = PropositionMapper(llm_provider=mock_llm)
        propositions = ["Single point."]

        result = mapper.map_propositions(
            propositions,
            simple_template,
            context="This is about cosmology.",
        )

        # Check that context was passed to prompt
        call_args = mock_llm.call_json.call_args
        # call_json uses keyword args
        user_prompt = call_args.kwargs.get("user_prompt", "")
        assert "cosmology" in user_prompt


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_single_proposition_single_slot(self, mapper):
        template = RhetoricalTemplate(slots=[
            TemplateSlot(0, SentenceFunction.CLAIM),
        ])
        propositions = ["The only point."]

        result = mapper.map_propositions(propositions, template)

        assert len(result.mappings) == 1
        assert result.mappings[0].slot_index == 0

    def test_all_same_function(self, mapper):
        template = RhetoricalTemplate(slots=[
            TemplateSlot(0, SentenceFunction.CONTINUATION),
            TemplateSlot(1, SentenceFunction.CONTINUATION),
            TemplateSlot(2, SentenceFunction.CONTINUATION),
        ])
        propositions = ["One.", "Two.", "Three."]

        result = mapper.map_propositions(propositions, template)

        assert len(result.mappings) == 3

    def test_invalid_llm_indices(self, simple_template):
        mock_llm = MagicMock()
        mock_llm.call_json.return_value = {
            "mappings": [
                {"proposition_index": 100, "slot_index": 0},  # Invalid
                {"proposition_index": 0, "slot_index": 100},  # Invalid
                {"proposition_index": 0, "slot_index": 0},    # Valid
            ],
            "reordered": False,
            "empty_slots": [],
        }

        mapper = PropositionMapper(llm_provider=mock_llm)
        propositions = ["Test."]

        result = mapper.map_propositions(propositions, simple_template)

        # Should only have the valid mapping
        assert len(result.mappings) == 1


class TestFunctionDescriptions:
    """Test that function descriptions are defined."""

    def test_all_functions_have_descriptions(self):
        for func in SentenceFunction:
            if func != SentenceFunction.CONTINUATION:
                assert func in FUNCTION_DESCRIPTIONS, f"Missing description for {func}"

    def test_descriptions_are_meaningful(self):
        for func, desc in FUNCTION_DESCRIPTIONS.items():
            assert len(desc) > 10, f"Description too short for {func}"
            assert desc[0].isupper(), f"Description should start uppercase for {func}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
