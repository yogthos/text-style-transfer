"""Regression tests for rhetorical template system.

These tests verify the integrated system works correctly and catches
breaking changes in the rhetorical template pipeline.
"""

import pytest
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.rhetorical import (
    SentenceFunction,
    SentenceFunctionClassifier,
    ClassifiedSentence,
    RhetoricalTemplateGenerator,
    RhetoricalTemplate,
    TemplateSlot,
    PropositionMapper,
    MappingResult,
    FUNCTION_DESCRIPTIONS,
)
from src.style.profile import SentenceFunctionProfile, AuthorStyleProfile
from src.utils.prompts import load_prompt, clear_prompt_cache


class TestFunctionClassifierRegression:
    """Regression tests for function classification."""

    @pytest.fixture
    def classifier(self):
        return SentenceFunctionClassifier()

    def test_question_with_question_mark(self, classifier):
        """Questions ending with ? must be classified as QUESTION."""
        sentences = [
            "What is the meaning of life?",
            "How does this work?",
            "Why would anyone do this?",
            "Is this correct?",
        ]
        for sent in sentences:
            result = classifier.classify(sent)
            assert result.function == SentenceFunction.QUESTION, f"Failed: {sent}"

    def test_contrast_markers_at_start(self, classifier):
        """Sentences starting with contrast markers must be CONTRAST."""
        sentences = [
            "But this is wrong.",
            "However, the evidence suggests otherwise.",
            "Yet we must consider alternatives.",
            "Nevertheless, the point stands.",
        ]
        for sent in sentences:
            result = classifier.classify(sent)
            assert result.function == SentenceFunction.CONTRAST, f"Failed: {sent}"

    def test_resolution_markers(self, classifier):
        """Sentences with resolution markers must be RESOLUTION."""
        sentences = [
            "Therefore, we conclude this is true.",
            "Thus, the hypothesis is confirmed.",
            "Consequently, we must act.",
            "In conclusion, the data supports our theory.",
        ]
        for sent in sentences:
            result = classifier.classify(sent)
            assert result.function == SentenceFunction.RESOLUTION, f"Failed: {sent}"

    def test_setup_markers_at_start(self, classifier):
        """Sentences starting with setup markers must be SETUP."""
        sentences = [
            "Consider the following example.",
            "Imagine a world without gravity.",
            "Suppose we accept this premise.",
        ]
        for sent in sentences:
            result = classifier.classify(sent)
            assert result.function == SentenceFunction.SETUP, f"Failed: {sent}"

    def test_profile_extraction_structure(self, classifier):
        """Profile extraction must return correct structure."""
        paragraphs = [
            "Consider this problem. What causes it? But the answer is complex. Therefore, we need more research.",
            "The evidence is clear. However, some disagree. Thus, debate continues.",
        ]
        profile = classifier.extract_function_profile(paragraphs)

        assert "function_distribution" in profile
        assert "function_transitions" in profile
        assert "initial_function_probs" in profile
        assert "function_samples" in profile

        # Must have detected multiple function types
        assert len(profile["function_distribution"]) >= 3

        # Probabilities must sum to ~1
        dist_sum = sum(profile["function_distribution"].values())
        assert 0.99 <= dist_sum <= 1.01

        initial_sum = sum(profile["initial_function_probs"].values())
        assert 0.99 <= initial_sum <= 1.01


class TestTemplateGeneratorRegression:
    """Regression tests for template generation."""

    @pytest.fixture
    def generator(self):
        transitions = {
            "claim": {"evidence": 0.4, "contrast": 0.3, "elaboration": 0.3},
            "evidence": {"claim": 0.5, "resolution": 0.3, "elaboration": 0.2},
            "contrast": {"resolution": 0.5, "claim": 0.3, "elaboration": 0.2},
            "resolution": {"elaboration": 0.5, "continuation": 0.5},
        }
        initial = {"claim": 0.5, "setup": 0.3, "question": 0.2}
        return RhetoricalTemplateGenerator(
            function_transitions=transitions,
            initial_function_probs=initial,
        )

    def test_template_length_matches_request(self, generator):
        """Generated template must have requested number of slots."""
        for n in [1, 3, 5, 10]:
            template = generator.generate(n)
            assert len(template) == n

    def test_empty_template_for_zero(self, generator):
        """Zero slots must produce empty template."""
        template = generator.generate(0)
        assert len(template) == 0
        assert template.slots == []

    def test_seed_reproducibility(self, generator):
        """Same seed must produce identical templates."""
        t1 = generator.generate(5, seed=42)
        t2 = generator.generate(5, seed=42)

        funcs1 = [s.function for s in t1.slots]
        funcs2 = [s.function for s in t2.slots]
        assert funcs1 == funcs2

    def test_no_excessive_repetition(self, generator):
        """No more than 2 consecutive same functions."""
        for seed in range(20):
            template = generator.generate(10, seed=seed)
            funcs = [s.function for s in template.slots]

            for i in range(len(funcs) - 2):
                same_three = funcs[i] == funcs[i+1] == funcs[i+2]
                assert not same_three, f"Found 3+ consecutive at seed {seed}"

    def test_slot_indices_correct(self, generator):
        """Slot indices must be sequential starting from 0."""
        template = generator.generate(5)
        for i, slot in enumerate(template.slots):
            assert slot.index == i


class TestPropositionMapperRegression:
    """Regression tests for proposition mapping."""

    @pytest.fixture
    def mapper(self):
        return PropositionMapper(llm_provider=None)

    @pytest.fixture
    def template(self):
        return RhetoricalTemplate(slots=[
            TemplateSlot(0, SentenceFunction.CLAIM),
            TemplateSlot(1, SentenceFunction.CONTRAST),
            TemplateSlot(2, SentenceFunction.RESOLUTION),
        ])

    def test_empty_propositions_returns_empty(self, mapper, template):
        """Empty propositions must return empty mappings."""
        result = mapper.map_propositions([], template)
        assert result.mappings == []
        assert not result.reordered

    def test_mapping_count_matches_min(self, mapper, template):
        """Mappings count must not exceed slots or propositions."""
        props = ["First.", "Second.", "Third.", "Fourth.", "Fifth."]
        result = mapper.map_propositions(props, template)

        # Can't map more than slots available
        assert len(result.mappings) <= len(template.slots)
        # Unmapped should contain the rest
        assert len(result.mappings) + len(result.unmapped_propositions) == len(props)

    def test_question_maps_to_question_slot(self, mapper):
        """Questions should map to QUESTION slots when available."""
        template = RhetoricalTemplate(slots=[
            TemplateSlot(0, SentenceFunction.CLAIM),
            TemplateSlot(1, SentenceFunction.QUESTION),
            TemplateSlot(2, SentenceFunction.RESOLUTION),
        ])
        props = [
            "This is a claim.",
            "Why does this happen?",
            "Therefore, we conclude.",
        ]
        result = mapper.map_propositions(props, template)

        # Find the question mapping
        question_mapping = next(
            (m for m in result.mappings if "?" in m.proposition), None
        )
        if question_mapping:
            assert question_mapping.function == SentenceFunction.QUESTION

    def test_mapping_result_methods(self, mapper, template):
        """MappingResult helper methods must work correctly."""
        props = ["A.", "B.", "C."]
        result = mapper.map_propositions(props, template)

        ordered = result.get_ordered_propositions()
        assert len(ordered) == len(result.mappings)

        functions = result.get_slot_functions()
        assert len(functions) == len(result.mappings)


class TestStyleProfileIntegration:
    """Test function profile integration with style profile."""

    def test_function_profile_serialization(self):
        """Function profile must serialize and deserialize correctly."""
        profile = SentenceFunctionProfile(
            function_distribution={"claim": 0.4, "contrast": 0.3, "resolution": 0.3},
            function_transitions={"claim": {"contrast": 0.5, "resolution": 0.5}},
            initial_function_probs={"claim": 0.6, "setup": 0.4},
            function_samples={"claim": ["This is a claim.", "Another claim."]},
        )

        # Serialize
        data = profile.to_dict()

        # Deserialize
        restored = SentenceFunctionProfile.from_dict(data)

        assert restored.function_distribution == profile.function_distribution
        assert restored.function_transitions == profile.function_transitions
        assert restored.initial_function_probs == profile.initial_function_probs
        assert restored.function_samples == profile.function_samples

    def test_empty_function_profile(self):
        """Empty function profile must not break serialization."""
        profile = SentenceFunctionProfile()

        data = profile.to_dict()
        restored = SentenceFunctionProfile.from_dict(data)

        assert restored.function_distribution == {}
        assert restored.function_transitions == {}


class TestPromptLoadingRegression:
    """Regression tests for prompt loading."""

    def test_proposition_mapper_prompt_exists(self):
        """Proposition mapper prompt must exist and load."""
        clear_prompt_cache()
        prompt = load_prompt("proposition_mapper_system")

        assert len(prompt) > 100
        assert "rhetorical" in prompt.lower()
        assert "JSON" in prompt

    def test_prompt_cache_works(self):
        """Prompt cache must return same object."""
        clear_prompt_cache()

        p1 = load_prompt("proposition_mapper_system")
        p2 = load_prompt("proposition_mapper_system")

        # Should be cached (same content)
        assert p1 == p2

    def test_missing_prompt_raises(self):
        """Missing prompt must raise FileNotFoundError."""
        clear_prompt_cache()

        with pytest.raises(FileNotFoundError):
            load_prompt("nonexistent_prompt_file")


class TestEndToEndPipeline:
    """End-to-end tests for the full rhetorical pipeline."""

    def test_corpus_to_template_pipeline(self):
        """Full pipeline: corpus → profile → template → mapping."""
        # 1. Classify corpus
        classifier = SentenceFunctionClassifier()
        paragraphs = [
            "Consider the nature of reality. What is consciousness? "
            "But this question has puzzled philosophers for centuries. "
            "Therefore, we may never have a complete answer.",

            "The evidence suggests a new theory. However, critics disagree. "
            "Admittedly, the data is limited. Thus, more research is needed.",
        ]
        profile_data = classifier.extract_function_profile(paragraphs)

        # 2. Create template generator from profile
        generator = RhetoricalTemplateGenerator(
            function_transitions=profile_data["function_transitions"],
            initial_function_probs=profile_data["initial_function_probs"],
        )

        # 3. Generate template for 4 propositions
        template = generator.generate(4, seed=123)
        assert len(template) == 4

        # 4. Map propositions to template
        mapper = PropositionMapper(llm_provider=None)
        propositions = [
            "The universe is vast.",
            "But how vast exactly?",
            "Scientists have measured it.",
            "Therefore, we know its size.",
        ]
        result = mapper.map_propositions(propositions, template)

        # 5. Verify result
        assert len(result.mappings) > 0
        ordered = result.get_ordered_propositions()
        assert len(ordered) == len(result.mappings)

    def test_default_transitions_work(self):
        """Generator with empty transitions must still work."""
        generator = RhetoricalTemplateGenerator(
            function_transitions={},
            initial_function_probs={},
        )

        template = generator.generate(5, seed=42)
        assert len(template) == 5

        # All slots should have valid functions
        for slot in template.slots:
            assert isinstance(slot.function, SentenceFunction)


class TestFunctionDescriptions:
    """Test that function descriptions are complete and useful."""

    def test_all_functions_described(self):
        """All non-CONTINUATION functions must have descriptions."""
        for func in SentenceFunction:
            if func != SentenceFunction.CONTINUATION:
                assert func in FUNCTION_DESCRIPTIONS, f"Missing: {func}"

    def test_descriptions_are_informative(self):
        """Descriptions must be substantive."""
        for func, desc in FUNCTION_DESCRIPTIONS.items():
            assert len(desc) >= 20, f"Too short for {func}"
            # Should describe what the function does
            assert any(word in desc.lower() for word in [
                "assert", "support", "question", "acknowledge",
                "opposition", "resolve", "expand", "introduce", "continue"
            ]), f"Not descriptive enough for {func}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
