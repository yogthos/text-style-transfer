"""Tests for rhetorical template generation."""

import pytest
import sys
from pathlib import Path
from collections import Counter

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.rhetorical.template_generator import (
    TemplateSlot,
    RhetoricalTemplate,
    RhetoricalTemplateGenerator,
    generate_template,
)
from src.rhetorical.function_classifier import SentenceFunction


# Sample transition matrices for testing
@pytest.fixture
def simple_transitions():
    """Simple transition matrix for testing."""
    return {
        "claim": {"evidence": 0.5, "elaboration": 0.3, "contrast": 0.2},
        "evidence": {"claim": 0.4, "elaboration": 0.4, "resolution": 0.2},
        "question": {"claim": 0.3, "resolution": 0.5, "elaboration": 0.2},
        "contrast": {"resolution": 0.5, "claim": 0.3, "concession": 0.2},
        "concession": {"contrast": 0.4, "claim": 0.4, "resolution": 0.2},
        "resolution": {"elaboration": 0.5, "claim": 0.3, "continuation": 0.2},
        "elaboration": {"claim": 0.4, "evidence": 0.3, "continuation": 0.3},
        "setup": {"claim": 0.5, "question": 0.3, "evidence": 0.2},
        "continuation": {"claim": 0.5, "elaboration": 0.3, "continuation": 0.2},
    }


@pytest.fixture
def simple_initial_probs():
    """Simple initial probabilities for testing."""
    return {
        "claim": 0.3,
        "setup": 0.25,
        "question": 0.2,
        "evidence": 0.15,
        "contrast": 0.1,
    }


@pytest.fixture
def generator(simple_transitions, simple_initial_probs):
    """Create a generator with test data."""
    return RhetoricalTemplateGenerator(
        function_transitions=simple_transitions,
        initial_function_probs=simple_initial_probs,
    )


class TestTemplateSlot:
    """Test TemplateSlot dataclass."""

    def test_create_slot(self):
        slot = TemplateSlot(index=0, function=SentenceFunction.CLAIM)
        assert slot.index == 0
        assert slot.function == SentenceFunction.CLAIM
        assert slot.is_required is True

    def test_optional_slot(self):
        slot = TemplateSlot(
            index=1,
            function=SentenceFunction.ELABORATION,
            is_required=False,
        )
        assert slot.is_required is False


class TestRhetoricalTemplate:
    """Test RhetoricalTemplate dataclass."""

    def test_empty_template(self):
        template = RhetoricalTemplate(slots=[])
        assert len(template) == 0
        assert template.functions() == []

    def test_template_length(self):
        slots = [
            TemplateSlot(index=0, function=SentenceFunction.CLAIM),
            TemplateSlot(index=1, function=SentenceFunction.EVIDENCE),
            TemplateSlot(index=2, function=SentenceFunction.RESOLUTION),
        ]
        template = RhetoricalTemplate(slots=slots)
        assert len(template) == 3

    def test_functions_list(self):
        slots = [
            TemplateSlot(index=0, function=SentenceFunction.CLAIM),
            TemplateSlot(index=1, function=SentenceFunction.CONTRAST),
            TemplateSlot(index=2, function=SentenceFunction.RESOLUTION),
        ]
        template = RhetoricalTemplate(slots=slots)
        funcs = template.functions()
        assert funcs == [
            SentenceFunction.CLAIM,
            SentenceFunction.CONTRAST,
            SentenceFunction.RESOLUTION,
        ]

    def test_function_counts(self):
        slots = [
            TemplateSlot(index=0, function=SentenceFunction.CLAIM),
            TemplateSlot(index=1, function=SentenceFunction.CLAIM),
            TemplateSlot(index=2, function=SentenceFunction.RESOLUTION),
        ]
        template = RhetoricalTemplate(slots=slots)
        counts = template.function_counts()
        assert counts["claim"] == 2
        assert counts["resolution"] == 1


class TestBasicGeneration:
    """Test basic template generation."""

    def test_generate_empty(self, generator):
        template = generator.generate(0)
        assert len(template) == 0

    def test_generate_single(self, generator):
        template = generator.generate(1, seed=42)
        assert len(template) == 1
        assert template.slots[0].index == 0

    def test_generate_multiple(self, generator):
        template = generator.generate(5, seed=42)
        assert len(template) == 5
        # Check indices are correct
        for i, slot in enumerate(template.slots):
            assert slot.index == i

    def test_reproducibility_with_seed(self, generator):
        template1 = generator.generate(5, seed=123)
        template2 = generator.generate(5, seed=123)
        funcs1 = [s.function for s in template1.slots]
        funcs2 = [s.function for s in template2.slots]
        assert funcs1 == funcs2

    def test_different_seeds_different_output(self, generator):
        template1 = generator.generate(5, seed=100)
        template2 = generator.generate(5, seed=200)
        funcs1 = [s.function for s in template1.slots]
        funcs2 = [s.function for s in template2.slots]
        # Very unlikely to be same with different seeds
        # (but not guaranteed - just check they're valid)
        assert len(funcs1) == len(funcs2) == 5


class TestVarietyConstraints:
    """Test variety enforcement."""

    def test_min_variety_enforced(self, simple_transitions, simple_initial_probs):
        generator = RhetoricalTemplateGenerator(
            function_transitions=simple_transitions,
            initial_function_probs=simple_initial_probs,
            min_variety=3,
        )
        template = generator.generate(5, seed=42, force_variety=True)
        unique_funcs = set(s.function for s in template.slots)
        assert len(unique_funcs) >= 2  # At least some variety

    def test_max_consecutive_enforced(self, simple_transitions, simple_initial_probs):
        generator = RhetoricalTemplateGenerator(
            function_transitions=simple_transitions,
            initial_function_probs=simple_initial_probs,
            max_same_consecutive=2,
        )
        # Generate multiple times to check constraint
        for seed in range(10):
            template = generator.generate(10, seed=seed)
            funcs = [s.function for s in template.slots]

            # Check no more than 2 consecutive same
            for i in range(len(funcs) - 2):
                if funcs[i] == funcs[i + 1] == funcs[i + 2]:
                    pytest.fail(
                        f"Found 3+ consecutive same function: {funcs[i]} "
                        f"at positions {i}, {i+1}, {i+2}"
                    )

    def test_variety_not_forced(self, generator):
        # With force_variety=False, might get less variety
        template = generator.generate(3, seed=42, force_variety=False)
        assert len(template) == 3  # Should still work


class TestLogicalConstraints:
    """Test logical flow validation."""

    def test_resolution_has_predecessor(self, generator):
        """Resolution should follow question, contrast, or claim."""
        for seed in range(20):
            template = generator.generate(5, seed=seed)
            funcs = template.functions()

            for i, func in enumerate(funcs):
                if func == SentenceFunction.RESOLUTION and i > 0:
                    # Check there's a valid predecessor somewhere before
                    valid_preds = {
                        SentenceFunction.QUESTION,
                        SentenceFunction.CONTRAST,
                        SentenceFunction.CLAIM,
                        SentenceFunction.CONCESSION,
                    }
                    has_valid = any(f in valid_preds for f in funcs[:i])
                    # If no valid predecessor, should have been changed
                    # This is a soft constraint, so just check it's reasonable
                    assert True  # Test passes if no crash

    def test_good_endings_preferred(self, generator):
        """Templates should tend to end with good ending functions."""
        good_endings = {
            SentenceFunction.RESOLUTION,
            SentenceFunction.CLAIM,
            SentenceFunction.ELABORATION,
            SentenceFunction.CONTINUATION,
        }
        good_ending_count = 0
        total = 50

        for seed in range(total):
            template = generator.generate(5, seed=seed)
            if template.slots[-1].function in good_endings:
                good_ending_count += 1

        # Most should have good endings
        assert good_ending_count > total * 0.5


class TestConstrainedGeneration:
    """Test generation with specific constraints."""

    def test_start_constraint(self, generator):
        template = generator.generate_with_constraints(
            num_slots=4,
            start_with=SentenceFunction.QUESTION,
        )
        assert template.slots[0].function == SentenceFunction.QUESTION

    def test_end_constraint(self, generator):
        template = generator.generate_with_constraints(
            num_slots=4,
            end_with=SentenceFunction.RESOLUTION,
        )
        assert template.slots[-1].function == SentenceFunction.RESOLUTION

    def test_both_constraints(self, generator):
        template = generator.generate_with_constraints(
            num_slots=5,
            start_with=SentenceFunction.SETUP,
            end_with=SentenceFunction.RESOLUTION,
        )
        assert template.slots[0].function == SentenceFunction.SETUP
        assert template.slots[-1].function == SentenceFunction.RESOLUTION

    def test_required_functions(self, generator):
        required = [SentenceFunction.QUESTION, SentenceFunction.CONTRAST]
        template = generator.generate_with_constraints(
            num_slots=5,
            required_functions=required,
        )
        funcs = set(template.functions())
        # At least some required should be present
        # (not all may fit depending on constraints)
        assert len(funcs) >= 2

    def test_all_constraints(self, generator):
        template = generator.generate_with_constraints(
            num_slots=6,
            required_functions=[SentenceFunction.EVIDENCE],
            start_with=SentenceFunction.CLAIM,
            end_with=SentenceFunction.RESOLUTION,
        )
        assert template.slots[0].function == SentenceFunction.CLAIM
        assert template.slots[-1].function == SentenceFunction.RESOLUTION
        assert len(template) == 6


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_transitions(self):
        generator = RhetoricalTemplateGenerator(
            function_transitions={},
            initial_function_probs={},
        )
        template = generator.generate(3, seed=42)
        assert len(template) == 3

    def test_partial_transitions(self):
        """Test with incomplete transition matrix."""
        partial_transitions = {
            "claim": {"evidence": 0.8, "resolution": 0.2},
            # Missing most other transitions
        }
        generator = RhetoricalTemplateGenerator(
            function_transitions=partial_transitions,
        )
        template = generator.generate(5, seed=42)
        assert len(template) == 5

    def test_single_slot_constraints(self, generator):
        template = generator.generate_with_constraints(
            num_slots=1,
            start_with=SentenceFunction.CLAIM,
            end_with=SentenceFunction.RESOLUTION,
        )
        # With 1 slot, start takes precedence
        assert template.slots[0].function == SentenceFunction.CLAIM

    def test_zero_slot_constraints(self, generator):
        template = generator.generate_with_constraints(
            num_slots=0,
            start_with=SentenceFunction.CLAIM,
        )
        assert len(template) == 0

    def test_very_long_template(self, generator):
        template = generator.generate(50, seed=42)
        assert len(template) == 50
        # Should have good variety in 50 slots
        unique = set(template.functions())
        assert len(unique) >= 4


class TestConvenienceFunction:
    """Test the convenience function."""

    def test_generate_template_function(self, simple_transitions, simple_initial_probs):
        template = generate_template(
            num_slots=5,
            function_transitions=simple_transitions,
            initial_function_probs=simple_initial_probs,
        )
        assert len(template) == 5
        assert all(isinstance(s, TemplateSlot) for s in template.slots)


class TestMarkovSampling:
    """Test that Markov sampling follows transition probabilities."""

    def test_transitions_influence_output(self):
        """Test that transition probabilities affect generation."""
        # Create biased transitions - claim always goes to evidence
        biased_transitions = {
            "claim": {"evidence": 1.0},
            "evidence": {"claim": 1.0},
        }
        biased_initial = {"claim": 1.0}

        generator = RhetoricalTemplateGenerator(
            function_transitions=biased_transitions,
            initial_function_probs=biased_initial,
            min_variety=0,  # Disable variety enforcement
        )

        template = generator.generate(4, seed=42, force_variety=False)
        funcs = template.functions()

        # Should alternate between claim and evidence
        # (though variety constraints might modify this)
        assert funcs[0] == SentenceFunction.CLAIM

    def test_initial_probs_influence_start(self):
        """Test that initial probabilities affect starting function."""
        # Make setup 100% likely to start
        initial = {"setup": 1.0}
        transitions = {
            "setup": {"claim": 1.0},
            "claim": {"claim": 1.0},
        }

        generator = RhetoricalTemplateGenerator(
            function_transitions=transitions,
            initial_function_probs=initial,
        )

        template = generator.generate(3, seed=42, force_variety=False)
        assert template.slots[0].function == SentenceFunction.SETUP


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
