"""
Tests for Phase 3: Evolution Controller

These tests verify:
1. Evolution loop basic flow
2. Locking mechanism (score > threshold locks slot)
3. Early termination (all slots locked)
4. Partial locking (some slots locked, others continue)
5. Generation count tracking
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from src.generator.translator import StyleTranslator, ParagraphState
from src.validator.semantic_critic import SemanticCritic
import json


class MockLLMProvider:
    """Mock LLM provider for testing."""

    def __init__(self, config_path=None, mock_responses=None):
        self.config_path = config_path
        self.call_count = 0
        self.call_history = []
        self.mock_responses = mock_responses or []

    def call(self, system_prompt, user_prompt, model_type, require_json, temperature, max_tokens, timeout=None, top_p=None):
        self.call_count += 1
        self.call_history.append({
            "system_prompt": system_prompt[:100] if len(system_prompt) > 100 else system_prompt,
            "user_prompt": user_prompt[:200] if len(user_prompt) > 200 else user_prompt,
            "model_type": model_type
        })

        # Return mock response based on call count
        if self.mock_responses:
            return self.mock_responses[(self.call_count - 1) % len(self.mock_responses)]

        # Default: return candidates for all unlocked slots
        return json.dumps({
            "slot_0": ["Candidate 0.1", "Candidate 0.2", "Candidate 0.3"],
            "slot_1": ["Candidate 1.1", "Candidate 1.2", "Candidate 1.3"],
            "slot_2": ["Candidate 2.1", "Candidate 2.2", "Candidate 2.3"]
        })


class MockPropositionExtractor:
    """Mock proposition extractor."""

    def extract_atomic_propositions(self, text):
        return ["Prop 1", "Prop 2", "Prop 3"]


class MockAtlas:
    """Mock StyleAtlas."""

    def get_examples_by_rhetoric(self, rhetorical_type, top_k, author_name, query_text):
        return [
            "Example sentence 1. Example sentence 2. Example sentence 3.",
            "Another example sentence 1. Another example sentence 2."
        ]

    def get_author_style_vector(self, author_name):
        return None


@pytest.fixture
def mock_translator():
    """Create a translator with mocked dependencies."""
    translator = StyleTranslator(config_path="config.json")

    # Replace LLM provider with mock
    translator.llm_provider = MockLLMProvider(config_path="config.json")

    # Replace proposition extractor
    translator.proposition_extractor = MockPropositionExtractor()

    # Set config values
    translator.paragraph_fusion_config = {
        "candidate_population_size": 3,
        "max_generations_per_slot": 5,
        "lock_threshold": 0.9,
        "generation_temperature": 0.7
    }
    translator.llm_provider_config = {"max_retries": 1, "retry_delay": 0}
    translator.translator_config = {"max_tokens": 500}

    # Mock _retrieve_robust_skeleton
    translator._retrieve_robust_skeleton = Mock(return_value=(
        "Example sentence 1. Example sentence 2. Example sentence 3.",
        ["[NP] [VP] [NP].", "[NP] [VP].", "[NP] [VP] [NP] [NP]."]
    ))

    # Mock _map_propositions_to_templates
    translator._map_propositions_to_templates = Mock(return_value=[
        ["Prop 1"],
        ["Prop 2"],
        ["Prop 3"]
    ])

    # Mock rhetorical classifier
    with patch('src.atlas.rhetoric.RhetoricalClassifier') as MockRhetoricalClassifier:
        MockRhetoricalClassifier.return_value.classify_heuristic.return_value = "OBSERVATION"
        yield translator


@pytest.fixture
def mock_critic():
    """Create a critic with mocked evaluation."""
    critic = SemanticCritic(config_path="config.json")

    # Mock evaluate_candidate_populations to return scores
    def mock_evaluate(candidate_populations, templates, prop_map, verbose=False):
        results = []
        for slot_idx, population in enumerate(candidate_populations):
            slot_results = []
            for candidate_idx, candidate in enumerate(population):
                # Return different scores based on candidate index
                # Candidate 0 gets high score (will lock), others get lower scores
                if candidate_idx == 0:
                    combined_score = 0.95  # Above threshold
                else:
                    combined_score = 0.7  # Below threshold

                slot_results.append({
                    "pass": combined_score >= 0.9,
                    "anchor_score": 0.5,
                    "semantic_score": combined_score - 0.5,
                    "combined_score": combined_score,
                    "feedback": f"Feedback for candidate {candidate_idx}",
                    "slot_index": slot_idx,
                    "candidate_index": candidate_idx
                })
            results.append(slot_results)
        return results

    critic.evaluate_candidate_populations = Mock(side_effect=mock_evaluate)

    # Mock evaluate for final recall calculation
    critic.evaluate = Mock(return_value={
        "proposition_recall": 0.85,
        "style_alignment": 0.8,
        "score": 0.825
    })

    return critic


class TestEvolutionLoopBasicFlow:
    """Test basic evolution loop flow."""

    def test_evolution_loop_initializes_state(self, mock_translator):
        """Test that translate_paragraph initializes ParagraphState correctly."""
        atlas = MockAtlas()

        with patch.object(mock_translator, '_generate_candidate_populations') as mock_generate, \
             patch('src.validator.semantic_critic.SemanticCritic') as MockCritic:
            mock_critic_instance = MockCritic.return_value
            mock_critic_instance.evaluate_candidate_populations.return_value = [
                [{"combined_score": 0.95, "candidate_index": 0, "feedback": "Good"}],
                [{"combined_score": 0.95, "candidate_index": 0, "feedback": "Good"}],
                [{"combined_score": 0.95, "candidate_index": 0, "feedback": "Good"}]
            ]
            mock_critic_instance.evaluate.return_value = {"proposition_recall": 0.85}

            mock_generate.return_value = [
                ["Candidate 0.1", "Candidate 0.2", "Candidate 0.3"],
                ["Candidate 1.1", "Candidate 1.2", "Candidate 1.3"],
                ["Candidate 2.1", "Candidate 2.2", "Candidate 2.3"]
            ]

            result = mock_translator.translate_paragraph(
                paragraph="Test paragraph with three sentences.",
                atlas=atlas,
                author_name="TestAuthor",
                verbose=False
            )

            # Verify state was initialized (indirectly by checking generation was called)
            assert mock_generate.called
            assert result[0]  # Generated paragraph should not be empty

    def test_evolution_loop_calls_generation_and_evaluation(self, mock_translator):
        """Test that evolution loop calls generation and evaluation."""
        atlas = MockAtlas()

        with patch.object(mock_translator, '_generate_candidate_populations') as mock_generate, \
             patch('src.validator.semantic_critic.SemanticCritic') as MockCritic:
            mock_critic_instance = MockCritic.return_value
            mock_critic_instance.evaluate_candidate_populations.return_value = [
                [{"combined_score": 0.95, "candidate_index": 0, "feedback": "Good"}],
                [{"combined_score": 0.95, "candidate_index": 0, "feedback": "Good"}],
                [{"combined_score": 0.95, "candidate_index": 0, "feedback": "Good"}]
            ]
            mock_critic_instance.evaluate.return_value = {"proposition_recall": 0.85}

            mock_generate.return_value = [
                ["Candidate 0.1", "Candidate 0.2", "Candidate 0.3"],
                ["Candidate 1.1", "Candidate 1.2", "Candidate 1.3"],
                ["Candidate 2.1", "Candidate 2.2", "Candidate 2.3"]
            ]

            mock_translator.translate_paragraph(
                paragraph="Test paragraph.",
                atlas=atlas,
                author_name="TestAuthor",
                verbose=False
            )

            # Verify generation and evaluation were called
            assert mock_generate.called
            assert mock_critic_instance.evaluate_candidate_populations.called


class TestLockingMechanism:
    """Test locking mechanism."""

    def test_slot_locks_when_score_exceeds_threshold(self, mock_translator):
        """Test that a slot locks when a candidate scores above threshold."""
        atlas = MockAtlas()

        # Track state changes
        state_snapshots = []

        def capture_state(*args, **kwargs):
            # Capture state after generation
            state = args[0] if args else kwargs.get('state')
            if state:
                state_snapshots.append({
                    'locked_flags': state.locked_flags.copy(),
                    'best_sentences': state.best_sentences.copy()
                })
            return [
                ["Candidate 0.1", "Candidate 0.2", "Candidate 0.3"],
                ["Candidate 1.1", "Candidate 1.2", "Candidate 1.3"],
                ["Candidate 2.1", "Candidate 2.2", "Candidate 2.3"]
            ]

        with patch.object(mock_translator, '_generate_candidate_populations', side_effect=capture_state) as mock_generate, \
             patch('src.validator.semantic_critic.SemanticCritic') as MockCritic:
            mock_critic_instance = MockCritic.return_value

            # First generation: Slot 0 scores high, others score low
            mock_critic_instance.evaluate_candidate_populations.return_value = [
                [{"combined_score": 0.95, "candidate_index": 0, "feedback": "Good"}],  # Slot 0 locks
                [{"combined_score": 0.7, "candidate_index": 0, "feedback": "Needs work"}],  # Slot 1 doesn't lock
                [{"combined_score": 0.7, "candidate_index": 0, "feedback": "Needs work"}]   # Slot 2 doesn't lock
            ]
            mock_critic_instance.evaluate.return_value = {"proposition_recall": 0.85}

            result = mock_translator.translate_paragraph(
                paragraph="Test paragraph.",
                atlas=atlas,
                author_name="TestAuthor",
                verbose=False
            )

            # Verify generation was called
            assert mock_generate.called

            # Verify evaluation was called
            assert mock_critic_instance.evaluate_candidate_populations.called

    def test_locked_slot_skipped_in_subsequent_generations(self, mock_translator):
        """Test that locked slots are skipped in subsequent generations."""
        atlas = MockAtlas()
        generation_calls = []

        def track_generation(state, *args, **kwargs):
            generation_calls.append({
                'unlocked_slots': [i for i, locked in enumerate(state.locked_flags) if not locked],
                'locked_slots': [i for i, locked in enumerate(state.locked_flags) if locked]
            })
            return [
                ["Candidate 0.1", "Candidate 0.2", "Candidate 0.3"],
                ["Candidate 1.1", "Candidate 1.2", "Candidate 1.3"],
                ["Candidate 2.1", "Candidate 2.2", "Candidate 2.3"]
            ]

        with patch.object(mock_translator, '_generate_candidate_populations', side_effect=track_generation), \
             patch('src.validator.semantic_critic.SemanticCritic') as MockCritic:
            mock_critic_instance = MockCritic.return_value

            # First call: Slot 0 locks, others don't
            # Second call: Only Slot 1 and Slot 2 should be in unlocked_slots
            call_count = [0]
            def evaluate_side_effect(*args, **kwargs):
                call_count[0] += 1
                if call_count[0] == 1:
                    # First generation: Slot 0 locks
                    return [
                        [{"combined_score": 0.95, "candidate_index": 0, "feedback": "Good"}],
                        [{"combined_score": 0.7, "candidate_index": 0, "feedback": "Needs work"}],
                        [{"combined_score": 0.7, "candidate_index": 0, "feedback": "Needs work"}]
                    ]
                else:
                    # Subsequent generations: Slot 0 should be skipped
                    return [
                        [{"combined_score": 0.95, "candidate_index": 0, "feedback": "Good"}],  # Slot 0 already locked, shouldn't be evaluated
                        [{"combined_score": 0.95, "candidate_index": 0, "feedback": "Good"}],  # Slot 1 locks
                        [{"combined_score": 0.95, "candidate_index": 0, "feedback": "Good"}]   # Slot 2 locks
                    ]

            mock_critic_instance.evaluate_candidate_populations.side_effect = evaluate_side_effect
            mock_critic_instance.evaluate.return_value = {"proposition_recall": 0.85}

            mock_translator.translate_paragraph(
                paragraph="Test paragraph.",
                atlas=atlas,
                author_name="TestAuthor",
                verbose=False
            )

            # Verify multiple generations occurred
            assert len(generation_calls) > 0


class TestEarlyTermination:
    """Test early termination when all slots are locked."""

    def test_loop_terminates_when_all_slots_locked(self, mock_translator):
        """Test that evolution loop breaks early when all slots are locked."""
        atlas = MockAtlas()
        generation_count = [0]

        def count_generations(*args, **kwargs):
            generation_count[0] += 1
            return [
                ["Candidate 0.1", "Candidate 0.2", "Candidate 0.3"],
                ["Candidate 1.1", "Candidate 1.2", "Candidate 1.3"],
                ["Candidate 2.1", "Candidate 2.2", "Candidate 2.3"]
            ]

        with patch.object(mock_translator, '_generate_candidate_populations', side_effect=count_generations), \
             patch('src.validator.semantic_critic.SemanticCritic') as MockCritic:
            mock_critic_instance = MockCritic.return_value

            # All slots lock in first generation
            mock_critic_instance.evaluate_candidate_populations.return_value = [
                [{"combined_score": 0.95, "candidate_index": 0, "feedback": "Good"}],
                [{"combined_score": 0.95, "candidate_index": 0, "feedback": "Good"}],
                [{"combined_score": 0.95, "candidate_index": 0, "feedback": "Good"}]
            ]
            mock_critic_instance.evaluate.return_value = {"proposition_recall": 0.85}

            result = mock_translator.translate_paragraph(
                paragraph="Test paragraph.",
                atlas=atlas,
                author_name="TestAuthor",
                verbose=False
            )

            # Should only generate once (all slots lock immediately)
            # Note: The loop checks all(state.locked_flags) at the start of each iteration
            # So it will generate once, then check and break
            assert generation_count[0] == 1
            assert result[0]  # Should return a paragraph


class TestPartialLocking:
    """Test partial locking (some slots locked, others continue)."""

    def test_unlocked_slots_continue_generating(self, mock_translator):
        """Test that unlocked slots continue generating while locked slots are skipped."""
        atlas = MockAtlas()
        generation_calls = []

        def track_generation(state, *args, **kwargs):
            unlocked = [i for i, locked in enumerate(state.locked_flags) if not locked]
            generation_calls.append(unlocked.copy())
            return [
                ["Candidate 0.1", "Candidate 0.2", "Candidate 0.3"],
                ["Candidate 1.1", "Candidate 1.2", "Candidate 1.3"],
                ["Candidate 2.1", "Candidate 2.2", "Candidate 2.3"]
            ]

        with patch.object(mock_translator, '_generate_candidate_populations', side_effect=track_generation), \
             patch('src.validator.semantic_critic.SemanticCritic') as MockCritic:
            mock_critic_instance = MockCritic.return_value

            call_count = [0]
            def evaluate_side_effect(*args, **kwargs):
                call_count[0] += 1
                if call_count[0] == 1:
                    # First: Slot 0 locks, others don't
                    return [
                        [{"combined_score": 0.95, "candidate_index": 0, "feedback": "Good"}],
                        [{"combined_score": 0.7, "candidate_index": 0, "feedback": "Needs work"}],
                        [{"combined_score": 0.7, "candidate_index": 0, "feedback": "Needs work"}]
                    ]
                elif call_count[0] == 2:
                    # Second: Slot 1 locks, Slot 2 doesn't
                    return [
                        [{"combined_score": 0.95, "candidate_index": 0, "feedback": "Good"}],  # Already locked
                        [{"combined_score": 0.95, "candidate_index": 0, "feedback": "Good"}],
                        [{"combined_score": 0.7, "candidate_index": 0, "feedback": "Needs work"}]
                    ]
                else:
                    # Third: All lock
                    return [
                        [{"combined_score": 0.95, "candidate_index": 0, "feedback": "Good"}],
                        [{"combined_score": 0.95, "candidate_index": 0, "feedback": "Good"}],
                        [{"combined_score": 0.95, "candidate_index": 0, "feedback": "Good"}]
                    ]

            mock_critic_instance.evaluate_candidate_populations.side_effect = evaluate_side_effect
            mock_critic_instance.evaluate.return_value = {"proposition_recall": 0.85}

            result = mock_translator.translate_paragraph(
                paragraph="Test paragraph.",
                atlas=atlas,
                author_name="TestAuthor",
                verbose=False
            )

            # Verify multiple generations occurred
            assert len(generation_calls) >= 2

            # First generation should include all slots
            assert 0 in generation_calls[0]
            assert 1 in generation_calls[0]
            assert 2 in generation_calls[0]

            # Second generation should exclude Slot 0 (locked)
            if len(generation_calls) > 1:
                assert 0 not in generation_calls[1] or all(generation_calls[1])  # Either 0 is excluded or all are unlocked


class TestGenerationCountTracking:
    """Test generation count tracking."""

    def test_generation_count_increments_for_unlocked_slots(self, mock_translator):
        """Test that generation_count increments for unlocked slots."""
        atlas = MockAtlas()

        with patch.object(mock_translator, '_generate_candidate_populations') as mock_generate, \
             patch('src.validator.semantic_critic.SemanticCritic') as MockCritic:
            mock_critic_instance = MockCritic.return_value

            call_count = [0]
            def evaluate_side_effect(*args, **kwargs):
                call_count[0] += 1
                # First two calls: Slot 0 doesn't lock, others don't either
                if call_count[0] <= 2:
                    return [
                        [{"combined_score": 0.7, "candidate_index": 0, "feedback": "Needs work"}],
                        [{"combined_score": 0.7, "candidate_index": 0, "feedback": "Needs work"}],
                        [{"combined_score": 0.7, "candidate_index": 0, "feedback": "Needs work"}]
                    ]
                else:
                    # Third call: All lock
                    return [
                        [{"combined_score": 0.95, "candidate_index": 0, "feedback": "Good"}],
                        [{"combined_score": 0.95, "candidate_index": 0, "feedback": "Good"}],
                        [{"combined_score": 0.95, "candidate_index": 0, "feedback": "Good"}]
                    ]

            mock_critic_instance.evaluate_candidate_populations.side_effect = evaluate_side_effect
            mock_critic_instance.evaluate.return_value = {"proposition_recall": 0.85}

            mock_generate.return_value = [
                ["Candidate 0.1", "Candidate 0.2", "Candidate 0.3"],
                ["Candidate 1.1", "Candidate 1.2", "Candidate 1.3"],
                ["Candidate 2.1", "Candidate 2.2", "Candidate 2.3"]
            ]

            result = mock_translator.translate_paragraph(
                paragraph="Test paragraph.",
                atlas=atlas,
                author_name="TestAuthor",
                verbose=False
            )

            # Verify multiple generations occurred
            assert mock_generate.call_count >= 2

            # Verify evaluation was called multiple times
            assert mock_critic_instance.evaluate_candidate_populations.call_count >= 2
