"""
Tests for Phase 2: Batch Logic

These tests verify:
1. Candidate generation with neighbor context (X-1, X+1)
2. Neighbor context priority (locked best sentences vs candidates)
3. Batch evaluation of all candidates
4. Empty population handling
"""

import json
import pytest
from unittest.mock import Mock, patch, MagicMock
from src.generator.translator import StyleTranslator, ParagraphState
from src.validator.semantic_critic import SemanticCritic


class TestCandidateGenerationWithNeighborContext:
    """Test candidate generation with neighbor context."""

    def test_generate_only_for_unlocked_slots(self):
        """Test that generation only produces candidates for unlocked slots."""
        translator = StyleTranslator(config_path="config.json")

        state = ParagraphState(
            original_text="Test paragraph.",
            templates=["[NP] [VP] [NP].", "[NP] [VP].", "[NP] [VP] [NP] [NP]."],
            prop_map=[["Prop 1"], ["Prop 2"], ["Prop 3"]],
            candidate_populations=[[], [], []],
            best_sentences=[None, None, None],
            locked_flags=[True, False, False],  # Slot 0 is locked
            feedback=[[], [], []],
            generation_count=[0, 0, 0]
        )

        # Mock LLM response
        mock_response = json.dumps({
            "slot_1": ["Candidate 1 for slot 1", "Candidate 2 for slot 1", "Candidate 3 for slot 1"],
            "slot_2": ["Candidate 1 for slot 2", "Candidate 2 for slot 2", "Candidate 3 for slot 2"]
        })

        with patch.object(translator.llm_provider, 'call') as mock_call:
            mock_call.return_value = mock_response

            populations = translator._generate_candidate_populations(
                state=state,
                author_name="TestAuthor",
                style_dna=None,
                population_size=3,
                verbose=False
            )

            # Slot 0 should be empty (locked)
            assert len(populations[0]) == 0
            # Slots 1 and 2 should have candidates
            assert len(populations[1]) == 3
            assert len(populations[2]) == 3

    def test_neighbor_context_uses_locked_best_sentence(self):
        """Test that neighbor context prioritizes locked best sentences."""
        translator = StyleTranslator(config_path="config.json")

        state = ParagraphState(
            original_text="Test paragraph.",
            templates=["[NP] [VP] [NP].", "[NP] [VP]."],
            prop_map=[["Prop 1"], ["Prop 2"]],
            candidate_populations=[[], []],
            best_sentences=["Locked sentence from slot 0", None],
            locked_flags=[True, False],  # Slot 0 is locked
            feedback=[[], []],
            generation_count=[0, 0]
        )

        # Mock LLM call and capture the prompt
        captured_prompt = []

        def capture_call(system_prompt, user_prompt, **kwargs):
            captured_prompt.append(user_prompt)
            return json.dumps({
                "slot_1": ["Candidate 1", "Candidate 2", "Candidate 3"]
            })

        with patch.object(translator.llm_provider, 'call') as mock_call:
            mock_call.side_effect = capture_call

            translator._generate_candidate_populations(
                state=state,
                author_name="TestAuthor",
                style_dna=None,
                population_size=3,
                verbose=False
            )

            # Verify prompt includes locked best sentence
            assert len(captured_prompt) > 0
            prompt_text = captured_prompt[0]
            assert "Locked sentence from slot 0" in prompt_text
            assert "Previous sentence:" in prompt_text

    def test_neighbor_context_uses_candidate_when_not_locked(self):
        """Test that neighbor context uses best candidate when slot is not locked."""
        translator = StyleTranslator(config_path="config.json")

        state = ParagraphState(
            original_text="Test paragraph.",
            templates=["[NP] [VP] [NP].", "[NP] [VP]."],
            prop_map=[["Prop 1"], ["Prop 2"]],
            candidate_populations=[["Candidate from slot 0"], []],
            best_sentences=[None, None],
            locked_flags=[False, False],  # Neither locked
            feedback=[[], []],
            generation_count=[0, 0]
        )

        # Mock LLM call and capture the prompt
        captured_prompt = []

        def capture_call(system_prompt, user_prompt, **kwargs):
            captured_prompt.append(user_prompt)
            return json.dumps({
                "slot_1": ["Candidate 1", "Candidate 2", "Candidate 3"]
            })

        with patch.object(translator.llm_provider, 'call') as mock_call:
            mock_call.side_effect = capture_call

            translator._generate_candidate_populations(
                state=state,
                author_name="TestAuthor",
                style_dna=None,
                population_size=3,
                verbose=False
            )

            # Verify prompt includes candidate from slot 0
            assert len(captured_prompt) > 0
            prompt_text = captured_prompt[0]
            assert "Candidate from slot 0" in prompt_text


class TestBatchEvaluation:
    """Test batch evaluation of candidate populations."""

    def test_evaluate_all_candidates(self):
        """Test that all candidates are evaluated."""
        critic = SemanticCritic(config_path="config.json")

        candidate_populations = [
            ["Candidate 1 for slot 0", "Candidate 2 for slot 0"],
            ["Candidate 1 for slot 1", "Candidate 2 for slot 1"]
        ]
        templates = ["[NP] [VP] [NP].", "[NP] [VP]."]
        prop_map = [["Prop 1"], ["Prop 2"]]

        with patch.object(critic, 'evaluate_sentence_fit') as mock_eval:
            # Mock evaluation results
            def eval_side_effect(draft, assigned_propositions, template, verbose=False):
                return {
                    "pass": True,
                    "anchor_score": 0.9,
                    "semantic_score": 0.9,
                    "feedback": "Pass"
                }

            mock_eval.side_effect = eval_side_effect

            results = critic.evaluate_candidate_populations(
                candidate_populations=candidate_populations,
                templates=templates,
                prop_map=prop_map,
                verbose=False
            )

            # Should have results for 2 slots
            assert len(results) == 2
            # Each slot should have 2 results (one per candidate)
            assert len(results[0]) == 2
            assert len(results[1]) == 2
            # Each result should have required fields
            for slot_results in results:
                for result in slot_results:
                    assert "pass" in result
                    assert "anchor_score" in result
                    assert "semantic_score" in result
                    assert "combined_score" in result
                    assert "feedback" in result
                    assert "slot_index" in result
                    assert "candidate_index" in result

    def test_evaluate_empty_populations(self):
        """Test that empty populations are handled gracefully."""
        critic = SemanticCritic(config_path="config.json")

        candidate_populations = [
            [],  # Empty (locked slot)
            ["Candidate 1", "Candidate 2"]
        ]
        templates = ["[NP] [VP] [NP].", "[NP] [VP]."]
        prop_map = [["Prop 1"], ["Prop 2"]]

        with patch.object(critic, 'evaluate_sentence_fit') as mock_eval:
            def eval_side_effect(draft, assigned_propositions, template, verbose=False):
                return {
                    "pass": True,
                    "anchor_score": 0.9,
                    "semantic_score": 0.9,
                    "feedback": "Pass"
                }

            mock_eval.side_effect = eval_side_effect

            results = critic.evaluate_candidate_populations(
                candidate_populations=candidate_populations,
                templates=templates,
                prop_map=prop_map,
                verbose=False
            )

            # Slot 0 should have empty results
            assert len(results[0]) == 0
            # Slot 1 should have 2 results
            assert len(results[1]) == 2

    def test_evaluation_result_structure(self):
        """Test that evaluation results have correct structure."""
        critic = SemanticCritic(config_path="config.json")

        candidate_populations = [["Test candidate"]]
        templates = ["[NP] [VP] [NP]."]
        prop_map = [["Prop 1"]]

        with patch.object(critic, 'evaluate_sentence_fit') as mock_eval:
            mock_eval.return_value = {
                "pass": True,
                "anchor_score": 0.85,
                "semantic_score": 0.95,
                "feedback": "Good"
            }

            results = critic.evaluate_candidate_populations(
                candidate_populations=candidate_populations,
                templates=templates,
                prop_map=prop_map,
                verbose=False
            )

            assert len(results) == 1
            assert len(results[0]) == 1

            result = results[0][0]
            assert result["pass"] is True
            assert result["anchor_score"] == 0.85
            assert result["semantic_score"] == 0.95
            assert abs(result["combined_score"] - 1.8) < 0.01  # 0.85 + 0.95 (allow floating point precision)
            assert result["slot_index"] == 0
            assert result["candidate_index"] == 0

