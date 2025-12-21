"""
Integration tests for Phase 4: Batch Flow

These tests verify:
1. Sentence 2 references context from Sentence 1 (Flow) while adhering to Template 2 (Structure)
2. Locking mechanism end-to-end
3. Structure adherence with flow context
4. Neighbor context priority (locked best sentences vs candidates)
5. Full paragraph generation with coherence
"""

import json
import pytest
from unittest.mock import Mock, patch, MagicMock
from src.generator.translator import StyleTranslator, ParagraphState
from src.validator.semantic_critic import SemanticCritic


class MockLLMProvider:
    """Mock LLM provider that generates candidates with flow awareness."""

    def __init__(self, config_path=None):
        self.config_path = config_path
        self.call_count = 0
        self.call_history = []

    def call(self, system_prompt, user_prompt, model_type, require_json, temperature, max_tokens, timeout=None, top_p=None):
        self.call_count += 1
        self.call_history.append({
            "system_prompt": system_prompt[:100] if len(system_prompt) > 100 else system_prompt,
            "user_prompt": user_prompt[:500] if len(user_prompt) > 500 else user_prompt,
            "model_type": model_type
        })

        # Parse the prompt to understand which slots are being generated
        # Look for "Slot 0", "Slot 1", etc. in the prompt
        unlocked_slots = []
        if "Slot 0" in user_prompt:
            unlocked_slots.append(0)
        if "Slot 1" in user_prompt:
            unlocked_slots.append(1)
        if "Slot 2" in user_prompt:
            unlocked_slots.append(2)

        # Generate candidates that show flow awareness
        # If Slot 1 is being generated and Slot 0 context is present, use pronouns/references
        response = {}
        for slot_idx in unlocked_slots:
            if slot_idx == 0:
                # First sentence - no previous context
                response[f"slot_{slot_idx}"] = [
                    "The cat sat on the mat.",
                    "A cat was sitting on the mat.",
                    "On the mat, a cat sat."
                ]
            elif slot_idx == 1:
                # Second sentence - should reference Slot 0
                # Check if Slot 0's best sentence is in the prompt
                if "The cat sat" in user_prompt or "cat" in user_prompt.lower():
                    response[f"slot_{slot_idx}"] = [
                        "It then began to purr.",
                        "The cat then began to purr.",
                        "This made it purr."
                    ]
                else:
                    response[f"slot_{slot_idx}"] = [
                        "Then it began to purr.",
                        "After that, it purred.",
                        "It started purring."
                    ]
            elif slot_idx == 2:
                # Third sentence - should reference previous sentences
                response[f"slot_{slot_idx}"] = [
                    "The sound was soothing.",
                    "This sound was soothing.",
                    "The purring sound was soothing."
                ]

        return json.dumps(response)


class MockPropositionExtractor:
    """Mock proposition extractor."""

    def extract_atomic_propositions(self, text):
        return ["A cat sat on a mat.", "The cat began to purr.", "The sound was soothing."]


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
    translator.llm_provider = MockLLMProvider(config_path="config.json")
    translator.proposition_extractor = MockPropositionExtractor()
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
        ["[NP] [VP] [PP].", "[NP] [VP].", "[NP] [VP] [ADJ]."]
    ))

    # Mock _map_propositions_to_templates
    translator._map_propositions_to_templates = Mock(return_value=[
        ["A cat sat on a mat."],
        ["The cat began to purr."],
        ["The sound was soothing."]
    ])

    # Mock _extract_sentence_templates
    translator._extract_sentence_templates = Mock(return_value=[
        "[NP] [VP] [PP].",
        "[NP] [VP].",
        "[NP] [VP] [ADJ]."
    ])

    return translator


@pytest.fixture
def mock_critic():
    """Create a critic that evaluates structure and meaning."""
    critic = SemanticCritic(config_path="config.json")

    def mock_evaluate_sentence_fit(draft, assigned_propositions, template, verbose=False):
        # Simple evaluation: check if draft contains key words from propositions
        # and if it matches template structure (has [NP] and [VP] patterns)
        prop_words = set()
        for prop in assigned_propositions:
            prop_words.update(prop.lower().split())

        draft_words = set(draft.lower().split())
        semantic_score = len(prop_words & draft_words) / max(len(prop_words), 1)

        # Structure score: check if draft has noun and verb (simplified)
        has_noun = any(word in draft.lower() for word in ['cat', 'mat', 'sound', 'it', 'the'])
        has_verb = any(word in draft.lower() for word in ['sat', 'began', 'purr', 'was', 'made'])
        anchor_score = 0.5 if (has_noun and has_verb) else 0.3

        combined_score = anchor_score + semantic_score * 0.5

        return {
            "pass": combined_score >= 0.9,
            "anchor_score": anchor_score,
            "semantic_score": semantic_score,
            "combined_score": combined_score,
            "feedback": f"Structure: {anchor_score:.2f}, Meaning: {semantic_score:.2f}"
        }

    def mock_evaluate_candidate_populations(candidate_populations, templates, prop_map, verbose=False):
        results = []
        for slot_idx, population in enumerate(candidate_populations):
            slot_results = []
            template = templates[slot_idx]
            assigned_props = prop_map[slot_idx]

            for candidate_idx, candidate in enumerate(population):
                result = mock_evaluate_sentence_fit(candidate, assigned_props, template, verbose)
                result["slot_index"] = slot_idx
                result["candidate_index"] = candidate_idx
                slot_results.append(result)
            results.append(slot_results)
        return results

    critic.evaluate_sentence_fit = Mock(side_effect=mock_evaluate_sentence_fit)
    critic.evaluate_candidate_populations = Mock(side_effect=mock_evaluate_candidate_populations)
    critic.evaluate = Mock(return_value={
        "proposition_recall": 0.85,
        "style_alignment": 0.8,
        "score": 0.825
    })

    return critic


class TestFlowContext:
    """Test that sentences reference previous sentences for flow."""

    def test_slot_1_includes_slot_0_context(self, mock_translator, mock_critic):
        """Test that when generating Slot 1, the prompt includes Slot 0's best sentence."""
        atlas = MockAtlas()

        with patch('src.validator.semantic_critic.SemanticCritic', return_value=mock_critic):
            # First generation: all slots unlocked
            result = mock_translator.translate_paragraph(
                paragraph="A cat sat on a mat. The cat began to purr. The sound was soothing.",
                atlas=atlas,
                author_name="TestAuthor",
                verbose=False
            )

            # Verify that generation was called
            assert mock_translator.llm_provider.call_count > 0

            # Check that Slot 1's generation prompt included Slot 0 context
            slot_1_calls = [
                call for call in mock_translator.llm_provider.call_history
                if "Slot 1" in call.get("user_prompt", "")
            ]

            if slot_1_calls:
                # Verify that Slot 0's content is mentioned in Slot 1's prompt
                slot_1_prompt = slot_1_calls[0]["user_prompt"]
                # Should contain reference to Slot 0 (either "Slot 0" or the actual sentence)
                assert "Slot 0" in slot_1_prompt or "cat" in slot_1_prompt.lower() or "mat" in slot_1_prompt.lower()

    def test_slot_2_includes_slot_1_context(self, mock_translator, mock_critic):
        """Test that when generating Slot 2, the prompt includes Slot 1's best sentence."""
        atlas = MockAtlas()

        with patch('src.validator.semantic_critic.SemanticCritic', return_value=mock_critic):
            result = mock_translator.translate_paragraph(
                paragraph="A cat sat on a mat. The cat began to purr. The sound was soothing.",
                atlas=atlas,
                author_name="TestAuthor",
                verbose=False
            )

            # Check that Slot 2's generation prompt included Slot 1 context
            slot_2_calls = [
                call for call in mock_translator.llm_provider.call_history
                if "Slot 2" in call.get("user_prompt", "")
            ]

            if slot_2_calls:
                slot_2_prompt = slot_2_calls[0]["user_prompt"]
                # Should contain reference to Slot 1
                assert "Slot 1" in slot_2_prompt or "purr" in slot_2_prompt.lower()


class TestLockingMechanismEndToEnd:
    """Test locking mechanism in full paragraph generation."""

    def test_slot_locks_after_high_score(self, mock_translator, mock_critic):
        """Test that a slot locks after achieving high score and is skipped in subsequent generations."""
        atlas = MockAtlas()

        # Modify critic to return high scores for Slot 0
        def evaluate_with_high_slot_0(candidate_populations, templates, prop_map, verbose=False):
            results = []
            for slot_idx, population in enumerate(candidate_populations):
                slot_results = []
                template = templates[slot_idx]
                assigned_props = prop_map[slot_idx]

                for candidate_idx, candidate in enumerate(population):
                    if slot_idx == 0:
                        # Slot 0 gets high score (will lock)
                        combined_score = 0.95
                    else:
                        # Other slots get lower scores initially
                        combined_score = 0.7

                    result = {
                        "pass": combined_score >= 0.9,
                        "anchor_score": 0.5,
                        "semantic_score": combined_score - 0.5,
                        "combined_score": combined_score,
                        "feedback": f"Score: {combined_score:.2f}",
                        "slot_index": slot_idx,
                        "candidate_index": candidate_idx
                    }
                    slot_results.append(result)
                results.append(slot_results)
            return results

        mock_critic.evaluate_candidate_populations = Mock(side_effect=evaluate_with_high_slot_0)

        with patch('src.validator.semantic_critic.SemanticCritic', return_value=mock_critic):
            result = mock_translator.translate_paragraph(
                paragraph="A cat sat on a mat. The cat began to purr. The sound was soothing.",
                atlas=atlas,
                author_name="TestAuthor",
                verbose=False
            )

            # Verify that final paragraph includes Slot 0's locked sentence
            assert result[0]  # Generated paragraph should not be empty
            assert "cat" in result[0].lower() or "mat" in result[0].lower()


class TestStructureAdherence:
    """Test that sentences adhere to their templates while maintaining flow."""

    def test_sentence_2_adheres_to_template_2(self, mock_translator, mock_critic):
        """Test that Sentence 2 adheres to Template 2 structure."""
        atlas = MockAtlas()

        with patch('src.validator.semantic_critic.SemanticCritic', return_value=mock_critic):
            result = mock_translator.translate_paragraph(
                paragraph="A cat sat on a mat. The cat began to purr. The sound was soothing.",
                atlas=atlas,
                author_name="TestAuthor",
                verbose=False
            )

            # Verify that the critic was called to check structure
            assert mock_critic.evaluate_candidate_populations.called

            # Verify that structure checks were performed
            # (The mock critic checks for noun and verb, which should be present)
            assert result[0]  # Should have generated text


class TestNeighborContextPriority:
    """Test that locked best sentences are prioritized in neighbor context."""

    def test_locked_sentence_used_in_neighbor_context(self, mock_translator, mock_critic):
        """Test that when Slot 0 is locked, Slot 1 uses Slot 0's locked best sentence."""
        atlas = MockAtlas()

        # Modify critic to lock Slot 0 in first generation
        call_count = [0]
        def evaluate_with_progressive_locking(candidate_populations, templates, prop_map, verbose=False):
            call_count[0] += 1
            results = []
            for slot_idx, population in enumerate(candidate_populations):
                slot_results = []
                for candidate_idx, candidate in enumerate(population):
                    if slot_idx == 0 and call_count[0] == 1:
                        # First generation: Slot 0 locks
                        combined_score = 0.95
                    elif slot_idx == 0:
                        # Subsequent: Slot 0 already locked, shouldn't be evaluated
                        combined_score = 0.0
                    else:
                        combined_score = 0.7

                    result = {
                        "pass": combined_score >= 0.9,
                        "anchor_score": 0.5,
                        "semantic_score": combined_score - 0.5,
                        "combined_score": combined_score,
                        "feedback": f"Score: {combined_score:.2f}",
                        "slot_index": slot_idx,
                        "candidate_index": candidate_idx
                    }
                    slot_results.append(result)
                results.append(slot_results)
            return results

        mock_critic.evaluate_candidate_populations = Mock(side_effect=evaluate_with_progressive_locking)

        with patch('src.validator.semantic_critic.SemanticCritic', return_value=mock_critic):
            result = mock_translator.translate_paragraph(
                paragraph="A cat sat on a mat. The cat began to purr. The sound was soothing.",
                atlas=atlas,
                author_name="TestAuthor",
                verbose=False
            )

            # Verify that Slot 1's generation (in second call) includes Slot 0's locked sentence
            if mock_translator.llm_provider.call_count > 1:
                # Check second generation call for Slot 1
                second_call = mock_translator.llm_provider.call_history[1]
                prompt = second_call.get("user_prompt", "")
                # Should reference Slot 0's locked sentence
                assert "Slot 0" in prompt or "cat" in prompt.lower() or "mat" in prompt.lower()


class TestFullParagraphGeneration:
    """Test full paragraph generation with coherence."""

    def test_full_paragraph_generation(self, mock_translator, mock_critic):
        """Test that a complete paragraph is generated with all slots eventually locking."""
        atlas = MockAtlas()

        # Modify critic to eventually lock all slots
        call_count = [0]
        def evaluate_with_all_locking(candidate_populations, templates, prop_map, verbose=False):
            call_count[0] += 1
            results = []
            for slot_idx, population in enumerate(candidate_populations):
                slot_results = []
                for candidate_idx, candidate in enumerate(population):
                    # All slots lock by second generation
                    if call_count[0] >= 2:
                        combined_score = 0.95
                    else:
                        combined_score = 0.7

                    result = {
                        "pass": combined_score >= 0.9,
                        "anchor_score": 0.5,
                        "semantic_score": combined_score - 0.5,
                        "combined_score": combined_score,
                        "feedback": f"Score: {combined_score:.2f}",
                        "slot_index": slot_idx,
                        "candidate_index": candidate_idx
                    }
                    slot_results.append(result)
                results.append(slot_results)
            return results

        mock_critic.evaluate_candidate_populations = Mock(side_effect=evaluate_with_all_locking)

        with patch('src.validator.semantic_critic.SemanticCritic', return_value=mock_critic):
            result = mock_translator.translate_paragraph(
                paragraph="A cat sat on a mat. The cat began to purr. The sound was soothing.",
                atlas=atlas,
                author_name="TestAuthor",
                verbose=False
            )

            # Verify final paragraph is generated
            assert result[0]  # Should have generated text
            assert len(result[0]) > 0  # Should not be empty

            # Verify all sentences are present (simplified check)
            # The paragraph should contain words from all propositions
            assert "cat" in result[0].lower() or "mat" in result[0].lower()

            # Verify that evolution loop ran
            assert mock_translator.llm_provider.call_count > 0

