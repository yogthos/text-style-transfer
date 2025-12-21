"""
Tests for forced paragraph fusion architecture (no fallbacks).

These tests verify that:
1. The paragraph fusion path is always used (no sentence-by-sentence fallback)
2. Errors are raised, not caught and hidden
3. Skeleton retrieval always returns non-empty
4. Planning always succeeds and assigns all propositions
5. Critic checks ideas, not specific words
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from src.generator.translator import StyleTranslator
from src.validator.semantic_critic import SemanticCritic


class TestNoFallbackOnErrors:
    """Test that errors in paragraph fusion raise exceptions, not return original paragraph."""

    def test_no_propositions_raises_error(self):
        """Test that missing propositions raises ValueError instead of returning original."""
        translator = StyleTranslator(config_path="config.json")
        atlas = Mock()

        with patch.object(translator, 'proposition_extractor') as mock_extractor:
            mock_extractor.extract_atomic_propositions.return_value = []

            with pytest.raises(ValueError, match="No propositions extracted"):
                translator.translate_paragraph(
                    paragraph="Test paragraph",
                    atlas=atlas,
                    author_name="Mao",
                    verbose=False
                )

    def test_no_skeleton_raises_error(self):
        """Test that skeleton retrieval failure raises error (should never happen due to guarantee)."""
        translator = StyleTranslator(config_path="config.json")
        atlas = Mock()

        with patch.object(translator, 'proposition_extractor') as mock_extractor:
            mock_extractor.extract_atomic_propositions.return_value = ["Prop 1", "Prop 2"]

            with patch.object(translator, '_retrieve_robust_skeleton') as mock_retrieve:
                # This should never happen, but if it does, we want an error
                mock_retrieve.return_value = []

                with pytest.raises(ValueError, match="Skeleton retrieval failed"):
                    translator.translate_paragraph(
                        paragraph="Test paragraph",
                        atlas=atlas,
                        author_name="Mao",
                        verbose=False
                    )

    def test_planning_failure_raises_error(self):
        """Test that planning failure raises ValueError instead of returning original."""
        translator = StyleTranslator(config_path="config.json")
        atlas = Mock()

        with patch.object(translator, 'proposition_extractor') as mock_extractor:
            mock_extractor.extract_atomic_propositions.return_value = ["Prop 1", "Prop 2"]

            with patch.object(translator, '_retrieve_robust_skeleton') as mock_retrieve:
                mock_retrieve.return_value = ["[NP] [VP] [NP]."]

                with patch.object(translator, '_map_propositions_to_templates') as mock_plan:
                    mock_plan.return_value = []

                    with pytest.raises(ValueError, match="Failed to map propositions"):
                        translator.translate_paragraph(
                            paragraph="Test paragraph",
                            atlas=atlas,
                            author_name="Mao",
                            verbose=False
                        )

    def test_sentence_generation_failure_raises_error(self):
        """Test that sentence generation failure raises ValueError."""
        translator = StyleTranslator(config_path="config.json")
        atlas = Mock()

        with patch.object(translator, 'proposition_extractor') as mock_extractor:
            mock_extractor.extract_atomic_propositions.return_value = ["Prop 1"]

            with patch.object(translator, '_retrieve_robust_skeleton') as mock_retrieve:
                mock_retrieve.return_value = ["[NP] [VP] [NP]."]

                with patch.object(translator, '_map_propositions_to_templates') as mock_plan:
                    mock_plan.return_value = [(0, ["Prop 1"], False)]

                    with patch.object(translator, '_write_and_refine_sentence') as mock_write:
                        mock_write.return_value = ("", False)  # Failed generation

                        with patch.object(translator, '_generate_literal_fallback') as mock_fallback:
                            mock_fallback.return_value = ""  # Fallback also fails

                            with pytest.raises(ValueError, match="Failed to generate sentences"):
                                translator.translate_paragraph(
                                    paragraph="Test paragraph",
                                    atlas=atlas,
                                    author_name="Mao",
                                    verbose=False
                                )


class TestSkeletonAlwaysReturned:
    """Test that _retrieve_robust_skeleton always returns non-empty list."""

    def test_skeleton_always_non_empty(self):
        """Test that skeleton retrieval always returns at least one template."""
        translator = StyleTranslator(config_path="config.json")
        atlas = Mock()

        # Test with various failure scenarios
        with patch.object(atlas, 'get_examples_by_rhetoric') as mock_get:
            # Scenario 1: Atlas returns empty
            mock_get.return_value = []

            result = translator._retrieve_robust_skeleton(
                rhetorical_type=Mock(),
                author_name="Mao",
                proposition_count=5,
                atlas=atlas,
                verbose=False
            )

            assert result is not None
            assert len(result) > 0, "Skeleton retrieval must always return non-empty list"

            # Scenario 2: Atlas raises exception
            mock_get.side_effect = Exception("Atlas error")

            result = translator._retrieve_robust_skeleton(
                rhetorical_type=Mock(),
                author_name="Mao",
                proposition_count=5,
                atlas=atlas,
                verbose=False
            )

            assert result is not None
            assert len(result) > 0, "Skeleton retrieval must always return non-empty list even on error"


class TestPlanningAlwaysSucceeds:
    """Test that planner assigns all propositions and every template gets at least 1."""

    def test_all_propositions_assigned(self):
        """Test that all propositions are assigned to templates."""
        translator = StyleTranslator(config_path="config.json")

        propositions = ["Prop 1", "Prop 2", "Prop 3", "Prop 4"]
        templates = ["[NP] [VP] [NP].", "[NP] is [ADJ]."]

        with patch.object(translator, 'llm_provider') as mock_llm:
            # Mock LLM response
            mock_response = [
                {"slot_index": 0, "propositions": ["Prop 1", "Prop 2"], "is_high_risk": False},
                {"slot_index": 1, "propositions": ["Prop 3", "Prop 4"], "is_high_risk": False}
            ]
            mock_llm.call.return_value = str(mock_response)

            result = translator._map_propositions_to_templates(
                propositions=propositions,
                templates=templates,
                verbose=False
            )

            # Verify all propositions are assigned
            assigned_props = set()
            for _, props, _ in result:
                assigned_props.update(props)

            assert assigned_props == set(propositions), "All propositions must be assigned"

    def test_every_template_gets_at_least_one_proposition(self):
        """Test that every template slot gets at least 1 proposition."""
        translator = StyleTranslator(config_path="config.json")

        propositions = ["Prop 1", "Prop 2"]
        templates = ["[NP] [VP] [NP].", "[NP] is [ADJ]."]

        with patch.object(translator, 'llm_provider') as mock_llm:
            # Mock LLM response that might leave a slot empty
            mock_response = [
                {"slot_index": 0, "propositions": ["Prop 1", "Prop 2"], "is_high_risk": False}
            ]
            mock_llm.call.return_value = str(mock_response)

            result = translator._map_propositions_to_templates(
                propositions=propositions,
                templates=templates,
                verbose=False
            )

            # Verify every template slot has at least 1 proposition
            for slot_idx, props, _ in result:
                assert len(props) >= 1, f"Template slot {slot_idx} must have at least 1 proposition"


class TestCriticChecksIdeasNotWords:
    """Test that critic accepts synonyms and rephrasing (concept matching, not word matching)."""

    def test_critic_accepts_synonyms(self):
        """Test that critic accepts 'nation-state' when proposition says 'country'."""
        critic = SemanticCritic(config_path="config.json")

        draft = "The nation-state collapsed."
        propositions = ["The country collapsed."]
        template = "[NP] [VP]."

        with patch.object(critic, 'llm_provider') as mock_llm:
            # Mock LLM to return high semantic score (concept match)
            mock_response = {
                "anchor_score": 1.0,
                "semantic_score": 0.95,  # High score because concept is present
                "anchor_feedback": "All anchors present.",
                "semantic_feedback": "Core meaning is present (nation-state = country).",
                "overall_feedback": "Good match.",
                "pass": True
            }
            mock_llm.call.return_value = str(mock_response)

            result = critic.evaluate_sentence_fit(
                draft=draft,
                assigned_propositions=propositions,
                template=template,
                verbose=False
            )

            # Critic should accept the synonym
            assert result["semantic_score"] >= 0.9, "Critic should accept synonyms (concept match)"
            assert result["pass"] is True, "Critic should pass when concept is present, even with different words"

    def test_critic_prompt_emphasizes_ideas_not_words(self):
        """Test that the critic prompt explicitly says to check ideas, not words."""
        critic = SemanticCritic(config_path="config.json")

        # Extract the prompt by mocking the LLM call
        captured_prompt = None

        def capture_prompt(*args, **kwargs):
            nonlocal captured_prompt
            captured_prompt = kwargs.get('user_prompt', '')
            return '{"anchor_score": 1.0, "semantic_score": 1.0, "anchor_feedback": "", "semantic_feedback": "", "overall_feedback": "", "pass": true}'

        with patch.object(critic, 'llm_provider') as mock_llm:
            mock_llm.call.side_effect = capture_prompt

            critic.evaluate_sentence_fit(
                draft="Test draft",
                assigned_propositions=["Test proposition"],
                template="[NP] [VP].",
                verbose=False
            )

            # Verify prompt contains key instructions
            assert "Do NOT check for specific words" in captured_prompt, "Prompt must explicitly avoid word-level checks"
            assert "Check if the *idea* is present" in captured_prompt, "Prompt must emphasize idea/concept checking"
            assert "core meaning" in captured_prompt.lower(), "Prompt must mention core meaning"


class TestEndToEndParagraphFusion:
    """Test complete pipeline from paragraph input to styled output."""

    def test_end_to_end_no_fallback(self):
        """Test that the full pipeline uses paragraph fusion, not sentence-by-sentence."""
        translator = StyleTranslator(config_path="config.json")
        atlas = Mock()

        # Mock all components
        with patch.object(translator, 'proposition_extractor') as mock_extractor:
            mock_extractor.extract_atomic_propositions.return_value = [
                "I spent my childhood scavenging for food.",
                "The Soviet Union was a ghost of the past."
            ]

            with patch.object(translator, '_retrieve_robust_skeleton') as mock_retrieve:
                mock_retrieve.return_value = [
                    "In [NP], there was [NP].",
                    "The [NP] [VP] [NP]."
                ]

                with patch.object(translator, '_map_propositions_to_templates') as mock_plan:
                    mock_plan.return_value = [
                        (0, ["I spent my childhood scavenging for food."], False),
                        (1, ["The Soviet Union was a ghost of the past."], False)
                    ]

                    with patch.object(translator, '_write_and_refine_sentence') as mock_write:
                        mock_write.return_value = ("Generated sentence 1.", True)

                        with patch.object(translator, 'rhetorical_classifier') as mock_classifier:
                            mock_classifier.classify_mode.return_value = "NARRATIVE"

                            with patch.object(SemanticCritic, '__init__', return_value=None):
                                with patch.object(SemanticCritic, 'evaluate') as mock_evaluate:
                                    mock_evaluate.return_value = {"proposition_recall": 0.95}

                                    result, _, _, recall = translator.translate_paragraph(
                                        paragraph="I spent my childhood scavenging for food. The Soviet Union was a ghost of the past.",
                                        atlas=atlas,
                                        author_name="Mao",
                                        verbose=False
                                    )

                                    # Verify paragraph fusion was used (not sentence-by-sentence)
                                    assert mock_retrieve.called, "Skeleton retrieval must be called (paragraph fusion path)"
                                    assert mock_plan.called, "Planning must be called (paragraph fusion path)"
                                    assert isinstance(result, str), "Result must be a string"
                                    assert len(result) > 0, "Result must not be empty"
                                    assert recall > 0, "Recall must be calculated"

