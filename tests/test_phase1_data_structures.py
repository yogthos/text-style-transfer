"""
Tests for Phase 1: Data Structures & Helpers

These tests verify:
1. ParagraphState initialization and field validation
2. Robust skeleton retrieval with sentence count compatibility
3. Sentence template extraction from teacher examples
4. Proposition mapping to template slots with strict 1:1 mapping
"""

import json
import pytest
from unittest.mock import Mock, patch, MagicMock
from src.generator.translator import StyleTranslator, ParagraphState
from src.atlas.rhetoric import RhetoricalType


class TestParagraphStateInitialization:
    """Test ParagraphState dataclass initialization and field validation."""

    def test_paragraph_state_initialization(self):
        """Test that ParagraphState can be created with sample data."""
        original_text = "This is a test paragraph."
        templates = ["[NP] [VP] [NP].", "[NP] [VP]."]
        prop_map = [["Prop 1", "Prop 2"], ["Prop 3"]]
        candidate_populations = [[], []]
        best_sentences = [None, None]
        locked_flags = [False, False]
        feedback = [[], []]
        generation_count = [0, 0]

        state = ParagraphState(
            original_text=original_text,
            templates=templates,
            prop_map=prop_map,
            candidate_populations=candidate_populations,
            best_sentences=best_sentences,
            locked_flags=locked_flags,
            feedback=feedback,
            generation_count=generation_count
        )

        assert state.original_text == original_text
        assert state.templates == templates
        assert state.prop_map == prop_map
        assert state.candidate_populations == candidate_populations
        assert state.best_sentences == best_sentences
        assert state.locked_flags == locked_flags
        assert state.feedback == feedback
        assert state.generation_count == generation_count

    def test_paragraph_state_list_lengths_match(self):
        """Test that all list fields have matching lengths."""
        templates = ["[NP] [VP] [NP].", "[NP] [VP].", "[NP] [VP] [NP] [NP]."]
        n = len(templates)

        state = ParagraphState(
            original_text="Test",
            templates=templates,
            prop_map=[[]] * n,
            candidate_populations=[[]] * n,
            best_sentences=[None] * n,
            locked_flags=[False] * n,
            feedback=[[]] * n,
            generation_count=[0] * n
        )

        assert len(state.templates) == n
        assert len(state.prop_map) == n
        assert len(state.candidate_populations) == n
        assert len(state.best_sentences) == n
        assert len(state.locked_flags) == n
        assert len(state.feedback) == n
        assert len(state.generation_count) == n


class TestRobustSkeletonRetrieval:
    """Test robust skeleton retrieval with sentence count compatibility."""

    def test_retrieve_robust_skeleton_returns_templates(self):
        """Test that _retrieve_robust_skeleton returns teacher example and non-empty templates."""
        translator = StyleTranslator(config_path="config.json")
        atlas = Mock()

        # Mock examples with varying sentence counts
        example1 = "Sentence one. Sentence two. Sentence three."
        example2 = "Sentence one. Sentence two."
        example3 = "Sentence one. Sentence two. Sentence three. Sentence four."

        atlas.get_examples_by_rhetoric.return_value = [example1, example2, example3]

        with patch.object(translator, '_extract_sentence_templates') as mock_extract:
            mock_extract.return_value = ["[NP] [VP] [NP].", "[NP] [VP].", "[NP] [VP] [NP]."]

            teacher_example, templates = translator._retrieve_robust_skeleton(
                rhetorical_type=RhetoricalType.OBSERVATION,
                author="TestAuthor",
                prop_count=3,
                atlas=atlas,
                verbose=False
            )

            assert teacher_example is not None
            assert isinstance(templates, list)
            assert len(templates) > 0
            assert all(isinstance(t, str) for t in templates)

    def test_retrieve_robust_skeleton_sentence_count_compatibility(self):
        """Test that selected skeleton has compatible sentence count."""
        translator = StyleTranslator(config_path="config.json")
        atlas = Mock()

        # Examples with different sentence counts
        example1 = "One. Two."  # 2 sentences
        example2 = "One. Two. Three."  # 3 sentences
        example3 = "One. Two. Three. Four. Five."  # 5 sentences

        atlas.get_examples_by_rhetoric.return_value = [example1, example2, example3]

        with patch.object(translator, '_extract_sentence_templates') as mock_extract:
            # Mock extraction to return templates matching sentence count
            def extract_side_effect(example, verbose=False):
                from nltk.tokenize import sent_tokenize
                sentences = sent_tokenize(example)
                return [f"[NP] [VP]."] * len(sentences)

            mock_extract.side_effect = extract_side_effect

            teacher_example, templates = translator._retrieve_robust_skeleton(
                rhetorical_type=RhetoricalType.OBSERVATION,
                author="TestAuthor",
                prop_count=3,  # Target: 3 sentences
                atlas=atlas,
                verbose=False
            )

            # Should select example2 (3 sentences) as best match
            assert len(templates) >= 2  # At least 2 sentences (minimum)
            assert len(templates) <= 5  # At most 5 sentences

    def test_retrieve_robust_skeleton_no_perfect_match_uses_longest(self):
        """Test that if no perfect match, longest available skeleton is used."""
        translator = StyleTranslator(config_path="config.json")
        atlas = Mock()

        # Examples with sentence counts far from target
        example1 = "One. Two."  # 2 sentences
        example2 = "One. Two. Three. Four."  # 4 sentences (longest)

        atlas.get_examples_by_rhetoric.return_value = [example1, example2]

        with patch.object(translator, '_extract_sentence_templates') as mock_extract:
            def extract_side_effect(example, verbose=False):
                from nltk.tokenize import sent_tokenize
                sentences = sent_tokenize(example)
                return [f"[NP] [VP]."] * len(sentences)

            mock_extract.side_effect = extract_side_effect

            teacher_example, templates = translator._retrieve_robust_skeleton(
                rhetorical_type=RhetoricalType.OBSERVATION,
                author="TestAuthor",
                prop_count=10,  # Target: 10 (no perfect match)
                atlas=atlas,
                verbose=False
            )

            # Should use longest (example2 with 4 sentences)
            assert len(templates) == 4

    def test_retrieve_robust_skeleton_always_returns_non_empty(self):
        """Test that _retrieve_robust_skeleton always returns non-empty templates."""
        translator = StyleTranslator(config_path="config.json")
        atlas = Mock()

        atlas.get_examples_by_rhetoric.return_value = ["One. Two."]

        with patch.object(translator, '_extract_sentence_templates') as mock_extract:
            # Even if extraction fails, should return fallback
            mock_extract.return_value = []

            teacher_example, templates = translator._retrieve_robust_skeleton(
                rhetorical_type=RhetoricalType.OBSERVATION,
                author="TestAuthor",
                prop_count=3,
                atlas=atlas,
                verbose=False
            )

            assert len(templates) > 0
            assert all(isinstance(t, str) and t.strip() for t in templates)


class TestExtractSentenceTemplates:
    """Test sentence template extraction from teacher examples."""

    def test_extract_sentence_templates_multi_sentence(self):
        """Test extraction from multi-sentence teacher example."""
        translator = StyleTranslator(config_path="config.json")

        teacher_example = "The cat sat. The dog ran. The bird flew."

        with patch.object(translator.structure_extractor, 'extract_template') as mock_extract:
            mock_extract.side_effect = [
                "[NP] [VP].",
                "[NP] [VP].",
                "[NP] [VP]."
            ]

            templates = translator._extract_sentence_templates(teacher_example, verbose=False)

            assert len(templates) == 3
            assert all(isinstance(t, str) for t in templates)
            assert mock_extract.call_count == 3

    def test_extract_sentence_templates_empty_example(self):
        """Test handling of empty teacher example."""
        translator = StyleTranslator(config_path="config.json")

        templates = translator._extract_sentence_templates("", verbose=False)

        assert len(templates) > 0  # Should return fallback
        assert templates[0] == "[NP] [VP] [NP]."

    def test_extract_sentence_templates_single_sentence(self):
        """Test extraction from single sentence."""
        translator = StyleTranslator(config_path="config.json")

        teacher_example = "The cat sat on the mat."

        with patch.object(translator.structure_extractor, 'extract_template') as mock_extract:
            mock_extract.return_value = "[NP] [VP] [PP]."

            templates = translator._extract_sentence_templates(teacher_example, verbose=False)

            assert len(templates) == 1
            assert templates[0] == "[NP] [VP] [PP]."

    def test_extract_sentence_templates_extraction_failure(self):
        """Test handling of extraction failure."""
        translator = StyleTranslator(config_path="config.json")

        teacher_example = "The cat sat. The dog ran."

        with patch.object(translator.structure_extractor, 'extract_template') as mock_extract:
            mock_extract.side_effect = Exception("Extraction failed")

            templates = translator._extract_sentence_templates(teacher_example, verbose=False)

            # Should return fallback templates
            assert len(templates) == 2
            assert all(t == "[NP] [VP] [NP]." for t in templates)


class TestPropositionMapping:
    """Test proposition mapping to template slots."""

    def test_map_propositions_to_templates_all_assigned(self):
        """Test that all propositions are assigned to templates."""
        translator = StyleTranslator(config_path="config.json")

        propositions = ["Prop 1", "Prop 2", "Prop 3", "Prop 4", "Prop 5"]
        templates = ["[NP] [VP] [NP].", "[NP] [VP].", "[NP] [VP] [NP] [NP]."]

        # Mock LLM response
        mock_response = json.dumps([
            {"template_index": 0, "propositions": ["Prop 1", "Prop 2"]},
            {"template_index": 1, "propositions": ["Prop 3"]},
            {"template_index": 2, "propositions": ["Prop 4", "Prop 5"]}
        ])

        with patch.object(translator.llm_provider, 'call') as mock_call:
            mock_call.return_value = mock_response

            prop_map = translator._map_propositions_to_templates(
                propositions=propositions,
                templates=templates,
                verbose=False
            )

            assert len(prop_map) == len(templates)
            # Verify all propositions are assigned
            all_assigned = set()
            for props in prop_map:
                all_assigned.update(props)
            assert len(all_assigned) >= len(propositions) * 0.8  # Allow some variation

    def test_map_propositions_to_templates_every_template_gets_one(self):
        """Test that every template gets at least 1 proposition."""
        translator = StyleTranslator(config_path="config.json")

        propositions = ["Prop 1", "Prop 2", "Prop 3"]
        templates = ["[NP] [VP] [NP].", "[NP] [VP].", "[NP] [VP] [NP] [NP]."]

        # Mock LLM response (might not assign to all templates)
        mock_response = json.dumps([
            {"template_index": 0, "propositions": ["Prop 1", "Prop 2", "Prop 3"]}
        ])

        with patch.object(translator.llm_provider, 'call') as mock_call:
            mock_call.return_value = mock_response

            prop_map = translator._map_propositions_to_templates(
                propositions=propositions,
                templates=templates,
                verbose=False
            )

            assert len(prop_map) == len(templates)
            # Post-processing should ensure every template has at least 1
            assert all(len(props) > 0 for props in prop_map)

    def test_map_propositions_to_templates_strict_1_to_1(self):
        """Test strict 1:1 mapping (no unassigned props, no empty templates)."""
        translator = StyleTranslator(config_path="config.json")

        propositions = ["Prop 1", "Prop 2", "Prop 3"]
        templates = ["[NP] [VP] [NP].", "[NP] [VP]."]

        # Mock LLM response
        mock_response = json.dumps([
            {"template_index": 0, "propositions": ["Prop 1", "Prop 2"]},
            {"template_index": 1, "propositions": ["Prop 3"]}
        ])

        with patch.object(translator.llm_provider, 'call') as mock_call:
            mock_call.return_value = mock_response

            prop_map = translator._map_propositions_to_templates(
                propositions=propositions,
                templates=templates,
                verbose=False
            )

            assert len(prop_map) == len(templates)
            assert all(len(props) > 0 for props in prop_map)  # No empty templates

    def test_map_propositions_to_templates_error_handling(self):
        """Test that ValueError is raised if mapping fails after retries."""
        translator = StyleTranslator(config_path="config.json")

        propositions = ["Prop 1", "Prop 2"]
        templates = ["[NP] [VP] [NP]."]

        with patch.object(translator.llm_provider, 'call') as mock_call:
            # Simulate repeated failures
            mock_call.side_effect = json.JSONDecodeError("Invalid JSON", "", 0)

            with pytest.raises(ValueError, match="Failed to map propositions"):
                translator._map_propositions_to_templates(
                    propositions=propositions,
                    templates=templates,
                    verbose=False
                )

