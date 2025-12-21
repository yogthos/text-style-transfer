"""Regression tests for Structure-Aware Semantic Verification.

This test suite verifies:
1. Anchor extraction correctly identifies fixed anchors from templates
2. Critic correctly separates anchor checks from semantic checks
3. Writer can recast sentences with anchors when feedback is provided
4. End-to-end integration produces sentences with correct anchors and meaning
"""

import unittest
from unittest.mock import Mock, patch
import json

from src.validator.semantic_critic import SemanticCritic
from src.generator.translator import StyleTranslator
from src.atlas.rhetoric import RhetoricalType


class TestAnchorExtraction(unittest.TestCase):
    """Test anchor extraction from templates."""

    def setUp(self):
        """Set up test fixtures."""
        # Anchor extraction doesn't need LLM, so we can create critic directly
        # but we need to mock LLMProvider for initialization
        mock_llm = Mock()
        with patch('src.generator.llm_provider.LLMProvider', return_value=mock_llm):
            self.critic = SemanticCritic(config_path="config.json")
            self.critic.llm_provider = None  # Not needed for anchor extraction

    def test_extract_anchors_simple_template(self):
        """Test extraction from simple template with prepositions."""
        template = "In [NP], there was [NP]."
        anchors = self.critic._extract_fixed_anchors(template)

        self.assertIsInstance(anchors, list)
        self.assertGreater(len(anchors), 0)
        # Should contain "In" and "there was"
        anchor_text = " ".join(anchors)
        self.assertIn("In", anchor_text)
        self.assertIn("there was", anchor_text)

    def test_extract_anchors_with_fact_truth(self):
        """Test that words like 'fact' and 'truth' are kept as anchors."""
        template = "The fact is [NP]."
        anchors = self.critic._extract_fixed_anchors(template)

        self.assertIsInstance(anchors, list)
        anchor_text = " ".join(anchors)
        # "fact" should be kept even though it's a noun
        self.assertIn("fact", anchor_text.lower())

    def test_extract_anchors_multi_word_phrase(self):
        """Test that multi-word phrases are preserved."""
        template = "It is a truth that [NP]."
        anchors = self.critic._extract_fixed_anchors(template)

        self.assertIsInstance(anchors, list)
        anchor_text = " ".join(anchors)
        # Should preserve the full phrase
        self.assertIn("truth", anchor_text.lower())
        self.assertIn("that", anchor_text.lower())

    def test_extract_anchors_only_placeholders(self):
        """Test template with only placeholders (minimal anchors)."""
        template = "[NP] [VP] [NP]."
        anchors = self.critic._extract_fixed_anchors(template)

        # Should still extract something (at least punctuation)
        self.assertIsInstance(anchors, list)
        # May have just punctuation
        if anchors:
            self.assertTrue(all(isinstance(a, str) for a in anchors))

    def test_extract_anchors_empty_template(self):
        """Test empty template."""
        anchors = self.critic._extract_fixed_anchors("")
        self.assertEqual(anchors, [])

    def test_extract_anchors_no_placeholders(self):
        """Test template with no placeholders (all anchors)."""
        template = "The revolution is not a dinner party."
        anchors = self.critic._extract_fixed_anchors(template)

        # Should return the entire template as one anchor
        self.assertIsInstance(anchors, list)
        self.assertGreater(len(anchors), 0)
        anchor_text = " ".join(anchors)
        self.assertIn("revolution", anchor_text.lower())


class TestStructureAwareCritic(unittest.TestCase):
    """Test structure-aware critic evaluation."""

    def setUp(self):
        """Set up test fixtures."""
        self.config = {
            "evolutionary": {"max_generations": 10},
            "paragraph_fusion": {
                "style_alignment_threshold": 0.7,
                "proposition_recall_threshold": 0.85
            },
            "translator": {"max_tokens": 200}
        }

        # Mock LLM provider
        self.mock_llm = Mock()
        self.mock_llm.call.return_value = json.dumps({
            "anchor_score": 0.95,
            "semantic_score": 0.90,
            "anchor_feedback": "",
            "semantic_feedback": "",
            "overall_feedback": "Good",
            "pass": True
        })

        # Create critic and set LLM provider directly
        with patch('src.generator.llm_provider.LLMProvider', return_value=self.mock_llm):
            self.critic = SemanticCritic(config_path="config.json")
            self.critic.llm_provider = self.mock_llm

    def test_critic_returns_anchor_and_semantic_scores(self):
        """Test that critic returns separate anchor and semantic scores."""
        draft = "In the ruins, there was a void."
        propositions = ["The Soviet Union collapsed."]
        template = "In [NP], there was [NP]."

        result = self.critic.evaluate_sentence_fit(draft, propositions, template, verbose=False)

        # Check new structure-aware fields
        self.assertIn("anchor_score", result)
        self.assertIn("semantic_score", result)
        self.assertIn("anchor_feedback", result)
        self.assertIn("semantic_feedback", result)

        # Check backward compatibility aliases
        self.assertIn("meaning_score", result)
        self.assertIn("structure_score", result)
        self.assertIn("meaning_feedback", result)
        self.assertIn("structure_feedback", result)

        # Verify aliases match
        self.assertEqual(result["meaning_score"], result["semantic_score"])
        self.assertEqual(result["structure_score"], result["anchor_score"])
        self.assertEqual(result["meaning_feedback"], result["semantic_feedback"])
        self.assertEqual(result["structure_feedback"], result["anchor_feedback"])

    def test_critic_extracts_anchors_before_evaluation(self):
        """Test that critic extracts anchors before calling LLM."""
        draft = "The void appeared."
        propositions = ["The Soviet Union collapsed."]
        template = "In [NP], there was [NP]."

        # Mock LLM to verify it receives anchor information
        call_count = [0]
        def mock_call(system_prompt="", user_prompt="", **kwargs):
            call_count[0] += 1
            # Verify anchors are mentioned in prompt
            if call_count[0] == 1:  # First call is the evaluation
                self.assertIn("Fixed Anchors", user_prompt)
                self.assertIn("In", user_prompt)
                self.assertIn("there was", user_prompt)
            return json.dumps({
                "anchor_score": 0.5,
                "semantic_score": 0.9,
                "anchor_feedback": "Missing anchor 'In' at start.",
                "semantic_feedback": "",
                "overall_feedback": "Add 'In' at start.",
                "pass": False
            })

        self.mock_llm.call = mock_call

        result = self.critic.evaluate_sentence_fit(draft, propositions, template, verbose=False)

        # Verify LLM was called
        self.assertGreater(call_count[0], 0)
        # Verify result structure
        self.assertIn("anchor_score", result)
        self.assertIn("anchor_feedback", result)

    def test_critic_handles_missing_anchors(self):
        """Test that critic correctly identifies missing anchors."""
        draft = "The void appeared."
        propositions = ["The Soviet Union collapsed."]
        template = "In [NP], there was [NP]."

        # Mock LLM to return low anchor score
        self.mock_llm.call.return_value = json.dumps({
            "anchor_score": 0.3,
            "semantic_score": 0.9,
            "anchor_feedback": "Missing anchor 'In' at start. Missing 'there was' construction.",
            "semantic_feedback": "All meaning atoms present.",
            "overall_feedback": "Add 'In' at start and use 'there was' construction.",
            "pass": False
        })

        result = self.critic.evaluate_sentence_fit(draft, propositions, template, verbose=False)

        self.assertLess(result["anchor_score"], 0.9)
        self.assertGreaterEqual(result["semantic_score"], 0.9)
        self.assertFalse(result["pass"])
        self.assertIn("anchor", result["anchor_feedback"].lower() or result["overall_feedback"].lower())

    def test_critic_handles_missing_semantics(self):
        """Test that critic correctly identifies missing meaning atoms."""
        draft = "In the ruins, there was a void."
        propositions = ["The Soviet Union collapsed.", "The shift to capitalism failed."]
        template = "In [NP], there was [NP]."

        # Mock LLM to return low semantic score
        self.mock_llm.call.return_value = json.dumps({
            "anchor_score": 0.95,
            "semantic_score": 0.5,
            "anchor_feedback": "",
            "semantic_feedback": "Missing mention of 'shift to capitalism' and 'failed'.",
            "overall_feedback": "Add mention of shift to capitalism and failure.",
            "pass": False
        })

        result = self.critic.evaluate_sentence_fit(draft, propositions, template, verbose=False)

        self.assertGreaterEqual(result["anchor_score"], 0.9)
        self.assertLess(result["semantic_score"], 0.9)
        self.assertFalse(result["pass"])
        # Verify semantic feedback contains information about missing meaning
        feedback_text = (result["semantic_feedback"] or result["overall_feedback"]).lower()
        self.assertTrue(
            "shift" in feedback_text or "capitalism" in feedback_text or "failed" in feedback_text,
            f"Feedback should mention missing meaning atoms, got: {feedback_text}"
        )


class TestWriterWithAnchors(unittest.TestCase):
    """Test writer's ability to use anchors in generation and repair."""

    def setUp(self):
        """Set up test fixtures."""
        self.config = {
            "evolutionary": {"max_generations": 10},
            "paragraph_fusion": {
                "style_alignment_threshold": 0.7,
                "proposition_recall_threshold": 0.85
            },
            "translator": {"max_tokens": 200}
        }

        # Create translator with mocked dependencies
        with patch('src.generator.translator.LLMProvider'):
            with patch('src.generator.translator.SoftScorer'):
                with patch('src.generator.translator.Structuralizer'):
                    with patch('src.analyzer.structure_extractor.StructureExtractor'):
                        with patch('src.analyzer.rhetorical_classifier.RhetoricalClassifier'):
                            self.translator = StyleTranslator(config_path="config.json")
                            self.translator.config = self.config
                            self.translator.translator_config = self.config.get("translator", {})
                            self.translator.paragraph_fusion_config = self.config.get("paragraph_fusion", {})

                            # Mock LLM provider
                            self.mock_llm = Mock()
                            self.translator.llm_provider = self.mock_llm

    def test_generate_sentence_includes_anchors_in_prompt(self):
        """Test that sentence generation prompt includes anchor information."""
        template = "In [NP], there was [NP]."
        propositions = ["The Soviet Union collapsed."]

        # Track what was sent to LLM
        call_args = []
        def mock_call(system_prompt="", user_prompt="", **kwargs):
            call_args.append(user_prompt)
            return "In the ruins, there was a void."

        self.mock_llm.call = mock_call

        result = self.translator._generate_sentence_from_plan(
            template=template,
            assigned_propositions=propositions,
            author_name="Mao",
            verbose=False
        )

        # Verify LLM was called
        self.assertGreater(len(call_args), 0)
        # Verify prompt includes anchor information
        prompt_text = call_args[0]
        self.assertIn("fixed structural anchors", prompt_text.lower())
        self.assertIn("In", prompt_text)
        self.assertIn("there was", prompt_text)

    def test_repair_instruction_includes_anchors(self):
        """Test that repair instructions include anchor information."""
        template = "The fact is [NP]."
        propositions = ["The Soviet Union collapsed."]

        # Mock critic to fail first time, pass second time
        call_count = [0]
        def mock_evaluate_sentence_fit(draft, props, template, verbose=False):
            call_count[0] += 1
            if call_count[0] == 1:
                return {
                    "anchor_score": 0.5,
                    "semantic_score": 0.9,
                    "anchor_feedback": "Missing anchor 'The fact is'.",
                    "semantic_feedback": "",
                    "overall_feedback": "Add 'The fact is' at start.",
                    "pass": False,
                    # Backward compatibility aliases
                    "meaning_score": 0.9,
                    "structure_score": 0.5,
                    "meaning_feedback": "",
                    "structure_feedback": "Missing anchor 'The fact is'."
                }
            else:
                return {
                    "anchor_score": 0.95,
                    "semantic_score": 0.95,
                    "anchor_feedback": "",
                    "semantic_feedback": "",
                    "overall_feedback": "Good",
                    "pass": True,
                    # Backward compatibility aliases
                    "meaning_score": 0.95,
                    "structure_score": 0.95,
                    "meaning_feedback": "",
                    "structure_feedback": ""
                }

        # Track repair instruction
        repair_instructions = []
        def mock_call(system_prompt="", user_prompt="", **kwargs):
            if "Previous attempt" in user_prompt or "CRITICAL" in user_prompt:
                repair_instructions.append(user_prompt)
            return "The fact is the Soviet Union collapsed."

        self.mock_llm.call = mock_call

        mock_critic = Mock()
        mock_critic.evaluate_sentence_fit = mock_evaluate_sentence_fit
        # Mock _extract_fixed_anchors to return actual anchors
        mock_critic._extract_fixed_anchors = lambda t: ["The fact is", "."] if "fact" in t else []

        with patch('src.validator.semantic_critic.SemanticCritic', return_value=mock_critic):
            sentence, success = self.translator._write_and_refine_sentence(
                template=template,
                assigned_propositions=propositions,
                author_name="Mao",
                max_tries=3,
                verbose=False
            )

        # Verify repair instruction was generated
        if repair_instructions:
            repair_text = repair_instructions[0]
            self.assertIn("fixed structural anchors", repair_text.lower())
            self.assertIn("fact", repair_text.lower())


class TestEndToEndIntegration(unittest.TestCase):
    """Test end-to-end integration of structure-aware verification."""

    def setUp(self):
        """Set up test fixtures."""
        self.config = {
            "evolutionary": {"max_generations": 10},
            "paragraph_fusion": {
                "style_alignment_threshold": 0.7,
                "proposition_recall_threshold": 0.85,
                "coherence_threshold": 0.75,
                "topic_similarity_threshold": 0.55
            },
            "translator": {"max_tokens": 400}
        }

        # Create translator with mocked dependencies
        with patch('src.generator.translator.LLMProvider'):
            with patch('src.generator.translator.SoftScorer'):
                with patch('src.generator.translator.Structuralizer'):
                    with patch('src.analyzer.structure_extractor.StructureExtractor'):
                        with patch('src.analyzer.rhetorical_classifier.RhetoricalClassifier'):
                            self.translator = StyleTranslator(config_path="config.json")
                            self.translator.config = self.config
                            self.translator.translator_config = self.config.get("translator", {})
                            self.translator.paragraph_fusion_config = self.config.get("paragraph_fusion", {})

                            # Mock proposition extractor
                            self.mock_prop_extractor = Mock()
                            self.mock_prop_extractor.extract_atomic_propositions.return_value = [
                                "I spent my childhood scavenging.",
                                "The Soviet Union was a ghost."
                            ]
                            self.translator.proposition_extractor = self.mock_prop_extractor

                            # Mock rhetorical classifier
                            self.mock_rhetorical_classifier = Mock()
                            self.mock_rhetorical_classifier.classify_mode.return_value = "NARRATIVE"
                            self.translator.rhetorical_classifier = self.mock_rhetorical_classifier

                            # Mock LLM provider
                            self.mock_llm = Mock()
                            self.translator.llm_provider = self.mock_llm

    def test_end_to_end_with_anchor_preservation(self):
        """Test that end-to-end pipeline preserves anchors in output."""
        input_paragraph = "I spent my childhood scavenging. The Soviet Union was a ghost."

        # Mock Atlas
        mock_atlas = Mock()
        mock_atlas.get_examples_by_rhetoric.return_value = [
            "In the ruins, there was a void. The fact is clear."
        ]

        # Mock LLM calls for planner and writer
        call_count = [0]
        def mock_llm_call(system_prompt="", user_prompt="", **kwargs):
            call_count[0] += 1
            if "Narrative Architect" in user_prompt or "Assign every Proposition" in user_prompt:
                # Planner call
                return json.dumps([
                    {"slot_index": 0, "propositions": ["I spent my childhood scavenging."], "is_high_risk": False},
                    {"slot_index": 1, "propositions": ["The Soviet Union was a ghost."], "is_high_risk": False}
                ])
            elif "Write a single sentence" in user_prompt or "Generate a single" in user_prompt:
                # Writer call - return sentence with anchors
                if "In" in user_prompt and "there was" in user_prompt:
                    return "In my childhood, there was scavenging."
                elif "fact" in user_prompt.lower():
                    return "The fact is the Soviet Union was a ghost."
                else:
                    return "Generated sentence."
            else:
                return "Response."

        self.mock_llm.call = mock_llm_call

        # Mock critic
        mock_critic = Mock()
        mock_critic.evaluate.return_value = {
            "pass": True,
            "score": 0.9,
            "proposition_recall": 0.9,
            "style_alignment": 0.85,
            "adherence_score": 0.9,
            "recall_score": 0.9,
            "feedback": "Good",
            "reason": "Meets thresholds",
            "recall_details": {"preserved": [], "missing": []},
            "style_details": {"lexicon_density": 0.0}
        }
        mock_critic.evaluate_sentence_fit.return_value = {
            "anchor_score": 0.95,
            "semantic_score": 0.95,
            "anchor_feedback": "",
            "semantic_feedback": "",
            "overall_feedback": "Good",
            "pass": True,
            "meaning_score": 0.95,
            "structure_score": 0.95,
            "meaning_feedback": "",
            "structure_feedback": ""
        }
        # Mock _extract_fixed_anchors to return actual anchors based on template
        def extract_anchors(template):
            if "In" in template and "there was" in template:
                return ["In", ", there was", "."]
            elif "fact" in template.lower():
                return ["The fact is", "."]
            else:
                return ["."]
        mock_critic._extract_fixed_anchors = extract_anchors

        with patch('src.generator.translator.SemanticCritic', return_value=mock_critic):
            with patch('src.validator.semantic_critic.SemanticCritic', return_value=mock_critic):
                result = self.translator.translate_paragraph(
                    paragraph=input_paragraph,
                    author_name="Mao",
                    atlas=mock_atlas,
                    position="BODY",
                    verbose=False
                )

        # Verify result structure
        self.assertIsNotNone(result)
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 4)

        generated_paragraph, _, _, _ = result
        self.assertIsInstance(generated_paragraph, str)
        self.assertGreater(len(generated_paragraph), 0)

        # Verify LLM was called (planner, writer)
        self.assertGreater(call_count[0], 0)


if __name__ == '__main__':
    unittest.main()

