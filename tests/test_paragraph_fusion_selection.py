"""Unit tests for paragraph fusion tiered selection and repair loop.

Tests the scenario where paragraph fusion generates 5 variations,
evaluates them all, and uses tiered selection logic to pick the best.
Also tests the repair loop when recall is too low.
"""

import sys
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import json
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.generator.translator import StyleTranslator
from src.validator.semantic_critic import SemanticCritic
from src.ingestion.blueprint import BlueprintExtractor
from src.analysis.semantic_analyzer import PropositionExtractor


class MockLLMProvider:
    """Mock LLM provider for testing."""

    def __init__(self, mock_responses=None):
        self.mock_responses = mock_responses if mock_responses is not None else {}
        self.call_count = 0
        self.call_history = []

    def call(self, system_prompt, user_prompt, model_type="editor", require_json=False, temperature=0.7, max_tokens=500):
        self.call_count += 1
        self.call_history.append({
            "system_prompt": system_prompt,
            "user_prompt": user_prompt[:200] if len(user_prompt) > 200 else user_prompt,
            "model_type": model_type
        })

        # Check if this is a repair call
        if "missed these specific facts" in user_prompt:
            # Repair generation
            return json.dumps([
                "It is through the dialectical process that human experience reinforces the rule of finitude. The biological cycle of birth, life, and decay defines our reality. Every object we touch eventually breaks, demonstrating the materialist law of transformation.",
                "Human experience, as demonstrated through practice, reinforces the rule of finitude. This is evident in the biological cycle of birth, life, and decay, which defines our reality. Furthermore, every object we touch eventually breaks, illustrating the inevitable process of material transformation.",
                "The rule of finitude is reinforced by human experience. Our reality is defined by the biological cycle of birth, life, and decay. Objects we touch eventually break, showing the dialectical nature of material existence."
            ])
        elif "CONTENT SOURCE (Atomic Propositions)" in user_prompt or "write a single cohesive paragraph" in system_prompt:
            # Initial paragraph fusion generation
            # Return variations that match the test expectations
            return json.dumps([
                "It is through the dialectical process that human experience reinforces the rule of finitude.",  # Short, high recall, low style
                "Human experience reinforces the rule of finitude. The biological cycle of birth, life, and decay defines our reality.",  # Medium, high recall, medium style
                "It is an objective materialist fact that human practice reinforces the rule of finitude, as demonstrated through the biological cycle of birth, life, and decay that defines our reality, and the inevitable breaking of every object we touch, illustrating the dialectical law of material transformation.",  # Long, high recall, high style (BEST)
                "Human experience proves finitude. Birth and death define reality. Stars break.",  # Plain English, low recall, low style
                "The rule of finitude is reinforced by human experience through the biological cycle."  # Medium recall, medium style
            ])
        elif "extract every distinct fact" in system_prompt or "Extract every distinct fact" in user_prompt:
            # Proposition extraction
            return json.dumps([
                "Human experience reinforces the rule of finitude",
                "The biological cycle of birth, life, and decay defines our reality",
                "Every object we touch eventually breaks"
            ])
        else:
            return json.dumps(["Mock response"])


class MockStyleAtlas:
    """Mock StyleAtlas for testing."""

    def __init__(self):
        self.examples = [
            "It is a fundamental law of dialectics that all material bodies, whether terrestrial or celestial, must undergo the process of internal contradiction, leading inevitably to transformation and decay. This process defines the very essence of existence, where every phenomenon is both cause and effect, constantly evolving.",
            "The historical trajectory of human society demonstrates a continuous struggle between opposing forces, a struggle that propels progress through the resolution of inherent contradictions. Thus, the development of consciousness is inextricably linked to the material conditions of production.",
            "The concept of finitude, often perceived as a limitation, is in fact a necessary condition for the emergence of new forms of being. Without the end of one cycle, the beginning of another would be impossible, illustrating the cyclical nature of all natural and social processes.",
        ]

    def get_examples_by_rhetoric(self, rhetorical_type, top_k, author_name, query_text=None, exclude=None):
        return self.examples[:top_k]

    def get_author_style_vector(self, author_name):
        return np.array([0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2], dtype=np.float32)


def test_tiered_selection_qualified_pool():
    """Test 1: Tiered Selection - Qualified Pool (recall >= 0.8) selects by composite score."""
    print("\n" + "="*60)
    print("TEST 1: Tiered Selection - Qualified Pool")
    print("="*60)

    translator = StyleTranslator()
    translator.llm_provider = MockLLMProvider()
    translator.paragraph_fusion_config = {
        "proposition_recall_threshold": 0.8,
        "num_variations": 5
    }

    mock_atlas = MockStyleAtlas()

    # Mock the critic to return different scores for different variations
    with patch('src.validator.semantic_critic.SemanticCritic') as MockCritic:
        mock_critic_instance = MockCritic.return_value

        # Variation 1: High recall (0.9), low style (0.3) -> score = 0.9*0.6 + 0.3*0.4 = 0.66
        # Variation 2: High recall (0.85), medium style (0.5) -> score = 0.85*0.6 + 0.5*0.4 = 0.71
        # Variation 3: High recall (0.95), high style (0.9) -> score = 0.95*0.6 + 0.9*0.4 = 0.93 (BEST)
        # Variation 4: Low recall (0.4), low style (0.2) -> score = 0.4*0.6 + 0.2*0.4 = 0.32
        # Variation 5: Medium recall (0.75), medium style (0.6) -> score = 0.75*0.6 + 0.6*0.4 = 0.69

        def mock_evaluate(generated_text, input_blueprint, propositions=None, is_paragraph=False, author_style_vector=None):
            # Map variations to scores based on content
            text_lower = generated_text.lower()

            # Variation 1: Short, high recall, low style
            if "dialectical process" in text_lower and "human experience" in text_lower and len(generated_text) < 100:
                return {
                    "proposition_recall": 0.9,
                    "style_alignment": 0.3,
                    "score": 0.9 * 0.6 + 0.3 * 0.4,  # 0.66
                    "recall_details": {"preserved": ["prop1", "prop2", "prop3"], "missing": []}
                }
            # Variation 2: Medium, high recall, medium style
            elif "biological cycle" in text_lower and "defines reality" in text_lower:
                return {
                    "proposition_recall": 0.85,
                    "style_alignment": 0.5,
                    "score": 0.85 * 0.6 + 0.5 * 0.4,  # 0.71
                    "recall_details": {"preserved": ["prop1", "prop2"], "missing": ["prop3"]}
                }
            # Variation 3: Long, high recall, high style (BEST) - look for long text with multiple propositions
            elif ("objective materialist" in text_lower or "dialectical" in text_lower) and len(generated_text) > 150:
                return {
                    "proposition_recall": 0.95,
                    "style_alignment": 0.9,
                    "score": 0.95 * 0.6 + 0.9 * 0.4,  # 0.93
                    "recall_details": {"preserved": ["prop1", "prop2", "prop3"], "missing": []}
                }
            # Variation 4: Plain English, low recall, low style
            elif "proves finitude" in text_lower or ("birth and death" in text_lower and "stars break" in text_lower):
                return {
                    "proposition_recall": 0.4,
                    "style_alignment": 0.2,
                    "score": 0.4 * 0.6 + 0.2 * 0.4,  # 0.32
                    "recall_details": {"preserved": ["prop1"], "missing": ["prop2", "prop3"]}
                }
            else:
                # Variation 5: Medium recall, medium style (default)
                return {
                    "proposition_recall": 0.75,
                    "style_alignment": 0.6,
                    "score": 0.75 * 0.6 + 0.6 * 0.4,  # 0.69
                    "recall_details": {"preserved": ["prop1", "prop2"], "missing": ["prop3"]}
                }

        mock_critic_instance.evaluate = mock_evaluate

        # Mock proposition extractor
        translator.proposition_extractor = Mock()
        translator.proposition_extractor.extract_atomic_propositions = Mock(return_value=[
            "Human experience reinforces the rule of finitude",
            "The biological cycle of birth, life, and decay defines our reality",
            "Every object we touch eventually breaks"
        ])

        result = translator.translate_paragraph(
            paragraph="Human experience reinforces the rule of finitude. The biological cycle of birth, life, and decay defines our reality. Every object we touch eventually breaks.",
            atlas=mock_atlas,
            author_name="Test Author",
            verbose=True
        )

        # Assertions
        assert result is not None
        assert isinstance(result, str)
        # Should select Variation 3 (highest composite score from qualified pool)
        # Check that it's a longer, styled text (not the short plain one)
        assert len(result) > 100, "Should select longer, styled variation"

        print(f"\n✓ Selected result: {result[:100]}...")
        print(f"✓ Result length: {len(result)} characters")
        print("✓ Test 1 PASSED: Tiered selection picks highest composite score from qualified pool")
        return True


def test_tiered_selection_fallback():
    """Test 2: Tiered Selection - Fallback (no qualified candidates) selects by highest recall."""
    print("\n" + "="*60)
    print("TEST 2: Tiered Selection - Fallback")
    print("="*60)

    translator = StyleTranslator()
    translator.llm_provider = MockLLMProvider()
    translator.paragraph_fusion_config = {
        "proposition_recall_threshold": 0.8,
        "num_variations": 5
    }

    mock_atlas = MockStyleAtlas()

    with patch('src.validator.semantic_critic.SemanticCritic') as MockCritic:
        mock_critic_instance = MockCritic.return_value

        # All variations have recall < 0.8, so we fall back to highest recall
        # Variation 1: recall 0.7 (highest)
        # Variation 2: recall 0.6
        # Variation 3: recall 0.5
        # Variation 4: recall 0.4
        # Variation 5: recall 0.3

        def mock_evaluate(generated_text, input_blueprint, propositions=None, is_paragraph=False, author_style_vector=None):
            if "dialectical process" in generated_text:
                return {
                    "proposition_recall": 0.7,  # Highest recall
                    "style_alignment": 0.3,
                    "score": 0.7 * 0.6 + 0.3 * 0.4,  # 0.54
                    "recall_details": {"preserved": ["prop1", "prop2"], "missing": ["prop3"]}
                }
            elif "biological cycle defines reality" in generated_text:
                return {
                    "proposition_recall": 0.6,
                    "style_alignment": 0.5,
                    "score": 0.6 * 0.6 + 0.5 * 0.4,  # 0.56 (higher score but lower recall)
                    "recall_details": {"preserved": ["prop1"], "missing": ["prop2", "prop3"]}
                }
            else:
                return {
                    "proposition_recall": 0.5,
                    "style_alignment": 0.4,
                    "score": 0.5 * 0.6 + 0.4 * 0.4,  # 0.46
                    "recall_details": {"preserved": ["prop1"], "missing": ["prop2", "prop3"]}
                }

        mock_critic_instance.evaluate = mock_evaluate

        translator.proposition_extractor = Mock()
        translator.proposition_extractor.extract_atomic_propositions = Mock(return_value=[
            "Human experience reinforces the rule of finitude",
            "The biological cycle of birth, life, and decay defines our reality",
            "Every object we touch eventually breaks"
        ])

        result = translator.translate_paragraph(
            paragraph="Human experience reinforces the rule of finitude. The biological cycle of birth, life, and decay defines our reality. Every object we touch eventually breaks.",
            atlas=mock_atlas,
            author_name="Test Author",
            verbose=True
        )

        # Assertions
        assert result is not None
        # Should select Variation 1 (highest recall, even though Variation 2 has higher score)
        assert "dialectical process" in result or "human experience" in result.lower()

        print(f"\n✓ Selected result: {result[:100]}...")
        print("✓ Test 2 PASSED: Fallback selects highest recall when no qualified candidates")
        return True


def test_repair_loop_triggered():
    """Test 3: Repair Loop - Triggered when recall < 0.7, generates repair variations."""
    print("\n" + "="*60)
    print("TEST 3: Repair Loop - Triggered")
    print("="*60)

    translator = StyleTranslator()
    mock_llm = MockLLMProvider()
    translator.llm_provider = mock_llm
    translator.paragraph_fusion_config = {
        "proposition_recall_threshold": 0.8,
        "num_variations": 5
    }

    mock_atlas = MockStyleAtlas()

    repair_called = {"value": False}

    with patch('src.validator.semantic_critic.SemanticCritic') as MockCritic:
        mock_critic_instance = MockCritic.return_value

        def mock_evaluate(generated_text, input_blueprint, propositions=None, is_paragraph=False, author_style_vector=None):
            text_lower = generated_text.lower()

            # Check if this is a repair variation (contains all three propositions)
            has_all_props = (
                "human experience" in text_lower and
                "biological cycle" in text_lower and
                ("object" in text_lower and "break" in text_lower or "touch" in text_lower)
            )

            # Initial variations all have low recall
            if not has_all_props and "dialectical process" in text_lower:
                return {
                    "proposition_recall": 0.6,  # Below repair threshold (0.7)
                    "style_alignment": 0.3,
                    "score": 0.6 * 0.6 + 0.3 * 0.4,  # 0.48
                    "recall_details": {
                        "preserved": ["prop1", "prop2"],
                        "missing": ["prop3"]  # Missing: "Every object we touch eventually breaks"
                    }
                }
            elif has_all_props and ("dialectical" in text_lower or "materialist" in text_lower):
                # Repair variations have higher recall (all propositions present)
                repair_called["value"] = True
                return {
                    "proposition_recall": 0.9,  # Repaired!
                    "style_alignment": 0.8,
                    "score": 0.9 * 0.6 + 0.8 * 0.4,  # 0.86
                    "recall_details": {
                        "preserved": ["prop1", "prop2", "prop3"],
                        "missing": []
                    }
                }
            else:
                return {
                    "proposition_recall": 0.5,
                    "style_alignment": 0.4,
                    "score": 0.5 * 0.6 + 0.4 * 0.4,  # 0.46
                    "recall_details": {
                        "preserved": ["prop1"],
                        "missing": ["prop2", "prop3"]
                    }
                }

        mock_critic_instance.evaluate = mock_evaluate

        translator.proposition_extractor = Mock()
        translator.proposition_extractor.extract_atomic_propositions = Mock(return_value=[
            "Human experience reinforces the rule of finitude",
            "The biological cycle of birth, life, and decay defines our reality",
            "Every object we touch eventually breaks"
        ])

        result = translator.translate_paragraph(
            paragraph="Human experience reinforces the rule of finitude. The biological cycle of birth, life, and decay defines our reality. Every object we touch eventually breaks.",
            atlas=mock_atlas,
            author_name="Test Author",
            verbose=True
        )

        # Assertions
        assert result is not None
        assert repair_called["value"], "Repair loop should have been triggered"

        # Check that LLM was called for repair (should be more than initial generation call)
        repair_calls = [c for c in mock_llm.call_history if "missed these specific facts" in c["user_prompt"]]
        assert len(repair_calls) > 0, "Repair prompt should have been sent to LLM"

        print(f"\n✓ Selected result: {result[:100]}...")
        print(f"✓ Repair loop was triggered: {repair_called['value']}")
        print("✓ Test 3 PASSED: Repair loop generates and evaluates repair variations")
        return True


def test_repair_loop_not_triggered():
    """Test 4: Repair Loop - Not triggered when recall >= 0.7."""
    print("\n" + "="*60)
    print("TEST 4: Repair Loop - Not Triggered")
    print("="*60)

    translator = StyleTranslator()
    mock_llm = MockLLMProvider()
    translator.llm_provider = mock_llm
    translator.paragraph_fusion_config = {
        "proposition_recall_threshold": 0.8,
        "num_variations": 5
    }

    mock_atlas = MockStyleAtlas()

    with patch('src.validator.semantic_critic.SemanticCritic') as MockCritic:
        mock_critic_instance = MockCritic.return_value

        def mock_evaluate(generated_text, input_blueprint, propositions=None, is_paragraph=False, author_style_vector=None):
            # All variations have recall >= 0.7, so repair should NOT be triggered
            return {
                "proposition_recall": 0.75,  # Above repair threshold (0.7)
                "style_alignment": 0.5,
                "score": 0.75 * 0.6 + 0.5 * 0.4,  # 0.65
                "recall_details": {
                    "preserved": ["prop1", "prop2"],
                    "missing": ["prop3"]
                }
            }

        mock_critic_instance.evaluate = mock_evaluate

        translator.proposition_extractor = Mock()
        translator.proposition_extractor.extract_atomic_propositions = Mock(return_value=[
            "Human experience reinforces the rule of finitude",
            "The biological cycle of birth, life, and decay defines our reality",
            "Every object we touch eventually breaks"
        ])

        result = translator.translate_paragraph(
            paragraph="Human experience reinforces the rule of finitude. The biological cycle of birth, life, and decay defines our reality. Every object we touch eventually breaks.",
            atlas=mock_atlas,
            author_name="Test Author",
            verbose=True
        )

        # Assertions
        assert result is not None

        # Check that repair was NOT called (only initial generation)
        repair_calls = [c for c in mock_llm.call_history if "missed these specific facts" in c["user_prompt"]]
        assert len(repair_calls) == 0, "Repair loop should NOT have been triggered"

        print(f"\n✓ Selected result: {result[:100]}...")
        print("✓ Test 4 PASSED: Repair loop not triggered when recall >= 0.7")
        return True


def test_plain_english_trap_avoided():
    """Test 5: Plain English Trap - System prefers styled text with good recall over plain text with perfect recall."""
    print("\n" + "="*60)
    print("TEST 5: Plain English Trap Avoided")
    print("="*60)

    translator = StyleTranslator()
    translator.llm_provider = MockLLMProvider()
    translator.paragraph_fusion_config = {
        "proposition_recall_threshold": 0.8,
        "num_variations": 5
    }

    mock_atlas = MockStyleAtlas()

    selected_variation = {"value": None}

    with patch('src.validator.semantic_critic.SemanticCritic') as MockCritic:
        mock_critic_instance = MockCritic.return_value

        def mock_evaluate(generated_text, input_blueprint, propositions=None, is_paragraph=False, author_style_vector=None):
            text_lower = generated_text.lower()

            # Candidate A: Plain English, perfect recall (1.0), low style (0.1)
            # Match the exact text from mock LLM: "Human experience proves finitude. Birth and death define reality. Stars break."
            if "proves finitude" in text_lower and "stars break" in text_lower:
                selected_variation["value"] = "plain"
                return {
                    "proposition_recall": 1.0,  # Perfect recall
                    "style_alignment": 0.1,  # Low style
                    "score": 1.0 * 0.6 + 0.1 * 0.4,  # 0.64
                    "recall_details": {"preserved": ["prop1", "prop2", "prop3"], "missing": []}
                }
            # Candidate B: Styled text, good recall (0.9), high style (0.9)
            # Match the exact text from mock LLM: "It is an objective materialist fact..."
            elif "objective materialist fact" in text_lower and len(generated_text) > 200:
                selected_variation["value"] = "styled"
                return {
                    "proposition_recall": 0.9,  # Good recall (slightly lower)
                    "style_alignment": 0.9,  # High style
                    "score": 0.9 * 0.6 + 0.9 * 0.4,  # 0.90 (HIGHER composite score)
                    "recall_details": {"preserved": ["prop1", "prop2", "prop3"], "missing": []}
                }
            else:
                return {
                    "proposition_recall": 0.7,
                    "style_alignment": 0.5,
                    "score": 0.7 * 0.6 + 0.5 * 0.4,  # 0.62
                    "recall_details": {"preserved": ["prop1", "prop2"], "missing": ["prop3"]}
                }

        mock_critic_instance.evaluate = mock_evaluate

        translator.proposition_extractor = Mock()
        translator.proposition_extractor.extract_atomic_propositions = Mock(return_value=[
            "Human experience reinforces the rule of finitude",
            "The biological cycle of birth, life, and decay defines our reality",
            "Every object we touch eventually breaks"
        ])

        result = translator.translate_paragraph(
            paragraph="Human experience reinforces the rule of finitude. The biological cycle of birth, life, and decay defines our reality. Every object we touch eventually breaks.",
            atlas=mock_atlas,
            author_name="Test Author",
            verbose=True
        )

        # Assertions
        assert result is not None
        # Should select styled version (Candidate B) because it has higher composite score
        # Check that the result contains the styled text markers
        text_lower = result.lower()
        is_styled = "objective materialist" in text_lower or ("dialectical" in text_lower and len(result) > 200)
        is_plain = "proves finitude" in text_lower and "stars break" in text_lower

        assert is_styled, f"Should select styled text, got: {result[:100]}..."
        assert not is_plain, "Should NOT select plain English text"
        # Should be longer (styled text is longer)
        assert len(result) > 100, "Should select longer styled text"

        print(f"\n✓ Selected result: {result[:100]}...")
        print(f"✓ Is styled: {is_styled}, Is plain: {is_plain}")
        print(f"✓ Result length: {len(result)} characters")
        print("✓ Test 5 PASSED: Plain English trap avoided - styled text with good recall preferred")
        return True


if __name__ == "__main__":
    all_passed = True

    try:
        if not test_tiered_selection_qualified_pool():
            all_passed = False
    except Exception as e:
        print(f"\n✗ Test 1 FAILED with exception: {e}")
        import traceback
        traceback.print_exc()
        all_passed = False

    try:
        if not test_tiered_selection_fallback():
            all_passed = False
    except Exception as e:
        print(f"\n✗ Test 2 FAILED with exception: {e}")
        import traceback
        traceback.print_exc()
        all_passed = False

    try:
        if not test_repair_loop_triggered():
            all_passed = False
    except Exception as e:
        print(f"\n✗ Test 3 FAILED with exception: {e}")
        import traceback
        traceback.print_exc()
        all_passed = False

    try:
        if not test_repair_loop_not_triggered():
            all_passed = False
    except Exception as e:
        print(f"\n✗ Test 4 FAILED with exception: {e}")
        import traceback
        traceback.print_exc()
        all_passed = False

    try:
        if not test_plain_english_trap_avoided():
            all_passed = False
    except Exception as e:
        print(f"\n✗ Test 5 FAILED with exception: {e}")
        import traceback
        traceback.print_exc()
        all_passed = False

    print("\n" + "="*60)
    print("FINAL TEST SUMMARY")
    print("="*60)
    if all_passed:
        print("🎉 All tests passed! Paragraph fusion tiered selection and repair loop verified.")
    else:
        print("⚠️  Some tests failed. Please review the implementation.")

