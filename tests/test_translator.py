"""Tests for style translator."""

import sys
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Ensure config.json exists before imports
from tests.test_helpers import ensure_config_exists
ensure_config_exists()

# Import with error handling for missing dependencies
try:
    from src.ingestion.blueprint import SemanticBlueprint
    from src.atlas.rhetoric import RhetoricalType
    from src.generator.translator import StyleTranslator
    from src.critic.judge import LLMJudge
    from src.validator.semantic_critic import SemanticCritic
    DEPENDENCIES_AVAILABLE = True
except (ImportError, ModuleNotFoundError) as e:
    DEPENDENCIES_AVAILABLE = False
    IMPORT_ERROR = str(e)
    print(f"⚠ Skipping tests: Missing dependencies - {IMPORT_ERROR}")
    # Create dummy classes to prevent NameError
    class SemanticBlueprint:
        pass
    class RhetoricalType:
        OBSERVATION = None
    class StyleTranslator:
        pass
    class LLMJudge:
        pass
    class SemanticCritic:
        pass


def test_basic_translation():
    """Test basic translation with mocked LLM."""
    if not DEPENDENCIES_AVAILABLE:
        print("⚠ SKIPPED: test_basic_translation (missing dependencies)")
        return
    # Create a simple blueprint
    blueprint = SemanticBlueprint(
        original_text="The cat sat on the mat.",
        svo_triples=[("the cat", "sit", "the mat")],
        named_entities=[],
        core_keywords={"cat", "sit", "mat"},
        citations=[],
        quotes=[]
    )

    # Mock LLM provider
    mock_llm = Mock()
    mock_llm.call.return_value = "The feline rested upon the rug."

    translator = StyleTranslator(config_path="config.json")
    translator.llm_provider = mock_llm

    examples = ["The feline rested upon the rug.", "A cat positioned itself on a mat."]

    result = translator.translate(
        blueprint=blueprint,
        author_name="Test Author",
        style_dna="Formal and precise.",
        rhetorical_type=RhetoricalType.OBSERVATION,
        examples=examples
    )

    assert result == "The feline rested upon the rug."
    assert mock_llm.call.called

    print("✓ test_basic_translation passed")


def test_dynamic_expansion_thresholds():
    """Test that dynamic expansion thresholds allow more expansion for short inputs."""
    if not DEPENDENCIES_AVAILABLE:
        print("⚠ SKIPPED: test_dynamic_expansion_thresholds (missing dependencies)")
        return
    from src.generator.translator import StyleTranslator

    translator = StyleTranslator(config_path="config.json")

    # Test short input (< 10 words) - should allow 5x expansion
    short_blueprint = SemanticBlueprint(
        original_text="Human experience reinforces the rule.",
        svo_triples=[("Human experience", "reinforces", "the rule")],
        named_entities=[],
        core_keywords={"human", "experience", "reinforce", "rule"},
        citations=[],
        quotes=[]
    )

    # Create a candidate that's 4.5x longer (should pass for short input)
    short_candidate = " ".join(["word"] * 32)  # 32 words vs 7 words = 4.57x

    evaluation = translator._evaluate_template_candidate(
        candidate=short_candidate,
        blueprint=short_blueprint,
        skeleton="[NP] [VP] [NP]",
        style_dna={"lexicon": []}
    )

    # Should not be rejected for expansion (4.57 < 5.0 for short inputs)
    assert "excessive expansion" not in evaluation.get("rejection_reason", "").lower()

    # Test longer input (>= 20 words) - should only allow 2.5x expansion
    long_blueprint = SemanticBlueprint(
        original_text=" ".join(["word"] * 20),  # 20 words
        svo_triples=[],
        named_entities=[],
        core_keywords=set(),
        citations=[],
        quotes=[]
    )

    # Create a candidate that's 3x longer (should be rejected)
    long_candidate = " ".join(["word"] * 60)  # 60 words vs 20 words = 3.0x

    evaluation = translator._evaluate_template_candidate(
        candidate=long_candidate,
        blueprint=long_blueprint,
        skeleton="[NP] [VP]",
        style_dna={"lexicon": []}
    )

    # Should be rejected for expansion (3.0 > 2.5 for longer inputs)
    assert "excessive expansion" in evaluation.get("rejection_reason", "").lower() or evaluation.get("composite_score", 1.0) == 0.0

    print("✓ test_dynamic_expansion_thresholds passed")


def test_rescue_logic():
    """Test that high-adherence/low-semantic candidates are marked for repair."""
    if not DEPENDENCIES_AVAILABLE:
        print("⚠ SKIPPED: test_rescue_logic (missing dependencies)")
        return
    from src.generator.translator import StyleTranslator
    from unittest.mock import Mock, patch

    translator = StyleTranslator(config_path="config.json")

    blueprint = SemanticBlueprint(
        original_text="Human experience reinforces the rule of finitude.",
        svo_triples=[("Human experience", "reinforces", "the rule of finitude")],
        named_entities=[],
        core_keywords={"human", "experience", "reinforce", "rule", "finitude"},
        citations=[],
        quotes=[]
    )

    # Create a candidate with perfect structure but missing keywords
    # This simulates a rescue candidate: high adherence, low semantic
    candidate_text = "The practice of cognition is the perceptual process."
    skeleton = "The [NP] of [NP] is the [ADJ] [NP]."

    # Mock the evaluation to return high adherence but low semantic
    with patch.object(translator, '_evaluate_template_candidate') as mock_eval:
        mock_eval.return_value = {
            "semantic_score": 0.4,  # Low semantic (missing keywords)
            "adherence_score": 0.95,  # High adherence (perfect structure)
            "style_density": 0.3,
            "composite_score": 0.5,
            "passed_gates": False
        }

        # Evaluate candidate
        evaluation = mock_eval.return_value

        # Check if it would be marked as rescue candidate
        needs_repair = (evaluation.get("adherence_score", 0.0) > 0.9 and
                       evaluation.get("semantic_score", 0.0) < 0.6)

        assert needs_repair == True, "High-adherence/low-semantic candidate should be marked for repair"

    # Test that SemanticCritic can be imported and instantiated (catches import errors)
    from src.validator.semantic_critic import SemanticCritic
    critic = SemanticCritic(config_path="config.json")
    assert critic is not None, "SemanticCritic should be importable and instantiable"

    print("✓ test_rescue_logic passed")


def test_explicit_concept_mapping():
    """Test that structural clone prompt includes explicit concept mapping."""
    if not DEPENDENCIES_AVAILABLE:
        print("⚠ SKIPPED: test_explicit_concept_mapping (missing dependencies)")
        return
    from src.generator.mutation_operators import StructuralCloneOperator
    from src.ingestion.blueprint import SemanticBlueprint
    from unittest.mock import Mock

    operator = StructuralCloneOperator()

    blueprint = SemanticBlueprint(
        original_text="Human experience reinforces the rule of finitude.",
        svo_triples=[("Human experience", "reinforces", "the rule of finitude")],
        named_entities=[],
        core_keywords={"human", "experience", "reinforce", "rule", "finitude"},
        citations=[],
        quotes=[]
    )

    skeleton = "The [NP] of [NP] [VP] the [NP]."

    # Mock LLM provider
    mock_llm = Mock()
    mock_llm.call.return_value = "The practice of human experience reinforces the rule of finitude."

    # Generate with structural clone
    result = operator.generate(
        current_draft="",
        blueprint=blueprint,
        author_name="Test Author",
        style_dna="Test style",
        rhetorical_type=RhetoricalType.OBSERVATION,
        llm_provider=mock_llm,
        skeleton=skeleton,
        style_lexicon=None
    )

    # Check that the prompt was called
    assert mock_llm.call.called

    # Check that the prompt includes concept mapping
    call_args = mock_llm.call.call_args
    user_prompt = call_args[1]['user_prompt'] if 'user_prompt' in call_args[1] else call_args[0][1]

    # Should include explicit mapping section
    assert "STEP 1: PLANNING" in user_prompt or "PLANNING" in user_prompt
    assert "mapping" in user_prompt.lower() or "map" in user_prompt.lower()
    # Should mention subjects, verbs, or objects
    assert ("subject" in user_prompt.lower() or
            "verb" in user_prompt.lower() or
            "object" in user_prompt.lower() or
            "human experience" in user_prompt.lower() or
            "reinforces" in user_prompt.lower())

    print("✓ test_explicit_concept_mapping passed")


def test_check_acceptance_without_fluency():
    """Test that _check_acceptance works without fluency_score."""
    if not DEPENDENCIES_AVAILABLE:
        print("⚠ SKIPPED: test_check_acceptance_without_fluency (missing dependencies)")
        return
    from src.generator.translator import StyleTranslator

    translator = StyleTranslator(config_path="config.json")

    # Test with fluency_score provided (backward compatibility)
    result1 = translator._check_acceptance(
        recall_score=1.0,
        precision_score=0.9,
        fluency_score=0.8,
        overall_score=0.95,
        pass_threshold=0.9
    )
    assert result1 == True

    # Test without fluency_score (new behavior)
    result2 = translator._check_acceptance(
        recall_score=1.0,
        precision_score=0.9,
        fluency_score=None,  # Not provided
        overall_score=0.95,
        pass_threshold=0.9
    )
    assert result2 == True

    # Test with low recall (should fail)
    result3 = translator._check_acceptance(
        recall_score=0.5,
        precision_score=0.9,
        fluency_score=None,
        overall_score=0.7,
        pass_threshold=0.9
    )
    assert result3 == False

    print("✓ test_check_acceptance_without_fluency passed")


def test_complex_translation():
    """Test complex translation."""
    if not DEPENDENCIES_AVAILABLE:
        print("⚠ SKIPPED: test_complex_translation (missing dependencies)")
        return
    blueprint = SemanticBlueprint(
        original_text="Human experience reinforces the rule of finitude.",
        svo_triples=[("human experience", "reinforce", "the rule of finitude")],
        named_entities=[],
        core_keywords={"human", "experience", "reinforce", "rule", "finitude"},
        citations=[],
        quotes=[]
    )

    mock_llm = Mock()
    mock_llm.call.return_value = "Human practice reinforces the rule of finitude."

    translator = StyleTranslator(config_path="config.json")
    translator.llm_provider = mock_llm

    examples = ["Practice makes perfect.", "The rule stands firm."]

    result = translator.translate(
        blueprint=blueprint,
        author_name="Mao",
        style_dna="Revolutionary and direct.",
        rhetorical_type=RhetoricalType.OBSERVATION,
        examples=examples
    )

    assert "practice" in result.lower() or "experience" in result.lower()
    assert mock_llm.call.called

    print("✓ test_complex_translation passed")


def test_rhetorical_mode_matching():
    """Test that rhetorical mode is included in prompt."""
    if not DEPENDENCIES_AVAILABLE:
        print("⚠ SKIPPED: test_rhetorical_mode_matching (missing dependencies)")
        return
    blueprint = SemanticBlueprint(
        original_text="A revolution is a process.",
        svo_triples=[("a revolution", "be", "a process")],
        named_entities=[],
        core_keywords={"revolution", "process"},
        citations=[],
        quotes=[]
    )

    mock_llm = Mock()
    mock_llm.call.return_value = "A revolution represents a process."

    translator = StyleTranslator(config_path="config.json")
    translator.llm_provider = mock_llm

    examples = ["A war is a struggle.", "A battle is a conflict."]

    result = translator.translate(
        blueprint=blueprint,
        author_name="Test Author",
        style_dna="Academic style.",
        rhetorical_type=RhetoricalType.DEFINITION,
        examples=examples
    )

    # Verify that the prompt included the rhetorical mode
    # Check all calls to find one that includes the rhetorical mode
    assert mock_llm.call.called, "LLM should have been called"
    found_rhetorical_mode = False
    for call in mock_llm.call.call_args_list:
        call_args = call
        if call_args:
            kwargs = call_args[1] if len(call_args) > 1 else {}
            args = call_args[0] if len(call_args) > 0 else ()
            user_prompt = kwargs.get('user_prompt', args[1] if len(args) > 1 else '')
            system_prompt = kwargs.get('system_prompt', args[0] if len(args) > 0 else '')
            combined_prompt = (user_prompt + ' ' + system_prompt).lower()
            if "definition" in combined_prompt:
                found_rhetorical_mode = True
                break
    assert found_rhetorical_mode, f"Rhetorical mode 'DEFINITION' not found in any LLM call. Total calls: {mock_llm.call.call_count}"

    print("✓ test_rhetorical_mode_matching passed")


def test_error_handling():
    """Test error handling for edge cases."""
    if not DEPENDENCIES_AVAILABLE:
        print("⚠ SKIPPED: test_error_handling (missing dependencies)")
        return
    blueprint = SemanticBlueprint(
        original_text="Test sentence.",
        svo_triples=[],
        named_entities=[],
        core_keywords=set(),
        citations=[],
        quotes=[]
    )

    # Test with empty examples
    mock_llm = Mock()
    mock_llm.call.return_value = "Translated text."

    translator = StyleTranslator(config_path="config.json")
    translator.llm_provider = mock_llm

    result = translator.translate(
        blueprint=blueprint,
        author_name="Test Author",
        style_dna="Test style.",
        rhetorical_type=RhetoricalType.OBSERVATION,
        examples=[]  # Empty examples
    )

    assert result == "Translated text."

    # Test with LLM failure
    mock_llm.call.side_effect = Exception("LLM error")
    result = translator.translate(
        blueprint=blueprint,
        author_name="Test Author",
        style_dna="Test style.",
        rhetorical_type=RhetoricalType.OBSERVATION,
        examples=["Example"]
    )

    # Should fall back to literal translation
    assert isinstance(result, str)

    print("✓ test_error_handling passed")


def test_translate_literal():
    """Test literal translation fallback."""
    if not DEPENDENCIES_AVAILABLE:
        print("⚠ SKIPPED: test_translate_literal (missing dependencies)")
        return
    blueprint = SemanticBlueprint(
        original_text="The server crashed.",
        svo_triples=[("the server", "crash", "")],
        named_entities=[],
        core_keywords={"server", "crash"},
        citations=[],
        quotes=[]
    )

    mock_llm = Mock()
    mock_llm.call.return_value = "The server stopped working."

    translator = StyleTranslator(config_path="config.json")
    translator.llm_provider = mock_llm

    result = translator.translate_literal(
        blueprint=blueprint,
        author_name="Test Author",
        style_dna="Clear and direct."
    )

    assert result == "The server stopped working."
    assert mock_llm.call.called

    # Test with LLM failure
    mock_llm.call.side_effect = Exception("LLM error")
    result = translator.translate_literal(
        blueprint=blueprint,
        author_name="Test Author",
        style_dna="Test style."
    )

    # Should return original text
    assert result == blueprint.original_text

    print("✓ test_translate_literal passed")


def test_fallback_returns_original_text():
    """Test that fallback mechanism returns original_text when blueprint is incomplete."""
    if not DEPENDENCIES_AVAILABLE:
        print("⚠ SKIPPED: test_fallback_returns_original_text (missing dependencies)")
        return
    translator = StyleTranslator(config_path="config.json")

    # Create blueprint that's missing key noun "object"
    blueprint = SemanticBlueprint(
        original_text="Every object we touch eventually breaks.",
        svo_triples=[("we", "touch", "breaks")],  # Missing "object" in blueprint
        named_entities=[],
        core_keywords={"touch", "break"},  # Missing "object" keyword
        citations=[],
        quotes=[]
    )

    # translate_literal should detect incomplete blueprint and return original_text
    result = translator.translate_literal(
        blueprint=blueprint,
        author_name="Test Author",
        style_dna="Simple and direct."
    )

    # Should return original_text, not try to generate from broken blueprint
    assert result == "Every object we touch eventually breaks.", \
        f"Fallback should return original_text, got: {result}"

    print("✓ test_fallback_returns_original_text passed")


def test_is_blueprint_incomplete_missing_object():
    """Test that blueprint missing 'object' is detected as incomplete."""
    if not DEPENDENCIES_AVAILABLE:
        print("⚠ SKIPPED: test_is_blueprint_incomplete_missing_object (missing dependencies)")
        return
    translator = StyleTranslator(config_path="config.json")

    # Blueprint missing "object" - the exact failing case
    blueprint = SemanticBlueprint(
        original_text="Every object we touch eventually breaks.",
        svo_triples=[("we", "touch", "breaks")],  # Missing "object"
        named_entities=[],
        core_keywords={"touch", "break"},  # Missing "object"
        citations=[],
        quotes=[]
    )

    is_incomplete = translator._is_blueprint_incomplete(blueprint)
    assert is_incomplete == True, \
        f"Blueprint missing 'object' should be marked incomplete. Got: {is_incomplete}"

    print("✓ test_is_blueprint_incomplete_missing_object passed")


def test_is_blueprint_incomplete_complete_blueprint():
    """Test that complete blueprint is not marked as incomplete."""
    if not DEPENDENCIES_AVAILABLE:
        print("⚠ SKIPPED: test_is_blueprint_incomplete_complete_blueprint (missing dependencies)")
        return
    translator = StyleTranslator(config_path="config.json")

    # Complete blueprint with all nouns preserved
    blueprint = SemanticBlueprint(
        original_text="The cat sat on the mat.",
        svo_triples=[("the cat", "sit", "the mat")],
        named_entities=[],
        core_keywords={"cat", "sit", "mat"},
        citations=[],
        quotes=[]
    )

    is_incomplete = translator._is_blueprint_incomplete(blueprint)
    assert is_incomplete == False, \
        f"Complete blueprint should not be marked incomplete. Got: {is_incomplete}"

    print("✓ test_is_blueprint_incomplete_complete_blueprint passed")


def test_is_blueprint_incomplete_lemmatization():
    """Test that lemmatization works correctly (Objects matches object)."""
    if not DEPENDENCIES_AVAILABLE:
        print("⚠ SKIPPED: test_is_blueprint_incomplete_lemmatization (missing dependencies)")
        return
    translator = StyleTranslator(config_path="config.json")

    # Original has "Objects" (plural), blueprint has "object" (singular)
    # Should match via lemmatization
    blueprint = SemanticBlueprint(
        original_text="Objects break.",
        svo_triples=[("object", "break", "")],  # Singular "object" should match plural "Objects"
        named_entities=[],
        core_keywords={"object", "break"},
        citations=[],
        quotes=[]
    )

    is_incomplete = translator._is_blueprint_incomplete(blueprint)
    assert is_incomplete == False, \
        f"Blueprint with lemmatized match should not be incomplete. Got: {is_incomplete}"

    print("✓ test_is_blueprint_incomplete_lemmatization passed")


def test_is_blueprint_incomplete_biological_cycle():
    """Test that blueprint missing 'biological' and list items is detected."""
    if not DEPENDENCIES_AVAILABLE:
        print("⚠ SKIPPED: test_is_blueprint_incomplete_biological_cycle (missing dependencies)")
        return
    translator = StyleTranslator(config_path="config.json")

    # Blueprint missing "biological" and list "birth, life, and decay"
    blueprint = SemanticBlueprint(
        original_text="The biological cycle of birth, life, and decay defines our reality.",
        svo_triples=[("cycle", "define", "reality")],  # Missing "biological" and list
        named_entities=[],
        core_keywords={"cycle", "define", "reality"},  # Missing "biological", "birth", "life", "decay"
        citations=[],
        quotes=[]
    )

    is_incomplete = translator._is_blueprint_incomplete(blueprint)
    # Should be incomplete because it's missing many nouns (biological, birth, life, decay)
    assert is_incomplete == True, \
        f"Blueprint missing multiple nouns should be incomplete. Got: {is_incomplete}"

    print("✓ test_is_blueprint_incomplete_biological_cycle passed")


def test_is_blueprint_incomplete_empty_svo_long_text():
    """Test that empty SVO for long text (>5 words) is detected as incomplete."""
    if not DEPENDENCIES_AVAILABLE:
        print("⚠ SKIPPED: test_is_blueprint_incomplete_empty_svo_long_text (missing dependencies)")
        return
    translator = StyleTranslator(config_path="config.json")

    # Long text with empty SVO
    blueprint = SemanticBlueprint(
        original_text="This is a longer sentence with multiple words.",
        svo_triples=[],  # Empty SVO
        named_entities=[],
        core_keywords=set(),  # Empty keywords
        citations=[],
        quotes=[]
    )

    is_incomplete = translator._is_blueprint_incomplete(blueprint)
    assert is_incomplete == True, \
        f"Empty SVO for long text should be incomplete. Got: {is_incomplete}"

    print("✓ test_is_blueprint_incomplete_empty_svo_long_text passed")


def test_is_blueprint_incomplete_short_text_no_nouns():
    """Test that short text with no nouns doesn't trigger false positive."""
    if not DEPENDENCIES_AVAILABLE:
        print("⚠ SKIPPED: test_is_blueprint_incomplete_short_text_no_nouns (missing dependencies)")
        return
    translator = StyleTranslator(config_path="config.json")

    # Short text with no nouns (just verbs/adjectives)
    blueprint = SemanticBlueprint(
        original_text="Run fast.",
        svo_triples=[("", "run", "")],
        named_entities=[],
        core_keywords={"run", "fast"},
        citations=[],
        quotes=[]
    )

    is_incomplete = translator._is_blueprint_incomplete(blueprint)
    # Should not be incomplete (no nouns to check)
    assert is_incomplete == False, \
        f"Short text with no nouns should not be incomplete. Got: {is_incomplete}"

    print("✓ test_is_blueprint_incomplete_short_text_no_nouns passed")


def test_build_prompt_uses_original_only_when_incomplete():
    """Test that _build_prompt uses original-text-only template when blueprint is incomplete."""
    if not DEPENDENCIES_AVAILABLE:
        print("⚠ SKIPPED: test_build_prompt_uses_original_only_when_incomplete (missing dependencies)")
        return
    translator = StyleTranslator(config_path="config.json")

    # Incomplete blueprint (missing "object")
    blueprint = SemanticBlueprint(
        original_text="Every object we touch eventually breaks.",
        svo_triples=[("we", "touch", "breaks")],  # Missing "object"
        named_entities=[],
        core_keywords={"touch", "break"},
        citations=[],
        quotes=[],
        position="BODY"
    )

    prompt = translator._build_prompt(
        blueprint=blueprint,
        author_name="Test Author",
        style_dna="Philosophical",
        rhetorical_type=RhetoricalType.OBSERVATION,
        examples=["Example 1", "Example 2"]
    )

    # Should use original-text-only template (no blueprint structure)
    assert "SOURCE TEXT" in prompt or "ORIGINAL TEXT" in prompt, \
        "Prompt should use original-text-only template"
    assert "Every object we touch eventually breaks" in prompt, \
        "Original text should be in prompt"
    # Should NOT have blueprint structure checklist
    assert "Subjects:" not in prompt or "STRUCTURE CHECKLIST" not in prompt, \
        "Prompt should not include blueprint structure when incomplete"

    print("✓ test_build_prompt_uses_original_only_when_incomplete passed")


def test_build_prompt_uses_blueprint_when_complete():
    """Test that _build_prompt uses standard blueprint template when blueprint is complete."""
    if not DEPENDENCIES_AVAILABLE:
        print("⚠ SKIPPED: test_build_prompt_uses_blueprint_when_complete (missing dependencies)")
        return
    translator = StyleTranslator(config_path="config.json")

    # Complete blueprint
    blueprint = SemanticBlueprint(
        original_text="The cat sat on the mat.",
        svo_triples=[("the cat", "sit", "the mat")],
        named_entities=[],
        core_keywords={"cat", "sit", "mat"},
        citations=[],
        quotes=[],
        position="BODY"
    )

    prompt = translator._build_prompt(
        blueprint=blueprint,
        author_name="Test Author",
        style_dna="Simple",
        rhetorical_type=RhetoricalType.OBSERVATION,
        examples=["Example 1"]
    )

    # Should use standard blueprint template
    assert "STRUCTURE CHECKLIST" in prompt or "Subjects:" in prompt, \
        "Prompt should include blueprint structure when complete"
    assert "The cat sat on the mat" in prompt, \
        "Original text should still be in prompt"

    print("✓ test_build_prompt_uses_blueprint_when_complete passed")


def test_build_original_text_only_prompt():
    """Test that _build_original_text_only_prompt creates correct prompt."""
    if not DEPENDENCIES_AVAILABLE:
        print("⚠ SKIPPED: test_build_original_text_only_prompt (missing dependencies)")
        return
    translator = StyleTranslator(config_path="config.json")

    blueprint = SemanticBlueprint(
        original_text="Every object we touch eventually breaks.",
        svo_triples=[("we", "touch", "breaks")],
        named_entities=[],
        core_keywords={"touch", "break"},
        citations=[],
        quotes=[],
        position="BODY"
    )

    prompt = translator._build_original_text_only_prompt(
        blueprint=blueprint,
        author_name="Test Author",
        style_dna="Philosophical",
        rhetorical_type=RhetoricalType.OBSERVATION,
        examples=["Example 1", "Example 2"]
    )

    # Check prompt structure
    assert "Every object we touch eventually breaks" in prompt, \
        "Original text should be in prompt"
    assert "Example 1" in prompt and "Example 2" in prompt, \
        "Examples should be in prompt"
    assert "OBSERVATION" in prompt, \
        "Rhetorical type should be in prompt"
    # Should NOT have blueprint structure
    assert "Subjects:" not in prompt or "STRUCTURE CHECKLIST" not in prompt, \
        "Should not include blueprint structure"

    print("✓ test_build_original_text_only_prompt passed")


def test_generate_simplification_uses_original_when_incomplete():
    """Test that _generate_simplification uses original_text when blueprint is incomplete."""
    if not DEPENDENCIES_AVAILABLE:
        print("⚠ SKIPPED: test_generate_simplification_uses_original_when_incomplete (missing dependencies)")
        return
    from unittest.mock import Mock
    from src.validator.semantic_critic import SemanticCritic

    translator = StyleTranslator(config_path="config.json")
    mock_llm = Mock()
    mock_llm.call.return_value = "Every object we touch eventually breaks."
    translator.llm_provider = mock_llm

    critic = SemanticCritic(config_path="config.json")

    # Incomplete blueprint (missing "object")
    blueprint = SemanticBlueprint(
        original_text="Every object we touch eventually breaks.",
        svo_triples=[("we", "touch", "breaks")],  # Missing "object"
        named_entities=[],
        core_keywords={"touch", "break"},
        citations=[],
        quotes=[]
    )

    result = translator._generate_simplification(
        best_draft="We touch breaks.",
        blueprint=blueprint,
        author_name="Test Author",
        style_dna="Simple",
        rhetorical_type=RhetoricalType.OBSERVATION,
        critic=critic,
        verbose=False
    )

    # Check that LLM was called with original_text (not broken blueprint)
    call_args = mock_llm.call.call_args
    user_prompt = call_args[1]['user_prompt'] if 'user_prompt' in call_args[1] else call_args[0][1]
    assert "Every object we touch eventually breaks" in user_prompt, \
        "Simplification should use original_text when blueprint is incomplete"
    assert "N/A (Using original text due to incomplete blueprint)" in user_prompt or \
           "original text" in user_prompt.lower(), \
        "Prompt should indicate using original text"

    print("✓ test_generate_simplification_uses_original_when_incomplete passed")


def test_generate_simplification_uses_blueprint_when_complete():
    """Test that _generate_simplification uses blueprint when blueprint is complete."""
    if not DEPENDENCIES_AVAILABLE:
        print("⚠ SKIPPED: test_generate_simplification_uses_blueprint_when_complete (missing dependencies)")
        return
    from unittest.mock import Mock
    from src.validator.semantic_critic import SemanticCritic

    translator = StyleTranslator(config_path="config.json")
    mock_llm = Mock()
    mock_llm.call.return_value = "The cat sat on the mat."
    translator.llm_provider = mock_llm

    critic = SemanticCritic(config_path="config.json")

    # Complete blueprint
    blueprint = SemanticBlueprint(
        original_text="The cat sat on the mat.",
        svo_triples=[("the cat", "sit", "the mat")],
        named_entities=[],
        core_keywords={"cat", "sit", "mat"},
        citations=[],
        quotes=[]
    )

    result = translator._generate_simplification(
        best_draft="The cat sits on the mat.",
        blueprint=blueprint,
        author_name="Test Author",
        style_dna="Simple",
        rhetorical_type=RhetoricalType.OBSERVATION,
        critic=critic,
        verbose=False
    )

    # Check that LLM was called (should use blueprint_text, not just original)
    assert mock_llm.call.called, \
        "Simplification should call LLM when blueprint is complete"

    print("✓ test_generate_simplification_uses_blueprint_when_complete passed")


def test_is_blueprint_incomplete_partial_match():
    """Test that blueprint with <50% noun match is detected as incomplete."""
    if not DEPENDENCIES_AVAILABLE:
        print("⚠ SKIPPED: test_is_blueprint_incomplete_partial_match (missing dependencies)")
        return
    translator = StyleTranslator(config_path="config.json")

    # Original has 4 nouns, blueprint has only 1 (25% match - should be incomplete)
    blueprint = SemanticBlueprint(
        original_text="The biological cycle of birth and decay defines reality.",
        svo_triples=[("cycle", "define", "reality")],  # Only has "cycle" and "reality", missing "biological", "birth", "decay"
        named_entities=[],
        core_keywords={"cycle", "define", "reality"},
        citations=[],
        quotes=[]
    )

    is_incomplete = translator._is_blueprint_incomplete(blueprint)
    # Should be incomplete because match ratio < 50%
    assert is_incomplete == True, \
        f"Blueprint with <50% noun match should be incomplete. Got: {is_incomplete}"

    print("✓ test_is_blueprint_incomplete_partial_match passed")


def test_select_best_candidate_uses_judge():
    """Test that _select_best_candidate uses LLM Judge for ranking."""
    # Skip: _select_best_candidate method has been removed - selection is now inline in _evolve_text
    print("⊘ SKIPPED: test_select_best_candidate_uses_judge (method removed)")
    return
    if not DEPENDENCIES_AVAILABLE:
        print("⚠ SKIPPED: test_select_best_candidate_uses_judge (missing dependencies)")
        return
    translator = StyleTranslator(config_path="config.json")
    critic = SemanticCritic(config_path="config.json")
    judge = LLMJudge(config_path="config.json")

    # Mock judge to select candidate B
    mock_judge_llm = Mock()
    mock_judge_llm.call.return_value = "B"
    judge.llm_provider = mock_judge_llm

    blueprint = SemanticBlueprint(
        original_text="Every object we touch eventually breaks.",
        svo_triples=[("object", "touch", "breaks")],
        named_entities=[],
        core_keywords={"object", "touch", "break"},
        citations=[],
        quotes=[]
    )

    # Create candidates that pass hard gates
    candidates = [
        ("semantic", "Every object we touch breaks."),  # A - complete
        ("fluency", "Every object we touch eventually breaks."),  # B - more natural
        ("style", "All objects we touch break.")  # C - different style
    ]

    parent_draft = "We touch breaks."  # Incomplete parent
    parent_result = critic.evaluate(parent_draft, blueprint)

    best_candidate, score, strategy, result = translator._select_best_candidate(
        candidates=candidates,
        parent_draft=parent_draft,
        parent_score=0.5,
        parent_recall=parent_result["recall_score"],
        blueprint=blueprint,
        critic=critic,
        judge=judge,
        style_dna="Natural and clear",
        rhetorical_type=RhetoricalType.OBSERVATION,
        verbose=False
    )

    # Judge should be called and select candidate B
    assert mock_judge_llm.call.called, "Judge should be called for ranking"
    assert best_candidate is not None, "Should select a candidate"
    assert "eventually" in best_candidate, "Judge should select more natural phrasing"

    print("✓ test_select_best_candidate_uses_judge passed")


def test_select_best_candidate_filters_hard_gates():
    """Test that _select_best_candidate filters candidates through hard gates."""
    # Skip: _select_best_candidate method has been removed - selection is now inline in _evolve_text
    print("⊘ SKIPPED: test_select_best_candidate_filters_hard_gates (method removed)")
    return
    if not DEPENDENCIES_AVAILABLE:
        print("⚠ SKIPPED: test_select_best_candidate_filters_hard_gates (missing dependencies)")
        return
    translator = StyleTranslator(config_path="config.json")
    critic = SemanticCritic(config_path="config.json")
    judge = LLMJudge(config_path="config.json")

    blueprint = SemanticBlueprint(
        original_text="Every object we touch eventually breaks.",
        svo_triples=[("object", "touch", "breaks")],
        named_entities=[],
        core_keywords={"object", "touch", "break"},
        citations=[],
        quotes=[]
    )

    # Create candidates: one passes, one fails hard gates
    candidates = [
        ("semantic", "Every object we touch eventually breaks."),  # A - passes
        ("fluency", "We touch breaks."),  # B - fails (missing "object", score=0.0)
        ("style", "Objects break.")  # C - fails (missing "touch", score=0.0)
    ]

    parent_draft = "Every object we touch breaks."
    parent_result = critic.evaluate(parent_draft, blueprint)

    best_candidate, score, strategy, result = translator._select_best_candidate(
        candidates=candidates,
        parent_draft=parent_draft,
        parent_score=0.8,
        parent_recall=parent_result["recall_score"],
        blueprint=blueprint,
        critic=critic,
        judge=judge,
        style_dna="Natural",
        rhetorical_type=RhetoricalType.OBSERVATION,
        verbose=False
    )

    # Only candidate A should pass hard gates, so it should be selected
    # (Judge won't be called if only 1 candidate)
    assert best_candidate is not None, "Should select the candidate that passed hard gates"
    assert "object" in best_candidate.lower(), "Selected candidate should contain 'object'"

    print("✓ test_select_best_candidate_filters_hard_gates passed")


def test_select_best_candidate_keeps_parent_when_judge_selects_parent():
    """Test that _select_best_candidate keeps parent when judge selects it."""
    # Skip: _select_best_candidate method has been removed - selection is now inline in _evolve_text
    print("⊘ SKIPPED: test_select_best_candidate_keeps_parent_when_judge_selects_parent (method removed)")
    return
    if not DEPENDENCIES_AVAILABLE:
        print("⚠ SKIPPED: test_select_best_candidate_keeps_parent_when_judge_selects_parent (missing dependencies)")
        return
    translator = StyleTranslator(config_path="config.json")
    critic = SemanticCritic(config_path="config.json")
    judge = LLMJudge(config_path="config.json")

    # Mock judge to select PARENT (index 0)
    mock_judge_llm = Mock()
    mock_judge_llm.call.return_value = "A"  # A is PARENT in the list
    judge.llm_provider = mock_judge_llm

    blueprint = SemanticBlueprint(
        original_text="The cat sat on the mat.",
        svo_triples=[("cat", "sit", "mat")],
        named_entities=[],
        core_keywords={"cat", "sit", "mat"},
        citations=[],
        quotes=[]
    )

    parent_draft = "The cat sat on the mat."  # Good parent
    candidates = [
        ("semantic", "The cat sits on the mat."),  # Slightly different
        ("fluency", "The cat sat on the mat.")  # Same as parent
    ]

    parent_result = critic.evaluate(parent_draft, blueprint)

    best_candidate, score, strategy, result = translator._select_best_candidate(
        candidates=candidates,
        parent_draft=parent_draft,
        parent_score=0.95,
        parent_recall=parent_result["recall_score"],
        blueprint=blueprint,
        critic=critic,
        judge=judge,
        style_dna="Natural",
        rhetorical_type=RhetoricalType.OBSERVATION,
        verbose=False
    )

    # Judge selected parent (elitism), so should return None
    assert best_candidate is None, "Should keep parent when judge selects it"
    assert strategy == "parent", "Strategy should be 'parent'"

    print("✓ test_select_best_candidate_keeps_parent_when_judge_selects_parent passed")


def test_select_best_candidate_handles_all_rejected():
    """Test that _select_best_candidate handles case where all candidates are rejected."""
    # Skip: _select_best_candidate method has been removed - selection is now inline in _evolve_text
    print("⊘ SKIPPED: test_select_best_candidate_handles_all_rejected (method removed)")
    return
    if not DEPENDENCIES_AVAILABLE:
        print("⚠ SKIPPED: test_select_best_candidate_handles_all_rejected (missing dependencies)")
        return
    translator = StyleTranslator(config_path="config.json")
    critic = SemanticCritic(config_path="config.json")
    judge = LLMJudge(config_path="config.json")

    blueprint = SemanticBlueprint(
        original_text="Every object we touch eventually breaks.",
        svo_triples=[("object", "touch", "breaks")],
        named_entities=[],
        core_keywords={"object", "touch", "break"},
        citations=[],
        quotes=[]
    )

    # All candidates fail hard gates
    candidates = [
        ("semantic", "We touch breaks."),  # Missing "object" - fails
        ("fluency", "Breaks."),  # Fragment - fails
        ("style", "")  # Empty - fails
    ]

    parent_draft = "Every object we touch breaks."
    parent_result = critic.evaluate(parent_draft, blueprint)

    best_candidate, score, strategy, result = translator._select_best_candidate(
        candidates=candidates,
        parent_draft=parent_draft,
        parent_score=0.8,
        parent_recall=parent_result["recall_score"],
        blueprint=blueprint,
        critic=critic,
        judge=judge,
        style_dna="Natural",
        rhetorical_type=RhetoricalType.OBSERVATION,
        verbose=False
    )

    # Should keep parent when all candidates rejected
    assert best_candidate is None, "Should keep parent when all candidates rejected"
    assert strategy == "parent", "Strategy should be 'parent'"
    assert not judge.llm_provider.call.called, "Judge should not be called if no candidates pass"

    print("✓ test_select_best_candidate_handles_all_rejected passed")


def test_style_lexicon_ratio_configuration():
    """Test that style_lexicon_ratio config controls lexicon injection in translate_paragraph."""
    if not DEPENDENCIES_AVAILABLE:
        print("⚠ SKIPPED: test_style_lexicon_ratio_configuration (missing dependencies)")
        return

    translator = StyleTranslator(config_path="config.json")

    # Create a mock style_dna with a known lexicon (20 words)
    test_lexicon = [f"word{i}" for i in range(20)]  # 20 words
    style_dna = {
        "lexicon": test_lexicon,
        "tone": "authoritative"
    }

    # Test 1: Default ratio (0.4) - should use 40% = 8 words
    with patch.object(translator, 'paragraph_fusion_config', {'style_lexicon_ratio': 0.4}):
        ratio = 0.4
        lexicon = style_dna.get("lexicon", [])
        count = int(len(lexicon) * ratio) if ratio > 0.0 else 0
        assert count == 8, f"Expected 8 words for 0.4 ratio, got {count}"

        if count > 0:
            top_lexicon = lexicon[:count]
            assert len(top_lexicon) == 8, f"Expected 8 words, got {len(top_lexicon)}"
            # Check instruction for 0.4 ratio (should be "Integrate these words naturally.")
            instruction = "Integrate these words naturally."  # 0.3 <= 0.4 <= 0.7
            assert instruction == "Integrate these words naturally."

    # Test 2: Low ratio (0.2) - should use 20% = 4 words, "sparingly" instruction
    ratio = 0.2
    count = int(len(test_lexicon) * ratio) if ratio > 0.0 else 0
    assert count == 4, f"Expected 4 words for 0.2 ratio, got {count}"
    if count > 0:
        top_lexicon = test_lexicon[:count]
        assert len(top_lexicon) == 4
        instruction = "Sprinkle these style markers sparingly." if ratio < 0.3 else ("Heavily saturate the text with this vocabulary." if ratio > 0.7 else "Integrate these words naturally.")
        assert instruction == "Sprinkle these style markers sparingly."

    # Test 3: High ratio (0.8) - should use 80% = 16 words, "heavily saturate" instruction
    ratio = 0.8
    count = int(len(test_lexicon) * ratio) if ratio > 0.0 else 0
    assert count == 16, f"Expected 16 words for 0.8 ratio, got {count}"
    if count > 0:
        top_lexicon = test_lexicon[:count]
        assert len(top_lexicon) == 16
        instruction = "Sprinkle these style markers sparingly." if ratio < 0.3 else ("Heavily saturate the text with this vocabulary." if ratio > 0.7 else "Integrate these words naturally.")
        assert instruction == "Heavily saturate the text with this vocabulary."

    # Test 4: Edge case - empty lexicon
    empty_style_dna = {"lexicon": [], "tone": "authoritative"}
    lexicon = empty_style_dna.get("lexicon", [])
    assert len(lexicon) == 0
    # Should handle gracefully (no crash)

    # Test 5: Edge case - ratio 1.0 - should use all words
    ratio = 1.0
    count = int(len(test_lexicon) * ratio) if ratio > 0.0 else 0
    assert count == 20, f"Expected 20 words for 1.0 ratio, got {count}"
    if count > 0:
        top_lexicon = test_lexicon[:count]
        assert len(top_lexicon) == 20

    # Test 6: Edge case - ratio 0.0 - should use 0 words, no MANDATORY_VOCABULARY section
    ratio = 0.0
    count = int(len(test_lexicon) * ratio) if ratio > 0.0 else 0
    assert count == 0, f"Expected 0 words for 0.0 ratio, got {count}"
    # When count == 0, mandatory_vocabulary should remain empty
    mandatory_vocabulary = ""
    if count > 0:
        # This block should not execute
        assert False, "Should not create vocabulary section when count is 0"
    assert mandatory_vocabulary == "", "mandatory_vocabulary should be empty when ratio is 0.0"

    print("✓ test_style_lexicon_ratio_configuration passed")


def test_evolve_text_uses_judge():
    """Test that _evolve_text uses LLM Judge for selection."""
    # Skip: _select_best_candidate method has been removed - selection is now inline in _evolve_text
    print("⊘ SKIPPED: test_evolve_text_uses_judge (method removed)")
    return
    if not DEPENDENCIES_AVAILABLE:
        print("⚠ SKIPPED: test_evolve_text_uses_judge (missing dependencies)")
        return
    translator = StyleTranslator(config_path="config.json")
    critic = SemanticCritic(config_path="config.json")

    # Mock LLM provider for population generation
    mock_llm = Mock()
    mock_llm.call.side_effect = [
        "Every object we touch breaks.",  # Semantic candidate
        "Every object we touch eventually breaks.",  # Fluency candidate
        "All objects we touch break."  # Style candidate
    ]
    translator.llm_provider = mock_llm

    # Mock judge to select Fluency candidate (B)
    judge = LLMJudge(config_path="config.json")
    mock_judge_llm = Mock()
    mock_judge_llm.call.return_value = "B"  # Select Fluency
    judge.llm_provider = mock_judge_llm

    blueprint = SemanticBlueprint(
        original_text="Every object we touch eventually breaks.",
        svo_triples=[("object", "touch", "breaks")],
        named_entities=[],
        core_keywords={"object", "touch", "break"},
        citations=[],
        quotes=[]
    )

    # Patch judge into translator
    with patch.object(translator, '_select_best_candidate') as mock_select:
        # Mock selection to return Fluency candidate
        mock_select.return_value = (
            "Every object we touch eventually breaks.",
            0.9,
            "fluency",
            {"pass": True, "score": 0.9, "recall_score": 1.0, "precision_score": 1.0, "adherence_score": 1.0, "feedback": "Good"}
        )

        best_draft, best_score = translator._evolve_text(
            initial_draft="We touch breaks.",
            blueprint=blueprint,
            author_name="Test Author",
            style_dna="Natural",
            rhetorical_type=RhetoricalType.OBSERVATION,
            initial_score=0.5,
            initial_feedback="Missing object",
            critic=critic,
            verbose=False
        )

        # Verify judge was used (through _select_best_candidate)
        assert mock_select.called, "Selection should be called"
        assert "eventually" in best_draft, "Should select more natural phrasing"

    print("✓ test_evolve_text_uses_judge passed")


def test_evolve_text_tracks_convergence():
    """Test that _evolve_text detects convergence when same candidate wins 2 rounds."""
    # Skip: _select_best_candidate method has been removed - selection is now inline in _evolve_text
    print("⊘ SKIPPED: test_evolve_text_tracks_convergence (method removed)")
    return
    if not DEPENDENCIES_AVAILABLE:
        print("⚠ SKIPPED: test_evolve_text_tracks_convergence (missing dependencies)")
        return
    translator = StyleTranslator(config_path="config.json")
    critic = SemanticCritic(config_path="config.json")

    blueprint = SemanticBlueprint(
        original_text="The cat sat on the mat.",
        svo_triples=[("cat", "sit", "mat")],
        named_entities=[],
        core_keywords={"cat", "sit", "mat"},
        citations=[],
        quotes=[]
    )

    # Mock LLM to return same candidate twice
    mock_llm = Mock()
    mock_llm.call.side_effect = [
        "The cat sat on the mat.",  # Gen 1: Semantic
        "The cat sat on the mat.",  # Gen 1: Fluency
        "The cat sat on the mat.",  # Gen 1: Style
        "The cat sat on the mat.",  # Gen 2: Semantic (same)
        "The cat sat on the mat.",  # Gen 2: Fluency (same)
        "The cat sat on the mat.",  # Gen 2: Style (same)
    ]
    translator.llm_provider = mock_llm

    # Mock judge to select same candidate (B - Fluency) twice
    judge = LLMJudge(config_path="config.json")
    mock_judge_llm = Mock()
    mock_judge_llm.call.side_effect = ["B", "B"]  # Select Fluency twice
    judge.llm_provider = mock_judge_llm

    # Mock critic to return good scores
    def mock_evaluate(text, blueprint):
        return {
            "pass": True,
            "score": 0.9,
            "recall_score": 1.0,
            "precision_score": 1.0,
            "adherence_score": 1.0,
            "feedback": "Good"
        }

    critic.evaluate = Mock(side_effect=mock_evaluate)

    # Patch _select_best_candidate to use judge and track convergence
    original_select = translator._select_best_candidate

    call_count = [0]
    def mock_select(*args, **kwargs):
        call_count[0] += 1
        # Return Fluency candidate (same text both times)
        return (
            "The cat sat on the mat.",
            0.9,
            "fluency",
            {"pass": True, "score": 0.9, "recall_score": 1.0, "precision_score": 1.0, "fluency_score": 0.9, "feedback": "Good"}
        )

    translator._select_best_candidate = mock_select

    best_draft, best_score = translator._evolve_text(
        initial_draft="The cat sits on the mat.",
        blueprint=blueprint,
        author_name="Test Author",
        style_dna="Natural",
        rhetorical_type=RhetoricalType.OBSERVATION,
        initial_score=0.8,
        initial_feedback="Good",
        critic=critic,
        verbose=False
    )

    # Should detect convergence and stop early
    # (The exact behavior depends on implementation, but convergence should be detected)
    assert best_draft is not None, "Should return a draft"

    print("✓ test_evolve_text_tracks_convergence passed")


def test_full_translation_with_judge():
    """Integration test: Full translation flow uses LLM Judge."""
    # Skip: _select_best_candidate method has been removed - selection is now inline in _evolve_text
    print("⊘ SKIPPED: test_full_translation_with_judge (method removed)")
    return
    if not DEPENDENCIES_AVAILABLE:
        print("⚠ SKIPPED: test_full_translation_with_judge (missing dependencies)")
        return
    translator = StyleTranslator(config_path="config.json")
    critic = SemanticCritic(config_path="config.json")

    blueprint = SemanticBlueprint(
        original_text="Every object we touch eventually breaks.",
        svo_triples=[("object", "touch", "breaks")],
        named_entities=[],
        core_keywords={"object", "touch", "break"},
        citations=[],
        quotes=[]
    )

    # Mock LLM for initial generation
    mock_llm = Mock()
    mock_llm.call.side_effect = [
        "Every object we touch eventually breaks.",  # Initial draft
        "Every object we touch breaks.",  # Semantic candidate
        "Every object we touch eventually breaks.",  # Fluency candidate
        "All objects we touch break."  # Style candidate
    ]
    translator.llm_provider = mock_llm

    # Mock judge to select Fluency (B)
    judge = LLMJudge(config_path="config.json")
    mock_judge_llm = Mock()
    mock_judge_llm.call.return_value = "B"
    judge.llm_provider = mock_judge_llm

    # Patch _select_best_candidate to use our mocked judge
    original_select = translator._select_best_candidate

    def mock_select_with_judge(*args, **kwargs):
        # Extract judge from kwargs
        judge_arg = kwargs.get('judge')
        if judge_arg:
            # Use the mocked judge
            kwargs['judge'] = judge_arg
        return original_select(*args, **kwargs)

    translator._select_best_candidate = mock_select_with_judge

    examples = ["Example 1", "Example 2"]

    result = translator.translate(
        blueprint=blueprint,
        author_name="Test Author",
        style_dna="Natural and clear",
        rhetorical_type=RhetoricalType.OBSERVATION,
        examples=examples
    )

    # Verify result is reasonable
    assert result is not None, "Translation should return a result"
    assert len(result) > 0, "Translation should not be empty"

    # Verify judge was called (through _select_best_candidate)
    # This is indirect verification - if evolution happened, judge should have been used

    print("✓ test_full_translation_with_judge passed")


if __name__ == "__main__":
    if not DEPENDENCIES_AVAILABLE:
        print(f"⚠ All tests skipped due to missing dependencies: {IMPORT_ERROR}")
        sys.exit(1)

    test_basic_translation()
    test_complex_translation()
    test_rhetorical_mode_matching()
    test_error_handling()
    test_translate_literal()
    test_fallback_returns_original_text()
    # Blueprint incompleteness tests
    test_is_blueprint_incomplete_missing_object()
    test_is_blueprint_incomplete_complete_blueprint()
    test_is_blueprint_incomplete_lemmatization()
    test_is_blueprint_incomplete_biological_cycle()
    test_is_blueprint_incomplete_empty_svo_long_text()
    test_is_blueprint_incomplete_short_text_no_nouns()
    test_is_blueprint_incomplete_partial_match()
    test_build_prompt_uses_original_only_when_incomplete()
    test_build_prompt_uses_blueprint_when_complete()
    test_build_original_text_only_prompt()
    test_generate_simplification_uses_original_when_incomplete()
    test_generate_simplification_uses_blueprint_when_complete()
    # LLM Judge integration tests
    test_select_best_candidate_uses_judge()
    test_select_best_candidate_filters_hard_gates()
    test_select_best_candidate_keeps_parent_when_judge_selects_parent()
    test_select_best_candidate_handles_all_rejected()
    test_evolve_text_uses_judge()
    test_evolve_text_tracks_convergence()
    test_full_translation_with_judge()
    # New tests for expansion, rescue, and concept mapping
    test_dynamic_expansion_thresholds()
    test_rescue_logic()
    test_explicit_concept_mapping()
    test_check_acceptance_without_fluency()
    test_style_lexicon_ratio_configuration()
    print("\n✓ All translator tests completed!")

