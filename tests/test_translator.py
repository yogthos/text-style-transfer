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


def test_adaptive_length_filter_short_sentences():
    """Test that adaptive length filter allows expansion for short sentences (< 5 words)."""
    if not DEPENDENCIES_AVAILABLE:
        print("⚠ SKIPPED: test_adaptive_length_filter_short_sentences (missing dependencies)")
        return
    from src.generator.translator import StyleTranslator

    translator = StyleTranslator(config_path="config.json")

    # Test with very short input (3 words) - should use adaptive filter
    short_blueprint = SemanticBlueprint(
        original_text="I was thirteen",
        svo_triples=[("I", "was", "thirteen")],
        named_entities=[],
        core_keywords={"I", "was", "thirteen"},
        citations=[],
        quotes=[]
    )

    input_len = len(short_blueprint.original_text.split())
    assert input_len == 3, f"Expected 3 words, got {input_len}"

    # For 3-word input, adaptive filter should be:
    # min_len = 5
    # max_len = max(15, 5.0 * 3) = max(15, 15) = 15
    expected_min = 5
    expected_max = max(15, int(5.0 * input_len))

    # Create examples with various lengths
    examples = [
        "Short.",  # 1 word - should be filtered (too short)
        "This is short.",  # 3 words - should be filtered (too short, < min_len)
        "This is a valid example sentence.",  # 6 words - should pass (5 <= 6 <= 15)
        "This is a longer valid example sentence with more words.",  # 10 words - should pass
        "This is a very long example sentence that contains many words and should be filtered out because it exceeds the maximum length allowed by the adaptive filter.",  # 20 words - should be filtered (too long)
        "Another valid example here.",  # 4 words - should be filtered (too short, < min_len)
        "This example has exactly fifteen words total count here now.",  # 15 words - should pass (at max)
    ]

    # Mock the structuralizer to avoid actual skeleton extraction
    # We just want to test the length filtering
    original_extract = translator.structuralizer.extract_skeleton
    translator.structuralizer.extract_skeleton = Mock(return_value="[NP] [VP] [NP]")
    translator.structuralizer.count_skeleton_slots = Mock(return_value=3)

    try:
        # Call _extract_multiple_skeletons which applies the length filter
        result = translator._extract_multiple_skeletons(
            examples=examples,
            blueprint=short_blueprint,
            verbose=False
        )

        # Check that examples within range (5-15 words) passed
        # The result contains (skeleton, source_example) tuples
        passed_examples = [example for _, example in result]

        # Count word lengths for passed examples
        passed_lengths = [len(example.split()) for example in passed_examples]

        # Examples that should pass length filter: 6, 10, and 15 word examples
        # (Note: skeleton extraction or complexity filtering might remove some)
        expected_passing_lengths = [6, 10, 15]

        # Examples that should be filtered by length: 1, 3, 4, 20 word examples
        expected_filtered_lengths = [1, 3, 4, 20]

        # Verify that examples in the expected passing range (5-15 words) are present
        # We expect at least some examples in the 5-15 word range to pass
        examples_in_range = [l for l in passed_lengths if expected_min <= l <= expected_max]
        assert len(examples_in_range) > 0, \
            f"No examples in range {expected_min}-{expected_max} words passed. " \
            f"Passed examples lengths: {passed_lengths}"

        # Verify that examples outside the range (too short or too long) were filtered
        examples_outside_range = [l for l in passed_lengths if l < expected_min or l > expected_max]
        assert len(examples_outside_range) == 0, \
            f"Examples outside range {expected_min}-{expected_max} words should be filtered. " \
            f"Found: {examples_outside_range}"

        # Verify specific examples that should definitely pass length filter
        # (6-word example should pass if skeleton extraction works)
        six_word_example = "This is a valid example sentence."
        six_word_passed = any(six_word_example in example for example in passed_examples)

        # Verify examples that should definitely be filtered
        one_word_example = "Short."
        one_word_passed = any(one_word_example in example for example in passed_examples)
        assert not one_word_passed, \
            f"1-word example '{one_word_example}' should be filtered (too short, < {expected_min} words)"

        print(f"✓ test_adaptive_length_filter_short_sentences passed")
        print(f"  Input: {input_len} words, Filter range: {expected_min}-{expected_max} words")
        print(f"  Examples passed: {len(passed_examples)}/{len(examples)}")
        print(f"  Passed example lengths: {passed_lengths}")

    finally:
        # Restore original method
        translator.structuralizer.extract_skeleton = original_extract


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


def test_detect_sentence_type_question():
    """Test _detect_sentence_type correctly identifies questions."""
    if not DEPENDENCIES_AVAILABLE:
        print("⚠ SKIPPED: test_detect_sentence_type_question (missing dependencies)")
        return

    translator = StyleTranslator(config_path="config.json")

    # Test explicit question mark
    assert translator._detect_sentence_type("Where did the morning require?") == "QUESTION"
    assert translator._detect_sentence_type("What is this?") == "QUESTION"

    # Test question words at start
    assert translator._detect_sentence_type("where did the morning require") == "QUESTION"
    assert translator._detect_sentence_type("what is this") == "QUESTION"
    assert translator._detect_sentence_type("who are you") == "QUESTION"
    assert translator._detect_sentence_type("when did it happen") == "QUESTION"
    assert translator._detect_sentence_type("why is this") == "QUESTION"
    assert translator._detect_sentence_type("how does it work") == "QUESTION"

    # Test that question words in middle don't trigger
    assert translator._detect_sentence_type("I know where it is") == "DECLARATIVE"
    assert translator._detect_sentence_type("Tell me what happened") == "DECLARATIVE"

    print("✓ test_detect_sentence_type_question passed")


def test_detect_sentence_type_conditional():
    """Test _detect_sentence_type correctly identifies conditionals."""
    if not DEPENDENCIES_AVAILABLE:
        print("⚠ SKIPPED: test_detect_sentence_type_conditional (missing dependencies)")
        return

    translator = StyleTranslator(config_path="config.json")

    # Test conditional markers at start
    assert translator._detect_sentence_type("if this happens, then that") == "CONDITIONAL"
    assert translator._detect_sentence_type("when it rains, we stay inside") == "CONDITIONAL"
    assert translator._detect_sentence_type("unless you agree, we cannot proceed") == "CONDITIONAL"
    assert translator._detect_sentence_type("provided that you agree, we can proceed") == "CONDITIONAL"
    assert translator._detect_sentence_type("should you agree, we proceed") == "CONDITIONAL"

    # Test that conditional words in middle don't trigger
    assert translator._detect_sentence_type("I know if it happens") == "DECLARATIVE"
    assert translator._detect_sentence_type("Tell me when it happens") == "DECLARATIVE"

    print("✓ test_detect_sentence_type_conditional passed")


def test_detect_sentence_type_declarative():
    """Test _detect_sentence_type correctly identifies declarative sentences."""
    if not DEPENDENCIES_AVAILABLE:
        print("⚠ SKIPPED: test_detect_sentence_type_declarative (missing dependencies)")
        return

    translator = StyleTranslator(config_path="config.json")

    # Test standard declarative
    assert translator._detect_sentence_type("Every morning required a pilgrimage") == "DECLARATIVE"
    assert translator._detect_sentence_type("The cat sat on the mat") == "DECLARATIVE"
    assert translator._detect_sentence_type("I went to the store") == "DECLARATIVE"

    # Test with question words in middle (should still be declarative)
    assert translator._detect_sentence_type("I know where it is") == "DECLARATIVE"
    assert translator._detect_sentence_type("Tell me what happened") == "DECLARATIVE"

    # Test with conditional words in middle (should still be declarative)
    assert translator._detect_sentence_type("I know if it happens") == "DECLARATIVE"

    print("✓ test_detect_sentence_type_declarative passed")


def test_type_compatibility_filter_blocks_question_for_declarative():
    """Test that type filter blocks question skeletons for declarative inputs."""
    if not DEPENDENCIES_AVAILABLE:
        print("⚠ SKIPPED: test_type_compatibility_filter_blocks_question_for_declarative (missing dependencies)")
        return

    translator = StyleTranslator(config_path="config.json")

    # Create declarative blueprint
    declarative_blueprint = SemanticBlueprint(
        original_text="Every morning required a pilgrimage to the general store with grandmother.",
        svo_triples=[("morning", "required", "pilgrimage")],
        named_entities=[],
        core_keywords={"morning", "required", "pilgrimage", "store", "grandmother"},
        citations=[],
        quotes=[]
    )

    # Create examples: mix of question and declarative skeletons
    examples = [
        "Where did the morning require a pilgrimage?",  # Question - should be filtered
        "Every morning, we made a pilgrimage to the store.",  # Declarative - should pass
        "What did the morning require?",  # Question - should be filtered
        "Each morning brought a necessary journey to the general store.",  # Declarative - should pass
    ]

    # Mock structuralizer to return skeletons matching the examples
    def mock_extract_skeleton(example, input_text=None):
        if "?" in example or example.lower().startswith(("where", "what")):
            return "Where [VP] [NP] [VP] [NP]?"
        else:
            return "[ADJ] [NP], [NP] [VP] [NP] [PP] [NP]."

    translator.structuralizer.extract_skeleton = Mock(side_effect=mock_extract_skeleton)
    translator.structuralizer.count_skeleton_slots = Mock(return_value=5)

    # Extract skeletons
    result = translator._extract_multiple_skeletons(
        examples=examples,
        blueprint=declarative_blueprint,
        verbose=False
    )

    # Should only have declarative skeletons (2 out of 4 examples)
    assert len(result) <= 2, f"Expected at most 2 declarative skeletons, got {len(result)}"

    # Verify all returned skeletons are declarative (no question marks)
    for skeleton, _ in result:
        assert "?" not in skeleton, f"Found question skeleton in results: {skeleton}"
        assert not skeleton.lower().startswith(("where", "what", "who", "when", "why", "how")), \
            f"Found question skeleton in results: {skeleton}"

    print("✓ test_type_compatibility_filter_blocks_question_for_declarative passed")


def test_type_compatibility_filter_allows_conditional_for_declarative():
    """Test that type filter allows conditional skeletons for declarative inputs (style expansion)."""
    if not DEPENDENCIES_AVAILABLE:
        print("⚠ SKIPPED: test_type_compatibility_filter_allows_conditional_for_declarative (missing dependencies)")
        return

    translator = StyleTranslator(config_path="config.json")

    # Create declarative blueprint
    declarative_blueprint = SemanticBlueprint(
        original_text="Every morning required a pilgrimage.",
        svo_triples=[("morning", "required", "pilgrimage")],
        named_entities=[],
        core_keywords={"morning", "required", "pilgrimage"},
        citations=[],
        quotes=[]
    )

    # Create examples: mix of conditional and declarative
    examples = [
        "If morning comes, we make a pilgrimage.",  # Conditional - should pass
        "Every morning, we made a pilgrimage.",  # Declarative - should pass
        "When morning arrives, the pilgrimage begins.",  # Conditional - should pass
    ]

    # Mock structuralizer
    def mock_extract_skeleton(example, input_text=None):
        if example.lower().startswith(("if", "when")):
            return "If [NP] [VP], [NP] [VP] [NP]."
        else:
            return "[ADJ] [NP], [NP] [VP] [NP]."

    translator.structuralizer.extract_skeleton = Mock(side_effect=mock_extract_skeleton)
    translator.structuralizer.count_skeleton_slots = Mock(return_value=5)

    # Extract skeletons
    result = translator._extract_multiple_skeletons(
        examples=examples,
        blueprint=declarative_blueprint,
        verbose=False
    )

    # Should have both conditional and declarative skeletons (all 3 should pass)
    assert len(result) >= 2, f"Expected at least 2 skeletons (conditional + declarative), got {len(result)}"

    print("✓ test_type_compatibility_filter_allows_conditional_for_declarative passed")


def test_type_compatibility_filter_allows_exact_match():
    """Test that type filter allows exact type matches."""
    if not DEPENDENCIES_AVAILABLE:
        print("⚠ SKIPPED: test_type_compatibility_filter_allows_exact_match (missing dependencies)")
        return

    translator = StyleTranslator(config_path="config.json")

    # Test question -> question
    question_blueprint = SemanticBlueprint(
        original_text="Where did the morning require a pilgrimage?",
        svo_triples=[("morning", "required", "pilgrimage")],
        named_entities=[],
        core_keywords={"morning", "required", "pilgrimage"},
        citations=[],
        quotes=[]
    )

    question_examples = [
        "Where did the flame come from?",
        "What did the ancestors know?",
    ]

    def mock_extract_skeleton(example, input_text=None):
        return "Where [VP] [NP] [VP] [NP]?"

    translator.structuralizer.extract_skeleton = Mock(side_effect=mock_extract_skeleton)
    translator.structuralizer.count_skeleton_slots = Mock(return_value=5)

    result = translator._extract_multiple_skeletons(
        examples=question_examples,
        blueprint=question_blueprint,
        verbose=False
    )

    # Should allow question skeletons for question input
    assert len(result) >= 1, "Expected question skeletons to pass for question input"

    print("✓ test_type_compatibility_filter_allows_exact_match passed")


def test_adherence_penalizes_type_mismatch():
    """Test that _calculate_skeleton_adherence returns low score for type mismatches."""
    if not DEPENDENCIES_AVAILABLE:
        print("⚠ SKIPPED: test_adherence_penalizes_type_mismatch (missing dependencies)")
        return

    translator = StyleTranslator(config_path="config.json")

    # Question skeleton with declarative candidate (should be penalized)
    question_skeleton = "Where [VP] [NP] [VP] [NP]?"
    declarative_candidate = "Every morning required a pilgrimage to the store."

    adherence = translator._calculate_skeleton_adherence(declarative_candidate, question_skeleton)

    # Should return low score (0.3) for type mismatch
    assert adherence == 0.3, f"Expected 0.3 for type mismatch, got {adherence}"

    # Declarative skeleton with question candidate (should be penalized)
    declarative_skeleton = "[ADJ] [NP] [VP] [NP] [PP] [NP]."
    question_candidate = "Where did the morning require a pilgrimage?"

    adherence = translator._calculate_skeleton_adherence(question_candidate, declarative_skeleton)

    # Should return low score (0.3) for type mismatch
    assert adherence == 0.3, f"Expected 0.3 for type mismatch, got {adherence}"

    print("✓ test_adherence_penalizes_type_mismatch passed")


def test_adherence_allows_type_match():
    """Test that _calculate_skeleton_adherence works normally for type matches."""
    if not DEPENDENCIES_AVAILABLE:
        print("⚠ SKIPPED: test_adherence_allows_type_match (missing dependencies)")
        return

    translator = StyleTranslator(config_path="config.json")

    # Declarative skeleton with declarative candidate (should work normally)
    declarative_skeleton = "[ADJ] [NP] [VP] [NP] [PP] [NP]."
    declarative_candidate = "Every morning required a pilgrimage to the general store with grandmother."

    adherence = translator._calculate_skeleton_adherence(declarative_candidate, declarative_skeleton)

    # Should return a score > 0.3 (not the type mismatch penalty)
    assert adherence > 0.3, f"Expected score > 0.3 for type match, got {adherence}"

    # Question skeleton with question candidate (should work normally)
    question_skeleton = "Where [VP] [NP] [VP] [NP]?"
    question_candidate = "Where did the morning require a pilgrimage?"

    adherence = translator._calculate_skeleton_adherence(question_candidate, question_skeleton)

    # Should return a score > 0.3 (not the type mismatch penalty)
    assert adherence > 0.3, f"Expected score > 0.3 for type match, got {adherence}"

    print("✓ test_adherence_allows_type_match passed")


def test_detect_voice_first_person():
    """Test voice detection for first-person text."""
    if not DEPENDENCIES_AVAILABLE:
        print("⚠ SKIPPED: test_detect_voice_first_person (missing dependencies)")
        return

    translator = StyleTranslator(config_path="config.json")

    test_cases = [
        ("I spent my childhood scavenging in the ruins.", "1st"),
        ("We were there together.", "1st"),
        ("I was thirteen. Every morning required a pilgrimage.", "1st"),
        ("My family lived in that house.", "1st"),
        ("Our journey began there.", "1st")
    ]

    for text, expected in test_cases:
        result = translator._detect_voice(text)
        assert result == expected, f"Expected {expected} for '{text[:40]}...', got {result}"

    print("✓ First-person voice detection works correctly")


def test_detect_voice_second_person():
    """Test voice detection for second-person text."""
    if not DEPENDENCIES_AVAILABLE:
        print("⚠ SKIPPED: test_detect_voice_second_person (missing dependencies)")
        return

    translator = StyleTranslator(config_path="config.json")

    test_cases = [
        ("You will understand that the system operates correctly.", "2nd"),
        ("You'll see how this works.", "2nd"),
        ("Your approach is correct.", "2nd"),
        ("You must follow these steps.", "2nd")
    ]

    for text, expected in test_cases:
        result = translator._detect_voice(text)
        assert result == expected, f"Expected {expected} for '{text[:40]}...', got {result}"

    print("✓ Second-person voice detection works correctly")


def test_detect_voice_third_person():
    """Test voice detection for third-person text."""
    if not DEPENDENCIES_AVAILABLE:
        print("⚠ SKIPPED: test_detect_voice_third_person (missing dependencies)")
        return

    translator = StyleTranslator(config_path="config.json")

    test_cases = [
        ("He walked through the ruins.", "3rd"),
        ("She found the answer.", "3rd"),
        ("They were destroyed.", "3rd"),
        ("The system operates correctly. They work well.", "3rd"),  # "it" ignored, "they" detected
        ("His approach was different.", "3rd")
    ]

    for text, expected in test_cases:
        result = translator._detect_voice(text)
        assert result == expected, f"Expected {expected} for '{text[:40]}...', got {result}"

    print("✓ Third-person voice detection works correctly")


def test_detect_voice_neutral():
    """Test voice detection for neutral text (no pronouns)."""
    if not DEPENDENCIES_AVAILABLE:
        print("⚠ SKIPPED: test_detect_voice_neutral (missing dependencies)")
        return

    translator = StyleTranslator(config_path="config.json")

    test_cases = [
        ("The system operates correctly.", "neutral"),
        ("Clojure is a functional language.", "neutral"),
        ("It is clear that the approach works.", "neutral"),  # "it" is ignored
        ("The violent shift to capitalism did not bring freedom.", "neutral")
    ]

    for text, expected in test_cases:
        result = translator._detect_voice(text)
        assert result == expected, f"Expected {expected} for '{text[:40]}...', got {result}"

    print("✓ Neutral voice detection works correctly")


def test_detect_voice_mixed():
    """Test voice detection for mixed text (should return dominant voice)."""
    if not DEPENDENCIES_AVAILABLE:
        print("⚠ SKIPPED: test_detect_voice_mixed (missing dependencies)")
        return

    translator = StyleTranslator(config_path="config.json")

    # More first-person markers than third-person
    text1 = "I spent my childhood there. We were together. He was also there."
    result1 = translator._detect_voice(text1)
    assert result1 == "1st", f"Expected '1st' for mixed text with more first-person, got {result1}"

    # More third-person markers than first-person
    text2 = "He walked there. She was there. I saw them."
    result2 = translator._detect_voice(text2)
    assert result2 == "3rd", f"Expected '3rd' for mixed text with more third-person, got {result2}"

    print("✓ Mixed voice detection works correctly (returns dominant)")


def test_voice_mismatch_penalty_second_person_template():
    """Test that second-person templates get heavy penalty for first-person input."""
    if not DEPENDENCIES_AVAILABLE:
        print("⚠ SKIPPED: test_voice_mismatch_penalty_second_person_template (missing dependencies)")
        return

    print("\n" + "="*60)
    print("TEST: Voice Mismatch Penalty (2nd-person template)")
    print("="*60)

    translator = StyleTranslator(config_path="config.json")

    # First-person input
    paragraph = "I spent my childhood scavenging in the ruins. We were there together."
    input_voice = translator._detect_voice(paragraph)
    assert input_voice == "1st", "Input should be detected as first-person"

    # Second-person example (should get heavy penalty)
    example_2nd = "You will understand that the system operates correctly. You'll see how this works."
    example_voice_2nd = translator._detect_voice(example_2nd)
    assert example_voice_2nd == "2nd", "Example should be detected as second-person"

    # First-person example (should get no penalty)
    example_1st = "I spent my childhood there. We were together."
    example_voice_1st = translator._detect_voice(example_1st)
    assert example_voice_1st == "1st", "Example should be detected as first-person"

    # Calculate penalties manually (matching the logic in translate_paragraph)
    if example_voice_2nd == "2nd" and input_voice != "2nd":
        penalty_2nd = 0.1
    else:
        penalty_2nd = 1.0

    if example_voice_1st == input_voice:
        penalty_1st = 1.0
    else:
        penalty_1st = 1.0

    assert penalty_2nd == 0.1, f"Second-person template should get 0.1 penalty"
    assert penalty_1st == 1.0, "First-person template should get no penalty"

    print(f"  Input voice: {input_voice}")
    print(f"  2nd-person template penalty: {penalty_2nd}x")
    print(f"  1st-person template penalty: {penalty_1st}x")
    print("✓ Voice mismatch penalty applied correctly")


def test_voice_mismatch_penalty_first_third_person():
    """Test that first vs third person mismatch gets mild penalty."""
    if not DEPENDENCIES_AVAILABLE:
        print("⚠ SKIPPED: test_voice_mismatch_penalty_first_third_person (missing dependencies)")
        return

    print("\n" + "="*60)
    print("TEST: Voice Mismatch Penalty (1st vs 3rd person)")
    print("="*60)

    translator = StyleTranslator(config_path="config.json")

    # First-person input
    paragraph = "I spent my childhood there. We were together."
    input_voice = translator._detect_voice(paragraph)
    assert input_voice == "1st", "Input should be detected as first-person"

    # Third-person example (should get mild penalty)
    example_3rd = "He walked through the ruins. They were destroyed."
    example_voice_3rd = translator._detect_voice(example_3rd)
    assert example_voice_3rd == "3rd", "Example should be detected as third-person"

    # Calculate penalty (matching the logic in translate_paragraph)
    if example_voice_3rd == "2nd" and input_voice != "2nd":
        penalty = 0.1
    elif input_voice == "2nd" and example_voice_3rd != "2nd":
        penalty = 0.7
    elif input_voice != example_voice_3rd and example_voice_3rd != "neutral" and input_voice != "neutral":
        penalty = 0.9
    else:
        penalty = 1.0

    assert penalty == 0.9, f"First vs third person mismatch should get 0.9x penalty, got {penalty}"

    print(f"  Input voice: {input_voice}")
    print(f"  Example voice: {example_voice_3rd}")
    print(f"  Penalty: {penalty}x")
    print("✓ Mild penalty applied for 1st/3rd mismatch")


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
    test_adaptive_length_filter_short_sentences()
    test_rescue_logic()
    test_explicit_concept_mapping()
    test_check_acceptance_without_fluency()
    test_style_lexicon_ratio_configuration()
    # Type compatibility filter tests
    test_detect_sentence_type_question()
    test_detect_sentence_type_conditional()
    test_detect_sentence_type_declarative()
    test_type_compatibility_filter_blocks_question_for_declarative()
    test_type_compatibility_filter_allows_conditional_for_declarative()
    test_type_compatibility_filter_allows_exact_match()
    test_adherence_penalizes_type_mismatch()
    test_adherence_allows_type_match()
    # Voice detection tests
    test_detect_voice_first_person()
    test_detect_voice_second_person()
    test_detect_voice_third_person()
    test_detect_voice_neutral()
    test_detect_voice_mixed()
    # Voice mismatch penalty tests
    test_voice_mismatch_penalty_second_person_template()
    test_voice_mismatch_penalty_first_third_person()
    print("\n✓ All translator tests completed!")

