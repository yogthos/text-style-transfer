"""Tests for hill climbing refinement evolution."""

import sys
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.generator.translator import StyleTranslator, REFINEMENT_PROMPT
from src.ingestion.blueprint import SemanticBlueprint
from src.atlas.rhetoric import RhetoricalType
from src.validator.semantic_critic import SemanticCritic


def test_refinement_prompt_construction():
    """Test that refinement prompt includes all required fields."""
    blueprint_text = "Subjects: cat | Actions: sit | Objects: mat"
    current_draft = "The cat sat on the mat."
    critique_feedback = "CRITICAL: Missing concepts: object."
    rhetorical_type = "OBSERVATION"

    prompt = REFINEMENT_PROMPT.format(
        blueprint_text=blueprint_text,
        current_draft=current_draft,
        critique_feedback=critique_feedback,
        rhetorical_type=rhetorical_type
    )

    assert blueprint_text in prompt, "Blueprint text should be in prompt"
    assert current_draft in prompt, "Current draft should be in prompt"
    assert critique_feedback in prompt, "Critique feedback should be in prompt"
    assert rhetorical_type in prompt, "Rhetorical type should be in prompt"

    print("✓ test_refinement_prompt_construction passed")


def test_refinement_prompt_includes_fluency():
    """Test that refinement prompt explicitly mentions fluency."""
    prompt = REFINEMENT_PROMPT.format(
        blueprint_text="Test",
        current_draft="We touch breaks.",
        critique_feedback="Missing object",
        rhetorical_type="OBSERVATION"
    )

    assert "fluency" in prompt.lower() or "grammatical" in prompt.lower(), \
        "Prompt should mention fluency or grammatical naturalness"
    assert "functional words" in prompt.lower() or "articles" in prompt.lower(), \
        "Prompt should mention functional words or articles"

    print("✓ test_refinement_prompt_includes_fluency passed")


def test_get_blueprint_text():
    """Test _get_blueprint_text helper method."""
    translator = StyleTranslator(config_path="config.json")

    # Test with full blueprint
    blueprint = SemanticBlueprint(
        original_text="The cat sat on the mat.",
        svo_triples=[("cat", "sit", "mat")],
        named_entities=[],
        core_keywords={"cat", "sit", "mat"},
        citations=[],
        quotes=[]
    )

    blueprint_text = translator._get_blueprint_text(blueprint)
    assert "cat" in blueprint_text or "Subjects" in blueprint_text
    assert "sit" in blueprint_text or "Actions" in blueprint_text
    assert "mat" in blueprint_text or "Objects" in blueprint_text

    # Test with empty blueprint
    empty_blueprint = SemanticBlueprint(
        original_text="Test sentence.",
        svo_triples=[],
        named_entities=[],
        core_keywords=set(),
        citations=[],
        quotes=[]
    )

    blueprint_text = translator._get_blueprint_text(empty_blueprint)
    assert blueprint_text == "Test sentence."

    print("✓ test_get_blueprint_text passed")


def test_evolution_accepts_improvements():
    """Test that evolution accepts candidates with higher scores."""
    translator = StyleTranslator(config_path="config.json")

    # Mock LLM provider
    translator.llm_provider = MagicMock()

    # Mock critic that returns improving scores
    mock_critic = MagicMock(spec=SemanticCritic)

    initial_draft = "We touch breaks."
    blueprint = SemanticBlueprint(
        original_text="Every object we touch eventually breaks.",
        svo_triples=[("object", "touch", "breaks")],
        named_entities=[],
        core_keywords={"object", "touch", "break"},
        citations=[],
        quotes=[]
    )

    # Mock critic evaluations: initial score 0.5, then 0.7, then 0.9
    def mock_evaluate(text, blueprint):
        if "object" in text.lower() and "break" in text.lower():
            return {"pass": True, "score": 0.9, "feedback": "Passed"}
        elif "object" in text.lower():
            return {"pass": False, "score": 0.7, "feedback": "Better"}
        else:
            return {"pass": False, "score": 0.5, "feedback": "Missing object"}

    mock_critic.evaluate = Mock(side_effect=mock_evaluate)

    # Mock LLM to return improving drafts
    def mock_call(system_prompt, user_prompt, temperature, max_tokens):
        if "object" in user_prompt.lower():
            return "Every object we touch eventually breaks."
        return "We touch breaks."

    translator.llm_provider.call = Mock(side_effect=mock_call)

    # Run evolution
    best_draft, best_score = translator._evolve_text(
        initial_draft=initial_draft,
        blueprint=blueprint,
        author_name="Test Author",
        style_dna="Test style",
        rhetorical_type=RhetoricalType.OBSERVATION,
        initial_score=0.5,
        initial_feedback="Missing object",
        critic=mock_critic,
        verbose=False
    )

    # Verify score improved
    assert best_score >= 0.5, "Score should not decrease"
    assert "object" in best_draft.lower(), "Improved draft should include missing concept"

    print("✓ test_evolution_accepts_improvements passed")


def test_evolution_rejects_degradations():
    """Test that evolution rejects candidates with lower scores."""
    translator = StyleTranslator(config_path="config.json")
    translator.llm_provider = MagicMock()

    mock_critic = MagicMock(spec=SemanticCritic)

    initial_draft = "The cat sat on the mat."
    blueprint = SemanticBlueprint(
        original_text="The cat sat on the mat.",
        svo_triples=[("cat", "sit", "mat")],
        named_entities=[],
        core_keywords={"cat", "sit", "mat"},
        citations=[],
        quotes=[]
    )

    # Mock critic: initial score 0.8, candidate score 0.4 (worse)
    def mock_evaluate(text, blueprint):
        if text == initial_draft:
            return {"pass": False, "score": 0.8, "feedback": "Good"}
        else:
            return {"pass": False, "score": 0.4, "feedback": "Worse"}

    mock_critic.evaluate = Mock(side_effect=mock_evaluate)
    translator.llm_provider.call = Mock(return_value="Bad draft")

    # Run evolution
    best_draft, best_score = translator._evolve_text(
        initial_draft=initial_draft,
        blueprint=blueprint,
        author_name="Test Author",
        style_dna="Test style",
        rhetorical_type=RhetoricalType.OBSERVATION,
        initial_score=0.8,
        initial_feedback="Good",
        critic=mock_critic,
        verbose=False
    )

    # Verify we kept the initial draft (score didn't decrease)
    assert best_score == 0.8, "Should keep initial draft when candidate is worse"
    assert best_draft == initial_draft, "Should return initial draft when no improvement"

    print("✓ test_evolution_rejects_degradations passed")


def test_evolution_stops_at_pass_threshold():
    """Test that evolution stops when pass threshold is reached."""
    translator = StyleTranslator(config_path="config.json")
    translator.llm_provider = MagicMock()

    mock_critic = MagicMock(spec=SemanticCritic)

    initial_draft = "Draft with score 0.85"
    blueprint = SemanticBlueprint(
        original_text="Test",
        svo_triples=[],
        named_entities=[],
        core_keywords=set(),
        citations=[],
        quotes=[]
    )

    # Mock critic: initial score 0.85, then 0.95 (above threshold 0.9)
    call_count = [0]
    def mock_evaluate(text, blueprint):
        call_count[0] += 1
        if call_count[0] == 1:
            return {"pass": False, "score": 0.85, "feedback": "Almost there"}
        else:
            return {"pass": True, "score": 0.95, "feedback": "Passed"}

    mock_critic.evaluate = Mock(side_effect=mock_evaluate)
    translator.llm_provider.call = Mock(return_value="Improved draft")

    # Run evolution
    best_draft, best_score = translator._evolve_text(
        initial_draft=initial_draft,
        blueprint=blueprint,
        author_name="Test Author",
        style_dna="Test style",
        rhetorical_type=RhetoricalType.OBSERVATION,
        initial_score=0.85,
        initial_feedback="Almost there",
        critic=mock_critic,
        verbose=False
    )

    # Verify we stopped early (should only call LLM once, then stop at threshold)
    assert best_score >= 0.9, "Should reach pass threshold"

    print("✓ test_evolution_stops_at_pass_threshold passed")


def test_evolution_respects_max_generations():
    """Test that evolution respects max_generations limit."""
    translator = StyleTranslator(config_path="config.json")
    translator.llm_provider = MagicMock()

    mock_critic = MagicMock(spec=SemanticCritic)

    initial_draft = "Initial draft"
    blueprint = SemanticBlueprint(
        original_text="Test",
        svo_triples=[],
        named_entities=[],
        core_keywords=set(),
        citations=[],
        quotes=[]
    )

    # Mock critic: always return score below threshold
    def mock_evaluate(text, blueprint):
        return {"pass": False, "score": 0.6, "feedback": "Needs work"}

    mock_critic.evaluate = Mock(side_effect=mock_evaluate)
    translator.llm_provider.call = Mock(return_value="Candidate draft")

    # Run evolution
    best_draft, best_score = translator._evolve_text(
        initial_draft=initial_draft,
        blueprint=blueprint,
        author_name="Test Author",
        style_dna="Test style",
        rhetorical_type=RhetoricalType.OBSERVATION,
        initial_score=0.6,
        initial_feedback="Needs work",
        critic=mock_critic,
        verbose=False
    )

    # Verify LLM was called max_generations times (3 by default)
    assert translator.llm_provider.call.call_count <= 3, "Should not exceed max_generations"

    print("✓ test_evolution_respects_max_generations passed")


def test_evolution_handles_empty_draft():
    """Test that evolution handles empty initial draft."""
    translator = StyleTranslator(config_path="config.json")
    translator.llm_provider = MagicMock()

    mock_critic = MagicMock(spec=SemanticCritic)

    initial_draft = ""
    blueprint = SemanticBlueprint(
        original_text="Test",
        svo_triples=[],
        named_entities=[],
        core_keywords=set(),
        citations=[],
        quotes=[]
    )

    def mock_evaluate(text, blueprint):
        return {"pass": False, "score": 0.0, "feedback": "Empty"}

    mock_critic.evaluate = Mock(side_effect=mock_evaluate)
    translator.llm_provider.call = Mock(return_value="")

    # Run evolution (should handle gracefully)
    best_draft, best_score = translator._evolve_text(
        initial_draft=initial_draft,
        blueprint=blueprint,
        author_name="Test Author",
        style_dna="Test style",
        rhetorical_type=RhetoricalType.OBSERVATION,
        initial_score=0.0,
        initial_feedback="Empty",
        critic=mock_critic,
        verbose=False
    )

    # Should return something (even if empty)
    assert isinstance(best_draft, str), "Should return string"
    assert best_score == 0.0, "Score should remain 0.0"

    print("✓ test_evolution_handles_empty_draft passed")


def test_evolution_skips_when_already_passing():
    """Test that evolution is skipped when initial draft already passes."""
    translator = StyleTranslator(config_path="config.json")
    translator.llm_provider = MagicMock()

    mock_critic = MagicMock(spec=SemanticCritic)

    initial_draft = "Perfect draft"
    blueprint = SemanticBlueprint(
        original_text="Test",
        svo_triples=[],
        named_entities=[],
        core_keywords=set(),
        citations=[],
        quotes=[]
    )

    # Mock critic: initial score already above threshold
    def mock_evaluate(text, blueprint):
        return {"pass": True, "score": 0.95, "feedback": "Perfect"}

    mock_critic.evaluate = Mock(side_effect=mock_evaluate)

    # Run evolution
    best_draft, best_score = translator._evolve_text(
        initial_draft=initial_draft,
        blueprint=blueprint,
        author_name="Test Author",
        style_dna="Test style",
        rhetorical_type=RhetoricalType.OBSERVATION,
        initial_score=0.95,
        initial_feedback="Perfect",
        critic=mock_critic,
        verbose=False
    )

    # Should not call LLM (already passing)
    assert translator.llm_provider.call.call_count == 0, "Should not call LLM when already passing"
    assert best_draft == initial_draft, "Should return initial draft unchanged"
    assert best_score == 0.95, "Score should remain unchanged"

    print("✓ test_evolution_skips_when_already_passing passed")


def test_evolution_score_never_decreases():
    """Test that evolution score never decreases (hill climbing property)."""
    translator = StyleTranslator(config_path="config.json")
    translator.llm_provider = MagicMock()

    mock_critic = MagicMock(spec=SemanticCritic)

    initial_draft = "Initial draft"
    blueprint = SemanticBlueprint(
        original_text="Test",
        svo_triples=[],
        named_entities=[],
        core_keywords=set(),
        citations=[],
        quotes=[]
    )

    # Track scores
    scores = [0.5, 0.4, 0.6, 0.7]  # Some worse, some better
    score_index = [0]

    def mock_evaluate(text, blueprint):
        idx = score_index[0]
        score_index[0] += 1
        if idx < len(scores):
            return {"pass": False, "score": scores[idx], "feedback": f"Score {scores[idx]}"}
        return {"pass": False, "score": 0.5, "feedback": "Default"}

    mock_critic.evaluate = Mock(side_effect=mock_evaluate)
    translator.llm_provider.call = Mock(return_value="Candidate")

    # Run evolution
    best_draft, best_score = translator._evolve_text(
        initial_draft=initial_draft,
        blueprint=blueprint,
        author_name="Test Author",
        style_dna="Test style",
        rhetorical_type=RhetoricalType.OBSERVATION,
        initial_score=0.5,
        initial_feedback="Initial",
        critic=mock_critic,
        verbose=False
    )

    # Verify score never decreased from initial
    assert best_score >= 0.5, "Final score should be >= initial score"

    print("✓ test_evolution_score_never_decreases passed")


def test_dynamic_temperature_increases_when_stuck():
    """Test that temperature increases when mutations fail to improve."""
    translator = StyleTranslator(config_path="config.json")
    translator.llm_provider = MagicMock()

    # Mock critic: always return same score (stuck)
    mock_critic = MagicMock(spec=SemanticCritic)
    mock_critic.evaluate.return_value = {"score": 0.7, "feedback": "Stuck", "pass": False}

    # Track temperature calls
    temperatures = []
    def track_temp(*args, **kwargs):
        temp = kwargs.get("temperature", 0.5)
        temperatures.append(temp)
        return "Candidate draft"

    translator.llm_provider.call = Mock(side_effect=track_temp)

    blueprint = SemanticBlueprint(
        original_text="Test",
        svo_triples=[],
        named_entities=[],
        core_keywords=set(),
        citations=[],
        quotes=[]
    )

    # Run evolution
    translator._evolve_text(
        initial_draft="Initial draft",
        blueprint=blueprint,
        author_name="Test Author",
        style_dna="Test style",
        rhetorical_type=RhetoricalType.OBSERVATION,
        initial_score=0.7,
        initial_feedback="Stuck",
        critic=mock_critic,
        verbose=False
    )

    # Verify temperature increased
    assert len(temperatures) > 1, "Should call LLM multiple times"
    assert temperatures[-1] > temperatures[0], \
        f"Temperature should increase when stuck (started at {temperatures[0]:.2f}, ended at {temperatures[-1]:.2f})"

    print("✓ test_dynamic_temperature_increases_when_stuck passed")


def test_dynamic_temperature_resets_on_improvement():
    """Test that temperature resets to initial when score improves."""
    translator = StyleTranslator(config_path="config.json")
    translator.llm_provider = MagicMock()

    # Mock critic: return improving scores
    mock_critic = MagicMock(spec=SemanticCritic)
    scores = [0.6, 0.7, 0.8]  # Improving
    score_index = [0]

    def mock_evaluate(text, blueprint):
        idx = score_index[0]
        score_index[0] += 1
        if idx < len(scores):
            return {"score": scores[idx], "feedback": f"Score {scores[idx]}", "pass": False}
        return {"score": 0.6, "feedback": "Default", "pass": False}

    mock_critic.evaluate = Mock(side_effect=mock_evaluate)

    # Track temperature calls
    temperatures = []
    def track_temp(*args, **kwargs):
        temp = kwargs.get("temperature", 0.5)
        temperatures.append(temp)
        return "Improved draft"

    translator.llm_provider.call = Mock(side_effect=track_temp)

    blueprint = SemanticBlueprint(
        original_text="Test",
        svo_triples=[],
        named_entities=[],
        core_keywords=set(),
        citations=[],
        quotes=[]
    )

    # Run evolution
    translator._evolve_text(
        initial_draft="Initial draft",
        blueprint=blueprint,
        author_name="Test Author",
        style_dna="Test style",
        rhetorical_type=RhetoricalType.OBSERVATION,
        initial_score=0.6,
        initial_feedback="Initial",
        critic=mock_critic,
        verbose=False
    )

    # Verify temperature resets after improvement
    # After first improvement, temperature should reset to initial (0.3)
    if len(temperatures) >= 2:
        # Temperature should reset after improvement
        assert temperatures[-1] <= temperatures[0] + 0.1, \
            "Temperature should reset to initial after improvement"

    print("✓ test_dynamic_temperature_resets_on_improvement passed")


if __name__ == "__main__":
    test_refinement_prompt_construction()
    test_refinement_prompt_includes_fluency()
    test_get_blueprint_text()
    test_evolution_accepts_improvements()
    test_evolution_rejects_degradations()
    test_evolution_stops_at_pass_threshold()
    test_evolution_respects_max_generations()
    test_evolution_handles_empty_draft()
    test_evolution_skips_when_already_passing()
    test_evolution_score_never_decreases()
    test_dynamic_temperature_increases_when_stuck()
    test_dynamic_temperature_resets_on_improvement()
    test_smart_patience_early_exit()
    test_fluency_forgiveness_accepts_perfect_recall()
    test_precision_trap_fix_end_to_end()
    print("\n✓ All refinement tests completed!")

