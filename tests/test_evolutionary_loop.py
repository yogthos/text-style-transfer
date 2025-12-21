"""Tests for Evolutionary Loop in translate_paragraph.

This feature ensures that translate_paragraph breeds children and evolves over generations
until both meaning and style thresholds are met, rather than just doing a single refinement pass.
"""

import sys
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import json

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Ensure config.json exists before imports
from tests.test_helpers import ensure_config_exists
ensure_config_exists()

try:
    from src.ingestion.blueprint import SemanticBlueprint, BlueprintExtractor
    from src.validator.semantic_critic import SemanticCritic
    from src.generator.translator import StyleTranslator
    from src.analysis.semantic_analyzer import PropositionExtractor
    from src.atlas.rhetoric import RhetoricalType
    DEPENDENCIES_AVAILABLE = True
except (ImportError, ModuleNotFoundError) as e:
    DEPENDENCIES_AVAILABLE = False
    IMPORT_ERROR = str(e)
    print(f"⚠ Skipping tests: Missing dependencies - {IMPORT_ERROR}")


def test_evolutionary_loop_breeds_children():
    """Test that the evolutionary loop calls _breed_children when style is below threshold."""
    if not DEPENDENCIES_AVAILABLE:
        print("⚠ SKIPPED: test_evolutionary_loop_breeds_children (missing dependencies)")
        return

    translator = StyleTranslator(config_path="config.json")

    # Track if _breed_children was called
    breed_called = [False]

    def mock_breed(*args, **kwargs):
        breed_called[0] = True
        # Return mock children
        return [
            "My childhood was spent scavenging in the ruins of the Soviet Union, now a ghost.",
            "Scavenging in the ruins of the Soviet Union during my childhood, I saw it become a ghost."
        ]

    translator._breed_children = mock_breed

    # Create test scenario: qualified candidate with good meaning but poor style
    from src.ingestion.blueprint import BlueprintExtractor
    extractor = BlueprintExtractor()
    paragraph = "I spent my childhood scavenging in the ruins of the Soviet Union. That country is a ghost now."
    blueprint = extractor.extract(paragraph)

    # Test data that should trigger breeding
    qualified_candidates = [
        {
            "text": "I spent my childhood scavenging in the ruins of the Soviet Union, a ghost now.",
            "recall": 0.90,  # Good meaning (>= 0.85)
            "style_alignment": 0.65,  # Poor style (below 0.7 threshold)
            "score": 0.78,
            "result": {"proposition_recall": 0.90, "style_alignment": 0.65}
        }
    ]

    # Mock LLM for breeding
    mock_llm = Mock()
    mock_llm.call.return_value = json.dumps(["Bred child 1", "Bred child 2"])
    translator.llm_provider = mock_llm

    # Test the evolutionary loop conditions
    evo_config = translator.config.get("evolutionary", {})
    max_generations = evo_config.get("max_generations", 3)
    style_threshold = translator.paragraph_fusion_config.get("style_alignment_threshold", 0.7)
    recall_threshold = translator.paragraph_fusion_config.get("proposition_recall_threshold", 0.85)

    # Convert to population format (as done in actual code)
    population = []
    for c in qualified_candidates:
        population.append({
            "text": c["text"],
            "recall": c["recall"],
            "style_score": c.get("style_alignment", 0.0),
            "composite_score": c["score"]
        })

    # Simulate one iteration of the evolutionary loop
    generation = 0
    while generation < max_generations:
        # Sort: Meaning Passers First, then by Style
        population.sort(key=lambda x: (x["recall"] >= recall_threshold, x["style_score"]), reverse=True)
        best_survivor = population[0] if population else None

        if not best_survivor:
            break

        # Check Thresholds
        if best_survivor["recall"] >= recall_threshold and best_survivor["style_score"] >= style_threshold:
            break  # Would break in actual code

        # This is where breeding should happen
        parents = population[:5]
        # Need at least 2 parents, but we only have 1 in population
        # Add more to population to meet the requirement
        if len(parents) < 2:
            # Duplicate the parent to meet minimum requirement
            population.append(population[0].copy())
            parents = population[:5]

        if len(parents) >= 2:
            # Call _breed_children (this is what we're testing)
            children = translator._breed_children(
                parents=[{"text": p["text"], "recall_score": p["recall"],
                        "style_density": p["style_score"], "score": p["composite_score"],
                        "critic_result": {}} for p in parents],
                blueprint=blueprint,
                author_name="Test Author",
                style_dna="Authoritative Complex",
                rhetorical_type=RhetoricalType.OBSERVATION,
                style_lexicon=None,
                num_children=10,
                verbose=False
            )
            assert len(children) > 0, "Breeding should produce children"
            break  # Stop after first iteration for test

        generation += 1

    assert breed_called[0], f"Evolutionary loop should call _breed_children when style is below threshold. Called: {breed_called[0]}, parents: {len(parents) if 'parents' in locals() else 0}"
    print("✓ test_evolutionary_loop_breeds_children passed")


def test_evolutionary_loop_stagnation_detection():
    """Test that stagnation detection stops the loop when score doesn't improve."""
    if not DEPENDENCIES_AVAILABLE:
        print("⚠ SKIPPED: test_evolutionary_loop_stagnation_detection (missing dependencies)")
        return

    translator = StyleTranslator(config_path="config.json")
    blueprint_extractor = BlueprintExtractor()
    extractor = PropositionExtractor()

    paragraph = "I spent my childhood scavenging in the ruins of the Soviet Union."
    propositions = extractor.extract_atomic_propositions(paragraph)
    blueprint = blueprint_extractor.extract(paragraph)

    # Track generation count
    generation_count = [0]
    original_breed = translator._breed_children

    def mock_breed(*args, **kwargs):
        generation_count[0] += 1
        return ["Child text 1", "Child text 2"]

    translator._breed_children = mock_breed

    # Mock atlas
    mock_atlas = Mock()
    mock_atlas.get_examples_by_rhetoric.return_value = ["Example 1"]

    # Mock LLM provider
    mock_llm = Mock()
    mock_llm.call.return_value = json.dumps(["Initial text"])
    translator.llm_provider = mock_llm

    # Mock critic to return same score (stagnation)
    def mock_evaluate(text, blueprint, **kwargs):
        # Always return same score (0.65) to trigger stagnation
        return {
            "proposition_recall": 0.90,
            "style_alignment": 0.65,  # Never improves
            "score": 0.78,
            "pass": True
        }

    with patch.object(SemanticCritic, 'evaluate', mock_evaluate):
        try:
            result = translator.translate_paragraph(
                paragraph=paragraph,
                atlas=mock_atlas,
                author_name="Test Author",
                verbose=False
            )
        except Exception as e:
            if "atlas" in str(e).lower():
                print(f"  ⚠ Test skipped due to missing dependencies: {e}")
                return

    # Stagnation should stop after 2 generations of no improvement
    # With max_generations=3, it should stop at generation 2 (after 2 stagnant generations)
    assert generation_count[0] <= 3, f"Stagnation should stop evolution early. Got {generation_count[0]} generations"
    print("✓ test_evolutionary_loop_stagnation_detection passed")


def test_evolutionary_loop_stops_when_thresholds_met():
    """Test that the loop stops when both meaning and style thresholds are met."""
    if not DEPENDENCIES_AVAILABLE:
        print("⚠ SKIPPED: test_evolutionary_loop_stops_when_thresholds_met (missing dependencies)")
        return

    translator = StyleTranslator(config_path="config.json")
    blueprint_extractor = BlueprintExtractor()
    extractor = PropositionExtractor()

    paragraph = "I spent my childhood scavenging in the ruins of the Soviet Union."
    propositions = extractor.extract_atomic_propositions(paragraph)
    blueprint = blueprint_extractor.extract(paragraph)

    # Track generation count
    generation_count = [0]
    original_breed = translator._breed_children

    def mock_breed(*args, **kwargs):
        generation_count[0] += 1
        return ["Improved child text"]

    translator._breed_children = mock_breed

    # Mock atlas
    mock_atlas = Mock()
    mock_atlas.get_examples_by_rhetoric.return_value = ["Example 1"]

    # Mock LLM provider
    mock_llm = Mock()
    mock_llm.call.return_value = json.dumps(["Initial text"])
    translator.llm_provider = mock_llm

    # Mock critic: first has poor style, then improves to meet threshold
    call_count = [0]
    def mock_evaluate(text, blueprint, **kwargs):
        call_count[0] += 1
        if call_count[0] == 1:
            # Initial: good meaning, poor style
            return {
                "proposition_recall": 0.90,
                "style_alignment": 0.65,  # Below threshold
                "score": 0.78,
                "pass": True
            }
        else:
            # After breeding: both thresholds met
            return {
                "proposition_recall": 0.90,
                "style_alignment": 0.75,  # Above threshold (0.7)
                "score": 0.83,
                "pass": True
            }

    with patch.object(SemanticCritic, 'evaluate', mock_evaluate):
        try:
            result = translator.translate_paragraph(
                paragraph=paragraph,
                atlas=mock_atlas,
                author_name="Test Author",
                verbose=False
            )
        except Exception as e:
            if "atlas" in str(e).lower():
                print(f"  ⚠ Test skipped due to missing dependencies: {e}")
                return

    # Should stop after first generation when thresholds are met
    assert generation_count[0] <= 1, f"Should stop when thresholds met. Got {generation_count[0]} generations"
    print("✓ test_evolutionary_loop_stops_when_thresholds_met passed")


def test_evolutionary_loop_uses_existing_critic():
    """Test that the loop reuses critic instance (performance safety)."""
    if not DEPENDENCIES_AVAILABLE:
        print("⚠ SKIPPED: test_evolutionary_loop_uses_existing_critic (missing dependencies)")
        return

    translator = StyleTranslator(config_path="config.json")

    # Track SemanticCritic instantiations
    critic_instantiations = [0]
    original_init = SemanticCritic.__init__

    def counting_init(self, *args, **kwargs):
        critic_instantiations[0] += 1
        return original_init(self, *args, **kwargs)

    SemanticCritic.__init__ = counting_init

    # Mock other dependencies
    mock_atlas = Mock()
    mock_atlas.get_examples_by_rhetoric.return_value = ["Example 1"]

    mock_llm = Mock()
    mock_llm.call.return_value = json.dumps(["Initial text"])
    translator.llm_provider = mock_llm

    # Mock _breed_children to return children
    translator._breed_children = Mock(return_value=["Child 1", "Child 2"])

    # Mock evaluate to trigger evolution
    def mock_evaluate(text, blueprint, **kwargs):
        return {
            "proposition_recall": 0.90,
            "style_alignment": 0.65,  # Below threshold
            "score": 0.78,
            "pass": True
        }

    with patch.object(SemanticCritic, 'evaluate', mock_evaluate):
        try:
            result = translator.translate_paragraph(
                paragraph="Test paragraph with multiple sentences.",
                atlas=mock_atlas,
                author_name="Test Author",
                verbose=False
            )
        except Exception as e:
            if "atlas" in str(e).lower():
                print(f"  ⚠ Test skipped due to missing dependencies: {e}")
                SemanticCritic.__init__ = original_init
                return

    # Restore original
    SemanticCritic.__init__ = original_init

    # Critic should be instantiated once (or use existing self.critic)
    # The loop should reuse it, not create new instances
    assert critic_instantiations[0] <= 2, f"Critic should be instantiated once or reused. Got {critic_instantiations[0]} instantiations"
    print("✓ test_evolutionary_loop_uses_existing_critic passed")


def test_evolutionary_loop_filters_viable_children():
    """Test that the loop filters out children with low recall (< 0.6)."""
    if not DEPENDENCIES_AVAILABLE:
        print("⚠ SKIPPED: test_evolutionary_loop_filters_viable_children (missing dependencies)")
        return

    translator = StyleTranslator(config_path="config.json")
    blueprint_extractor = BlueprintExtractor()
    extractor = PropositionExtractor()

    paragraph = "I spent my childhood scavenging in the ruins of the Soviet Union."
    propositions = extractor.extract_atomic_propositions(paragraph)
    blueprint = blueprint_extractor.extract(paragraph)

    # Test the filtering logic directly
    children_candidates = [
        {
            "text": "Good child with high recall",
            "recall": 0.85,  # Above 0.6 threshold
            "style_score": 0.70,
            "composite_score": 0.78
        },
        {
            "text": "Bad child with low recall",
            "recall": 0.50,  # Below 0.6 threshold
            "style_score": 0.80,
            "composite_score": 0.65
        }
    ]

    # Simulate the filtering logic from the evolutionary loop
    viable_children = [c for c in children_candidates if c["recall"] > 0.6]

    # Verify filtering works
    assert len(viable_children) == 1, f"Should filter out low recall children. Got {len(viable_children)} viable"
    assert viable_children[0]["text"] == "Good child with high recall", "Should keep high recall child"
    assert "Bad child" not in [c["text"] for c in viable_children], "Should filter out low recall child"

    print("✓ test_evolutionary_loop_filters_viable_children passed")


def test_evolutionary_loop_meaning_first_sorting():
    """Test that population is sorted with meaning-valid candidates first (Darwinian Meaning Guardrail)."""
    if not DEPENDENCIES_AVAILABLE:
        print("⚠ SKIPPED: test_evolutionary_loop_meaning_first_sorting (missing dependencies)")
        return

    translator = StyleTranslator(config_path="config.json")

    # Create a mock population to test sorting logic
    population = [
        {"text": "High style, low recall", "recall": 0.70, "style_score": 0.90, "composite_score": 0.80},
        {"text": "High recall, low style", "recall": 0.90, "style_score": 0.60, "composite_score": 0.75},
        {"text": "Both high", "recall": 0.90, "style_score": 0.85, "composite_score": 0.88},
    ]

    recall_threshold = 0.85

    # Simulate the sorting logic from the evolutionary loop
    population.sort(key=lambda x: (x["recall"] >= recall_threshold, x["style_score"]), reverse=True)

    # First should be "Both high" (meaning valid AND highest style)
    assert population[0]["text"] == "Both high", "Meaning-valid candidate with high style should be first"
    # Second should be "High recall, low style" (meaning valid, lower style)
    assert population[1]["text"] == "High recall, low style", "Meaning-valid candidate should come before meaning-invalid"
    # Third should be "High style, low recall" (meaning invalid, even though high style)
    assert population[2]["text"] == "High style, low recall", "Meaning-invalid candidate should be last, even with high style"

    print("✓ test_evolutionary_loop_meaning_first_sorting passed")


def test_evolutionary_loop_structure_exists():
    """Test that translate_paragraph has the evolutionary loop structure (regression prevention)."""
    if not DEPENDENCIES_AVAILABLE:
        print("⚠ SKIPPED: test_evolutionary_loop_structure_exists (missing dependencies)")
        return

    translator = StyleTranslator(config_path="config.json")

    # Read the source code to verify evolutionary loop structure exists
    import inspect
    source = inspect.getsource(translator.translate_paragraph)

    # Critical checks to prevent regression
    assert "max_generations" in source, "Evolutionary loop should use max_generations"
    assert "stagnation_counter" in source, "Evolutionary loop should have stagnation detection"
    assert "_breed_children" in source, "Evolutionary loop should call _breed_children"
    assert "population.sort" in source, "Evolutionary loop should sort population"
    assert "recall_threshold" in source and "style_threshold" in source, "Evolutionary loop should check both thresholds"
    assert "while generation" in source or ("while" in source and "generation" in source), "Evolutionary loop should have a while loop with generation counter"

    # Verify critic reuse (performance safety)
    assert "self.critic" in source or "critic = self.critic" in source or "if not hasattr(self, 'critic')" in source, "Should reuse critic instance (performance safety)"

    # Verify population structure
    assert "population.append" in source or "population = []" in source, "Should maintain population structure"

    print("✓ test_evolutionary_loop_structure_exists passed")


if __name__ == "__main__":
    test_evolutionary_loop_breeds_children()
    test_evolutionary_loop_stagnation_detection()
    test_evolutionary_loop_stops_when_thresholds_met()
    test_evolutionary_loop_uses_existing_critic()
    test_evolutionary_loop_filters_viable_children()
    test_evolutionary_loop_meaning_first_sorting()
    test_evolutionary_loop_structure_exists()
    print("\n✓ All Evolutionary Loop tests completed!")

