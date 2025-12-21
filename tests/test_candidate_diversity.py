"""Tests for candidate diversity - ensuring each candidate uses a different seed sentence."""

import sys
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock, call

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Ensure config.json exists before imports
from tests.test_helpers import ensure_config_exists
ensure_config_exists()

# Import with error handling for missing dependencies
try:
    from src.generator.translator import StyleTranslator
    from src.atlas.rhetoric import RhetoricalType
    DEPENDENCIES_AVAILABLE = True
except (ImportError, ModuleNotFoundError) as e:
    DEPENDENCIES_AVAILABLE = False
    IMPORT_ERROR = str(e)
    print(f"⚠ Skipping tests: Missing dependencies - {IMPORT_ERROR}")


def test_candidates_use_different_seed_sentences():
    """Test that each candidate is seeded with a different sentence from the corpus."""
    if not DEPENDENCIES_AVAILABLE:
        print("⚠ SKIPPED: test_candidates_use_different_seed_sentences (missing dependencies)")
        return

    translator = StyleTranslator(config_path="config.json")

    sentence_plan = {
        "text": "I spent my childhood scavenging in the ruins of the Soviet Union.",
        "propositions": [
            "I spent my childhood scavenging in the ruins of the Soviet Union.",
            "The Soviet Union is a ghost now."
        ]
    }

    # Create multiple distinct corpus examples (at least 5 for population_size=5)
    corpus_examples = [
        "My childhood was spent scavenging amidst the ruins of the Soviet Union.",
        "Throughout my youth, I scavenged in the Soviet Union's ruins.",
        "I scavenged in Soviet ruins during my childhood years.",
        "The ruins of the Soviet Union were where I spent my childhood scavenging.",
        "Scavenging in Soviet ruins defined my childhood experience.",
        "In the ruins of the Soviet Union, I spent my childhood scavenging.",
        "My youth involved scavenging through Soviet Union ruins.",
        "The Soviet Union's ruins became my childhood scavenging ground."
    ]

    # Mock atlas to return these examples
    mock_atlas = Mock()
    mock_atlas.get_examples_by_rhetoric.return_value = corpus_examples

    # Track which examples were used for each candidate
    used_examples = []

    def track_build_prompt(*args, **kwargs):
        """Track which examples are passed to _build_prompt."""
        examples = kwargs.get('examples', [])
        if examples:
            used_examples.append(examples[0])  # Track the first (seed) example
        # Return a mock prompt
        return "Mock prompt with examples"

    # Mock _build_prompt to track examples
    with patch.object(translator, '_build_prompt', side_effect=track_build_prompt):
        # Mock LLM to return different responses based on call count
        call_count = [0]
        def mock_llm_call(*args, **kwargs):
            call_count[0] += 1
            # Return different responses to simulate diversity
            responses = [
                "Candidate 1: My childhood was spent scavenging amidst the ruins.",
                "Candidate 2: Throughout my youth, I scavenged in the Soviet ruins.",
                "Candidate 3: I scavenged in Soviet ruins during my childhood.",
                "Candidate 4: The ruins of the Soviet Union were where I scavenged.",
                "Candidate 5: Scavenging in Soviet ruins defined my childhood."
            ]
            return responses[call_count[0] - 1] if call_count[0] <= len(responses) else responses[-1]

        translator.llm_provider = Mock()
        translator.llm_provider.call.side_effect = mock_llm_call

        # Mock critic to return passing scores immediately (no evolution needed)
        mock_critic = Mock()
        mock_critic.evaluate.return_value = {
            "proposition_recall": 0.90,
            "style_alignment": 0.80,
            "score": 0.85,
            "pass": True,
            "feedback": "Good",
            "reason": ""
        }

        with patch('src.generator.translator.SemanticCritic', return_value=mock_critic):
            with patch('src.ingestion.blueprint.BlueprintExtractor') as mock_extractor_class:
                with patch('src.atlas.rhetoric.RhetoricalClassifier') as mock_classifier_class:
                    # Mock blueprint extractor
                    mock_extractor = Mock()
                    mock_blueprint = Mock()
                    mock_blueprint.original_text = sentence_plan["text"]
                    mock_blueprint.svo_triples = []
                    mock_blueprint.core_keywords = {"childhood", "scavenging", "Soviet Union"}
                    mock_blueprint.named_entities = []
                    mock_blueprint.citations = []
                    mock_blueprint.quotes = []
                    mock_blueprint.position = "BODY"
                    mock_blueprint.previous_context = None
                    mock_blueprint.get_subjects = Mock(return_value=[])
                    mock_blueprint.get_verbs = Mock(return_value=[])
                    mock_blueprint.get_objects = Mock(return_value=[])
                    mock_extractor.extract.return_value = mock_blueprint
                    mock_extractor_class.return_value = mock_extractor

                    # Mock rhetorical classifier
                    mock_classifier = Mock()
                    mock_classifier.classify_heuristic.return_value = RhetoricalType.OBSERVATION
                    mock_classifier_class.return_value = mock_classifier

                    # Mock NLP to return None (use fallback sentence counting)
                    with patch.object(translator, '_get_nlp', return_value=None):
                        result = translator._evolve_sentence_unit(
                            sentence_plan=sentence_plan,
                            context=[],
                            atlas=mock_atlas,
                            author_name="Test Author",
                            style_dna=None,
                            verbose=False
                        )

    # Verify that different seed sentences were used
    assert len(used_examples) >= 5, f"Expected at least 5 candidates, got {len(used_examples)}"

    # Verify that at least some seed sentences are different
    unique_seeds = set(used_examples)
    assert len(unique_seeds) > 1, \
        f"All candidates used the same seed sentence! Seeds used: {used_examples}"

    # Verify that we're using different examples from the corpus
    # (not just the same one repeated)
    assert len(unique_seeds) >= 3, \
        f"Expected at least 3 unique seed sentences, got {len(unique_seeds)}. Seeds: {unique_seeds}"

    print(f"✓ test_candidates_use_different_seed_sentences passed")
    print(f"  Used {len(unique_seeds)} unique seed sentences from {len(used_examples)} candidates")


def test_candidates_are_diverse():
    """Test that generated candidates are actually different from each other."""
    if not DEPENDENCIES_AVAILABLE:
        print("⚠ SKIPPED: test_candidates_are_diverse (missing dependencies)")
        return

    translator = StyleTranslator(config_path="config.json")

    sentence_plan = {
        "text": "I spent my childhood scavenging in the ruins of the Soviet Union.",
        "propositions": [
            "I spent my childhood scavenging in the ruins of the Soviet Union."
        ]
    }

    # Create diverse corpus examples
    corpus_examples = [
        "My childhood was spent scavenging amidst the ruins of the Soviet Union.",
        "Throughout my youth, I scavenged in the Soviet Union's ruins.",
        "I scavenged in Soviet ruins during my childhood years.",
        "The ruins of the Soviet Union were where I spent my childhood scavenging.",
        "Scavenging in Soviet ruins defined my childhood experience.",
        "In the ruins of the Soviet Union, I spent my childhood scavenging.",
        "My youth involved scavenging through Soviet Union ruins.",
        "The Soviet Union's ruins became my childhood scavenging ground."
    ]

    mock_atlas = Mock()
    mock_atlas.get_examples_by_rhetoric.return_value = corpus_examples

    # Track generated candidates
    generated_candidates = []

    call_count = [0]
    def mock_llm_call(*args, **kwargs):
        call_count[0] += 1
        # Return distinctly different responses
        responses = [
            "My childhood was spent scavenging amidst the ruins of the Soviet Union.",
            "Throughout my youth, I scavenged in the Soviet Union's ruins.",
            "I scavenged in Soviet ruins during my childhood years.",
            "The ruins of the Soviet Union were where I spent my childhood scavenging.",
            "Scavenging in Soviet ruins defined my childhood experience."
        ]
        response = responses[call_count[0] - 1] if call_count[0] <= len(responses) else responses[-1]
        generated_candidates.append(response)
        return response

    translator.llm_provider = Mock()
    translator.llm_provider.call.side_effect = mock_llm_call

    # Mock critic to return passing scores
    mock_critic = Mock()
    mock_critic.evaluate.return_value = {
        "proposition_recall": 0.90,
        "style_alignment": 0.80,
        "score": 0.85,
        "pass": True,
        "feedback": "Good",
        "reason": ""
    }

    with patch('src.generator.translator.SemanticCritic', return_value=mock_critic):
        with patch('src.ingestion.blueprint.BlueprintExtractor') as mock_extractor_class:
            with patch('src.atlas.rhetoric.RhetoricalClassifier') as mock_classifier_class:
                # Mock blueprint extractor
                mock_extractor = Mock()
                mock_blueprint = Mock()
                mock_blueprint.original_text = sentence_plan["text"]
                mock_blueprint.svo_triples = []
                mock_blueprint.core_keywords = {"childhood", "scavenging", "Soviet Union"}
                mock_blueprint.named_entities = []
                mock_blueprint.citations = []
                mock_blueprint.quotes = []
                mock_blueprint.position = "BODY"
                mock_blueprint.previous_context = None
                mock_blueprint.get_subjects = Mock(return_value=[])
                mock_blueprint.get_verbs = Mock(return_value=[])
                mock_blueprint.get_objects = Mock(return_value=[])
                mock_extractor.extract.return_value = mock_blueprint
                mock_extractor_class.return_value = mock_extractor

                # Mock rhetorical classifier
                mock_classifier = Mock()
                mock_classifier.classify_heuristic.return_value = RhetoricalType.OBSERVATION
                mock_classifier_class.return_value = mock_classifier

                # Mock NLP
                with patch.object(translator, '_get_nlp', return_value=None):
                    result = translator._evolve_sentence_unit(
                        sentence_plan=sentence_plan,
                        context=[],
                        atlas=mock_atlas,
                        author_name="Test Author",
                        style_dna=None,
                        verbose=False
                    )

    # Verify that we got multiple candidates
    assert len(generated_candidates) >= 5, \
        f"Expected at least 5 candidates, got {len(generated_candidates)}"

    # Verify candidates are unique
    unique_candidates = set(generated_candidates)
    assert len(unique_candidates) > 1, \
        f"All candidates are identical! Candidates: {generated_candidates}"

    # Verify we have at least 3 unique candidates
    assert len(unique_candidates) >= 3, \
        f"Expected at least 3 unique candidates, got {len(unique_candidates)}. " \
        f"Candidates: {generated_candidates}"

    print(f"✓ test_candidates_are_diverse passed")
    print(f"  Generated {len(unique_candidates)} unique candidates from {len(generated_candidates)} total")


def test_fetches_enough_examples():
    """Test that the code fetches enough examples to seed all candidates."""
    if not DEPENDENCIES_AVAILABLE:
        print("⚠ SKIPPED: test_fetches_enough_examples (missing dependencies)")
        return

    translator = StyleTranslator(config_path="config.json")

    sentence_plan = {
        "text": "I spent my childhood scavenging in the ruins of the Soviet Union.",
        "propositions": ["I spent my childhood scavenging in the ruins of the Soviet Union."]
    }

    # Create many corpus examples
    corpus_examples = [
        f"Example sentence {i} from the corpus with unique content {i}."
        for i in range(30)  # More than enough for population_size=5
    ]

    mock_atlas = Mock()

    # Track how many examples were requested
    requested_top_k = []

    def track_get_examples(rhetorical_type, top_k=None, **kwargs):
        requested_top_k.append(top_k)
        return corpus_examples[:top_k] if top_k else corpus_examples

    mock_atlas.get_examples_by_rhetoric.side_effect = track_get_examples

    translator.llm_provider = Mock()
    translator.llm_provider.call.return_value = "Generated candidate"

    mock_critic = Mock()
    mock_critic.evaluate.return_value = {
        "proposition_recall": 0.90,
        "style_alignment": 0.80,
        "score": 0.85,
        "pass": True,
        "feedback": "Good",
        "reason": ""
    }

    with patch('src.generator.translator.SemanticCritic', return_value=mock_critic):
        with patch('src.ingestion.blueprint.BlueprintExtractor') as mock_extractor_class:
            with patch('src.atlas.rhetoric.RhetoricalClassifier') as mock_classifier_class:
                mock_extractor = Mock()
                mock_blueprint = Mock()
                mock_blueprint.original_text = sentence_plan["text"]
                mock_blueprint.svo_triples = []
                mock_blueprint.core_keywords = {"childhood", "scavenging"}
                mock_blueprint.named_entities = []
                mock_blueprint.citations = []
                mock_blueprint.quotes = []
                mock_blueprint.position = "BODY"
                mock_blueprint.previous_context = None
                mock_blueprint.get_subjects = Mock(return_value=[])
                mock_blueprint.get_verbs = Mock(return_value=[])
                mock_blueprint.get_objects = Mock(return_value=[])
                mock_extractor.extract.return_value = mock_blueprint
                mock_extractor_class.return_value = mock_extractor

                mock_classifier = Mock()
                mock_classifier.classify_heuristic.return_value = RhetoricalType.OBSERVATION
                mock_classifier_class.return_value = mock_classifier

                with patch.object(translator, '_get_nlp', return_value=None):
                    translator._evolve_sentence_unit(
                        sentence_plan=sentence_plan,
                        context=[],
                        atlas=mock_atlas,
                        author_name="Test Author",
                        style_dna=None,
                        verbose=False
                    )

    # Verify that we requested enough examples
    # population_size=5, so we should request at least max(20, 5*4) = 20
    assert len(requested_top_k) > 0, "Should have requested examples"
    assert requested_top_k[0] >= 20, \
        f"Should request at least 20 examples (got {requested_top_k[0]}). " \
        f"Formula: max(20, population_size * 4) = max(20, 20) = 20"

    print(f"✓ test_fetches_enough_examples passed")
    print(f"  Requested {requested_top_k[0]} examples (expected >= 20)")


if __name__ == '__main__':
    test_candidates_use_different_seed_sentences()
    test_candidates_are_diverse()
    test_fetches_enough_examples()
    print("\n✅ All candidate diversity tests passed!")

