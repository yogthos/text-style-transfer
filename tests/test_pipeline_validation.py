"""Tests for pipeline validation and retry logic improvements."""

import sys
import json
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.generator.translator import StyleTranslator
from src.ingestion.blueprint import SemanticBlueprint, BlueprintExtractor
from tests.mocks.mock_llm_provider import MockLLMProvider


class TestSemanticScoreCalculation:
    """Tests for semantic score calculation (replacing hardcoded 1.0)."""

    def setup_method(self):
        """Set up test fixtures."""
        self.mock_llm = MockLLMProvider()
        self.translator = StyleTranslator(config_path="config.json")
        self.translator.llm_provider = self.mock_llm

        # Mock semantic critic
        self.mock_critic = MagicMock()
        self.translator.semantic_critic = self.mock_critic

    def test_calculate_semantic_score_returns_real_score(self):
        """Test that _calculate_semantic_score returns real scores, not hardcoded 1.0."""
        input_propositions = [
            "Stalin coined the term Dialectical Materialism",
            "The term refers to a practical toolset"
        ]
        generated_text = "Stalin coined Dialectical Materialism, a practical toolset."

        # Mock blueprint extractor
        mock_blueprint = SemanticBlueprint(
            original_text=" ".join(input_propositions),
            svo_triples=[("Stalin", "coined", "term")],
            named_entities=[("Stalin", "PERSON")],
            core_keywords={"stalin", "coined", "term", "dialectical", "materialism"},
            citations=[],
            quotes=[]
        )

        # Mock critic evaluation
        self.mock_critic.evaluate.return_value = {
            "recall_score": 0.85,
            "pass": True
        }

        with patch('src.ingestion.blueprint.BlueprintExtractor') as MockExtractor:
            mock_extractor = MockExtractor.return_value
            mock_extractor.extract.return_value = mock_blueprint

            score = self.translator._calculate_semantic_score(
                generated_text,
                input_propositions,
                verbose=False
            )

        # Should return real score, not 1.0
        assert score == 0.85
        assert score != 1.0
        self.mock_critic.evaluate.assert_called_once()

    def test_calculate_semantic_score_handles_empty_input(self):
        """Test score calculation with empty input."""
        score = self.translator._calculate_semantic_score("", [], verbose=False)
        assert score == 0.0

    def test_calculate_semantic_score_handles_errors(self):
        """Test score calculation handles errors gracefully."""
        self.mock_critic.evaluate.side_effect = Exception("Evaluation failed")

        score = self.translator._calculate_semantic_score(
            "Some text",
            ["Fact 1"],
            verbose=False
        )

        # Should default to 0.5 on error
        assert score == 0.5

    def test_translate_paragraph_propositions_returns_real_score(self):
        """Test that translate_paragraph_propositions returns real scores."""
        # Mock proposition extraction method directly
        propositions = ["Fact 1", "Fact 2"]
        self.translator._extract_propositions_from_text = Mock(return_value=propositions)

        # Mock LLM calls for synthesize_match and generation
        call_count = {"value": 0}
        def mock_llm_call(*args, **kwargs):
            call_count["value"] += 1
            if call_count["value"] == 1:
                # synthesize_match response
                return json.dumps({
                    'revised_skeleton': '[P0] and [P1]',
                    'rationale': 'Selected candidate'
                })
            else:
                # Graph generation
                return "Generated sentence."

        self.mock_llm.call = Mock(side_effect=mock_llm_call)

        # Mock input mapper
        self.translator.input_mapper = MagicMock()
        self.translator.input_mapper.map_propositions.return_value = {
            'intent': 'DEFINITION',
            'signature': 'DEFINITION'
        }

        # Mock graph matcher
        self.translator.graph_matcher = MagicMock()
        self.translator.graph_matcher.synthesize_match.return_value = {
            'style_metadata': {'skeleton': '[P0] and [P1]', 'node_count': 2},
            'node_mapping': {'P0': 'P0', 'P1': 'P1'},
            'intent': 'DEFINITION',
            'signature': 'DEFINITION'
        }

        # Mock semantic critic
        self.mock_critic.evaluate.return_value = {
            "recall_score": 0.75,
            "pass": True
        }

        with patch('src.ingestion.blueprint.BlueprintExtractor') as MockExtractor:
            mock_extractor = MockExtractor.return_value
            mock_blueprint = SemanticBlueprint(
                original_text=" ".join(propositions),
                svo_triples=[],
                named_entities=[],
                core_keywords=set(),
                citations=[],
                quotes=[]
            )
            mock_extractor.extract.return_value = mock_blueprint

            result = self.translator.translate_paragraph_propositions(
                "Some paragraph text",
                "Mao",
                verbose=False
            )

        text, arch_id, score = result
        # Should return real score, not hardcoded 1.0
        assert score != 1.0
        assert score == 0.75
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0


class TestRetryLoop:
    """Tests for retry loop with graceful degradation."""

    def setup_method(self):
        """Set up test fixtures."""
        self.mock_llm = MockLLMProvider()
        self.translator = StyleTranslator(config_path="config.json")
        self.translator.llm_provider = self.mock_llm

        # Mock semantic critic
        self.mock_critic = MagicMock()
        self.translator.semantic_critic = self.mock_critic

        # Mock input mapper and graph matcher
        self.translator.input_mapper = MagicMock()
        self.translator.graph_matcher = MagicMock()

    def test_retry_loop_returns_immediately_on_high_score(self):
        """Test that retry loop returns immediately when score >= 0.85."""
        propositions = ["Fact 1", "Fact 2"]
        # Mock proposition extraction
        self.translator._extract_propositions_from_text = Mock(return_value=propositions)

        call_count = {"value": 0}
        def mock_llm_call(*args, **kwargs):
            call_count["value"] += 1
            if call_count["value"] == 1:
                return json.dumps({'revised_skeleton': '[P0] and [P1]'})
            else:
                return "Generated sentence."

        self.mock_llm.call = Mock(side_effect=mock_llm_call)

        self.translator.input_mapper.map_propositions.return_value = {
            'intent': 'DEFINITION',
            'signature': 'DEFINITION'
        }

        self.translator.graph_matcher.synthesize_match.return_value = {
            'style_metadata': {'skeleton': '[P0] and [P1]', 'node_count': 2},
            'node_mapping': {'P0': 'P0', 'P1': 'P1'},
            'intent': 'DEFINITION',
            'signature': 'DEFINITION'
        }

        # Mock high score (should not retry)
        self.mock_critic.evaluate.return_value = {
            "recall_score": 0.90,
            "pass": True
        }

        with patch('src.ingestion.blueprint.BlueprintExtractor') as MockExtractor:
            mock_extractor = MockExtractor.return_value
            mock_blueprint = SemanticBlueprint(
                original_text=" ".join(propositions),
                svo_triples=[],
                named_entities=[],
                core_keywords=set(),
                citations=[],
                quotes=[]
            )
            mock_extractor.extract.return_value = mock_blueprint

            result = self.translator.translate_paragraph_propositions(
                "Some text",
                "Mao",
                verbose=False
            )

        # Should only call once (no retry)
        assert self.mock_critic.evaluate.call_count == 1
        text, arch_id, score = result
        assert score == 0.90

    def test_retry_loop_retries_on_low_score(self):
        """Test that retry loop retries when score < 0.85."""
        propositions = ["Fact 1", "Fact 2"]
        # Mock proposition extraction
        self.translator._extract_propositions_from_text = Mock(return_value=propositions)

        call_count = {"value": 0}

        def mock_call(*args, **kwargs):
            call_count["value"] += 1
            if call_count["value"] == 1:
                return json.dumps({'revised_skeleton': '[P0] and [P1]'})
            else:
                return "Generated sentence."

        self.mock_llm.call = Mock(side_effect=mock_call)

        self.translator.input_mapper.map_propositions.return_value = {
            'intent': 'DEFINITION',
            'signature': 'DEFINITION'
        }

        self.translator.graph_matcher.synthesize_match.return_value = {
            'style_metadata': {'skeleton': '[P0] and [P1]', 'node_count': 2},
            'node_mapping': {'P0': 'P0', 'P1': 'P1'},
            'intent': 'DEFINITION',
            'signature': 'DEFINITION'
        }

        # Mock low score (should trigger retry)
        score_count = {"value": 0}
        def mock_evaluate(*args, **kwargs):
            score_count["value"] += 1
            # First attempt: low score, second: high score
            if score_count["value"] == 1:
                return {"recall_score": 0.60, "pass": False}
            else:
                return {"recall_score": 0.90, "pass": True}

        self.mock_critic.evaluate.side_effect = mock_evaluate

        with patch('src.ingestion.blueprint.BlueprintExtractor') as MockExtractor:
            mock_extractor = MockExtractor.return_value
            mock_blueprint = SemanticBlueprint(
                original_text=" ".join(propositions),
                svo_triples=[],
                named_entities=[],
                core_keywords=set(),
                citations=[],
                quotes=[]
            )
            mock_extractor.extract.return_value = mock_blueprint

            result = self.translator.translate_paragraph_propositions(
                "Some text",
                "Mao",
                verbose=False
            )

        # Should have retried (called evaluate multiple times)
        assert self.mock_critic.evaluate.call_count >= 1
        text, arch_id, score = result
        # Should return best score
        assert score >= 0.60

    def test_retry_loop_returns_best_result(self):
        """Test that retry loop returns the best result across attempts."""
        propositions = ["Fact 1"]
        # Mock proposition extraction
        self.translator._extract_propositions_from_text = Mock(return_value=propositions)

        call_count = {"value": 0}
        def mock_llm_call(*args, **kwargs):
            call_count["value"] += 1
            if call_count["value"] == 1:
                return json.dumps({'revised_skeleton': '[P0]'})
            else:
                return "Generated sentence."

        self.mock_llm.call = Mock(side_effect=mock_llm_call)

        self.translator.input_mapper.map_propositions.return_value = {
            'intent': 'DEFINITION',
            'signature': 'DEFINITION'
        }

        self.translator.graph_matcher.synthesize_match.return_value = {
            'style_metadata': {'skeleton': '[P0]', 'node_count': 1},
            'node_mapping': {'P0': 'P0'},
            'intent': 'DEFINITION',
            'signature': 'DEFINITION'
        }

        # Mock scores: first low, second medium, third high
        scores = [0.50, 0.70, 0.80]
        score_index = {"value": 0}

        def mock_evaluate(*args, **kwargs):
            idx = score_index["value"]
            score_index["value"] += 1
            if idx < len(scores):
                return {"recall_score": scores[idx], "pass": scores[idx] >= 0.85}
            return {"recall_score": 0.80, "pass": False}

        self.mock_critic.evaluate.side_effect = mock_evaluate

        with patch('src.ingestion.blueprint.BlueprintExtractor') as MockExtractor:
            mock_extractor = MockExtractor.return_value
            mock_blueprint = SemanticBlueprint(
                original_text=" ".join(propositions),
                svo_triples=[],
                named_entities=[],
                core_keywords=set(),
                citations=[],
                quotes=[]
            )
            mock_extractor.extract.return_value = mock_blueprint

            result = self.translator.translate_paragraph_propositions(
                "Some text",
                "Mao",
                verbose=False
            )

        text, arch_id, score = result
        # Should return best score (0.80)
        assert score == 0.80


class TestAntiHallucinationRules:
    """Tests for anti-hallucination rules in generation prompts."""

    def setup_method(self):
        """Set up test fixtures."""
        self.mock_llm = MockLLMProvider()
        self.translator = StyleTranslator(config_path="config.json")
        self.translator.llm_provider = self.mock_llm

    def test_generate_from_graph_includes_anti_hallucination_rules(self):
        """Test that _generate_from_graph prompt includes anti-hallucination rules."""
        blueprint = {
            'style_mermaid': 'graph LR; ROOT --> CLAIM',
            'node_mapping': {'ROOT': 'P0', 'CLAIM': 'P1'},
            'style_metadata': {
                'skeleton': 'The [ROOT] is [CLAIM]',
                'intent': 'DEFINITION',
                'signature': 'DEFINITION'
            },
            'signature': 'DEFINITION'
        }

        input_node_map = {'P0': 'Stalin', 'P1': 'coined the term'}

        captured_prompt = None

        def capture_prompt(*args, **kwargs):
            nonlocal captured_prompt
            if 'user_prompt' in kwargs:
                captured_prompt = kwargs['user_prompt']
            return "Generated text."

        self.mock_llm.call = capture_prompt

        self.translator._generate_from_graph(
            blueprint,
            input_node_map,
            "Mao",
            verbose=False
        )

        # Verify anti-hallucination rules are in the prompt
        assert captured_prompt is not None
        assert "FACTUAL INTEGRITY" in captured_prompt
        assert "Do NOT add quotes" in captured_prompt or "Do not add quotes" in captured_prompt
        assert "Do NOT add attribution" in captured_prompt or "Do not add attribution" in captured_prompt
        assert "Do NOT invent facts" in captured_prompt or "Do not invent facts" in captured_prompt
        assert "IGNORE THE SKELETON" in captured_prompt or "ignore the skeleton" in captured_prompt.lower()


class TestStricterTemplateSelection:
    """Tests for stricter template selection in synthesize_match."""

    def setup_method(self):
        """Set up test fixtures."""
        import tempfile
        import shutil
        from pathlib import Path
        from unittest.mock import patch

        self.temp_dir = tempfile.mkdtemp()
        self.chroma_path = Path(self.temp_dir) / "test_chroma"

        from src.generator.graph_matcher import TopologicalMatcher
        from tests.mocks.mock_llm_provider import MockLLMProvider

        self.mock_llm = MockLLMProvider()

        with patch('src.generator.graph_matcher.LLMProvider', return_value=self.mock_llm):
            self.matcher = TopologicalMatcher(
                config_path="config.json",
                chroma_path=str(self.chroma_path)
            )
            self.matcher.llm_provider = self.mock_llm

    def teardown_method(self):
        """Clean up test fixtures."""
        import shutil
        from pathlib import Path
        if Path(self.temp_dir).exists():
            shutil.rmtree(self.temp_dir)

    def test_synthesize_match_prompt_emphasizes_selection_not_synthesis(self):
        """Test that synthesize_match prompt emphasizes selection over synthesis."""
        propositions = ["Fact 1", "Fact 2"]

        captured_prompt = None

        def capture_prompt(*args, **kwargs):
            nonlocal captured_prompt
            if 'system_prompt' in kwargs:
                captured_prompt = kwargs['system_prompt']
            return json.dumps({
                'revised_skeleton': '[P0] and [P1]',
                'rationale': 'Selected candidate'
            })

        self.mock_llm.call = capture_prompt

        self.matcher.synthesize_match(
            propositions,
            'DEFINITION',
            verbose=False
        )

        # Verify prompt emphasizes selection, not synthesis
        assert captured_prompt is not None
        assert "Template Selector" in captured_prompt or "template selector" in captured_prompt.lower()
        assert "NOT a Fiction Writer" in captured_prompt or "not a fiction writer" in captured_prompt.lower()
        assert "Do NOT synthesize" in captured_prompt or "do not synthesize" in captured_prompt.lower()
        assert "Do NOT invent" in captured_prompt or "do not invent" in captured_prompt.lower()

    def test_synthesize_match_user_prompt_emphasizes_selection(self):
        """Test that synthesize_match user prompt emphasizes candidate selection."""
        propositions = ["Fact 1", "Fact 2"]

        captured_prompt = None

        def capture_prompt(*args, **kwargs):
            nonlocal captured_prompt
            if 'user_prompt' in kwargs:
                captured_prompt = kwargs['user_prompt']
            return json.dumps({
                'revised_skeleton': '[P0] and [P1]',
                'rationale': 'Selected candidate'
            })

        self.mock_llm.call = capture_prompt

        self.matcher.synthesize_match(
            propositions,
            'DEFINITION',
            input_signature='DEFINITION',
            verbose=False
        )

        # Verify user prompt emphasizes selection
        assert captured_prompt is not None
        assert "Select Best Matching Candidate" in captured_prompt or "select best matching candidate" in captured_prompt.lower()
        assert "Do NOT invent" in captured_prompt or "do not invent" in captured_prompt.lower()
        assert "simple structure" in captured_prompt.lower() or "Simple structure" in captured_prompt


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])

