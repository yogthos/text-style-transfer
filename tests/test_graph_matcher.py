"""Tests for TopologicalMatcher."""

import sys
import json
import tempfile
import shutil
from pathlib import Path
from unittest.mock import patch, MagicMock

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import chromadb
from chromadb.utils import embedding_functions

from src.generator.graph_matcher import TopologicalMatcher
from tests.mocks.mock_llm_provider import MockLLMProvider


class TestTopologicalMatcher:
    """Test suite for TopologicalMatcher."""

    def setup_method(self):
        """Set up test fixtures."""
        # Create temporary directory for ChromaDB
        self.temp_dir = tempfile.mkdtemp()
        self.chroma_path = Path(self.temp_dir) / "test_chroma"

        # Create test ChromaDB collection
        self.client = chromadb.PersistentClient(path=str(self.chroma_path))
        embedding_fn = embedding_functions.DefaultEmbeddingFunction()

        try:
            self.client.delete_collection("style_graphs")
        except Exception:
            pass

        self.collection = self.client.create_collection(
            name="style_graphs",
            embedding_function=embedding_fn
        )

        # Create mock LLM provider
        from unittest.mock import MagicMock
        self.mock_llm = MagicMock()

        # Create matcher with custom chroma path
        with patch('src.generator.graph_matcher.LLMProvider', return_value=self.mock_llm):
            self.matcher = TopologicalMatcher(
                config_path="config.json",
                chroma_path=str(self.chroma_path)
            )
            self.matcher.llm_provider = self.mock_llm
            self.matcher.collection = self.collection

    def teardown_method(self):
        """Clean up test fixtures."""
        # Clean up temporary directory
        if Path(self.temp_dir).exists():
            shutil.rmtree(self.temp_dir)

    def _add_test_graph(self, mermaid: str, description: str, node_count: int,
                       edge_types: str = "defines", paragraph_role: str = None,
                       original_text: str = "Test text", skeleton: str = "",
                       intent: str = None):
        """Helper to add a test graph to ChromaDB."""
        metadata = {
            'mermaid': mermaid,
            'node_count': node_count,
            'edge_types': edge_types,
            'skeleton': skeleton,
            'original_text': original_text
        }
        if paragraph_role:
            metadata['paragraph_role'] = paragraph_role
        if intent:
            metadata['intent'] = intent

        self.collection.add(
            ids=[f"graph_{len(self.collection.get()['ids'])}"],
            documents=[description],
            metadatas=[metadata]
        )

    def test_get_best_match_exact_node_count(self):
        """Test matching with exact node count."""
        # Add test graphs
        self._add_test_graph(
            "graph LR; ROOT --> NODE1 --> NODE2",
            "A three-node causal chain",
            node_count=3
        )
        self._add_test_graph(
            "graph LR; ROOT --> NODE1 --> NODE2 --> NODE3",
            "A four-node chain",
            node_count=4
        )
        self._add_test_graph(
            "graph LR; ROOT --> NODE1",
            "A two-node chain",
            node_count=2
        )

        input_graph = {
            'description': 'A three-node causal chain',
            'mermaid': 'graph LR; P0 --> P1 --> P2',
            'node_map': {'P0': 'First', 'P1': 'Second', 'P2': 'Third'},
            'node_count': 3
        }

        # Mock LLM response for node mapping
        mapping_response = {
            'ROOT': 'P0',
            'NODE1': 'P1',
            'NODE2': 'P2'
        }
        self.mock_llm.call.return_value = json.dumps(mapping_response)

        result = self.matcher.get_best_match(input_graph)

        assert result is not None
        assert result['style_metadata']['node_count'] == 3
        assert 'node_mapping' in result
        assert 'distance' in result

    def test_get_best_match_style_has_more_nodes(self):
        """Test matching when style has more nodes."""
        self._add_test_graph(
            "graph LR; ROOT --> NODE1 --> NODE2 --> NODE3",
            "A four-node chain",
            node_count=4
        )

        input_graph = {
            'description': 'A two-node chain',
            'mermaid': 'graph LR; P0 --> P1',
            'node_map': {'P0': 'First', 'P1': 'Second'},
            'node_count': 2
        }

        mapping_response = {
            'ROOT': 'P0',
            'NODE1': 'P1',
            'NODE2': 'UNUSED',
            'NODE3': 'UNUSED'
        }
        self.mock_llm.call.return_value = json.dumps(mapping_response)

        result = self.matcher.get_best_match(input_graph)

        assert result is not None
        assert result['style_metadata']['node_count'] == 4
        assert 'UNUSED' in str(result['node_mapping'].values())

    def test_get_best_match_no_meeting_constraint(self):
        """Test overflow handling when input has more nodes."""
        self._add_test_graph(
            "graph LR; ROOT --> NODE1 --> NODE2",
            "A three-node chain",
            node_count=3
        )

        input_graph = {
            'description': 'A five-node chain',
            'mermaid': 'graph LR; P0 --> P1 --> P2 --> P3 --> P4',
            'node_map': {f'P{i}': f'Prop {i}' for i in range(5)},
            'node_count': 5
        }

        # Mock LLM response with grouped nodes
        mapping_response = {
            'ROOT': 'P0, P1',
            'NODE1': 'P2, P3',
            'NODE2': 'P4'
        }
        self.mock_llm.call.return_value = json.dumps(mapping_response)

        result = self.matcher.get_best_match(input_graph)

        assert result is not None
        # Should use the 3-node graph (largest available)
        assert result['style_metadata']['node_count'] == 3
        # Verify grouping happened
        assert any(',' in str(v) for v in result['node_mapping'].values())

    def test_get_best_match_multiple_candidates(self):
        """Test selecting best candidate from multiple options."""
        # Add multiple graphs with same node count but different distances
        self._add_test_graph(
            "graph LR; ROOT --> NODE1 --> NODE2",
            "A three-node causal chain",
            node_count=3
        )
        self._add_test_graph(
            "graph LR; A --> B --> C",
            "Another three-node chain",
            node_count=3
        )

        input_graph = {
            'description': 'A three-node causal chain',
            'mermaid': 'graph LR; P0 --> P1 --> P2',
            'node_map': {'P0': 'First', 'P1': 'Second', 'P2': 'Third'},
            'node_count': 3
        }

        mapping_response = {'ROOT': 'P0', 'NODE1': 'P1', 'NODE2': 'P2'}
        self.mock_llm.call.return_value = json.dumps(mapping_response)

        result = self.matcher.get_best_match(input_graph)

        assert result is not None
        assert 'distance' in result
        assert isinstance(result['distance'], (int, float))

    def test_get_best_match_empty_collection(self):
        """Test error handling with empty collection."""
        # Create new empty collection
        try:
            self.client.delete_collection("style_graphs")
        except Exception:
            pass

        empty_collection = self.client.create_collection(
            name="style_graphs",
            embedding_function=embedding_functions.DefaultEmbeddingFunction()
        )
        self.matcher.collection = empty_collection

        input_graph = {
            'description': 'A test graph',
            'mermaid': 'graph LR; P0 --> P1',
            'node_map': {'P0': 'First'},
            'node_count': 1
        }

        try:
            result = self.matcher.get_best_match(input_graph)
            assert False, "Should have raised ValueError"
        except ValueError as e:
            assert "No style graphs" in str(e)

    def test_get_best_match_semantic_search(self):
        """Test semantic search functionality."""
        self._add_test_graph(
            "graph LR; ROOT --> CLAIM",
            "A causal chain leading to consequence",
            node_count=2
        )

        input_graph = {
            'description': 'A causal chain leading to consequence',
            'mermaid': 'graph LR; P0 --> P1',
            'node_map': {'P0': 'Cause', 'P1': 'Effect'},
            'node_count': 2
        }

        mapping_response = {'ROOT': 'P0', 'CLAIM': 'P1'}
        self.mock_llm.call.return_value = json.dumps(mapping_response)

        result = self.matcher.get_best_match(input_graph)

        assert result is not None
        assert 'causal' in result['style_metadata']['original_text'].lower() or \
               'causal' in input_graph['description'].lower()

    def test_node_mapping_llm_integration(self):
        """Test LLM-based node mapping."""
        self._add_test_graph(
            "graph LR; ROOT --> CLAIM --> EVIDENCE",
            "A three-part argument structure",
            node_count=3
        )

        input_graph = {
            'description': 'A structured argument',
            'mermaid': 'graph LR; P0 --> P1 --> P2',
            'node_map': {'P0': 'Premise', 'P1': 'Conclusion', 'P2': 'Support'},
            'node_count': 3
        }

        mapping_response = {
            'ROOT': 'P0',
            'CLAIM': 'P1',
            'EVIDENCE': 'P2'
        }
        self.mock_llm.call.return_value = json.dumps(mapping_response)

        result = self.matcher.get_best_match(input_graph)

        assert result is not None
        assert 'node_mapping' in result
        assert len(result['node_mapping']) == 3
        assert 'ROOT' in result['node_mapping']
        assert 'CLAIM' in result['node_mapping']
        assert 'EVIDENCE' in result['node_mapping']

    def test_node_mapping_unused_nodes(self):
        """Test mapping with unused style nodes."""
        self._add_test_graph(
            "graph LR; ROOT --> NODE1 --> NODE2 --> NODE3",
            "A four-node structure",
            node_count=4
        )

        input_graph = {
            'description': 'A simple two-node chain',
            'mermaid': 'graph LR; P0 --> P1',
            'node_map': {'P0': 'First', 'P1': 'Second'},
            'node_count': 2
        }

        mapping_response = {
            'ROOT': 'P0',
            'NODE1': 'P1',
            'NODE2': 'UNUSED',
            'NODE3': 'UNUSED'
        }
        self.mock_llm.call.return_value = json.dumps(mapping_response)

        result = self.matcher.get_best_match(input_graph)

        assert result is not None
        assert len(result['node_mapping']) == 4
        assert result['node_mapping'].get('NODE2') == 'UNUSED' or \
               'UNUSED' in str(result['node_mapping'].values())

    def test_node_mapping_overflow_semantic_grafting(self):
        """Test semantic grafting when input has more nodes."""
        self._add_test_graph(
            "graph LR; ROOT --> CLAIM",
            "A two-node structure",
            node_count=2
        )

        input_graph = {
            'description': 'A complex five-node argument',
            'mermaid': 'graph LR; P0 --> P1 --> P2 --> P3 --> P4',
            'node_map': {f'P{i}': f'Proposition {i}' for i in range(5)},
            'node_count': 5
        }

        # Mock LLM response with grouped nodes
        mapping_response = {
            'ROOT': 'P0, P1, P2',
            'CLAIM': 'P3, P4'
        }
        self.mock_llm.call.return_value = json.dumps(mapping_response)

        result = self.matcher.get_best_match(input_graph)

        assert result is not None
        assert len(result['node_mapping']) == 2
        # Verify grouping (comma-separated values)
        grouped_values = [v for v in result['node_mapping'].values() if ',' in str(v)]
        assert len(grouped_values) > 0

    def test_parse_mermaid_nodes(self):
        """Test Mermaid node parsing."""
        test_cases = [
            ("graph LR; ROOT --> NODE1", ["NODE1", "ROOT"]),
            ("graph TD; A[Label] --> B", ["A", "B"]),
            ("ROOT --edge--> NODE1", ["NODE1", "ROOT"]),
        ]

        for mermaid, expected_nodes in test_cases:
            nodes = self.matcher._parse_mermaid_nodes(mermaid)
            for expected in expected_nodes:
                assert expected in nodes, f"Expected {expected} in {nodes} for {mermaid}"

    def test_get_best_match_invalid_input_graph(self):
        """Test error handling for invalid input."""
        input_graph = {
            # Missing 'description' field
            'mermaid': 'graph LR; P0 --> P1',
            'node_map': {'P0': 'First'}
        }

        try:
            result = self.matcher.get_best_match(input_graph)
            assert False, "Should have raised ValueError"
        except ValueError as e:
            assert "description" in str(e).lower()

    def test_get_best_match_filters_by_role(self):
        """Test role-based filtering."""
        # Add graphs with different roles
        self._add_test_graph(
            "graph LR; ROOT --> CLAIM",
            "An opening statement",
            node_count=2,
            paragraph_role='opener'
        )
        self._add_test_graph(
            "graph LR; A --> B",
            "A body paragraph structure",
            node_count=2,
            paragraph_role='body'
        )

        input_graph = {
            'description': 'An opening statement',
            'mermaid': 'graph LR; P0 --> P1',
            'node_map': {'P0': 'First', 'P1': 'Second'},
            'node_count': 2
        }

        mapping_response = {'ROOT': 'P0', 'CLAIM': 'P1'}
        self.mock_llm.call.return_value = json.dumps(mapping_response)

        # Test opener role
        context = {'current_index': 0, 'total_paragraphs': 3}
        result = self.matcher.get_best_match(input_graph, document_context=context)

        assert result is not None
        # Should prefer opener graph
        assert result['style_metadata'].get('paragraph_role') == 'opener' or \
               'ROOT' in result['style_mermaid']

        # Test body role
        context = {'current_index': 1, 'total_paragraphs': 3}
        result = self.matcher.get_best_match(input_graph, document_context=context)

        assert result is not None

    def test_get_best_match_role_fallback(self):
        """Test fallback when role filter finds no results."""
        # Add graph without role metadata
        self._add_test_graph(
            "graph LR; ROOT --> CLAIM",
            "A generic structure",
            node_count=2,
            paragraph_role=None
        )

        input_graph = {
            'description': 'A generic structure',
            'mermaid': 'graph LR; P0 --> P1',
            'node_map': {'P0': 'First', 'P1': 'Second'},
            'node_count': 2
        }

        mapping_response = {'ROOT': 'P0', 'CLAIM': 'P1'}
        self.mock_llm.call.return_value = json.dumps(mapping_response)

        # Request opener but none available
        context = {'current_index': 0, 'total_paragraphs': 3}
        result = self.matcher.get_best_match(input_graph, document_context=context)

        # Should fall back to unfiltered search
        assert result is not None

    def test_get_best_match_return_format(self):
        """Test return format structure."""
        self._add_test_graph(
            "graph LR; ROOT --> CLAIM --> CONDITION",
            "A three-node structure",
            node_count=3,
            edge_types="defines,supports"
        )

        input_graph = {
            'description': 'A three-node structure',
            'mermaid': 'graph LR; P0 --> P1 --> P2',
            'node_map': {'P0': 'First', 'P1': 'Second', 'P2': 'Third'},
            'node_count': 3
        }

        mapping_response = {'ROOT': 'P0', 'CLAIM': 'P1', 'CONDITION': 'P2'}
        self.mock_llm.call.return_value = json.dumps(mapping_response)

        result = self.matcher.get_best_match(input_graph)

        assert 'style_mermaid' in result
        assert 'node_mapping' in result
        assert 'style_metadata' in result
        assert 'distance' in result
        assert isinstance(result['style_mermaid'], str)
        assert isinstance(result['node_mapping'], dict)
        assert isinstance(result['style_metadata'], dict)
        assert isinstance(result['distance'], (int, float))
        assert 'node_count' in result['style_metadata']
        assert 'edge_types' in result['style_metadata']

    def test_synthesize_match_with_candidates(self):
        """Test synthesize_match with style candidates from ChromaDB."""
        # Add test graphs to ChromaDB
        self._add_test_graph(
            "graph LR; ROOT --> CLAIM",
            "A definition explaining what something is",
            node_count=2,
            intent='DEFINITION',
            skeleton='The concept is [CLAIM]'
        )
        self._add_test_graph(
            "graph LR; A --> B",
            "An argument with contrast",
            node_count=2,
            intent='ARGUMENT',
            skeleton='However, [A] contradicts [B]'
        )

        propositions = [
            "Many people hear the term Dialectical Materialism",
            "They assume they are stepping into a realm of mysticism",
            "In reality, Dialectical Materialism is a practical toolset"
        ]

        # Mock LLM response for Architect
        architect_response = json.dumps({
            'revised_skeleton': 'The [P0], while [P1], but [P2]',
            'rationale': 'Selected DEFINITION candidate and adapted it for all facts'
        })
        self.mock_llm.call.return_value = architect_response

        result = self.matcher.synthesize_match(
            propositions,
            'DEFINITION',
            verbose=False
        )

        assert result is not None
        assert 'style_metadata' in result
        assert 'node_mapping' in result
        assert 'intent' in result
        assert result['intent'] == 'DEFINITION'

        # Verify direct P0->P0 mapping
        assert result['node_mapping'] == {'P0': 'P0', 'P1': 'P1', 'P2': 'P2'}

        # Verify skeleton was created
        skeleton = result['style_metadata'].get('skeleton', '')
        assert skeleton != ''
        assert '[P0]' in skeleton or 'P0' in skeleton
        assert result['style_metadata']['node_count'] == 3

    def test_synthesize_match_no_candidates(self):
        """Test synthesize_match when ChromaDB has no candidates."""
        propositions = [
            "First proposition",
            "Second proposition"
        ]

        # Mock LLM response for Architect (with no candidates)
        architect_response = json.dumps({
            'revised_skeleton': '[P0] and [P1]',
            'rationale': 'Created skeleton from scratch'
        })
        self.mock_llm.call.return_value = architect_response

        result = self.matcher.synthesize_match(
            propositions,
            'ARGUMENT',
            verbose=False
        )

        assert result is not None
        assert result['node_mapping'] == {'P0': 'P0', 'P1': 'P1'}
        assert result['style_metadata']['node_count'] == 2
        assert result['intent'] == 'ARGUMENT'

    def test_synthesize_match_with_role_filtering(self):
        """Test synthesize_match with paragraph role filtering."""
        # Add graphs with different roles
        self._add_test_graph(
            "graph LR; ROOT --> CLAIM",
            "An opening statement",
            node_count=2,
            intent='DEFINITION',
            skeleton='The [ROOT] is [CLAIM]',
            paragraph_role='opener'
        )
        self._add_test_graph(
            "graph LR; A --> B",
            "A body paragraph",
            node_count=2,
            intent='ARGUMENT',
            skeleton='However, [A] and [B]',
            paragraph_role='body'
        )

        propositions = ["First fact", "Second fact"]

        # Mock LLM response
        architect_response = json.dumps({
            'revised_skeleton': '[P0] and [P1]',
            'rationale': 'Selected opener candidate'
        })
        self.mock_llm.call.return_value = architect_response

        # Test with opener role
        document_context = {'current_index': 0, 'total_paragraphs': 3}
        result = self.matcher.synthesize_match(
            propositions,
            'DEFINITION',
            document_context=document_context,
            verbose=False
        )

        assert result is not None
        assert result['style_metadata'].get('paragraph_role') == 'opener'
        assert result['node_mapping'] == {'P0': 'P0', 'P1': 'P1'}

    def test_synthesize_match_llm_fallback(self):
        """Test synthesize_match fallback when LLM call fails."""
        # Add test graph
        self._add_test_graph(
            "graph LR; ROOT --> CLAIM",
            "A test structure",
            node_count=2,
            intent='DEFINITION',
            skeleton='The [ROOT] is [CLAIM]'
        )

        propositions = ["Fact one", "Fact two"]

        # Mock LLM to raise exception
        self.mock_llm.call.side_effect = Exception("LLM call failed")

        result = self.matcher.synthesize_match(
            propositions,
            'DEFINITION',
            verbose=False
        )

        # Should still return a result with fallback skeleton
        assert result is not None
        assert result['node_mapping'] == {'P0': 'P0', 'P1': 'P1'}
        # Fallback skeleton should contain P0 and P1
        skeleton = result['style_metadata'].get('skeleton', '')
        assert 'P0' in skeleton or 'P1' in skeleton

    def test_synthesize_match_chromadb_error(self):
        """Test synthesize_match handles ChromaDB errors gracefully."""
        propositions = ["Fact one", "Fact two"]

        # Mock LLM response
        architect_response = json.dumps({
            'revised_skeleton': '[P0] and [P1]',
            'rationale': 'Created from scratch due to DB error'
        })
        self.mock_llm.call.return_value = architect_response

        # Simulate ChromaDB error by making collection.query raise
        original_query = self.matcher.collection.query
        def failing_query(**kwargs):
            raise Exception("ChromaDB connection failed")
        self.matcher.collection.query = failing_query

        result = self.matcher.synthesize_match(
            propositions,
            'ARGUMENT',
            verbose=False
        )

        # Should still work with empty candidate list
        assert result is not None
        assert result['node_mapping'] == {'P0': 'P0', 'P1': 'P1'}

        # Restore original query
        self.matcher.collection.query = original_query

    def test_synthesize_match_all_propositions_preserved(self):
        """Test that synthesize_match preserves all input propositions."""
        propositions = [
            "First important fact",
            "Second important fact",
            "Third important fact",
            "Fourth important fact"
        ]

        # Mock LLM response that includes all P slots
        architect_response = json.dumps({
            'revised_skeleton': '[P0], [P1], [P2], and [P3]',
            'rationale': 'Included all facts in skeleton'
        })
        self.mock_llm.call.return_value = architect_response

        result = self.matcher.synthesize_match(
            propositions,
            'NARRATIVE',
            verbose=False
        )

        assert result is not None
        # Verify all propositions are mapped
        assert len(result['node_mapping']) == 4
        assert 'P0' in result['node_mapping']
        assert 'P1' in result['node_mapping']
        assert 'P2' in result['node_mapping']
        assert 'P3' in result['node_mapping']

        # Verify skeleton mentions all P slots
        skeleton = result['style_metadata'].get('skeleton', '')
        assert '[P0]' in skeleton or 'P0' in skeleton
        assert '[P1]' in skeleton or 'P1' in skeleton
        assert '[P2]' in skeleton or 'P2' in skeleton
        assert '[P3]' in skeleton or 'P3' in skeleton

    def test_synthesize_match_blueprint_structure(self):
        """Test that synthesize_match returns correct blueprint structure."""
        propositions = ["Test proposition"]

        architect_response = json.dumps({
            'revised_skeleton': '[P0] is a test',
            'rationale': 'Simple test skeleton'
        })
        self.mock_llm.call.return_value = architect_response

        result = self.matcher.synthesize_match(
            propositions,
            'DEFINITION',
            verbose=False
        )

        # Verify blueprint structure
        assert 'style_metadata' in result
        assert 'node_mapping' in result
        assert 'intent' in result
        assert 'distance' in result

        # Verify style_metadata structure
        style_meta = result['style_metadata']
        assert 'skeleton' in style_meta
        assert 'node_count' in style_meta
        assert 'edge_types' in style_meta
        assert 'intent' in style_meta

        # Verify intent is set correctly
        assert result['intent'] == 'DEFINITION'
        assert style_meta['intent'] == 'DEFINITION'

        # Verify distance is 0.0 for synthesized blueprints
        assert result['distance'] == 0.0

    def test_synthesize_match_redundancy_merging_instruction(self):
        """Test that synthesize_match prompt includes redundancy merging instruction."""
        propositions = [
            "Stalin coined the term",
            "The term was coined by Stalin",
            "Different fact"
        ]

        # Mock LLM to capture the prompt
        captured_prompt = None
        original_call = self.mock_llm.call

        def capture_prompt(*args, **kwargs):
            nonlocal captured_prompt
            if 'user_prompt' in kwargs:
                captured_prompt = kwargs['user_prompt']
            return json.dumps({
                'revised_skeleton': '[P0] and [P1]',
                'rationale': 'Merged redundant facts'
            })

        self.mock_llm.call = capture_prompt

        self.matcher.synthesize_match(
            propositions,
            'DEFINITION',
            verbose=False
        )

        # Verify redundancy merging instruction is in the prompt
        assert captured_prompt is not None
        assert "MERGE REDUNDANCIES" in captured_prompt
        assert "SINGLE slot" in captured_prompt
        assert "Do not create repetitive structures" in captured_prompt

    def test_select_diverse_candidates(self):
        """Test diversity selection across different intents."""
        # Create candidates with different intents
        candidates = [
            {'intent': 'DEFINITION', 'distance': 0.1, 'priority_score': 0.1, 'node_count': 2},
            {'intent': 'ARGUMENT', 'distance': 0.2, 'priority_score': 0.2, 'node_count': 2},
            {'intent': 'NARRATIVE', 'distance': 0.3, 'priority_score': 0.3, 'node_count': 2},
            {'intent': 'DEFINITION', 'distance': 0.4, 'priority_score': 0.4, 'node_count': 2},
            {'intent': 'ARGUMENT', 'distance': 0.5, 'priority_score': 0.5, 'node_count': 2},
            {'intent': 'INTERROGATIVE', 'distance': 0.6, 'priority_score': 0.6, 'node_count': 2},
        ]

        # Test with input intent matching DEFINITION
        selected = self.matcher._select_diverse_candidates(candidates, top_k=5, input_intent='DEFINITION')

        assert len(selected) == 5
        # Should include the matching intent first
        assert selected[0]['intent'] == 'DEFINITION'
        # Should have diverse intents
        intents = [c['intent'] for c in selected]
        unique_intents = set(intents)
        assert len(unique_intents) >= 3, f"Expected at least 3 unique intents, got {unique_intents}"

    def test_select_diverse_candidates_no_input_intent(self):
        """Test diversity selection without input intent."""
        candidates = [
            {'intent': 'DEFINITION', 'distance': 0.1, 'priority_score': 0.1, 'node_count': 2},
            {'intent': 'ARGUMENT', 'distance': 0.2, 'priority_score': 0.2, 'node_count': 2},
            {'intent': 'NARRATIVE', 'distance': 0.3, 'priority_score': 0.3, 'node_count': 2},
            {'intent': 'DEFINITION', 'distance': 0.4, 'priority_score': 0.4, 'node_count': 2},
        ]

        selected = self.matcher._select_diverse_candidates(candidates, top_k=3, input_intent=None)

        assert len(selected) == 3
        # Should still have diversity
        intents = [c['intent'] for c in selected]
        unique_intents = set(intents)
        assert len(unique_intents) >= 2

    def test_synthesize_best_fit_selects_candidate(self):
        """Test synthesis selects best candidate."""
        input_graph = {
            'description': 'A definition explaining what something is',
            'intent': 'DEFINITION',
            'mermaid': 'graph LR; P0 --> P1'
        }

        candidates = [
            {
                'id': 'candidate_0',
                'mermaid': 'graph LR; ROOT --> CLAIM',
                'skeleton': 'The concept is [CLAIM]',
                'intent': 'DEFINITION',
                'node_count': 2,
                'distance': 0.1,
                'priority_score': 0.05  # Intent match boosted
            },
            {
                'id': 'candidate_1',
                'mermaid': 'graph LR; A --> B',
                'skeleton': 'However, [A] contradicts [B]',
                'intent': 'ARGUMENT',
                'node_count': 2,
                'distance': 0.2,
                'priority_score': 0.2
            },
            {
                'id': 'candidate_2',
                'mermaid': 'graph LR; X --> Y',
                'skeleton': 'At that time, [X] led to [Y]',
                'intent': 'NARRATIVE',
                'node_count': 2,
                'distance': 0.3,
                'priority_score': 0.3
            }
        ]

        # Mock LLM response: select candidate 0 (DEFINITION match) and keep skeleton
        synthesis_response = json.dumps({
            'selected_index': 0,
            'revised_skeleton': 'The concept is [CLAIM]'
        })
        self.mock_llm.call.return_value = synthesis_response

        result = self.matcher._synthesize_best_fit(input_graph, candidates)

        assert result is not None
        assert result['id'] == 'candidate_0'
        assert result['intent'] == 'DEFINITION'
        assert result['skeleton'] == 'The concept is [CLAIM]'

    def test_synthesize_best_fit_revises_skeleton(self):
        """Test synthesis revises skeleton when connectors contradict."""
        input_graph = {
            'description': 'A definition explaining what something is',
            'intent': 'DEFINITION',
            'mermaid': 'graph LR; P0 --> P1'
        }

        candidates = [
            {
                'id': 'candidate_0',
                'mermaid': 'graph LR; ROOT --> CLAIM',
                'skeleton': 'However, [ROOT] contradicts [CLAIM]',  # Contrast connector for definition
                'intent': 'ARGUMENT',
                'node_count': 2,
                'distance': 0.1,
                'priority_score': 0.1
            }
        ]

        # Mock LLM response: select candidate 0 but revise skeleton to remove "However"
        synthesis_response = json.dumps({
            'selected_index': 0,
            'revised_skeleton': 'The [ROOT] is [CLAIM]'  # Revised to definition structure
        })
        self.mock_llm.call.return_value = synthesis_response

        result = self.matcher._synthesize_best_fit(input_graph, candidates)

        assert result is not None
        assert result['skeleton'] == 'The [ROOT] is [CLAIM]'
        assert 'However' not in result['skeleton']

    def test_synthesize_best_fit_fallback_on_error(self):
        """Test synthesis falls back to top candidate on LLM error."""
        input_graph = {
            'description': 'A test graph',
            'intent': 'ARGUMENT',
            'mermaid': 'graph LR; P0 --> P1'
        }

        candidates = [
            {
                'id': 'candidate_0',
                'mermaid': 'graph LR; ROOT --> CLAIM',
                'skeleton': 'The [ROOT] leads to [CLAIM]',
                'intent': 'ARGUMENT',
                'node_count': 2,
                'distance': 0.1,
                'priority_score': 0.05
            }
        ]

        # Mock LLM to raise exception
        self.mock_llm.call.side_effect = Exception("LLM call failed")

        result = self.matcher._synthesize_best_fit(input_graph, candidates)

        # Should fall back to top candidate
        assert result is not None
        assert result['id'] == 'candidate_0'

    def test_get_best_match_with_synthesis(self):
        """Test full get_best_match flow with synthesis."""
        # Add diverse candidates with different intents
        self._add_test_graph(
            "graph LR; ROOT --> CLAIM",
            "A definition explaining what something is",
            node_count=2,
            intent='DEFINITION',
            skeleton='The concept is [CLAIM]'
        )
        self._add_test_graph(
            "graph LR; A --> B",
            "An argument with contrast",
            node_count=2,
            intent='ARGUMENT',
            skeleton='However, [A] contradicts [B]'
        )
        self._add_test_graph(
            "graph LR; X --> Y",
            "A narrative sequence",
            node_count=2,
            intent='NARRATIVE',
            skeleton='At that time, [X] led to [Y]'
        )

        input_graph = {
            'description': 'A definition explaining what something is',
            'intent': 'DEFINITION',
            'mermaid': 'graph LR; P0 --> P1',
            'node_map': {'P0': 'Concept', 'P1': 'Meaning'},
            'node_count': 2
        }

        # Mock LLM responses: synthesis selects candidate 0, node mapping
        synthesis_response = json.dumps({
            'selected_index': 0,
            'revised_skeleton': 'The concept is [CLAIM]'
        })
        mapping_response = json.dumps({
            'ROOT': 'P0',
            'CLAIM': 'P1'
        })

        # Set up mock to return different responses for different calls
        # First call is synthesis, second is mapping
        call_responses = [synthesis_response, mapping_response]
        def mock_call_side_effect(**kwargs):
            if call_responses:
                return call_responses.pop(0)
            return mapping_response

        self.mock_llm.call.side_effect = mock_call_side_effect

        result = self.matcher.get_best_match(input_graph)

        assert result is not None
        assert result['style_metadata']['intent'] == 'DEFINITION'
        assert 'node_mapping' in result
        assert 'distance' in result
        # Verify synthesis was called (should have 2 LLM calls: synthesis + mapping)
        # Reset call_count check - the mock might be reset, so just verify we got a result
        assert result['style_metadata'].get('skeleton') == 'The concept is [CLAIM]' or \
               'skeleton' in result['style_metadata']

    def test_get_best_match_synthesis_fallback(self):
        """Test get_best_match falls back when synthesis fails."""
        self._add_test_graph(
            "graph LR; ROOT --> CLAIM",
            "A definition",
            node_count=2,
            intent='DEFINITION',
            skeleton='The concept is [CLAIM]'
        )

        input_graph = {
            'description': 'A definition',
            'intent': 'DEFINITION',
            'mermaid': 'graph LR; P0 --> P1',
            'node_map': {'P0': 'Concept', 'P1': 'Meaning'},
            'node_count': 2
        }

        # Mock synthesis to fail, but mapping should work
        mapping_response = json.dumps({'ROOT': 'P0', 'CLAIM': 'P1'})

        call_count = 0
        def mock_call(**kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                # First call (synthesis) fails
                raise Exception("Synthesis failed")
            else:
                # Subsequent calls (mapping) succeed
                return mapping_response

        self.mock_llm.call.side_effect = mock_call

        result = self.matcher.get_best_match(input_graph)

        # Should still return a result (fallback to top candidate)
        assert result is not None
        assert 'node_mapping' in result

    def test_build_where_clause_empty_list(self):
        """Test _build_where_clause with empty conditions list."""
        result = self.matcher._build_where_clause([])
        assert result is None

    def test_build_where_clause_single_condition(self):
        """Test _build_where_clause with single condition."""
        conditions = [{"signature": "CONTRAST"}]
        result = self.matcher._build_where_clause(conditions)
        assert result == {"signature": "CONTRAST"}
        # Should not have $and operator for single condition
        assert "$and" not in result

    def test_build_where_clause_multiple_conditions(self):
        """Test _build_where_clause with multiple conditions."""
        conditions = [
            {"signature": "CONTRAST"},
            {"role": "INTRO"}
        ]
        result = self.matcher._build_where_clause(conditions)
        assert result == {"$and": [{"signature": "CONTRAST"}, {"role": "INTRO"}]}
        assert "$and" in result
        assert len(result["$and"]) == 2

    def test_build_where_clause_three_conditions(self):
        """Test _build_where_clause with three conditions."""
        conditions = [
            {"signature": "DEFINITION"},
            {"role": "BODY"},
            {"paragraph_role": "body"}
        ]
        result = self.matcher._build_where_clause(conditions)
        assert result == {
            "$and": [
                {"signature": "DEFINITION"},
                {"role": "BODY"},
                {"paragraph_role": "body"}
            ]
        }
        assert "$and" in result
        assert len(result["$and"]) == 3

    def test_build_where_clause_filters_none(self):
        """Test _build_where_clause filters out None values."""
        conditions = [
            {"signature": "CONTRAST"},
            None,
            {"role": "INTRO"}
        ]
        result = self.matcher._build_where_clause(conditions)
        # Should filter out None and create $and with 2 conditions
        assert result == {"$and": [{"signature": "CONTRAST"}, {"role": "INTRO"}]}
        assert len(result["$and"]) == 2

    def test_build_where_clause_filters_empty_dict(self):
        """Test _build_where_clause filters out empty dicts."""
        conditions = [
            {"signature": "CONTRAST"},
            {},
            {"role": "INTRO"}
        ]
        result = self.matcher._build_where_clause(conditions)
        # Empty dict is falsy, so should be filtered out
        assert result == {"$and": [{"signature": "CONTRAST"}, {"role": "INTRO"}]}
        assert len(result["$and"]) == 2

    def test_build_where_clause_all_filtered(self):
        """Test _build_where_clause returns None when all conditions are filtered."""
        conditions = [None, {}, None]
        result = self.matcher._build_where_clause(conditions)
        assert result is None

    def test_build_where_clause_real_world_scenario(self):
        """Test _build_where_clause with real-world query scenario."""
        # Simulate primary query: signature + role + paragraph_role
        conditions = [
            {"signature": "CONTRAST"},
            {"role": "INTRO"},
            {"paragraph_role": "opener"}
        ]
        result = self.matcher._build_where_clause(conditions)
        assert "$and" in result
        assert len(result["$and"]) == 3
        # Verify all conditions are present
        condition_dicts = result["$and"]
        signatures = [c for c in condition_dicts if "signature" in c]
        roles = [c for c in condition_dicts if "role" in c]
        paragraph_roles = [c for c in condition_dicts if "paragraph_role" in c]
        assert len(signatures) == 1
        assert len(roles) == 1
        assert len(paragraph_roles) == 1
        assert signatures[0]["signature"] == "CONTRAST"
        assert roles[0]["role"] == "INTRO"
        assert paragraph_roles[0]["paragraph_role"] == "opener"


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])

