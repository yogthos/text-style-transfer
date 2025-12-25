"""Tests for Graph-Based Skeleton Builder.

Tests the graph-to-graph transformation logic that builds skeletons
by traversing input graphs and mapping logical edges to stylistic connectors.
"""

import sys
import re
from pathlib import Path
from unittest.mock import patch, MagicMock
from typing import List, Dict, Any, Tuple

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.generator.graph_matcher import TopologicalMatcher
from tests.mocks.mock_llm_provider import MockLLMProvider


class MockMatcher(TopologicalMatcher):
    """Mock TopologicalMatcher that avoids loading spaCy models."""

    def __init__(self, *args, **kwargs):
        """Initialize with mocked NLP components."""
        # Don't call super().__init__ to avoid loading ChromaDB
        # Instead, set up minimal required attributes
        self._nlp_cache = None
        self.llm_provider = kwargs.get('llm_provider', MockLLMProvider())

    def _get_nlp(self):
        """Return None to avoid loading spaCy."""
        return None

    def _is_valid_connector(self, text: str) -> bool:
        """Simplified connector validation for testing."""
        if not text or not text.strip():
            return True
        # Simple heuristic: reject if too long or contains content words
        if len(text.split()) > 5:
            return False
        # Reject if contains common content words
        content_words = ['world', 'there', 'would', 'no', 'the', 'a', 'an']
        text_lower = text.lower()
        if any(word in text_lower for word in content_words if len(word) > 2):
            # Allow structural words
            structural_words = ['is', 'are', 'but', 'and', 'or', 'because', 'then']
            if not any(word in text_lower for word in structural_words):
                return False
        return True

    def _analyze_connector_grammar(self, connector_text: str) -> str:
        """Mock grammar analysis for testing."""
        if not connector_text or not connector_text.strip():
            return 'PUNCT'

        conn_lower = connector_text.lower()

        # Simple keyword-based classification
        if any(word in conn_lower for word in ['is', 'are', 'was', 'were', 'be', 'defines', 'constitutes']):
            return 'VERB'
        elif any(word in conn_lower for word in ['but', 'however', 'whereas', 'and', 'or', 'because', 'then']):
            return 'CONJ'
        elif ',' in connector_text or ';' in connector_text:
            return 'PUNCT'
        return 'OTHER'


def create_mermaid_graph(edges: List[Tuple[str, str, str]]) -> str:
    """Create a mermaid graph string from edge list.

    Args:
        edges: List of (source, target, label) tuples

    Returns:
        Mermaid graph string
    """
    edge_strings = []
    for source, target, label in edges:
        if label and label != 'SEQUENCE':
            edge_strings.append(f"{source} --{label.lower()}--> {target}")
        else:
            edge_strings.append(f"{source} --> {target}")
    return f"graph LR; {'; '.join(edge_strings)}"


def create_input_graph(mermaid: str, node_map: Dict[str, str] = None) -> Dict[str, Any]:
    """Create an input graph dictionary.

    Args:
        mermaid: Mermaid graph string
        node_map: Optional node mapping dictionary

    Returns:
        Input graph dictionary
    """
    # Extract nodes from mermaid
    nodes = set()
    edge_pattern = r'([A-Z_][A-Z0-9_]*)\s*--[^>]*-->\s*([A-Z_][A-Z0-9_]*)'
    matches = re.findall(edge_pattern, mermaid)
    for match in matches:
        nodes.add(match[0])
        nodes.add(match[1])

    simple_pattern = r'([A-Z_][A-Z0-9_]*)\s*-->\s*([A-Z_][A-Z0-9_]*)'
    simple_matches = re.findall(simple_pattern, mermaid)
    for match in simple_matches:
        nodes.add(match[0])
        nodes.add(match[1])

    if node_map is None:
        node_map = {node: f"Text for {node}" for node in sorted(nodes)}

    return {
        'mermaid': mermaid,
        'node_map': node_map,
        'node_count': len(nodes)
    }


class TestGraphSkeletonBuilder:
    """Test suite for graph-based skeleton building."""

    def setup_method(self):
        """Set up test fixtures."""
        self.matcher = MockMatcher(llm_provider=MockLLMProvider())

    # ========== Vocabulary Harvesting Tests ==========

    def test_harvest_categorizes_connectors(self):
        """Test that _harvest_style_vocab correctly categorizes connectors."""
        candidates = [
            {'skeleton': '[P0] but [P1]'},
            {'skeleton': '[P0] however [P1]'},
            {'skeleton': '[P0] because [P1]'},
            {'skeleton': '[P0] then [P1]'},
        ]

        style_vocab = self.matcher._harvest_style_vocab(candidates, verbose=False)

        # Check categorization
        assert 'but' in style_vocab['CONTRAST'] or any('but' in c for c in style_vocab['CONTRAST'])
        assert 'however' in style_vocab['CONTRAST'] or any('however' in c for c in style_vocab['CONTRAST'])
        assert 'because' in style_vocab['CAUSALITY'] or any('because' in c for c in style_vocab['CAUSALITY'])
        assert 'then' in style_vocab['SEQUENCE'] or any('then' in c for c in style_vocab['SEQUENCE'])

    def test_harvest_handles_empty_candidates(self):
        """Test that _harvest_style_vocab handles empty candidate list."""
        candidates = []
        style_vocab = self.matcher._harvest_style_vocab(candidates, verbose=False)

        # All buckets should be empty
        for logic_type, connectors in style_vocab.items():
            assert len(connectors) == 0, f"{logic_type} should be empty"

    def test_harvest_filters_invalid_connectors(self):
        """Test that _harvest_style_vocab filters out invalid connectors."""
        candidates = [
            {'skeleton': '[P0] but [P1]'},  # Valid
            {'skeleton': '[P0] there would be no world [P1]'},  # Invalid (content phrase)
        ]

        style_vocab = self.matcher._harvest_style_vocab(candidates, verbose=False)

        # Should only have 'but', not the invalid phrase
        contrast_connectors = ' '.join(style_vocab['CONTRAST'])
        assert 'but' in contrast_connectors
        assert 'there would be no world' not in contrast_connectors

    # ========== Skeleton Building Tests ==========

    def test_skeleton_simple_contrast(self):
        """Test building skeleton from simple contrast graph."""
        mermaid = "graph LR; P0 --contrast--> P1"
        input_graph = create_input_graph(mermaid)
        style_vocab = {'CONTRAST': [' but ']}

        skeleton = self.matcher._build_skeleton_from_graph(
            input_graph, style_vocab, 2, verbose=False
        )

        assert skeleton is not None
        assert '[P0]' in skeleton
        assert '[P1]' in skeleton
        assert ' but ' in skeleton
        # Should end with period
        assert skeleton.endswith('.')

    def test_skeleton_multi_hop_logic(self):
        """Test building skeleton from multi-hop graph with different logic types."""
        mermaid = "graph LR; P0 --contrast--> P1; P1 --cause--> P2"
        input_graph = create_input_graph(mermaid)
        style_vocab = {
            'CONTRAST': ['; however, '],
            'CAUSALITY': [' because ']
        }

        skeleton = self.matcher._build_skeleton_from_graph(
            input_graph, style_vocab, 3, verbose=False
        )

        assert skeleton is not None
        assert '[P0]' in skeleton
        assert '[P1]' in skeleton
        assert '[P2]' in skeleton
        assert '; however, ' in skeleton
        assert ' because ' in skeleton
        # Check order: P0 should come before P1, P1 before P2
        assert skeleton.index('[P0]') < skeleton.index('[P1]')
        assert skeleton.index('[P1]') < skeleton.index('[P2]')

    def test_skeleton_fallback_defaults(self):
        """Test that skeleton building falls back to defaults when vocab is empty."""
        mermaid = "graph LR; P0 --> P1"
        input_graph = create_input_graph(mermaid)
        style_vocab = {}  # Empty vocabulary

        skeleton = self.matcher._build_skeleton_from_graph(
            input_graph, style_vocab, 2, verbose=False
        )

        assert skeleton is not None
        assert '[P0]' in skeleton
        assert '[P1]' in skeleton
        # Should use SEQUENCE fallback (", then ")
        assert ', then ' in skeleton or skeleton.endswith('.')

    def test_skeleton_complex_definition(self):
        """Test building skeleton from complex definition graph."""
        mermaid = "graph LR; P0 --definition--> P1; P1 --> P2"
        input_graph = create_input_graph(mermaid)
        style_vocab = {
            'DEFINITION': [' is '],
            'SEQUENCE': [', ']
        }

        skeleton = self.matcher._build_skeleton_from_graph(
            input_graph, style_vocab, 3, verbose=False
        )

        assert skeleton is not None
        assert '[P0]' in skeleton
        assert '[P1]' in skeleton
        assert '[P2]' in skeleton
        assert ' is ' in skeleton
        # P1 should be connected to P2 with comma
        assert ', ' in skeleton or skeleton.count(',') > 0

    def test_skeleton_single_node(self):
        """Test building skeleton from graph with single node."""
        # Single node graphs have no edges, so we need to handle this case
        # The implementation currently requires at least one edge
        # For a single node, we can create a self-loop or just test the edge case
        mermaid = "graph LR; P0 --> P0"  # Self-loop to create an edge
        input_graph = create_input_graph(mermaid)
        style_vocab = {}

        skeleton = self.matcher._build_skeleton_from_graph(
            input_graph, style_vocab, 1, verbose=False
        )

        # With self-loop, should still work
        assert skeleton is not None
        assert '[P0]' in skeleton
        assert skeleton.endswith('.')

    def test_skeleton_handles_unknown_edge_label(self):
        """Test that skeleton building handles unknown edge labels gracefully."""
        mermaid = "graph LR; P0 --unknown--> P1"
        input_graph = create_input_graph(mermaid)
        style_vocab = {}  # Empty vocabulary

        skeleton = self.matcher._build_skeleton_from_graph(
            input_graph, style_vocab, 2, verbose=False
        )

        assert skeleton is not None
        assert '[P0]' in skeleton
        assert '[P1]' in skeleton
        # Should fall back to default connector

    # ========== Node Ordering & Topology Tests ==========

    def test_nodes_are_sorted_topologically(self):
        """Test that nodes are sorted correctly even if added in random order."""
        # Create graph where P1 depends on P0
        mermaid = "graph LR; P1 --cause--> P0"
        input_graph = create_input_graph(mermaid)
        style_vocab = {'CAUSALITY': [' because ']}

        skeleton = self.matcher._build_skeleton_from_graph(
            input_graph, style_vocab, 2, verbose=False
        )

        assert skeleton is not None
        # Nodes should be sorted by numeric index (P0 before P1)
        assert skeleton.index('[P0]') < skeleton.index('[P1]')

    def test_nodes_sorted_with_multiple_edges(self):
        """Test node sorting with complex graph structure."""
        mermaid = "graph LR; P2 --> P0; P1 --> P2"
        input_graph = create_input_graph(mermaid)
        style_vocab = {'SEQUENCE': [', then ']}

        skeleton = self.matcher._build_skeleton_from_graph(
            input_graph, style_vocab, 3, verbose=False
        )

        assert skeleton is not None
        # Should be sorted: P0, P1, P2
        assert skeleton.index('[P0]') < skeleton.index('[P1]')
        assert skeleton.index('[P1]') < skeleton.index('[P2]')

    # ========== Edge Parsing Tests ==========

    def test_parse_mermaid_edges_labeled(self):
        """Test parsing labeled edges from mermaid."""
        mermaid = "graph LR; P0 --contrast--> P1; P1 --cause--> P2"
        edges = self.matcher._parse_mermaid_edges(mermaid)

        assert len(edges) == 2
        assert ('P0', 'P1', 'CONTRAST') in edges
        assert ('P1', 'P2', 'CAUSALITY') in edges

    def test_parse_mermaid_edges_unlabeled(self):
        """Test parsing unlabeled edges (defaults to SEQUENCE)."""
        mermaid = "graph LR; P0 --> P1"
        edges = self.matcher._parse_mermaid_edges(mermaid)

        assert len(edges) == 1
        assert ('P0', 'P1', 'SEQUENCE') in edges

    def test_parse_mermaid_edges_mixed(self):
        """Test parsing mixed labeled and unlabeled edges."""
        mermaid = "graph LR; P0 --contrast--> P1; P1 --> P2"
        edges = self.matcher._parse_mermaid_edges(mermaid)

        assert len(edges) == 2
        assert ('P0', 'P1', 'CONTRAST') in edges
        assert ('P1', 'P2', 'SEQUENCE') in edges

    def test_parse_mermaid_edges_normalizes_labels(self):
        """Test that edge labels are normalized correctly."""
        mermaid = "graph LR; P0 --cause--> P1; P1 --define--> P2"
        edges = self.matcher._parse_mermaid_edges(mermaid)

        assert ('P0', 'P1', 'CAUSALITY') in edges  # 'cause' -> 'CAUSALITY'
        assert ('P1', 'P2', 'DEFINITION') in edges  # 'define' -> 'DEFINITION'

    # ========== Connector Selection Tests ==========

    def test_select_connector_uses_most_frequent(self):
        """Test that _select_connector selects most frequent connector."""
        style_vocab = {
            'CONTRAST': [' but ', ' however ', ' but ', ' but ']  # 'but' appears 3 times
        }

        connector = self.matcher._select_connector(style_vocab, 'CONTRAST', verbose=False)

        assert connector == ' but '  # Most frequent

    def test_select_connector_fallback(self):
        """Test that _select_connector falls back to defaults when vocab is empty."""
        style_vocab = {}

        connector = self.matcher._select_connector(style_vocab, 'CONTRAST', verbose=False)

        assert connector == " but "  # Fallback default

    def test_select_connector_unknown_type(self):
        """Test that _select_connector handles unknown logic types."""
        style_vocab = {}

        connector = self.matcher._select_connector(style_vocab, 'UNKNOWN_TYPE', verbose=False)

        assert connector == ", "  # Generic fallback

    # ========== Integration Tests ==========

    def test_full_pipeline_simple(self):
        """Test full pipeline: harvest vocab -> build skeleton."""
        # Create candidates
        candidates = [
            {'skeleton': '[P0] but [P1]'},
            {'skeleton': '[P0] however [P1]'},
        ]

        # Harvest vocabulary
        style_vocab = self.matcher._harvest_style_vocab(candidates, verbose=False)

        # Build skeleton
        mermaid = "graph LR; P0 --contrast--> P1"
        input_graph = create_input_graph(mermaid)
        skeleton = self.matcher._build_skeleton_from_graph(
            input_graph, style_vocab, 2, verbose=False
        )

        assert skeleton is not None
        assert '[P0]' in skeleton
        assert '[P1]' in skeleton
        # Should use one of the harvested connectors
        assert ('but' in skeleton or 'however' in skeleton)

    def test_full_pipeline_complex(self):
        """Test full pipeline with complex multi-edge graph."""
        # Create candidates with various connectors
        candidates = [
            {'skeleton': '[P0] but [P1]'},
            {'skeleton': '[P0] because [P1]'},
            {'skeleton': '[P0] then [P1]'},
        ]

        # Harvest vocabulary
        style_vocab = self.matcher._harvest_style_vocab(candidates, verbose=False)

        # Build skeleton from complex graph
        mermaid = "graph LR; P0 --contrast--> P1; P1 --cause--> P2"
        input_graph = create_input_graph(mermaid)
        skeleton = self.matcher._build_skeleton_from_graph(
            input_graph, style_vocab, 3, verbose=False
        )

        assert skeleton is not None
        assert '[P0]' in skeleton
        assert '[P1]' in skeleton
        assert '[P2]' in skeleton
        # Should use appropriate connectors for each edge type
        assert ('but' in skeleton or 'however' in skeleton)  # CONTRAST
        assert 'because' in skeleton  # CAUSALITY


if __name__ == '__main__':
    import pytest
    pytest.main([__file__, '-v'])

