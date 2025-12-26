"""Tests for information ordering by graph structure (Phase 2.3)."""

import pytest
from src.models.graph import PropositionNode, RelationshipEdge, RelationshipType
from src.planning.sentence_planner import PropositionClusterer
from src.planning.rhythm_planner import RhythmPattern


class TestPropositionOrdering:
    """Tests for ordering propositions within clusters based on graph structure."""

    @pytest.fixture
    def clusterer(self):
        return PropositionClusterer()

    def test_single_proposition_unchanged(self, clusterer):
        """Single proposition cluster should remain unchanged."""
        prop = PropositionNode(
            id="p1", text="Main point.", subject="point", verb="is"
        )
        cluster = [prop]
        edges = []

        result = clusterer._order_by_graph_structure(cluster, edges)

        assert result == [prop]

    def test_root_proposition_comes_first(self, clusterer):
        """Root propositions (no incoming edges) should come first."""
        root = PropositionNode(id="p1", text="Main claim.", subject="claim", verb="is")
        dependent = PropositionNode(id="p2", text="Supporting detail.", subject="detail", verb="is")

        # p1 ELABORATES p2 (p1 is root, p2 is dependent)
        edges = [
            RelationshipEdge(source_id="p1", target_id="p2", relationship=RelationshipType.ELABORATES)
        ]

        # Original order has dependent first
        cluster = [dependent, root]
        result = clusterer._order_by_graph_structure(cluster, edges)

        # Root should be first now
        assert result[0].id == "p1"
        assert result[1].id == "p2"

    def test_causal_chain_ordering(self, clusterer):
        """Propositions in a causal chain should be ordered cause -> effect."""
        cause = PropositionNode(id="p1", text="Cause.", subject="cause", verb="is")
        effect = PropositionNode(id="p2", text="Effect.", subject="effect", verb="is")

        edges = [
            RelationshipEdge(source_id="p1", target_id="p2", relationship=RelationshipType.CAUSES)
        ]

        # Start with reverse order
        cluster = [effect, cause]
        result = clusterer._order_by_graph_structure(cluster, edges)

        assert result[0].id == "p1"  # cause
        assert result[1].id == "p2"  # effect

    def test_contrast_ordering(self, clusterer):
        """Contrasting proposition should follow what it contrasts."""
        statement = PropositionNode(id="p1", text="Statement.", subject="it", verb="is")
        contrast = PropositionNode(id="p2", text="Contrast.", subject="it", verb="differs")

        edges = [
            RelationshipEdge(source_id="p1", target_id="p2", relationship=RelationshipType.CONTRASTS)
        ]

        cluster = [contrast, statement]
        result = clusterer._order_by_graph_structure(cluster, edges)

        assert result[0].id == "p1"  # statement comes first
        assert result[1].id == "p2"  # contrast follows

    def test_multiple_roots_preserved(self, clusterer):
        """Multiple root propositions should all appear before dependents."""
        root1 = PropositionNode(id="p1", text="First root.", subject="first", verb="is")
        root2 = PropositionNode(id="p2", text="Second root.", subject="second", verb="is")
        dependent = PropositionNode(id="p3", text="Depends on p1.", subject="it", verb="depends")

        edges = [
            RelationshipEdge(source_id="p1", target_id="p3", relationship=RelationshipType.ELABORATES)
        ]

        cluster = [dependent, root2, root1]
        result = clusterer._order_by_graph_structure(cluster, edges)

        # Both roots should come before dependent
        root_positions = [i for i, p in enumerate(result) if p.id in ("p1", "p2")]
        dependent_position = next(i for i, p in enumerate(result) if p.id == "p3")

        assert all(rp < dependent_position for rp in root_positions)

    def test_disconnected_propositions_preserved(self, clusterer):
        """Disconnected propositions should still be included."""
        connected = PropositionNode(id="p1", text="Connected.", subject="it", verb="is")
        disconnected = PropositionNode(id="p2", text="Disconnected.", subject="it", verb="is")

        # Only one proposition has edges
        edges = []  # No edges at all

        cluster = [connected, disconnected]
        result = clusterer._order_by_graph_structure(cluster, edges)

        # Both should be in result
        assert len(result) == 2
        assert {p.id for p in result} == {"p1", "p2"}

    def test_complex_graph_ordering(self, clusterer):
        """Complex graph with multiple edges should be properly ordered."""
        # Graph: p1 -> p2 -> p4
        #        p1 -> p3
        p1 = PropositionNode(id="p1", text="Root.", subject="root", verb="is")
        p2 = PropositionNode(id="p2", text="First child.", subject="child1", verb="is")
        p3 = PropositionNode(id="p3", text="Second child.", subject="child2", verb="is")
        p4 = PropositionNode(id="p4", text="Grandchild.", subject="grandchild", verb="is")

        edges = [
            RelationshipEdge(source_id="p1", target_id="p2", relationship=RelationshipType.ELABORATES),
            RelationshipEdge(source_id="p1", target_id="p3", relationship=RelationshipType.ELABORATES),
            RelationshipEdge(source_id="p2", target_id="p4", relationship=RelationshipType.CAUSES),
        ]

        cluster = [p4, p3, p2, p1]  # Reverse order
        result = clusterer._order_by_graph_structure(cluster, edges)

        # p1 should be first
        assert result[0].id == "p1"

        # p4 should be after p2
        p2_idx = next(i for i, p in enumerate(result) if p.id == "p2")
        p4_idx = next(i for i, p in enumerate(result) if p.id == "p4")
        assert p2_idx < p4_idx


class TestClusterOrderingIntegration:
    """Tests for ordering being applied during clustering."""

    @pytest.fixture
    def clusterer(self):
        return PropositionClusterer()

    def test_clusters_are_ordered(self, clusterer):
        """Full cluster method should return ordered propositions."""
        root = PropositionNode(id="p1", text="Root claim.", subject="claim", verb="is")
        dep1 = PropositionNode(id="p2", text="First detail.", subject="detail", verb="is")
        dep2 = PropositionNode(id="p3", text="Second detail.", subject="detail", verb="is")

        edges = [
            RelationshipEdge(source_id="p1", target_id="p2", relationship=RelationshipType.ELABORATES),
            RelationshipEdge(source_id="p1", target_id="p3", relationship=RelationshipType.ELABORATES),
        ]

        # Propositions in random order
        propositions = [dep2, root, dep1]

        # Rhythm pattern that creates one big cluster
        rhythm = RhythmPattern(lengths=[30], burstiness=0.1, avg_length=30.0)

        clusters = clusterer.cluster(propositions, edges, rhythm)

        # Should have one cluster
        assert len(clusters) == 1

        # Root should be first in the cluster
        assert clusters[0][0].id == "p1"
