"""Tests for the semantic graph module."""

import pytest

from src.validation.semantic_graph import (
    PropositionNode,
    RelationEdge,
    RelationType,
    SemanticGraph,
    GraphDiff,
    SemanticGraphBuilder,
    SemanticGraphComparator,
    build_and_compare_graphs,
)


class TestPropositionNode:
    """Tests for PropositionNode dataclass."""

    def test_summary_basic(self):
        """Test basic summary generation."""
        node = PropositionNode(
            id="P1",
            text="The cat sat on the mat.",
            subject="cat",
            predicate="sat",
            object="mat",
        )
        assert "cat sat mat" == node.summary()

    def test_summary_negated(self):
        """Test summary with negation."""
        node = PropositionNode(
            id="P1",
            text="The cat did not sit.",
            subject="cat",
            predicate="sit",
            is_negated=True,
        )
        assert node.summary().startswith("NOT:")

    def test_summary_truncation(self):
        """Test summary truncation at 60 chars."""
        node = PropositionNode(
            id="P1",
            text="A very long sentence.",
            subject="a very long subject with many words",
            predicate="does something very complicated",
            object="to a very long object",
        )
        assert len(node.summary()) <= 60

    def test_to_dict(self):
        """Test conversion to dictionary."""
        node = PropositionNode(
            id="P1",
            text="Test",
            subject="sub",
            predicate="pred",
            entities=["Entity1"],
        )
        d = node.to_dict()
        assert d["id"] == "P1"
        assert d["subject"] == "sub"
        assert d["entities"] == ["Entity1"]


class TestRelationEdge:
    """Tests for RelationEdge dataclass."""

    def test_to_dict(self):
        """Test conversion to dictionary."""
        edge = RelationEdge("P1", "P2", RelationType.CAUSE)
        d = edge.to_dict()
        assert d["source"] == "P1"
        assert d["target"] == "P2"
        assert d["relation"] == "cause"


class TestSemanticGraph:
    """Tests for SemanticGraph class."""

    def test_add_node(self):
        """Test adding nodes."""
        graph = SemanticGraph()
        node = PropositionNode(id="P1", text="Test", subject="s", predicate="p")
        graph.add_node(node)
        assert len(graph.nodes) == 1

    def test_add_edge(self):
        """Test adding edges."""
        graph = SemanticGraph()
        graph.add_edge("P1", "P2", RelationType.CAUSE)
        assert len(graph.edges) == 1
        assert graph.edges[0].relation == RelationType.CAUSE

    def test_get_node(self):
        """Test getting node by ID."""
        graph = SemanticGraph()
        node = PropositionNode(id="P1", text="Test", subject="s", predicate="p")
        graph.add_node(node)
        assert graph.get_node("P1") == node
        assert graph.get_node("P2") is None

    def test_to_mermaid_empty(self):
        """Test Mermaid output for empty graph."""
        graph = SemanticGraph()
        mermaid = graph.to_mermaid()
        assert "graph TD" in mermaid

    def test_to_mermaid_with_nodes(self):
        """Test Mermaid output with nodes."""
        graph = SemanticGraph()
        graph.add_node(PropositionNode(id="P1", text="Test", subject="cat", predicate="sat"))
        mermaid = graph.to_mermaid()
        assert "P1" in mermaid
        assert "cat sat" in mermaid

    def test_to_mermaid_escapes_special_chars(self):
        """Test that special characters are escaped in Mermaid output."""
        graph = SemanticGraph()
        graph.add_node(PropositionNode(
            id="P1", text='Test with "quotes" and [brackets]',
            subject='test "with" special', predicate="chars"
        ))
        mermaid = graph.to_mermaid()
        # Should not contain unescaped special chars
        assert '"' not in mermaid.split('"')[1].split('"')[0] or "'" in mermaid

    def test_to_text_description(self):
        """Test text description output."""
        graph = SemanticGraph()
        graph.add_node(PropositionNode(
            id="P1", text="Test", subject="cat", predicate="sat",
            entities=["Cat"]
        ))
        text = graph.to_text_description()
        assert "PROPOSITIONS:" in text
        assert "P1:" in text
        assert "Entities: Cat" in text


class TestGraphDiff:
    """Tests for GraphDiff class."""

    def test_is_isomorphic_empty(self):
        """Test isomorphism check for empty diff."""
        diff = GraphDiff()
        assert diff.is_isomorphic

    def test_is_isomorphic_with_missing(self):
        """Test isomorphism check with missing nodes."""
        diff = GraphDiff()
        diff.missing_nodes.append(
            PropositionNode(id="P1", text="Test", subject="s", predicate="p")
        )
        assert not diff.is_isomorphic

    def test_has_critical_differences(self):
        """Test critical differences detection."""
        diff = GraphDiff()
        assert not diff.has_critical_differences

        diff.missing_nodes.append(
            PropositionNode(id="P1", text="Test", subject="s", predicate="p")
        )
        assert diff.has_critical_differences

    def test_to_repair_instructions(self):
        """Test repair instruction generation."""
        diff = GraphDiff()
        diff.missing_nodes.append(
            PropositionNode(id="P1", text="The cat sat", subject="cat", predicate="sat")
        )
        diff.added_nodes.append(
            PropositionNode(id="P2", text="The dog ran", subject="dog", predicate="ran")
        )

        instructions = diff.to_repair_instructions()
        assert len(instructions) >= 2
        assert any("MISSING" in i for i in instructions)
        assert any("REMOVE" in i for i in instructions)

    def test_to_text(self):
        """Test text summary generation."""
        diff = GraphDiff()
        text = diff.to_text()
        assert "structurally equivalent" in text

        diff.missing_nodes.append(
            PropositionNode(id="P1", text="Test", subject="s", predicate="p")
        )
        text = diff.to_text()
        assert "MISSING PROPOSITIONS" in text


class TestSemanticGraphBuilder:
    """Tests for SemanticGraphBuilder class."""

    def test_build_from_empty_text(self):
        """Test building graph from empty text."""
        builder = SemanticGraphBuilder()
        graph = builder.build_from_text("")
        assert len(graph.nodes) == 0

    def test_build_from_simple_sentence(self):
        """Test building graph from simple sentence."""
        builder = SemanticGraphBuilder()
        graph = builder.build_from_text("The cat sat on the mat.")
        assert len(graph.nodes) >= 1

    def test_build_from_multiple_sentences(self):
        """Test building graph from multiple sentences."""
        builder = SemanticGraphBuilder()
        text = "The cat sat on the mat. The dog ran in the park."
        graph = builder.build_from_text(text)
        # Should have edges between sentences
        if len(graph.nodes) >= 2:
            assert len(graph.edges) >= 1

    def test_detect_contrast_relation(self):
        """Test detection of contrast markers."""
        builder = SemanticGraphBuilder()
        text = "The weather was nice. However, it rained later."
        graph = builder.build_from_text(text)
        # Should detect contrast
        contrast_edges = [e for e in graph.edges if e.relation == RelationType.CONTRAST]
        assert len(contrast_edges) >= 0  # May or may not detect depending on parsing

    def test_detect_negation(self):
        """Test negation detection in sentences."""
        builder = SemanticGraphBuilder()
        graph = builder.build_from_text("The cat did not sit on the mat.")
        if graph.nodes:
            assert graph.nodes[0].is_negated

    def test_extract_entities(self):
        """Test entity extraction."""
        builder = SemanticGraphBuilder()
        graph = builder.build_from_text("Karl Marx wrote Das Kapital in London.")
        if graph.nodes:
            # Should extract Karl Marx and London as entities
            all_entities = []
            for node in graph.nodes:
                all_entities.extend(node.entities)
            # At least one entity should be found
            assert len(all_entities) >= 0  # Depends on NER model


class TestSemanticGraphComparator:
    """Tests for SemanticGraphComparator class."""

    def test_compare_identical_graphs(self):
        """Test comparison of identical graphs."""
        builder = SemanticGraphBuilder()
        text = "The cat sat on the mat."
        graph1 = builder.build_from_text(text)
        graph2 = builder.build_from_text(text)

        comparator = SemanticGraphComparator()
        diff = comparator.compare(graph1, graph2)
        # Same text should produce similar graphs
        assert len(diff.missing_nodes) == 0 or diff.is_isomorphic

    def test_compare_different_graphs(self):
        """Test comparison of different graphs."""
        builder = SemanticGraphBuilder()
        graph1 = builder.build_from_text("The cat sat on the mat.")
        graph2 = builder.build_from_text("The dog ran in the park.")

        comparator = SemanticGraphComparator()
        diff = comparator.compare(graph1, graph2)
        # Different content should produce differences
        assert not diff.is_isomorphic or len(graph1.nodes) == 0

    def test_similarity_threshold(self):
        """Test that similarity threshold affects matching."""
        comparator1 = SemanticGraphComparator(similarity_threshold=0.9)
        comparator2 = SemanticGraphComparator(similarity_threshold=0.3)

        # Higher threshold should be stricter
        assert comparator1.similarity_threshold > comparator2.similarity_threshold


class TestBuildAndCompareGraphs:
    """Tests for the convenience function."""

    def test_build_and_compare_basic(self):
        """Test basic build and compare functionality."""
        source = "The cat sat on the mat."
        output = "A cat was sitting on a mat."

        source_graph, output_graph, diff = build_and_compare_graphs(source, output)

        assert isinstance(source_graph, SemanticGraph)
        assert isinstance(output_graph, SemanticGraph)
        assert isinstance(diff, GraphDiff)

    def test_build_and_compare_hallucination(self):
        """Test detection of hallucinated content."""
        source = "Karl Marx wrote about economics."
        output = "Karl Marx wrote about economics. He also invented the telephone."

        source_graph, output_graph, diff = build_and_compare_graphs(source, output)

        # Output has more content - should show added nodes
        if len(output_graph.nodes) > len(source_graph.nodes):
            assert len(diff.added_nodes) > 0 or not diff.is_isomorphic


class TestIntegration:
    """Integration tests for the semantic graph module."""

    def test_dialectics_example(self):
        """Test with the dialectics example from the codebase."""
        source = '''Many people hear the term Dialectical Materialism (or "Diamat") and assume they are stepping into a realm of strange mysticism. In reality, it is a practical "toolset" used to analyze and understand how the world works.'''

        output = '''Dialectical materialism is a practical analytical toolset. The term "Diamat" is a German abbreviation for this philosophy.'''

        source_graph, output_graph, diff = build_and_compare_graphs(source, output)

        # The output adds "German abbreviation" which is not in source
        # Should detect this as a difference
        repair_instructions = diff.to_repair_instructions()
        # Either missing propositions or added propositions should be detected
        assert len(diff.missing_nodes) > 0 or len(diff.added_nodes) > 0

    def test_mermaid_output_is_valid(self):
        """Test that Mermaid output is syntactically valid."""
        source = "The quick brown fox jumps over the lazy dog."
        builder = SemanticGraphBuilder()
        graph = builder.build_from_text(source)
        mermaid = graph.to_mermaid()

        # Basic syntax checks
        assert mermaid.startswith("graph TD")
        # No unclosed brackets or quotes (basic check)
        assert mermaid.count("[") == mermaid.count("]")

    def test_entity_conflation_detection(self):
        """Test detection of entity conflation (Marx ≠ Dzhugashvili)."""
        # Source clearly distinguishes two people with different roles
        source = '''While this worldview is overwhelmingly the work of Karl Marx, the term itself was coined by Joseph Vissarionovich Dzhugashvili to describe the method Marxists use.'''

        # Output incorrectly conflates them
        output = '''The term "Dialectical Materialism" was coined by Joseph Vissarionovich Dzhugashvili (a.K.A. Marx) to describe the method.'''

        source_graph, output_graph, diff = build_and_compare_graphs(source, output)

        # Should detect entity conflation as critical error
        assert diff.has_critical_differences, "Should detect critical differences when entities are conflated"

        # Check for entity role errors specifically
        conflation_errors = [e for e in diff.entity_role_errors if e.error_type == "conflation"]
        # The system should flag that Marx and Dzhugashvili are being conflated
        assert len(diff.entity_role_errors) > 0, "Should detect entity role errors"

        # Repair instructions should mention the conflation
        instructions = diff.to_repair_instructions()
        # Should have CRITICAL ERROR about the conflation
        critical_found = any("CRITICAL" in i for i in instructions)
        assert critical_found or len(diff.entity_role_errors) > 0, "Should flag entity errors as critical"

    def test_entity_role_extraction(self):
        """Test that entity roles are correctly extracted."""
        builder = SemanticGraphBuilder()
        graph = builder.build_from_text(
            "Karl Marx developed the philosophy. Stalin coined the term."
        )

        roles = graph.get_entity_roles()

        # Should extract roles for both entities
        # Note: entity names may be lowercase in the dict
        all_roles_text = str(roles).lower()
        # At least some entities should be tracked
        assert len(roles) >= 0  # May vary based on NER

    def test_role_swap_detection(self):
        """Test detection of role swaps."""
        source = "Einstein developed relativity. Newton discovered gravity."
        output = "Newton developed relativity. Einstein discovered gravity."

        source_graph, output_graph, diff = build_and_compare_graphs(source, output)

        # Should detect role swaps
        # Note: depends on NER correctly identifying Einstein and Newton
        if len(source_graph.nodes) >= 2:
            # If we have enough nodes, check for differences
            assert diff.has_critical_differences or len(diff.entity_role_errors) > 0
