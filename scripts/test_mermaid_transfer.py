#!/usr/bin/env python3
"""Test Mermaid graph syntax for style transfer.

This script tests whether the base model can:
1. Understand Mermaid graph syntax
2. Extract meaning from graph nodes
3. Produce faithful prose output
4. Apply author style

Usage:
    python scripts/test_mermaid_transfer.py --adapter lora_adapters/author
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def build_mermaid_from_graph(graph) -> str:
    """Convert semantic graph to enhanced Mermaid syntax.

    Uses relationship types as edge labels for clarity.
    Deduplicates nodes with identical text.
    """
    from src.validation.semantic_graph import RelationType

    lines = ["graph TD"]

    # Map relation types to readable labels
    relation_labels = {
        RelationType.CAUSE: "CAUSES",
        RelationType.CONTRAST: "BUT",
        RelationType.EXAMPLE: "EXAMPLE",
        RelationType.ELABORATION: "DETAIL",
        RelationType.SEQUENCE: "THEN",
        RelationType.CONDITION: "IF",
        RelationType.SUPPORT: "BECAUSE",
        RelationType.RESTATEMENT: "I.E.",
    }

    # Create node definitions - DEDUPLICATE by text content
    node_ids = {}  # original node.id -> mermaid node id
    text_to_node_id = {}  # normalized text -> mermaid node id
    unique_nodes = []
    node_counter = 0

    for node in graph.nodes:
        # Get and normalize text
        text = node.text.strip() if node.text else node.summary()
        text_normalized = text.lower()

        # Skip if we've seen this text before
        if text_normalized in text_to_node_id:
            # Map this node's ID to the existing mermaid node
            node_ids[node.id] = text_to_node_id[text_normalized]
            continue

        # New unique node
        node_id = f"N{node_counter}"
        node_counter += 1

        node_ids[node.id] = node_id
        text_to_node_id[text_normalized] = node_id

        # Escape brackets and quotes
        text_display = text.replace("[", "(").replace("]", ")")
        text_display = text_display.replace('"', "'")

        # Truncate very long nodes
        if len(text_display) > 100:
            text_display = text_display[:97] + "..."

        lines.append(f'    {node_id}["{text_display}"]')
        unique_nodes.append(node_id)

    # Create edges with relationship labels (skip self-loops from dedup)
    seen_edges = set()
    for edge in graph.edges:
        if edge.source_id in node_ids and edge.target_id in node_ids:
            src = node_ids[edge.source_id]
            tgt = node_ids[edge.target_id]

            # Skip self-loops (can happen after deduplication)
            if src == tgt:
                continue

            # Skip duplicate edges
            edge_key = (src, tgt)
            if edge_key in seen_edges:
                continue
            seen_edges.add(edge_key)

            label = relation_labels.get(edge.relation, "RELATES")
            lines.append(f'    {src} -->|{label}| {tgt}')

    # If no edges, create sequence edges between unique nodes
    if not seen_edges and len(unique_nodes) > 1:
        for i in range(len(unique_nodes) - 1):
            src = unique_nodes[i]
            tgt = unique_nodes[i + 1]
            lines.append(f'    {src} -->|THEN| {tgt}')

    return "\n".join(lines)


def build_bullet_format(graph) -> str:
    """Convert semantic graph to bullet point format for comparison."""
    lines = []
    seen_texts = set()
    for node in graph.nodes:
        text = node.text.strip() if node.text else node.summary()
        # Deduplicate by text content
        text_normalized = text.lower()
        if text_normalized in seen_texts:
            continue
        seen_texts.add(text_normalized)
        lines.append(f"• {text}")
    return "\n".join(lines)


def build_prose_format(graph) -> str:
    """Convert semantic graph to plain prose for comparison."""
    sentences = []
    seen_texts = set()
    for node in graph.nodes:
        text = node.text.strip() if node.text else node.summary()
        # Deduplicate by text content
        text_normalized = text.lower()
        if text_normalized in seen_texts:
            continue
        seen_texts.add(text_normalized)
        sentences.append(text)
    return " ".join(sentences)


def test_format(generator, content: str, format_name: str, author: str, target_words: int):
    """Test a specific input format."""
    print(f"\n{'='*60}")
    print(f"FORMAT: {format_name}")
    print(f"{'='*60}")
    print(f"INPUT ({len(content.split())} words):")
    print(content[:500] + "..." if len(content) > 500 else content)
    print()

    try:
        # Use LoRAStyleGenerator's generate method
        output = generator.generate(
            content=content,
            author=author,
            max_tokens=target_words * 2,
        )
        output = output.strip()

        print(f"OUTPUT ({len(output.split())} words):")
        print(output)
        print()

        return output
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        return None


def evaluate_coverage(source_graph, output_text, builder, comparator):
    """Evaluate how many propositions from source appear in output."""
    output_graph = builder.build_from_text(output_text)
    diff = comparator.compare(source_graph, output_graph)

    total = len(source_graph.nodes)
    missing = len(diff.missing_nodes)
    coverage = (total - missing) / total if total > 0 else 0

    return {
        "total_propositions": total,
        "missing": missing,
        "coverage": coverage,
        "missing_nodes": [n.summary()[:50] for n in diff.missing_nodes[:5]]
    }


def main():
    parser = argparse.ArgumentParser(description="Test Mermaid graph format for style transfer")
    parser.add_argument("--adapter", type=str, help="Path to LoRA adapter")
    parser.add_argument("--author", type=str, default="the author", help="Author name")
    parser.add_argument("--model", type=str, help="Base model (defaults to config)")
    args = parser.parse_args()

    # Test paragraphs with varying complexity
    test_paragraphs = [
        # Simple - few propositions
        """Dialectical Materialism is a practical toolset used to analyze how the world works.
        Karl Marx developed this worldview. Joseph Vissarionovich Dzhugashvili coined the term
        to describe the method Marxists use to examine reality.""",

        # Medium - multiple relationships
        """Consider the illusion of the static object: think of a smartphone. We perceive it as
        a finished, stationary object, but dialectically, that phone is merely a temporary snapshot
        of multiple intersecting flows. These include the flow of raw materials, such as lithium
        from Chile and cobalt from the Congo; the flow of labor, comprising thousands of hours of
        human toil and engineering; and the flow of information, where the phone acts as a node
        in a global network of satellites and server farms.""",

        # Complex - analogies and cause-effect
        """Dialectics teaches us that when you move one part of a system, everything moves with it.
        To visualize this interconnectivity, consider the mechanism of a traditional mechanical watch.
        It is a strictly dynamic system where every component exists in a state of active, functional
        tension. Each gear is physically interlocked with the next, meaning that motion is not a
        local event but a systemic one. If a single gear stops turning or is removed, the entire
        system does not simply slow down; it ceases to function as a clock.""",
    ]

    print("=" * 60)
    print("MERMAID GRAPH FORMAT TEST")
    print("=" * 60)

    # Initialize components
    from src.validation.semantic_graph import SemanticGraphBuilder, SemanticGraphComparator
    from src.generation.lora_generator import LoRAStyleGenerator, GenerationConfig

    builder = SemanticGraphBuilder(use_rebel=False)
    comparator = SemanticGraphComparator()

    # Initialize generator
    print("\nLoading model...")
    config = GenerationConfig(temperature=0.7)

    if args.adapter:
        generator = LoRAStyleGenerator(
            adapter_path=args.adapter,
            base_model=args.model if args.model else "mlx-community/Qwen3-8B-4bit",
            config=config,
        )
        print(f"Loaded adapter: {args.adapter}")
    else:
        generator = LoRAStyleGenerator(
            adapter_path=None,
            base_model=args.model if args.model else "mlx-community/Qwen3-8B-4bit",
            config=config,
        )
        print("Using base model (no adapter)")

    results = []

    for i, paragraph in enumerate(test_paragraphs):
        print(f"\n{'#'*60}")
        print(f"TEST PARAGRAPH {i+1}")
        print(f"{'#'*60}")
        print(f"\nORIGINAL ({len(paragraph.split())} words):")
        print(paragraph.strip())

        # Build semantic graph
        graph = builder.build_from_text(paragraph)
        print(f"\nGraph: {len(graph.nodes)} nodes, {len(graph.edges)} edges")

        target_words = len(paragraph.split())

        # Test different formats
        formats = {
            "mermaid": build_mermaid_from_graph(graph),
            "bullets": build_bullet_format(graph),
            "prose": build_prose_format(graph),
            "raw": paragraph.strip(),  # Original text for comparison
        }

        format_results = {}

        for format_name, content in formats.items():
            output = test_format(generator, content, format_name, args.author, target_words)

            if output:
                coverage = evaluate_coverage(graph, output, builder, comparator)
                format_results[format_name] = coverage
                print(f"COVERAGE: {coverage['coverage']:.0%} ({coverage['total_propositions'] - coverage['missing']}/{coverage['total_propositions']} propositions)")
                if coverage['missing_nodes']:
                    print(f"MISSING: {coverage['missing_nodes']}")

        results.append({
            "paragraph": i + 1,
            "results": format_results
        })

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")

    for fmt in ["mermaid", "bullets", "prose", "raw"]:
        coverages = [r["results"].get(fmt, {}).get("coverage", 0) for r in results]
        avg = sum(coverages) / len(coverages) if coverages else 0
        print(f"{fmt:10s}: avg coverage {avg:.0%}")

    print("\nConclusion:")
    print("- Higher coverage = better proposition preservation")
    print("- Compare output quality manually for style transfer")


if __name__ == "__main__":
    main()
