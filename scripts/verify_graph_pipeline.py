#!/usr/bin/env python3
"""Verification script for graph pipeline.

Tests the complete graph pipeline with a sample paragraph.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.atlas.input_mapper import InputLogicMapper
from src.generator.graph_matcher import TopologicalMatcher
# NOTE: Verified class name in src/generator/translator.py - it is StyleTranslator
from src.generator.translator import StyleTranslator
from src.utils.llm_provider import LLMProvider

def main():
    # Sample paragraph with more propositions to test fracturing
    sample_paragraph = (
        "The smartphone is a tool. "
        "It connects us to the world. "
        "But if it loses power, it becomes a brick. "
        "Technology requires energy to function. "
        "Without electricity, devices are useless. "
        "This dependency creates vulnerability. "
        "We must understand these limitations. "
        "Only then can we use technology wisely."
    )

    print("=" * 60)
    print("Graph Pipeline Verification")
    print("=" * 60)
    print(f"\nInput Paragraph:\n{sample_paragraph}\n")

    # Initialize components
    config_path = "config.json"
    llm_provider = LLMProvider(config_path=config_path)
    input_mapper = InputLogicMapper(llm_provider)
    graph_matcher = TopologicalMatcher(config_path=config_path)
    # NOTE: Verified class name in src/generator/translator.py - it is StyleTranslator
    translator = StyleTranslator(config_path=config_path)

    # Step 1: Extract propositions
    print("Step 1: Extracting propositions...")
    propositions = translator._extract_propositions_from_text(sample_paragraph)
    print(f"  Extracted {len(propositions)} propositions:")
    for i, prop in enumerate(propositions):
        print(f"    P{i}: {prop}")

    if not propositions:
        print("  ⚠ No propositions extracted. Exiting.")
        return 1

    # Step 2: Map to input graph
    print("\nStep 2: Mapping to input graph...")
    input_graph = input_mapper.map_propositions(propositions)
    if not input_graph:
        print("  ⚠ Graph mapping failed. Exiting.")
        return 1

    print(f"  Input Graph Mermaid:\n  {input_graph['mermaid']}")
    print(f"  Description: {input_graph['description']}")
    print(f"  Node Count: {input_graph['node_count']}")

    # Step 3: Match style graph (force opener role)
    print("\nStep 3: Matching style graph (forcing 'opener' role)...")
    document_context = {'current_index': 0, 'total_paragraphs': 1}
    style_match = graph_matcher.get_best_match(input_graph, document_context)

    if not style_match:
        print("  ⚠ No style graph match found. Exiting.")
        return 1

    print(f"  Style Graph Mermaid:\n  {style_match['style_mermaid']}")
    print(f"  Distance: {style_match.get('distance', 'N/A'):.4f}")
    print(f"  Skeleton: {style_match['style_metadata'].get('skeleton', 'N/A')}")
    print(f"  Node Mapping: {style_match['node_mapping']}")

    # Step 4: Generate text
    print("\nStep 4: Generating text from graph...")
    final_text = translator._generate_from_graph(
        style_match,
        input_graph['node_map'],
        "Mao",  # Default author
        verbose=True
    )

    print(f"\n{'=' * 60}")
    print("Final Generated Text:")
    print(f"{'=' * 60}")
    print(final_text)
    print()

    return 0

if __name__ == "__main__":
    sys.exit(main())

