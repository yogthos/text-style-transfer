#!/usr/bin/env python3
"""Helper script for curating golden examples.

This script helps create new golden examples by generating outputs
and calculating semantic similarity scores.
"""

import json
import sys
import argparse
from pathlib import Path
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.generator.translator import StyleTranslator
from src.validator.semantic_critic import SemanticCritic
from tests.test_helpers import ensure_config_exists


def calculate_similarity(text1: str, text2: str) -> float:
    """Calculate semantic similarity between two texts."""
    critic = SemanticCritic()
    return critic._calculate_semantic_similarity(text1, text2)


def create_golden_example(
    input_text: str,
    author_name: str,
    output_id: str,
    output_text: Optional[str] = None,
    archetype_id: Optional[int] = None,
    perspective: Optional[str] = None,
    notes: str = ""
) -> Dict:
    """Create a golden example entry.

    Args:
        input_text: Original input paragraph
        author_name: Target author name
        output_id: Unique identifier (e.g., "golden_001")
        output_text: Generated output (if None, will generate)
        archetype_id: Optional archetype ID used
        perspective: Optional perspective override
        notes: Optional notes about this example

    Returns:
        Dictionary with golden example data
    """
    ensure_config_exists()

    translator = StyleTranslator(config_path="config.json")

    # Generate output if not provided
    if output_text is None:
        print(f"Generating output for {output_id}...")
        output_text, used_archetype_id, compliance_score = translator.translate_paragraph_statistical(
            paragraph=input_text,
            author_name=author_name,
            perspective=perspective,
            verbose=True
        )
        if archetype_id is None:
            archetype_id = used_archetype_id
        print(f"Generated output (compliance: {compliance_score:.2f})")
    else:
        # If output provided, still need to get archetype if not specified
        if archetype_id is None:
            # Generate once to get archetype, but use provided output
            _, used_archetype_id, _ = translator.translate_paragraph_statistical(
                paragraph=input_text,
                author_name=author_name,
                perspective=perspective,
                verbose=False
            )
            archetype_id = used_archetype_id

    # Calculate similarity (input vs output as baseline)
    similarity = calculate_similarity(input_text, output_text)

    # Detect perspective if not provided
    if perspective is None:
        perspective = translator._detect_input_perspective(input_text)

    # Create golden example
    golden_example = {
        "id": output_id,
        "input_text": input_text,
        "author_name": author_name,
        "archetype_id": archetype_id,
        "perspective": perspective,
        "expected_output": output_text,
        "expected_similarity": similarity,
        "metadata": {
            "created_date": datetime.now().isoformat(),
            "verified_by": "manual",
            "notes": notes
        }
    }

    return golden_example


def save_golden_example(golden_example: Dict, examples_dir: Path):
    """Save golden example to JSON file."""
    examples_dir.mkdir(parents=True, exist_ok=True)

    output_id = golden_example["id"]
    output_file = examples_dir / f"{output_id}.json"

    with open(output_file, 'w') as f:
        json.dump(golden_example, f, indent=2, ensure_ascii=False)

    print(f"✓ Saved golden example to {output_file}")
    return output_file


def main():
    parser = argparse.ArgumentParser(
        description="Curate golden examples for regression testing"
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Input text (paragraph to translate)"
    )
    parser.add_argument(
        "--author",
        required=True,
        help="Target author name"
    )
    parser.add_argument(
        "--output-id",
        required=True,
        help="Unique identifier (e.g., golden_001)"
    )
    parser.add_argument(
        "--output-text",
        help="Pre-generated output text (optional, will generate if not provided)"
    )
    parser.add_argument(
        "--archetype-id",
        type=int,
        help="Archetype ID (optional, will detect if not provided)"
    )
    parser.add_argument(
        "--perspective",
        help="Perspective override (first_person_singular, third_person, etc.)"
    )
    parser.add_argument(
        "--notes",
        default="",
        help="Notes about this golden example"
    )
    parser.add_argument(
        "--examples-dir",
        default=None,
        help="Directory to save examples (default: tests/golden_set/examples)"
    )

    args = parser.parse_args()

    # Determine examples directory
    if args.examples_dir:
        examples_dir = Path(args.examples_dir)
    else:
        examples_dir = Path(__file__).parent / "examples"

    # Create golden example
    golden_example = create_golden_example(
        input_text=args.input,
        author_name=args.author,
        output_id=args.output_id,
        output_text=args.output_text,
        archetype_id=args.archetype_id,
        perspective=args.perspective,
        notes=args.notes
    )

    # Save to file
    save_golden_example(golden_example, examples_dir)

    print(f"\nGolden example created:")
    print(f"  ID: {golden_example['id']}")
    print(f"  Similarity: {golden_example['expected_similarity']:.3f}")
    print(f"  Perspective: {golden_example['perspective']}")
    print(f"  Archetype: {golden_example['archetype_id']}")


if __name__ == "__main__":
    main()

