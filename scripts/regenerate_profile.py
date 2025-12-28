#!/usr/bin/env python3
"""Regenerate style profile with structural_dna for an existing author."""

import json
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.analysis.style_profiler import StyleProfiler

def regenerate_profile(author_name: str, corpus_file: str):
    """Regenerate style profile with structural_dna."""
    # Load corpus
    with open(corpus_file, 'r', encoding='utf-8') as f:
        corpus_text = f.read()

    # Analyze style
    profiler = StyleProfiler()
    style_profile = profiler.analyze_style(corpus_text)

    # Save to existing profile location
    author_lower = author_name.lower()
    profile_path = Path(f"atlas_cache/paragraph_atlas/{author_lower}/style_profile.json")

    if not profile_path.exists():
        print(f"Error: Profile not found at {profile_path}")
        return

    with open(profile_path, 'w') as f:
        json.dump(style_profile, f, indent=2)

    print(f"✓ Regenerated style profile with structural_dna")
    print(f"  avg_words_per_sentence: {style_profile.get('structural_dna', {}).get('avg_words_per_sentence', 'N/A')}")
    print(f"  complexity_ratio: {style_profile.get('structural_dna', {}).get('complexity_ratio', 'N/A')}")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python regenerate_profile.py <author_name> <corpus_file>")
        print("Example: python regenerate_profile.py Mao data/corpus/mao.txt")
        sys.exit(1)

    regenerate_profile(sys.argv[1], sys.argv[2])