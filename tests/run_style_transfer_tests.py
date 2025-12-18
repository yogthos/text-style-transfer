#!/usr/bin/env python3
"""Run style transfer tests with better error reporting."""

import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Check if config.json exists
config_path = project_root / "config.json"
if not config_path.exists():
    print("⚠ Warning: config.json not found. Some tests may fail.")
    print("   Creating a minimal config.json for testing...")
    import json
    minimal_config = {
        "semantic_critic": {
            "similarity_threshold": 0.7,
            "recall_threshold": 0.80,
            "precision_threshold": 0.75,
            "fluency_threshold": 0.8
        },
        "paragraph_fusion": {
            "proposition_recall_threshold": 0.8,
            "meaning_weight": 0.6,
            "style_alignment_weight": 0.4
        },
        "translator": {
            "max_tokens": 500
        }
    }
    with open(config_path, 'w') as f:
        json.dump(minimal_config, f, indent=2)
    print("   ✓ Created minimal config.json")

# Now run the tests
if __name__ == "__main__":
    from test_style_transfer import run_all_tests
    success = run_all_tests()
    sys.exit(0 if success else 1)

