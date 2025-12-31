#!/usr/bin/env python3
"""Test different LoRA configurations to find optimal settings.

Tests combinations of:
- lora_scale: How much LoRA influences output
- temperature: Generation randomness
- prompt format: Different instruction styles
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.generation.lora_generator import LoRAStyleGenerator, GenerationConfig

# Test inputs - simple, clear facts
TEST_CASES = [
    {
        "name": "Simple facts",
        "input": "Karl Marx developed Dialectical Materialism. Joseph Stalin coined the term.",
        "required": ["Karl Marx", "Dialectical Materialism", "Joseph Stalin", "coined"],
        "forbidden": ["Lenin", "1908", "1906", "Vladimir"],
    },
    {
        "name": "With example",
        "input": "Consider the smartphone. It contains lithium from Chile and cobalt from the Congo.",
        "required": ["smartphone", "lithium", "Chile", "cobalt", "Congo"],
        "forbidden": ["battery", "mining", "Africa"],
    },
    {
        "name": "Cause and effect",
        "input": "When one gear stops, the entire watch stops. Motion is systemic, not local.",
        "required": ["gear", "watch", "stops", "Motion", "systemic"],
        "forbidden": ["clock", "mechanism", "spring"],
    },
]

# Prompt templates to test
PROMPTS = {
    "minimal": "Rewrite in {author}'s style (~{words} words). Preserve all facts.\n\n",
    "strict": """Rewrite in {author}'s style (~{words} words).
RULES: Keep ALL names/facts exactly. Add NOTHING new.

""",
    "roleplay": """You are {author}. Rewrite this in your voice.
Keep every fact unchanged. Add nothing.

""",
}


def check_output(output: str, test_case: dict) -> dict:
    """Check if output contains required terms and avoids forbidden ones."""
    output_lower = output.lower()

    missing = []
    for term in test_case["required"]:
        if term.lower() not in output_lower:
            missing.append(term)

    hallucinated = []
    for term in test_case["forbidden"]:
        if term.lower() in output_lower:
            hallucinated.append(term)

    return {
        "missing": missing,
        "hallucinated": hallucinated,
        "score": (len(test_case["required"]) - len(missing)) / len(test_case["required"]),
        "clean": len(hallucinated) == 0,
    }


def test_config(adapter_path: str, author: str, lora_scale: float, temp: float, prompt_key: str):
    """Test a specific configuration."""
    config = GenerationConfig(
        temperature=temp,
        lora_scale=lora_scale,
        top_p=0.9,
    )

    generator = LoRAStyleGenerator(
        adapter_path=adapter_path,
        base_model="mlx-community/Qwen3-8B-4bit",
        config=config,
    )

    prompt_template = PROMPTS[prompt_key]

    results = []
    for test in TEST_CASES:
        # Build custom system prompt
        system = prompt_template.format(author=author, words=len(test["input"].split()))

        output = generator.generate(
            content=test["input"],
            author=author,
            system_override=system,
            max_tokens=80,
        )

        check = check_output(output, test)
        results.append({
            "test": test["name"],
            "input": test["input"],
            "output": output.strip(),
            **check,
        })

    return results


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter", default="lora_adapters/lovecraft")
    parser.add_argument("--author", default="H.P. Lovecraft")
    args = parser.parse_args()

    # Configurations to test
    configs = [
        # (lora_scale, temperature, prompt_key)
        (0.3, 0.3, "minimal"),
        (0.3, 0.5, "strict"),
        (0.5, 0.3, "minimal"),
        (0.5, 0.5, "strict"),
        (0.5, 0.5, "roleplay"),
        (0.7, 0.5, "strict"),
        (1.0, 0.5, "strict"),
    ]

    print("=" * 70)
    print("LORA CONFIGURATION TEST")
    print("=" * 70)
    print(f"Adapter: {args.adapter}")
    print(f"Author: {args.author}")
    print()

    all_results = []

    for lora_scale, temp, prompt_key in configs:
        print(f"\n{'='*70}")
        print(f"CONFIG: lora_scale={lora_scale}, temp={temp}, prompt={prompt_key}")
        print("=" * 70)

        results = test_config(args.adapter, args.author, lora_scale, temp, prompt_key)

        total_score = 0
        total_clean = 0

        for r in results:
            status = "✓" if r["clean"] and r["score"] == 1.0 else "✗"
            print(f"\n{status} {r['test']}")
            print(f"  IN:  {r['input'][:60]}...")
            print(f"  OUT: {r['output'][:80]}...")
            print(f"  Score: {r['score']:.0%} | Clean: {r['clean']}")
            if r["missing"]:
                print(f"  Missing: {r['missing']}")
            if r["hallucinated"]:
                print(f"  Hallucinated: {r['hallucinated']}")

            total_score += r["score"]
            total_clean += 1 if r["clean"] else 0

        avg_score = total_score / len(results)
        clean_rate = total_clean / len(results)

        all_results.append({
            "config": f"scale={lora_scale}, temp={temp}, prompt={prompt_key}",
            "avg_score": avg_score,
            "clean_rate": clean_rate,
            "combined": avg_score * 0.5 + clean_rate * 0.5,
        })

        print(f"\n  SUMMARY: Avg Score={avg_score:.0%}, Clean Rate={clean_rate:.0%}")

    # Final ranking
    print("\n" + "=" * 70)
    print("RANKING (by combined score)")
    print("=" * 70)

    all_results.sort(key=lambda x: x["combined"], reverse=True)
    for i, r in enumerate(all_results, 1):
        print(f"{i}. {r['config']}")
        print(f"   Score={r['avg_score']:.0%}, Clean={r['clean_rate']:.0%}, Combined={r['combined']:.0%}")


if __name__ == "__main__":
    main()
