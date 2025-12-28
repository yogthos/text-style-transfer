#!/usr/bin/env python3
"""Command-line interface for text style transfer."""

import argparse
import json
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))


def cmd_ingest(args):
    """Ingest corpus files and extract style profile."""
    from pathlib import Path
    from src.style.extractor import StyleProfileExtractor
    from src.utils.nlp import split_into_sentences

    print(f"Loading corpus from: {args.corpus}")

    # Load corpus
    corpus_path = Path(args.corpus)
    if not corpus_path.exists():
        print(f"Error: Corpus file not found: {args.corpus}")
        sys.exit(1)

    corpus_text = corpus_path.read_text()
    paragraphs = [p.strip() for p in corpus_text.split('\n\n') if p.strip()]

    # Count statistics
    all_sentences = []
    for para in paragraphs:
        all_sentences.extend(split_into_sentences(para))
    word_count = sum(len(s.split()) for s in all_sentences)

    print(f"Loaded: {len(paragraphs)} paragraphs, {len(all_sentences)} sentences, {word_count} words")

    # Extract style profile
    print(f"\nExtracting style profile for '{args.author}'...")
    extractor = StyleProfileExtractor()
    profile = extractor.extract(paragraphs, args.author)

    # Print profile summary
    lp = profile.length_profile
    tp = profile.transition_profile
    rp = profile.register_profile

    print(f"\nStyle Profile: {profile.author_name}")
    print(f"  Sentence Length: mean={lp.mean:.1f}, std={lp.std:.1f}")
    print(f"  Burstiness: {lp.burstiness:.3f}")
    print(f"  No-Transition Ratio: {tp.no_transition_ratio:.1%}")
    print(f"  Formality Score: {rp.formality_score:.2f}")

    # Top transitions per category
    for cat_name, probs in [
        ("Causal", tp.causal),
        ("Adversative", tp.adversative),
        ("Additive", tp.additive),
    ]:
        if probs:
            top = sorted(probs.items(), key=lambda x: -x[1])[:3]
            words = ", ".join(f"{w}({p:.0%})" for w, p in top)
            print(f"  {cat_name}: {words}")

    # Show Markov transitions
    print(f"\n  Length Transitions (Markov):")
    for from_cat, to_probs in lp.length_transitions.items():
        trans = ", ".join(f"{to}:{p:.0%}" for to, p in sorted(to_probs.items()))
        print(f"    {from_cat} -> {trans}")

    # Save profile
    profile_path = Path(args.index_path) / f"{args.author}_style_profile.json"
    profile_path.parent.mkdir(parents=True, exist_ok=True)
    profile.save(str(profile_path))

    print(f"\nProfile saved to: {profile_path}")
    print("Done!")


def cmd_transfer(args):
    """Transfer text to target author's style using pre-computed profile."""
    from pathlib import Path
    from src.config import load_config
    from src.style.profile import AuthorStyleProfile
    from src.generation.data_driven_generator import DataDrivenStyleTransfer
    from src.llm.deepseek import DeepSeekProvider
    from src.vocabulary import create_controlled_generator

    # Load config
    try:
        config = load_config(args.config)
    except FileNotFoundError:
        print(f"Error: Config file not found: {args.config}")
        print("Please create config.json from config.json.sample")
        sys.exit(1)

    # Load input text
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: Input file not found: {args.input}")
        sys.exit(1)

    input_text = input_path.read_text()
    input_paragraphs = [p.strip() for p in input_text.split('\n\n') if p.strip()]
    print(f"Loaded input: {len(input_paragraphs)} paragraphs")

    # Try to load pre-computed profile
    profile_path = Path(args.index_path) / f"{args.author}_style_profile.json"
    profile = None

    if profile_path.exists():
        print(f"Loading pre-computed profile for '{args.author}'...")
        profile = AuthorStyleProfile.load(str(profile_path))
        print(f"  Corpus: {profile.corpus_word_count} words, {profile.corpus_sentence_count} sentences")
    else:
        print(f"No pre-computed profile found for '{args.author}'")
        print(f"Run: python cli.py ingest <corpus_file> --author {args.author}")
        sys.exit(1)

    # Initialize LLM provider
    provider_config = config.llm.get_provider_config()
    llm_provider = DeepSeekProvider(
        config=provider_config,
        retry_config={
            "max_retries": config.llm.max_retries,
            "base_delay": config.llm.base_delay,
            "max_delay": config.llm.max_delay,
        }
    )

    # Create vocabulary-controlled generator if enabled
    use_vocab_control = not getattr(args, 'no_vocab_control', False)

    if use_vocab_control:
        controlled = create_controlled_generator(
            llm_provider=llm_provider,
            profile=profile,
            provider_name="deepseek",
        )
        # Set input text to filter content words
        controlled.set_input_text(input_text)
        llm_generate = controlled.generate
        vocab_stats = controlled.get_stats()
        if vocab_stats.get("enabled"):
            print(f"  Vocabulary control: {vocab_stats['penalty_words']} penalties, {vocab_stats['boost_words']} boosts")
    else:
        # Fallback: regular LLM callable
        def llm_generate(prompt: str) -> str:
            return llm_provider.call(
                system_prompt="You are a skilled writer. Follow instructions exactly.",
                user_prompt=prompt,
                temperature=0.7,
                max_tokens=200
            )

    # Create transfer pipeline with pre-computed profile
    transfer = DataDrivenStyleTransfer(
        llm_generate=llm_generate,
        profile=profile,
        population_size=args.population_size,
        max_generations=args.max_generations,
    )

    # Print profile summary
    print("\n" + transfer.get_profile_summary())

    # Prepare output
    output_path = Path(args.output) if args.output else None
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text("")

    # Transfer with progress callback
    def on_paragraph_complete(idx, result):
        """Callback after each paragraph."""
        text = result.transferred
        v = result.verification
        e = result.entailment

        # Write incrementally
        if output_path:
            with open(output_path, 'a') as f:
                if idx > 0:
                    f.write("\n\n")
                f.write(text)

        # Show progress with entailment info
        preview = text[:70] + "..." if len(text) > 70 else text
        style_ok = v.is_acceptable
        content_ok = e.is_acceptable if e else True
        status = "✓" if (style_ok and content_ok) else "!"
        entail_info = f"content={e.coverage_ratio:.0%}" if e else ""
        print(f"  [{idx+1}/{len(input_paragraphs)}] {status} style={v.overall_score:.2f} {entail_info} {preview}")

    print("\nTransferring style...")
    result = transfer.transfer_document(input_paragraphs, on_paragraph_complete)

    # Summary
    print("\n" + "="*60)
    print("Summary:")
    print(f"  Paragraphs: {len(result.paragraphs)}")
    print(f"  Average score: {result.avg_score:.2f}")
    print(f"  Overall verification: {'PASS' if result.overall_verification.is_acceptable else 'FAIL'}")

    # Show component scores
    ov = result.overall_verification
    print(f"\nStyle Scores:")
    print(f"  Length match: {ov.length_score:.2f}")
    print(f"  Burstiness match: {ov.burstiness_score:.2f}")
    print(f"  Transition match: {ov.transition_score:.2f}")
    print(f"  Delta match: {ov.delta_score:.2f}")

    # Content preservation summary
    total_props = sum(len(r.propositions) for r in result.paragraphs)
    entailed_props = sum(
        r.entailment.propositions_entailed if r.entailment else len(r.propositions)
        for r in result.paragraphs
    )
    content_ratio = entailed_props / total_props if total_props > 0 else 1.0
    print(f"\nContent Preservation:")
    print(f"  Propositions: {entailed_props}/{total_props} entailed ({content_ratio:.0%})")

    # Show lost propositions if any
    lost = []
    for r in result.paragraphs:
        if r.entailment and r.entailment.lost_propositions:
            lost.extend(r.entailment.lost_propositions)
    if lost:
        print(f"  Lost content ({len(lost)} items):")
        for l in lost[:3]:
            print(f"    - {l[:60]}...")

    if ov.issues:
        print(f"\nIssues:")
        for issue in ov.issues[:5]:
            print(f"  - {issue}")

    if ov.suggestions:
        print(f"\nSuggestions:")
        for sug in ov.suggestions[:3]:
            print(f"  - {sug}")

    if output_path:
        print(f"\nOutput written to: {args.output}")
    else:
        print("\n" + "="*60)
        print("TRANSFORMED TEXT:")
        print("="*60)
        print(result.text)

    print("\nDone!")


def cmd_analyze(args):
    """Analyze a text file for style metrics."""
    from src.style.extractor import StyleProfileExtractor
    from src.utils.nlp import split_into_sentences
    import numpy as np

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: File not found: {args.input}")
        sys.exit(1)

    text = input_path.read_text()
    paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]

    # Gather sentences
    all_sentences = []
    for para in paragraphs:
        all_sentences.extend(split_into_sentences(para))

    # Basic stats
    lengths = [len(s.split()) for s in all_sentences]
    total_words = sum(lengths)

    print(f"Analysis of: {args.input}")
    print(f"\nStructure:")
    print(f"  Paragraphs: {len(paragraphs)}")
    print(f"  Sentences: {len(all_sentences)}")
    print(f"  Words: {total_words}")

    if lengths:
        mean = np.mean(lengths)
        std = np.std(lengths)
        burstiness = std / mean if mean > 0 else 0

        print(f"\nSentence Length:")
        print(f"  Mean: {mean:.1f} words")
        print(f"  Std: {std:.1f}")
        print(f"  Range: {min(lengths)} - {max(lengths)} words")
        print(f"  Burstiness: {burstiness:.3f}")

    # If requested, extract full profile
    if args.full:
        print("\nExtracting full style profile...")
        extractor = StyleProfileExtractor()
        profile = extractor.extract(paragraphs, "analyzed_text")

        tp = profile.transition_profile
        print(f"\nTransitions:")
        print(f"  No-transition ratio: {tp.no_transition_ratio:.1%}")

        for cat_name, probs in [
            ("Causal", tp.causal),
            ("Adversative", tp.adversative),
            ("Additive", tp.additive),
        ]:
            if probs:
                top = sorted(probs.items(), key=lambda x: -x[1])[:5]
                words = ", ".join(f"{w}({p:.0%})" for w, p in top)
                print(f"  {cat_name}: {words}")


def cmd_list_profiles(args):
    """List saved style profiles."""
    index_path = Path(args.index_path)
    if not index_path.exists():
        print("No profiles found. Run 'ingest' first.")
        return

    profiles = list(index_path.glob("*_style_profile.json"))
    if not profiles:
        print("No style profiles found.")
        return

    print("Saved style profiles:")
    for profile_path in profiles:
        from src.style.profile import AuthorStyleProfile
        profile = AuthorStyleProfile.load(str(profile_path))
        lp = profile.length_profile
        tp = profile.transition_profile
        print(f"\n  {profile.author_name}:")
        print(f"    Corpus: {profile.corpus_sentence_count} sentences, {profile.corpus_word_count} words")
        print(f"    Sentence length: mean={lp.mean:.1f}, std={lp.std:.1f}")
        print(f"    Burstiness: {lp.burstiness:.3f}")
        print(f"    No-transition ratio: {tp.no_transition_ratio:.1%}")


def main():
    parser = argparse.ArgumentParser(
        description="Text Style Transfer - Transform text to match an author's writing style"
    )
    parser.add_argument(
        "--config", "-c",
        default="config.json",
        help="Path to configuration file (default: config.json)"
    )
    parser.add_argument(
        "--index-path", "-i",
        default="atlas_cache/",
        help="Path to profile storage directory (default: atlas_cache/)"
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Ingest command
    ingest_parser = subparsers.add_parser(
        "ingest",
        help="Extract and save style profile from corpus"
    )
    ingest_parser.add_argument(
        "corpus",
        help="Corpus file containing author's text"
    )
    ingest_parser.add_argument(
        "--author", "-a",
        required=True,
        help="Author name for this corpus"
    )
    ingest_parser.set_defaults(func=cmd_ingest)

    # Transfer command
    transfer_parser = subparsers.add_parser(
        "transfer",
        help="Transfer text to target author's style (requires pre-ingested profile)"
    )
    transfer_parser.add_argument(
        "input",
        help="Input text file to transform"
    )
    transfer_parser.add_argument(
        "--author", "-a",
        required=True,
        help="Target author name (must be ingested first)"
    )
    transfer_parser.add_argument(
        "--output", "-o",
        help="Output file (prints to stdout if not specified)"
    )
    transfer_parser.add_argument(
        "--population-size", "-p",
        type=int,
        default=5,
        help="Population size for evolutionary generation (default: 5)"
    )
    transfer_parser.add_argument(
        "--max-generations", "-g",
        type=int,
        default=3,
        help="Maximum generations for evolution (default: 3)"
    )
    transfer_parser.add_argument(
        "--no-vocab-control",
        action="store_true",
        dest="no_vocab_control",
        help="Disable vocabulary control (logit bias for LLM-speak words)"
    )
    transfer_parser.set_defaults(func=cmd_transfer, vocab_control=True)

    # Analyze command
    analyze_parser = subparsers.add_parser(
        "analyze",
        help="Analyze text for style metrics"
    )
    analyze_parser.add_argument(
        "input",
        help="Text file to analyze"
    )
    analyze_parser.add_argument(
        "--full", "-f",
        action="store_true",
        help="Extract full style profile with transitions"
    )
    analyze_parser.set_defaults(func=cmd_analyze)

    # List profiles command
    list_parser = subparsers.add_parser(
        "list",
        help="List saved style profiles"
    )
    list_parser.set_defaults(func=cmd_list_profiles)

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    args.func(args)


if __name__ == "__main__":
    main()
