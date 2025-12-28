"""
Build Pre-Computed Syntactic Templates from Author Corpus.

This script analyzes the author's corpus using spaCy to extract syntactic skeletons,
categorizes them by rhetorical type, mood, and capacity, and saves them as a JSON library
for deterministic template selection.
"""

import spacy
import json
import os
import sys
import argparse
import re
from pathlib import Path
from collections import defaultdict

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

try:
    from src.utils.nlp_manager import NLPManager
    nlp = NLPManager.get_nlp()
except (ImportError, OSError, RuntimeError):
    print("Warning: Could not load spaCy model. Attempting direct load...")
    try:
        nlp = spacy.load("en_core_web_md")
    except OSError:
        print("Error: spaCy model 'en_core_web_md' not found.")
        print("Please install it with: python -m spacy download en_core_web_md")
        sys.exit(1)


def get_sentence_mood(doc):
    """Detects grammatical mood: Narrative, Imperative, Definition, Interrogative.

    Args:
        doc: spaCy Doc object for a sentence

    Returns:
        String mood: "narrative", "imperative", "definition", "interrogative"
    """
    text = doc.text.lower()

    # 1. Interrogative
    if text.strip().endswith("?"):
        return "interrogative"

    # 2. Imperative (Commands / Duty)
    if any(t.lemma_ in ["must", "should", "let", "need", "ought"] for t in doc):
        return "imperative"

    # 3. Definition (Copula Roots)
    roots = [t for t in doc if t.dep_ == "ROOT"]
    if roots and roots[0].lemma_ == "be":
        # Check if it's not just a passive construction auxiliary
        if roots[0].pos_ == "AUX" and not any(c.dep_ == "auxpass" for c in roots[0].children):
            return "definition"

    return "narrative"


def get_rhetorical_type(doc):
    """Detects rhetorical structure: Contrast, List, Causal, General.

    Args:
        doc: spaCy Doc object for a sentence

    Returns:
        String rhetoric: "contrast", "list", "causal", "general"
    """
    text = doc.text.lower()

    if "but" in text or "however" in text or "yet" in text:
        return "contrast"

    # Lists usually have multiple commas and 'and'/'or'
    commas = text.count(",")
    if commas >= 2 and ("and" in text or "or" in text):
        return "list"

    if "because" in text or "since" in text or "therefore" in text:
        return "causal"

    return "general"


def create_skeleton(doc):
    """
    Converts sentence to abstract skeleton.
    Masks content words, preserves structural anchors.

    Args:
        doc: spaCy Doc object for a sentence

    Returns:
        String skeleton with placeholders for content words
    """
    skeleton = []

    # Structural Lemmas (Keep these)
    structural_verbs = {"be", "have", "do", "will", "can", "could", "must", "should", "may", "might", "would"}
    structural_pronouns = {"it", "this", "that", "there", "what", "which", "who"}

    for token in doc:
        # 1. Punctuation & Conjunctions (Always Keep)
        if token.pos_ in ["PUNCT", "CCONJ", "SCONJ"]:
            skeleton.append(token.text.lower() if token.pos_ != "PUNCT" else token.text)
            continue

        # 2. Negations (Always Keep)
        if token.dep_ == "neg" or token.text.lower() == "not":
            skeleton.append("not")
            continue

        # 3. Structural Verbs/Auxiliaries (Keep)
        if token.lemma_ in structural_verbs:
            skeleton.append(token.text.lower())
            continue

        # 4. Prepositions (Keep for rhythm)
        if token.pos_ == "ADP":
            skeleton.append(token.text.lower())
            continue

        # 5. Articles (Keep)
        if token.tag_ == "DT":
            skeleton.append(token.text.lower())
            continue

        # 6. Structural Pronouns (Keep)
        if token.text.lower() in structural_pronouns:
            skeleton.append(token.text.lower())
            continue

        # --- MASKING CONTENT ---

        if token.pos_ in ["NOUN", "PROPN", "PRON"]:
            skeleton.append("[Noun]")
        elif token.pos_ == "VERB":
            skeleton.append("[Action]")
        elif token.pos_ == "ADJ":
            skeleton.append("[Adj]")
        elif token.pos_ == "ADV":
            skeleton.append("[Adv]")
        elif token.pos_ == "NUM":
            skeleton.append("[Num]")
        else:
            skeleton.append("[X]")

    # Post-processing to merge adjacent masks for cleaner templates
    # e.g. "The [Adj] [Noun]" -> "The [NounPhrase]"
    raw = " ".join(skeleton)

    # Reduce noise
    raw = re.sub(r'(\[Adj\]\s*)+\[Noun\]', '[NounPhrase]', raw)
    raw = re.sub(r'\[Noun\]\s*\[Noun\]', '[NounPhrase]', raw)

    # Clean up multiple spaces
    raw = " ".join(raw.split())

    return raw


def calculate_capacity(skeleton):
    """Counts how many 'slots' (bracketed items) are in the skeleton.

    Args:
        skeleton: Template skeleton string

    Returns:
        Integer capacity (number of slots)
    """
    return skeleton.count("[")


def build_templates(corpus_path: str, output_path: str, min_sentence_length: int = 5, max_sentence_length: int = 50):
    """
    Builds syntactic template library from corpus.

    Args:
        corpus_path: Path to corpus text file
        output_path: Path to output JSON file
        min_sentence_length: Minimum tokens per sentence
        max_sentence_length: Maximum tokens per sentence
    """
    print(f"Reading corpus from {corpus_path}...")
    if not os.path.exists(corpus_path):
        raise FileNotFoundError(f"Corpus file not found: {corpus_path}")

    with open(corpus_path, "r", encoding="utf-8") as f:
        text = f.read()

    doc = nlp(text)

    # Organization: Type -> Mood -> Capacity -> List of Templates
    library = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))

    print("Analyzing sentences...")
    count = 0
    seen = set()

    for sent in doc.sents:
        # Filter garbage
        if len(sent) < min_sentence_length or len(sent) > max_sentence_length:
            continue
        if sent.text.count("[") > 0:
            continue  # Skip text that already looks like code

        try:
            rhetoric = get_rhetorical_type(sent)
            mood = get_sentence_mood(sent)
            skeleton = create_skeleton(sent)
            capacity = calculate_capacity(skeleton)

            # Deduplicate
            if skeleton not in seen:
                seen.add(skeleton)
                library[rhetoric][mood][capacity].append(skeleton)
                count += 1
        except Exception as e:
            print(f"Warning: Error processing sentence '{sent.text[:50]}...': {e}")
            continue

    # Save
    print(f"Extracted {count} unique templates.")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Convert defaultdict to regular dict for JSON
    json_ready = {}
    for r, moods in library.items():
        json_ready[r] = {}
        for m, caps in moods.items():
            json_ready[r][m] = {}
            for c, templates in caps.items():
                json_ready[r][m][str(c)] = templates

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(json_ready, f, indent=2, ensure_ascii=False)
    print(f"Saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Build syntactic templates from author corpus")
    parser.add_argument("corpus_path", help="Path to corpus text file")
    parser.add_argument("--author", required=True, help="Author name (e.g., 'mao')")
    parser.add_argument("--output-dir", default="atlas_cache/paragraph_atlas",
                       help="Output directory for templates (default: atlas_cache/paragraph_atlas)")
    parser.add_argument("--min-length", type=int, default=5,
                       help="Minimum sentence length in tokens (default: 5)")
    parser.add_argument("--max-length", type=int, default=50,
                       help="Maximum sentence length in tokens (default: 50)")

    args = parser.parse_args()

    # Save to author-specific directory
    output_dir = Path(args.output_dir) / args.author.lower()
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "syntactic_templates.json"

    build_templates(args.corpus_path, str(output_path), args.min_length, args.max_length)

    print(f"✓ Built syntactic templates library: {output_path}")


if __name__ == "__main__":
    main()
