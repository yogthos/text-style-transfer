"""OpenIE-based text flattening for style transfer training.

Uses Stanford OpenIE to extract fact triples, combined with spaCy
for entity extraction. This creates a fact-preserving neutral
representation that differs significantly in structure from the
original styled prose.

The output is intentionally "choppy" - this forces the LoRA to learn
how to combine facts into flowing prose with the target author's style.

Requirements:
    - Java JRE (for Stanford CoreNLP)
    - pip install stanford-openie spacy
    - python -m spacy download en_core_web_md
"""

import atexit
import re
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass
import spacy

# Lazy-loaded resources
_nlp = None
_openie_available = None
_openie_client = None


def get_nlp():
    """Get or load spaCy model."""
    global _nlp
    if _nlp is None:
        _nlp = spacy.load("en_core_web_md")
    return _nlp


def is_openie_available() -> bool:
    """Check if Stanford OpenIE is available."""
    global _openie_available
    if _openie_available is None:
        try:
            from openie import StanfordOpenIE
            _openie_available = True
        except ImportError:
            _openie_available = False
    return _openie_available


def get_openie_client():
    """Get or create singleton OpenIE client.

    The client starts a Java server on first use and reuses it for all
    subsequent calls. This avoids the overhead of starting a new server
    for each extraction.
    """
    global _openie_client
    if _openie_client is None:
        if not is_openie_available():
            raise ImportError("stanford-openie not installed. Run: pip install stanford-openie")
        from openie import StanfordOpenIE
        _openie_client = StanfordOpenIE()
        _openie_client.__enter__()  # Start the server
    return _openie_client


def shutdown_openie():
    """Shutdown the OpenIE server if running."""
    global _openie_client
    if _openie_client is not None:
        try:
            _openie_client.__exit__(None, None, None)
        except Exception:
            pass
        _openie_client = None


# Register shutdown handler for clean exit
atexit.register(shutdown_openie)


@dataclass
class FlattenedResult:
    """Result of flattening operation."""
    text: str
    triples: List[Tuple[str, str, str]]
    entities: List[Tuple[str, str]]
    word_overlap: float
    structural_diff: Dict[str, float]


def extract_entities(text: str) -> List[Tuple[str, str]]:
    """Extract named entities using spaCy.

    Returns list of (entity_text, entity_label) tuples.
    """
    nlp = get_nlp()
    doc = nlp(text)

    entities = []
    seen = set()

    for ent in doc.ents:
        # Only keep informative entity types
        if ent.label_ in ("DATE", "TIME", "ORG", "PERSON", "GPE", "LOC",
                          "MONEY", "PERCENT", "QUANTITY", "CARDINAL"):
            key = (ent.text.lower(), ent.label_)
            if key not in seen:
                seen.add(key)
                entities.append((ent.text, ent.label_))

    return entities


def extract_triples_openie(text: str) -> List[Tuple[str, str, str]]:
    """Extract fact triples using Stanford OpenIE.

    Returns list of (subject, relation, object) tuples.
    Uses singleton client to avoid starting a new Java server per call.
    """
    client = get_openie_client()
    raw_triples = client.annotate(text)

    # Deduplicate - keep most informative version of each relation
    seen_relations = {}
    for t in raw_triples:
        subj = t['subject'].strip()
        rel = t['relation'].strip()
        obj = t['object'].strip()

        # Skip empty or too short
        if len(subj) < 2 or len(obj) < 2:
            continue

        key = (subj.lower(), rel.lower())
        # Keep longer/more informative objects
        if key not in seen_relations or len(obj) > len(seen_relations[key]):
            seen_relations[key] = obj

    triples = []
    for (subj, rel), obj in seen_relations.items():
        # Restore original case for subject
        subj = subj[0].upper() + subj[1:] if subj else subj
        triples.append((subj, rel, obj))

    return triples


def extract_triples_spacy(text: str) -> List[Tuple[str, str, str]]:
    """Fallback triple extraction using spaCy dependency parsing.

    Less comprehensive than OpenIE but doesn't require Java.
    """
    nlp = get_nlp()
    doc = nlp(text)
    triples = []

    for sent in doc.sents:
        root = sent.root

        # Find subject
        subject = None
        for child in root.children:
            if child.dep_ in ("nsubj", "nsubjpass"):
                # Get noun phrase
                subj_tokens = [t.text for t in child.subtree
                              if t.dep_ not in ("relcl", "acl")]
                subject = " ".join(subj_tokens).strip()
                break

        if not subject or subject.lower() in ("it", "this", "that", "which"):
            continue

        # Get verb
        verb = root.lemma_

        # Get objects
        objects = []
        for child in root.children:
            if child.dep_ in ("dobj", "attr", "acomp"):
                obj_tokens = [t.text for t in child.subtree]
                objects.append(" ".join(obj_tokens))
            elif child.dep_ == "prep":
                prep_text = child.text
                for gc in child.children:
                    if gc.dep_ == "pobj":
                        obj_tokens = [t.text for t in gc.subtree]
                        prep_text += " " + " ".join(obj_tokens)
                objects.append(prep_text)

        # Create triples
        if objects:
            for obj in objects:
                triples.append((subject, verb, obj))
        else:
            triples.append((subject, verb, ""))

    return triples


def format_triples_as_text(
    triples: List[Tuple[str, str, str]],
    entities: List[Tuple[str, str]],
) -> str:
    """Format extracted triples and entities as neutral text.

    Output format is intentionally choppy to maximize structural
    difference from flowing prose.
    """
    lines = []
    seen = set()

    # Add entity markers first (important dates, orgs, etc.)
    for ent_text, ent_label in entities:
        if ent_label in ("DATE", "ORG", "PERSON", "GPE"):
            marker = f"[{ent_label}] {ent_text}."
            if marker.lower() not in seen:
                seen.add(marker.lower())
                lines.append(marker)

    # Add triples as simple sentences
    for subj, rel, obj in triples:
        if obj:
            sent = f"{subj} {rel} {obj}."
        else:
            sent = f"{subj} {rel}."

        # Clean up
        sent = re.sub(r'\s+', ' ', sent)
        sent = re.sub(r'\s+\.', '.', sent)

        norm = sent.lower()
        if norm not in seen and len(sent) > 5:
            seen.add(norm)
            lines.append(sent)

    return " ".join(lines)


def calculate_metrics(original: str, flattened: str) -> Dict[str, float]:
    """Calculate structural difference metrics."""
    nlp = get_nlp()

    orig_doc = nlp(original)
    flat_doc = nlp(flattened)

    # Word overlap
    orig_words = set(w.lower().strip('.,;:!?"\'()[]')
                    for w in original.split() if len(w) > 2)
    flat_words = set(w.lower().strip('.,;:!?"\'()[]')
                    for w in flattened.split() if len(w) > 2)
    word_overlap = len(orig_words & flat_words) / len(orig_words) if orig_words else 0

    # Sentence metrics
    orig_sents = list(orig_doc.sents)
    flat_sents = list(flat_doc.sents)

    orig_avg_len = sum(len(list(s)) for s in orig_sents) / len(orig_sents) if orig_sents else 0
    flat_avg_len = sum(len(list(s)) for s in flat_sents) / len(flat_sents) if flat_sents else 0

    # Subordinate clause count
    orig_subordinate = sum(1 for t in orig_doc if t.dep_ in ("mark", "advcl", "relcl", "ccomp"))
    flat_subordinate = sum(1 for t in flat_doc if t.dep_ in ("mark", "advcl", "relcl", "ccomp"))

    return {
        "word_overlap": word_overlap,
        "orig_sentences": len(orig_sents),
        "flat_sentences": len(flat_sents),
        "orig_avg_sent_len": orig_avg_len,
        "flat_avg_sent_len": flat_avg_len,
        "orig_subordinate_clauses": orig_subordinate,
        "flat_subordinate_clauses": flat_subordinate,
    }


def flatten_text(text: str, use_openie: bool = True) -> FlattenedResult:
    """Flatten styled text to neutral fact representation.

    Args:
        text: Input styled text.
        use_openie: If True, use Stanford OpenIE (requires Java).
                   If False, use spaCy fallback.

    Returns:
        FlattenedResult with flattened text and metrics.
    """
    # Extract entities
    entities = extract_entities(text)

    # Extract triples
    if use_openie and is_openie_available():
        triples = extract_triples_openie(text)
    else:
        triples = extract_triples_spacy(text)

    # Format as text
    flattened = format_triples_as_text(triples, entities)

    # Calculate metrics
    metrics = calculate_metrics(text, flattened)

    return FlattenedResult(
        text=flattened,
        triples=triples,
        entities=entities,
        word_overlap=metrics["word_overlap"],
        structural_diff=metrics,
    )


def flatten_text_simple(text: str) -> str:
    """Simple interface - just return flattened text.

    Uses OpenIE if available, otherwise spaCy fallback.
    """
    result = flatten_text(text, use_openie=is_openie_available())
    return result.text


# Alias for compatibility with existing code
def deterministic_flatten(text: str) -> str:
    """Alias for flatten_text_simple for API compatibility."""
    return flatten_text_simple(text)


if __name__ == "__main__":
    # Test
    sample = """The vault to which I refer is of ancient granite, weathered and
    discoloured by the mists and dampness of generations. The structure is visible
    only at the entrance. The door, a ponderous and forbidding slab of stone,
    hangs upon rusted iron hinges."""

    print("Testing OpenIE flattening...")
    print(f"\nOriginal:\n{sample}")

    result = flatten_text(sample)
    print(f"\nFlattened:\n{result.text}")
    print(f"\nTriples ({len(result.triples)}):")
    for t in result.triples[:5]:
        print(f"  {t}")
    print(f"\nEntities: {result.entities}")
    print(f"\nWord overlap: {result.word_overlap:.0%}")
    print(f"Structural metrics: {result.structural_diff}")
