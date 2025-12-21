"""spaCy Linguistics Utilities

This module provides utility functions for extracting linguistic features from
pre-processed spaCy Doc objects. All functions accept Doc objects (not raw strings)
to ensure efficient processing - text should be processed once at entry points.

⚠️ CRITICAL: These functions must accept pre-processed Doc objects.
Do NOT call nlp() inside these functions - it will cause massive performance issues.
"""

from typing import Set, List, Dict
from spacy.tokens import Doc


def get_stop_words(doc: Doc) -> Set[str]:
    """Extract stop words from a pre-processed spaCy Doc.

    Args:
        doc: Pre-processed spaCy Doc object

    Returns:
        Set of stop word strings (lowercased)
    """
    return {token.text.lower() for token in doc if token.is_stop}


def get_conjunctions(doc: Doc) -> List[str]:
    """Extract coordinating and subordinating conjunctions.

    Args:
        doc: Pre-processed spaCy Doc object

    Returns:
        List of conjunction strings (lowercased)
    """
    return [token.text.lower() for token in doc if token.pos_ in ["CCONJ", "SCONJ"]]


def get_relative_pronouns(doc: Doc) -> List[str]:
    """Extract relative pronouns using dependency parsing.

    Relative pronouns typically have dependency relations like "relcl" (relative clause),
    or appear as pronouns in specific contexts.

    Args:
        doc: Pre-processed spaCy Doc object

    Returns:
        List of relative pronoun strings (lowercased)
    """
    relative_pronouns = []
    for token in doc:
        if token.pos_ == "PRON":
            # Check for relative clause markers or common relative pronouns
            if token.dep_ == "relcl" or token.text.lower() in ["which", "who", "that", "where", "when", "whom", "whose"]:
                relative_pronouns.append(token.text.lower())
    return relative_pronouns


def get_pov_pronouns(doc: Doc) -> Dict[str, Set[str]]:
    """Classify POV pronouns (first singular, first plural, third person).

    Args:
        doc: Pre-processed spaCy Doc object

    Returns:
        Dictionary with 'first_singular', 'first_plural', 'third_person' sets
    """
    first_singular = set()
    first_plural = set()
    third_person = set()

    for token in doc:
        if token.pos_ == "PRON" or (token.pos_ == "DET" and token.text.lower() in ["my", "our", "his", "her", "its", "their"]):
            token_lower = token.text.lower()

            # First person singular
            if token_lower in ["i", "me", "my", "myself", "mine"]:
                first_singular.add(token_lower)
            # First person plural
            elif token_lower in ["we", "us", "our", "ourselves", "ours"]:
                first_plural.add(token_lower)
            # Third person
            elif token_lower in ["he", "she", "it", "they", "him", "her", "them", "his", "hers", "its", "their", "theirs"]:
                third_person.add(token_lower)

    return {
        "first_singular": first_singular,
        "first_plural": first_plural,
        "third_person": third_person
    }


def get_main_verbs(doc: Doc) -> List[str]:
    """Extract main verbs using POS and dependency parsing.

    Main verbs are typically root verbs or verbs with specific tags indicating
    they are the main verb of a clause.

    Args:
        doc: Pre-processed spaCy Doc object

    Returns:
        List of main verb strings (lowercased)
    """
    main_verbs = []
    for token in doc:
        if token.pos_ == "VERB":
            # Root verbs or verbs with specific tags (VBZ=3rd person singular, VBP=non-3rd person, VBD=past tense)
            if token.dep_ == "ROOT" or token.tag_ in ["VBZ", "VBP", "VBD", "VBN", "VBG"]:
                main_verbs.append(token.text.lower())
    return main_verbs


def is_question_sentence(doc: Doc) -> bool:
    """Detect if sentence is a question.

    Args:
        doc: Pre-processed spaCy Doc object

    Returns:
        True if sentence is a question
    """
    # Check if sentence ends with question mark
    if doc.text.strip().endswith("?"):
        return True

    # Check for interrogative words (WH-words) or question tags
    for token in doc:
        # WH-determiners, WH-pronouns, WH-adverbs
        if token.tag_ in ["WDT", "WP", "WP$", "WRB"]:
            return True
        # Check for question starters at beginning
        if token.i == 0 and token.text.lower() in ["do", "does", "did", "is", "are", "was", "were", "can", "could", "will", "would", "should", "may", "might"]:
            return True

    return False


def get_discourse_markers(doc: Doc) -> List[str]:
    """Extract logical/narrative discourse markers.

    Discourse markers are words that connect ideas and indicate relationships
    between clauses or sentences.

    Args:
        doc: Pre-processed spaCy Doc object

    Returns:
        List of discourse marker strings (lowercased)
    """
    discourse_markers = []
    for token in doc:
        # Check for discourse markers by dependency and POS
        if token.dep_ in ["advmod", "mark", "cc"] and token.pos_ in ["ADV", "SCONJ", "CCONJ"]:
            discourse_markers.append(token.text.lower())
    return discourse_markers


def get_action_verbs(doc: Doc) -> List[str]:
    """Extract action verbs (all verbs in the document).

    This is a simple implementation that returns all verbs.
    For more sophisticated action verb detection, semantic analysis could be added.

    Args:
        doc: Pre-processed spaCy Doc object

    Returns:
        List of action verb strings (lowercased)
    """
    return [token.text.lower() for token in doc if token.pos_ == "VERB"]


def has_copula(doc: Doc) -> bool:
    """Check if sentence has a copula construction (e.g., "X is Y").

    Args:
        doc: Pre-processed spaCy Doc object

    Returns:
        True if sentence contains a copula
    """
    return any(token.dep_ == "cop" for token in doc)


def is_imperative_sentence(doc: Doc) -> bool:
    """Detect if sentence is imperative (command form).

    Imperatives typically have a root verb without an explicit subject.

    Args:
        doc: Pre-processed spaCy Doc object

    Returns:
        True if sentence appears to be imperative
    """
    if not doc:
        return False

    # Check if first token is a verb and is the root
    first_token = doc[0]
    if first_token.pos_ == "VERB" and first_token.dep_ == "ROOT":
        # Check if there's no explicit subject (nsubj)
        has_subject = any(token.dep_ == "nsubj" for token in doc)
        return not has_subject

    return False


def get_main_verbs_excluding_auxiliaries(doc: Doc) -> Set[str]:
    """Extract main verbs excluding auxiliaries and stopwords.

    Args:
        doc: Pre-processed spaCy Doc object

    Returns:
        Set of verb lemmas (lowercased) that are main verbs, not auxiliaries
    """
    main_verbs = set()
    for token in doc:
        if token.pos_ == "VERB" and not token.is_stop:
            # Exclude auxiliary verbs by checking dependency and tag
            # Auxiliaries typically have tags like "AUX" or dependencies like "aux", "auxpass"
            # Note: In spaCy, auxiliaries can have pos_="AUX" or pos_="VERB" with dep_="aux"/"auxpass"
            if token.pos_ != "AUX" and token.dep_ not in ["aux", "auxpass"]:
                main_verbs.add(token.lemma_.lower())
    return main_verbs


def detect_moralizing_patterns(doc: Doc) -> Dict[str, any]:
    """Detect moralizing/abstract patterns in text using spaCy Matcher.

    Args:
        doc: Pre-processed spaCy Doc object

    Returns:
        Dictionary with:
        - 'has_moralizing': bool
        - 'abstract_ratio': float (ratio of abstract to concrete nouns)
        - 'concrete_count': int (count of concrete markers)
        - 'abstract_lemmas': set of abstract lemmas found
    """
    from spacy.matcher import Matcher

    # Initialize matcher with shared vocab
    matcher = Matcher(doc.vocab)

    # Pattern 1: Abstract conclusion phrases
    # "In conclusion", "Ultimately", "In essence", "The lesson is"
    abstract_phrases = [
        [{"LOWER": {"IN": ["in", "ultimately", "essentially", "fundamentally"]}}],
        [{"LOWER": "the"}, {"LOWER": {"IN": ["lesson", "moral", "point", "meaning"]}}],
        [{"LOWER": {"IN": ["thus", "therefore", "hence", "consequently"]}}],
    ]

    # Pattern 2: Abstract nouns that signal moralizing
    abstract_nouns = [
        [{"POS": "NOUN", "LEMMA": {"IN": ["lesson", "meaning", "significance", "importance",
                                          "understanding", "wisdom", "truth", "reality",
                                          "nature", "essence", "purpose", "value"]}}],
    ]

    # Add patterns to matcher
    matcher.add("ABSTRACT_PHRASE", abstract_phrases)
    matcher.add("ABSTRACT_NOUN", abstract_nouns)

    # Find matches
    matches = matcher(doc)
    has_moralizing = len(matches) > 0

    # Calculate concrete vs abstract ratio
    concrete_markers = sum(1 for t in doc
                          if t.pos_ in ["PROPN", "NUM"] or
                          (t.pos_ == "NOUN" and t.ent_type_))  # Named entity nouns are concrete

    abstract_lemmas = {"lesson", "structure", "system", "humanity", "society",
                      "understanding", "abstract", "meaning", "significance",
                      "essence", "nature", "purpose", "value", "wisdom", "truth"}

    abstract_count = sum(1 for t in doc if t.lemma_.lower() in abstract_lemmas)
    total_nouns = sum(1 for t in doc if t.pos_ == "NOUN" and not t.is_stop)

    abstract_ratio = abstract_count / total_nouns if total_nouns > 0 else 0.0

    return {
        "has_moralizing": has_moralizing,
        "abstract_ratio": abstract_ratio,
        "concrete_count": concrete_markers,
        "abstract_lemmas": {t.lemma_.lower() for t in doc if t.lemma_.lower() in abstract_lemmas}
    }

