"""Semantic blueprint extraction module.

This module extracts semantic meaning from text, stripping away style
to create a "blueprint" of the content (SVOs, entities, keywords).
"""

from dataclasses import dataclass
from typing import List, Tuple, Set, Optional
import spacy
import re


@dataclass
class SemanticBlueprint:
    """Semantic representation of text meaning, stripped of style."""
    original_text: str
    svo_triples: List[Tuple[str, str, str]]  # (Subject, Verb, Object)
    named_entities: List[Tuple[str, str]]  # (Text, Label)
    core_keywords: Set[str]  # Lemmatized nouns/verbs/adjectives
    citations: List[Tuple[str, int]]  # List of (citation_text, position_in_sentence) tuples
    quotes: List[Tuple[str, int]]  # List of (quote_text, position_in_sentence) tuples
    # Positional metadata for contextual anchoring
    paragraph_id: int = 0  # Which paragraph (0-indexed)
    position: str = "BODY"  # "OPENER", "BODY", "CLOSER", "SINGLETON"
    previous_context: Optional[str] = None  # Generated text from previous sentence

    def get_subjects(self) -> List[str]:
        """Extract unique subjects from SVO triples."""
        return list(set([svo[0] for svo in self.svo_triples if svo[0]]))

    def get_verbs(self) -> List[str]:
        """Extract unique verbs from SVO triples."""
        return list(set([svo[1] for svo in self.svo_triples if svo[1]]))

    def get_objects(self) -> List[str]:
        """Extract unique objects from SVO triples."""
        return list(set([svo[2] for svo in self.svo_triples if svo[2]]))


class BlueprintExtractor:
    """Extracts semantic blueprint from text using spaCy."""

    def __init__(self):
        """Initialize the extractor with spaCy model."""
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            # Model not found, try to download it
            import subprocess
            import sys
            try:
                subprocess.check_call(
                    [sys.executable, "-m", "spacy", "download", "en_core_web_sm"],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL
                )
                self.nlp = spacy.load("en_core_web_sm")
            except Exception:
                raise RuntimeError(
                    "spaCy model 'en_core_web_sm' not found. "
                    "Please install it with: python -m spacy download en_core_web_sm"
                )

    def extract(self, text: str, paragraph_id: int = 0, position: str = "BODY",
                previous_context: Optional[str] = None) -> SemanticBlueprint:
        """Extract semantic blueprint from text.

        Args:
            text: Input text to analyze.
            paragraph_id: Which paragraph this sentence belongs to (0-indexed).
            position: Position of sentence in paragraph: "OPENER", "BODY", "CLOSER", "SINGLETON".
            previous_context: Generated text from previous sentence (for context).

        Returns:
            SemanticBlueprint with SVOs, entities, keywords, and positional metadata.
        """
        if not text or not text.strip():
            return SemanticBlueprint(
                original_text=text or "",
                svo_triples=[],
                named_entities=[],
                core_keywords=set(),
                citations=[],
                quotes=[],
                paragraph_id=paragraph_id,
                position=position,
                previous_context=previous_context
            )

        try:
            doc = self.nlp(text)
        except Exception:
            # If parsing fails, return empty blueprint
            return SemanticBlueprint(
                original_text=text,
                svo_triples=[],
                named_entities=[],
                core_keywords=set(),
                citations=[],
                quotes=[],
                paragraph_id=paragraph_id,
                position=position,
                previous_context=previous_context
            )

        # Extract SVO triples
        svo_triples = self._extract_svos(doc)

        # Extract named entities
        named_entities = [(ent.text, ent.label_) for ent in doc.ents]

        # Extract core keywords (lemmatized nouns, verbs, adjectives)
        core_keywords = self._extract_keywords(doc)

        # Extract citations and quotes
        citations = self._extract_citations(text)
        quotes = self._extract_quotes(text)

        return SemanticBlueprint(
            original_text=text,
            svo_triples=svo_triples,
            named_entities=named_entities,
            core_keywords=core_keywords,
            citations=citations,
            quotes=quotes,
            paragraph_id=paragraph_id,
            position=position,
            previous_context=previous_context
        )

    def _extract_svos(self, doc) -> List[Tuple[str, str, str]]:
        """Extract Subject-Verb-Object triples using dependency parsing."""
        svos = []
        for sent in doc.sents:
            # First, find the ROOT token (regardless of POS)
            root = None
            for token in sent:
                if token.dep_ == "ROOT":
                    root = token
                    break

            if not root:
                continue

            # Strategy 1: ROOT is a verb (normal case)
            if root.pos_ == "VERB":
                # Find subject (nsubj or nsubjpass for passive)
                subject = None
                for token in sent:
                    if token.dep_ in ("nsubj", "nsubjpass") and token.head == root:
                        subject = self._extract_phrase(token)
                        break

                # Find object (dobj)
                obj = None
                for token in sent:
                    if token.dep_ == "dobj" and token.head == root:
                        obj = self._extract_phrase(token)
                        break

                # Also check for prepositional objects (pobj) if no direct object
                if not obj:
                    for token in sent:
                        if token.dep_ == "pobj" and token.head.head == root:
                            obj = self._extract_phrase(token)
                            break

                if subject:
                    verb_lemma = root.lemma_.lower()
                    subject_clean = self._clean_span(subject)
                    obj_clean = self._clean_span(obj) if obj else ""
                    svos.append((subject_clean, verb_lemma, obj_clean))
                continue

            # Strategy 2: ROOT is not a verb - find the main verb or handle special cases
            # (Sometimes spaCy misparses verbs as nouns, e.g., "repeats" as NOUN)

            # First, try to find a verb in the sentence
            verb_root = None
            for token in sent:
                if token.pos_ == "VERB":
                    # Check if this verb has a subject or object
                    has_subject = any(
                        child.dep_ in ("nsubj", "nsubjpass") and child.head == token
                        for child in sent
                    )
                    has_object = any(
                        child.dep_ == "dobj" and child.head == token
                        for child in sent
                    )
                    if has_subject or has_object:
                        verb_root = token
                        break

            # If no verb found, look for any verb
            if not verb_root:
                for token in sent:
                    if token.pos_ == "VERB":
                        verb_root = token
                        break

            # Special case: If ROOT is a noun that looks like a verb (e.g., "repeats" parsed as NOUN)
            # and it has compound/nmod children, it might actually be the verb
            if root.pos_ == "NOUN" and not verb_root:
                # Check if the ROOT word could be a verb (ends in -s, -ed, -ing, etc.)
                root_text_lower = root.text.lower()
                verb_suffixes = ["s", "ed", "ing", "es", "ies"]
                if any(root_text_lower.endswith(suffix) for suffix in verb_suffixes):
                    # Check if it has noun children that could be the subject
                    noun_children = [
                        child for child in root.children
                        if child.pos_ == "NOUN" and child.dep_ in ("compound", "nmod")
                    ]

                    if noun_children:
                        # Treat the ROOT as verb, and the compound noun as subject
                        # Extract the subject phrase including all modifiers
                        # In "A rigid pattern repeats":
                        # - "pattern" (NOUN, compound) is the subject noun
                        # - "rigid" (ADJ, amod) and "A" (DET, det) are modifiers of root
                        #   but they actually modify "pattern"
                        subject_noun = noun_children[0]

                        # Collect all tokens that form the subject phrase
                        subject_tokens = [subject_noun]

                        # Add modifiers that are children of root (these modify the subject)
                        for child in root.children:
                            if child != subject_noun and child.dep_ in ("det", "amod"):
                                subject_tokens.append(child)

                        # Sort by token order
                        subject_tokens.sort(key=lambda t: t.i)
                        subject = " ".join([t.text for t in subject_tokens])

                        if subject:
                            verb_lemma = root.lemma_.lower()
                            subject_clean = self._clean_span(subject)
                            svos.append((subject_clean, verb_lemma, ""))
                            continue  # Skip to next sentence

            # Strategy 3: ROOT is a noun that is the subject, find the actual verb
            # e.g., "The cycle defines reality" where "cycle" is ROOT but "defines" is the verb
            if root.pos_ == "NOUN" and verb_root:
                # Check if root is subject of the verb
                if root.head == verb_root and root.dep_ in ("nsubj", "nsubjpass"):
                    subject = self._extract_phrase(root)

                    # Find object
                    obj = None
                    for token in sent:
                        if token.dep_ == "dobj" and token.head == verb_root:
                            obj = self._extract_phrase(token)
                            break

                    verb_lemma = verb_root.lemma_.lower()
                    subject_clean = self._clean_span(subject)
                    obj_clean = self._clean_span(obj) if obj else ""
                    svos.append((subject_clean, verb_lemma, obj_clean))
                    continue

            # Strategy 4: ROOT is a noun, verb exists, but root might not be directly linked
            # Look for any noun that is subject of the verb
            if root.pos_ == "NOUN" and verb_root:
                # Find subject of the verb
                subject_token = None
                for token in sent:
                    if token.dep_ in ("nsubj", "nsubjpass") and token.head == verb_root:
                        subject_token = token
                        break

                # If no explicit subject found, the ROOT noun might be the subject
                # (even if dependency is wrong, e.g., "cycle defines" where cycle is ROOT)
                if not subject_token and root.head == verb_root:
                    subject_token = root
                elif not subject_token:
                    # Try to find the ROOT noun as subject (spaCy might have wrong dependency)
                    # In "The cycle defines", "cycle" is ROOT but should be subject of "defines"
                    subject_token = root

                if subject_token:
                    subject = self._extract_phrase(subject_token)

                    # Find object
                    obj = None
                    for t in sent:
                        if t.dep_ == "dobj" and t.head == verb_root:
                            obj = self._extract_phrase(t)
                            break

                    verb_lemma = verb_root.lemma_.lower()
                    subject_clean = self._clean_span(subject)
                    obj_clean = self._clean_span(obj) if obj else ""
                    svos.append((subject_clean, verb_lemma, obj_clean))
                    continue

        return svos

    def _clean_span(self, span) -> str:
        """Removes determiners and restricted modifiers to leave core concept.

        Args:
            span: Either a string or a spaCy token/span.

        Returns:
            Cleaned string with determiners and punctuation removed.
        """
        if isinstance(span, str):
            # If span is already a string, parse it
            doc = self.nlp(span)
            tokens = [t for t in doc if t.pos_ not in ["DET", "PUNCT", "SPACE"]]
        else:
            # If span is a token or list of tokens
            if hasattr(span, '__iter__') and not hasattr(span, 'pos_'):
                # It's a list/collection of tokens
                tokens = [t for t in span if hasattr(t, 'pos_') and t.pos_ not in ["DET", "PUNCT", "SPACE"]]
            else:
                # Single token or span
                if hasattr(span, 'pos_'):
                    # Single token
                    tokens = [span] if span.pos_ not in ["DET", "PUNCT", "SPACE"] else []
                else:
                    # spaCy span
                    tokens = [t for t in span if t.pos_ not in ["DET", "PUNCT", "SPACE"]]

        return " ".join([t.text for t in tokens])

    def _extract_phrase(self, token) -> str:
        """Extract full phrase from a token (including modifiers).

        This includes:
        - Determiners (the, a, an)
        - Adjectives (rigid, biological)
        - Compound nouns
        - Prepositional phrases (of birth, life, and decay)
        """
        # Collect all tokens in the noun phrase
        phrase_tokens = [token]
        visited = {token.i}

        # Recursively collect modifiers and related tokens
        def collect_phrase_tokens(t):
            # Add children that are part of the phrase
            for child in t.children:
                if child.i in visited:
                    continue
                # Include determiners, adjectives, compounds, noun modifiers, etc.
                if child.dep_ in ("det", "amod", "compound", "nmod", "nummod", "poss", "prep", "pobj", "conj", "cc"):
                    phrase_tokens.append(child)
                    visited.add(child.i)
                    # For prepositional phrases, also collect their objects
                    if child.dep_ == "prep":
                        collect_phrase_tokens(child)
                    # For conjunctions (and, or), collect the conjoined items
                    # CRITICAL: Recursively collect all conjoined items to preserve lists
                    elif child.dep_ == "conj":
                        collect_phrase_tokens(child)
                    # For coordinating conjunctions (cc), also collect what they connect
                    elif child.dep_ == "cc":
                        # The conjunction itself is already added, now collect siblings
                        # that are connected by this conjunction
                        for sibling in t.children:
                            if sibling.dep_ == "conj" and sibling.i not in visited:
                                phrase_tokens.append(sibling)
                                visited.add(sibling.i)
                                collect_phrase_tokens(sibling)
                # Also include tokens that modify this token
                elif child.head == t and child.dep_ in ("amod", "compound", "nmod"):
                    phrase_tokens.append(child)
                    visited.add(child.i)

            # Also check if this token is part of a larger phrase
            # (e.g., "cycle of birth" - "cycle" is head, "of birth" is nmod)
            if t.head != t and t.head.i not in visited:
                if t.dep_ in ("nmod", "prep") or (t.head.pos_ == "NOUN" and t.dep_ in ("pobj",)):
                    # This is part of a noun phrase, include the head
                    phrase_tokens.append(t.head)
                    visited.add(t.head.i)
                    collect_phrase_tokens(t.head)

        collect_phrase_tokens(token)

        # Sort by token order
        phrase_tokens.sort(key=lambda t: t.i)

        # Extract text, preserving original case for proper nouns
        return " ".join([t.text for t in phrase_tokens])

    def _extract_keywords(self, doc) -> Set[str]:
        """Extract lemmatized keywords (nouns, verbs, adjectives).

        Filters out:
        - Stop words (the, a, an, is, are, etc.)
        - Very short words (< 3 chars)
        - Common fluff words (however, therefore, etc.)
        """
        # Extended list of fluff words to filter
        fluff_words = {
            "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
            "have", "has", "had", "do", "does", "did", "will", "would", "could",
            "should", "may", "might", "must", "can", "this", "that", "these", "those",
            "however", "therefore", "thus", "hence", "moreover", "furthermore",
            "nevertheless", "nonetheless", "indeed", "rather", "quite", "very",
            "just", "only", "also", "too", "as", "so", "such", "more", "most",
            "some", "any", "all", "each", "every", "both", "either", "neither"
        }

        keywords = set()
        for token in doc:
            if token.pos_ in {"NOUN", "VERB", "ADJ"}:
                lemma = token.lemma_.lower()
                # Filter stop words, fluff words, and very short words
                if (not token.is_stop and
                    lemma not in fluff_words and
                    len(lemma) > 2):
                    keywords.add(lemma)
        return keywords

    def _extract_citations(self, text: str) -> List[Tuple[str, int]]:
        """Extract all [^number] style citations with their positions.

        Args:
            text: Input text to analyze.

        Returns:
            List of (citation_text, position) tuples where position is the
            character index where the citation starts.
        """
        citations = []
        # Pattern matches [^number] where number is one or more digits
        citation_pattern = r'\[\^\d+\]'
        for match in re.finditer(citation_pattern, text):
            citation_text = match.group(0)
            position = match.start()
            citations.append((citation_text, position))
        return citations

    def _extract_quotes(self, text: str) -> List[Tuple[str, int]]:
        """Extract all direct quotations (both single and double quotes) with positions.

        Args:
            text: Input text to analyze.

        Returns:
            List of (quote_text, position) tuples where position is the
            character index where the quote starts. Only includes substantial
            quotes (length > 2 after stripping quotes).
        """
        quotes = []
        # Pattern matches both single and double quotes, handling escaped quotes
        quotation_pattern = r'["\'](?:[^"\']|(?<=\\)["\'])*["\']'
        for match in re.finditer(quotation_pattern, text):
            quote_text = match.group(0)
            # Only consider substantial quotations (more than just punctuation)
            if len(quote_text.strip('"\'')) > 2:
                position = match.start()
                quotes.append((quote_text, position))
        return quotes

