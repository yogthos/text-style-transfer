"""Proposition extraction from text using spaCy and LLM."""

import re
from dataclasses import dataclass, field
from typing import List, Optional, Tuple
import uuid

from ..utils.nlp import get_nlp, split_into_sentences, extract_citations
from ..utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class SVOTriple:
    """Subject-Verb-Object triple extracted from text."""
    subject: str
    verb: str
    object: Optional[str]
    full_text: str


@dataclass
class PropositionNode:
    """A node representing an atomic proposition in a semantic graph."""

    id: str
    text: str  # The proposition text
    subject: str = ""
    verb: str = ""
    object: Optional[str] = None
    entities: List[str] = field(default_factory=list)
    keywords: List[str] = field(default_factory=list)
    source_sentence_idx: int = 0
    is_citation: bool = False
    is_quotation: bool = False
    attached_citations: List[str] = field(default_factory=list)

    # RST-like rhetorical role
    rst_role: str = "nucleus"  # "nucleus" (main claim) or "satellite" (supporting)
    rst_relation: str = "none"  # relation to previous: "example", "evidence", "elaboration", "contrast", "cause", "none"
    parent_nucleus_idx: Optional[int] = None  # index of the nucleus this satellite supports


class PropositionExtractor:
    """Extracts atomic propositions from text using dependency parsing.

    A proposition represents an atomic claim or fact that can be expressed
    as a subject-verb-object triple with associated entities and keywords.
    """

    # Citation patterns to preserve
    CITATION_PATTERN = re.compile(r'\[\^\d+\]|\[\d+\]')

    # Quotation patterns
    QUOTATION_PATTERN = re.compile(r'["\u201c]([^"\u201d]+)["\u201d]')

    def __init__(self):
        """Initialize extractor with spaCy model."""
        self._nlp = None

    @property
    def nlp(self):
        """Lazy-load spaCy model."""
        if self._nlp is None:
            self._nlp = get_nlp()
        return self._nlp

    def extract_from_text(self, text: str) -> List[PropositionNode]:
        """Extract propositions from a paragraph of text.

        Args:
            text: Text to extract propositions from.

        Returns:
            List of PropositionNode instances.
        """
        if not text or not text.strip():
            return []

        # Extract citations for later attachment
        citations = extract_citations(text)

        # Split into sentences
        sentences = split_into_sentences(text)

        propositions = []
        for sent_idx, sentence in enumerate(sentences):
            sent_propositions = self._extract_from_sentence(
                sentence, sent_idx, citations
            )
            propositions.extend(sent_propositions)

        # Classify RST roles (nucleus vs satellite)
        propositions = self.classify_rst_roles(propositions)

        logger.debug(
            f"Extracted {len(propositions)} propositions from {len(sentences)} sentences"
        )

        return propositions

    def _extract_from_sentence(
        self,
        sentence: str,
        sentence_idx: int,
        all_citations: List[Tuple[str, int]]
    ) -> List[PropositionNode]:
        """Extract propositions from a single sentence.

        Args:
            sentence: Sentence text.
            sentence_idx: Index of sentence in paragraph.
            all_citations: All citations in the paragraph.

        Returns:
            List of PropositionNode instances.
        """
        # Check if this is a quotation
        is_quotation = bool(self.QUOTATION_PATTERN.search(sentence))

        # Extract citations attached to this sentence
        attached_citations = [
            cit for cit, pos in all_citations
            if cit in sentence
        ]

        # Remove citations for parsing
        clean_sentence = self.CITATION_PATTERN.sub('', sentence).strip()

        if not clean_sentence:
            return []

        # Parse with spaCy
        doc = self.nlp(clean_sentence)

        # Extract SVO triples
        triples = self._extract_svo_triples(doc)

        # If no triples found, create a single proposition from the whole sentence
        if not triples:
            triples = [SVOTriple(
                subject=self._get_sentence_subject(doc),
                verb=self._get_main_verb(doc),
                object=self._get_sentence_object(doc),
                full_text=clean_sentence
            )]

        # Extract entities and keywords
        entities = [ent.text for ent in doc.ents]
        keywords = self._extract_keywords(doc)

        # Create proposition nodes
        propositions = []
        for i, triple in enumerate(triples):
            prop_id = f"p_{sentence_idx}_{i}_{uuid.uuid4().hex[:8]}"
            propositions.append(PropositionNode(
                id=prop_id,
                text=triple.full_text,
                subject=triple.subject,
                verb=triple.verb,
                object=triple.object,
                entities=entities,
                keywords=keywords,
                source_sentence_idx=sentence_idx,
                is_citation=bool(attached_citations),
                is_quotation=is_quotation,
                attached_citations=attached_citations
            ))

        return propositions

    def _extract_svo_triples(self, doc) -> List[SVOTriple]:
        """Extract subject-verb-object triples from parsed doc.

        Args:
            doc: spaCy Doc object.

        Returns:
            List of SVOTriple instances.
        """
        triples = []

        for token in doc:
            # Look for main verbs (ROOT or with nsubj)
            if token.pos_ == "VERB" and token.dep_ in ("ROOT", "conj"):
                subject = self._find_subject(token)
                obj = self._find_object(token)

                if subject:
                    # Get the clause text
                    clause_tokens = self._get_clause_tokens(token)
                    clause_text = " ".join(t.text for t in clause_tokens)

                    triples.append(SVOTriple(
                        subject=subject,
                        verb=token.lemma_,
                        object=obj,
                        full_text=clause_text
                    ))

        return triples

    def _find_subject(self, verb_token) -> Optional[str]:
        """Find the subject of a verb.

        Args:
            verb_token: spaCy Token for the verb.

        Returns:
            Subject text or None.
        """
        for child in verb_token.children:
            if child.dep_ in ("nsubj", "nsubjpass"):
                # Get the full noun phrase
                return self._get_noun_phrase(child)

        # Check in ancestors if verb is part of a clause
        for ancestor in verb_token.ancestors:
            if ancestor.pos_ == "VERB":
                for child in ancestor.children:
                    if child.dep_ in ("nsubj", "nsubjpass"):
                        return self._get_noun_phrase(child)

        return None

    def _find_object(self, verb_token) -> Optional[str]:
        """Find the direct object of a verb.

        Args:
            verb_token: spaCy Token for the verb.

        Returns:
            Object text or None.
        """
        for child in verb_token.children:
            if child.dep_ in ("dobj", "attr", "pobj"):
                return self._get_noun_phrase(child)

        # Check for prepositional objects
        for child in verb_token.children:
            if child.dep_ == "prep":
                for pobj in child.children:
                    if pobj.dep_ == "pobj":
                        return self._get_noun_phrase(pobj)

        return None

    def _get_noun_phrase(self, token) -> str:
        """Get the full noun phrase for a token.

        Args:
            token: spaCy Token at the head of a noun phrase.

        Returns:
            Full noun phrase text.
        """
        # Get all tokens in the subtree
        subtree_tokens = list(token.subtree)

        # Filter to reasonable phrase length
        if len(subtree_tokens) > 10:
            # Just use the token and its immediate modifiers
            tokens = [token]
            for child in token.children:
                if child.dep_ in ("det", "amod", "compound", "poss"):
                    tokens.append(child)
            subtree_tokens = sorted(tokens, key=lambda t: t.i)

        return " ".join(t.text for t in subtree_tokens)

    def _get_clause_tokens(self, verb_token) -> List:
        """Get all tokens belonging to a clause headed by a verb.

        Args:
            verb_token: spaCy Token for the verb.

        Returns:
            List of tokens in the clause.
        """
        # Get subtree but limit scope
        clause_tokens = []
        prev_token = None

        for token in verb_token.subtree:
            # Stop at coordinating conjunctions that start truly new clauses
            # but NOT at comparative constructs like "rather than", "as well as"
            if token.dep_ == "cc" and token.i > verb_token.i:
                # Check if preceded by "rather" or part of comparative construct
                is_comparative = (
                    prev_token and prev_token.text.lower() in ("rather", "as", "well", "other")
                ) or token.text.lower() in ("than", "as")

                if not is_comparative:
                    # Check if this truly starts a new independent clause
                    # by looking for a new subject/verb after it
                    remaining_tokens = [t for t in verb_token.subtree if t.i > token.i]
                    has_new_clause = any(
                        t.dep_ in ("nsubj", "nsubjpass") and t.head.i > token.i
                        for t in remaining_tokens
                    )
                    if has_new_clause:
                        break

            clause_tokens.append(token)
            prev_token = token

        # Sort by position
        clause_tokens.sort(key=lambda t: t.i)
        return clause_tokens

    def _get_sentence_subject(self, doc) -> str:
        """Get the main subject of a sentence.

        Args:
            doc: spaCy Doc object.

        Returns:
            Subject text or empty string.
        """
        for token in doc:
            if token.dep_ in ("nsubj", "nsubjpass"):
                return self._get_noun_phrase(token)
        return ""

    def _get_main_verb(self, doc) -> str:
        """Get the main verb of a sentence.

        Args:
            doc: spaCy Doc object.

        Returns:
            Verb lemma or empty string.
        """
        for token in doc:
            if token.dep_ == "ROOT" and token.pos_ == "VERB":
                return token.lemma_
        # Fallback to any verb
        for token in doc:
            if token.pos_ == "VERB":
                return token.lemma_
        return ""

    def _get_sentence_object(self, doc) -> Optional[str]:
        """Get the main object of a sentence.

        Args:
            doc: spaCy Doc object.

        Returns:
            Object text or None.
        """
        for token in doc:
            if token.dep_ in ("dobj", "attr"):
                return self._get_noun_phrase(token)
        return None

    def _extract_keywords(self, doc) -> List[str]:
        """Extract keywords (significant nouns and verbs) from doc.

        Args:
            doc: spaCy Doc object.

        Returns:
            List of lemmatized keywords.
        """
        keywords = []
        for token in doc:
            if token.pos_ in ("NOUN", "PROPN", "VERB") and not token.is_stop:
                lemma = token.lemma_.lower()
                if lemma not in keywords and len(lemma) > 2:
                    keywords.append(lemma)
        return keywords[:10]  # Limit to top 10

    def classify_rst_roles(self, propositions: List[PropositionNode]) -> List[PropositionNode]:
        """Classify propositions into nucleus (main claims) vs satellite (supporting).

        Uses spaCy to identify rhetorical roles based on:
        - Sentence position (first sentences often nuclei)
        - Discourse markers (e.g., "for example", "because")
        - Specificity (examples are more concrete/specific)

        Args:
            propositions: List of extracted propositions.

        Returns:
            Propositions with rst_role, rst_relation, and parent_nucleus_idx set.
        """
        if not propositions:
            return propositions

        last_nucleus_idx = 0

        for i, prop in enumerate(propositions):
            doc = self.nlp(prop.text)

            # Classify based on discourse markers and position
            role, relation = self._classify_rst_role(doc, i == 0)

            prop.rst_role = role
            prop.rst_relation = relation

            if role == "nucleus":
                last_nucleus_idx = i
                prop.parent_nucleus_idx = None
            else:
                prop.parent_nucleus_idx = last_nucleus_idx

        return propositions

    def _classify_rst_role(self, doc, is_first: bool) -> Tuple[str, str]:
        """Classify a single sentence's RST role.

        Args:
            doc: spaCy Doc object.
            is_first: Whether this is the first sentence.

        Returns:
            Tuple of (role, relation) where role is "nucleus" or "satellite"
            and relation is the type of connection.
        """
        text_lower = doc.text.lower()
        first_token = doc[0] if doc else None

        # Check for example markers - these are SATELLITES
        example_markers = [
            "for example", "for instance", "such as", "like",
            "consider", "take", "imagine", "suppose"
        ]
        if any(marker in text_lower for marker in example_markers):
            return ("satellite", "example")

        # Check for evidence/citation markers - SATELLITES
        if "[^" in doc.text or "according to" in text_lower or "studies show" in text_lower:
            return ("satellite", "evidence")

        # Check for discourse connectors at start
        if first_token:
            first_pos = first_token.pos_
            first_lemma = first_token.lemma_.lower()

            # Contrast markers often indicate satellite elaboration
            if first_pos == "CCONJ" and first_lemma in ("but", "yet", "however"):
                return ("satellite", "contrast")

            # Causal markers - satellite providing reason
            if first_pos == "SCONJ" and first_lemma in ("because", "since", "as"):
                return ("satellite", "cause")

            # Conditional - satellite
            if first_lemma == "if":
                return ("satellite", "condition")

            # "This/These/That" referring back often indicates elaboration
            if first_lemma in ("this", "these", "that", "such") and first_pos == "DET":
                return ("satellite", "elaboration")

        # Check for specific/concrete entities (examples tend to be more concrete)
        named_entities = [ent for ent in doc.ents]
        concrete_nouns = [t for t in doc if t.pos_ == "NOUN" and not t.is_stop]

        # If sentence has specific named entities and isn't first, likely an example
        if named_entities and not is_first:
            # Check if entities are specific (person names, places, etc.)
            specific_ent_types = {"PERSON", "ORG", "GPE", "LOC", "PRODUCT"}
            if any(ent.label_ in specific_ent_types for ent in named_entities):
                return ("satellite", "example")

        # Modal verbs indicating claims/assertions are often nuclei
        has_modal_claim = any(
            t.tag_ == "MD" and t.lemma_ in ("must", "should", "will", "would", "can")
            for t in doc
        )
        if has_modal_claim:
            return ("nucleus", "none")

        # First sentence is typically nucleus
        if is_first:
            return ("nucleus", "none")

        # Default: check sentence complexity
        # Short, direct sentences after nuclei are often elaborations
        if len(doc) < 8:
            return ("satellite", "elaboration")

        return ("nucleus", "none")
