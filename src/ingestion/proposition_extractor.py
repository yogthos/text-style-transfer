"""Proposition extraction from text using spaCy and LLM.

Enhanced with epistemic stance detection, logical relation preservation,
and content anchor identification for semantic fidelity in style transfer.
"""

import re
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict
import uuid

from ..utils.nlp import get_nlp, split_into_sentences, extract_citations
from ..utils.logging import get_logger

logger = get_logger(__name__)


# =============================================================================
# Epistemic and Modal Markers (for detecting stance)
# =============================================================================

# Markers that indicate APPEARANCE (things seem/appear to be X, not ARE X)
APPEARANCE_MARKERS = {
    "seem", "seems", "seemed", "seeming",
    "appear", "appears", "appeared", "appearing",
    "look", "looks", "looked", "looking",
}

APPEARANCE_PHRASES = [
    "as if", "as though", "looks like", "seems like", "appears to be",
    "gives the impression", "on the surface", "at first glance",
    "one might think", "it would seem", "apparently",
    # Perception-as constructions (see X as Y = perceive X to be Y, not X IS Y)
    "see it as", "see them as", "see this as", "see the", "see things as",
    "view it as", "view them as", "view this as", "view the",
    "perceive it as", "perceive them as", "perceive this as",
    "regard it as", "regard them as", "regard this as",
    "think of it as", "think of them as", "think of this as",
    "consider it", "consider them", "consider this",
    "treat it as", "treat them as", "treat this as",
    "conditioned to see", "conditioned to view", "conditioned to perceive",
]

# Regex patterns for more complex appearance constructions
APPEARANCE_PATTERNS = [
    r"see\s+(?:the\s+)?[\w\s]+\s+as\b",      # see X as Y
    r"view\s+(?:the\s+)?[\w\s]+\s+as\b",     # view X as Y (e.g., "view this phenomenon as")
    r"perceive\s+(?:the\s+)?[\w\s]+\s+as\b", # perceive X as Y
    r"regard\s+(?:the\s+)?[\w\s]+\s+as\b",   # regard X as Y
    r"treat\s+(?:the\s+)?[\w\s]+\s+as\b",    # treat X as Y
    r"think\s+of\s+[\w\s]+\s+as\b",          # think of X as Y
    r"considered\s+(?:to\s+be\s+)?",         # considered (to be) X
]

# Markers that indicate CONDITIONAL statements
CONDITIONAL_MARKERS = {"if", "unless", "provided", "assuming", "given", "when", "whenever"}

# Markers that indicate HYPOTHETICAL statements
HYPOTHETICAL_MODALS = {"would", "could", "might", "may"}

# Hedging markers (author distancing from claim)
HEDGING_WORDS = {
    "perhaps", "possibly", "probably", "maybe", "somewhat", "relatively",
    "fairly", "rather", "quite", "largely", "generally", "typically",
    "often", "sometimes", "occasionally", "tends", "suggest", "suggests",
    "indicate", "indicates", "imply", "implies", "appear", "appears",
}

HEDGING_PHRASES = [
    "to some extent", "in some ways", "in a sense", "more or less",
    "it is possible", "it seems", "it appears", "one might argue",
    "it could be", "there is evidence", "research suggests",
]

# Booster markers (author strengthening claim)
BOOSTER_WORDS = {
    "certainly", "definitely", "clearly", "obviously", "undoubtedly",
    "surely", "absolutely", "indeed", "truly", "necessarily",
    "always", "never", "must", "every", "all", "none", "fundamental",
    "essential", "critical", "crucial", "vital", "key", "primary",
}

# Attribution markers (claim sourced from another)
ATTRIBUTION_PATTERNS = [
    r"according to (\w+(?:\s+\w+)?)",
    r"(\w+(?:\s+\w+)?) (?:claims?|argues?|states?|suggests?|contends?|maintains?|asserts?)",
    r"as (\w+(?:\s+\w+)?) (?:noted|observed|pointed out|argued|claimed)",
]

# =============================================================================
# Logical Relation Markers
# =============================================================================

CONTRAST_MARKERS = {
    "but", "however", "yet", "although", "though", "whereas", "while",
    "nevertheless", "nonetheless", "on the other hand", "conversely",
    "in contrast", "despite", "even though", "still",
}

CAUSE_MARKERS = {
    "because", "since", "therefore", "thus", "hence", "consequently",
    "as a result", "for this reason", "due to", "owing to", "so",
    "accordingly", "thereby",
}

CONDITION_MARKERS = {
    "if", "unless", "provided that", "assuming", "given that",
    "in case", "on condition that", "supposing",
}

EXAMPLE_MARKERS = {
    "for example", "for instance", "such as", "like", "including",
    "e.g.", "namely", "specifically", "in particular", "consider",
}

REFORMULATION_MARKERS = {
    "that is", "in other words", "i.e.", "namely", "put differently",
    "to put it another way", "meaning", "which means",
}


@dataclass
class ContentAnchor:
    """Content that MUST be preserved verbatim in output."""
    text: str
    anchor_type: str  # "example", "statistic", "quote", "citation", "entity", "technical_term"
    must_preserve: bool = True
    context: str = ""  # What this anchor supports


@dataclass
class LogicalRelation:
    """Logical relation that must be preserved in output."""
    type: str  # "contrast", "cause", "condition", "example", "elaboration", "reformulation"
    source_marker: str  # The actual word/phrase that signals this
    must_preserve: bool = True


@dataclass
class EpistemicStance:
    """Epistemic stance information for a proposition."""
    stance: str = "factual"  # "factual", "appearance", "conditional", "hypothetical"
    hedging_level: float = 0.0  # 0.0 = absolute, 1.0 = fully hedged
    modal_markers: List[str] = field(default_factory=list)
    is_negated: bool = False
    source_attribution: Optional[str] = None  # "According to X"


@dataclass
class SVOTriple:
    """Subject-Verb-Object triple extracted from text."""
    subject: str
    verb: str
    object: Optional[str]
    full_text: str


@dataclass
class PropositionNode:
    """A node representing an atomic proposition in a semantic graph.

    Enhanced with epistemic stance detection, logical relations, and content anchors
    for semantic fidelity preservation during style transfer.
    """

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

    # === NEW: Epistemic stance (must be preserved in output) ===
    epistemic_stance: EpistemicStance = field(default_factory=EpistemicStance)

    # === NEW: Logical relations (must be preserved in output) ===
    logical_relations: List[LogicalRelation] = field(default_factory=list)

    # === NEW: Content anchors (must appear verbatim in output) ===
    content_anchors: List[ContentAnchor] = field(default_factory=list)


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

        # === NEW: Extract epistemic stance, logical relations, and anchors ===
        # Note: Use original sentence for anchors (to capture citations), clean for parsing
        epistemic_stance = self.extract_epistemic_stance(doc, clean_sentence)
        logical_relations = self.extract_logical_relations(doc, clean_sentence)
        content_anchors = self.extract_content_anchors(doc, sentence, entities)  # Use original sentence

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
                attached_citations=attached_citations,
                # === NEW: Add semantic fidelity fields ===
                epistemic_stance=epistemic_stance,
                logical_relations=logical_relations,
                content_anchors=content_anchors,
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

    # =========================================================================
    # NEW: Epistemic Stance Detection
    # =========================================================================

    def extract_epistemic_stance(self, doc, text: str) -> EpistemicStance:
        """Extract epistemic stance from a sentence.

        Detects whether the proposition is:
        - Factual: direct assertion of truth
        - Appearance: stated as how things seem/appear (not necessarily true)
        - Conditional: dependent on a condition
        - Hypothetical: speculative, using modal verbs like could/might

        Args:
            doc: spaCy Doc object.
            text: Original text.

        Returns:
            EpistemicStance with detected stance, hedging level, and markers.
        """
        text_lower = text.lower()
        stance = "factual"
        hedging_level = 0.0
        modal_markers = []
        is_negated = False
        source_attribution = None

        # Check for APPEARANCE markers (most important for semantic fidelity)
        for marker in APPEARANCE_MARKERS:
            if marker in text_lower:
                stance = "appearance"
                modal_markers.append(marker)

        for phrase in APPEARANCE_PHRASES:
            if phrase in text_lower:
                stance = "appearance"
                modal_markers.append(phrase)

        # Check for appearance patterns using regex
        for pattern in APPEARANCE_PATTERNS:
            match = re.search(pattern, text_lower)
            if match:
                stance = "appearance"
                modal_markers.append(match.group())

        # Check for CONDITIONAL markers
        if stance == "factual":
            for token in doc:
                if token.lemma_.lower() in CONDITIONAL_MARKERS and token.dep_ in ("mark", "advmod"):
                    stance = "conditional"
                    modal_markers.append(token.text)
                    break

        # Check for HYPOTHETICAL modals
        if stance == "factual":
            for token in doc:
                if token.tag_ == "MD" and token.lemma_.lower() in HYPOTHETICAL_MODALS:
                    stance = "hypothetical"
                    modal_markers.append(token.text)
                    break

        # Calculate hedging level
        hedge_count = 0
        boost_count = 0

        for token in doc:
            if token.lemma_.lower() in HEDGING_WORDS:
                hedge_count += 1
                modal_markers.append(token.text)

            if token.lemma_.lower() in BOOSTER_WORDS:
                boost_count += 1

        for phrase in HEDGING_PHRASES:
            if phrase in text_lower:
                hedge_count += 1
                modal_markers.append(phrase)

        # Hedging level: positive = hedged, negative = boosted
        total_markers = hedge_count + boost_count
        if total_markers > 0:
            hedging_level = (hedge_count - boost_count) / max(total_markers, 1)
            hedging_level = max(-1.0, min(1.0, hedging_level))

        # Check for negation
        for token in doc:
            if token.dep_ == "neg":
                is_negated = True
                break

        # Check for attribution
        for pattern in ATTRIBUTION_PATTERNS:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                source_attribution = match.group(1)
                break

        return EpistemicStance(
            stance=stance,
            hedging_level=hedging_level,
            modal_markers=list(set(modal_markers)),  # Deduplicate
            is_negated=is_negated,
            source_attribution=source_attribution
        )

    # =========================================================================
    # NEW: Logical Relation Detection
    # =========================================================================

    def extract_logical_relations(self, doc, text: str) -> List[LogicalRelation]:
        """Extract logical relations signaled in the text.

        Detects relations like:
        - Contrast: but, however, yet
        - Cause: because, therefore, thus
        - Condition: if, unless
        - Example: for example, such as
        - Reformulation: that is, in other words

        Args:
            doc: spaCy Doc object.
            text: Original text.

        Returns:
            List of LogicalRelation instances.
        """
        text_lower = text.lower()
        relations = []

        # Check for contrast markers
        for marker in CONTRAST_MARKERS:
            if marker in text_lower:
                # Verify it's used as a discourse marker, not just the word
                if self._is_discourse_marker(doc, marker):
                    relations.append(LogicalRelation(
                        type="contrast",
                        source_marker=marker,
                        must_preserve=True
                    ))
                    break  # Only record first contrast marker

        # Check for cause markers
        for marker in CAUSE_MARKERS:
            if marker in text_lower:
                if self._is_discourse_marker(doc, marker):
                    relations.append(LogicalRelation(
                        type="cause",
                        source_marker=marker,
                        must_preserve=True
                    ))
                    break

        # Check for condition markers
        for marker in CONDITION_MARKERS:
            if marker in text_lower:
                relations.append(LogicalRelation(
                    type="condition",
                    source_marker=marker,
                    must_preserve=True
                ))
                break

        # Check for example markers
        for marker in EXAMPLE_MARKERS:
            if marker in text_lower:
                relations.append(LogicalRelation(
                    type="example",
                    source_marker=marker,
                    must_preserve=True
                ))
                break

        # Check for reformulation markers
        for marker in REFORMULATION_MARKERS:
            if marker in text_lower:
                relations.append(LogicalRelation(
                    type="reformulation",
                    source_marker=marker,
                    must_preserve=True
                ))
                break

        return relations

    def _is_discourse_marker(self, doc, marker: str) -> bool:
        """Check if a marker is used as a discourse connector.

        Args:
            doc: spaCy Doc object.
            marker: The marker word/phrase.

        Returns:
            True if used as discourse marker.
        """
        # Single word markers
        marker_words = marker.split()
        if len(marker_words) == 1:
            for token in doc:
                if token.text.lower() == marker:
                    # Discourse markers are typically at start or after comma
                    if token.i == 0 or (token.i > 0 and doc[token.i - 1].text == ","):
                        return True
                    # Or used as SCONJ/CCONJ
                    if token.pos_ in ("SCONJ", "CCONJ", "ADV"):
                        return True
            return False
        else:
            # Multi-word markers - just check presence
            return True

    # =========================================================================
    # NEW: Content Anchor Detection
    # =========================================================================

    def extract_content_anchors(self, doc, text: str, entities: List[str]) -> List[ContentAnchor]:
        """Extract content that must be preserved verbatim in output.

        Anchors include:
        - Named entities in examples
        - Statistics and numbers
        - Direct quotes
        - Citations
        - Technical terms

        Args:
            doc: spaCy Doc object.
            text: Original text.
            entities: Already-extracted named entities.

        Returns:
            List of ContentAnchor instances.
        """
        anchors = []

        # 1. Named entities are anchors (especially in example contexts)
        for ent in doc.ents:
            # High-value entity types that must be preserved
            if ent.label_ in ("PERSON", "ORG", "GPE", "LOC", "PRODUCT", "EVENT", "WORK_OF_ART"):
                anchors.append(ContentAnchor(
                    text=ent.text,
                    anchor_type="entity",
                    must_preserve=True,
                    context=f"Named entity ({ent.label_})"
                ))

        # 2. Statistics and numbers with units
        stat_pattern = r'\b(\d+(?:\.\d+)?(?:\s*(?:%|percent|million|billion|thousand|kg|km|miles?|hours?|years?|dollars?|\$))?)\b'
        for match in re.finditer(stat_pattern, text, re.IGNORECASE):
            stat_text = match.group(1)
            if len(stat_text) > 1:  # Skip single digits
                anchors.append(ContentAnchor(
                    text=stat_text,
                    anchor_type="statistic",
                    must_preserve=True,
                    context="Numeric data"
                ))

        # 3. Direct quotes (must be preserved exactly)
        quote_pattern = r'["\u201c]([^"\u201d]+)["\u201d]'
        for match in re.finditer(quote_pattern, text):
            quote_text = match.group(1)
            anchors.append(ContentAnchor(
                text=quote_text,
                anchor_type="quote",
                must_preserve=True,
                context="Direct quotation"
            ))

        # 4. Citations
        citation_pattern = r'\[\^?\d+\]'
        for match in re.finditer(citation_pattern, text):
            anchors.append(ContentAnchor(
                text=match.group(),
                anchor_type="citation",
                must_preserve=True,
                context="Citation reference"
            ))

        # 5. Items in example lists (after "such as", "like", "including")
        example_list_pattern = r'(?:such as|like|including|for example|e\.g\.)\s+([^.;]+)'
        for match in re.finditer(example_list_pattern, text, re.IGNORECASE):
            example_items = match.group(1)
            # Split by commas and 'and'
            items = re.split(r',\s*|\s+and\s+', example_items)
            for item in items:
                item = item.strip()
                if item and len(item) > 2:
                    anchors.append(ContentAnchor(
                        text=item,
                        anchor_type="example",
                        must_preserve=True,
                        context="Example item"
                    ))

        # 6. Technical terms (capitalized multi-word phrases not at sentence start)
        tech_term_pattern = r'(?<!^)(?<!\. )([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)'
        for match in re.finditer(tech_term_pattern, text):
            term = match.group(1)
            # Exclude if it's already an entity
            if term not in [a.text for a in anchors]:
                anchors.append(ContentAnchor(
                    text=term,
                    anchor_type="technical_term",
                    must_preserve=True,
                    context="Technical term"
                ))

        return anchors
