"""
Stage 1: Semantic Extraction Module

Extracts meaning from input text, independent of surface form.
Produces a structured JSON representation of:
- Entities: people, places, concepts, technical terms
- Claims: propositions and arguments
- Relationships: cause-effect, comparison, sequence
- Preserved elements: citations, proper nouns, technical terms
"""

import spacy
import re
import json
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict
from collections import defaultdict


@dataclass
class Entity:
    """A named entity or concept."""
    text: str
    label: str  # PERSON, ORG, CONCEPT, TECHNICAL, etc.
    start: int
    end: int


@dataclass
class Claim:
    """A proposition or statement being made."""
    text: str
    subject: str
    predicate: str
    objects: List[str]
    modifiers: List[str]
    citations: List[str]  # [^N] references
    confidence: float  # How certain the claim is expressed


@dataclass
class Relationship:
    """A relationship between concepts."""
    source: str
    target: str
    relation_type: str  # CAUSE_EFFECT, COMPARISON, SEQUENCE, PART_OF, etc.
    evidence: str  # The text that establishes this relationship


@dataclass
class SemanticContent:
    """Complete semantic extraction from a text."""
    entities: List[Entity]
    claims: List[Claim]
    relationships: List[Relationship]
    preserved_elements: Dict[str, List[str]]  # citations, technical_terms, proper_nouns
    paragraph_structure: List[Dict[str, Any]]  # Logical structure of paragraphs

    def to_dict(self) -> Dict:
        return {
            'entities': [asdict(e) for e in self.entities],
            'claims': [asdict(c) for c in self.claims],
            'relationships': [asdict(r) for r in self.relationships],
            'preserved_elements': self.preserved_elements,
            'paragraph_structure': self.paragraph_structure
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)


class SemanticExtractor:
    """
    Extracts semantic content from text using spaCy NLP.

    The goal is to capture WHAT is being said, not HOW it's said.
    This allows the synthesizer to re-express the same meaning
    in a completely different style.
    """

    def __init__(self):
        """Initialize the semantic extractor with spaCy model."""
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            print("Downloading spaCy model 'en_core_web_sm'...")
            import subprocess
            subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"])
            self.nlp = spacy.load("en_core_web_sm")

        # Patterns for identifying special elements
        self.citation_pattern = re.compile(r'\[\^(\d+)\]')
        self.technical_term_pattern = re.compile(r'\b[A-Z][a-z]*(?:[A-Z][a-z]*)+\b')  # CamelCase

        # Relationship markers
        self.cause_markers = {'because', 'since', 'therefore', 'thus', 'hence',
                             'consequently', 'as a result', 'due to', 'owing to',
                             'leads to', 'causes', 'results in'}
        self.contrast_markers = {'but', 'however', 'although', 'though', 'yet',
                                'nevertheless', 'nonetheless', 'contrary to',
                                'in contrast', 'on the other hand', 'whereas'}
        self.sequence_markers = {'first', 'then', 'next', 'finally', 'subsequently',
                                'before', 'after', 'previously', 'following'}
        self.comparison_markers = {'like', 'similar to', 'compared to', 'as...as',
                                  'more than', 'less than', 'unlike'}

        # Hedging words that indicate uncertainty
        self.hedging_words = {'might', 'may', 'could', 'possibly', 'perhaps',
                             'likely', 'probably', 'suggests', 'appears', 'seems'}

    def extract(self, text: str) -> SemanticContent:
        """
        Extract semantic content from input text.

        Args:
            text: Input text to analyze

        Returns:
            SemanticContent with entities, claims, relationships, and preserved elements
        """
        # Pre-process: extract citations before NLP
        preserved = self._extract_preserved_elements(text)

        # Process with spaCy
        doc = self.nlp(text)

        # Extract components
        entities = self._extract_entities(doc)
        claims = self._extract_claims(doc, preserved['citations'])
        relationships = self._extract_relationships(doc)
        paragraph_structure = self._analyze_paragraph_structure(text, doc)

        return SemanticContent(
            entities=entities,
            claims=claims,
            relationships=relationships,
            preserved_elements=preserved,
            paragraph_structure=paragraph_structure
        )

    def _extract_preserved_elements(self, text: str) -> Dict[str, List[str]]:
        """Extract elements that must be preserved exactly."""
        preserved = {
            'citations': [],
            'technical_terms': [],
            'proper_nouns': [],
            'numbers': [],
            'quoted_text': []
        }

        # Extract citations [^N]
        citations = self.citation_pattern.findall(text)
        preserved['citations'] = [f'[^{c}]' for c in citations]

        # Extract quoted text
        quotes = re.findall(r'"([^"]+)"', text)
        preserved['quoted_text'] = quotes

        # Extract numbers with context
        numbers = re.findall(r'\b\d+(?:\.\d+)?(?:\s*(?:percent|%|million|billion|years?|centuries?))?\b', text)
        preserved['numbers'] = numbers

        return preserved

    def _extract_entities(self, doc) -> List[Entity]:
        """Extract named entities and key concepts."""
        entities = []

        # Get spaCy named entities
        for ent in doc.ents:
            entities.append(Entity(
                text=ent.text,
                label=ent.label_,
                start=ent.start_char,
                end=ent.end_char
            ))

        # Extract noun chunks as concepts
        seen_texts = {e.text.lower() for e in entities}
        for chunk in doc.noun_chunks:
            # Skip if already captured as named entity
            if chunk.text.lower() in seen_texts:
                continue

            # Check if it's a significant concept (has adjectives or is compound)
            if len(list(chunk.root.children)) > 0 or chunk.root.pos_ == 'PROPN':
                entities.append(Entity(
                    text=chunk.text,
                    label='CONCEPT',
                    start=chunk.start_char,
                    end=chunk.end_char
                ))
                seen_texts.add(chunk.text.lower())

        return entities

    def _extract_claims(self, doc, citations: List[str]) -> List[Claim]:
        """Extract propositions and claims from each sentence."""
        claims = []

        for sent in doc.sents:
            claim = self._analyze_sentence_claim(sent, citations)
            if claim:
                claims.append(claim)

        return claims

    def _analyze_sentence_claim(self, sent, citations: List[str]) -> Optional[Claim]:
        """Analyze a sentence to extract its core claim."""
        # Find the root verb
        root = None
        for token in sent:
            if token.dep_ == 'ROOT':
                root = token
                break

        if not root:
            return None

        # Extract subject
        subject = ''
        for child in root.children:
            if child.dep_ in ('nsubj', 'nsubjpass'):
                # Get the full subject phrase
                subject = self._get_subtree_text(child)
                break

        # Extract objects
        objects = []
        for child in root.children:
            if child.dep_ in ('dobj', 'pobj', 'attr', 'acomp'):
                objects.append(self._get_subtree_text(child))

        # Extract modifiers (adverbs, prepositional phrases)
        modifiers = []
        for child in root.children:
            if child.dep_ in ('advmod', 'prep', 'advcl'):
                modifiers.append(self._get_subtree_text(child))

        # Check for citations in this sentence
        sent_citations = [c for c in citations if c in sent.text]

        # Determine confidence based on hedging
        confidence = self._calculate_confidence(sent)

        return Claim(
            text=sent.text.strip(),
            subject=subject,
            predicate=root.lemma_,
            objects=objects,
            modifiers=modifiers,
            citations=sent_citations,
            confidence=confidence
        )

    def _get_subtree_text(self, token) -> str:
        """Get the full text of a token's subtree."""
        subtree = list(token.subtree)
        subtree.sort(key=lambda t: t.i)
        return ' '.join(t.text for t in subtree)

    def _calculate_confidence(self, sent) -> float:
        """Calculate how certain a claim is expressed (1.0 = certain, 0.0 = uncertain)."""
        text_lower = sent.text.lower()

        # Check for hedging words
        hedge_count = sum(1 for word in self.hedging_words if word in text_lower)

        # Check for negation
        has_negation = any(token.dep_ == 'neg' for token in sent)

        # Check for conditional markers
        has_conditional = any(token.text.lower() in ('if', 'unless', 'whether') for token in sent)

        # Calculate confidence score
        confidence = 1.0
        confidence -= hedge_count * 0.15
        if has_negation:
            confidence -= 0.1
        if has_conditional:
            confidence -= 0.2

        return max(0.0, min(1.0, confidence))

    def _extract_relationships(self, doc) -> List[Relationship]:
        """Extract relationships between concepts."""
        relationships = []

        for sent in doc.sents:
            text_lower = sent.text.lower()

            # Check for cause-effect relationships
            for marker in self.cause_markers:
                if marker in text_lower:
                    rel = self._parse_cause_effect(sent, marker)
                    if rel:
                        relationships.append(rel)
                    break

            # Check for contrast relationships
            for marker in self.contrast_markers:
                if marker in text_lower:
                    rel = self._parse_contrast(sent, marker)
                    if rel:
                        relationships.append(rel)
                    break

            # Check for comparison relationships
            for marker in self.comparison_markers:
                if marker in text_lower:
                    rel = self._parse_comparison(sent, marker)
                    if rel:
                        relationships.append(rel)
                    break

        return relationships

    def _parse_cause_effect(self, sent, marker: str) -> Optional[Relationship]:
        """Parse a cause-effect relationship from a sentence."""
        text = sent.text
        marker_idx = text.lower().find(marker)

        if marker_idx == -1:
            return None

        # Determine which part is cause and which is effect based on marker
        if marker in ('because', 'since', 'due to', 'owing to'):
            # Effect comes before, cause comes after
            effect = text[:marker_idx].strip()
            cause = text[marker_idx + len(marker):].strip()
        else:
            # Cause comes before, effect comes after
            cause = text[:marker_idx].strip()
            effect = text[marker_idx + len(marker):].strip()

        if cause and effect:
            return Relationship(
                source=cause,
                target=effect,
                relation_type='CAUSE_EFFECT',
                evidence=text
            )
        return None

    def _parse_contrast(self, sent, marker: str) -> Optional[Relationship]:
        """Parse a contrast relationship from a sentence."""
        text = sent.text
        marker_idx = text.lower().find(marker)

        if marker_idx == -1:
            return None

        first_part = text[:marker_idx].strip()
        second_part = text[marker_idx + len(marker):].strip()

        if first_part and second_part:
            return Relationship(
                source=first_part,
                target=second_part,
                relation_type='CONTRAST',
                evidence=text
            )
        return None

    def _parse_comparison(self, sent, marker: str) -> Optional[Relationship]:
        """Parse a comparison relationship from a sentence."""
        text = sent.text
        marker_idx = text.lower().find(marker)

        if marker_idx == -1:
            return None

        first_part = text[:marker_idx].strip()
        second_part = text[marker_idx + len(marker):].strip()

        if first_part and second_part:
            return Relationship(
                source=first_part,
                target=second_part,
                relation_type='COMPARISON',
                evidence=text
            )
        return None

    def _analyze_paragraph_structure(self, text: str, doc) -> List[Dict[str, Any]]:
        """Analyze the logical structure of paragraphs."""
        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
        structure = []

        for i, para in enumerate(paragraphs):
            para_doc = self.nlp(para)
            sentences = list(para_doc.sents)

            # Analyze paragraph function
            para_info = {
                'index': i,
                'sentence_count': len(sentences),
                'function': self._classify_paragraph_function(para_doc),
                'key_concepts': self._get_key_concepts(para_doc),
                'opening_type': self._classify_opening(sentences[0] if sentences else None),
                'closing_type': self._classify_closing(sentences[-1] if sentences else None),
                'logical_flow': self._analyze_logical_flow(sentences)
            }
            structure.append(para_info)

        return structure

    def _classify_paragraph_function(self, doc) -> str:
        """Classify what function a paragraph serves."""
        text_lower = doc.text.lower()

        # Check for introduction markers
        if any(marker in text_lower for marker in ['introduce', 'first', 'begin', 'overview']):
            return 'INTRODUCTION'

        # Check for conclusion markers
        if any(marker in text_lower for marker in ['conclude', 'finally', 'in summary', 'therefore']):
            return 'CONCLUSION'

        # Check for example/evidence
        if any(marker in text_lower for marker in ['for example', 'for instance', 'such as', 'evidence']):
            return 'EVIDENCE'

        # Check for argument/claim
        if any(marker in text_lower for marker in ['argue', 'claim', 'propose', 'suggest']):
            return 'ARGUMENT'

        # Check for explanation
        if any(marker in text_lower for marker in ['means', 'explains', 'reason', 'because']):
            return 'EXPLANATION'

        return 'EXPOSITION'

    def _get_key_concepts(self, doc) -> List[str]:
        """Extract key concepts from a paragraph."""
        # Get noun chunks that appear to be important
        concepts = []
        for chunk in doc.noun_chunks:
            # Filter out pronouns and very short chunks
            if chunk.root.pos_ != 'PRON' and len(chunk.text) > 2:
                concepts.append(chunk.text)

        # Deduplicate while preserving order
        seen = set()
        unique_concepts = []
        for c in concepts:
            c_lower = c.lower()
            if c_lower not in seen:
                seen.add(c_lower)
                unique_concepts.append(c)

        return unique_concepts[:10]  # Top 10 concepts

    def _classify_opening(self, sent) -> Optional[str]:
        """Classify how a sentence opens."""
        if not sent:
            return None

        first_token = list(sent)[0] if sent else None
        if not first_token:
            return 'UNKNOWN'

        # Check first token type
        if first_token.pos_ == 'PRON':
            return 'PRONOUN'
        elif first_token.pos_ == 'DET':
            return 'DETERMINER'
        elif first_token.pos_ == 'VERB':
            return 'VERB'
        elif first_token.pos_ == 'ADV':
            return 'ADVERB'
        elif first_token.pos_ == 'SCONJ':
            return 'SUBORDINATE'
        elif first_token.pos_ in ('NOUN', 'PROPN'):
            return 'NOUN'
        else:
            return 'OTHER'

    def _classify_closing(self, sent) -> Optional[str]:
        """Classify how a sentence closes."""
        if not sent:
            return None

        text = sent.text.strip()
        if text.endswith('?'):
            return 'QUESTION'
        elif text.endswith('!'):
            return 'EXCLAMATION'
        elif text.endswith('.'):
            return 'STATEMENT'
        else:
            return 'OTHER'

    def _analyze_logical_flow(self, sentences) -> List[str]:
        """Analyze the logical progression through sentences."""
        flow = []

        for sent in sentences:
            text_lower = sent.text.lower()
            first_word = list(sent)[0].text.lower() if sent else ''

            # Determine the sentence's role in logical flow
            if any(marker in text_lower for marker in self.cause_markers):
                flow.append('CAUSE_EFFECT')
            elif any(marker in text_lower for marker in self.contrast_markers):
                flow.append('CONTRAST')
            elif any(marker in text_lower for marker in self.sequence_markers):
                flow.append('SEQUENCE')
            elif first_word in ('this', 'that', 'these', 'those', 'it'):
                flow.append('CONTINUATION')
            elif '?' in sent.text:
                flow.append('QUESTION')
            else:
                flow.append('STATEMENT')

        return flow


def extract_semantics(text: str) -> SemanticContent:
    """Convenience function to extract semantics from text."""
    extractor = SemanticExtractor()
    return extractor.extract(text)


# Test function
if __name__ == '__main__':
    test_text = """Human experience reinforces the rule of finitude. The biological cycle of birth, life, and decay defines our reality. Every object we touch eventually breaks. Every star burning in the night sky eventually succumbs to erosion. But we encounter a logical trap when we apply that same finiteness to the universe itself. A cosmos with a definitive beginning and a hard boundary implies a container. Logic demands we ask a difficult question. If the universe has edges, what exists outside them?

A truly finite universe must exist within a larger context. Anything with limits implies the existence of an exterior. A bottle possesses a finite volume because the bottle sits within a room. The room exists within a house. The observable universe sits within the expanse of the greater cosmos. We must consider the possibility that our reality is a component of a grander whole.

We can resolve the paradox if we embrace the concept of an infinite cosmos. A system that stretches forever with no beginning or end requires no external container. The system is complete. But a structure without walls must rely on internal rules to hold its shape. Self-containment becomes a physical possibility if we treat information as a fundamental property of the universe. Tom Stonier proposed that information is interconvertible with energy and conserved alongside energy[^155]. The informational architecture of the cosmos provides the intrinsic scaffolding for the structure. The code is embedded in every particle and field. Such an internal framework eliminates the need for an external container. Cosmological models of an infinite universe support this view."""

    extractor = SemanticExtractor()
    result = extractor.extract(test_text)

    print("=== Semantic Extraction Results ===\n")
    print(f"Entities found: {len(result.entities)}")
    for e in result.entities[:5]:
        print(f"  - {e.text} ({e.label})")

    print(f"\nClaims found: {len(result.claims)}")
    for c in result.claims[:3]:
        print(f"  - Subject: {c.subject}")
        print(f"    Predicate: {c.predicate}")
        print(f"    Confidence: {c.confidence:.2f}")

    print(f"\nRelationships found: {len(result.relationships)}")
    for r in result.relationships[:3]:
        print(f"  - {r.relation_type}: {r.source[:30]}... -> {r.target[:30]}...")

    print(f"\nPreserved elements:")
    for key, values in result.preserved_elements.items():
        if values:
            print(f"  - {key}: {values}")

    print(f"\nParagraph structure: {len(result.paragraph_structure)} paragraphs")
    for p in result.paragraph_structure:
        print(f"  - Para {p['index']}: {p['function']} ({p['sentence_count']} sentences)")

