"""Paragraph-level rhetorical graph for document coherence.

Tracks relationships between paragraphs to give LLM context about
document structure, preventing the isolated, mechanical generation
that flags AI-generated text.

Key insight: Human writers maintain a mental model of how paragraphs
relate to each other. LLMs generate each paragraph in isolation.
This module bridges that gap.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from enum import Enum
import re

from ..utils.nlp import get_nlp
from ..utils.logging import get_logger

logger = get_logger(__name__)


class ParagraphRelation(Enum):
    """Rhetorical relations between paragraphs."""

    INTRODUCTION = "introduction"  # Sets up the document thesis
    SUPPORTS = "supports"          # Provides evidence/argument for previous
    CONTRASTS = "contrasts"        # Opposes or qualifies previous
    ELABORATES = "elaborates"      # Expands on previous with more detail
    EXEMPLIFIES = "exemplifies"    # Provides concrete example of previous
    FOLLOWS_FROM = "follows_from"  # Logical consequence of previous
    TRANSITIONS = "transitions"    # Shifts to new topic/aspect
    CONCLUSION = "conclusion"      # Synthesizes/summarizes previous


@dataclass
class ParagraphNode:
    """A node in the paragraph graph."""

    index: int
    summary: str  # 1-2 sentence summary of paragraph content
    main_claim: str  # The central claim/topic of the paragraph
    key_entities: List[str] = field(default_factory=list)
    relation_to_previous: Optional[ParagraphRelation] = None
    relation_to_thesis: Optional[str] = None  # How it connects to document thesis

    def to_context_string(self) -> str:
        """Generate context string for prompt injection."""
        parts = [f"[P{self.index + 1}] {self.summary}"]
        if self.relation_to_previous:
            parts.append(f"  Relation: {self.relation_to_previous.value}")
        return "\n".join(parts)


@dataclass
class ParagraphGraph:
    """Graph tracking paragraph relationships across a document.

    This is the key to breaking AI detection patterns - it gives the LLM
    awareness of document structure so it can generate contextually
    appropriate text instead of isolated, mechanical sentences.
    """

    nodes: List[ParagraphNode] = field(default_factory=list)
    document_thesis: str = ""  # Main argument/thesis of document
    document_intent: str = ""  # Purpose: explain, argue, describe, etc.

    _nlp = None

    @property
    def nlp(self):
        if self._nlp is None:
            self._nlp = get_nlp()
        return self._nlp

    def add_source_paragraph(self, paragraph: str, index: int) -> ParagraphNode:
        """Analyze source paragraph and add to graph.

        Extracts:
        - Summary (first sentence or key claim)
        - Main claim/topic
        - Key entities
        - Relation to previous paragraph
        """
        doc = self.nlp(paragraph)
        sentences = list(doc.sents)

        # Extract summary (first sentence, cleaned up)
        summary = sentences[0].text.strip() if sentences else paragraph[:100]
        if len(summary) > 150:
            summary = summary[:147] + "..."

        # Extract main claim (look for assertive statements)
        main_claim = self._extract_main_claim(sentences)

        # Extract key entities
        entities = [ent.text for ent in doc.ents
                   if ent.label_ in ('ORG', 'PERSON', 'GPE', 'NORP', 'PRODUCT', 'EVENT')]

        # Determine relation to previous paragraph
        relation = self._classify_relation_to_previous(paragraph, index)

        # Set document thesis from first paragraph
        if index == 0:
            self.document_thesis = main_claim
            relation = ParagraphRelation.INTRODUCTION

        node = ParagraphNode(
            index=index,
            summary=summary,
            main_claim=main_claim,
            key_entities=entities[:5],  # Limit to 5
            relation_to_previous=relation,
        )

        self.nodes.append(node)
        logger.info(f"[GRAPH] P{index + 1}: {relation.value if relation else 'none'} - {summary[:50]}...")

        return node

    def _extract_main_claim(self, sentences) -> str:
        """Extract the main claim from paragraph sentences."""
        if not sentences:
            return ""

        # Look for sentences with strong verbs or assertions
        for sent in sentences[:3]:  # Check first 3 sentences
            text = sent.text.strip()

            # Skip questions
            if text.endswith('?'):
                continue

            # Prefer sentences with assertive patterns
            doc = self.nlp(text)
            has_root_verb = any(t.dep_ == 'ROOT' and t.pos_ == 'VERB' for t in doc)

            if has_root_verb and len(text.split()) >= 5:
                return text[:200]

        # Fallback to first sentence
        return sentences[0].text.strip()[:200]

    def _classify_relation_to_previous(
        self,
        paragraph: str,
        index: int
    ) -> Optional[ParagraphRelation]:
        """Classify how this paragraph relates to the previous one."""
        if index == 0:
            return ParagraphRelation.INTRODUCTION

        if not self.nodes:
            return ParagraphRelation.FOLLOWS_FROM

        doc = self.nlp(paragraph)
        first_sentence = list(doc.sents)[0].text.lower() if list(doc.sents) else ""

        # Check for explicit discourse markers
        contrast_markers = ['however', 'but', 'yet', 'although', 'nevertheless',
                           'on the other hand', 'in contrast', 'conversely']
        example_markers = ['for example', 'for instance', 'consider', 'such as',
                          'to illustrate', 'specifically']
        elaboration_markers = ['furthermore', 'moreover', 'in addition', 'also',
                              'indeed', 'in fact', 'that is']
        consequence_markers = ['therefore', 'thus', 'consequently', 'as a result',
                              'hence', 'so', 'this means']
        conclusion_markers = ['in conclusion', 'to summarize', 'finally',
                             'in summary', 'ultimately']

        # Check first sentence for markers
        for marker in contrast_markers:
            if marker in first_sentence:
                return ParagraphRelation.CONTRASTS

        for marker in example_markers:
            if marker in first_sentence:
                return ParagraphRelation.EXEMPLIFIES

        for marker in elaboration_markers:
            if marker in first_sentence:
                return ParagraphRelation.ELABORATES

        for marker in consequence_markers:
            if marker in first_sentence:
                return ParagraphRelation.FOLLOWS_FROM

        for marker in conclusion_markers:
            if marker in first_sentence:
                return ParagraphRelation.CONCLUSION

        # Check for entity continuity (shared entities = elaboration)
        if self.nodes:
            prev_entities = set(e.lower() for e in self.nodes[-1].key_entities)
            curr_entities = set(ent.text.lower() for ent in doc.ents)

            if prev_entities & curr_entities:
                return ParagraphRelation.ELABORATES

        # Default
        return ParagraphRelation.SUPPORTS

    def get_context_for_paragraph(self, index: int) -> str:
        """Generate context string for generating paragraph at index.

        This is injected into the LLM prompt to give it awareness of
        document structure and prevent isolated generation.
        """
        parts = []

        # Document-level context (keep brief)
        if self.document_thesis and len(self.document_thesis) < 100:
            parts.append(f"THESIS: {self.document_thesis[:100]}")

        # Previous paragraphs context - only show paragraphs BEFORE current index
        previous_nodes = [n for n in self.nodes if n.index < index]
        if previous_nodes:
            # Only show the immediately previous paragraph to avoid prompt bloat
            prev = previous_nodes[-1]
            parts.append(f"PREVIOUS: {prev.summary[:80]}...")
            if prev.relation_to_previous:
                parts.append(f"(was: {prev.relation_to_previous.value})")

        # Show relation for CURRENT paragraph if we have it
        if index < len(self.nodes):
            current = self.nodes[index]
            if current.relation_to_previous:
                parts.append(f"THIS PARAGRAPH: {current.relation_to_previous.value}")

        return "\n".join(parts) if parts else ""

    def get_transition_guidance(self, index: int) -> str:
        """Get guidance on how to transition into this paragraph.

        Returns specific instruction for opening the paragraph based on
        its relation to previous paragraphs.
        """
        if index == 0:
            return ""

        if not self.nodes:
            return ""

        prev_node = self.nodes[-1] if self.nodes else None

        # Based on previous paragraph, suggest transition
        guidance_map = {
            ParagraphRelation.INTRODUCTION: "Build on the introduction by...",
            ParagraphRelation.SUPPORTS: "Continue supporting the argument by...",
            ParagraphRelation.CONTRASTS: "Acknowledge the contrast but...",
            ParagraphRelation.ELABORATES: "Extend this elaboration by...",
            ParagraphRelation.EXEMPLIFIES: "Move from examples to broader points...",
            ParagraphRelation.FOLLOWS_FROM: "Draw the next logical consequence...",
        }

        if prev_node and prev_node.relation_to_previous:
            return guidance_map.get(prev_node.relation_to_previous, "")

        return ""

    def to_dict(self) -> Dict:
        """Serialize for debugging/logging."""
        return {
            "document_thesis": self.document_thesis,
            "document_intent": self.document_intent,
            "nodes": [
                {
                    "index": n.index,
                    "summary": n.summary,
                    "main_claim": n.main_claim,
                    "relation": n.relation_to_previous.value if n.relation_to_previous else None,
                    "entities": n.key_entities,
                }
                for n in self.nodes
            ]
        }


def build_paragraph_graph(source_paragraphs: List[str]) -> ParagraphGraph:
    """Build a paragraph graph from source document.

    This should be called BEFORE generation to analyze the source
    document structure, which is then used to guide generation.
    """
    graph = ParagraphGraph()

    for i, para in enumerate(source_paragraphs):
        graph.add_source_paragraph(para, i)

    logger.info(f"[GRAPH] Built graph with {len(graph.nodes)} paragraphs")
    logger.info(f"[GRAPH] Thesis: {graph.document_thesis[:80]}...")

    return graph
