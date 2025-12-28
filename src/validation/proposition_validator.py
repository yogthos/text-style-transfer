"""Proposition-based validation for style transfer.

Tracks propositions from source text and validates they are preserved
in generated output. Provides specific repair instructions for missing
or hallucinated content.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Set
import re

from ..ingestion.proposition_extractor import (
    PropositionExtractor as RichPropositionExtractor,
    PropositionNode,
    ContentAnchor,
)
from ..utils.nlp import get_nlp, split_into_sentences
from ..utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class PropositionMatch:
    """Result of matching a source proposition to generated text."""
    proposition: PropositionNode
    is_preserved: bool
    match_score: float  # 0.0 to 1.0
    matched_text: Optional[str] = None  # The generated text that matches
    missing_elements: List[str] = field(default_factory=list)  # What's missing


@dataclass
class HallucinatedContent:
    """Content in generated text not found in source."""
    text: str
    content_type: str  # "entity", "claim", "example", "statistic"
    severity: str  # "critical", "warning"


@dataclass
class ValidationResult:
    """Complete result of proposition-based validation."""
    is_valid: bool
    proposition_coverage: float  # % of propositions preserved
    anchor_coverage: float  # % of content anchors preserved

    preserved_propositions: List[PropositionMatch] = field(default_factory=list)
    missing_propositions: List[PropositionMatch] = field(default_factory=list)
    hallucinated_content: List[HallucinatedContent] = field(default_factory=list)

    # Specific issues for repair
    missing_entities: List[str] = field(default_factory=list)
    missing_facts: List[str] = field(default_factory=list)
    added_entities: List[str] = field(default_factory=list)
    stance_violations: List[str] = field(default_factory=list)

    def get_repair_instructions(self) -> List[str]:
        """Generate specific repair instructions."""
        instructions = []

        # Missing content
        if self.missing_entities:
            instructions.append(
                f"INCLUDE these missing names/entities: {', '.join(self.missing_entities[:5])}"
            )

        if self.missing_facts:
            for fact in self.missing_facts[:3]:
                instructions.append(f"INCLUDE this fact: {fact}")

        # Hallucinated content
        if self.added_entities:
            instructions.append(
                f"REMOVE these added names/entities not in source: {', '.join(self.added_entities[:5])}"
            )

        # Stance violations
        if self.stance_violations:
            for violation in self.stance_violations[:2]:
                instructions.append(violation)

        return instructions


class PropositionValidator:
    """Validates generated text against source propositions.

    Uses rich proposition extraction to:
    1. Extract atomic propositions from source
    2. Check each proposition is preserved in output
    3. Identify hallucinated content not in source
    4. Generate specific repair instructions
    """

    def __init__(
        self,
        proposition_threshold: float = 0.7,  # Min coverage for propositions
        anchor_threshold: float = 0.8,  # Min coverage for content anchors
    ):
        self.proposition_threshold = proposition_threshold
        self.anchor_threshold = anchor_threshold
        self.prop_extractor = RichPropositionExtractor()
        self._nlp = None

    @property
    def nlp(self):
        if self._nlp is None:
            self._nlp = get_nlp()
        return self._nlp

    def extract_propositions(self, text: str) -> List[PropositionNode]:
        """Extract propositions from text."""
        return self.prop_extractor.extract_from_text(text)

    def validate(
        self,
        source_text: str,
        generated_text: str,
        source_propositions: Optional[List[PropositionNode]] = None,
    ) -> ValidationResult:
        """Validate generated text preserves source propositions.

        Args:
            source_text: Original source text.
            generated_text: Generated/transformed text.
            source_propositions: Pre-extracted propositions (optional).

        Returns:
            ValidationResult with detailed analysis.
        """
        # Extract propositions if not provided
        if source_propositions is None:
            source_propositions = self.extract_propositions(source_text)

        logger.debug(f"Validating {len(source_propositions)} propositions")

        # Parse generated text
        gen_doc = self.nlp(generated_text)
        gen_text_lower = generated_text.lower()

        # Track results
        preserved = []
        missing = []
        all_anchors = []
        preserved_anchors = 0

        # Check each proposition
        for prop in source_propositions:
            match = self._check_proposition_preserved(prop, generated_text, gen_doc)

            if match.is_preserved:
                preserved.append(match)
            else:
                missing.append(match)

            # Track anchor coverage
            for anchor in prop.content_anchors:
                all_anchors.append(anchor)
                if anchor.text.lower() in gen_text_lower:
                    preserved_anchors += 1

        # Check for hallucinated content
        hallucinated = self._find_hallucinations(source_text, generated_text, gen_doc)

        # Calculate coverage
        prop_coverage = len(preserved) / len(source_propositions) if source_propositions else 1.0
        anchor_coverage = preserved_anchors / len(all_anchors) if all_anchors else 1.0

        # Collect specific issues
        missing_entities = []
        missing_facts = []
        added_entities = []
        stance_violations = []

        for match in missing:
            # Extract what's specifically missing
            for anchor in match.proposition.content_anchors:
                if anchor.anchor_type == "entity" and anchor.text.lower() not in gen_text_lower:
                    missing_entities.append(anchor.text)

            # Check if the core claim is missing
            if match.match_score < 0.3:
                # Very low match - the whole proposition is missing
                missing_facts.append(self._summarize_proposition(match.proposition))

            # Check stance violations
            stance = match.proposition.epistemic_stance
            if stance.stance == "appearance":
                # Check if appearance markers are preserved
                has_appearance = any(
                    marker in gen_text_lower
                    for marker in ["seem", "appear", "look like", "as if"]
                )
                if not has_appearance:
                    stance_violations.append(
                        "PRESERVE epistemic stance: use 'seems/appears' - source presents as perception, not fact"
                    )

        # Collect hallucinated entities
        for h in hallucinated:
            if h.content_type == "entity":
                added_entities.append(h.text)

        # Determine validity
        is_valid = (
            prop_coverage >= self.proposition_threshold and
            anchor_coverage >= self.anchor_threshold and
            len([h for h in hallucinated if h.severity == "critical"]) == 0
        )

        return ValidationResult(
            is_valid=is_valid,
            proposition_coverage=prop_coverage,
            anchor_coverage=anchor_coverage,
            preserved_propositions=preserved,
            missing_propositions=missing,
            hallucinated_content=hallucinated,
            missing_entities=list(set(missing_entities)),
            missing_facts=missing_facts,
            added_entities=list(set(added_entities)),
            stance_violations=list(set(stance_violations)),
        )

    def _check_proposition_preserved(
        self,
        prop: PropositionNode,
        generated_text: str,
        gen_doc,
    ) -> PropositionMatch:
        """Check if a proposition is preserved in generated text."""
        gen_lower = generated_text.lower()
        prop_lower = prop.text.lower()

        # Calculate match score based on keyword overlap
        prop_keywords = set(prop.keywords)

        # Extract keywords from generated text
        gen_keywords = set()
        for token in gen_doc:
            if token.pos_ in ("NOUN", "PROPN", "VERB") and not token.is_stop:
                gen_keywords.add(token.lemma_.lower())

        # Calculate overlap
        if prop_keywords:
            overlap = prop_keywords & gen_keywords
            keyword_score = len(overlap) / len(prop_keywords)
        else:
            keyword_score = 0.5  # Default if no keywords

        # Check entity preservation
        entity_score = 1.0
        missing_elements = []

        for entity in prop.entities:
            if entity.lower() not in gen_lower:
                entity_score -= 1.0 / max(len(prop.entities), 1)
                missing_elements.append(f"entity: {entity}")

        # Check anchor preservation
        anchor_score = 1.0
        for anchor in prop.content_anchors:
            if anchor.must_preserve and anchor.text.lower() not in gen_lower:
                anchor_score -= 1.0 / max(len(prop.content_anchors), 1)
                missing_elements.append(f"{anchor.anchor_type}: {anchor.text}")

        # Combined score
        match_score = (keyword_score * 0.4 + entity_score * 0.3 + anchor_score * 0.3)

        # Find matching text (the sentence that best matches)
        matched_text = None
        best_sentence_score = 0

        for sent in gen_doc.sents:
            sent_keywords = set(
                t.lemma_.lower() for t in sent
                if t.pos_ in ("NOUN", "PROPN", "VERB") and not t.is_stop
            )
            if prop_keywords:
                sent_score = len(prop_keywords & sent_keywords) / len(prop_keywords)
                if sent_score > best_sentence_score:
                    best_sentence_score = sent_score
                    matched_text = sent.text

        return PropositionMatch(
            proposition=prop,
            is_preserved=match_score >= 0.5,
            match_score=match_score,
            matched_text=matched_text,
            missing_elements=missing_elements,
        )

    def _find_hallucinations(
        self,
        source_text: str,
        generated_text: str,
        gen_doc,
    ) -> List[HallucinatedContent]:
        """Find content in generated text not present in source."""
        hallucinated = []
        source_lower = source_text.lower()
        source_doc = self.nlp(source_text)

        # Extract source entities and concepts
        source_entities = set()
        for ent in source_doc.ents:
            source_entities.add(ent.text.lower())
            # Also add stems for partial matching
            source_entities.add(ent.text.lower()[:5] if len(ent.text) > 5 else ent.text.lower())

        source_concepts = set()
        for chunk in source_doc.noun_chunks:
            source_concepts.add(chunk.root.lemma_.lower())

        # Check generated entities
        for ent in gen_doc.ents:
            ent_lower = ent.text.lower()
            ent_stem = ent_lower[:5] if len(ent_lower) > 5 else ent_lower

            # Check if entity or its stem is in source
            if ent_lower not in source_lower and ent_stem not in source_entities:
                # Determine severity based on entity type
                if ent.label_ in ("PERSON", "ORG", "GPE", "EVENT", "DATE"):
                    severity = "critical"
                else:
                    severity = "warning"

                hallucinated.append(HallucinatedContent(
                    text=ent.text,
                    content_type="entity",
                    severity=severity,
                ))

        # Check for added claims (new verb phrases with subjects not in source)
        # This is a heuristic - look for new subject-verb patterns
        for token in gen_doc:
            if token.dep_ == "nsubj" and token.head.pos_ == "VERB":
                subject_text = token.text.lower()
                verb_text = token.head.lemma_.lower()

                # Check if this subject appears in source
                if subject_text not in source_lower and len(subject_text) > 3:
                    # New subject not in source - possible hallucination
                    if token.ent_type_ in ("PERSON", "ORG", "GPE"):
                        hallucinated.append(HallucinatedContent(
                            text=f"{token.text} {token.head.text}",
                            content_type="claim",
                            severity="warning",
                        ))

        return hallucinated

    def _summarize_proposition(self, prop: PropositionNode) -> str:
        """Create a short summary of a proposition for repair instructions."""
        # Use the first 100 chars or the subject-verb-object if available
        if prop.subject and prop.verb:
            summary = f"{prop.subject} {prop.verb}"
            if prop.object:
                summary += f" {prop.object}"
            return summary[:100]
        return prop.text[:100]

    def get_repair_prompt(
        self,
        author: str,
        source_text: str,
        validation_result: ValidationResult,
    ) -> str:
        """Generate a detailed repair prompt based on validation result.

        Args:
            author: Author name for style.
            source_text: Original source text.
            validation_result: Result from validate().

        Returns:
            System prompt for repair generation.
        """
        instructions = validation_result.get_repair_instructions()

        if not instructions:
            return f"""You are {author}. Rewrite the following text in your distinctive voice.

RULES:
1. Preserve ALL facts, names, and claims from the source
2. Do NOT add new information, examples, or claims
3. Complete all sentences
4. Vary your vocabulary"""

        # Build specific repair prompt
        instruction_text = "\n".join(f"- {inst}" for inst in instructions)

        return f"""You are {author}. Rewrite the following text in your distinctive voice.

CRITICAL - Fix these specific issues:
{instruction_text}

RULES:
1. Preserve ALL facts, names, and claims from the source
2. Do NOT add new information not in the source
3. Complete all sentences"""


def create_proposition_validator(
    proposition_threshold: float = 0.7,
    anchor_threshold: float = 0.8,
) -> PropositionValidator:
    """Create a proposition validator with specified thresholds."""
    return PropositionValidator(
        proposition_threshold=proposition_threshold,
        anchor_threshold=anchor_threshold,
    )
