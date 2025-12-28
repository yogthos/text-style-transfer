"""Semantic verification for generated text.

Ensures that generated text preserves the semantic meaning of the source,
including epistemic stance, logical relations, and content anchors.
"""

import re
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from ..ingestion.proposition_extractor import (
    PropositionNode,
    EpistemicStance,
    LogicalRelation,
    ContentAnchor,
    APPEARANCE_MARKERS,
    APPEARANCE_PHRASES,
    APPEARANCE_PATTERNS,
    CONTRAST_MARKERS,
    CAUSE_MARKERS,
    CONDITION_MARKERS,
    EXAMPLE_MARKERS,
)
from ..utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class VerificationIssue:
    """A semantic verification issue."""
    issue_type: str  # "missing_anchor", "stance_mismatch", "relation_missing", "meaning_inversion"
    severity: str  # "critical", "warning", "info"
    description: str
    source_text: str = ""
    generated_text: str = ""


@dataclass
class VerificationResult:
    """Result of semantic verification."""
    passed: bool
    issues: List[VerificationIssue] = field(default_factory=list)
    anchor_coverage: float = 1.0  # Percentage of anchors preserved
    stance_preserved: bool = True
    relations_preserved: bool = True

    @property
    def critical_issues(self) -> List[VerificationIssue]:
        """Get only critical issues."""
        return [i for i in self.issues if i.severity == "critical"]

    @property
    def has_critical_issues(self) -> bool:
        """Check if there are any critical issues."""
        return len(self.critical_issues) > 0


class SemanticVerifier:
    """Verifies semantic fidelity between source and generated text.

    Checks:
    1. Content anchors are preserved (entities, examples, statistics)
    2. Epistemic stance is maintained (appearance vs factual)
    3. Logical relations are preserved (contrast, cause, condition)
    4. No meaning inversion (negation flip, conditional -> factual)
    """

    def verify(
        self,
        source_proposition: PropositionNode,
        generated_text: str,
    ) -> VerificationResult:
        """Verify that generated text preserves source semantics.

        Args:
            source_proposition: The source proposition with semantic metadata.
            generated_text: The generated text to verify.

        Returns:
            VerificationResult with pass/fail and detailed issues.
        """
        issues = []

        # 1. Check anchor preservation
        anchor_issues, anchor_coverage = self._verify_anchors(
            source_proposition.content_anchors,
            generated_text
        )
        issues.extend(anchor_issues)

        # 2. Check epistemic stance preservation
        stance_issues, stance_preserved = self._verify_stance(
            source_proposition.epistemic_stance,
            source_proposition.text,
            generated_text
        )
        issues.extend(stance_issues)

        # 3. Check logical relation preservation
        relation_issues, relations_preserved = self._verify_relations(
            source_proposition.logical_relations,
            generated_text
        )
        issues.extend(relation_issues)

        # 4. Check for meaning inversion
        inversion_issues = self._check_meaning_inversion(
            source_proposition,
            generated_text
        )
        issues.extend(inversion_issues)

        # Determine overall pass/fail
        has_critical = any(i.severity == "critical" for i in issues)
        passed = not has_critical and anchor_coverage >= 0.8

        return VerificationResult(
            passed=passed,
            issues=issues,
            anchor_coverage=anchor_coverage,
            stance_preserved=stance_preserved,
            relations_preserved=relations_preserved
        )

    def _verify_anchors(
        self,
        anchors: List[ContentAnchor],
        generated_text: str
    ) -> Tuple[List[VerificationIssue], float]:
        """Verify that content anchors are preserved.

        Args:
            anchors: List of content anchors from source.
            generated_text: Generated text to check.

        Returns:
            Tuple of (issues, coverage_ratio).
        """
        issues = []
        if not anchors:
            return issues, 1.0

        preserved = 0
        generated_lower = generated_text.lower()

        for anchor in anchors:
            if not anchor.must_preserve:
                preserved += 1
                continue

            anchor_lower = anchor.text.lower()

            # Check if anchor is present (case-insensitive)
            if anchor_lower in generated_lower:
                preserved += 1
            else:
                # For entities, check if a similar form is present
                # (e.g., "Smith" vs "Prof. Smith")
                words = anchor_lower.split()
                if any(word in generated_lower for word in words if len(word) > 3):
                    preserved += 0.5  # Partial match
                    issues.append(VerificationIssue(
                        issue_type="partial_anchor",
                        severity="warning",
                        description=f"Anchor '{anchor.text}' partially present",
                        source_text=anchor.text,
                        generated_text=generated_text[:100]
                    ))
                else:
                    severity = "critical" if anchor.anchor_type in ("quote", "citation", "statistic") else "warning"
                    issues.append(VerificationIssue(
                        issue_type="missing_anchor",
                        severity=severity,
                        description=f"Missing {anchor.anchor_type}: '{anchor.text}'",
                        source_text=anchor.text,
                        generated_text=generated_text[:100]
                    ))

        coverage = preserved / len(anchors) if anchors else 1.0
        return issues, coverage

    def _verify_stance(
        self,
        source_stance: EpistemicStance,
        source_text: str,
        generated_text: str
    ) -> Tuple[List[VerificationIssue], bool]:
        """Verify that epistemic stance is preserved.

        Critical check: If source says "seems/appears to be X", generated
        should NOT say "is X" (factual assertion of something only stated
        as appearance).

        Args:
            source_stance: Epistemic stance from source.
            source_text: Original source text.
            generated_text: Generated text to check.

        Returns:
            Tuple of (issues, stance_preserved).
        """
        issues = []
        generated_lower = generated_text.lower()

        # Most critical: appearance -> factual is a meaning inversion
        if source_stance.stance == "appearance":
            # Check if generated text maintains appearance markers
            has_appearance_marker = False

            for marker in APPEARANCE_MARKERS:
                if marker in generated_lower:
                    has_appearance_marker = True
                    break

            for phrase in APPEARANCE_PHRASES:
                if phrase in generated_lower:
                    has_appearance_marker = True
                    break

            for pattern in APPEARANCE_PATTERNS:
                if re.search(pattern, generated_lower):
                    has_appearance_marker = True
                    break

            if not has_appearance_marker:
                # This is a critical issue - appearance stated as fact
                issues.append(VerificationIssue(
                    issue_type="stance_mismatch",
                    severity="critical",
                    description="Appearance stance lost: source presents as appearance/perception, but generated states as fact",
                    source_text=source_text[:100],
                    generated_text=generated_text[:100]
                ))
                return issues, False

        # Conditional -> factual is also problematic
        if source_stance.stance == "conditional":
            has_conditional = any(
                marker in generated_lower
                for marker in ("if", "unless", "when", "provided", "assuming")
            )
            if not has_conditional:
                issues.append(VerificationIssue(
                    issue_type="stance_mismatch",
                    severity="warning",
                    description="Conditional stance may be lost",
                    source_text=source_text[:100],
                    generated_text=generated_text[:100]
                ))

        # Hypothetical -> factual
        if source_stance.stance == "hypothetical":
            has_modal = any(
                modal in generated_lower
                for modal in ("would", "could", "might", "may")
            )
            if not has_modal:
                issues.append(VerificationIssue(
                    issue_type="stance_mismatch",
                    severity="warning",
                    description="Hypothetical stance may be lost",
                    source_text=source_text[:100],
                    generated_text=generated_text[:100]
                ))

        # Check negation preservation
        if source_stance.is_negated:
            has_negation = any(
                neg in generated_lower
                for neg in ("not", "n't", "no ", "never", "neither", "nor")
            )
            if not has_negation:
                issues.append(VerificationIssue(
                    issue_type="negation_lost",
                    severity="critical",
                    description="Source negation not preserved in generated text",
                    source_text=source_text[:100],
                    generated_text=generated_text[:100]
                ))
                return issues, False

        return issues, len(issues) == 0

    def _verify_relations(
        self,
        relations: List[LogicalRelation],
        generated_text: str
    ) -> Tuple[List[VerificationIssue], bool]:
        """Verify that logical relations are preserved.

        Args:
            relations: Logical relations from source.
            generated_text: Generated text to check.

        Returns:
            Tuple of (issues, relations_preserved).
        """
        issues = []
        generated_lower = generated_text.lower()

        for relation in relations:
            if not relation.must_preserve:
                continue

            preserved = False

            # Check for the specific marker or equivalent
            if relation.type == "contrast":
                preserved = any(m in generated_lower for m in CONTRAST_MARKERS)
            elif relation.type == "cause":
                preserved = any(m in generated_lower for m in CAUSE_MARKERS)
            elif relation.type == "condition":
                preserved = any(m in generated_lower for m in CONDITION_MARKERS)
            elif relation.type == "example":
                preserved = any(m in generated_lower for m in EXAMPLE_MARKERS)
            else:
                # For other relations, check if the original marker is present
                preserved = relation.source_marker.lower() in generated_lower

            if not preserved:
                issues.append(VerificationIssue(
                    issue_type="relation_missing",
                    severity="warning",
                    description=f"Logical relation '{relation.type}' (marker: '{relation.source_marker}') not preserved",
                    source_text=relation.source_marker,
                    generated_text=generated_text[:100]
                ))

        return issues, len(issues) == 0

    def _check_meaning_inversion(
        self,
        source: PropositionNode,
        generated_text: str
    ) -> List[VerificationIssue]:
        """Check for meaning inversion patterns.

        Detects cases where the generated text says the opposite of the source.

        Args:
            source: Source proposition.
            generated_text: Generated text.

        Returns:
            List of inversion issues.
        """
        issues = []
        source_lower = source.text.lower()
        generated_lower = generated_text.lower()

        # Pattern 1: Source has negation, generated doesn't (or vice versa)
        source_negated = any(
            neg in source_lower
            for neg in ("not ", "n't", " no ", "never ", "neither", "nor ")
        )
        generated_negated = any(
            neg in generated_lower
            for neg in ("not ", "n't", " no ", "never ", "neither", "nor ")
        )

        # Only flag if one has negation and other doesn't
        # (both having or both lacking is fine)
        if source_negated != generated_negated:
            # Check if it's the same claim being negated/affirmed
            # by looking for shared key terms
            source_terms = set(source_lower.split())
            generated_terms = set(generated_lower.split())
            overlap = source_terms & generated_terms
            # Filter out stopwords
            overlap = {w for w in overlap if len(w) > 3}

            if len(overlap) >= 3:  # Likely discussing same topic
                issues.append(VerificationIssue(
                    issue_type="meaning_inversion",
                    severity="critical",
                    description="Possible meaning inversion: negation status differs between source and generated",
                    source_text=source.text[:100],
                    generated_text=generated_text[:100]
                ))

        return issues


def verify_semantic_fidelity(
    source_propositions: List[PropositionNode],
    generated_text: str,
) -> VerificationResult:
    """Convenience function to verify semantic fidelity.

    Args:
        source_propositions: List of source propositions.
        generated_text: Generated text to verify.

    Returns:
        Combined VerificationResult.
    """
    verifier = SemanticVerifier()
    all_issues = []
    anchor_coverages = []
    stance_preserved = True
    relations_preserved = True

    for prop in source_propositions:
        result = verifier.verify(prop, generated_text)
        all_issues.extend(result.issues)
        anchor_coverages.append(result.anchor_coverage)
        stance_preserved = stance_preserved and result.stance_preserved
        relations_preserved = relations_preserved and result.relations_preserved

    avg_coverage = sum(anchor_coverages) / len(anchor_coverages) if anchor_coverages else 1.0
    has_critical = any(i.severity == "critical" for i in all_issues)

    return VerificationResult(
        passed=not has_critical and avg_coverage >= 0.8,
        issues=all_issues,
        anchor_coverage=avg_coverage,
        stance_preserved=stance_preserved,
        relations_preserved=relations_preserved
    )
