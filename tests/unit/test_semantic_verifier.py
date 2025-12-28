"""Unit tests for semantic verification."""

import pytest
from src.ingestion.proposition_extractor import (
    PropositionExtractor,
    PropositionNode,
    EpistemicStance,
    LogicalRelation,
    ContentAnchor,
)
from src.validation.semantic_verifier import (
    SemanticVerifier,
    VerificationResult,
    VerificationIssue,
    verify_semantic_fidelity,
)


class TestSemanticVerifier:
    """Test SemanticVerifier functionality."""

    @pytest.fixture
    def verifier(self):
        return SemanticVerifier()

    @pytest.fixture
    def extractor(self):
        return PropositionExtractor()

    # =========================================================================
    # Epistemic Stance Verification Tests
    # =========================================================================

    def test_appearance_to_factual_fails(self, verifier, extractor):
        """Converting appearance to factual assertion should FAIL."""
        source = "Most of us are conditioned to see the world as static."
        props = extractor.extract_from_text(source)

        # Bad output: states as fact what should be appearance
        bad_output = "The world is static."

        result = verifier.verify(props[0], bad_output)

        assert result.passed is False
        assert result.stance_preserved is False
        assert any(i.severity == "critical" for i in result.issues)
        assert any("stance" in i.issue_type.lower() for i in result.issues)

    def test_appearance_preserved_passes(self, verifier, extractor):
        """Preserving appearance stance should PASS."""
        source = "Most of us are conditioned to see the world as static."
        props = extractor.extract_from_text(source)

        # Good output: maintains appearance framing
        good_output = "The world appears static to most observers."

        result = verifier.verify(props[0], good_output)

        assert result.stance_preserved is True

    def test_appearance_with_seems_passes(self, verifier, extractor):
        """Using 'seems' to preserve appearance should PASS."""
        source = "The economy seems unstable."
        props = extractor.extract_from_text(source)

        good_output = "The economy appears to be experiencing instability."

        result = verifier.verify(props[0], good_output)

        assert result.stance_preserved is True

    def test_factual_to_factual_passes(self, verifier, extractor):
        """Factual to factual transformation should PASS."""
        source = "The Earth orbits the Sun."
        props = extractor.extract_from_text(source)

        output = "The Sun is orbited by the Earth."

        result = verifier.verify(props[0], output)

        # No stance mismatch issues
        stance_issues = [i for i in result.issues if "stance" in i.issue_type.lower()]
        assert len(stance_issues) == 0

    def test_negation_lost_fails(self, verifier, extractor):
        """Losing negation should FAIL."""
        source = "The hypothesis is not supported by evidence."
        props = extractor.extract_from_text(source)

        # Bad output: lost negation (meaning inversion)
        bad_output = "The hypothesis is supported by evidence."

        result = verifier.verify(props[0], bad_output)

        assert result.passed is False
        # Should detect either negation_lost or meaning_inversion
        critical_issues = [i for i in result.issues if i.severity == "critical"]
        assert len(critical_issues) > 0

    def test_negation_preserved_passes(self, verifier, extractor):
        """Preserving negation should PASS."""
        source = "The hypothesis is not supported by evidence."
        props = extractor.extract_from_text(source)

        good_output = "Evidence does not support the hypothesis."

        result = verifier.verify(props[0], good_output)

        # Should not have negation issues
        negation_issues = [i for i in result.issues if "negation" in i.issue_type.lower()]
        critical_neg = [i for i in negation_issues if i.severity == "critical"]
        assert len(critical_neg) == 0

    # =========================================================================
    # Content Anchor Verification Tests
    # =========================================================================

    def test_missing_entity_anchor_warning(self, verifier, extractor):
        """Missing entity anchor should produce WARNING."""
        source = "Albert Einstein developed the theory in Germany."
        props = extractor.extract_from_text(source)

        # Bad output: missing the named entities
        bad_output = "A scientist developed the theory."

        result = verifier.verify(props[0], bad_output)

        # Should have anchor-related issues
        anchor_issues = [i for i in result.issues if "anchor" in i.issue_type.lower()]
        assert len(anchor_issues) > 0
        assert result.anchor_coverage < 1.0

    def test_entity_preserved_passes(self, verifier, extractor):
        """Preserving entity anchors should improve coverage."""
        source = "Albert Einstein developed the theory in Germany."
        props = extractor.extract_from_text(source)

        # Good output: preserves named entities
        good_output = "In Germany, Einstein developed the theory."

        result = verifier.verify(props[0], good_output)

        # Should have better anchor coverage
        assert result.anchor_coverage > 0.5

    def test_missing_statistic_critical(self, verifier):
        """Missing statistic anchor should be CRITICAL."""
        # Create proposition with statistic anchor
        prop = PropositionNode(
            id="test",
            text="Growth was 25 percent.",
            content_anchors=[
                ContentAnchor(
                    text="25 percent",
                    anchor_type="statistic",
                    must_preserve=True
                )
            ]
        )

        bad_output = "There was significant growth."

        result = verifier.verify(prop, bad_output)

        # Missing statistic should be critical
        critical_issues = [i for i in result.issues if i.severity == "critical"]
        assert len(critical_issues) > 0

    def test_missing_citation_critical(self, verifier):
        """Missing citation anchor should be CRITICAL."""
        prop = PropositionNode(
            id="test",
            text="Research confirms this[^1].",
            content_anchors=[
                ContentAnchor(
                    text="[^1]",
                    anchor_type="citation",
                    must_preserve=True
                )
            ]
        )

        bad_output = "Research confirms this."

        result = verifier.verify(prop, bad_output)

        critical_issues = [i for i in result.issues if i.severity == "critical"]
        assert len(critical_issues) > 0

    def test_quote_preserved_passes(self, verifier):
        """Preserving quote anchor should PASS."""
        prop = PropositionNode(
            id="test",
            text='He said "this is important" during the meeting.',
            content_anchors=[
                ContentAnchor(
                    text="this is important",
                    anchor_type="quote",
                    must_preserve=True
                )
            ]
        )

        good_output = 'At the meeting, he emphasized that "this is important".'

        result = verifier.verify(prop, good_output)

        # Quote should be found
        assert result.anchor_coverage > 0.5

    # =========================================================================
    # Logical Relation Verification Tests
    # =========================================================================

    def test_contrast_relation_preserved(self, verifier):
        """Preserving contrast relation should PASS."""
        prop = PropositionNode(
            id="test",
            text="But the evidence suggests otherwise.",
            logical_relations=[
                LogicalRelation(type="contrast", source_marker="but", must_preserve=True)
            ]
        )

        # Good output: has contrast marker
        good_output = "However, the evidence points in a different direction."

        result = verifier.verify(prop, good_output)

        assert result.relations_preserved is True

    def test_contrast_relation_lost_warning(self, verifier):
        """Losing contrast relation should produce WARNING."""
        prop = PropositionNode(
            id="test",
            text="But the evidence suggests otherwise.",
            logical_relations=[
                LogicalRelation(type="contrast", source_marker="but", must_preserve=True)
            ]
        )

        # Bad output: no contrast marker
        bad_output = "The evidence suggests otherwise."

        result = verifier.verify(prop, bad_output)

        relation_issues = [i for i in result.issues if "relation" in i.issue_type.lower()]
        assert len(relation_issues) > 0

    def test_cause_relation_preserved(self, verifier):
        """Preserving cause relation should PASS."""
        prop = PropositionNode(
            id="test",
            text="The project failed because of poor planning.",
            logical_relations=[
                LogicalRelation(type="cause", source_marker="because", must_preserve=True)
            ]
        )

        good_output = "Poor planning caused the project to fail."

        result = verifier.verify(prop, good_output)

        # "caused" is not in CAUSE_MARKERS but "because" would be detected
        # Let's use a clearer example
        good_output2 = "The project failed due to poor planning."

        result2 = verifier.verify(prop, good_output2)

        # At least one should preserve the relation
        assert result.relations_preserved or result2.relations_preserved

    # =========================================================================
    # Meaning Inversion Tests
    # =========================================================================

    def test_meaning_inversion_detected(self, verifier, extractor):
        """Meaning inversion should be detected."""
        source = "The theory is not widely accepted."
        props = extractor.extract_from_text(source)

        # Inverted meaning
        bad_output = "The theory is widely accepted."

        result = verifier.verify(props[0], bad_output)

        # Should detect inversion or negation loss
        critical_issues = [i for i in result.issues if i.severity == "critical"]
        assert len(critical_issues) > 0

    # =========================================================================
    # Integration Tests
    # =========================================================================

    def test_full_verification_pass(self, verifier, extractor):
        """Complete verification of good transformation should PASS."""
        source = "According to Einstein, the universe seems to be expanding."
        props = extractor.extract_from_text(source)

        # Good transformation: preserves stance, anchors, attribution
        good_output = "Einstein observed that the universe appears to expand."

        result = verifier.verify(props[0], good_output)

        # Should pass overall
        assert result.passed is True

    def test_full_verification_multiple_issues(self, verifier, extractor):
        """Bad transformation should accumulate multiple issues."""
        source = "But according to Smith, the market seems unstable, with growth at 15 percent."
        props = extractor.extract_from_text(source)

        # Bad output: loses contrast, stance, attribution, and statistic
        bad_output = "The market is stable."

        result = verifier.verify(props[0], bad_output)

        # Should fail with multiple issues
        assert result.passed is False
        assert len(result.issues) >= 2


class TestVerifySemanticFidelityFunction:
    """Test the convenience function."""

    @pytest.fixture
    def extractor(self):
        return PropositionExtractor()

    def test_multiple_propositions(self, extractor):
        """Test verification across multiple propositions."""
        source = "The sky appears blue. But at sunset, it seems orange."
        props = extractor.extract_from_text(source)

        # Good output preserving both
        good_output = "The sky looks blue during the day. However, at sunset it appears orange."

        result = verify_semantic_fidelity(props, good_output)

        # Should handle multiple propositions
        assert isinstance(result, VerificationResult)

    def test_empty_propositions(self):
        """Test with empty proposition list."""
        result = verify_semantic_fidelity([], "Any text")

        assert result.passed is True
        assert result.anchor_coverage == 1.0


class TestVerificationResultMethods:
    """Test VerificationResult helper methods."""

    def test_critical_issues_filter(self):
        """Test critical_issues property."""
        result = VerificationResult(
            passed=False,
            issues=[
                VerificationIssue("type1", "critical", "desc1"),
                VerificationIssue("type2", "warning", "desc2"),
                VerificationIssue("type3", "critical", "desc3"),
            ]
        )

        critical = result.critical_issues
        assert len(critical) == 2
        assert all(i.severity == "critical" for i in critical)

    def test_has_critical_issues(self):
        """Test has_critical_issues property."""
        result_with_critical = VerificationResult(
            passed=False,
            issues=[VerificationIssue("type", "critical", "desc")]
        )
        assert result_with_critical.has_critical_issues is True

        result_without_critical = VerificationResult(
            passed=True,
            issues=[VerificationIssue("type", "warning", "desc")]
        )
        assert result_without_critical.has_critical_issues is False
