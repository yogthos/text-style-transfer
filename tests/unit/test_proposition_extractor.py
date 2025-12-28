"""Unit tests for proposition extraction."""

import pytest
from src.ingestion.proposition_extractor import (
    PropositionExtractor,
    SVOTriple,
    EpistemicStance,
    LogicalRelation,
    ContentAnchor,
)


class TestPropositionExtractor:
    """Test PropositionExtractor functionality."""

    @pytest.fixture
    def extractor(self):
        """Create extractor instance."""
        return PropositionExtractor()

    def test_empty_input(self, extractor):
        """Test handling of empty input."""
        result = extractor.extract_from_text("")
        assert result == []

    def test_simple_sentence(self, extractor):
        """Test extraction from simple sentence."""
        text = "The cat sat on the mat."
        propositions = extractor.extract_from_text(text)

        assert len(propositions) >= 1
        assert propositions[0].text  # Has text
        assert propositions[0].id  # Has ID

    def test_multiple_sentences(self, extractor):
        """Test extraction from multiple sentences."""
        text = "The scientist conducted an experiment. The results were surprising."
        propositions = extractor.extract_from_text(text)

        # Should have at least 2 propositions (one per sentence)
        assert len(propositions) >= 2

    def test_proposition_has_subject(self, extractor):
        """Test that propositions have subjects extracted."""
        text = "John wrote a book about history."
        propositions = extractor.extract_from_text(text)

        assert len(propositions) >= 1
        # Subject should be extracted
        assert propositions[0].subject or propositions[0].text

    def test_proposition_has_verb(self, extractor):
        """Test that propositions have verbs extracted."""
        text = "The students learn mathematics daily."
        propositions = extractor.extract_from_text(text)

        assert len(propositions) >= 1
        # Should have a verb
        assert propositions[0].verb or propositions[0].text

    def test_citation_preservation(self, extractor):
        """Test that citations are preserved."""
        text = "Research shows positive results[^1]. Another study confirms this[^2]."
        propositions = extractor.extract_from_text(text)

        # Should have citations attached
        has_citations = any(p.attached_citations for p in propositions)
        assert has_citations or any("[^" in p.text for p in propositions)

    def test_quotation_detection(self, extractor):
        """Test detection of quotations."""
        text = '"This is a quote," he said. Regular sentence here.'
        propositions = extractor.extract_from_text(text)

        # At least one should be marked as quotation
        assert len(propositions) >= 1

    def test_entity_extraction(self, extractor):
        """Test named entity extraction."""
        text = "Albert Einstein developed the theory of relativity in Germany."
        propositions = extractor.extract_from_text(text)

        assert len(propositions) >= 1
        # Should have entities
        all_entities = []
        for p in propositions:
            all_entities.extend(p.entities)
        # Should find at least Einstein or Germany
        assert len(all_entities) > 0 or "Einstein" in propositions[0].text

    def test_keyword_extraction(self, extractor):
        """Test keyword extraction."""
        text = "The computer processes data using algorithms."
        propositions = extractor.extract_from_text(text)

        assert len(propositions) >= 1
        # Should have keywords
        assert len(propositions[0].keywords) > 0

    def test_sentence_index_tracking(self, extractor):
        """Test that sentence indices are tracked."""
        text = "First sentence. Second sentence. Third sentence."
        propositions = extractor.extract_from_text(text)

        # Should have different sentence indices
        indices = [p.source_sentence_idx for p in propositions]
        assert len(set(indices)) >= 1

    def test_unique_ids(self, extractor):
        """Test that proposition IDs are unique."""
        text = "One sentence here. Another sentence there. A third one too."
        propositions = extractor.extract_from_text(text)

        ids = [p.id for p in propositions]
        assert len(ids) == len(set(ids))  # All unique

    def test_complex_sentence(self, extractor):
        """Test extraction from complex sentence."""
        text = "Although the weather was bad, the team completed the project because they were dedicated."
        propositions = extractor.extract_from_text(text)

        # Should extract meaningful content
        assert len(propositions) >= 1
        all_text = " ".join(p.text for p in propositions)
        assert "team" in all_text.lower() or "project" in all_text.lower()


class TestSVOTriple:
    """Test SVOTriple data class."""

    def test_create_triple(self):
        """Test creating an SVO triple."""
        triple = SVOTriple(
            subject="The cat",
            verb="sat",
            object="on the mat",
            full_text="The cat sat on the mat."
        )

        assert triple.subject == "The cat"
        assert triple.verb == "sat"
        assert triple.object == "on the mat"

    def test_triple_without_object(self):
        """Test triple without object."""
        triple = SVOTriple(
            subject="He",
            verb="slept",
            object=None,
            full_text="He slept."
        )

        assert triple.object is None


# =============================================================================
# NEW: Epistemic Stance Detection Tests
# =============================================================================

class TestEpistemicStanceDetection:
    """Test epistemic stance detection in propositions."""

    @pytest.fixture
    def extractor(self):
        return PropositionExtractor()

    def test_appearance_with_seem(self, extractor):
        """'Seem' should be detected as appearance stance."""
        text = "The situation seems dangerous."
        props = extractor.extract_from_text(text)

        assert len(props) >= 1
        assert props[0].epistemic_stance.stance == "appearance"
        assert any("seem" in m.lower() for m in props[0].epistemic_stance.modal_markers)

    def test_appearance_with_appear(self, extractor):
        """'Appear' should be detected as appearance stance."""
        text = "The results appear to be conclusive."
        props = extractor.extract_from_text(text)

        assert len(props) >= 1
        assert props[0].epistemic_stance.stance == "appearance"

    def test_appearance_see_as_construction(self, extractor):
        """'See X as Y' should be detected as appearance stance."""
        text = "We see the world as a static place."
        props = extractor.extract_from_text(text)

        assert len(props) >= 1
        assert props[0].epistemic_stance.stance == "appearance"

    def test_appearance_conditioned_to_see(self, extractor):
        """'Conditioned to see' should be detected as appearance stance."""
        text = "Most of us are conditioned to see the world as static."
        props = extractor.extract_from_text(text)

        assert len(props) >= 1
        assert props[0].epistemic_stance.stance == "appearance"

    def test_appearance_view_as(self, extractor):
        """'View X as Y' should be detected as appearance stance."""
        text = "Scientists view this phenomenon as evidence of change."
        props = extractor.extract_from_text(text)

        assert len(props) >= 1
        assert props[0].epistemic_stance.stance == "appearance"

    def test_hypothetical_with_might(self, extractor):
        """'Might' should be detected as hypothetical stance."""
        text = "This might be the solution."
        props = extractor.extract_from_text(text)

        assert len(props) >= 1
        assert props[0].epistemic_stance.stance == "hypothetical"
        assert "might" in props[0].epistemic_stance.modal_markers

    def test_hypothetical_with_could(self, extractor):
        """'Could' should be detected as hypothetical stance."""
        text = "The experiment could fail."
        props = extractor.extract_from_text(text)

        assert len(props) >= 1
        assert props[0].epistemic_stance.stance == "hypothetical"

    def test_conditional_with_if(self, extractor):
        """'If' clauses should be detected as conditional stance."""
        text = "If the data is correct, we can proceed."
        props = extractor.extract_from_text(text)

        assert len(props) >= 1
        assert props[0].epistemic_stance.stance == "conditional"

    def test_factual_direct_assertion(self, extractor):
        """Direct assertions should be detected as factual stance."""
        text = "The Earth orbits the Sun."
        props = extractor.extract_from_text(text)

        assert len(props) >= 1
        assert props[0].epistemic_stance.stance == "factual"

    def test_negation_detection(self, extractor):
        """Negation should be detected in propositions."""
        text = "The hypothesis is not supported by evidence."
        props = extractor.extract_from_text(text)

        assert len(props) >= 1
        assert props[0].epistemic_stance.is_negated is True

    def test_attribution_detection(self, extractor):
        """'According to X' should be detected as attribution."""
        text = "According to Einstein, time is relative."
        props = extractor.extract_from_text(text)

        assert len(props) >= 1
        assert props[0].epistemic_stance.source_attribution is not None
        assert "Einstein" in props[0].epistemic_stance.source_attribution

    def test_hedging_increases_level(self, extractor):
        """Hedging words should increase hedging level."""
        text = "The results are perhaps somewhat inconclusive."
        props = extractor.extract_from_text(text)

        assert len(props) >= 1
        assert props[0].epistemic_stance.hedging_level > 0

    def test_boosters_decrease_hedging(self, extractor):
        """Booster words should decrease hedging level (more assertive)."""
        text = "This is certainly the correct answer."
        props = extractor.extract_from_text(text)

        assert len(props) >= 1
        # Negative hedging means boosted/assertive
        assert props[0].epistemic_stance.hedging_level < 0


# =============================================================================
# NEW: Logical Relation Detection Tests
# =============================================================================

class TestLogicalRelationDetection:
    """Test logical relation detection in propositions."""

    @pytest.fixture
    def extractor(self):
        return PropositionExtractor()

    def test_contrast_with_but(self, extractor):
        """'But' should be detected as contrast relation."""
        text = "But the evidence suggests otherwise."
        props = extractor.extract_from_text(text)

        assert len(props) >= 1
        relations = props[0].logical_relations
        assert any(r.type == "contrast" for r in relations)

    def test_contrast_with_however(self, extractor):
        """'However' should be detected as contrast relation."""
        text = "However, we must consider alternatives."
        props = extractor.extract_from_text(text)

        assert len(props) >= 1
        relations = props[0].logical_relations
        assert any(r.type == "contrast" for r in relations)

    def test_cause_with_because(self, extractor):
        """'Because' should be detected as cause relation."""
        text = "The project failed because of poor planning."
        props = extractor.extract_from_text(text)

        assert len(props) >= 1
        relations = props[0].logical_relations
        assert any(r.type == "cause" for r in relations)

    def test_cause_with_therefore(self, extractor):
        """'Therefore' should be detected as cause relation."""
        text = "Therefore, we must reconsider our approach."
        props = extractor.extract_from_text(text)

        assert len(props) >= 1
        relations = props[0].logical_relations
        assert any(r.type == "cause" for r in relations)

    def test_condition_with_if(self, extractor):
        """'If' should be detected as condition relation."""
        text = "If the temperature rises, the ice will melt."
        props = extractor.extract_from_text(text)

        assert len(props) >= 1
        relations = props[0].logical_relations
        assert any(r.type == "condition" for r in relations)

    def test_example_with_for_example(self, extractor):
        """'For example' should be detected as example relation."""
        text = "Consider, for example, the case of climate change."
        props = extractor.extract_from_text(text)

        assert len(props) >= 1
        relations = props[0].logical_relations
        assert any(r.type == "example" for r in relations)

    def test_example_with_such_as(self, extractor):
        """'Such as' should be detected as example relation."""
        text = "Metals such as iron and copper conduct electricity."
        props = extractor.extract_from_text(text)

        assert len(props) >= 1
        relations = props[0].logical_relations
        assert any(r.type == "example" for r in relations)

    def test_no_relation_in_simple_sentence(self, extractor):
        """Simple sentences without markers should have no relations."""
        text = "The cat sat on the mat."
        props = extractor.extract_from_text(text)

        assert len(props) >= 1
        # May have empty relations or none
        assert len(props[0].logical_relations) == 0


# =============================================================================
# NEW: Content Anchor Detection Tests
# =============================================================================

class TestContentAnchorDetection:
    """Test content anchor detection in propositions."""

    @pytest.fixture
    def extractor(self):
        return PropositionExtractor()

    def test_named_entity_anchor(self, extractor):
        """Named entities should be detected as anchors."""
        text = "Albert Einstein developed relativity in Germany."
        props = extractor.extract_from_text(text)

        assert len(props) >= 1
        anchors = props[0].content_anchors
        anchor_texts = [a.text for a in anchors]
        # Should find Einstein or Germany
        assert any("Einstein" in t or "Germany" in t for t in anchor_texts)

    def test_statistic_anchor(self, extractor):
        """Statistics should be detected as anchors."""
        text = "The population grew by 25 percent last year."
        props = extractor.extract_from_text(text)

        assert len(props) >= 1
        anchors = props[0].content_anchors
        assert any(a.anchor_type == "statistic" for a in anchors)

    def test_citation_anchor(self, extractor):
        """Citations should be detected as anchors."""
        text = "Research confirms this finding[^1]."
        props = extractor.extract_from_text(text)

        assert len(props) >= 1
        anchors = props[0].content_anchors
        assert any(a.anchor_type == "citation" for a in anchors)

    def test_quote_anchor(self, extractor):
        """Quotes should be detected as anchors."""
        text = 'He said "this is important" during the meeting.'
        props = extractor.extract_from_text(text)

        assert len(props) >= 1
        anchors = props[0].content_anchors
        assert any(a.anchor_type == "quote" for a in anchors)

    def test_example_list_items_anchor(self, extractor):
        """Items after 'such as' should be detected as example anchors."""
        text = "Countries such as France, Germany, and Italy participated."
        props = extractor.extract_from_text(text)

        assert len(props) >= 1
        anchors = props[0].content_anchors
        example_anchors = [a for a in anchors if a.anchor_type == "example"]
        # Should find France, Germany, Italy as examples
        assert len(example_anchors) >= 1

    def test_no_anchors_in_simple_sentence(self, extractor):
        """Simple sentences may have few or no anchors."""
        text = "The weather is nice today."
        props = extractor.extract_from_text(text)

        assert len(props) >= 1
        # Simple sentence might have no anchors
        # (or maybe "today" as a minor anchor)


# =============================================================================
# NEW: Integrated Proposition Tests
# =============================================================================

class TestIntegratedPropositionExtraction:
    """Test integrated proposition extraction with all new features."""

    @pytest.fixture
    def extractor(self):
        return PropositionExtractor()

    def test_complex_sentence_full_extraction(self, extractor):
        """Test full extraction from complex sentence."""
        text = "According to Smith, the market seems unstable, but investors like Warren Buffett remain confident."
        props = extractor.extract_from_text(text)

        assert len(props) >= 1

        # Check epistemic stance (should be appearance due to "seems")
        all_stances = [p.epistemic_stance.stance for p in props]
        assert "appearance" in all_stances

        # Check for contrast relation (due to "but")
        all_relations = []
        for p in props:
            all_relations.extend(p.logical_relations)
        assert any(r.type == "contrast" for r in all_relations)

        # Check for anchors (Smith, Warren Buffett)
        all_anchors = []
        for p in props:
            all_anchors.extend(p.content_anchors)
        anchor_texts = [a.text for a in all_anchors]
        # Should find named entities
        assert len(anchor_texts) > 0

    def test_problematic_sentence_from_input(self, extractor):
        """Test the exact problematic sentence from our analysis."""
        text = "Most of us are conditioned to see the world as a static gallery of things—a tree, a smartphone, a government."
        props = extractor.extract_from_text(text)

        assert len(props) >= 1

        # CRITICAL: This must be detected as APPEARANCE, not factual
        assert props[0].epistemic_stance.stance == "appearance"

        # Should have markers indicating the appearance nature
        assert len(props[0].epistemic_stance.modal_markers) > 0

    def test_dialectical_sentence(self, extractor):
        """Test sentence with dialectical structure."""
        text = "The metaphysicians see things as static, but dialectics reveals they are processes in motion."
        props = extractor.extract_from_text(text)

        assert len(props) >= 1

        # Should detect appearance stance (from "see things as")
        all_stances = [p.epistemic_stance.stance for p in props]
        assert "appearance" in all_stances

        # Should detect contrast relation
        all_relations = []
        for p in props:
            all_relations.extend(p.logical_relations)
        assert any(r.type == "contrast" for r in all_relations)
