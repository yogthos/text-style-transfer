"""
Stage 4: Verification Module

Validates synthesis output independently:
- Semantic check: Are all claims and facts from input present?
- Style check: Does output match sample's stylistic patterns?
- Structural check: Does output use patterns appropriate for each role?
- Preserved check: Are citations, technical terms intact?

Returns detailed TransformationHints for iterative refinement.
"""

import re
import json
import math
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, asdict, field
from collections import Counter
from pathlib import Path

from semantic_extractor import SemanticContent, SemanticExtractor, Claim
from style_analyzer import StyleProfile, StyleAnalyzer
from structural_analyzer import (
    StructuralAnalyzer,
    TransformationHint,
    SentenceAnalysis,
    StructuralPattern
)


@dataclass
class SemanticVerification:
    """Results of semantic verification."""
    passed: bool
    claim_coverage: float  # Percentage of claims preserved
    missing_claims: List[str]
    preserved_citations: bool
    missing_citations: List[str]
    preserved_technical_terms: bool
    missing_terms: List[str]
    meaning_drift_score: float  # 0 = no drift, 1 = complete drift


@dataclass
class StyleVerification:
    """Results of style verification."""
    passed: bool
    sentence_length_match: float  # How close to target distribution
    opener_diversity: float  # Variety of sentence openers
    vocabulary_match: float  # Overlap with target vocabulary patterns
    formality_match: float  # How close to target formality
    pattern_coverage: float  # NEW: How many distinctive patterns are used
    discourse_marker_usage: float  # NEW: How well discourse markers match target
    issues: List[str]


@dataclass
class PreservationVerification:
    """Results of preservation verification."""
    passed: bool
    citations_preserved: bool
    missing_citations: List[str]
    numbers_preserved: bool
    missing_numbers: List[str]
    quotes_preserved: bool
    missing_quotes: List[str]


@dataclass
class VerificationResult:
    """Complete verification result."""
    overall_passed: bool
    semantic: SemanticVerification
    style: StyleVerification
    preservation: PreservationVerification
    recommendations: List[str]
    transformation_hints: List[TransformationHint] = field(default_factory=list)
    iteration: int = 0
    improvement_score: float = 0.0  # Track improvement across iterations

    def to_dict(self) -> Dict:
        return {
            'overall_passed': self.overall_passed,
            'semantic': asdict(self.semantic),
            'style': asdict(self.style),
            'preservation': asdict(self.preservation),
            'recommendations': self.recommendations,
            'transformation_hints': [asdict(h) for h in self.transformation_hints],
            'iteration': self.iteration,
            'improvement_score': self.improvement_score
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    def get_hint_summary(self) -> str:
        """Generate a summary of hints for the LLM."""
        if not self.transformation_hints:
            return ""

        summary = ["## TRANSFORMATION HINTS (address these issues):\n"]

        # Group by priority
        priority_1 = [h for h in self.transformation_hints if h.priority == 1]
        priority_2 = [h for h in self.transformation_hints if h.priority == 2]
        priority_3 = [h for h in self.transformation_hints if h.priority == 3]

        if priority_1:
            summary.append("### CRITICAL (must fix):")
            for h in priority_1[:5]:
                summary.append(f"- Sentence {h.sentence_index + 1} ({h.structural_role}): {h.issue}")
                summary.append(f"  FIX: {h.suggestion}")

        if priority_2:
            summary.append("\n### IMPORTANT (should fix):")
            for h in priority_2[:5]:
                summary.append(f"- Sentence {h.sentence_index + 1}: {h.issue}")
                summary.append(f"  FIX: {h.suggestion}")

        if priority_3:
            summary.append("\n### MINOR (nice to have):")
            for h in priority_3[:3]:
                summary.append(f"- {h.issue}")

        return '\n'.join(summary)


class Verifier:
    """
    Verifies that synthesis output meets quality requirements.

    Four independent verification passes:
    1. Semantic: All meaning from input is preserved
    2. Style: Output matches target style patterns
    3. Structural: Output uses patterns appropriate for each role
    4. Preservation: Citations, terms, etc. are intact

    Returns TransformationHints for iterative refinement.
    """

    def __init__(self, config_path: str = None):
        """Initialize verifier with NLP tools and configurable thresholds."""
        self.semantic_extractor = SemanticExtractor()
        self.style_analyzer = StyleAnalyzer()
        self.structural_analyzer = StructuralAnalyzer()

        # Cache for structural patterns
        self._role_patterns = None
        self._sample_hash = None

        # Load thresholds from config or use defaults
        self._load_thresholds(config_path)

    def _load_thresholds(self, config_path: str = None):
        """Load verification thresholds from config file."""
        import json

        # Default thresholds
        self.min_claim_coverage = 0.80  # At least 80% of claims
        self.max_meaning_drift = 0.50  # Max 50% drift
        self.min_style_match = 0.60    # At least 60% style match
        self.min_structural_match = 0.40  # At least 40% structural match

        # Try to load from config
        if config_path is None:
            config_path = Path(__file__).parent / "config.json"
        else:
            config_path = Path(config_path)

        if config_path.exists():
            try:
                with open(config_path, 'r', encoding='utf-8') as f:
                    config = json.load(f)

                verification = config.get('verification', {})

                if 'max_meaning_drift' in verification:
                    self.max_meaning_drift = verification['max_meaning_drift']
                if 'min_claim_coverage' in verification:
                    self.min_claim_coverage = verification['min_claim_coverage']
                if 'min_style_match' in verification:
                    self.min_style_match = verification['min_style_match']
                if 'min_structural_match' in verification:
                    self.min_structural_match = verification['min_structural_match']

            except (json.JSONDecodeError, IOError):
                pass  # Use defaults if config can't be read

    def _load_sample_patterns(self) -> Dict[str, List[StructuralPattern]]:
        """Load structural patterns from sample text."""
        sample_path = Path(__file__).parent / "prompts" / "sample.txt"
        if sample_path.exists():
            with open(sample_path, 'r', encoding='utf-8') as f:
                sample_text = f.read()
            return self.structural_analyzer.analyze_sample(sample_text)
        return {}

    def get_role_patterns(self) -> Dict[str, List[StructuralPattern]]:
        """Get cached role patterns, loading if necessary."""
        if self._role_patterns is None:
            self._role_patterns = self._load_sample_patterns()
        return self._role_patterns

    def verify(self,
               input_text: str,
               output_text: str,
               input_semantics: Optional[SemanticContent] = None,
               target_style: Optional[StyleProfile] = None,
               iteration: int = 0,
               previous_score: float = 0.0) -> VerificationResult:
        """
        Verify synthesis output against input and target style.

        Args:
            input_text: Original input text
            output_text: Synthesized output text
            input_semantics: Pre-extracted input semantics (optional)
            target_style: Target style profile (optional)
            iteration: Current iteration number (for tracking improvement)
            previous_score: Score from previous iteration

        Returns:
            VerificationResult with pass/fail, details, and transformation hints
        """
        # Extract semantics if not provided
        if input_semantics is None:
            input_semantics = self.semantic_extractor.extract(input_text)

        # Run verification passes
        semantic_result = self._verify_semantics(input_text, output_text, input_semantics)
        style_result = self._verify_style(output_text, target_style)
        preservation_result = self._verify_preservation(input_semantics, output_text)

        # Determine overall pass/fail
        overall_passed = (
            semantic_result.passed and
            style_result.passed and
            preservation_result.passed
        )

        # Generate recommendations
        recommendations = self._generate_recommendations(
            semantic_result, style_result, preservation_result
        )

        # Generate transformation hints based on structural analysis
        transformation_hints = self._generate_transformation_hints(
            input_text, output_text, style_result, target_style
        )

        # Calculate overall improvement score
        current_score = self._calculate_overall_score(
            semantic_result, style_result, preservation_result
        )
        improvement = current_score - previous_score if previous_score > 0 else 0

        return VerificationResult(
            overall_passed=overall_passed,
            semantic=semantic_result,
            style=style_result,
            preservation=preservation_result,
            recommendations=recommendations,
            transformation_hints=transformation_hints,
            iteration=iteration,
            improvement_score=improvement
        )

    def _calculate_overall_score(self, semantic: SemanticVerification,
                                  style: StyleVerification,
                                  preservation: PreservationVerification) -> float:
        """Calculate an overall quality score for tracking improvement."""
        scores = [
            semantic.claim_coverage,
            1 - semantic.meaning_drift_score,
            style.sentence_length_match,
            style.pattern_coverage,
            style.discourse_marker_usage,
            style.formality_match,
            1.0 if preservation.citations_preserved else 0.0,
            1.0 if preservation.numbers_preserved else 0.0,
        ]
        return sum(scores) / len(scores)

    def _generate_transformation_hints(self,
                                        input_text: str,
                                        output_text: str,
                                        style_result: StyleVerification,
                                        target_style: Optional[StyleProfile]) -> List[TransformationHint]:
        """Generate detailed transformation hints based on structural analysis."""
        hints = []

        # Get structural patterns
        role_patterns = self.get_role_patterns()

        # Analyze input structure
        input_sentences = self.structural_analyzer.analyze_input_structure(input_text)

        # Generate hints from structural analyzer
        structural_hints = self.structural_analyzer.generate_transformation_hints(
            input_sentences, output_text, role_patterns
        )
        hints.extend(structural_hints)

        # Add hints based on style issues
        if style_result.pattern_coverage < 0.4:
            hints.append(TransformationHint(
                sentence_index=-1,  # Document-level
                current_text="",
                structural_role="document",
                expected_patterns=["section_opener patterns", "paragraph_opener patterns"],
                issue=f"Overall pattern coverage too low ({style_result.pattern_coverage:.0%})",
                suggestion="Use more distinctive patterns like 'Contrary to...', 'Hence,...', 'The X method therefore holds that...'",
                priority=1
            ))

        if style_result.discourse_marker_usage < 0.3:
            hints.append(TransformationHint(
                sentence_index=-1,
                current_text="",
                structural_role="document",
                expected_patterns=["therefore", "hence", "consequently", "however"],
                issue=f"Insufficient discourse markers ({style_result.discourse_marker_usage:.0%})",
                suggestion="Add connectors: 'therefore' at clause starts, 'however' parenthetically, 'hence' to start sentences",
                priority=1
            ))

        # Check for AI patterns that should be removed - CRITICAL priority
        ai_patterns = self._detect_ai_patterns(output_text)
        if ai_patterns:
            # All AI patterns are priority 1 (critical)
            for pattern in ai_patterns:
                hints.append(TransformationHint(
                    sentence_index=-1,
                    current_text=pattern,
                    structural_role="document",
                    expected_patterns=[],
                    issue=f"🚨 AI FINGERPRINT DETECTED: '{pattern}' - MUST BE REMOVED",
                    suggestion=f"ELIMINATE '{pattern}' completely. Use direct, assertive phrasing from the target style instead.",
                    priority=1
                ))

        return hints

    def _detect_ai_patterns(self, text: str) -> List[str]:
        """Detect AI-typical patterns that should be removed."""
        text_lower = text.lower()

        ai_patterns = [
            # === PHRASES (common AI slop) ===
            'conventional notions', 'local perspective', 'internal complexity',
            'grander whole', 'intrinsic scaffolding', 'broader context',
            'key insight', 'importantly', 'it is worth noting',
            'it should be noted', 'interestingly', 'notably',
            'unique perspective', 'nuanced understanding', 'rich tapestry',
            'serves as a', 'stands as a', 'acts as a reminder',
            'in the realm of', 'at its core', 'when it comes to',
            'in today\'s world', 'in this day and age', 'moving forward',
            'at the end of the day', 'the fact that', 'in order to',
            'due to the fact', 'for the purpose of', 'with regard to',
            'taking into account', 'a wide range of', 'a variety of',
            'plays a crucial role', 'is of utmost importance',
            'it goes without saying', 'needless to say',
            'last but not least', 'first and foremost',

            # === WEAK/HEDGING VERBS ===
            'this suggests', 'this implies', 'this indicates',
            'we can see', 'we observe', 'we find', 'we note',
            'could potentially', 'might possibly', 'seems to suggest',
            'appears to be', 'tends to be', 'is likely to',
            'may or may not', 'it seems', 'it appears',
            'one could argue', 'some might say', 'many believe',

            # === GENERIC OPENERS (AI loves these) ===
            'we must consider', 'we must acknowledge', 'we must recognize',
            'consider the fact', 'consider how', 'consider that',
            'it is important to', 'it is essential to', 'it is necessary to',
            'it is clear that', 'it is evident that', 'it is obvious that',
            'there is no doubt', 'without a doubt', 'undoubtedly',

            # === FILLER/PADDING ===
            'in essence', 'essentially', 'basically', 'fundamentally',
            'to put it simply', 'in other words', 'that is to say',
            'as such', 'as a result', 'as a consequence',
            'on the other hand', 'by the same token',
            'for instance', 'for example', 'such as',

            # === OVER-QUALIFIED STATEMENTS ===
            'to some extent', 'to a certain degree', 'in some ways',
            'in many ways', 'in a sense', 'in a way',
            'relatively speaking', 'generally speaking', 'broadly speaking',

            # === BUZZWORDS ===
            'paradigm shift', 'game changer', 'cutting edge',
            'state of the art', 'best practices', 'synergy',
            'leverage', 'optimize', 'streamline', 'holistic',
            'robust', 'scalable', 'innovative', 'transformative',

            # === SPECIFIC TO THIS OUTPUT ===
            'teeming with', 'sits within', 'operates under',
        ]

        found = []
        for pattern in ai_patterns:
            if pattern in text_lower:
                found.append(pattern)

        return found

    def _verify_semantics(self,
                          input_text: str,
                          output_text: str,
                          input_semantics: SemanticContent) -> SemanticVerification:
        """Verify semantic content is preserved."""
        # Extract semantics from output
        output_semantics = self.semantic_extractor.extract(output_text)

        # Check claim coverage
        input_claims = input_semantics.claims
        output_text_lower = output_text.lower()

        preserved_claims = []
        missing_claims = []

        for claim in input_claims:
            # Check if key concepts from claim appear in output
            claim_preserved = self._claim_appears_in_output(claim, output_text_lower)
            if claim_preserved:
                preserved_claims.append(claim.text)
            else:
                missing_claims.append(claim.text)

        claim_coverage = len(preserved_claims) / max(1, len(input_claims))

        # Check citations
        input_citations = input_semantics.preserved_elements.get('citations', [])
        missing_citations = [c for c in input_citations if c not in output_text]
        citations_preserved = len(missing_citations) == 0

        # Check technical terms (from entities)
        input_terms = [e.text for e in input_semantics.entities
                       if e.label in ('TECHNICAL', 'PERSON', 'ORG', 'GPE')]
        missing_terms = [t for t in input_terms if t.lower() not in output_text_lower]
        terms_preserved = len(missing_terms) <= len(input_terms) * 0.1  # Allow 10% loss

        # Calculate meaning drift (based on key concept overlap)
        input_concepts = set()
        for claim in input_claims:
            words = claim.subject.lower().split() + [claim.predicate.lower()]
            words += [w.lower() for obj in claim.objects for w in obj.split()]
            input_concepts.update(words)

        output_concepts = set()
        for claim in output_semantics.claims:
            words = claim.subject.lower().split() + [claim.predicate.lower()]
            words += [w.lower() for obj in claim.objects for w in obj.split()]
            output_concepts.update(words)

        # Remove common stopwords
        stopwords = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been',
                    'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would',
                    'could', 'should', 'may', 'might', 'must', 'this', 'that',
                    'it', 'they', 'we', 'you', 'i', 'he', 'she', 'of', 'in',
                    'to', 'for', 'with', 'on', 'at', 'by', 'from', 'as'}
        input_concepts = input_concepts - stopwords
        output_concepts = output_concepts - stopwords

        if len(input_concepts) > 0:
            overlap = len(input_concepts & output_concepts)
            meaning_drift = 1 - (overlap / len(input_concepts))
        else:
            meaning_drift = 0

        # Determine if passed
        passed = (
            claim_coverage >= self.min_claim_coverage and
            citations_preserved and
            meaning_drift <= self.max_meaning_drift
        )

        return SemanticVerification(
            passed=passed,
            claim_coverage=claim_coverage,
            missing_claims=missing_claims[:5],  # Limit to first 5
            preserved_citations=citations_preserved,
            missing_citations=missing_citations,
            preserved_technical_terms=terms_preserved,
            missing_terms=missing_terms[:5],
            meaning_drift_score=meaning_drift
        )

    def _claim_appears_in_output(self, claim: Claim, output_lower: str) -> bool:
        """Check if a claim's key content appears in output."""
        # Extract key content words from claim
        key_words = []

        # Subject words (excluding articles and common words)
        subject_words = [w.lower() for w in claim.subject.split()
                        if len(w) > 3 and w.lower() not in ('the', 'this', 'that', 'these', 'those')]
        key_words.extend(subject_words[:3])  # Top 3 subject words

        # Objects
        for obj in claim.objects:
            obj_words = [w.lower() for w in obj.split()
                        if len(w) > 3 and w.lower() not in ('the', 'this', 'that')]
            key_words.extend(obj_words[:2])

        if not key_words:
            return True  # No key words to check

        # Check if majority of key words appear in output
        found = sum(1 for w in key_words if w in output_lower)
        return found >= len(key_words) * 0.6  # 60% of key words found

    def _verify_style(self,
                      output_text: str,
                      target_style: Optional[StyleProfile]) -> StyleVerification:
        """Verify output matches target style."""
        issues = []

        # Analyze output style
        output_style = self.style_analyzer.analyze(output_text)

        # If no target style provided, just check for reasonable patterns
        if target_style is None:
            # Load and analyze sample text
            from pathlib import Path
            sample_path = Path(__file__).parent / "prompts" / "sample.txt"
            if sample_path.exists():
                with open(sample_path, 'r', encoding='utf-8') as f:
                    target_style = self.style_analyzer.analyze(f.read())
            else:
                # Return passing result with neutral scores
                return StyleVerification(
                    passed=True,
                    sentence_length_match=0.7,
                    opener_diversity=0.7,
                    vocabulary_match=0.7,
                    formality_match=0.7,
                    pattern_coverage=0.5,
                    discourse_marker_usage=0.5,
                    issues=[]
                )

        # Compare sentence length distributions
        target_dist = target_style.sentences.length_distribution
        output_dist = output_style.sentences.length_distribution

        length_diff = sum(
            abs(target_dist.get(k, 0) - output_dist.get(k, 0))
            for k in ('short', 'medium', 'long')
        ) / 3
        sentence_length_match = 1 - length_diff

        if sentence_length_match < 0.5:
            issues.append(f"Sentence length distribution differs significantly from target")

        # Compare opener diversity
        output_openers = output_style.sentences.opener_distribution
        opener_diversity = len([k for k, v in output_openers.items() if v > 0.05])
        opener_diversity = min(1.0, opener_diversity / 5)  # Normalize: 5+ types = 1.0

        if opener_diversity < 0.4:
            issues.append("Low variety in sentence openers")

        # Compare formality
        target_formality = target_style.vocabulary.formality_score
        output_formality = output_style.vocabulary.formality_score
        formality_match = 1 - abs(target_formality - output_formality)

        if formality_match < 0.6:
            issues.append(f"Formality level differs from target (output: {output_formality:.2f}, target: {target_formality:.2f})")

        # Check for AI-typical words from style profile
        output_lower = output_text.lower()
        ai_words_found = [w for w in target_style.vocabulary.forbidden_words
                         if w in output_lower]
        if ai_words_found:
            issues.append(f"Contains AI-typical words: {', '.join(ai_words_found[:3])}")

        # CRITICAL: Check for AI fingerprints from expanded detection
        ai_fingerprints = self._detect_ai_patterns(output_text)
        if ai_fingerprints:
            issues.append(f"🚨 AI FINGERPRINTS DETECTED ({len(ai_fingerprints)}): {', '.join(ai_fingerprints[:5])}")
            # Each AI fingerprint is a serious issue
            for fp in ai_fingerprints:
                issues.append(f"  - REMOVE: '{fp}'")

        # Calculate vocabulary match (rough approximation)
        # Check if output uses similar transition words
        target_transitions = set(w for w, _ in target_style.vocabulary.transition_words[:10])
        output_transitions = set(w for w, _ in output_style.vocabulary.transition_words[:10])

        if target_transitions:
            vocab_match = len(target_transitions & output_transitions) / len(target_transitions)
        else:
            vocab_match = 0.7

        # NEW: Calculate pattern coverage
        pattern_coverage = self._calculate_pattern_coverage(output_text, target_style)

        if pattern_coverage < 0.2:
            issues.append("Low usage of distinctive style patterns from target")

        # NEW: Calculate discourse marker usage match
        discourse_marker_usage = self._calculate_discourse_marker_match(output_text, target_style)

        if discourse_marker_usage < 0.3:
            issues.append("Discourse marker usage differs from target style")

        # Overall style match (now includes pattern metrics)
        overall_match = (
            sentence_length_match +
            opener_diversity +
            formality_match +
            vocab_match +
            pattern_coverage +
            discourse_marker_usage
        ) / 6

        # STRICT: Fail if ANY AI fingerprints are detected
        has_ai_fingerprints = len(ai_fingerprints) > 0
        passed = (
            overall_match >= self.min_style_match and
            len(ai_words_found) == 0 and
            not has_ai_fingerprints  # NO AI patterns allowed
        )

        return StyleVerification(
            passed=passed,
            sentence_length_match=sentence_length_match,
            opener_diversity=opener_diversity,
            vocabulary_match=vocab_match,
            formality_match=formality_match,
            pattern_coverage=pattern_coverage,
            discourse_marker_usage=discourse_marker_usage,
            issues=issues
        )

    def _calculate_pattern_coverage(self, output_text: str, target_style: StyleProfile) -> float:
        """
        Calculate how many distinctive patterns from target style appear in output.

        Returns a score from 0 to 1 indicating pattern usage.
        """
        if not target_style.distinctive_patterns:
            return 0.5  # Neutral score if no patterns extracted

        output_lower = output_text.lower()
        patterns_found = 0
        total_patterns = 0

        # Check phrasal patterns
        if target_style.distinctive_patterns.phrasal_patterns:
            top_phrases = target_style.distinctive_patterns.phrasal_patterns[:15]
            total_patterns += len(top_phrases)

            for pattern in top_phrases:
                if pattern.phrase in output_lower:
                    patterns_found += 1

        # Check for syntactic construction indicators
        # We can't check exact templates, but we can check for key indicators
        if target_style.distinctive_patterns.syntactic_constructions:
            construction_indicators = {
                'contrastive': ['contrary to', 'not', 'but', 'however'],
                'definition': ['is called', 'is the', 'means that'],
                'enumeration': ['first', 'second', 'finally'],
                'causal': ['therefore', 'hence', 'thus', 'consequently'],
                'explanatory': ['in other words', 'that is to say', 'means that'],
                'quotation': ['says', 'wrote', 'declared']
            }

            for construction in target_style.distinctive_patterns.syntactic_constructions[:8]:
                total_patterns += 1
                indicators = construction_indicators.get(construction.construction_type, [])

                for indicator in indicators:
                    if indicator in output_lower:
                        patterns_found += 1
                        break  # Count each construction type once

        if total_patterns == 0:
            return 0.5

        return min(1.0, patterns_found / total_patterns)

    def _calculate_discourse_marker_match(self, output_text: str, target_style: StyleProfile) -> float:
        """
        Calculate how well discourse marker usage matches the target style.

        Returns a score from 0 to 1.
        """
        if not target_style.distinctive_patterns or not target_style.distinctive_patterns.discourse_markers:
            return 0.5  # Neutral score if no markers extracted

        output_lower = output_text.lower()

        # Count target markers found in output
        target_markers = target_style.distinctive_patterns.discourse_markers[:15]
        markers_found = 0

        for marker_usage in target_markers:
            marker = marker_usage.marker.lower()
            # Use word boundary pattern to avoid partial matches
            pattern = r'\b' + re.escape(marker) + r'\b'
            if re.search(pattern, output_lower):
                markers_found += 1

        if len(target_markers) == 0:
            return 0.5

        # Score based on proportion of markers used
        base_score = markers_found / len(target_markers)

        # Bonus for using high-frequency markers from target
        high_freq_markers = [m for m in target_markers if m.frequency_per_100_sentences > 1.0]
        high_freq_found = 0

        for marker_usage in high_freq_markers:
            marker = marker_usage.marker.lower()
            pattern = r'\b' + re.escape(marker) + r'\b'
            if re.search(pattern, output_lower):
                high_freq_found += 1

        if high_freq_markers:
            high_freq_score = high_freq_found / len(high_freq_markers)
            # Weight: 70% base score, 30% high-frequency bonus
            return 0.7 * base_score + 0.3 * high_freq_score

        return base_score

    def _verify_preservation(self,
                             input_semantics: SemanticContent,
                             output_text: str) -> PreservationVerification:
        """Verify that must-preserve elements are intact."""
        preserved = input_semantics.preserved_elements

        # Check citations
        input_citations = preserved.get('citations', [])
        missing_citations = [c for c in input_citations if c not in output_text]
        citations_ok = len(missing_citations) == 0

        # Check numbers
        input_numbers = preserved.get('numbers', [])
        missing_numbers = []
        for num in input_numbers:
            # Numbers might be reformatted, so check loosely
            core_num = re.search(r'\d+', num)
            if core_num and core_num.group() not in output_text:
                missing_numbers.append(num)
        numbers_ok = len(missing_numbers) <= len(input_numbers) * 0.1  # Allow 10%

        # Check quoted text
        input_quotes = preserved.get('quoted_text', [])
        missing_quotes = []
        for quote in input_quotes:
            # Quotes should be preserved verbatim
            if quote not in output_text and quote.lower() not in output_text.lower():
                missing_quotes.append(quote)
        quotes_ok = len(missing_quotes) == 0

        passed = citations_ok and numbers_ok and quotes_ok

        return PreservationVerification(
            passed=passed,
            citations_preserved=citations_ok,
            missing_citations=missing_citations,
            numbers_preserved=numbers_ok,
            missing_numbers=missing_numbers,
            quotes_preserved=quotes_ok,
            missing_quotes=missing_quotes
        )

    def _generate_recommendations(self,
                                   semantic: SemanticVerification,
                                   style: StyleVerification,
                                   preservation: PreservationVerification) -> List[str]:
        """Generate actionable recommendations for improvement."""
        recommendations = []

        # Semantic recommendations
        if not semantic.passed:
            if semantic.claim_coverage < self.min_claim_coverage:
                recommendations.append(
                    f"Increase claim coverage ({semantic.claim_coverage:.0%} < {self.min_claim_coverage:.0%}). "
                    f"Missing claims about: {', '.join(c[:30] for c in semantic.missing_claims[:2])}"
                )

            if semantic.meaning_drift_score > self.max_meaning_drift:
                recommendations.append(
                    f"Reduce meaning drift ({semantic.meaning_drift_score:.0%}). "
                    "Output may have diverged from original meaning."
                )

            if not semantic.preserved_citations:
                recommendations.append(
                    f"Restore missing citations: {', '.join(semantic.missing_citations)}"
                )

        # Style recommendations
        if not style.passed:
            for issue in style.issues[:3]:
                recommendations.append(f"Style issue: {issue}")

        # Pattern-specific recommendations (NEW)
        if style.pattern_coverage < 0.3:
            recommendations.append(
                f"Pattern coverage is low ({style.pattern_coverage:.0%}). "
                "Use more distinctive phrases and syntactic constructions from the target style."
            )

        if style.discourse_marker_usage < 0.3:
            recommendations.append(
                f"Discourse marker usage is low ({style.discourse_marker_usage:.0%}). "
                "Incorporate more connectors like 'therefore', 'however', 'moreover' as used in the target."
            )

        # Preservation recommendations
        if not preservation.passed:
            if not preservation.citations_preserved:
                recommendations.append(
                    f"Critical: Restore citations {', '.join(preservation.missing_citations)}"
                )

            if not preservation.numbers_preserved and preservation.missing_numbers:
                recommendations.append(
                    f"Restore numeric data: {', '.join(preservation.missing_numbers[:3])}"
                )

            if not preservation.quotes_preserved and preservation.missing_quotes:
                recommendations.append(
                    f"Restore quoted text: {preservation.missing_quotes[0][:50]}..."
                )

        return recommendations


def verify_synthesis(input_text: str, output_text: str) -> VerificationResult:
    """Convenience function to verify synthesis output."""
    verifier = Verifier()
    return verifier.verify(input_text, output_text)


# Test function
if __name__ == '__main__':
    input_text = """Human experience reinforces the rule of finitude. The biological cycle of birth, life, and decay defines our reality. Every object we touch eventually breaks. Every star burning in the night sky eventually succumbs to erosion. But we encounter a logical trap when we apply that same finiteness to the universe itself. A cosmos with a definitive beginning and a hard boundary implies a container. Logic demands we ask a difficult question. If the universe has edges, what exists outside them?

Tom Stonier proposed that information is interconvertible with energy and conserved alongside energy[^155]. One theory envisions the overarching structure as a fractal[^25]."""

    # Simulate a synthesis output (for testing)
    output_text = """The human condition teaches us about endings. Birth gives way to life, and life surrenders to decay. Objects break under our hands. Stars in the night sky fade into nothing through erosion. Yet a problem emerges when we apply finite thinking to the cosmos. A universe with clear beginnings and hard edges requires something to contain it. We must ask: what lies beyond those edges?

According to Tom Stonier, information can convert to energy and follows conservation laws[^155]. Some theorists describe the greater structure as fractal in nature[^25]."""

    print("=== Verification Test ===\n")
    print("Input text:")
    print(input_text[:200] + "...\n")
    print("Output text:")
    print(output_text[:200] + "...\n")

    verifier = Verifier()
    result = verifier.verify(input_text, output_text)

    print("=== Results ===\n")
    print(f"Overall passed: {result.overall_passed}")
    print(f"\nSemantic verification:")
    print(f"  - Passed: {result.semantic.passed}")
    print(f"  - Claim coverage: {result.semantic.claim_coverage:.0%}")
    print(f"  - Meaning drift: {result.semantic.meaning_drift_score:.0%}")
    print(f"  - Citations preserved: {result.semantic.preserved_citations}")

    print(f"\nStyle verification:")
    print(f"  - Passed: {result.style.passed}")
    print(f"  - Sentence length match: {result.style.sentence_length_match:.0%}")
    print(f"  - Opener diversity: {result.style.opener_diversity:.0%}")
    print(f"  - Formality match: {result.style.formality_match:.0%}")
    print(f"  - Pattern coverage: {result.style.pattern_coverage:.0%}")
    print(f"  - Discourse marker usage: {result.style.discourse_marker_usage:.0%}")

    print(f"\nPreservation verification:")
    print(f"  - Passed: {result.preservation.passed}")
    print(f"  - Citations: {result.preservation.citations_preserved}")
    print(f"  - Numbers: {result.preservation.numbers_preserved}")

    if result.recommendations:
        print(f"\nRecommendations:")
        for rec in result.recommendations:
            print(f"  - {rec}")

