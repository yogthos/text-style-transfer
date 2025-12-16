"""
Stage 4: Verification Module

Validates synthesis output independently:
- Semantic check: Are all claims and facts from input present?
- Style check: Does output match sample's stylistic patterns?
- Statistical check: Does output match statistical profile of sample?
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
from style_statistics import StyleStatisticsAnalyzer
from sentence_validator import SentenceValidator
from template_generator import ParagraphTemplate


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
    issues: List[str] = field(default_factory=list)  # Issues found during verification
    phrase_repetition_detected: bool = False  # NEW: Detected recently used opener phrase
    repeated_phrase: Optional[str] = None  # NEW: The phrase that was repeated


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
        """Generate a concise, actionable summary of hints for the LLM."""
        if not self.transformation_hints:
            return ""

        summary = ["## FIX THESE ISSUES:\n"]

        # Group by priority
        priority_1 = [h for h in self.transformation_hints if h.priority == 1]
        priority_2 = [h for h in self.transformation_hints if h.priority == 2]

        if priority_1:
            summary.append("### MUST FIX:")
            for h in priority_1[:5]:
                summary.append(f"- {h.issue}")
                summary.append(f"  → {h.suggestion}")

        if priority_2:
            summary.append("\n### SHOULD FIX:")
            for h in priority_2[:3]:
                if h.current_text:
                    summary.append(f"- \"{h.current_text}\"")
                    summary.append(f"  → {h.suggestion}")
                else:
                    summary.append(f"- {h.issue}: {h.suggestion}")

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
        self.style_stats = StyleStatisticsAnalyzer()

        # Cache for structural patterns
        self._role_patterns = None
        self._sample_hash = None
        self._stats_initialized = False

        # Load config
        if config_path is None:
            config_path = Path(__file__).parent / "config.json"
        else:
            config_path = Path(config_path)

        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                self.config = json.load(f)
        except (IOError, json.JSONDecodeError):
            self.config = {}

        # Load thresholds from config or use defaults
        self._load_thresholds(config_path)

        # Initialize style statistics from sample
        self._init_style_statistics()

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

    def _init_style_statistics(self):
        """Initialize style statistics from sample text."""
        # Get sample path from config, with fallback to default
        sample_file = self.config.get("sample", {}).get("file", "prompts/sample.txt")
        sample_path = Path(__file__).parent / sample_file
        if sample_path.exists():
            try:
                with open(sample_path, 'r', encoding='utf-8') as f:
                    sample_text = f.read()
                self.style_stats.set_sample_profile(sample_text)
                self._stats_initialized = True
            except Exception as e:
                print(f"  [Verifier] Warning: Could not initialize style statistics: {e}")
                self._stats_initialized = False

    def _load_sample_patterns(self) -> Dict[str, List[StructuralPattern]]:
        """Load structural patterns from sample text."""
        # Get sample path from config, with fallback to default
        sample_file = self.config.get("sample", {}).get("file", "prompts/sample.txt")
        sample_path = Path(__file__).parent / sample_file
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
               previous_score: float = 0.0,
               accumulated_context: Optional[str] = None,
               position_in_document: Optional[Tuple[int, int]] = None,
               used_phrases: Optional[List[str]] = None,
               paragraph_template: Optional[ParagraphTemplate] = None,
               sentence_validator: Optional[SentenceValidator] = None) -> VerificationResult:
        """
        Verify synthesis output against input and target style.

        Args:
            input_text: Original input text
            output_text: Synthesized output text
            input_semantics: Pre-extracted input semantics (optional)
            target_style: Target style profile (optional)
            iteration: Current iteration number (for tracking improvement)
            previous_score: Score from previous iteration
            accumulated_context: Full transformed text so far (for statistical analysis)
            position_in_document: Tuple of (paragraph_index, total_paragraphs) for position-aware checks
            used_phrases: List of opener phrases recently used (to detect repetition)

        Returns:
            VerificationResult with pass/fail, details, and transformation hints
        """
        # Extract semantics if not provided
        if input_semantics is None:
            input_semantics = self.semantic_extractor.extract(input_text)

        # Run verification passes
        semantic_result = self._verify_semantics(input_text, output_text, input_semantics)

        # Use accumulated context for statistical style checks if available
        style_result = self._verify_style(
            output_text, target_style,
            accumulated_context=accumulated_context,
            position_in_document=position_in_document,
            used_phrases=used_phrases
        )
        preservation_result = self._verify_preservation(input_semantics, output_text)

        # Determine overall pass/fail (phrase repetition is a critical failure)
        overall_passed = (
            semantic_result.passed and
            style_result.passed and
            preservation_result.passed and
            not style_result.phrase_repetition_detected  # CRITICAL: Fail if phrase repetition detected
        )

        # Generate recommendations
        recommendations = self._generate_recommendations(
            semantic_result, style_result, preservation_result
        )

        # Generate transformation hints based on structural analysis
        # Pass accumulated context for document-aware hints
        transformation_hints = self._generate_transformation_hints(
            input_text, output_text, style_result, target_style,
            accumulated_context=accumulated_context,
            position_in_document=position_in_document
        )

        # Add sentence template validation hints (CRITICAL - must be checked)
        sentence_template_hints = self._validate_sentence_templates(
            output_text, paragraph_template, sentence_validator
        )
        if sentence_template_hints:
            transformation_hints.extend(sentence_template_hints)

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
                                        target_style: Optional[StyleProfile],
                                        accumulated_context: Optional[str] = None,
                                        position_in_document: Optional[Tuple[int, int]] = None) -> List[TransformationHint]:
        """
        Generate FOCUSED transformation hints for iterative refinement.

        Key principle: Fewer, more specific hints work better than many generic ones.
        We prioritize:
        1. AI fingerprints (must be removed)
        2. Overused words (statistical issue - checked against full document)
        3. Position-appropriate style (opening vs closing paragraphs)
        4. Specific sentence rewrites (not generic pattern suggestions)
        """
        hints = []

        # Use accumulated context for document-wide statistics
        text_for_statistics = accumulated_context if accumulated_context else output_text

        # CRITICAL: Phrase repetition check - must be fixed immediately
        if style_result.phrase_repetition_detected and style_result.repeated_phrase:
            hints.append(TransformationHint(
                sentence_index=0,  # First sentence
                current_text=output_text[:100] if len(output_text) > 100 else output_text,
                structural_role="paragraph_opener",
                expected_patterns=[],
                issue=f"🚨 CRITICAL: Output starts with recently used phrase '{style_result.repeated_phrase}'",
                suggestion=f"REWRITE the first sentence to use a COMPLETELY DIFFERENT opener phrase. Do NOT use '{style_result.repeated_phrase}' or any variation of it. Use a different opener type entirely.",
                priority=1  # Highest priority - must fix immediately
            ))

        # CRITICAL: AI fingerprints first - these must be addressed
        ai_patterns = self._detect_ai_patterns(output_text)
        for pattern in ai_patterns[:5]:  # Limit to top 5 most important
            hints.append(TransformationHint(
                sentence_index=-1,
                current_text=pattern,
                structural_role="document",
                expected_patterns=[],
                issue=f"🚨 REMOVE: '{pattern}'",
                suggestion=f"Delete or replace '{pattern}' with direct phrasing",
                priority=1
            ))

        # STATISTICAL: Overused words checked against FULL DOCUMENT
        if self._stats_initialized:
            _, stat_metrics = self.style_stats.score_text(text_for_statistics)
            if 'overused_markers' in stat_metrics:
                for marker, ratio in stat_metrics['overused_markers'][:3]:
                    # Check if this paragraph specifically uses the overused word
                    if marker.lower() in output_text.lower():
                        hints.append(TransformationHint(
                            sentence_index=-1,
                            current_text=marker,
                            structural_role="document",
                            expected_patterns=[],
                            issue=f"🔄 '{marker}' overused in document ({ratio:.1f}x) - AVOID in this paragraph",
                            suggestion=f"Do NOT use '{marker}' here. Use 'thus', 'as a result', 'accordingly', or omit entirely.",
                            priority=1
                        ))

            # Top 3 worst sentences in current output
            rejections = self.style_stats.get_rejection_sentences(output_text, threshold=0.4)
            for sent, score, sent_issues in rejections[:3]:
                short_sent = sent[:60] + "..." if len(sent) > 60 else sent
                hints.append(TransformationHint(
                    sentence_index=-1,
                    current_text=short_sent,
                    structural_role="sentence",
                    expected_patterns=[],
                    issue=f"📊 Rewrite needed ({score:.0%}): {sent_issues[0] if sent_issues else 'style mismatch'}",
                    suggestion="REWRITE: make longer with subordinate clauses, or shorter if too complex",
                    priority=2
                ))

        # POSITION-AWARE hints
        if position_in_document:
            para_idx, total_paras = position_in_document
            position_ratio = para_idx / max(total_paras - 1, 1) if total_paras > 1 else 0.5

            # Opening paragraph guidance
            if position_ratio < 0.2:
                hints.append(TransformationHint(
                    sentence_index=-1,
                    current_text="",
                    structural_role="paragraph_opener",
                    expected_patterns=["declarative statement", "thesis establishment"],
                    issue=f"📍 This is paragraph {para_idx+1}/{total_paras} (opening section)",
                    suggestion="Use strong declarative opening. Avoid 'However', 'Furthermore'. State the main point directly.",
                    priority=3
                ))

            # Closing paragraph guidance
            elif position_ratio > 0.8:
                hints.append(TransformationHint(
                    sentence_index=-1,
                    current_text="",
                    structural_role="paragraph_closer",
                    expected_patterns=["conclusion", "synthesis", "final statement"],
                    issue=f"📍 This is paragraph {para_idx+1}/{total_paras} (closing section)",
                    suggestion="Use concluding tone: 'Therefore', 'Thus', 'In the final analysis'. Synthesize the argument.",
                    priority=3
                ))

        # STYLE: Only add these if we're really struggling
        if style_result.pattern_coverage < 0.3:
            hints.append(TransformationHint(
                sentence_index=-1,
                current_text="",
                structural_role="document",
                expected_patterns=[],
                issue=f"Pattern coverage only {style_result.pattern_coverage:.0%}",
                suggestion="Add 2-3 sentences starting with 'Contrary to...' or 'Hence, ...'",
                priority=2
            ))

        # LIMIT: Return max 15 hints, sorted by priority
        hints.sort(key=lambda h: h.priority)
        return hints[:15]

    def _validate_sentence_templates(self,
                                    output_text: str,
                                    paragraph_template: Optional[ParagraphTemplate],
                                    sentence_validator: Optional[SentenceValidator]) -> List[TransformationHint]:
        """
        Validate each sentence in output against its template.

        Returns list of transformation hints for sentences that don't match their templates.
        """
        hints = []

        if not paragraph_template or not sentence_validator:
            return hints

        # Split output into sentences
        doc = self.semantic_extractor.nlp(output_text)
        output_sentences = list(doc.sents)

        if not output_sentences:
            return hints

        # Match each output sentence to its template
        for i, output_sent in enumerate(output_sentences):
            if i >= len(paragraph_template.sentences):
                break  # More sentences than template (might be OK, but could be an issue)

            sentence_template = paragraph_template.sentences[i]
            sentence_text = output_sent.text.strip()

            # Validate sentence against template
            issues = sentence_validator.validate_sentence(sentence_text, sentence_template)

            for issue in issues:
                # Determine priority: mismatches are critical, others are important
                priority = 1 if "mismatch" in issue.description.lower() else 2

                hints.append(TransformationHint(
                    sentence_index=i,
                    current_text=sentence_text[:100],
                    structural_role=paragraph_template.structural_role,
                    expected_patterns=[sentence_template.to_template_string()],
                    issue=f"S{i+1}: {issue.description}",
                    suggestion=f"S{i+1}: {issue.fix_guidance}",
                    priority=priority
                ))

        return hints

    def _extract_opener_phrase(self, text: str) -> str:
        """Extract the actual opener phrase (first 2-3 words) from text.

        Returns normalized phrase (lowercase, trimmed) for consistent matching.
        """
        if not text:
            return ""

        # Use semantic_extractor's nlp for consistency
        doc = self.semantic_extractor.nlp(text)
        sentences = list(doc.sents)
        if not sentences:
            return ""

        first_sent = sentences[0]
        tokens = [t for t in first_sent if not t.is_space and not t.is_punct]

        # Get first 2-3 words (normalized to lowercase, trimmed)
        if len(tokens) >= 3:
            phrase = ' '.join([t.text.lower().strip() for t in tokens[:3]])
        elif len(tokens) >= 2:
            phrase = ' '.join([t.text.lower().strip() for t in tokens[:2]])
        elif tokens:
            phrase = tokens[0].text.lower().strip()
        else:
            return ""

        # Normalize whitespace
        phrase = ' '.join(phrase.split())
        return phrase

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
                      target_style: Optional[StyleProfile],
                      accumulated_context: Optional[str] = None,
                      position_in_document: Optional[Tuple[int, int]] = None,
                      used_phrases: Optional[List[str]] = None) -> StyleVerification:
        """
        Verify output matches target style with full document awareness.

        Args:
            output_text: The current paragraph/text being verified
            target_style: Target style profile
            accumulated_context: Full transformed document so far (for statistical checks)
            position_in_document: (current_index, total) for position-aware checks
            used_phrases: List of opener phrases recently used (to detect repetition)
        """
        issues = []

        # CRITICAL: Check for phrase repetition (recently used opener phrases)
        phrase_repetition_detected = False
        repeated_phrase = None
        if used_phrases and output_text:
            output_opener_phrase = self._extract_opener_phrase(output_text)
            if output_opener_phrase:
                # Normalize for comparison
                normalized_output = output_opener_phrase.lower().strip()
                # Check against last 5 used phrases
                for used_phrase in used_phrases[-5:]:
                    if used_phrase:
                        normalized_used = used_phrase.lower().strip()
                        # Check for exact match or prefix match
                        if (normalized_output == normalized_used or
                            normalized_output.startswith(normalized_used) or
                            normalized_used.startswith(normalized_output)):
                            phrase_repetition_detected = True
                            repeated_phrase = used_phrase
                            issues.append(f"🚨 CRITICAL: Output starts with recently used phrase '{used_phrase}' - this is REPEATED and must be changed")
                            break

        # Use accumulated context for statistical analysis if available
        # This ensures we catch overuse patterns across the whole document
        text_for_statistics = accumulated_context if accumulated_context else output_text

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
                    phrase_repetition_detected=False,
                    repeated_phrase=None,
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

        # NEW: Statistical style verification using FULL DOCUMENT CONTEXT
        statistical_score = 1.0
        if self._stats_initialized:
            # Use accumulated context for document-wide statistics
            # This catches overuse patterns across the whole document
            stat_score, stat_metrics = self.style_stats.score_text(text_for_statistics)
            statistical_score = stat_score

            # Check for overused markers (now across full document)
            if 'overused_markers' in stat_metrics and stat_metrics['overused_markers']:
                for marker, ratio in stat_metrics['overused_markers']:
                    issues.append(f"🔄 OVERUSED (document-wide): '{marker}' used {ratio:.1f}x more than sample")

            # Get sentences that should be rejected from current output only
            rejections = self.style_stats.get_rejection_sentences(output_text, threshold=0.35)
            if rejections:
                for sent, score, sent_issues in rejections[:3]:  # Top 3 problematic
                    short_sent = sent[:50] + "..." if len(sent) > 50 else sent
                    issues.append(f"📊 Low statistical match ({score:.2f}): '{short_sent}'")
                    for issue in sent_issues[:2]:
                        issues.append(f"     → {issue}")

        # Position-aware checks: opening vs middle vs closing paragraphs have different expectations
        if position_in_document:
            para_idx, total_paras = position_in_document
            position_ratio = para_idx / max(total_paras - 1, 1) if total_paras > 1 else 0.5

            # Opening paragraph (first 20%) should be more declarative
            if position_ratio < 0.2:
                # Check for weak openers that don't suit an opening paragraph
                output_lower = output_text.lower()
                weak_openers = ['however', 'furthermore', 'additionally', 'moreover']
                if any(output_lower.strip().startswith(w) for w in weak_openers):
                    issues.append(f"📍 Position: Opening paragraph starts with continuation word")

            # Closing paragraph (last 20%) should have concluding tone
            elif position_ratio > 0.8:
                output_lower = output_text.lower()
                concluding_signals = ['therefore', 'thus', 'hence', 'in conclusion', 'finally', 'in the end']
                has_conclusion = any(s in output_lower for s in concluding_signals)
                if not has_conclusion and total_paras > 3:
                    issues.append(f"📍 Position: Final paragraph lacks concluding tone")

        # Overall style match (now includes pattern metrics and statistics)
        overall_match = (
            sentence_length_match +
            opener_diversity +
            formality_match +
            vocab_match +
            pattern_coverage +
            discourse_marker_usage +
            statistical_score
        ) / 7

        # STRICT: Fail if ANY AI fingerprints are detected OR phrase repetition detected
        has_ai_fingerprints = len(ai_fingerprints) > 0
        passed = (
            overall_match >= self.min_style_match and
            len(ai_words_found) == 0 and
            not has_ai_fingerprints and  # NO AI patterns allowed
            not phrase_repetition_detected  # CRITICAL: No phrase repetition allowed
        )

        return StyleVerification(
            passed=passed,
            sentence_length_match=sentence_length_match,
            opener_diversity=opener_diversity,
            vocabulary_match=vocab_match,
            formality_match=formality_match,
            pattern_coverage=pattern_coverage,
            discourse_marker_usage=discourse_marker_usage,
            phrase_repetition_detected=phrase_repetition_detected,
            repeated_phrase=repeated_phrase,
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

