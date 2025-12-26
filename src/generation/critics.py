"""Validation critics for generated content."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from enum import Enum

from ..llm.provider import LLMProvider
from ..models.plan import SentenceNode, SentencePlan
from ..models.style import StyleProfile
from ..ingestion.context_analyzer import GlobalContext
from ..utils.nlp import NLPManager
from ..utils.logging import get_logger
from .style_metrics import StyleScorer

logger = get_logger(__name__)


class CriticType(Enum):
    """Types of critics."""
    SEMANTIC = "semantic"
    STYLE = "style"
    LENGTH = "length"
    FLUENCY = "fluency"
    KEYWORD = "keyword"
    VOICE = "voice"
    PUNCTUATION = "punctuation"


@dataclass
class CriticFeedback:
    """Feedback from a critic."""
    critic_type: CriticType
    score: float  # 0.0 to 1.0
    passed: bool
    issues: List[str] = field(default_factory=list)
    suggestions: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "critic_type": self.critic_type.value,
            "score": self.score,
            "passed": self.passed,
            "issues": self.issues,
            "suggestions": self.suggestions,
        }


@dataclass
class ValidationResult:
    """Combined validation result from all critics."""
    feedbacks: List[CriticFeedback] = field(default_factory=list)

    @property
    def all_passed(self) -> bool:
        """Check if all critics passed."""
        return all(f.passed for f in self.feedbacks)

    @property
    def overall_score(self) -> float:
        """Calculate weighted overall score."""
        if not self.feedbacks:
            return 0.0
        return sum(f.score for f in self.feedbacks) / len(self.feedbacks)

    @property
    def failing_critics(self) -> List[CriticFeedback]:
        """Get list of critics that failed."""
        return [f for f in self.feedbacks if not f.passed]

    def get_consolidated_feedback(self) -> str:
        """Get consolidated feedback for revision."""
        issues = []
        for feedback in self.failing_critics:
            for issue in feedback.issues:
                issues.append(f"[{feedback.critic_type.value}] {issue}")
        return " | ".join(issues) if issues else "All checks passed."


class Critic(ABC):
    """Base class for validation critics."""

    def __init__(self, threshold: float = 0.7):
        """Initialize critic.

        Args:
            threshold: Score threshold for passing (0.0-1.0).
        """
        self.threshold = threshold

    @property
    @abstractmethod
    def critic_type(self) -> CriticType:
        """Get the type of this critic."""
        pass

    @abstractmethod
    def evaluate(
        self,
        generated_text: str,
        node: SentenceNode,
        context: Optional[Dict] = None
    ) -> CriticFeedback:
        """Evaluate generated text.

        Args:
            generated_text: The text to evaluate.
            node: Original sentence specification.
            context: Additional context for evaluation.

        Returns:
            CriticFeedback with score and issues.
        """
        pass


class LengthCritic(Critic):
    """Critic that evaluates sentence length accuracy."""

    def __init__(self, threshold: float = 0.7, tolerance: int = 5):
        """Initialize length critic.

        Args:
            threshold: Pass/fail threshold.
            tolerance: Acceptable word count deviation.
        """
        super().__init__(threshold)
        self.tolerance = tolerance

    @property
    def critic_type(self) -> CriticType:
        return CriticType.LENGTH

    def evaluate(
        self,
        generated_text: str,
        node: SentenceNode,
        context: Optional[Dict] = None
    ) -> CriticFeedback:
        """Evaluate sentence length.

        Args:
            generated_text: Generated sentence.
            node: Sentence specification.
            context: Not used.

        Returns:
            CriticFeedback with length analysis.
        """
        word_count = len(generated_text.split())
        target = node.target_length
        diff = abs(word_count - target)

        # Calculate score (1.0 when exact, decreases with difference)
        if target == 0:
            score = 1.0
        else:
            score = max(0.0, 1.0 - (diff / target))

        passed = diff <= self.tolerance
        issues = []
        suggestions = []

        if not passed:
            if word_count > target:
                issues.append(f"Sentence too long: {word_count} words vs target {target}")
                suggestions.append("Shorten the sentence by removing non-essential words")
            else:
                issues.append(f"Sentence too short: {word_count} words vs target {target}")
                suggestions.append("Expand the sentence with more detail")

        return CriticFeedback(
            critic_type=self.critic_type,
            score=score,
            passed=passed,
            issues=issues,
            suggestions=suggestions
        )


class KeywordCritic(Critic):
    """Critic that checks required keywords are present."""

    @property
    def critic_type(self) -> CriticType:
        return CriticType.KEYWORD

    def evaluate(
        self,
        generated_text: str,
        node: SentenceNode,
        context: Optional[Dict] = None
    ) -> CriticFeedback:
        """Evaluate keyword inclusion.

        Args:
            generated_text: Generated sentence.
            node: Sentence specification with keywords.
            context: Not used.

        Returns:
            CriticFeedback with keyword analysis.
        """
        if not node.keywords:
            return CriticFeedback(
                critic_type=self.critic_type,
                score=1.0,
                passed=True,
                issues=[],
                suggestions=[]
            )

        text_lower = generated_text.lower()
        missing = []
        found = 0

        for keyword in node.keywords:
            if keyword.lower() in text_lower:
                found += 1
            else:
                missing.append(keyword)

        score = found / len(node.keywords)
        passed = score >= self.threshold

        issues = []
        suggestions = []
        if missing:
            issues.append(f"Missing keywords: {', '.join(missing)}")
            suggestions.append(f"Include: {', '.join(missing)}")

        return CriticFeedback(
            critic_type=self.critic_type,
            score=score,
            passed=passed,
            issues=issues,
            suggestions=suggestions
        )


class FluencyCritic(Critic):
    """Critic that evaluates sentence fluency and grammar."""

    def __init__(self, threshold: float = 0.7, nlp_manager: Optional[NLPManager] = None):
        """Initialize fluency critic.

        Args:
            threshold: Pass/fail threshold.
            nlp_manager: NLP manager for analysis.
        """
        super().__init__(threshold)
        self.nlp = nlp_manager or NLPManager()

    @property
    def critic_type(self) -> CriticType:
        return CriticType.FLUENCY

    def evaluate(
        self,
        generated_text: str,
        node: SentenceNode,
        context: Optional[Dict] = None
    ) -> CriticFeedback:
        """Evaluate sentence fluency.

        Args:
            generated_text: Generated sentence.
            node: Sentence specification.
            context: Not used.

        Returns:
            CriticFeedback with fluency analysis.
        """
        issues = []
        penalties = 0.0

        # Check sentence has proper structure
        doc = self.nlp.process(generated_text)
        sentences = list(doc.sents)

        if len(sentences) > 1:
            issues.append("Generated text contains multiple sentences")
            penalties += 0.2

        # Check for basic grammatical issues
        if sentences:
            sent = sentences[0]

            # Check for subject
            has_subject = any(
                token.dep_ in ('nsubj', 'nsubjpass', 'expl')
                for token in sent
            )
            if not has_subject:
                issues.append("Sentence may lack a clear subject")
                penalties += 0.1

            # Check for verb
            has_verb = any(
                token.pos_ in ('VERB', 'AUX')
                for token in sent
            )
            if not has_verb:
                issues.append("Sentence may lack a verb")
                penalties += 0.15

        # Check ending punctuation
        if generated_text and generated_text[-1] not in '.!?':
            issues.append("Sentence lacks proper ending punctuation")
            penalties += 0.1

        # Check for repetition
        words = generated_text.lower().split()
        if len(words) > 3:
            word_counts = {}
            for w in words:
                if len(w) > 3:  # Only check significant words
                    word_counts[w] = word_counts.get(w, 0) + 1

            repeated = [w for w, c in word_counts.items() if c > 2]
            if repeated:
                issues.append(f"Excessive repetition: {', '.join(repeated)}")
                penalties += 0.1

        score = max(0.0, 1.0 - penalties)
        passed = score >= self.threshold

        suggestions = []
        if issues:
            suggestions.append("Revise for better grammatical structure")

        return CriticFeedback(
            critic_type=self.critic_type,
            score=score,
            passed=passed,
            issues=issues,
            suggestions=suggestions
        )


class SemanticCritic(Critic):
    """Critic that evaluates semantic preservation."""

    def __init__(
        self,
        threshold: float = 0.7,
        llm_provider: Optional[LLMProvider] = None,
        nlp_manager: Optional[NLPManager] = None
    ):
        """Initialize semantic critic.

        Args:
            threshold: Pass/fail threshold.
            llm_provider: Optional LLM for advanced checking.
            nlp_manager: NLP manager for similarity.
        """
        super().__init__(threshold)
        self.llm_provider = llm_provider
        self.nlp = nlp_manager or NLPManager()

    @property
    def critic_type(self) -> CriticType:
        return CriticType.SEMANTIC

    def evaluate(
        self,
        generated_text: str,
        node: SentenceNode,
        context: Optional[Dict] = None
    ) -> CriticFeedback:
        """Evaluate semantic preservation.

        Args:
            generated_text: Generated sentence.
            node: Sentence specification with propositions.
            context: Optional additional context.

        Returns:
            CriticFeedback with semantic analysis.
        """
        if not node.propositions:
            return CriticFeedback(
                critic_type=self.critic_type,
                score=1.0,
                passed=True,
                issues=[],
                suggestions=[]
            )

        # Get source proposition text
        source_text = node.get_proposition_text()

        # Calculate semantic similarity
        score = self._calculate_similarity(source_text, generated_text)

        passed = score >= self.threshold
        issues = []
        suggestions = []

        if not passed:
            issues.append("Generated text may not fully preserve source meaning")
            suggestions.append("Ensure all key concepts from source are included")

            # Check for missing concepts
            missing = self._find_missing_concepts(source_text, generated_text)
            if missing:
                issues.append(f"Potentially missing concepts: {', '.join(missing)}")

        return CriticFeedback(
            critic_type=self.critic_type,
            score=score,
            passed=passed,
            issues=issues,
            suggestions=suggestions
        )

    def _calculate_similarity(self, source: str, generated: str) -> float:
        """Calculate semantic similarity between texts.

        Args:
            source: Source proposition text.
            generated: Generated text.

        Returns:
            Similarity score 0.0 to 1.0.
        """
        source_doc = self.nlp.process(source)
        generated_doc = self.nlp.process(generated)

        # Use spaCy's similarity if vectors available
        if source_doc.has_vector and generated_doc.has_vector:
            return source_doc.similarity(generated_doc)

        # Fallback to word overlap
        source_words = set(
            token.lemma_.lower() for token in source_doc
            if not token.is_stop and not token.is_punct
        )
        generated_words = set(
            token.lemma_.lower() for token in generated_doc
            if not token.is_stop and not token.is_punct
        )

        if not source_words:
            return 1.0

        overlap = len(source_words & generated_words)
        return overlap / len(source_words)

    def _find_missing_concepts(self, source: str, generated: str) -> List[str]:
        """Find concepts in source not present in generated.

        Args:
            source: Source text.
            generated: Generated text.

        Returns:
            List of potentially missing concepts.
        """
        source_doc = self.nlp.process(source)
        generated_doc = self.nlp.process(generated)

        # Extract significant terms from source
        source_terms = set()
        for token in source_doc:
            if token.pos_ in ('NOUN', 'VERB', 'ADJ') and not token.is_stop:
                source_terms.add(token.lemma_.lower())

        # Extract terms from generated
        generated_terms = set()
        for token in generated_doc:
            if not token.is_stop:
                generated_terms.add(token.lemma_.lower())

        missing = source_terms - generated_terms
        return list(missing)[:5]


class StyleCritic(Critic):
    """Critic that evaluates style adherence using comprehensive style metrics."""

    def __init__(
        self,
        style_profile: StyleProfile,
        threshold: float = 0.6,
        nlp_manager: Optional[NLPManager] = None
    ):
        """Initialize style critic.

        Args:
            style_profile: Target style profile.
            threshold: Pass/fail threshold.
            nlp_manager: NLP manager for analysis (unused, kept for API compat).
        """
        super().__init__(threshold)
        self.style_profile = style_profile
        self.style_scorer = StyleScorer(style_profile)

    @property
    def critic_type(self) -> CriticType:
        return CriticType.STYLE

    def evaluate(
        self,
        generated_text: str,
        node: SentenceNode,
        context: Optional[Dict] = None
    ) -> CriticFeedback:
        """Evaluate style adherence using StyleScorer.

        Args:
            generated_text: Generated sentence.
            node: Sentence specification.
            context: Optional additional context.

        Returns:
            CriticFeedback with style analysis.
        """
        # Use the comprehensive StyleScorer
        style_score = self.style_scorer.score(generated_text)

        issues = []

        # Report specific issues
        if style_score.vocabulary_overlap < 0.3:
            issues.append("Low vocabulary alignment with target style")
        if style_score.voice_match < 0.5:
            target_ratio = self.style_profile.primary_author.voice_ratio
            voice_type = "active" if target_ratio > 0.5 else "passive"
            issues.append(f"Voice doesn't match target (should be more {voice_type})")
        if style_score.sentence_length_match < 0.5:
            issues.append("Sentence length doesn't match target style")
        if style_score.punctuation_match < 0.4:
            issues.append("Punctuation patterns don't match target style")

        passed = style_score.overall >= self.threshold

        suggestions = []
        if not passed:
            # Get specific feedback from StyleScorer
            feedback = self.style_scorer.get_feedback(generated_text)
            if feedback and feedback != "Style looks good":
                suggestions.append(feedback)

        return CriticFeedback(
            critic_type=self.critic_type,
            score=style_score.overall,
            passed=passed,
            issues=issues,
            suggestions=suggestions
        )


class VoiceCritic(Critic):
    """Critic that specifically evaluates active/passive voice matching."""

    def __init__(
        self,
        style_profile: StyleProfile,
        threshold: float = 0.6
    ):
        """Initialize voice critic.

        Args:
            style_profile: Target style profile.
            threshold: Pass/fail threshold.
        """
        super().__init__(threshold)
        self.style_profile = style_profile
        self.style_scorer = StyleScorer(style_profile)

    @property
    def critic_type(self) -> CriticType:
        return CriticType.VOICE

    def evaluate(
        self,
        generated_text: str,
        node: SentenceNode,
        context: Optional[Dict] = None
    ) -> CriticFeedback:
        """Evaluate voice matching.

        Args:
            generated_text: Generated sentence.
            node: Sentence specification.
            context: Optional additional context.

        Returns:
            CriticFeedback with voice analysis.
        """
        voice_scorer = self.style_scorer.voice_scorer
        score = voice_scorer.score(generated_text)
        analysis = voice_scorer.analyze_voice(generated_text)

        target_ratio = voice_scorer.target_voice_ratio
        actual_ratio = analysis["active_ratio"]

        issues = []
        suggestions = []

        passed = score >= self.threshold

        if not passed:
            if target_ratio > 0.6 and actual_ratio < 0.5:
                issues.append(f"Text is {actual_ratio*100:.0f}% active voice, target is {target_ratio*100:.0f}%")
                suggestions.append("Rewrite passive constructions to active voice")
            elif target_ratio < 0.4 and actual_ratio > 0.5:
                issues.append(f"Text is {actual_ratio*100:.0f}% active voice, target is {target_ratio*100:.0f}%")
                suggestions.append("Use more passive constructions")

        return CriticFeedback(
            critic_type=self.critic_type,
            score=score,
            passed=passed,
            issues=issues,
            suggestions=suggestions
        )


class PunctuationCritic(Critic):
    """Critic that evaluates punctuation pattern matching."""

    def __init__(
        self,
        style_profile: StyleProfile,
        threshold: float = 0.5
    ):
        """Initialize punctuation critic.

        Args:
            style_profile: Target style profile.
            threshold: Pass/fail threshold.
        """
        super().__init__(threshold)
        self.style_profile = style_profile
        self.style_scorer = StyleScorer(style_profile)

    @property
    def critic_type(self) -> CriticType:
        return CriticType.PUNCTUATION

    def evaluate(
        self,
        generated_text: str,
        node: SentenceNode,
        context: Optional[Dict] = None
    ) -> CriticFeedback:
        """Evaluate punctuation pattern matching.

        Args:
            generated_text: Generated sentence.
            node: Sentence specification.
            context: Optional additional context.

        Returns:
            CriticFeedback with punctuation analysis.
        """
        punc_scorer = self.style_scorer.punctuation_scorer
        score = punc_scorer.score(generated_text)
        analysis = punc_scorer.analyze_punctuation(generated_text)
        target_patterns = punc_scorer.target_patterns

        issues = []
        suggestions = []

        passed = score >= self.threshold

        if not passed and target_patterns:
            # Find significant mismatches
            for punc_type, target_data in target_patterns.items():
                target_freq = target_data.get("per_sentence", 0)
                actual_freq = analysis.get(punc_type, 0)

                if target_freq > 0.2 and actual_freq < 0.1:
                    issues.append(f"Missing {punc_type} (author uses {target_freq:.1f} per sentence)")
                    suggestions.append(f"Consider using {punc_type}")

        return CriticFeedback(
            critic_type=self.critic_type,
            score=score,
            passed=passed,
            issues=issues,
            suggestions=suggestions
        )


class CriticPanel:
    """Panel of critics for comprehensive validation."""

    def __init__(
        self,
        style_profile: StyleProfile,
        global_context: GlobalContext,
        llm_provider: Optional[LLMProvider] = None,
        include_voice: bool = True,
        include_punctuation: bool = True
    ):
        """Initialize critic panel.

        Args:
            style_profile: Target style profile.
            global_context: Global document context.
            llm_provider: Optional LLM for advanced critics.
            include_voice: Whether to include voice critic.
            include_punctuation: Whether to include punctuation critic.
        """
        self.nlp = NLPManager()
        self.style_profile = style_profile

        self.critics = [
            LengthCritic(threshold=0.7, tolerance=5),
            KeywordCritic(threshold=0.8),
            FluencyCritic(threshold=0.7, nlp_manager=self.nlp),
            SemanticCritic(threshold=0.7, llm_provider=llm_provider, nlp_manager=self.nlp),
            StyleCritic(style_profile, threshold=0.6),
        ]

        # Add optional critics for detailed style matching
        if include_voice:
            self.critics.append(VoiceCritic(style_profile, threshold=0.6))
        if include_punctuation:
            self.critics.append(PunctuationCritic(style_profile, threshold=0.5))

    def validate(
        self,
        generated_text: str,
        node: SentenceNode,
        context: Optional[Dict] = None
    ) -> ValidationResult:
        """Run all critics on generated text.

        Args:
            generated_text: Text to validate.
            node: Sentence specification.
            context: Additional context.

        Returns:
            ValidationResult with all feedbacks.
        """
        feedbacks = []
        for critic in self.critics:
            try:
                feedback = critic.evaluate(generated_text, node, context)
                feedbacks.append(feedback)
            except Exception as e:
                logger.warning(f"Critic {critic.critic_type.value} failed: {e}")
                # Add neutral feedback on error
                feedbacks.append(CriticFeedback(
                    critic_type=critic.critic_type,
                    score=0.5,
                    passed=True,
                    issues=[f"Evaluation error: {str(e)}"],
                    suggestions=[]
                ))

        result = ValidationResult(feedbacks=feedbacks)

        logger.debug(
            f"Validation: score={result.overall_score:.2f}, "
            f"passed={result.all_passed}, "
            f"failing={[f.critic_type.value for f in result.failing_critics]}"
        )

        return result

    def validate_paragraph(
        self,
        sentences: List[str],
        plan: SentencePlan
    ) -> List[ValidationResult]:
        """Validate all sentences in a paragraph.

        Args:
            sentences: List of generated sentences.
            plan: Sentence plan.

        Returns:
            List of ValidationResults.
        """
        results = []
        for sentence, node in zip(sentences, plan.nodes):
            result = self.validate(sentence, node)
            results.append(result)
        return results

    def get_revision_feedback(
        self,
        generated_text: str,
        node: SentenceNode
    ) -> str:
        """Get consolidated feedback for revision.

        Args:
            generated_text: Text to evaluate.
            node: Sentence specification.

        Returns:
            Feedback string for LLM revision prompt.
        """
        result = self.validate(generated_text, node)
        return result.get_consolidated_feedback()
