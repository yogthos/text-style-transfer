"""Sentence function classification using spaCy.

Classifies sentences by rhetorical function:
- CLAIM: Makes an assertion/statement of position
- EVIDENCE: Supports a claim with facts/examples
- QUESTION: Poses a question (rhetorical or genuine)
- CONCESSION: Acknowledges opposing view/limitation
- CONTRAST: Sets up opposition to previous statement
- RESOLUTION: Resolves a tension/answers a question
- ELABORATION: Expands on previous statement
- SETUP: Introduces context/premise for what follows
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional
from collections import Counter, defaultdict
from enum import Enum

from ..utils.nlp import get_nlp
from ..utils.logging import get_logger

logger = get_logger(__name__)


class SentenceFunction(Enum):
    """Rhetorical function of a sentence."""
    CLAIM = "claim"
    EVIDENCE = "evidence"
    QUESTION = "question"
    CONCESSION = "concession"
    CONTRAST = "contrast"
    RESOLUTION = "resolution"
    ELABORATION = "elaboration"
    SETUP = "setup"
    CONTINUATION = "continuation"  # Default/neutral


@dataclass
class ClassifiedSentence:
    """A sentence with its classified function."""
    text: str
    function: SentenceFunction
    confidence: float
    markers_found: List[str] = field(default_factory=list)


class SentenceFunctionClassifier:
    """Classify sentences by rhetorical function using spaCy.

    Uses linguistic markers and syntactic patterns to determine
    the rhetorical role each sentence plays in discourse.
    """

    def __init__(self):
        self._nlp = None

    @property
    def nlp(self):
        if self._nlp is None:
            self._nlp = get_nlp()
        return self._nlp

    def classify(self, sentence: str) -> ClassifiedSentence:
        """Classify a single sentence's rhetorical function.

        Args:
            sentence: The sentence to classify.

        Returns:
            ClassifiedSentence with function and confidence.
        """
        doc = self.nlp(sentence)
        text_lower = sentence.lower().strip()

        # Check each function type in priority order
        # Priority matters for sentences with multiple markers

        # 1. QUESTION - highest priority, clear syntactic marker
        if self._is_question(sentence, doc):
            return ClassifiedSentence(
                text=sentence,
                function=SentenceFunction.QUESTION,
                confidence=0.95,
                markers_found=["?"]
            )

        # 2. CONTRAST - opposition markers at sentence start
        contrast_markers = self._find_contrast_markers(text_lower, doc)
        if contrast_markers:
            return ClassifiedSentence(
                text=sentence,
                function=SentenceFunction.CONTRAST,
                confidence=0.85,
                markers_found=contrast_markers
            )

        # 3. CONCESSION - acknowledging limitations
        concession_markers = self._find_concession_markers(text_lower, doc)
        if concession_markers:
            return ClassifiedSentence(
                text=sentence,
                function=SentenceFunction.CONCESSION,
                confidence=0.80,
                markers_found=concession_markers
            )

        # 4. RESOLUTION - concluding/resolving markers
        resolution_markers = self._find_resolution_markers(text_lower, doc)
        if resolution_markers:
            return ClassifiedSentence(
                text=sentence,
                function=SentenceFunction.RESOLUTION,
                confidence=0.80,
                markers_found=resolution_markers
            )

        # 5. EVIDENCE - supporting data/examples
        evidence_markers = self._find_evidence_markers(text_lower, doc)
        if evidence_markers:
            return ClassifiedSentence(
                text=sentence,
                function=SentenceFunction.EVIDENCE,
                confidence=0.75,
                markers_found=evidence_markers
            )

        # 6. ELABORATION - expansion/clarification
        elaboration_markers = self._find_elaboration_markers(text_lower, doc)
        if elaboration_markers:
            return ClassifiedSentence(
                text=sentence,
                function=SentenceFunction.ELABORATION,
                confidence=0.70,
                markers_found=elaboration_markers
            )

        # 7. SETUP - context/premise markers
        setup_markers = self._find_setup_markers(text_lower, doc)
        if setup_markers:
            return ClassifiedSentence(
                text=sentence,
                function=SentenceFunction.SETUP,
                confidence=0.70,
                markers_found=setup_markers
            )

        # 8. CLAIM - strong assertions (check verb patterns)
        if self._is_claim(doc):
            return ClassifiedSentence(
                text=sentence,
                function=SentenceFunction.CLAIM,
                confidence=0.60,
                markers_found=["declarative_pattern"]
            )

        # 9. Default: CONTINUATION
        return ClassifiedSentence(
            text=sentence,
            function=SentenceFunction.CONTINUATION,
            confidence=0.50,
            markers_found=[]
        )

    def classify_paragraph(self, paragraph: str) -> List[ClassifiedSentence]:
        """Classify all sentences in a paragraph.

        Args:
            paragraph: The paragraph text.

        Returns:
            List of ClassifiedSentence objects.
        """
        doc = self.nlp(paragraph)
        results = []

        for sent in doc.sents:
            sent_text = sent.text.strip()
            if sent_text:
                classified = self.classify(sent_text)
                results.append(classified)

        return results

    def extract_function_profile(
        self,
        paragraphs: List[str]
    ) -> Dict:
        """Extract function distribution and transitions from corpus.

        Args:
            paragraphs: List of paragraph texts.

        Returns:
            Dict with function_distribution, function_transitions,
            initial_function_probs, and function_samples.
        """
        function_counts = Counter()
        transitions = defaultdict(Counter)
        initial_counts = Counter()
        samples = defaultdict(list)

        total_sentences = 0

        for para in paragraphs:
            classified = self.classify_paragraph(para)

            if not classified:
                continue

            # Track initial function (paragraph starters)
            initial_counts[classified[0].function.value] += 1

            # Track all functions and transitions
            prev_function = None
            for sent in classified:
                func = sent.function.value
                function_counts[func] += 1
                total_sentences += 1

                # Collect samples (limit to 10 per function)
                if len(samples[func]) < 10:
                    samples[func].append(sent.text)

                # Track transition
                if prev_function is not None:
                    transitions[prev_function][func] += 1

                prev_function = func

        # Normalize to probabilities
        function_distribution = {
            func: count / total_sentences
            for func, count in function_counts.items()
        } if total_sentences > 0 else {}

        # Normalize transitions to conditional probabilities
        function_transitions = {}
        for from_func, to_counts in transitions.items():
            total = sum(to_counts.values())
            if total > 0:
                function_transitions[from_func] = {
                    to_func: count / total
                    for to_func, count in to_counts.items()
                }

        # Normalize initial probabilities
        total_initial = sum(initial_counts.values())
        initial_function_probs = {
            func: count / total_initial
            for func, count in initial_counts.items()
        } if total_initial > 0 else {}

        logger.info(
            f"Extracted function profile: {total_sentences} sentences, "
            f"{len(function_distribution)} function types"
        )

        return {
            "function_distribution": function_distribution,
            "function_transitions": function_transitions,
            "initial_function_probs": initial_function_probs,
            "function_samples": dict(samples),
        }

    def _is_question(self, text: str, doc) -> bool:
        """Check if sentence is a question."""
        # Direct check for question mark
        if text.strip().endswith("?"):
            return True

        # Check for interrogative structure without question mark
        first_token = doc[0] if len(doc) > 0 else None
        if first_token:
            # WH-words at start
            if first_token.tag_ in {"WDT", "WP", "WP$", "WRB"}:
                return True
            # Auxiliary inversion
            if first_token.pos_ == "AUX" and first_token.dep_ == "aux":
                return True

        return False

    def _find_contrast_markers(self, text_lower: str, doc) -> List[str]:
        """Find contrast/opposition markers at sentence start."""
        markers = []

        # Sentence-initial contrast markers
        contrast_starters = [
            "but ", "however,", "however ", "yet ", "still,", "still ",
            "nevertheless,", "nonetheless,", "on the contrary,",
            "in contrast,", "conversely,", "whereas ", "while ",
            "although ", "though ", "even though ", "despite ",
            "in spite of ", "contrary to ",
        ]

        for marker in contrast_starters:
            if text_lower.startswith(marker):
                markers.append(marker.strip().rstrip(","))
                break

        # Also check first token
        if len(doc) > 0:
            first = doc[0].text.lower()
            if first in {"but", "however", "yet", "still", "nevertheless", "nonetheless"}:
                if first not in markers:
                    markers.append(first)

        return markers

    def _find_concession_markers(self, text_lower: str, doc) -> List[str]:
        """Find concession markers (acknowledging limitations)."""
        markers = []

        # Concession patterns - phrases that acknowledge opposing views
        concession_patterns = [
            "admittedly,", "granted,", "to be sure,", "of course,",
            "it is true that", "true,", "certainly,",
            "no doubt,", "undoubtedly,", "i concede that",
            "it must be admitted that", "one might argue that",
            "while it is true", "although it is true",
        ]
        # Note: "we must acknowledge" removed - too generic, often just a claim

        for pattern in concession_patterns:
            if pattern in text_lower:
                markers.append(pattern.strip().rstrip(","))

        return markers

    def _find_resolution_markers(self, text_lower: str, doc) -> List[str]:
        """Find resolution/conclusion markers."""
        markers = []

        resolution_starters = [
            "therefore,", "therefore ", "thus,", "thus ", "hence,", "hence ",
            "consequently,", "as a result,", "in conclusion,", "to conclude,",
            "ultimately,", "finally,", "in the end,", "this means that",
            "this shows that", "this proves that", "we can conclude",
            "it follows that", "the answer is", "the solution is",
            "this resolves", "this explains",
        ]

        for marker in resolution_starters:
            if text_lower.startswith(marker) or marker in text_lower[:50]:
                markers.append(marker.strip().rstrip(","))

        return markers

    def _find_evidence_markers(self, text_lower: str, doc) -> List[str]:
        """Find evidence/example markers."""
        markers = []

        evidence_patterns = [
            "for example,", "for instance,", "such as ", "including ",
            "specifically,", "in particular,", "notably,", "namely,",
            "studies show", "research shows",
            "data shows", "evidence shows", "according to ", "as shown by",
            "statistics indicate", "surveys show", "experiments demonstrate",
            "the data", "the evidence", "the research",
        ]
        # Note: "consider" removed - it's a SETUP marker when at start

        for pattern in evidence_patterns:
            if pattern in text_lower:
                markers.append(pattern.strip().rstrip(","))

        return markers

    def _find_elaboration_markers(self, text_lower: str, doc) -> List[str]:
        """Find elaboration/expansion markers."""
        markers = []

        elaboration_patterns = [
            "that is,", "that is to say,", "in other words,", "i.e.,",
            "to put it another way,", "more specifically,", "to clarify,",
            "what this means is", "by this i mean", "to elaborate,",
            "in fact,", "indeed,", "actually,", "more precisely,",
        ]

        for pattern in elaboration_patterns:
            if text_lower.startswith(pattern) or pattern in text_lower[:30]:
                markers.append(pattern.strip().rstrip(","))

        return markers

    def _find_setup_markers(self, text_lower: str, doc) -> List[str]:
        """Find setup/context markers."""
        markers = []

        setup_patterns = [
            "consider ", "imagine ", "suppose ", "let us ", "let's ",
            "to understand ", "to begin,", "first,", "initially,",
            "in order to ", "before we ", "to start,", "historically,",
            "traditionally,", "in the past,", "once upon a time",
            "picture ", "think about ", "it is worth noting",
        ]

        for pattern in setup_patterns:
            if text_lower.startswith(pattern):
                markers.append(pattern.strip().rstrip(","))

        return markers

    def _is_claim(self, doc) -> bool:
        """Check if sentence is a strong claim/assertion.

        Uses syntactic patterns to identify declarative assertions.
        """
        # Look for strong assertion patterns
        for token in doc:
            # Strong modal verbs indicate claims
            if token.pos_ == "AUX" and token.lemma_ in {"must", "should", "need"}:
                return True

            # Assertive verbs
            if token.pos_ == "VERB" and token.lemma_ in {
                "be", "prove", "demonstrate", "show", "reveal", "establish",
                "confirm", "indicate", "suggest", "require", "demand",
                "constitute", "represent", "define", "determine",
            }:
                # Check if it's the root verb (main assertion)
                if token.dep_ == "ROOT":
                    return True

        # Check for definitional patterns ("X is Y")
        root = None
        for token in doc:
            if token.dep_ == "ROOT":
                root = token
                break

        if root and root.lemma_ == "be":
            # Simple "X is Y" pattern is often a claim
            return True

        return False


def classify_sentence_function(sentence: str) -> ClassifiedSentence:
    """Convenience function to classify a single sentence."""
    classifier = SentenceFunctionClassifier()
    return classifier.classify(sentence)
