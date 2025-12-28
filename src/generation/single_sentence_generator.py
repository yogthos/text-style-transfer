"""Single sentence generator with data-driven style constraints.

Generates one sentence at a time with:
1. Target length from Markov chain
2. Transition selection from author's distribution
3. Per-sentence verification
4. Repair loop if needed
"""

import random
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Callable
from collections import Counter

from ..style.profile import AuthorStyleProfile
from ..style.verifier import StyleVerifier, SentenceVerification
from ..utils.nlp import get_nlp, split_into_sentences
from ..utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class GenerationState:
    """State maintained across sentence generation."""

    previous_sentence: str = ""
    previous_length_category: str = "medium"
    used_transitions: Counter = field(default_factory=Counter)
    sentence_lengths: List[int] = field(default_factory=list)
    entities_introduced: set = field(default_factory=set)
    source_keywords: set = field(default_factory=set)


@dataclass
class SentenceSpec:
    """Specification for generating a single sentence."""

    proposition: str  # What to express
    target_length: int  # Target word count
    transition: Optional[str]  # Transition word to use (or None)
    required_keywords: List[str]  # Keywords that must appear
    position_in_paragraph: int  # 0-indexed position
    is_first: bool
    is_last: bool


class SingleSentenceGenerator:
    """Generate sentences one at a time with style constraints."""

    def __init__(
        self,
        profile: AuthorStyleProfile,
        llm_generate: Callable[[str], str],
        max_repair_attempts: int = 3,
    ):
        """Initialize generator.

        Args:
            profile: Author's extracted style profile.
            llm_generate: Function to call LLM with prompt.
            max_repair_attempts: Max attempts to repair a sentence.
        """
        self.profile = profile
        self.llm_generate = llm_generate
        self.max_repair_attempts = max_repair_attempts
        self.verifier = StyleVerifier(profile)
        self._nlp = None

    @property
    def nlp(self):
        if self._nlp is None:
            self._nlp = get_nlp()
        return self._nlp

    def select_target_length(self, state: GenerationState) -> int:
        """Select target length using author's Markov transitions.

        Args:
            state: Current generation state.

        Returns:
            Target word count for next sentence.
        """
        length_profile = self.profile.length_profile
        transitions = length_profile.length_transitions

        # Get transition probabilities from current category
        current_cat = state.previous_length_category
        if current_cat not in transitions:
            current_cat = "medium"

        probs = transitions[current_cat]

        # Sample next category
        categories = list(probs.keys())
        weights = [probs[cat] for cat in categories]
        next_category = random.choices(categories, weights=weights)[0]

        # Sample length from that category's range
        percentiles = length_profile.percentiles

        if next_category == "short":
            min_len = percentiles.get(10, 5)
            max_len = percentiles.get(25, 12)
        elif next_category == "long":
            min_len = percentiles.get(75, 25)
            max_len = percentiles.get(90, 45)
        else:  # medium
            min_len = percentiles.get(25, 12)
            max_len = percentiles.get(75, 25)

        return random.randint(min_len, max_len)

    def should_use_transition(
        self,
        position: int,
        state: GenerationState,
    ) -> Tuple[bool, Optional[str]]:
        """Decide if this sentence should have a transition.

        Based on author's actual usage patterns.

        Args:
            position: Position in paragraph (0-indexed).
            state: Current generation state.

        Returns:
            Tuple of (should_use, transition_word or None).
        """
        trans_profile = self.profile.transition_profile

        # First sentence rarely has transition
        if position == 0:
            return False, None

        # Roll against author's no-transition frequency
        if random.random() < trans_profile.no_transition_ratio:
            return False, None

        # Select category (for now, simple heuristic)
        # In full implementation, would consider context
        categories = [
            ("causal", trans_profile.causal),
            ("adversative", trans_profile.adversative),
            ("additive", trans_profile.additive),
        ]

        # Filter to non-empty categories
        categories = [(name, words) for name, words in categories if words]
        if not categories:
            return False, None

        # Prefer causal for non-first sentences
        category_name, word_probs = random.choice(categories)

        # Sample word from distribution, avoiding overused ones
        words = list(word_probs.keys())
        weights = list(word_probs.values())

        # Penalize already-used transitions
        adjusted_weights = []
        for word, weight in zip(words, weights):
            usage_count = state.used_transitions.get(word, 0)
            penalty = 0.5 ** usage_count  # Halve weight for each use
            adjusted_weights.append(weight * penalty)

        if sum(adjusted_weights) == 0:
            return False, None

        selected = random.choices(words, weights=adjusted_weights)[0]
        return True, selected

    def extract_keywords(self, source_sentence: str) -> List[str]:
        """Extract keywords that MUST appear in output.

        Args:
            source_sentence: Source text.

        Returns:
            List of required keywords.
        """
        doc = self.nlp(source_sentence)

        keywords = []
        for token in doc:
            # Nouns, proper nouns (entities)
            if token.pos_ in ("NOUN", "PROPN") and not token.is_stop:
                keywords.append(token.lemma_.lower())
            # Main verb
            elif token.pos_ == "VERB" and token.dep_ == "ROOT":
                keywords.append(token.lemma_.lower())

        # Named entities
        for ent in doc.ents:
            keywords.append(ent.text.lower())

        return list(set(keywords))

    def build_prompt(
        self,
        spec: SentenceSpec,
        state: GenerationState,
        style_example: Optional[str] = None,
    ) -> str:
        """Build minimal prompt for single sentence generation.

        Args:
            spec: Sentence specification.
            state: Current generation state.
            style_example: Optional example from corpus.

        Returns:
            Prompt string.
        """
        parts = []

        # Style example (if available)
        if style_example:
            parts.append(f'Style example: "{style_example}"')
            parts.append("")

        # Previous sentence context
        if state.previous_sentence:
            parts.append(f'Previous: "{state.previous_sentence}"')
        else:
            parts.append("This is the first sentence.")
        parts.append("")

        # What to express
        parts.append(f"Express: {spec.proposition}")
        parts.append("")

        # Constraints
        parts.append(f"Length: approximately {spec.target_length} words")

        if spec.transition:
            parts.append(f"Start with: {spec.transition}")
        else:
            parts.append("No transition word needed - start directly with content")

        if spec.required_keywords:
            parts.append(f"Must include: {', '.join(spec.required_keywords[:5])}")

        parts.append("")
        parts.append("Write ONLY the sentence:")

        return "\n".join(parts)

    def generate_sentence(
        self,
        proposition: str,
        state: GenerationState,
        position: int,
        is_first: bool = False,
        is_last: bool = False,
        style_example: Optional[str] = None,
    ) -> Tuple[str, GenerationState]:
        """Generate a single sentence with verification and repair.

        Args:
            proposition: What to express.
            state: Current generation state.
            position: Position in paragraph.
            is_first: Is this the first sentence?
            is_last: Is this the last sentence?
            style_example: Optional example from corpus.

        Returns:
            Tuple of (generated_sentence, updated_state).
        """
        # Determine target length
        target_length = self.select_target_length(state)

        # Determine transition
        use_transition, transition_word = self.should_use_transition(position, state)

        # Extract keywords from proposition
        keywords = self.extract_keywords(proposition)

        # Build specification
        spec = SentenceSpec(
            proposition=proposition,
            target_length=target_length,
            transition=transition_word if use_transition else None,
            required_keywords=keywords,
            position_in_paragraph=position,
            is_first=is_first,
            is_last=is_last,
        )

        # Generate with repair loop
        best_sentence = None
        best_verification = None

        for attempt in range(self.max_repair_attempts):
            prompt = self.build_prompt(spec, state, style_example)
            sentence = self.llm_generate(prompt).strip()

            # Clean up response
            sentence = self._clean_sentence(sentence)

            # Verify
            verification = self.verifier.verify_sentence(sentence, target_length)

            if verification.is_acceptable:
                best_sentence = sentence
                best_verification = verification
                break

            # Track best attempt
            if best_sentence is None or verification.length_deviation < best_verification.length_deviation:
                best_sentence = sentence
                best_verification = verification

            # Adjust for next attempt
            if verification.length_deviation > 0.25:
                # Adjust target length based on deviation
                actual_len = len(sentence.split())
                if actual_len > target_length:
                    spec.target_length = max(5, target_length - 5)
                else:
                    spec.target_length = target_length + 5

        # Update state
        new_state = self._update_state(state, best_sentence, spec)

        logger.debug(
            f"Generated: {best_sentence[:50]}... "
            f"(target={target_length}, actual={len(best_sentence.split())}, "
            f"transition={transition_word})"
        )

        return best_sentence, new_state

    def _clean_sentence(self, sentence: str) -> str:
        """Clean up LLM response."""
        # Remove quotes if wrapped
        sentence = sentence.strip()
        if sentence.startswith('"') and sentence.endswith('"'):
            sentence = sentence[1:-1]
        if sentence.startswith("'") and sentence.endswith("'"):
            sentence = sentence[1:-1]

        # Ensure ends with punctuation
        if sentence and sentence[-1] not in ".!?":
            sentence += "."

        return sentence

    def _update_state(
        self,
        state: GenerationState,
        sentence: str,
        spec: SentenceSpec,
    ) -> GenerationState:
        """Update generation state after producing a sentence."""
        length = len(sentence.split())
        category = self.profile.length_profile.get_length_category(length)

        new_state = GenerationState(
            previous_sentence=sentence,
            previous_length_category=category,
            used_transitions=state.used_transitions.copy(),
            sentence_lengths=state.sentence_lengths + [length],
            entities_introduced=state.entities_introduced.copy(),
            source_keywords=state.source_keywords.copy(),
        )

        # Track transition usage
        if spec.transition:
            new_state.used_transitions[spec.transition] += 1

        return new_state


class ParagraphOrchestrator:
    """Orchestrate sentence-by-sentence paragraph generation."""

    def __init__(
        self,
        profile: AuthorStyleProfile,
        llm_generate: Callable[[str], str],
        style_retriever: Optional[Callable[[str, int], Optional[str]]] = None,
    ):
        """Initialize orchestrator.

        Args:
            profile: Author's style profile.
            llm_generate: Function to call LLM.
            style_retriever: Optional function to retrieve style examples.
        """
        self.generator = SingleSentenceGenerator(profile, llm_generate)
        self.style_retriever = style_retriever
        self.profile = profile
        self.verifier = StyleVerifier(profile)

    def generate_paragraph(
        self,
        propositions: List[str],
    ) -> str:
        """Generate a paragraph from propositions.

        Args:
            propositions: List of propositions to express.

        Returns:
            Generated paragraph text.
        """
        if not propositions:
            return ""

        state = GenerationState()
        sentences = []

        for i, proposition in enumerate(propositions):
            is_first = i == 0
            is_last = i == len(propositions) - 1

            # Retrieve style example if available
            style_example = None
            if self.style_retriever:
                target_length = self.generator.select_target_length(state)
                style_example = self.style_retriever(proposition, target_length)

            # Generate sentence
            sentence, state = self.generator.generate_sentence(
                proposition=proposition,
                state=state,
                position=i,
                is_first=is_first,
                is_last=is_last,
                style_example=style_example,
            )

            sentences.append(sentence)

        # Join into paragraph
        paragraph = " ".join(sentences)

        # Verify full paragraph
        verification = self.verifier.verify_paragraph(paragraph)
        if not verification.is_acceptable:
            logger.warning(
                f"Paragraph verification issues: {verification.issues}"
            )

        return paragraph
