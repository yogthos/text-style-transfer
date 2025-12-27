"""Evolutionary sentence generator with selection pressure.

Uses genetic algorithm principles to evolve sentences toward target style:
1. Generate diverse population of candidates
2. Score with multi-objective fitness function
3. Select best candidates
4. Apply targeted mutations based on specific failures
5. Iterate until convergence
"""

import random
import re
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Callable, Tuple
from collections import Counter
import numpy as np

from ..style.profile import AuthorStyleProfile
from ..style.verifier import StyleVerifier
from ..utils.nlp import get_nlp, split_into_sentences
from ..utils.logging import get_logger
from ..humanization.pattern_injector import PatternInjector, HumanizationConfig
from ..humanization.corpus_patterns import HumanPatterns

logger = get_logger(__name__)


@dataclass
class Candidate:
    """A candidate sentence with fitness scores."""

    text: str
    generation: int = 0

    # Fitness components (0-1, higher is better)
    length_fitness: float = 0.0
    transition_fitness: float = 0.0
    vocabulary_fitness: float = 0.0
    fluency_fitness: float = 0.0

    # Diagnostic info
    word_count: int = 0
    target_length: int = 0
    has_transition: bool = False
    issues: List[str] = field(default_factory=list)

    @property
    def total_fitness(self) -> float:
        """Weighted fitness score."""
        return (
            self.length_fitness * 0.45 +      # Increased from 0.35
            self.transition_fitness * 0.20 +  # Decreased from 0.25
            self.vocabulary_fitness * 0.20 +  # Decreased from 0.25
            self.fluency_fitness * 0.15
        )

    def __lt__(self, other):
        return self.total_fitness < other.total_fitness


@dataclass
class GenerationState:
    """State maintained across sentence generation."""

    previous_sentences: List[str] = field(default_factory=list)
    previous_length_category: str = "medium"
    previous_structure_type: str = "simple"  # Track sentence structure
    previous_discourse_relation: str = "continuation"  # Track discourse flow
    used_transitions: Counter = field(default_factory=Counter)
    paragraph_lengths: List[int] = field(default_factory=list)
    target_burstiness: float = 0.5

    # Context tracking to avoid repetition
    used_phrases: set = field(default_factory=set)  # N-grams already used
    used_openings: set = field(default_factory=set)  # Sentence opening patterns
    mentioned_concepts: List[str] = field(default_factory=list)  # For implicit refs
    key_nouns: List[str] = field(default_factory=list)  # Nouns that can be pronominalized

    # Source discourse relation for current sentence (to preserve logic)
    required_discourse_relation: Optional[str] = None


class EvolutionarySentenceGenerator:
    """Generate sentences using evolutionary optimization.

    Key features:
    - Population-based search (not single-shot)
    - Multi-objective fitness (length, transitions, vocabulary)
    - Targeted mutations based on specific failures
    - Selection pressure toward convergence
    """

    def __init__(
        self,
        profile: AuthorStyleProfile,
        llm_generate: Callable[[str], str],
        population_size: int = 5,
        max_generations: int = 3,
        elite_count: int = 2,
        mutation_rate: float = 0.8,
    ):
        """Initialize evolutionary generator.

        Args:
            profile: Target author's style profile.
            llm_generate: Function to call LLM.
            population_size: Number of candidates per generation.
            max_generations: Maximum evolution iterations.
            elite_count: Number of top candidates to preserve.
            mutation_rate: Probability of applying mutations.
        """
        self.profile = profile
        self.llm_generate = llm_generate
        self.population_size = population_size
        self.max_generations = max_generations
        self.elite_count = elite_count
        self.mutation_rate = mutation_rate

        self.verifier = StyleVerifier(profile)
        self._nlp = None

        # Extract vocabulary from profile for fitness scoring
        self.target_vocabulary = set(profile.delta_profile.mfw_frequencies.keys())
        self.transition_words = profile.transition_profile.get_all_transitions()

        # Extract top content words (exclude function words) for vocabulary hints
        # Use spaCy to identify function words (closed-class POS tags)
        mfw = profile.delta_profile.mfw_frequencies
        self.vocab_hints = self._extract_content_words(mfw, limit=20)

        # Store structure samples and transitions from profile
        # Use raw_samples (original text) for prompts - better style matching
        # Fall back to sanitized samples if raw not available
        self.structure_samples = (
            profile.structure_profile.raw_samples
            if profile.structure_profile.raw_samples
            else profile.structure_profile.structure_samples
        )
        self.structure_transitions = profile.structure_profile.structure_transitions
        self.proposition_capacity = profile.structure_profile.proposition_capacity

        # Store discourse relation data for logic preservation
        self.discourse_samples = profile.discourse_profile.relation_samples
        self.discourse_transitions = profile.discourse_profile.relation_transitions
        self.discourse_connectives = profile.discourse_profile.relation_connectives

        # Initialize humanization pattern injector from profile
        self.pattern_injector = self._create_pattern_injector(profile.human_patterns)

    def _create_pattern_injector(self, human_patterns: Dict) -> PatternInjector:
        """Create PatternInjector from profile's human_patterns dict."""
        if not human_patterns:
            logger.info("No human patterns in profile, using default humanization config")
            return PatternInjector(patterns=None, config=HumanizationConfig())

        # Reconstruct HumanPatterns from dict
        patterns = HumanPatterns(
            fragments=human_patterns.get("fragments", []),
            questions=human_patterns.get("questions", []),
            asides=human_patterns.get("asides", []),
            dash_patterns=human_patterns.get("dash_patterns", []),
            colloquialisms=human_patterns.get("colloquialisms", []),
            short_sentences=human_patterns.get("short_sentences", []),
            long_sentences=human_patterns.get("long_sentences", []),
            unconventional_openers=human_patterns.get("unconventional_openers", []),
            fragment_ratio=human_patterns.get("fragment_ratio", 0.0),
            question_ratio=human_patterns.get("question_ratio", 0.0),
            dash_ratio=human_patterns.get("dash_ratio", 0.0),
            short_ratio=human_patterns.get("short_ratio", 0.0),
            long_ratio=human_patterns.get("long_ratio", 0.0),
        )

        logger.info(
            f"Initialized humanization from corpus: "
            f"fragments={len(patterns.fragments)}, "
            f"questions={len(patterns.questions)}, "
            f"dashes={len(patterns.dash_patterns)}"
        )

        return PatternInjector(patterns=patterns)

    @property
    def nlp(self):
        if self._nlp is None:
            self._nlp = get_nlp()
        return self._nlp

    def _extract_phrases(self, text: str) -> set:
        """Extract significant phrases (2-4 grams) from text using spaCy."""
        doc = self.nlp(text.lower())
        phrases = set()

        # Extract noun chunks
        for chunk in doc.noun_chunks:
            if len(chunk.text.split()) >= 2:
                phrases.add(chunk.text)

        # Extract 2-grams and 3-grams of content words
        content_tokens = [t for t in doc if not t.is_stop and t.is_alpha and len(t.text) > 2]
        for i in range(len(content_tokens) - 1):
            bigram = f"{content_tokens[i].text} {content_tokens[i+1].text}"
            phrases.add(bigram)
            if i < len(content_tokens) - 2:
                trigram = f"{content_tokens[i].text} {content_tokens[i+1].text} {content_tokens[i+2].text}"
                phrases.add(trigram)

        return phrases

    def _extract_key_nouns(self, text: str) -> List[str]:
        """Extract key nouns that can be referred to with pronouns."""
        doc = self.nlp(text)
        nouns = []
        for token in doc:
            if token.pos_ == 'NOUN' and not token.is_stop and len(token.text) > 3:
                nouns.append(token.text.lower())
        return nouns

    def _get_sentence_opening(self, text: str) -> str:
        """Extract the opening pattern of a sentence (first 3-4 words structure)."""
        doc = self.nlp(text)
        if len(doc) < 3:
            return ""
        # Get POS pattern of first 4 tokens
        opening = " ".join([t.pos_ for t in doc[:4]])
        return opening

    def _build_context_constraints(self, state: GenerationState) -> str:
        """Build prompt section about what to avoid based on context.

        These are structural/content constraints, not style instructions.
        """
        constraints = []

        # Phrases to avoid (prevents repetition)
        if state.used_phrases:
            recent_phrases = list(state.used_phrases)[-15:]  # Last 15 phrases
            constraints.append(f"AVOID these phrases already used: {', '.join(recent_phrases)}")

        # Previous sentence for context
        if state.previous_sentences:
            last = state.previous_sentences[-1]
            constraints.append(f"Previous sentence: \"{last}\"")

        return "\n".join(constraints)

    def _extract_content_words(self, mfw: Dict[str, float], limit: int = 20) -> List[str]:
        """Extract content words from MFW using spaCy.

        Uses spaCy's is_stop property and POS tagging to identify
        content words vs function words. Both are derived from
        linguistic data, not hardcoded word lists.
        """
        content_words = []
        # Sort by frequency descending
        sorted_words = sorted(mfw.items(), key=lambda x: -x[1])

        for word, freq in sorted_words:
            if len(word) < 3:  # Skip very short words
                continue

            # Use spaCy to analyze the word
            doc = self.nlp(word)
            if doc and len(doc) > 0:
                token = doc[0]
                # Skip stop words (function words) - spaCy's is_stop is linguistically derived
                if token.is_stop:
                    continue
                # Skip punctuation and symbols
                if token.is_punct or not token.is_alpha:
                    continue
                content_words.append(word)
                if len(content_words) >= limit:
                    break

        return content_words

    def _classify_source_discourse_relation(self, prev_prop: str, curr_prop: str) -> str:
        """Classify the discourse relation between two source propositions.

        Uses spaCy to identify logical relationships.
        """
        if not prev_prop:
            return "continuation"

        # Parse both propositions
        doc_prev = self.nlp(prev_prop)
        doc_curr = self.nlp(curr_prop)

        # Check first token of current proposition for discourse markers
        if len(doc_curr) > 0:
            first_token = doc_curr[0]
            first_pos = first_token.pos_

            if first_pos == 'CCONJ':
                if first_token.lemma_.lower() in ('but', 'yet'):
                    return "contrast"
                return "continuation"

            if first_pos == 'SCONJ':
                return "cause"

            if first_pos == 'ADV':
                return "elaboration"

        # Check for semantic contrast (opposing concepts)
        prev_verbs = [t.lemma_ for t in doc_prev if t.pos_ == 'VERB']
        curr_verbs = [t.lemma_ for t in doc_curr if t.pos_ == 'VERB']

        # Check for negation patterns suggesting contrast
        prev_has_neg = any(t.dep_ == 'neg' for t in doc_prev)
        curr_has_neg = any(t.dep_ == 'neg' for t in doc_curr)

        if prev_has_neg != curr_has_neg:
            return "contrast"

        # Default to elaboration if there's semantic similarity
        return "elaboration"

    def _get_discourse_connective(self, relation: str) -> Optional[str]:
        """Get an appropriate connective for a discourse relation from the corpus."""
        if not self.discourse_connectives or relation not in self.discourse_connectives:
            return None

        connectives = self.discourse_connectives[relation]
        if not connectives:
            return None

        # Sample based on corpus frequency
        items = list(connectives.items())
        words = [w for w, _ in items]
        probs = [p for _, p in items]

        return random.choices(words, weights=probs)[0]

    def _get_few_shot_example(self, relation: str) -> Optional[Tuple[str, str]]:
        """Get a few-shot example sentence pair for a discourse relation."""
        if not self.discourse_samples or relation not in self.discourse_samples:
            return None

        samples = self.discourse_samples[relation]
        if not samples:
            return None

        return random.choice(samples)

    def _get_target_structure_type(self, state: GenerationState) -> str:
        """Select next sentence structure type using Markov model.

        Uses the structure transition probabilities to determine
        what type of sentence to generate next.
        """
        if not self.structure_transitions:
            logger.debug("[STRUCTURE] No transitions available, defaulting to simple")
            return "simple"

        prev_type = state.previous_structure_type
        if prev_type not in self.structure_transitions:
            prev_type = "simple"

        transitions = self.structure_transitions.get(prev_type, {})
        if not transitions:
            return "simple"

        # Sample from transition probabilities
        types = list(transitions.keys())
        probs = [transitions[t] for t in types]
        selected = random.choices(types, weights=probs)[0]

        logger.debug(
            f"[STRUCTURE] Markov: {prev_type} -> {selected} "
            f"(probs: {dict(zip(types, [f'{p:.2f}' for p in probs]))})"
        )
        return selected

    def _classify_generated_structure(self, text: str) -> str:
        """Classify the structure of a generated sentence."""
        doc = self.nlp(text)

        num_verbs = sum(1 for t in doc if t.pos_ == 'VERB' and t.dep_ in ('ROOT', 'conj', 'ccomp', 'xcomp', 'advcl', 'relcl'))
        has_cconj = any(t.pos_ == 'CCONJ' for t in doc)
        has_sconj = any(t.pos_ == 'SCONJ' for t in doc)
        has_relcl = any(t.dep_ == 'relcl' for t in doc)
        has_advcl = any(t.dep_ == 'advcl' for t in doc)

        has_subordinate = has_sconj or has_relcl or has_advcl

        if num_verbs <= 1:
            return "simple"
        elif has_cconj and has_subordinate:
            return "compound_complex"
        elif has_cconj:
            return "compound"
        elif has_subordinate:
            return "complex"
        else:
            return "simple"

    def select_target_length(self, state: GenerationState) -> int:
        """Select target length using Markov chain."""
        length_profile = self.profile.length_profile
        transitions = length_profile.length_transitions

        current_cat = state.previous_length_category
        if current_cat not in transitions:
            current_cat = "medium"

        probs = transitions[current_cat]
        categories = list(probs.keys())
        weights = [probs[cat] for cat in categories]
        next_category = random.choices(categories, weights=weights)[0]

        percentiles = length_profile.percentiles

        if next_category == "short":
            min_len = percentiles.get(10, 5)
            max_len = percentiles.get(25, 12)
        elif next_category == "long":
            min_len = percentiles.get(75, 25)
            max_len = percentiles.get(90, 45)
        else:
            min_len = percentiles.get(25, 12)
            max_len = percentiles.get(75, 25)

        return random.randint(min_len, max_len), next_category

    def should_use_transition(
        self,
        position: int,
        state: GenerationState,
    ) -> Tuple[bool, Optional[str]]:
        """Decide if sentence should have a transition."""
        trans_profile = self.profile.transition_profile

        if position == 0:
            return False, None

        if random.random() < trans_profile.no_transition_ratio:
            return False, None

        # Select category
        categories = [
            ("causal", trans_profile.causal),
            ("adversative", trans_profile.adversative),
            ("additive", trans_profile.additive),
        ]
        categories = [(name, words) for name, words in categories if words]
        if not categories:
            return False, None

        category_name, word_probs = random.choice(categories)

        words = list(word_probs.keys())
        weights = list(word_probs.values())

        # Penalize overused transitions
        adjusted_weights = []
        for word, weight in zip(words, weights):
            usage_count = state.used_transitions.get(word, 0)
            penalty = 0.5 ** usage_count
            adjusted_weights.append(weight * penalty)

        if sum(adjusted_weights) == 0:
            return False, None

        selected = random.choices(words, weights=adjusted_weights)[0]
        return True, selected

    def _get_position_label(self, position: int, total_positions: int = 5) -> str:
        """Convert numeric position to label for humanization."""
        if position == 0:
            return "start"
        elif position >= total_positions - 1:
            return "end"
        else:
            return "middle"

    def generate_sentence(
        self,
        proposition: str,
        state: GenerationState,
        position: int,
        style_hint: str = "",
        total_sentences: int = 5,
    ) -> Tuple[str, GenerationState]:
        """Generate a sentence using evolutionary optimization.

        Args:
            proposition: What to express.
            state: Current generation state.
            style_hint: Additional context for style guidance.
            position: Position in paragraph.
            total_sentences: Expected total sentences in paragraph.

        Returns:
            Tuple of (best_sentence, updated_state).
        """
        # Determine targets
        target_length, length_category = self.select_target_length(state)
        use_transition, transition_word = self.should_use_transition(position, state)

        # Get humanization pattern request
        position_label = self._get_position_label(position, total_sentences)
        pattern_request = self.pattern_injector.get_pattern_request(position_label)

        # Apply humanization burstiness modification
        target_length = self.pattern_injector.modify_length_target(target_length, position_label)

        # Check if we need more burstiness (on top of humanization)
        need_length_variation = self._check_burstiness_pressure(state, target_length)
        if need_length_variation:
            # Force a different length category to increase burstiness
            target_length = self._adjust_for_burstiness(state, target_length)

        logger.info(
            f"[GEN] Position {position}: target_length={target_length}, "
            f"category={length_category}, transition={transition_word or 'none'}"
        )
        logger.debug(f"[GEN] Proposition: {proposition[:80]}...")
        if pattern_request:
            logger.debug(f"[GEN] Humanization: {pattern_request}")

        # Generate initial population
        population = self._generate_initial_population(
            proposition, state, target_length, transition_word, style_hint, pattern_request
        )

        logger.debug(f"[GEN] Initial population: {len(population)} candidates")
        for i, c in enumerate(population[:3]):
            logger.debug(f"[GEN]   [{i}] len={c.word_count} fit={c.total_fitness:.2f}: {c.text[:50]}...")

        # Evolve population
        best_candidate = self._evolve_population(
            population, proposition, state, target_length, transition_word, style_hint, pattern_request
        )

        # Update state
        new_state = self._update_state(
            state, best_candidate.text, length_category, transition_word
        )

        # Detailed logging of result
        length_diff = abs(best_candidate.word_count - target_length)
        length_status = "OK" if length_diff <= 3 else f"MISS by {length_diff}"
        logger.info(
            f"[GEN] Result: len={best_candidate.word_count}/{target_length} ({length_status}), "
            f"fitness={best_candidate.total_fitness:.2f}"
        )
        logger.debug(f"[GEN] Output: {best_candidate.text}")

        return best_candidate.text, new_state

    def _check_burstiness_pressure(
        self,
        state: GenerationState,
        proposed_length: int
    ) -> bool:
        """Check if we need to adjust length for burstiness."""
        if len(state.paragraph_lengths) < 2:
            return False

        # Calculate current burstiness
        lengths = state.paragraph_lengths + [proposed_length]
        mean = np.mean(lengths)
        std = np.std(lengths)
        current_burstiness = std / mean if mean > 0 else 0

        target_burstiness = self.profile.length_profile.burstiness

        # If we're significantly below target, apply pressure
        return current_burstiness < target_burstiness * 0.7

    def _adjust_for_burstiness(
        self,
        state: GenerationState,
        current_target: int
    ) -> int:
        """Adjust target length to increase burstiness."""
        if not state.paragraph_lengths:
            return current_target

        recent_mean = np.mean(state.paragraph_lengths[-3:])
        percentiles = self.profile.length_profile.percentiles

        # If recent sentences are medium/long, go short
        if recent_mean > percentiles.get(50, 18):
            return random.randint(
                percentiles.get(10, 5),
                percentiles.get(25, 12)
            )
        # If recent sentences are short, go long
        else:
            return random.randint(
                percentiles.get(75, 25),
                percentiles.get(90, 45)
            )

    def _generate_initial_population(
        self,
        proposition: str,
        state: GenerationState,
        target_length: int,
        transition_word: Optional[str],
        style_hint: str = "",
        pattern_request: Optional[str] = None,
    ) -> List[Candidate]:
        """Generate initial population of diverse candidates."""
        population = []

        # Generate diverse prompts with different instructions
        prompt_variants = self._create_prompt_variants(
            proposition, state, target_length, transition_word, style_hint, pattern_request
        )

        for i, prompt in enumerate(prompt_variants[:self.population_size]):
            try:
                response = self.llm_generate(prompt)
                text = self._clean_sentence(response)

                candidate = self._evaluate_candidate(
                    text, target_length, transition_word, state
                )
                candidate.generation = 0
                population.append(candidate)
            except Exception as e:
                logger.warning(f"Failed to generate candidate {i}: {e}")

        return population

    def _create_prompt_variants(
        self,
        proposition: str,
        state: GenerationState,
        target_length: int,
        transition_word: Optional[str],
        style_hint: str = "",
        pattern_request: Optional[str] = None,
    ) -> List[str]:
        """Create diverse prompt variants for population diversity.

        Style comes ONLY from corpus samples - no hardcoded style descriptions.
        Variants differ only in target length (for burstiness) and structural constraints.
        Humanization patterns (fragments, questions, dashes) are injected for variety.
        """
        variants = []

        # Pass through any RST/entity context hints (these are content, not style)
        context_hint = style_hint if style_hint else ""

        # Variant 1: Standard prompt with base target length + humanization pattern
        variants.append(self._build_prompt(
            proposition, target_length, transition_word, state,
            style_hint=context_hint, pattern_request=pattern_request
        ))

        # Variant 2: Emphasize exact length with pattern
        length_hint = f"{context_hint} EXACTLY {target_length} words." if context_hint else f"EXACTLY {target_length} words."
        variants.append(self._build_prompt(
            proposition, target_length, transition_word, state,
            style_hint=length_hint, pattern_request=pattern_request
        ))

        # Variant 3: Shorter target (for burstiness variation) with pattern
        variants.append(self._build_prompt(
            proposition, max(8, target_length - 8), transition_word, state,
            style_hint=context_hint, pattern_request=pattern_request
        ))

        # Variant 4: Longer target (for burstiness variation) with pattern
        variants.append(self._build_prompt(
            proposition, target_length + 10, transition_word, state,
            style_hint=context_hint, pattern_request=pattern_request
        ))

        # Variant 5: Much longer (for high burstiness authors) with pattern
        variants.append(self._build_prompt(
            proposition, min(50, target_length + 20), transition_word, state,
            style_hint=context_hint, pattern_request=pattern_request
        ))

        # Variant 6: Without pattern request (for diversity)
        variants.append(self._build_prompt(
            proposition, target_length, transition_word, state,
            style_hint=context_hint, pattern_request=None
        ))

        return variants

    def _build_prompt(
        self,
        proposition: str,
        target_length: int,
        transition_word: Optional[str],
        state: GenerationState,
        style_hint: str = "",
        pattern_request: Optional[str] = None,
    ) -> str:
        """Build a generation prompt using corpus examples only - no hardcoded style.

        Humanization patterns (fragments, questions, dashes) are injected to
        address AI detection signals like mechanical precision and predictable syntax.
        """
        parts = []

        # Determine target structure type from Markov model
        target_structure = self._get_target_structure_type(state)

        # Show corpus examples for STRUCTURE/RHYTHM only
        parts.append(f"STYLE REFERENCE (target: {target_structure} sentence):")
        parts.append("")

        # Select samples matching target structure type - focused selection
        samples_shown = 0
        selected_samples = []

        if hasattr(self, 'structure_samples') and self.structure_samples:
            # First, try to get samples matching target structure
            if target_structure in self.structure_samples and self.structure_samples[target_structure]:
                available = self.structure_samples[target_structure]
                selected_samples = random.sample(available, min(3, len(available)))
                logger.debug(f"[SAMPLE] Selected {len(selected_samples)} {target_structure} samples")

            # If not enough, fill from other types
            if len(selected_samples) < 2:
                for struct_type in ['simple', 'compound', 'complex']:
                    if struct_type != target_structure and struct_type in self.structure_samples:
                        available = self.structure_samples[struct_type]
                        if available:
                            selected_samples.append(random.choice(available))
                            if len(selected_samples) >= 3:
                                break

        for sample in selected_samples[:3]:
            parts.append(f'"{sample}"')
            samples_shown += 1
            logger.debug(f"[SAMPLE] Using: {sample[:60]}...")

        if samples_shown == 0:
            parts.append("(No style samples available)")
            logger.warning("[SAMPLE] No style samples available!")

        parts.append("")
        parts.append("(Match the sentence RHYTHM and STRUCTURE above, not the topics.)")
        parts.append("---")

        # Context from previous sentences
        if state.previous_sentences:
            parts.append(f"Previous: \"{state.previous_sentences[-1]}\"")

        # The task - STRICT content preservation
        parts.append("")
        parts.append("CONTENT TO EXPRESS (do not add, remove, or change meaning):")
        parts.append(f'"{proposition}"')
        parts.append("")
        parts.append(f"Length: ~{target_length} words")

        # Transition guidance from corpus data
        # IMPORTANT: Only add transition if explicitly requested (76.7% should have NO transition)
        discourse_relation = state.required_discourse_relation or "continuation"
        has_explicit_transition = False

        if discourse_relation == "contrast":
            connective = self._get_discourse_connective("contrast")
            if connective:
                parts.append(f"Start with: \"{connective.capitalize()}\"")
                has_explicit_transition = True
        elif transition_word:
            parts.append(f"Start with: \"{transition_word.capitalize()}\"")
            has_explicit_transition = True

        # If no transition requested, explicitly tell LLM not to add one
        if not has_explicit_transition:
            parts.append("Do NOT start with a transition word (no 'So', 'But', 'However', etc.)")

        if style_hint:
            parts.append(style_hint)

        # HUMANIZATION: Add pattern request if available
        # This addresses AI detection signals by requesting human writing patterns
        if pattern_request:
            parts.append(f"Structure: {pattern_request}")

        parts.append("")
        parts.append("CRITICAL: Express ONLY the idea above. Do not add new concepts, examples, or embellishments.")
        parts.append("Write ONE sentence matching the style examples:")

        return "\n".join(parts)

    def _evolve_population(
        self,
        population: List[Candidate],
        proposition: str,
        state: GenerationState,
        target_length: int,
        transition_word: Optional[str],
        style_hint: str = "",
        pattern_request: Optional[str] = None,
    ) -> Candidate:
        """Evolve population through selection and mutation."""
        if not population:
            # Fallback: generate a single candidate
            prompt = self._build_prompt(
                proposition, target_length, transition_word, state, style_hint,
                pattern_request=pattern_request
            )
            text = self._clean_sentence(self.llm_generate(prompt))
            return self._evaluate_candidate(text, target_length, transition_word, state)

        for gen in range(self.max_generations):
            # Sort by fitness
            population.sort(reverse=True)

            # Check for convergence
            best = population[0]
            if best.total_fitness > 0.85:
                logger.debug(f"Converged at generation {gen}")
                break

            # Selection: keep elite
            elite = population[:self.elite_count]

            # Generate new candidates through mutation
            new_candidates = []
            for candidate in elite:
                if random.random() < self.mutation_rate:
                    mutated = self._mutate_candidate(
                        candidate, proposition, target_length,
                        transition_word, state
                    )
                    if mutated:
                        mutated.generation = gen + 1
                        new_candidates.append(mutated)

            # Combine elite with mutants
            population = elite + new_candidates

            # Fill remaining slots with new random candidates if needed
            while len(population) < self.population_size:
                prompt = random.choice(self._create_prompt_variants(
                    proposition, state, target_length, transition_word,
                    pattern_request=pattern_request
                ))
                try:
                    text = self._clean_sentence(self.llm_generate(prompt))
                    candidate = self._evaluate_candidate(
                        text, target_length, transition_word, state
                    )
                    candidate.generation = gen + 1
                    population.append(candidate)
                except:
                    break

        # Return best
        population.sort(reverse=True)
        return population[0]

    def _mutate_candidate(
        self,
        candidate: Candidate,
        proposition: str,
        target_length: int,
        transition_word: Optional[str],
        state: GenerationState,
    ) -> Optional[Candidate]:
        """Apply targeted mutation based on candidate's weaknesses."""
        # Identify the main issue
        issues = candidate.issues

        mutation_prompt = None

        if candidate.length_fitness < 0.7:
            # Length is off - request specific adjustment (structural, not style)
            length_diff = candidate.word_count - target_length
            if length_diff > 0:
                mutation_prompt = (
                    f'Shorten to {target_length} words (currently {candidate.word_count}):\n'
                    f'"{candidate.text}"\n'
                    f'Keep the meaning. Write ONLY the result:'
                )
            else:
                mutation_prompt = (
                    f'Expand to {target_length} words (currently {candidate.word_count}):\n'
                    f'"{candidate.text}"\n'
                    f'Keep the meaning. Write ONLY the result:'
                )

        elif candidate.transition_fitness < 0.7:
            # Transition issue (structural constraint from corpus)
            if transition_word and not candidate.has_transition:
                mutation_prompt = (
                    f'Rewrite starting with "{transition_word.capitalize()}":\n'
                    f'"{candidate.text}"\n'
                    f'Write ONLY the result:'
                )
            elif not transition_word and candidate.has_transition:
                mutation_prompt = (
                    f'Rewrite WITHOUT a transition word at the start:\n'
                    f'"{candidate.text}"\n'
                    f'Write ONLY the result:'
                )

        elif candidate.vocabulary_fitness < 0.6:
            # Vocabulary mismatch - show corpus examples instead of describing style
            vocab_sample = ', '.join(self.vocab_hints[:10]) if self.vocab_hints else ''
            mutation_prompt = (
                f'Rewrite using words like: {vocab_sample}\n'
                f'"{candidate.text}"\n'
                f'Write ONLY the result:'
            )

        else:
            # General refinement - just adjust length
            mutation_prompt = (
                f'Rewrite in {target_length} words:\n'
                f'"{candidate.text}"\n'
                f'Write ONLY the result:'
            )

        if mutation_prompt:
            try:
                response = self.llm_generate(mutation_prompt)
                text = self._clean_sentence(response)
                return self._evaluate_candidate(text, target_length, transition_word, state)
            except Exception as e:
                logger.warning(f"Mutation failed: {e}")

        return None

    def _evaluate_candidate(
        self,
        text: str,
        target_length: int,
        transition_word: Optional[str],
        state: Optional[GenerationState] = None,
    ) -> Candidate:
        """Evaluate a candidate sentence on multiple fitness dimensions."""
        candidate = Candidate(text=text)

        # Word count
        words = text.split()
        candidate.word_count = len(words)
        candidate.target_length = target_length

        # Length fitness (gaussian around target, tighter sigma)
        length_diff = abs(candidate.word_count - target_length)
        # Sigma of 3 means: 3 words off = 0.61 fitness, 6 words off = 0.14 fitness
        candidate.length_fitness = np.exp(-0.5 * (length_diff / 3) ** 2)
        if length_diff > target_length * 0.2:
            candidate.issues.append("length_deviation")

        # Transition fitness
        first_words = " ".join(words[:3]).lower() if words else ""
        has_any_transition = any(t in first_words for t in self.transition_words)
        candidate.has_transition = has_any_transition

        if transition_word:
            # Should have specific transition
            if transition_word.lower() in first_words:
                candidate.transition_fitness = 1.0
            elif has_any_transition:
                candidate.transition_fitness = 0.5  # Has a transition, but wrong one
                candidate.issues.append("wrong_transition")
            else:
                candidate.transition_fitness = 0.2
                candidate.issues.append("missing_transition")
        else:
            # Should NOT have transition
            if has_any_transition:
                candidate.transition_fitness = 0.3
                candidate.issues.append("unwanted_transition")
            else:
                candidate.transition_fitness = 1.0

        # Vocabulary fitness (overlap with author's MFW)
        text_words = set(w.lower() for w in re.findall(r'\b[a-z]+\b', text.lower()))
        if text_words and self.target_vocabulary:
            overlap = len(text_words & self.target_vocabulary)
            candidate.vocabulary_fitness = min(1.0, overlap / max(5, len(text_words) * 0.3))
        else:
            candidate.vocabulary_fitness = 0.5

        # Fluency fitness (basic checks)
        candidate.fluency_fitness = 1.0
        if not text.strip():
            candidate.fluency_fitness = 0.0
        elif text[-1] not in '.!?':
            candidate.fluency_fitness *= 0.8
        elif text.count('"') % 2 != 0:
            candidate.fluency_fitness *= 0.9

        # Repetition penalty - penalize reuse of phrases from previous sentences
        if state and state.used_phrases:
            candidate_phrases = self._extract_phrases(text)
            overlap = candidate_phrases & state.used_phrases
            if overlap:
                # Penalize based on how many phrases are repeated
                penalty = 0.8 ** len(overlap)
                candidate.fluency_fitness *= penalty
                if len(overlap) > 2:
                    candidate.issues.append("phrase_repetition")

            # Also check opening pattern similarity
            opening = self._get_sentence_opening(text)
            if opening and opening in state.used_openings:
                candidate.fluency_fitness *= 0.7
                candidate.issues.append("similar_opening")

        return candidate

    def _clean_sentence(self, text: str) -> str:
        """Clean LLM response to get a single sentence."""
        text = text.strip()

        # Remove common prefixes
        prefixes = ["Here's", "Here is", "Sentence:", "Output:", "Result:", "Sure:"]
        for prefix in prefixes:
            if text.lower().startswith(prefix.lower()):
                text = text[len(prefix):].strip()
                if text.startswith(":"):
                    text = text[1:].strip()

        # Remove quotes
        if text.startswith('"') and text.endswith('"'):
            text = text[1:-1]
        if text.startswith("'") and text.endswith("'"):
            text = text[1:-1]

        # Remove newlines
        text = ' '.join(text.split())

        # Keep only the first sentence if multiple were returned
        # Look for sentence boundaries but be careful with abbreviations
        sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', text)
        if len(sentences) > 1:
            # Take the longest sentence (likely the main content)
            text = max(sentences, key=len)

        # Ensure punctuation
        if text and text[-1] not in '.!?':
            text += '.'

        return text

    def _update_state(
        self,
        state: GenerationState,
        sentence: str,
        length_category: str,
        transition_word: Optional[str],
    ) -> GenerationState:
        """Update generation state after producing a sentence."""
        # Extract phrases and nouns from the new sentence
        new_phrases = self._extract_phrases(sentence)
        new_nouns = self._extract_key_nouns(sentence)
        new_opening = self._get_sentence_opening(sentence)
        new_structure = self._classify_generated_structure(sentence)

        new_state = GenerationState(
            previous_sentences=state.previous_sentences + [sentence],
            previous_length_category=length_category,
            previous_structure_type=new_structure,
            used_transitions=state.used_transitions.copy(),
            paragraph_lengths=state.paragraph_lengths + [len(sentence.split())],
            target_burstiness=state.target_burstiness,
            # Context tracking
            used_phrases=state.used_phrases | new_phrases,
            used_openings=state.used_openings | ({new_opening} if new_opening else set()),
            mentioned_concepts=state.mentioned_concepts + new_nouns,
            key_nouns=state.key_nouns + new_nouns,
        )

        if transition_word:
            new_state.used_transitions[transition_word] += 1

        return new_state


class EvolutionaryParagraphGenerator:
    """Generate paragraphs using evolutionary sentence generation."""

    def __init__(
        self,
        profile: AuthorStyleProfile,
        llm_generate: Callable[[str], str],
        population_size: int = 5,
        max_generations: int = 3,
    ):
        """Initialize paragraph generator."""
        self.generator = EvolutionarySentenceGenerator(
            profile=profile,
            llm_generate=llm_generate,
            population_size=population_size,
            max_generations=max_generations,
        )
        self.profile = profile
        self.verifier = StyleVerifier(profile)

    def generate_paragraph(
        self,
        propositions: List[str],
        rst_info: Optional[List[dict]] = None,
    ) -> str:
        """Generate a paragraph from propositions with RST awareness.

        KEY: Groups propositions based on target sentence structure complexity.
        A compound_complex sentence might combine 3-4 propositions.

        Args:
            propositions: List of propositions to express.
            rst_info: Optional RST info for each proposition with keys:
                - role: "nucleus" or "satellite"
                - relation: type of relation (example, evidence, contrast, etc.)
                - parent_idx: index of parent nucleus for satellites
                - entities: list of entities in the proposition
                - citations: list of citations attached to proposition

        Returns:
            Generated paragraph text.
        """
        if not propositions:
            return ""

        logger.info(f"[PARA] Generating paragraph from {len(propositions)} propositions")

        state = GenerationState(
            target_burstiness=self.profile.length_profile.burstiness
        )
        sentences = []
        sentence_position = 0

        # Track entity mentions for coherent references (implicit refs)
        mentioned_entities = set()
        mentioned_nouns = []

        # Group propositions based on target structure complexity
        prop_groups = self._group_propositions(propositions, rst_info, state)

        logger.info(f"[PARA] Grouped into {len(prop_groups)} sentence groups")

        for group_idx, group in enumerate(prop_groups):
            group_props = group["propositions"]
            group_rst = group.get("rst_info", [])
            target_structure = group["target_structure"]
            citations = group.get("citations", [])

            logger.debug(
                f"[PARA] Group {group_idx}: {len(group_props)} props, "
                f"structure={target_structure}, citations={citations}"
            )

            # Combine propositions into a single content block
            if len(group_props) == 1:
                combined_content = group_props[0]
            else:
                # Join with semicolons or logical connectors for LLM to reformulate
                combined_content = "; ".join(group_props)

            # Build context hint for implicit references
            context_hint = ""
            if mentioned_nouns:
                # Suggest using implicit references to previously mentioned concepts
                recent_nouns = mentioned_nouns[-5:]
                context_hint = f"[Can refer to: {', '.join(recent_nouns)}]"

            # Add RST context if satellite
            if group_rst:
                primary_rst = group_rst[0]
                relation = primary_rst.get("relation", "none")
                if relation != "none" and sentence_position > 0:
                    state.required_discourse_relation = relation
            else:
                state.required_discourse_relation = None

            # Add citation to content if present
            if citations:
                combined_content += f" {' '.join(citations)}"

            # Generate the sentence
            sentence, state = self.generator.generate_sentence(
                proposition=combined_content,
                state=state,
                position=sentence_position,
                style_hint=context_hint,
                total_sentences=len(prop_groups),
            )

            # Validate citation preservation
            if citations:
                missing_citations = [c for c in citations if c not in sentence]
                if missing_citations:
                    logger.warning(f"[PARA] Missing citations: {missing_citations}")
                    # Append missing citations
                    sentence = sentence.rstrip('.!?') + ' ' + ' '.join(missing_citations) + '.'

            # Prevent duplicate sentences
            if sentences and sentence.strip() == sentences[-1].strip():
                logger.warning(f"[PARA] Duplicate sentence detected, regenerating...")
                sentence, state = self.generator.generate_sentence(
                    proposition=combined_content,
                    state=state,
                    position=sentence_position,
                    style_hint=context_hint,
                    total_sentences=len(prop_groups),
                )

            sentences.append(sentence)
            sentence_position += 1

            # Update mentioned entities for implicit references
            for rst in group_rst:
                mentioned_entities.update(rst.get("entities", []))

            # Extract nouns from generated sentence for future implicit refs
            new_nouns = self.generator._extract_key_nouns(sentence)
            mentioned_nouns.extend(new_nouns)

        paragraph = " ".join(sentences)

        # Verify and log
        verification = self.verifier.verify_paragraph(paragraph)
        logger.info(
            f"[PARA] Result: {len(sentences)} sentences, "
            f"score={verification.overall_score:.2f}, "
            f"issues={verification.issues if verification.issues else 'none'}"
        )

        return paragraph

    def _group_propositions(
        self,
        propositions: List[str],
        rst_info: Optional[List[dict]],
        state: GenerationState,
    ) -> List[dict]:
        """Group propositions based on target sentence structure complexity.

        Uses Markov model to determine structure type, then groups
        propositions according to the capacity of that structure.
        """
        if not propositions:
            return []

        groups = []
        prop_idx = 0
        capacity_map = self.profile.structure_profile.proposition_capacity

        while prop_idx < len(propositions):
            # Determine target structure using Markov model
            target_structure = self.generator._get_target_structure_type(state)
            capacity = capacity_map.get(target_structure, 1)

            # Don't exceed remaining propositions
            actual_capacity = min(capacity, len(propositions) - prop_idx)

            # Collect propositions for this group
            group_props = propositions[prop_idx:prop_idx + actual_capacity]
            group_rst = []
            group_citations = []

            if rst_info:
                for i in range(prop_idx, min(prop_idx + actual_capacity, len(rst_info))):
                    group_rst.append(rst_info[i])
                    # Collect citations from RST info
                    citations = rst_info[i].get("citations", [])
                    group_citations.extend(citations)

            groups.append({
                "propositions": group_props,
                "rst_info": group_rst,
                "target_structure": target_structure,
                "citations": group_citations,
            })

            logger.debug(
                f"[GROUP] Created group: {actual_capacity} props for {target_structure} "
                f"(capacity={capacity})"
            )

            # Update state for next Markov transition
            state = GenerationState(
                previous_structure_type=target_structure,
                target_burstiness=state.target_burstiness,
            )

            prop_idx += actual_capacity

        return groups
