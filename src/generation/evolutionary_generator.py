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
        self.structure_samples = profile.structure_profile.structure_samples
        self.structure_transitions = profile.structure_profile.structure_transitions
        self.proposition_capacity = profile.structure_profile.proposition_capacity

        # Store discourse relation data for logic preservation
        self.discourse_samples = profile.discourse_profile.relation_samples
        self.discourse_transitions = profile.discourse_profile.relation_transitions
        self.discourse_connectives = profile.discourse_profile.relation_connectives

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
        """Build prompt section about what to avoid based on context."""
        constraints = []

        # Phrases to avoid
        if state.used_phrases:
            recent_phrases = list(state.used_phrases)[-15:]  # Last 15 phrases
            constraints.append(f"AVOID these phrases already used: {', '.join(recent_phrases)}")

        # Previous sentence for flow
        if state.previous_sentences:
            last = state.previous_sentences[-1]
            constraints.append(f"Previous sentence: \"{last}\"")
            constraints.append("Continue naturally from this context.")

            # If we have mentioned concepts, suggest implicit reference
            if state.key_nouns:
                recent_nouns = state.key_nouns[-3:]
                constraints.append(f"You may refer to '{recent_nouns[-1]}' using 'it', 'this', or 'such'")

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

    def _get_current_structure_type(self, state: GenerationState) -> str:
        """Select next sentence structure type using Markov model."""
        if not self.structure_transitions:
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
        return random.choices(types, weights=probs)[0]

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

    def generate_sentence(
        self,
        proposition: str,
        state: GenerationState,
        position: int,
        style_hint: str = "",
    ) -> Tuple[str, GenerationState]:
        """Generate a sentence using evolutionary optimization.

        Args:
            proposition: What to express.
            state: Current generation state.
            style_hint: Additional context for style guidance.
            position: Position in paragraph.

        Returns:
            Tuple of (best_sentence, updated_state).
        """
        # Determine targets
        target_length, length_category = self.select_target_length(state)
        use_transition, transition_word = self.should_use_transition(position, state)

        # Check if we need more burstiness
        need_length_variation = self._check_burstiness_pressure(state, target_length)
        if need_length_variation:
            # Force a different length category to increase burstiness
            target_length = self._adjust_for_burstiness(state, target_length)

        logger.debug(
            f"Generating sentence: target_length={target_length}, "
            f"transition={transition_word}, position={position}"
        )

        # Generate initial population
        population = self._generate_initial_population(
            proposition, state, target_length, transition_word, style_hint
        )

        # Evolve population
        best_candidate = self._evolve_population(
            population, proposition, state, target_length, transition_word, style_hint
        )

        # Update state
        new_state = self._update_state(
            state, best_candidate.text, length_category, transition_word
        )

        logger.debug(
            f"Best candidate: fitness={best_candidate.total_fitness:.3f}, "
            f"length={best_candidate.word_count}/{target_length}, "
            f"generation={best_candidate.generation}"
        )

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
    ) -> List[Candidate]:
        """Generate initial population of diverse candidates."""
        population = []

        # Generate diverse prompts with different instructions
        prompt_variants = self._create_prompt_variants(
            proposition, state, target_length, transition_word, style_hint
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
    ) -> List[str]:
        """Create diverse prompt variants for population diversity."""
        variants = []

        # Build context hint from RST/entity info
        base_hint = style_hint if style_hint else "Write naturally, varying structure from previous sentences."

        # Variant 1: Standard prompt with full context
        variants.append(self._build_prompt(
            proposition, target_length, transition_word, state,
            style_hint=base_hint
        ))

        # Variant 2: Emphasize length with context
        length_hint = f"{base_hint} The sentence MUST be exactly {target_length} words."
        variants.append(self._build_prompt(
            proposition, target_length, transition_word, state,
            style_hint=length_hint
        ))

        # Variant 3: Shorter target (for burstiness)
        variants.append(self._build_prompt(
            proposition, max(8, target_length - 8), transition_word, state,
            style_hint="Be concise. Use a different structure than previous sentences."
        ))

        # Variant 4: Longer target (for burstiness)
        variants.append(self._build_prompt(
            proposition, target_length + 10, transition_word, state,
            style_hint="Develop with concrete examples. Vary the sentence structure."
        ))

        # Variant 5: Much longer (for high burstiness)
        variants.append(self._build_prompt(
            proposition, min(50, target_length + 20), transition_word, state,
            style_hint="Write a complex sentence with multiple clauses."
        ))

        # Variant 6: Implicit reference focus
        if state.key_nouns:
            variants.append(self._build_prompt(
                proposition, target_length, transition_word, state,
                style_hint=f"Refer back to previous concepts using 'this', 'it', or 'such'."
            ))

        return variants

    def _build_prompt(
        self,
        proposition: str,
        target_length: int,
        transition_word: Optional[str],
        state: GenerationState,
        style_hint: str = "",
    ) -> str:
        """Build a generation prompt with full context awareness and discourse preservation."""
        parts = []

        # Get required discourse relation (preserving source logic)
        discourse_relation = state.required_discourse_relation or "continuation"

        # Add few-shot example from corpus for this discourse relation
        if discourse_relation != "continuation":
            example = self._get_few_shot_example(discourse_relation)
            if example:
                parts.append("EXAMPLE of the required logical flow:")
                parts.append(f"  Previous: \"{example[0]}\"")
                parts.append(f"  Current ({discourse_relation.upper()}): \"{example[1]}\"")
                parts.append("")

        # Context from previous sentences
        context_constraints = self._build_context_constraints(state)
        if context_constraints:
            parts.append(context_constraints)

        parts.append(f"Write ONE SINGLE sentence expressing: {proposition}")
        parts.append(f"REQUIRED: The sentence must be {target_length} words (minimum {max(10, target_length - 5)} words)")

        # Discourse relation guidance - balance source logic with target author's style
        # Critical relations (contrast, cause) take priority; others respect transition decision
        if discourse_relation == "contrast":
            # CONTRAST is critical for logic - always signal it
            connective = self._get_discourse_connective("contrast")
            if connective:
                parts.append(f"This sentence CONTRASTS with the previous. Start with: \"{connective.capitalize()}\"")
            else:
                parts.append("This sentence CONTRASTS with the previous (show opposition or concession)")
        elif discourse_relation == "cause" and transition_word:
            # CAUSE - only if we're allowed to use a transition
            parts.append(f"This sentence shows CAUSE/RESULT. Start with: \"{transition_word.capitalize()}\"")
        elif transition_word:
            # We're allowed a transition - use the selected word
            parts.append(f"Start with: \"{transition_word.capitalize()}\"")
        else:
            # No transition allowed - flow naturally without explicit markers
            parts.append("Do NOT start with a transition word (but, however, so, therefore, etc.)")

        # Style guidance - use actual samples from corpus
        if hasattr(self, 'structure_samples') and self.structure_samples:
            structure_type = self._get_current_structure_type(state)
            if structure_type in self.structure_samples and self.structure_samples[structure_type]:
                sample = random.choice(self.structure_samples[structure_type])
                parts.append(f"Match this style/register: \"{sample}\"")

        if style_hint:
            parts.append(style_hint)

        parts.append("Output ONLY the single sentence (no explanation, no multiple sentences):")

        return "\n".join(parts)

    def _evolve_population(
        self,
        population: List[Candidate],
        proposition: str,
        state: GenerationState,
        target_length: int,
        transition_word: Optional[str],
        style_hint: str = "",
    ) -> Candidate:
        """Evolve population through selection and mutation."""
        if not population:
            # Fallback: generate a single candidate
            prompt = self._build_prompt(
                proposition, target_length, transition_word, state, style_hint
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
                    proposition, state, target_length, transition_word
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
            # Length is off - request specific adjustment
            length_diff = candidate.word_count - target_length
            if length_diff > 0:
                mutation_prompt = (
                    f'Shorten this sentence to exactly {target_length} words '
                    f'(currently {candidate.word_count} words):\n'
                    f'"{candidate.text}"\n'
                    f'Keep the core meaning. Write ONLY the shortened sentence:'
                )
            else:
                # Sentence is too short - expand it with concrete details
                mutation_prompt = (
                    f'This sentence is too short ({candidate.word_count} words). '
                    f'Expand it to EXACTLY {target_length} words by adding:\n'
                    f'- Specific examples or details\n'
                    f'- Qualifications or context\n'
                    f'- Related consequences or implications\n\n'
                    f'Original: "{candidate.text}"\n\n'
                    f'Write ONLY the expanded {target_length}-word sentence:'
                )

        elif candidate.transition_fitness < 0.7:
            # Transition issue
            if transition_word and not candidate.has_transition:
                mutation_prompt = (
                    f'Rewrite this sentence to start with "{transition_word.capitalize()}":\n'
                    f'"{candidate.text}"\n'
                    f'Write ONLY the rewritten sentence:'
                )
            elif not transition_word and candidate.has_transition:
                mutation_prompt = (
                    f'Rewrite this sentence WITHOUT any transition word at the start:\n'
                    f'"{candidate.text}"\n'
                    f'Start directly with content. Write ONLY the rewritten sentence:'
                )

        elif candidate.vocabulary_fitness < 0.6:
            # Vocabulary doesn't match - simplify
            mutation_prompt = (
                f'Rewrite using simpler, more direct words:\n'
                f'"{candidate.text}"\n'
                f'Avoid fancy vocabulary. Write ONLY the rewritten sentence:'
            )

        else:
            # General improvement
            mutation_prompt = (
                f'Improve this sentence while keeping its meaning:\n'
                f'"{candidate.text}"\n'
                f'Target: {target_length} words. Write ONLY the improved sentence:'
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

        Uses RST nucleus-satellite structure to maintain coherence.
        Satellites (examples, evidence) reference their parent nuclei.

        Args:
            propositions: List of propositions to express.
            rst_info: Optional RST info for each proposition with keys:
                - role: "nucleus" or "satellite"
                - relation: type of relation (example, evidence, contrast, etc.)
                - parent_idx: index of parent nucleus for satellites
                - entities: list of entities in the proposition

        Returns:
            Generated paragraph text.
        """
        if not propositions:
            return ""

        state = GenerationState(
            target_burstiness=self.profile.length_profile.burstiness
        )
        sentences = []
        sentence_position = 0

        # Track generated sentences by proposition index for back-references
        generated_by_prop_idx = {}
        # Track entity mentions for coherent references
        mentioned_entities = set()

        for prop_idx, proposition in enumerate(propositions):
            # Get RST info for this proposition
            if rst_info and prop_idx < len(rst_info):
                rst = rst_info[prop_idx]
                role = rst.get("role", "nucleus")
                relation = rst.get("relation", "none")
                parent_idx = rst.get("parent_idx")
                prop_entities = rst.get("entities", [])
            else:
                role = "nucleus"
                relation = "none"
                parent_idx = None
                prop_entities = []

            # Set discourse relation from RST
            if relation != "none" and sentence_position > 0:
                # Map RST relations to discourse relations
                relation_map = {
                    "example": "elaboration",
                    "evidence": "cause",
                    "elaboration": "elaboration",
                    "contrast": "contrast",
                    "cause": "cause",
                    "condition": "cause",
                }
                state.required_discourse_relation = relation_map.get(relation, "continuation")
            else:
                state.required_discourse_relation = None

            # Build context hint for satellites that reference nuclei
            context_hint = ""
            if role == "satellite" and parent_idx is not None and parent_idx in generated_by_prop_idx:
                parent_sentence = generated_by_prop_idx[parent_idx]
                # Tell LLM to connect back to the parent claim
                if relation == "example":
                    context_hint = f"This is an EXAMPLE supporting: \"{parent_sentence[:60]}...\". Connect it clearly."
                elif relation == "evidence":
                    context_hint = f"This provides EVIDENCE for: \"{parent_sentence[:60]}...\". Reference the claim."
                elif relation == "elaboration":
                    context_hint = f"This ELABORATES on: \"{parent_sentence[:60]}...\". Expand naturally."

            # Add entity context for coherent references
            if mentioned_entities and prop_entities:
                shared = mentioned_entities.intersection(set(prop_entities))
                if shared:
                    context_hint += f" Entities already mentioned: {', '.join(list(shared)[:3])}"

            # Generate the sentence with context
            sentence, state = self.generator.generate_sentence(
                proposition=proposition,
                state=state,
                position=sentence_position,
                style_hint=context_hint,
            )

            sentences.append(sentence)
            generated_by_prop_idx[prop_idx] = sentence
            sentence_position += 1

            # Update mentioned entities
            mentioned_entities.update(prop_entities)

        paragraph = " ".join(sentences)

        # Verify and log
        verification = self.verifier.verify_paragraph(paragraph)
        if not verification.is_acceptable:
            logger.warning(
                f"Paragraph verification: score={verification.overall_score:.2f}, "
                f"issues={verification.issues}"
            )

        return paragraph
