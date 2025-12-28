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
from typing import List, Dict, Optional, Callable, Tuple, TYPE_CHECKING
from collections import Counter
import numpy as np

from ..style.profile import AuthorStyleProfile
from ..style.verifier import StyleVerifier

if TYPE_CHECKING:
    from ..style.blender import GhostVector
from ..utils.nlp import get_nlp, split_into_sentences
from ..utils.logging import get_logger
from ..humanization.pattern_injector import PatternInjector, HumanizationConfig
from ..humanization.corpus_patterns import HumanPatterns
from ..rhetorical import (
    SentenceFunction,
    RhetoricalTemplateGenerator,
    PropositionMapper,
    FUNCTION_DESCRIPTIONS,
)
from .paragraph_graph import ParagraphGraph
from ..style.clause_extractor import ClausePatternExtractor, ClausePatternProfile

logger = get_logger(__name__)


# Instruction templates for prompt variety (research shows this prevents attention collapse)
# Each template frames the same task differently to avoid pattern lock-in
INSTRUCTION_TEMPLATES = [
    "Rewrite the following content using the author's style and rhythm:",
    "Express this idea in the voice and manner shown in the examples:",
    "Render this content matching the prose patterns above:",
    "Compose this content emulating the style of the examples:",
    "Transform this content to match the author's characteristic voice:",
    "Craft this content in the distinctive style demonstrated:",
    "Write this content with the rhythm and flow shown in examples:",
    "Convey this meaning using the author's syntactic patterns:",
    "Present this content in the manner of the style samples:",
    "Channel the author's voice to express this content:",
]

# System prompt variants for additional variety
SYSTEM_PROMPTS = [
    "You are an expert at emulating literary styles. Match the examples precisely.",
    "You are a skilled writer capturing authorial voice. Replicate the patterns shown.",
    "You are a style transfer specialist. Mirror the sentence rhythms exactly.",
    "You are a prose craftsman. Adopt the demonstrated writing patterns.",
    "You are channeling an author's voice. Match their style precisely.",
]


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
    content_fitness: float = 1.0  # Content preservation score
    style_fitness: float = 0.0    # Ghost vector similarity (for blending)

    # Diagnostic info
    word_count: int = 0
    target_length: int = 0
    has_transition: bool = False
    issues: List[str] = field(default_factory=list)

    # Whether style blending is active (affects weight distribution)
    _use_style_blending: bool = False

    @property
    def total_fitness(self) -> float:
        """Weighted fitness score - CONTENT is highest priority.

        When style blending is active, style_fitness gets 15% weight
        (taken proportionally from other components).
        """
        if self._use_style_blending and self.style_fitness > 0:
            # With blending: redistribute weights to include style
            return (
                self.content_fitness * 0.35 +     # Slightly reduced
                self.style_fitness * 0.15 +       # Ghost vector similarity
                self.length_fitness * 0.18 +
                self.transition_fitness * 0.12 +
                self.vocabulary_fitness * 0.12 +
                self.fluency_fitness * 0.08
            )
        else:
            # Standard weights without blending
            return (
                self.content_fitness * 0.40 +
                self.length_fitness * 0.20 +
                self.transition_fitness * 0.15 +
                self.vocabulary_fitness * 0.15 +
                self.fluency_fitness * 0.10
            )

    def __lt__(self, other):
        return self.total_fitness < other.total_fitness


@dataclass
class DocumentState:
    """State maintained across the entire document to prevent AI patterns.

    Key insight: LLMs generate text in isolation, leading to repetitive vocabulary
    and mechanical patterns. This class tracks document-wide context to give
    the LLM specific constraints that prevent these AI tells.
    """

    # Vocabulary tracking - prevent word repetition across document
    word_counts: Counter = field(default_factory=Counter)  # All content words used

    # Thresholds for "overused" - be conservative to avoid content loss
    overuse_threshold: int = 3  # Word appearing more than this is "overused"

    # Paragraph graph for document structure awareness
    paragraph_graph: Optional[ParagraphGraph] = None

    # Current paragraph index being generated
    current_paragraph_index: int = 0

    # Previous paragraph text for context injection
    previous_paragraph: str = ""

    # Sentence variety tracking across document
    recent_first_words: List[str] = field(default_factory=list)

    def get_overused_words(self, exclude_function_words: bool = True) -> List[str]:
        """Get words that have been used too frequently.

        These should be avoided in future generation to prevent
        the mechanical repetition that flags AI-generated text.
        """
        # Function words are fine to repeat (the, a, is, etc.)
        function_words = {
            'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
            'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
            'should', 'may', 'might', 'must', 'can', 'to', 'of', 'in', 'for',
            'on', 'with', 'at', 'by', 'from', 'as', 'into', 'through', 'during',
            'before', 'after', 'above', 'below', 'between', 'under', 'again',
            'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why',
            'how', 'all', 'each', 'every', 'both', 'few', 'more', 'most', 'other',
            'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so',
            'than', 'too', 'very', 'just', 'also', 'now', 'and', 'but', 'or',
            'if', 'because', 'although', 'while', 'that', 'which', 'this', 'these',
            'those', 'it', 'its', 'they', 'their', 'them', 'we', 'our', 'us',
            'you', 'your', 'he', 'she', 'him', 'her', 'his', 'i', 'me', 'my',
        }

        overused = []
        for word, count in self.word_counts.most_common():
            if count > self.overuse_threshold:
                if exclude_function_words and word.lower() in function_words:
                    continue
                overused.append(word)

        return overused[:15]  # Return top 15 most overused

    def update_from_text(self, text: str, nlp=None):
        """Update word counts from generated text.

        Args:
            text: Generated text to analyze
            nlp: Optional spaCy nlp object for better tokenization
        """
        if nlp:
            doc = nlp(text.lower())
            for token in doc:
                if token.is_alpha and not token.is_stop and len(token.text) > 3:
                    self.word_counts[token.text] += 1
        else:
            # Fallback: simple word splitting
            import re
            words = re.findall(r'\b[a-z]{4,}\b', text.lower())
            self.word_counts.update(words)

    def get_paragraph_context(self) -> str:
        """Get context string for current paragraph from the graph.

        This gives the LLM awareness of document structure,
        preventing isolated, mechanical generation.
        """
        if self.paragraph_graph:
            return self.paragraph_graph.get_context_for_paragraph(
                self.current_paragraph_index
            )
        return ""

    def get_transition_guidance(self) -> str:
        """Get guidance on transitioning into current paragraph."""
        if self.paragraph_graph:
            return self.paragraph_graph.get_transition_guidance(
                self.current_paragraph_index
            )
        return ""

    def add_first_word(self, word: str):
        """Track sentence first words for variety enforcement."""
        if word:
            self.recent_first_words.append(word.lower())
            # Keep only last 10 for memory efficiency
            if len(self.recent_first_words) > 10:
                self.recent_first_words = self.recent_first_words[-10:]

    def get_words_to_avoid_opening(self) -> List[str]:
        """Get first words to avoid for variety.

        Returns words used in last 3 sentences that aren't common function words.
        """
        ok_to_repeat = {'the', 'a', 'an', 'in', 'on', 'at', 'to', 'for', 'and', 'but', 'or', 'if', 'when', 'as'}
        return [w for w in self.recent_first_words[-3:] if w and w.lower() not in ok_to_repeat]


@dataclass
class GenerationState:
    """State maintained across sentence generation within a paragraph."""

    previous_sentences: List[str] = field(default_factory=list)
    previous_length_category: str = "medium"
    previous_structure_type: str = "simple"  # Track sentence structure
    previous_discourse_relation: str = "continuation"  # Track discourse flow
    used_transitions: Counter = field(default_factory=Counter)
    paragraph_lengths: List[int] = field(default_factory=list)
    target_burstiness: float = 0.5

    # Context tracking to avoid repetition
    used_phrases: set = field(default_factory=set)  # N-grams already used
    used_openings: set = field(default_factory=set)  # Sentence opening patterns (POS)
    recent_first_words: List[str] = field(default_factory=list)  # Actual first words (for variety)
    mentioned_concepts: List[str] = field(default_factory=list)  # For implicit refs
    key_nouns: List[str] = field(default_factory=list)  # Nouns that can be pronominalized

    # Source discourse relation for current sentence (to preserve logic)
    required_discourse_relation: Optional[str] = None

    # Rhetorical function for current sentence (from template)
    rhetorical_function: Optional[SentenceFunction] = None

    # Reference to document-level state for vocabulary control
    document_state: Optional[DocumentState] = None


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
        ghost_vector: Optional["GhostVector"] = None,
    ):
        """Initialize evolutionary generator.

        Args:
            profile: Target author's style profile.
            llm_generate: Function to call LLM.
            population_size: Number of candidates per generation.
            max_generations: Maximum evolution iterations.
            elite_count: Number of top candidates to preserve.
            mutation_rate: Probability of applying mutations.
            ghost_vector: Optional blended style target for SLERP-based scoring.
        """
        self.profile = profile
        self.llm_generate = llm_generate
        self.population_size = population_size
        self.max_generations = max_generations
        self.elite_count = elite_count
        self.mutation_rate = mutation_rate
        self.ghost_vector = ghost_vector  # For style blending

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

        # NEW: Store flow samples (150-400 word chunks) for multi-sentence style
        # These capture paragraph transitions where "style lives"
        self.flow_samples = profile.structure_profile.flow_samples or []
        if self.flow_samples:
            logger.info(f"Loaded {len(self.flow_samples)} flow samples for style transfer")

        # Extract clause patterns from structure samples for syntactic guidance
        # This gives the LLM structural skeletons to follow
        self.clause_pattern_profile = self._extract_clause_patterns(self.structure_samples)

        # Store discourse relation data for logic preservation
        self.discourse_samples = profile.discourse_profile.relation_samples
        self.discourse_transitions = profile.discourse_profile.relation_transitions
        self.discourse_connectives = profile.discourse_profile.relation_connectives

        # Initialize humanization pattern injector from profile
        self.pattern_injector = self._create_pattern_injector(profile.human_patterns)

        # Store vocabulary palette for prompt hints
        self.vocabulary_palette = profile.vocabulary_palette
        # Get author's STYLISTIC words for prompts - NOT content words
        # Use connectives and intensifiers (these are style, not content)
        self.author_stylistic_words = (
            self.vocabulary_palette.connectives[:8] +
            self.vocabulary_palette.intensifiers[:6]
        ) if self.vocabulary_palette.connectives else []
        # Get LLM-speak to avoid
        self.llm_avoid = list(self.vocabulary_palette.llm_replacements.keys())[:10] if self.vocabulary_palette.llm_replacements else []

        # Store rhetorical function profile for template generation
        self.function_profile = profile.function_profile
        self.function_samples = (
            self.function_profile.function_samples
            if self.function_profile else {}
        )

        # NEW: Store voice profile data for prompt injection
        self.assertiveness_profile = profile.assertiveness_profile
        self.rhetorical_profile = profile.rhetorical_profile
        self.phrase_patterns = profile.phrase_patterns

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

    def _extract_clause_patterns(
        self,
        structure_samples: Dict[str, List[str]]
    ) -> Optional[ClausePatternProfile]:
        """Extract clause patterns from structure samples.

        These patterns give the LLM syntactic skeletons to follow,
        improving sentence variety and reducing mechanical structures.
        """
        if not structure_samples:
            return None

        try:
            extractor = ClausePatternExtractor()
            profile = extractor.extract_profile(structure_samples)

            # Log extraction results
            for struct_type, patterns in profile.patterns_by_type.items():
                if patterns:
                    logger.debug(
                        f"[CLAUSE] Extracted {len(patterns)} {struct_type} patterns, "
                        f"avg depth={profile.avg_clause_depth.get(struct_type, 0):.1f}"
                    )

            return profile
        except Exception as e:
            logger.warning(f"Failed to extract clause patterns: {e}")
            return None

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
        """Extract key nouns and noun phrases that can be referred to with pronouns.

        Prioritizes:
        1. Main subject of the sentence (most important for anaphora)
        2. Noun phrases (compound nouns)
        3. Standalone nouns
        """
        doc = self.nlp(text)
        nouns = []

        # First, find the main subject - this is the primary candidate for "it/this"
        for token in doc:
            if token.dep_ in ('nsubj', 'nsubjpass') and token.head.dep_ == 'ROOT':
                # Get the full noun phrase if it's part of one
                if token.head.pos_ == 'VERB':
                    # This is the main subject
                    noun_phrase = ' '.join([t.text for t in token.subtree
                                           if t.pos_ in ('NOUN', 'PROPN', 'ADJ', 'DET')
                                           and not t.is_stop])
                    if noun_phrase and len(noun_phrase) > 3:
                        nouns.append(noun_phrase.lower())
                        break

        # Then extract other significant nouns
        for chunk in doc.noun_chunks:
            # Skip very short or stopword-only chunks
            content_words = [t for t in chunk if not t.is_stop and t.pos_ in ('NOUN', 'PROPN', 'ADJ')]
            if content_words:
                phrase = ' '.join([t.text for t in content_words])
                if phrase and len(phrase) > 3 and phrase.lower() not in nouns:
                    nouns.append(phrase.lower())

        # Fallback to simple nouns
        if not nouns:
            for token in doc:
                if token.pos_ == 'NOUN' and not token.is_stop and len(token.text) > 3:
                    nouns.append(token.text.lower())

        return nouns[:5]  # Limit to top 5

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

    def _extract_content_anchors_for_prompt(self, proposition: str) -> List[str]:
        """Extract content anchors that MUST be preserved in output.

        Content anchors include:
        - Named entities (places, people, organizations)
        - Numbers and statistics
        - Specific examples and concrete nouns
        - Technical terms
        """
        anchors = []
        doc = self.nlp(proposition)

        # Extract named entities
        for ent in doc.ents:
            if ent.label_ in ('GPE', 'LOC', 'ORG', 'PERSON', 'NORP', 'FAC', 'PRODUCT'):
                anchors.append(ent.text)

        # Extract numbers and percentages
        for token in doc:
            if token.like_num or token.pos_ == 'NUM':
                # Get the number with its unit if present
                num_phrase = token.text
                if token.i + 1 < len(doc):
                    next_tok = doc[token.i + 1]
                    if next_tok.pos_ == 'NOUN' or next_tok.text in ('percent', '%', 'million', 'billion'):
                        num_phrase = f"{token.text} {next_tok.text}"
                anchors.append(num_phrase)

        # Extract specific nouns (concrete things, not abstract concepts)
        concrete_nouns = []
        for token in doc:
            if token.pos_ == 'NOUN' and not token.is_stop:
                # Check if it's a concrete noun (has hyponym in WordNet or is capitalized)
                if token.text[0].isupper() or token.text in (
                    'smartphone', 'phone', 'watch', 'gear', 'lithium', 'cobalt',
                    'electricity', 'software', 'chip', 'silicon', 'device', 'tool',
                    'tree', 'government', 'gallery', 'object', 'waste', 'material',
                ):
                    concrete_nouns.append(token.text)

        # Add concrete nouns but avoid duplicates
        for noun in concrete_nouns[:3]:  # Limit to 3 to avoid prompt bloat
            if noun not in anchors:
                anchors.append(noun)

        # Extract quoted content
        import re
        quotes = re.findall(r'"([^"]+)"', proposition)
        anchors.extend(quotes)

        # Deduplicate while preserving order
        seen = set()
        unique_anchors = []
        for a in anchors:
            if a.lower() not in seen:
                seen.add(a.lower())
                unique_anchors.append(a)

        return unique_anchors[:5]  # Limit to 5 most important anchors

    def _build_voice_profile_section(self) -> str:
        """Build prompt section for author voice injection.

        Uses corpus-extracted assertiveness and rhetorical profiles
        to inject the author's characteristic voice patterns.
        """
        parts = []

        # Assertiveness profile
        if self.assertiveness_profile:
            ap = self.assertiveness_profile

            # Commitment level description
            if ap.average_commitment > 0.5:
                parts.append("VOICE: Write with STRONG CONVICTION. Use assertive, direct language.")
            elif ap.average_commitment < -0.3:
                parts.append("VOICE: Write with ACADEMIC CAUTION. Use hedged, qualified language.")

            # Top booster words (for assertive authors)
            if ap.author_boosters and ap.average_commitment > 0.3:
                top_boosters = list(ap.author_boosters.keys())[:5]
                parts.append(f"Use emphatic words like: {', '.join(top_boosters)}")

            # Top hedge words (for cautious authors)
            if ap.author_hedges and ap.average_commitment < -0.3:
                top_hedges = list(ap.author_hedges.keys())[:5]
                parts.append(f"Use qualifying words like: {', '.join(top_hedges)}")

            # Assertion patterns (sentence starters)
            if ap.assertion_patterns:
                sample_patterns = ap.assertion_patterns[:3]
                parts.append(f"Sentence patterns: \"{sample_patterns[0]}...\"")

        # Rhetorical profile
        if self.rhetorical_profile:
            rp = self.rhetorical_profile

            # High contrast usage
            if rp.contrast_frequency > 0.15:
                markers = rp.contrast_markers[:4] if rp.contrast_markers else ["but", "however"]
                parts.append(f"Use contrast markers freely: {', '.join(markers)}")

            # Dialectical patterns ("not X but Y")
            if rp.negation_affirmation_ratio > 0.02:
                parts.append('Use dialectical "not X, but Y" constructions when appropriate.')
                if rp.pattern_samples and "negation_affirmation" in rp.pattern_samples:
                    example = rp.pattern_samples["negation_affirmation"][0][:80]
                    parts.append(f'Example: "{example}..."')

            # Direct address
            if rp.direct_address_ratio > 0.03:
                parts.append("Address the reader directly with 'you must', 'we should', etc.")

            # Resolution markers
            if rp.resolution_frequency > 0.05 and rp.resolution_markers:
                res_markers = rp.resolution_markers[:3]
                parts.append(f"Use conclusion markers: {', '.join(res_markers)}")

        # Phrase patterns
        if self.phrase_patterns:
            pp = self.phrase_patterns

            # Characteristic constructions
            if pp.characteristic_constructions:
                example = pp.characteristic_constructions[0][:60]
                parts.append(f'Construction style: "{example}..."')

            # Emphasis patterns (dashes, etc.)
            if pp.emphasis_patterns:
                example = pp.emphasis_patterns[0][:60]
                parts.append(f'Emphasis style: "{example}..."')

        if not parts:
            return ""

        return "---\nAUTHOR VOICE:\n" + "\n".join(parts)

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
                    text, target_length, transition_word, state, proposition
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

        # Add clause pattern hint for syntactic guidance
        if hasattr(self, 'clause_pattern_profile') and self.clause_pattern_profile:
            from ..style.clause_extractor import get_clause_template_for_prompt
            clause_hint = get_clause_template_for_prompt(
                target_structure, self.clause_pattern_profile
            )
            if clause_hint:
                parts.append(f"Structure hint: {clause_hint}")

        parts.append("")
        parts.append("(Match the sentence RHYTHM and STRUCTURE above, not the topics.)")

        # NEW: Show a flow sample (multi-sentence chunk) for paragraph-level rhythm
        # This helps the LLM understand how sentences connect and flow
        if hasattr(self, 'flow_samples') and self.flow_samples:
            # Pick a random flow sample to show diverse paragraph styles
            flow_sample = random.choice(self.flow_samples)
            # Truncate if too long (show first ~200 words)
            flow_words = flow_sample.split()
            if len(flow_words) > 200:
                flow_sample = " ".join(flow_words[:200]) + "..."
            parts.append("")
            parts.append("PARAGRAPH FLOW EXAMPLE (notice how sentences connect):")
            parts.append(f'"""{flow_sample}"""')
            parts.append("(Capture this flowing, connected style - not choppy isolated sentences.)")

        # Add vocabulary hints - author's STYLISTIC words only (not content)
        if self.author_stylistic_words:
            stylistic_sample = ", ".join(self.author_stylistic_words[:8])
            parts.append(f"Style words: {stylistic_sample}")

        # Add LLM-speak to avoid
        if self.llm_avoid:
            avoid_sample = ", ".join(self.llm_avoid[:6])
            parts.append(f"Avoid LLM-speak: {avoid_sample}")

        # Add document-level overused words - AVOID these to prevent mechanical repetition
        # Distinguish between moderately overused (>3) and severely overused (>5)
        if state.document_state:
            overused = state.document_state.get_overused_words()
            if overused:
                # Check for severely overused words (>5 occurrences)
                severe = [w for w in overused[:8]
                         if state.document_state.word_counts.get(w, 0) > 5]
                moderate = [w for w in overused[:8] if w not in severe]

                if severe:
                    parts.append(f"AVOID these overused words: {', '.join(severe)}")
                if moderate:
                    parts.append(f"Prefer synonyms for: {', '.join(moderate)}")
                logger.debug(f"[VOCAB] Avoid: {severe}, Prefer synonyms: {moderate}")

        parts.append("---")

        # NEW: Add voice profile section (Sprint 2)
        voice_section = self._build_voice_profile_section()
        if voice_section:
            parts.append(voice_section)

        # PARAGRAPH CONTEXT: Give LLM awareness of document structure
        # This prevents isolated, mechanical generation by showing how this
        # paragraph fits into the broader argument
        if state.document_state:
            para_context = state.document_state.get_paragraph_context()
            if para_context:
                parts.append("")
                parts.append(para_context)

            transition_guidance = state.document_state.get_transition_guidance()
            if transition_guidance:
                parts.append(f"Approach: {transition_guidance}")

        # Context from previous sentences
        if state.previous_sentences:
            parts.append(f"Previous sentence: \"{state.previous_sentences[-1]}\"")

            # COHESION: Suggest anaphoric references to connect sentences naturally
            # This prevents the choppy, disconnected feel of AI-generated text
            if state.key_nouns and len(state.key_nouns) > 0:
                recent_nouns = state.key_nouns[-5:]  # Last 5 key nouns
                if recent_nouns:
                    parts.append(f"FLOW NATURALLY from above. You may reference: 'this {recent_nouns[-1]}', 'such {recent_nouns[0]}s', or 'it/they' for clarity")

            # Bridge phrases based on discourse relation
            discourse = state.required_discourse_relation or "continuation"
            if discourse == "continuation":
                parts.append("Connect smoothly - don't restart the topic abruptly")
            elif discourse == "elaboration":
                parts.append("Expand on the previous point - use 'this' or 'such' to refer back")
            elif discourse == "contrast":
                parts.append("Contrast with the previous point while acknowledging it")

        # SENTENCE VARIETY: Prevent repetitive sentence openings across DOCUMENT
        # This addresses the mechanical, predictable patterns that flag AI text
        # NOTE: Don't add transitions just for variety - the transition ratio is controlled elsewhere
        if state.document_state:
            avoid_words = state.document_state.get_words_to_avoid_opening()
            if avoid_words:
                parts.append(f"VARY YOUR OPENING (use different subject/topic, NOT transition words): avoid '{', '.join(set(avoid_words))}'")

        # The task - STRICT content preservation
        # Use rotating instruction templates to prevent attention collapse
        instruction = random.choice(INSTRUCTION_TEMPLATES)
        parts.append("")
        parts.append(instruction)
        parts.append(f'"{proposition}"')
        parts.append("(CRITICAL: preserve ALL content details)")

        # Extract and enforce content anchors
        content_anchors = self._extract_content_anchors_for_prompt(proposition)
        if content_anchors:
            parts.append("")
            parts.append(f"MUST INCLUDE these specific details: {', '.join(content_anchors)}")

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

        # RHETORICAL FUNCTION: Add function hint if available
        # This creates varied argumentative flow instead of monotonous statements
        rhetorical_function = state.rhetorical_function
        if rhetorical_function and rhetorical_function != SentenceFunction.CONTINUATION:
            func_desc = FUNCTION_DESCRIPTIONS.get(rhetorical_function, "")
            if func_desc:
                parts.append(f"Role: {rhetorical_function.value.upper()} - {func_desc}")
                # Add example from corpus if available
                if hasattr(self, 'function_samples') and self.function_samples:
                    samples = self.function_samples.get(rhetorical_function.value, [])
                    if samples:
                        example = random.choice(samples[:5])  # Pick from top 5
                        parts.append(f'Example: "{example[:80]}..."' if len(example) > 80 else f'Example: "{example}"')

        parts.append("")
        parts.append("Write ONE sentence expressing this. NEVER repeat words from 'Previous'.")
        parts.append("Sentence:")

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
            return self._evaluate_candidate(text, target_length, transition_word, state, proposition)

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
                        text, target_length, transition_word, state, proposition
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
                return self._evaluate_candidate(text, target_length, transition_word, state, proposition)
            except Exception as e:
                logger.warning(f"Mutation failed: {e}")

        return None

    def _extract_proposition_content(self, text: str) -> set:
        """Extract key content words from proposition that must be preserved."""
        doc = self.nlp(text.lower())
        content_words = set()
        for token in doc:
            # Keep nouns, verbs, adjectives (not stopwords, not short)
            if (token.pos_ in ('NOUN', 'VERB', 'ADJ', 'PROPN') and
                not token.is_stop and len(token.lemma_) > 3):
                content_words.add(token.lemma_)
        return content_words

    def _evaluate_candidate(
        self,
        text: str,
        target_length: int,
        transition_word: Optional[str],
        state: Optional[GenerationState] = None,
        proposition: Optional[str] = None,
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

        # CONTENT FITNESS - Check that key content words from proposition are preserved
        # This is CRITICAL for accurate translation
        if proposition:
            prop_content = self._extract_proposition_content(proposition)
            gen_content = self._extract_proposition_content(text)

            if prop_content:
                # Calculate overlap ratio
                preserved = prop_content & gen_content
                content_ratio = len(preserved) / len(prop_content)

                # Be lenient: some rephrasing is OK, but major content should be there
                # Score: 100% = 1.0, 50% = 0.5, 0% = 0.0
                candidate.content_fitness = content_ratio

                if content_ratio < 0.5:
                    candidate.issues.append(f"content_loss:{1-content_ratio:.0%}")
                    logger.debug(f"[CONTENT] Low preservation {content_ratio:.0%}: missing {prop_content - gen_content}")

        # Penalty for using severely overused words (>5 occurrences in document)
        if state and state.document_state:
            overused = state.document_state.get_overused_words()
            if overused:
                severe_overused = {w.lower() for w in overused
                                  if state.document_state.word_counts.get(w, 0) > 5}
                used_severe = text_words & severe_overused
                if used_severe:
                    # 0.7 penalty per severely overused word
                    penalty = 0.7 ** len(used_severe)
                    candidate.vocabulary_fitness *= penalty
                    candidate.issues.append(f"overused_vocab:{','.join(used_severe)}")

        # Fluency fitness (basic checks)
        candidate.fluency_fitness = 1.0
        if not text.strip():
            candidate.fluency_fitness = 0.0
        elif text[-1] not in '.!?':
            candidate.fluency_fitness *= 0.8
        elif text.count('"') % 2 != 0:
            candidate.fluency_fitness *= 0.9

        # COHESION bonus - reward sentences that use anaphoric references
        # when there are previous concepts to refer to
        if state and state.key_nouns and state.previous_sentences:
            text_lower = text.lower()
            # Check for cohesive devices
            cohesive_markers = ['this ', 'these ', 'such ', 'that ', 'those ']
            has_cohesive_device = any(marker in text_lower for marker in cohesive_markers)

            # Check for pronoun references (when not starting a paragraph)
            pronoun_refs = text_lower.startswith(('it ', 'they '))

            if has_cohesive_device or pronoun_refs:
                candidate.fluency_fitness *= 1.05  # 5% bonus for cohesion (reduced from 10%)
                logger.debug(f"[COHESION] Bonus for anaphoric reference in: {text[:50]}...")
            # NOTE: Removed orphan sentence penalty - it was hurting content preservation
            # Content fitness now handles this more appropriately

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

            # STRONG penalty for repeating the exact same first word across DOCUMENT
            # This prevents the mechanical "It... It... It..." pattern
            if state.document_state and state.document_state.recent_first_words:
                first_word = text.split()[0].lower() if text.split() else ""
                recent_words = state.document_state.recent_first_words
                ok_to_repeat = {'the', 'a', 'an', 'in', 'on', 'at', 'to', 'for', 'and', 'but'}
                if first_word and first_word not in ok_to_repeat:
                    # Check if same as immediately previous sentence
                    if recent_words and first_word == recent_words[-1]:
                        candidate.fluency_fitness *= 0.5  # Strong penalty
                        candidate.issues.append("repeated_first_word")
                    # Check if same as any of last 3 sentences
                    elif first_word in recent_words[-3:]:
                        candidate.fluency_fitness *= 0.75
                        candidate.issues.append("recent_first_word")

        # STYLE BLENDING: Score against ghost vector if available
        # This computes cosine similarity to the SLERP-interpolated style target
        if self.ghost_vector is not None:
            candidate.style_fitness = self.ghost_vector.score_text(text)
            candidate._use_style_blending = True
            logger.debug(f"[GHOST] Style fitness: {candidate.style_fitness:.3f} for: {text[:50]}...")

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

        # Extract first word for variety tracking at DOCUMENT level
        first_word = sentence.split()[0].lower() if sentence.split() else ""
        if state.document_state and first_word:
            state.document_state.add_first_word(first_word)

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
            recent_first_words=state.recent_first_words,  # Keep for backward compat, but use document_state
            mentioned_concepts=state.mentioned_concepts + new_nouns,
            key_nouns=state.key_nouns + new_nouns,
            # Preserve document state reference
            document_state=state.document_state,
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
        ghost_vector: Optional["GhostVector"] = None,
    ):
        """Initialize paragraph generator.

        Args:
            profile: Target author's style profile.
            llm_generate: Function to call LLM.
            population_size: Number of candidates per generation.
            max_generations: Maximum evolution iterations.
            ghost_vector: Optional SLERP-blended style target for Critic scoring.
        """
        self.ghost_vector = ghost_vector
        self.generator = EvolutionarySentenceGenerator(
            profile=profile,
            llm_generate=llm_generate,
            population_size=population_size,
            max_generations=max_generations,
            ghost_vector=ghost_vector,
        )
        self.profile = profile
        self.verifier = StyleVerifier(profile)

        # Initialize rhetorical template generator
        # Use profile's function data if available, otherwise use default transitions
        if profile.function_profile and profile.function_profile.function_transitions:
            func_profile = profile.function_profile
            transitions = func_profile.function_transitions
            initial_probs = func_profile.initial_function_probs
            logger.info("[PARA] Using corpus-derived rhetorical patterns")
        else:
            # Default transitions for natural argumentative flow
            transitions = {
                "claim": {"evidence": 0.3, "elaboration": 0.3, "contrast": 0.2, "continuation": 0.2},
                "evidence": {"claim": 0.3, "elaboration": 0.3, "resolution": 0.2, "continuation": 0.2},
                "question": {"claim": 0.4, "resolution": 0.4, "elaboration": 0.2},
                "contrast": {"resolution": 0.4, "claim": 0.3, "elaboration": 0.3},
                "resolution": {"elaboration": 0.4, "claim": 0.3, "continuation": 0.3},
                "setup": {"claim": 0.5, "question": 0.3, "evidence": 0.2},
            }
            initial_probs = {"claim": 0.4, "setup": 0.3, "question": 0.2, "evidence": 0.1}
            logger.info("[PARA] Using default rhetorical patterns")

        self.template_generator = RhetoricalTemplateGenerator(
            function_transitions=transitions,
            initial_function_probs=initial_probs,
            min_variety=2,
            max_same_consecutive=2,
        )
        self.proposition_mapper = PropositionMapper(llm_provider=None)

    def generate_paragraph(
        self,
        propositions: List[str],
        rst_info: Optional[List[dict]] = None,
        document_state: Optional[DocumentState] = None,
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
            document_state: Optional document-level state for vocabulary tracking.

        Returns:
            Generated paragraph text.
        """
        if not propositions:
            return ""

        logger.info(f"[PARA] Generating paragraph from {len(propositions)} propositions")

        state = GenerationState(
            target_burstiness=self.profile.length_profile.burstiness,
            document_state=document_state,
        )
        sentences = []
        sentence_position = 0

        # Track entity mentions for coherent references (implicit refs)
        mentioned_entities = set()
        mentioned_nouns = []

        # Group propositions based on target structure complexity
        prop_groups = self._group_propositions(propositions, rst_info, state)

        logger.info(f"[PARA] Grouped into {len(prop_groups)} sentence groups")

        # Generate rhetorical template and map propositions to slots
        template = self.template_generator.generate(len(prop_groups))
        group_props_text = [g["propositions"][0] for g in prop_groups if g["propositions"]]

        mapping_result = self.proposition_mapper.map_propositions(
            group_props_text, template
        )

        # Reorder groups if mapping suggests reordering
        if mapping_result.reordered and mapping_result.mappings:
            reordered_groups = [None] * len(prop_groups)
            for mapping in mapping_result.mappings:
                if mapping.slot_index < len(reordered_groups):
                    orig_idx = mapping.original_index
                    if orig_idx < len(prop_groups):
                        reordered_groups[mapping.slot_index] = prop_groups[orig_idx]

            prop_groups = [g for g in reordered_groups if g is not None]
            logger.info(f"[PARA] Reordered propositions for rhetorical flow")

        # Extract functions in order
        slot_functions = [
            template.slots[i].function if i < len(template.slots) else SentenceFunction.CONTINUATION
            for i in range(len(prop_groups))
        ]
        logger.info(f"[PARA] Rhetorical template: {[f.value for f in slot_functions]}")

        for group_idx, group in enumerate(prop_groups):
            group_props = group["propositions"]
            group_rst = group.get("rst_info", [])
            target_structure = group["target_structure"]
            citations = group.get("citations", [])

            logger.debug(
                f"[PARA] Group {group_idx}: {len(group_props)} props, "
                f"structure={target_structure}, citations={citations}"
            )

            # CONTENT PRESERVATION FIX: Use only 1 proposition per sentence
            # Grouping multiple propositions causes details to be lost
            effective_props = group_props[:1]  # Changed from [:2] to [:1]

            combined_content = effective_props[0] if effective_props else ""

            # Build context hint for implicit references
            context_hint = ""
            if mentioned_nouns and sentence_position > 0:
                # Suggest using pronouns for previously mentioned concepts
                recent_nouns = list(set(mentioned_nouns[-5:]))  # dedupe
                context_hint = f"[Use 'it/this/these' instead of repeating: {', '.join(recent_nouns)}]"

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

            # Set rhetorical function for this sentence
            state.rhetorical_function = (
                slot_functions[group_idx] if group_idx < len(slot_functions) else None
            )

            # NEW: Build semantic constraints from stance info
            semantic_hint = ""
            if group_rst:
                primary_rst = group_rst[0]
                stance = primary_rst.get("epistemic_stance")
                if stance and hasattr(stance, 'stance'):
                    if stance.stance == "appearance":
                        semantic_hint = "[PRESERVE APPEARANCE: Use 'seems', 'appears', 'conditioned to see as', NOT factual 'is']"
                    elif stance.stance == "conditional":
                        semantic_hint = "[PRESERVE CONDITIONAL: Keep 'if', 'unless', 'when' constructions]"
                    elif stance.stance == "hypothetical":
                        semantic_hint = "[PRESERVE HYPOTHETICAL: Keep 'might', 'could', 'may' modal verbs]"
                    if stance.is_negated:
                        semantic_hint += " [KEEP NEGATION]"

                # Check for logical relations
                relations = primary_rst.get("logical_relations", [])
                for rel in relations[:1]:  # First relation
                    if hasattr(rel, 'type'):
                        if rel.type == "contrast":
                            semantic_hint += " [PRESERVE CONTRAST with 'but'/'however']"
                        elif rel.type == "cause":
                            semantic_hint += " [PRESERVE CAUSATION with 'because'/'therefore']"

            if semantic_hint:
                context_hint = f"{context_hint} {semantic_hint}".strip()

            # Generate the sentence
            sentence, state = self.generator.generate_sentence(
                proposition=combined_content,
                state=state,
                position=sentence_position,
                style_hint=context_hint,
                total_sentences=len(prop_groups),
            )

            # Validate citation preservation (dedupe first)
            unique_citations = list(set(citations))
            if unique_citations:
                missing_citations = [c for c in unique_citations if c not in sentence]
                if missing_citations:
                    logger.warning(f"[PARA] Missing citations: {missing_citations}")
                    # Append missing citations
                    sentence = sentence.rstrip('.!?') + ' ' + ' '.join(missing_citations) + '.'
                # Also remove duplicate citations that LLM may have added
                for cit in unique_citations:
                    while sentence.count(cit) > 1:
                        # Remove first occurrence, keep last (typically at end)
                        sentence = sentence.replace(cit, '', 1).replace('  ', ' ')

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

            # CRITICAL: Update document-level vocabulary tracking
            # This prevents word repetition across the entire document
            if document_state:
                document_state.update_from_text(sentence, self.generator.nlp)

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

            # CONTENT PRESERVATION FIX: Always use capacity=1 to preserve all details
            # Grouping propositions causes content loss (specific examples, anchors)
            capacity = 1  # Changed from min(raw_capacity, 2) to 1

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
