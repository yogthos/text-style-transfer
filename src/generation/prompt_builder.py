"""Prompt building for sentence generation."""

import re
from collections import Counter
from dataclasses import dataclass
from typing import List, Optional, Dict, TYPE_CHECKING, Tuple

from ..models.plan import SentencePlan, SentenceNode, TransitionType
from ..models.graph import SemanticGraph
from ..models.style import StyleProfile
from ..ingestion.context_analyzer import GlobalContext, ParagraphContext
from ..utils.logging import get_logger

if TYPE_CHECKING:
    from ..corpus.indexer import CorpusIndexer

logger = get_logger(__name__)


# Default transition word suggestions (fallback when author has none)
DEFAULT_TRANSITION_WORDS = {
    TransitionType.NONE: [],
    TransitionType.CAUSAL: [
        "therefore", "thus", "consequently", "as a result",
        "hence", "accordingly", "for this reason"
    ],
    TransitionType.ADVERSATIVE: [
        "however", "but", "yet", "nevertheless", "nonetheless",
        "on the other hand", "conversely", "in contrast"
    ],
    TransitionType.ADDITIVE: [
        "moreover", "furthermore", "additionally", "also",
        "in addition", "besides", "what's more"
    ],
    TransitionType.TEMPORAL: [
        "then", "next", "subsequently", "afterward",
        "finally", "meanwhile", "previously"
    ],
}

# Map TransitionType to author profile transition categories
TRANSITION_TYPE_TO_CATEGORY = {
    TransitionType.NONE: None,
    TransitionType.CAUSAL: "causal",
    TransitionType.ADVERSATIVE: "adversative",
    TransitionType.ADDITIVE: "additive",
    TransitionType.TEMPORAL: "temporal",
}


@dataclass
class GenerationPrompt:
    """A complete prompt for generation."""
    system_prompt: str
    user_prompt: str

    def to_messages(self) -> List[Dict[str, str]]:
        """Convert to message format for LLM."""
        return [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": self.user_prompt},
        ]


class PromptBuilder:
    """Builds prompts for sentence generation.

    Creates structured prompts that guide the LLM to:
    1. Express specific propositions
    2. Match target sentence lengths
    3. Use appropriate transitions (from author's vocabulary)
    4. Follow the target style
    5. Use pronouns for previously introduced entities
    6. Follow style examples from corpus (when indexer available)
    """

    def __init__(
        self,
        global_context: GlobalContext,
        style_profile: Optional[StyleProfile] = None,
        indexer: Optional["CorpusIndexer"] = None
    ):
        """Initialize prompt builder.

        Args:
            global_context: Document-level context.
            style_profile: Target author's style profile (for author-specific transitions).
            indexer: Optional corpus indexer for retrieving style examples.
        """
        self.global_context = global_context
        self.style_profile = style_profile
        self.indexer = indexer
        self._introduced_entities: set = set()  # Entities already introduced in output
        self._used_transitions: Counter = Counter()  # Track transition word usage
        self._used_openers: Counter = Counter()  # Track sentence opener usage
        self._sentence_lengths: List[int] = []  # Track sentence lengths for burstiness
        self._total_sentences: int = 0  # Track sentence count for frequency calculations
        self._author_transition_dist = self._build_author_transition_distribution()

    def _build_author_transition_distribution(self) -> Dict[str, Dict[str, float]]:
        """Build normalized transition distribution from author's profile.

        Returns:
            Dict mapping category to dict of word -> target proportion.
        """
        if not self.style_profile or not self.style_profile.primary_author:
            return {}

        author = self.style_profile.primary_author
        distribution = {}

        for category, word_freqs in author.transitions.items():
            if not word_freqs:
                continue

            # Calculate total for this category
            total = sum(freq for _, freq in word_freqs)
            if total == 0:
                continue

            # Normalize to proportions (case-insensitive)
            cat_dist = {}
            for word, freq in word_freqs:
                word_lower = word.lower()
                if word_lower in cat_dist:
                    cat_dist[word_lower] += freq / total
                else:
                    cat_dist[word_lower] = freq / total

            distribution[category] = cat_dist

        return distribution

    def reset_tracking(self) -> None:
        """Reset all tracking for a new document."""
        self._introduced_entities = set()
        self._used_transitions = Counter()
        self._used_openers = Counter()
        self._sentence_lengths = []
        self._total_sentences = 0

    def reset_entity_tracking(self) -> None:
        """Reset entity tracking for a new paragraph."""
        self._introduced_entities = set()

    def register_introduced_entities(self, entities: List[str]) -> None:
        """Register entities that have been introduced in generated text.

        Args:
            entities: List of entity names/strings.
        """
        for entity in entities:
            self._introduced_entities.add(entity.lower().strip())

    def get_reference_guidance(self, propositions) -> str:
        """Generate guidance for entity references based on what's already introduced.

        Args:
            propositions: List of PropositionNode for current sentence.

        Returns:
            Guidance string for the LLM about entity references.
        """
        if not self._introduced_entities:
            return ""

        # Find entities in current propositions that were already introduced
        current_entities = set()
        for prop in propositions:
            for entity in prop.entities:
                current_entities.add(entity.lower().strip())

        # Check for overlap
        repeated = current_entities & self._introduced_entities
        if not repeated:
            return ""

        # Generate guidance
        repeated_list = list(repeated)[:3]  # Limit to 3 for readability
        return (
            f"\nENTITY REFERENCE NOTE: The following entities were already introduced: "
            f"{', '.join(repeated_list)}. "
            f"Use pronouns (it, they, this, etc.) or shortened references instead of repeating full names."
        )

    def register_generated_sentence(self, sentence: str) -> None:
        """Register a generated sentence to track transitions, openers, and lengths.

        Args:
            sentence: The generated sentence text.
        """
        self._total_sentences += 1

        # Track sentence length
        word_count = len(sentence.split())
        self._sentence_lengths.append(word_count)

        # Track sentence opener (first word)
        words = sentence.strip().split()
        if words:
            opener = words[0].strip('"\'"(')
            self._used_openers[opener] += 1

        # Check for transition words in all categories
        sentence_lower = sentence.lower()

        for category, word_dist in self._author_transition_dist.items():
            for word in word_dist.keys():
                # Use word boundary matching
                pattern = r'\b' + re.escape(word) + r'\b'
                matches = re.findall(pattern, sentence_lower)
                if matches:
                    self._used_transitions[word] += len(matches)

    def get_transition_balance_guidance(self, transition_type: TransitionType) -> str:
        """Get guidance on which transitions to prefer based on current usage balance.

        Args:
            transition_type: The type of transition needed.

        Returns:
            Guidance string suggesting under-used transitions.
        """
        if transition_type == TransitionType.NONE:
            return ""

        # Map transition type to category
        type_to_category = {
            TransitionType.CAUSAL: "causal",
            TransitionType.ADVERSATIVE: "adversative",
            TransitionType.ADDITIVE: "additive",
            TransitionType.TEMPORAL: "temporal",
        }

        category = type_to_category.get(transition_type)
        if not category or category not in self._author_transition_dist:
            return ""

        target_dist = self._author_transition_dist[category]
        if not target_dist:
            return ""

        # Calculate current usage proportions
        total_used = sum(self._used_transitions.get(w, 0) for w in target_dist.keys())

        if total_used == 0:
            # No transitions used yet - suggest the author's most common one
            top_word = max(target_dist.items(), key=lambda x: x[1])[0]
            return f"(author prefers '{top_word}' for this connection type)"

        # Find under-used transitions (current usage < target proportion)
        under_used = []
        over_used = []

        for word, target_prop in target_dist.items():
            current_count = self._used_transitions.get(word, 0)
            current_prop = current_count / total_used if total_used > 0 else 0

            # Significant deviation from target
            if target_prop > 0.1 and current_prop < target_prop * 0.5:
                under_used.append((word, target_prop))
            elif current_prop > target_prop * 2 and current_count > 2:
                over_used.append(word)

        if not under_used and not over_used:
            return ""

        guidance_parts = []

        if under_used:
            # Sort by target proportion (suggest most characteristic first)
            under_used.sort(key=lambda x: -x[1])
            suggestions = [w for w, _ in under_used[:2]]
            guidance_parts.append(f"consider '{suggestions[0]}'" if len(suggestions) == 1
                                  else f"consider '{suggestions[0]}' or '{suggestions[1]}'")

        if over_used:
            guidance_parts.append(f"vary from '{over_used[0]}'")

        return f"({'; '.join(guidance_parts)})" if guidance_parts else ""

    def get_opener_diversity_guidance(self) -> str:
        """Get guidance to avoid repetitive sentence openers.

        Returns:
            Guidance string if an opener is overused, empty otherwise.
        """
        if not self._used_openers or self._total_sentences < 3:
            return ""

        # Find the most used opener
        most_common = self._used_openers.most_common(1)
        if not most_common:
            return ""

        top_opener, count = most_common[0]

        # If an opener is used more than 30% of sentences, suggest variety
        usage_ratio = count / self._total_sentences
        if usage_ratio > 0.3 and count >= 2:
            # Get author's openers as alternatives
            author_openers = self.global_context.author_openers[:5]
            alternatives = [o for o in author_openers if o.lower() != top_opener.lower()][:3]

            if alternatives:
                return f"(Vary your opener - avoid starting with '{top_opener}' again; try: {', '.join(alternatives)})"
            else:
                return f"(Vary your opener - avoid starting with '{top_opener}' again)"

        return ""

    def get_vocabulary_guidance(self, n_words: int = 10) -> str:
        """Get vocabulary guidance from author's profile.

        NOTE: We deliberately provide minimal vocabulary guidance here to avoid
        the LLM mechanically stuffing words. Style should come from examples.

        Args:
            n_words: Number of vocabulary words to include.

        Returns:
            Formatted vocabulary guidance string (minimal or empty).
        """
        # Deliberately return empty - vocabulary should come from style examples
        # and the style DNA in the system prompt, not from explicit word lists
        # that cause mechanical stuffing
        return ""

    def get_style_examples(
        self,
        query_text: str,
        role: Optional[str] = None,
        target_length: Optional[int] = None,
        n_examples: int = 3
    ) -> str:
        """Retrieve style examples from corpus to include in prompts.

        Args:
            query_text: Text to search for similar examples.
            role: Optional paragraph role filter (INTRO, BODY, CONCLUSION).
            target_length: Optional target sentence length for filtering.
            n_examples: Number of examples to retrieve.

        Returns:
            Formatted string with style examples, or empty if unavailable.
        """
        if not self.indexer or not self.style_profile:
            return ""

        try:
            author_name = self.style_profile.get_author_name()
            results = self.indexer.query_fragments(
                query_text=query_text,
                author=author_name,
                role=role,
                n_results=n_examples * 2  # Fetch extra to allow filtering
            )

            if not results:
                return ""

            # Filter by approximate length if specified
            if target_length:
                results = [
                    r for r in results
                    if abs(r.get("metadata", {}).get("avg_sentence_length", 0) - target_length) < target_length * 0.5
                ]

            # Take top n_examples
            results = results[:n_examples]

            if not results:
                return ""

            # Format examples
            examples = []
            for i, result in enumerate(results):
                text = result.get("text", "")
                # Truncate long paragraphs to first 2 sentences
                sentences = text.split(". ")
                if len(sentences) > 2:
                    text = ". ".join(sentences[:2]) + "."
                examples.append(f"{i+1}. \"{text}\"")

            if not examples:
                return ""

            return (
                f"\nSTYLE EXAMPLES (model your writing after these):\n"
                f"{chr(10).join(examples)}"
            )

        except Exception as e:
            logger.warning(f"Failed to retrieve style examples: {e}")
            return ""

    def get_transition_words(self, transition_type: TransitionType) -> List[str]:
        """Get transition words for a given type, preferring author's vocabulary.

        Args:
            transition_type: The type of transition needed.

        Returns:
            List of transition words, prioritizing author's usage.
        """
        if transition_type == TransitionType.NONE:
            return []

        # Try to get from author's profile first
        if self.style_profile and self.style_profile.primary_author:
            category = TRANSITION_TYPE_TO_CATEGORY.get(transition_type)
            if category:
                author_words = self.style_profile.primary_author.get_transition_words(
                    category, limit=5
                )
                if author_words:
                    return author_words

        # Fall back to defaults
        return DEFAULT_TRANSITION_WORDS.get(transition_type, [])

    def build_paragraph_prompt(
        self,
        plan: SentencePlan,
        paragraph_context: ParagraphContext,
        previous_sentences: Optional[List[str]] = None
    ) -> GenerationPrompt:
        """Build prompt for generating a full paragraph.

        Args:
            plan: Sentence plan for the paragraph.
            paragraph_context: Paragraph-specific context.
            previous_sentences: Previously generated sentences (for continuation).

        Returns:
            GenerationPrompt ready for LLM.
        """
        system_prompt = self._build_system_prompt()
        user_prompt = self._build_paragraph_user_prompt(
            plan, paragraph_context, previous_sentences
        )

        return GenerationPrompt(
            system_prompt=system_prompt,
            user_prompt=user_prompt
        )

    def build_sentence_prompt(
        self,
        sentence_node: SentenceNode,
        plan: SentencePlan,
        previous_sentence: Optional[str] = None
    ) -> GenerationPrompt:
        """Build prompt for generating a single sentence.

        Args:
            sentence_node: The sentence to generate.
            plan: Full sentence plan (for context).
            previous_sentence: The immediately previous sentence.

        Returns:
            GenerationPrompt ready for LLM.
        """
        system_prompt = self._build_system_prompt()
        user_prompt = self._build_sentence_user_prompt(
            sentence_node, plan, previous_sentence
        )

        return GenerationPrompt(
            system_prompt=system_prompt,
            user_prompt=user_prompt
        )

    def build_revision_prompt(
        self,
        original_sentence: str,
        feedback: str,
        sentence_node: SentenceNode
    ) -> GenerationPrompt:
        """Build prompt for revising a sentence.

        Args:
            original_sentence: The sentence to revise.
            feedback: What needs improvement.
            sentence_node: Original sentence specifications.

        Returns:
            GenerationPrompt for revision.
        """
        system_prompt = self._build_system_prompt()

        user_prompt = f"""Revise this sentence based on the feedback:

ORIGINAL: {original_sentence}

FEEDBACK: {feedback}

REQUIREMENTS:
- Express these propositions: {sentence_node.get_proposition_text()}
- Target length: ~{sentence_node.target_length} words
- Role: {sentence_node.role.value}
- Required keywords: {', '.join(sentence_node.keywords) if sentence_node.keywords else 'none'}

Provide ONLY the revised sentence, nothing else."""

        return GenerationPrompt(
            system_prompt=system_prompt,
            user_prompt=user_prompt
        )

    def _build_system_prompt(self) -> str:
        """Build the system prompt from global context.

        Returns:
            System prompt string.
        """
        return self.global_context.to_system_prompt()

    def _build_paragraph_user_prompt(
        self,
        plan: SentencePlan,
        paragraph_context: ParagraphContext,
        previous_sentences: Optional[List[str]]
    ) -> str:
        """Build user prompt for paragraph generation.

        Args:
            plan: Sentence plan.
            paragraph_context: Paragraph context.
            previous_sentences: Previously generated sentences.

        Returns:
            User prompt string.
        """
        # Build sentence specifications
        sentence_specs = []
        for i, node in enumerate(plan.nodes):
            spec = self._format_sentence_spec(node, i + 1)
            sentence_specs.append(spec)

        specs_text = "\n".join(sentence_specs)

        # Build previous context if any
        if previous_sentences:
            prev_text = f"\nPrevious sentences in this paragraph:\n{chr(10).join(previous_sentences)}\n"
        else:
            prev_text = ""

        # Build the prompt
        prompt = f"""{paragraph_context.to_prompt_section()}
{prev_text}
SENTENCE SPECIFICATIONS:
{specs_text}

Generate the paragraph following these specifications exactly.
Each sentence should:
- Express the specified propositions completely
- Match the target word count closely (+/- 3 words)
- Use the indicated transition if specified
- Include all required keywords

OUTPUT FORMAT: Write only the paragraph text, one sentence per line."""

        return prompt

    def _build_sentence_user_prompt(
        self,
        sentence_node: SentenceNode,
        plan: SentencePlan,
        previous_sentence: Optional[str]
    ) -> str:
        """Build user prompt for single sentence generation.

        Args:
            sentence_node: Sentence to generate.
            plan: Full sentence plan.
            previous_sentence: Previous sentence for flow.

        Returns:
            User prompt string.
        """
        # Build context from previous sentence
        if previous_sentence:
            context = f"Previous sentence: {previous_sentence}"
        else:
            context = "This is the first sentence of the paragraph."

        # Transition guidance - subtle, not prescriptive
        transition_type = sentence_node.transition

        # Get balance guidance based on what's been used vs author's distribution
        balance_hint = self.get_transition_balance_guidance(transition_type)

        if transition_type.value == "CAUSAL":
            transition_hint = f"Show logical consequence {balance_hint}".strip()
        elif transition_type.value == "ADVERSATIVE":
            transition_hint = f"Show contrast or qualification {balance_hint}".strip()
        elif transition_type.value == "ADDITIVE":
            transition_hint = f"Build on the previous idea {balance_hint}".strip()
        elif transition_type.value == "TEMPORAL":
            transition_hint = f"Show sequence or temporal relationship {balance_hint}".strip()
        else:
            transition_hint = "Flow naturally from previous sentence"

        # Build individual propositions list for clarity
        props_list = self._format_propositions_list(sentence_node.propositions)

        # Get entity reference guidance (for avoiding repetition)
        reference_guidance = self.get_reference_guidance(sentence_node.propositions)

        # Get opener diversity guidance (avoid repetitive sentence starts)
        opener_guidance = self.get_opener_diversity_guidance()

        # Get style examples from corpus (RAG) - these are PRIMARY guidance
        query_text = sentence_node.get_proposition_text() if hasattr(sentence_node, 'get_proposition_text') else ""
        style_examples = self.get_style_examples(
            query_text=query_text,
            role=plan.paragraph_role if plan else None,
            target_length=sentence_node.target_length,
            n_examples=3  # More examples for better few-shot learning
        )

        # Combine guidance strings
        guidance = ""
        if transition_hint and transition_hint != "Flow naturally from previous sentence":
            guidance += f"\n{transition_hint}"
        if reference_guidance:
            guidance += reference_guidance
        if opener_guidance:
            guidance += f"\n{opener_guidance}"

        # Build the prompt - EXAMPLE-DRIVEN, not prescriptive
        skeleton_hint = self._get_skeleton_hint(sentence_node)

        if style_examples:
            # Few-shot style: examples first, then the task
            prompt = f"""Here are examples of the author's writing style:
{style_examples}

Now write a similar sentence that expresses: {sentence_node.get_proposition_text()}

{context}{guidance}
{skeleton_hint}"""
        else:
            # Fallback when no examples available
            prompt = f"""{context}

Express this idea: {sentence_node.get_proposition_text()}{guidance}
{skeleton_hint}"""

        return prompt.strip()

    def _format_sentence_spec(self, node: SentenceNode, index: int) -> str:
        """Format a sentence specification.

        Args:
            node: Sentence node.
            index: 1-based sentence index.

        Returns:
            Formatted specification string.
        """
        transition_words = self.get_transition_words(node.transition)
        trans_hint = f" (consider: {transition_words[0]})" if transition_words else ""

        keywords = f" [must include: {', '.join(node.keywords)}]" if node.keywords else ""

        return f"""Sentence {index}:
  - Role: {node.role.value}
  - Propositions: {node.get_proposition_text()}
  - Length: ~{node.target_length} words
  - Transition: {node.transition.value}{trans_hint}{keywords}"""

    def _format_propositions_list(self, propositions) -> str:
        """Format propositions as a numbered list.

        Args:
            propositions: List of PropositionNode.

        Returns:
            Formatted string with numbered propositions.
        """
        if not propositions:
            return "(none)"

        lines = []
        for i, prop in enumerate(propositions, 1):
            lines.append(f"{i}. {prop.text}")

        return "\n".join(lines)

    def _get_skeleton_hint(self, node: SentenceNode) -> str:
        """Get skeleton hint if available.

        Args:
            node: Sentence node.

        Returns:
            Skeleton hint string or empty.
        """
        if node.target_skeleton:
            return f"STRUCTURE HINT: Follow pattern {node.target_skeleton}"
        if node.style_template:
            return f"STYLE TEMPLATE: Model after: \"{node.style_template}\""
        return ""


class MultiSentencePromptBuilder(PromptBuilder):
    """Extended prompt builder for multi-pass generation."""

    def build_alternatives_prompt(
        self,
        sentence_node: SentenceNode,
        previous_sentence: Optional[str],
        num_alternatives: int = 3
    ) -> GenerationPrompt:
        """Build prompt to generate multiple sentence alternatives.

        Args:
            sentence_node: Sentence specification.
            previous_sentence: Previous sentence for context.
            num_alternatives: Number of alternatives to generate.

        Returns:
            GenerationPrompt for alternatives.
        """
        system_prompt = self._build_system_prompt()

        context = f"Previous: {previous_sentence}" if previous_sentence else "First sentence."

        user_prompt = f"""Generate {num_alternatives} different versions of a sentence.

CONTEXT: {context}

REQUIREMENTS:
- Express: {sentence_node.get_proposition_text()}
- Target length: ~{sentence_node.target_length} words
- Role: {sentence_node.role.value}
- Keywords: {', '.join(sentence_node.keywords) if sentence_node.keywords else 'none'}

OUTPUT: Number each alternative (1, 2, 3...), one per line. Make each stylistically distinct while preserving meaning."""

        return GenerationPrompt(
            system_prompt=system_prompt,
            user_prompt=user_prompt
        )

    def build_scoring_prompt(
        self,
        candidates: List[str],
        sentence_node: SentenceNode,
        style_exemplar: Optional[str] = None
    ) -> GenerationPrompt:
        """Build prompt to score candidate sentences.

        Args:
            candidates: Candidate sentences to score.
            sentence_node: Original specification.
            style_exemplar: Example sentence from target style.

        Returns:
            GenerationPrompt for scoring.
        """
        system_prompt = """You are a style critic evaluating sentence candidates.
Score each candidate on: semantic accuracy, style match, and flow.
Output format: candidate_number:score (0-10)"""

        candidates_text = "\n".join(
            f"{i+1}. {c}" for i, c in enumerate(candidates)
        )

        exemplar_text = f"\nStyle exemplar: \"{style_exemplar}\"" if style_exemplar else ""

        user_prompt = f"""Score these sentence candidates:

{candidates_text}

REQUIREMENTS:
- Must express: {sentence_node.get_proposition_text()}
- Target length: ~{sentence_node.target_length} words{exemplar_text}

Score each candidate (1-10), format: number:score
Example output:
1:8
2:6
3:9"""

        return GenerationPrompt(
            system_prompt=system_prompt,
            user_prompt=user_prompt
        )
