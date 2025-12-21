"""Style translator for Pipeline 2.0.

This module translates semantic blueprints into styled text using
few-shot examples from a rhetorically-indexed style atlas.
"""

import json
import re
import time
import requests
from pathlib import Path
from typing import List, Tuple, Dict, Optional, Set, TYPE_CHECKING
from dataclasses import dataclass

if TYPE_CHECKING:
    import numpy as np
else:
    try:
        import numpy as np
    except ImportError:
        np = None
from src.ingestion.blueprint import SemanticBlueprint
from src.atlas.rhetoric import RhetoricalType
from src.generator.llm_provider import LLMProvider
from src.generator.llm_interface import clean_generated_text
from src.critic.judge import LLMJudge
from src.critic.scorer import SoftScorer
from src.validator.semantic_critic import SemanticCritic
from src.generator.mutation_operators import (
    get_operator, OP_SEMANTIC_INJECTION, OP_GRAMMAR_REPAIR, OP_STYLE_POLISH, OP_DYNAMIC_STYLE, OP_STRUCTURAL_CLONE
)
from src.analyzer.structuralizer import Structuralizer
from src.analysis.semantic_analyzer import PropositionExtractor


def _load_prompt_template(template_name: str) -> str:
    """Load a prompt template from the prompts directory.

    Args:
        template_name: Name of the template file (e.g., 'translator_system.md')

    Returns:
        Template content as string.
    """
    prompts_dir = Path(__file__).parent.parent.parent / "prompts"
    template_path = prompts_dir / template_name
    if not template_path.exists():
        raise FileNotFoundError(f"Prompt template not found: {template_path}")
    return template_path.read_text().strip()


def _load_positional_instructions() -> Dict[str, str]:
    """Load positional instructions from prompt file.

    Returns:
        Dictionary mapping position names to instruction text.
    """
    content = _load_prompt_template("translator_positional_instructions.md")
    instructions = {}
    current_section = None
    current_text = []

    for line in content.split('\n'):
        if line.startswith('## '):
            if current_section:
                instructions[current_section] = '\n'.join(current_text).strip()
            current_section = line[3:].strip()
            current_text = []
        elif current_section:
            current_text.append(line)

    if current_section:
        instructions[current_section] = '\n'.join(current_text).strip()

    return instructions


@dataclass
class ParagraphState:
    """Tracks the full state of a paragraph during batch evolution.

    Attributes:
        original_text: Original source text
        templates: Sentence-level templates (fixed)
        prop_map: Propositions assigned to each template (fixed, index-aligned)
        narrative_roles: Narrative role for each template slot (fixed, index-aligned)
        candidate_populations: For each slot, a list of candidate sentences (dynamic)
        best_sentences: Best passing sentence per slot (or None if none pass) (dynamic)
        locked_flags: True if slot is locked (score > threshold) (dynamic)
        feedback: Per-candidate feedback per slot
        generation_count: Track how many generations per slot
    """
    original_text: str
    templates: List[str]  # Sentence-level templates (fixed)
    prop_map: List[List[str]]  # Propositions assigned to each template (fixed, index-aligned)
    narrative_roles: List[str]  # Narrative role for each template slot (fixed, index-aligned)
    candidate_populations: List[List[str]]  # For each slot, a list of candidate sentences (dynamic)
    best_sentences: List[Optional[str]]  # Best passing sentence per slot (or None if none pass) (dynamic)
    locked_flags: List[bool]  # True if slot is locked (score > threshold) (dynamic)
    feedback: List[List[Optional[str]]]  # Per-candidate feedback per slot
    generation_count: List[int]  # Track how many generations per slot


class StyleTranslator:
    """Translates semantic blueprints into styled text using few-shot examples."""

    def __init__(self, config_path: str = "config.json"):
        """Initialize the translator.

        Args:
            config_path: Path to configuration file.
        """
        self.llm_provider = LLMProvider(config_path=config_path)
        self.config_path = config_path
        with open(config_path, 'r') as f:
            self.config = json.load(f)
        self.translator_config = self.config.get("translator", {})
        # Load positional instructions
        self.positional_instructions = _load_positional_instructions()
        # Cache spaCy model for blueprint completeness checks
        self._nlp_cache = None
        # Initialize soft scorer for fitness-based evolution
        self.soft_scorer = SoftScorer(config_path=config_path)
        # Initialize structuralizer for JIT structural templating
        self.structuralizer = Structuralizer(config_path=config_path)
        # Initialize proposition extractor for paragraph fusion
        self.proposition_extractor = PropositionExtractor(config_path=config_path)
        # Initialize structure extractor for template bleaching
        from src.analyzer.structure_extractor import StructureExtractor
        self.structure_extractor = StructureExtractor(config_path=config_path)
        # Load paragraph fusion config
        self.paragraph_fusion_config = self.config.get("paragraph_fusion", {})
        # Load LLM provider config (for retry settings)
        self.llm_provider_config = self.config.get("llm_provider", {})
        # Initialize skeleton cache for atlas memorization
        self._skeleton_cache = {}  # Key: (author, rhetorical_type, prop_count_bucket) -> (teacher_example, templates)

    def _get_nlp(self):
        """Get or load spaCy model for noun extraction."""
        if self._nlp_cache is None:
            try:
                from src.utils.nlp_manager import NLPManager
                self._nlp_cache = NLPManager.get_nlp()
            except (OSError, ImportError, RuntimeError):
                # If spaCy not available, return None (check will be skipped)
                self._nlp_cache = False
        return self._nlp_cache if self._nlp_cache is not False else None

    def _is_blueprint_incomplete(self, blueprint: SemanticBlueprint) -> bool:
        """Check if blueprint is semantically incomplete (missing critical nouns).

        Uses spaCy to extract and compare nouns from original_text vs blueprint.
        Uses lemmatization for matching (e.g., "Objects" matches "object").

        Args:
            blueprint: Semantic blueprint to check.

        Returns:
            True if blueprint is incomplete (missing > 50% of original nouns or empty SVO for long text).
        """
        nlp = self._get_nlp()
        if not nlp:
            # If spaCy not available, be conservative: assume complete
            return False

        if not blueprint.original_text or not blueprint.original_text.strip():
            return False

        # Extract nouns from original_text (lemmatized, non-stop words)
        original_doc = nlp(blueprint.original_text)
        original_nouns = set()
        for token in original_doc:
            if token.pos_ == "NOUN" and not token.is_stop:
                original_nouns.add(token.lemma_.lower())

        # If original has no nouns, can't determine completeness
        if not original_nouns:
            # Check if original has > 5 words but blueprint SVO is empty
            original_word_count = len(blueprint.original_text.split())
            if original_word_count > 5 and not blueprint.svo_triples:
                return True
            return False

        # Enhanced check: If blueprint has keywords but no SVO triples, mark as incomplete
        original_word_count = len(blueprint.original_text.split())
        if original_word_count > 5 and blueprint.core_keywords and not blueprint.svo_triples:
            return True

        # Extract nouns from blueprint (subjects + objects, lemmatized)
        blueprint_nouns = set()

        # Extract from subjects
        for subject in blueprint.get_subjects():
            if subject:
                subj_doc = nlp(subject.lower())
                for token in subj_doc:
                    if token.pos_ == "NOUN" and not token.is_stop:
                        blueprint_nouns.add(token.lemma_.lower())

        # Extract from objects
        for obj in blueprint.get_objects():
            if obj:
                obj_doc = nlp(obj.lower())
                for token in obj_doc:
                    if token.pos_ == "NOUN" and not token.is_stop:
                        blueprint_nouns.add(token.lemma_.lower())

        # Also extract from core_keywords (handles cases where standalone words are mis-tagged)
        # This is important because words like "object" can be tagged as VERB when standalone
        # Only use this as a fallback when we haven't found enough nouns from SVO triples
        if blueprint.core_keywords:
            for keyword in blueprint.core_keywords:
                if keyword:
                    keyword_doc = nlp(keyword.lower())
                    for token in keyword_doc:
                        # Accept if it's a NOUN
                        if token.pos_ == "NOUN" and not token.is_stop:
                            blueprint_nouns.add(token.lemma_.lower())
                        # Also accept if lemma matches an original noun (handles mis-tagging like "object" -> VERB)
                        # But only if we haven't found enough nouns yet (to avoid over-matching)
                        elif len(blueprint_nouns) == 0 and token.lemma_.lower() in original_nouns:
                            # Only use this fallback if we have zero nouns from SVO (handles lemmatization case)
                            blueprint_nouns.add(token.lemma_.lower())

        # Check: if blueprint has zero matching nouns, it's incomplete
        matching_nouns = original_nouns.intersection(blueprint_nouns)
        if len(matching_nouns) == 0 and len(original_nouns) > 0:
            return True

        # Check: if blueprint has < 50% of original nouns, it's incomplete
        if len(original_nouns) > 0:
            match_ratio = len(matching_nouns) / len(original_nouns)
            if match_ratio < 0.5:
                return True

        # Also check: if original has > 5 words but blueprint SVO is empty
        original_word_count = len(blueprint.original_text.split())
        if original_word_count > 5 and not blueprint.svo_triples:
            return True

        return False

    def _extract_multiple_skeletons(
        self,
        examples: List[str],
        blueprint: SemanticBlueprint,
        verbose: bool = False
    ) -> List[Tuple[str, str]]:
        """Extract skeletons with length filtering and deduplication.

        Implements "Wide Net, Strict Filter" strategy:
        1. Pre-filter by length (0.5x to 2.5x) before skeleton extraction
        2. Extract skeletons and apply complexity gate
        3. Deduplicate similar skeletons (preserving ChromaDB relevance order)
        4. Return top 5 distinct skeletons

        Args:
            examples: List of example sentences from ChromaDB (already sorted by relevance).
            blueprint: Semantic blueprint with original text.
            verbose: Whether to print debug information.

        Returns:
            List of (skeleton, source_example) tuples (top 5 distinct skeletons).
        """
        if not examples:
            return []

        input_len = len(blueprint.original_text.split())

        # Step 1: Length Filter (Pre-filter before skeleton extraction)
        # Adaptive filter: more lenient for very short sentences
        length_filtered = []
        for example in examples:
            example_len = len(example.split())

            # Adaptive length filter: more lenient for very short sentences
            if input_len < 5:
                # For short inputs, we WANT expansion.
                # Ensure min_len allows at least short valid sentences (e.g. 5 words)
                # Ensure max_len allows enough room (e.g. at least 15 words for very short inputs)
                # This handles edge case where input_len could be 0
                min_len = 5
                max_len = max(15, int(5.0 * input_len))
            else:
                # Standard filter: 0.5x to 2.5x length
                min_len = int(0.5 * input_len)
                max_len = int(2.5 * input_len)

            if min_len <= example_len <= max_len:
                length_filtered.append(example)

        if verbose:
            if input_len < 5:
                print(f"    Length filter: {len(length_filtered)}/{len(examples)} examples passed (adaptive: {min_len}-{max_len} words for {input_len}-word input)")
            else:
                print(f"    Length filter: {len(length_filtered)}/{len(examples)} examples passed (0.5x-2.5x length range)")

        # Step 2: Extract skeletons from length-filtered examples
        skeleton_candidates = []
        for example in length_filtered:
            try:
                # Pass input text for skeleton pruning (if single sentence, truncate to first sentence)
                skeleton = self.structuralizer.extract_skeleton(example, input_text=blueprint.original_text)
                if not skeleton:
                    continue

                skeleton_slots = self.structuralizer.count_skeleton_slots(skeleton)

                # Complexity gate: 0.5x to 3.0x slots
                if 0.5 * input_len <= skeleton_slots <= 3.0 * input_len:
                    skeleton_candidates.append((skeleton, example, skeleton_slots))
                elif skeleton_slots > input_len * 2:
                    # Too long: try to compress it
                    adapted_skeleton = self.structuralizer.adapt_skeleton(skeleton, input_len)
                    adapted_slots = self.structuralizer.count_skeleton_slots(adapted_skeleton)
                    # Check if adaptation helped
                    if 0.5 * input_len <= adapted_slots <= 3.0 * input_len:
                        skeleton_candidates.append((adapted_skeleton, example, adapted_slots))
                elif skeleton_slots < input_len * 0.5:
                    # Too short: try to expand it
                    adapted_skeleton = self.structuralizer.adapt_skeleton(skeleton, input_len)
                    adapted_slots = self.structuralizer.count_skeleton_slots(adapted_skeleton)
                    # Check if adaptation helped
                    if 0.5 * input_len <= adapted_slots <= 3.0 * input_len:
                        skeleton_candidates.append((adapted_skeleton, example, adapted_slots))
            except Exception:
                # Skip if skeleton extraction/adaptation fails
                continue

        if verbose:
            print(f"    Complexity gate: {len(skeleton_candidates)} skeletons passed (0.5x-3.0x slots)")

        # Step 2.5: Type Compatibility Filter
        # Detect input type once
        input_type = self._detect_sentence_type(blueprint.original_text)
        original_count = len(skeleton_candidates)
        compatible_candidates = []

        for skeleton, example, slots in skeleton_candidates:
            skeleton_type = self._detect_sentence_type(skeleton)

            # Logic:
            # 1. Exact match is always allowed (Decl -> Decl, Quest -> Quest, Cond -> Cond)
            # 2. Conditional -> Declarative is allowed (Common style expansion)
            # 3. Question -> Declarative is BANNED (Causes the logic errors seen in logs)

            is_compatible = False

            if input_type == skeleton_type:
                is_compatible = True
            elif input_type == "DECLARATIVE" and skeleton_type == "CONDITIONAL":
                is_compatible = True  # Allow expanding facts into conditionals
            # Note: Explicitly NOT allowing Declarative -> Question

            if is_compatible:
                compatible_candidates.append((skeleton, example, slots))
            elif verbose:
                print(f"    Type Mismatch: Dropped {skeleton_type} skeleton for {input_type} input.")

        # Fallback: If we filtered EVERYTHING, relax the constraint to avoid crashing
        if not compatible_candidates and verbose:
            print("    ⚠ Warning: All skeletons filtered by type. Relaxing filter.")
            # Keep empty list to trigger standard generation fallback

        if verbose:
            print(f"    Type filter: {len(compatible_candidates)}/{original_count} skeletons passed type compatibility")

        # Replace the list for the next steps
        skeleton_candidates = compatible_candidates

        # Step 3: Deduplication - Remove similar skeletons (preserving order)
        # Process in order to preserve ChromaDB relevance ranking
        unique_skeletons = []
        for skeleton, example, slots in skeleton_candidates:
            is_duplicate = False
            for existing_skeleton, _, _ in unique_skeletons:
                similarity = self._skeleton_similarity(skeleton, existing_skeleton)
                if similarity > 0.9:  # >90% similar
                    is_duplicate = True
                    if verbose:
                        print(f"    Deduplication: Skipping duplicate skeleton (similarity: {similarity:.2f})")
                    break

            if not is_duplicate:
                unique_skeletons.append((skeleton, example, slots))

        # Step 4: Preserve ChromaDB relevance ranking and return Top 5
        # DO NOT re-sort by length - the examples list is already sorted by relevance from ChromaDB
        # Re-sorting by length would destroy the rhetorical fit signal
        # Just take the first 5 unique skeletons that survived all filters

        if verbose:
            print(f"    Deduplication: {len(unique_skeletons)} unique skeletons from {len(skeleton_candidates)} candidates")
            print(f"    Selection: Returning top {min(5, len(unique_skeletons))} distinct skeletons (preserving ChromaDB relevance order)")

        # Return top 5 distinct skeletons (preserving ChromaDB relevance order)
        return [(skeleton, example) for skeleton, example, _ in unique_skeletons[:5]]

    def _skeleton_similarity(self, skeleton1: str, skeleton2: str) -> float:
        """Calculate similarity between two skeletons.

        Uses Jaccard similarity on normalized token sets.

        Args:
            skeleton1: First skeleton string.
            skeleton2: Second skeleton string.

        Returns:
            Similarity score 0.0-1.0 (1.0 = identical).
        """
        # Normalize skeletons (remove whitespace differences)
        s1_norm = re.sub(r'\s+', ' ', skeleton1.strip())
        s2_norm = re.sub(r'\s+', ' ', skeleton2.strip())

        if s1_norm == s2_norm:
            return 1.0

        # Calculate Jaccard similarity on tokens
        tokens1 = set(s1_norm.split())
        tokens2 = set(s2_norm.split())

        if not tokens1 or not tokens2:
            return 0.0

        intersection = len(tokens1 & tokens2)
        union = len(tokens1 | tokens2)

        return intersection / union if union > 0 else 0.0

    def _detect_voice(self, text: str) -> str:
        """Detect the voice/perspective of a text (1st, 2nd, 3rd person, or neutral).

        Args:
            text: Input text to analyze.

        Returns:
            "1st", "2nd", "3rd", or "neutral" based on pronoun usage.
        """
        if not text or not text.strip():
            return "neutral"

        import re
        text_lower = text.lower()

        # First-person markers (use word boundaries to avoid false matches)
        first_person_patterns = [
            r'\bi\b', r'\bme\b', r'\bmy\b', r'\bmine\b',
            r'\bwe\b', r'\bus\b', r'\bour\b', r'\bours\b'
        ]
        first_count = sum(len(re.findall(pattern, text_lower)) for pattern in first_person_patterns)

        # Second-person markers
        second_person_patterns = [
            r'\byou\b', r'\byour\b', r'\byours\b'
        ]
        second_count = sum(len(re.findall(pattern, text_lower)) for pattern in second_person_patterns)

        # Third-person markers (explicitly ignore "it" to avoid noise)
        third_person_patterns = [
            r'\bhe\b', r'\bhim\b', r'\bhis\b',
            r'\bshe\b', r'\bher\b', r'\bhers\b',
            r'\bthey\b', r'\bthem\b', r'\btheir\b', r'\btheirs\b'
        ]
        third_count = sum(len(re.findall(pattern, text_lower)) for pattern in third_person_patterns)

        # Return the category with the highest count
        counts = {
            "1st": first_count,
            "2nd": second_count,
            "3rd": third_count
        }

        max_count = max(counts.values())
        if max_count == 0:
            return "neutral"

        # Return the category with the highest count
        for voice, count in counts.items():
            if count == max_count:
                return voice

        return "neutral"

    def _detect_sentence_type(self, text: str) -> str:
        """Robustly detects if a sentence/skeleton is QUESTION, CONDITIONAL, or DECLARATIVE.

        Args:
            text: Sentence text or skeleton to analyze.

        Returns:
            "QUESTION", "DECLARATIVE", or "CONDITIONAL"
        """
        if not text or not text.strip():
            return "DECLARATIVE"

        text = text.strip().lower()

        # 1. Explicit Question: ends with ? or starts with question words
        # Check for question mark first (most definitive)
        if text.endswith("?"):
            return "QUESTION"

        # Check for question words at start (who, what, where, why, how)
        # Note: "when" is ambiguous, so handle it separately
        if text.startswith(("who ", "what ", "where ", "why ", "how ")):
            return "QUESTION"

        # "when" can be question or conditional - check for question pattern first
        # Questions: "when did/do/does/will/would/can/could/should..."
        # Conditionals: "when [subject] [verb]..." (no auxiliary)
        if text.startswith("when "):
            # Check if it's a question pattern (auxiliary verb after "when")
            if re.match(r"^when\s+(did|do|does|will|would|can|could|should|is|are|was|were|has|have|had)\s", text):
                return "QUESTION"
            # Otherwise treat as conditional
            return "CONDITIONAL"

        # 2. Conditional: Check for structure at start
        # Matches: "If X, Y" or "Unless X, Y" or "Provided that X, Y"
        if re.match(r"^(if|unless|provided that|should)\s", text):
            return "CONDITIONAL"

        return "DECLARATIVE"

    def _calculate_skeleton_adherence(self, candidate: str, skeleton: str) -> float:
        """Calculate skeleton adherence score using anchor word overlap.

        Extracts function words (anchor words) from skeleton and checks
        what percentage appear in candidate in roughly the same order.

        Also checks type compatibility - if skeleton type doesn't match candidate type,
        returns a lower score to penalize type mismatches.

        Args:
            candidate: Generated candidate text.
            skeleton: Skeleton template with placeholders.

        Returns:
            Adherence score 0.0-1.0 (matched_anchors / total_anchors).
        """
        if not candidate or not skeleton:
            return 0.0

        # Type compatibility check: if types don't match, penalize heavily
        skeleton_type = self._detect_sentence_type(skeleton)
        candidate_type = self._detect_sentence_type(candidate)

        if skeleton_type != candidate_type:
            # Type mismatch - return a low score to indicate poor adherence
            # This helps filter out candidates that don't match the skeleton's sentence type
            return 0.3  # Low score for type mismatch

        # Function words (anchor words) that should be preserved
        # These are structural words that appear in the skeleton
        function_words = {
            'the', 'a', 'an', 'of', 'in', 'on', 'at', 'to', 'for', 'with', 'by',
            'and', 'or', 'but', 'if', 'then', 'when', 'where', 'while', 'as',
            'from', 'into', 'onto', 'upon', 'over', 'under', 'above', 'below',
            'through', 'during', 'before', 'after', 'since', 'until', 'about',
            'against', 'between', 'among', 'within', 'without', 'across',
            'that', 'this', 'these', 'those', 'which', 'who', 'whom', 'whose',
            'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had',
            'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might',
            'must', 'can', 'not', 'no', 'nor', 'so', 'than', 'too', 'very', 'more',
            'most', 'some', 'any', 'all', 'each', 'every', 'both', 'either', 'neither'
        }

        # Extract anchor words from skeleton (non-placeholder words)
        # First, remove placeholders [NP], [VP], [ADJ] from skeleton
        skeleton_clean = re.sub(r'\[NP\]|\[VP\]|\[ADJ\]', '', skeleton, flags=re.IGNORECASE)
        skeleton_tokens = re.findall(r'\b\w+\b', skeleton_clean.lower())
        anchor_words = []
        anchor_positions = []

        for i, token in enumerate(skeleton_tokens):
            # Include function words
            if token in function_words:
                anchor_words.append(token)
                anchor_positions.append(i)

        if not anchor_words:
            # No anchor words found, return neutral score
            return 0.5

        # Extract tokens from candidate
        candidate_tokens = re.findall(r'\b\w+\b', candidate.lower())

        # Count matched anchors (with position tolerance ±2)
        matched_count = 0
        candidate_idx = 0

        for anchor_idx, anchor_word in enumerate(anchor_words):
            # Search for anchor word in candidate within position tolerance
            skeleton_pos = anchor_positions[anchor_idx]
            search_start = max(0, candidate_idx - 2)
            search_end = min(len(candidate_tokens), candidate_idx + 10)

            for i in range(search_start, search_end):
                if candidate_tokens[i] == anchor_word:
                    matched_count += 1
                    candidate_idx = i + 1  # Move forward
                    break

        adherence_score = matched_count / len(anchor_words) if anchor_words else 0.0
        return adherence_score

    def _calculate_style_density(self, candidate: str, style_lexicon: Optional[List[str]]) -> float:
        """Calculate style density score (ratio of style lexicon words in candidate).

        Args:
            candidate: Generated candidate text.
            style_lexicon: Optional list of style words to check for.

        Returns:
            Style density score 0.0-1.0 (count(output_words in lexicon) / total_words).
        """
        if not candidate or not style_lexicon:
            return 0.0

        # Tokenize candidate
        candidate_tokens = re.findall(r'\b\w+\b', candidate.lower())
        if not candidate_tokens:
            return 0.0

        # Create set of style lexicon words (lowercase)
        style_words_set = {word.lower().strip() for word in style_lexicon}

        # Count style words in candidate
        style_word_count = sum(1 for token in candidate_tokens if token in style_words_set)

        style_density = style_word_count / len(candidate_tokens)
        return style_density

    def _generate_batch(
        self,
        skeleton: str,
        blueprint: SemanticBlueprint,
        author_name: str,
        style_dna: str,
        rhetorical_type: RhetoricalType,
        style_lexicon: Optional[List[str]] = None,
        batch_size: int = 20,
        verbose: bool = False
    ) -> List[str]:
        """Generate batch of variants in a single API call.

        Args:
            skeleton: Structural skeleton template
            blueprint: Semantic blueprint with meaning
            style_lexicon: List of style words to prioritize
            batch_size: Number of variants to generate (default 20)
            verbose: Whether to print debug information

        Returns:
            List of generated variant strings
        """
        from src.generator.mutation_operators import BATCH_GENERATION_PROMPT

        # Format prompt with blueprint data
        subjects_list = blueprint.get_subjects()[:5] if blueprint.get_subjects() else []
        verbs_list = blueprint.get_verbs()[:5] if blueprint.get_verbs() else []
        objects_list = blueprint.get_objects()[:5] if blueprint.get_objects() else []

        subjects = ", ".join(subjects_list) if subjects_list else "None"
        verbs = ", ".join(verbs_list) if verbs_list else "None"
        objects = ", ".join(objects_list) if objects_list else "None"
        lexicon_text = ", ".join(style_lexicon[:20]) if style_lexicon else "None"

        # Get primary subject and verb for anchoring and mapping
        primary_subject = subjects_list[0] if subjects_list else "None"
        primary_verb = verbs_list[0] if verbs_list else "None"

        # Build keyword checklist from core keywords
        core_keywords = list(blueprint.core_keywords)[:10] if blueprint.core_keywords else []
        keywords_text = ", ".join(core_keywords) if core_keywords else "None"

        # TASK 3: Subject Anchoring - Pre-fill first [NP] slot with input subject
        anchored_skeleton = skeleton
        if primary_subject != "None" and "[NP]" in skeleton:
            # Replace first [NP] with the actual subject (bolded for emphasis)
            anchored_skeleton = skeleton.replace("[NP]", f"**{primary_subject}**", 1)
            if verbose:
                print(f"    🔒 Subject anchor: Injected '{primary_subject}' into first [NP] slot")
                print(f"    📐 Original skeleton: {skeleton[:80]}...")
                print(f"    📐 Anchored skeleton: {anchored_skeleton[:80]}...")

        prompt = BATCH_GENERATION_PROMPT.format(
            subjects=subjects,
            verbs=verbs,
            objects=objects,
            original_text=blueprint.original_text,
            skeleton=anchored_skeleton,  # Use anchored skeleton
            style_lexicon=lexicon_text,
            keywords=keywords_text,
            subject=primary_subject,
            verb=primary_verb
        )

        if verbose:
            print(f"    📋 Keyword checklist: {keywords_text}")
            print(f"    🗺️  Concept mapping: Subject '{primary_subject}' -> [NP], Verb '{primary_verb}' -> [VP]")

        # Call LLM with JSON mode enabled (high temperature for diversity)
        # Add retry logic with exponential backoff for timeout errors
        import time
        max_retries = self.llm_provider_config.get("max_retries", 3)
        retry_delay = self.llm_provider_config.get("retry_delay", 2)  # Start delay in seconds

        for attempt in range(max_retries):
            try:
                if verbose:
                    if attempt > 0:
                        print(f"    🔄 Retry attempt {attempt + 1}/{max_retries} (after {retry_delay}s delay)")
                    else:
                        print(f"    📤 Calling LLM for batch generation (batch_size={batch_size}, temp=0.8)")

                response = self.llm_provider.call(
                    system_prompt="You are a precision batch generator. Output ONLY valid JSON.",
                    user_prompt=prompt,
                    model_type="editor",
                    require_json=True,
                    temperature=0.8,  # High temp for diversity
                    max_tokens=self.translator_config.get("max_tokens", 500)
                )

                if verbose:
                    print(f"    📥 Received response ({len(response)} chars)")

                # Parse JSON response
                candidates = json.loads(response)
                if isinstance(candidates, list):
                    # Filter out empty strings and validate
                    candidates = [c.strip() for c in candidates if c and c.strip()]
                    if verbose:
                        print(f"    ✅ Parsed {len(candidates)} candidates from JSON")
                    return candidates[:batch_size]  # Ensure we don't exceed batch_size
                else:
                    if verbose:
                        print(f"    ⚠ Batch generation returned non-list, attempting extraction")
                    return self._extract_json_list(response)

            except json.JSONDecodeError as e:
                if verbose:
                    print(f"    ⚠ JSON decode error: {e}, attempting extraction")
                    print(f"    📝 Response preview: {response[:200] if 'response' in locals() else 'N/A'}...")
                # Fallback: try to extract JSON from response
                if 'response' in locals():
                    extracted = self._extract_json_list(response)
                    if verbose:
                        print(f"    {'✅' if extracted else '✗'} Extracted {len(extracted)} candidates from text")
                    return extracted
                return []

            except (RuntimeError, requests.exceptions.RequestException, requests.exceptions.Timeout, requests.exceptions.ConnectionError) as e:
                error_str = str(e)
                is_timeout = "timeout" in error_str.lower() or "timed out" in error_str.lower() or "Read timed out" in error_str

                if attempt < max_retries - 1 and is_timeout:
                    # Retry on timeout with exponential backoff
                    if verbose:
                        print(f"    ⚠ Timeout error (attempt {attempt + 1}/{max_retries}): {error_str[:100]}...")
                        print(f"    ⏳ Waiting {retry_delay}s before retry...")
                    time.sleep(retry_delay)
                    retry_delay *= 2  # Exponential backoff
                    continue
                else:
                    # Final attempt failed or non-timeout error - try smaller batches
                    if verbose:
                        print(f"    ⚠ Batch generation failed after {attempt + 1} attempts: {error_str[:100]}...")
                        if is_timeout:
                            print(f"    🔄 Falling back to smaller batch generation...")

                    # Fallback: Generate smaller batches
                    return self._generate_batch_fallback(
                        skeleton, blueprint, author_name, style_dna, rhetorical_type,
                        style_lexicon, batch_size, verbose, primary_subject, primary_verb,
                        subjects, verbs, objects, keywords_text, lexicon_text
                    )

            except Exception as e:
                if verbose:
                    print(f"    ✗ Batch generation failed: {e}")
                    import traceback
                    traceback.print_exc()
                # Try fallback on any other error
                if attempt == max_retries - 1:
                    return self._generate_batch_fallback(
                        skeleton, blueprint, author_name, style_dna, rhetorical_type,
                        style_lexicon, batch_size, verbose, primary_subject, primary_verb,
                        subjects, verbs, objects, keywords_text, lexicon_text
                    )
                continue

        # All retries exhausted
        if verbose:
            print(f"    ✗ All retry attempts exhausted")
        return []

    def _extract_json_list(self, text: str) -> List[str]:
        """Extract JSON list from text that may contain extra content.

        Tries to find JSON array pattern in the response.

        Args:
            text: Text that may contain a JSON array

        Returns:
            List of strings extracted from JSON array, or empty list if extraction fails
        """
        # Try to find JSON array pattern
        json_match = re.search(r'\[.*\]', text, re.DOTALL)
        if json_match:
            try:
                candidates = json.loads(json_match.group())
                if isinstance(candidates, list):
                    return [c.strip() for c in candidates if c and c.strip()]
            except json.JSONDecodeError:
                pass

        # Fallback: return empty list
        return []

    def _generate_batch_fallback(
        self,
        skeleton: str,
        blueprint: SemanticBlueprint,
        author_name: str,
        style_dna: str,
        rhetorical_type: RhetoricalType,
        style_lexicon: Optional[List[str]],
        batch_size: int,
        verbose: bool,
        primary_subject: str,
        primary_verb: str,
        subjects: str,
        verbs: str,
        objects: str,
        keywords_text: str,
        lexicon_text: str
    ) -> List[str]:
        """Fallback: Generate smaller batches if full batch times out.

        Generates batches of 5 candidates at a time until we reach batch_size.
        """
        from src.generator.mutation_operators import BATCH_GENERATION_PROMPT

        if verbose:
            print(f"    🔄 Fallback: Generating in smaller batches of 5...")

        all_candidates = []
        chunk_size = 5
        num_chunks = (batch_size + chunk_size - 1) // chunk_size  # Ceiling division

        # Update prompt for smaller batch
        fallback_prompt_template = BATCH_GENERATION_PROMPT.replace("20 distinct variations", f"{chunk_size} distinct variations")
        fallback_prompt_template = fallback_prompt_template.replace(
            "- **Candidates 1-5", f"- **Candidates 1-{chunk_size}"
        )
        # Remove the other candidate category instructions for smaller batches
        fallback_prompt_template = re.sub(
            r'- \*\*Candidates \d+-\d+.*?\n',
            '',
            fallback_prompt_template
        )

        # Subject anchoring
        anchored_skeleton = skeleton
        if primary_subject != "None" and "[NP]" in skeleton:
            anchored_skeleton = skeleton.replace("[NP]", f"**{primary_subject}**", 1)

        prompt = fallback_prompt_template.format(
            subjects=subjects,
            verbs=verbs,
            objects=objects,
            original_text=blueprint.original_text,
            skeleton=anchored_skeleton,
            style_lexicon=lexicon_text,
            keywords=keywords_text,
            subject=primary_subject,
            verb=primary_verb
        )

        for chunk_idx in range(num_chunks):
            try:
                if verbose:
                    print(f"      📦 Generating chunk {chunk_idx + 1}/{num_chunks} ({chunk_size} candidates)...")

                response = self.llm_provider.call(
                    system_prompt="You are a precision batch generator. Output ONLY valid JSON.",
                    user_prompt=prompt,
                    model_type="editor",
                    require_json=True,
                    temperature=0.8,
                    max_tokens=self.translator_config.get("max_tokens", 300)  # Smaller for chunk
                )

                candidates = json.loads(response)
                if isinstance(candidates, list):
                    candidates = [c.strip() for c in candidates if c and c.strip()]
                    all_candidates.extend(candidates)
                    if verbose:
                        print(f"      ✅ Chunk {chunk_idx + 1}: Got {len(candidates)} candidates")
                else:
                    extracted = self._extract_json_list(response)
                    all_candidates.extend(extracted)
                    if verbose:
                        print(f"      ✅ Chunk {chunk_idx + 1}: Extracted {len(extracted)} candidates")

                # Stop if we have enough
                if len(all_candidates) >= batch_size:
                    break

                # Small delay between chunks to avoid rate limiting
                if chunk_idx < num_chunks - 1:
                    time.sleep(1)

            except Exception as e:
                if verbose:
                    print(f"      ⚠ Chunk {chunk_idx + 1} failed: {e}")
                # Continue with next chunk
                continue

        if verbose:
            print(f"    ✅ Fallback complete: Generated {len(all_candidates)} total candidates")

        return all_candidates[:batch_size]

    def _run_arena(
        self,
        candidates: List[Dict[str, any]],  # List of {"text": str, "skeleton": str, ...}
        blueprint: SemanticBlueprint,
        style_dna_dict: Optional[Dict[str, any]] = None,
        verbose: bool = False
    ) -> List[Dict[str, any]]:
        """Evaluate all candidates and return survivors.

        Filters:
        1. Structure Filter: Adherence to skeleton (kill if < 0.7)
        2. Logic Filter: Logic failure check (kill if logic_fail=True)
        3. Meaning Filter: Keyword presence (kill if < 50%)

        Returns:
            Top K survivors sorted by style score
        """
        from src.validator.semantic_critic import SemanticCritic

        # Load config
        evolutionary_config = self.config.get("evolutionary", {})
        top_k = evolutionary_config.get("top_k_parents", 5)
        min_keyword_presence = evolutionary_config.get("min_keyword_presence", 0.5)

        # Extract style lexicon
        style_lexicon = None
        if style_dna_dict and isinstance(style_dna_dict, dict):
            style_lexicon = style_dna_dict.get("lexicon")

        # Initialize critic
        critic = SemanticCritic(config_path=self.config_path)

        survivors = []

        if verbose:
            print(f"  🏟️  Arena: Evaluating {len(candidates)} candidates")
            print(f"    📐 Skeleton structure: {candidates[0].get('skeleton', 'N/A')[:100] if candidates else 'N/A'}...")
            print(f"    🎯 Target keywords: {', '.join(list(blueprint.core_keywords)[:5]) if blueprint.core_keywords else 'None'}")

        for idx, candidate_data in enumerate(candidates):
            candidate_text = candidate_data.get("text", "")
            skeleton = candidate_data.get("skeleton", "")

            if not candidate_text or not candidate_text.strip():
                if verbose:
                    print(f"    [{idx+1}/{len(candidates)}] ✗ Empty candidate, skipping")
                continue

            if verbose:
                print(f"    [{idx+1}/{len(candidates)}] Evaluating: {candidate_text[:60]}...")

            # Filter 1: Structure Filter (Adherence Check)
            adherence_score = self._calculate_skeleton_adherence(candidate_text, skeleton)
            if adherence_score < 0.7:
                if verbose:
                    print(f"      ✗ Structure filter FAILED: Adherence {adherence_score:.2f} < 0.7")
                    print(f"         📐 Skeleton: {skeleton[:80]}...")
                    print(f"         📝 Candidate: {candidate_text[:80]}...")
                continue
            else:
                if verbose:
                    print(f"      ✓ Structure filter PASSED: Adherence {adherence_score:.2f}")

            # Filter 2 & 3: Logic Filter and Meaning Filter (both use critic evaluation)
            # Infer skeleton type from skeleton pattern for logic verification
            skeleton_type = None
            if skeleton:
                skeleton_lower = skeleton.lower()
                if "?" in skeleton or candidate_text.strip().endswith("?"):
                    skeleton_type = "RHETORICAL_QUESTION"
                elif any(cond in skeleton_lower for cond in ["if", "when", "unless", "provided that"]):
                    skeleton_type = "CONDITIONAL"
                else:
                    skeleton_type = "DECLARATIVE"

            try:
                critic_result = critic.evaluate(
                    candidate_text,
                    blueprint,
                    allowed_style_words=style_lexicon,
                    skeleton=skeleton,
                    skeleton_type=skeleton_type
                )

                # Filter 2: Logic Filter (Logic Failure Check)
                if critic_result.get("logic_fail", False):
                    if verbose:
                        print(f"      ✗ Logic filter FAILED: Logic failure detected")
                        print(f"         💬 Feedback: {critic_result.get('feedback', 'N/A')[:100]}...")
                    continue
                else:
                    if verbose:
                        print(f"      ✓ Logic filter PASSED")

                # Filter 3: Meaning Filter (Keyword Presence Check with Synonym Awareness)
                recall_score = critic_result.get("recall_score", 0.0)
                semantic_similarity = None

                # TASK 2: Check semantic similarity if recall is low
                if recall_score < min_keyword_presence:
                    # Calculate semantic similarity as fallback
                    try:
                        # Use the critic's semantic model if available
                        if hasattr(critic, 'semantic_model') and critic.semantic_model:
                            semantic_similarity = critic._calculate_semantic_similarity(
                                blueprint.original_text, candidate_text
                            )
                            if verbose:
                                print(f"      ⚠ Low recall ({recall_score:.2f}), checking semantic similarity: {semantic_similarity:.2f}")

                            # If semantic similarity is high (>0.85), allow through (synonyms used)
                            if semantic_similarity and semantic_similarity > 0.85:
                                if verbose:
                                    print(f"      ✓ Meaning filter PASSED (synonym bridge): Recall {recall_score:.2f} < {min_keyword_presence}, but semantic similarity {semantic_similarity:.2f} > 0.85")
                            else:
                                if verbose:
                                    semantic_sim_str = f"{semantic_similarity:.2f}" if semantic_similarity is not None else "N/A"
                                    print(f"      ✗ Meaning filter FAILED: Recall {recall_score:.2f} < {min_keyword_presence} AND semantic similarity {semantic_sim_str} <= 0.85")
                                    print(f"         📝 Original: {blueprint.original_text[:80]}...")
                                    print(f"         📝 Candidate: {candidate_text[:80]}...")
                                    print(f"         💬 Feedback: {critic_result.get('feedback', 'N/A')[:100]}...")
                                continue
                        else:
                            # No semantic model, reject on low recall
                            if verbose:
                                print(f"      ✗ Meaning filter FAILED: Recall {recall_score:.2f} < {min_keyword_presence} (no semantic model for synonym check)")
                                print(f"         📝 Original: {blueprint.original_text[:80]}...")
                                print(f"         📝 Candidate: {candidate_text[:80]}...")
                                print(f"         💬 Feedback: {critic_result.get('feedback', 'N/A')[:100]}...")
                            continue
                    except Exception as e:
                        if verbose:
                            print(f"      ⚠ Semantic similarity check failed: {e}, rejecting on recall alone")
                            import traceback
                            traceback.print_exc()
                        if verbose:
                            print(f"      ✗ Meaning filter FAILED: Recall {recall_score:.2f} < {min_keyword_presence}")
                            print(f"         📝 Original: {blueprint.original_text[:80]}...")
                            print(f"         📝 Candidate: {candidate_text[:80]}...")
                        continue
                else:
                    if verbose:
                        print(f"      ✓ Meaning filter PASSED: Recall {recall_score:.2f} >= {min_keyword_presence}")

            except Exception as e:
                if verbose:
                    print(f"      ⚠ Critic evaluation failed: {e}")
                    import traceback
                    traceback.print_exc()
                # Reject candidate if evaluation fails
                continue

            # Calculate style density for ranking
            style_density = self._calculate_style_density(candidate_text, style_lexicon)

            # Candidate passed all filters - add to survivors
            if verbose:
                print(f"      ✅ CANDIDATE SURVIVED ALL FILTERS")
                print(f"         📊 Scores: Adherence {adherence_score:.2f}, Recall {recall_score:.2f}, Style {style_density:.2f}, Overall {critic_result.get('score', 0.0):.2f}")

            # Store meaning failure flags for hierarchical filtering
            context_leak_detected = critic_result.get("context_leak_detected", False)
            hallucination = critic_result.get("score", 0.0) == 0.0 and recall_score < min_keyword_presence
            meaning_failure = (critic_result.get("score", 0.0) == 0.0 or
                              recall_score < min_keyword_presence or
                              context_leak_detected or
                              hallucination)

            survivors.append({
                "text": candidate_text,
                "skeleton": skeleton,
                "source_example": candidate_data.get("source_example", ""),
                "adherence_score": adherence_score,
                "recall_score": recall_score,
                "style_density": style_density,
                "score": critic_result.get("score", 0.0),
                "semantic_similarity": semantic_similarity,
                "critic_result": critic_result,
                "context_leak_detected": context_leak_detected,
                "hallucination": hallucination,
                "meaning_failure": meaning_failure,
                "repair_attempts": 1 if meaning_failure else 0  # Initialize repair attempts
            })

        # MEANING GATE: Filter by meaning FIRST, then rank by style
        min_viable_recall = evolutionary_config.get("min_viable_recall", 0.85)
        valid_candidates = [
            c for c in survivors
            if c['recall_score'] >= min_viable_recall
            and not c.get('context_leak_detected', False)
            and not c.get('hallucination', False)
        ]

        if valid_candidates:
            # Sort by style (meaning-valid candidates compete on style)
            valid_candidates.sort(key=lambda x: x["style_density"], reverse=True)
            if verbose:
                print(f"  🏟️  Arena Results: {len(valid_candidates)} meaning-valid survivors from {len(candidates)} candidates")
                if valid_candidates:
                    print(f"    🏆 Top {min(3, len(valid_candidates))} survivors (meaning-valid):")
                    for i, surv in enumerate(valid_candidates[:3]):
                        print(f"      {i+1}. Style: {surv['style_density']:.2f}, Recall: {surv['recall_score']:.2f}, Score: {surv['score']:.2f}")
                        print(f"         Text: {surv['text'][:80]}...")
            return valid_candidates[:top_k]
        else:
            # Emergency: All failed meaning - select best recall for repair
            survivors.sort(key=lambda x: x["recall_score"], reverse=True)
            if verbose:
                print(f"  🏟️  Arena Results: {len(survivors)} survivors (ALL FAILED MEANING GATE - entering repair mode)")
                if survivors:
                    print(f"    🔧 Top {min(3, len(survivors))} candidates for repair (sorted by recall):")
                    for i, surv in enumerate(survivors[:3]):
                        print(f"      {i+1}. Recall: {surv['recall_score']:.2f}, Style: {surv['style_density']:.2f}, Score: {surv['score']:.2f}")
                        print(f"         Text: {surv['text'][:80]}...")
            return survivors[:top_k]

    def _breed_children(
        self,
        parents: List[Dict[str, any]],  # Top 5 parents from arena
        blueprint: SemanticBlueprint,
        author_name: str,
        style_dna: str,
        rhetorical_type: RhetoricalType,
        style_lexicon: Optional[List[str]] = None,
        num_children: int = 10,
        verbose: bool = False
    ) -> List[str]:
        """Breed children from parents to fix specific defects.

        Analyzes each parent's weaknesses:
        - Parent A: Great style but missing keywords
        - Parent B: Perfect keywords but boring style

        Generates children that combine strengths.

        Args:
            parents: List of parent candidates with evaluation data
            blueprint: Original semantic blueprint
            author_name: Target author name
            style_dna: Style DNA description
            rhetorical_type: Rhetorical mode
            style_lexicon: Optional list of style words
            num_children: Number of children to generate (default 10)
            verbose: Whether to print debug information

        Returns:
            List of generated child strings
        """
        if not parents or len(parents) < 2:
            if verbose:
                print(f"    ⚠ Breeding requires at least 2 parents, got {len(parents)}")
            return []

        # Analyze parents to identify "Delta" (specific defects)
        parent_deltas = []
        for i, parent in enumerate(parents[:3]):  # Analyze top 3 parents
            text = parent.get("text", "")
            critic_result = parent.get("critic_result", {})
            recall_score = parent.get("recall_score", 0.0)
            style_density = parent.get("style_density", 0.0)
            logic_fail = critic_result.get("logic_fail", False)
            context_leak_detected = parent.get("context_leak_detected", False)
            hallucination = parent.get("hallucination", False)
            repair_attempts = parent.get("repair_attempts", 0)
            feedback = critic_result.get("feedback", "")

            deltas = []
            meaning_failure = False

            # MEANING GATE FAILURES (critical - must repair)
            if recall_score < 0.85:
                meaning_failure = True
                # Missing propositions - extract from recall details if available
                recall_details = critic_result.get("recall_details", {})
                missing_propositions = recall_details.get("missing", [])
                if missing_propositions:
                    deltas.append(f"Missing propositions: {', '.join(missing_propositions[:3])}")
                else:
                    # Fallback: extract missing keywords from blueprint
                    missing_keywords = []
                    blueprint_keywords = list(blueprint.core_keywords)[:5]
                    text_lower = text.lower()
                    for keyword in blueprint_keywords:
                        if keyword.lower() not in text_lower:
                            missing_keywords.append(keyword)
                    if missing_keywords:
                        deltas.append(f"Missing keywords: {', '.join(missing_keywords[:3])}")
                    else:
                        deltas.append(f"Proposition recall too low ({recall_score:.2f} < 0.85)")

            if context_leak_detected or hallucination:
                meaning_failure = True
                leaked_keywords = critic_result.get("leaked_keywords", [])
                if leaked_keywords:
                    deltas.append(f"Hallucination detected: {', '.join(leaked_keywords[:3])}")
                else:
                    deltas.append("Hallucination detected (context leak)")

            # STYLE ISSUES (non-critical)
            if style_density < 0.1:
                deltas.append("Boring style, needs jargon")

            if logic_fail:
                deltas.append("Logic contradiction detected")

            parent_deltas.append({
                "text": text,
                "deltas": deltas,
                "strengths": [],
                "meaning_failure": meaning_failure,
                "repair_attempts": repair_attempts
            })

            # Identify strengths
            if style_density > 0.2:
                parent_deltas[-1]["strengths"].append("Great style")
            if recall_score > 0.9:
                parent_deltas[-1]["strengths"].append("Perfect keywords")
            if parent.get("adherence_score", 0.0) > 0.9:
                parent_deltas[-1]["strengths"].append("Strong structure")

        # Check if we need repair mode or structure swap
        best_parent = parent_deltas[0] if parent_deltas else None
        needs_repair = best_parent and best_parent.get("meaning_failure", False)
        repair_attempts = best_parent.get("repair_attempts", 0) if best_parent else 0
        use_structure_swap = needs_repair and repair_attempts >= 2

        # Extract missing propositions for repair prompts
        missing_propositions = []
        if needs_repair and best_parent:
            for delta in best_parent.get("deltas", []):
                if "Missing propositions:" in delta:
                    # Extract propositions from delta text
                    prop_text = delta.split("Missing propositions:")[1].strip()
                    missing_propositions = [p.strip() for p in prop_text.split(",")[:5]]
                elif "Missing keywords:" in delta:
                    kw_text = delta.split("Missing keywords:")[1].strip()
                    missing_propositions = [k.strip() for k in kw_text.split(",")[:5]]

        # Build breeding prompt
        parent_descriptions = []
        for i, parent_data in enumerate(parent_deltas):
            deltas_text = ", ".join(parent_data["deltas"]) if parent_data["deltas"] else "No major defects"
            strengths_text = ", ".join(parent_data["strengths"]) if parent_data["strengths"] else "No major strengths"
            parent_descriptions.append(
                f"Parent {chr(65+i)}: \"{parent_data['text']}\"\n"
                f"  Strengths: {strengths_text}\n"
                f"  Defects: {deltas_text}"
            )

        if use_structure_swap:
            # STRUCTURE SWAP MODE: Discard template, use simple S-V-O structure
            breeding_prompt = f"""CRITICAL STRUCTURE SWAP:
The structure template is incompatible with the meaning. Previous repair attempts failed.

### ERROR
{best_parent.get('deltas', ['Meaning failure'])[0] if best_parent else 'Meaning failure'}

### ORIGINAL PROPOSITIONS
{', '.join(missing_propositions) if missing_propositions else blueprint.original_text}

### TASK
Generate {num_children} children that:
1. DISCARD the template structure entirely (do NOT use the parent's structure).
2. Rewrite the propositions using simple Subject-Verb-Object structure.
3. Ensure ALL propositions are preserved: {', '.join(missing_propositions) if missing_propositions else 'all original meaning'}.
4. THEN apply the target style and vocabulary: {', '.join(style_lexicon[:20]) if style_lexicon else 'None'}.
5. Do NOT force incompatible rhetorical structures (e.g., don't use list format for narratives).

### OUTPUT FORMAT
Output PURE JSON. A single list of strings:
[
  "Child 1 text...",
  "Child 2 text...",
  ...
  "Child {num_children} text..."
]
"""
        elif needs_repair:
            # REPAIR MODE: Fix meaning while preserving structure
            error_msg = best_parent.get('deltas', ['Meaning failure'])[0] if best_parent else 'Meaning failure'
            breeding_prompt = f"""CRITICAL REPAIR TASK:
The text below has correct Structure but failed the Meaning Check.

### ERROR
{error_msg}

### ORIGINAL TEXT
{best_parent.get('text', '') if best_parent else ''}

### INPUT PROPOSITIONS
{', '.join(missing_propositions) if missing_propositions else blueprint.original_text}

### TASK
Generate {num_children} children that:
1. Keep the exact sentence structure and rhythm from the original text.
2. Remove hallucinated words/phrases (if any).
3. Replace them with correct facts from Input Propositions: {', '.join(missing_propositions) if missing_propositions else 'all original meaning'}.
4. Do NOT change the style or vocabulary: {', '.join(style_lexicon[:20]) if style_lexicon else 'None'}.

### OUTPUT FORMAT
Output PURE JSON. A single list of strings:
[
  "Child 1 text...",
  "Child 2 text...",
  ...
  "Child {num_children} text..."
]
"""
        else:
            # STANDARD BREEDING MODE
            breeding_prompt = f"""You are a master literary breeder. Your task is to generate children that combine the strengths of multiple parent sentences while fixing their specific defects.

### PARENTS TO BREED:
{chr(10).join(parent_descriptions)}

### TASK
Generate {num_children} children that:
1. Combine the strengths of the parents (e.g., Parent A's style + Parent B's keywords)
2. Fix the specific defects identified (e.g., add missing keywords, inject jargon, fix logic)
3. Preserve the core meaning: "{blueprint.original_text}"

### STYLE PALETTE
Vocabulary to prioritize: {', '.join(style_lexicon[:20]) if style_lexicon else 'None'}

### OUTPUT FORMAT
Output PURE JSON. A single list of strings:
[
  "Child 1 text...",
  "Child 2 text...",
  ...
  "Child {num_children} text..."
]
"""

        # Generate children using batch generation approach
        try:
            response = self.llm_provider.call(
                system_prompt="You are a precision breeding generator. Output ONLY valid JSON.",
                user_prompt=breeding_prompt,
                model_type="editor",
                require_json=True,
                temperature=0.7,  # Moderate temperature for focused breeding
                max_tokens=self.translator_config.get("max_tokens", 500)
            )

            # Parse JSON response
            children = json.loads(response)
            if isinstance(children, list):
                children = [c.strip() for c in children if c and c.strip()]
                return children[:num_children]
            else:
                if verbose:
                    print(f"    ⚠ Breeding returned non-list, attempting extraction")
                return self._extract_json_list(response)[:num_children]

        except json.JSONDecodeError as e:
            if verbose:
                print(f"    ⚠ Breeding JSON decode error: {e}, attempting extraction")
            return self._extract_json_list(response)[:num_children]
        except Exception as e:
            if verbose:
                print(f"    ✗ Breeding failed: {e}")
            return []

    def _evaluate_template_candidate(
        self,
        candidate: str,
        blueprint: SemanticBlueprint,
        skeleton: str,
        style_dna: Optional[Dict[str, any]] = None
    ) -> Dict[str, any]:
        """Evaluate template candidate using composite scoring.

        Calculates semantic score, structural adherence score, and style density score.
        Applies hard gates (reject if semantic < 0.7 or adherence < 0.7).

        Args:
            candidate: Generated candidate text.
            blueprint: Original semantic blueprint.
            skeleton: Skeleton template used for generation.
            style_dna: Optional style DNA dictionary with lexicon.

        Returns:
            Dictionary with scores and pass status:
            {
                "semantic_score": float,
                "adherence_score": float,
                "style_density": float,
                "composite_score": float,
                "passed_gates": bool
            }
        """
        # GHOSTBUSTER: Reject candidates containing skeleton placeholder brackets
        # These are generation failures (raw skeletons), not valid candidates
        # Note: Citation references like [^155] are allowed - only reject [NP], [VP], [ADJ] placeholders
        skeleton_placeholder_pattern = r'\[(?:NP|VP|ADJ)\]'
        if re.search(skeleton_placeholder_pattern, candidate, re.IGNORECASE):
            return {
                "semantic_score": 0.0,
                "adherence_score": 0.0,
                "style_density": 0.0,
                "composite_score": 0.0,
                "passed_gates": False,
                "rejection_reason": "Candidate contains skeleton placeholder brackets ([NP], [VP], [ADJ]) - generation failure"
            }

        from src.validator.semantic_critic import SemanticCritic

        # Load weights from config
        weights = self.config.get("weights", {})
        semantic_weight = weights.get("semantic", 0.4)
        adherence_weight = weights.get("structure_adherence", 0.3)
        style_weight = weights.get("style_density", 0.3)

        # Load thresholds from config
        template_config = self.translator_config.get("template_evolution", {})
        semantic_threshold = template_config.get("semantic_threshold", 0.7)
        adherence_threshold = template_config.get("adherence_threshold", 0.7)

        # Extract style lexicon
        style_lexicon = None
        if style_dna and isinstance(style_dna, dict):
            style_lexicon = style_dna.get("lexicon")

        # 1. Semantic Score (40% weight)
        critic = SemanticCritic(config_path=self.config_path)
        semantic_result = self.soft_scorer.evaluate_with_raw_score(
            candidate, blueprint, style_lexicon=style_lexicon, skeleton=skeleton
        )
        semantic_score = semantic_result.get("raw_score", semantic_result.get("score", 0.0))

        # 2. Structural Adherence Score (30% weight)
        adherence_score = self._calculate_skeleton_adherence(candidate, skeleton)

        # 3. Style Density Score (30% weight)
        style_density = self._calculate_style_density(candidate, style_lexicon)

        # 4. Expansion Ratio Gate (Dynamic): Reject excessive expansion
        input_words = blueprint.original_text.split()
        candidate_words = candidate.split()
        input_len = len(input_words)
        expansion_ratio = len(candidate_words) / max(1, input_len)

        # Dynamic expansion thresholds: Allow more expansion for short inputs
        # Short sentences need more room to breathe when transformed to complex styles (e.g., Maoist dialectic)
        if input_len < 10:
            max_ratio = 6.5  # Allow 6.5x expansion for very short inputs (e.g., 5 words -> 32 words, or 7 words -> 45 words)
        elif input_len < 20:
            max_ratio = 4.0  # Allow 4x expansion for short inputs
        else:
            max_ratio = 2.5  # Allow 2.5x expansion for longer inputs

        # Hard rejection: If ratio > max_ratio, reject immediately
        # Exception: Allow if style_density > 0.4 (very high style usage indicates intentional expansion)
        if expansion_ratio > max_ratio and style_density <= 0.4:
            return {
                "semantic_score": semantic_score,
                "adherence_score": adherence_score,
                "style_density": style_density,
                "composite_score": 0.0,
                "passed_gates": False,
                "rejection_reason": f"Candidate rejected due to excessive expansion (Ratio: {expansion_ratio:.2f} > {max_ratio:.1f} for {input_len}-word input)."
            }

        # Hard gates: reject if semantic or adherence below threshold
        passed_gates = (semantic_score >= semantic_threshold and
                       adherence_score >= adherence_threshold)

        # Calculate composite score
        if not passed_gates:
            composite_score = 0.0
        else:
            composite_score = (
                semantic_score * semantic_weight +
                adherence_score * adherence_weight +
                style_density * style_weight
            )

        return {
            "semantic_score": semantic_score,
            "adherence_score": adherence_score,
            "style_density": style_density,
            "composite_score": composite_score,
            "passed_gates": passed_gates
        }

    def translate(
        self,
        blueprint: SemanticBlueprint,
        author_name: str,
        style_dna: str,
        rhetorical_type: RhetoricalType,
        examples: List[str],  # 3 examples from atlas
        verbose: bool = False
    ) -> str:
        # Initialize style_dna_dict at function start to ensure it's always defined
        style_dna_dict = None
        """Translate blueprint into styled text.

        Args:
            blueprint: Semantic blueprint to translate.
            author_name: Target author name.
            style_dna: Style DNA description.
            rhetorical_type: Rhetorical mode.
            examples: Few-shot examples from atlas.

        Returns:
            Generated text in target style.
        """
        # Initialize style_dna_dict at function start to ensure it's always defined
        style_dna_dict = None

        if not examples:
            # Fallback if no examples provided
            examples = ["Example text in the target style."]

        # STRICT FALLBACK RULE: If only 1 example, force fallback unless perfect match
        if len(examples) == 1:
            # Extract style_dna_dict from single example (needed for standard generation fallback)
            try:
                from src.analyzer.style_extractor import StyleExtractor
                style_extractor = StyleExtractor(config_path=self.config_path)
                if examples:
                    style_dna_dict = style_extractor.extract_style_dna(examples)
            except Exception:
                style_dna_dict = None

            # Check sentence type compatibility
            def _is_sentence_question(text):
                """Check if sentence is a question."""
                if not text or not text.strip():
                    return False
                question_starters = ["where", "what", "how", "why", "when", "do", "does", "is", "are"]
                first_word = text.strip().split()[0].lower() if text.strip().split() else ""
                return first_word in question_starters or text.strip().endswith("?")

            example_is_question = _is_sentence_question(examples[0])
            input_is_question = _is_sentence_question(blueprint.original_text)

            # Check rhetorical type match (should already match, but verify)
            from src.atlas.rhetoric import RhetoricalClassifier
            classifier = RhetoricalClassifier()
            example_rhetorical_type = classifier.classify_heuristic(examples[0])

            # If ANY mismatch, skip template evolution
            if example_is_question != input_is_question or example_rhetorical_type != rhetorical_type:
                if verbose:
                    print(f"  ⚠ Only 1 example retrieved. Sentence type match: {example_is_question == input_is_question}, "
                          f"Rhetorical type match: {example_rhetorical_type == rhetorical_type}. Skipping template evolution.")
                # Skip template evolution, use standard generation
                compatible_skeletons = []
            else:
                # Perfect match - allow template evolution to proceed
                if verbose:
                    print(f"  ✓ Only 1 example, but perfect match (sentence type and rhetorical type). Proceeding with template evolution.")
                compatible_skeletons = self._extract_multiple_skeletons(examples, blueprint, verbose=verbose)
        else:
            # Multiple examples - proceed normally
            # Extract style_dna_dict from examples if available
            style_dna_dict = None
            try:
                from src.analyzer.style_extractor import StyleExtractor
                style_extractor = StyleExtractor(config_path=self.config_path)
                if examples:
                    style_dna_dict = style_extractor.extract_style_dna(examples)
            except Exception:
                style_dna_dict = None

            # Phase 1: Smart Skeleton Selection (Pre-Filter) - LAZY EVALUATION
            # Only extract skeletons if template evolution is enabled and we have sufficient examples
            template_config = self.translator_config.get("template_evolution", {})
            template_evolution_enabled = template_config.get("enabled", True)  # Default to True for backward compatibility

            compatible_skeletons = []  # Default: no skeletons (will use standard generation)

            if template_evolution_enabled:
                # Check if rhetorical type actually needs skeleton extraction
                # Refined: Only skip if sentence is both short AND semantically empty (no SVO triples)
                # Complete short sentences with semantic structure should still use template evolution
                input_text = blueprint.original_text
                word_count = len(input_text.split())
                has_svo_triples = len(blueprint.svo_triples) > 0
                has_semantic_content = has_svo_triples or len(blueprint.core_keywords) > 0

                # Only skip if: short (< 10 words) AND no semantic structure (no SVOs, no keywords)
                is_simple_sentence = word_count < 10 and not has_semantic_content

                # Skip skeleton extraction only for truly simple/fragmentary sentences
                if is_simple_sentence:
                    if verbose:
                        print(f"  Skipping skeleton extraction: simple sentence detected (no semantic structure)")
                else:
                    num_examples = template_config.get("num_examples", 5)
                    candidates_per_template = template_config.get("candidates_per_template", 2)

                    # Retrieve top N examples (request more to have options after filtering)
                    if len(examples) < num_examples:
                        # Use all available examples
                        candidate_examples = examples
                    else:
                        # Use top N examples
                        candidate_examples = examples[:num_examples]

                    if verbose:
                        print(f"  Parallel Template Evolution: Processing {len(candidate_examples)} examples")

                    # Extract multiple skeletons with complexity filtering and deduplication
                    compatible_skeletons = self._extract_multiple_skeletons(candidate_examples, blueprint, verbose=verbose)

        if verbose:
            print(f"  Skeletons extracted: {len(compatible_skeletons)} compatible (after complexity filtering)")

        if not compatible_skeletons:
            if verbose:
                print(f"  No compatible skeletons found, falling back to standard generation")
        else:
            # Phase 2: Mass Generation (Evolutionary Architecture)
            evolutionary_config = self.config.get("evolutionary", {})
            batch_size = evolutionary_config.get("batch_size", 20)
            max_generations = evolutionary_config.get("max_generations", 3)
            convergence_threshold = evolutionary_config.get("convergence_threshold", 0.95)
            top_k_parents = evolutionary_config.get("top_k_parents", 5)
            breeding_children = evolutionary_config.get("breeding_children", 10)

            # Extract style lexicon from style_dna_dict
            style_lexicon = None
            if style_dna_dict and isinstance(style_dna_dict, dict):
                style_lexicon = style_dna_dict.get("lexicon")

            all_candidates = []
            input_type = self._detect_sentence_type(blueprint.original_text)

            for skeleton, source_example in compatible_skeletons:
                # Early rejection: Check skeleton type vs input type before generating candidates
                skeleton_type = self._detect_sentence_type(skeleton)

                # Same logic as in _extract_multiple_skeletons: reject Question -> Declarative
                is_compatible = False
                if input_type == skeleton_type:
                    is_compatible = True
                elif input_type == "DECLARATIVE" and skeleton_type == "CONDITIONAL":
                    is_compatible = True

                if not is_compatible:
                    if verbose:
                        print(f"  Skipping skeleton: type mismatch ({skeleton_type} skeleton for {input_type} input)")
                    continue

                if verbose:
                    print(f"  Generating batch of {batch_size} variants for skeleton: {skeleton[:60]}...")

                try:
                    # Generate batch of variants in single API call
                    batch = self._generate_batch(
                        skeleton=skeleton,
                        blueprint=blueprint,
                        author_name=author_name,
                        style_dna=style_dna,
                        rhetorical_type=rhetorical_type,
                        style_lexicon=style_lexicon,
                        batch_size=batch_size,
                        verbose=verbose
                    )

                    # Filter out logic mismatch signals and add to candidates
                    for variant in batch:
                        if variant and variant.strip():
                            # ESCAPE HATCH: Check for logic mismatch signal
                            if "SKIPPING: LOGIC_MISMATCH" in variant.upper() or "LOGIC_MISMATCH" in variant.upper():
                                if verbose:
                                    print(f"    ⚠ Logic mismatch detected for skeleton, skipping this variant")
                                continue

                            all_candidates.append({
                                "text": variant,
                                "skeleton": skeleton,
                                "source_example": source_example
                            })
                except Exception as e:
                    if verbose:
                        print(f"    ✗ Batch generation failed for skeleton: {e}")
                    continue

            if verbose:
                print(f"  Total candidates generated: {len(all_candidates)}")

            # Phase 3: The Arena (Batch Evaluation)
            survivors = self._run_arena(
                candidates=all_candidates,
                blueprint=blueprint,
                style_dna_dict=style_dna_dict,
                verbose=verbose
            )

            # Phase 4: Convergence Check
            best_survivor = survivors[0] if survivors else None
            if best_survivor and best_survivor.get("score", 0.0) >= convergence_threshold:
                if verbose:
                    print(f"  ✓ Perfect match found (score: {best_survivor.get('score', 0.0):.2f} >= {convergence_threshold})")
                best_text = self._restore_citations_and_quotes(best_survivor["text"], blueprint)
                return best_text

            # Phase 5: Evolutionary Feedback (if needed)
            if best_survivor:
                for generation in range(1, max_generations):
                    if best_survivor.get("score", 0.0) >= convergence_threshold:
                        break

                    if verbose:
                        print(f"  Generation {generation + 1}/{max_generations}: Best score {best_survivor.get('score', 0.0):.2f} < {convergence_threshold}, breeding children...")

                    # Get fresh generation ratio from config
                    fresh_ratio = evolutionary_config.get("fresh_generation_ratio", 0.33)
                    num_improvements = max(1, int(breeding_children * (1 - fresh_ratio)))
                    num_fresh = breeding_children - num_improvements

                    if verbose:
                        print(f"    Population strategy: {num_improvements} improvements + {num_fresh} fresh (ratio: {fresh_ratio:.2f})")

                    # Breed children from top parents (improvements)
                    top_parents = survivors[:top_k_parents]
                    improvement_children = []
                    if num_improvements > 0:
                        improvement_children = self._breed_children(
                            parents=top_parents,
                            blueprint=blueprint,
                            author_name=author_name,
                            style_dna=style_dna,
                            rhetorical_type=rhetorical_type,
                            style_lexicon=style_lexicon,
                            num_children=num_improvements,
                            verbose=verbose
                        )

                    # Generate fresh children from scratch
                    fresh_children = []
                    if num_fresh > 0:
                        # Get examples for fresh generation (use best survivor's skeleton source if available)
                        fresh_examples = examples if examples else []
                        if best_survivor and best_survivor.get("source_example"):
                            fresh_examples = [best_survivor.get("source_example")] + (fresh_examples[:2] if fresh_examples else [])

                        fresh_candidates = self._generate_fresh_candidates(
                            blueprint=blueprint,
                            author_name=author_name,
                            style_dna=style_dna,
                            rhetorical_type=rhetorical_type,
                            temperature=0.8,  # High temp for diversity
                            num_candidates=num_fresh,
                            verbose=verbose,
                            style_lexicon=style_lexicon,
                            examples=fresh_examples
                        )
                        fresh_children = [c[1] for c in fresh_candidates]  # Extract text from tuples

                    # Combine all children
                    children = improvement_children + fresh_children

                    if not children:
                        if verbose:
                            print(f"    ⚠ No children generated, stopping evolution")
                        break

                    # Evaluate children in arena
                    # Track which children are fresh (don't have skeletons)
                    child_candidates = []
                    for i, c in enumerate(children):
                        # Fresh children are those beyond the improvement_children count
                        is_fresh = i >= len(improvement_children) if improvement_children else False
                        child_candidates.append({
                            "text": c,
                            "skeleton": best_survivor["skeleton"] if not is_fresh else "",  # Fresh children don't use parent skeleton
                            "source_example": best_survivor.get("source_example", "") if not is_fresh else ""
                        })
                    child_survivors = self._run_arena(
                        candidates=child_candidates,
                        blueprint=blueprint,
                        style_dna_dict=style_dna_dict,
                        verbose=verbose
                    )

                    # Track repair attempts: if parent had meaning failure, increment repair_attempts for children
                    parent_repair_attempts = best_survivor.get("repair_attempts", 0)
                    parent_meaning_failure = (best_survivor.get("score", 0.0) == 0.0 or
                                             best_survivor.get("recall_score", 0.0) < 0.85 or
                                             best_survivor.get("context_leak_detected", False) or
                                             best_survivor.get("hallucination", False))

                    for child in child_survivors:
                        # If child also has meaning failure, increment repair attempts
                        child_meaning_failure = (child.get("score", 0.0) == 0.0 or
                                                child.get("recall_score", 0.0) < 0.85 or
                                                child.get("context_leak_detected", False) or
                                                child.get("hallucination", False))
                        if child_meaning_failure:
                            if parent_meaning_failure:
                                # Both parent and child failed - increment repair attempts
                                child["repair_attempts"] = parent_repair_attempts + 1
                            else:
                                # New failure - start tracking
                                child["repair_attempts"] = 1
                        else:
                            # Child passed meaning gate - reset repair attempts
                            child["repair_attempts"] = 0

                    # Update survivors (keep best from parents + children)
                    all_survivors = survivors + child_survivors
                    all_survivors.sort(key=lambda x: x.get("score", 0.0), reverse=True)
                    survivors = all_survivors[:top_k_parents]
                    best_survivor = survivors[0] if survivors else None

                    if not best_survivor:
                        break

            # Final selection: Return best survivor or fallback
            if best_survivor:
                if verbose:
                    print(f"  ✓ Final selection: Best survivor (score: {best_survivor.get('score', 0.0):.2f})")
                best_text = self._restore_citations_and_quotes(best_survivor["text"], blueprint)
                return best_text

        # Fallback: Standard generation if evolutionary architecture fails
        if verbose:
            print(f"  Falling back to standard generation")

        # Step 2: Standard generation (fallback if structural cloning fails or not applicable)
        # Style-Infused Fallback: Ensure we use style DNA even in fallback
        prompt = self._build_prompt(blueprint, author_name, style_dna, rhetorical_type, examples)

        # Extract style lexicon for prompt injection
        style_lexicon_text = ""
        if style_dna_dict and isinstance(style_dna_dict, dict):
            style_lexicon = style_dna_dict.get("lexicon", [])
            if style_lexicon:
                lexicon_preview = ", ".join(style_lexicon[:15])  # Show first 15 words
                style_lexicon_text = f"\n\n**STYLE VOCABULARY (Use these words):** {lexicon_preview}"

        # Build examples text for style injection
        examples_text = ""
        if examples:
            examples_preview = "\n".join([f"- \"{ex}\"" for ex in examples[:3]])
            examples_text = f"\n\n**STYLE EXAMPLES (Match this voice):**\n{examples_preview}"

        # Inject style guidance into prompt
        if style_lexicon_text or examples_text:
            style_injection = f"""
=== STYLE REQUIREMENT (CRITICAL - DO NOT BE BORING) ===
You are writing in the style of these examples. Do NOT write generic corporate English.{examples_text}{style_lexicon_text}

**CRITICAL:** Transform the sentence to match the author's distinctive voice. Use the vocabulary provided. Be distinct, not generic.
=====================================================
"""
            prompt = style_injection + prompt

        system_prompt_template = _load_prompt_template("translator_system.md")
        system_prompt = system_prompt_template.format(
            author_name=author_name,
            style_dna=style_dna
        )

        try:
            # Phase 2: Two-Pass Pipeline
            # Pass 1: Draft generation (high temperature for meaning preservation)
            draft_temperature = self.translator_config.get("draft_temperature", 0.75)
            draft = self.llm_provider.call(
                system_prompt=system_prompt,
                user_prompt=prompt,
                temperature=draft_temperature,
                max_tokens=self.translator_config.get("max_tokens", 300)
            )
            draft = clean_generated_text(draft)
            draft = draft.strip()

            # Pass 2: Polish for natural English (low temperature for refinement)
            polished = self._polish_draft(draft, blueprint, author_name, style_dna)

            # Restore citations and quotes if missing
            polished = self._restore_citations_and_quotes(polished, blueprint)
            return polished
        except Exception as e:
            # Fallback on error
            return self.translate_literal(blueprint, author_name, style_dna)

    def _build_prompt(
        self,
        blueprint: SemanticBlueprint,
        author_name: str,
        style_dna: str,
        rhetorical_type: RhetoricalType,
        examples: List[str]
    ) -> str:
        """Build few-shot prompt with contextual anchoring.

        Args:
            blueprint: Semantic blueprint (with positional metadata).
            author_name: Target author name.
            style_dna: Style DNA description.
            rhetorical_type: Rhetorical mode.
            examples: Few-shot examples.

        Returns:
            Formatted prompt string with position instructions and context.
        """
        # Check if blueprint is semantically complete
        if self._is_blueprint_incomplete(blueprint):
            # Use original-text-only prompt (no blueprint structure)
            return self._build_original_text_only_prompt(blueprint, author_name, style_dna, rhetorical_type, examples)

        examples_text = "\n".join([f"Example {i+1}: \"{ex}\"" for i, ex in enumerate(examples)])

        # Get position-specific instruction
        pos_instruction = self.positional_instructions.get(
            blueprint.position,
            self.positional_instructions.get("BODY", "")
        )

        # Build context block (only for BODY/CLOSER positions)
        context_block = ""
        if blueprint.position in ["BODY", "CLOSER"] and blueprint.previous_context:
            context_block = f"""
=== PREVIOUS CONTEXT (The sentence you just wrote) ===
"{blueprint.previous_context}"
(Your rewriting MUST logically follow this sentence.)
======================================================
"""

        # FAILSAFE: If blueprint is empty, use original text directly
        if not blueprint.svo_triples and not blueprint.core_keywords:
            # Build citations and quotes sections even for empty blueprint
            citations_text = ""
            if blueprint.citations:
                citation_list = [cit[0] for cit in blueprint.citations]
                citations_text = f"- Citations to preserve: {', '.join(citation_list)}"

            quotes_text = ""
            if blueprint.quotes:
                quote_list = [quote[0] for quote in blueprint.quotes]
                quotes_text = f"- Direct quotes to preserve exactly: {', '.join(quote_list)}"

            preservation_section = ""
            if citations_text or quotes_text:
                preservation_section = f"""
=== CRITICAL PRESERVATION REQUIREMENTS (NON-NEGOTIABLE) ===
{citations_text}
{quotes_text}

CRITICAL RULES:
- ALL citations MUST use EXACTLY the [^number] format shown above (e.g., [^155])
- DO NOT use (Author, Year), (Smith 42), or any other citation formats - ONLY [^number] format is allowed
- ALL citations [^number] MUST be included in your output (you may place them at the end of the sentence if needed)
- ALL direct quotes MUST be preserved EXACTLY word-for-word as shown above
- DO NOT modify, paraphrase, or change any quoted text
- DO NOT remove or relocate citations"""

            template = _load_prompt_template("translator_user_empty_blueprint_template.md")
            return template.format(
                rhetorical_type=rhetorical_type.value,
                context_block=context_block,
                examples_text=examples_text,
                original_text=blueprint.original_text,
                preservation_section=preservation_section,
                pos_instruction=pos_instruction
            )

        # Normal blueprint path
        subjects = blueprint.get_subjects()
        verbs = blueprint.get_verbs()
        objects = blueprint.get_objects()
        entities = blueprint.named_entities
        keywords = sorted(blueprint.core_keywords)

        entities_text = ', '.join([f"{ent[0]} ({ent[1]})" for ent in entities]) if entities else "None"

        # Build citations and quotes sections
        citations_text = ""
        if blueprint.citations:
            citation_list = [cit[0] for cit in blueprint.citations]
            citations_text = f"- Citations to preserve: {', '.join(citation_list)}"

        quotes_text = ""
        if blueprint.quotes:
            quote_list = [quote[0] for quote in blueprint.quotes]
            quotes_text = f"- Direct quotes to preserve exactly: {', '.join(quote_list)}"

        preservation_section = ""
        if citations_text or quotes_text:
            preservation_section = f"""
=== CRITICAL PRESERVATION REQUIREMENTS (NON-NEGOTIABLE) ===
{citations_text}
{quotes_text}

CRITICAL RULES:
- ALL citations MUST use EXACTLY the [^number] format shown above (e.g., [^155])
- DO NOT use (Author, Year), (Smith 42), or any other citation formats - ONLY [^number] format is allowed
- ALL citations [^number] MUST be included in your output (you may place them at the end of the sentence if needed)
- ALL direct quotes MUST be preserved EXACTLY word-for-word as shown above
- DO NOT modify, paraphrase, or change any quoted text
- DO NOT remove or relocate citations"""

        template = _load_prompt_template("translator_user_template.md")
        return template.format(
            rhetorical_type=rhetorical_type.value,
            context_block=context_block,
            examples_text=examples_text,
            original_text=blueprint.original_text,
            subjects=', '.join(subjects) if subjects else 'None',
            verbs=', '.join(verbs) if verbs else 'None',
            objects=', '.join(objects) if objects else 'None',
            entities=entities_text,
            keywords=', '.join(keywords) if keywords else 'None',
            preservation_section=preservation_section,
            pos_instruction=pos_instruction
        )

    def _build_original_text_only_prompt(
        self,
        blueprint: SemanticBlueprint,
        author_name: str,
        style_dna: str,
        rhetorical_type: RhetoricalType,
        examples: List[str]
    ) -> str:
        """Build prompt using only original text (no blueprint structure).

        Used when blueprint is semantically incomplete to avoid generating
        broken sentences from incomplete blueprints.

        Args:
            blueprint: Semantic blueprint (with original_text).
            author_name: Target author name.
            style_dna: Style DNA description.
            rhetorical_type: Rhetorical mode.
            examples: Few-shot examples.

        Returns:
            Formatted prompt string using original text only.
        """
        examples_text = "\n".join([f"Example {i+1}: \"{ex}\"" for i, ex in enumerate(examples)])

        # Get position-specific instruction
        pos_instruction = self.positional_instructions.get(
            blueprint.position,
            self.positional_instructions.get("BODY", "")
        )

        # Build context block (only for BODY/CLOSER positions)
        context_block = ""
        if blueprint.position in ["BODY", "CLOSER"] and blueprint.previous_context:
            context_block = f"""
=== PREVIOUS CONTEXT (The sentence you just wrote) ===
"{blueprint.previous_context}"
(Your rewriting MUST logically follow this sentence.)
======================================================
"""

        # Build citations and quotes sections
        citations_text = ""
        if blueprint.citations:
            citation_list = [cit[0] for cit in blueprint.citations]
            citations_text = f"- Citations to preserve: {', '.join(citation_list)}"

        quotes_text = ""
        if blueprint.quotes:
            quote_list = [quote[0] for quote in blueprint.quotes]
            quotes_text = f"- Direct quotes to preserve exactly: {', '.join(quote_list)}"

        preservation_section = ""
        if citations_text or quotes_text:
            preservation_section = f"""
=== CRITICAL PRESERVATION REQUIREMENTS (NON-NEGOTIABLE) ===
{citations_text}
{quotes_text}

CRITICAL RULES:
- ALL citations MUST use EXACTLY the [^number] format shown above (e.g., [^155])
- DO NOT use (Author, Year), (Smith 42), or any other citation formats - ONLY [^number] format is allowed
- ALL citations [^number] MUST be included in your output (you may place them at the end of the sentence if needed)
- ALL direct quotes MUST be preserved EXACTLY word-for-word as shown above
- DO NOT modify, paraphrase, or change any quoted text
- DO NOT remove or relocate citations"""

        template = _load_prompt_template("translator_user_original_only_template.md")
        return template.format(
            rhetorical_type=rhetorical_type.value,
            context_block=context_block,
            examples_text=examples_text,
            original_text=blueprint.original_text,
            preservation_section=preservation_section,
            pos_instruction=pos_instruction
        )

    def _polish_draft(
        self,
        draft: str,
        blueprint: SemanticBlueprint,
        author_name: str,
        style_dna: str
    ) -> str:
        """Polish a draft sentence for natural English flow.

        This is the second pass in the two-pass pipeline. Takes a draft that
        preserves meaning and refines it for natural English, fixing passive
        voice and stilted phrasing.

        Args:
            draft: Draft sentence to polish.
            blueprint: Original semantic blueprint (for reference).
            author_name: Target author name.
            style_dna: Style DNA description.

        Returns:
            Polished sentence in natural English.
        """
        if not draft or not draft.strip():
            return draft

        polish_template = _load_prompt_template("translator_polish.md")
        polish_prompt = polish_template.format(
            draft_text=draft,
            original_text=blueprint.original_text
        )

        system_prompt_template = _load_prompt_template("translator_system.md")
        system_prompt = system_prompt_template.format(
            author_name=author_name,
            style_dna=style_dna
        )

        try:
            polished = self.llm_provider.call(
                system_prompt=system_prompt,
                user_prompt=polish_prompt,
                temperature=self.translator_config.get("polish_temperature", 0.25),
                max_tokens=self.translator_config.get("max_tokens", 300)
            )
            polished = clean_generated_text(polished)
            polished = polished.strip()

            # If polish fails or returns empty, return original draft
            if not polished:
                return draft

            return polished
        except Exception as e:
            # If polish fails, return original draft
            return draft

    def translate_literal(
        self,
        blueprint: SemanticBlueprint,
        author_name: str,
        style_dna: str,
        rhetorical_type: Optional[RhetoricalType] = None,
        examples: Optional[List[str]] = None
    ) -> str:
        """Style-preserving fallback (loose style transfer when blueprint constraints fail).

        Args:
            blueprint: Semantic blueprint.
            author_name: Target author name.
            style_dna: Style DNA description.
            rhetorical_type: Optional rhetorical mode (for style-preserving fallback).
            examples: Optional few-shot examples (for style-preserving fallback).

        Returns:
            Style-transferred text (never returns original text verbatim).
        """
        # If blueprint is incomplete, return original text (don't try to generate from broken blueprint)
        if self._is_blueprint_incomplete(blueprint):
            return blueprint.original_text

        # FAILSAFE: If blueprint is empty, use style-preserving fallback
        if not blueprint.svo_triples and not blueprint.core_keywords:
            return self._translate_style_fallback(blueprint, author_name, style_dna, rhetorical_type, examples)

        # Normal blueprint path - use literal translation template
        subjects = blueprint.get_subjects()
        verbs = blueprint.get_verbs()
        objects = blueprint.get_objects()

        # Build citations and quotes sections for literal translation too
        citations_text = ""
        if blueprint.citations:
            citation_list = [cit[0] for cit in blueprint.citations]
            citations_text = f"\nCitations to include: {', '.join(citation_list)}"

        quotes_text = ""
        if blueprint.quotes:
            quote_list = [quote[0] for quote in blueprint.quotes]
            quotes_text = f"\nQuotes to preserve exactly: {', '.join(quote_list)}"

        prompt_template = _load_prompt_template("translator_literal_user.md")
        # CRITICAL: Include original_text so LLM can see full content even if blueprint is incomplete
        prompt = prompt_template.format(
            original_text=blueprint.original_text,
            subjects=', '.join(subjects) if subjects else 'None',
            verbs=', '.join(verbs) if verbs else 'None',
            objects=', '.join(objects) if objects else 'None',
            citations_text=citations_text,
            quotes_text=quotes_text
        )

        system_prompt_template = _load_prompt_template("translator_literal_system.md")
        system_prompt = system_prompt_template.format(author_name=author_name)

        try:
            generated = self.llm_provider.call(
                system_prompt=system_prompt,
                user_prompt=prompt,
                temperature=self.translator_config.get("literal_temperature", 0.3),
                max_tokens=self.translator_config.get("literal_max_tokens", 200)
            )
            generated = clean_generated_text(generated)
            generated = generated.strip()
            # Restore citations and quotes if missing
            generated = self._restore_citations_and_quotes(generated, blueprint)
            return generated
        except Exception:
            # If normal path fails, use style-preserving fallback
            return self._translate_style_fallback(blueprint, author_name, style_dna, rhetorical_type, examples)

    def _translate_style_fallback(
        self,
        blueprint: SemanticBlueprint,
        author_name: str,
        style_dna: str,
        rhetorical_type: Optional[RhetoricalType] = None,
        examples: Optional[List[str]] = None
    ) -> str:
        """Style-preserving fallback: do style transfer without structural constraints.

        This is the "Hail Mary" attempt when blueprint is incomplete or normal translation fails.
        It still does style transfer, just without strict blueprint constraints.

        Args:
            blueprint: Semantic blueprint (with original_text).
            author_name: Target author name.
            style_dna: Style DNA description.
            rhetorical_type: Optional rhetorical mode.
            examples: Optional few-shot examples.

        Returns:
            Style-transferred text.
        """
        # Use OBSERVATION as default if not provided
        if rhetorical_type is None:
            rhetorical_type = RhetoricalType.OBSERVATION

        # Use empty examples if not provided
        if examples is None:
            examples = []

        # Build prompt using original-text-only template
        prompt = self._build_original_text_only_prompt(
            blueprint=blueprint,
            author_name=author_name,
            style_dna=style_dna,
            rhetorical_type=rhetorical_type,
            examples=examples
        )

        # Load system prompt
        system_prompt_template = _load_prompt_template("translator_system.md")
        system_prompt = system_prompt_template.format(
            author_name=author_name,
            style_dna=style_dna
        )

        try:
            generated = self.llm_provider.call(
                system_prompt=system_prompt,
                user_prompt=prompt,
                temperature=self.translator_config.get("literal_temperature", 0.5),  # Slightly higher for style
                max_tokens=self.translator_config.get("literal_max_tokens", 200)
            )
            generated = clean_generated_text(generated)
            generated = generated.strip()
            # Restore citations and quotes if missing
            generated = self._restore_citations_and_quotes(generated, blueprint)
            return generated
        except Exception:
            # Ultimate fallback: still try to do style transfer with minimal prompt
            # This should never happen, but if it does, at least attempt transformation
            minimal_prompt = f"""Rewrite this text in the style of {author_name}: "{blueprint.original_text}"

Style: {style_dna}

Do NOT copy the text verbatim. Transform it into the target style while preserving meaning."""
            try:
                generated = self.llm_provider.call(
                    system_prompt=f"You are {author_name}.",
                    user_prompt=minimal_prompt,
                    temperature=0.5,
                    max_tokens=200
                )
                generated = clean_generated_text(generated)
                generated = generated.strip()
                return generated
            except Exception:
                # Only return original if ALL attempts fail
                return blueprint.original_text

    def _check_acceptance(
        self,
        recall_score: float,
        precision_score: float,
        fluency_score: Optional[float] = None,
        overall_score: float = 0.0,
        pass_threshold: float = 0.9,
        original_text: str = "",
        generated_text: str = ""
    ) -> bool:
        """Check if draft should be accepted using Fluency Forgiveness logic.

        HARD GATE: Length heuristic - reject if input > 6 words and output < 4 words.
        HARD GATE: Fluency must be above minimum threshold (if provided).
        RULE 1: Recall is King. If we miss keywords, we fail.
        RULE 2: Precision is Flexible. If Recall is perfect, we can accept lower precision.
        Fallback: High overall score.

        Args:
            recall_score: Recall score (0-1)
            precision_score: Precision score (0-1)
            fluency_score: Optional fluency score (0-1). If None, fluency check is skipped.
            overall_score: Weighted overall score (0-1)
            pass_threshold: Default pass threshold
            original_text: Original input text (for length check)
            generated_text: Generated text (for length check)

        Returns:
            True if draft should be accepted, False otherwise
        """
        # HARD GATE: Length heuristic - reject if input > 6 words and output < 4 words
        # This catches "We touch breaks" type garbage output
        if original_text and generated_text:
            input_word_count = len(original_text.split())
            output_word_count = len(generated_text.split())
            if input_word_count > 6 and output_word_count < 4:
                return False  # Too short to be a valid translation

        # HARD GATE: Fluency must be above minimum (if fluency_score is provided)
        # Note: Fluency check removed from SemanticCritic in two-gate simplification
        # This check is kept for backward compatibility but defaults to passing if not provided
        if fluency_score is not None and fluency_score < 0.7:
            return False  # HARD REJECT regardless of other scores

        # RULE 1: Recall is King. If we miss keywords, we fail.
        if recall_score < 1.0:
            # Must have all keywords - use strict threshold
            return overall_score >= pass_threshold

        # RULE 2: Precision is Flexible.
        # If Recall is perfect, we can accept lower precision (fluency glue).
        if precision_score >= 0.80:
            return True

        # Fallback: High overall score
        return overall_score >= pass_threshold

    def _get_blueprint_text(self, blueprint: SemanticBlueprint) -> str:
        """Get text representation of blueprint for refinement prompt.

        Returns a concise summary of blueprint content.

        Args:
            blueprint: Semantic blueprint to extract text from.

        Returns:
            String representation of blueprint content.
        """
        if not blueprint.svo_triples and not blueprint.core_keywords:
            return blueprint.original_text

        subjects = blueprint.get_subjects()
        verbs = blueprint.get_verbs()
        objects = blueprint.get_objects()

        parts = []
        if subjects:
            parts.append(f"Subjects: {', '.join(subjects)}")
        if verbs:
            parts.append(f"Actions: {', '.join(verbs)}")
        if objects:
            parts.append(f"Objects: {', '.join(objects)}")

        return " | ".join(parts) if parts else blueprint.original_text

    def _generate_simplification(
        self,
        best_draft: str,
        blueprint: SemanticBlueprint,
        author_name: str,
        style_dna: str,
        rhetorical_type: RhetoricalType,
        critic: 'SemanticCritic',
        verbose: bool = False
    ) -> str:
        """Generate a simplified version when stuck at low scores.

        This is a "Hail Mary" attempt that strips the sentence down to basics
        when the evolution loop has stagnated at a low score.

        Args:
            best_draft: Current best draft (may be ignored in favor of blueprint)
            blueprint: Original semantic blueprint
            author_name: Target author name
            style_dna: Style DNA description
            rhetorical_type: Rhetorical mode
            critic: SemanticCritic instance for validation
            verbose: Enable verbose logging

        Returns:
            Simplified text generated from blueprint, or best_draft if simplification fails validation
        """
        # CRITICAL: Check if blueprint is incomplete
        # If incomplete, use original_text as source (not broken blueprint)
        if self._is_blueprint_incomplete(blueprint):
            # Use original text directly as the source
            source_text = blueprint.original_text
            blueprint_text = "N/A (Using original text due to incomplete blueprint)"
        else:
            # Use blueprint text as before
            blueprint_text = self._get_blueprint_text(blueprint)
            source_text = blueprint.original_text

        repair_prompt_template = _load_prompt_template("translator_repair.md")
        repair_prompt = repair_prompt_template.format(
            original_text=source_text,
            blueprint_text=blueprint_text
        )

        system_prompt_template = _load_prompt_template("translator_simplification_system.md")
        system_prompt = system_prompt_template.format(
            author_name=author_name,
            style_dna=style_dna
        )

        try:
            simplified = self.llm_provider.call(
                system_prompt=system_prompt,
                user_prompt=repair_prompt,
                temperature=0.2,  # Low temperature for repair
                max_tokens=self.translator_config.get("max_tokens", 200)
            )
            simplified = clean_generated_text(simplified)
            simplified = simplified.strip()
            # Restore citations and quotes if missing
            simplified = self._restore_citations_and_quotes(simplified, blueprint)

            # Validate simplification output - don't return broken fragments
            result = critic.evaluate(simplified, blueprint)
            # Note: fluency_score may not be present (removed in two-gate simplification)
            # Use recall_score as proxy for validation
            if result.get("recall_score", 0.0) < 0.7:
                if verbose:
                    print("  Simplification produced low-fluency output, reverting to best draft")
                return best_draft

            return simplified
        except Exception as e:
            # Fallback: return best draft if simplification fails
            if verbose:
                print(f"  Simplification failed with exception: {e}, reverting to best draft")
            return best_draft

    def _evolve_text(
        self,
        initial_draft: str,
        blueprint: SemanticBlueprint,
        author_name: str,
        style_dna: str,
        rhetorical_type: RhetoricalType,
        initial_score: float,
        initial_feedback: str,
        critic: 'SemanticCritic',
        verbose: bool = False,
        style_dna_dict: Optional[Dict[str, any]] = None,
        examples: Optional[List[str]] = None,
        force_semantic_injection: bool = False
    ) -> Tuple[str, float]:
        """Evolve text using population-based beam search with tournament selection.

        Uses a 3-pronged evolution strategy:
        1. Semantic Repair: Focuses on adding missing keywords (recall)
        2. Fluency Polish: Focuses on grammar and flow (fluency)
        3. Style Enhancement: Focuses on matching target voice (style)

        Each generation generates 3 candidates, evaluates them all, and selects the best
        using tournament selection with anti-regression (rejects candidates with lower recall)
        and elitism (only replaces parent if child is strictly better).

        Args:
            initial_draft: First generated draft
            blueprint: Original semantic blueprint
            author_name: Target author name
            style_dna: Style DNA description
            rhetorical_type: Rhetorical mode
            initial_score: Score from initial evaluation
            initial_feedback: Feedback from initial evaluation
            critic: SemanticCritic instance for evaluation
            verbose: Enable verbose logging

        Returns:
            Tuple of (best_draft, best_score)
        """
        # Initialize best draft and score
        best_draft = initial_draft
        best_score = initial_score
        best_feedback = initial_feedback

        # Initialize LLM Judge for ranking
        judge = LLMJudge(config_path=self.config_path)

        # Convergence tracking: track judge's selections
        last_judge_winner = None
        convergence_counter = 0
        convergence_threshold = 2  # Same winner for 2 rounds = converged

        # Load refinement config
        refinement_config = self.config.get("refinement", {})
        max_generations = refinement_config.get("max_generations", 3)
        pass_threshold = refinement_config.get("pass_threshold", 0.9)

        # Smart Patience parameters
        patience_counter = 0
        patience_threshold = refinement_config.get("patience_threshold", 3)
        patience_min_score = refinement_config.get("patience_min_score", 0.80)

        # Stagnation Breaker parameters (separate from patience)
        stagnation_counter = 0
        stagnation_threshold = 3

        # Dynamic temperature parameters
        current_temp = refinement_config.get("initial_temperature",
                                             refinement_config.get("refinement_temperature", 0.3))
        temperature_increment = refinement_config.get("temperature_increment", 0.2)
        max_temperature = refinement_config.get("max_temperature", 0.9)

        if verbose:
            print(f"  Evolution: Starting with score {best_score:.2f}")
            print(f"    Max generations: {max_generations}, Pass threshold: {pass_threshold}")
            print(f"    Patience: {patience_threshold} (min score: {patience_min_score})")
            print(f"    Initial temperature: {current_temp:.2f}")
            print(f"    Convergence threshold: {convergence_threshold} rounds")

        # Get blueprint text representation
        blueprint_text = self._get_blueprint_text(blueprint)

        # Extract style lexicon from style_dna_dict if available
        style_lexicon = None
        style_structure = None
        style_tone = None
        rag_example = None
        if style_dna_dict:
            style_lexicon = style_dna_dict.get("lexicon", [])
            style_structure = style_dna_dict.get("structure")
            style_tone = style_dna_dict.get("tone")
        # Get top RAG example for stylistic repair
        if examples and len(examples) > 0:
            rag_example = examples[0]

        # Get initial raw_score for fitness-based evolution (with style lexicon for density bonus)
        initial_eval = self.soft_scorer.evaluate_with_raw_score(best_draft, blueprint, style_lexicon=style_lexicon)
        best_raw_score = initial_eval.get("raw_score", best_score)

        # Track logic failures in evolution loop (Initialize OUTSIDE loop)
        logic_failure_count = 0
        max_logic_failures = 3

        # Evolution loop
        for gen in range(max_generations):
            # Check if we've reached acceptance criteria (using Fluency Forgiveness)
            # Evaluate current best draft to get recall/precision (with style whitelist)
            best_result = critic.evaluate(best_draft, blueprint, allowed_style_words=style_lexicon)
            if self._check_acceptance(
                recall_score=best_result["recall_score"],
                precision_score=best_result["precision_score"],
                fluency_score=best_result.get("fluency_score"),  # Optional - may not be present
                overall_score=best_score,
                pass_threshold=pass_threshold,
                original_text=blueprint.original_text,
                generated_text=best_draft
            ):
                if verbose:
                    print(f"  Evolution: Draft accepted (recall: {best_result['recall_score']:.2f}, precision: {best_result['precision_score']:.2f}, score: {best_score:.2f})")
                break

            if verbose:
                print(f"  Evolution Generation {gen + 1}/{max_generations}")
                print(f"    Current Score: {best_score:.2f}, Raw Score: {best_raw_score:.2f}")
                print(f"    Temperature: {current_temp:.2f}")

            try:
                # Step A: Diagnosis - Analyze current draft to select mutation strategy
                # If force_semantic_injection is True, prioritize semantic injection for rescue candidates
                if force_semantic_injection and gen == 0:
                    operator_type = OP_SEMANTIC_INJECTION
                    if verbose:
                        print(f"    Forcing semantic injection for rescue candidate repair")
                else:
                    operator_type = self._diagnose_draft(best_draft, blueprint, critic)

                    # Check if we're stuck in logic failure loop
                    # Evaluate current best draft to check for logic failure
                    current_result = critic.evaluate(best_draft, blueprint, allowed_style_words=style_lexicon)
                    if current_result.get("logic_fail", False):
                        logic_failure_count += 1
                        if logic_failure_count >= max_logic_failures:
                            # CRITICAL: Force semantic injection to repair logic by re-inserting
                            # missing causal words that restore original relationship
                            operator_type = OP_SEMANTIC_INJECTION
                            if verbose:
                                print(f"    Forcing logic repair after {logic_failure_count} consecutive failures")
                            # Note: Counter resets only when logic_fail becomes False (see below)
                    else:
                        # Reset counter only when logic failure is resolved
                        if logic_failure_count > 0:
                            if verbose:
                                print(f"    Logic failure resolved, resetting counter")
                            logic_failure_count = 0

                # Use dynamic style if style lexicon is available and we're doing style polish
                if style_lexicon and operator_type == OP_STYLE_POLISH:
                    operator_type = OP_DYNAMIC_STYLE  # Use OP_DYNAMIC_STYLE when style DNA is available
                if verbose:
                    print(f"    Diagnosis: Selected operator '{operator_type}'")

                # Step B: Population - Generate candidates using ratio-based strategy
                # Get fresh generation ratio from config
                evolutionary_config = self.config.get("evolutionary", {})
                fresh_ratio = evolutionary_config.get("fresh_generation_ratio", 0.33)
                num_candidates = 3
                num_improvements = max(1, int(num_candidates * (1 - fresh_ratio)))
                num_fresh = num_candidates - num_improvements

                if verbose:
                    print(f"    Population strategy: {num_improvements} improvements + {num_fresh} fresh (ratio: {fresh_ratio:.2f})")

                # Generate improvements to promising candidates
                improvement_candidates = []
                if num_improvements > 0:
                    improvement_candidates = self._generate_population_with_operator(
                        parent_draft=best_draft,
                        blueprint=blueprint,
                        author_name=author_name,
                        style_dna=style_dna,
                        rhetorical_type=rhetorical_type,
                        operator_type=operator_type,
                        temperature=current_temp,
                        num_candidates=num_improvements,
                        verbose=verbose,
                        style_lexicon=style_lexicon,
                        style_structure=style_structure,
                        style_tone=style_tone,
                        rag_example=rag_example
                    )

                # Generate fresh candidates from scratch
                fresh_candidates = []
                if num_fresh > 0:
                    fresh_candidates = self._generate_fresh_candidates(
                        blueprint=blueprint,
                        author_name=author_name,
                        style_dna=style_dna,
                        rhetorical_type=rhetorical_type,
                        temperature=min(current_temp + 0.2, 0.9),  # Higher temp for diversity
                        num_candidates=num_fresh,
                        verbose=verbose,
                        style_lexicon=style_lexicon,
                        examples=examples
                    )

                # Combine all candidates
                candidates = improvement_candidates + fresh_candidates

                # Step C: Scoring - Get raw_score for all candidates
                scored_candidates = []
                for strategy, candidate_text in candidates:
                    if not candidate_text or not candidate_text.strip():
                        continue

                    # Get both critic evaluation and raw_score (with style whitelist and density bonus)
                    candidate_result = critic.evaluate(candidate_text, blueprint, allowed_style_words=style_lexicon)
                    candidate_eval = self.soft_scorer.evaluate_with_raw_score(candidate_text, blueprint, style_lexicon=style_lexicon)
                    candidate_raw_score = candidate_eval.get("raw_score", candidate_result.get("score", 0.0))

                    scored_candidates.append({
                        "strategy": strategy,
                        "text": candidate_text,
                        "score": candidate_result.get("score", 0.0),
                        "raw_score": candidate_raw_score,
                        "pass": candidate_result.get("pass", False),
                        "result": candidate_result,
                        "recall": candidate_result.get("recall_score", 0.0)
                    })

                # Step D: Selection - Pick candidate with highest raw_score if it improves over parent
                # CRITICAL: Accept improvement even if pass=False (fitness-based selection)
                best_candidate = None
                candidate_score = best_score
                candidate_raw_score = best_raw_score
                winning_strategy = None
                candidate_result = best_result

                if scored_candidates:
                    # Find candidate with highest raw_score
                    best_scored = max(scored_candidates, key=lambda c: c["raw_score"])

                    # Evolution Logic: Accept if raw_score improves (even if pass=False)
                    # CRITICAL: Also accept if score improves significantly even if raw_score is same
                    # This handles cases where critic gives good scores (0.84, 0.80) but pass=False
                    score_improvement = best_scored["score"] > best_score
                    raw_score_improvement = best_scored["raw_score"] > best_raw_score

                    if raw_score_improvement or (score_improvement and best_scored["score"] > 0.7):
                        best_candidate = best_scored["text"]
                        candidate_score = best_scored["score"]
                        candidate_raw_score = best_scored["raw_score"]
                        winning_strategy = best_scored["strategy"]
                        candidate_result = best_scored["result"]

                        if verbose:
                            reason = "raw_score" if raw_score_improvement else "score"
                            print(f"    ✓ Fitness Improvement: {winning_strategy} "
                                  f"({reason}: {candidate_raw_score if raw_score_improvement else candidate_score:.2f} > "
                                  f"{best_raw_score if raw_score_improvement else best_score:.2f}, "
                                  f"pass={best_scored['pass']}, score={best_scored['score']:.2f})")
                    else:
                        if verbose:
                            print(f"    ↻ No fitness improvement "
                                  f"(best raw_score: {best_scored['raw_score']:.2f} <= {best_raw_score:.2f}, "
                                  f"best score: {best_scored['score']:.2f} <= {best_score:.2f})")

                # Check if we have a winner (fitness improvement found)
                if best_candidate is not None:
                    # Improvement found: reset temperature, patience, and stagnation counter
                    current_temp = refinement_config.get("initial_temperature",
                                                         refinement_config.get("refinement_temperature", 0.3))
                    patience_counter = 0
                    stagnation_counter = 0

                    # Convergence check: same winner 2 rounds in a row?
                    if best_candidate == last_judge_winner:
                        convergence_counter += 1
                        if verbose:
                            print(f"    Convergence: Same candidate selected {convergence_counter}/{convergence_threshold} rounds")
                    else:
                        convergence_counter = 0
                        last_judge_winner = best_candidate

                    best_draft = best_candidate
                    best_score = candidate_score
                    best_raw_score = candidate_raw_score  # Update raw_score for next iteration
                    best_feedback = candidate_result["feedback"]
                    best_result = candidate_result
                    if verbose:
                        print(f"    ✓ Fitness Winner: {winning_strategy} "
                              f"(raw_score={best_raw_score:.2f}, score={best_score:.2f}, "
                              f"pass={candidate_result.get('pass', False)}, temp reset to {current_temp:.2f})")

                    # Convergence: Same winner for 2 rounds → stop
                    if convergence_counter >= convergence_threshold:
                        if verbose:
                            print(f"  Evolution: CONVERGED - Judge selected same candidate for {convergence_threshold} rounds. Stopping.")
                        break

                    # Check if improved draft meets acceptance criteria
                    if self._check_acceptance(
                        recall_score=candidate_result["recall_score"],
                        precision_score=candidate_result["precision_score"],
                        fluency_score=candidate_result.get("fluency_score"),  # Optional - may not be present
                        overall_score=candidate_score,
                        pass_threshold=pass_threshold,
                        original_text=blueprint.original_text,
                        generated_text=best_candidate
                    ):
                        if verbose:
                            print(f"  Evolution: Draft accepted after improvement "
                                  f"(recall: {candidate_result['recall_score']:.2f}, "
                                  f"precision: {candidate_result['precision_score']:.2f}, "
                                  f"score: {candidate_score:.2f})")
                        break
                else:
                    # No improvement (elitism kept parent): increment patience, stagnation, and increase temperature
                    patience_counter += 1
                    stagnation_counter += 1
                    current_temp = min(current_temp + temperature_increment, max_temperature)
                    if verbose:
                        print(f"    ↻ Stuck at {best_score:.2f}, increasing temperature to {current_temp:.2f} (patience: {patience_counter}/{patience_threshold}, stagnation: {stagnation_counter}/{stagnation_threshold})")

                    # Stagnation Breaker: triggers regardless of score after 3 non-improvements
                    if stagnation_counter >= stagnation_threshold:
                        if verbose:
                            print(f"  DEBUG: Stagnation detected (3 gens at {best_score:.2f}).")

                        # Check for logic failure before accepting stagnation exit
                        logic_fail = False
                        try:
                            stagnation_result = critic.evaluate(best_draft, blueprint, allowed_style_words=style_lexicon)
                            logic_fail = stagnation_result.get("logic_fail", False)
                        except Exception:
                            pass

                        if best_score >= 0.85 and not logic_fail:
                            if verbose:
                                print("  DEBUG: Score is acceptable. Early exit.")
                            break
                        elif logic_fail:
                            if verbose:
                                print(f"  DEBUG: Logic failure detected (score {best_score:.2f}), continuing evolution.")
                        else:
                            # Never trade a good score (>= 0.5) for a potentially worse simplification
                            # Only attempt simplification if best_score is truly bad (< 0.5)
                            if best_score < 0.5:
                                if verbose:
                                    print("  DEBUG: Score is very low (< 0.5). Attempting 'Simplification Pivot'...")
                                # Try one last radical simplification before giving up
                                final_attempt = self._generate_simplification(best_draft, blueprint, author_name, style_dna, rhetorical_type, critic, verbose)
                                return (final_attempt, best_score)
                            else:
                                # Return the best draft we have (B-grade is better than risking F-grade)
                                if verbose:
                                    print(f"  DEBUG: Returning best draft (score: {best_score:.2f}). Not risking simplification.")
                                return (best_draft, best_score)

                    # Smart Patience: early exit if stuck at good enough score
                    if patience_counter >= patience_threshold and best_score >= patience_min_score:
                        if verbose:
                            print(f"  Evolution converged at {best_score:.2f}. Early exit triggered (patience: {patience_counter})")
                        break

            except Exception as e:
                if verbose:
                    print(f"    ✗ Evolution Failed: Exception during refinement: {e}")
                    import traceback
                    traceback.print_exc()
                # Don't continue - let it fall through to next generation
                # This ensures we don't silently fail
                continue

        if verbose:
            print(f"  Evolution: Final score {best_score:.2f} (improvement: {best_score - initial_score:+.2f})")

        # Soft Pass Logic: Only accept if style is present OR score is very high
        # Ban boring sentences - force evolution to continue if no style
        # LOGIC VETO: Never soft pass if logic failure detected
        style_lexicon = None
        if style_dna_dict and isinstance(style_dna_dict, dict):
            style_lexicon = style_dna_dict.get("lexicon")

        style_density = 0.0
        if style_lexicon and best_draft:
            style_density = self._calculate_style_density(best_draft, style_lexicon)

        # Check for logic failure in the current best result
        logic_fail = False
        try:
            current_result = critic.evaluate(best_draft, blueprint, allowed_style_words=style_lexicon)
            logic_fail = current_result.get("logic_fail", False)
        except Exception:
            pass  # If evaluation fails, continue without logic check

        # Conditional Soft Pass:
        # - If style present (density > 0.1): Accept if score >= 0.85 AND no logic failure
        # - If boring (density == 0): Only accept if score >= 0.95 AND no logic failure
        # - NEVER accept if logic_fail is True (even with high score)
        if best_score >= 0.85 and not logic_fail:
            if style_density > 0.1:
                # Style is present, accept early
                if verbose and best_score < pass_threshold:
                    print(f"  Evolution: Soft pass accepted (score {best_score:.2f} >= 0.85, style_density {style_density:.2f} > 0.1)")
                return (best_draft, best_score)
            elif best_score >= 0.95:
                # Very high score even without style, accept
                if verbose:
                    print(f"  Evolution: Soft pass accepted (score {best_score:.2f} >= 0.95, style_density {style_density:.2f})")
                return (best_draft, best_score)
            else:
                # Boring sentence with mediocre score - reject soft pass, keep evolving
                if verbose:
                    print(f"  Evolution: Soft pass rejected (score {best_score:.2f} but style_density {style_density:.2f} <= 0.1, requiring >= 0.95 for boring sentences)")
        elif logic_fail:
            # Logic failure detected - reject soft pass regardless of score
            if verbose:
                print(f"  Evolution: Soft pass rejected (logic failure detected, score {best_score:.2f})")

        return (best_draft, best_score)


    def _calculate_composite_score(self, recall: float, precision: float, fluency: float) -> float:
        """Calculate weighted composite score prioritizing recall.

        Formula: (recall * 2 + fluency + precision) / 4

        Args:
            recall: Recall score (0-1).
            precision: Precision score (0-1).
            fluency: Fluency score (0-1).

        Returns:
            Composite score (0-1).
        """
        return (recall * 2.0 + fluency + precision) / 4.0

    def _diagnose_draft(
        self,
        draft: str,
        blueprint: SemanticBlueprint,
        critic: 'SemanticCritic'
    ) -> str:
        """Diagnose the draft to select mutation strategy.

        Args:
            draft: Current draft to diagnose.
            blueprint: Original semantic blueprint.
            critic: SemanticCritic instance.

        Returns:
            Mutation operator type (OP_SEMANTIC_INJECTION, OP_GRAMMAR_REPAIR, or OP_STYLE_POLISH).
        """
        result = critic.evaluate(draft, blueprint)
        recall = result.get("recall_score", 0.0)
        fluency = result.get("fluency_score", 1.0)  # Default to 1.0 if not present (fluency checks removed)
        logic_fail = result.get("logic_fail", False)

        # Diagnosis logic:
        # - If logic failure: Force semantic injection to fix meaning
        # - If recall < 1.0: Missing keywords → Semantic Injection
        # - Elif fluency < 0.8: Grammar issues → Grammar Repair (only if fluency_score available)
        # - Else: Style needs enhancement → Style Polish
        if logic_fail:
            return OP_SEMANTIC_INJECTION  # Force semantic injection to fix logic
        elif recall < 1.0:
            return OP_SEMANTIC_INJECTION
        elif fluency is not None and fluency < 0.8:
            return OP_GRAMMAR_REPAIR
        else:
            return OP_STYLE_POLISH

    def _generate_population_with_operator(
        self,
        parent_draft: str,
        blueprint: SemanticBlueprint,
        author_name: str,
        style_dna: str,
        rhetorical_type: RhetoricalType,
        operator_type: str,
        temperature: float = 0.6,
        num_candidates: int = 3,
        verbose: bool = False,
        style_lexicon: Optional[List[str]] = None,
        style_structure: Optional[str] = None,
        style_tone: Optional[str] = None,
        rag_example: Optional[str] = None
    ) -> List[Tuple[str, str]]:
        """Generate population using a specific mutation operator.

        Args:
            parent_draft: Current best draft.
            blueprint: Original semantic blueprint.
            author_name: Target author name.
            style_dna: Style DNA description.
            rhetorical_type: Rhetorical mode.
            operator_type: Mutation operator type.
            temperature: Temperature for generation.
            num_candidates: Number of candidates to generate.
            verbose: Enable verbose logging.

        Returns:
            List of (strategy_name, candidate_text) tuples.
        """
        operator = get_operator(operator_type)
        candidates = []

        for i in range(num_candidates):
            try:
                if verbose:
                    print(f"    Generating {operator_type} candidate {i+1}/{num_candidates}...")

                # Pass style DNA to operators that support it
                generate_kwargs = {
                    "current_draft": parent_draft,
                    "blueprint": blueprint,
                    "author_name": author_name,
                    "style_dna": style_dna,
                    "rhetorical_type": rhetorical_type,
                    "llm_provider": self.llm_provider,
                    "temperature": temperature,
                    "max_tokens": self.translator_config.get("max_tokens", 300)
                }
                # Add style DNA parameters if operator supports them
                if hasattr(operator, 'generate') and operator_type in [OP_STYLE_POLISH, OP_DYNAMIC_STYLE]:
                    generate_kwargs["style_lexicon"] = style_lexicon
                    generate_kwargs["style_structure"] = style_structure
                    if operator_type == OP_DYNAMIC_STYLE:
                        generate_kwargs["style_tone"] = style_tone

                # Pass style_lexicon and rag_example to semantic injection for stylistic repair
                if operator_type == OP_SEMANTIC_INJECTION:
                    generate_kwargs["style_lexicon"] = style_lexicon
                    generate_kwargs["rag_example"] = rag_example

                candidate = operator.generate(**generate_kwargs)

                # Restore citations and quotes
                candidate = self._restore_citations_and_quotes(candidate, blueprint)

                if candidate and candidate.strip():
                    candidates.append((operator_type, candidate))
                else:
                    if verbose:
                        print(f"    ✗ {operator_type} candidate {i+1} failed (empty)")
            except Exception as e:
                if verbose:
                    print(f"    ✗ {operator_type} candidate {i+1} failed: {e}")

        return candidates

    def _generate_fresh_candidates(
        self,
        blueprint: SemanticBlueprint,
        author_name: str,
        style_dna: str,
        rhetorical_type: RhetoricalType,
        temperature: float = 0.8,
        num_candidates: int = 1,
        verbose: bool = False,
        style_lexicon: Optional[List[str]] = None,
        examples: Optional[List[str]] = None
    ) -> List[Tuple[str, str]]:
        """Generate fresh candidates from scratch (not based on parents).

        Generates completely new sentences from the blueprint to avoid local maxima.
        Uses high temperature for diversity.

        Args:
            blueprint: Original semantic blueprint.
            author_name: Target author name.
            style_dna: Style DNA description.
            rhetorical_type: Rhetorical mode.
            temperature: Temperature for generation (default 0.8 for diversity).
            num_candidates: Number of fresh candidates to generate.
            verbose: Enable verbose logging.
            style_lexicon: Optional list of style words.
            examples: Optional few-shot examples.

        Returns:
            List of ("FRESH", candidate_text) tuples.
        """
        candidates = []

        # Build a fresh generation prompt
        subjects_list = blueprint.get_subjects()[:5] if blueprint.get_subjects() else []
        verbs_list = blueprint.get_verbs()[:5] if blueprint.get_verbs() else []
        objects_list = blueprint.get_objects()[:5] if blueprint.get_objects() else []

        subjects = ", ".join(subjects_list) if subjects_list else "None"
        verbs = ", ".join(verbs_list) if verbs_list else "None"
        objects = ", ".join(objects_list) if objects_list else "None"
        lexicon_text = ", ".join(style_lexicon[:20]) if style_lexicon else "None"

        core_keywords = list(blueprint.core_keywords)[:10] if blueprint.core_keywords else []
        keywords_text = ", ".join(core_keywords) if core_keywords else "None"

        examples_text = ""
        if examples:
            examples_preview = "\n".join([f"- \"{ex}\"" for ex in examples[:3]])
            examples_text = f"\n\n**STYLE EXAMPLES (Match this voice):**\n{examples_preview}"

        fresh_prompt = f"""Generate {num_candidates} completely NEW and DISTINCT sentences from scratch. Do NOT base these on any existing text. Explore different approaches to express the same meaning.

### MEANING (MUST PRESERVE):
- Original: "{blueprint.original_text}"
- Subjects: {subjects}
- Verbs: {verbs}
- Objects: {objects}
- Core Keywords (MUST include): {keywords_text}

### STYLE REQUIREMENTS:
- Author: {author_name}
- Style DNA: {style_dna}
- Vocabulary to use: {lexicon_text}{examples_text}

### TASK:
Generate {num_candidates} fresh variations that:
1. Express the EXACT same meaning as the original
2. Include ALL core keywords
3. Use the author's distinctive vocabulary and style
4. Explore DIFFERENT sentence structures and phrasings
5. Be creative - avoid repeating patterns from previous generations

### OUTPUT FORMAT:
Output PURE JSON. A single list of strings:
[
  "Fresh variation 1...",
  "Fresh variation 2...",
  ...
  "Fresh variation {num_candidates}..."
]
"""

        try:
            if verbose:
                print(f"    Generating {num_candidates} fresh candidate(s) from scratch...")

            response = self.llm_provider.call(
                system_prompt="You are a creative sentence generator. Output ONLY valid JSON.",
                user_prompt=fresh_prompt,
                model_type="editor",
                require_json=True,
                temperature=temperature,
                max_tokens=self.translator_config.get("max_tokens", 400)
            )

            # Parse JSON response
            fresh_candidates = json.loads(response)
            if isinstance(fresh_candidates, list):
                fresh_candidates = [c.strip() for c in fresh_candidates if c and c.strip()]
                for candidate in fresh_candidates[:num_candidates]:
                    # Restore citations and quotes
                    candidate = self._restore_citations_and_quotes(candidate, blueprint)
                    if candidate and candidate.strip():
                        candidates.append(("FRESH", candidate))
            else:
                extracted = self._extract_json_list(response)
                for candidate in extracted[:num_candidates]:
                    candidate = self._restore_citations_and_quotes(candidate, blueprint)
                    if candidate and candidate.strip():
                        candidates.append(("FRESH", candidate))

            if verbose:
                print(f"    ✓ Generated {len(candidates)} fresh candidate(s)")

        except Exception as e:
            if verbose:
                print(f"    ✗ Fresh generation failed: {e}")

        return candidates


    def _remove_phantom_citations(self, text: str, expected_citations: set) -> str:
        """Remove phantom citations (citations not in original input) from text.

        Args:
            text: Text that may contain phantom citations
            expected_citations: Set of valid citation strings from original input

        Returns:
            Text with phantom citations removed
        """
        if not text:
            return text

        citation_pattern = r'\[\^\d+\]'
        found_citations = set(re.findall(citation_pattern, text))
        phantom_citations = found_citations - expected_citations

        if phantom_citations:
            # Remove each phantom citation from text
            for phantom in phantom_citations:
                # Remove citation with optional space before and after
                text = re.sub(r'\s*' + re.escape(phantom) + r'\s*', ' ', text)
                # Clean up any double spaces
                text = re.sub(r'\s+', ' ', text)

        return text

    def _restore_citations_and_quotes(self, generated: str, blueprint: SemanticBlueprint) -> str:
        """Ensure all citations and quotes from blueprint are present in generated text.

        Citations can be appended to the end of the sentence if missing.
        Quotes must be present exactly - if missing or modified, this indicates
        a critical failure that should be caught by the critic.

        Also removes any non-standard citation formats (e.g., (Author, Year), (Smith 42)).

        CRITICAL: Only restores citations that actually exist in the original input text.
        This prevents phantom citations from being added.

        Args:
            generated: Generated text from LLM.
            blueprint: Original blueprint with citations and quotes.

        Returns:
            Generated text with citations restored (if missing) and non-standard formats removed.
        """
        if not generated:
            return generated

        # Remove all non-standard citation formats BEFORE checking for valid citations
        # Remove (Author, Year, p. #) format
        generated = re.sub(r'\([A-Z][a-z]+,?\s+\d{4}(?:,\s*p\.\s*#?)?\)', '', generated)
        # Remove (Author, Year, p. #) template pattern
        generated = re.sub(r'\(Author,?\s+Year,?\s+p\.\s*#\)', '', generated, flags=re.IGNORECASE)
        # Remove (Smith 42) format
        generated = re.sub(r'\([A-Z][a-z]+\s+\d+\)', '', generated)

        # Extract valid citations from generated text (only [^number] format)
        citation_pattern = r'\[\^\d+\]'
        generated_citations = set(re.findall(citation_pattern, generated))

        # CRITICAL FIX: Verify citations actually exist in original input text
        # Extract citations from original text to ensure we only restore real ones
        # Ensure original_text is a string (not a tuple)
        original_text = blueprint.original_text if isinstance(blueprint.original_text, str) else str(blueprint.original_text)
        original_citations = set(re.findall(citation_pattern, original_text))

        # Remove phantom citations from generated text (citations not in original input)
        phantom_citations = generated_citations - original_citations
        if phantom_citations:
            # Remove each phantom citation from generated text
            for phantom in phantom_citations:
                # Remove the citation, handling spacing
                generated = re.sub(re.escape(phantom) + r'\s*', '', generated)
                generated = re.sub(r'\s+' + re.escape(phantom), '', generated)
            # Re-extract citations after removal
            generated_citations = set(re.findall(citation_pattern, generated))

        # Only consider citations that are both in the blueprint AND in the original text
        # This prevents phantom citations from being restored
        valid_blueprint_citations = set([cit[0] for cit in blueprint.citations
                                         if cit[0] in original_citations])

        # Check which valid citations from blueprint are missing from generated text
        missing_citations = valid_blueprint_citations - generated_citations

        # Append missing citations to end of sentence
        if missing_citations:
            # Remove any trailing punctuation that might interfere
            generated = generated.rstrip('.!?')
            # Append citations
            citations_to_add = ' '.join(sorted(missing_citations))
            generated = f"{generated} {citations_to_add}"

        # Note: We don't restore quotes here because they must be exact word-for-word.
        # If quotes are missing or modified, the critic will catch it and fail validation.
        # This is intentional - quotes cannot be automatically restored.

        return generated

    def _extract_sentence_templates(self, teacher_example: str, verbose: bool = False) -> List[str]:
        """Extract sentence-level templates from a teacher example.

        Splits the teacher example into sentences and extracts a bleached template
        for each sentence using the structure extractor.

        Args:
            teacher_example: Multi-sentence paragraph text to extract templates from.
            verbose: Whether to print debug information.

        Returns:
            List of template strings (one per sentence).
        """
        if not teacher_example or not teacher_example.strip():
            if verbose:
                print("  Warning: Empty teacher example, returning fallback template")
            return ["[NP] [VP] [NP]."]

        try:
            from nltk.tokenize import sent_tokenize
        except ImportError:
            # Fallback: simple sentence splitting
            sentences = re.split(r'[.!?]+\s+', teacher_example)
            sentences = [s.strip() for s in sentences if s.strip()]
        else:
            sentences = sent_tokenize(teacher_example)
            sentences = [s.strip() for s in sentences if s.strip()]

        if not sentences:
            if verbose:
                print("  Warning: No sentences found, returning fallback template")
            return ["[NP] [VP] [NP]."]

        templates = []
        for i, sentence in enumerate(sentences):
            try:
                template = self.structure_extractor.extract_template(sentence)
                if template and template.strip():
                    templates.append(template)
                else:
                    if verbose:
                        print(f"  Warning: Empty template for sentence {i+1}, using fallback")
                    templates.append("[NP] [VP] [NP].")
            except Exception as e:
                if verbose:
                    print(f"  Warning: Template extraction failed for sentence {i+1}: {e}, using fallback")
                templates.append("[NP] [VP] [NP].")

        if not templates:
            if verbose:
                print("  Warning: No templates extracted, returning fallback template")
            return ["[NP] [VP] [NP]."]

        if verbose:
            print(f"  Extracted {len(templates)} sentence templates")

        return templates

    def _extract_narrative_arc(self, teacher_text: str, verbose: bool = False) -> List[str]:
        """Extract narrative roles for each sentence in teacher example.

        Analyzes the teacher paragraph sentence by sentence and identifies
        the Narrative Role (rhetorical function) of each sentence.

        Args:
            teacher_text: Multi-sentence paragraph text to analyze.
            verbose: Whether to print debug information.

        Returns:
            List of narrative role strings (one per sentence).
        """
        if not teacher_text or not teacher_text.strip():
            if verbose:
                print("  Warning: Empty teacher text, returning fallback roles")
            return ["BODY"]

        try:
            from nltk.tokenize import sent_tokenize
        except ImportError:
            # Fallback: simple sentence splitting
            sentences = re.split(r'[.!?]+\s+', teacher_text)
            sentences = [s.strip() for s in sentences if s.strip()]
        else:
            sentences = sent_tokenize(teacher_text)
            sentences = [s.strip() for s in sentences if s.strip()]

        if not sentences:
            if verbose:
                print("  Warning: No sentences found, returning fallback roles")
            return ["BODY"]

        # Build prompt for narrative role extraction
        system_prompt = """You are a Narrative Analyst. Your task is to identify the rhetorical function (Narrative Role) of each sentence in a paragraph.

Analyze the paragraph sentence by sentence and identify the Narrative Role of each sentence.

For each sentence, define its **Narrative Role** in the author's argument. Choose from:
- Setup/Introduction
- Observation/Evidence
- Theoretical Analysis
- Rebuttal/Counterargument
- Historical Context
- Transition
- Conclusion/Call to Action

Output: JSON array of strings, one role per sentence in order.
Example: ["Observation of material conditions", "Theoretical implication", "Final deduction"]"""

        user_prompt = f"""Analyze this paragraph sentence by sentence and identify the Narrative Role of each sentence.

Text: {teacher_text}

For each sentence, define its **Narrative Role** in the author's argument. Choose from:
- Setup/Introduction
- Observation/Evidence
- Theoretical Analysis
- Rebuttal/Counterargument
- Historical Context
- Transition
- Conclusion/Call to Action

Output: JSON array of strings, one role per sentence in order.
Example: ["Observation of material conditions", "Theoretical implication", "Final deduction"]"""

        max_retries = self.llm_provider_config.get("max_retries", 3)
        retry_delay = self.llm_provider_config.get("retry_delay", 2)

        for attempt in range(max_retries):
            try:
                if verbose:
                    if attempt > 0:
                        print(f"  🔄 Retry attempt {attempt + 1}/{max_retries} for narrative arc extraction")
                    else:
                        print(f"  📤 Calling LLM for narrative arc extraction ({len(sentences)} sentences)")

                response = self.llm_provider.call(
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    model_type="editor",
                    require_json=True,
                    temperature=0.3,  # Low temperature for consistent classification
                    max_tokens=self.translator_config.get("max_tokens", 500)
                )

                if verbose:
                    print(f"  📥 Received response ({len(response)} chars)")

                # Parse JSON response
                narrative_roles = json.loads(response)
                if not isinstance(narrative_roles, list):
                    raise ValueError(f"Expected JSON array, got {type(narrative_roles)}")

                # Validate length matches sentence count
                if len(narrative_roles) != len(sentences):
                    if verbose:
                        print(f"  ⚠ Warning: Narrative roles count ({len(narrative_roles)}) doesn't match sentence count ({len(sentences)}), padding or truncating")
                    # Pad or truncate to match sentence count
                    if len(narrative_roles) < len(sentences):
                        # Pad with "BODY" for missing roles
                        narrative_roles.extend(["BODY"] * (len(sentences) - len(narrative_roles)))
                    else:
                        # Truncate to match
                        narrative_roles = narrative_roles[:len(sentences)]

                # Validate all roles are strings
                narrative_roles = [str(role).strip() if role else "BODY" for role in narrative_roles]

                if verbose:
                    print(f"  ✅ Extracted {len(narrative_roles)} narrative roles: {narrative_roles}")

                return narrative_roles

            except json.JSONDecodeError as e:
                if verbose:
                    print(f"  ⚠ JSON decode error: {e}")
                if attempt < max_retries - 1:
                    time.sleep(retry_delay)
                    continue
                else:
                    if verbose:
                        print("  ⚠ Failed to parse JSON after retries, using fallback roles")
                    return ["BODY"] * len(sentences)

            except Exception as e:
                if verbose:
                    print(f"  ⚠ Error extracting narrative arc: {e}")
                if attempt < max_retries - 1:
                    time.sleep(retry_delay)
                    continue
                else:
                    if verbose:
                        print("  ⚠ Failed to extract narrative arc after retries, using fallback roles")
                    return ["BODY"] * len(sentences)

        # Final fallback
        if verbose:
            print("  ⚠ Using fallback narrative roles")
        return ["BODY"] * len(sentences)

    def _get_prop_count_bucket(self, prop_count: int) -> int:
        """Bucket proposition counts for caching.

        Groups similar proposition counts together:
        - 1-2 props -> bucket 2
        - 3-4 props -> bucket 4
        - 5-7 props -> bucket 6
        - 8+ props -> bucket 10

        Args:
            prop_count: Number of propositions

        Returns:
            Bucketed proposition count
        """
        if prop_count <= 2:
            return 2
        elif prop_count <= 4:
            return 4
        elif prop_count <= 7:
            return 6
        else:
            return 10

    def _retrieve_robust_skeleton(
        self,
        rhetorical_type: str,
        author: str,
        prop_count: int,
        atlas,
        verbose: bool = False
    ) -> Tuple[str, List[str]]:
        """Retrieve a robust skeleton (teacher example) with sentence count compatible with proposition count.

        Fetches paragraphs from Atlas and filters by sentence count compatibility.
        Guarantees to always return a non-empty list of templates.

        Args:
            rhetorical_type: Rhetorical type for filtering examples.
            author: Author name for filtering examples.
            prop_count: Number of propositions (used to determine target sentence count).
            atlas: StyleAtlas instance for retrieving examples.
            verbose: Whether to print debug information.

        Returns:
            Tuple of (teacher_example, templates) where templates is a non-empty list.

        Raises:
            ValueError: If no examples can be retrieved from Atlas.
        """
        # Convert rhetorical_type to enum for consistent caching
        from src.atlas.rhetoric import RhetoricalType
        try:
            # Try to convert string to RhetoricalType enum if needed
            if isinstance(rhetorical_type, str):
                # Try to find matching enum value
                rt_enum = None
                for rt in RhetoricalType:
                    if rt.value.lower() == rhetorical_type.lower():
                        rt_enum = rt
                        break
                if rt_enum is None:
                    rt_enum = RhetoricalType.OBSERVATION  # Fallback
                rhetorical_type = rt_enum
        except Exception:
            # If conversion fails, use OBSERVATION as fallback
            rhetorical_type = RhetoricalType.OBSERVATION

        # Generate cache key
        prop_count_bucket = self._get_prop_count_bucket(prop_count)
        cache_key = (author, rhetorical_type.value, prop_count_bucket)

        # Check cache
        if cache_key in self._skeleton_cache:
            if verbose:
                print(f"  Cache hit for skeleton (author={author}, type={rhetorical_type.value}, bucket={prop_count_bucket})")
            return self._skeleton_cache[cache_key]

        if verbose:
            print(f"  Retrieving 20 examples for skeleton selection...")

        # Get examples with metadata to access template scores
        raw_examples = atlas.get_examples_by_rhetoric(
            rhetorical_type,
            top_k=20,
            author_name=author,
            query_text=None  # Don't filter by input length
        )

        if not raw_examples:
            # Fallback: try any examples from author
            raw_examples = atlas.get_examples_by_rhetoric(
                RhetoricalType.OBSERVATION,
                top_k=20,
                author_name=author,
                query_text=None
            )

        if not raw_examples:
            raise ValueError(f"No examples found in Atlas for author '{author}' and rhetorical type '{rhetorical_type}'")

        if verbose:
            print(f"  Retrieved {len(raw_examples)} examples from Atlas")

        # Get metadata for examples to access template scores
        # Try to get collection and metadata
        example_metadata_map = {}
        try:
            if hasattr(atlas, '_collection'):
                collection = atlas._collection
                # Get all entries to find metadata
                # This is a bit inefficient but necessary to get template metadata
                all_results = collection.get(where={"author_id": author} if author else None, limit=1000)
                if all_results and all_results.get('documents'):
                    for idx, doc in enumerate(all_results['documents']):
                        if doc in raw_examples:
                            metadata = all_results['metadatas'][idx] if all_results['metadatas'] else {}
                            example_metadata_map[doc] = metadata
        except Exception as e:
            if verbose:
                print(f"  ⚠ Could not retrieve metadata: {e}, continuing without template filtering")

        # Filter by sentence count compatibility and template quality
        from nltk.tokenize import sent_tokenize

        best_example = None
        best_score = float('inf')
        longest_example = None
        longest_sentence_count = 0

        # Score examples based on template metadata
        scored_examples = []

        for example in raw_examples:
            try:
                example_sentences = sent_tokenize(example)
                sentence_count = len([s for s in example_sentences if s.strip()])

                if sentence_count < 2:
                    continue  # Skip fragments

                # Track longest example as fallback
                if sentence_count > longest_sentence_count:
                    longest_example = example
                    longest_sentence_count = sentence_count

                # Get template metadata if available
                metadata = example_metadata_map.get(example, {})
                skeletons_json = metadata.get('skeletons', '[]')

                # Calculate template quality score
                template_quality_score = 0.0
                ideal_prop_match_score = 0.0

                try:
                    skeletons = json.loads(skeletons_json)
                    if isinstance(skeletons, list):
                        # Find best template in this example
                        best_template_score = 0
                        best_template_capacity = None

                        for skeleton in skeletons:
                            if isinstance(skeleton, dict):
                                style_score = skeleton.get('style_score', 3)
                                ideal_prop = skeleton.get('ideal_prop_count', 2)

                                # Track best template
                                if style_score > best_template_score:
                                    best_template_score = style_score
                                    best_template_capacity = ideal_prop

                        # Template quality: prefer style_score >= 3
                        if best_template_score >= 3:
                            template_quality_score = best_template_score / 5.0  # Normalize to 0-1

                        # Capacity match: prefer templates where ideal_prop_count matches prop_count
                        if best_template_capacity is not None:
                            capacity_diff = abs(best_template_capacity - prop_count)
                            ideal_prop_match_score = 1.0 / (1.0 + capacity_diff)  # Closer = higher score
                except (json.JSONDecodeError, TypeError):
                    pass  # No metadata available, use defaults

                # Combined score: sentence count match (primary) + template quality + capacity match
                sentence_match_score = 1.0 / (1.0 + abs(sentence_count - prop_count))
                combined_score = (
                    sentence_match_score * 0.5 +  # 50% weight on sentence count
                    template_quality_score * 0.3 +  # 30% weight on template quality
                    ideal_prop_match_score * 0.2  # 20% weight on capacity match
                )

                scored_examples.append((example, combined_score, sentence_count))

                # Legacy scoring for backward compatibility
                score = abs(sentence_count - prop_count)
                if score < best_score:
                    best_score = score
                    best_example = example
            except Exception as e:
                if verbose:
                    print(f"  Warning: Failed to process example: {e}")
                continue

        # Sort by combined score (best first) if we have template metadata
        if scored_examples and any(score > 0 for _, score, _ in scored_examples):
            scored_examples.sort(key=lambda x: x[1], reverse=True)
            # Use top-scored example
            best_example = scored_examples[0][0]
            if verbose:
                print(f"  Selected example based on template quality and capacity match")

        # Select best match, or longest if no good match
        selected_example = best_example if best_example else longest_example

        if not selected_example:
            # Last resort: use first example
            selected_example = raw_examples[0]
            if verbose:
                print(f"  Warning: Using first example as last resort")

        if verbose:
            example_sentences = sent_tokenize(selected_example)
            sentence_count = len([s for s in example_sentences if s.strip()])
            print(f"  Selected teacher example with {sentence_count} sentences (target: {prop_count} propositions)")

        # Extract templates from selected example
        templates = self._extract_sentence_templates(selected_example, verbose=verbose)

        # Guarantee: Must always return non-empty list
        if not templates:
            if verbose:
                print("  Warning: No templates extracted, using fallback")
            templates = ["[NP] [VP] [NP]."]

        result = (selected_example, templates)

        # Cache result (limit cache size to 100 entries with LRU eviction)
        if len(self._skeleton_cache) >= 100:
            # Remove oldest entry (simple FIFO)
            oldest_key = next(iter(self._skeleton_cache))
            del self._skeleton_cache[oldest_key]
        self._skeleton_cache[cache_key] = result

        return result

    def _map_propositions_to_templates(
        self,
        propositions: List[str],
        templates: List[str],
        narrative_roles: List[str],
        verbose: bool = False
    ) -> List[List[str]]:
        """Map propositions to template slots using LLM.

        Assigns each proposition to exactly one template slot, ensuring every
        template gets at least 1 proposition (strict 1:1 mapping).
        Considers narrative roles to match propositions to appropriate rhetorical functions.

        Args:
            propositions: List of proposition strings to assign.
            templates: List of template strings (one per sentence slot).
            narrative_roles: List of narrative role strings (one per template slot).
            verbose: Whether to print debug information.

        Returns:
            List of lists where index i contains propositions for template i.

        Raises:
            ValueError: If mapping fails after retries.
        """
        if not propositions:
            raise ValueError("Cannot map propositions: propositions list is empty")
        if not templates:
            raise ValueError("Cannot map propositions: templates list is empty")
        if not narrative_roles:
            raise ValueError("Cannot map propositions: narrative_roles list is empty")
        if len(narrative_roles) != len(templates):
            raise ValueError(f"Mismatched lengths: templates={len(templates)}, narrative_roles={len(narrative_roles)}")

        max_retries = self.llm_provider_config.get("max_retries", 3)
        retry_delay = self.llm_provider_config.get("retry_delay", 2)

        # Build prompt
        system_prompt = """You are a Structural Architect constructing a logical narrative. Your task is to assign propositions to sentence templates.

You will receive:
- A list of propositions (atomic meaning units)
- A list of sentence templates (structural blueprints)
- A list of narrative roles (rhetorical function for each template)

Your job: Assign every proposition to exactly one template where it fits best contextually, considering both semantic fit and narrative role alignment.

**Critical Constraints:**
1. **Role Alignment:** Prioritize matching facts to the Slot's Narrative Role (e.g., 'Evidence' facts -> 'Evidence' slot).
2. **Load Balancing (CRITICAL):** Do NOT assign more than 3 propositions to a single slot.
   - If a fact does not fit the available roles perfectly, assign it to the most logical 'Body' or 'Analysis' slot rather than overloading the perfect match.
   - It is better to have a 'Setup' fact in a 'Body' slot than to break the sentence structure with too many facts.
   - If a proposition does not fit the specific Narrative Role of the remaining slots, assign it to the slot where it can serve as *supporting context*, but do NOT overload any single slot (max 3 props).
3. **Dependency Rule:** If two propositions are grammatically linked in the source (e.g., Action + Location like 'Scavenging in ruins', Cause + Effect, Subject + Modifier), you MUST assign them to the **SAME** template slot. Do not split dependent propositions.
   - Before assigning, identify prepositional phrases, adverbial modifiers, and causal chains. Keep these together.
   - Examples:
     - Bad: Slot 1: 'Scavenging', Slot 2: 'In ruins'
     - Good: Slot 1: 'Scavenging in ruins'
4. **Subject Consistency Rule:** If multiple propositions share the same subject (e.g., 'The Soviet Union'), group them together rather than scattering them, unless you are creating a deliberate list. This maintains narrative coherence and prevents fragmented references to the same entity.
5. Every proposition must be assigned to exactly one template
6. Every template must receive at least 1 proposition

Output format: JSON array of objects, each with:
- "template_index": integer (0-based index of the template)
- "propositions": array of strings (propositions assigned to this template)

Example:
[
  {"template_index": 0, "propositions": ["Prop 1", "Prop 2"]},
  {"template_index": 1, "propositions": ["Prop 3"]},
  {"template_index": 2, "propositions": ["Prop 4", "Prop 5"]}
]"""

        # Build structure with roles
        structure_lines = []
        for i, (role, template) in enumerate(zip(narrative_roles, templates)):
            structure_lines.append(f"{i}. Role: {role} | Template: {template}")

        user_prompt = f"""Propositions ({len(propositions)} total):
{chr(10).join(f"{i}. {prop}" for i, prop in enumerate(propositions))}

Structure:
{chr(10).join(structure_lines)}

Assign each proposition to exactly one template. Every template must have at least 1 proposition.
Ensure the facts assigned to each slot fit the role of that slot. If perfect role match is not possible, assign to the most compatible slot while maintaining load balance (max 3 props per slot).
**Distribute propositions evenly** - avoid overloading any single template with more than 3 propositions.

**CRITICAL:** Analyze each proposition for grammatical dependencies (prepositions, modifiers, causal links). If Proposition A modifies or depends on Proposition B, assign them to the same slot.

**CRITICAL:** Identify propositions that share the same subject and group them together in the same template slot to maintain subject consistency.

Return JSON array with template_index and propositions for each template."""

        for attempt in range(max_retries):
            try:
                if verbose:
                    if attempt > 0:
                        print(f"  🔄 Retry attempt {attempt + 1}/{max_retries} for proposition mapping")
                    else:
                        print(f"  📤 Calling LLM for proposition mapping ({len(propositions)} props -> {len(templates)} templates)")

                response = self.llm_provider.call(
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    model_type="editor",
                    require_json=True,
                    temperature=0.3,  # Low temperature for consistent assignment
                    max_tokens=self.translator_config.get("max_tokens", 1000)
                )

                if verbose:
                    print(f"  📥 Received response ({len(response)} chars)")

                # Parse JSON response
                assignments = json.loads(response)
                if not isinstance(assignments, list):
                    raise ValueError(f"Expected JSON array, got {type(assignments)}")

                # Build prop_map: List[List[str]] where index i is propositions for template i
                prop_map = [[] for _ in templates]
                assigned_props = set()

                for assignment in assignments:
                    if not isinstance(assignment, dict):
                        continue
                    template_idx = assignment.get("template_index")
                    assigned_props_list = assignment.get("propositions", [])

                    if template_idx is None or not isinstance(template_idx, int):
                        continue
                    if template_idx < 0 or template_idx >= len(templates):
                        continue
                    if not isinstance(assigned_props_list, list):
                        continue

                    # Add propositions to this template slot
                    for prop in assigned_props_list:
                        if isinstance(prop, str) and prop.strip():
                            prop_map[template_idx].append(prop.strip())
                            assigned_props.add(prop.strip())

                # Post-process: Ensure every template gets at least 1 proposition
                unassigned_props = [p for p in propositions if p not in assigned_props]

                # Redistribute unassigned propositions
                if unassigned_props:
                    if verbose:
                        print(f"  ⚠ Found {len(unassigned_props)} unassigned propositions, redistributing...")
                    # Distribute evenly to templates with fewest propositions
                    for prop in unassigned_props:
                        # Find template with fewest propositions
                        min_idx = min(range(len(prop_map)), key=lambda i: len(prop_map[i]))
                        prop_map[min_idx].append(prop)

                # Ensure every template has at least 1 proposition
                empty_templates = [i for i, props in enumerate(prop_map) if not props]
                if empty_templates:
                    if verbose:
                        print(f"  ⚠ Found {len(empty_templates)} empty templates, redistributing...")
                    # Redistribute from templates with multiple propositions
                    for empty_idx in empty_templates:
                        # Find template with most propositions
                        max_idx = max(range(len(prop_map)), key=lambda i: len(prop_map[i]) if i != empty_idx else -1)
                        if prop_map[max_idx]:
                            # Move one proposition from max to empty
                            prop_map[empty_idx].append(prop_map[max_idx].pop())

                # Final validation
                all_assigned = all(len(props) > 0 for props in prop_map)
                if not all_assigned:
                    raise ValueError("Post-processing failed: some templates still have no propositions")

                # Verify all propositions are assigned
                all_props_assigned = set()
                for props in prop_map:
                    all_props_assigned.update(props)

                # Check if all original propositions are accounted for (allowing for minor variations)
                if len(all_props_assigned) < len(propositions) * 0.8:  # Allow 20% variation
                    if verbose:
                        print(f"  ⚠ Warning: Only {len(all_props_assigned)}/{len(propositions)} propositions assigned")

                # Load balancing: Redistribute if any slot is overloaded
                # Heuristic: If any slot has >50% of total propositions OR >3 propositions when others have ≤1
                total_props = sum(len(props) for props in prop_map)
                max_props = max(len(props) for props in prop_map) if prop_map else 0
                min_props = min(len(props) for props in prop_map) if prop_map else 0

                needs_redistribution = False
                if max_props > 3 and min_props <= 1:
                    # One slot has >3 props while others have ≤1
                    needs_redistribution = True
                elif max_props > total_props * 0.5 and len(prop_map) > 1:
                    # One slot has >50% of all propositions
                    needs_redistribution = True

                if needs_redistribution:
                    if verbose:
                        print(f"  ⚠ Unbalanced distribution detected: {[len(props) for props in prop_map]}, redistributing...")

                    # Find overloaded slots (those with >3 props or >50% of total)
                    overloaded_slots = []
                    for i, props in enumerate(prop_map):
                        if len(props) > 3 or (len(props) > total_props * 0.5 and len(prop_map) > 1):
                            overloaded_slots.append(i)

                    # Redistribute from overloaded slots to underloaded ones
                    for overloaded_idx in overloaded_slots:
                        excess_props = prop_map[overloaded_idx][3:]  # Keep first 3, move the rest
                        if not excess_props:
                            continue

                        # Remove excess from overloaded slot
                        prop_map[overloaded_idx] = prop_map[overloaded_idx][:3]

                        # Distribute excess to slots with fewest propositions
                        for prop in excess_props:
                            # Find slot with fewest propositions (excluding the overloaded one)
                            min_idx = min(
                                (i for i in range(len(prop_map)) if i != overloaded_idx),
                                key=lambda i: len(prop_map[i]),
                                default=None
                            )
                            if min_idx is not None:
                                prop_map[min_idx].append(prop)
                            else:
                                # Fallback: put it back in the overloaded slot
                                prop_map[overloaded_idx].append(prop)

                    if verbose:
                        print(f"  ✅ Redistributed to: {[len(props) for props in prop_map]} propositions per template")

                if verbose:
                    print(f"  ✅ Mapped propositions: {[len(props) for props in prop_map]} propositions per template")

                return prop_map

            except (json.JSONDecodeError, ValueError, KeyError) as e:
                if verbose:
                    print(f"  ❌ Mapping failed: {e}")
                if attempt < max_retries - 1:
                    time.sleep(retry_delay)
                    retry_delay *= 2  # Exponential backoff
                    continue
                else:
                    raise ValueError(f"Failed to map propositions to templates after {max_retries} attempts: {e}")
            except Exception as e:
                if verbose:
                    print(f"  ❌ Unexpected error: {e}")
                if attempt < max_retries - 1:
                    time.sleep(retry_delay)
                    retry_delay *= 2
                    continue
                else:
                    raise ValueError(f"Unexpected error mapping propositions: {e}")

        # Should never reach here, but just in case
        raise ValueError("Failed to map propositions to templates")

    def _smooth_paragraph(
        self,
        draft_text: str,
        verbose: bool = False
    ) -> str:
        """Apply a smoothing pass to fix grammar and awkward phrasing.

        Fixes grammatical errors (subject-verb agreement) and awkward transitions
        while preserving sentence structure and vocabulary (style).

        Args:
            draft_text: The generated paragraph text to smooth.
            verbose: Whether to print debug information.

        Returns:
            Smoothed paragraph text.
        """
        if not draft_text or not draft_text.strip():
            return draft_text

        if not self.llm_provider:
            if verbose:
                print(f"  ⚠ LLM provider not available, skipping smoothing pass")
            return draft_text

        try:
            if verbose:
                print(f"  Applying smoothing pass to paragraph ({len(draft_text)} chars)...")

            system_prompt = """You are a Coherence Editor. Your task is to rewrite text that was generated by fusing facts into rigid templates, which may result in 'Word Salad' or unnatural jargon. Make the text logically coherent and natural while preserving the author's voice."""

            user_prompt = f"""**Draft:** {draft_text}

**Issue:** The draft was generated by fusing facts into a rigid template. It may sound like 'Word Salad' or unnatural jargon.

**Task:** Rewrite the paragraph to make it **Logically Coherent** and **Natural**.

**Constraints:**
- **MUST keep** the Author's *Tone* (e.g., didactic, complex, philosophical)
- **MUST keep** all the *Facts* (Meaning) - do not remove or change facts
- **MAY adjust** the sentence structure if the template made it nonsensical
- **MAY simplify** complex clauses that don't make logical sense
- **MAY reorder** phrases to improve flow and readability
- DO ensure every sentence is a complete, grammatical English sentence
- DO fix incomplete sentences, missing verbs, or grammatical fragments

**Output:** Return only the rewritten text, no explanations."""

            smoothed = self.llm_provider.call(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                model_type="editor",
                require_json=False,
                temperature=0.2,  # Low temperature for minimal changes
                max_tokens=len(draft_text.split()) * 2  # Enough for smoothed version
            )

            # Clean up the response (remove any markdown formatting or explanations)
            smoothed = smoothed.strip()
            # Remove markdown code blocks if present
            if smoothed.startswith("```"):
                lines = smoothed.split('\n')
                # Remove first and last lines if they're markdown fences
                if lines[0].strip().startswith('```'):
                    lines = lines[1:]
                if lines and lines[-1].strip().startswith('```'):
                    lines = lines[:-1]
                smoothed = '\n'.join(lines).strip()

            if verbose:
                if smoothed != draft_text:
                    print(f"  ✅ Smoothing pass applied (changed {len(draft_text)} -> {len(smoothed)} chars)")
                else:
                    print(f"  ✅ Smoothing pass completed (no changes needed)")

            return smoothed if smoothed else draft_text

        except Exception as e:
            if verbose:
                print(f"  ⚠ Smoothing pass failed: {e}, using original text")
            return draft_text

    def _simplify_paragraph(
        self,
        draft_text: str,
        verbose: bool = False
    ) -> str:
        """Emergency simplification pass for incoherent text.

        If coherence fails after smoothing, simplify the grammar and structure
        to make the text readable while preserving facts.

        Args:
            draft_text: The text to simplify.
            verbose: Whether to print debug information.

        Returns:
            Simplified paragraph text.
        """
        if not draft_text or not draft_text.strip():
            return draft_text

        if not self.llm_provider:
            if verbose:
                print(f"  ⚠ LLM provider not available, skipping simplification pass")
            return draft_text

        try:
            if verbose:
                print(f"  Applying simplification pass to paragraph ({len(draft_text)} chars)...")

            system_prompt = """You are a Text Simplifier. Your task is to simplify incoherent text by reducing structural complexity while preserving all facts and meaning."""

            user_prompt = f"""**Draft:** {draft_text}

**Issue:** The text is incoherent due to overly complex template structures.

**Task:** Simplify the grammar and structure to make it readable and coherent.

**Constraints:**
- **MUST keep** all facts and meaning (do not remove information)
- **MUST keep** the author's tone (if possible)
- **MAY simplify** complex clauses that don't make sense
- **MAY break** long, convoluted sentences into shorter, clearer ones
- **MAY remove** redundant or nonsensical phrases
- **MAY reorder** words/phrases for clarity
- DO ensure every sentence is a complete, grammatical English sentence

**Output:** Return only the simplified text, no explanations."""

            simplified = self.llm_provider.call(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                model_type="editor",
                require_json=False,
                temperature=0.2,  # Low temperature for minimal changes
                max_tokens=len(draft_text.split()) * 2  # Enough for simplified version
            )

            # Clean up the response (remove any markdown formatting or explanations)
            simplified = simplified.strip()
            # Remove markdown code blocks if present
            if simplified.startswith("```"):
                lines = simplified.split('\n')
                # Remove first and last lines if they're markdown fences
                if lines[0].strip().startswith('```'):
                    lines = lines[1:]
                if lines and lines[-1].strip().startswith('```'):
                    lines = lines[:-1]
                simplified = '\n'.join(lines).strip()

            if verbose:
                if simplified != draft_text:
                    print(f"  ✅ Simplification pass applied (changed {len(draft_text)} -> {len(simplified)} chars)")
                else:
                    print(f"  ✅ Simplification pass completed (no changes needed)")

            return simplified if simplified else draft_text

        except Exception as e:
            if verbose:
                print(f"  ⚠ Simplification pass failed: {e}, using original text")
            return draft_text

    def _generate_candidate_populations(
        self,
        state: ParagraphState,
        author_name: str,
        style_dna: Optional[Dict],
        population_size: int,
        verbose: bool = False
    ) -> List[List[str]]:
        """Generate multiple candidates for UNLOCKED slots in one LLM call with neighbor context.

        For each unlocked slot, provides context from neighboring slots (X-1, X+1) to ensure flow.
        Uses locked best sentences if available, otherwise uses best candidates as placeholders.

        Args:
            state: ParagraphState containing current paragraph state.
            author_name: Target author name.
            style_dna: Optional style DNA dictionary.
            population_size: Number of candidates to generate per unlocked slot.
            verbose: Whether to print debug information.

        Returns:
            List[List[str]] where index i is the population for slot i (empty list for locked slots).

        Raises:
            ValueError: If generation fails.
        """
        # Identify unlocked slots
        unlocked_slots = [i for i, locked in enumerate(state.locked_flags) if not locked]

        if not unlocked_slots:
            if verbose:
                print("  All slots are locked, no candidates to generate")
            return [[] for _ in state.templates]

        if verbose:
            print(f"  Generating candidates for {len(unlocked_slots)} unlocked slots (population_size={population_size})")

        # Build context for each unlocked slot
        slot_contexts = []
        for slot_idx in unlocked_slots:
            # Get previous context (Slot i-1)
            prev_context = ""
            if slot_idx > 0:
                if state.locked_flags[slot_idx - 1] and state.best_sentences[slot_idx - 1]:
                    prev_context = state.best_sentences[slot_idx - 1]
                elif state.candidate_populations[slot_idx - 1]:
                    # Use first candidate as placeholder (could be improved to use best scoring)
                    prev_context = state.candidate_populations[slot_idx - 1][0]

            # Get next context (Slot i+1)
            next_context = ""
            if slot_idx < len(state.templates) - 1:
                if state.locked_flags[slot_idx + 1] and state.best_sentences[slot_idx + 1]:
                    next_context = state.best_sentences[slot_idx + 1]
                elif state.candidate_populations[slot_idx + 1]:
                    # Use first candidate as placeholder
                    next_context = state.candidate_populations[slot_idx + 1][0]

            # Get narrative role for this slot
            narrative_role = state.narrative_roles[slot_idx] if slot_idx < len(state.narrative_roles) else "BODY"

            # Determine role with descriptive label
            if slot_idx == 0:
                role = "OPENER (Establish the context)"
            elif slot_idx == len(state.templates) - 1:
                role = "CLOSER (Summarize or conclude)"
            else:
                role = "BODY (Develop the argument)"

            # Build context hint
            prev_sentence_text = prev_context if prev_context else "None (Start of paragraph)"
            context_hint = f"""
**Paragraph Context:**
- Position: Sentence {slot_idx + 1} of {len(state.templates)} ({role}).
- Narrative Role: {narrative_role}
- Previous Sentence: "{prev_sentence_text}"
- Instruction: Write a sentence that fulfills the rhetorical function of '{narrative_role}' while connecting logically to the previous sentence.
"""

            slot_contexts.append({
                "slot_index": slot_idx,
                "role": role,
                "template": state.templates[slot_idx],
                "propositions": state.prop_map[slot_idx],
                "prev_context": prev_context,
                "next_context": next_context,
                "context_hint": context_hint
            })

        # Extract synthesized feedback and elite candidate for each slot
        for ctx in slot_contexts:
            slot_idx = ctx['slot_index']
            # Extract synthesized feedback and elite candidate if present
            synthesized_feedback = None
            elite_text = None
            elite_feedback = None
            if state.feedback[slot_idx]:
                for fb in state.feedback[slot_idx]:
                    if isinstance(fb, str):
                        if fb.startswith("[SYNTHESIZED]"):
                            synthesized_feedback = fb.replace("[SYNTHESIZED]", "").strip()
                        elif fb.startswith("[ELITE_TEXT]"):
                            elite_text = fb.replace("[ELITE_TEXT]", "").strip()
                        elif fb.startswith("[ELITE_FEEDBACK]"):
                            elite_feedback = fb.replace("[ELITE_FEEDBACK]", "").strip()
            ctx['synthesized_feedback'] = synthesized_feedback
            ctx['elite_text'] = elite_text
            ctx['elite_feedback'] = elite_feedback

        # Build prompt
        system_prompt = """You are a Narrative Architect.

**Task:** Write candidate sentences for each slot in a paragraph.

For each unlocked slot, you will receive:
- **Paragraph Context:** Position, role, narrative role, and flow instructions
- **Template:** Structural blueprint (fixed anchors must be preserved)
- **Propositions:** Meaning atoms to express
- **Neighbor Context:** Previous and next sentences for flow

**Style Translation Rule:** Do not use the literal phrasing of the propositions. You must **Elevate the Register** to match the Target Author's vocabulary and tone.
- Examples:
  - Input Prop: 'brought void and hunger'
  - Target (Mao): 'precipitated a material crisis of subsistence'
  - Target (Hemingway): 'left nothing but the empty stomach'
- **Task:** Rewrite the proposition to fit the Template's Tone *before* inserting it. Translate the *meaning* of the proposition, not the words.

**Grammar Supremacy Rule:**
Grammar correctness ALWAYS overrides template tag requirements.
- If the template says `[ADJ]` but your fact is a Noun, use it as a Noun.
- If changing a tag from `[ADJ]` to `[NP]` makes the sentence grammatically correct, DO IT.
- The template tags are GUIDELINES, not strict requirements.
- **Your goal:** Create a grammatically perfect English sentence that conveys the meaning.

**CRITICAL ANTI-PATTERNS (DO NOT DO THIS):**
- ❌ "It is [ADV] the [NOUN] is [ADJ]" → Creates nonsense like "It is profoundly the ghost is historical"
- ✅ Instead: "It is [ADV] that the [NOUN] is [ADJ]" OR restructure completely
- ❌ Forcing adjectives where nouns belong just to match template tags
- ✅ Use the correct part of speech for the concept, even if it doesn't match the template tag

**Red Flags to Avoid:**
- Sentences that sound like machine translation errors
- Awkward constructions that prioritize template matching over natural English
- Phrases that make no logical sense when read aloud

**Constraints:**
1. **Meaning:** Express the assigned propositions by *translating* their meaning into the target style, not by copying their wording. DO NOT paste proposition text literally. Transform it to match the author's register.
2. **Structure:** Follow the template exactly (preserve fixed anchors)
3. **Narrative Role:** Each slot has a Narrative Role that must be fulfilled (e.g., 'Theoretical Analysis', 'Evidence'). Your sentence must fulfill this rhetorical function.
4. **Flow:** Connect logically to the previous sentence (DO NOT rewrite it)
5. **Style:** Match the target author's voice
6. **CRITICAL GRAMMAR RULE:** You must replace every `[NP]`, `[VP]`, `[ADJ]`, `[ADV]` placeholder with actual words. The result must be a complete, grammatical English sentence.
   - BAD: "Consequently I can neither the small..." (Missing verb in [VP] slot)
   - GOOD: "Consequently I can **tolerate** neither the small..." (Complete verb phrase)
   - BAD: "In the ruins, there was..." (Incomplete, missing object)
   - GOOD: "In the ruins, there was **scavenging**." (Complete sentence)
   - DO NOT skip placeholders or leave them empty
   - DO NOT create fragments or incomplete sentences
   - Every placeholder must be filled with appropriate words

7. **Grammar Overrides Template (ENHANCED):**
   The template placeholders (e.g., `[ADJ]`, `[NP]`, `[VP]`) are suggestions for the *kind of complexity* required, not strict grammatical rules.
   - **Grammar > Template Tags:** If adhering to the template tag would create nonsense, BREAK THE TEMPLATE TAG.
   - If the template says `[ADJ]` but your fact is a Noun, **change it to a Noun**.
   - If the template says `[NP]` but your fact is an Adjective, **change it to an Adjective**.
   - **Constraint:** You must maintain the *sentence structure* (clauses, punctuation, word order), but you may swap Parts of Speech to create valid English.
   - **DO NOT write nonsense** like "The ghost is historical" just to fit an `[ADJ]` slot. If "ghost" is a noun, use it as a noun.
   - **CRITICAL EXAMPLE:** Template `It is [ADV] the "[ADJ]" is [ADJ]` with fact "ghost" →
     * ❌ WRONG: `It is profoundly the "ghost" is historical` (gibberish - "the ghost is historical" makes no sense)
     * ✅ CORRECT: `It is profoundly the ghost is historical` (noun in noun position, but still awkward)
     * ✅ BEST: `It is profoundly that the ghost is historical` (restructured with "that" clause)
     * ✅ ALTERNATIVE: `The ghost is profoundly historical` (complete restructure, grammar-first)

**Output Format:** JSON object with keys for each slot:
- "slot_0_plan": "Brief explanation of your approach (e.g., 'I need to convert the noun 'ghost' to fit the template structure...')"
- "slot_0": ["Candidate 1", "Candidate 2", ...]
- "slot_1_plan": "..."
- "slot_1": [...]

The `_plan` field forces you to think through the grammar/template trade-offs before generating.

Example:
{{
  "slot_0_plan": "The template has [ADJ] but my fact is a noun 'ghost'. I'll restructure to use 'ghost' as a noun in a grammatically correct way.",
  "slot_0": ["Candidate 1 for slot 0", "Candidate 2 for slot 0", ...],
  "slot_1_plan": "The template requires [VP] but the proposition is a state. I'll use a verb form that captures the state.",
  "slot_1": ["Candidate 1 for slot 1", "Candidate 2 for slot 1", ...]
}}""".format(population_size=population_size)

        user_prompt_parts = [
            f"Original text: {state.original_text}",
            "",
            "Unlocked slots to generate:"
        ]

        # Build set of locked sentence texts to avoid duplicates
        locked_texts = set()
        for i, (locked, best) in enumerate(zip(state.locked_flags, state.best_sentences)):
            if locked and best:
                locked_texts.add(best.strip())

        for ctx in slot_contexts:
            user_prompt_parts.append(f"\nSlot {ctx['slot_index']} {ctx['role']}:")
            user_prompt_parts.append(ctx['context_hint'])
            user_prompt_parts.append(f"  Template: {ctx['template']}")
            user_prompt_parts.append(f"  Propositions: {', '.join(ctx['propositions'])}")
            # Only include next_context if it's not already in locked_sentences
            if ctx['next_context'] and ctx['next_context'].strip() not in locked_texts:
                user_prompt_parts.append(f"  Next sentence (for reference): {ctx['next_context']}")

            # Add synthesized feedback if available
            if ctx.get('synthesized_feedback'):
                user_prompt_parts.append(f"  **Population Analysis:** {ctx['synthesized_feedback']}")

            # Add elite candidate if available
            if ctx.get('elite_text') and ctx.get('elite_feedback'):
                user_prompt_parts.append(f"  **Best Previous Attempt (Near Miss):**")
                user_prompt_parts.append(f"    Draft: '{ctx['elite_text']}'")
                user_prompt_parts.append(f"    Error: {ctx['elite_feedback']}")
                user_prompt_parts.append(f"    Task: Fix THIS specific error while preserving what worked.")

        # Add locked slots context for global awareness
        # Only include locked sentences that are adjacent to unlocked slots or the opener
        locked_sentences = []
        locked_indices = set()

        # Always include opener (slot 0) if it exists and is locked
        if state.locked_flags[0] and state.best_sentences[0]:
            locked_sentences.append(f"Slot 0 (locked, opener): {state.best_sentences[0]}")
            locked_indices.add(0)

        # Include adjacent locked sentences (i-1, i+1) for each unlocked slot
        for slot_idx in unlocked_slots:
            # Previous slot (i-1)
            if slot_idx > 0:
                prev_idx = slot_idx - 1
                if prev_idx not in locked_indices and state.locked_flags[prev_idx] and state.best_sentences[prev_idx]:
                    locked_sentences.append(f"Slot {prev_idx} (locked, previous): {state.best_sentences[prev_idx]}")
                    locked_indices.add(prev_idx)

            # Next slot (i+1)
            if slot_idx < len(state.templates) - 1:
                next_idx = slot_idx + 1
                if next_idx not in locked_indices and state.locked_flags[next_idx] and state.best_sentences[next_idx]:
                    locked_sentences.append(f"Slot {next_idx} (locked, next): {state.best_sentences[next_idx]}")
                    locked_indices.add(next_idx)

        if locked_sentences:
            user_prompt_parts.append("\nLocked sentences (for global context):")
            user_prompt_parts.extend(locked_sentences)

        # Add style DNA if available
        if style_dna:
            style_info = []
            if style_dna.get("tone"):
                style_info.append(f"Tone: {style_dna['tone']}")
            if style_dna.get("lexicon"):
                lexicon_sample = style_dna['lexicon'][:10] if isinstance(style_dna['lexicon'], list) else []
                if lexicon_sample:
                    style_info.append(f"Lexicon sample: {', '.join(lexicon_sample)}")
            if style_info:
                user_prompt_parts.append(f"\nStyle DNA: {', '.join(style_info)}")

        user_prompt = "\n".join(user_prompt_parts)

        # Call LLM
        max_retries = self.llm_provider_config.get("max_retries", 3)
        retry_delay = self.llm_provider_config.get("retry_delay", 2)
        temperature = self.paragraph_fusion_config.get("generation_temperature", 0.7)

        for attempt in range(max_retries):
            try:
                if verbose:
                    if attempt > 0:
                        print(f"  🔄 Retry attempt {attempt + 1}/{max_retries} for candidate generation")
                    else:
                        print(f"  📤 Calling LLM for candidate generation ({len(unlocked_slots)} slots, {population_size} candidates each)")

                response = self.llm_provider.call(
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    model_type="editor",
                    require_json=True,
                    temperature=temperature,
                    max_tokens=self.translator_config.get("max_tokens", 2000)
                )

                if verbose:
                    print(f"  📥 Received response ({len(response)} chars)")

                # Parse JSON response
                result = json.loads(response)
                if not isinstance(result, dict):
                    raise ValueError(f"Expected JSON object, got {type(result)}")

                # Build populations: List[List[str]] where index i is population for slot i
                populations = [[] for _ in state.templates]

                for slot_idx in unlocked_slots:
                    slot_key = f"slot_{slot_idx}"
                    # Ignore _plan fields (Chain of Thought planning step)
                    plan_key = f"slot_{slot_idx}_plan"

                    # Extract plan if present (for potential logging/debugging, but not used)
                    if plan_key in result and verbose:
                        plan_text = result[plan_key]
                        if plan_text:
                            print(f"  📋 Slot {slot_idx} plan: {plan_text[:100]}{'...' if len(plan_text) > 100 else ''}")

                    if slot_key in result and isinstance(result[slot_key], list):
                        candidates = [str(c).strip() for c in result[slot_key] if c and str(c).strip()]
                        # Ensure we have exactly population_size candidates
                        if len(candidates) < population_size:
                            if verbose:
                                print(f"  ⚠ Slot {slot_idx}: Only {len(candidates)} candidates, expected {population_size}")
                            # Pad with empty strings or repeat last candidate
                            while len(candidates) < population_size:
                                candidates.append(candidates[-1] if candidates else "")
                        elif len(candidates) > population_size:
                            candidates = candidates[:population_size]
                        populations[slot_idx] = candidates
                    else:
                        if verbose:
                            print(f"  ⚠ Slot {slot_idx}: Missing or invalid candidates in response")
                        # Fallback: generate empty list (will be handled by evolution loop)
                        populations[slot_idx] = []

                if verbose:
                    total_candidates = sum(len(p) for p in populations)
                    print(f"  ✅ Generated {total_candidates} total candidates across {len(unlocked_slots)} slots")

                return populations

            except (json.JSONDecodeError, ValueError, KeyError) as e:
                if verbose:
                    print(f"  ❌ Generation failed: {e}")
                if attempt < max_retries - 1:
                    time.sleep(retry_delay)
                    retry_delay *= 2  # Exponential backoff
                    continue
                else:
                    raise ValueError(f"Failed to generate candidate populations after {max_retries} attempts: {e}")
            except Exception as e:
                if verbose:
                    print(f"  ❌ Unexpected error: {e}")
                if attempt < max_retries - 1:
                    time.sleep(retry_delay)
                    retry_delay *= 2
                    continue
                else:
                    raise ValueError(f"Unexpected error generating candidates: {e}")

        # Should never reach here
        raise ValueError("Failed to generate candidate populations")

    def _sanity_check_candidates(
        self,
        candidates: List[str],
        template: str,
        verbose: bool = False
    ) -> List[str]:
        """Fast regex-based sanity check to filter gibberish before critic evaluation.

        Detects common gibberish patterns like "It is [adv] the [noun] is [adj]"
        which creates nonsensical constructions.

        Args:
            candidates: List of candidate sentences
            template: Template string for context (not currently used, but available)
            verbose: Enable verbose logging

        Returns:
            Filtered list of candidates (gibberish removed)
        """
        if not candidates:
            return []

        filtered = []
        gibberish_patterns = [
            # Pattern: "It is [adv] the [noun] is [adj]" - the classic gibberish
            r'^It\s+is\s+\w+\s+the\s+\w+\s+is\s+\w+',
            # Pattern: "It is [adv] the [noun] [verb]" - similar issue
            r'^It\s+is\s+\w+\s+the\s+\w+\s+\w+\s+\w+',
            # Pattern: Multiple "is" in close proximity (often indicates structure issues)
            r'\b\w+\s+is\s+\w+\s+is\s+\w+',
        ]

        for candidate in candidates:
            if not candidate or not candidate.strip():
                continue

            candidate_lower = candidate.strip()
            is_gibberish = False

            # Check against patterns
            for pattern in gibberish_patterns:
                if re.search(pattern, candidate_lower, re.IGNORECASE):
                    is_gibberish = True
                    if verbose:
                        print(f"      🚫 Filtered gibberish: {candidate[:80]}{'...' if len(candidate) > 80 else ''}")
                    break

            if not is_gibberish:
                filtered.append(candidate)

        if verbose and len(filtered) < len(candidates):
            print(f"      ⚠ Filtered {len(candidates) - len(filtered)} gibberish candidates")

        return filtered

    def _synthesize_slot_feedback(
        self,
        slot_results: List[Dict],
        slot_candidates: List[str],
        slot_template: str,
        slot_propositions: List[str],
        prev_context: Optional[str] = None,
        verbose: bool = False
    ) -> Tuple[Optional[str], Optional[str], Optional[str]]:
        """Synthesize feedback from entire candidate population using elite forwarding.

        Analyzes all candidates to identify partial successes and creates targeted
        directives for the next generation. Uses "elite forwarding" - forwards entire
        feedback strings from leader candidates rather than parsing them.

        Args:
            slot_results: List of evaluation result dicts (one per candidate)
            slot_candidates: List of candidate sentence strings
            slot_template: Template string for this slot
            slot_propositions: List of propositions assigned to this slot
            prev_context: Previous sentence context (for narrative flow)
            verbose: Enable verbose logging

        Returns:
            Tuple of (directive_string, elite_candidate_text, elite_candidate_feedback)
            Returns (None, None, None) if slot is passing or no candidates exist
        """
        if not slot_results or not slot_candidates:
            return (None, None, None)

        if len(slot_results) != len(slot_candidates):
            if verbose:
                print(f"      ⚠ Warning: Mismatch between results ({len(slot_results)}) and candidates ({len(slot_candidates)})")
            return (None, None, None)

        # Extract scores and identify leaders
        structure_leader_idx = None
        structure_leader_score = -1.0
        meaning_leader_idx = None
        meaning_leader_score = -1.0
        logic_leader_idx = None
        logic_leader_score = -1.0
        max_narrative = -1.0

        elite_failing_idx = None
        elite_failing_score = -1.0

        for idx, result in enumerate(slot_results):
            anchor_score = result.get("anchor_score", 0.0)
            semantic_score = result.get("semantic_score", 0.0)
            narrative_score = result.get("narrative_score", 0.0)
            logic_score = result.get("logic_score", 0.0)
            combined_score = result.get("combined_score", 0.0)
            passed = result.get("pass", False)

            # Track leaders
            if anchor_score > structure_leader_score:
                structure_leader_score = anchor_score
                structure_leader_idx = idx

            if semantic_score > meaning_leader_score:
                meaning_leader_score = semantic_score
                meaning_leader_idx = idx

            if logic_score > logic_leader_score:
                logic_leader_score = logic_score
                logic_leader_idx = idx

            if narrative_score > max_narrative:
                max_narrative = narrative_score

            # Track best failing candidate (highest combined score that didn't pass)
            if not passed and combined_score > elite_failing_score:
                elite_failing_score = combined_score
                elite_failing_idx = idx

        # Check if slot is passing (all thresholds met)
        if (structure_leader_score >= 1.0 and
            meaning_leader_score >= 0.95 and
            max_narrative >= 0.8 and
            logic_leader_score >= 0.8):
            if verbose:
                print(f"      ✅ Slot is passing, no synthesis needed")
            return (None, None, None)

        # Build directive parts
        directive_parts = []

        # Structural failure
        if structure_leader_idx is not None and structure_leader_score < 1.0:
            structure_leader_feedback = slot_results[structure_leader_idx].get("feedback", "")
            if structure_leader_feedback:
                directive_parts.append(
                    f"The best structural attempt (Candidate #{structure_leader_idx}) failed. "
                    f"Feedback: {structure_leader_feedback}"
                )

        # Semantic failure
        if meaning_leader_idx is not None and meaning_leader_score < 0.95:
            meaning_leader_feedback = slot_results[meaning_leader_idx].get("feedback", "")
            if meaning_leader_feedback:
                directive_parts.append(
                    f"The best semantic attempt (Candidate #{meaning_leader_idx}) failed. "
                    f"Feedback: {meaning_leader_feedback}"
                )

        # Cross-pollination: if different leaders, suggest combining
        if (structure_leader_idx is not None and
            meaning_leader_idx is not None and
            structure_leader_idx != meaning_leader_idx and
            structure_leader_score < 1.0 and
            meaning_leader_score < 0.95):
            directive_parts.append(
                f"Candidate #{structure_leader_idx} had good structure. "
                f"Candidate #{meaning_leader_idx} had good meaning. Combine their strengths."
            )

        # Narrative failure
        if max_narrative < 0.8:
            if prev_context:
                directive_parts.append(
                    f"The narrative connection is weak. Ensure you connect to the previous sentence: '{prev_context}'."
                )
            else:
                directive_parts.append(
                    "The narrative connection is weak. Ensure proper flow and coherence."
                )

        # Logic failure
        if logic_leader_idx is not None and logic_leader_score < 0.8:
            logic_leader_feedback = slot_results[logic_leader_idx].get("feedback", "")
            if logic_leader_feedback:
                directive_parts.append(
                    f"Logical coherence issue. Feedback: {logic_leader_feedback}"
                )

        # Combine directive parts
        directive = " ".join(directive_parts) if directive_parts else None

        # Get elite failing candidate
        elite_text = None
        elite_feedback = None
        if elite_failing_idx is not None:
            elite_text = slot_candidates[elite_failing_idx]
            elite_feedback = slot_results[elite_failing_idx].get("feedback", "")

        if verbose and directive:
            print(f"      📊 Synthesized feedback: {directive[:100]}{'...' if len(directive) > 100 else ''}")
            if elite_text:
                print(f"      🏆 Elite candidate: {elite_text[:60]}{'...' if len(elite_text) > 60 else ''}")

        return (directive, elite_text, elite_feedback)

    def translate_paragraph(
        self,
        paragraph: str,
        atlas,
        author_name: str,
        style_dna: Optional[Dict] = None,
        position: str = "BODY",
        structure_tracker: Optional[object] = None,
        used_examples: Optional[Set[str]] = None,
        secondary_author: Optional[str] = None,
        blend_ratio: float = 0.5,
        verbose: bool = False,
        global_context: Optional[Dict] = None,
        is_opener: bool = False
    ) -> tuple[str, Optional[List[Dict]], Optional[str], float]:
        """Translate a paragraph holistically using paragraph fusion.

        Extracts atomic propositions from the paragraph, retrieves complex style examples,
        and generates a single cohesive paragraph that combines all propositions in the
        target author's style.

        Args:
            paragraph: Input paragraph text (multiple sentences).
            atlas: StyleAtlas instance for retrieving examples.
            author_name: Target author name.
            style_dna: Optional pre-extracted style DNA dictionary.
            verbose: Whether to print debug information.

        Returns:
            Tuple of (generated_paragraph, rhythm_map, teacher_example, internal_recall).
            internal_recall is the best proposition recall achieved during evaluation/repair loop.
        """
        if not paragraph or not paragraph.strip():
            return paragraph, None, None, 0.0

        # Step 1: Extract atomic propositions (with citations bound to facts)
        if verbose:
            print(f"  Extracting atomic propositions from paragraph...")
        propositions = self.proposition_extractor.extract_atomic_propositions(paragraph)
        if verbose:
            print(f"  Extracted {len(propositions)} propositions: {propositions[:3]}...")

        if not propositions:
            raise ValueError("No propositions extracted from paragraph")

        # Extract quotes and citations for later verification/restoration
        quote_pattern = r'["\'](?:[^"\']|(?<=\\)["\'])*["\']'
        extracted_quotes = []
        for match in re.finditer(quote_pattern, paragraph):
            quote_text = match.group(0)
            if len(quote_text.strip('"\'')) > 2:
                extracted_quotes.append(quote_text)

        if verbose and extracted_quotes:
            print(f"  Extracted {len(extracted_quotes)} direct quotations")

        citation_pattern = r'\[\^\d+\]'
        expected_citations = set(re.findall(citation_pattern, paragraph))
        if verbose and expected_citations:
            print(f"  Expected citations: {sorted(expected_citations)}")

        # Step 2: Retrieve robust skeleton and extract sentence templates
        from src.atlas.rhetoric import RhetoricalType, RhetoricalClassifier
        classifier = RhetoricalClassifier()
        rhetorical_type = classifier.classify_heuristic(paragraph)

        if verbose:
            print(f"  Retrieving robust skeleton for {len(propositions)} propositions...")

        try:
            teacher_example, templates = self._retrieve_robust_skeleton(
                rhetorical_type=rhetorical_type.value if hasattr(rhetorical_type, 'value') else str(rhetorical_type),
                author=author_name,
                prop_count=len(propositions),
                atlas=atlas,
                verbose=verbose
            )
        except Exception as e:
            if verbose:
                print(f"  ❌ Failed to retrieve skeleton: {e}")
            raise ValueError(f"Failed to retrieve robust skeleton: {e}")

        if verbose:
            print(f"  ✅ Retrieved skeleton with {len(templates)} sentence templates")

        # Step 2.5: Extract narrative roles from teacher example
        if verbose:
            print(f"  Extracting narrative arc from teacher example...")
        narrative_roles = self._extract_narrative_arc(teacher_example, verbose=verbose)
        if verbose:
            print(f"  ✅ Extracted {len(narrative_roles)} narrative roles")

        # Step 3: Map propositions to templates
        if verbose:
            print(f"  Mapping {len(propositions)} propositions to {len(templates)} templates...")

        try:
            prop_map = self._map_propositions_to_templates(
                propositions=propositions,
                templates=templates,
                narrative_roles=narrative_roles,
                verbose=verbose
            )
        except Exception as e:
            if verbose:
                print(f"  ❌ Failed to map propositions: {e}")
            raise ValueError(f"Failed to map propositions to templates: {e}")

        if verbose:
            print(f"  ✅ Mapped propositions to templates")

        # Step 4: Extract style DNA if not provided
        # Use the teacher_example that was already retrieved for style DNA extraction
        if not style_dna:
            try:
                from src.analyzer.style_extractor import StyleExtractor
                style_extractor = StyleExtractor(config_path=self.config_path)
                # Use the teacher_example as a source for style DNA
                style_dna = style_extractor.extract_style_dna([teacher_example])
                if verbose:
                    print(f"  Extracted style DNA: {style_dna.get('tone', 'Unknown')} tone")
            except Exception as e:
                if verbose:
                    print(f"  ⚠ Style DNA extraction failed: {e}, continuing without style DNA")
                style_dna = None

        style_lexicon = None
        if style_dna and isinstance(style_dna, dict):
            style_lexicon = style_dna.get("lexicon", [])

        # Extract rhythm_map from teacher_example for return value
        rhythm_map = None
        try:
            from src.analyzer.structuralizer import extract_paragraph_rhythm
            rhythm_map = extract_paragraph_rhythm(teacher_example)
        except Exception:
            pass

        # Step 5: Initialize ParagraphState and run evolution loop
        if not style_dna:
            try:
                from src.analyzer.style_extractor import StyleExtractor
                style_extractor = StyleExtractor(config_path=self.config_path)

                # Get examples for style DNA extraction
                examples_for_dna = atlas.get_examples_by_rhetoric(
                    rhetorical_type,
                    top_k=5,
                    author_name=author_name,
                    query_text=None
                )

                if secondary_author:
                    style_dna = style_extractor.extract_style_dna(examples_for_dna) if examples_for_dna else None

                    # Extract secondary author's DNA separately for lexicon fusion
                    secondary_examples = atlas.get_examples_by_rhetoric(
                        RhetoricalType.OBSERVATION,
                        top_k=5,
                        author_name=secondary_author,
                        query_text=None
                    )
                    secondary_dna = style_extractor.extract_style_dna(secondary_examples) if secondary_examples else None

                    # Merge lexicons
                    if secondary_dna and isinstance(secondary_dna, dict) and isinstance(style_dna, dict):
                        primary_lexicon = style_dna.get("lexicon", [])[:15]
                        secondary_lexicon = secondary_dna.get("lexicon", [])[:15]
                        blended_lexicon = list(set(primary_lexicon) | set(secondary_lexicon))
                        style_dna["lexicon"] = blended_lexicon

                        if verbose:
                            print(f"  Blended lexicon: {len(blended_lexicon)} words")
                else:
                    style_dna = style_extractor.extract_style_dna(examples_for_dna) if examples_for_dna else None

                if verbose and style_dna:
                    print(f"  Extracted style DNA: {style_dna.get('tone', 'Unknown')} tone")
            except Exception as e:
                if verbose:
                    print(f"  ⚠ Style DNA extraction failed: {e}, continuing without style DNA")
                style_dna = None

        # Extract style_lexicon for use in evaluation
        style_lexicon = None
        if style_dna and isinstance(style_dna, dict):
            style_lexicon = style_dna.get("lexicon", [])

        # Step 4: Initialize ParagraphState and run evolution loop
        # Get config values
        population_size = self.paragraph_fusion_config.get("candidate_population_size", 5)
        max_generations = self.paragraph_fusion_config.get("max_generations_per_slot", 10)
        lock_threshold = self.paragraph_fusion_config.get("lock_threshold", 0.9)

        # Initialize ParagraphState
        num_slots = len(templates)
        state = ParagraphState(
            original_text=paragraph,
            templates=templates,
            prop_map=prop_map,
            narrative_roles=narrative_roles,
            candidate_populations=[[] for _ in range(num_slots)],
            best_sentences=[None] * num_slots,
            locked_flags=[False] * num_slots,
            feedback=[[None] * population_size for _ in range(num_slots)],
            generation_count=[0] * num_slots
        )

        # Initialize critic and run evolution loop
        from src.validator.semantic_critic import SemanticCritic
        critic = SemanticCritic(config_path=self.config_path)

        if verbose:
            print(f"  Starting evolution loop (max {max_generations} generations per slot)...")

        for generation in range(max_generations):
            # Check if all slots are locked
            if all(state.locked_flags):
                if verbose:
                    print(f"  ✅ All slots locked after {generation} generations")
                break

            # Generate candidates for unlocked slots
            if verbose:
                unlocked_count = sum(1 for locked in state.locked_flags if not locked)
                print(f"  Generation {generation + 1}/{max_generations}: Generating for {unlocked_count} unlocked slots...")

            try:
                new_populations = self._generate_candidate_populations(
                    state=state,
                    author_name=author_name,
                    style_dna=style_dna,
                    population_size=population_size,
                    verbose=verbose
                )

                # Update candidate populations for unlocked slots
                for i, (locked, new_pop) in enumerate(zip(state.locked_flags, new_populations)):
                    if not locked:
                        # Apply sanity check to filter gibberish
                        filtered_pop = self._sanity_check_candidates(
                            candidates=new_pop,
                            template=state.templates[i],
                            verbose=verbose
                        )
                        state.candidate_populations[i] = filtered_pop
                        state.generation_count[i] += 1
                        if verbose:
                            print(f"    Slot {i}: Generated {len(new_pop)} candidates, {len(filtered_pop)} after sanity check")
                            for idx, candidate in enumerate(filtered_pop[:3]):  # Show first 3
                                print(f"      [{idx}] {candidate[:80]}{'...' if len(candidate) > 80 else ''}")
                            if len(filtered_pop) > 3:
                                print(f"      ... and {len(filtered_pop) - 3} more")

                # Evaluate all candidates
                if verbose:
                    print(f"  Evaluating candidates...")

                evaluation_results = critic.evaluate_candidate_populations(
                    candidate_populations=state.candidate_populations,
                    templates=state.templates,
                    prop_map=state.prop_map,
                    narrative_roles=state.narrative_roles,
                    best_sentences=state.best_sentences,
                    verbose=verbose
                )

                # Update best sentences and lock flags
                for slot_idx, slot_results in enumerate(evaluation_results):
                    if state.locked_flags[slot_idx]:
                        continue  # Skip locked slots

                    if verbose:
                        print(f"    Slot {slot_idx} evaluation results:")
                        for candidate_idx, result in enumerate(slot_results):
                            anchor = result.get("anchor_score", 0.0)
                            semantic = result.get("semantic_score", 0.0)
                            narrative = result.get("narrative_score", 0.0)
                            combined = result.get("combined_score", 0.0)
                            pass_flag = result.get("pass", False)
                            feedback = result.get("feedback", "")[:60]
                            candidate = state.candidate_populations[slot_idx][candidate_idx][:60]
                            status = "✓" if pass_flag else "✗"
                            print(f"      [{candidate_idx}] {status} anchor={anchor:.2f} semantic={semantic:.2f} narrative={narrative:.2f} combined={combined:.2f}")
                            print(f"          Text: {candidate}{'...' if len(state.candidate_populations[slot_idx][candidate_idx]) > 60 else ''}")
                            if feedback:
                                print(f"          Feedback: {feedback}{'...' if len(result.get('feedback', '')) > 60 else ''}")

                    # Find best candidate for this slot
                    best_candidate_idx = None
                    best_score = -1.0

                    for candidate_idx, result in enumerate(slot_results):
                        combined_score = result.get("combined_score", 0.0)
                        if combined_score > best_score:
                            best_score = combined_score
                            best_candidate_idx = candidate_idx

                    if best_candidate_idx is not None:
                        best_result = slot_results[best_candidate_idx]
                        best_candidate = state.candidate_populations[slot_idx][best_candidate_idx]

                        # Update best sentence if this is better
                        if state.best_sentences[slot_idx] is None or best_score > 0.0:
                            state.best_sentences[slot_idx] = best_candidate
                            state.feedback[slot_idx] = [r.get("feedback") for r in slot_results]

                        # Stricter locking: require complete meaning, structure, AND narrative flow
                        anchor_score = best_result.get("anchor_score", 0.0)
                        semantic_score = best_result.get("semantic_score", 0.0)
                        narrative_score = best_result.get("narrative_score", 0.0)

                        if anchor_score >= 1.0 and semantic_score >= 0.95 and narrative_score >= 0.8:
                            state.locked_flags[slot_idx] = True
                            if verbose:
                                print(f"  🔒 Slot {slot_idx} locked (anchor={anchor_score:.2f} >= 1.0, semantic={semantic_score:.2f} >= 0.95, narrative={narrative_score:.2f} >= 0.8)")
                                print(f"      Selected: {best_candidate[:100]}{'...' if len(best_candidate) > 100 else ''}")
                        else:
                            if verbose:
                                reason = []
                                if anchor_score < 1.0:
                                    reason.append(f"anchor={anchor_score:.2f} < 1.0")
                                if semantic_score < 0.95:
                                    reason.append(f"semantic={semantic_score:.2f} < 0.95")
                                print(f"  ⚠ Slot {slot_idx} NOT locked: {', '.join(reason)}")
                                print(f"      Best candidate: {best_candidate[:100]}{'...' if len(best_candidate) > 100 else ''}")

                            # Synthesize feedback for unlocked slots that failed to lock
                            directive, elite_text, elite_feedback = self._synthesize_slot_feedback(
                                slot_results=slot_results,
                                slot_candidates=state.candidate_populations[slot_idx],
                                slot_template=state.templates[slot_idx],
                                slot_propositions=state.prop_map[slot_idx],
                                prev_context=state.best_sentences[slot_idx - 1] if slot_idx > 0 else None,
                                verbose=verbose
                            )

                            # Store synthesized feedback and elite candidate info
                            if directive or elite_text:
                                if state.feedback[slot_idx] is None:
                                    state.feedback[slot_idx] = []
                                # Store as structured format: [SYNTHESIZED] directive | [ELITE_TEXT] text | [ELITE_FEEDBACK] feedback
                                # Insert at beginning to preserve order
                                if directive:
                                    state.feedback[slot_idx].insert(0, f"[SYNTHESIZED] {directive}")
                                if elite_text and elite_feedback:
                                    # Insert after synthesized directive if it exists
                                    insert_pos = 1 if directive else 0
                                    state.feedback[slot_idx].insert(insert_pos, f"[ELITE_TEXT] {elite_text}")
                                    state.feedback[slot_idx].insert(insert_pos + 1, f"[ELITE_FEEDBACK] {elite_feedback}")

            except Exception as e:
                if verbose:
                    print(f"  ⚠ Generation {generation + 1} failed: {e}")
                # Continue to next generation
                continue

        # Build final paragraph from best sentences
        if verbose:
            print(f"  Assembling final paragraph from {len(state.best_sentences)} slots...")
        final_sentences = []
        for i, best_sentence in enumerate(state.best_sentences):
            if best_sentence:
                final_sentences.append(best_sentence)
                if verbose:
                    print(f"    Slot {i}: Using best sentence ({len(best_sentence)} chars)")
                    print(f"      \"{best_sentence[:100]}{'...' if len(best_sentence) > 100 else ''}\"")
            else:
                # Fallback: use first candidate if available, or empty string
                if state.candidate_populations[i]:
                    fallback = state.candidate_populations[i][0]
                    final_sentences.append(fallback)
                    if verbose:
                        print(f"    Slot {i}: ⚠ No best sentence, using first candidate ({len(fallback)} chars)")
                        print(f"      \"{fallback[:100]}{'...' if len(fallback) > 100 else ''}\"")
                else:
                    if verbose:
                        print(f"    Slot {i}: ⚠ No valid candidates, using empty string")
                    final_sentences.append("")

        final_text = " ".join(final_sentences)
        if verbose:
            print(f"  Final paragraph assembled: {len(final_sentences)} sentences, {len(final_text)} chars")
            print(f"  Full text: {final_text[:200]}{'...' if len(final_text) > 200 else ''}")

        # Apply smoothing pass to fix grammar and awkward phrasing
        try:
            final_text = self._smooth_paragraph(final_text, verbose=verbose)
        except Exception as e:
            if verbose:
                print(f"  ⚠ Smoothing pass failed: {e}, using unsmoothed text")

        # Check coherence after smoothing, apply simplification if needed
        try:
            from src.validator.semantic_critic import SemanticCritic
            coherence_critic = SemanticCritic(config_path=self.config_path)
            coherence_score, coherence_reason = coherence_critic._verify_coherence(final_text, verbose=verbose)

            if verbose:
                print(f"  Coherence check after smoothing: {coherence_score:.2f}")

            # If coherence is still low, apply simplification
            if coherence_score < 0.75:
                if verbose:
                    print(f"  ⚠ Coherence still low ({coherence_score:.2f} < 0.75), applying simplification pass...")
                    print(f"      Reason: {coherence_reason}")
                try:
                    final_text = self._simplify_paragraph(final_text, verbose=verbose)
                    # Re-check coherence after simplification
                    coherence_score_after, _ = coherence_critic._verify_coherence(final_text, verbose=False)
                    if verbose:
                        print(f"  Coherence after simplification: {coherence_score_after:.2f}")
                except Exception as e:
                    if verbose:
                        print(f"  ⚠ Simplification pass failed: {e}, using smoothed text")
        except Exception as e:
            if verbose:
                print(f"  ⚠ Coherence check failed: {e}, skipping simplification")

        # Extract rhythm_map from teacher_example for return value
        rhythm_map = None
        try:
            from src.analyzer.structuralizer import extract_paragraph_rhythm
            rhythm_map = extract_paragraph_rhythm(teacher_example)
        except Exception:
            pass

        # Calculate internal_recall (proposition recall)
        try:
            from src.ingestion.blueprint import BlueprintExtractor
            extractor = BlueprintExtractor()
            blueprint = extractor.extract(paragraph)

            # Get author style vector
            try:
                author_style_vector = atlas.get_author_style_vector(author_name)
            except Exception:
                author_style_vector = None

            # Evaluate final text
            final_result = critic.evaluate(
                generated_text=final_text,
                input_blueprint=blueprint,
                propositions=propositions,
                is_paragraph=True,
                author_style_vector=author_style_vector,
                style_lexicon=style_lexicon,
                verbose=verbose
            )
            internal_recall = final_result.get("proposition_recall", 0.0)
        except Exception as e:
            if verbose:
                print(f"  ⚠ Failed to calculate recall: {e}")
            internal_recall = 0.0

        # Restore citations and quotes
        try:
            from src.ingestion.blueprint import BlueprintExtractor
            extractor = BlueprintExtractor()
            blueprint = extractor.extract(paragraph)
            final_text = self._restore_citations_and_quotes(final_text, blueprint)
        except Exception:
            pass

        if verbose:
            print(f"  ✅ Evolution complete. Final paragraph: {len(final_text)} chars, recall: {internal_recall:.2f}")

        return final_text, rhythm_map, teacher_example, internal_recall
