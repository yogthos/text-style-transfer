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
from concurrent.futures import ThreadPoolExecutor, as_completed

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
from src.atlas.paragraph_atlas import ParagraphAtlas
from src.atlas.style_rag import StyleRAG
from src.generator.semantic_translator import SemanticTranslator
from src.generator.content_planner import ContentPlanner
from src.generator.refiner import ParagraphRefiner
from src.validator.statistical_critic import StatisticalCritic
from src.utils.nlp_manager import NLPManager
from src.utils.text_processing import check_zipper_merge, parse_variants_from_response


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
        # Load LLM provider config (for retry settings)
        self.llm_provider_config = self.config.get("llm_provider", {})

        # Initialize ParagraphAtlas and SemanticTranslator
        atlas_path = self.config.get("paragraph_atlas", {}).get("path", "atlas_cache/paragraph_atlas")
        # Author will be set when translate_paragraph_statistical is called
        self.atlas_path = atlas_path
        self.paragraph_atlas = None  # Will be initialized per-author
        self.semantic_translator = SemanticTranslator(config_path=config_path)

        # Initialize StyleRAG (lazy initialization per author)
        self.style_rag = {}  # Dict keyed by author name

        # Load generation config
        self.generation_config = self.config.get("generation", {})

        # Initialize statistical critic
        self.statistical_critic = StatisticalCritic(config_path=config_path)

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

    def _analyze_reference_structure(self, reference_text: str) -> tuple[int, str]:
        """Analyze reference paragraph structure for skeleton mapping.

        Args:
            reference_text: The reference paragraph text

        Returns:
            Tuple of (sentence_count, sentence_analysis_string)
        """
        if not reference_text or not reference_text.strip():
            return 0, "No reference available."

        # Split into sentences (simple approach - can be improved)
        # Remove common abbreviations that end with periods
        import re
        text = reference_text.strip()

        # Split on sentence-ending punctuation, but be careful with abbreviations
        # This is a simple heuristic - could use spaCy for better results
        sentences = re.split(r'(?<=[.!?])\s+', text)
        sentences = [s.strip() for s in sentences if s.strip()]

        # Filter out very short fragments that might be false splits
        sentences = [s for s in sentences if len(s.split()) >= 3]

        sentence_count = len(sentences)

        if sentence_count == 0:
            return 0, "No sentences detected in reference."

        # Analyze each sentence
        analysis_parts = []
        for i, sent in enumerate(sentences, 1):
            word_count = len(sent.split())

            # Detect punctuation
            has_semicolon = ';' in sent
            has_colon = ':' in sent and sent.index(':') < len(sent) - 1  # Not just at end
            has_dash = '—' in sent or '--' in sent or '–' in sent

            # Detect complexity (rough heuristic)
            comma_count = sent.count(',')
            clause_indicators = sum(1 for word in ['that', 'which', 'who', 'when', 'where', 'while', 'although', 'because', 'if'] if word in sent.lower())

            complexity = "simple"
            if word_count > 30 or comma_count >= 3 or clause_indicators >= 2:
                complexity = "complex"
            elif word_count > 15 or comma_count >= 1 or clause_indicators >= 1:
                complexity = "moderate"
            else:
                complexity = "short"

            # Build description
            desc_parts = [f"Sentence {i}: {word_count} words ({complexity})"]
            if has_semicolon:
                desc_parts.append("uses semicolon")
            if has_colon:
                desc_parts.append("uses colon")
            if has_dash:
                desc_parts.append("uses dash")
            if clause_indicators > 0:
                desc_parts.append(f"{clause_indicators} subordinate clause(s)")

            analysis_parts.append(" - ".join(desc_parts))

        analysis_text = "\n".join(analysis_parts)
        return sentence_count, analysis_text

    def _detect_input_perspective(self, text: str) -> str:
        """Detect the perspective of input text.

        Args:
            text: Input text to analyze

        Returns:
            Normalized perspective: 'first_person_singular', 'first_person_plural', or 'third_person'
        """
        if not text or not text.strip():
            return "third_person"

        # Use spaCy for POV detection
        from src.utils.spacy_linguistics import get_pov_pronouns
        from src.utils.nlp_manager import NLPManager

        try:
            nlp = NLPManager.get_nlp()
            doc = nlp(text)
            pov_dict = get_pov_pronouns(doc)

            first_singular = pov_dict["first_singular"]
            first_plural = pov_dict["first_plural"]
            third_person = pov_dict["third_person"]

            # Count occurrences in text
            text_lower = text.lower()
            singular_count = sum(1 for word in first_singular if word in text_lower)
            plural_count = sum(1 for word in first_plural if word in text_lower)
            third_count = sum(1 for word in third_person if word in text_lower)
        except Exception:
            # Fallback to regex if spaCy fails
            import re
            text_lower = text.lower()

            singular_patterns = [r'\bi\b', r'\bme\b', r'\bmy\b', r'\bmine\b', r'\bmyself\b']
            singular_count = sum(len(re.findall(pattern, text_lower)) for pattern in singular_patterns)

            plural_patterns = [r'\bwe\b', r'\bus\b', r'\bour\b', r'\bours\b', r'\bourselves\b']
            plural_count = sum(len(re.findall(pattern, text_lower)) for pattern in plural_patterns)

            third_patterns = [
                r'\bhe\b', r'\bhim\b', r'\bhis\b',
                r'\bshe\b', r'\bher\b', r'\bhers\b',
                r'\bthey\b', r'\bthem\b', r'\btheir\b', r'\btheirs\b'
            ]
            third_count = sum(len(re.findall(pattern, text_lower)) for pattern in third_patterns)

        # Determine dominant perspective
        if singular_count > 0 and singular_count >= plural_count:
            return "first_person_singular"
        elif plural_count > 0 and plural_count > singular_count:
            return "first_person_plural"
        elif third_count > 0:
            return "third_person"
        else:
            # Default to third person if no clear indicators
            return "third_person"

    def _normalize_perspective(self, pov: str, pov_breakdown: Optional[Dict] = None) -> str:
        """Normalize perspective string to standard format.

        Args:
            pov: Perspective string from style profile or input
            pov_breakdown: Optional breakdown dictionary with first_singular/first_plural counts

        Returns:
            Normalized: 'first_person_singular', 'first_person_plural', or 'third_person'
        """
        if not pov:
            return "third_person"

        pov_lower = pov.lower()

        # Check if already in normalized format
        if pov_lower in ["first_person_singular", "first_person_plural", "third_person"]:
            return pov_lower

        # Handle "First Person" - need to check breakdown
        if "first person" in pov_lower:
            if pov_breakdown:
                first_singular = pov_breakdown.get("first_singular", 0)
                first_plural = pov_breakdown.get("first_plural", 0)
                if first_singular > first_plural:
                    return "first_person_singular"
                elif first_plural > first_singular:
                    return "first_person_plural"
                # If equal or both zero, default to singular
                return "first_person_singular"
            else:
                # No breakdown, default to singular
                return "first_person_singular"

        # Handle explicit "First Person Singular" or "First Person Plural"
        if "first person singular" in pov_lower or "first_singular" in pov_lower:
            return "first_person_singular"
        if "first person plural" in pov_lower or "first_plural" in pov_lower:
            return "first_person_plural"

        # Handle "Third Person"
        if "third person" in pov_lower or "third_person" in pov_lower:
            return "third_person"

        # Default to third person
        return "third_person"

    def _load_style_profile(self, author_name: str) -> Optional[Dict]:
        """Load style profile for author.

        Args:
            author_name: Name of the author

        Returns:
            Style profile dictionary or None if not found
        """
        try:
            author_lower = author_name.lower()
            style_profile_path = Path(self.atlas_path) / author_lower / "style_profile.json"

            if not style_profile_path.exists():
                style_profile_path = Path(self.atlas_path) / author_name / "style_profile.json"
                if not style_profile_path.exists():
                    return None

            with open(style_profile_path, 'r') as f:
                return json.load(f)
        except Exception:
            return None

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
                # Use spaCy for question detection
                from src.utils.spacy_linguistics import is_question_sentence
                from src.utils.nlp_manager import NLPManager
                try:
                    nlp = NLPManager.get_nlp()
                    doc = nlp(text)
                    return is_question_sentence(doc)
                except Exception:
                    # Fallback to simple check
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

    def translate_paragraph_statistical(
        self,
        paragraph: str,
        author_name: str,
        prev_archetype_id: Optional[int] = None,
        perspective: Optional[str] = None,
        verbose: bool = False
    ) -> tuple[str, int, float]:
        """Translate paragraph using statistical archetype generation with iterative refinement.

        Uses parallel generation of multiple candidates, selects the best, and iteratively
        refines if compliance is below threshold.

        Args:
            paragraph: Input paragraph to translate
            author_name: Target author name
            prev_archetype_id: Previous archetype ID for Markov chain
            perspective: Optional perspective override (first_person_singular, first_person_plural, third_person)
            verbose: Enable verbose output

        Returns:
            Tuple of (generated_paragraph, archetype_id_used, compliance_score)
        """
        if not paragraph or not paragraph.strip():
            return paragraph, 0, 1.0

        # Initialize ParagraphAtlas for this author if needed
        if self.paragraph_atlas is None or getattr(self.paragraph_atlas, 'author', None) != author_name:
            if verbose:
                print(f"  Initializing ParagraphAtlas for {author_name}...")
            self.paragraph_atlas = ParagraphAtlas(self.atlas_path, author_name)

        # Determine target perspective (before neutralization)
        # Priority: User override > Author profile > Input detection > Default
        if perspective:
            target_pov = self._normalize_perspective(perspective)
            if verbose:
                print(f"  Using user-specified perspective: {target_pov}")
        else:
            # Check config for default perspective
            default_perspective = self.generation_config.get("default_perspective")
            if default_perspective:
                target_pov = self._normalize_perspective(default_perspective)
                if verbose:
                    print(f"  Using config default perspective: {target_pov}")
            else:
                # Try to get from style profile
                style_profile = self._load_style_profile(author_name)
                if style_profile and style_profile.get("pov"):
                    pov_from_profile = style_profile["pov"]
                    pov_breakdown = style_profile.get("pov_breakdown", {})
                    target_pov = self._normalize_perspective(pov_from_profile, pov_breakdown)
                    if verbose:
                        print(f"  Using author profile perspective: {target_pov} (from {pov_from_profile})")
                else:
                    # Detect from input
                    detected = self._detect_input_perspective(paragraph)
                    target_pov = detected if detected != "neutral" else "third_person"
                    if verbose:
                        print(f"  Detected input perspective: {target_pov}")

        # Step 1: Neutralize with perspective
        if verbose:
            print(f"  Extracting neutral summary with perspective: {target_pov}...")
        neutral_text = self.semantic_translator.extract_neutral_summary(paragraph, target_perspective=target_pov)
        if verbose:
            print(f"  Neutral summary: {neutral_text[:100]}{'...' if len(neutral_text) > 100 else ''}")

        # Step 1.5: Retrieve style palette using StyleRAG
        style_palette_fragments = []
        if author_name not in self.style_rag:
            try:
                self.style_rag[author_name] = StyleRAG(self.atlas_path, author_name, self.config_path)
            except Exception as e:
                if verbose:
                    print(f"  ⚠ Could not initialize StyleRAG: {e}")
                self.style_rag[author_name] = None

        if self.style_rag.get(author_name):
            style_rag_config = self.config.get("style_rag", {})
            num_fragments = style_rag_config.get("num_fragments", 8)
            style_palette_fragments = self.style_rag[author_name].retrieve_palette(neutral_text, n=num_fragments)
            if verbose:
                if style_palette_fragments:
                    print(f"  Retrieved {len(style_palette_fragments)} style fragments")
                else:
                    print(f"  ⚠ No style fragments retrieved (collection may not exist)")

        # Format style palette as newline-joined string
        style_palette_text = "\n".join([f'"{fragment}"' for fragment in style_palette_fragments]) if style_palette_fragments else ""

        # Step 2: Select next archetype
        if verbose:
            print(f"  Selecting next archetype (prev: {prev_archetype_id})...")
        target_arch_id = self.paragraph_atlas.select_next_archetype(prev_archetype_id)
        if verbose:
            print(f"  Selected archetype: {target_arch_id}")

        # Step 3: Get archetype description
        archetype_desc = self.paragraph_atlas.get_archetype_description(target_arch_id)
        if verbose:
            print(f"  Archetype stats: {archetype_desc['avg_len']} words/sent, {archetype_desc['avg_sents']} sents, {archetype_desc['burstiness']} burstiness")

        # Step 3.5: Generate style directives by comparing neutral text vs target archetype
        style_directives = self._generate_style_directives(neutral_text, archetype_desc)
        if verbose and style_directives:
            print(f"  Style directives: {style_directives[:100]}{'...' if len(style_directives) > 100 else ''}")

        # Step 3.6: Build style constraints from forensic profile
        style_constraints = self._build_style_constraints(author_name)
        if verbose and style_constraints:
            print(f"  Style constraints loaded from forensic profile")

        # Step 4: MANDATORY - Fetch full paragraph from ChromaDB
        if verbose:
            print(f"  Fetching full example paragraph from ChromaDB...")
        rhythm_reference = self.paragraph_atlas.get_example_paragraph(target_arch_id)

        # Fallback to truncated JSON snippet if ChromaDB fails
        if not rhythm_reference:
            if verbose:
                print(f"  ⚠ ChromaDB retrieval failed, using truncated JSON snippet")
            rhythm_reference = archetype_desc.get('example', '')

        if verbose and rhythm_reference:
            print(f"  Rhythm reference: {rhythm_reference[:100]}{'...' if len(rhythm_reference) > 100 else ''}")

        # Analyze reference structure for skeleton mapping
        reference_sentence_count, reference_sentence_analysis = self._analyze_reference_structure(rhythm_reference)
        if verbose:
            print(f"  Reference structure: {reference_sentence_count} sentences detected")

        # ASSEMBLY LINE ARCHITECTURE: Get structure map (blueprint)
        if verbose:
            print(f"  Extracting structure map from archetype {target_arch_id}...")
        structure_map = self.paragraph_atlas.get_structure_map(target_arch_id)
        if not structure_map:
            if verbose:
                print(f"  ⚠ No structure map available, falling back to holistic generation")
            # Fallback to old approach if structure map unavailable
            return self._translate_paragraph_holistic(
                paragraph, author_name, prev_archetype_id, perspective, verbose,
                neutral_text, target_pov, target_arch_id, archetype_desc,
                style_palette_text, rhythm_reference, reference_sentence_count,
                reference_sentence_analysis, style_directives, style_constraints
            )

        if verbose:
            structure_summary = ", ".join([f"{s['target_len']}w" for s in structure_map])
            print(f"  Structure map: [{structure_summary}] ({len(structure_map)} sentences)")

        # ASSEMBLY LINE ARCHITECTURE: Plan content distribution
        if verbose:
            print(f"  Planning content distribution into {len(structure_map)} slots...")
        content_planner = ContentPlanner(self.config_path)
        content_slots = content_planner.plan_content(neutral_text, structure_map, author_name)
        if verbose:
            print(f"  Content distributed into {len(content_slots)} slots")

        # Get generation config
        max_sentence_retries = self.generation_config.get("max_retries", 3)
        generation_temp = self.generation_config.get("temperature", 0.8)
        generation_max_tokens = self.generation_config.get("max_tokens", 1500)
        compliance_fuzziness = self.generation_config.get("compliance_fuzziness", 0.05)
        sentence_variants_per_attempt = self.generation_config.get("sentence_variants_per_attempt", 5)

        # Load system prompt for sentence generation
        prompts_dir = Path(__file__).parent.parent.parent / "prompts"
        try:
            system_prompt_path = prompts_dir / "translator_statistical_system.md"
            system_prompt = system_prompt_path.read_text().strip()
            system_prompt = system_prompt.format(
                style_palette=style_palette_text if style_palette_text else "",
                author_name=author_name
            )
        except (FileNotFoundError, KeyError):
            system_prompt = f"You are a style translator for {author_name}. Generate sentences that match target lengths while preserving semantic content and author voice."

        # Perspective pronoun mapping
        perspective_pronouns = {
            "first_person_singular": "I, Me, My, Myself, Mine",
            "first_person_plural": "We, Us, Our, Ourselves, Ours",
            "third_person": "The subject, The narrator, or specific names"
        }
        perspective_pronoun_list = perspective_pronouns.get(target_pov, "The subject, The narrator")

        # ASSEMBLY LINE ARCHITECTURE: Build sentence by sentence
        final_sentences = []
        context_so_far = ""
        active_structure_map = []  # NEW: Track only slots that were actually generated

        for slot_idx, slot in enumerate(structure_map):
            content = content_slots[slot_idx] if slot_idx < len(content_slots) else ""
            target_len = slot.get('target_len', 20)
            slot_type = slot.get('type', 'moderate')

            # Check if this is the last sentence
            is_last_sentence = (slot_idx == len(structure_map) - 1)

            # NEW: Handle EMPTY slots
            if not content or content.strip().upper() == "EMPTY":
                if verbose:
                    print(f"  Skipping EMPTY slot {slot_idx + 1}")
                # Skip this slot entirely - don't add to active_structure_map
                continue

            # If valid, add to active map BEFORE generation
            active_structure_map.append(slot)

            if verbose:
                print(f"  Building sentence {slot_idx + 1}/{len(structure_map)}: target {target_len} words ({slot_type})...")

            # Inner loop: Generate and validate THIS sentence
            sentence = None
            for attempt in range(max_sentence_retries):
                # 1. Generate batch of variants
                variants = self._generate_sentence_variants(
                    content=content,
                    target_length=target_len,
                    prev_context=context_so_far,
                    author_name=author_name,
                    n=sentence_variants_per_attempt,
                    slot_type=slot_type,
                    target_perspective=target_pov,
                    perspective_pronouns=perspective_pronoun_list,
                    style_palette=style_palette_text,
                    system_prompt=system_prompt,
                    temperature=generation_temp,
                    max_tokens=generation_max_tokens,
                    is_last_sentence=is_last_sentence,
                    verbose=verbose and attempt == 0
                )

                if verbose and attempt == 0:
                    print(f"    Generated {len(variants)} variants for attempt {attempt + 1}")

                if not variants:
                    if verbose and attempt == 0:
                        print(f"    ⚠ No variants generated, retrying...")
                    continue  # Skip empty results

                # 2. Select best candidate (Zipper-Aware: filters during selection)
                sentence = self._select_best_sentence_variant(
                    variants, target_len, context_so_far, verbose=verbose and attempt == 0
                )

                if not sentence:
                    if verbose:
                        print(f"    ⚠ No valid variant found, retrying...")
                    continue

                if verbose and target_len:
                    word_count = len(sentence.split())
                    print(f"    Selected best variant: {word_count} words (target: {target_len}) from {len(variants)} candidates")

                # Validate strict compliance (math only)
                score, feedback = self.statistical_critic.evaluate_sentence(
                    sentence, target_len
                )

                if verbose:
                    word_count = len(sentence.split())
                    print(f"    Attempt {attempt + 1}: {word_count} words, score: {score:.2f}")

                # Check if this is the final attempt
                is_final_attempt = (attempt == max_sentence_retries - 1)

                # Apply fuzzy threshold on final attempt
                if is_final_attempt:
                    fuzzy_threshold = 1.0 - compliance_fuzziness
                    if score >= fuzzy_threshold:
                        final_sentences.append(sentence)
                        context_so_far += sentence + " "
                        if verbose:
                            print(f"    ✓ Sentence {slot_idx + 1} passed validation (fuzzy threshold: {score:.2f} >= {fuzzy_threshold:.2f})")
                        break  # Success! Move to next sentence
                elif score >= 1.0:  # Passed on non-final attempt!
                    final_sentences.append(sentence)
                    context_so_far += sentence + " "
                    if verbose:
                        print(f"    ✓ Sentence {slot_idx + 1} passed validation")
                    break  # Success! Move to next sentence

                # Retry with specific feedback
                if attempt < max_sentence_retries - 1:
                    # Use feedback to refine (pass string, not dict)
                    refined_result = self._refine_sentence(
                        sentence, feedback, target_len, context_so_far,
                        author_name, target_pov, perspective_pronoun_list,
                        style_palette_text, system_prompt, generation_temp,
                        generation_max_tokens, verbose
                    )
                    # Extract text safely
                    if isinstance(refined_result, dict):
                        sentence = refined_result.get('text', sentence)
                    else:
                        sentence = refined_result if refined_result else sentence

            # If we exhausted retries, use best attempt
            if len(final_sentences) == slot_idx and sentence:
                final_sentences.append(sentence)  # Use last attempt
                context_so_far += sentence + " "
                if verbose:
                    print(f"    ⚠ Sentence {slot_idx + 1} used after {max_sentence_retries} attempts")

        # Combine sentences into paragraph
        final_paragraph = " ".join(final_sentences)

        # BLUEPRINT-AWARE GRADING: Calculate targets from ACTIVE structure map, not original
        # This ensures validation uses the correct sentence count (6 instead of 9 if 3 were EMPTY)
        # CRITICAL: Skip burstiness check for short paragraphs (sentence_count < 3) to avoid low-sample noise
        # The burstiness is already baked into the structure map (e.g., [Short, Long])
        sentence_count = len(active_structure_map)  # Use active count, not original
        blueprint_target_stats = {
            "avg_sents": sentence_count,
            "avg_len": sum(s['target_len'] for s in active_structure_map) / sentence_count if active_structure_map else 0,
            # Skip burstiness check for short paragraphs (statistically noisy)
            # For longer paragraphs, use archetype's burstiness
            "burstiness": None if sentence_count < 3 else archetype_desc.get("burstiness", "Low"),
            "style": archetype_desc.get("style", ""),
            "id": archetype_desc.get("id", 0)
        }

        # Final validation: Check overall compliance against BLUEPRINT targets
        if verbose:
            print(f"  Final paragraph: {len(final_sentences)} sentences")
            print(f"  Blueprint targets: {blueprint_target_stats['avg_sents']} sents, {blueprint_target_stats['avg_len']:.1f} words/sent")
        best_text, best_score, qualitative_feedback = self.statistical_critic.select_best_candidate(
            [final_paragraph], blueprint_target_stats  # Use blueprint, not archetype
        )
        if verbose:
            print(f"  Final compliance score: {best_score:.2f}")

        # LEXICAL DIVERSITY GUARD: Check for repetitive phrasing
        # Check for repetition and action echo
        repetition_issues = self.statistical_critic.check_repetition(final_paragraph)

        # Check for action echo (repeated action verbs across sentences)
        import spacy
        try:
            doc = self.statistical_critic.nlp(final_paragraph)
            sentences = [sent.text.strip() for sent in doc.sents]
            action_echo_issues = self.statistical_critic.check_action_echo(sentences)
            repetition_issues.extend(action_echo_issues)
        except Exception:
            # If action echo check fails, continue with just repetition issues
            pass

        original_repetition_count = len(repetition_issues)  # Store count before repair
        if repetition_issues:
            if verbose:
                print(f"  ⚠ Lexical repetition detected: {len(repetition_issues)} issues")
                for issue in repetition_issues[:3]:  # Show first 3 issues
                    print(f"    - {issue}")
            # Add repetition issues to feedback - repair loop will handle it
            if qualitative_feedback:
                qualitative_feedback += " " + " ".join(repetition_issues)
            else:
                qualitative_feedback = " ".join(repetition_issues)
            # Optionally lower score slightly to ensure repair is triggered if close to threshold
            if best_score >= 0.85:
                best_score = min(best_score, 0.84)  # Force repair for repetition issues

        # Check if refinement is needed (flow/coherence issues OR repetition issues)
        compliance_threshold = self.generation_config.get("compliance_threshold", 0.85)
        compliance_fuzziness = self.generation_config.get("compliance_fuzziness", 0.05)
        fuzzy_threshold = compliance_threshold - compliance_fuzziness

        # Apply fuzzy threshold: if score is close enough, accept without repair
        if best_score >= fuzzy_threshold and best_score < compliance_threshold:
            if verbose:
                print(f"  Score within fuzzy threshold ({best_score:.2f} >= {fuzzy_threshold:.2f}), accepting without repair")
            # Skip repair, accept the paragraph as-is
        elif best_score < fuzzy_threshold and qualitative_feedback:
            if verbose:
                print(f"  Score below fuzzy threshold ({best_score:.2f} < {fuzzy_threshold:.2f}), attempting repair...")

            # Initialize refiner if needed
            if not hasattr(self, 'refiner') or self.refiner is None:
                self.refiner = ParagraphRefiner(self.config_path)

            # Extract forbidden phrases from repetition issues
            forbidden_phrases = []
            for issue in repetition_issues:
                # Parse "Repeated phrase: 'profound void' (appears 2 times)"
                if "'" in issue:
                    try:
                        phrase = issue.split("'")[1]  # Extract text between single quotes
                        forbidden_phrases.append(phrase)
                    except IndexError:
                        pass

            # Attempt repair via repair plan
            try:
                refined_text, structure_delta = self.refiner.refine_via_repair_plan(
                    best_text,
                    qualitative_feedback,
                    structure_map,
                    author_name,
                    verbose,
                    forbidden_phrases=forbidden_phrases  # NEW ARGUMENT
                )

                # Safety check: Handle empty or None repair results
                if not refined_text or len(refined_text.strip()) == 0:
                    if verbose:
                        print(f"  ⚠ Repair returned empty text, keeping original")
                    refined_text = best_text

                if refined_text == best_text:
                    if verbose:
                        print(f"  Repair did not produce changes, keeping original")
                else:
                    # Calculate Delta based on REALITY, not the plan
                    # Count actual sentences in original and refined text
                    try:
                        nlp = NLPManager.get_nlp()
                        original_doc = nlp(best_text)
                        refined_doc = nlp(refined_text)
                        original_count = len(list(original_doc.sents))
                        new_count = len(list(refined_doc.sents))
                        actual_structure_delta = new_count - original_count
                    except Exception:
                        # Fallback: simple sentence count using periods
                        original_count = len([s for s in best_text.split('.') if s.strip()])
                        new_count = len([s for s in refined_text.split('.') if s.strip()])
                        actual_structure_delta = new_count - original_count

                    if verbose and actual_structure_delta != structure_delta:
                        print(f"  Structure delta: planned={structure_delta:+d}, actual={actual_structure_delta:+d} (using actual)")

                    # Use actual delta, not planned delta
                    structure_delta = actual_structure_delta

                    # DYNAMIC RUBRIC ADJUSTMENT: Update blueprint stats to match repair changes
                    adjusted_stats = blueprint_target_stats.copy()

                    if structure_delta != 0:
                        # 1. Update Sentence Count
                        old_count = adjusted_stats.get('avg_sents', blueprint_target_stats.get('avg_sents', 1))
                        new_count = max(1, old_count + structure_delta)
                        adjusted_stats['avg_sents'] = new_count

                        # 2. Update Average Length (Conservation of Mass)
                        # Total words should remain roughly constant, just redistributed
                        old_avg_len = adjusted_stats.get('avg_len', 0)
                        if old_avg_len > 0 and old_count > 0:
                            total_mass = old_avg_len * old_count
                            adjusted_stats['avg_len'] = total_mass / new_count if new_count > 0 else old_avg_len

                        if verbose:
                            print(f"  Adjusting rubric for repair: {old_count} -> {new_count} sents, "
                                  f"{old_avg_len:.1f} -> {adjusted_stats['avg_len']:.1f} avg len")
                    else:
                        # No structural change, use original stats
                        adjusted_stats = blueprint_target_stats

                    # Re-validate refined text with adjusted rubric
                    refined_text, refined_score, _ = self.statistical_critic.select_best_candidate(
                        [refined_text], adjusted_stats
                    )

                    # Check if repetition improved
                    # Check for repetition and action echo in refined text
                    new_repetition_issues = self.statistical_critic.check_repetition(refined_text)
                    try:
                        doc = self.statistical_critic.nlp(refined_text)
                        sentences = [sent.text.strip() for sent in doc.sents]
                        new_action_echo_issues = self.statistical_critic.check_action_echo(sentences)
                        new_repetition_issues.extend(new_action_echo_issues)
                    except Exception:
                        pass
                    new_repetition_count = len(new_repetition_issues)
                    repetition_improved = original_repetition_count > 0 and new_repetition_count < original_repetition_count

                    # Structural Bias: Prefer action over inaction when structural changes occur
                    structure_changed = (structure_delta != 0)
                    score_tied_or_better = (refined_score >= best_score)
                    score_acceptable = (refined_score >= 0.40)  # Safety floor

                    # Improvement-Based Acceptance Logic:
                    # Accept if:
                    # A) Score improved (standard case)
                    # B) OR: Repetition was fixed (quality > strict structure compliance)
                    # C) OR: Structural change achieved and score is acceptable and tied/better (tie-breaker)
                    accept = False
                    reason = ""

                    if refined_score > best_score:
                        # Case A: Clear win
                        accept = True
                        reason = "score_improved"
                    elif repetition_improved:
                        # Case B: Quality win
                        accept = True
                        reason = "repetition_reduced"
                    elif structure_changed and score_acceptable and score_tied_or_better:
                        # Case C: Structural win (Tie-breaker)
                        # If we ordered a split and got a split, we keep it even if length score is messy
                        accept = True
                        reason = "structure_restored"

                    if accept:
                        if verbose:
                            print(f"  ✓ Repair accepted ({reason}). Score: {refined_score:.2f} vs {best_score:.2f}" +
                                  (f", structure change: {structure_delta:+d}" if structure_changed else ""))
                        best_text = refined_text
                        best_score = refined_score
                    else:
                        if verbose:
                            print(f"  Repair rejected: Score {refined_score:.2f} <= {best_score:.2f}, " +
                                  f"repetition {new_repetition_count} >= {original_repetition_count}" +
                                  (f", structure change: {structure_delta:+d}" if structure_changed else ""))
            except Exception as e:
                import traceback
                error_msg = str(e)
                error_type = type(e).__name__

                # Always log repair failures with full context
                print(f"  ⚠ Repair failed ({error_type}): {error_msg}")

                if verbose:
                    print(f"  ⚠ Full traceback:")
                    traceback.print_exc()

                # Check for JSON-related errors
                if "JSON" in str(e) or "json" in str(e).lower() or "sent_index" in str(e):
                    print(f"  ⚠ JSON parsing/access error detected")
                    print(f"  ⚠ This usually means the LLM did not return valid JSON format")
                    print(f"  ⚠ Check the repair_plan_user.md prompt and LLM provider JSON enforcement")

                # Truncate for final message if needed
                if len(error_msg) > 100:
                    error_msg = error_msg[:100] + "..."
                print(f"  ⚠ Keeping original text due to repair failure")

        return best_text, target_arch_id, best_score

    def _build_style_constraints(self, author_name: str) -> str:
        """Build style constraints from forensic profile.

        Loads the style profile JSON and formats it as prompt text.

        Args:
            author_name: Name of the author

        Returns:
            Formatted style constraints string, or empty string if profile not found
        """
        try:
            # Use lowercase author name for directory lookup
            author_lower = author_name.lower()
            style_profile_path = Path(self.atlas_path) / author_lower / "style_profile.json"

            if not style_profile_path.exists():
                # Fallback: try exact case
                style_profile_path = Path(self.atlas_path) / author_name / "style_profile.json"
                if not style_profile_path.exists():
                    return ""

            with open(style_profile_path, 'r') as f:
                profile = json.load(f)

            # Format constraints
            pov = profile.get("pov", "Third Person")
            pov_breakdown = profile.get("pov_breakdown", {})
            rhythm_desc = profile.get("rhythm_desc", "Unknown")
            burstiness = profile.get("burstiness", 0.0)
            common_openers = profile.get("common_openers", [])
            keywords = profile.get("keywords", [])

            # Format POV breakdown
            pov_details = []
            if pov_breakdown.get("first_singular", 0) > 0:
                pov_details.append(f"1st singular: {pov_breakdown['first_singular']}")
            if pov_breakdown.get("first_plural", 0) > 0:
                pov_details.append(f"1st plural: {pov_breakdown['first_plural']}")
            if pov_breakdown.get("third_person", 0) > 0:
                pov_details.append(f"3rd person: {pov_breakdown['third_person']}")
            pov_breakdown_str = ", ".join(pov_details) if pov_details else "N/A"

            # Format openers (top 5)
            openers_str = ", ".join(common_openers[:5]) if common_openers else "N/A"

            # Format keywords (top 15)
            keywords_str = ", ".join(keywords[:15]) if keywords else "N/A"

            constraints = f"""**Voice Constraints (Forensic Profile - MANDATORY):**
1. **Point of View:** {pov} ({pov_breakdown_str}).

2. **Rhythm Constraint:** You must mimic a Burstiness of {burstiness}. Vary sentence lengths aggressively—mix very short sentences (5-10 words) with longer ones (30+ words) to create the characteristic 'spiky' rhythm. Target: {rhythm_desc}.

3. **Opener Constraint (MANDATORY):** At least 30% of your sentences MUST begin with one of the following Author Signature Openers: {openers_str}. Avoid academic transitions like 'Therefore' or 'However' unless they appear in the list.

4. **Vocabulary Seeding:** Before expanding your syntax, anchor your text with these signature terms: {keywords_str}. These words should appear naturally throughout the paragraph, not just once."""

            return constraints

        except Exception as e:
            # Silently fail - return empty string if profile can't be loaded
            return ""

    def _generate_style_directives(self, neutral_text: str, target_stats: Dict) -> str:
        """Generate dynamic style directives by comparing neutral text stats vs target archetype stats.

        Args:
            neutral_text: The neutral logical summary text
            target_stats: Target archetype dictionary with stats (noun_ratio, verb_ratio, adj_ratio, clause_density, etc.)

        Returns:
            String containing qualitative style transformation directives
        """
        if not neutral_text or not neutral_text.strip():
            return ""

        # Analyze neutral text using spaCy
        nlp = NLPManager.get_nlp()
        doc = nlp(neutral_text)
        sents = list(doc.sents)

        if not sents:
            return ""

        # Calculate neutral text stats (same metrics as in build_paragraph_atlas.py)
        total_tokens = len(doc)
        verb_count = len([t for t in doc if t.pos_ == "VERB"])

        neutral_noun_ratio = len([t for t in doc if t.pos_ in ["NOUN", "PROPN"]]) / total_tokens if total_tokens > 0 else 0
        neutral_verb_ratio = verb_count / total_tokens if total_tokens > 0 else 0
        neutral_adj_ratio = len([t for t in doc if t.pos_ == "ADJ"]) / total_tokens if total_tokens > 0 else 0

        # Clause density
        clause_count = len([t for t in doc if t.dep_ in ["mark", "advcl"]])
        neutral_clause_density = clause_count / len(sents) if len(sents) > 0 else 0

        # Get target stats (with fallbacks if not present)
        target_noun_ratio = target_stats.get("noun_ratio", 0.2)
        target_verb_ratio = target_stats.get("verb_ratio", 0.15)
        target_adj_ratio = target_stats.get("adj_ratio", 0.1)
        target_clause_density = target_stats.get("clause_density", 1.0)

        directives = []

        # Compare noun ratio (nominalization)
        noun_delta = target_noun_ratio - neutral_noun_ratio
        if noun_delta > 0.05:  # Target has significantly more nouns
            directives.append("Use nominalization: turn actions into nouns (e.g., 'the transformation' instead of 'transforms').")
        elif noun_delta < -0.05:  # Target has significantly fewer nouns
            directives.append("Use more action verbs: prefer active constructions over nominalizations.")

        # Compare verb ratio
        verb_delta = target_verb_ratio - neutral_verb_ratio
        if verb_delta > 0.05:
            directives.append("Increase verb usage: use more action-oriented language.")
        elif verb_delta < -0.05:
            directives.append("Reduce verb density: use more nominal constructions.")

        # Compare adjective ratio
        adj_delta = target_adj_ratio - neutral_adj_ratio
        if adj_delta < -0.03:  # Target has fewer adjectives
            directives.append("Remove decorative adjectives: focus on essential descriptive words only.")
        elif adj_delta > 0.03:  # Target has more adjectives
            directives.append("Add descriptive adjectives: enrich the text with appropriate modifiers.")

        # Compare clause density (complexity)
        clause_delta = target_clause_density - neutral_clause_density
        if clause_delta > 0.5:  # Target has significantly more complex clauses
            directives.append("Use complex nested clauses: build sophisticated sentence structures with subordinate and relative clauses.")
        elif clause_delta < -0.5:  # Target has simpler structure
            directives.append("Simplify clause structure: reduce nested clauses and use more straightforward constructions.")

        # Check burstiness if available
        target_burstiness = target_stats.get("burstiness", "Low")
        if isinstance(target_burstiness, str) and target_burstiness.lower() == "high":
            # Calculate actual burstiness of neutral text
            sent_lens = [len(sent) for sent in sents]
            if len(sent_lens) > 1:
                import numpy as np
                neutral_avg_len = np.mean(sent_lens)
                neutral_burstiness = np.std(sent_lens) / neutral_avg_len if neutral_avg_len > 0 else 0
                if neutral_burstiness < 0.4:  # Neutral is too uniform
                    directives.append("Alternate strictly between very short and very long sentences: create high variation in sentence length for natural rhythm.")

        if not directives:
            return "Maintain the current stylistic balance while matching the target archetype."

        return " ".join(directives)

    def _generate_sentence_for_slot(
        self,
        content: str,
        target_length: int,
        prev_context: str,
        author_name: str,
        slot_type: str,
        target_perspective: str,
        perspective_pronouns: str,
        style_palette: str,
        system_prompt: str,
        temperature: float,
        max_tokens: int,
        is_last_sentence: bool = False,
        verbose: bool = False
    ) -> str:
        """Generate one sentence for a specific slot.

        Args:
            content: Content to express in this sentence
            target_length: Target word count
            prev_context: Previous sentences for flow
            author_name: Author name
            slot_type: Sentence type (simple/moderate/complex)
            target_perspective: Target POV
            perspective_pronouns: Pronoun list for POV
            style_palette: Style fragments
            system_prompt: System prompt
            temperature: Generation temperature
            max_tokens: Max tokens
            verbose: Verbose output

        Returns:
            Generated sentence (string, never dict)
        """
        prompts_dir = Path(__file__).parent.parent.parent / "prompts"
        try:
            template_path = prompts_dir / "sentence_worker_user.md"
            template = template_path.read_text().strip()
        except FileNotFoundError:
            # Fallback template
            template = """# Task: Write ONE Sentence

## Content to Express:
{slot_content}

## Target Length:
{target_length} words (strict constraint)

## Previous Context:
{prev_context}

## Author Voice:
Adopt the voice of {author_name}.

## Instructions:
1. Write exactly ONE sentence expressing the content above.
2. The sentence must be approximately {target_length} words (within 15% tolerance).
3. The sentence should flow naturally from the previous context.
4. Use the author's distinctive voice and vocabulary.

Output only the sentence, no explanations.
"""

        # 1. Build the Constraint String (with safety fallback)
        anti_echo_section = ""  # Default to empty string
        if prev_context:
            prev_words = prev_context.strip().split()
            if len(prev_words) >= 3:
                # Protect against Markdown injection in the previous text
                clean_prev = " ".join(prev_words[:4]).replace("`", "").replace("*", "")
                anti_echo_section = f"**DO NOT start with:** '{clean_prev}...'"

        # 2. Build ending constraint for last sentence
        ending_constraint = ""
        ending_constraint_instruction = ""
        if is_last_sentence:
            ending_constraint = (
                "**GROUNDING CONSTRAINT:** This is the final sentence of the paragraph. "
                "Do NOT summarize, moralize, or use abstract concepts like 'lesson', 'meaning', or 'significance'. "
                "End with a sensory detail (sight, sound, touch, smell) or a specific concrete object. "
                "Avoid phrases like 'in conclusion', 'ultimately', 'the lesson is', or abstract nouns."
            )
            ending_constraint_instruction = "6. **GROUNDING:** " + ending_constraint.replace("**GROUNDING CONSTRAINT:**", "").strip()

        # 3. Safe Formatting (use dictionary to ensure all keys present)
        prompt_params = {
            "slot_content": content,
            "target_length": target_length,
            "prev_context": prev_context,
            "anti_echo_section": anti_echo_section,  # Must be present even if empty
            "author_name": author_name,
            "ending_constraint": ending_constraint,  # New parameter
            "ending_constraint_instruction": ending_constraint_instruction  # New parameter
        }

        user_prompt = template.format(**prompt_params)

        try:
            result = self.llm_provider.call(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                model_type="editor",
                require_json=False,
                temperature=temperature,
                max_tokens=max_tokens
            )
            # CRITICAL: Return string, never dict
            sentence = result.strip() if result else ""
            # Remove any trailing punctuation issues
            if sentence and not sentence.endswith(('.', '!', '?')):
                sentence += '.'
            return sentence
        except Exception as e:
            if verbose:
                print(f"    ⚠ Error generating sentence: {e}")
            return ""

    def _generate_sentence_variants(
        self,
        content: str,
        target_length: int,
        prev_context: str,
        author_name: str,
        n: int = 5,
        slot_type: str = "moderate",
        target_perspective: str = "first_person_singular",
        perspective_pronouns: str = "I, Me, My, Myself, Mine",
        style_palette: str = "",
        system_prompt: str = "",
        temperature: float = 0.8,
        max_tokens: int = 1500,
        is_last_sentence: bool = False,
        verbose: bool = False
    ) -> List[str]:
        """Generate N variants of a sentence for a specific slot.

        Args:
            content: Content to express in this sentence
            target_length: Target word count
            prev_context: Previous sentences for flow
            author_name: Author name
            n: Number of variants to generate
            slot_type: Sentence type (simple/moderate/complex)
            target_perspective: Target POV
            perspective_pronouns: Pronoun list for POV
            style_palette: Style fragments
            system_prompt: System prompt
            temperature: Generation temperature
            max_tokens: Max tokens
            verbose: Verbose output

        Returns:
            List of variant strings
        """
        prompts_dir = Path(__file__).parent.parent.parent / "prompts"
        try:
            template_path = prompts_dir / "sentence_worker_user.md"
            template = template_path.read_text().strip()
        except FileNotFoundError:
            # Fallback template
            template = """# Task: Write ONE Sentence

## Content to Express:
{slot_content}

## Target Length:
{target_length} words (strict constraint)

## Previous Context:
{prev_context}

{anti_echo_section}

## Author Voice:
Adopt the voice of {author_name}.

## Instructions:
1. Write exactly ONE sentence expressing the content above.
2. The sentence must be approximately {target_length} words (within 15% tolerance).
3. The sentence should flow naturally from the previous context.
4. **ANTI-ECHO:** Do NOT start with the same words as the Previous Context.
5. Use the author's distinctive voice and vocabulary.

Output only the sentence, no explanations.
"""

        # Build prompt requesting N variants
        user_content = f"""Generate {n} different variants of the sentence.
Strictly follow the constraints below.
**Output Format:** Output each variant on a new line starting with 'VAR:'. Do not number them.

## Content to Express:
{content}

## Target Length:
{target_length} words (strict constraint)

## Previous Context:
{prev_context}
"""

        # 1. Build the Constraint String (with safety fallback)
        anti_echo_section = ""  # Default to empty string
        if prev_context:
            prev_words = prev_context.strip().split()
            if len(prev_words) >= 3:
                # Protect against Markdown injection in the previous text
                clean_prev = " ".join(prev_words[:4]).replace("`", "").replace("*", "")
                anti_echo_section = f"**DO NOT start with:** '{clean_prev}...'"

        # Build ending constraint for last sentence
        ending_constraint = ""
        if is_last_sentence:
            ending_constraint = (
                "**GROUNDING CONSTRAINT:** This is the final sentence of the paragraph. "
                "Do NOT summarize, moralize, or use abstract concepts like 'lesson', 'meaning', or 'significance'. "
                "End with a sensory detail (sight, sound, touch, smell) or a specific concrete object. "
                "Avoid phrases like 'in conclusion', 'ultimately', 'the lesson is', or abstract nouns."
            )

        # Build ending constraint instruction
        ending_constraint_instruction = ""
        if is_last_sentence and ending_constraint:
            ending_constraint_instruction = f"5. **GROUNDING:** {ending_constraint.replace('**GROUNDING CONSTRAINT:**', '').strip()}"
            final_instruction = "6. Use the author's distinctive voice and vocabulary."
        else:
            final_instruction = "5. Use the author's distinctive voice and vocabulary."

        user_content += f"""
{anti_echo_section}
{ending_constraint}

## Author Voice:
Adopt the voice of {author_name}.

## Instructions:
1. Write exactly ONE sentence expressing the content above.
2. The sentence must be approximately {target_length} words (within 15% tolerance).
3. The sentence should flow naturally from the previous context.
4. **ANTI-ECHO:** Do NOT start with the same words as the Previous Context.
{ending_constraint_instruction}
{final_instruction}

Output only the sentence, no explanations.
"""

        try:
            response = self.llm_provider.call(
                system_prompt=f"You are a sentence generator. Generate {n} variants. Output each on a new line with 'VAR:' prefix.",
                user_prompt=user_content,
                model_type="editor",
                require_json=False,
                temperature=temperature,
                max_tokens=max_tokens
            )

            # Parse variants using shared utility
            variants = parse_variants_from_response(response, verbose=verbose)

            if variants:
                return variants
            else:
                # Fallback: if no variants found, return empty list
                if verbose:
                    print(f"      ⚠ No variants parsed from response, using fallback")
                return []

        except Exception as e:
            if verbose:
                print(f"      ⚠ Error generating variants: {e}, using fallback")
            return []

    def _select_best_sentence_variant(
        self,
        variants: List[str],
        target_length: int,
        prev_context: str,
        verbose: bool = False
    ) -> Optional[str]:
        """Select best sentence variant based on format compliance, zipper check, and length proximity.

        Args:
            variants: List of variant strings
            target_length: Target word count
            prev_context: Previous context for zipper check
            verbose: Verbose output

        Returns:
            Best variant string, or None if no valid variants
        """
        if not variants:
            return None

        valid_candidates = []

        for v in variants:
            # 1. Format check (basic validation)
            if not v or len(v.split()) < 3:
                continue

            # 2. ZIPPER CHECK (The Optimization)
            # If this variant echoes the previous sentence, disqualify it immediately
            if prev_context and check_zipper_merge(prev_context, v):
                if verbose:
                    print(f"      Variant filtered (Zipper Check): {v[:50]}...")
                continue

            valid_candidates.append(v)

        if not valid_candidates:
            # Fallback: ignore zipper check if all failed (let the main loop handle the retry)
            if verbose:
                print(f"      ⚠ All variants failed zipper check, using best anyway")
            valid_candidates = variants

        if not valid_candidates:
            return None

        # 3. Select by Length from VALID candidates only
        best = min(valid_candidates, key=lambda v: abs(len(v.split()) - target_length))
        return best

    def _refine_sentence(
        self,
        sentence: str,
        feedback: str,
        target_length: int,
        prev_context: str,
        author_name: str,
        target_perspective: str,
        perspective_pronouns: str,
        style_palette: str,
        system_prompt: str,
        temperature: float,
        max_tokens: int,
        verbose: bool = False
    ) -> str:
        """Refine a sentence based on feedback.

        Args:
            sentence: Current sentence to refine
            feedback: Feedback from critic
            target_length: Target word count
            prev_context: Previous sentences
            author_name: Author name
            target_perspective: Target POV
            perspective_pronouns: Pronoun list
            style_palette: Style fragments
            system_prompt: System prompt
            temperature: Generation temperature
            max_tokens: Max tokens
            verbose: Verbose output

        Returns:
            Refined sentence (string, never dict)
        """
        prompts_dir = Path(__file__).parent.parent.parent / "prompts"
        try:
            template_path = prompts_dir / "sentence_refinement_user.md"
            template = template_path.read_text().strip()
        except FileNotFoundError:
            # Fallback template
            template = """# Task: Fix Sentence Length

## Current Sentence:
"{current_sentence}"

## Feedback:
{feedback}

## Target Length:
{target_length} words

## Previous Context:
{prev_context}

## Instructions:
Apply the feedback to fix the sentence length. Do not change the meaning or author's voice.
Output only the corrected sentence.
"""

        user_prompt = template.format(
            current_sentence=sentence,
            feedback=feedback,
            target_length=target_length,
            prev_context=prev_context
        )

        try:
            result = self.llm_provider.call(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                model_type="editor",
                require_json=False,
                temperature=temperature,
                max_tokens=max_tokens
            )
            # CRITICAL: Return string, never dict
            refined = result.strip() if result else sentence
            # Remove any trailing punctuation issues
            if refined and not refined.endswith(('.', '!', '?')):
                refined += '.'
            return refined
        except Exception as e:
            if verbose:
                print(f"    ⚠ Error refining sentence: {e}")
            return sentence  # Return original on error

    def _calculate_burstiness_from_map(self, structure_map: List[Dict]) -> str:
        """Calculate burstiness classification from structure map variation.

        Args:
            structure_map: List of slot dicts with target_len

        Returns:
            Burstiness classification: "High", "Low", or "Moderate"
        """
        if not structure_map or len(structure_map) < 2:
            return "Low"  # Single sentence or empty = low variation

        lengths = [s.get('target_len', 0) for s in structure_map]
        mean_len = sum(lengths) / len(lengths) if lengths else 0

        if mean_len == 0:
            return "Low"

        # Calculate coefficient of variation (std dev / mean)
        variance = sum((x - mean_len) ** 2 for x in lengths) / len(lengths)
        std_dev = variance ** 0.5
        cv = std_dev / mean_len if mean_len > 0 else 0

        # Classify based on coefficient of variation
        if cv > 0.4:
            return "High"
        elif cv < 0.15:
            return "Low"
        else:
            return "Moderate"

    def _translate_paragraph_holistic(
        self,
        paragraph: str,
        author_name: str,
        prev_archetype_id: Optional[int],
        perspective: Optional[str],
        verbose: bool,
        neutral_text: str,
        target_pov: str,
        target_arch_id: int,
        archetype_desc: Dict,
        style_palette_text: str,
        rhythm_reference: str,
        reference_sentence_count: int,
        reference_sentence_analysis: str,
        style_directives: str,
        style_constraints: str
    ) -> tuple[str, int, float]:
        """Fallback holistic generation when structure map unavailable.

        This is the old holistic approach, kept as fallback.
        """
        if verbose:
            print(f"  Using holistic generation (fallback)")
        # For now, return neutral text with low score
        # In a full implementation, this would run the old holistic loop
        return neutral_text, target_arch_id, 0.5

