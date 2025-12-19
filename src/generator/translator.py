"""Style translator for Pipeline 2.0.

This module translates semantic blueprints into styled text using
few-shot examples from a rhetorically-indexed style atlas.
"""

import json
import re
import time
import requests
from pathlib import Path
from typing import List, Tuple, Dict, Optional, Set
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
from src.generator.mutation_operators import PARAGRAPH_FUSION_PROMPT


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
        # Load paragraph fusion config
        self.paragraph_fusion_config = self.config.get("paragraph_fusion", {})
        # Load LLM provider config (for retry settings)
        self.llm_provider_config = self.config.get("llm_provider", {})

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
        # This saves token costs by filtering before expensive skeleton extraction
        length_filtered = []
        for example in examples:
            example_len = len(example.split())

            # Filter: 0.5x to 2.5x length (stricter than complexity gate)
            if 0.5 * input_len <= example_len <= 2.5 * input_len:
                length_filtered.append(example)

        if verbose:
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

    def _calculate_skeleton_adherence(self, candidate: str, skeleton: str) -> float:
        """Calculate skeleton adherence score using anchor word overlap.

        Extracts function words (anchor words) from skeleton and checks
        what percentage appear in candidate in roughly the same order.

        Args:
            candidate: Generated candidate text.
            skeleton: Skeleton template with placeholders.

        Returns:
            Adherence score 0.0-1.0 (matched_anchors / total_anchors).
        """
        if not candidate or not skeleton:
            return 0.0

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

            survivors.append({
                "text": candidate_text,
                "skeleton": skeleton,
                "source_example": candidate_data.get("source_example", ""),
                "adherence_score": adherence_score,
                "recall_score": recall_score,
                "style_density": style_density,
                "score": critic_result.get("score", 0.0),
                "semantic_similarity": semantic_similarity,
                "critic_result": critic_result
            })

        # Sort survivors by style_density (descending) and return Top K
        survivors.sort(key=lambda x: x["style_density"], reverse=True)

        if verbose:
            print(f"  🏟️  Arena Results: {len(survivors)} survivors from {len(candidates)} candidates")
            if survivors:
                print(f"    🏆 Top {min(3, len(survivors))} survivors:")
                for i, surv in enumerate(survivors[:3]):
                    print(f"      {i+1}. Style: {surv['style_density']:.2f}, Recall: {surv['recall_score']:.2f}, Score: {surv['score']:.2f}")
                    print(f"         Text: {surv['text'][:80]}...")
            else:
                print(f"    ⚠ No survivors - all candidates failed filters")

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

            deltas = []
            if recall_score < 0.8:
                # Missing keywords - extract missing ones from blueprint
                missing_keywords = []
                blueprint_keywords = list(blueprint.core_keywords)[:5]
                text_lower = text.lower()
                for keyword in blueprint_keywords:
                    if keyword.lower() not in text_lower:
                        missing_keywords.append(keyword)
                if missing_keywords:
                    deltas.append(f"Missing keywords: {', '.join(missing_keywords[:3])}")

            if style_density < 0.1:
                deltas.append("Boring style, needs jargon")

            if logic_fail:
                deltas.append("Logic contradiction detected")

            parent_deltas.append({
                "text": text,
                "deltas": deltas,
                "strengths": []
            })

            # Identify strengths
            if style_density > 0.2:
                parent_deltas[-1]["strengths"].append("Great style")
            if recall_score > 0.9:
                parent_deltas[-1]["strengths"].append("Perfect keywords")
            if parent.get("adherence_score", 0.0) > 0.9:
                parent_deltas[-1]["strengths"].append("Strong structure")

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
            for skeleton, source_example in compatible_skeletons:
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

                    # Breed children from top parents
                    top_parents = survivors[:top_k_parents]
                    children = self._breed_children(
                        parents=top_parents,
                        blueprint=blueprint,
                        author_name=author_name,
                        style_dna=style_dna,
                        rhetorical_type=rhetorical_type,
                        style_lexicon=style_lexicon,
                        num_children=breeding_children,
                        verbose=verbose
                    )

                    if not children:
                        if verbose:
                            print(f"    ⚠ No children generated, stopping evolution")
                        break

                    # Evaluate children in arena
                    child_candidates = [{
                        "text": c,
                        "skeleton": best_survivor["skeleton"],
                        "source_example": best_survivor.get("source_example", "")
                    } for c in children]
                    child_survivors = self._run_arena(
                        candidates=child_candidates,
                        blueprint=blueprint,
                        style_dna_dict=style_dna_dict,
                        verbose=verbose
                    )

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

                # Step B: Population - Generate 3 candidates using selected strategy
                candidates = self._generate_population_with_operator(
                    parent_draft=best_draft,
                    blueprint=blueprint,
                    author_name=author_name,
                    style_dna=style_dna,
                    rhetorical_type=rhetorical_type,
                    operator_type=operator_type,
                    temperature=current_temp,
                    num_candidates=3,
                    verbose=verbose,
                    style_lexicon=style_lexicon,
                    style_structure=style_structure,
                    style_tone=style_tone,
                    rag_example=rag_example
                )

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
        verbose: bool = False
    ) -> tuple[str, Optional[List[Dict]], Optional[str]]:
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
            Generated paragraph in target style.
        """
        if not paragraph or not paragraph.strip():
            return paragraph, None, None

        # Step 1: Extract atomic propositions (with citations bound to facts)
        if verbose:
            print(f"  Extracting atomic propositions from paragraph...")
        propositions = self.proposition_extractor.extract_atomic_propositions(paragraph)
        if verbose:
            print(f"  Extracted {len(propositions)} propositions: {propositions[:3]}...")

        if not propositions:
            # Fallback: use original paragraph
            return paragraph, None, None

        # Step 1.5: Extract direct quotations separately (for quote preservation)
        quote_pattern = r'["\'](?:[^"\']|(?<=\\)["\'])*["\']'
        extracted_quotes = []
        for match in re.finditer(quote_pattern, paragraph):
            quote_text = match.group(0)
            # Only consider substantial quotations (more than just punctuation)
            if len(quote_text.strip('"\'')) > 2:
                extracted_quotes.append(quote_text)

        if verbose and extracted_quotes:
            print(f"  Extracted {len(extracted_quotes)} direct quotations")

        # Extract expected citations from input paragraph (for verification)
        citation_pattern = r'\[\^\d+\]'
        expected_citations = set(re.findall(citation_pattern, paragraph))
        if verbose and expected_citations:
            print(f"  Expected citations: {sorted(expected_citations)}")

        # Step 2: Retrieve complex/long style examples
        # For paragraph fusion, we want LONG examples (ignore length mismatch filters)
        num_examples = self.paragraph_fusion_config.get("num_style_examples", 5)
        retrieval_pool_size = self.paragraph_fusion_config.get("retrieval_pool_size", 20)

        if verbose:
            print(f"  Retrieving {retrieval_pool_size} examples for complexity filtering...")

        # Get examples using existing method (will use 4-tier fallback)
        from src.atlas.rhetoric import RhetoricalType, RhetoricalClassifier
        classifier = RhetoricalClassifier()
        rhetorical_type = classifier.classify_heuristic(paragraph)

        # Retrieve examples (wide net - don't filter by input length)
        # Support dual-author blending
        if secondary_author:
            # Split retrieval pool between two authors based on blend_ratio
            # ratio=0.7 means 70% from primary, 30% from secondary
            primary_count = max(1, int(retrieval_pool_size * blend_ratio))
            secondary_count = retrieval_pool_size - primary_count

            # Retrieve from primary author
            raw_examples_primary = atlas.get_examples_by_rhetoric(
                rhetorical_type,
                top_k=primary_count,
                author_name=author_name,
                query_text=None  # Don't filter by input length
            )

            # Retrieve from secondary author
            raw_examples_secondary = atlas.get_examples_by_rhetoric(
                rhetorical_type,
                top_k=secondary_count,
                author_name=secondary_author,
                query_text=None  # Don't filter by input length
            )

            # Pool them for teacher selection
            raw_examples = raw_examples_primary + raw_examples_secondary

            if verbose:
                print(f"  Dual-author retrieval: {len(raw_examples_primary)} from {author_name}, "
                      f"{len(raw_examples_secondary)} from {secondary_author} (ratio: {blend_ratio:.2f})")

            # Fallback if primary retrieval failed
            if not raw_examples_primary:
                raw_examples_primary = atlas.get_examples_by_rhetoric(
                    RhetoricalType.OBSERVATION,
                    top_k=primary_count,
                    author_name=author_name,
                    query_text=None
                )
                raw_examples = raw_examples_primary + raw_examples_secondary

            # Fallback if secondary retrieval failed (use primary only)
            if not raw_examples_secondary and raw_examples_primary:
                if verbose:
                    print(f"  ⚠ No examples from {secondary_author}, using primary author only")
                raw_examples = raw_examples_primary
        else:
            # Single author (existing logic)
            raw_examples = atlas.get_examples_by_rhetoric(
                rhetorical_type,
                top_k=retrieval_pool_size,  # Wide net for complexity filtering
                author_name=author_name,
                query_text=None  # Don't filter by input length
            )

            if not raw_examples:
                # Fallback: try any examples from author
                raw_examples = atlas.get_examples_by_rhetoric(
                    RhetoricalType.OBSERVATION,
                    top_k=retrieval_pool_size,
                    author_name=author_name,
                    query_text=None  # Don't filter by input length
                )

        # Filter by complexity (word count, sentence count, structure)
        complex_examples = self._select_complex_examples(
            raw_examples,
            min_words=self.paragraph_fusion_config.get("min_word_count", 30),
            min_sentences=self.paragraph_fusion_config.get("min_sentence_count", 2),
            top_k=num_examples,
            verbose=verbose
        )

        if verbose:
            print(f"  Selected {len(complex_examples)} complex examples after filtering")

        # Step 2.5: Select "Teacher Example" for structural cloning
        # Select the example whose sentence count best matches our proposition count
        import math
        from nltk.tokenize import sent_tokenize

        n_props = len(propositions)
        target_sentences = math.ceil(n_props * 0.6)  # Target ratio: ~1.5 props per sentence

        teacher_example = None  # Will be set when best_match is selected
        rhythm_map = None

        if complex_examples:
            # Composite scoring: sentence count + diversity + positional fit + freshness
            # Load weights from config
            diversity_config = self.paragraph_fusion_config.get("structure_diversity", {})
            count_weight = diversity_config.get("count_match_weight", 0.5)  # Increased from 0.3 to prioritize structure fit
            diversity_weight = diversity_config.get("diversity_weight", 0.4)
            positional_weight = diversity_config.get("positional_weight", 0.3)
            freshness_weight = diversity_config.get("freshness_weight", 0.1)  # Reduced from 2.0 to prevent "Fresh but Terrible" selection
            enabled = diversity_config.get("enabled", True)

            # Positional keyword sets
            OPENER_KEYWORDS = {'the', 'in', 'it', 'this', 'a', 'an', 'when', 'how', 'what', 'why', 'where', 'who'}
            CLOSER_KEYWORDS = {'thus', 'ultimately', 'therefore', 'hence', 'consequently', 'in conclusion', 'finally', 'in the final analysis', 'in sum', 'to conclude'}
            TRANSITION_KEYWORDS = {'however', 'furthermore', 'moreover', 'additionally', 'nevertheless', 'nonetheless', 'meanwhile', 'alternatively'}

            best_match = None
            best_score = -1.0

            # Import functions for structure analysis
            from src.analyzer.structuralizer import extract_paragraph_rhythm, generate_structure_signature

            for example in complex_examples:
                try:
                    # 1. Sentence count match
                    example_sentences = sent_tokenize(example)
                    sentence_count = len([s for s in example_sentences if s.strip()])

                    # Hard filter: Reject examples that are too short to hold the content
                    # (Prevent 13 propositions -> 2 sentence template)
                    min_sentences = max(2, int(target_sentences * 0.5))
                    if sentence_count < min_sentences:
                        if verbose:
                            print(f"    Skipping example with {sentence_count} sentences (minimum: {min_sentences})")
                        continue

                    count_diff = abs(sentence_count - target_sentences)
                    count_match = 1.0 - (count_diff / max(target_sentences, 5))  # Normalized
                    count_match = max(0.0, count_match)  # Ensure non-negative

                    # 2. Extract rhythm map and get diversity score
                    candidate_rhythm_map = extract_paragraph_rhythm(example)
                    if not candidate_rhythm_map:
                        continue  # Skip if rhythm extraction fails

                    signature = generate_structure_signature(candidate_rhythm_map)
                    diversity_score = 1.0  # Default: no penalty
                    if enabled and structure_tracker:
                        diversity_score = structure_tracker.get_diversity_score(signature, candidate_rhythm_map)

                    # 3. Positional fit
                    opener_val = candidate_rhythm_map[0].get('opener') if candidate_rhythm_map else None
                    opener = (opener_val or "none").lower()
                    positional_fit = 0.8  # Default neutral

                    if position == "OPENER":
                        if opener in OPENER_KEYWORDS:
                            positional_fit = 1.0  # Good
                        elif opener in TRANSITION_KEYWORDS:
                            positional_fit = 0.3  # Bad - don't start with "However"
                        elif opener in CLOSER_KEYWORDS:
                            positional_fit = 0.2  # Very bad
                        elif opener == 'none' or not opener:
                            positional_fit = 0.9  # Neutral opener is fine
                        else:
                            positional_fit = 0.7  # Neutral
                    elif position == "CLOSER":
                        if opener in CLOSER_KEYWORDS:
                            positional_fit = 1.0  # Good
                        elif opener in OPENER_KEYWORDS:
                            positional_fit = 0.6  # Acceptable but not ideal
                        elif opener == 'none' or not opener:
                            positional_fit = 0.8  # Neutral
                        else:
                            positional_fit = 0.7  # Neutral
                    else:  # BODY
                        positional_fit = 0.8  # Most structures acceptable

                    # 4. Opener diversity penalty
                    if enabled and structure_tracker and opener and opener != 'none':
                        opener_penalty = structure_tracker.get_opener_penalty(opener)
                        diversity_score *= opener_penalty

                    # 5. Freshness score (prefer unused examples)
                    is_used = used_examples and (example in used_examples)
                    freshness_score = 0.0 if is_used else 1.0

                    # 6. Composite score
                    composite_score = (
                        count_match * count_weight +
                        diversity_score * diversity_weight +
                        positional_fit * positional_weight +
                        freshness_score * freshness_weight
                    )

                    # Track best candidate
                    if composite_score > best_score:
                        best_score = composite_score
                        best_match = example
                        rhythm_map = candidate_rhythm_map  # Store rhythm map for best match
                        teacher_example = example  # Store for tracking

                except Exception as e:
                    # If analysis fails, skip this example
                    if verbose:
                        print(f"    ⚠ Error analyzing example: {e}")
                    continue

            # Fallback: If composite scoring failed for all examples, use simple length matching
            if best_match is None and complex_examples:
                if verbose:
                    print(f"  ⚠ Composite scoring failed for all examples. Falling back to simple length matching.")

                # Fallback: simple sentence count matching (original behavior)
                best_match = None
                best_diff = float('inf')

                for example in complex_examples:
                    try:
                        example_sentences = sent_tokenize(example)
                        sentence_count = len([s for s in example_sentences if s.strip()])
                        diff = abs(sentence_count - target_sentences)

                        if diff < best_diff:
                            best_diff = diff
                            best_match = example
                    except Exception:
                        continue

                if best_match:
                    teacher_example = best_match
                    from src.analyzer.structuralizer import extract_paragraph_rhythm
                    rhythm_map = extract_paragraph_rhythm(teacher_example)
                    if verbose:
                        sentence_count = len(rhythm_map) if rhythm_map else 0
                        print(f"  Selected fallback teacher example with {sentence_count} sentences (target: {target_sentences})")
                else:
                    teacher_example = None

            if best_match:
                # teacher_example already set above (or in fallback)
                # rhythm_map already extracted above (or set in fallback)

                # Validation: Check rhythm_map sentence count before generation
                if rhythm_map and len(rhythm_map) < target_sentences * 0.4:
                    if verbose:
                        print(f"  ⚠ Warning: Template too short ({len(rhythm_map)} vs {target_sentences}). Quality may be affected.")

                if verbose:
                    if rhythm_map:
                        sentence_count = len(rhythm_map)
                        mismatch = abs(sentence_count - target_sentences)
                        print(f"  Selected teacher example with {sentence_count} sentences (target: {target_sentences})")
                        if mismatch > 2:
                            print(f"  ⚠ Warning: Large sentence count mismatch ({mismatch} sentences). Quality may be affected.")
                        rhythm_summary = [f"{r['length']} {r['type']}" for r in rhythm_map[:3]]
                        print(f"  Rhythm map: {rhythm_summary}...")
                    else:
                        print(f"  Selected teacher example (no rhythm map extracted)")

        # Step 3: Extract style DNA if not provided
        if not style_dna:
            try:
                from src.analyzer.style_extractor import StyleExtractor
                style_extractor = StyleExtractor(config_path=self.config_path)

                if secondary_author:
                    # Extract DNA from primary author's examples (from complex_examples pool)
                    # Note: complex_examples already contains mixed examples, but we extract
                    # primary DNA from the primary examples we retrieved earlier
                    style_dna = style_extractor.extract_style_dna(complex_examples)

                    # Extract secondary author's DNA separately for lexicon fusion
                    # Get examples from secondary author for DNA extraction
                    secondary_examples = atlas.get_examples_by_rhetoric(
                        RhetoricalType.OBSERVATION,  # Use generic type for DNA extraction
                        top_k=5,
                        author_name=secondary_author,
                        query_text=None
                    )
                    secondary_dna = style_extractor.extract_style_dna(secondary_examples) if secondary_examples else None

                    # Merge lexicons (union of top words from both)
                    if secondary_dna and isinstance(secondary_dna, dict) and isinstance(style_dna, dict):
                        primary_lexicon = style_dna.get("lexicon", [])[:15]
                        secondary_lexicon = secondary_dna.get("lexicon", [])[:15]
                        blended_lexicon = list(set(primary_lexicon) | set(secondary_lexicon))
                        style_dna["lexicon"] = blended_lexicon

                        if verbose:
                            print(f"  Blended lexicon: {len(blended_lexicon)} words "
                                  f"({len(primary_lexicon)} from {author_name}, "
                                  f"{len(secondary_lexicon)} from {secondary_author})")
                else:
                    style_dna = style_extractor.extract_style_dna(complex_examples)

                if verbose:
                    print(f"  Extracted style DNA: {style_dna.get('tone', 'Unknown')} tone")
            except Exception as e:
                # Gracefully handle style DNA extraction failure
                if verbose:
                    print(f"  ⚠ Style DNA extraction failed: {e}, continuing without style DNA")
                style_dna = None  # Continue without style DNA

        # Extract style_lexicon for use in evaluation and repair
        style_lexicon = None
        if style_dna and isinstance(style_dna, dict):
            style_lexicon = style_dna.get("lexicon", [])

        # Step 4: Format and call LLM with PARAGRAPH_FUSION_PROMPT
        propositions_list = "\n".join([f"- {prop}" for prop in propositions])
        style_examples_text = "\n\n".join([f"Example {i+1}: \"{ex}\"" for i, ex in enumerate(complex_examples[:3])])

        # Extract lexicon and format mandatory vocabulary section
        mandatory_vocabulary = ""
        if style_dna and isinstance(style_dna, dict):
            lexicon = style_dna.get("lexicon", [])
            if lexicon:
                # Get ratio from config (default to 1.0 if missing for backward compatibility)
                ratio = self.paragraph_fusion_config.get("style_lexicon_ratio", 1.0)

                # Calculate count based on ratio (allow 0 if ratio is 0.0 to disable style injection)
                count = int(len(lexicon) * ratio) if ratio > 0.0 else 0

                # Only include MANDATORY_VOCABULARY section if count > 0
                if count > 0:
                    top_lexicon = lexicon[:count]
                    lexicon_text = ", ".join(top_lexicon)

                    # Dynamic instruction based on ratio
                    if ratio < 0.3:
                        instruction = "Sprinkle these style markers sparingly."
                    elif ratio > 0.7:
                        instruction = "Heavily saturate the text with this vocabulary."
                    else:
                        instruction = "Integrate these words naturally."

                    mandatory_vocabulary = f"""### MANDATORY VOCABULARY:
You MUST use at least 3-5 distinct words from this list in your paragraph: {lexicon_text}
These words are characteristic of the target author's voice. {instruction}"""
                # If count == 0, mandatory_vocabulary remains empty string (no section added)

        # Extract rhetorical connectors from examples
        rhetorical_connectors = ""
        # Common transition phrases to look for in examples
        common_connectors = [
            "furthermore", "moreover", "consequently", "therefore", "thus", "hence",
            "it follows that", "in this way", "in this manner", "accordingly",
            "as a result", "for this reason", "indeed", "in fact", "specifically",
            "in particular", "notably", "significantly", "importantly", "crucially"
        ]
        # Extract connectors found in examples
        found_connectors = []
        examples_text_combined = " ".join(complex_examples[:3]).lower()
        for connector in common_connectors:
            if connector in examples_text_combined:
                found_connectors.append(connector)

        if found_connectors:
            connectors_text = ", ".join(found_connectors[:10])  # Limit to 10
            rhetorical_connectors = f"""### RHETORICAL CONNECTORS:
Use these transition phrases to link the propositions naturally: {connectors_text}
These connectors match the author's style and help create flowing, complex sentences."""

        # Build citation instruction dynamically - only include if citations exist
        # Silence is golden: if no citations, don't mention them (prevents LLM hallucination)
        if expected_citations:
            citation_instruction = """5. **Citations:** The Source Propositions contain citations (e.g., `[^1]`, `[^2]`). You MUST include these citations in your output, placed immediately after the claim they support. Do not drop or swap them. Each citation must stay with its original fact."""
            citation_output_instruction = "- Include all citations from the propositions (placed after their relevant claims)"
        else:
            citation_instruction = ""
            citation_output_instruction = ""

        # Format structural blueprint from rhythm map
        structural_blueprint = ""
        if rhythm_map and len(rhythm_map) > 0:
            blueprint_lines = []
            blueprint_lines.append("### STRUCTURAL BLUEPRINT:")
            blueprint_lines.append("You must structure your paragraph to match this rhythm exactly. Distribute your Atomic Propositions into this container. Merge or split them as needed to fit the sentence types. Follow this sentence-by-sentence blueprint exactly. If the blueprint asks for a Short Sentence, do not write a long one.")
            blueprint_lines.append("")
            for i, spec in enumerate(rhythm_map):
                length = spec['length']
                sent_type = spec['type']
                opener = spec['opener']

                # Build description
                desc_parts = [length]
                if sent_type == 'question':
                    desc_parts.append('rhetorical question')
                elif sent_type == 'conditional':
                    desc_parts.append('conditional')
                else:
                    desc_parts.append('declarative statement')

                opener_text = ""
                if opener:
                    opener_text = f" starting with '{opener}'"

                blueprint_lines.append(f"Sentence {i+1}: {' '.join(desc_parts).capitalize()}{opener_text}.")

            structural_blueprint = "\n".join(blueprint_lines)
        else:
            # Fallback: no structural blueprint
            structural_blueprint = ""

        prompt = PARAGRAPH_FUSION_PROMPT.format(
            propositions_list=propositions_list,
            proposition_count=len(propositions),
            style_examples=style_examples_text,
            mandatory_vocabulary=mandatory_vocabulary,
            rhetorical_connectors=rhetorical_connectors,
            citation_instruction=citation_instruction,
            citation_output_instruction=citation_output_instruction,
            structural_blueprint=structural_blueprint
        )

        if verbose:
            print(f"  Generating paragraph fusion with {len(propositions)} propositions...")

        # Get timeout from config (default 60 seconds for paragraph fusion)
        api_timeout = self.paragraph_fusion_config.get("api_timeout", 60)

        # Add retry logic with exponential backoff for timeout errors
        import time
        import requests
        max_retries = self.llm_provider_config.get("max_retries", 3)
        retry_delay = self.llm_provider_config.get("retry_delay", 2)  # Start delay in seconds

        response = None
        try:
            for attempt in range(max_retries):
                try:
                    response = self.llm_provider.call(
                        system_prompt="You are a ghostwriter creating paragraphs in a specific style. Output ONLY valid JSON arrays.",
                        user_prompt=prompt,
                        model_type="editor",
                        require_json=True,
                        temperature=0.7,  # Moderate temperature for style variation
                        max_tokens=self.translator_config.get("max_tokens", 500),
                        timeout=api_timeout
                    )
                    break  # Success, exit retry loop
                except (RuntimeError, requests.exceptions.RequestException, requests.exceptions.Timeout, requests.exceptions.ConnectionError) as e:
                    error_str = str(e)
                    is_timeout = "timeout" in error_str.lower() or "timed out" in error_str.lower() or "Read timed out" in error_str

                    if attempt < max_retries - 1 and is_timeout:
                        # Retry on timeout with exponential backoff
                        if verbose:
                            print(f"  ⚠ Timeout error (attempt {attempt + 1}/{max_retries}): {error_str[:100]}...")
                            print(f"  ⏳ Waiting {retry_delay}s before retry...")
                        time.sleep(retry_delay)
                        retry_delay *= 2  # Exponential backoff
                        continue
                    else:
                        # Final attempt failed or non-timeout error - re-raise
                        raise
        except Exception as e:
            # Gracefully handle LLM generation failure
            if verbose:
                print(f"  ⚠ LLM generation failed: {e}, returning original paragraph")
            return paragraph, rhythm_map, teacher_example  # Fallback to original

        if response is None:
            if verbose:
                print(f"  ⚠ No response from LLM, returning original paragraph")
            return paragraph, rhythm_map, teacher_example  # Fallback to original

        try:

            # Parse JSON response
            variations = self._extract_json_list(response)
            num_variations = self.paragraph_fusion_config.get("num_variations", 5)
            variations = variations[:num_variations]

            if not variations:
                if verbose:
                    print(f"  ⚠ No variations generated, using fallback")
                return paragraph, rhythm_map, teacher_example  # Fallback to original

            # Step 5: Evaluate all variations and select best using tiered selection logic
            if verbose:
                print(f"  Generated {len(variations)} variations, evaluating all...")

            # Initialize critic for evaluation
            from src.validator.semantic_critic import SemanticCritic
            from src.ingestion.blueprint import BlueprintExtractor
            critic = SemanticCritic(config_path=self.config_path)
            # Pass rhythm map to critic for verification (soft check)
            if rhythm_map:
                critic._rhythm_map = rhythm_map
            extractor = BlueprintExtractor()

            # Get author style vector for holistic evaluation
            try:
                author_style_vector = atlas.get_author_style_vector(author_name)
            except Exception:
                author_style_vector = None

            # Create a minimal blueprint for evaluation (critic needs it for paragraph mode)
            blueprint = extractor.extract(paragraph)

            # Evaluate all variations
            evaluated_candidates = []
            proposition_recall_threshold = self.paragraph_fusion_config.get("proposition_recall_threshold", 0.8)

            for i, variation in enumerate(variations):
                if verbose:
                    print(f"    Evaluating variation {i+1}/{len(variations)}...")

                # Clean variation (remove "Variation X:" prefix if present)
                cleaned_variation = re.sub(r'^Variation \d+:\s*', '', variation.strip())

                # Task 4: Verify citations and quotes are present
                found_citations = set(re.findall(citation_pattern, cleaned_variation))
                missing_citations = expected_citations - found_citations

                # Check quotes (exact match required)
                found_quotes = []
                for match in re.finditer(quote_pattern, cleaned_variation):
                    quote_text = match.group(0)
                    if len(quote_text.strip('"\'')) > 2:
                        found_quotes.append(quote_text)
                missing_quotes = [q for q in extracted_quotes if q not in found_quotes]

                # If citations or quotes are missing, mark for repair (but still evaluate)
                needs_artifact_repair = len(missing_citations) > 0 or len(missing_quotes) > 0

                if verbose and needs_artifact_repair:
                    if missing_citations:
                        print(f"    Variation {i+1}: Missing citations: {sorted(missing_citations)}")
                    if missing_quotes:
                        print(f"    Variation {i+1}: Missing quotes: {len(missing_quotes)}")

                critic_result = critic.evaluate(
                    generated_text=cleaned_variation,
                    input_blueprint=blueprint,
                    propositions=propositions,
                    is_paragraph=True,
                    author_style_vector=author_style_vector,
                    style_lexicon=style_lexicon
                )

                evaluated_candidates.append({
                    "text": cleaned_variation,
                    "recall": critic_result.get("proposition_recall", 0.0),
                    "style_alignment": critic_result.get("style_alignment", 0.0),
                    "score": critic_result.get("score", 0.0),
                    "result": critic_result,
                    "missing_citations": missing_citations,
                    "missing_quotes": missing_quotes,
                    "needs_artifact_repair": needs_artifact_repair
                })

                if verbose:
                    print(f"    Variation {i+1}: recall={critic_result.get('proposition_recall', 0.0):.2f}, "
                          f"style={critic_result.get('style_alignment', 0.0):.2f}, "
                          f"score={critic_result.get('score', 0.0):.2f}")

            # Tiered Selection Logic:
            # 1. Filter qualified candidates (recall >= threshold)
            qualified_candidates = [
                c for c in evaluated_candidates
                if c["recall"] >= proposition_recall_threshold
            ]

            if qualified_candidates:
                # 2. From qualified pool, prioritize candidates without missing citations/quotes
                # Then pick highest composite score (Style + Meaning)
                complete_candidates = [c for c in qualified_candidates if not c.get("needs_artifact_repair", False)]
                if complete_candidates:
                    best_candidate = max(complete_candidates, key=lambda x: x["score"])
                else:
                    # If all have missing artifacts, pick best and repair
                    best_candidate = max(qualified_candidates, key=lambda x: x["score"])

                # Repair missing citations/quotes if needed
                if best_candidate.get("needs_artifact_repair", False):
                    repaired = self._repair_missing_artifacts(
                        best_candidate["text"],
                        best_candidate.get("missing_citations", set()),
                        best_candidate.get("missing_quotes", []),
                        style_lexicon=style_lexicon,
                        verbose=verbose
                    )
                    if repaired:
                        best_candidate["text"] = repaired
                        if verbose:
                            print(f"  ✓ Repaired missing citations/quotes")

                if verbose:
                    print(f"  ✓ Selected qualified candidate: recall={best_candidate['recall']:.2f}, "
                          f"score={best_candidate['score']:.2f}")

                # Remove phantom citations before returning
                final_text = self._remove_phantom_citations(best_candidate["text"], expected_citations)
                # Restore valid citations if needed
                from src.ingestion.blueprint import BlueprintExtractor
                extractor = BlueprintExtractor()
                blueprint = extractor.extract(paragraph)
                final_text = self._restore_citations_and_quotes(final_text, blueprint)

                return final_text, rhythm_map, teacher_example
            else:
                # 3. Fallback: No qualified candidates, pick highest recall (salvage meaning)
                best_candidate = max(evaluated_candidates, key=lambda x: x["recall"])
                if verbose:
                    print(f"  ⚠ No qualified candidates (recall >= {proposition_recall_threshold}), "
                          f"using best recall: {best_candidate['recall']:.2f}")

                # Task 3: "Delta" Repair Loop - if recall < threshold, attempt repair
                repair_threshold = 0.7
                max_repair_attempts = 2
                repair_attempt = 0
                current_best = best_candidate

                while current_best["recall"] < proposition_recall_threshold and repair_attempt < max_repair_attempts:
                    repair_attempt += 1
                    if verbose:
                        print(f"  🔧 Repair loop attempt {repair_attempt}/{max_repair_attempts}: recall {current_best['recall']:.2f} < {proposition_recall_threshold:.2f}")

                    # Progressive threshold relaxation: use lower threshold (0.25) for second repair attempt
                    use_relaxed_threshold = (repair_attempt == 2 and current_best["recall"] < proposition_recall_threshold)
                    if use_relaxed_threshold and verbose:
                        print(f"  ⚠ Using relaxed threshold (0.25) for second repair attempt")

                    # Identify missing propositions from current best candidate
                    recall_details = current_best["result"].get("recall_details", {})
                    missing_propositions = recall_details.get("missing", [])

                    if not missing_propositions:
                        break  # No missing propositions, no need to repair

                    if verbose:
                        print(f"  Missing {len(missing_propositions)} propositions:")
                        for prop in missing_propositions[:5]:
                            print(f"    - {prop}")
                        if len(missing_propositions) > 5:
                            print(f"    ... and {len(missing_propositions) - 5} more")

                    # Get preserved propositions for full checklist
                    preserved_propositions = recall_details.get("preserved", [])
                    all_propositions = preserved_propositions + missing_propositions

                    # Extract similarity scores and best matches for missing propositions
                    similarity_scores = recall_details.get("scores", {})
                    missing_with_scores = []
                    for prop in missing_propositions[:10]:
                        score = similarity_scores.get(prop, 0.0)
                        best_match_key = f"{prop}_best_match"
                        best_match = similarity_scores.get(best_match_key, "N/A")
                        missing_with_scores.append((prop, score, best_match))

                    # Generate repair prompt with FULL proposition checklist
                    all_propositions_list = "\n".join([f"{i+1}. {prop}" for i, prop in enumerate(all_propositions)])

                    # Build missing list with similarity scores and examples
                    missing_list_parts = []
                    for i, (prop, score, best_match) in enumerate(missing_with_scores, 1):
                        match_preview = best_match[:80] + "..." if isinstance(best_match, str) and len(best_match) > 80 else (str(best_match)[:80] + "..." if best_match != "N/A" else "No close match found")
                        missing_list_parts.append(
                            f"{i}. {prop}\n"
                            f"   Similarity: {score:.3f} (below threshold)\n"
                            f"   Best match in generated text: \"{match_preview}\"\n"
                            f"   → You need to include the KEY PHRASES from this proposition more explicitly."
                        )
                    missing_list = "\n".join(missing_list_parts)

                    # Add style preservation instructions if lexicon available
                    style_preservation = ""
                    if style_lexicon:
                        lexicon_text = ", ".join(style_lexicon[:15])  # Limit to 15
                        style_preservation = f"""

**CRITICAL: Do not lose the style of the original draft. You must maintain the Vocabulary ({lexicon_text}) and the Complex Sentence Structure. Do not simplify the text just to add the facts."""

                    repair_prompt = f"""You need to re-write this paragraph to ensure ALL facts are present.

**THE FULL CHECKLIST:**
You must ensure EVERY one of the following propositions is present in the text. Check them off one by one as you write:
{all_propositions_list}

**MISSING ITEMS (CRITICAL - These were not detected in the last draft):**
{missing_list}

**HOW TO FIX:**
For each missing proposition above, you must include its KEY WORDS explicitly. For example:
- If the proposition is "The core message is an admission", you MUST include the words "core", "message", and "admission" in close proximity.
- Do NOT paraphrase too heavily - the semantic similarity detector needs to see the actual content words.
- You can embed them in complex sentences like: "At its heart, this admission's core message reveals that..." or "The fundamental essence of this recognition is an admission that..."

**Style Constraint:** Maintain the complex sentence structure and vocabulary you used before. Do not simplify the text just to add the facts.{style_preservation}

Original paragraph (for style reference):
"{current_best['text']}"

**Task:** Rewrite the paragraph completely. Ensure ALL propositions from the checklist above are present with their KEY WORDS explicitly included. Integrate them naturally into flowing, complex sentences. Do not just append facts at the end.

Generate 3 new variations that include ALL facts from the checklist. Output as a JSON array of strings:
[
  "Repaired variation 1...",
  "Repaired variation 2...",
  "Repaired variation 3..."
]"""

                    try:
                        # Generate repair variations
                        repair_response = self.llm_provider.call(
                            system_prompt="You are a ghostwriter repairing a paragraph to include missing facts. Output ONLY valid JSON arrays.",
                            user_prompt=repair_prompt,
                            model_type="editor",
                            require_json=True,
                            temperature=0.6,  # Slightly lower temperature for focused repair
                            max_tokens=self.translator_config.get("max_tokens", 500)
                        )

                        repair_variations = self._extract_json_list(repair_response)
                        repair_variations = repair_variations[:3]  # Limit to 3

                        if repair_variations:
                            if verbose:
                                print(f"  Generated {len(repair_variations)} repair variations, evaluating...")

                            # Evaluate repair variations
                            for i, repair_var in enumerate(repair_variations):
                                cleaned_repair = re.sub(r'^Repaired variation \d+:\s*', '', repair_var.strip())

                                if verbose:
                                    print(f"    Repair {i+1} text (first 100 chars): {cleaned_repair[:100]}...")

                                # Verify citations and quotes in repair variations
                                found_citations_repair = set(re.findall(citation_pattern, cleaned_repair))
                                missing_citations_repair = expected_citations - found_citations_repair

                                found_quotes_repair = []
                                for match in re.finditer(quote_pattern, cleaned_repair):
                                    quote_text = match.group(0)
                                    if len(quote_text.strip('"\'')) > 2:
                                        found_quotes_repair.append(quote_text)
                                missing_quotes_repair = [q for q in extracted_quotes if q not in found_quotes_repair]

                                needs_artifact_repair_repair = len(missing_citations_repair) > 0 or len(missing_quotes_repair) > 0

                                # Use relaxed threshold for second repair attempt if first didn't improve
                                if use_relaxed_threshold:
                                    # For second repair attempt, use relaxed threshold (0.25) for proposition recall
                                    # but still evaluate style with normal critic
                                    repair_result = critic.evaluate(
                                        generated_text=cleaned_repair,
                                        input_blueprint=blueprint,
                                        propositions=propositions,
                                        is_paragraph=True,
                                        author_style_vector=author_style_vector
                                    )
                                    # Re-check proposition recall with relaxed threshold (0.25)
                                    relaxed_recall, relaxed_details = critic._check_proposition_recall(
                                        cleaned_repair,
                                        propositions,
                                        similarity_threshold=0.25  # Relaxed threshold for second attempt
                                    )
                                    # Update the result with relaxed recall
                                    repair_result["proposition_recall"] = relaxed_recall
                                    repair_result["recall_details"] = relaxed_details
                                    # Recalculate score with relaxed recall
                                    meaning_weight = self.paragraph_fusion_config.get("meaning_weight", 0.6)
                                    style_weight = self.paragraph_fusion_config.get("style_alignment_weight", 0.4)
                                    style_alignment = repair_result.get("style_alignment", 0.0)
                                    repair_result["score"] = (relaxed_recall * meaning_weight) + (style_alignment * style_weight)
                                    # Update pass status based on relaxed recall
                                    repair_result["pass"] = relaxed_recall >= proposition_recall_threshold
                                else:
                                    repair_result = critic.evaluate(
                                        generated_text=cleaned_repair,
                                        input_blueprint=blueprint,
                                        propositions=propositions,
                                        is_paragraph=True,
                                        author_style_vector=author_style_vector
                                    )

                                # Extract similarity scores for debugging
                                repair_recall_details = repair_result.get("recall_details", {})
                                repair_similarity_scores = repair_recall_details.get("scores", {})
                                repair_missing = repair_recall_details.get("missing", [])

                                if verbose and repair_missing:
                                    print(f"    Repair {i+1} missing propositions with similarity scores:")
                                    for missing_prop in repair_missing[:5]:
                                        score = repair_similarity_scores.get(missing_prop, 0.0)
                                        best_match_key = f"{missing_prop}_best_match"
                                        best_match = repair_similarity_scores.get(best_match_key, "N/A")
                                        if best_match == "N/A" and score > 0:
                                            # Fallback: try to find best match from generated sentences
                                            best_match = "See generated text"
                                        print(f"      [FAIL] '{missing_prop[:60]}...' - Best Match: '{best_match[:60] if isinstance(best_match, str) else str(best_match)[:60]}...' (Score: {score:.3f})")

                                evaluated_candidates.append({
                                    "text": cleaned_repair,
                                    "recall": repair_result.get("proposition_recall", 0.0),
                                    "style_alignment": repair_result.get("style_alignment", 0.0),
                                    "score": repair_result.get("score", 0.0),
                                    "result": repair_result,
                                    "missing_citations": missing_citations_repair,
                                    "missing_quotes": missing_quotes_repair,
                                    "needs_artifact_repair": needs_artifact_repair_repair
                                })

                                if verbose:
                                    print(f"    Repair {i+1}: recall={repair_result.get('proposition_recall', 0.0):.2f}, "
                                          f"style={repair_result.get('style_alignment', 0.0):.2f}, "
                                          f"score={repair_result.get('score', 0.0):.2f}")

                            # Re-select best from merged pool (original + repairs)
                            qualified_after_repair = [
                                c for c in evaluated_candidates
                                if c["recall"] >= proposition_recall_threshold
                            ]

                            if qualified_after_repair:
                                best_after_repair = max(qualified_after_repair, key=lambda x: x["score"])
                                if verbose:
                                    print(f"  ✓ Repair {repair_attempt} successful: recall={best_after_repair['recall']:.2f}, "
                                          f"score={best_after_repair['score']:.2f}")

                                # Remove phantom citations before returning
                                final_text = self._remove_phantom_citations(best_after_repair["text"], expected_citations)
                                # Restore valid citations if needed
                                from src.ingestion.blueprint import BlueprintExtractor
                                extractor = BlueprintExtractor()
                                blueprint = extractor.extract(paragraph)
                                final_text = self._restore_citations_and_quotes(final_text, blueprint)

                                return final_text, rhythm_map, teacher_example
                            else:
                                # Still no qualified, but pick best from all (including repairs)
                                best_after_repair = max(evaluated_candidates, key=lambda x: x["recall"])
                                if verbose:
                                    print(f"  ⚠ Repair {repair_attempt} improved to recall={best_after_repair['recall']:.2f} (still below threshold {proposition_recall_threshold:.2f})")

                                # Update current_best for potential second repair pass
                                current_best = best_after_repair
                                # Continue loop to try second repair if needed
                        else:
                            if verbose:
                                print(f"  ⚠ Repair {repair_attempt} generation failed, stopping repair loop")
                            break  # Stop repair loop if generation fails
                    except Exception as e:
                        if verbose:
                            print(f"  ⚠ Repair {repair_attempt} error: {e}, stopping repair loop")
                        break  # Stop repair loop on error

                # Repair missing citations/quotes if needed before returning
                if current_best.get("needs_artifact_repair", False):
                    repaired = self._repair_missing_artifacts(
                        current_best["text"],
                        current_best.get("missing_citations", set()),
                        current_best.get("missing_quotes", []),
                        style_lexicon=style_lexicon,
                        verbose=verbose
                    )
                    if repaired:
                        current_best["text"] = repaired
                        if verbose:
                            print(f"  ✓ Repaired missing citations/quotes before returning")

                # Step 3: Explicit phantom citation removal (sanitize output)
                # Remove any citations that don't exist in the original paragraph
                final_text = current_best["text"]

                # Remove phantom citations
                citation_pattern = r'\[\^\d+\]'
                found_citations = set(re.findall(citation_pattern, final_text))
                phantom_citations = found_citations - expected_citations
                if phantom_citations and verbose:
                    print(f"  🧹 Removing {len(phantom_citations)} phantom citations: {sorted(phantom_citations)}")
                final_text = self._remove_phantom_citations(final_text, expected_citations)

                # Step 4: Use standard restoration tool (integrate with existing system)
                # Create a blueprint from the original paragraph for citation restoration
                from src.ingestion.blueprint import BlueprintExtractor
                extractor = BlueprintExtractor()
                blueprint = extractor.extract(paragraph)
                final_text = self._restore_citations_and_quotes(final_text, blueprint)

                # Return best candidate (either original or after repair attempts)
                if verbose and current_best["recall"] < proposition_recall_threshold:
                    print(f"  ⚠ Final recall {current_best['recall']:.2f} below threshold {proposition_recall_threshold:.2f}, returning best available")
                return final_text, rhythm_map, teacher_example

        except Exception as e:
            error_str = str(e)
            is_timeout = "timeout" in error_str.lower() or "timed out" in error_str.lower() or "Read timed out" in error_str

            if verbose:
                if is_timeout:
                    print(f"  ✗ Paragraph fusion failed: Network timeout after {max_retries} retry attempts. Using fallback.")
                else:
                    print(f"  ✗ Paragraph fusion failed: {error_str[:200]}, using fallback")
            return paragraph, None, None  # Fallback to original

    def _repair_missing_artifacts(
        self,
        candidate_text: str,
        missing_citations: set,
        missing_quotes: List[str],
        style_lexicon: Optional[List[str]] = None,
        verbose: bool = False
    ) -> Optional[str]:
        """Repair missing citations and quotes in generated text.

        Args:
            candidate_text: Generated text that may be missing citations/quotes.
            missing_citations: Set of citation strings (e.g., {"[^1]", "[^2]"}).
            missing_quotes: List of quote strings that should be present.
            style_lexicon: Optional list of style words to preserve in the repair.
            verbose: Whether to print debug information.

        Returns:
            Repaired text with citations/quotes inserted, or None if repair fails.
        """
        if not missing_citations and not missing_quotes:
            return candidate_text  # Nothing to repair

        if verbose:
            print(f"  🔧 Repairing missing artifacts: {len(missing_citations)} citations, {len(missing_quotes)} quotes")

        # Build repair prompt
        repair_parts = []
        if missing_citations:
            citations_list = ", ".join(sorted(missing_citations))
            repair_parts.append(f"Missing citations: {citations_list}. Please re-insert them next to their relevant claims.")
        if missing_quotes:
            quotes_list = "\n".join([f"- {quote}" for quote in missing_quotes[:5]])  # Limit to 5
            repair_parts.append(f"Missing direct quotes (preserve exactly):\n{quotes_list}")

        # Add style preservation instructions if lexicon provided
        style_preservation = ""
        if style_lexicon:
            lexicon_text = ", ".join(style_lexicon[:15])  # Limit to 15
            style_preservation = f"""

**CRITICAL: Do not lose the style of the original draft. You must maintain the Vocabulary ({lexicon_text}) and the Complex Sentence Structure. Do not simplify the text just to add the citations/quotes."""

        repair_prompt = f"""The following paragraph is good but is missing some citations and/or quotes:

{chr(10).join(repair_parts)}

Original paragraph:
"{candidate_text}"{style_preservation}

Rewrite the paragraph to include the missing citations and quotes. Place citations immediately after the claims they support. Include quotes exactly as shown. Maintain the style and structure.

Output as a single paragraph (not JSON array):"""

        try:
            repair_response = self.llm_provider.call(
                system_prompt="You are a ghostwriter repairing a paragraph to include missing citations and quotes. Output the repaired paragraph text directly, not JSON.",
                user_prompt=repair_prompt,
                model_type="editor",
                require_json=False,
                temperature=0.5,  # Lower temperature for precise repair
                max_tokens=self.translator_config.get("max_tokens", 500)
            )

            repaired_text = repair_response.strip()

            # Remove JSON array brackets if present (LLM sometimes returns JSON despite instructions)
            if repaired_text.startswith('[') and repaired_text.endswith(']'):
                try:
                    parsed = json.loads(repaired_text)
                    if isinstance(parsed, list) and len(parsed) > 0:
                        repaired_text = parsed[0]  # Take first element if it's a list
                except json.JSONDecodeError:
                    # If JSON parsing fails, try to extract text between brackets
                    pass

            # Verify repair was successful
            citation_pattern = r'\[\^\d+\]'
            found_citations = set(re.findall(citation_pattern, repaired_text))
            if missing_citations:
                still_missing = missing_citations - found_citations
                if still_missing and verbose:
                    print(f"  ⚠ Some citations still missing after repair: {sorted(still_missing)}")

            # Check quotes (exact match)
            quote_pattern = r'["\'](?:[^"\']|(?<=\\)["\'])*["\']'
            found_quotes = []
            for match in re.finditer(quote_pattern, repaired_text):
                quote_text = match.group(0)
                if len(quote_text.strip('"\'')) > 2:
                    found_quotes.append(quote_text)
            if missing_quotes:
                still_missing_quotes = [q for q in missing_quotes if q not in found_quotes]
                if still_missing_quotes and verbose:
                    print(f"  ⚠ Some quotes still missing after repair: {len(still_missing_quotes)}")

            return repaired_text

        except Exception as e:
            if verbose:
                print(f"  ⚠ Artifact repair failed: {e}")
            return None  # Return None to indicate repair failed

    def _count_words(self, text: str) -> int:
        """Count words in text.

        Args:
            text: Text string.

        Returns:
            Word count.
        """
        return len(text.split())

    def _count_sentences(self, text: str) -> int:
        """Count sentences in text.

        Uses sentence-ending punctuation (. ! ?) to count sentences.

        Args:
            text: Text string.

        Returns:
            Sentence count.
        """
        # Count sentence-ending punctuation
        sentence_endings = text.count('.') + text.count('!') + text.count('?')
        # If no punctuation, treat as single sentence
        return max(1, sentence_endings)

    def _count_clauses(self, text: str) -> int:
        """Count clauses as proxy for complexity.

        Counts commas and conjunctions as indicators of clause density.

        Args:
            text: Text string.

        Returns:
            Approximate clause count.
        """
        # Count commas (often indicate clauses)
        comma_count = text.count(',')
        # Count common conjunctions
        conjunctions = ['and', 'but', 'or', 'nor', 'for', 'so', 'yet', 'because', 'although', 'while']
        conjunction_count = sum(text.lower().count(conj) for conj in conjunctions)
        # Return approximate clause count (commas + conjunctions + 1 base clause)
        return comma_count + conjunction_count + 1

    def _calculate_text_similarity(self, text1: str, text2: str) -> float:
        """Calculate Jaccard similarity between two texts.

        Uses token overlap (intersection over union) for fast deduplication.
        Formula: len(set(a) & set(b)) / len(set(a) | set(b))

        Args:
            text1: First text string.
            text2: Second text string.

        Returns:
            Similarity score 0.0-1.0 (1.0 = identical).
        """
        tokens1 = set(text1.lower().split())
        tokens2 = set(text2.lower().split())

        if not tokens1 or not tokens2:
            return 0.0

        intersection = len(tokens1 & tokens2)
        union = len(tokens1 | tokens2)

        return intersection / union if union > 0 else 0.0

    def _select_complex_examples(
        self,
        examples: List[str],
        min_words: int = 30,
        min_sentences: int = 2,
        top_k: int = 5,
        verbose: bool = False
    ) -> List[str]:
        """Select complex examples based on word count, sentence count, and structure.

        Filters examples to find the most complex, dense paragraphs that can serve
        as "expansion templates" for paragraph fusion.

        Args:
            examples: List of raw example strings from ChromaDB.
            min_words: Minimum word count threshold (default: 30).
            min_sentences: Minimum sentence count threshold (default: 2).
            top_k: Number of examples to return (default: 5).
            verbose: Whether to print debug information.

        Returns:
            List of top K complex examples, sorted by length (descending).
        """
        if not examples:
            return []

        # Step 1: Word Count Filter
        word_filtered = []
        for ex in examples:
            word_count = self._count_words(ex)
            if word_count >= min_words:
                word_filtered.append((ex, word_count))

        if verbose:
            print(f"    Word filter ({min_words}+ words): {len(word_filtered)}/{len(examples)} passed")

        if not word_filtered:
            # If no examples meet word threshold, return empty
            return []

        # Step 2: Sentence Count Filter
        sentence_filtered = []
        for ex, word_count in word_filtered:
            sentence_count = self._count_sentences(ex)
            if sentence_count >= min_sentences:
                sentence_filtered.append((ex, word_count, sentence_count))

        if verbose:
            print(f"    Sentence filter ({min_sentences}+ sentences): {len(sentence_filtered)}/{len(word_filtered)} passed")

        if not sentence_filtered:
            # If no examples meet sentence threshold, return empty
            return []

        # Step 3: Sort by Length (Descending) - CRITICAL: This gives model "expansion permission"
        # Sort by word count descending to prioritize longest, most complex examples
        sentence_filtered.sort(key=lambda x: x[1], reverse=True)

        # Step 4: Deduplication (Remove similar examples, keep longer one)
        unique_examples = []
        for ex, word_count, sentence_count in sentence_filtered:
            is_duplicate = False
            for existing_ex, _, _ in unique_examples:
                similarity = self._calculate_text_similarity(ex, existing_ex)
                if similarity > 0.8:  # >80% token overlap = near duplicate
                    is_duplicate = True
                    if verbose:
                        print(f"    Deduplication: Skipping duplicate (similarity: {similarity:.2f})")
                    break

            if not is_duplicate:
                unique_examples.append((ex, word_count, sentence_count))

        if verbose:
            print(f"    Deduplication: {len(unique_examples)} unique examples from {len(sentence_filtered)} candidates")

        # Step 5: Return Top K (already sorted by length descending)
        result = [ex for ex, _, _ in unique_examples[:top_k]]

        if verbose:
            if result:
                avg_words = sum(self._count_words(ex) for ex in result) / len(result)
                avg_sentences = sum(self._count_sentences(ex) for ex in result) / len(result)
                print(f"    Selected {len(result)} examples (avg: {avg_words:.1f} words, {avg_sentences:.1f} sentences)")

        return result

