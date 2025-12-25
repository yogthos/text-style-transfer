"""Pipeline: Extract-Translate-Refine architecture.

This module implements the pipeline that:
1. Extracts semantic blueprints from input
2. Classifies rhetorical mode
3. Retrieves few-shot examples
4. Translates using StyleTranslator
5. Validates using SemanticCritic
"""

import json
import re
from pathlib import Path
from typing import List, Optional, Callable, Dict
from concurrent.futures import ThreadPoolExecutor
from src.ingestion.blueprint import SemanticBlueprint, BlueprintExtractor
from src.atlas.rhetoric import RhetoricalType, RhetoricalClassifier
from src.generator.translator import StyleTranslator
from src.validator.semantic_critic import SemanticCritic
from src.atlas.builder import StyleAtlas, load_atlas
from src.atlas.style_registry import StyleRegistry
from src.analyzer.style_extractor import StyleExtractor
# PropositionExtractor removed in Phase 1 - replaced by SemanticTranslator
from src.utils.structure_tracker import StructureTracker
from src.analyzer.global_context import GlobalContextAnalyzer
from src.utils.nlp_manager import NLPManager
from src.utils.vocabulary_budget import VocabularyBudget


def _split_into_paragraphs(text: str) -> List[str]:
    """Split text into paragraphs.

    Args:
        text: Input text.

    Returns:
        List of paragraph strings (non-empty).
    """
    # Try double newlines first
    paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]

    # If no double newlines, try single newlines
    if len(paragraphs) == 1 and '\n' in text:
        paragraphs = [p.strip() for p in text.split('\n') if p.strip()]

    # If still only one, treat entire text as one paragraph
    if not paragraphs:
        paragraphs = [text.strip()] if text.strip() else []

    return paragraphs


def _split_into_sentences_safe(paragraph: str) -> List[str]:
    """Split paragraph into sentences, preserving citations.

    This function avoids splitting on periods that are part of citations
    like [^155] or within quoted text.

    Args:
        paragraph: Input paragraph text.

    Returns:
        List of sentence strings (non-empty).
    """
    if not paragraph or not paragraph.strip():
        return []

    # First, identify citation positions to protect them
    citation_pattern = r'\[\^\d+\]'
    citation_positions = []
    for match in re.finditer(citation_pattern, paragraph):
        citation_positions.append((match.start(), match.end()))

    # Find sentence boundaries (periods, exclamation, question marks)
    # But avoid splitting if the period is within a citation bracket
    sentences = []
    current_start = 0
    i = 0

    while i < len(paragraph):
        # Check if we're at a sentence-ending punctuation
        if paragraph[i] in '.!?':
            # Check if this punctuation is part of a citation
            is_in_citation = False
            for cit_start, cit_end in citation_positions:
                if cit_start <= i < cit_end:
                    is_in_citation = True
                    break

            if not is_in_citation:
                # Check if there's whitespace after (sentence boundary)
                # Look ahead for whitespace or end of string
                if i + 1 >= len(paragraph) or paragraph[i + 1].isspace():
                    # Found sentence boundary
                    sentence = paragraph[current_start:i + 1].strip()
                    if sentence:
                        sentences.append(sentence)
                    # Skip the punctuation and any following whitespace
                    i += 1
                    while i < len(paragraph) and paragraph[i].isspace():
                        i += 1
                    current_start = i
                    continue

        i += 1

    # Add remaining text as final sentence
    if current_start < len(paragraph):
        remaining = paragraph[current_start:].strip()
        if remaining:
            sentences.append(remaining)

    # Fallback: if no sentences found, use original splitting method
    if not sentences:
        sentences = re.split(r'[.!?]+\s+', paragraph)
        sentences = [s.strip() for s in sentences if s.strip()]

    return sentences


def _prepare_paragraphs_parallel(
    paragraphs: List[str],
    atlas: StyleAtlas,
    author_name: str,
    extractor: BlueprintExtractor,
    proposition_extractor,  # Removed - no longer used
    translator: StyleTranslator,
    classifier: RhetoricalClassifier,
    verbose: bool = False
) -> List[Dict]:
    """Prepare all paragraphs in parallel (proposition extraction, mapping, skeleton retrieval).

    This function performs independent operations in parallel:
    - Extract propositions
    - Extract blueprints
    - Classify rhetorical type
    - Retrieve skeletons (uses cache from Phase 1.2)

    Args:
        paragraphs: List of paragraph strings to prepare
        atlas: StyleAtlas instance for skeleton retrieval
        author_name: Author name for skeleton retrieval
        extractor: BlueprintExtractor instance
        proposition_extractor: PropositionExtractor instance
        translator: StyleTranslator instance (for skeleton retrieval)
        classifier: RhetoricalClassifier instance
        verbose: Enable verbose logging

    Returns:
        List of preparation results, one per paragraph. Each dict contains:
        - paragraph: str - Original paragraph text
        - propositions: List[str] - Extracted propositions
        - blueprint: SemanticBlueprint - Extracted blueprint
        - rhetorical_type: RhetoricalType - Classified rhetorical type
        - templates: List[str] - Retrieved sentence templates
        - teacher_example: str - Teacher example used for templates
    """
    def prepare_paragraph(para_idx: int, paragraph: str) -> Dict:
        """Prepare a single paragraph (runs in parallel)."""
        try:
            # Extract propositions
            propositions = proposition_extractor.extract_atomic_propositions(paragraph)

            # Extract blueprint
            blueprint = extractor.extract(paragraph)

            # Classify rhetoric
            rhetorical_type = classifier.classify_heuristic(paragraph)

            # Retrieve skeleton (uses cache from Phase 1.2)
            try:
                teacher_example, templates = translator._retrieve_robust_skeleton(
                    rhetorical_type=rhetorical_type.value if hasattr(rhetorical_type, 'value') else str(rhetorical_type),
                    author=author_name,
                    prop_count=len(propositions),
                    atlas=atlas,
                    verbose=verbose
                )
            except Exception as e:
                if verbose:
                    print(f"  Warning: Skeleton retrieval failed for para {para_idx}: {e}")
                templates = ["[NP] [VP] [NP]."]  # Fallback
                teacher_example = None

            return {
                'paragraph': paragraph,
                'propositions': propositions,
                'blueprint': blueprint,
                'rhetorical_type': rhetorical_type,
                'templates': templates,
                'teacher_example': teacher_example
            }
        except Exception as e:
            if verbose:
                print(f"  Error preparing paragraph {para_idx}: {e}")
            # Return minimal result on error
            return {
                'paragraph': paragraph,
                'propositions': [],
                'blueprint': None,
                'rhetorical_type': RhetoricalType.OBSERVATION,
                'templates': ["[NP] [VP] [NP]."],
                'teacher_example': None
            }

    # Execute in parallel
    if verbose:
        print(f"  Preparing {len(paragraphs)} paragraphs in parallel...")

    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(prepare_paragraph, idx, para)
                  for idx, para in enumerate(paragraphs)]
        results = [f.result() for f in futures]

    if verbose:
        print(f"  ✅ Prepared {len(results)} paragraphs")

    return results


def process_text(
    input_text: str,
    atlas: StyleAtlas,
    author_name: str,
    style_dna: str,
    config_path: str = "config.json",
    max_retries: int = 3,
    perspective: Optional[str] = None,
    verbose: bool = False,
    write_callback: Optional[Callable[[Optional[str], bool, bool], None]] = None
) -> List[str]:
    """Process text through the pipeline.

    Args:
        input_text: Input text to transform.
        atlas: StyleAtlas with rhetorical indexing.
        author_name: Target author name.
        style_dna: Style DNA description.
        config_path: Path to config file.
        max_retries: Maximum retry attempts per sentence.
        verbose: Enable verbose logging.
        write_callback: Optional callback function(sentence, is_new_paragraph) to write sentences incrementally.

    Returns:
        List of generated paragraphs (each paragraph is a space-joined string of sentences).
    """
    # Initialize spaCy model early (shared across all components)
    if verbose:
        print("Initializing NLP pipeline...")
    try:
        NLPManager.get_nlp()
        if verbose:
            print("✅ NLP pipeline ready")
    except Exception as e:
        if verbose:
            print(f"⚠ Warning: spaCy model not available: {e}")
            print("  Some features may be limited")

    # Read blending configuration
    try:
        import json
        with open(config_path, 'r') as f:
            config = json.load(f)
        blend_config = config.get("blend", {})
        blend_authors = blend_config.get("authors", [])
        blend_ratio = blend_config.get("ratio", 0.5)

        # Determine primary and secondary authors
        if len(blend_authors) >= 2:
            # Dual-author mode
            primary_author = blend_authors[0]
            secondary_author = blend_authors[1]
            # Override author_name if different from config
            if author_name != primary_author:
                if verbose:
                    print(f"  Note: Using primary author from config: {primary_author} (was: {author_name})")
                author_name = primary_author
        elif len(blend_authors) == 1:
            # Single-author mode (ratio ignored)
            primary_author = blend_authors[0]
            secondary_author = None
            blend_ratio = 0.5  # Not used, but set for consistency
            if author_name != primary_author:
                if verbose:
                    print(f"  Note: Using author from config: {primary_author} (was: {author_name})")
                author_name = primary_author
        else:
            # No blend config or empty, use provided author_name
            secondary_author = None
            blend_ratio = 0.5
    except Exception as e:
        if verbose:
            print(f"  ⚠ Could not read blend config: {e}, using single-author mode")
        secondary_author = None
        blend_ratio = 0.5

    extractor = BlueprintExtractor()
    classifier = RhetoricalClassifier()
    translator = StyleTranslator(config_path=config_path)
    critic = SemanticCritic(config_path=config_path)
    style_extractor = StyleExtractor(config_path=config_path)
    # PropositionExtractor removed - no longer needed

    # Split into paragraphs first
    paragraphs = _split_into_paragraphs(input_text)

    if not paragraphs:
        return []

    # Extract global context if enabled (Read First step)
    global_context = None
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        global_context_config = config.get("global_context", {})
        if global_context_config.get("enabled", True):
            if verbose:
                print("  Extracting Global Context...")
            analyzer = GlobalContextAnalyzer(config_path=config_path)
            global_context = analyzer.analyze_document(input_text, verbose=verbose)
            if verbose:
                thesis_preview = global_context.get('thesis', '')[:100] if global_context.get('thesis') else ''
                if thesis_preview:
                    print(f"  Context Thesis: {thesis_preview}...")
                print(f"  Context Intent: {global_context.get('intent', 'informing')}")
                keywords = global_context.get('keywords', [])[:5]
                if keywords:
                    print(f"  Context Keywords: {', '.join(keywords)}")

        # Initialize vocabulary budget if enabled
        vocab_config = config.get("vocabulary_budget", {})
        if vocab_config.get("enabled", False):
            static_restricted = vocab_config.get("restricted_words", [])

            # Get dynamic blocklist based on perspective (e.g., author name blocking)
            dynamic_restricted = translator._get_dynamic_blocklist(author_name)

            # Merge static and dynamic restricted words
            all_restricted = static_restricted + dynamic_restricted

            max_per_chapter = vocab_config.get("max_per_chapter", 2)
            clustering_distance = vocab_config.get("clustering_distance", 50)
            vocabulary_budget = VocabularyBudget(
                restricted_words=all_restricted,
                max_per_chapter=max_per_chapter,
                clustering_distance=clustering_distance
            )
            if verbose:
                total_restricted = len(all_restricted)
                if dynamic_restricted:
                    print(f"  Vocabulary Budget: Tracking {total_restricted} restricted words ({len(static_restricted)} static + {len(dynamic_restricted)} dynamic, max {max_per_chapter} per chapter)")
                else:
                    print(f"  Vocabulary Budget: Tracking {total_restricted} restricted words (max {max_per_chapter} per chapter)")
        else:
            vocabulary_budget = None
    except Exception as e:
        if verbose:
            print(f"  ⚠ Global context extraction failed: {e}, continuing without context")
        global_context = None
        vocabulary_budget = None

    generated_paragraphs = []
    current_paragraph_sentences = []  # Track sentences for current paragraph
    is_first_paragraph = True

    # Track used examples to prevent repetition
    used_examples = set()

    # Structure tracking for paragraph diversity
    structure_tracker = StructureTracker()
    total_paragraphs = len(paragraphs)

    # Track previous archetype ID for Markov chain continuity
    prev_archetype_id = None

    # Track previous paragraph ID for context reset
    previous_paragraph_id = -1
    previous_generated_text = ""

    # Track previous paragraph text for context continuity (Phase 5)
    prev_paragraph_text = None

    for para_idx, paragraph in enumerate(paragraphs):
        if verbose:
            print(f"\n{'='*60}")
            print(f"Processing Paragraph {para_idx + 1}/{len(paragraphs)}")
            print(f"{'='*60}")
            print(f"Input: {paragraph[:100]}{'...' if len(paragraph) > 100 else ''}")

        # Split paragraph into sentences (preserving citations)
        sentences = _split_into_sentences_safe(paragraph)

        if not sentences:
            continue

        if verbose:
            print(f"Sentences in paragraph: {len(sentences)}")

        # Reset context if crossing paragraph boundary
        if para_idx != previous_paragraph_id:
            previous_generated_text = ""
            previous_paragraph_id = para_idx
            if verbose:
                print(f"  Context reset (new paragraph)")

        # Check if we should skip title case headers (headings)
        # Load config to check both generation.skip_title_case_headers and paragraph_fusion.transcribe_headings_as_is
        skip_title_case_headers = False
        transcribe_headings_as_is = False
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
            # Check generation.skip_title_case_headers (newer setting)
            generation_config = config.get("generation", {})
            skip_title_case_headers = generation_config.get("skip_title_case_headers", False)
            # Also check paragraph_fusion.transcribe_headings_as_is (legacy setting)
            paragraph_fusion_config = config.get("paragraph_fusion", {})
            transcribe_headings_as_is = paragraph_fusion_config.get("transcribe_headings_as_is", False)
        except Exception:
            pass

        # If either setting is enabled and single sentence, check if it looks like a heading
        should_skip_headers = skip_title_case_headers or transcribe_headings_as_is

        if should_skip_headers and len(sentences) == 1:
            single_sentence = sentences[0].strip()
            # Heuristics for heading detection:
            # 1. Short length (typically headings are < 100 chars)
            # 2. All caps or no lowercase letters (common in headings)
            # 3. Title case (most words capitalized, like "The Toolset of Reality: A Practical Introduction")
            # 4. Ends with no sentence-ending punctuation or just colon
            is_short = len(single_sentence) < 100
            is_all_caps = single_sentence.isupper() and len(single_sentence) > 3
            has_no_lowercase = not any(c.islower() for c in single_sentence)

            # Check for title case: most words (excluding first word and common short words) are capitalized
            words = single_sentence.split()
            if len(words) > 1:
                # Count capitalized words (excluding first word and common short words like "a", "of", "the", "to", "in")
                short_words = {'a', 'an', 'the', 'of', 'to', 'in', 'on', 'at', 'for', 'and', 'or', 'but', 'is', 'are', 'was', 'were'}
                capitalized_count = 0
                total_count = 0
                for i, word in enumerate(words):
                    # Strip punctuation from word for comparison (e.g., "Reality:" -> "Reality")
                    clean_word = word.rstrip(':,;.!?')
                    # Skip first word (always capitalized) and short common words
                    if i > 0 and clean_word.lower() not in short_words:
                        total_count += 1
                        if clean_word and clean_word[0].isupper():
                            capitalized_count += 1

                # Consider title case if >50% of non-short words are capitalized
                is_title_case = total_count > 0 and (capitalized_count / total_count) > 0.5
            else:
                is_title_case = False

            stripped = single_sentence.rstrip()
            ends_with_colon = stripped.endswith(':')
            ends_with_sentence_punct = stripped.endswith(('.', '!', '?'))
            # Heading if: ends with colon OR doesn't end with sentence punctuation
            ends_like_heading = ends_with_colon or not ends_with_sentence_punct

            # Consider it a heading if it's short and (all caps OR no lowercase OR title case) and ends like a heading
            looks_like_heading = is_short and (is_all_caps or has_no_lowercase or is_title_case) and ends_like_heading

            if looks_like_heading:
                if verbose:
                    print(f"  ℹ Detected heading, transcribing as-is: {single_sentence[:50]}{'...' if len(single_sentence) > 50 else ''}")
                generated_paragraphs.append(single_sentence)
                if write_callback:
                    is_new_paragraph = True
                    write_callback(single_sentence, is_new_paragraph, is_first_paragraph)
                    if is_first_paragraph:
                        is_first_paragraph = False
                continue  # Skip restyling for this paragraph

        # Try graph-based generation first, fall back to statistical if needed
        if verbose:
            print(f"  Processing paragraph {para_idx + 1}/{total_paragraphs}...")

        # Build document context for graph matching
        document_context = {
            "current_index": para_idx,
            "total_paragraphs": total_paragraphs,
            "prev_paragraph_end_type": None  # Could be enhanced to track previous paragraph end type
        }

        # Check if graph pipeline is enabled
        graph_config = config.get("graph_pipeline", {})
        graph_enabled = graph_config.get("enabled", True)

        # Try graph-based generation first (if enabled)
        generated_paragraph = None
        arch_id = 0
        compliance_score = 1.0

        if graph_enabled:
            try:
                if verbose:
                    print(f"  Attempting graph-based generation...")
                generated_paragraph, arch_id, compliance_score = translator.translate_paragraph_propositions(
                    paragraph,
                    author_name,
                    document_context=document_context,
                    paragraph_index=para_idx,
                    total_paragraphs=total_paragraphs,
                    prev_paragraph_text=prev_paragraph_text,
                    verbose=verbose
                )

                # If graph-based generation returned empty, fall back to statistical
                if generated_paragraph and generated_paragraph.strip():
                    if verbose:
                        print(f"  ✓ Graph-based generation succeeded")
                else:
                    if verbose:
                        print(f"  ⚠ Graph-based generation returned empty, falling back to statistical method")
                    generated_paragraph = None

            except Exception as e:
                if verbose:
                    print(f"  ⚠ Graph-based generation failed: {e}, falling back to statistical method")
                generated_paragraph = None
        else:
            if verbose:
                print(f"  Graph pipeline disabled in config, using statistical method")

        # Fall back to statistical method if graph-based failed or disabled
        if generated_paragraph is None:
            if verbose:
                print(f"  Using statistical paragraph generation mode")
            generated_paragraph, arch_id, compliance_score = translator.translate_paragraph_statistical(
                paragraph,
                author_name,
                prev_archetype_id=prev_archetype_id,
                perspective=perspective,
                verbose=verbose,
                global_context=global_context,
                vocabulary_budget=vocabulary_budget,
                document_context=document_context
            )

        # Update previous archetype ID for Markov chain continuity
        prev_archetype_id = arch_id

        # Check compliance score
        if verbose:
            print(f"  Compliance score: {compliance_score:.2f}")

        # Validate vocabulary budget if enabled
        if vocabulary_budget:
            try:
                vocab_config = config.get("vocabulary_budget", {})
            except:
                vocab_config = {}
            auto_regenerate = vocab_config.get("auto_regenerate_on_violation", True)
            enforce_clustering = vocab_config.get("enforce_clustering", True)
            force_programmatic_fix = vocab_config.get("force_programmatic_fix", False)
            max_retries_before_force = vocab_config.get("max_retries_before_force", None)  # None = wait for all retries

            # Find restricted words in generated paragraph
            found_words = vocabulary_budget.find_restricted_words(generated_paragraph)

            violations = []
            for word, position in found_words:
                # Calculate absolute position in document
                absolute_position = vocabulary_budget._total_words + position

                # Check if allowed
                allowed, reason = vocabulary_budget.check_word_allowed(word, absolute_position)
                if not allowed:
                    violations.append((word, reason))
                    if verbose:
                        print(f"  ⚠ Vocabulary violation: '{word}' - {reason}")

            # Track retry attempts for vocabulary violations
            vocab_retry_count = 0
            max_vocab_retries = max_retries_before_force if max_retries_before_force is not None else 1

            # If violations found, repair the text instead of falling back to statistical generation
            if violations:
                if verbose:
                    print(f"  🔧 Repairing vocabulary violations (preserving graph structure)...")
                # Use vocabulary repair to fix violations while preserving graph structure
                generated_paragraph = translator._repair_vocabulary(
                    generated_paragraph,
                    violations,
                    verbose=verbose
                )
                # Re-check violations after repair
                found_words = vocabulary_budget.find_restricted_words(generated_paragraph)
                violations = []
                for word, position in found_words:
                    absolute_position = vocabulary_budget._total_words + position
                    allowed, reason = vocabulary_budget.check_word_allowed(word, absolute_position)
                    if not allowed:
                        violations.append((word, reason))
                        if verbose:
                            print(f"  ⚠ Vocabulary violation persists after repair: '{word}' - {reason}")

            # Apply programmatic cleanup (sledgehammer) if violations persist and enabled
            # Apply if: force_programmatic_fix is enabled AND (violations exist AND (retries exhausted OR auto-regenerate disabled))
            should_apply_cleanup = (
                force_programmatic_fix and
                violations and
                (vocab_retry_count >= max_vocab_retries or not auto_regenerate)
            )

            if should_apply_cleanup:
                if verbose:
                    print(f"  🔨 Applying programmatic cleanup (semantic scalpel) to remove violations...")
                # Extract just the words from violations (which are tuples of (word, reason))
                violation_words = [v[0] if isinstance(v, tuple) else v for v in violations]
                generated_paragraph = translator._programmatic_cleanup(generated_paragraph, violation_words, author_name=author_name)
                # Re-check after cleanup to verify violations are gone
                found_words = vocabulary_budget.find_restricted_words(generated_paragraph)
                violations = []
                for word, position in found_words:
                    absolute_position = vocabulary_budget._total_words + position
                    allowed, reason = vocabulary_budget.check_word_allowed(word, absolute_position)
                    if not allowed:
                        violations.append((word, reason))
                if violations:
                    if verbose:
                        print(f"  ⚠ Warning: Some violations may remain after cleanup: {violations}")
                else:
                    if verbose:
                        print(f"  ✓ Programmatic cleanup successful - all violations removed")

            # Record valid word usage
            if not violations or not auto_regenerate:
                for word, position in found_words:
                    absolute_position = vocabulary_budget._total_words + position
                    vocabulary_budget.record_word(word, absolute_position, generated_paragraph)

                # Update total word count
                word_count = len(generated_paragraph.split())
                vocabulary_budget.update_total_words(word_count)

        # Accept the generated paragraph
        if verbose:
            print(f"  ✓ Generated paragraph accepted")

        generated_paragraphs.append(generated_paragraph)

        # Update previous paragraph text for next iteration (Phase 5)
        prev_paragraph_text = generated_paragraph

        # Write paragraph if callback provided
        if write_callback:
            is_new_paragraph = True
            write_callback(generated_paragraph, is_new_paragraph, is_first_paragraph)
            if is_first_paragraph:
                is_first_paragraph = False

        # Continue to next paragraph (old sentence-by-sentence logic removed)
        continue

    return generated_paragraphs


def run_pipeline(
    input_file: Optional[str] = None,
    input_text: Optional[str] = None,
    config_path: str = "config.json",
    output_file: Optional[str] = None,
    max_retries: int = 3,
    atlas_cache_path: Optional[str] = None,
    blend_ratio: Optional[float] = None,
    perspective: Optional[str] = None,
    verbose: bool = False
) -> List[str]:
    """Run the pipeline with file I/O.

    Args:
        input_file: Path to input text file (optional if input_text provided).
        input_text: Input text as string (optional if input_file provided).
        config_path: Path to configuration file.
        output_file: Optional path to save output.
        max_retries: Maximum number of retries per generation.
        atlas_cache_path: Optional path to ChromaDB persistence directory (falls back to config.json if None).
        blend_ratio: Optional blend ratio for style mixing (ignored in Pipeline 2.0, kept for compatibility).

    Returns:
        List of generated paragraphs (each paragraph is a space-joined string of sentences).
    """
    # Note: NLP pipeline initialization is handled by process_text() which is called below
    # No need to initialize here to avoid duplicate initialization messages

    # Load input text
    if input_text is None:
        if input_file is None:
            raise ValueError("Either input_file or input_text must be provided")
        with open(input_file, 'r', encoding='utf-8') as f:
            input_text = f.read()

    # Load config
    with open(config_path, 'r') as f:
        config = json.load(f)

    # Resolve atlas_cache_path from config if not provided
    if atlas_cache_path is None:
        atlas_cache_path = config.get("atlas", {}).get("persist_path")

    # Load atlas
    if atlas_cache_path:
        atlas_file = Path(atlas_cache_path) / "atlas.json"
        if atlas_file.exists():
            atlas = load_atlas(str(atlas_file), persist_directory=atlas_cache_path)
        else:
            raise FileNotFoundError(
                f"Atlas file not found: {atlas_file}. "
                "Please build the atlas first using: python scripts/load_style.py"
            )
    else:
        raise ValueError("Atlas cache path not specified. Set in config.json or via --atlas-cache")

    # Get author name from config
    blend_config = config.get("blend", {})
    authors = blend_config.get("authors", [])
    if not authors:
        raise ValueError("No authors specified in config.json. Set 'blend.authors' to a list of author names.")
    author_name = authors[0]  # Use first author as primary (for backward compatibility)

    # Get style DNA from registry for all authors
    style_dna = ""
    if atlas_cache_path:
        registry = StyleRegistry(atlas_cache_path)

        # Load style DNA for all authors in blend config
        for author in authors:
            exists, suggestion = registry.validate_author(author)
            if not exists:
                if verbose:
                    print(f"Warning: Author '{author}' not found in registry.")
                    if suggestion:
                        print(f"  {suggestion}")

            author_dna = registry.get_dna(author)
            if not author_dna:
                # Fallback: try to get from atlas
                author_dna = atlas.author_style_dna.get(author, "")
                if not author_dna:
                    if verbose:
                        print(f"Warning: No Style DNA found for '{author}'. Using empty DNA.")
                        available = list(registry.get_all_profiles().keys())
                        if available:
                            print(f"Available authors in registry: {', '.join(sorted(available))}")
                        print(f"Generate Style DNA using: python scripts/generate_style_dna.py --author '{author}'")
            else:
                if verbose:
                    print(f"  ✓ Loaded Style DNA from registry for '{author}'")

        # Get style DNA for primary author (for backward compatibility with single-author mode)
        style_dna = registry.get_dna(author_name)
        if not style_dna:
            style_dna = atlas.author_style_dna.get(author_name, "")
    else:
        style_dna = atlas.author_style_dna.get(author_name, "")

    # Open output file early if specified
    output_file_handle = None
    write_callback = None

    if output_file:
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_file_handle = open(output_path, 'w', encoding='utf-8')

        # Track state for incremental writing
        is_first_sentence = True
        is_first_in_paragraph = True

        def write_sentence(sentence: Optional[str], is_new_paragraph: bool, is_first_paragraph: bool):
            """Write sentence incrementally to file.

            Args:
                sentence: Sentence text to write, or None for paragraph break.
                is_new_paragraph: Whether this is the first sentence of a new paragraph.
                is_first_paragraph: Whether this is the very first paragraph.
            """
            nonlocal is_first_sentence, is_first_in_paragraph

            if sentence is None:
                # Paragraph break
                output_file_handle.write('\n\n')
                is_first_in_paragraph = True
            else:
                # Write sentence
                if is_new_paragraph and not is_first_paragraph:
                    # Start new paragraph (add paragraph break before)
                    output_file_handle.write('\n\n')
                    is_first_in_paragraph = True  # Reset for new paragraph

                if is_first_in_paragraph:
                    # First sentence in paragraph - no leading space
                    output_file_handle.write(sentence)
                    is_first_in_paragraph = False
                else:
                    # Subsequent sentence - add space before
                    output_file_handle.write(' ' + sentence)

                is_first_sentence = False
                output_file_handle.flush()  # Ensure immediate write to disk

        write_callback = write_sentence

    try:
        # Process text
        output = process_text(
            verbose=verbose,
            input_text=input_text,
            atlas=atlas,
            author_name=author_name,
            style_dna=style_dna,
            config_path=config_path,
            max_retries=max_retries,
            perspective=perspective,
            write_callback=write_callback
        )
    finally:
        # Close file if we opened it
        if output_file_handle:
            output_file_handle.close()

    return output

