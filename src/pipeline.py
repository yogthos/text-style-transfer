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
from typing import List, Optional, Callable
from src.ingestion.blueprint import SemanticBlueprint, BlueprintExtractor
from src.atlas.rhetoric import RhetoricalType, RhetoricalClassifier
from src.generator.translator import StyleTranslator
from src.validator.semantic_critic import SemanticCritic
from src.atlas.builder import StyleAtlas, load_atlas
from src.atlas.style_registry import StyleRegistry


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


def process_text(
    input_text: str,
    atlas: StyleAtlas,
    author_name: str,
    style_dna: str,
    config_path: str = "config.json",
    max_retries: int = 3,
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
    extractor = BlueprintExtractor()
    classifier = RhetoricalClassifier()
    translator = StyleTranslator(config_path=config_path)
    critic = SemanticCritic(config_path=config_path)

    # Split into paragraphs first
    paragraphs = _split_into_paragraphs(input_text)

    if not paragraphs:
        return []

    generated_paragraphs = []
    current_paragraph_sentences = []  # Track sentences for current paragraph
    is_first_paragraph = True

    # Context tracking for contextual anchoring
    previous_generated_text = ""
    previous_paragraph_id = -1

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

        generated_sentences = []

        for sent_idx, sentence in enumerate(sentences):
            # Determine position in paragraph
            if len(sentences) == 1:
                position = "SINGLETON"
            elif sent_idx == 0:
                position = "OPENER"
            elif sent_idx == len(sentences) - 1:
                position = "CLOSER"
            else:
                position = "BODY"

            if verbose:
                print(f"  Position: {position}")
                if previous_generated_text:
                    print(f"  Previous context: {previous_generated_text[:50]}...")
            # DEBUG: Verify sentence is unique
            sentence_hash = hash(sentence)
            if verbose:
                print(f"\n  [{sent_idx + 1}/{len(sentences)}] Sentence: {sentence[:50]}{'...' if len(sentence) > 50 else ''}")
                print(f"  DEBUG: Sentence hash: {sentence_hash}")

            # Check for duplicate input sentences (shouldn't happen, but verify)
            if sent_idx > 0:
                prev_sentences = [s for i, s in enumerate(sentences) if i < sent_idx]
                if sentence in prev_sentences:
                    print(f"  ⚠ WARNING: Duplicate input sentence detected at index {sent_idx}!")
                    if verbose:
                        print(f"    This sentence appeared at indices: {[i for i, s in enumerate(sentences) if s == sentence]}")

            # Step 1: Extract blueprint with positional metadata
            try:
                blueprint = extractor.extract(
                    sentence,
                    paragraph_id=para_idx,
                    position=position,
                    previous_context=previous_generated_text
                )
                # DEBUG: Verify blueprint is unique
                blueprint_hash = hash(str(blueprint.svo_triples) + str(blueprint.core_keywords))
                if verbose:
                    print(f"  Blueprint:")
                    print(f"    Subjects: {blueprint.get_subjects()}")
                    print(f"    Verbs: {blueprint.get_verbs()}")
                    print(f"    Objects: {blueprint.get_objects()}")
                    keywords_list = sorted(list(blueprint.core_keywords))
                    print(f"    Keywords: {keywords_list[:10]}{'...' if len(keywords_list) > 10 else ''}")
                    print(f"  DEBUG: Blueprint hash: {blueprint_hash}")
            except Exception as e:
                print(f"Warning: Failed to extract blueprint from '{sentence}': {e}")
                # Fallback: use original sentence
                # Check for duplicate before appending
                if sentence in generated_sentences:
                    print(f"  ⚠ WARNING: Duplicate generated sentence detected (fallback)!")
                generated_sentences.append(sentence)
                continue

            # Step 2: Classify rhetoric
            rhetorical_type = classifier.classify_heuristic(sentence)
            if verbose:
                print(f"  Rhetorical Type: {rhetorical_type.value}")

            # Step 3: Retrieve examples
            examples = atlas.get_examples_by_rhetoric(rhetorical_type, top_k=3, author_name=author_name)
            if not examples:
                # Fallback to any examples from same author
                examples = atlas.get_examples_by_rhetoric(RhetoricalType.OBSERVATION, top_k=3, author_name=author_name)
                if not examples:
                    # Ultimate fallback: empty examples (translator will handle it)
                    examples = []

            if verbose:
                print(f"  Examples retrieved: {len(examples)}")
                for i, ex in enumerate(examples[:2]):
                    print(f"    Example {i+1}: {ex[:80]}{'...' if len(ex) > 80 else ''}")

            # Step 4: Generate initial draft
            result = None
            generated = None

            try:
                # Get examples for rhetorical type
                examples = atlas.get_examples_by_rhetoric(rhetorical_type, top_k=3, author_name=author_name)
                if not examples:
                    # Fallback to any examples from same author
                    examples = atlas.get_examples_by_rhetoric(RhetoricalType.OBSERVATION, top_k=3, author_name=author_name)
                    if not examples:
                        examples = []

                # Generate initial draft
                generated = translator.translate(
                    blueprint=blueprint,
                    author_name=author_name,
                    style_dna=style_dna,
                    rhetorical_type=rhetorical_type,
                    examples=examples
                )

                if verbose:
                    print(f"  Generated (initial draft): {generated}")

                # Step 5: Validate initial draft
                result = critic.evaluate(generated, blueprint)

                if verbose:
                    print(f"  Critic Result: pass={result['pass']}, score={result['score']:.2f}")
                    print(f"    Recall: {result['recall_score']:.2f}, Precision: {result['precision_score']:.2f}")
                    if not result['pass']:
                        print(f"    Feedback: {result['feedback']}")

                # If initial draft fails, evolve it using hill climbing
                if not result["pass"]:
                    if verbose:
                        print(f"  ↻ Initial draft failed, evolving with hill climbing...")

                    try:
                        generated, final_score = translator._evolve_text(
                            initial_draft=generated,
                            blueprint=blueprint,
                            author_name=author_name,
                            style_dna=style_dna,
                            rhetorical_type=rhetorical_type,
                            initial_score=result["score"],
                            initial_feedback=result["feedback"],
                            critic=critic,
                            verbose=verbose
                        )

                        # Re-evaluate evolved draft
                        result = critic.evaluate(generated, blueprint)

                        if verbose:
                            print(f"  Evolution Result: pass={result['pass']}, score={result['score']:.2f}")
                            if result['pass']:
                                print(f"    ✓ Evolution successful: Score improved to {result['score']:.2f}")
                            else:
                                print(f"    ✗ Evolution failed: Final score {result['score']:.2f}")
                    except Exception as e:
                        if verbose:
                            print(f"  ⚠ Evolution failed with exception: {e}")
                        # Continue with initial draft

            except Exception as e:
                print(f"Warning: Initial generation failed: {e}")
                result = None

            # Step 6: Accept or fallback
            if result and result["pass"]:
                # DEBUG: Before appending, verify we haven't seen this before
                if generated in generated_sentences:
                    print(f"  ⚠ WARNING: Duplicate generated sentence detected!")
                    if verbose:
                        print(f"    Generated text: {generated[:80]}...")
                        print(f"    This sentence was already generated at an earlier index")
                        # Find where it was first generated
                        first_idx = next(i for i, s in enumerate(generated_sentences) if s == generated)
                        print(f"    First occurrence was at sentence index {first_idx}")
                    # Skip duplicate - don't append again
                    if verbose:
                        print(f"  ✓ Accepted (but skipped duplicate)")
                else:
                    generated_sentences.append(generated)
                    current_paragraph_sentences.append(generated)

                    # Update context for next iteration
                    previous_generated_text = generated

                    # Write sentence immediately if callback provided
                    if write_callback:
                        is_new_paragraph = (sent_idx == 0)
                        write_callback(generated, is_new_paragraph, is_first_paragraph)
                        if is_new_paragraph and is_first_paragraph:
                            is_first_paragraph = False

                    if verbose:
                        print(f"  ✓ Accepted")
            else:
                # Evolution failed or initial generation failed, use literal translation fallback
                if verbose:
                    print(f"  ↻ Evolution/Generation failed, using literal translation")
                try:
                    generated = translator.translate_literal(blueprint, author_name, style_dna)
                    if verbose:
                        print(f"  Literal translation: {generated}")

                    # DEBUG: Check for duplicate before appending
                    if generated in generated_sentences:
                        print(f"  ⚠ WARNING: Duplicate generated sentence detected (literal fallback)!")
                        if verbose:
                            print(f"    Generated text: {generated[:80]}...")
                    else:
                        generated_sentences.append(generated)
                        current_paragraph_sentences.append(generated)

                        # Update context for next iteration
                        previous_generated_text = generated

                        # Write sentence immediately if callback provided
                        if write_callback:
                            is_new_paragraph = (sent_idx == 0)
                            write_callback(generated, is_new_paragraph, is_first_paragraph)
                            if is_new_paragraph and is_first_paragraph:
                                is_first_paragraph = False
                except Exception as e:
                    print(f"Warning: Literal translation failed: {e}")
                    # Ultimate fallback: use original sentence
                    if verbose:
                        print(f"  ↻ Using original sentence as fallback")

                    # DEBUG: Check for duplicate before appending
                    if sentence in generated_sentences:
                        print(f"  ⚠ WARNING: Duplicate generated sentence detected (ultimate fallback)!")
                    else:
                        generated_sentences.append(sentence)
                        current_paragraph_sentences.append(sentence)

                        # Update context for next iteration (use original sentence as context)
                        previous_generated_text = sentence

                        # Write sentence immediately if callback provided
                        if write_callback:
                            is_new_paragraph = (sent_idx == 0)
                            write_callback(sentence, is_new_paragraph, is_first_paragraph)
                            if is_new_paragraph and is_first_paragraph:
                                is_first_paragraph = False

        # Join sentences within paragraph with spaces
        if generated_sentences:
            para_text = ' '.join(generated_sentences)
            if verbose:
                print(f"\n  Paragraph output: {para_text[:100]}{'...' if len(para_text) > 100 else ''}")
            generated_paragraphs.append(para_text)

            # Reset for next paragraph
            current_paragraph_sentences = []

    return generated_paragraphs


def run_pipeline(
    input_file: Optional[str] = None,
    input_text: Optional[str] = None,
    config_path: str = "config.json",
    output_file: Optional[str] = None,
    max_retries: int = 3,
    atlas_cache_path: Optional[str] = None,
    blend_ratio: Optional[float] = None,
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
    author_name = authors[0]  # Use first author (Pipeline 2.0 doesn't support blending yet)

    # Get style DNA from registry
    if atlas_cache_path:
        registry = StyleRegistry(atlas_cache_path)
        style_dna = registry.get_dna(author_name)
        if not style_dna:
            # Fallback: try to get from atlas
            style_dna = atlas.author_style_dna.get(author_name, "")
            if not style_dna:
                print(f"Warning: No Style DNA found for '{author_name}'. Using empty DNA.")
                print(f"Generate Style DNA using: python scripts/generate_style_dna.py --author '{author_name}'")
                style_dna = ""
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
            write_callback=write_callback
        )
    finally:
        # Close file if we opened it
        if output_file_handle:
            output_file_handle.close()

    return output

