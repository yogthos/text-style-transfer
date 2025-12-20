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
from src.analyzer.style_extractor import StyleExtractor
from src.analysis.semantic_analyzer import PropositionExtractor
from src.utils.structure_tracker import StructureTracker
from src.analyzer.global_context import GlobalContextAnalyzer


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
    proposition_extractor = PropositionExtractor(config_path=config_path)

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
    except Exception as e:
        if verbose:
            print(f"  ⚠ Global context extraction failed: {e}, continuing without context")
        global_context = None

    generated_paragraphs = []
    current_paragraph_sentences = []  # Track sentences for current paragraph
    is_first_paragraph = True

    # Track used examples to prevent repetition
    used_examples = set()

    # Structure tracking for paragraph diversity
    structure_tracker = StructureTracker()
    total_paragraphs = len(paragraphs)

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

        # Check if we should transcribe single-sentence paragraphs as-is (likely headings)
        # Load config once for both heading detection and paragraph fusion settings
        paragraph_fusion_config = {}
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
            paragraph_fusion_config = config.get("paragraph_fusion", {})
        except Exception:
            pass

        transcribe_headings_as_is = paragraph_fusion_config.get("transcribe_headings_as_is", False)

        # If enabled and single sentence, check if it looks like a heading
        if transcribe_headings_as_is and len(sentences) == 1:
            single_sentence = sentences[0].strip()
            # Heuristics for heading detection:
            # 1. Short length (typically headings are < 100 chars)
            # 2. All caps or no lowercase letters (common in headings)
            # 3. Ends with no sentence-ending punctuation or just colon
            is_short = len(single_sentence) < 100
            is_all_caps = single_sentence.isupper() and len(single_sentence) > 3
            has_no_lowercase = not any(c.islower() for c in single_sentence)
            stripped = single_sentence.rstrip()
            ends_with_colon = stripped.endswith(':')
            ends_with_sentence_punct = stripped.endswith(('.', '!', '?'))
            # Heading if: ends with colon OR doesn't end with sentence punctuation
            ends_like_heading = ends_with_colon or not ends_with_sentence_punct

            # Consider it a heading if it's short and (all caps OR no lowercase) and ends like a heading
            looks_like_heading = is_short and (is_all_caps or has_no_lowercase) and ends_like_heading

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

        # Check if this is a multi-sentence paragraph (use paragraph fusion)
        is_multi_sentence = len(sentences) > 1
        min_sentences_for_fusion = 2
        if paragraph_fusion_config.get("enabled", True):
            min_sentences_for_fusion = paragraph_fusion_config.get("min_sentences_for_fusion", 2)

        # Use paragraph fusion if enabled and paragraph has multiple sentences
        if is_multi_sentence and len(sentences) >= min_sentences_for_fusion:
            if verbose:
                print(f"  Using paragraph fusion mode ({len(sentences)} sentences)")

            # Extract style DNA from examples
            style_dna_dict = None
            try:
                # Get examples for style extraction
                examples = atlas.get_examples_by_rhetoric(
                    RhetoricalType.OBSERVATION,
                    top_k=5,
                    author_name=author_name,
                    query_text=paragraph
                )
                if examples:
                    style_dna_dict = style_extractor.extract_style_dna(examples)
            except Exception:
                pass

            # Determine paragraph position (OPENER, BODY, CLOSER)
            if para_idx == 0:
                position = "OPENER"
            elif para_idx == total_paragraphs - 1:
                position = "CLOSER"
            else:
                position = "BODY"

            # Translate paragraph holistically
            try:
                generated_paragraph, teacher_rhythm_map, teacher_example, internal_recall = translator.translate_paragraph(
                    paragraph,
                    atlas,
                    author_name,
                    style_dna=style_dna_dict,
                    position=position,
                    structure_tracker=structure_tracker,
                    used_examples=used_examples,
                    secondary_author=secondary_author,
                    blend_ratio=blend_ratio,
                    verbose=verbose,
                    global_context=global_context,
                    is_opener=(para_idx == 0)
                )

                # Use internal_recall from translator (includes repair loop context and relaxed thresholds)
                # Re-evaluate to get coherence and topic similarity scores for sanity gate
                # Load thresholds from config for tiered evaluation
                try:
                    with open(config_path, 'r') as f:
                        config = json.load(f)
                    paragraph_fusion_config = config.get("paragraph_fusion", {})
                    ideal_threshold = paragraph_fusion_config.get("proposition_recall_threshold", 0.85)
                    min_viable = paragraph_fusion_config.get("min_viable_recall_threshold", 0.70)
                    coherence_threshold = paragraph_fusion_config.get("coherence_threshold", 0.8)
                    topic_similarity_threshold = paragraph_fusion_config.get("topic_similarity_threshold", 0.6)
                except Exception:
                    # Fallback to defaults if config read fails
                    ideal_threshold = 0.85
                    min_viable = 0.70
                    coherence_threshold = 0.8
                    topic_similarity_threshold = 0.6

                # Re-evaluate final paragraph to get coherence and topic similarity
                # (These are expensive checks that only run on promising candidates)
                # Note: critic and proposition_extractor are already initialized at function start
                propositions = proposition_extractor.extract_atomic_propositions(paragraph)

                # Create blueprint for evaluation
                blueprint = extractor.extract(paragraph)

                # Re-evaluate to get coherence and topic similarity
                critic_result = critic.evaluate(
                    generated_text=generated_paragraph,
                    input_blueprint=blueprint,
                    propositions=propositions,
                    is_paragraph=True,
                    author_style_vector=None,  # Not needed for coherence check
                    style_lexicon=None
                )

                # Get coherence and topic similarity from critic result
                coherence_score = critic_result.get('coherence_score', 1.0)
                topic_similarity = critic_result.get('topic_similarity', 1.0)

                # internal_recall is already available from translate_paragraph return value
                # This is the best recall achieved during the repair loop, with proper context

                # Tiered evaluation logic with sanity gate
                if internal_recall >= ideal_threshold and coherence_score >= coherence_threshold and topic_similarity >= topic_similarity_threshold:
                    # Scenario A: Perfect Pass (high recall, coherent, and preserves topic)
                    if verbose:
                        print(f"  ✓ Fusion Success: Recall {internal_recall:.2f} >= {ideal_threshold}, Coherence {coherence_score:.2f} >= {coherence_threshold}, Topic similarity {topic_similarity:.2f} >= {topic_similarity_threshold}")
                    # Override critic_result to ensure pass=True
                    critic_result["pass"] = True
                    critic_result["score"] = 1.0  # Perfect score
                    pass_status = "PERFECT PASS"
                elif internal_recall >= ideal_threshold:
                    # High recall but low coherence/similarity = gibberish
                    if verbose:
                        print(f"  ⚠ High recall ({internal_recall:.2f}) but coherence ({coherence_score:.2f}) or topic similarity ({topic_similarity:.2f}) too low")
                    pass_status = "COHERENCE FAIL"
                    critic_result["pass"] = False
                    # Don't override score - let it reflect the failure

                elif internal_recall >= min_viable:
                    # Scenario B: Soft Pass (The Fix)
                    if verbose:
                        print(f"  ⚠ Soft Pass: Recall {internal_recall:.2f} is below ideal ({ideal_threshold}) but viable (>= {min_viable}). Accepting.")
                    # Create critic_result for soft pass
                    critic_result = {"pass": True, "score": 0.8, "proposition_recall": internal_recall}
                    pass_status = "SOFT PASS"

                else:
                    # Scenario C: Hard Fail -> Trigger Sentence-by-Sentence Fallback
                    if verbose:
                        print(f"  ✗ Fusion Failed: Recall {internal_recall:.2f} below viability floor ({min_viable}).")
                    # Create a minimal critic_result for logging consistency
                    critic_result = {"pass": False, "score": 0.0, "proposition_recall": internal_recall}
                    pass_status = "HARD FAIL"

                # Logging with status indicator
                if verbose:
                    pass_value = critic_result.get('pass', False)
                    # Color codes: green for Perfect/Soft Pass, red for Hard Fail
                    if pass_value:
                        pass_color = '\033[92m'  # Green
                    else:
                        pass_color = '\033[91m'  # Red
                    reset_color = '\033[0m'
                    pass_str = f"{pass_color}{pass_value}{reset_color}"
                    composite_score = critic_result.get('score', 0.0)
                    print(f"  Paragraph fusion result: {pass_status}, pass={pass_str}, "
                          f"recall={internal_recall:.2f}, "
                          f"composite_score={composite_score:.2f}")

                # Use generated paragraph if it passes (Perfect or Soft Pass), otherwise fall back
                if critic_result.get('pass', False):
                    # Record structure for diversity tracking
                    if teacher_rhythm_map:
                        from src.analyzer.structuralizer import generate_structure_signature
                        signature = generate_structure_signature(teacher_rhythm_map)
                        structure_tracker.add_structure(signature, teacher_rhythm_map)

                    # Track teacher example to prevent reuse
                    if teacher_example:
                        used_examples.add(teacher_example)

                    generated_paragraphs.append(generated_paragraph)
                    if write_callback:
                        # Each paragraph is a new paragraph (except we track is_first_paragraph separately)
                        is_new_paragraph = True
                        write_callback(generated_paragraph, is_new_paragraph, is_first_paragraph)
                        if is_first_paragraph:
                            is_first_paragraph = False
                    previous_generated_text = generated_paragraph  # Update context
                    continue  # Skip sentence-by-sentence processing
                else:
                    if verbose:
                        print(f"  Paragraph fusion failed, falling back to sentence-by-sentence")
                    # Fall through to sentence-by-sentence processing
            except Exception as e:
                if verbose:
                    print(f"  Paragraph fusion error: {e}, falling back to sentence-by-sentence")
                # Fall through to sentence-by-sentence processing

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

            # Step 3: Retrieve examples (exclude previously used ones)
            # Pass sentence as query_text for length window filtering
            # Fetch 15 examples to provide wide net for filtering
            examples = atlas.get_examples_by_rhetoric(
                rhetorical_type,
                top_k=15,
                author_name=author_name,
                exclude=list(used_examples),
                query_text=sentence
            )
            if not examples:
                # Fallback to any examples from same author
                examples = atlas.get_examples_by_rhetoric(
                    RhetoricalType.OBSERVATION,
                    top_k=15,
                    author_name=author_name,
                    exclude=list(used_examples),
                    query_text=sentence
                )
                if not examples:
                    # Ultimate fallback: empty examples (translator will handle it)
                    examples = []

            # Track used examples
            used_examples.update(examples)

            if verbose:
                print(f"  Examples retrieved: {len(examples)}")
                for i, ex in enumerate(examples[:2]):
                    print(f"    Example {i+1}: {ex[:80]}{'...' if len(ex) > 80 else ''}")

            # Step 3.5: Extract style DNA from examples (RAG-driven)
            style_dna_dict = None
            style_lexicon = None
            if examples:
                try:
                    style_dna_dict = style_extractor.extract_style_dna(examples)
                    style_lexicon = style_dna_dict.get("lexicon", [])
                    if verbose:
                        print(f"  Style DNA extracted:")
                        print(f"    Lexicon: {', '.join(style_lexicon[:5])}{'...' if len(style_lexicon) > 5 else ''}")
                        print(f"    Tone: {style_dna_dict.get('tone', 'N/A')}")
                        print(f"    Structure: {style_dna_dict.get('structure', 'N/A')[:60]}...")
                except Exception as e:
                    if verbose:
                        print(f"  ⚠ Style DNA extraction failed: {e}")
                    style_dna_dict = None
                    style_lexicon = None

            # Step 4: Generate initial draft
            result = None
            generated = None

            try:
                # Generate initial draft (examples already retrieved above)

                # Generate initial draft
                generated = translator.translate(
                    blueprint=blueprint,
                    author_name=author_name,
                    style_dna=style_dna,
                    rhetorical_type=rhetorical_type,
                    examples=examples,
                    verbose=verbose
                )

                if verbose:
                    print(f"  Generated (initial draft): {generated}")

                # Step 5: Validate initial draft (with style whitelist)
                result = critic.evaluate(generated, blueprint, allowed_style_words=style_lexicon)

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
                            verbose=verbose,
                            style_dna_dict=style_dna_dict,
                            examples=examples
                        )

                        # Re-evaluate evolved draft
                        result = critic.evaluate(generated, blueprint)

                        if verbose:
                            print(f"  Evolution Result: pass={result['pass']}, score={result['score']:.2f}")
                            if result['pass']:
                                print(f"    ✓ Evolution successful: Score improved to {result['score']:.2f}")
                                print(f"    Evolved text: {generated}")
                            else:
                                print(f"    ✗ Evolution failed: Final score {result['score']:.2f}")
                                print(f"    Evolved text: {generated}")
                    except Exception as e:
                        if verbose:
                            print(f"  ⚠ Evolution failed with exception: {e}")
                        # Continue with initial draft

            except Exception as e:
                print(f"Warning: Initial generation failed: {e}")
                result = None

            # Step 6: Accept or fallback
            # Soft Landing: If we have a decent draft (Score >= 0.75), KEEP IT.
            # This prevents throwing away good work (e.g., 0.81 scores) just because pass=False
            if result and result.get("score", 0.0) >= 0.75:
                # Accept even if pass=False - score >= 0.75 is good enough
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
                        pass_status = "PASS" if result.get("pass", False) else "SOFT PASS"
                        print(f"  ✓ Accepted ({pass_status}, score: {result.get('score', 0.0):.2f}): {generated}")
            elif result and result.get("score", 0.0) < 0.60:
                # Evolution failed or initial generation failed, use literal translation fallback
                # Lowered threshold to 0.60 to avoid discarding "diamonds in the rough" (good style, minor issues)
                if verbose:
                    print(f"  ↻ Evolution/Generation failed (score < 0.60), using literal translation")
                try:
                    # Pass rhetorical_type and examples for style-preserving fallback
                    generated = translator.translate_literal(
                        blueprint=blueprint,
                        author_name=author_name,
                        style_dna=style_dna,
                        rhetorical_type=rhetorical_type,
                        examples=examples
                    )
                    if verbose:
                        print(f"  Style-preserving fallback: {generated}")

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

                        if verbose:
                            print(f"  ✓ Accepted (literal fallback): {generated}")
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

                        if verbose:
                            print(f"  ✓ Accepted (original fallback): {sentence}")

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
            write_callback=write_callback
        )
    finally:
        # Close file if we opened it
        if output_file_handle:
            output_file_handle.close()

    return output

