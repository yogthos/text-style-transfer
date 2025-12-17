"""Main pipeline for Style Atlas-based text style transfer.

This module implements the Style Atlas pipeline:
1. Build/load Style Atlas from sample text
2. Extract meaning from input
3. Navigate style clusters
4. Retrieve style references
5. Generate with adversarial critic loop
"""

import json
from pathlib import Path
from typing import List, Optional

from src.models import ContentUnit
from src.ingestion.semantic import extract_meaning
from src.atlas import (
    StyleAtlas,
    build_style_atlas,
    save_atlas,
    load_atlas,
    build_cluster_markov,
    predict_next_cluster,
    find_situation_match,
    find_structure_match,
    StructureNavigator,
    StyleBlender
)
from src.generator.llm_interface import generate_sentence
from src.validator.critic import generate_with_critic, ConvergenceError
from src.analyzer.style_metrics import get_style_vector
from src.ingestion.semantic import _detect_sentiment


def _determine_current_cluster(
    generated_text_so_far: List[str],
    atlas: StyleAtlas
) -> Optional[int]:
    """Determine the current style cluster from generated text.

    Args:
        generated_text_so_far: List of previously generated sentences/paragraphs.
        atlas: StyleAtlas with cluster information.

    Returns:
        Current cluster ID, or None if cannot determine.
    """
    if not generated_text_so_far:
        return None

    # Get style vector of most recent generated text
    recent_text = " ".join(generated_text_so_far[-3:])  # Last 3 sentences
    current_style_vec = get_style_vector(recent_text)

    # Find closest cluster center
    if len(atlas.cluster_centers) == 0:
        return None

    distances = []
    for center in atlas.cluster_centers:
        dist = ((current_style_vec - center) ** 2).sum()
        distances.append(dist)

    closest_cluster_idx = min(range(len(distances)), key=lambda i: distances[i])
    return closest_cluster_idx


def process_text(
    input_text: str,
    config_path: str = "config.json",
    max_retries: int = 3,
    output_file: Optional[str] = None,
    atlas_cache_path: Optional[str] = None,
    blend_ratio: Optional[float] = None
) -> List[str]:
    """Process input text through Style Atlas pipeline.

    Args:
        input_text: The input text to transform.
        config_path: Path to configuration file.
        max_retries: Maximum number of retries per generation (default: 3).
        output_file: Optional path to write output incrementally.
        atlas_cache_path: Optional path to ChromaDB persistence directory.

    Returns:
        List of generated paragraphs.
    """
    # Load config
    with open(config_path, 'r') as f:
        config = json.load(f)

    atlas_config = config.get("atlas", {})
    num_clusters = atlas_config.get("num_clusters", 5)
    similarity_threshold = atlas_config.get("similarity_threshold", 0.3)
    collection_name = atlas_config.get("collection_name", "style_atlas")

    # Read critic configuration
    critic_config = config.get("critic", {})
    critic_min_score = critic_config.get("min_score", 0.75)
    critic_min_pipeline_score = critic_config.get("min_pipeline_score", 0.6)
    critic_max_retries = critic_config.get("max_retries", 5)
    critic_max_pipeline_retries = critic_config.get("max_pipeline_retries", 2)

    # Detect blend mode from config
    blend_config = config.get("blend", {})
    blend_authors = blend_config.get("authors", [])
    blend_ratio_config = blend_config.get("ratio", 0.5)
    # CLI blend_ratio overrides config
    final_blend_ratio = blend_ratio if blend_ratio is not None else blend_ratio_config

    # Determine if we're in blend mode (multiple authors) or single-author mode
    is_blend_mode = len(blend_authors) > 1
    style_blender = None

    if is_blend_mode:
        print(f"Phase 1: Blend mode detected with authors: {blend_authors}")
        print(f"  Blend ratio: {final_blend_ratio} (0.0 = All {blend_authors[0]}, 1.0 = All {blend_authors[1]})")
    else:
        print(f"Phase 1: Single-author mode")

    # Phase 1: Load existing Style Atlas from ChromaDB
    print("Phase 1: Loading Style Atlas from ChromaDB...")
    print(f"  Collection: {collection_name}")
    print(f"  Persist directory: {atlas_cache_path or '(in-memory)'}")
    atlas = None

    # Try to load existing atlas from ChromaDB
    if atlas_cache_path:
        cache_file = Path(atlas_cache_path) / "atlas.json"
        if cache_file.exists():
            try:
                print(f"  Attempting to load from cache: {cache_file}")
                atlas = load_atlas(str(cache_file), persist_directory=atlas_cache_path)
                # Verify collection exists and has data
                if hasattr(atlas, '_collection'):
                    try:
                        collection = atlas._collection
                        test_results = collection.get(limit=1)
                        if test_results['ids'] and len(test_results['ids']) > 0:
                            total_count = collection.count() if hasattr(collection, 'count') else len(atlas.cluster_ids)
                            print(f"  ✓ Loaded atlas from cache with {total_count} paragraphs")
                        else:
                            atlas = None
                            raise ValueError("ChromaDB collection is empty")
                    except ValueError:
                        raise
                    except Exception as e:
                        atlas = None
                        raise ValueError(f"ChromaDB collection not accessible: {e}")
                else:
                    atlas = None
                    raise ValueError("Cached atlas has no collection connection")
            except ValueError:
                raise
            except Exception as e:
                print(f"  ⚠ Failed to load from cache: {e}")
                atlas = None

    # If cache load failed, try to connect directly to ChromaDB
    if atlas is None:
        try:
            import chromadb
            from chromadb.config import Settings

            print(f"  Connecting directly to ChromaDB...")
            if atlas_cache_path:
                client = chromadb.PersistentClient(path=atlas_cache_path)
            else:
                client = chromadb.Client(Settings(anonymized_telemetry=False))

            # Check if collection exists
            try:
                collection = client.get_collection(name=collection_name)
            except Exception as e:
                error_msg = (
                    f"ChromaDB collection '{collection_name}' does not exist.\n"
                    f"Persist directory: {atlas_cache_path or '(in-memory)'}\n"
                    f"Please load author styles first using:\n"
                    f"  python scripts/load_style.py --style-file <file> --author <name>"
                )
                raise ValueError(error_msg) from e

            # Check if collection has data
            try:
                test_results = collection.get(limit=1)
            except Exception as e:
                error_msg = (
                    f"Failed to access ChromaDB collection '{collection_name}': {e}\n"
                    f"Persist directory: {atlas_cache_path or '(in-memory)'}"
                )
                raise ValueError(error_msg) from e

            if not test_results.get('ids') or len(test_results['ids']) == 0:
                error_msg = (
                    f"ChromaDB collection '{collection_name}' is empty.\n"
                    f"Persist directory: {atlas_cache_path or '(in-memory)'}\n"
                    f"Please load author styles first using:\n"
                    f"  python scripts/load_style.py --style-file <file> --author <name>"
                )
                raise ValueError(error_msg)

            # Create a minimal atlas object from existing collection
            # We'll need cluster_ids, but we can get them from metadata
            all_results = collection.get(include=["metadatas"])
            cluster_ids = {}
            for idx, meta in enumerate(all_results.get('metadatas', [])):
                para_id = all_results['ids'][idx]
                cluster_id = meta.get('cluster_id', 0)
                cluster_ids[para_id] = cluster_id

            # Create StyleAtlas object
            from src.atlas.builder import StyleAtlas
            atlas = StyleAtlas(
                collection_name=collection_name,
                cluster_ids=cluster_ids,
                cluster_centers=np.array([]),  # Not needed for retrieval
                style_vectors=[],  # Not needed for retrieval
                num_clusters=len(set(cluster_ids.values()))
            )
            atlas._client = client
            atlas._collection = collection

            total_count = collection.count() if hasattr(collection, 'count') else len(cluster_ids)
            print(f"  ✓ Connected to ChromaDB collection with {total_count} paragraphs")

        except ValueError:
            # Re-raise ValueError with our helpful message
            raise
        except Exception as e:
            error_msg = (
                f"ChromaDB collection '{collection_name}' is not accessible.\n"
                f"Persist directory: {atlas_cache_path or '(in-memory)'}\n"
                f"Error: {type(e).__name__}: {e}\n"
                f"Please load author styles first using:\n"
                f"  python scripts/load_style.py --style-file <file> --author <name>"
            )
            raise ValueError(error_msg) from e

    # Initialize StyleBlender if in blend mode
    if is_blend_mode:
        try:
            style_blender = StyleBlender(atlas)
            print(f"  ✓ Initialized StyleBlender for blend mode")
        except Exception as e:
            print(f"  ⚠ Failed to initialize StyleBlender: {e}")
            print(f"  Falling back to single-author mode")
            is_blend_mode = False

    # Build cluster Markov chain
    print("Phase 1: Building cluster Markov chain...")
    cluster_markov = build_cluster_markov(atlas)
    print(f"  ✓ Built Markov chain with {len(cluster_markov[1])} clusters")

    # Initialize StructureNavigator for stochastic selection
    print("Phase 1: Initializing structure navigator...")
    structure_navigator = StructureNavigator(history_limit=3)
    print(f"  ✓ Initialized structure navigator")

    # Global vocabulary extraction
    # In blend mode, vocabulary is handled by StyleBlender
    # In single-author mode, we use situation matches for vocabulary, so global vocab is optional
    global_vocab_dict = {'positive': [], 'negative': [], 'neutral': []}
    if not is_blend_mode:
        # For single-author mode, vocabulary comes from situation matches
        # Global vocabulary injection is optional and can be skipped
        print("Phase 1: Using situation matches for vocabulary (global vocab injection disabled)")

    # Phase 2: Extract meaning from input
    print("Phase 2: Extracting meaning...")
    content_units = extract_meaning(input_text)
    print(f"  Extracted {len(content_units)} content units")

    # Group content units by paragraph
    paragraphs_content = []
    current_para = []
    current_para_idx = -1

    for unit in content_units:
        if unit.paragraph_idx != current_para_idx:
            if current_para:
                paragraphs_content.append(current_para)
            current_para = [unit]
            current_para_idx = unit.paragraph_idx
        else:
            current_para.append(unit)

    if current_para:
        paragraphs_content.append(current_para)

    print(f"  Grouped into {len(paragraphs_content)} paragraphs")

    # Phase 3-5: Process each paragraph
    final_output = []
    generated_text_so_far = []
    current_cluster = None

    # Open output file for incremental writing if provided
    output_handle = None
    if output_file:
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_handle = open(output_file, 'w', encoding='utf-8')
        print(f"Writing output incrementally to: {output_file}")

    try:
        for para_idx, para_units in enumerate(paragraphs_content):
            print(f"\nProcessing paragraph {para_idx + 1}/{len(paragraphs_content)}")

            # Determine current cluster
            if current_cluster is None:
                # First paragraph - use first cluster or predict from start
                current_cluster = 0 if atlas.num_clusters > 0 else None
            else:
                # Predict next cluster
                current_cluster = predict_next_cluster(current_cluster, cluster_markov)

            print(f"  Current style cluster: {current_cluster}")

            # Process each sentence in paragraph
            para_output = []

            for unit_idx, content_unit in enumerate(para_units):
                print(f"  Processing sentence {unit_idx + 1}/{len(para_units)}")
                print(f"    Original: {content_unit.original_text[:80]}...")

                # Retrieve dual RAG references (blend mode vs single-author mode)
                if is_blend_mode and style_blender and len(blend_authors) >= 2:
                    # Blend mode: use StyleBlender to find bridge texts
                    author_a = blend_authors[0]
                    author_b = blend_authors[1]

                    # Get blended structure match (bridge text)
                    structure_match = style_blender.retrieve_blended_template(
                        author_a=author_a,
                        author_b=author_b,
                        blend_ratio=final_blend_ratio
                    )

                    # For situation match, still use semantic similarity (could be enhanced later)
                    situation_match = find_situation_match(
                        atlas,
                        content_unit.original_text,
                        similarity_threshold=similarity_threshold
                    )
                else:
                    # Single-author mode: use existing retrieval
                    situation_match = find_situation_match(
                        atlas,
                        content_unit.original_text,
                        similarity_threshold=similarity_threshold
                    )

                    structure_match = find_structure_match(
                        atlas,
                        current_cluster,
                        input_text=content_unit.original_text,
                        length_tolerance=0.3,
                        navigator=structure_navigator
                    )

                if situation_match:
                    print(f"    Retrieved situation match (vocabulary): {situation_match[:80]}...")
                else:
                    print(f"    ⚠ No situation match found (similarity < {similarity_threshold}). Using structure match only.")

                if structure_match:
                    print(f"    Retrieved structure match (rhythm): {structure_match[:80]}...")
                else:
                    print(f"    ⚠ No structure match found for cluster {current_cluster}. Cannot proceed.")
                    para_output.append(content_unit.original_text)
                    generated_text_so_far.append(content_unit.original_text)
                    continue

                # Determine vocabulary list
                if is_blend_mode and style_blender and len(blend_authors) >= 2:
                    # Blend mode: get hybrid vocabulary
                    author_a = blend_authors[0]
                    author_b = blend_authors[1]
                    global_vocab_list = style_blender.get_hybrid_vocab(
                        author_a=author_a,
                        author_b=author_b,
                        ratio=final_blend_ratio
                    )
                else:
                    # Single-author mode: vocabulary comes from situation matches
                    # Global vocabulary injection is optional (can be empty)
                    global_vocab_list = []

                # Calculate adaptive threshold based on structure match quality
                # If structure match is very different from input, lower the threshold
                input_word_count = len(content_unit.original_text.split())
                structure_word_count = len(structure_match.split()) if structure_match else input_word_count
                length_ratio = structure_word_count / input_word_count if input_word_count > 0 else 1.0

                # If length ratio is very different (>2x or <0.5x), lower the threshold
                adaptive_min_score = critic_min_score
                if length_ratio > 2.0 or length_ratio < 0.5:
                    # Structure match is very different, be more lenient
                    adaptive_min_score = max(0.6, critic_min_score - 0.15)
                    print(f"    ⚠ Structure match length ratio {length_ratio:.2f} is very different, lowering threshold to {adaptive_min_score:.2f}")
                elif length_ratio > 1.5 or length_ratio < 0.67:
                    # Structure match is moderately different, slightly lower threshold
                    adaptive_min_score = max(0.65, critic_min_score - 0.1)
                    print(f"    ⚠ Structure match length ratio {length_ratio:.2f} is different, adjusting threshold to {adaptive_min_score:.2f}")

                # Generate with critic loop and pipeline-level retries
                generated = None
                critic_result = None
                final_score = 0.0

                try:
                    # Pipeline-level retry loop
                    for pipeline_attempt in range(critic_max_pipeline_retries + 1):
                        try:
                            # Determine author names for prompt (for blend mode)
                            author_names = None
                            if is_blend_mode and len(blend_authors) >= 2:
                                author_names = blend_authors

                            generated, critic_result = generate_with_critic(
                                generate_fn=lambda cu, struct_match, sit_match, cfg, hint=None: generate_sentence(
                                    cu, struct_match, sit_match, cfg, hint=hint,
                                    global_vocab_list=global_vocab_list,
                                    author_names=author_names,
                                    blend_ratio=final_blend_ratio if is_blend_mode else None
                                ),
                                content_unit=content_unit,
                                structure_match=structure_match,
                                situation_match=situation_match,
                                config_path=config_path,
                                max_retries=critic_max_retries,
                                min_score=adaptive_min_score  # Use adaptive threshold
                            )

                            score = critic_result.get("score", 0.0)
                            final_score = score

                            # Log attempt
                            if pipeline_attempt > 0:
                                print(f"    Pipeline retry {pipeline_attempt}: Score {score:.3f}")

                            # Check if score is acceptable for pipeline
                            if score >= critic_min_pipeline_score:
                                break  # Good enough, proceed

                            # If score too low and we have more retries, continue
                            if pipeline_attempt < critic_max_pipeline_retries:
                                print(f"    ⚠ Score {score:.3f} below pipeline threshold {critic_min_pipeline_score}, retrying...")
                                # Optionally refresh structure/situation matches for next attempt
                                # (keeping same matches for now, but could refresh)
                            else:
                                # Last attempt, accept what we have but warn
                                print(f"    ⚠ Final score {score:.3f} below pipeline threshold {critic_min_pipeline_score} after all retries")

                        except ConvergenceError as e:
                            # Critic loop failed to converge
                            if pipeline_attempt < critic_max_pipeline_retries:
                                print(f"    ⚠ Critic convergence failed: {e}")
                                print(f"    Retrying at pipeline level (attempt {pipeline_attempt + 1}/{critic_max_pipeline_retries + 1})...")
                                continue
                            else:
                                # No more retries, raise the error
                                raise

                    # Log final result
                    passed = critic_result.get("pass", False) if critic_result else False
                    print(f"    Generated: {generated}")
                    print(f"    Critic score: {final_score:.3f} (pass: {passed})")

                    if not passed:
                        feedback = critic_result.get("feedback", "") if critic_result else ""
                        if feedback:
                            print(f"    Critic feedback: {feedback[:100]}...")

                    # Warn if score is still low
                    if final_score < critic_min_pipeline_score:
                        print(f"    ⚠ WARNING: Final score {final_score:.3f} is below pipeline threshold {critic_min_pipeline_score}")

                    para_output.append(generated)
                    generated_text_so_far.append(generated)

                    # Print final score summary before moving to next sentence
                    print(f"    ✓ Final score: {final_score:.3f} (pass: {passed})")

                    # Add blank line before next sentence for readability
                    if unit_idx < len(para_units) - 1:
                        print()

                except ConvergenceError as e:
                    print(f"    ✗ CRITICAL: Failed to converge after all retries: {e}")
                    # Use original text as fallback
                    para_output.append(content_unit.original_text)
                    generated_text_so_far.append(content_unit.original_text)
                    print(f"    ⚠ Using original text due to convergence failure")

                except Exception as e:
                    print(f"    ⚠ Generation failed: {e}")
                    import traceback
                    traceback.print_exc()
                    # Fallback: use original text
                    para_output.append(content_unit.original_text)
                    generated_text_so_far.append(content_unit.original_text)

            # Combine paragraph sentences
            para_text = " ".join(para_output)
            final_output.append(para_text)

            # Write to file if provided
            if output_handle:
                output_handle.write(para_text)
                if para_idx < len(paragraphs_content) - 1:
                    output_handle.write("\n\n")
                output_handle.flush()

            # Update current cluster based on generated text
            current_cluster = _determine_current_cluster(generated_text_so_far, atlas)
            if current_cluster is not None:
                print(f"  ✓ Paragraph complete. Next cluster: {current_cluster}")

    finally:
        if output_handle:
            output_handle.close()
            print(f"\n✓ Output file closed: {output_file}")

    return final_output


def run_pipeline(
    input_file: Optional[str] = None,
    input_text: Optional[str] = None,
    config_path: str = "config.json",
    output_file: Optional[str] = None,
    max_retries: int = 3,
    atlas_cache_path: Optional[str] = None,
    blend_ratio: Optional[float] = None
) -> List[str]:
    """Run the Style Atlas pipeline with file I/O.

    Args:
        input_file: Path to input text file (optional if input_text provided).
        input_text: Input text as string (optional if input_file provided).
        config_path: Path to configuration file.
        output_file: Optional path to save output.
        max_retries: Maximum number of retries per generation.
        atlas_cache_path: Optional path to ChromaDB persistence directory (falls back to config.json if None).
        blend_ratio: Optional blend ratio for style mixing (overrides config.json).

    Returns:
        List of generated paragraphs.
    """
    # Load input text
    if input_text is None:
        if input_file is None:
            raise ValueError("Either input_file or input_text must be provided")
        with open(input_file, 'r', encoding='utf-8') as f:
            input_text = f.read()

    # Resolve atlas_cache_path from config if not provided
    if atlas_cache_path is None:
        with open(config_path, 'r') as f:
            config = json.load(f)
        atlas_cache_path = config.get("atlas", {}).get("persist_path")

    # Run pipeline
    output = process_text(
        input_text=input_text,
        config_path=config_path,
        max_retries=max_retries,
        output_file=output_file,
        atlas_cache_path=atlas_cache_path,
        blend_ratio=blend_ratio
    )

    return output
