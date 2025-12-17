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
    StructureNavigator
)
from src.generator.llm_interface import generate_sentence
from src.validator.critic import generate_with_critic, ConvergenceError
from src.analyzer.style_metrics import get_style_vector
from src.analyzer.vocabulary import extract_global_vocabulary
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
    sample_text: str,
    config_path: str = "config.json",
    max_retries: int = 3,
    output_file: Optional[str] = None,
    atlas_cache_path: Optional[str] = None,
    clear_db: bool = False
) -> List[str]:
    """Process input text through Style Atlas pipeline.

    Args:
        input_text: The input text to transform.
        sample_text: The sample text defining the target style.
        config_path: Path to configuration file.
        max_retries: Maximum number of retries per generation (default: 3).
        output_file: Optional path to write output incrementally.
        atlas_cache_path: Optional path to cache/load Style Atlas.

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

    # Clear ChromaDB if requested
    if clear_db:
        from src.atlas.builder import clear_chromadb_collection
        print("Clearing ChromaDB collection...")
        cleared = clear_chromadb_collection(
            collection_name=collection_name,
            persist_directory=atlas_cache_path
        )
        if cleared:
            print("  ✓ Cleared ChromaDB collection")
        else:
            print("  ⚠ Collection did not exist (nothing to clear)")

    # Phase 1: Build or load Style Atlas
    print("Phase 1: Building Style Atlas...")
    atlas = None

    # If we cleared the DB, we must rebuild (don't load from cache)
    if not clear_db and atlas_cache_path:
        cache_file = Path(atlas_cache_path) / "atlas.json"
        if cache_file.exists():
            try:
                print(f"  Loading atlas from cache: {cache_file}")
                atlas = load_atlas(str(cache_file), persist_directory=atlas_cache_path)
                # Verify collection exists and has data
                if hasattr(atlas, '_collection'):
                    try:
                        collection = atlas._collection
                        test_results = collection.get(limit=1)
                        if test_results['ids'] and len(test_results['ids']) > 0:
                            print(f"  ✓ Loaded atlas with {len(atlas.cluster_ids)} paragraphs")
                        else:
                            print(f"  ⚠ Cached atlas collection is empty, rebuilding...")
                            atlas = None
                    except Exception:
                        print(f"  ⚠ Cached atlas collection not accessible, rebuilding...")
                        atlas = None
                else:
                    print(f"  ⚠ Cached atlas has no collection connection, rebuilding...")
                    atlas = None
            except Exception as e:
                print(f"  ⚠ Failed to load cache: {e}")
                atlas = None

    if atlas is None:
        print("  Building new atlas...")
        persist_dir = atlas_cache_path if atlas_cache_path else None
        atlas = build_style_atlas(
            sample_text,
            num_clusters=num_clusters,
            collection_name=collection_name,
            persist_directory=persist_dir
        )
        print(f"  ✓ Built atlas with {len(atlas.cluster_ids)} paragraphs, {atlas.num_clusters} clusters")

        # Save to cache if path provided
        if atlas_cache_path:
            cache_file = Path(atlas_cache_path) / "atlas.json"
            cache_file.parent.mkdir(parents=True, exist_ok=True)
            save_atlas(atlas, str(cache_file))
            print(f"  ✓ Saved atlas to cache")

    # Build cluster Markov chain
    print("Phase 1: Building cluster Markov chain...")
    cluster_markov = build_cluster_markov(atlas)
    print(f"  ✓ Built Markov chain with {len(cluster_markov[1])} clusters")

    # Initialize StructureNavigator for stochastic selection
    print("Phase 1: Initializing structure navigator...")
    structure_navigator = StructureNavigator(history_limit=3)
    print(f"  ✓ Initialized structure navigator")

    # Extract global vocabulary from sample text
    print("Phase 1: Extracting global vocabulary...")
    global_vocab_dict = extract_global_vocabulary(sample_text, top_n=200)
    print(f"  ✓ Extracted vocabulary: {len(global_vocab_dict.get('positive', []))} positive, "
          f"{len(global_vocab_dict.get('negative', []))} negative, "
          f"{len(global_vocab_dict.get('neutral', []))} neutral words")

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

                # Retrieve dual RAG references
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

                # Determine sentiment and select appropriate vocabulary
                sentiment = _detect_sentiment(content_unit.original_text)
                sentiment_key = sentiment.lower() if sentiment else 'neutral'
                global_vocab_list = global_vocab_dict.get(sentiment_key, global_vocab_dict.get('neutral', []))

                # Generate with critic loop and pipeline-level retries
                generated = None
                critic_result = None
                final_score = 0.0

                try:
                    # Pipeline-level retry loop
                    for pipeline_attempt in range(critic_max_pipeline_retries + 1):
                        try:
                            generated, critic_result = generate_with_critic(
                                generate_fn=lambda cu, struct_match, sit_match, cfg, hint=None: generate_sentence(
                                    cu, struct_match, sit_match, cfg, hint=hint, global_vocab_list=global_vocab_list
                                ),
                                content_unit=content_unit,
                                structure_match=structure_match,
                                situation_match=situation_match,
                                config_path=config_path,
                                max_retries=critic_max_retries,
                                min_score=critic_min_score
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
    sample_file: Optional[str] = None,
    input_text: Optional[str] = None,
    sample_text: Optional[str] = None,
    config_path: str = "config.json",
    output_file: Optional[str] = None,
    max_retries: int = 3,
    atlas_cache_path: Optional[str] = None,
    clear_db: bool = False
) -> List[str]:
    """Run the Style Atlas pipeline with file I/O.

    Args:
        input_file: Path to input text file (optional if input_text provided).
        sample_file: Path to sample text file (optional if sample_text provided).
        input_text: Input text as string (optional if input_file provided).
        sample_text: Sample text as string (optional if sample_file provided).
        config_path: Path to configuration file.
        output_file: Optional path to save output.
        max_retries: Maximum number of retries per generation.
        atlas_cache_path: Optional path to cache Style Atlas.
        clear_db: Whether to clear ChromaDB collection before building atlas.

    Returns:
        List of generated paragraphs.
    """
    # Load input text
    if input_text is None:
        if input_file is None:
            raise ValueError("Either input_file or input_text must be provided")
        with open(input_file, 'r', encoding='utf-8') as f:
            input_text = f.read()

    # Load sample text
    if sample_text is None:
        if sample_file is None:
            # Try to load from config
            with open(config_path, 'r') as f:
                config = json.load(f)
            sample_file = config.get('sample', {}).get('file', 'prompts/sample_mao.txt')

        with open(sample_file, 'r', encoding='utf-8') as f:
            sample_text = f.read()

    # Run pipeline
    output = process_text(
        input_text=input_text,
        sample_text=sample_text,
        config_path=config_path,
        max_retries=max_retries,
        output_file=output_file,
        atlas_cache_path=atlas_cache_path,
        clear_db=clear_db
    )

    return output
