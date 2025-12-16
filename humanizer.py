"""
Main Pipeline: Style Transfer Pipeline

Integrates all stages into a complete style transfer system:
1. Semantic Extraction - Extract meaning from input
2. Style Analysis - Analyze target style from sample
3. Synthesis - Generate new text with extracted meaning in target style
4. Verification - Validate and iteratively refine with hints

Features:
- Iterative refinement with transformation hints (genetic algorithm approach)
- Structural role awareness for context-aware synthesis
- SQLite caching for structural patterns
"""

import sys
import json
import argparse
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple
from dataclasses import dataclass, asdict, field

from semantic_extractor import SemanticExtractor, SemanticContent
from style_analyzer import StyleAnalyzer, StyleProfile
from synthesizer import Synthesizer, SynthesisResult
from verifier import Verifier, VerificationResult, TransformationHint
from structural_analyzer import StructuralAnalyzer, StructuralPattern
from cadence_analyzer import CadenceAnalyzer, CadenceProfile
from semantic_regrouper import SemanticRegrouper, SemanticChunk
from phrase_distribution_analyzer import PhraseDistributionAnalyzer
from sentence_validator import SentenceValidator


@dataclass
class PipelineResult:
    """Result of the complete pipeline."""
    success: bool
    output_text: str
    iterations: int
    final_verification: Optional[VerificationResult]
    semantic_content: Optional[SemanticContent]
    style_profile: Optional[StyleProfile]
    error: Optional[str]
    improvement_history: List[float] = field(default_factory=list)
    convergence_achieved: bool = False
    opener_type_used: Optional[str] = None  # For tracking variety across paragraphs
    opener_phrase_used: Optional[str] = None  # For exact repetition avoidance

    def to_dict(self) -> Dict:
        return {
            'success': self.success,
            'output_text': self.output_text,
            'iterations': self.iterations,
            'final_verification': self.final_verification.to_dict() if self.final_verification else None,
            'error': self.error,
            'improvement_history': self.improvement_history,
            'convergence_achieved': self.convergence_achieved,
            'opener_type_used': self.opener_type_used,
            'opener_phrase_used': self.opener_phrase_used
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)


class StyleTransferPipeline:
    """
    Complete style transfer pipeline with iterative refinement.

    Implements the 4-stage architecture:
    1. Stage 1: Semantic Extraction - Extract meaning from input
    2. Stage 2: Style Analysis - Analyze target style from sample
    3. Stage 3: Synthesis - Generate text with meaning in target style
    4. Stage 4: Verification - Validate and iteratively refine with hints

    The pipeline supports:
    - Genetic algorithm-style iterative refinement
    - Transformation hints for targeted improvements
    - Structural role awareness (section openers, paragraph openers, etc.)
    - SQLite caching for structural patterns
    - Convergence detection to stop when improvements plateau
    """

    def __init__(self, config_path: str = None, max_retries: int = None):
        """
        Initialize the pipeline.

        Args:
            config_path: Path to config.json for LLM provider settings
            max_retries: Maximum synthesis attempts (overrides config if provided)
        """
        self.config_path = config_path

        # Load pipeline settings from config
        self._load_pipeline_config(max_retries)

        # Initialize components
        print("Initializing pipeline components...")
        self.semantic_extractor = SemanticExtractor()
        self.style_analyzer = StyleAnalyzer()
        self.synthesizer = Synthesizer(config_path)
        self.verifier = Verifier(config_path)
        self.structural_analyzer = StructuralAnalyzer()

        # Track current position for context-aware verification
        self._current_position: Optional[Tuple[int, int]] = None

        # Log verification thresholds
        print(f"  Verification thresholds: drift<={self.verifier.max_meaning_drift:.0%}, claims>={self.verifier.min_claim_coverage:.0%}")

        # Load and cache sample style
        print("Analyzing sample text style...")
        self._style_profile = self.synthesizer.get_style_profile()

        # Load structural patterns from cache or generate
        print("Loading structural patterns...")
        self._role_patterns = self.synthesizer.get_role_patterns()

        # Initialize cadence analyzer (if enabled)
        self.cadence_analyzer = None
        self.cadence_profile = None
        cadence_config = self._get_cadence_config()
        if cadence_config.get("enabled", True):
            print("Analyzing sample text cadence...")
            sample_text = self.synthesizer.sample_text
            if sample_text:
                sequence_window = cadence_config.get("sequence_window_size", 5)
                position_segments = cadence_config.get("position_segments", 4)
                self.cadence_analyzer = CadenceAnalyzer(
                    sample_text,
                    self.semantic_extractor,
                    sequence_window_size=sequence_window,
                    position_segments=position_segments
                )
                self.cadence_profile = self.cadence_analyzer.analyze_cadence()
                print(f"  - Analyzed {self.cadence_profile.paragraph_count} paragraphs")
                print(f"  - Avg length: {self.cadence_profile.avg_paragraph_length:.0f} words")

        # Initialize phrase distribution analyzer (if enabled)
        self.phrase_analyzer = None
        phrase_config = self._get_phrase_config()
        if phrase_config.get("enabled", True):
            print("Analyzing phrase distribution...")
            sample_text = self.synthesizer.sample_text
            if sample_text:
                phrase_length = phrase_config.get("phrase_length", 3)
                self.phrase_analyzer = PhraseDistributionAnalyzer(sample_text, phrase_length=phrase_length)
                dist = self.phrase_analyzer.get_distribution()
                print(f"  - Analyzed {dist.total_paragraphs} paragraphs")
                print(f"  - Found {len(dist.phrase_frequencies)} unique opener phrases")

        # Initialize sentence validator (if enabled)
        self.sentence_validator = None
        sentence_config = self._get_sentence_config()
        if sentence_config.get("enabled", True):
            print("Initializing sentence validator...")
            word_count_tolerance = sentence_config.get("word_count_tolerance", 0.2)
            require_exact_opener = sentence_config.get("require_exact_opener", True)
            self.sentence_validator = SentenceValidator(
                word_count_tolerance=word_count_tolerance,
                require_exact_opener=require_exact_opener
            )
            print(f"  - Word count tolerance: ±{word_count_tolerance*100:.0f}%")
            print(f"  - Require exact opener: {require_exact_opener}")

        print(f"Pipeline ready (provider: {self.synthesizer.llm.provider}, max_retries: {self.max_retries})")

    def _load_pipeline_config(self, max_retries_override: Optional[int] = None):
        """Load pipeline configuration from config.json."""
        # Default values
        self.max_retries = 10
        self.convergence_threshold = 0.02  # Stop if improvement < 2%
        self.min_iterations = 2  # Always do at least 2 iterations

        # Try to load from config
        if self.config_path is None:
            config_path = Path(__file__).parent / "config.json"
        else:
            config_path = Path(self.config_path)

        if config_path.exists():
            try:
                with open(config_path, 'r', encoding='utf-8') as f:
                    config = json.load(f)

                pipeline_config = config.get('pipeline', {})

                if 'max_retries' in pipeline_config:
                    self.max_retries = pipeline_config['max_retries']
                if 'convergence_threshold' in pipeline_config:
                    self.convergence_threshold = pipeline_config['convergence_threshold']
                if 'min_iterations' in pipeline_config:
                    self.min_iterations = pipeline_config['min_iterations']

            except (json.JSONDecodeError, IOError):
                pass  # Use defaults if config can't be read

        # Override with provided value (CLI takes precedence)
        if max_retries_override is not None:
            self.max_retries = max_retries_override

    def _get_cadence_config(self) -> Dict[str, Any]:
        """Get cadence matching configuration from config file."""
        if self.config_path is None:
            config_path = Path(__file__).parent / "config.json"
        else:
            config_path = Path(self.config_path)

        if config_path.exists():
            try:
                with open(config_path, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                    return config.get("cadence_matching", {})
            except (json.JSONDecodeError, IOError):
                pass

        return {}  # Return empty dict with defaults

    def _get_phrase_config(self) -> Dict[str, Any]:
        """Get phrase distribution configuration from config file."""
        if self.config_path is None:
            config_path = Path(__file__).parent / "config.json"
        else:
            config_path = Path(self.config_path)

        if config_path.exists():
            try:
                with open(config_path, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                    return config.get("phrase_distribution", {})
            except (json.JSONDecodeError, IOError):
                pass

        return {}  # Return empty dict with defaults

    def _get_sentence_config(self) -> Dict[str, Any]:
        """Get sentence validation configuration from config file."""
        if self.config_path is None:
            config_path = Path(__file__).parent / "config.json"
        else:
            config_path = Path(self.config_path)

        if config_path.exists():
            try:
                with open(config_path, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                    return config.get("sentence_validation", {})
            except (json.JSONDecodeError, IOError):
                pass

        return {}  # Return empty dict with defaults

    def _estimate_tokens(self, text: str) -> int:
        """Estimate token count (rough: ~4 chars per token for English)."""
        return len(text) // 4

    def transform(self, input_text: str,
                  verbose: bool = False) -> PipelineResult:
        """
        Transform input text to target style using streaming paragraph-by-paragraph processing.

        Always uses streaming mode for consistency:
        - Each paragraph is processed with full document context
        - Statistical checks accumulate across the document
        - Position-aware hints guide opening vs closing paragraphs
        - No truncation risk regardless of document length

        Args:
            input_text: Text to transform
            verbose: Print detailed progress

        Returns:
            PipelineResult with transformed text and metadata
        """
        try:
            estimated_tokens = self._estimate_tokens(input_text)
            paragraphs = [p.strip() for p in input_text.split('\n\n') if p.strip()]

            if verbose:
                print(f"\n=== Pre-flight Check ===")
                print(f"  - Input: ~{estimated_tokens} tokens, {len(paragraphs)} paragraphs")
                print(f"  - Mode: STREAMING (context-aware paragraph processing)")

            # Stage 1: Semantic Extraction (do once, share across all paragraphs)
            if verbose:
                print("\n=== Stage 1: Semantic Extraction ===")

            semantic_content = self.semantic_extractor.extract(input_text)

            if verbose:
                print(f"  - Extracted {len(semantic_content.claims)} claims")
                print(f"  - Found {len(semantic_content.entities)} entities")
                print(f"  - Identified {len(semantic_content.relationships)} relationships")
                print(f"  - Preserved elements: {list(semantic_content.preserved_elements.keys())}")

            # Cadence-based regrouping (if enabled)
            semantic_chunks = None
            if self.cadence_profile and self.cadence_analyzer:
                cadence_config = self._get_cadence_config()
                if cadence_config.get("enabled", True):
                    if verbose:
                        print("\n=== Cadence-Based Regrouping ===")

                    regrouper = SemanticRegrouper(
                        semantic_content,
                        self.cadence_profile,
                        self.cadence_analyzer,
                        min_chunk_claims=cadence_config.get("min_chunk_claims", 1),
                        max_chunk_claims=cadence_config.get("max_chunk_claims", 5),
                        semantic_similarity_threshold=cadence_config.get("semantic_similarity_threshold", 0.6)
                    )
                    semantic_chunks = regrouper.regroup_to_cadence()

                    if verbose:
                        print(f"  - Regrouped into {len(semantic_chunks)} semantic chunks")
                        for i, chunk in enumerate(semantic_chunks[:3], 1):
                            print(f"    Chunk {i}: {len(chunk.claims)} claims, "
                                  f"{chunk.target_length} words target, {chunk.role}")

            # Stage 2: Style Analysis (already cached)
            if verbose:
                print("\n=== Stage 2: Style Analysis ===")
                print(f"  - Formality: {self._style_profile.vocabulary.formality_score:.2f}")
                print(f"  - Avg sentence length: {self._style_profile.sentences.avg_length:.1f}")
                print(f"  - Sentence variety: {self._style_profile.sentences.length_distribution}")
                print(f"  - Structural patterns: {len(self._role_patterns)} roles cached")

            # Stage 3 & 4: Stream through paragraphs with context
            return self._transform_streaming(
                input_text, semantic_content, verbose, semantic_chunks
            )

        except Exception as e:
            import traceback
            traceback.print_exc()
            return PipelineResult(
                success=False,
                output_text="",
                iterations=0,
                final_verification=None,
                semantic_content=None,
                style_profile=self._style_profile,
                error=str(e)
            )

    def _transform_streaming(self, input_text: str,
                               semantic_content: SemanticContent,
                               verbose: bool,
                               semantic_chunks: Optional[List[SemanticChunk]] = None) -> PipelineResult:
        """
        Transform text using streaming paragraph-by-paragraph processing.

        Each paragraph is processed with:
        - Full document context (the original text)
        - Preceding transformed output (for consistency)
        - Accumulated statistics (for overuse detection)
        - Position awareness (for opening/closing style)
        """
        # Use semantic chunks if available, otherwise use paragraphs
        if semantic_chunks:
            items_to_process = semantic_chunks
            item_type = "semantic chunks"
        else:
            paragraphs = [p.strip() for p in input_text.split('\n\n') if p.strip()]
            items_to_process = paragraphs
            item_type = "paragraphs"

        if verbose:
            print(f"\n=== Stage 3 & 4: Streaming Transform ({len(items_to_process)} {item_type}) ===")
            print(f"  - Document has {len(semantic_content.claims)} total claims")
            print(f"  - Context-aware processing enabled")

        full_document_semantics = semantic_content

        transformed_paragraphs = []
        total_iterations = 0
        all_verifications = []

        # Track accumulated output for statistical verification
        accumulated_hints = []  # Hints that apply to the whole document

        # Track used opener types and phrases for variety across paragraphs
        used_openers = []
        used_phrases = []

        # Track phrase usage counts for distribution matching
        output_phrase_counts: Dict[str, int] = {}
        output_total_paragraphs = 0

        for i, item in enumerate(items_to_process):
            if verbose:
                print(f"\n--- {item_type.capitalize().rstrip('s')} {i + 1}/{len(items_to_process)} ---")

            # Build context: preceding output for consistency
            preceding_output = '\n\n'.join(transformed_paragraphs) if transformed_paragraphs else None

            # Pass accumulated hints from whole-document analysis
            doc_level_hints = accumulated_hints if accumulated_hints else None

            # Position tracking for position-aware style checks
            position = (i, len(items_to_process))

            # Transform based on type
            if semantic_chunks:
                # Transform semantic chunk
                result = self._transform_chunk_with_context(
                    semantic_chunk=item,
                    full_document=input_text,
                    full_semantics=full_document_semantics,
                    preceding_output=preceding_output,
                    doc_level_hints=doc_level_hints,
                    position_in_document=position,
                    used_openers=used_openers,
                    used_phrases=used_phrases,
                    output_phrase_counts=output_phrase_counts,
                    output_total_paragraphs=output_total_paragraphs,
                    verbose=verbose
                )
            else:
                # Transform paragraph (existing logic)
                result = self._transform_paragraph_with_context(
                    paragraph=item,
                    full_document=input_text,
                    full_semantics=full_document_semantics,
                    preceding_output=preceding_output,
                    doc_level_hints=doc_level_hints,
                    position_in_document=position,
                    used_openers=used_openers,  # Pass for variety
                    used_phrases=used_phrases,  # Pass for exact repetition avoidance
                    output_phrase_counts=output_phrase_counts,
                    output_total_paragraphs=output_total_paragraphs,
                    verbose=verbose
                )

            if result.output_text:
                transformed_paragraphs.append(result.output_text)

                # Track the opener type and phrase used for variety
                if result.opener_type_used:
                    used_openers.append(result.opener_type_used)
                if result.opener_phrase_used:
                    used_phrases.append(result.opener_phrase_used)
                    # Update phrase counts for distribution matching
                    phrase_key = result.opener_phrase_used.lower().strip()
                    output_phrase_counts[phrase_key] = output_phrase_counts.get(phrase_key, 0) + 1
                    output_total_paragraphs += 1
                if verbose and (result.opener_type_used or result.opener_phrase_used):
                    avoiding_types = used_openers[-3:] if len(used_openers) > 1 else []
                    avoiding_phrases = used_phrases[-5:] if len(used_phrases) > 1 else []
                    print(f"  [Variety] Opener type: {result.opener_type_used or 'N/A'}, phrase: '{result.opener_phrase_used or 'N/A'}'")
                    if avoiding_types or avoiding_phrases:
                        print(f"  [Variety] Avoiding types: {avoiding_types}, phrases: {avoiding_phrases[:3]}")

                # Every 3 paragraphs, run statistical check on accumulated output
                if (i + 1) % 3 == 0 and self.verifier._stats_initialized:
                    accumulated_text = '\n\n'.join(transformed_paragraphs)
                    _, stat_metrics = self.verifier.style_stats.score_text(accumulated_text)

                    # Update accumulated hints based on statistical issues
                    accumulated_hints = []
                    if 'overused_markers' in stat_metrics:
                        for marker, ratio in stat_metrics['overused_markers']:
                            accumulated_hints.append(f"AVOID '{marker}' (already overused {ratio:.1f}x)")

                    if verbose and accumulated_hints:
                        print(f"  [Stats] Document-level hints: {len(accumulated_hints)}")
            else:
                # Fallback to original if transformation failed
                transformed_paragraphs.append(para)
                if verbose:
                    print(f"  WARNING: Using original paragraph (transformation failed)")

            total_iterations += result.iterations
            if result.final_verification:
                all_verifications.append(result.final_verification)

            if verbose:
                status = "PASS" if result.success else "PARTIAL"
                print(f"  Status: {status} ({result.iterations} iterations)")

        # Combine results
        combined_output = '\n\n'.join(transformed_paragraphs)

        # Final cleanup pass - ensure no AI fingerprints remain
        if self.synthesizer.ai_word_replacer:
            final_output = self.synthesizer.ai_word_replacer.replace_ai_words(combined_output)
            if final_output != combined_output:
                if verbose:
                    print("\n  [Final Pass] Removed remaining AI fingerprints from combined output")
                combined_output = final_output

        # Run final verification on combined output
        if verbose:
            print("\n=== Final Verification ===")

        final_verification = self.verifier.verify(
            input_text=input_text,
            output_text=combined_output,
            target_style=self._style_profile
        )

        return PipelineResult(
            success=final_verification.overall_passed,
            output_text=combined_output,
            iterations=total_iterations,
            final_verification=final_verification,
            semantic_content=full_document_semantics,
            style_profile=self._style_profile,
            error=None
        )

    def _transform_paragraph_with_context(self,
                                          paragraph: str,
                                          full_document: str,
                                          full_semantics: SemanticContent,
                                          preceding_output: Optional[str],
                                          doc_level_hints: Optional[List[str]] = None,
                                          position_in_document: Optional[Tuple[int, int]] = None,
                                          used_openers: Optional[List[str]] = None,
                                          used_phrases: Optional[List[str]] = None,
                                          output_phrase_counts: Optional[Dict[str, int]] = None,
                                          output_total_paragraphs: int = 0,
                                          verbose: bool = False) -> PipelineResult:
        """
        Transform a single paragraph with full document context and iterative refinement.

        Args:
            paragraph: The paragraph to transform
            full_document: The complete original document
            full_semantics: Semantics extracted from the full document
            preceding_output: Already-transformed preceding paragraphs
            doc_level_hints: Document-level hints (e.g., avoid overused words)
            position_in_document: (current_index, total_paragraphs) for position-aware checks
            used_openers: List of opener types already used in document (for variety)
            used_phrases: List of opener phrases already used (for exact repetition avoidance)
            verbose: Print progress information
        """
        # Extract paragraph-specific semantics
        para_semantics = self.semantic_extractor.extract(paragraph)

        best_output = ""
        best_verification = None
        best_score = 0.0
        improvement_history = []
        transformation_hints = None
        convergence_achieved = False
        last_opener_type = None  # Track the opener type used for variety
        last_opener_phrase = None  # Track the opener phrase used for exact repetition avoidance

        # Store position for use in verification
        self._current_position = position_in_document

        # Use fewer iterations per paragraph (we'll iterate on the whole document too)
        max_para_iterations = min(5, self.max_retries)

        # If we have document-level hints (e.g., overused words), add them
        if doc_level_hints:
            from structural_analyzer import TransformationHint
            transformation_hints = [
                TransformationHint(
                    sentence_index=-1,
                    current_text=hint,
                    structural_role="document",
                    expected_patterns=[],
                    issue=hint,
                    suggestion=hint,
                    priority=1
                )
                for hint in doc_level_hints
            ]

        # Generate and print template info once (before iterations)
        if verbose and self.synthesizer.template_generator and position_in_document:
            para_idx, total_paras = position_in_document
            position_ratio = para_idx / max(total_paras - 1, 1) if total_paras > 1 else 0.5

            # Determine role based on position
            if para_idx == 0:
                role = 'section_opener'
            elif position_ratio > 0.9:
                role = 'closer'
            elif position_ratio < 0.1:
                role = 'paragraph_opener'
            else:
                role = 'body'

            # Generate template to show structure
            template = self.synthesizer.template_generator.generate_template_from_context(
                input_text=paragraph,
                preceding_output=preceding_output,
                example_selector=self.synthesizer.example_selector,
                role=role,
                position_ratio=position_ratio,
                semantic_weight=len(para_semantics.claims) // 2,
                used_openers=used_openers,
                used_phrases=used_phrases,
                paragraph_index=para_idx,
                phrase_analyzer=self.phrase_analyzer,
                output_phrase_counts=output_phrase_counts or {},
                output_total_paragraphs=output_total_paragraphs
            )

            if template:
                context_info = "with context" if preceding_output else "position-based"
                print(f"  [Template] Role: {role}, {context_info}")
                print(f"  [Template] Structure: {template.sentence_count} sentences, ~{template.total_word_count} words")
                if template.sentences:
                    for i, sent in enumerate(template.sentences[:3], 1):  # Show first 3 sentences
                        opener_desc = sent.opener_type.replace('_', ' ')
                        complexity = f"{sent.clause_count} clause(s)" if sent.clause_count > 1 else "simple"
                        sub = " + subordinate" if sent.has_subordinate_clause else ""
                        print(f"  [Template] S{i}: ~{sent.word_count} words, {complexity}{sub}, opener: {opener_desc}")
                    if len(template.sentences) > 3:
                        print(f"  [Template] ... and {len(template.sentences) - 3} more sentence(s)")

        for iteration in range(1, max_para_iterations + 1):
            if verbose:
                hint_info = f" (with {len(transformation_hints)} hints)" if transformation_hints else ""
                print(f"    Iteration {iteration}/{max_para_iterations}{hint_info}...")

            # Synthesize with document context, preceding output, hints, and variety tracking
            synthesis_result = self.synthesizer.synthesize(
                paragraph,
                semantic_content=para_semantics,
                style_profile=self._style_profile,
                document_context=full_document,
                preceding_output=preceding_output,
                transformation_hints=transformation_hints,
                iteration=iteration - 1,
                position_in_document=self._current_position,
                used_openers=used_openers,  # For template variety
                used_phrases=used_phrases,  # For exact repetition avoidance
                verbose=verbose  # Print template details
            )

            if not synthesis_result.output_text.strip():
                if verbose:
                    print("    WARNING: Empty output from synthesizer")
                continue

            # Validate sentences against template BEFORE verification
            sentence_hints = []
            paragraph_template = synthesis_result.paragraph_template
            if paragraph_template and self.sentence_validator:
                sentence_hints = self.verifier._validate_sentence_templates(
                    synthesis_result.output_text,
                    paragraph_template,
                    self.sentence_validator
                )

            # If CRITICAL sentence hints exist, add to transformation hints and continue iteration
            critical_sentence_hints = [h for h in sentence_hints if h.priority == 1]
            if critical_sentence_hints:
                if transformation_hints:
                    transformation_hints.extend(critical_sentence_hints)
                else:
                    transformation_hints = critical_sentence_hints

                # Force another iteration (don't accept this output)
                if verbose:
                    print(f"    REJECTED: {len(critical_sentence_hints)} sentences don't match templates")
                    for hint in critical_sentence_hints[:3]:
                        print(f"      - {hint.issue}")
                continue  # Skip verification, regenerate

            # Track opener type and phrase for variety
            if synthesis_result.template_opener_type:
                last_opener_type = synthesis_result.template_opener_type
            if synthesis_result.template_opener_phrase:
                last_opener_phrase = synthesis_result.template_opener_phrase

            # Build accumulated context for full-document statistical analysis
            if preceding_output:
                accumulated_text = preceding_output + "\n\n" + synthesis_result.output_text
            else:
                accumulated_text = synthesis_result.output_text

            # Verify this paragraph with full document context
            verification = self.verifier.verify(
                input_text=paragraph,
                output_text=synthesis_result.output_text,
                input_semantics=para_semantics,
                target_style=self._style_profile,
                iteration=iteration,
                previous_score=best_score,
                accumulated_context=accumulated_text,
                position_in_document=self._current_position,  # Set by caller
                used_phrases=used_phrases,  # For phrase repetition detection
                paragraph_template=paragraph_template,  # For sentence validation
                sentence_validator=self.sentence_validator  # For sentence validation
            )

            # Calculate score and track improvement
            current_score = self._calculate_score(verification)
            improvement_history.append(current_score)

            # Track best result
            if current_score > best_score:
                best_output = synthesis_result.output_text
                best_verification = verification
                best_score = current_score
                transformation_hints = verification.transformation_hints

            # Check if passed
            if verification.overall_passed:
                return PipelineResult(
                    success=True,
                    output_text=synthesis_result.output_text,
                    iterations=iteration,
                    final_verification=verification,
                    semantic_content=para_semantics,
                    style_profile=self._style_profile,
                    error=None,
                    improvement_history=improvement_history,
                    convergence_achieved=True,
                    opener_type_used=last_opener_type,
                    opener_phrase_used=last_opener_phrase
                )

            # Check for convergence
            if iteration >= 2 and len(improvement_history) >= 2:
                recent_improvement = abs(improvement_history[-1] - improvement_history[-2])
                if recent_improvement < self.convergence_threshold:
                    convergence_achieved = True
                    if verbose:
                        print(f"    Convergence reached")
                    break

        # Return best result
        return PipelineResult(
            success=best_verification.overall_passed if best_verification else False,
            output_text=best_output,
            iterations=len(improvement_history),
            final_verification=best_verification,
            semantic_content=para_semantics,
            style_profile=self._style_profile,
            error=None if best_output else "Failed to generate valid output",
            improvement_history=improvement_history,
            convergence_achieved=convergence_achieved,
            opener_type_used=last_opener_type,
            opener_phrase_used=last_opener_phrase
        )

    def _transform_chunk_with_context(self,
                                     semantic_chunk: SemanticChunk,
                                     full_document: str,
                                     full_semantics: SemanticContent,
                                     preceding_output: Optional[str],
                                     doc_level_hints: Optional[List[str]] = None,
                                     position_in_document: Optional[Tuple[int, int]] = None,
                                     used_openers: Optional[List[str]] = None,
                                     used_phrases: Optional[List[str]] = None,
                                     output_phrase_counts: Optional[Dict[str, int]] = None,
                                     output_total_paragraphs: int = 0,
                                     verbose: bool = False) -> PipelineResult:
        """
        Transform a semantic chunk with full document context and iterative refinement.

        Similar to _transform_paragraph_with_context but uses semantic chunk directly.

        Args:
            semantic_chunk: The semantic chunk to transform
            full_document: The complete original document
            full_semantics: Semantics extracted from the full document
            preceding_output: Already-transformed preceding paragraphs
            doc_level_hints: Document-level hints (e.g., avoid overused words)
            position_in_document: (current_index, total_chunks) for position-aware checks
            used_openers: List of opener types already used in document (for variety)
            used_phrases: List of opener phrases already used (for exact repetition avoidance)
            verbose: Print progress information
        """
        # Create semantic content from chunk
        from semantic_extractor import SemanticContent
        chunk_semantics = SemanticContent(
            entities=semantic_chunk.entities,
            claims=semantic_chunk.claims,
            relationships=semantic_chunk.relationships,
            preserved_elements=full_semantics.preserved_elements,  # Use full document preserved elements
            paragraph_structure=[]  # Will be generated during synthesis
        )

        best_output = ""
        best_verification = None
        best_score = 0.0
        improvement_history = []
        transformation_hints = None
        convergence_achieved = False
        last_opener_type = None
        last_opener_phrase = None

        # Store position for use in verification
        self._current_position = position_in_document

        # Use fewer iterations per chunk
        max_chunk_iterations = min(5, self.max_retries)

        # If we have document-level hints, add them
        if doc_level_hints:
            from structural_analyzer import TransformationHint
            transformation_hints = [
                TransformationHint(
                    sentence_index=-1,
                    current_text=hint,
                    structural_role="document",
                    expected_patterns=[],
                    issue=hint,
                    suggestion="",
                    priority=2
                ) for hint in doc_level_hints
            ]

        para_idx, total_paras = position_in_document if position_in_document else (0, 1)
        position_ratio = para_idx / max(total_paras - 1, 1)

        # Use chunk's role and position
        role = semantic_chunk.role
        position_ratio = semantic_chunk.position

        for iteration in range(max_chunk_iterations):
            if verbose and iteration > 0:
                print(f"    Iteration {iteration + 1}/{max_chunk_iterations}")

            # Generate template
            template = self.synthesizer.template_generator.generate_template_from_context(
                input_text="",  # No input text, using semantic chunk
                preceding_output=preceding_output,
                example_selector=self.synthesizer.example_selector,
                role=role,
                position_ratio=position_ratio,
                semantic_weight=len(chunk_semantics.claims),
                used_openers=used_openers,
                used_phrases=used_phrases,
                phrase_analyzer=self.phrase_analyzer,
                output_phrase_counts=output_phrase_counts or {},
                output_total_paragraphs=output_total_paragraphs
            )

            # Synthesize from semantic chunk
            synthesis_result = self.synthesizer.synthesize_from_chunk(
                semantic_chunk=chunk_semantics,
                style_profile=self._style_profile,
                document_context=full_document,
                preceding_output=preceding_output,
                transformation_hints=transformation_hints,
                iteration=iteration,
                position_in_document=position_in_document,
                used_openers=used_openers,
                used_phrases=used_phrases,
                target_length=semantic_chunk.target_length,
                target_sentence_count=semantic_chunk.target_sentence_count,
                verbose=verbose
            )

            if not synthesis_result.output_text:
                continue

            # Validate sentences against template BEFORE verification
            sentence_hints = []
            paragraph_template = synthesis_result.paragraph_template
            if paragraph_template and self.sentence_validator:
                sentence_hints = self.verifier._validate_sentence_templates(
                    synthesis_result.output_text,
                    paragraph_template,
                    self.sentence_validator
                )

            # If CRITICAL sentence hints exist, add to transformation hints and continue iteration
            critical_sentence_hints = [h for h in sentence_hints if h.priority == 1]
            if critical_sentence_hints:
                if transformation_hints:
                    transformation_hints.extend(critical_sentence_hints)
                else:
                    transformation_hints = critical_sentence_hints

                # Force another iteration (don't accept this output)
                if verbose:
                    print(f"    REJECTED: {len(critical_sentence_hints)} sentences don't match templates")
                continue  # Skip verification, regenerate

            # Verify (create dummy input text from claims for verification)
            dummy_input_text = " ".join([claim.text for claim in chunk_semantics.claims[:3]])
            verification = self.verifier.verify(
                input_text=dummy_input_text,
                output_text=synthesis_result.output_text,
                input_semantics=chunk_semantics,
                target_style=self._style_profile,
                iteration=iteration,
                position_in_document=position_in_document,
                paragraph_template=paragraph_template,  # For sentence validation
                sentence_validator=self.sentence_validator  # For sentence validation
            )

            score = self._calculate_score(verification)
            improvement_history.append(score)

            if verbose:
                print(f"    Score: {score:.3f} (drift={verification.semantic.meaning_drift_score:.2f}, "
                      f"coverage={verification.semantic.claim_coverage:.2f})")

            # Update best if better
            if not best_verification or self._is_better(verification, best_verification):
                best_output = synthesis_result.output_text
                best_verification = verification
                best_score = score

                # Extract opener info from synthesis result if available
                if hasattr(synthesis_result, 'opener_type_used'):
                    last_opener_type = synthesis_result.opener_type_used
                if hasattr(synthesis_result, 'opener_phrase_used'):
                    last_opener_phrase = synthesis_result.opener_phrase_used

            # Check if verification passed
            if verification.overall_passed:
                if verbose:
                    print(f"    Verification passed after {iteration + 1} iterations")
                return PipelineResult(
                    success=True,
                    output_text=synthesis_result.output_text,
                    iterations=iteration + 1,
                    final_verification=verification,
                    semantic_content=chunk_semantics,
                    style_profile=self._style_profile,
                    error=None,
                    improvement_history=improvement_history,
                    convergence_achieved=True,
                    opener_type_used=last_opener_type,
                    opener_phrase_used=last_opener_phrase
                )

            # Check for convergence
            if iteration >= 2 and len(improvement_history) >= 2:
                recent_improvement = abs(improvement_history[-1] - improvement_history[-2])
                if recent_improvement < self.convergence_threshold:
                    convergence_achieved = True
                    if verbose:
                        print(f"    Convergence reached")
                    break

        # Return best result
        return PipelineResult(
            success=best_verification.overall_passed if best_verification else False,
            output_text=best_output,
            iterations=len(improvement_history),
            final_verification=best_verification,
            semantic_content=chunk_semantics,
            style_profile=self._style_profile,
            error=None if best_output else "Failed to generate valid output",
            improvement_history=improvement_history,
            convergence_achieved=convergence_achieved,
            opener_type_used=last_opener_type,
            opener_phrase_used=last_opener_phrase
        )

    def _calculate_score(self, verification: VerificationResult) -> float:
        """Calculate a normalized quality score (0-1) for tracking improvement."""
        # Weights for different components
        semantic_weight = 0.35
        style_weight = 0.35
        preservation_weight = 0.30

        # Semantic score
        semantic_score = (
            verification.semantic.claim_coverage * 0.5 +
            (1 - verification.semantic.meaning_drift_score) * 0.5
        )

        # Style score - include pattern coverage and discourse markers
        style_score = (
            verification.style.sentence_length_match * 0.2 +
            verification.style.formality_match * 0.2 +
            verification.style.pattern_coverage * 0.3 +  # Weight patterns heavily
            verification.style.discourse_marker_usage * 0.2 +
            verification.style.opener_diversity * 0.1
        )

        # Preservation score
        preservation_score = (
            (1.0 if verification.preservation.citations_preserved else 0.0) * 0.4 +
            (1.0 if verification.preservation.numbers_preserved else 0.0) * 0.3 +
            (1.0 if verification.preservation.quotes_preserved else 0.0) * 0.3
        )

        # Combined score
        return (
            semantic_score * semantic_weight +
            style_score * style_weight +
            preservation_score * preservation_weight
        )

    def _is_better(self, new: VerificationResult, old: VerificationResult) -> bool:
        """Determine if new verification result is better than old."""
        # Use the new scoring method
        new_score = self._calculate_score(new)
        old_score = self._calculate_score(old)
        return new_score > old_score

    def _is_better_legacy(self, new: VerificationResult, old: VerificationResult) -> bool:
        """Legacy comparison method - kept for reference."""
        # Score based on passing checks
        new_score = (
            (1 if new.semantic.passed else 0) * 3 +  # Weight semantics highest
            (1 if new.style.passed else 0) * 2 +
            (1 if new.preservation.passed else 0) * 3 +  # Preservation critical
            new.semantic.claim_coverage +
            (1 - new.semantic.meaning_drift_score)
        )

        old_score = (
            (1 if old.semantic.passed else 0) * 3 +
            (1 if old.style.passed else 0) * 2 +
            (1 if old.preservation.passed else 0) * 3 +
            old.semantic.claim_coverage +
            (1 - old.semantic.meaning_drift_score)
        )

        return new_score > old_score

    def analyze_input(self, input_text: str) -> Dict[str, Any]:
        """Analyze input text without transforming (for debugging)."""
        semantic = self.semantic_extractor.extract(input_text)
        return {
            'claims': len(semantic.claims),
            'entities': len(semantic.entities),
            'relationships': len(semantic.relationships),
            'preserved_elements': semantic.preserved_elements,
            'paragraph_structure': semantic.paragraph_structure
        }

    def analyze_style(self, text: str) -> Dict[str, Any]:
        """Analyze style of any text (for debugging)."""
        style = self.style_analyzer.analyze(text)
        return {
            'formality': style.vocabulary.formality_score,
            'avg_sentence_length': style.sentences.avg_length,
            'length_distribution': style.sentences.length_distribution,
            'opener_distribution': style.sentences.opener_distribution,
            'assertiveness': style.tone.assertiveness
        }


def main():
    """Command-line interface for the pipeline."""
    parser = argparse.ArgumentParser(
        description='Style Transfer Pipeline - Transform text to match sample style',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  python humanizer.py input/document.md                    # Basic usage
  python humanizer.py input/document.md -o output/doc.md   # Custom output
  python humanizer.py input/document.md --verbose          # Detailed progress
  python humanizer.py input/document.md -r 20              # More iterations

Processing Mode:
  Always uses streaming paragraph-by-paragraph processing for:
  - Consistent output quality regardless of document length
  - Context-aware statistical verification (detects overused words)
  - Position-aware style hints (opening vs closing paragraphs)
'''
    )

    parser.add_argument('input_file', help='Input file to transform')
    parser.add_argument('-o', '--output', help='Output file path (default: output/<input_name>)')
    parser.add_argument('-c', '--config', help='Config file path (default: config.json)')
    parser.add_argument('-r', '--retries', type=int, default=None, help='Max retry attempts (overrides config.json, default: from config or 10)')
    parser.add_argument('-v', '--verbose', action='store_true', help='Verbose output')
    parser.add_argument('--analyze-only', action='store_true', help='Only analyze input, no transformation')
    parser.add_argument('--json', action='store_true', help='Output results as JSON')

    args = parser.parse_args()

    # Read input file
    input_path = Path(args.input_file)
    if not input_path.exists():
        print(f"Error: Input file not found: {input_path}")
        sys.exit(1)

    with open(input_path, 'r', encoding='utf-8') as f:
        input_text = f.read()

    # Determine output path
    if args.output:
        output_path = Path(args.output)
    else:
        output_dir = Path(__file__).parent / "output"
        output_dir.mkdir(exist_ok=True)
        output_path = output_dir / input_path.name

    # Initialize pipeline
    print(f"Input: {input_path}")
    print(f"Output: {output_path}")
    print()

    pipeline = StyleTransferPipeline(
        config_path=args.config,
        max_retries=args.retries
    )

    # Analyze-only mode
    if args.analyze_only:
        print("\n=== Input Analysis ===")
        analysis = pipeline.analyze_input(input_text)
        if args.json:
            print(json.dumps(analysis, indent=2))
        else:
            print(f"Claims: {analysis['claims']}")
            print(f"Entities: {analysis['entities']}")
            print(f"Relationships: {analysis['relationships']}")
            print(f"Preserved elements: {list(analysis['preserved_elements'].keys())}")
        sys.exit(0)

    # Run transformation
    print("\n" + "=" * 50)
    print("Starting style transfer...")
    print("=" * 50)

    result = pipeline.transform(
        input_text,
        verbose=args.verbose
    )

    # Output results
    if args.json:
        print("\n" + result.to_json())
    else:
        print("\n" + "=" * 50)
        print("RESULTS")
        print("=" * 50)
        print(f"Success: {result.success}")
        print(f"Iterations: {result.iterations}")

        if result.final_verification:
            v = result.final_verification
            print(f"\nVerification:")
            print(f"  Semantic: {'PASS' if v.semantic.passed else 'FAIL'} "
                  f"(claims: {v.semantic.claim_coverage:.0%}, drift: {v.semantic.meaning_drift_score:.0%})")
            print(f"  Style: {'PASS' if v.style.passed else 'FAIL'}")
            print(f"  Preservation: {'PASS' if v.preservation.passed else 'FAIL'}")

            if v.recommendations:
                print(f"\nRecommendations:")
                for rec in v.recommendations:
                    print(f"  - {rec}")

        if result.error:
            print(f"\nError: {result.error}")

    # Write output
    if result.output_text:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(result.output_text)
        print(f"\nOutput written to: {output_path}")
    else:
        print("\nNo output generated.")
        sys.exit(1)


if __name__ == '__main__':
    main()

