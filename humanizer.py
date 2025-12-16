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

    def to_dict(self) -> Dict:
        return {
            'success': self.success,
            'output_text': self.output_text,
            'iterations': self.iterations,
            'final_verification': self.final_verification.to_dict() if self.final_verification else None,
            'error': self.error,
            'improvement_history': self.improvement_history,
            'convergence_achieved': self.convergence_achieved,
            'opener_type_used': self.opener_type_used
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

            # Stage 2: Style Analysis (already cached)
            if verbose:
                print("\n=== Stage 2: Style Analysis ===")
                print(f"  - Formality: {self._style_profile.vocabulary.formality_score:.2f}")
                print(f"  - Avg sentence length: {self._style_profile.sentences.avg_length:.1f}")
                print(f"  - Sentence variety: {self._style_profile.sentences.length_distribution}")
                print(f"  - Structural patterns: {len(self._role_patterns)} roles cached")

            # Stage 3 & 4: Stream through paragraphs with context
            return self._transform_streaming(
                input_text, semantic_content, verbose
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
                               verbose: bool) -> PipelineResult:
        """
        Transform text using streaming paragraph-by-paragraph processing.

        Each paragraph is processed with:
        - Full document context (the original text)
        - Preceding transformed output (for consistency)
        - Accumulated statistics (for overuse detection)
        - Position awareness (for opening/closing style)
        """
        paragraphs = [p.strip() for p in input_text.split('\n\n') if p.strip()]

        if verbose:
            print(f"\n=== Stage 3 & 4: Streaming Transform ({len(paragraphs)} paragraphs) ===")
            print(f"  - Document has {len(semantic_content.claims)} total claims")
            print(f"  - Context-aware processing enabled")

        full_document_semantics = semantic_content

        transformed_paragraphs = []
        total_iterations = 0
        all_verifications = []

        # Track accumulated output for statistical verification
        accumulated_hints = []  # Hints that apply to the whole document

        # Track used opener types for variety across paragraphs
        used_openers = []

        for i, para in enumerate(paragraphs):
            if verbose:
                print(f"\n--- Paragraph {i + 1}/{len(paragraphs)} ---")

            # Build context: preceding output for consistency
            preceding_output = '\n\n'.join(transformed_paragraphs) if transformed_paragraphs else None

            # Pass accumulated hints from whole-document analysis
            doc_level_hints = accumulated_hints if accumulated_hints else None

            # Position tracking for position-aware style checks
            position = (i, len(paragraphs))

            # Transform this paragraph with full document context and variety tracking
            result = self._transform_paragraph_with_context(
                paragraph=para,
                full_document=input_text,
                full_semantics=full_document_semantics,
                preceding_output=preceding_output,
                doc_level_hints=doc_level_hints,
                position_in_document=position,
                used_openers=used_openers,  # Pass for variety
                verbose=verbose
            )

            if result.output_text:
                transformed_paragraphs.append(result.output_text)

                # Track the opener type used for variety
                if result.opener_type_used:
                    used_openers.append(result.opener_type_used)
                    if verbose:
                        print(f"  [Variety] Opener type: {result.opener_type_used} (avoiding: {used_openers[-3:] if len(used_openers) > 1 else 'none yet'})")

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
                used_openers=used_openers  # For template variety
            )

            if not synthesis_result.output_text.strip():
                if verbose:
                    print("    WARNING: Empty output from synthesizer")
                continue

            # Track opener type for variety
            if synthesis_result.template_opener_type:
                last_opener_type = synthesis_result.template_opener_type

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
                position_in_document=self._current_position  # Set by caller
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
                    opener_type_used=last_opener_type
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
            opener_type_used=last_opener_type
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

