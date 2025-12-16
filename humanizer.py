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
from typing import Optional, Dict, Any, List
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

    def to_dict(self) -> Dict:
        return {
            'success': self.success,
            'output_text': self.output_text,
            'iterations': self.iterations,
            'final_verification': self.final_verification.to_dict() if self.final_verification else None,
            'error': self.error,
            'improvement_history': self.improvement_history,
            'convergence_achieved': self.convergence_achieved
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

    def __init__(self, config_path: str = None, max_retries: int = 10):
        """
        Initialize the pipeline.

        Args:
            config_path: Path to config.json for LLM provider settings
            max_retries: Maximum synthesis attempts (default 10 for genetic approach)
        """
        self.config_path = config_path
        self.max_retries = max_retries

        # Convergence settings
        self.convergence_threshold = 0.02  # Stop if improvement < 2%
        self.min_iterations = 2  # Always do at least 2 iterations

        # Initialize components
        print("Initializing pipeline components...")
        self.semantic_extractor = SemanticExtractor()
        self.style_analyzer = StyleAnalyzer()
        self.synthesizer = Synthesizer(config_path)
        self.verifier = Verifier(config_path)
        self.structural_analyzer = StructuralAnalyzer()

        # Log verification thresholds
        print(f"  Verification thresholds: drift<={self.verifier.max_meaning_drift:.0%}, claims>={self.verifier.min_claim_coverage:.0%}")

        # Load and cache sample style
        print("Analyzing sample text style...")
        self._style_profile = self.synthesizer.get_style_profile()

        # Load structural patterns from cache or generate
        print("Loading structural patterns...")
        self._role_patterns = self.synthesizer.get_role_patterns()

        print(f"Pipeline ready (provider: {self.synthesizer.llm.provider}, max_retries: {max_retries})")

    def transform(self, input_text: str,
                  verbose: bool = False,
                  chunk_mode: bool = False) -> PipelineResult:
        """
        Transform input text to target style.

        Args:
            input_text: Text to transform
            verbose: Print detailed progress
            chunk_mode: Process paragraph by paragraph for long texts

        Returns:
            PipelineResult with transformed text and metadata
        """
        try:
            if chunk_mode:
                return self._transform_chunked(input_text, verbose)
            else:
                return self._transform_full(input_text, verbose)
        except Exception as e:
            return PipelineResult(
                success=False,
                output_text="",
                iterations=0,
                final_verification=None,
                semantic_content=None,
                style_profile=self._style_profile,
                error=str(e)
            )

    def _transform_full(self, input_text: str, verbose: bool) -> PipelineResult:
        """Transform entire text at once with iterative refinement."""
        # Stage 1: Semantic Extraction
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

        # Stage 3 & 4: Iterative Synthesis with Verification and Hints
        if verbose:
            print("\n=== Stage 3 & 4: Iterative Refinement (Genetic Approach) ===")

        best_output = ""
        best_verification = None
        best_score = 0.0
        improvement_history = []
        transformation_hints = None
        convergence_achieved = False

        for iteration in range(1, self.max_retries + 1):
            if verbose:
                print(f"\n  Iteration {iteration}/{self.max_retries}...")
                if transformation_hints:
                    print(f"    Applying {len(transformation_hints)} transformation hints")

            # Stage 3: Synthesis with hints from previous iteration
            synthesis_result = self.synthesizer.synthesize(
                input_text,
                semantic_content=semantic_content,
                style_profile=self._style_profile,
                transformation_hints=transformation_hints,
                iteration=iteration - 1  # 0-indexed for synthesis
            )

            if not synthesis_result.output_text.strip():
                if verbose:
                    print("  WARNING: Empty output from synthesizer")
                continue

            # Stage 4: Verification with hint generation
            verification = self.verifier.verify(
                input_text=input_text,
                output_text=synthesis_result.output_text,
                input_semantics=semantic_content,
                target_style=self._style_profile,
                iteration=iteration,
                previous_score=best_score
            )

            # Calculate current score
            current_score = self._calculate_score(verification)
            improvement = current_score - best_score
            improvement_history.append(current_score)

            if verbose:
                print(f"  - Score: {current_score:.2%} (improvement: {improvement:+.1%})")
                print(f"  - Semantic: {'PASS' if verification.semantic.passed else 'FAIL'} "
                      f"(claims: {verification.semantic.claim_coverage:.0%})")
                print(f"  - Style: {'PASS' if verification.style.passed else 'FAIL'} "
                      f"(pattern coverage: {verification.style.pattern_coverage:.0%})")
                print(f"  - Preservation: {'PASS' if verification.preservation.passed else 'FAIL'}")
                if verification.transformation_hints:
                    print(f"  - Generated {len(verification.transformation_hints)} hints for next iteration")

            # Track best result
            if current_score > best_score:
                best_output = synthesis_result.output_text
                best_verification = verification
                best_score = current_score

                # Get hints for next iteration
                transformation_hints = verification.transformation_hints

            # Check if passed
            if verification.overall_passed:
                if verbose:
                    print(f"\n  SUCCESS on iteration {iteration}!")

                return PipelineResult(
                    success=True,
                    output_text=synthesis_result.output_text,
                    iterations=iteration,
                    final_verification=verification,
                    semantic_content=semantic_content,
                    style_profile=self._style_profile,
                    error=None,
                    improvement_history=improvement_history,
                    convergence_achieved=True
                )

            # Check for convergence (improvement plateau)
            if iteration >= self.min_iterations and len(improvement_history) >= 2:
                recent_improvement = abs(improvement_history[-1] - improvement_history[-2])
                if recent_improvement < self.convergence_threshold:
                    convergence_achieved = True
                    if verbose:
                        print(f"\n  CONVERGENCE reached (improvement: {recent_improvement:.1%} < threshold {self.convergence_threshold:.1%})")
                    break

            # Log issues for retry
            if verbose and verification.recommendations:
                print("  Issues to address:")
                for rec in verification.recommendations[:3]:
                    print(f"    - {rec}")
                if transformation_hints:
                    print("  Top hints for next iteration:")
                    for hint in transformation_hints[:2]:
                        print(f"    - {hint.issue}")

        # Return best result even if not perfect
        if verbose:
            print(f"\n  Returning best result after {len(improvement_history)} iterations")
            print(f"  Final score: {best_score:.2%}")

        return PipelineResult(
            success=best_verification.overall_passed if best_verification else False,
            output_text=best_output,
            iterations=len(improvement_history),
            final_verification=best_verification,
            semantic_content=semantic_content,
            style_profile=self._style_profile,
            error=None if best_output else "Failed to generate valid output",
            improvement_history=improvement_history,
            convergence_achieved=convergence_achieved
        )

    def _transform_chunked(self, input_text: str, verbose: bool) -> PipelineResult:
        """Transform text paragraph by paragraph with full document context."""
        paragraphs = [p.strip() for p in input_text.split('\n\n') if p.strip()]

        if verbose:
            print(f"\n=== Chunked Mode: {len(paragraphs)} paragraphs ===")

        # Extract semantics from the FULL document first for holistic understanding
        if verbose:
            print("\n  Extracting full document semantics...")
        full_document_semantics = self.semantic_extractor.extract(input_text)

        if verbose:
            print(f"  - Document has {len(full_document_semantics.claims)} total claims")
            print(f"  - Found {len(full_document_semantics.entities)} entities")
            print(f"  - Identified {len(full_document_semantics.relationships)} relationships")

        transformed_paragraphs = []
        total_iterations = 0
        all_verifications = []

        for i, para in enumerate(paragraphs):
            if verbose:
                print(f"\n--- Paragraph {i + 1}/{len(paragraphs)} ---")

            # Build context: preceding output for consistency
            preceding_output = '\n\n'.join(transformed_paragraphs) if transformed_paragraphs else None

            # Transform this paragraph with full document context
            result = self._transform_paragraph_with_context(
                paragraph=para,
                full_document=input_text,
                full_semantics=full_document_semantics,
                preceding_output=preceding_output,
                verbose=verbose
            )

            if result.output_text:
                transformed_paragraphs.append(result.output_text)
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
                                          verbose: bool) -> PipelineResult:
        """Transform a single paragraph with full document context and iterative refinement."""
        # Extract paragraph-specific semantics
        para_semantics = self.semantic_extractor.extract(paragraph)

        best_output = ""
        best_verification = None
        best_score = 0.0
        improvement_history = []
        transformation_hints = None
        convergence_achieved = False

        # Use fewer iterations per paragraph (we'll iterate on the whole document too)
        max_para_iterations = min(5, self.max_retries)

        for iteration in range(1, max_para_iterations + 1):
            if verbose:
                hint_info = f" (with {len(transformation_hints)} hints)" if transformation_hints else ""
                print(f"    Iteration {iteration}/{max_para_iterations}{hint_info}...")

            # Synthesize with document context, preceding output, and hints
            synthesis_result = self.synthesizer.synthesize(
                paragraph,
                semantic_content=para_semantics,
                style_profile=self._style_profile,
                document_context=full_document,
                preceding_output=preceding_output,
                transformation_hints=transformation_hints,
                iteration=iteration - 1
            )

            if not synthesis_result.output_text.strip():
                if verbose:
                    print("    WARNING: Empty output from synthesizer")
                continue

            # Verify this paragraph with hint generation
            verification = self.verifier.verify(
                input_text=paragraph,
                output_text=synthesis_result.output_text,
                input_semantics=para_semantics,
                target_style=self._style_profile,
                iteration=iteration,
                previous_score=best_score
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
                    convergence_achieved=True
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
            convergence_achieved=convergence_achieved
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
  python pipeline.py input/document.md                    # Basic usage
  python pipeline.py input/document.md -o output/doc.md  # Custom output
  python pipeline.py input/document.md --verbose         # Detailed progress
  python pipeline.py input/document.md --chunked         # Process by paragraph
'''
    )

    parser.add_argument('input_file', help='Input file to transform')
    parser.add_argument('-o', '--output', help='Output file path (default: output/<input_name>)')
    parser.add_argument('-c', '--config', help='Config file path (default: config.json)')
    parser.add_argument('-r', '--retries', type=int, default=3, help='Max retry attempts (default: 3)')
    parser.add_argument('-v', '--verbose', action='store_true', help='Verbose output')
    parser.add_argument('--chunked', action='store_true', help='Process paragraph by paragraph')
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
        verbose=args.verbose,
        chunk_mode=args.chunked
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

