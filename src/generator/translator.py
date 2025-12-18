"""Style translator for Pipeline 2.0.

This module translates semantic blueprints into styled text using
few-shot examples from a rhetorically-indexed style atlas.
"""

import json
import re
from typing import List, Tuple
from src.ingestion.blueprint import SemanticBlueprint
from src.atlas.rhetoric import RhetoricalType
from src.generator.llm_provider import LLMProvider
from src.generator.llm_interface import clean_generated_text


# Positional instructions for contextual anchoring
POSITIONAL_INSTRUCTIONS = {
    "OPENER": """
### CONTEXT: PARAGRAPH OPENER
- **Role:** This sentence initializes a new thought unit.
- **Requirement:** Establish the central theme or subject immediately.
- **Constraint:** Do not use transitional words that imply a missing previous sentence (like "Therefore," "However," or "As a result").
- **Tone:** Be declarative, framing, and authoritative. Set the stage for the sentences to follow.
""",
    "BODY": """
### CONTEXT: CONTINUATION (INTERNAL BODY)
- **Role:** This sentence develops the argument established in the [PREVIOUS CONTEXT].
- **Requirement:** MAINTAIN FLOW. Your output must logically follow the [PREVIOUS CONTEXT].
- **Critical Instruction:** If the Input Text contains metaphors (e.g., "objects break", "stars die"), interpret them **in the context of the Previous Sentence**. Do not treat them as literal isolated statements.
- **Tone:** Connective and expansive. Use the generated style to bridge the gap between the previous thought and this one.
""",
    "CLOSER": """
### CONTEXT: PARAGRAPH CONCLUSION
- **Role:** This sentence terminates the current thought unit.
- **Requirement:** Synthesize the preceding arguments into a final point.
- **Constraint:** Do not introduce entirely new evidence or unrelated topics.
- **Tone:** Conclusive, resonant, and final. Provide a sense of closure to the reader.
""",
    "SINGLETON": """
### CONTEXT: STANDALONE SENTENCE
- **Role:** This sentence is a complete thought unit by itself.
- **Requirement:** Be self-contained and complete.
- **Tone:** Declarative and authoritative.
"""
}

# Refinement prompt for hill climbing evolution (fluency-aware)
REFINEMENT_PROMPT = """
### ROLE: Expert Editor
You are refining a draft text to improve its accuracy and flow.

### INPUT DATA
1. **Original Blueprint:** "{blueprint_text}" (This is the TRUTH. Do not deviate.)
2. **Current Draft:** {current_draft}
3. **Error Report:** {critique_feedback}

### STRATEGY
- The Critic has flagged specific issues.
- If the Error Report says "Missing Concepts", **ADD** them from the Blueprint.
- If the Error Report says "Hallucinated", **REMOVE** the extra words.
- If the Error Report says "Incomplete", **simplify the syntax**.

### TASK: Mutate and Polish
- **Priority 1 (Fix Errors):** Address the specific "Missing" or "Hallucinated" concepts listed in the feedback.
- **Priority 2 (Fluency):** Ensure the sentence is grammatically natural and stylistically resonant.
    - *Permission:* You MAY add necessary functional words (articles, prepositions, auxiliary verbs) to ensure flow.
    - *Constraint:* Do not change the core meaning or subject matter.
- **Style:** Maintain the requested rhetorical style: {rhetorical_type}.

### COMMAND
Rewrite the draft to fix the errors while maintaining the Blueprint's meaning.

### OUTPUT
Return ONLY the refined sentence.
"""

# Simplification prompt for emergency pivot when stuck
SIMPLIFICATION_PROMPT = """
### EMERGENCY REWRITE
The previous drafts are too complex or hallucinated.
Rewrite the following Blueprint into a SINGLE, SIMPLE, DECLARATIVE sentence.
Use ONLY the keywords provided. Do not be poetic.

Blueprint: "{blueprint_text}"
"""


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

    def translate(
        self,
        blueprint: SemanticBlueprint,
        author_name: str,
        style_dna: str,
        rhetorical_type: RhetoricalType,
        examples: List[str]  # 3 examples from atlas
    ) -> str:
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
        if not examples:
            # Fallback if no examples provided
            examples = ["Example text in the target style."]

        prompt = self._build_prompt(blueprint, author_name, style_dna, rhetorical_type, examples)

        system_prompt = f"""You are {author_name}.
Your writing style: {style_dna}

Your task is to rewrite semantic blueprints into your authentic voice.
Do NOT copy words from examples. Use them only as style references.
Focus on the meaning (subjects, actions, objects) and express it in your style."""

        try:
            generated = self.llm_provider.call(
                system_prompt=system_prompt,
                user_prompt=prompt,
                temperature=self.translator_config.get("temperature", 0.7),
                max_tokens=self.translator_config.get("max_tokens", 300)
            )
            generated = clean_generated_text(generated)
            generated = generated.strip()
            # Restore citations and quotes if missing
            generated = self._restore_citations_and_quotes(generated, blueprint)
            return generated
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
        examples_text = "\n".join([f"Example {i+1}: \"{ex}\"" for i, ex in enumerate(examples)])

        # Get position-specific instruction
        pos_instruction = POSITIONAL_INSTRUCTIONS.get(
            blueprint.position,
            POSITIONAL_INSTRUCTIONS["BODY"]
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

            return f"""TASK: Rewrite the following text into your voice.
RHETORICAL MODE: {rhetorical_type.value}

{context_block}=== STYLE EXAMPLES (Use ONLY for tone/style reference, DO NOT copy words) ===
HERE IS HOW YOU WRITE {rhetorical_type.value}s:
{examples_text}

=== INPUT TEXT (This is the content you must preserve) ===
{blueprint.original_text}
{preservation_section}

{pos_instruction}
=== INSTRUCTIONS ===
1. Use the STYLE EXAMPLES above to understand the tone and voice.
2. Use the INPUT TEXT above to understand the content you must preserve.
3. DO NOT copy words from the examples - they are only style references.
4. Express the input text's meaning in the style shown in the examples.
{preservation_section}

REWRITE (output ONLY the rewritten text, no explanations or instructions):"""

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

        return f"""TASK: Rewrite the following "Semantic Blueprint" into your voice.
RHETORICAL MODE: {rhetorical_type.value}

{context_block}=== STYLE EXAMPLES (Use ONLY for tone/style reference, DO NOT copy words) ===
HERE IS HOW YOU WRITE {rhetorical_type.value}s:
{examples_text}

=== INPUT BLUEPRINT (This is the MEANING you must preserve) ===
- Subjects: {', '.join(subjects) if subjects else 'None'}
- Actions: {', '.join(verbs) if verbs else 'None'}
- Objects: {', '.join(objects) if objects else 'None'}
- Entities: {entities_text}
- Keywords: {', '.join(keywords) if keywords else 'None'}
{preservation_section}

{pos_instruction}
=== INSTRUCTIONS ===
1. Use the STYLE EXAMPLES above to understand the tone and voice.
2. Use the INPUT BLUEPRINT above to understand the meaning you must preserve.
3. DO NOT copy words from the examples - they are only style references.
4. DO NOT ignore the blueprint - you must include all subjects, actions, and objects.
5. Express the blueprint's meaning in the style shown in the examples.
{preservation_section}

REWRITE (output ONLY the rewritten text, no explanations or instructions):"""

    def translate_literal(
        self,
        blueprint: SemanticBlueprint,
        author_name: str,
        style_dna: str
    ) -> str:
        """Literal translation fallback (no style, just meaning preservation).

        Args:
            blueprint: Semantic blueprint.
            author_name: Target author name.
            style_dna: Style DNA description.

        Returns:
            Literally translated text.
        """
        # FAILSAFE: If blueprint is empty, use original text directly
        if not blueprint.svo_triples and not blueprint.core_keywords:
            # Build citations and quotes sections even for empty blueprint
            citations_text = ""
            if blueprint.citations:
                citation_list = [cit[0] for cit in blueprint.citations]
                citations_text = f"\nCitations to include: {', '.join(citation_list)}"

            quotes_text = ""
            if blueprint.quotes:
                quote_list = [quote[0] for quote in blueprint.quotes]
                quotes_text = f"\nQuotes to preserve exactly: {', '.join(quote_list)}"

            prompt = f"""Rewrite this sentence clearly. Do not worry about style. Just preserve meaning.

INPUT TEXT: {blueprint.original_text}{citations_text}{quotes_text}

CRITICAL:
- Include all citations using EXACTLY the [^number] format shown above
- DO NOT use (Author, Year), (Smith 42), or any other citation formats - ONLY [^number] format is allowed
- Preserve all quotes exactly as shown

Rewrite (output ONLY the rewritten text, no explanations or instructions):"""
        else:
            # Normal blueprint path
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

            prompt = f"""Rewrite this sentence clearly. Do not worry about style. Just preserve meaning.

Subjects: {', '.join(subjects) if subjects else 'None'}
Actions: {', '.join(verbs) if verbs else 'None'}
Objects: {', '.join(objects) if objects else 'None'}{citations_text}{quotes_text}

CRITICAL:
- Include all citations using EXACTLY the [^number] format shown above
- DO NOT use (Author, Year), (Smith 42), or any other citation formats - ONLY [^number] format is allowed
- Preserve all quotes exactly as shown

Rewrite (output ONLY the rewritten text, no explanations or instructions):"""

        system_prompt = f"You are {author_name}. Write clearly and preserve meaning."

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
            # Ultimate fallback: return original text
            return blueprint.original_text

    def _check_acceptance(
        self,
        recall_score: float,
        precision_score: float,
        overall_score: float,
        pass_threshold: float
    ) -> bool:
        """Check if draft should be accepted using Fluency Forgiveness logic.

        RULE 1: Recall is King. If we miss keywords, we fail.
        RULE 2: Precision is Flexible. If Recall is perfect, we can accept lower precision.
        Fallback: High overall score.

        Args:
            recall_score: Recall score (0-1)
            precision_score: Precision score (0-1)
            overall_score: Weighted overall score (0-1)
            pass_threshold: Default pass threshold

        Returns:
            True if draft should be accepted, False otherwise
        """
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
        rhetorical_type: RhetoricalType
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

        Returns:
            Simplified text generated from blueprint
        """
        blueprint_text = self._get_blueprint_text(blueprint)

        simplification_prompt = SIMPLIFICATION_PROMPT.format(
            blueprint_text=blueprint_text
        )

        system_prompt = f"""You are {author_name}.
Your writing style: {style_dna}

Your task is to create a simple, declarative sentence from the blueprint.
Be literal and direct. Do not be poetic or complex."""

        try:
            simplified = self.llm_provider.call(
                system_prompt=system_prompt,
                user_prompt=simplification_prompt,
                temperature=0.2,  # Low temperature for simplicity
                max_tokens=self.translator_config.get("max_tokens", 200)
            )
            simplified = clean_generated_text(simplified)
            simplified = simplified.strip()
            # Restore citations and quotes if missing
            simplified = self._restore_citations_and_quotes(simplified, blueprint)
            return simplified
        except Exception as e:
            # Fallback: return best draft if simplification fails
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
        verbose: bool = False
    ) -> Tuple[str, float]:
        """Evolve text using hill climbing refinement.

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

        # Get blueprint text representation
        blueprint_text = self._get_blueprint_text(blueprint)

        # Evolution loop
        for gen in range(max_generations):
            # Check if we've reached acceptance criteria (using Fluency Forgiveness)
            # Evaluate current best draft to get recall/precision
            best_result = critic.evaluate(best_draft, blueprint)
            if self._check_acceptance(
                recall_score=best_result["recall_score"],
                precision_score=best_result["precision_score"],
                overall_score=best_score,
                pass_threshold=pass_threshold
            ):
                if verbose:
                    print(f"  Evolution: Draft accepted (recall: {best_result['recall_score']:.2f}, precision: {best_result['precision_score']:.2f}, score: {best_score:.2f})")
                break

            if verbose:
                print(f"  Evolution Generation {gen + 1}/{max_generations}")
                print(f"    Current Score: {best_score:.2f}")

            # Build refinement prompt
            refinement_prompt = REFINEMENT_PROMPT.format(
                blueprint_text=blueprint_text,
                current_draft=best_draft,
                critique_feedback=best_feedback,
                rhetorical_type=rhetorical_type.value
            )

            # System prompt for refinement
            system_prompt = f"""You are {author_name}, an editor refining a draft.
Your writing style: {style_dna}

Your task is to refine the draft based on specific critique.
Keep what works, fix what doesn't."""

            try:
                # Call LLM for refinement with dynamic temperature
                if verbose:
                    print(f"    Temperature: {current_temp:.2f}")
                candidate_draft = self.llm_provider.call(
                    system_prompt=system_prompt,
                    user_prompt=refinement_prompt,
                    temperature=current_temp,
                    max_tokens=self.translator_config.get("max_tokens", 300)
                )

                # Clean and restore citations/quotes
                candidate_draft = clean_generated_text(candidate_draft)
                candidate_draft = candidate_draft.strip()
                candidate_draft = self._restore_citations_and_quotes(candidate_draft, blueprint)

                if not candidate_draft:
                    if verbose:
                        print(f"    ✗ Mutation Failed: Empty candidate draft")
                    continue

                # Evaluate candidate
                candidate_result = critic.evaluate(candidate_draft, blueprint)
                candidate_score = candidate_result["score"]
                candidate_feedback = candidate_result["feedback"]

                if verbose:
                    print(f"    Candidate Score: {candidate_score:.2f}")

                # Hill climbing: only accept improvements
                if candidate_score > best_score:
                    # Improvement: reset temperature, patience, and stagnation counter
                    current_temp = refinement_config.get("initial_temperature",
                                                         refinement_config.get("refinement_temperature", 0.3))
                    patience_counter = 0
                    stagnation_counter = 0
                    best_draft = candidate_draft
                    best_score = candidate_score
                    best_feedback = candidate_feedback
                    if verbose:
                        print(f"    ✓ Hill Climb: Score improved to {best_score:.2f} (temp reset to {current_temp:.2f})")

                    # Check if improved draft meets acceptance criteria
                    if self._check_acceptance(
                        recall_score=candidate_result["recall_score"],
                        precision_score=candidate_result["precision_score"],
                        overall_score=candidate_score,
                        pass_threshold=pass_threshold
                    ):
                        if verbose:
                            print(f"  Evolution: Draft accepted after improvement (recall: {candidate_result['recall_score']:.2f}, precision: {candidate_result['precision_score']:.2f}, score: {candidate_score:.2f})")
                        break
                else:
                    # No improvement: increment patience, stagnation, and increase temperature
                    patience_counter += 1
                    stagnation_counter += 1
                    current_temp = min(current_temp + temperature_increment, max_temperature)
                    if verbose:
                        print(f"    ↻ Stuck at {best_score:.2f}, increasing temperature to {current_temp:.2f} (patience: {patience_counter}/{patience_threshold}, stagnation: {stagnation_counter}/{stagnation_threshold})")

                    # Stagnation Breaker: triggers regardless of score after 3 non-improvements
                    if stagnation_counter >= stagnation_threshold:
                        if verbose:
                            print(f"  DEBUG: Stagnation detected (3 gens at {best_score:.2f}).")

                        if best_score >= 0.85:
                            if verbose:
                                print("  DEBUG: Score is acceptable. Early exit.")
                            break
                        else:
                            if verbose:
                                print("  DEBUG: Score is low. Attempting 'Simplification Pivot'...")
                            # Try one last radical simplification before giving up
                            final_attempt = self._generate_simplification(best_draft, blueprint, author_name, style_dna, rhetorical_type)
                            return (final_attempt, best_score)

                    # Smart Patience: early exit if stuck at good enough score
                    if patience_counter >= patience_threshold and best_score >= patience_min_score:
                        if verbose:
                            print(f"  Evolution converged at {best_score:.2f}. Early exit triggered (patience: {patience_counter})")
                        break

            except Exception as e:
                if verbose:
                    print(f"    ✗ Mutation Failed: Exception during refinement: {e}")
                continue

        if verbose:
            print(f"  Evolution: Final score {best_score:.2f} (improvement: {best_score - initial_score:+.2f})")

        return (best_draft, best_score)

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
        original_citations = set(re.findall(citation_pattern, blueprint.original_text))

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

