"""Paragraph Refiner for surgical repair of flow and coherence issues.

This module implements a "Holistic Audit -> Repair Plan -> Execution Loop"
approach to fix flow issues in assembly-line generated paragraphs.
"""

import json
import re
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from src.generator.llm_provider import LLMProvider
from src.utils.parsing import extract_json_from_text
from src.utils.text_processing import parse_variants_from_response


class ParagraphRefiner:
    """Refines paragraphs using a repair plan system."""

    def __init__(self, config_path: str = "config.json"):
        """Initialize the paragraph refiner.

        Args:
            config_path: Path to configuration file
        """
        self.config_path = config_path
        with open(config_path, 'r') as f:
            self.config = json.load(f)
        self.llm_provider = LLMProvider(config_path=config_path)

    def refine_via_repair_plan(
        self,
        draft: str,
        feedback: str,
        structure_map: List[Dict],
        author_name: str,
        verbose: bool = False,
        forbidden_phrases: Optional[List[str]] = None
    ) -> Tuple[str, int]:
        """Refine paragraph using holistic audit and surgical repair.

        Args:
            draft: Draft paragraph to refine
            feedback: Qualitative feedback from critic
            structure_map: Structure map used to build the paragraph
            author_name: Author name for context
            verbose: Enable verbose output
            forbidden_phrases: Optional list of phrases to avoid

        Returns:
            Tuple of (refined paragraph text, structure_delta)
            structure_delta: Change in sentence count (+1 for split, -1 for merge/combine, 0 otherwise)
        """
        if not draft or not draft.strip():
            return draft, 0

        # Step 1: Holistic Audit -> Repair Plan
        if verbose:
            print(f"  Performing holistic audit...")
        repair_plan = self._generate_repair_plan(draft, feedback, structure_map, author_name, verbose)

        if not repair_plan:
            if verbose:
                print(f"  No repair plan generated, returning original draft")
            return draft, 0

        # Calculate anticipated structural change
        structure_delta = 0
        for task in repair_plan:
            if not isinstance(task, dict):
                continue
            action = task.get('action', '').lower()
            instruction = task.get('instruction', '').lower()

            # Check for split operations
            if 'split' in action or 'split' in instruction:
                structure_delta += 1
            # Check for combine/merge operations
            elif 'combine' in action or 'merge' in action or 'combine' in instruction or 'merge' in instruction:
                structure_delta -= 1

        if verbose and structure_delta != 0:
            print(f"  Anticipated structure change: {structure_delta:+d} sentences")

        # Step 2: Execution Loop
        if verbose:
            print(f"  Executing {len(repair_plan)} repair instructions...")
        refined = self._execute_repair_plan(draft, repair_plan, author_name, verbose, forbidden_phrases=forbidden_phrases)

        return refined, structure_delta

    def _generate_repair_plan(
        self,
        draft: str,
        feedback: str,
        structure_map: List[Dict],
        author_name: str,
        verbose: bool = False
    ) -> List[Dict]:
        """Generate a repair plan from holistic audit.

        Args:
            draft: Draft paragraph
            feedback: Qualitative feedback
            structure_map: Original blueprint structure map
            author_name: Author name
            verbose: Verbose output

        Returns:
            List of repair instructions: [{"sent_index": 1, "action": "...", "instruction": "..."}, ...]
        """
        prompts_dir = Path(__file__).parent.parent.parent / "prompts"
        try:
            template_path = prompts_dir / "repair_plan_user.md"
            template = template_path.read_text().strip()
        except FileNotFoundError:
            # Fallback template
            template = """# Task: Generate Repair Plan

## Draft Paragraph:
{draft}

## Original Blueprint (Structure Map):
{structure_map_info}

## Feedback:
{feedback}

## Instructions:
The original plan required: {structure_map_info}
The current draft has errors: {feedback}
Create a repair plan to match the original structure map.

Analyze the draft for flow, coherence, and natural transitions. Generate a JSON repair plan with specific instructions for each sentence that needs fixing.

Output a JSON array of repair instructions. Each instruction should have:
- "sent_index": 1-based sentence index (1 = first sentence)
- "action": One of: "shorten", "lengthen", "merge_with_next", "split", "rewrite_connector", "add_transition", "simplify", "expand"
- "instruction": Specific instruction for the LLM (e.g., "Shorten to 25 words by removing adjectives")
- "target_len": Integer (optional), target word count if action is "shorten" or "lengthen"

Output ONLY the JSON array, no other text.
"""

        # Format structure map for prompt
        structure_map_info = self._format_structure_map(structure_map)

        user_prompt = template.format(
            draft=draft,
            feedback=feedback,
            structure_map_info=structure_map_info,
            author_name=author_name
        )

        try:
            response = self.llm_provider.call(
                system_prompt="You are a paragraph editor. Analyze drafts for flow issues and generate surgical repair plans.",
                user_prompt=user_prompt,
                model_type="editor",
                require_json=True,
                temperature=0.3,
                max_tokens=1000
            )

            # Parse JSON response using robust extractor
            plan = None
            if isinstance(response, str):
                plan = extract_json_from_text(response, verbose=True)  # Always verbose for debugging
                if not plan:
                    # Always log the full response for debugging JSON failures
                    print(f"  ⚠ CRITICAL: Failed to extract JSON from repair plan response")
                    print(f"  ⚠ Full response ({len(response)} chars):")
                    print(f"  {'='*60}")
                    print(f"  {response}")
                    print(f"  {'='*60}")
                    print(f"  ⚠ This suggests the LLM did not return valid JSON format")
                    print(f"  ⚠ Check the repair_plan_user.md prompt to ensure JSON format is enforced")
            else:
                plan = response

            # Validate repair plan structure
            if not plan:
                if verbose:
                    print(f"  ⚠ Failed to extract JSON from repair plan response")
                return []

            if not isinstance(plan, list):
                if verbose:
                    print(f"  ⚠ Repair plan is not a list, converting...")
                plan = [plan] if plan else []

            # Validate each instruction has required fields
            validated_plan = []
            for instruction in plan:
                if not isinstance(instruction, dict):
                    if verbose:
                        print(f"  ⚠ Skipping invalid instruction (not a dict): {instruction}")
                    continue
                if 'sent_index' not in instruction or 'action' not in instruction:
                    if verbose:
                        print(f"  ⚠ Skipping instruction missing required fields: {instruction}")
                    continue
                validated_plan.append(instruction)

            if verbose and len(validated_plan) < len(plan):
                print(f"  ⚠ Validated {len(validated_plan)}/{len(plan)} instructions")

            return validated_plan

        except Exception as e:
            if verbose:
                print(f"  ⚠ Error generating repair plan: {e}")
            return []

    def _execute_repair_plan(
        self,
        draft: str,
        repair_plan: List[Dict],
        author_name: str,
        verbose: bool = False,
        forbidden_phrases: Optional[List[str]] = None
    ) -> str:
        """Execute repair plan by applying each instruction sequentially.

        Args:
            draft: Original draft paragraph
            repair_plan: List of repair instructions
            author_name: Author name
            verbose: Verbose output

        Returns:
            Refined paragraph
        """
        # Split draft into sentences
        try:
            import spacy
            try:
                nlp = spacy.load("en_core_web_sm")
            except OSError:
                from spacy.cli import download
                download("en_core_web_sm")
                nlp = spacy.load("en_core_web_sm")

            doc = nlp(draft)
            sentences = [sent.text.strip() for sent in doc.sents]
        except Exception as e:
            if verbose:
                print(f"  ⚠ Error splitting sentences: {e}")
            # Fallback: simple split
            sentences = [s.strip() for s in draft.split('.') if s.strip()]
            sentences = [s + '.' if not s.endswith(('.', '!', '?')) else s for s in sentences]

        if not sentences:
            return draft

        # Apply repairs in order
        modified_sentences = sentences.copy()

        for instruction in repair_plan:
            # Validate instruction structure
            if not isinstance(instruction, dict):
                if verbose:
                    print(f"  ⚠ Skipping invalid instruction (not a dict): {instruction}")
                continue

            # sent_index is 1-based in the plan, convert to 0-based
            sent_index_1based = instruction.get('sent_index')
            if sent_index_1based is None:
                if verbose:
                    print(f"  ⚠ Instruction missing 'sent_index' field: {instruction}")
                continue

            try:
                sent_index_1based = int(sent_index_1based)
            except (ValueError, TypeError):
                if verbose:
                    print(f"  ⚠ Invalid 'sent_index' value (must be int): {instruction.get('sent_index')}")
                continue

            sent_index = sent_index_1based - 1  # Convert to 0-based

            action = instruction.get('action', '')
            instruction_text = instruction.get('instruction', '')
            target_len = instruction.get('target_len')  # Optional target length

            if sent_index < 0 or sent_index >= len(modified_sentences):
                if verbose:
                    print(f"  ⚠ Invalid sentence index {sent_index_1based} (1-based, max={len(modified_sentences)}), skipping")
                continue

            if verbose:
                print(f"    Repairing sentence {sent_index_1based}: {action}" + (f" (target: {target_len} words)" if target_len else ""))

            # Execute repair with validation loop
            try:
                fixed = self._execute_sentence_repair(
                    modified_sentences[sent_index],
                    instruction_text,
                    target_len,
                    action,
                    modified_sentences,
                    sent_index,
                    author_name,
                    verbose,
                    forbidden_phrases=forbidden_phrases
                )

                if fixed:
                    # Update the sentence(s) in place
                    if action == "merge_with_next" and sent_index + 1 < len(modified_sentences):
                        # Merge current and next
                        modified_sentences[sent_index] = fixed
                        modified_sentences.pop(sent_index + 1)
                    else:
                        # Replace single sentence
                        modified_sentences[sent_index] = fixed
            except Exception as e:
                if verbose:
                    print(f"    ⚠ Error applying repair: {e}")

        # Rejoin sentences
        refined = " ".join(modified_sentences)
        return refined

    def _generate_repair_variants(
        self,
        sentence: str,
        instruction: str,
        action: str,
        target_len: Optional[int],
        author_name: str,
        n: int = 5,
        prev_context: str = "",
        next_sent: str = "",
        forbidden_phrases: Optional[List[str]] = None,
        verbose: bool = False
    ) -> List[str]:
        """Generate N variants using delimiter-based format.

        Args:
            sentence: Current sentence to repair
            instruction: Specific repair instruction
            action: Action type
            target_len: Optional target word count
            author_name: Author name
            n: Number of variants to generate
            prev_context: Previous context
            next_sent: Next sentence (for merging)
            forbidden_phrases: Optional list of phrases to avoid
            verbose: Verbose output

        Returns:
            List of variant strings
        """
        # Build prompt (similar to current user_content but request N variants)
        user_content = f"""Generate {n} different variants of the fixed sentence.
Strictly follow the constraints below.
**Output Format:** Output each variant on a new line starting with 'VAR:'. Do not number them.

## Target Sentence:
"{sentence}"

## Action:
{action}

## Instruction:
{instruction}
"""

        # 1. Special Handling for "Split" (must come before length constraint)
        is_split = action.lower() == "split"
        if is_split:
            # Clarify that target applies to EACH resulting sentence, not total
            if target_len:
                user_content += f"\n**SPLIT TARGET: Split into two sentences of approximately {target_len} words EACH.**\n"
            else:
                user_content += "\n**SPLIT TARGET: Split into two roughly equal sentences.**\n"
            # STRICT DELIMITER REQUIREMENT
            user_content += "**CRITICAL: You MUST place '|||' between the two sentences. This is mandatory.**\n"
            user_content += "Example: 'First sentence here.|||Second sentence here.'\n"
        else:
            # Hard Length Constraint - only for non-split operations
            if target_len:
                user_content += f"\n**HARD CONSTRAINT: Output must be close to {target_len} words. This is mandatory.**\n"
            # CRITICAL: Prevent accidental splitting
            user_content += "\n**CONSTRAINT: Output exactly ONE sentence. Do not split.**\n"

        # 2. Negative Constraints (The Ban List)
        if forbidden_phrases:
            phrases_str = ", ".join([f"'{p}'" for p in forbidden_phrases])
            user_content += f"\n**FORBIDDEN PHRASES (Do NOT use these): {phrases_str}**\n"
            user_content += "You must find synonyms or alternative phrasing to avoid these exact phrases.\n"

        # Add context
        if prev_context:
            user_content += f"\n## Previous Context:\n{prev_context}\n"
        if next_sent and action == "merge_with_next":
            user_content += f"\n## Next Sentence (for merging):\n{next_sent}\n"

        user_content += f"\n## Instructions:\nApply the repair instruction EXACTLY. Maintain {author_name}'s voice.\n"
        user_content += "Output ONLY the fixed sentence(s), no explanations.\n"

        try:
            response = self.llm_provider.call(
                system_prompt=f"You are a surgical editor. Generate {n} variants. Output each on a new line with 'VAR:' prefix.",
                user_prompt=user_content,
                model_type="editor",
                require_json=False,
                temperature=0.5,
                max_tokens=500
            )

            # Parse variants using shared utility
            variants = parse_variants_from_response(response, verbose=verbose)

            if variants:
                return variants
            else:
                # Final fallback: if no variants found, return original sentence as single variant
                if verbose:
                    print(f"      ⚠ No variants parsed from response, using fallback")
                return [sentence]

        except Exception as e:
            if verbose:
                print(f"      ⚠ Error generating variants: {e}, using fallback")
            # Fallback to original sentence
            return [sentence]

    def _select_best_variant(
        self,
        variants: List[str],
        action: str,
        target_len: Optional[int],
        verbose: bool = False
    ) -> Optional[str]:
        """Select best variant based on format compliance and length proximity.

        Args:
            variants: List of variant strings
            action: Action type
            target_len: Optional target word count
            verbose: Verbose output

        Returns:
            Best variant string, or None if no valid variants
        """
        if not variants:
            return None

        is_split = "split" in action.lower()

        # Priority 1: Filter by format compliance
        compliant_variants = []
        for v in variants:
            if is_split:
                if "|||" in v:
                    compliant_variants.append(v)
            else:
                # Basic validation: non-empty, has sentence structure
                if v and v.strip() and (v.endswith(('.', '!', '?')) or len(v.split()) > 0):
                    compliant_variants.append(v)

        if verbose:
            print(f"      Format-compliant variants: {len(compliant_variants)}/{len(variants)}")

        # Priority 2: Select by length proximity
        if compliant_variants:
            if target_len:
                best = min(compliant_variants,
                          key=lambda v: abs(len(v.split()) - target_len))
            else:
                best = compliant_variants[0]  # First compliant if no target
            return best

        # Fallback: Best effort from all variants
        if target_len:
            return min(variants, key=lambda v: abs(len(v.split()) - target_len))
        return variants[0] if variants else None

    def _execute_sentence_repair(
        self,
        sentence: str,
        instruction: str,
        target_len: Optional[int],
        action: str,
        sentences: List[str],
        sent_index: int,
        author_name: str,
        verbose: bool = False,
        forbidden_phrases: Optional[List[str]] = None
    ) -> Optional[str]:
        """Execute repair on a single sentence with validation loop.

        Args:
            sentence: Current sentence to repair
            instruction: Specific repair instruction
            target_len: Optional target word count
            action: Action type
            sentences: List of all sentences (for context)
            sent_index: Index of current sentence
            author_name: Author name
            verbose: Verbose output

        Returns:
            Fixed sentence text, or None if error
        """
        prev_context = " ".join(sentences[:sent_index]) if sent_index > 0 else ""
        next_sent = sentences[sent_index + 1] if sent_index + 1 < len(sentences) else ""

        prompts_dir = Path(__file__).parent.parent.parent / "prompts"
        try:
            template_path = prompts_dir / "repair_executor_user.md"
            template = template_path.read_text().strip()
        except FileNotFoundError:
            # Fallback template
            template = """# Task: Execute Repair Instruction

## Current Sentence:
"{current_sentence}"

## Action:
{action}

## Instruction:
{instruction}

## Target Length (if specified):
{target_length}

## Previous Context:
{prev_context}

## Next Sentence (if merging):
{next_sentence}

## Instructions:
Apply the repair instruction to the current sentence. Maintain the author's voice ({author_name}).
Output only the fixed sentence(s), no explanations.
"""

        # Get max_retries from config (critic section)
        critic_config = self.config.get("critic", {})
        max_retries = critic_config.get("max_retries", 3)

        # Get number of variants per attempt from config
        refinement_config = self.config.get("refinement", {})
        n_variants = refinement_config.get("repair_variants_per_attempt", 5)

        best_attempt = None
        best_length_diff = float('inf')

        for attempt in range(max_retries):
            # 1. Generate batch of variants
            variants = self._generate_repair_variants(
                sentence, instruction, action, target_len, author_name,
                n=n_variants, prev_context=prev_context, next_sent=next_sent,
                forbidden_phrases=forbidden_phrases, verbose=verbose
            )

            if verbose:
                print(f"      Generated {len(variants)} variants for attempt {attempt + 1}")

            # 2. Select best candidate
            best_candidate = self._select_best_variant(variants, action, target_len, verbose=verbose)

            if not best_candidate:
                if verbose:
                    print(f"      ⚠ No valid variant found, using first variant as fallback")
                best_candidate = variants[0] if variants else sentence

            if verbose and target_len:
                word_count = len(best_candidate.split())
                print(f"      Selected best variant: {word_count} words (target: {target_len}) from {len(variants)} candidates")

            # 3. Validate candidate (existing validation logic)
            fixed = best_candidate
            is_split = action.lower() == "split"

            try:
                # STRICT VALIDATION: Check Split Compliance
                if is_split:
                    if "|||" not in fixed:
                        if verbose:
                            print(f"      ⚠ Attempt {attempt + 1} missing '|||' separator, retrying...")
                        # Track this as best attempt if it's the first valid attempt (even without |||)
                        if best_attempt is None:
                            best_attempt = fixed
                            best_length_diff = abs(len(fixed.split()) - (target_len * 2 if target_len else 0))
                        continue  # Fail immediately, force retry

                # Post-process for Split: Handle ||| separator (prevent double punctuation)
                if is_split and "|||" in fixed:
                    parts = fixed.split("|||")
                    if len(parts) >= 2:
                        part1 = parts[0].strip().rstrip('.')  # Remove existing dot to prevent "Sentence A.. Sentence B"
                        part2 = parts[1].strip()

                        # Capitalize Part 2 if needed
                        if part2 and part2[0].islower():
                            part2 = part2[0].upper() + part2[1:]

                        fixed = f"{part1}. {part2}"
                    else:
                        # Fallback: try to split on period
                        if verbose:
                            print(f"      ⚠ Split format invalid, attempting fallback")

                # Validation: Check if target length is met (if specified)
                if target_len and fixed:
                    word_count = len(fixed.split())
                    length_diff = abs(word_count - target_len)

                    # Track best attempt (closest to target)
                    if length_diff < best_length_diff:
                        best_attempt = fixed
                        best_length_diff = length_diff

                    diff_ratio = length_diff / target_len if target_len > 0 else 1.0
                    tolerance = 0.20  # 20% tolerance for repairs

                    if diff_ratio <= tolerance:
                        if verbose and attempt > 0:
                            print(f"      ✓ Sentence repair validated (attempt {attempt + 1}): {word_count} words")
                        return fixed
                    else:
                        if verbose:
                            print(f"      ⚠ Attempt {attempt + 1}: {word_count} words (target: {target_len}), retrying...")
                        sentence = fixed  # Use this as base for next attempt
                        continue
                else:
                    # No target length, accept the result
                    return fixed

            except Exception as e:
                if verbose:
                    print(f"      ⚠ Error executing repair (attempt {attempt + 1}): {e}")
                if attempt == max_retries - 1:
                    # On final failure, return best attempt if available, otherwise original
                    if best_attempt and best_length_diff < float('inf'):
                        if verbose:
                            print(f"      ⚠ Using best attempt after error: {len(best_attempt.split())} words (target: {target_len})")
                        return best_attempt
                    return sentence  # Return original on final failure

        # If we exhausted retries, check for heuristic backup (especially for splits)
        # Check if we need to force a split: action was split but no ||| separator found
        needs_forced_split = is_split and ("|||" not in (best_attempt or sentence))
        if needs_forced_split:
            # Heuristic Backup: Programmatic split if LLM failed to split (The Sledgehammer)
            if verbose:
                print(f"      ⚠ LLM refused to split after {max_retries} attempts. Applying programmatic sledgehammer...")

            # Strategy: Find the strongest punctuation mark near the middle
            import re
            text = (best_attempt or sentence).strip()
            mid_point = len(text) // 2

            # Look for semicolon or comma, prioritizing proximity to middle
            # Find all punctuation indices
            punct_matches = list(re.finditer(r'[,;](?=\s+\w)', text))

            if punct_matches:
                # Pick the one closest to the middle
                best_punct = min(punct_matches, key=lambda m: abs(m.start() - mid_point))

                # Perform the cut
                split_idx = best_punct.start()
                part1 = text[:split_idx].strip()
                # Remove the punctuation mark and replace with period
                part2 = text[split_idx + 1:].strip()

                # Capitalize Part 2
                if part2:
                    part2 = part2[0].upper() + part2[1:] if part2[0].islower() else part2

                # Ensure proper punctuation
                if not part1.endswith(('.', '!', '?')):
                    part1 = part1.rstrip(',;') + '.'
                if not part2.endswith(('.', '!', '?')):
                    part2 = part2.rstrip(',;') + '.'

                # Create split format with ||| so post-processing handles it
                programmatic_split = f"{part1} ||| {part2}"

                # Process it through the same post-processing logic
                if "|||" in programmatic_split:
                    parts = programmatic_split.split("|||")
                    if len(parts) >= 2:
                        part1_clean = parts[0].strip().rstrip('.')
                        part2_clean = parts[1].strip()
                        if part2_clean and part2_clean[0].islower():
                            part2_clean = part2_clean[0].upper() + part2_clean[1:]
                        best_attempt = f"{part1_clean}. {part2_clean}"
                        if verbose:
                            print(f"      ✓ Programmatic split successful: {len(part1_clean.split())} + {len(part2_clean.split())} words")
                        return best_attempt
            else:
                # Desperation: Split at the nearest word boundary to the middle
                # (Rare fallback)
                split_idx = text.rfind(' ', 0, mid_point + 10)
                if split_idx != -1:
                    part1 = text[:split_idx].strip()
                    part2 = text[split_idx:].strip()
                    if part2:
                        part2 = part2[0].upper() + part2[1:] if part2[0].islower() else part2
                    if not part1.endswith(('.', '!', '?')):
                        part1 = part1 + '.'
                    if not part2.endswith(('.', '!', '?')):
                        part2 = part2 + '.'
                    best_attempt = f"{part1} {part2}"
                    if verbose:
                        print(f"      ✓ Programmatic split at word boundary: {len(part1.split())} + {len(part2.split())} words")
                    return best_attempt

        # If we exhausted retries, return best attempt (or original if none)
        if best_attempt and best_length_diff < float('inf'):
            if verbose:
                print(f"      ⚠ Max retries reached, using best attempt: {len(best_attempt.split())} words (target: {target_len})")
            return best_attempt

        # Final fallback: return original
        return sentence

    def _format_structure_map(self, structure_map: List[Dict]) -> str:
        """Format structure map for display in prompts.

        Args:
            structure_map: Structure map list

        Returns:
            Formatted string describing the structure map
        """
        if not structure_map:
            return "No structure map available"

        parts = []
        for i, slot in enumerate(structure_map, 1):
            target_len = slot.get('target_len', 0)
            slot_type = slot.get('type', 'moderate')
            parts.append(f"Sentence {i}: {target_len} words ({slot_type})")

        return "\n".join(parts)

