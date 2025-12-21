"""Paragraph Refiner for surgical repair of flow and coherence issues.

This module implements a "Holistic Audit -> Repair Plan -> Execution Loop"
approach to fix flow issues in assembly-line generated paragraphs.
"""

import json
import re
from pathlib import Path
from typing import List, Dict, Optional
from src.generator.llm_provider import LLMProvider
from src.utils.parsing import extract_json_from_text


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
        verbose: bool = False
    ) -> str:
        """Refine paragraph using holistic audit and surgical repair.

        Args:
            draft: Draft paragraph to refine
            feedback: Qualitative feedback from critic
            structure_map: Structure map used to build the paragraph
            author_name: Author name for context
            verbose: Enable verbose output

        Returns:
            Refined paragraph text
        """
        if not draft or not draft.strip():
            return draft

        # Step 1: Holistic Audit -> Repair Plan
        if verbose:
            print(f"  Performing holistic audit...")
        repair_plan = self._generate_repair_plan(draft, feedback, structure_map, author_name, verbose)

        if not repair_plan:
            if verbose:
                print(f"  No repair plan generated, returning original draft")
            return draft

        # Step 2: Execution Loop
        if verbose:
            print(f"  Executing {len(repair_plan)} repair instructions...")
        refined = self._execute_repair_plan(draft, repair_plan, author_name, verbose)

        return refined

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
        verbose: bool = False
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
                    verbose
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

    def _execute_sentence_repair(
        self,
        sentence: str,
        instruction: str,
        target_len: Optional[int],
        action: str,
        sentences: List[str],
        sent_index: int,
        author_name: str,
        verbose: bool = False
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

        max_retries = 3
        for attempt in range(max_retries):
            user_prompt = template.format(
                current_sentence=sentence,
                action=action,
                instruction=instruction,
                target_length=f"{target_len} words" if target_len else "Not specified",
                prev_context=prev_context,
                next_sentence=next_sent if action == "merge_with_next" else "",
                author_name=author_name
            )

            try:
                result = self.llm_provider.call(
                    system_prompt=f"You are a surgical editor fixing specific sentences. Maintain {author_name}'s voice.",
                    user_prompt=user_prompt,
                    model_type="editor",
                    require_json=False,
                    temperature=0.5,
                    max_tokens=500
                )

                fixed = result.strip() if result else sentence

                # Validation: Check if target length is met (if specified)
                if target_len and fixed:
                    word_count = len(fixed.split())
                    diff_ratio = abs(word_count - target_len) / target_len if target_len > 0 else 1.0
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
                    return sentence  # Return original on final failure

        # If we exhausted retries, return the last attempt
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

