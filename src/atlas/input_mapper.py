"""Input Logic Mapper.

Converts unstructured propositions into structured logical dependency graphs
using rhetorical topology analysis.
"""

import json
import re
from typing import List, Optional, Dict, Any

from src.generator.llm_provider import LLMProvider


class InputLogicMapper:
    """Maps propositions to logical dependency graphs using LLM analysis."""

    def __init__(self, llm_provider: LLMProvider):
        """Initialize the Input Logic Mapper.

        Args:
            llm_provider: LLM provider instance for graph generation.
        """
        self.llm_provider = llm_provider

    def _strip_markdown_code_blocks(self, text: str) -> str:
        """Strip markdown code blocks from text.

        Args:
            text: Text that may contain markdown code blocks.

        Returns:
            Text with code blocks removed.
        """
        # Remove ```json ... ``` blocks
        text = re.sub(r'```json\s*\n?(.*?)\n?```', r'\1', text, flags=re.DOTALL)
        # Remove ``` ... ``` blocks (generic)
        text = re.sub(r'```\s*\n?(.*?)\n?```', r'\1', text, flags=re.DOTALL)
        # Remove any remaining ``` markers
        text = text.replace('```', '')
        return text.strip()

    def map_propositions(self, propositions: List[str], prev_paragraph_summary: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """Map propositions to a logical dependency graph.

        Args:
            propositions: List of atomic facts (e.g., ['The phone needs power', 'Without it, it dies']).
            prev_paragraph_summary: Optional summary of the previous paragraph for context-aware role detection.

        Returns:
            Dictionary with 'mermaid', 'description', 'node_map', 'node_count', 'signature', and 'role',
            or None if mapping fails.
        """
        if not propositions:
            raise ValueError("Propositions list cannot be empty")

        system_prompt = (
            "You are a Rhetorical Topologist. Analyze the logical flow. "
            "When describing the graph, use structural terms like 'contrast', 'causality', "
            "'concession', 'enumeration', and 'definition' to match the style index."
        )

        # Format propositions for the prompt
        propositions_text = "\n".join([f"{i}. {prop}" for i, prop in enumerate(propositions)])

        # Build previous context section if available
        prev_context_section = ""
        if prev_paragraph_summary:
            prev_context_section = f"""
**Previous Paragraph Context:**
{prev_paragraph_summary}
"""

        user_prompt = f"""Propositions:
{propositions_text}
{prev_context_section}
Task:
1. **CRITICAL: De-duplicate facts.** If the text repeats an idea (e.g., 'He made it' and 'It was created by him'), extract it ONLY ONCE. Remove redundant propositions before creating the graph.
2. Create a Mermaid graph using IDs P0, P1, P2... corresponding to the list index (after deduplication).
3. Label edges with logic (cause, contrast, support).
4. Write a 1-sentence description of the flow using rhetorical topology terms (causal, contrastive, conditional, enumeration, concession, definition).
5. **Generate Structural Summary (CRITICAL):** Describe the **Rhetorical Structure** of these propositions. Ignore specific names, topics, or entities. Use abstract terms like 'Contrast', 'Definition', 'Attribution', 'Conditional', 'Causality', 'Sequence', 'List'. Keep it under 15 words. Focus on the LOGICAL FORM, not the CONTENT.
   - Example: "A definition of a concept followed by its historical origin."
   - Example: "A set of misconceptions followed by a clarification."
   - Example: "Historical attribution of a creation action to a named agent."
   - Example: "A contrast between a false view and a true statement."
6. Analyze the propositions. What is the primary rhetorical intent? Choose one: `DEFINITION`, `ARGUMENT`, `NARRATIVE`, `INTERROGATIVE`, `IMPERATIVE`.
   - `DEFINITION`: Explaining what something is (e.g., "The phone is a tool.").
   - `ARGUMENT`: Persuading or debating (e.g., "Therefore, we must reject...").
   - `NARRATIVE`: Telling a sequence of events (e.g., "At that time, the army moved...").
   - `INTERROGATIVE`: Asking rhetorical questions.
   - `IMPERATIVE`: Giving commands/directives.
7. Determine the **Logical Signature** of this chunk (Choose ONE based on the relationship between propositions):
   - Does it correct a misconception? -> CONTRAST
   - Does it explain a cause? -> CAUSALITY
   - Does it define a term? -> DEFINITION
   - Does it describe a sequence? -> SEQUENCE
   - Does it state a condition? -> CONDITIONAL
   - Does it enumerate items? -> LIST
   - If classification is ambiguous, default to DEFINITION.
8. Analyze the **Narrative Role** relative to the previous context:
   - Is this an `INTRO` (New Topic)? - Introduces a new subject or argument.
   - Is this an `ELABORATION` (Continuing from previous)? - Expands on or continues the previous paragraph's point.
   - Is this a `CONCLUSION` (Wrapping up)? - Summarizes or concludes the argument.
   - If no previous context is provided, default to `INTRO`.

Output JSON:
{{
  "mermaid": "graph LR; P0 --cause--> P1",
  "description": "A causal chain...",
  "structural_summary": "A causal relationship between two propositions.",
  "intent": "DEFINITION",
  "signature": "CAUSALITY",
  "role": "ELABORATION",
  "node_map": {{ "P0": "text of prop 0", "P1": "text of prop 1" }}
}}"""

        try:
            # Call LLM with require_json=True
            response = self.llm_provider.call(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                model_type="editor",
                require_json=True,
                temperature=0.3,
                max_tokens=500
            )

            # Strip markdown code blocks before parsing
            response = self._strip_markdown_code_blocks(response)

            # Parse JSON
            result = json.loads(response)

            # Validate required fields
            required_fields = ['mermaid', 'description', 'node_map']
            if not all(field in result for field in required_fields):
                print(f"Warning: Missing required fields in LLM response. Expected: {required_fields}")
                return None

            # Generate structural_summary if not provided (fallback)
            if 'structural_summary' not in result or not result.get('structural_summary'):
                # Fallback: create from description, but make it more abstract
                result['structural_summary'] = result.get('description', 'A logical structure.')

            # Validate intent if present, or set default
            valid_intents = ['DEFINITION', 'ARGUMENT', 'NARRATIVE', 'INTERROGATIVE', 'IMPERATIVE']
            if 'intent' not in result:
                # Try to infer from description if intent not provided
                description_lower = result.get('description', '').lower()
                if 'definition' in description_lower or 'define' in description_lower:
                    result['intent'] = 'DEFINITION'
                elif 'narrative' in description_lower or 'sequence' in description_lower or 'event' in description_lower:
                    result['intent'] = 'NARRATIVE'
                elif 'question' in description_lower or 'interrogative' in description_lower:
                    result['intent'] = 'INTERROGATIVE'
                elif 'command' in description_lower or 'imperative' in description_lower or 'directive' in description_lower:
                    result['intent'] = 'IMPERATIVE'
                else:
                    result['intent'] = 'ARGUMENT'  # Default fallback
            elif result.get('intent') not in valid_intents:
                print(f"Warning: Invalid intent '{result.get('intent')}', defaulting to 'ARGUMENT'")
                result['intent'] = 'ARGUMENT'

            # Validate signature if present, or set default
            valid_signatures = ['CONTRAST', 'CAUSALITY', 'DEFINITION', 'SEQUENCE', 'CONDITIONAL', 'LIST']
            if 'signature' not in result:
                # Try to infer from description if signature not provided
                description_lower = result.get('description', '').lower()
                if 'contrast' in description_lower or 'negation' in description_lower or 'correction' in description_lower:
                    result['signature'] = 'CONTRAST'
                elif 'causal' in description_lower or 'cause' in description_lower or 'leads to' in description_lower:
                    result['signature'] = 'CAUSALITY'
                elif 'sequence' in description_lower or 'temporal' in description_lower or 'order' in description_lower:
                    result['signature'] = 'SEQUENCE'
                elif 'conditional' in description_lower or 'if' in description_lower or 'hypothetical' in description_lower:
                    result['signature'] = 'CONDITIONAL'
                elif 'enumeration' in description_lower or 'list' in description_lower or 'grouping' in description_lower:
                    result['signature'] = 'LIST'
                else:
                    result['signature'] = 'DEFINITION'  # Default fallback
            elif result.get('signature') not in valid_signatures:
                print(f"Warning: Invalid signature '{result.get('signature')}', defaulting to 'DEFINITION'")
                result['signature'] = 'DEFINITION'

            # Validate role if present, or set default
            valid_roles = ['INTRO', 'ELABORATION', 'CONCLUSION']
            if 'role' not in result:
                # Default to INTRO if no previous context, ELABORATION if context exists
                if prev_paragraph_summary:
                    result['role'] = 'ELABORATION'
                else:
                    result['role'] = 'INTRO'
            elif result.get('role') not in valid_roles:
                print(f"Warning: Invalid role '{result.get('role')}', defaulting based on context")
                if prev_paragraph_summary:
                    result['role'] = 'ELABORATION'
                else:
                    result['role'] = 'INTRO'

            # Validate types
            if not isinstance(result['mermaid'], str):
                print("Warning: Mermaid field is not a string")
                return None

            if not isinstance(result['description'], str):
                print("Warning: Description field is not a string")
                return None

            if not isinstance(result['node_map'], dict):
                print("Warning: Node map field is not a dictionary")
                return None

            # Validate node_map keys match P0, P1, P2... pattern
            node_keys = list(result['node_map'].keys())
            expected_keys = [f"P{i}" for i in range(len(propositions))]
            if set(node_keys) != set(expected_keys):
                print(f"Warning: Node map keys don't match expected pattern. Got: {node_keys}, Expected: {expected_keys}")
                # Try to normalize - this is lenient but logs a warning
                normalized_map = {}
                for i, prop in enumerate(propositions):
                    key = f"P{i}"
                    if key in result['node_map']:
                        normalized_map[key] = result['node_map'][key]
                    else:
                        # Use the proposition text if key not found
                        normalized_map[key] = prop
                result['node_map'] = normalized_map

            # Add node_count derived from node_map length
            result['node_count'] = len(result['node_map'])

            return result

        except json.JSONDecodeError as e:
            print(f"Warning: Failed to parse JSON response: {e}")
            return None
        except Exception as e:
            print(f"Warning: LLM call failed: {e}")
            return None
