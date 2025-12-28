"""Proposition-to-template mapping using LLM.

Maps propositions to rhetorical template slots for natural argumentative flow.
Allows reordering when logical coherence requires it.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional

from .function_classifier import SentenceFunction
from .template_generator import RhetoricalTemplate, TemplateSlot
from ..utils.logging import get_logger
from ..utils.prompts import load_prompt

logger = get_logger(__name__)


# Function descriptions for the LLM
FUNCTION_DESCRIPTIONS = {
    SentenceFunction.CLAIM: "Makes an assertion or statement of position that can be argued for or against",
    SentenceFunction.EVIDENCE: "Supports a claim with facts, examples, data, or specific instances",
    SentenceFunction.QUESTION: "Poses a question (rhetorical or genuine) to engage the reader",
    SentenceFunction.CONCESSION: "Acknowledges an opposing view, limitation, or valid counterpoint",
    SentenceFunction.CONTRAST: "Sets up opposition or contradiction to a previous statement",
    SentenceFunction.RESOLUTION: "Resolves a tension, answers a question, or draws a conclusion",
    SentenceFunction.ELABORATION: "Expands on or clarifies a previous statement with more detail",
    SentenceFunction.SETUP: "Introduces context, premise, or scenario for what follows",
    SentenceFunction.CONTINUATION: "Continues the narrative flow without strong rhetorical function",
}


@dataclass
class MappedProposition:
    """A proposition mapped to a template slot."""
    proposition: str
    slot_index: int
    function: SentenceFunction
    original_index: int  # Original position in input list
    confidence: float = 1.0
    notes: str = ""  # LLM reasoning


@dataclass
class MappingResult:
    """Result of proposition-to-template mapping."""
    mappings: List[MappedProposition]
    template: RhetoricalTemplate
    reordered: bool = False
    unmapped_propositions: List[str] = field(default_factory=list)
    empty_slots: List[int] = field(default_factory=list)

    def get_ordered_propositions(self) -> List[str]:
        """Get propositions in template order."""
        sorted_mappings = sorted(self.mappings, key=lambda m: m.slot_index)
        return [m.proposition for m in sorted_mappings]

    def get_slot_functions(self) -> List[SentenceFunction]:
        """Get functions in template order."""
        sorted_mappings = sorted(self.mappings, key=lambda m: m.slot_index)
        return [m.function for m in sorted_mappings]


class PropositionMapper:
    """Map propositions to rhetorical template slots using LLM.

    Uses the LLM to analyze proposition content and assign each to the
    best template slot, potentially reordering for coherence.
    """

    def __init__(self, llm_provider=None):
        """Initialize the mapper.

        Args:
            llm_provider: LLM provider for mapping. If None, uses fallback heuristics.
        """
        self.llm_provider = llm_provider

    def map_propositions(
        self,
        propositions: List[str],
        template: RhetoricalTemplate,
        context: Optional[str] = None,
    ) -> MappingResult:
        """Map propositions to template slots.

        Args:
            propositions: List of propositions to map.
            template: Rhetorical template with slots.
            context: Optional context about the paragraph topic.

        Returns:
            MappingResult with mappings and metadata.
        """
        if not propositions:
            return MappingResult(
                mappings=[],
                template=template,
                reordered=False,
            )

        if not template.slots:
            # No template - just return propositions in order as CONTINUATION
            mappings = [
                MappedProposition(
                    proposition=prop,
                    slot_index=i,
                    function=SentenceFunction.CONTINUATION,
                    original_index=i,
                )
                for i, prop in enumerate(propositions)
            ]
            return MappingResult(mappings=mappings, template=template)

        if self.llm_provider is not None:
            return self._map_with_llm(propositions, template, context)
        else:
            return self._map_with_heuristics(propositions, template)

    def _map_with_llm(
        self,
        propositions: List[str],
        template: RhetoricalTemplate,
        context: Optional[str] = None,
    ) -> MappingResult:
        """Map propositions using LLM."""
        user_prompt = self._build_mapping_prompt(propositions, template, context)
        system_prompt = load_prompt("proposition_mapper_system")

        try:
            response = self.llm_provider.call_json(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                temperature=0.3,  # Low temperature for consistent mapping
                max_tokens=1024,
            )

            return self._parse_llm_response(response, propositions, template)

        except Exception as e:
            logger.warning(f"LLM mapping failed, using heuristics: {e}")
            return self._map_with_heuristics(propositions, template)

    def _build_mapping_prompt(
        self,
        propositions: List[str],
        template: RhetoricalTemplate,
        context: Optional[str] = None,
    ) -> str:
        """Build the prompt for LLM mapping."""
        lines = []

        if context:
            lines.append(f"Context: {context}")
            lines.append("")

        lines.append("## Template Slots")
        lines.append("Each slot needs a proposition that fits its rhetorical function:")
        lines.append("")

        for slot in template.slots:
            func = slot.function
            desc = FUNCTION_DESCRIPTIONS.get(func, "General continuation")
            lines.append(f"Slot {slot.index}: {func.value.upper()}")
            lines.append(f"  Description: {desc}")
            lines.append("")

        lines.append("## Propositions to Map")
        for i, prop in enumerate(propositions):
            lines.append(f"Proposition {i}: \"{prop}\"")
        lines.append("")

        lines.append("## Instructions")
        lines.append("1. Analyze each proposition's rhetorical nature")
        lines.append("2. Assign each proposition to the best-fitting slot")
        lines.append("3. You may reorder for logical coherence")
        lines.append("4. If there are more propositions than slots, combine related ones")
        lines.append("5. If there are more slots than propositions, mark slots as empty")
        lines.append("")

        lines.append("## Response Format")
        lines.append("Respond with JSON in this exact format:")
        lines.append("```json")
        lines.append("{")
        lines.append('  "mappings": [')
        lines.append('    {')
        lines.append('      "proposition_index": 0,')
        lines.append('      "slot_index": 0,')
        lines.append('      "reasoning": "This proposition makes a claim about..."')
        lines.append('    }')
        lines.append('  ],')
        lines.append('  "reordered": false,')
        lines.append('  "empty_slots": [],')
        lines.append('  "combined_propositions": []')
        lines.append("}")
        lines.append("```")

        return "\n".join(lines)

    def _parse_llm_response(
        self,
        response: Dict,
        propositions: List[str],
        template: RhetoricalTemplate,
    ) -> MappingResult:
        """Parse LLM response into MappingResult."""
        mappings = []
        unmapped = []
        empty_slots = response.get("empty_slots", [])

        # Track which propositions have been mapped
        mapped_prop_indices = set()

        for mapping_data in response.get("mappings", []):
            prop_idx = mapping_data.get("proposition_index", 0)
            slot_idx = mapping_data.get("slot_index", 0)
            reasoning = mapping_data.get("reasoning", "")

            if prop_idx < 0 or prop_idx >= len(propositions):
                continue
            if slot_idx < 0 or slot_idx >= len(template.slots):
                continue

            mapped_prop_indices.add(prop_idx)

            # Get the function for this slot
            slot = template.slots[slot_idx]

            mappings.append(MappedProposition(
                proposition=propositions[prop_idx],
                slot_index=slot_idx,
                function=slot.function,
                original_index=prop_idx,
                confidence=0.9,
                notes=reasoning,
            ))

        # Check for unmapped propositions
        for i, prop in enumerate(propositions):
            if i not in mapped_prop_indices:
                unmapped.append(prop)

        # Check if reordering happened
        original_order = sorted(mappings, key=lambda m: m.original_index)
        slot_order = sorted(mappings, key=lambda m: m.slot_index)
        reordered = [m.original_index for m in original_order] != [m.original_index for m in slot_order]

        return MappingResult(
            mappings=mappings,
            template=template,
            reordered=reordered,
            unmapped_propositions=unmapped,
            empty_slots=empty_slots,
        )

    def _map_with_heuristics(
        self,
        propositions: List[str],
        template: RhetoricalTemplate,
    ) -> MappingResult:
        """Fallback: map propositions using simple heuristics.

        Uses keyword matching to assign propositions to slots.
        """
        mappings = []
        used_slots = set()
        unmapped = []
        empty_slots = []

        # Simple keyword-based matching
        function_keywords = {
            SentenceFunction.QUESTION: ["?", "why", "how", "what", "when", "where", "who"],
            SentenceFunction.CONTRAST: ["but", "however", "yet", "although", "though", "whereas"],
            SentenceFunction.CONCESSION: ["admittedly", "granted", "of course", "true"],
            SentenceFunction.RESOLUTION: ["therefore", "thus", "hence", "consequently", "in conclusion"],
            SentenceFunction.EVIDENCE: ["for example", "for instance", "studies show", "data", "according to"],
            SentenceFunction.ELABORATION: ["that is", "in other words", "more specifically"],
            SentenceFunction.SETUP: ["consider", "imagine", "suppose", "let us"],
        }

        # First pass: match propositions to slots by keywords
        for i, prop in enumerate(propositions):
            prop_lower = prop.lower()
            best_slot = None
            best_score = -1

            for slot in template.slots:
                if slot.index in used_slots:
                    continue

                func = slot.function
                if func in function_keywords:
                    score = sum(1 for kw in function_keywords[func] if kw in prop_lower)
                    if score > best_score:
                        best_score = score
                        best_slot = slot

            if best_slot and best_score > 0:
                mappings.append(MappedProposition(
                    proposition=prop,
                    slot_index=best_slot.index,
                    function=best_slot.function,
                    original_index=i,
                    confidence=0.5,
                ))
                used_slots.add(best_slot.index)
            else:
                unmapped.append((i, prop))

        # Second pass: assign unmapped propositions to remaining slots in order
        available_slots = [s for s in template.slots if s.index not in used_slots]
        for (i, prop), slot in zip(unmapped[:len(available_slots)], available_slots):
            mappings.append(MappedProposition(
                proposition=prop,
                slot_index=slot.index,
                function=slot.function,
                original_index=i,
                confidence=0.3,
            ))
            used_slots.add(slot.index)

        # Remaining unmapped
        final_unmapped = [prop for (i, prop) in unmapped[len(available_slots):]]

        # Find empty slots
        empty_slots = [s.index for s in template.slots if s.index not in used_slots]

        # Check if order changed
        sorted_by_original = sorted(mappings, key=lambda m: m.original_index)
        sorted_by_slot = sorted(mappings, key=lambda m: m.slot_index)
        reordered = [m.original_index for m in sorted_by_original] != [m.original_index for m in sorted_by_slot]

        return MappingResult(
            mappings=mappings,
            template=template,
            reordered=reordered,
            unmapped_propositions=final_unmapped,
            empty_slots=empty_slots,
        )


def map_propositions_to_template(
    propositions: List[str],
    template: RhetoricalTemplate,
    llm_provider=None,
    context: Optional[str] = None,
) -> MappingResult:
    """Convenience function to map propositions to a template.

    Args:
        propositions: List of propositions to map.
        template: Rhetorical template with slots.
        llm_provider: Optional LLM provider for intelligent mapping.
        context: Optional context about the paragraph topic.

    Returns:
        MappingResult with mappings and metadata.
    """
    mapper = PropositionMapper(llm_provider=llm_provider)
    return mapper.map_propositions(propositions, template, context)
