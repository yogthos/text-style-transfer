"""Rhetorical template generation using Markov models.

Generates paragraph templates with varied sentence functions (claim, question,
contrast, resolution) BEFORE generating content. Propositions are then mapped
to template slots for natural argumentative flow.
"""

from dataclasses import dataclass
from typing import List, Dict, Optional
import random
from collections import Counter

from .function_classifier import SentenceFunction
from ..utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class TemplateSlot:
    """A slot in the rhetorical template."""
    index: int
    function: SentenceFunction
    is_required: bool = True  # False for optional elaboration slots


@dataclass
class RhetoricalTemplate:
    """A rhetorical template for paragraph generation."""
    slots: List[TemplateSlot]

    def __len__(self) -> int:
        return len(self.slots)

    def functions(self) -> List[SentenceFunction]:
        """Get list of functions in order."""
        return [slot.function for slot in self.slots]

    def function_counts(self) -> Dict[str, int]:
        """Count occurrences of each function."""
        counts = Counter()
        for slot in self.slots:
            counts[slot.function.value] += 1
        return dict(counts)


class RhetoricalTemplateGenerator:
    """Generate rhetorical templates using Markov models.

    Uses the function transition probabilities extracted from the author's
    corpus to generate natural rhetorical sequences.
    """

    # Logical constraints: some functions should follow others
    LOGICAL_PRECEDENCE = {
        # resolution should follow question, contrast, or claim
        SentenceFunction.RESOLUTION: {
            SentenceFunction.QUESTION,
            SentenceFunction.CONTRAST,
            SentenceFunction.CLAIM,
            SentenceFunction.CONCESSION,
        },
        # evidence should follow claim
        SentenceFunction.EVIDENCE: {
            SentenceFunction.CLAIM,
            SentenceFunction.RESOLUTION,
        },
    }

    # Functions that work well as endings
    GOOD_ENDINGS = {
        SentenceFunction.RESOLUTION,
        SentenceFunction.CLAIM,
        SentenceFunction.ELABORATION,
        SentenceFunction.CONTINUATION,
    }

    def __init__(
        self,
        function_transitions: Dict[str, Dict[str, float]],
        initial_function_probs: Optional[Dict[str, float]] = None,
        min_variety: int = 2,  # Minimum different function types
        max_same_consecutive: int = 2,  # Max same function in a row
    ):
        """Initialize the template generator.

        Args:
            function_transitions: Markov model P(next|current) from corpus.
            initial_function_probs: Probability of each function starting a paragraph.
            min_variety: Minimum number of different function types to include.
            max_same_consecutive: Maximum same function type in a row.
        """
        self.function_transitions = function_transitions
        self.initial_function_probs = initial_function_probs or {}
        self.min_variety = min_variety
        self.max_same_consecutive = max_same_consecutive

        # Build reverse lookup for string -> enum
        self._func_lookup = {f.value: f for f in SentenceFunction}

    def generate(
        self,
        num_slots: int,
        seed: Optional[int] = None,
        force_variety: bool = True,
    ) -> RhetoricalTemplate:
        """Generate a rhetorical template with the specified number of slots.

        Args:
            num_slots: Number of slots (sentences) to generate.
            seed: Random seed for reproducibility.
            force_variety: If True, ensure minimum variety in functions.

        Returns:
            RhetoricalTemplate with ordered slots.
        """
        if seed is not None:
            random.seed(seed)

        if num_slots <= 0:
            return RhetoricalTemplate(slots=[])

        if num_slots == 1:
            # Single slot - use initial probability or default to CLAIM
            func = self._sample_initial()
            return RhetoricalTemplate(slots=[
                TemplateSlot(index=0, function=func)
            ])

        # Generate sequence using Markov model
        functions = self._generate_sequence(num_slots)

        # Apply variety constraint if needed
        if force_variety and num_slots >= 3:
            functions = self._ensure_variety(functions)

        # Validate logical constraints
        functions = self._validate_logical_flow(functions)

        # Create template slots
        slots = [
            TemplateSlot(index=i, function=func)
            for i, func in enumerate(functions)
        ]

        template = RhetoricalTemplate(slots=slots)

        logger.debug(
            f"Generated template with {num_slots} slots: "
            f"{[f.value for f in functions]}"
        )

        return template

    def _sample_initial(self) -> SentenceFunction:
        """Sample initial function for paragraph start."""
        if not self.initial_function_probs:
            # Default to good starters
            return random.choice([
                SentenceFunction.CLAIM,
                SentenceFunction.SETUP,
            ])

        # Weighted sampling from initial probabilities
        funcs = []
        probs = []
        for func_str, prob in self.initial_function_probs.items():
            if func_str in self._func_lookup:
                funcs.append(self._func_lookup[func_str])
                probs.append(prob)

        if not funcs:
            return SentenceFunction.CLAIM

        # Normalize probabilities
        total = sum(probs)
        if total > 0:
            probs = [p / total for p in probs]
        else:
            probs = [1 / len(probs)] * len(probs)

        return random.choices(funcs, weights=probs, k=1)[0]

    def _sample_next(self, current: SentenceFunction) -> SentenceFunction:
        """Sample next function given current function."""
        current_str = current.value

        if current_str not in self.function_transitions:
            # Fallback: uniform over all functions
            return random.choice(list(SentenceFunction))

        transitions = self.function_transitions[current_str]

        funcs = []
        probs = []
        for func_str, prob in transitions.items():
            if func_str in self._func_lookup:
                funcs.append(self._func_lookup[func_str])
                probs.append(prob)

        if not funcs:
            return random.choice(list(SentenceFunction))

        # Normalize probabilities
        total = sum(probs)
        if total > 0:
            probs = [p / total for p in probs]
        else:
            probs = [1 / len(probs)] * len(probs)

        return random.choices(funcs, weights=probs, k=1)[0]

    def _generate_sequence(self, num_slots: int) -> List[SentenceFunction]:
        """Generate a sequence of functions using Markov model."""
        functions = []

        # Start with initial function
        current = self._sample_initial()
        functions.append(current)

        # Generate rest of sequence
        consecutive_same = 1
        for i in range(1, num_slots):
            next_func = self._sample_next(current)

            # Enforce max consecutive same
            if next_func == current:
                consecutive_same += 1
                if consecutive_same > self.max_same_consecutive:
                    # Force a different function
                    alternatives = [f for f in SentenceFunction if f != current]
                    next_func = self._sample_with_alternatives(current, alternatives)
                    consecutive_same = 1
            else:
                consecutive_same = 1

            functions.append(next_func)
            current = next_func

        return functions

    def _sample_with_alternatives(
        self,
        current: SentenceFunction,
        alternatives: List[SentenceFunction],
    ) -> SentenceFunction:
        """Sample from alternatives using transition probabilities."""
        if not alternatives:
            return current

        current_str = current.value
        if current_str not in self.function_transitions:
            return random.choice(alternatives)

        transitions = self.function_transitions[current_str]

        # Build weighted choices from alternatives
        funcs = []
        probs = []
        for alt in alternatives:
            alt_str = alt.value
            prob = transitions.get(alt_str, 0.1)  # Small default probability
            funcs.append(alt)
            probs.append(prob)

        # Normalize
        total = sum(probs)
        if total > 0:
            probs = [p / total for p in probs]
        else:
            probs = [1 / len(probs)] * len(probs)

        return random.choices(funcs, weights=probs, k=1)[0]

    def _ensure_variety(
        self,
        functions: List[SentenceFunction],
    ) -> List[SentenceFunction]:
        """Ensure minimum variety in function types."""
        unique_funcs = set(functions)

        if len(unique_funcs) >= self.min_variety:
            return functions

        # Need to inject variety
        # Find positions with repeated functions
        result = functions.copy()
        counts = Counter(result)

        # Functions not yet used
        unused = [f for f in SentenceFunction if f not in unique_funcs]

        # Replace some repeated functions with unused ones
        needed = self.min_variety - len(unique_funcs)
        for _ in range(min(needed, len(unused))):
            if not unused:
                break

            # Find most repeated function
            most_common = counts.most_common(1)[0][0]
            if counts[most_common] <= 1:
                break

            # Find a position to replace (not first or last)
            for i in range(1, len(result) - 1):
                if result[i] == most_common:
                    new_func = unused.pop(0)
                    result[i] = new_func
                    counts[most_common] -= 1
                    counts[new_func] = 1
                    break

        return result

    def _would_create_triple(
        self,
        functions: List[SentenceFunction],
        index: int,
        new_func: SentenceFunction,
    ) -> bool:
        """Check if replacing functions[index] with new_func creates 3+ consecutive same."""
        # Check backwards
        count = 1
        for j in range(index - 1, -1, -1):
            if functions[j] == new_func:
                count += 1
            else:
                break
        # Check forwards
        for j in range(index + 1, len(functions)):
            if functions[j] == new_func:
                count += 1
            else:
                break
        return count >= 3

    def _validate_logical_flow(
        self,
        functions: List[SentenceFunction],
    ) -> List[SentenceFunction]:
        """Validate and fix logical flow constraints."""
        if len(functions) <= 1:
            return functions

        result = functions.copy()

        # Check logical precedence constraints
        for i in range(1, len(result)):
            func = result[i]

            if func in self.LOGICAL_PRECEDENCE:
                valid_predecessors = self.LOGICAL_PRECEDENCE[func]

                # Check if any predecessor in the sequence is valid
                has_valid_predecessor = False
                for j in range(i):
                    if result[j] in valid_predecessors:
                        has_valid_predecessor = True
                        break

                if not has_valid_predecessor:
                    # Insert a valid predecessor if possible
                    # For now, just swap with a more neutral function
                    if i > 1:  # Don't change if too early
                        # Try ELABORATION first, then CONTINUATION, then others
                        replacement_candidates = [
                            SentenceFunction.ELABORATION,
                            SentenceFunction.CONTINUATION,
                            SentenceFunction.CLAIM,
                        ]
                        for candidate in replacement_candidates:
                            if not self._would_create_triple(result, i, candidate):
                                result[i] = candidate
                                break

        # Ensure good ending if possible
        if result[-1] not in self.GOOD_ENDINGS and len(result) > 2:
            # Check if RESOLUTION would be valid here
            for good_end in [SentenceFunction.RESOLUTION, SentenceFunction.CLAIM]:
                if good_end in self.GOOD_ENDINGS:
                    # Check if there's a valid predecessor
                    if good_end in self.LOGICAL_PRECEDENCE:
                        valid_preds = self.LOGICAL_PRECEDENCE[good_end]
                        if any(f in valid_preds for f in result[:-1]):
                            result[-1] = good_end
                            break
                    else:
                        result[-1] = good_end
                        break

        return result

    def generate_with_constraints(
        self,
        num_slots: int,
        required_functions: Optional[List[SentenceFunction]] = None,
        start_with: Optional[SentenceFunction] = None,
        end_with: Optional[SentenceFunction] = None,
    ) -> RhetoricalTemplate:
        """Generate template with specific constraints.

        Args:
            num_slots: Number of slots to generate.
            required_functions: Functions that must appear in template.
            start_with: Force specific starting function.
            end_with: Force specific ending function.

        Returns:
            RhetoricalTemplate satisfying constraints.
        """
        if num_slots <= 0:
            return RhetoricalTemplate(slots=[])

        # Generate base template
        template = self.generate(num_slots, force_variety=True)
        functions = [slot.function for slot in template.slots]

        # Apply start constraint
        if start_with is not None:
            functions[0] = start_with

        # Apply end constraint (only if different slot from start)
        if end_with is not None and len(functions) > 1:
            functions[-1] = end_with

        # Apply required functions
        if required_functions:
            existing = set(functions)
            missing = [f for f in required_functions if f not in existing]

            # Insert missing functions (avoiding first and last if constrained)
            start_idx = 1 if start_with else 0
            end_idx = len(functions) - 1 if end_with else len(functions)

            for i, missing_func in enumerate(missing):
                if start_idx + i < end_idx:
                    functions[start_idx + i] = missing_func

        # Validate logical flow
        functions = self._validate_logical_flow(functions)

        # Rebuild template
        slots = [
            TemplateSlot(index=i, function=func)
            for i, func in enumerate(functions)
        ]

        return RhetoricalTemplate(slots=slots)


def generate_template(
    num_slots: int,
    function_transitions: Dict[str, Dict[str, float]],
    initial_function_probs: Optional[Dict[str, float]] = None,
) -> RhetoricalTemplate:
    """Convenience function to generate a rhetorical template."""
    generator = RhetoricalTemplateGenerator(
        function_transitions=function_transitions,
        initial_function_probs=initial_function_probs,
    )
    return generator.generate(num_slots)
