"""Structure tracker for paragraph diversity management.

This module tracks paragraph structures (rhythm maps) across a document
to prevent repetitive patterns and ensure structural diversity.
"""

from typing import List, Dict, Set, Optional


class StructureTracker:
    """Tracks used paragraph structures to ensure diversity.

    Prevents repetitive paragraph patterns by tracking:
    - Structure signatures (normalized rhythm patterns)
    - Opener usage frequency
    - Rhythm maps for similarity comparison
    """

    def __init__(self):
        """Initialize empty tracking."""
        self._used_signatures: Set[str] = set()
        self._used_rhythm_maps: List[List[Dict]] = []
        self._opener_counts: Dict[str, int] = {}
        self._total_paragraphs: int = 0

    def add_structure(self, signature: str, rhythm_map: List[Dict]) -> None:
        """Record a used structure.

        Args:
            signature: Normalized structure signature string.
            rhythm_map: List of sentence specifications from extract_paragraph_rhythm().
        """
        self._used_signatures.add(signature)
        self._used_rhythm_maps.append(rhythm_map)
        self._total_paragraphs += 1

        # Track opener
        if rhythm_map and rhythm_map[0].get('opener'):
            opener = rhythm_map[0]['opener'].lower()
            self._opener_counts[opener] = self._opener_counts.get(opener, 0) + 1

    def get_diversity_score(self, signature: str) -> float:
        """Score how novel a structure is.

        Args:
            signature: Structure signature to check.

        Returns:
            Score from 0.0 (identical to used) to 1.0 (completely new).
        """
        if signature not in self._used_signatures:
            return 1.0

        # Exact match = 0.0 (completely repetitive)
        # TODO: Add similarity calculation for partial matches in future
        return 0.0

    def get_opener_penalty(self, opener_word: str, threshold: float = 0.3) -> float:
        """Get penalty multiplier for overused opener.

        Args:
            opener_word: Opener word to check (case-insensitive).
            threshold: Frequency threshold above which penalty applies (default 0.3 = 30%).

        Returns:
            Penalty multiplier from 0.0 (max penalty) to 1.0 (no penalty).
        """
        if not opener_word or self._total_paragraphs == 0:
            return 1.0

        opener = opener_word.lower()
        count = self._opener_counts.get(opener, 0)
        frequency = count / max(self._total_paragraphs, 1)

        if frequency <= threshold:
            return 1.0

        # Penalty increases as frequency exceeds threshold
        excess = frequency - threshold
        # Scale penalty: at 50% usage, penalty = 0.6; at 70% usage, penalty = 0.2
        penalty = max(0.0, 1.0 - (excess * 2.0))
        return penalty

    def record_opener(self, opener_word: str) -> None:
        """Record opener usage (alternative to add_structure for opener-only tracking).

        Args:
            opener_word: Opener word used (case-insensitive).
        """
        if opener_word:
            opener = opener_word.lower()
            self._opener_counts[opener] = self._opener_counts.get(opener, 0) + 1

    def reset(self) -> None:
        """Clear tracking for new document."""
        self._used_signatures.clear()
        self._used_rhythm_maps.clear()
        self._opener_counts.clear()
        self._total_paragraphs = 0

    def get_used_signatures(self) -> Set[str]:
        """Get set of used structure signatures (for debugging).

        Returns:
            Set of signature strings.
        """
        return self._used_signatures.copy()

    def get_opener_frequency(self, opener_word: str) -> float:
        """Get frequency of an opener (for debugging).

        Args:
            opener_word: Opener word to check.

        Returns:
            Frequency as float (0.0 to 1.0).
        """
        if not opener_word or self._total_paragraphs == 0:
            return 0.0

        opener = opener_word.lower()
        count = self._opener_counts.get(opener, 0)
        return count / max(self._total_paragraphs, 1)

