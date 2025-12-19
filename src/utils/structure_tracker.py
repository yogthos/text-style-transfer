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

    def _compare_patterns(self, list1: List[str], list2: List[str]) -> float:
        """Calculate overlap between two lists of attributes.

        Args:
            list1: First list of attributes (e.g., lengths, types).
            list2: Second list of attributes.

        Returns:
            Overlap score from 0.0 (no match) to 1.0 (identical).
        """
        if not list1 or not list2:
            return 0.0

        # Truncate to shorter length to compare alignment
        min_len = min(len(list1), len(list2))
        matches = sum(1 for i in range(min_len) if list1[i] == list2[i])

        return matches / max(len(list1), len(list2))  # Normalize by longest

    def _calculate_rhythm_similarity(self, rhythm_map1: List[Dict], rhythm_map2: List[Dict]) -> float:
        """Calculate similarity between two rhythm maps.

        Args:
            rhythm_map1: First rhythm map to compare.
            rhythm_map2: Second rhythm map to compare.

        Returns:
            Similarity score from 0.0 (completely different) to 1.0 (identical).
        """
        # 1. Sentence count similarity (weight: 0.15)
        count_diff = abs(len(rhythm_map1) - len(rhythm_map2))
        max_count = max(len(rhythm_map1), len(rhythm_map2), 1)
        count_similarity = 1.0 - (count_diff / max_count)

        # 2. Opener match (weight: 0.35) - Most important for detecting repetition
        opener1 = rhythm_map1[0].get('opener') if rhythm_map1 else None
        opener2 = rhythm_map2[0].get('opener') if rhythm_map2 else None
        # Normalize to lowercase strings, handling None values
        opener1_str = (opener1.lower() if opener1 else 'none')
        opener2_str = (opener2.lower() if opener2 else 'none')
        opener_match = 1.0 if opener1_str == opener2_str else 0.0

        # 3. Length pattern similarity (weight: 0.15)
        lengths1 = [r.get('length', 'medium') for r in rhythm_map1]
        lengths2 = [r.get('length', 'medium') for r in rhythm_map2]
        length_similarity = self._compare_patterns(lengths1, lengths2)

        # 4. Type pattern similarity (weight: 0.35) - Also very important
        types1 = [r.get('type', 'standard') for r in rhythm_map1]
        types2 = [r.get('type', 'standard') for r in rhythm_map2]
        type_similarity = self._compare_patterns(types1, types2)

        # Weighted average (Opener and Type are most important)
        similarity = (
            count_similarity * 0.15 +
            opener_match * 0.35 +
            length_similarity * 0.15 +
            type_similarity * 0.35
        )

        return similarity  # 0.0 = different, 1.0 = identical

    def get_diversity_score(self, signature: str, rhythm_map: Optional[List[Dict]] = None) -> float:
        """Score how novel a structure is compared to history.

        Args:
            signature: Structure signature to check.
            rhythm_map: Optional rhythm map for similarity calculation.

        Returns:
            Score from 0.0 (very similar to used) to 1.0 (completely new).
        """
        # 1. Exact Match Check (Fast Fail)
        if signature in self._used_signatures:
            return 0.0  # Exact repetition is bad

        # 2. If no map provided, we can't do deep comparison, assume novel
        if not rhythm_map or not self._used_rhythm_maps:
            return 1.0

        # 3. Similarity Check (The "Fuzzy" Match)
        # We look for the *closest* match in history (max similarity)
        max_similarity = 0.0
        for used_map in self._used_rhythm_maps:
            # Returns 0.0 (different) to 1.0 (identical)
            sim = self._calculate_rhythm_similarity(rhythm_map, used_map)
            if sim > max_similarity:
                max_similarity = sim

        # Diversity is the inverse of the closest match
        # If closest match was 0.9 (very similar), Diversity is 0.1 (poor)
        return 1.0 - max_similarity

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

