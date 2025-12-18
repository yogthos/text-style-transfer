"""Style Registry for managing author Style DNA profiles.

This module provides a sidecar JSON storage system for author Style DNA,
allowing human-readable and editable style profiles that persist across
ChromaDB index rebuilds.
"""

import json
import os
from datetime import datetime
from typing import Dict, Optional, Tuple


class StyleRegistry:
    """Manages author Style DNA profiles in a sidecar JSON file."""

    def __init__(self, cache_dir: str):
        """Initialize the Style Registry.

        Args:
            cache_dir: Path to the cache directory (e.g., "atlas_cache/").
                      The registry file will be stored as `author_profiles.json` in this directory.
        """
        self.cache_dir = cache_dir
        # Ensure cache directory exists
        os.makedirs(cache_dir, exist_ok=True)
        self.path = os.path.join(cache_dir, "author_profiles.json")
        self.profiles = self._load()

    def _load(self) -> Dict[str, Dict[str, str]]:
        """Load profiles from JSON file.

        Returns:
            Dictionary mapping author names to profile data.
            Returns empty dict if file doesn't exist.
        """
        if os.path.exists(self.path):
            try:
                with open(self.path, 'r') as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError) as e:
                print(f"    ⚠ Warning: Failed to load author profiles: {e}. Starting with empty registry.")
                return {}
        return {}

    def get_dna(self, author_name: str) -> str:
        """Retrieve Style DNA for an author.

        Args:
            author_name: Name of the author.

        Returns:
            Style DNA string, or empty string if not found.
        """
        # Try exact match first (fast path)
        profile = self.profiles.get(author_name, {})
        if profile:
            return profile.get("style_dna", "")

        # Try case-insensitive match (handles "Mao" vs "mao" etc.)
        author_lower = author_name.lower()
        for key, profile in self.profiles.items():
            if key.lower() == author_lower:
                return profile.get("style_dna", "")

        return ""

    def set_dna(self, author_name: str, dna: str):
        """Store Style DNA for an author.

        Args:
            author_name: Name of the author.
            dna: Style DNA string to store.
        """
        if author_name not in self.profiles:
            self.profiles[author_name] = {}

        self.profiles[author_name]["style_dna"] = dna
        self.profiles[author_name]["last_updated"] = datetime.now().isoformat()

        # Save to file
        try:
            with open(self.path, 'w') as f:
                json.dump(self.profiles, f, indent=2)
        except IOError as e:
            print(f"    ⚠ Warning: Failed to save author profile for {author_name}: {e}")

    def has_dna(self, author_name: str) -> bool:
        """Check if Style DNA exists for an author.

        Args:
            author_name: Name of the author.

        Returns:
            True if DNA exists, False otherwise.
        """
        return bool(self.get_dna(author_name))

    def get_all_profiles(self) -> Dict[str, Dict[str, str]]:
        """Get all stored profiles.

        Returns:
            Dictionary of all author profiles.
        """
        return self.profiles.copy()

    def validate_author(self, author_name: str) -> Tuple[bool, str]:
        """Validate if an author exists in the registry.

        Args:
            author_name: Name of the author to check.

        Returns:
            Tuple of (exists: bool, suggestion: str)
            If author doesn't exist, suggestion contains similar author names.
        """
        # Check exact match
        if author_name in self.profiles:
            return True, ""

        # Check case-insensitive match
        author_lower = author_name.lower()
        matches = [key for key in self.profiles.keys() if key.lower() == author_lower]
        if matches:
            return True, f"Found as '{matches[0]}' (case difference)"

        # Find similar names (simple fuzzy match)
        available = list(self.profiles.keys())
        suggestions = []
        for key in available:
            if author_lower in key.lower() or key.lower() in author_lower:
                suggestions.append(key)

        suggestion_msg = ""
        if suggestions:
            suggestion_msg = f"Did you mean: {', '.join(suggestions[:3])}?"
        elif available:
            suggestion_msg = f"Available authors: {', '.join(sorted(available))}"

        return False, suggestion_msg

