"""Style Registry for managing author Style DNA profiles.

This module provides a sidecar JSON storage system for author Style DNA,
allowing human-readable and editable style profiles that persist across
ChromaDB index rebuilds.
"""

import json
import os
from datetime import datetime
from typing import Dict, Optional


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
        profile = self.profiles.get(author_name, {})
        return profile.get("style_dna", "")

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

