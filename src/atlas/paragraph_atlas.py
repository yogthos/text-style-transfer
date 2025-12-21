"""Paragraph Atlas loader for statistical archetype generation.

This module loads JSON artifacts (archetypes.json, transition_matrix.json)
and handles Markov chain selection for paragraph generation.
"""

import json
import random
from pathlib import Path
from typing import Dict, Optional, List
import chromadb
from chromadb.config import Settings


class ParagraphAtlas:
    """Loads and manages paragraph archetype data for statistical generation."""

    def __init__(self, atlas_dir: str, author: str):
        """Initialize the paragraph atlas.

        Args:
            atlas_dir: Base directory for paragraph atlas (e.g., "atlas_cache/paragraph_atlas")
            author: Author name (e.g., "Mao" or "mao")
        """
        self.atlas_dir = Path(atlas_dir)
        self.author = author
        # Use lowercase author name for directory lookup
        author_lower = author.lower()
        self.author_dir = self.atlas_dir / author_lower

        # Load archetypes
        archetypes_path = self.author_dir / "archetypes.json"
        if not archetypes_path.exists():
            raise FileNotFoundError(f"Archetypes file not found: {archetypes_path}")

        with open(archetypes_path, 'r') as f:
            archetypes_data = json.load(f)

        # Filter out metadata
        self.archetypes = {
            int(k): v for k, v in archetypes_data.items()
            if k != "_metadata" and isinstance(k, str) and k.isdigit()
        }

        # Load transition matrix
        transition_path = self.author_dir / "transition_matrix.json"
        if not transition_path.exists():
            raise FileNotFoundError(f"Transition matrix file not found: {transition_path}")

        with open(transition_path, 'r') as f:
            transition_data = json.load(f)

        self.transition_matrix = transition_data.get("matrix", {})

        # Connect to ChromaDB for example retrieval
        chroma_dir = self.author_dir / "chroma"
        try:
            self.client = chromadb.PersistentClient(path=str(chroma_dir))
            collection_name = f"paragraph_archetypes_{author.lower()}"
            self.collection = self.client.get_or_create_collection(collection_name)
        except Exception as e:
            print(f"Warning: Could not connect to ChromaDB: {e}")
            self.client = None
            self.collection = None

    def select_next_archetype(self, current_id: Optional[int] = None) -> int:
        """Select next archetype ID based on weighted probabilities from transition matrix.

        Args:
            current_id: Current archetype ID (None for first paragraph)

        Returns:
            Next archetype ID
        """
        # If no current ID, use default archetype or random
        if current_id is None:
            # Try to get default from config, otherwise use archetype 0
            return 0

        # Get transition probabilities for current archetype
        transitions = self.transition_matrix.get(str(current_id), {})

        if not transitions:
            # No transitions found, return default or random archetype
            return 0

        # Convert to list of (target_id, probability) tuples
        targets = [(int(k), v) for k, v in transitions.items()]

        # Weighted random selection
        target_ids = [t[0] for t in targets]
        weights = [t[1] for t in targets]

        selected = random.choices(target_ids, weights=weights, k=1)[0]
        return selected

    def get_archetype_description(self, archetype_id: int) -> Dict:
        """Return stats for the archetype for prompt building.

        Args:
            archetype_id: Archetype ID

        Returns:
            Dictionary with stats (avg_sents, avg_len, burstiness, style, example)
        """
        archetype = self.archetypes.get(archetype_id)
        if not archetype:
            raise ValueError(f"Archetype {archetype_id} not found")

        return {
            "id": archetype.get("id"),
            "avg_sents": archetype.get("avg_sents"),
            "avg_len": archetype.get("avg_len"),
            "burstiness": archetype.get("burstiness"),
            "style": archetype.get("style"),
            "example": archetype.get("example", "")  # Truncated snippet
        }

    def get_example_paragraph(self, archetype_id: int) -> Optional[str]:
        """Retrieve full example paragraph from ChromaDB collection.

        Args:
            archetype_id: Archetype ID to retrieve example for

        Returns:
            Full paragraph text, or None if not found or ChromaDB unavailable
        """
        if not self.collection:
            return None

        try:
            # Use .get() for metadata filtering (not .query() which requires embeddings)
            # Note: archetype_id is stored as integer in metadata, not string
            results = self.collection.get(
                where={"archetype_id": archetype_id},
                limit=1
            )

            if results and results.get("documents") and len(results["documents"]) > 0:
                return results["documents"][0]

            return None
        except Exception as e:
            print(f"Warning: Could not retrieve example from ChromaDB: {e}")
            return None

    def get_structure_map(self, archetype_id: int) -> List[Dict]:
        """Extract exact sentence structure from exemplar paragraph.

        Returns a blueprint of sentence lengths and types for assembly-line construction.

        Args:
            archetype_id: Archetype ID to get structure map for

        Returns:
            List of dicts with structure information:
            [{'target_len': 12, 'type': 'simple', 'position': 0}, ...]
        """
        # Get exemplar paragraph
        exemplar = self.get_example_paragraph(archetype_id)
        if not exemplar:
            # Fallback: use truncated example from archetype description
            archetype = self.archetypes.get(archetype_id)
            if archetype:
                exemplar = archetype.get('example', '')

        if not exemplar:
            # Last resort: return structure based on averages
            archetype = self.archetypes.get(archetype_id)
            if archetype:
                avg_len = archetype.get('avg_len', 20)
                avg_sents = round(archetype.get('avg_sents', 3))
                # Create uniform structure map
                return [
                    {'target_len': round(avg_len), 'type': 'moderate', 'position': i}
                    for i in range(avg_sents)
                ]
            return []

        # Parse exemplar with spaCy
        try:
            import spacy
            try:
                nlp = spacy.load("en_core_web_sm")
            except OSError:
                from spacy.cli import download
                download("en_core_web_sm")
                nlp = spacy.load("en_core_web_sm")

            doc = nlp(exemplar)
            sentences = list(doc.sents)

            if not sentences:
                return []

            structure_map = []
            for i, sent in enumerate(sentences):
                # Count words (tokens, excluding punctuation-only tokens)
                words = [token for token in sent if not token.is_punct]
                word_count = len(words)

                # Classify sentence type
                if word_count < 15:
                    sent_type = 'simple'
                elif word_count <= 25:
                    sent_type = 'moderate'
                else:
                    sent_type = 'complex'

                structure_map.append({
                    'target_len': word_count,
                    'type': sent_type,
                    'position': i
                })

            return structure_map

        except Exception as e:
            print(f"Warning: Could not parse exemplar for structure map: {e}")
            # Fallback to averages
            archetype = self.archetypes.get(archetype_id)
            if archetype:
                avg_len = archetype.get('avg_len', 20)
                avg_sents = round(archetype.get('avg_sents', 3))
                return [
                    {'target_len': round(avg_len), 'type': 'moderate', 'position': i}
                    for i in range(avg_sents)
                ]
            return []

