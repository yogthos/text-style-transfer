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
            error_msg = (
                f"\n{'='*70}\n"
                f"ERROR: Paragraph Atlas not found for author '{self.author}'\n"
                f"{'='*70}\n"
                f"Missing file: {archetypes_path}\n\n"
                f"To fix this, build the paragraph atlas for '{self.author}':\n\n"
                f"  Option 1 (Recommended - sets up everything):\n"
                f"    python3 scripts/init_author.py --author \"{self.author}\" --style-file styles/sample_{self.author.lower()}.txt\n\n"
                f"  Option 2 (Just build the atlas):\n"
                f"    python3 scripts/build_paragraph_atlas.py styles/sample_{self.author.lower()}.txt --author \"{self.author}\"\n\n"
                f"  If you get 'No valid paragraphs found', try:\n"
                f"    python3 scripts/build_paragraph_atlas.py styles/sample_{self.author.lower()}.txt --author \"{self.author}\" --relaxed\n"
                f"{'='*70}\n"
            )
            raise FileNotFoundError(error_msg)

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
            error_msg = (
                f"\n{'='*70}\n"
                f"ERROR: Transition matrix not found for author '{self.author}'\n"
                f"{'='*70}\n"
                f"Missing file: {transition_path}\n\n"
                f"This file should be created when building the paragraph atlas.\n"
                f"To fix this, rebuild the paragraph atlas:\n\n"
                f"    python3 scripts/build_paragraph_atlas.py styles/sample_{self.author.lower()}.txt --author \"{self.author}\"\n\n"
                f"Or use the turnkey script:\n"
                f"    python3 scripts/init_author.py --author \"{self.author}\" --style-file styles/sample_{self.author.lower()}.txt\n"
                f"{'='*70}\n"
            )
            raise FileNotFoundError(error_msg)

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

    def find_style_matched_archetype(self, target_sentence_count: int, tolerance: int = 3) -> Optional[int]:
        """
        Finds the 'nearest neighbor' archetype in terms of sentence count to borrow style metadata.
        Used when grafting author style onto a synthetic structure.

        Args:
            target_sentence_count: Desired sentence count (from synthetic structure)
            tolerance: Maximum acceptable difference in sentence count (default: 3)

        Returns:
            Archetype ID of the best match, or None if no match within tolerance
        """
        best_match_id = None
        best_divergence = float('inf')

        # Iterate through all available archetypes for this author
        for arch_id, arch_data in self.archetypes.items():
            # Get sentence count from archetype data
            arch_sents = arch_data.get('avg_sents', 0)

            divergence = abs(arch_sents - target_sentence_count)

            # We want the closest match within tolerance
            if divergence < best_divergence:
                best_divergence = divergence
                best_match_id = arch_id

        # Only return if it's a reasonable match (within tolerance)
        # Otherwise we risk mapping a 5-sentence style to a 20-sentence input
        if best_divergence <= tolerance:
            return best_match_id

        return None

    def get_author_avg_sentence_length(self) -> float:
        """
        Calculate the average sentence length (words per sentence) across all archetypes.
        Used to determine target density for elastic content mapping.

        Returns:
            Average words per sentence, or 25.0 as default if no archetypes available
        """
        if not self.archetypes:
            return 25.0  # Default to moderate density

        total_avg_len = 0.0
        count = 0

        for arch_id, arch_data in self.archetypes.items():
            avg_len = arch_data.get('avg_len', 0)
            if avg_len > 0:
                total_avg_len += avg_len
                count += 1

        if count > 0:
            return total_avg_len / count

        return 25.0  # Fallback default

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

    def _create_synthetic_archetype(self, input_text: str, target_density: float = 25.0, author_profile: Optional[Dict] = None) -> Dict:
        """
        Creates a temporary archetype based on the input text's structure,
        reshaped to match the target author's sentence density.

        Args:
            input_text: Original input paragraph text
            target_density: Target words per sentence (from author's average)
            author_profile: Optional author style profile containing structural_dna

        Returns:
            Dictionary with structure_map, content_map (grouped sentences), and stats
        """
        try:
            from nltk.tokenize import sent_tokenize
            sentences = sent_tokenize(input_text)
        except (ImportError, Exception):
            # Fallback: split by periods
            sentences = [s.strip() for s in input_text.split('.') if len(s.strip()) > 5]

        structure_map = []
        content_map = []  # Store which sentences go into which slot
        total_words = 0

        # DYNAMIC INFLATION CALCULATION
        # Try to get stats from profile, otherwise fallback to density heuristic
        struct_stats = author_profile.get('structural_dna', {}) if author_profile else {}
        avg_len = struct_stats.get('avg_words_per_sentence')

        if avg_len:
            # Dynamic Formula: Calculate inflation based on author's base density
            neutral_baseline = 15.0  # The input "Logical Beat" length
            raw_ratio = avg_len / neutral_baseline

            # Dampening Function: Pull the ratio closer to 1.0 to avoid runaway expansion
            # e.g., Mao (27.0) / Neutral (15.0) = 1.8
            # inflation = 1.0 + (1.8 - 1.0) * 0.5 = 1.4 (40% expansion)
            inflation_factor = 1.0 + (raw_ratio - 1.0) * 0.5

            # Clamp for safety (between 0.8 and 1.5)
            inflation_factor = max(0.8, min(inflation_factor, 1.5))
        else:
            # Fallback Legacy Logic (if no structural_dna in profile)
            if target_density > 25:
                inflation_factor = 1.20  # +20% budget for verbose authors
            elif target_density < 15:
                inflation_factor = 0.90  # -10% budget for concise authors
            else:
                inflation_factor = 1.10  # +10% standard buffer for style overhead

        # Helper function to avoid code duplication
        def create_slot(word_count, grouping):
            """Create a slot entry with inflation applied."""
            # Store raw length BEFORE inflation
            raw_len = word_count  # NEW: Capture the floor value

            # Apply Inflation
            adjusted_target = int(word_count * inflation_factor)

            # --- THE SAFETY CAP ---
            # Current LLMs struggle to maintain coherence beyond ~60 words.
            # Even if the author writes 100-word sentences, we cap at 60 to ensure success.
            # This prevents infinite retry loops when targets exceed model capabilities.
            MAX_SENTENCE_CEILING = 60

            # Ensure reasonable bounds (don't shrink below 3 or grow beyond ceiling)
            final_target = max(3, min(adjusted_target, MAX_SENTENCE_CEILING))

            # Determine slot type based on FINAL target
            if final_target < 10:
                slot_type = "simple"
            elif final_target < 25:
                slot_type = "moderate"
            else:
                slot_type = "complex"

            return {
                'target_len': final_target,
                'raw_len': raw_len,  # NEW: Store the floor value
                'type': slot_type
            }

        current_group = []
        current_word_count = 0
        complex_streak = 0  # NEW: Track streak of complex sentences

        for i, sent in enumerate(sentences):
            if not sent.strip():
                continue

            sent_len = len(sent.split())
            potential_len = current_word_count + sent_len

            # Elastic Grouping Logic with Burstiness Variation:
            # If target_density is high (>25), introduce variation to create natural rhythm
            # This prevents uniform chunk sizes (e.g., all 29w, 31w) for bursty authors
            if target_density > 25:
                # Add random variation to capacity (±20%)
                import random
                variation_factor = random.uniform(0.8, 1.2)
                adjusted_capacity = target_density * 1.5 * variation_factor
            else:
                adjusted_capacity = target_density * 1.5

            # RHYTHM BREAKER: Check if we need to force a break
            # If we have 2 complex slots in a row, don't let the next one get too big.
            # "Moderate" threshold is roughly 60% of target density.
            force_break = False
            if complex_streak >= 2:
                if potential_len > (target_density * 0.6):
                    force_break = True

            # Modified Merge Logic
            if not force_break and (current_word_count == 0 or potential_len < adjusted_capacity):
                # Merge: Add to current group
                current_group.append(sent)
                current_word_count += sent_len
            else:
                # FLUSH GROUP
                if current_group:
                    # Create slot with inflation applied
                    slot = create_slot(current_word_count, current_group)
                    structure_map.append(slot)
                    content_map.append(" ".join(current_group))
                    total_words += current_word_count

                    # Update Streak Logic
                    if slot['type'] == 'complex':
                        complex_streak += 1
                    else:
                        complex_streak = 0  # Reset streak on simple/moderate slots

                # Start new group with current sentence
                current_group = [sent]
                current_word_count = sent_len

        # Flush remaining group
        if current_group:
            slot = create_slot(current_word_count, current_group)
            structure_map.append(slot)
            content_map.append(" ".join(current_group))
            total_words += current_word_count
            # Note: Final slot streak tracking not needed as this is the last slot

        avg_words = total_words / len(structure_map) if structure_map else 0

        return {
            "id": "synthetic_fallback",
            "structure_map": structure_map,
            "content_map": content_map,  # NEW: Grouped content for direct mapping
            "stats": {
                "sentence_count": len(structure_map),
                "avg_words_per_sent": avg_words,
                "avg_len": avg_words,
                "avg_sents": len(structure_map)
            }
        }

