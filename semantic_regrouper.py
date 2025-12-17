"""
Semantic Regrouper Module

Regroups semantic content (claims, relationships, entities) to match
the cadence patterns from sample text. Uses:
- Topic clustering to group related claims
- Relationship-aware grouping to preserve logical flow
- Cadence matching to ensure target paragraph structures
"""

import spacy
import numpy as np
from typing import Dict, List, Optional, Set, Tuple
from dataclasses import dataclass, field
from collections import defaultdict, Counter

from semantic_extractor import SemanticContent, Claim, Relationship, Entity
from cadence_analyzer import CadenceProfile, ParagraphTarget, CadenceAnalyzer


@dataclass
class SemanticChunk:
    """A regrouped chunk of semantic content for a paragraph."""
    claims: List[Claim]
    relationships: List[Relationship]
    entities: List[Entity]
    target_length: int  # Target word count from cadence
    target_sentence_count: int
    position: float  # Position in document (0.0-1.0)
    role: str  # section_opener, body, closer
    key_concepts: List[str] = field(default_factory=list)


class SemanticRegrouper:
    """
    Regroups semantic content to match sample text cadence.

    Strategy:
    1. Cluster claims by topic similarity
    2. Group claims with logical relationships
    3. Match chunks to cadence targets
    4. Validate all content is included
    """

    def __init__(self, semantic_content: SemanticContent,
                 cadence_profile: CadenceProfile,
                 cadence_analyzer: CadenceAnalyzer,
                 min_chunk_claims: int = 1,
                 max_chunk_claims: int = 5,
                 semantic_similarity_threshold: float = 0.6,
                 force_one_claim_per_chunk: bool = False):
        """
        Initialize semantic regrouper.

        Args:
            semantic_content: Extracted semantic content to regroup
            cadence_profile: Cadence profile from sample text
            cadence_analyzer: CadenceAnalyzer instance for getting targets
            min_chunk_claims: Minimum claims per chunk
            max_chunk_claims: Maximum claims per chunk
            semantic_similarity_threshold: Threshold for topic clustering
            force_one_claim_per_chunk: If True, override max_chunk_claims to 1
        """
        self.semantic_content = semantic_content
        self.cadence_profile = cadence_profile
        self.cadence_analyzer = cadence_analyzer

        # Override if forcing one claim per chunk
        if force_one_claim_per_chunk:
            self.min_chunk_claims = 1
            self.max_chunk_claims = 1
        else:
            self.min_chunk_claims = min_chunk_claims
            self.max_chunk_claims = max_chunk_claims

        self.semantic_similarity_threshold = semantic_similarity_threshold

        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            import subprocess
            subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"])
            self.nlp = spacy.load("en_core_web_sm")

        # Initialize sentence transformers for semantic similarity (optional)
        self.embedder = None
        try:
            from sentence_transformers import SentenceTransformer
            self.embedder = SentenceTransformer('all-MiniLM-L6-v2')
        except ImportError:
            pass

    def regroup_to_cadence(self) -> List[SemanticChunk]:
        """
        Regroup semantic content to match cadence.

        Returns:
            List of SemanticChunk objects ready for synthesis
        """
        if not self.semantic_content.claims:
            # Return empty chunks if no claims
            return []

        # Step 1: Cluster claims by topic
        topic_clusters = self._cluster_claims_by_topic()

        # Step 2: Group by relationships
        relationship_groups = self._group_by_relationships(topic_clusters)

        # Step 3: Create initial chunks
        initial_chunks = self._create_initial_chunks(relationship_groups)

        # Step 4: Match to cadence (adjust sizes, assign positions/roles)
        final_chunks = self._match_to_cadence(initial_chunks)

        # Step 5: Validate
        self._validate_regrouping(final_chunks)

        return final_chunks

    def _cluster_claims_by_topic(self) -> List[List[Claim]]:
        """
        Cluster claims by topic similarity using word embeddings.

        Returns:
            List of claim clusters, each cluster contains related claims
        """
        if not self.semantic_content.claims:
            return []

        claims = self.semantic_content.claims

        # Extract key concepts from each claim
        claim_concepts = []
        for claim in claims:
            # Combine subject, predicate, objects for concept extraction
            concept_text = f"{claim.subject} {claim.predicate} {' '.join(claim.objects)}"
            claim_concepts.append(concept_text)

        # Use embeddings if available, otherwise use simple word overlap
        if self.embedder:
            # Get embeddings
            embeddings = self.embedder.encode(claim_concepts)

            # Simple clustering: group claims with high similarity
            clusters = []
            used = set()

            for i, claim in enumerate(claims):
                if i in used:
                    continue

                cluster = [claim]
                used.add(i)

                for j, other_claim in enumerate(claims):
                    if j in used or j == i:
                        continue

                    # Calculate cosine similarity
                    similarity = np.dot(embeddings[i], embeddings[j]) / (
                        np.linalg.norm(embeddings[i]) * np.linalg.norm(embeddings[j])
                    )

                    if similarity >= self.semantic_similarity_threshold:
                        cluster.append(other_claim)
                        used.add(j)

                clusters.append(cluster)
        else:
            # Fallback: use word overlap
            clusters = []
            used = set()

            for i, claim in enumerate(claims):
                if i in used:
                    continue

                cluster = [claim]
                used.add(i)

                # Extract words from claim
                claim_words = set(
                    self.nlp(claim_concepts[i].lower()).text.split()
                )

                for j, other_claim in enumerate(claims):
                    if j in used or j == i:
                        continue

                    other_words = set(
                        self.nlp(claim_concepts[j].lower()).text.split()
                    )

                    # Calculate Jaccard similarity
                    intersection = len(claim_words & other_words)
                    union = len(claim_words | other_words)
                    similarity = intersection / union if union > 0 else 0

                    if similarity >= self.semantic_similarity_threshold:
                        cluster.append(other_claim)
                        used.add(j)

                clusters.append(cluster)

        return clusters

    def _group_by_relationships(self, topic_clusters: List[List[Claim]]) -> List[List[Claim]]:
        """
        Group claims that have logical relationships.

        Preserves cause-effect, contrast, sequence relationships.
        """
        if not self.semantic_content.relationships:
            return topic_clusters

        # Build claim index (use claim text as key since Claim objects aren't hashable)
        claim_to_cluster = {}
        for cluster_idx, cluster in enumerate(topic_clusters):
            for claim in cluster:
                claim_to_cluster[claim.text] = cluster_idx

        # Find relationships between clusters
        cluster_relationships = defaultdict(set)
        for rel in self.semantic_content.relationships:
            # Find clusters containing source and target
            source_cluster = None
            target_cluster = None

            for cluster_idx, cluster in enumerate(topic_clusters):
                for claim in cluster:
                    if rel.source.lower() in claim.text.lower():
                        source_cluster = cluster_idx
                    if rel.target.lower() in claim.text.lower():
                        target_cluster = cluster_idx

            if source_cluster is not None and target_cluster is not None:
                if source_cluster != target_cluster:
                    cluster_relationships[source_cluster].add(target_cluster)

        # Merge clusters with strong relationships
        merged = [False] * len(topic_clusters)
        merged_clusters = []

        for i, cluster in enumerate(topic_clusters):
            if merged[i]:
                continue

            # Start with this cluster
            new_cluster = list(cluster)
            merged[i] = True

            # Add related clusters
            related = cluster_relationships.get(i, set())
            for related_idx in related:
                if not merged[related_idx]:
                    new_cluster.extend(topic_clusters[related_idx])
                    merged[related_idx] = True

            merged_clusters.append(new_cluster)

        return merged_clusters

    def _create_initial_chunks(self, claim_groups: List[List[Claim]]) -> List[SemanticChunk]:
        """
        Create initial semantic chunks from claim groups.

        Splits large groups and merges small ones.
        """
        chunks = []

        for group in claim_groups:
            # Split large groups
            if len(group) > self.max_chunk_claims:
                # Split into multiple chunks
                for i in range(0, len(group), self.max_chunk_claims):
                    chunk_claims = group[i:i + self.max_chunk_claims]
                    chunks.append(self._create_chunk_from_claims(chunk_claims))
            elif len(group) >= self.min_chunk_claims:
                # Use group as-is
                chunks.append(self._create_chunk_from_claims(group))
            else:
                # Small group - will merge later if needed
                chunks.append(self._create_chunk_from_claims(group))

        return chunks

    def _create_chunk_from_claims(self, claims: List[Claim]) -> SemanticChunk:
        """Create a semantic chunk from a list of claims."""
        # Find related relationships
        relationships = []
        for rel in self.semantic_content.relationships:
            for claim in claims:
                if rel.source.lower() in claim.text.lower() or \
                   rel.target.lower() in claim.text.lower():
                    relationships.append(rel)
                    break

        # Find related entities
        entities = []
        for entity in self.semantic_content.entities:
            for claim in claims:
                if entity.text.lower() in claim.text.lower():
                    entities.append(entity)
                    break

        # Extract key concepts
        key_concepts = []
        for claim in claims:
            key_concepts.extend([claim.subject, claim.predicate])
            key_concepts.extend(claim.objects)
        key_concepts = list(set([c.lower() for c in key_concepts if c.strip()]))

        return SemanticChunk(
            claims=claims,
            relationships=relationships,
            entities=entities,
            target_length=0,  # Will be set in match_to_cadence
            target_sentence_count=0,
            position=0.0,
            role='body',
            key_concepts=key_concepts[:10]  # Limit to top 10
        )

    def _match_to_cadence(self, chunks: List[SemanticChunk]) -> List[SemanticChunk]:
        """
        Match chunks to cadence targets.

        Assigns positions, roles, and target lengths to each chunk.
        """
        if not chunks:
            return []

        # Determine target number of paragraphs based on cadence
        target_para_count = max(
            self.cadence_profile.paragraph_count,
            len(chunks)  # At least one paragraph per chunk
        )

        # Merge small chunks if needed
        merged_chunks = self._merge_small_chunks(chunks)

        # Split large chunks if needed
        split_chunks = self._split_large_chunks(merged_chunks)

        # Assign positions and roles
        final_chunks = []
        for i, chunk in enumerate(split_chunks):
            position = i / max(len(split_chunks) - 1, 1)

            # Determine role
            if i == 0:
                role = 'section_opener'
            elif i >= len(split_chunks) - 2:
                role = 'closer'
            elif i < 3:
                role = 'paragraph_opener'
            else:
                role = 'body'

            # Get target structure
            target = self.cadence_analyzer.get_target_paragraph_structure(
                position, role, self.cadence_profile
            )

            chunk.target_length = target.target_length
            chunk.target_sentence_count = target.target_sentence_count
            chunk.position = position
            chunk.role = role

            final_chunks.append(chunk)

        return final_chunks

    def _merge_small_chunks(self, chunks: List[SemanticChunk]) -> List[SemanticChunk]:
        """Merge chunks that are too small."""
        if not chunks:
            return []

        merged = []
        current = None

        for chunk in chunks:
            if current is None:
                current = chunk
            elif len(current.claims) + len(chunk.claims) <= self.max_chunk_claims:
                # Merge
                current.claims.extend(chunk.claims)
                current.relationships.extend(chunk.relationships)
                current.entities.extend(chunk.entities)
                current.key_concepts.extend(chunk.key_concepts)
                # Deduplicate (using text/identity for non-hashable objects)
                # Deduplicate relationships by evidence text
                seen_rels = set()
                unique_rels = []
                for rel in current.relationships:
                    if rel.evidence not in seen_rels:
                        seen_rels.add(rel.evidence)
                        unique_rels.append(rel)
                current.relationships = unique_rels
                # Deduplicate entities by text
                seen_entities = set()
                unique_entities = []
                for ent in current.entities:
                    if ent.text not in seen_entities:
                        seen_entities.add(ent.text)
                        unique_entities.append(ent)
                current.entities = unique_entities
                # Deduplicate key concepts (strings are hashable)
                current.key_concepts = list(set(current.key_concepts))[:10]
            else:
                merged.append(current)
                current = chunk

        if current:
            merged.append(current)

        return merged

    def _split_large_chunks(self, chunks: List[SemanticChunk]) -> List[SemanticChunk]:
        """Split chunks that are too large."""
        split = []

        for chunk in chunks:
            if len(chunk.claims) <= self.max_chunk_claims:
                split.append(chunk)
            else:
                # Split into multiple chunks
                for i in range(0, len(chunk.claims), self.max_chunk_claims):
                    split_claims = chunk.claims[i:i + self.max_chunk_claims]
                    split_chunk = self._create_chunk_from_claims(split_claims)
                    split_chunk.key_concepts = chunk.key_concepts  # Preserve concepts
                    split.append(split_chunk)

        return split

    def _validate_regrouping(self, chunks: List[SemanticChunk]):
        """
        Validate that all semantic content is included.

        Ensures all claims are included, adding missing ones to appropriate chunks.
        """
        # Collect all claims from chunks (use text as identifier since Claim isn't hashable)
        chunk_claim_texts = set()
        for chunk in chunks:
            for claim in chunk.claims:
                chunk_claim_texts.add(claim.text)

        # Check if all original claims are included
        original_claim_texts = {claim.text for claim in self.semantic_content.claims}
        missing = original_claim_texts - chunk_claim_texts

        if missing:
            print(f"  [SemanticRegrouper] Warning: {len(missing)} claims not included in regrouping, adding them now")
            # Add missing claims to the last chunk (or create a new one if needed)
            missing_claims = [c for c in self.semantic_content.claims if c.text in missing]
            if chunks:
                # Add to last chunk
                chunks[-1].claims.extend(missing_claims)
                # Update target length to accommodate additional claims
                chunks[-1].target_length += len(missing_claims) * 20  # Estimate 20 words per claim
                chunks[-1].target_sentence_count = max(chunks[-1].target_sentence_count, len(chunks[-1].claims))
            else:
                # Create a new chunk for missing claims
                if missing_claims:
                    # Find related relationships and entities for missing claims
                    missing_claim_texts = {c.text for c in missing_claims}
                    related_relationships = [
                        r for r in self.semantic_content.relationships
                        if r.source in missing_claim_texts or r.target in missing_claim_texts
                    ]
                    related_entities = [
                        e for e in self.semantic_content.entities
                        if any(e.text in claim.text for claim in missing_claims)
                    ]
                    new_chunk = SemanticChunk(
                        claims=missing_claims,
                        relationships=related_relationships,
                        entities=related_entities,
                        target_length=len(missing_claims) * 20,
                        target_sentence_count=max(1, len(missing_claims)),
                        position=1.0,
                        role='body'
                    )
                    chunks.append(new_chunk)

        # Check relationships (use evidence text as identifier)
        chunk_rel_evidence = set()
        for chunk in chunks:
            for rel in chunk.relationships:
                chunk_rel_evidence.add(rel.evidence)

        original_rel_evidence = {rel.evidence for rel in self.semantic_content.relationships}
        missing_rel = original_rel_evidence - chunk_rel_evidence

        if missing_rel:
            print(f"  [SemanticRegrouper] Warning: {len(missing_rel)} relationships not included")


# Test function
if __name__ == '__main__':
    from pathlib import Path
    from semantic_extractor import SemanticExtractor

    # Test with sample file
    sample_path = Path(__file__).parent / "prompts" / "sample_mao.txt"
    if sample_path.exists():
        with open(sample_path, 'r', encoding='utf-8') as f:
            sample_text = f.read()

        print("=== Semantic Regrouper Test ===\n")

        extractor = SemanticExtractor()
        analyzer = CadenceAnalyzer(sample_text, extractor)
        profile = analyzer.analyze_cadence()

        # Extract semantics from a test text
        test_text = """
        The United States has shifted its strategy from being a global hegemon to a competitor.
        This transition is evident in recent policy documents that acknowledge the end of the American century.
        The new approach focuses on reindustrialization and protectionism.
        Free trade policies have hollowed out the middle class.
        Therefore, a new economic strategy is needed.
        """

        semantic_content = extractor.extract(test_text)
        print(f"Extracted {len(semantic_content.claims)} claims")
        print(f"Extracted {len(semantic_content.relationships)} relationships\n")

        regrouper = SemanticRegrouper(semantic_content, profile, analyzer)
        chunks = regrouper.regroup_to_cadence()

        print(f"Regrouped into {len(chunks)} chunks:\n")
        for i, chunk in enumerate(chunks, 1):
            print(f"Chunk {i} ({chunk.role}, position {chunk.position:.0%}):")
            print(f"  Claims: {len(chunk.claims)}")
            print(f"  Target: {chunk.target_length} words, {chunk.target_sentence_count} sentences")
            print(f"  Key concepts: {', '.join(chunk.key_concepts[:5])}")
            print()
    else:
        print("No sample file found.")

