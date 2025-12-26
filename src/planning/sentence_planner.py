"""Sentence planning from semantic graphs."""

import uuid
from typing import List, Optional, Dict

from ..models.graph import (
    SemanticGraph,
    PropositionNode,
    RelationshipEdge,
    RelationshipType,
)
from ..models.plan import (
    SentenceNode,
    SentencePlan,
    SentenceRole,
    TransitionType,
)
from ..models.style import StyleProfile
from ..utils.logging import get_logger
from .graph_matcher import GraphMatcher, MatchedStyleGraph, FallbackMatcher
from .rhythm_planner import RhythmPlanner, RhythmPattern

logger = get_logger(__name__)


# Map relationship types to transitions
RELATIONSHIP_TO_TRANSITION = {
    RelationshipType.CAUSES: TransitionType.CAUSAL,
    RelationshipType.CONTRASTS: TransitionType.ADVERSATIVE,
    RelationshipType.ELABORATES: TransitionType.ADDITIVE,
    RelationshipType.REFERENCES: TransitionType.ADDITIVE,
    RelationshipType.FOLLOWS: TransitionType.NONE,
}


class PropositionClusterer:
    """Clusters propositions into sentence-sized groups.

    Respects:
    - Target sentence lengths from rhythm plan
    - Semantic boundaries (related propositions stay together)
    - Information density balance
    - Graph structure (root propositions first, dependents follow)
    """

    def __init__(self, words_per_proposition: float = 8.0):
        """Initialize clusterer.

        Args:
            words_per_proposition: Estimated words per proposition.
        """
        self.words_per_prop = words_per_proposition

    def cluster(
        self,
        propositions: List[PropositionNode],
        edges: List[RelationshipEdge],
        rhythm: RhythmPattern
    ) -> List[List[PropositionNode]]:
        """Cluster propositions to match rhythm pattern.

        Args:
            propositions: Propositions to cluster.
            edges: Relationships between propositions.
            rhythm: Target rhythm pattern.

        Returns:
            List of proposition clusters (one per sentence), with
            propositions ordered by graph structure within each cluster.
        """
        if not propositions:
            return []

        num_sentences = len(rhythm.lengths)
        if num_sentences == 0:
            return [[p] for p in propositions]

        if len(propositions) <= num_sentences:
            # One proposition per sentence or less
            return [[p] for p in propositions]

        # Build adjacency info for semantic boundaries
        adjacency = self._build_adjacency(propositions, edges)

        # Greedy clustering respecting semantic boundaries
        clusters = self._greedy_cluster(
            propositions,
            rhythm.lengths,
            adjacency
        )

        # Order propositions within each cluster by graph structure
        ordered_clusters = [
            self._order_by_graph_structure(cluster, edges)
            for cluster in clusters
        ]

        return ordered_clusters

    def _order_by_graph_structure(
        self,
        cluster: List[PropositionNode],
        edges: List[RelationshipEdge]
    ) -> List[PropositionNode]:
        """Order propositions within a cluster based on graph relationships.

        Ordering rules:
        1. Root propositions (sources with no incoming edges) come first
        2. Dependent propositions follow their sources
        3. Contrasting propositions are positioned after what they contrast

        Args:
            cluster: Propositions in this cluster.
            edges: All edges in the graph.

        Returns:
            Ordered list of propositions.
        """
        if len(cluster) <= 1:
            return cluster

        cluster_ids = {p.id for p in cluster}
        id_to_prop = {p.id: p for p in cluster}

        # Find incoming and outgoing edges within this cluster
        incoming: Dict[str, List[str]] = {p.id: [] for p in cluster}
        outgoing: Dict[str, List[str]] = {p.id: [] for p in cluster}

        for edge in edges:
            if edge.source_id in cluster_ids and edge.target_id in cluster_ids:
                incoming[edge.target_id].append(edge.source_id)
                outgoing[edge.source_id].append(edge.target_id)

        # Find roots (no incoming edges within cluster)
        roots = [p.id for p in cluster if not incoming[p.id]]

        # If no clear roots, use original order
        if not roots:
            return cluster

        # Topological sort within cluster (BFS from roots)
        ordered = []
        visited = set()
        queue = list(roots)

        while queue:
            node_id = queue.pop(0)
            if node_id in visited:
                continue
            visited.add(node_id)
            ordered.append(id_to_prop[node_id])

            # Add children (targets of outgoing edges)
            for child_id in outgoing.get(node_id, []):
                if child_id not in visited and child_id in cluster_ids:
                    queue.append(child_id)

        # Add any remaining unvisited nodes (disconnected)
        for p in cluster:
            if p.id not in visited:
                ordered.append(p)

        return ordered

    def _build_adjacency(
        self,
        propositions: List[PropositionNode],
        edges: List[RelationshipEdge]
    ) -> Dict[str, List[str]]:
        """Build adjacency map from edges.

        Args:
            propositions: All propositions.
            edges: Relationship edges.

        Returns:
            Map of node_id to list of related node_ids.
        """
        adjacency = {p.id: [] for p in propositions}

        for edge in edges:
            if edge.source_id in adjacency:
                adjacency[edge.source_id].append(edge.target_id)
            if edge.target_id in adjacency:
                adjacency[edge.target_id].append(edge.source_id)

        return adjacency

    def _greedy_cluster(
        self,
        propositions: List[PropositionNode],
        target_lengths: List[int],
        adjacency: Dict[str, List[str]]
    ) -> List[List[PropositionNode]]:
        """Greedy clustering algorithm.

        Args:
            propositions: Propositions to cluster.
            target_lengths: Target word counts per cluster.
            adjacency: Adjacency relationships.

        Returns:
            List of clusters.
        """
        n_clusters = len(target_lengths)
        n_props = len(propositions)

        # Convert targets to proposition counts
        target_counts = [
            max(1, int(length / self.words_per_prop + 0.5))
            for length in target_lengths
        ]

        # Adjust to ensure all propositions are assigned
        total_target = sum(target_counts)
        if total_target < n_props:
            # Distribute extra props
            diff = n_props - total_target
            for i in range(diff):
                target_counts[i % n_clusters] += 1
        elif total_target > n_props:
            # Reduce target counts
            diff = total_target - n_props
            for i in range(diff):
                idx = n_clusters - 1 - (i % n_clusters)
                if target_counts[idx] > 1:
                    target_counts[idx] -= 1

        # Assign propositions to clusters
        clusters = []
        prop_idx = 0

        for cluster_idx, count in enumerate(target_counts):
            cluster = []
            for _ in range(count):
                if prop_idx < n_props:
                    cluster.append(propositions[prop_idx])
                    prop_idx += 1
            if cluster:
                clusters.append(cluster)

        # Handle remaining propositions
        while prop_idx < n_props:
            if clusters:
                clusters[-1].append(propositions[prop_idx])
            else:
                clusters.append([propositions[prop_idx]])
            prop_idx += 1

        return clusters


class SentencePlanner:
    """Creates sentence plans from semantic graphs.

    Orchestrates:
    1. Style graph matching
    2. Rhythm planning
    3. Proposition clustering
    4. Role and transition assignment
    """

    def __init__(
        self,
        style_profile: StyleProfile,
        graph_matcher: Optional[GraphMatcher] = None
    ):
        """Initialize sentence planner.

        Args:
            style_profile: Target style profile.
            graph_matcher: Optional graph matcher with ChromaDB.
        """
        self.style_profile = style_profile
        self.graph_matcher = graph_matcher
        self.rhythm_planner = RhythmPlanner(style_profile)
        self.clusterer = PropositionClusterer()
        self.fallback = FallbackMatcher(style_profile)

    def create_plan(self, semantic_graph: SemanticGraph) -> SentencePlan:
        """Create a sentence plan from a semantic graph.

        Args:
            semantic_graph: Source semantic graph.

        Returns:
            SentencePlan for generating the paragraph.
        """
        propositions = semantic_graph.nodes
        edges = semantic_graph.edges

        if not propositions:
            return SentencePlan(
                nodes=[],
                paragraph_intent=semantic_graph.intent.value,
                paragraph_signature=self._detect_signature(edges),
                paragraph_role=semantic_graph.role.value,
                source_graph=semantic_graph
            )

        # Step 1: Try to find matching style graph
        matched_graph = None
        if self.graph_matcher:
            matched_graph = self.graph_matcher.find_best_match(
                semantic_graph, self.style_profile
            )

        # Step 2: Plan rhythm
        rhythm = self.rhythm_planner.plan_for_propositions(
            len(propositions), semantic_graph
        )

        # Step 3: Cluster propositions
        clusters = self.clusterer.cluster(propositions, edges, rhythm)

        # Step 4: Create sentence nodes
        sentence_nodes = []
        for i, (cluster, target_length) in enumerate(zip(clusters, rhythm.lengths)):
            role = self._determine_role(i, len(clusters), semantic_graph)
            transition = self._determine_transition(i, cluster, edges)

            # Get skeleton from matched graph if available
            skeleton = None
            if matched_graph and matched_graph.skeleton:
                skeleton = self._extract_skeleton_segment(
                    matched_graph.skeleton, i, len(clusters)
                )

            # Ensure target length is viable for the proposition content
            # Minimum = 60% of proposition word count (allowing for compression)
            min_viable_length = self._calculate_min_viable_length(cluster)
            adjusted_target = max(target_length, min_viable_length)

            node = SentenceNode(
                id=f"s_{i}_{uuid.uuid4().hex[:8]}",
                propositions=cluster,
                role=role,
                transition=transition,
                target_length=adjusted_target,
                target_skeleton=skeleton,
                keywords=self._extract_keywords(cluster)
            )
            sentence_nodes.append(node)

        # Step 5: Create plan
        plan = SentencePlan(
            nodes=sentence_nodes,
            paragraph_intent=semantic_graph.intent.value,
            paragraph_signature=self._detect_signature(edges),
            paragraph_role=semantic_graph.role.value,
            source_graph=semantic_graph,
            matched_style_graph_id=matched_graph.id if matched_graph else None
        )

        logger.debug(
            f"Created plan: {len(sentence_nodes)} sentences, "
            f"matched_graph={'yes' if matched_graph else 'no'}"
        )

        return plan

    def _determine_role(
        self,
        sentence_idx: int,
        total_sentences: int,
        graph: SemanticGraph
    ) -> SentenceRole:
        """Determine the role of a sentence.

        Args:
            sentence_idx: Index of sentence (0-based).
            total_sentences: Total sentences in paragraph.
            graph: Source semantic graph.

        Returns:
            SentenceRole enum value.
        """
        # First sentence in intro paragraph is often thesis
        if sentence_idx == 0:
            if graph.role.value == "INTRO":
                return SentenceRole.THESIS
            return SentenceRole.THESIS if total_sentences > 2 else SentenceRole.ELABORATION

        # Last sentence
        if sentence_idx == total_sentences - 1:
            if total_sentences > 2:
                return SentenceRole.CONCLUSION
            return SentenceRole.ELABORATION

        # Middle sentences
        return SentenceRole.ELABORATION

    def _determine_transition(
        self,
        sentence_idx: int,
        cluster: List[PropositionNode],
        edges: List[RelationshipEdge]
    ) -> TransitionType:
        """Determine transition type for a sentence.

        Args:
            sentence_idx: Index of sentence.
            cluster: Propositions in this sentence.
            edges: All edges in the graph.

        Returns:
            TransitionType enum value.
        """
        if sentence_idx == 0:
            return TransitionType.NONE

        # Check if any proposition in cluster has an incoming edge
        # that suggests a transition
        cluster_ids = {p.id for p in cluster}

        for edge in edges:
            if edge.target_id in cluster_ids:
                transition = RELATIONSHIP_TO_TRANSITION.get(
                    edge.relationship, TransitionType.NONE
                )
                if transition != TransitionType.NONE:
                    return transition

        return TransitionType.NONE

    def _calculate_min_viable_length(
        self,
        cluster: List[PropositionNode]
    ) -> int:
        """Calculate minimum viable sentence length for a proposition cluster.

        The LLM cannot compress content infinitely. This ensures the target
        length is at least 60% of the total proposition word count, allowing
        for reasonable compression while preventing content loss.

        Args:
            cluster: List of propositions to express.

        Returns:
            Minimum viable word count.
        """
        if not cluster:
            return 10  # Default minimum

        # Count total words in all propositions
        total_words = sum(
            len(p.text.split()) for p in cluster
        )

        # Allow 40% compression (60% retention)
        # This is generous - most content needs near 1:1
        min_viable = int(total_words * 0.6)

        # Absolute minimum of 10 words
        return max(10, min_viable)

    def _detect_signature(self, edges: List[RelationshipEdge]) -> str:
        """Detect the dominant signature from edges.

        Args:
            edges: Relationship edges.

        Returns:
            Signature string (CONTRAST, CAUSALITY, SEQUENCE, etc.)
        """
        if not edges:
            return "SEQUENCE"

        # Count relationship types
        counts = {}
        for edge in edges:
            rel = edge.relationship.value
            counts[rel] = counts.get(rel, 0) + 1

        # Find dominant
        dominant = max(counts.items(), key=lambda x: x[1])[0]

        # Map to signature
        if dominant == "CONTRASTS":
            return "CONTRAST"
        elif dominant == "CAUSES":
            return "CAUSALITY"
        elif dominant == "ELABORATES":
            return "ELABORATION"
        else:
            return "SEQUENCE"

    def _extract_skeleton_segment(
        self,
        skeleton: str,
        idx: int,
        total: int
    ) -> Optional[str]:
        """Extract a segment of the skeleton for one sentence.

        Args:
            skeleton: Full skeleton template.
            idx: Sentence index.
            total: Total sentences.

        Returns:
            Skeleton segment or None.
        """
        # Simple approach: if skeleton has [S1], [S2], etc., extract relevant one
        import re
        pattern = rf'\[S{idx + 1}\]'
        if re.search(pattern, skeleton):
            return f"[S{idx + 1}]"

        # If not found, return None (no skeleton guidance)
        return None

    def _estimate_min_length(self, cluster: List[PropositionNode]) -> int:
        """Estimate minimum word count needed to express propositions.

        The target length should be sufficient to express the semantic
        content without losing information. We estimate based on the
        actual text of the propositions, with some compression allowance.

        Args:
            cluster: Propositions to express.

        Returns:
            Minimum target word count.
        """
        if not cluster:
            return 5

        # Count words in all propositions
        total_words = 0
        for prop in cluster:
            total_words += len(prop.text.split())

        # Allow for some compression (style transfer can condense)
        # but ensure we have enough space to preserve key content
        # Use 70% of original as minimum
        min_length = int(total_words * 0.7)

        # But at least 5 words
        return max(5, min_length)

    def _extract_keywords(self, cluster: List[PropositionNode]) -> List[str]:
        """Extract keywords that must appear in the sentence.

        Args:
            cluster: Propositions for this sentence.

        Returns:
            List of required keywords.
        """
        keywords = set()
        for prop in cluster:
            # Include entities
            for entity in prop.entities[:2]:  # Limit to 2 per proposition
                keywords.add(entity)
            # Include subject if significant
            if prop.subject and len(prop.subject) > 3:
                keywords.add(prop.subject.split()[0])  # First word

        return list(keywords)[:5]  # Limit total
