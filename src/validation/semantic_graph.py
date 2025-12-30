"""Semantic graph representation for meaning preservation validation.

Represents paragraphs as directed graphs where:
- Nodes are propositions (claims, facts, statements)
- Edges are relationships (cause, contrast, example, sequence)

Graphs can be:
- Encoded as MermaidJS for LLM understanding
- Compared to find structural differences
- Used to generate precise repair instructions
"""

from dataclasses import dataclass, field
from typing import List, Dict, Set, Optional, Tuple
from enum import Enum

from ..utils.logging import get_logger

logger = get_logger(__name__)


class RelationType(Enum):
    """Types of relationships between propositions."""
    CAUSE = "cause"           # A causes/leads to B
    CONTRAST = "contrast"     # A but/however B
    EXAMPLE = "example"       # A, for example B
    ELABORATION = "elaborates"  # B expands on A
    SEQUENCE = "then"         # A, then B
    CONDITION = "if"          # If A then B
    SUPPORT = "supports"      # B supports/evidences A
    RESTATEMENT = "restates"  # B restates A differently


@dataclass
class PropositionNode:
    """A node in the semantic graph representing a proposition."""
    id: str                          # Unique identifier (P1, P2, etc.)
    text: str                        # Full proposition text
    subject: str                     # Main subject
    predicate: str                   # Main verb/predicate
    object: Optional[str] = None     # Object if present
    entities: List[str] = field(default_factory=list)  # Named entities
    is_negated: bool = False         # Whether proposition is negated
    epistemic: str = "factual"       # factual, hypothetical, appearance

    def summary(self) -> str:
        """Short summary for graph display."""
        parts = [self.subject, self.predicate]
        if self.object:
            parts.append(self.object)
        summary = " ".join(parts)
        if self.is_negated:
            summary = f"NOT: {summary}"
        return summary[:60]

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "text": self.text,
            "subject": self.subject,
            "predicate": self.predicate,
            "object": self.object,
            "entities": self.entities,
            "is_negated": self.is_negated,
            "epistemic": self.epistemic,
        }


@dataclass
class RelationEdge:
    """An edge in the semantic graph representing a relationship."""
    source_id: str          # Source node ID
    target_id: str          # Target node ID
    relation: RelationType  # Type of relationship

    def to_dict(self) -> dict:
        return {
            "source": self.source_id,
            "target": self.target_id,
            "relation": self.relation.value,
        }


@dataclass
class SemanticGraph:
    """A semantic graph representing the meaning structure of a paragraph."""
    nodes: List[PropositionNode] = field(default_factory=list)
    edges: List[RelationEdge] = field(default_factory=list)

    def add_node(self, node: PropositionNode) -> None:
        """Add a proposition node."""
        self.nodes.append(node)

    def add_edge(self, source_id: str, target_id: str, relation: RelationType) -> None:
        """Add a relationship edge."""
        self.edges.append(RelationEdge(source_id, target_id, relation))

    def get_node(self, node_id: str) -> Optional[PropositionNode]:
        """Get node by ID."""
        for node in self.nodes:
            if node.id == node_id:
                return node
        return None

    def get_entity_roles(self) -> Dict[str, List[str]]:
        """Extract entity-role mappings from the graph.

        Returns:
            Dict mapping entity names to their roles/actions.
            E.g., {"Karl Marx": ["developed worldview"], "Dzhugashvili": ["coined term"]}
        """
        entity_roles: Dict[str, List[str]] = {}

        for node in self.nodes:
            # Get the action from predicate
            action = node.predicate or ""
            if node.object:
                action = f"{action} {node.object}"

            # Map entities in this node to the action
            for entity in node.entities:
                entity_lower = entity.lower()
                if entity_lower not in entity_roles:
                    entity_roles[entity_lower] = []

                if action and action not in entity_roles[entity_lower]:
                    entity_roles[entity_lower].append(action)

            # Also track the subject if it's a named entity
            if node.subject:
                subject_lower = node.subject.lower()
                # Check if subject looks like a proper noun (capitalized or in entities)
                if (node.subject[0].isupper() or
                    any(node.subject.lower() in e.lower() for e in node.entities)):
                    if subject_lower not in entity_roles:
                        entity_roles[subject_lower] = []
                    if action and action not in entity_roles[subject_lower]:
                        entity_roles[subject_lower].append(action)

        return entity_roles

    def to_mermaid(self) -> str:
        """Encode graph as MermaidJS flowchart.

        Returns:
            MermaidJS flowchart definition.
        """
        lines = ["graph TD"]

        # Add nodes with their summaries
        for node in self.nodes:
            # Escape special chars for Mermaid
            summary = node.summary()
            # Replace characters that break Mermaid syntax
            for char, replacement in [('"', "'"), ("[", "("), ("]", ")"),
                                       ("|", "/"), ("{", "("), ("}", ")"),
                                       ("<", ""), (">", ""), ("#", "")]:
                summary = summary.replace(char, replacement)
            label = f'{node.id}["{summary}"]'
            lines.append(f"    {label}")

        # Add edges with relationship labels
        for edge in self.edges:
            arrow = self._get_arrow_style(edge.relation)
            label = edge.relation.value
            lines.append(f"    {edge.source_id} {arrow}|{label}| {edge.target_id}")

        return "\n".join(lines)

    def _get_arrow_style(self, relation: RelationType) -> str:
        """Get Mermaid arrow style for relation type."""
        if relation == RelationType.CONTRAST:
            return "-.->>"  # Dotted arrow for contrast
        elif relation == RelationType.CAUSE:
            return "==>"    # Thick arrow for causation
        elif relation == RelationType.CONDITION:
            return "-->"    # Normal arrow for condition
        else:
            return "-->"    # Default arrow

    def to_text_description(self) -> str:
        """Generate text description of the graph structure.

        This is useful for LLMs that may not parse MermaidJS well.
        """
        lines = ["SEMANTIC STRUCTURE:"]
        lines.append("")
        lines.append("PROPOSITIONS:")
        for node in self.nodes:
            neg = "[NEGATED] " if node.is_negated else ""
            epist = f"[{node.epistemic}] " if node.epistemic != "factual" else ""
            lines.append(f"  {node.id}: {neg}{epist}{node.summary()}")
            if node.entities:
                lines.append(f"      Entities: {', '.join(node.entities)}")

        if self.edges:
            lines.append("")
            lines.append("RELATIONSHIPS:")
            for edge in self.edges:
                source = self.get_node(edge.source_id)
                target = self.get_node(edge.target_id)
                if source and target:
                    lines.append(
                        f"  {edge.source_id} --[{edge.relation.value}]--> {edge.target_id}"
                    )

        return "\n".join(lines)

    def to_neutral_prose(self) -> str:
        """Generate neutral prose from the graph propositions.

        This creates a deterministic neutral description that preserves
        ALL propositions and their relationships. No LLM needed.

        Returns:
            Neutral prose representation suitable for LoRA transformation.
        """
        if not self.nodes:
            return ""

        # Build sentences in order, deduplicating by sentence text
        # (multiple propositions can come from the same sentence)
        sentences = []
        seen_sentences = set()

        for node in self.nodes:
            sentence = self._proposition_to_sentence(node)
            if not sentence:
                continue

            # Deduplicate by normalized sentence text
            normalized = sentence.strip().lower()
            if normalized in seen_sentences:
                continue

            seen_sentences.add(normalized)
            sentences.append(sentence)

        return " ".join(sentences)

    def to_narrative_flow(self) -> str:
        """Generate a clear narrative flow representation.

        Creates a simple, linear narrative that shows the progression
        of ideas using transition words based on edge relationships.

        Returns:
            A narrative flow string for LLM consumption.
        """
        if not self.nodes:
            return ""

        # Build a map of node_id -> outgoing edges
        outgoing = {}
        for edge in self.edges:
            if edge.source_id not in outgoing:
                outgoing[edge.source_id] = []
            outgoing[edge.source_id].append(edge)

        # Map relation types to transition phrases
        transitions = {
            RelationType.CAUSE: "Therefore",
            RelationType.CONTRAST: "However",
            RelationType.EXAMPLE: "For example",
            RelationType.ELABORATION: "Furthermore",
            RelationType.SEQUENCE: "Then",
            RelationType.CONDITION: "If so",
            RelationType.SUPPORT: "This is because",
            RelationType.RESTATEMENT: "In other words",
        }

        # Generate narrative lines
        lines = []
        seen_nodes = set()
        seen_texts = set()

        for i, node in enumerate(self.nodes):
            if node.id in seen_nodes:
                continue
            seen_nodes.add(node.id)

            # Get the proposition text
            text = node.text.strip() if node.text else node.summary()

            # Deduplicate by text content
            text_lower = text.lower()
            if text_lower in seen_texts:
                continue
            seen_texts.add(text_lower)

            # Find transition from previous node
            transition = ""
            if i > 0:
                prev_node = self.nodes[i - 1]
                if prev_node.id in outgoing:
                    for edge in outgoing[prev_node.id]:
                        if edge.target_id == node.id:
                            transition = transitions.get(edge.relation, "")
                            break

            # Format the line
            if transition:
                lines.append(f"→ {transition}: {text}")
            else:
                lines.append(f"• {text}")

        return "\n".join(lines)

    def _proposition_to_sentence(self, node: PropositionNode) -> str:
        """Convert a proposition node to a neutral sentence.

        Args:
            node: The proposition node to convert.

        Returns:
            A neutral sentence expressing the proposition.
        """
        # If we have the original text, use a cleaned version
        if node.text and len(node.text) > 10:
            # Use the original sentence text
            return node.text.strip()

        # Otherwise, build from subject-predicate-object
        parts = []

        # Handle negation
        if node.is_negated:
            parts.append("It is not the case that")

        # Handle epistemic stance
        if node.epistemic == "hypothetical":
            parts.append("It might be that")
        elif node.epistemic == "appearance":
            parts.append("It appears that")

        # Subject
        if node.subject:
            parts.append(node.subject)

        # Predicate
        if node.predicate:
            parts.append(node.predicate)

        # Object
        if node.object:
            parts.append(node.object)

        if not parts:
            return ""

        sentence = " ".join(parts)

        # Ensure proper ending
        if sentence and sentence[-1] not in ".!?":
            sentence += "."

        return sentence


@dataclass
class EntityRoleError:
    """An error in entity-role assignment."""
    error_type: str  # "conflation", "role_swap", "role_loss"
    entity: str
    source_role: str  # What the entity does in source
    output_role: Optional[str]  # What the entity does in output (if any)
    conflated_with: Optional[str] = None  # For conflation errors


@dataclass
class GraphDiff:
    """Difference between two semantic graphs."""
    missing_nodes: List[PropositionNode] = field(default_factory=list)
    added_nodes: List[PropositionNode] = field(default_factory=list)
    missing_edges: List[RelationEdge] = field(default_factory=list)
    added_edges: List[RelationEdge] = field(default_factory=list)
    modified_nodes: List[Tuple[PropositionNode, PropositionNode]] = field(default_factory=list)
    entity_role_errors: List[EntityRoleError] = field(default_factory=list)

    @property
    def is_isomorphic(self) -> bool:
        """Check if graphs are structurally equivalent."""
        return (
            len(self.missing_nodes) == 0 and
            len(self.added_nodes) == 0 and
            len(self.missing_edges) == 0 and
            len(self.added_edges) == 0 and
            len(self.modified_nodes) == 0 and
            len(self.entity_role_errors) == 0
        )

    @property
    def has_critical_differences(self) -> bool:
        """Check if there are critical structural differences.

        Only missing NODES are critical - these represent lost propositions.
        Missing edges and entity role variations are acceptable as long as
        the core ideas are preserved. The goal is semantic preservation,
        not exact structural matching.
        """
        return len(self.missing_nodes) > 0

    def to_repair_instructions(self) -> List[str]:
        """Generate repair instructions from diff."""
        instructions = []

        for node in self.missing_nodes:
            instructions.append(
                f"MISSING PROPOSITION: Must include the idea that '{node.summary()}'"
            )
            if node.entities:
                instructions.append(f"  - Include entities: {', '.join(node.entities)}")

        for edge in self.missing_edges:
            instructions.append(
                f"MISSING RELATIONSHIP: Show {edge.relation.value} connection between ideas"
            )

        for node in self.added_nodes:
            instructions.append(
                f"REMOVE: The claim '{node.summary()}' is not in the source"
            )

        for source_node, output_node in self.modified_nodes:
            if source_node.is_negated != output_node.is_negated:
                if source_node.is_negated:
                    instructions.append(
                        f"FIX NEGATION: '{source_node.summary()}' should be NEGATED"
                    )
                else:
                    instructions.append(
                        f"FIX NEGATION: '{source_node.summary()}' should NOT be negated"
                    )
            if source_node.epistemic != output_node.epistemic:
                instructions.append(
                    f"FIX EPISTEMIC: '{source_node.summary()}' should be {source_node.epistemic}, "
                    f"not {output_node.epistemic}"
                )

        # Entity role errors are CRITICAL - they change meaning entirely
        for error in self.entity_role_errors:
            if error.error_type == "conflation":
                instructions.insert(0,  # Put at top - most critical
                    f"CRITICAL ERROR: '{error.entity}' and '{error.conflated_with}' are DIFFERENT people/things. "
                    f"'{error.entity}' {error.source_role}. Do NOT conflate them."
                )
            elif error.error_type == "role_swap":
                instructions.insert(0,
                    f"CRITICAL ERROR: '{error.entity}' should {error.source_role}, NOT {error.output_role}"
                )
            elif error.error_type == "role_loss":
                instructions.insert(0,
                    f"CRITICAL ERROR: Missing that '{error.entity}' {error.source_role}"
                )

        return instructions

    def to_text(self) -> str:
        """Generate text summary of differences."""
        lines = ["GRAPH COMPARISON:"]

        if self.is_isomorphic:
            lines.append("  ✓ Graphs are structurally equivalent")
            return "\n".join(lines)

        if self.missing_nodes:
            lines.append(f"\nMISSING PROPOSITIONS ({len(self.missing_nodes)}):")
            for node in self.missing_nodes:
                lines.append(f"  - {node.id}: {node.summary()}")

        if self.added_nodes:
            lines.append(f"\nADDED PROPOSITIONS ({len(self.added_nodes)}):")
            for node in self.added_nodes:
                lines.append(f"  + {node.id}: {node.summary()}")

        if self.missing_edges:
            lines.append(f"\nMISSING RELATIONSHIPS ({len(self.missing_edges)}):")
            for edge in self.missing_edges:
                lines.append(f"  - {edge.source_id} --{edge.relation.value}--> {edge.target_id}")

        if self.added_edges:
            lines.append(f"\nADDED RELATIONSHIPS ({len(self.added_edges)}):")
            for edge in self.added_edges:
                lines.append(f"  + {edge.source_id} --{edge.relation.value}--> {edge.target_id}")

        if self.modified_nodes:
            lines.append(f"\nMODIFIED PROPOSITIONS ({len(self.modified_nodes)}):")
            for source, output in self.modified_nodes:
                lines.append(f"  ~ {source.id}: negation or epistemic stance changed")

        return "\n".join(lines)


class SemanticGraphBuilder:
    """Builds semantic graphs from text using REBEL triplet extraction.

    Uses the REBEL model for accurate (subject, relation, object) triplet
    extraction, with spaCy as fallback for entity/discourse detection.
    """

    def __init__(self, use_rebel: bool = False):
        """Initialize the builder.

        Args:
            use_rebel: Whether to use REBEL model (slower, often less accurate for complex sentences).
                       Default is False - use spaCy which is faster and handles passive voice better.
        """
        self._nlp = None
        self._use_rebel = use_rebel
        self._triplet_extractor = None

    @property
    def nlp(self):
        if self._nlp is None:
            from ..utils.nlp import get_nlp
            self._nlp = get_nlp()
        return self._nlp

    @property
    def triplet_extractor(self):
        if self._triplet_extractor is None:
            try:
                from .triplet_extractor import get_triplet_extractor
                self._triplet_extractor = get_triplet_extractor(use_rebel=self._use_rebel)
            except Exception as e:
                logger.warning(f"Triplet extractor not available: {e}")
                self._triplet_extractor = None
        return self._triplet_extractor

    def build_from_text(self, text: str) -> SemanticGraph:
        """Build a semantic graph from paragraph text.

        Uses REBEL for triplet extraction when available, falls back to
        spaCy-based extraction otherwise.

        Args:
            text: Paragraph text to analyze.

        Returns:
            SemanticGraph representing the paragraph's meaning structure.
        """
        graph = SemanticGraph()

        if not text or not text.strip():
            return graph

        # Use spaCy-based extraction by default (faster, handles complex sentences better)
        if not self._use_rebel:
            return self._build_from_spacy(text)

        # Try REBEL-based extraction if explicitly requested
        if self.triplet_extractor:
            try:
                return self._build_from_triplets(text)
            except Exception as e:
                logger.warning(f"REBEL extraction failed, falling back to spaCy: {e}")

        # Fallback to spaCy-based extraction
        return self._build_from_spacy(text)

    def _build_from_triplets(self, text: str) -> SemanticGraph:
        """Build graph using REBEL triplet extraction.

        This produces more accurate entity-role assignments.
        """
        graph = SemanticGraph()
        doc = self.nlp(text)

        # Extract triplets using REBEL
        triplets = self.triplet_extractor.extract(text)

        if not triplets:
            # Fallback if REBEL returns nothing
            return self._build_from_spacy(text)

        logger.debug(f"REBEL extracted {len(triplets)} triplets")

        # Create nodes from triplets
        for i, triplet in enumerate(triplets):
            node_id = f"P{i + 1}"

            # Find the sentence containing this triplet for context
            sent_text = self._find_sentence_for_triplet(doc, triplet)

            # Detect negation and epistemic from sentence context
            is_negated = False
            epistemic = "factual"
            if sent_text:
                sent_doc = self.nlp(sent_text)
                for sent in sent_doc.sents:
                    is_negated = self._detect_negation(sent)
                    epistemic = self._detect_epistemic(sent)
                    break

            # Collect entities associated with this triplet
            entities = []
            for ent in doc.ents:
                if ent.label_ in ("PERSON", "ORG", "GPE", "WORK_OF_ART"):
                    # Check if entity appears in subject or object
                    ent_lower = ent.text.lower()
                    if (ent_lower in triplet.subject.lower() or
                        ent_lower in triplet.object.lower()):
                        entities.append(ent.text)

            node = PropositionNode(
                id=node_id,
                text=sent_text or f"{triplet.subject} {triplet.relation} {triplet.object}",
                subject=triplet.subject,
                predicate=triplet.relation,
                object=triplet.object,
                entities=entities,
                is_negated=is_negated,
                epistemic=epistemic,
            )
            graph.add_node(node)

        # Add discourse relationships between nodes
        self._add_discourse_edges(doc, graph)

        return graph

    def _find_sentence_for_triplet(self, doc, triplet) -> Optional[str]:
        """Find the sentence that contains a triplet's subject and object."""
        subj_lower = triplet.subject.lower()
        obj_lower = triplet.object.lower()

        for sent in doc.sents:
            sent_lower = sent.text.lower()
            # Check if both subject and object appear in sentence
            if subj_lower in sent_lower or obj_lower in sent_lower:
                return sent.text.strip()

        return None

    def _add_discourse_edges(self, doc, graph: SemanticGraph) -> None:
        """Add discourse relationship edges between graph nodes."""
        sentences = list(doc.sents)
        nodes = graph.nodes

        if len(nodes) < 2:
            return

        # Connect sequential nodes with discourse relations
        for i in range(len(nodes) - 1):
            # Try to detect relation from the next relevant sentence
            if i + 1 < len(sentences):
                relation = self._detect_relation(sentences[i + 1])
                if relation:
                    graph.add_edge(nodes[i].id, nodes[i + 1].id, relation)
                else:
                    graph.add_edge(nodes[i].id, nodes[i + 1].id, RelationType.SEQUENCE)

    def _build_from_spacy(self, text: str) -> SemanticGraph:
        """Build graph using spaCy dependency parsing.

        Extracts MULTIPLE propositions from complex sentences by:
        - Finding all verb clauses
        - Extracting "by X" agent phrases (passive voice)
        - Extracting "of X" possession phrases
        """
        doc = self.nlp(text)
        graph = SemanticGraph()

        sentences = list(doc.sents)
        node_id = 0
        sentence_nodes: Dict[int, str] = {}

        for sent_idx, sent in enumerate(sentences):
            sent_text = sent.text.strip()
            if len(sent_text) < 10:
                continue

            # Extract ALL propositions from this sentence
            propositions = self._extract_all_propositions(sent)

            is_negated = self._detect_negation(sent)
            epistemic = self._detect_epistemic(sent)
            sent_entities = [ent.text for ent in sent.ents if ent.label_ in ("PERSON", "ORG", "GPE", "NORP")]

            for subj, pred, obj in propositions:
                if not subj or not pred:
                    continue

                node_id += 1
                node_id_str = f"P{node_id}"

                # Find entities relevant to this proposition
                prop_text = f"{subj} {pred} {obj or ''}"
                entities = [e for e in sent_entities if e.lower() in prop_text.lower()]

                node = PropositionNode(
                    id=node_id_str,
                    text=sent_text,
                    subject=subj,
                    predicate=pred,
                    object=obj,
                    entities=entities,
                    is_negated=is_negated,
                    epistemic=epistemic,
                )
                graph.add_node(node)
                sentence_nodes[sent_idx] = node_id_str

        # Extract relationships
        sent_indices = sorted(sentence_nodes.keys())
        for i in range(len(sent_indices) - 1):
            curr_idx = sent_indices[i]
            next_idx = sent_indices[i + 1]

            curr_id = sentence_nodes[curr_idx]
            next_id = sentence_nodes[next_idx]

            if next_idx < len(sentences):
                next_sent = sentences[next_idx]
                relation = self._detect_relation(next_sent)
                if relation:
                    graph.add_edge(curr_id, next_id, relation)
                else:
                    graph.add_edge(curr_id, next_id, RelationType.SEQUENCE)

        return graph

    def _extract_all_propositions(self, sent) -> List[Tuple[str, str, Optional[str]]]:
        """Extract ALL propositions from a sentence, including from clauses.

        Handles:
        - Main clause SVO
        - Subordinate clauses (while X, although X)
        - Passive constructions with agents (was coined by X)
        - Possession constructions (work of X)
        """
        propositions = []

        # Find all verbs (including auxiliary and root)
        verbs = [t for t in sent if t.pos_ in ("VERB", "AUX") and t.dep_ in ("ROOT", "relcl", "advcl", "ccomp", "conj")]

        for verb in verbs:
            subj = None
            obj = None
            agent = None

            # Find subject for this verb
            for child in verb.children:
                if child.dep_ in ("nsubj", "nsubjpass"):
                    subj = self._get_noun_phrase(child)
                elif child.dep_ in ("dobj", "attr", "acomp"):
                    obj = self._get_noun_phrase(child)
                elif child.dep_ == "agent":
                    # Passive voice: "was coined by X"
                    for grandchild in child.children:
                        if grandchild.dep_ == "pobj":
                            agent = self._get_noun_phrase(grandchild)
                elif child.dep_ == "prep":
                    prep_text = child.text.lower()
                    for grandchild in child.children:
                        if grandchild.dep_ == "pobj":
                            pobj = self._get_noun_phrase(grandchild)
                            if prep_text == "by" and not agent:
                                agent = pobj
                            elif prep_text == "of" and not obj:
                                # "work of Karl Marx" -> subject=work, predicate=of, object=Karl Marx
                                obj = pobj

            # Build proposition
            if subj and verb.lemma_:
                if agent:
                    # Passive: "term was coined by X" -> "X coined term"
                    propositions.append((agent, verb.lemma_, subj))
                else:
                    propositions.append((subj, verb.lemma_, obj))

        # Also check for "X of Y" constructions without verbs
        for chunk in sent.noun_chunks:
            # Look for "the work of Karl Marx" patterns
            for token in chunk:
                if token.dep_ == "prep" and token.text.lower() == "of":
                    for child in token.children:
                        if child.dep_ == "pobj":
                            # Found "X of Y" - add as proposition
                            possessor = self._get_noun_phrase(child)
                            possessed = chunk.root.text
                            if possessor and possessed:
                                # Check if this creates duplicate
                                existing = [(s, p, o) for s, p, o in propositions
                                           if possessor.lower() in (s or "").lower()]
                                if not existing:
                                    propositions.append((possessor, "has", possessed))

        # Fallback: if no propositions found, try basic SVO
        if not propositions:
            subj, pred, obj = self._extract_svo(sent)
            if subj and pred:
                propositions.append((subj, pred, obj))

        return propositions

    def _extract_svo(self, sent) -> Tuple[str, str, Optional[str]]:
        """Extract basic subject-verb-object from sentence."""
        subject = None
        verb = None
        obj = None

        for token in sent:
            if token.dep_ in ("nsubj", "nsubjpass", "expl") and subject is None:
                # Get the full noun phrase
                subject = self._get_noun_phrase(token)
            elif token.dep_ == "ROOT":
                if token.pos_ == "VERB":
                    verb = token.lemma_
                elif token.pos_ == "AUX":
                    # Handle "it is X" constructions
                    verb = token.lemma_
            elif token.dep_ in ("dobj", "attr", "pobj", "acomp") and obj is None:
                obj = self._get_noun_phrase(token)

        # Fallback: if no explicit subject, use first noun chunk
        if not subject and verb:
            for chunk in sent.noun_chunks:
                subject = chunk.text
                break

        # Fallback: if no verb found, use the root
        if not verb:
            for token in sent:
                if token.dep_ == "ROOT":
                    verb = token.text
                    break

        return subject, verb, obj

    def _get_noun_phrase(self, token) -> str:
        """Get the full noun phrase for a token."""
        # Get subtree but limit to avoid overly long phrases
        phrase_tokens = []
        for t in token.subtree:
            if t.dep_ not in ("punct", "cc", "conj"):
                phrase_tokens.append(t.text)
            if len(phrase_tokens) > 6:
                break
        return " ".join(phrase_tokens)

    def _detect_negation(self, sent) -> bool:
        """Detect if sentence contains negation."""
        negation_markers = {"not", "no", "never", "neither", "nor", "n't", "cannot", "won't", "don't", "doesn't"}
        for token in sent:
            if token.text.lower() in negation_markers or token.dep_ == "neg":
                return True
        return False

    def _detect_epistemic(self, sent) -> str:
        """Detect epistemic stance of sentence."""
        sent_lower = sent.text.lower()

        # Appearance/perception markers
        if any(m in sent_lower for m in ["seem", "appear", "look like", "as if", "as though"]):
            return "appearance"

        # Hypothetical markers
        if any(m in sent_lower for m in ["might", "could", "would", "may", "perhaps", "possibly"]):
            return "hypothetical"

        # Conditional
        if sent_lower.startswith("if ") or " if " in sent_lower:
            return "conditional"

        return "factual"

    def _detect_relation(self, sent) -> Optional[RelationType]:
        """Detect discourse relation from sentence markers."""
        sent_lower = sent.text.lower()
        first_word = sent_lower.split()[0] if sent_lower.split() else ""

        # Contrast markers
        contrast_markers = ["but", "however", "yet", "although", "though", "whereas", "while", "nevertheless"]
        if any(sent_lower.startswith(m) or f" {m} " in sent_lower[:50] for m in contrast_markers):
            return RelationType.CONTRAST

        # Cause markers
        cause_markers = ["because", "therefore", "thus", "hence", "so", "consequently", "as a result"]
        if any(m in sent_lower for m in cause_markers):
            return RelationType.CAUSE

        # Example markers
        example_markers = ["for example", "for instance", "such as", "e.g.", "including"]
        if any(m in sent_lower for m in example_markers):
            return RelationType.EXAMPLE

        # Condition markers
        if first_word == "if" or sent_lower.startswith("when "):
            return RelationType.CONDITION

        # Elaboration markers
        elaboration_markers = ["in other words", "that is", "specifically", "in particular", "namely"]
        if any(m in sent_lower for m in elaboration_markers):
            return RelationType.ELABORATION

        return None


class SemanticGraphComparator:
    """Compares two semantic graphs to find differences."""

    def __init__(
        self,
        similarity_threshold: Optional[float] = None,
        entity_match_threshold: Optional[float] = None,
        structural_match_threshold: Optional[float] = None,
    ):
        """Initialize comparator with configurable thresholds.

        Args:
            similarity_threshold: Minimum score for nodes to be considered equivalent.
                Lower values = more lenient matching (0.5 = "same general idea").
            entity_match_threshold: Weight for entity overlap (soft, not required).
            structural_match_threshold: Minimum subject-predicate-object alignment.

        All thresholds default to config values, or fallback defaults.
        """
        # Load from config if not provided
        config = self._load_validation_config()

        # Lower default threshold (0.5) since we want semantic preservation,
        # not exact matching. The output is an interpretation, not a translation.
        self.similarity_threshold = similarity_threshold or config.get("graph_similarity_threshold", 0.5)
        self.entity_match_threshold = entity_match_threshold or config.get("entity_match_threshold", 0.5)
        self.structural_match_threshold = structural_match_threshold or config.get("structural_match_threshold", 0.4)
        self._nlp = None

    @staticmethod
    def _load_validation_config() -> dict:
        """Load validation config from config.json."""
        import json
        from pathlib import Path

        config_paths = [
            Path("config.json"),
            Path(__file__).parent.parent.parent / "config.json",
        ]

        for config_path in config_paths:
            if config_path.exists():
                try:
                    with open(config_path) as f:
                        full_config = json.load(f)
                    return full_config.get("validation", {})
                except Exception:
                    pass

        return {}  # No config found, use defaults

    @property
    def nlp(self):
        if self._nlp is None:
            from ..utils.nlp import get_nlp
            self._nlp = get_nlp()
        return self._nlp

    def compare(self, source_graph: SemanticGraph, output_graph: SemanticGraph) -> GraphDiff:
        """Compare two graphs and return their differences.

        Args:
            source_graph: Graph from source text (ground truth).
            output_graph: Graph from generated output.

        Returns:
            GraphDiff with missing, added, and modified elements.
        """
        diff = GraphDiff()

        # Build mapping between source and output nodes based on semantic similarity
        node_mapping: Dict[str, str] = {}  # source_id -> output_id
        matched_output_ids: Set[str] = set()

        for source_node in source_graph.nodes:
            best_match = None
            best_score = 0.0

            for output_node in output_graph.nodes:
                if output_node.id in matched_output_ids:
                    continue

                score = self._compute_node_similarity(source_node, output_node)
                if score > best_score and score >= self.similarity_threshold:
                    best_score = score
                    best_match = output_node

            if best_match:
                node_mapping[source_node.id] = best_match.id
                matched_output_ids.add(best_match.id)

                # Check for modifications (negation, epistemic changes)
                if (source_node.is_negated != best_match.is_negated or
                    source_node.epistemic != best_match.epistemic):
                    diff.modified_nodes.append((source_node, best_match))
            else:
                # No match found - this proposition is missing
                diff.missing_nodes.append(source_node)

        # Find added nodes (in output but not matched to any source)
        for output_node in output_graph.nodes:
            if output_node.id not in matched_output_ids:
                diff.added_nodes.append(output_node)

        # Compare edges (relationships)
        for source_edge in source_graph.edges:
            # Map to output node IDs
            mapped_source = node_mapping.get(source_edge.source_id)
            mapped_target = node_mapping.get(source_edge.target_id)

            if not mapped_source or not mapped_target:
                # One of the nodes is missing, so edge is missing
                diff.missing_edges.append(source_edge)
                continue

            # Check if equivalent edge exists in output
            edge_found = False
            for output_edge in output_graph.edges:
                if (output_edge.source_id == mapped_source and
                    output_edge.target_id == mapped_target and
                    output_edge.relation == source_edge.relation):
                    edge_found = True
                    break

            if not edge_found:
                diff.missing_edges.append(source_edge)

        # Find added edges
        reverse_mapping = {v: k for k, v in node_mapping.items()}
        for output_edge in output_graph.edges:
            # Check if this edge has a corresponding source edge
            source_source = reverse_mapping.get(output_edge.source_id)
            source_target = reverse_mapping.get(output_edge.target_id)

            if not source_source or not source_target:
                # Edge involves an added node
                diff.added_edges.append(output_edge)
                continue

            # Check if equivalent edge exists in source
            edge_found = False
            for source_edge in source_graph.edges:
                if (source_edge.source_id == source_source and
                    source_edge.target_id == source_target):
                    edge_found = True
                    break

            if not edge_found:
                diff.added_edges.append(output_edge)

        # Compare entity roles - detect conflation and role swaps
        entity_errors = self._compare_entity_roles(source_graph, output_graph)
        diff.entity_role_errors.extend(entity_errors)

        return diff

    def _compare_entity_roles(
        self,
        source_graph: SemanticGraph,
        output_graph: SemanticGraph,
    ) -> List[EntityRoleError]:
        """Compare entity-role assignments between graphs.

        Detects:
        - Conflation: Two distinct entities treated as the same
        - Role swap: Entity assigned a different action than in source
        - Role loss: Entity's role from source is missing

        Args:
            source_graph: Graph from source text.
            output_graph: Graph from generated output.

        Returns:
            List of EntityRoleError instances.
        """
        errors = []

        source_roles = source_graph.get_entity_roles()
        output_roles = output_graph.get_entity_roles()

        # Check for conflation in output text
        # Look for patterns like "X (a.k.a. Y)" or "X, also known as Y"
        output_text = " ".join(node.text.lower() for node in output_graph.nodes)

        for entity1 in source_roles:
            for entity2 in source_roles:
                if entity1 >= entity2:  # Avoid duplicate checks
                    continue

                # Check if output conflates these entities
                conflation_patterns = [
                    f"{entity1} (a.k.a. {entity2}",
                    f"{entity2} (a.k.a. {entity1}",
                    f"{entity1}, also known as {entity2}",
                    f"{entity2}, also known as {entity1}",
                    f"{entity1} or {entity2}",
                    f"{entity2} or {entity1}",
                    f"{entity1} ({entity2})",
                    f"{entity2} ({entity1})",
                ]

                for pattern in conflation_patterns:
                    if pattern.lower() in output_text.lower():
                        # Found conflation!
                        source_role1 = source_roles.get(entity1, ["has a role"])[0]
                        errors.append(EntityRoleError(
                            error_type="conflation",
                            entity=entity1,
                            source_role=source_role1,
                            output_role=None,
                            conflated_with=entity2,
                        ))
                        break

        # Check for role swaps and losses
        for entity, source_actions in source_roles.items():
            if not source_actions:
                continue

            output_actions = output_roles.get(entity, [])

            if not output_actions:
                # Entity's role is completely missing
                for action in source_actions[:2]:  # Limit to first 2 roles
                    errors.append(EntityRoleError(
                        error_type="role_loss",
                        entity=entity,
                        source_role=action,
                        output_role=None,
                    ))
            else:
                # Check for role swaps - entity has different action
                for source_action in source_actions:
                    action_found = any(
                        self._actions_similar(source_action, out_action)
                        for out_action in output_actions
                    )
                    if not action_found:
                        # Role swap - entity does something different
                        errors.append(EntityRoleError(
                            error_type="role_swap",
                            entity=entity,
                            source_role=source_action,
                            output_role=output_actions[0] if output_actions else None,
                        ))

        return errors

    def _actions_similar(self, action1: str, action2: str) -> bool:
        """Check if two actions are semantically similar."""
        if not action1 or not action2:
            return False

        # Normalize
        a1 = action1.lower().strip()
        a2 = action2.lower().strip()

        # Exact match
        if a1 == a2:
            return True

        # Significant word overlap
        words1 = set(w for w in a1.split() if len(w) > 3)
        words2 = set(w for w in a2.split() if len(w) > 3)

        if words1 and words2:
            overlap = len(words1 & words2) / max(len(words1), len(words2))
            return overlap >= 0.5

        return False

    def _compute_node_similarity(self, node1: PropositionNode, node2: PropositionNode) -> float:
        """Compute similarity between two proposition nodes.

        The goal is semantic preservation - do these nodes express the same idea?
        We use soft matching rather than strict requirements, since the output
        is an interpretation in the author's voice, not a literal translation.

        Uses configurable thresholds:
        - structural_match_threshold: Minimum S-P-O alignment
        - entity_match_threshold: Bonus weight for entity overlap (not required)
        """
        # Check structural match (subject-predicate-object alignment)
        struct_score = self._compute_structural_match(node1, node2)

        # Entity overlap contributes to score but doesn't cause hard failure
        ents1 = set(e.lower() for e in node1.entities)
        ents2 = set(e.lower() for e in node2.entities)

        if ents1 and ents2:
            entity_sim = len(ents1 & ents2) / len(ents1)
        elif ents1:
            # Source has entities, output doesn't - penalize but don't fail
            entity_sim = 0.3
        else:
            entity_sim = 1.0

        # Semantic similarity using spaCy vectors
        doc1 = self.nlp(node1.text[:200])
        doc2 = self.nlp(node2.text[:200])

        vector_sim = 0.0
        if doc1.has_vector and doc2.has_vector:
            try:
                vector_sim = doc1.similarity(doc2)
            except (ValueError, RuntimeWarning):
                vector_sim = 0.0

        # Content word overlap (lemma-based)
        words1 = set(t.lemma_.lower() for t in doc1 if t.pos_ in ("NOUN", "VERB", "PROPN", "ADJ") and not t.is_stop)
        words2 = set(t.lemma_.lower() for t in doc2 if t.pos_ in ("NOUN", "VERB", "PROPN", "ADJ") and not t.is_stop)

        if words1 and words2:
            keyword_sim = len(words1 & words2) / max(len(words1), len(words2))
        else:
            keyword_sim = 0.0

        # Weighted combination - emphasize semantic similarity over exact matching
        if doc1.has_vector and doc2.has_vector:
            # Vector similarity is a good proxy for "same meaning"
            base_score = vector_sim * 0.4 + keyword_sim * 0.25 + struct_score * 0.25 + entity_sim * 0.1
        else:
            # No vectors - rely more on keyword overlap
            base_score = keyword_sim * 0.5 + struct_score * 0.35 + entity_sim * 0.15

        return base_score

    def _compute_structural_match(self, node1: PropositionNode, node2: PropositionNode) -> float:
        """Compare subject-predicate-object structure strictly."""
        score = 0.0
        matches = 0
        total = 0

        # Compare subjects
        if node1.subject and node2.subject:
            total += 1
            subj1_lemma = self._lemmatize(node1.subject)
            subj2_lemma = self._lemmatize(node2.subject)
            if subj1_lemma == subj2_lemma:
                matches += 1
            elif self._are_synonymous(subj1_lemma, subj2_lemma):
                matches += 0.7

        # Compare predicates
        if node1.predicate and node2.predicate:
            total += 1
            pred1_lemma = self._lemmatize(node1.predicate)
            pred2_lemma = self._lemmatize(node2.predicate)
            if pred1_lemma == pred2_lemma:
                matches += 1
            elif self._are_synonymous(pred1_lemma, pred2_lemma):
                matches += 0.7

        # Compare objects
        if node1.object and node2.object:
            total += 1
            obj1_lemma = self._lemmatize(node1.object)
            obj2_lemma = self._lemmatize(node2.object)
            if obj1_lemma == obj2_lemma:
                matches += 1
            elif self._are_synonymous(obj1_lemma, obj2_lemma):
                matches += 0.7

        if total > 0:
            score = matches / total
        else:
            score = 0.5  # No structure to compare

        return score

    def _lemmatize(self, text: str) -> str:
        """Get lemmatized form of text."""
        if not text:
            return ""
        doc = self.nlp(text.lower())
        return " ".join(t.lemma_ for t in doc if not t.is_punct and not t.is_space)

    def _are_synonymous(self, word1: str, word2: str) -> bool:
        """Check if two words/phrases are synonymous using spaCy."""
        if not word1 or not word2:
            return False

        # Quick exact match
        if word1 == word2:
            return True

        # Use spaCy vector similarity (only if vectors available)
        doc1 = self.nlp(word1)
        doc2 = self.nlp(word2)

        # Check if vectors are available (avoid W008 warning)
        if not doc1.has_vector or not doc2.has_vector:
            # Fall back to lemma comparison
            lemmas1 = set(t.lemma_.lower() for t in doc1 if not t.is_stop)
            lemmas2 = set(t.lemma_.lower() for t in doc2 if not t.is_stop)
            return bool(lemmas1 & lemmas2)

        try:
            sim = doc1.similarity(doc2)
            return sim > 0.85  # High threshold for synonymy
        except (ValueError, RuntimeWarning):
            return False


def build_and_compare_graphs(source_text: str, output_text: str) -> Tuple[SemanticGraph, SemanticGraph, GraphDiff]:
    """Convenience function to build and compare graphs.

    Args:
        source_text: Source paragraph text.
        output_text: Generated output text.

    Returns:
        Tuple of (source_graph, output_graph, diff).
    """
    builder = SemanticGraphBuilder()
    comparator = SemanticGraphComparator()

    source_graph = builder.build_from_text(source_text)
    output_graph = builder.build_from_text(output_text)
    diff = comparator.compare(source_graph, output_graph)

    return source_graph, output_graph, diff
