"""Topological Graph Matcher.

Matches input logical graphs to author style graphs from ChromaDB
and maps input nodes to style nodes.
"""

import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import List, Optional, Dict, Any

import chromadb
from chromadb.utils import embedding_functions

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.generator.llm_provider import LLMProvider


class TopologicalMatcher:
    """Matches input graphs to style graphs and maps nodes."""

    def __init__(self, config_path: str = "config.json", chroma_path: Optional[str] = None, llm_provider: Optional[LLMProvider] = None):
        """Initialize the Topological Matcher.

        Args:
            config_path: Path to configuration file.
            chroma_path: Optional custom path for ChromaDB. Defaults to atlas_cache/chroma.
            llm_provider: Optional LLM provider instance. If None, creates a new one.
        """
        self.config_path = config_path

        # Initialize ChromaDB client
        if chroma_path:
            self.chroma_path = Path(chroma_path)
        else:
            self.chroma_path = project_root / "atlas_cache" / "chroma"

        self.chroma_path.mkdir(parents=True, exist_ok=True)

        print(f"Initializing ChromaDB at {self.chroma_path}...")
        self.client = chromadb.PersistentClient(path=str(self.chroma_path))

        # Get collection
        collection_name = "style_graphs"
        try:
            self.collection = self.client.get_collection(name=collection_name)
            print(f"Using existing collection: {collection_name}")
        except Exception:
            # Create collection with default embedding function
            embedding_fn = embedding_functions.DefaultEmbeddingFunction()
            self.collection = self.client.create_collection(
                name=collection_name,
                embedding_function=embedding_fn
            )
            print(f"Created new collection: {collection_name}")

        # Initialize LLM provider (use provided one or create new)
        if llm_provider is not None:
            self.llm_provider = llm_provider
        else:
            print("Initializing LLM provider...")
            self.llm_provider = LLMProvider(config_path=config_path)

        # Load config for potential future use
        with open(config_path, 'r') as f:
            self.config = json.load(f)

        # Initialize spaCy model cache (lazy loading)
        self._nlp_cache = None

    def _strip_markdown_code_blocks(self, text: str) -> str:
        """Strip markdown code blocks from text.

        Args:
            text: Text that may contain markdown code blocks.

        Returns:
            Text with code blocks removed.
        """
        # Remove ```json ... ``` blocks
        text = re.sub(r'```json\s*\n?(.*?)\n?```', r'\1', text, flags=re.DOTALL)
        # Remove ``` ... ``` blocks (generic)
        text = re.sub(r'```\s*\n?(.*?)\n?```', r'\1', text, flags=re.DOTALL)
        # Remove any remaining ``` markers
        text = text.replace('```', '')
        return text.strip()

    def _parse_mermaid_nodes(self, mermaid: str) -> List[str]:
        """Parse Mermaid graph string to extract node names.

        Args:
            mermaid: Mermaid graph string (e.g., "graph LR; ROOT --> NODE1").

        Returns:
            List of unique node identifiers.
        """
        nodes = set()

        # Remove graph declaration if present
        mermaid = re.sub(r'^graph\s+\w+\s*;?\s*', '', mermaid, flags=re.IGNORECASE)

        # Pattern 1: ROOT --> NODE1 or ROOT --edge--> NODE1
        # Pattern 2: NODE1[Label] --> NODE2
        # Pattern 3: ROOT --> NODE1 --> NODE2 (chain)

        # Extract all node identifiers
        # Match node names (alphanumeric, underscore, can have brackets)
        node_pattern = r'([A-Z_][A-Z0-9_]*)(?:\[[^\]]*\])?'

        # Find all nodes in the graph
        matches = re.findall(node_pattern, mermaid)
        nodes.update(matches)

        # Also look for nodes in edge definitions
        # ROOT --label--> NODE1
        edge_pattern = r'([A-Z_][A-Z0-9_]*)\s*--[^>]*-->\s*([A-Z_][A-Z0-9_]*)'
        edge_matches = re.findall(edge_pattern, mermaid)
        for match in edge_matches:
            nodes.add(match[0])
            nodes.add(match[1])

        # Also handle simple arrow notation: ROOT --> NODE1
        simple_edge_pattern = r'([A-Z_][A-Z0-9_]*)\s*-->\s*([A-Z_][A-Z0-9_]*)'
        simple_matches = re.findall(simple_edge_pattern, mermaid)
        for match in simple_matches:
            nodes.add(match[0])
            nodes.add(match[1])

        # Filter out invalid single-underscore nodes (parsing artifacts)
        nodes = {node for node in nodes if node != '_'}

        return sorted(list(nodes))

    def _parse_mermaid_edges(self, mermaid: str) -> List[tuple]:
        """Parse Mermaid graph string to extract edges with labels.

        Args:
            mermaid: Mermaid graph string (e.g., "graph LR; P0 --cause--> P1; P1 --contrast--> P2")

        Returns:
            List of tuples (source, target, label) where label is the edge type
        """
        edges = []

        # Remove graph declaration if present
        mermaid_clean = re.sub(r'^graph\s+\w+\s*;?\s*', '', mermaid, flags=re.IGNORECASE)

        # Pattern 1: P0 --label--> P1 (labeled edge)
        labeled_edge_pattern = r'([A-Z_][A-Z0-9_]*)\s*--([^-]+)-->\s*([A-Z_][A-Z0-9_]*)'
        labeled_matches = re.findall(labeled_edge_pattern, mermaid_clean)
        for match in labeled_matches:
            source, label, target = match
            # Normalize label (remove extra spaces, convert to uppercase)
            label = label.strip().upper()
            # Map common edge labels to standard types
            label_map = {
                'CAUSE': 'CAUSALITY',
                'CONTRAST': 'CONTRAST',
                'DEFINE': 'DEFINITION',
                'DEFINITION': 'DEFINITION',
                'SEQUENCE': 'SEQUENCE',
                'SUPPORT': 'SUPPORT',
                'CONDITION': 'CONDITIONAL',
                'CONDITIONAL': 'CONDITIONAL',
                'LIST': 'LIST',
                'ENUMERATION': 'LIST'
            }
            normalized_label = label_map.get(label, label)
            edges.append((source, target, normalized_label))

        # Pattern 2: P0 --> P1 (unlabeled edge, infer from context or use SEQUENCE)
        simple_edge_pattern = r'([A-Z_][A-Z0-9_]*)\s*-->\s*([A-Z_][A-Z0-9_]*)'
        simple_matches = re.findall(simple_edge_pattern, mermaid_clean)
        for match in simple_matches:
            source, target = match
            # Check if this edge was already captured with a label
            if not any(e[0] == source and e[1] == target for e in edges):
                # Default to SEQUENCE for unlabeled edges
                edges.append((source, target, 'SEQUENCE'))

        return edges

    def _select_diverse_candidates(
        self,
        candidates: List[Dict[str, Any]],
        top_k: int,
        input_intent: Optional[str]
    ) -> List[Dict[str, Any]]:
        """Select diverse candidates covering different intents.

        Args:
            candidates: List of candidate dictionaries (already sorted by priority_score).
            top_k: Number of candidates to select.
            input_intent: Input intent for prioritization.

        Returns:
            List of top_k diverse candidates.
        """
        if len(candidates) <= top_k:
            return candidates[:top_k]

        # Group candidates by intent
        intent_groups = {}
        for candidate in candidates:
            intent = candidate.get('intent', 'UNKNOWN')
            if intent not in intent_groups:
                intent_groups[intent] = []
            intent_groups[intent].append(candidate)

        # Prioritize input intent if it exists
        selected = []
        seen_intents = set()

        # First, add best candidate with matching intent (if exists)
        if input_intent:
            for intent, group in intent_groups.items():
                if intent.upper() == input_intent.upper():
                    selected.append(group[0])
                    seen_intents.add(intent)
                    break

        # Then, add one candidate from each intent group (diversity)
        for intent, group in intent_groups.items():
            if intent not in seen_intents and len(selected) < top_k:
                selected.append(group[0])
                seen_intents.add(intent)

        # Fill remaining slots with best candidates (regardless of intent)
        remaining = top_k - len(selected)
        if remaining > 0:
            for candidate in candidates:
                if candidate not in selected and len(selected) < top_k:
                    selected.append(candidate)

        return selected[:top_k]

    def _synthesize_best_fit(
        self,
        input_graph: Dict[str, Any],
        candidates: List[Dict[str, Any]],
        verbose: bool = False
    ) -> Dict[str, Any]:
        """Use LLM to select best skeleton or synthesize a fixed version.

        Args:
            input_graph: Input graph data with 'description', 'intent', 'mermaid', etc.
            candidates: List of candidate dictionaries.

        Returns:
            Selected candidate dictionary with potentially updated 'skeleton' field.
        """
        input_description = input_graph.get('description', '')
        input_intent = input_graph.get('intent', 'UNKNOWN')
        input_mermaid = input_graph.get('mermaid', '')

        # Format candidates for prompt
        candidates_text = []
        for i, candidate in enumerate(candidates):
            skeleton = candidate.get('skeleton', '')
            intent = candidate.get('intent', 'UNKNOWN')
            distance = candidate.get('distance', float('inf'))
            node_count = candidate.get('node_count', 0)
            candidates_text.append(
                f"Candidate {i}:\n"
                f"  Intent: {intent}\n"
                f"  Skeleton: {skeleton}\n"
                f"  Distance: {distance:.3f}\n"
                f"  Node Count: {node_count}"
            )

        system_prompt = """You are a Skeleton Selector and Synthesizer. Your task is to:
1. Identify logical mismatches between input logic and candidate skeletons.
2. Select the best base candidate.
3. If needed, rewrite the skeleton to fix connector mismatches."""

        user_prompt = f"""**Input Logic:**
{input_description}

**Input Intent:** {input_intent}

**Input Graph:** {input_mermaid}

**Candidates:**
{chr(10).join(candidates_text)}

**Task:**
1. **Identify Logical Mismatch:** Check if any candidate's skeleton has connectors that contradict the input logic.
   - Example: Input is "Definition" (explaining what X is), but skeleton has "However" (contrast connector).
   - Example: Input is "And" (addition), but skeleton has "But" (contrast).

2. **Select Best Base Candidate:**
   - Choose the candidate whose skeleton structure best matches the input logic.
   - Prioritize intent matches, but also consider structural fit.

3. **Rewrite the Skeleton (if needed):**
   - If the selected candidate's connectors contradict the input (e.g., Skeleton has "However" but Input is "And"),
     you MUST output a modified skeleton with the correct connectors.
   - Keep the same structural complexity and rhythm.
   - Only change connectors/logical words, not the overall sentence structure.
   - If no rewrite is needed, return the original skeleton as-is.

**Output JSON:**
{{
  "selected_index": 0,
  "revised_skeleton": "The original or modified skeleton text here..."
}}

**Rules:**
- `selected_index` must be between 0 and {len(candidates) - 1}.
- `revised_skeleton` must be a complete, grammatically valid sentence structure.
- If the skeleton is already correct, return it unchanged.
- Only modify connectors/logical words that contradict the input intent."""

        try:
            response = self.llm_provider.call(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                model_type="editor",
                require_json=True,
                temperature=0.3,
                max_tokens=800
            )

            # Strip markdown code blocks
            response = self._strip_markdown_code_blocks(response)

            # Parse JSON
            result = json.loads(response)

            selected_index = result.get('selected_index', 0)
            revised_skeleton = result.get('revised_skeleton', '')

            # Validate selected_index
            if not isinstance(selected_index, int) or selected_index < 0 or selected_index >= len(candidates):
                print(f"Warning: Invalid selected_index {selected_index}, using 0")
                selected_index = 0

            # Get selected candidate
            selected_candidate = candidates[selected_index].copy()
            original_skeleton = selected_candidate.get('skeleton', '')

            # Update skeleton if revised
            if revised_skeleton and revised_skeleton.strip():
                selected_candidate['skeleton'] = revised_skeleton.strip()
                if verbose:
                    if revised_skeleton.strip() != original_skeleton:
                        print(f"     ✏️  Skeleton revised:")
                        print(f"        Original: {original_skeleton[:80]}...")
                        print(f"        Revised:  {revised_skeleton.strip()[:80]}...")
                    else:
                        print(f"     ✓ Skeleton unchanged: {original_skeleton[:80]}...")
            elif verbose:
                print(f"     ✓ Selected candidate {selected_index}: {original_skeleton[:80]}...")

            if verbose:
                selected_intent = selected_candidate.get('intent', 'UNKNOWN')
                selected_distance = selected_candidate.get('distance', float('inf'))
                print(f"     Selected: Intent={selected_intent}, Distance={selected_distance:.3f}")

            return selected_candidate

        except (json.JSONDecodeError, Exception) as e:
            print(f"Warning: Synthesis LLM call failed: {e}")
            # Fallback: return top candidate (lowest distance)
            return candidates[0]

    def synthesize_match(
        self,
        propositions: List[str],
        input_intent: str,
        document_context: Optional[Dict[str, Any]] = None,
        style_profile: Optional[Dict[str, Any]] = None,
        input_signature: Optional[str] = None,
        global_index: Optional[int] = None,
        input_role: Optional[str] = None,
        prev_paragraph_summary: Optional[str] = None,
        input_graph: Optional[Dict[str, Any]] = None,
        verbose: bool = False
    ) -> Dict[str, Any]:
        """Synthesize a custom blueprint from propositions using The Architect pattern.

        Args:
            propositions: List of input proposition strings.
            input_intent: Rhetorical intent (DEFINITION, ARGUMENT, NARRATIVE, etc.).
            document_context: Optional dict with 'current_index' and 'total_paragraphs' for role filtering.
            style_profile: Optional dict with author's style profile for style injection.
            input_signature: Logical signature (CONTRAST, CAUSALITY, DEFINITION, etc.) for filtering.
            global_index: Paragraph index in document (0-based) for hard filtering.
            input_role: Narrative role (INTRO, ELABORATION, CONCLUSION) detected by mapper.
            prev_paragraph_summary: Optional summary of previous paragraph for context continuity.
            verbose: Enable verbose logging.

        Returns:
            Dict with 'style_metadata', 'node_mapping', 'intent', and 'signature' (effective signature used).
        """
        if verbose:
            print(f"  🏗️  Assembler: Synthesizing blueprint from {len(propositions)} propositions")
            print(f"     Input intent: {input_intent}")
            if input_signature:
                print(f"     Input signature: {input_signature}")
            if global_index is not None:
                print(f"     Global index: {global_index}")
            if input_role:
                print(f"     Input role: {input_role}")

        # Default input_signature to DEFINITION if not provided
        if not input_signature:
            input_signature = 'DEFINITION'
            if verbose:
                print(f"     No input signature provided, defaulting to DEFINITION")

        # Step A: Retrieve top_k=20 candidates from ChromaDB with signature filtering
        candidates = []
        effective_signature = input_signature  # Track which signature was actually used

        try:
            # Get structural summary from input_graph if available, otherwise generate it
            structural_summary = None
            if input_graph and 'structural_summary' in input_graph:
                structural_summary = input_graph['structural_summary']
                if verbose:
                    print(f"     Using structural_summary from input_graph: {structural_summary[:80]}...")

            # If no structural_summary available, generate one from propositions
            if not structural_summary:
                # Generate structural summary using LLM
                structural_summary = self._generate_structural_summary(propositions, input_intent, input_signature, verbose=verbose)
                if verbose:
                    print(f"     Generated structural_summary: {structural_summary[:80]}...")

            # Fallback to content description if structural summary generation fails
            if not structural_summary:
                structural_summary = f"{' '.join(propositions[:2])[:200]}..." if len(propositions) > 2 else ' '.join(propositions)
                if verbose:
                    print(f"     ⚠ Using content description as fallback for structural query")

            # Determine target role for filtering
            target_role = self._determine_role(document_context)

            # Check if signature and role fields exist in metadata
            has_signature_field = False
            has_role_field = False
            has_narrative_role_field = False
            try:
                sample = self.collection.get(limit=1, include=['metadatas'])
                if sample.get('metadatas') and len(sample['metadatas']) > 0:
                    sample_meta = sample['metadatas'][0]
                    has_signature_field = 'signature' in sample_meta
                    has_role_field = 'paragraph_role' in sample_meta
                    has_narrative_role_field = 'role' in sample_meta  # Check for narrative role field
            except Exception:
                pass

            # Primary Query: Filter by signature and role (hard constraints)
            # Use structural_summary for querying (content-blind structural matching)
            query_text = structural_summary if structural_summary else structural_query
            if verbose:
                print(f"     Querying with structural summary: {query_text[:80]}...")
            query_kwargs = {
                'query_texts': [query_text],
                'n_results': 20
            }

            # Build where clause with signature and role filters
            conditions = []
            if has_signature_field and input_signature:
                conditions.append({"signature": input_signature})

            # Hard constraint: First paragraph MUST use INTRO skeletons
            if global_index == 0 and has_narrative_role_field:
                conditions.append({"role": "INTRO"})
                if verbose:
                    print(f"     Hard filter: global_index=0, forcing role=INTRO")
            # Hard constraint: CONCLUSION paragraphs MUST use CONCLUSION skeletons
            elif input_role == 'CONCLUSION' and has_narrative_role_field:
                conditions.append({"role": "CONCLUSION"})
                if verbose:
                    print(f"     Hard filter: input_role=CONCLUSION, forcing role=CONCLUSION")
            # For BODY/ELABORATION: Prefer BODY role but allow flexibility
            elif input_role in ['ELABORATION', 'BODY'] and has_narrative_role_field:
                conditions.append({"role": "BODY"})
                if verbose:
                    print(f"     Soft filter: input_role={input_role}, preferring role=BODY")

            # Also include paragraph_role if available (for backward compatibility)
            if target_role and has_role_field:
                conditions.append({"paragraph_role": target_role})

            where_clause = self._build_where_clause(conditions)
            if where_clause:
                query_kwargs['where'] = where_clause

            try:
                results = self.collection.query(**query_kwargs)
                if results and results.get('ids') and results['ids'][0] and len(results['ids'][0]) >= 3:
                    candidates = self._extract_candidates_from_results(results)
                    if verbose:
                        print(f"     ✓ Primary query (signature={input_signature}) returned {len(candidates)} candidates")
            except Exception as e:
                if verbose:
                    print(f"     ⚠ Primary query failed: {e}")
                candidates = []

            # Secondary Fallback: Query with DEFINITION signature (keep role filter if hard constraint)
            if len(candidates) < 3 and has_signature_field:
                if verbose:
                    print(f"     ⚠ Primary query returned < 3 candidates, falling back to DEFINITION")
                query_kwargs_fallback = {
                    'query_texts': [query_text],  # Use structural query
                    'n_results': 20
                }
                conditions_fallback = [{"signature": "DEFINITION"}]
                # Keep role filter if it was a hard constraint (INTRO for first, CONCLUSION for conclusion)
                if global_index == 0 and has_narrative_role_field:
                    conditions_fallback.append({"role": "INTRO"})
                elif input_role == 'CONCLUSION' and has_narrative_role_field:
                    conditions_fallback.append({"role": "CONCLUSION"})
                # For BODY/ELABORATION, drop role filter in fallback to allow more flexibility
                if target_role and has_role_field:
                    conditions_fallback.append({"paragraph_role": target_role})

                where_clause_fallback = self._build_where_clause(conditions_fallback)
                if where_clause_fallback:
                    query_kwargs_fallback['where'] = where_clause_fallback

                try:
                    results = self.collection.query(**query_kwargs_fallback)
                    if results and results.get('ids') and results['ids'][0] and len(results['ids'][0]) >= 3:
                        candidates = self._extract_candidates_from_results(results)
                        effective_signature = 'DEFINITION'
                        if verbose:
                            print(f"     ✓ Secondary fallback (DEFINITION) returned {len(candidates)} candidates")
                except Exception as e:
                    if verbose:
                        print(f"     ⚠ Secondary fallback failed: {e}")
                    candidates = []

            # Emergency Fallback: No metadata filter (pure vector search)
            if len(candidates) < 3:
                if verbose:
                    print(f"     ⚠ Secondary fallback returned < 3 candidates, using emergency fallback (no filter)")
                query_kwargs_emergency = {
                    'query_texts': [query_text],  # Use structural query
                    'n_results': 20
                }
                # Only add role filter if available, but no signature filter
                if target_role and has_role_field:
                    query_kwargs_emergency['where'] = {'paragraph_role': target_role}

                try:
                    results = self.collection.query(**query_kwargs_emergency)
                    if results and results.get('ids') and results['ids'][0]:
                        candidates = self._extract_candidates_from_results(results)
                        effective_signature = None  # No signature filter used
                        if verbose:
                            print(f"     ✓ Emergency fallback returned {len(candidates)} candidates")
                except Exception as e:
                    if verbose:
                        print(f"     ⚠ Emergency fallback failed: {e}")
                    candidates = []

        except Exception as e:
            if verbose:
                print(f"     ⚠ Error retrieving candidates: {e}, using empty candidate list")
            candidates = []

        if verbose:
            print(f"     Retrieved {len(candidates)} style candidates")
            print(f"     Effective signature: {effective_signature or 'None (emergency fallback)'}")

        # Graph-Based Skeleton Building: Use input graph structure instead of guessing
        if input_graph and input_graph.get('mermaid'):
            if verbose:
                print(f"     🕸️  Building skeleton from input graph structure")

            # Harvest style vocabulary from candidates
            style_vocab = self._harvest_style_vocab(candidates, verbose=verbose)

            # Build skeleton by traversing the input graph
            graph_skeleton = self._build_skeleton_from_graph(
                input_graph,
                style_vocab,
                len(propositions),
                global_signature=input_signature,  # Pass global signature for contextual override
                verbose=verbose
            )

            if graph_skeleton:
                if verbose:
                    print(f"     ✓ Built skeleton from graph: {graph_skeleton[:80]}...")
                return {
                    'style_metadata': {
                        'mermaid': input_graph.get('mermaid', f"graph LR; {' '.join([f'P{i} --> P{i+1}' for i in range(len(propositions)-1)])}" if len(propositions) > 1 else f"graph LR; P0"),
                        'skeleton': graph_skeleton,
                        'intent': input_intent,
                        'signature': input_signature,
                        'node_count': len(propositions),
                        'edge_types': ['graph_based'],
                        'is_graph_based': True
                    },
                    'node_mapping': {f'P{i}': prop for i, prop in enumerate(propositions)},
                    'intent': input_intent,
                    'signature': input_signature,
                    'source_method': 'graph_traversal'  # Tag as graph-based
                }

        # Fallback: If graph-based building fails, use statistical synthesis
        STRUCTURAL_DISTANCE_THRESHOLD = 0.4  # If best match distance > 0.4, synthesize from candidates
        if candidates:
            best_distance = candidates[0].get('distance', float('inf'))
            if best_distance > STRUCTURAL_DISTANCE_THRESHOLD:
                if verbose:
                    print(f"     ⚠ Graph-based building unavailable, synthesizing statistical template")

                # Synthesize template from candidates using statistical analysis
                synthesized_skeleton = self._synthesize_template_from_candidates(
                    candidates=candidates,  # Pass the full list of 20 candidates
                    input_signature=input_signature,
                    num_propositions=len(propositions),
                    input_intent=input_intent,
                    verbose=verbose
                )

                if synthesized_skeleton:
                    if verbose:
                        print(f"     ✓ Synthesized statistical template: {synthesized_skeleton[:80]}...")
                    return {
                        'style_metadata': {
                            'mermaid': f"graph LR; {' '.join([f'P{i} --> P{i+1}' for i in range(len(propositions)-1)])}" if len(propositions) > 1 else f"graph LR; P0",
                            'skeleton': synthesized_skeleton,
                            'intent': input_intent,
                            'signature': input_signature,
                            'node_count': len(propositions),
                            'edge_types': ['synthesized'],
                            'is_synthesized': True
                        },
                        'node_mapping': {f'P{i}': prop for i, prop in enumerate(propositions)},
                        'intent': input_intent,
                        'signature': input_signature,
                        'source_method': 'statistical_synthesis'  # Tag as statistical synthesis
                    }
                else:
                    # Fallback to generic if synthesis fails
                    if verbose:
                        print(f"     ⚠ Synthesis failed, falling back to generic template")
                    generic_skeleton = self._generate_generic_template(input_signature, len(propositions))
                    if generic_skeleton:
                        return {
                            'style_metadata': {
                                'mermaid': f"graph LR; {' '.join([f'P{i} --> P{i+1}' for i in range(len(propositions)-1)])}" if len(propositions) > 1 else f"graph LR; P0",
                                'skeleton': generic_skeleton,
                                'intent': input_intent,
                                'signature': input_signature,
                                'node_count': len(propositions),
                                'edge_types': ['generic'],
                                'is_generic': True
                            },
                            'node_mapping': {f'P{i}': prop for i, prop in enumerate(propositions)},
                            'intent': input_intent,
                            'signature': input_signature,
                            'source_method': 'generic_template'  # Tag as generic fallback
                        }

        # Step B: The Grafter - Call LLM to construct custom blueprint via topological grafting
        # Format indexed propositions
        indexed_propositions = []
        for i, prop in enumerate(propositions):
            indexed_propositions.append(f"P{i}: {prop}")
        indexed_propositions_text = "\n".join(indexed_propositions)

        # Format candidates
        candidates_list = []
        if candidates:
            for i, candidate in enumerate(candidates):
                skeleton = candidate.get('skeleton', '')
                intent = candidate.get('intent', 'UNKNOWN')
                distance = candidate.get('distance', float('inf'))
                candidates_list.append(
                    f"Candidate {i+1}:\n"
                    f"  Intent: {intent}\n"
                    f"  Skeleton: {skeleton}\n"
                    f"  Distance: {distance:.3f}"
                )
        else:
            candidates_list.append("No style candidates available. Create a skeleton from scratch.")

        candidates_text = "\n\n".join(candidates_list)

        system_prompt = f"""You are a Template Selector (NOT a Fiction Writer).
Your Goal: Select the ONE candidate that BEST matches the Input Logic ({input_intent}, {input_signature}). Do NOT synthesize new skeletons.
Constraints:
1. DO NOT invent new skeletons. Only select from the provided Candidates.
2. DO NOT write generic English (e.g., "In reality," "However").
3. HARVEST the exact connectors and rhetorical structures from the Candidates.
4. If no candidate matches well, you may SPLICE two candidates together, but ONLY use phrases that exist in the Candidates.
5. **CRITICAL:** If the Input Logic does not match any candidate well, return a simple structure that preserves the Input Propositions without adding invented connectors."""

        # Build style guidance from style_profile if available
        style_guidance = ""
        if style_profile:
            style_description = style_profile.get('description') or style_profile.get('style_description')
            tone_markers = style_profile.get('tone_markers', [])
            if style_description:
                style_guidance = f"\n**AUTHOR'S VOICE:** {style_description}\n"
            if tone_markers:
                if isinstance(tone_markers, list):
                    style_guidance += f"**TONE MARKERS:** {', '.join(tone_markers)}\n"
                else:
                    style_guidance += f"**TONE MARKERS:** {tone_markers}\n"

        # Build previous context section if available
        prev_context_section = ""
        if prev_paragraph_summary:
            prev_context_section = f"""
**PREVIOUS PARAGRAPH CONTEXT:**
{prev_paragraph_summary}

**CONTEXT INSTRUCTION:** Use connectors that flow naturally from the previous paragraph. For example, if the previous paragraph ended with a question, use connectors like "This question leads us to..." or "To answer this...". If it ended with a premise, use connectors like "Following this logic..." or "Building on this foundation...".
"""

        user_prompt = f"""**INPUT PROPOSITIONS:**
{indexed_propositions_text}

**STYLE CANDIDATES (The Spare Parts):**
{candidates_text}
{style_guidance}{prev_context_section}
**TASK:**
1. **Select Best Matching Candidate:**
   - The Input Logic is: {input_intent} with signature {input_signature}.
   - Find the candidate whose skeleton structure BEST matches this logic.
   - **CRITICAL:** Do NOT invent a new skeleton. If none match well, select the closest one and adapt minimally.
   - **If NO candidate matches:** Return a simple structure like "[P0] and [P1]" that preserves facts without style.

2. **Harvest Connectors (The Frankenstein Step):**
   - Look through the 20 candidates for phrases that match this topology.
   - *Example Match:* If Input is "Misconception", find a Candidate that says: *"Some comrades naively imagine..."* or *"It is a fundamental error to view..."*
   - *Example Match:* If Input is "Correction", find a Candidate that says: *"...whereas the objective reality is..."*
   - **Seek:** "It is a fundamental error...", "We must strictly distinguish...", "The objective reality manifests..."
   - **Avoid:** "Basically", "In reality", "However" (unless a candidate explicitly uses them).

3. **MERGE REDUNDANCIES:**
   - **CRITICAL:** If multiple propositions express the same fact (e.g., "Stalin coined the term" and "The term was coined by Stalin"), you MUST merge them into a SINGLE slot.
   - Do not create repetitive structures. Redundant propositions should share one `[P#]` slot.
   - Example: If P0 and P1 are redundant, use `[P0]` for both and skip `[P1]` in the skeleton.

4. **Construct the Skeleton:**
   - Stitch these harvested phrases around the Input Slots `[P0]`, `[P1]`.
   - **CRITICAL:** You must use the **exact vocabulary** of the candidates (e.g., use "constitutes", "manifests as", "stem from" instead of "is").
   - If you must combine phrases from multiple candidates, SPLICE them together (e.g., Candidate 2's opening + Candidate 14's transition).
   - **Constraint:** The final skeleton MUST consist of **Real Author Phrases** + `[P#]` slots.

5. **Fallback:**
   - If you absolutely cannot find a match, you may adapt a candidate, but you MUST maintain the **dense, archaic, or revolutionary tone** of the group.

**OUTPUT JSON:**
{{
  "selected_source_indices": [2, 14],
  "revised_skeleton": "It is a naive assumption among some to view [P0] as [P1]; on the contrary, the objective reality [P2]...",
  "rationale": "Spliced Candidate 2 (Misconception Opening) with Candidate 14 (Dialectical Correction)."
}}"""

        try:
            response = self.llm_provider.call(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                model_type="editor",
                require_json=True,
                temperature=0.3,
                max_tokens=800
            )

            # Strip markdown code blocks
            response = self._strip_markdown_code_blocks(response)

            # Parse JSON
            result = json.loads(response)
            revised_skeleton = result.get('revised_skeleton', '')
            rationale = result.get('rationale', '')
            # Support both old format (selected_donor_index) and new format (selected_source_indices)
            selected_source_indices = result.get('selected_source_indices', [])
            if not selected_source_indices:
                # Fallback to old format for backward compatibility
                selected_donor_index = result.get('selected_donor_index', 0)
                if selected_donor_index is not None:
                    selected_source_indices = [selected_donor_index]
            original_donor_connectors = result.get('original_donor_connectors', [])

            if verbose:
                print(f"     ✓ Assembler created skeleton: {revised_skeleton[:80]}...")
                if rationale:
                    print(f"     Rationale: {rationale[:100]}...")
                if selected_source_indices:
                    indices_str = ', '.join(map(str, selected_source_indices[:5]))
                    if len(selected_source_indices) > 5:
                        indices_str += "..."
                    print(f"     Source candidate indices: [{indices_str}]")
                if original_donor_connectors:
                    connectors_str = ', '.join(original_donor_connectors[:3])
                    if len(original_donor_connectors) > 3:
                        connectors_str += "..."
                    print(f"     Harvested connectors: {connectors_str}")

        except (json.JSONDecodeError, Exception) as e:
            if verbose:
                print(f"     ⚠ Assembler LLM call failed: {e}, using fallback skeleton")
            # Fallback: create simple skeleton with all propositions
            revised_skeleton = " ".join([f"[P{i}]" for i in range(len(propositions))])
            if len(propositions) > 1:
                revised_skeleton = " ".join([f"[P{i}]" if i == 0 else f"and [P{i}]" for i in range(len(propositions))])

        # Step C: Create blueprint with direct P0->P0 mapping
        node_mapping = {f'P{i}': f'P{i}' for i in range(len(propositions))}

        # Use effective_signature (may be different from input_signature if fallback was used)
        # Default to DEFINITION if effective_signature is None (emergency fallback)
        final_signature = effective_signature if effective_signature is not None else 'DEFINITION'

        blueprint = {
            'style_metadata': {
                'skeleton': revised_skeleton,
                'node_count': len(propositions),
                'edge_types': [],
                'original_text': '',
                'paragraph_role': target_role,
                'intent': input_intent,
                'signature': final_signature  # Store effective signature
            },
            'node_mapping': node_mapping,
            'intent': input_intent,
            'signature': final_signature,  # Also store at top level for easy access
            'distance': 0.0,  # No distance for synthesized blueprints
            'source_method': 'llm_grafting'  # Tag as LLM-based grafting
        }

        if verbose:
            print(f"     ✓ Blueprint created with {len(node_mapping)} node mappings")

        return blueprint

    def _extract_candidates_from_results(self, results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract candidate dictionaries from ChromaDB query results.

        Args:
            results: ChromaDB query results dict.

        Returns:
            List of candidate dictionaries.
        """
        candidates = []
        ids = results.get('ids', [])
        if not ids or not ids[0]:
            return candidates

        distances = results.get('distances', [[]])[0] if results.get('distances') else [0.0] * len(ids[0])
        metadatas = results.get('metadatas', [[]])[0] if results.get('metadatas') else [{}] * len(ids[0])

        for i, graph_id in enumerate(ids[0]):
            metadata = metadatas[i] if i < len(metadatas) else {}
            distance = distances[i] if i < len(distances) else float('inf')

            candidates.append({
                'id': graph_id,
                'mermaid': metadata.get('mermaid', ''),
                'node_count': metadata.get('node_count', 0),
                'edge_types': metadata.get('edge_types', ''),
                'skeleton': metadata.get('skeleton', ''),
                'intent': metadata.get('intent'),
                'paragraph_role': metadata.get('paragraph_role'),
                'original_text': metadata.get('original_text', ''),
                'distance': distance
            })

        return candidates

    def _determine_role(self, document_context: Optional[Dict[str, Any]]) -> Optional[str]:
        """Determine paragraph role from document context.

        Args:
            document_context: Dict with 'current_index' and 'total_paragraphs'.

        Returns:
            Role string: 'opener', 'body', 'closer', or None.
        """
        if document_context is None:
            return None

        current_index = document_context.get('current_index', 0)
        total_paragraphs = document_context.get('total_paragraphs', 1)

        if current_index == 0:
            return 'opener'
        elif current_index == total_paragraphs - 1:
            return 'closer'
        else:
            return 'body'

    def _get_nlp(self):
        """Get or load spaCy model for POS tagging.

        Returns:
            spaCy Language model or None if unavailable
        """
        if self._nlp_cache is None:
            try:
                from src.utils.nlp_manager import NLPManager
                self._nlp_cache = NLPManager.get_nlp()
            except (OSError, ImportError, RuntimeError):
                # If spaCy not available, mark as unavailable
                self._nlp_cache = False
        return self._nlp_cache if self._nlp_cache is not False else None

    def _is_valid_connector(self, text: str) -> bool:
        """Check if a text fragment is a valid connector using spaCy POS tagging.

        Uses spaCy to reject connectors containing content words (nouns, adjectives,
        non-auxiliary verbs). Rejects segments containing substantive content.

        Args:
            text: Text fragment to validate

        Returns:
            True if valid connector, False if content phrase
        """
        if not text or not text.strip():
            return True  # Empty is valid (just juxtaposition)

        # Heuristic: Connectors shouldn't be overly long
        if len(text.split()) > 5:
            return False

        # Get spaCy model
        nlp = self._get_nlp()
        if nlp is None:
            # Fallback: If spaCy unavailable, use simple length check
            return len(text.split()) <= 3

        # Process text with spaCy
        doc = nlp(text)

        # BAN LIST: Weak subordinators that shouldn't be main structural pivots
        # These create awkward skeletons like "[P0] that [P1] that [P2]"
        banned_lemmas = {'that', 'which', 'who', 'whom', 'whose', 'where', 'when'}

        # Check if connector is primarily a banned subordinator
        token_lemmas = [token.lemma_.lower() for token in doc]
        if len(token_lemmas) <= 2:
            # Short connectors: if they're primarily banned words, reject
            if any(lemma in banned_lemmas for lemma in token_lemmas):
                return False

        for token in doc:
            # 1. Reject Content Words
            # NOUN, PROPN (Proper Noun), ADJ (Adjective), NUM (Number)
            if token.pos_ in ['NOUN', 'PROPN', 'ADJ', 'NUM']:
                return False

            # 2. Reject Action Verbs (but allow Auxiliaries/Copulas like 'is', 'are')
            # We allow 'AUX' (is, are, was). We reject 'VERB' unless it's 'be' lemma.
            if token.pos_ == 'VERB':
                # Only allow 'be' lemma (copula verbs are structural in definitions)
                if token.lemma_ != 'be':
                    return False

        return True

    def _harvest_style_vocab(self, candidates: List[Dict[str, Any]], verbose: bool = False) -> Dict[str, List[str]]:
        """Harvest style vocabulary from candidates, grouping connectors by logic type.

        Extracts connectors from candidate skeletons and classifies them by logical type
        (CONTRAST, CAUSALITY, DEFINITION, etc.) using spaCy analysis.

        Args:
            candidates: List of candidate dictionaries with 'skeleton' keys
            verbose: Enable verbose logging

        Returns:
            Dictionary mapping logic types to lists of connectors found in candidates
        """
        style_vocab = {
            'CONTRAST': [],
            'CAUSALITY': [],
            'DEFINITION': [],
            'SEQUENCE': [],
            'CONDITIONAL': [],
            'LIST': []
        }

        # Logic type keywords for classification
        logic_keywords = {
            'CONTRAST': ['but', 'however', 'whereas', 'yet', 'on the contrary', 'instead', 'rather', 'nevertheless'],
            'CAUSALITY': ['because', 'since', 'therefore', 'thus', 'as a result', 'so', 'consequently', 'causes', 'leads to'],
            'DEFINITION': ['is', 'are', 'defines', 'constitutes', 'means', 'represents', 'refers to'],
            'SEQUENCE': ['then', 'subsequently', 'after', 'next', 'following', 'first', 'second', 'finally'],
            'CONDITIONAL': ['if', 'provided', 'unless', 'when', 'then'],
            'LIST': [',', ';', 'and', 'or', 'nor', 'as well as', 'along with']
        }

        for c in candidates:
            skel = c.get('skeleton', '')
            if not skel:
                continue

            # Extract connectors from skeleton
            conn_matches = re.findall(r'\](.*?)\[', skel, re.DOTALL)
            for cm in conn_matches:
                clean_conn = cm.strip()
                if len(clean_conn) > 1 and self._is_valid_connector(clean_conn):
                    # Analyze grammar FIRST to apply logic-specific constraints
                    grammar_type = self._analyze_connector_grammar(clean_conn)

                    # Classify connector by logic type
                    conn_lower = clean_conn.lower()
                    classified = False

                    for logic_type, keywords in logic_keywords.items():
                        if any(kw in conn_lower for kw in keywords):
                            # Logic-Specific Grammar Constraints
                            # CONTRAST: REJECT VERB connectors (e.g., "are not", "differs")
                            # These are assertions, not logical pivots
                            if logic_type == 'CONTRAST' and grammar_type == 'VERB':
                                if verbose:
                                    print(f"     Rejecting CONTRAST connector '{clean_conn[:30]}...' (VERB type, not a logical pivot)")
                                continue  # Skip "is not", "differs", etc.

                            # DEFINITION: Only accept VERB connectors (is, defines, means)
                            if logic_type == 'DEFINITION' and grammar_type != 'VERB':
                                if verbose:
                                    print(f"     Rejecting DEFINITION connector '{clean_conn[:30]}...' (not VERB type)")
                                continue  # Definitions NEED verbs ("is", "means")

                            style_vocab[logic_type].append(clean_conn)
                            classified = True
                            break

                    # If not classified, use grammar analysis to infer
                    if not classified:
                        # Map grammar to logic (heuristic) with constraints
                        if grammar_type == 'VERB':
                            # Only add to DEFINITION, not CONTRAST
                            style_vocab['DEFINITION'].append(clean_conn)
                        elif grammar_type == 'CONJ':
                            # Default conjunctions to CONTRAST (most common)
                            style_vocab['CONTRAST'].append(clean_conn)
                        elif grammar_type == 'PUNCT':
                            style_vocab['LIST'].append(clean_conn)

        if verbose:
            for logic_type, connectors in style_vocab.items():
                if connectors:
                    print(f"     Harvested {len(connectors)} {logic_type} connectors")

        return style_vocab

    def _select_connector(self, style_vocab: Dict[str, List[str]], logic_type: str, verbose: bool = False) -> str:
        """Select the best connector for a given logic type from style vocabulary.

        Args:
            style_vocab: Dictionary mapping logic types to connector lists
            logic_type: The logic type needed (CONTRAST, CAUSALITY, etc.)
            verbose: Enable verbose logging

        Returns:
            Best connector string for the logic type, or fallback default
        """
        connectors = style_vocab.get(logic_type, [])

        if connectors:
            # Select most frequent connector
            connector_counts = Counter(connectors)
            best_conn = connector_counts.most_common(1)[0][0]
            if verbose:
                print(f"     Selected {logic_type} connector: '{best_conn[:50]}...' (appeared {connector_counts[best_conn]} times)")
            return best_conn

        # Fallback defaults
        fallback_map = {
            'CONTRAST': " but ",
            'CAUSALITY': " because ",
            'DEFINITION': " is ",
            'SEQUENCE': ", then ",
            'CONDITIONAL': ", if ",
            'LIST': ", "
        }
        fallback = fallback_map.get(logic_type, ", ")
        if verbose:
            print(f"     No {logic_type} connectors found, using fallback: '{fallback}'")
        return fallback

    def _build_skeleton_from_graph(
        self,
        input_graph: Dict[str, Any],
        style_vocab: Dict[str, List[str]],
        num_propositions: int,
        global_signature: Optional[str] = None,
        verbose: bool = False
    ) -> Optional[str]:
        """Build skeleton by traversing the input graph and styling edges.

        Constructs a skeleton by walking the input proposition graph and replacing
        abstract edges with stylistic connectors from candidates.

        Args:
            input_graph: Input graph dictionary with 'mermaid' and 'node_map'
            style_vocab: Dictionary mapping logic types to connector lists
            num_propositions: Number of propositions (for validation)
            global_signature: Global paragraph signature (for contextual override)
            verbose: Enable verbose logging

        Returns:
            Synthesized skeleton string or None if building fails
        """
        mermaid = input_graph.get('mermaid', '')
        if not mermaid:
            if verbose:
                print(f"     ⚠ No mermaid graph in input_graph, using fallback")
            return None

        # Parse edges from mermaid
        edges = self._parse_mermaid_edges(mermaid)
        if not edges:
            if verbose:
                print(f"     ⚠ No edges found in mermaid graph, using fallback")
            return None

        # Build node order from edges (topological sort or simple path)
        # Extract all nodes and sort them
        all_nodes = set()
        for source, target, _ in edges:
            all_nodes.add(source)
            all_nodes.add(target)

        # Sort nodes by their numeric index (P0, P1, P2...)
        try:
            ordered_nodes = sorted(all_nodes, key=lambda x: int(re.search(r'\d+', x).group()) if re.search(r'\d+', x) else 999)
        except:
            # Fallback: just use the order they appear
            ordered_nodes = list(all_nodes)

        if verbose:
            print(f"     Graph traversal: {len(ordered_nodes)} nodes, {len(edges)} edges")
            print(f"     Node order: {', '.join(ordered_nodes)}")

        # Build skeleton by walking the path
        parts = []
        used_connectors = set()  # Track used connectors to prevent repetition

        for i, current_node in enumerate(ordered_nodes):
            parts.append(f"[{current_node}]")

            if i < len(ordered_nodes) - 1:
                next_node = ordered_nodes[i + 1]

                # Find edge between current and next node
                edge_label = 'SEQUENCE'  # Default
                for source, target, label in edges:
                    if source == current_node and target == next_node:
                        edge_label = label
                        break

                # CONTEXTUAL OVERRIDE: If paragraph is globally CONTRAST, treat generic SEQUENCE as CONTRAST
                # A contrastive paragraph often structures propositions sequentially, but the logical
                # connection should be adversarial ("but") rather than temporal ("then")
                if global_signature == 'CONTRAST' and edge_label == 'SEQUENCE':
                    if verbose:
                        print(f"     Overriding SEQUENCE edge with CONTRAST (global signature: {global_signature})")
                    edge_label = 'CONTRAST'

                # CONNECTOR CYCLING: Prevent repetition by selecting unused connectors
                connectors_for_logic = style_vocab.get(edge_label, [])

                # Filter invalid connectors (e.g., VERBs in CONTRAST)
                valid_connectors = []
                for conn in connectors_for_logic:
                    # Check if connector is grammatically valid for this logic type
                    grammar_type = self._analyze_connector_grammar(conn)
                    if edge_label == 'CONTRAST' and grammar_type == 'VERB':
                        continue  # Skip VERB connectors for CONTRAST
                    if edge_label == 'DEFINITION' and grammar_type != 'VERB':
                        continue  # Only VERB connectors for DEFINITION
                    valid_connectors.append(conn)

                # SELECT WITH CYCLING: Pick first unused connector
                selected_connector = None
                for conn in valid_connectors:
                    if conn not in used_connectors:
                        selected_connector = conn
                        used_connectors.add(conn)  # Mark as used
                        if verbose:
                            print(f"     Selected {edge_label} connector (cycling): '{conn[:30]}...'")
                        break

                # Fallback: If all valid connectors are used, reset history and pick the best one
                if not selected_connector and valid_connectors:
                    # Reset history for this logic type (allow reuse after all have been used)
                    used_connectors = {c for c in used_connectors if c not in valid_connectors}
                    if valid_connectors:
                        from collections import Counter
                        connector_counts = Counter(valid_connectors)
                        selected_connector = connector_counts.most_common(1)[0][0]
                        used_connectors.add(selected_connector)
                        if verbose:
                            print(f"     All {edge_label} connectors used, resetting and selecting best: '{selected_connector[:30]}...'")

                # Ultimate fallback: Use default connector
                if not selected_connector:
                    selected_connector = self._select_connector(style_vocab, edge_label, verbose=False)
                    if verbose:
                        print(f"     Using fallback {edge_label} connector: '{selected_connector[:30]}...'")

                parts.append(selected_connector)

        skeleton = "".join(parts)

        # Add closing punctuation if not present
        if not skeleton.endswith(('.', '!', '?')):
            skeleton += "."

        if verbose:
            print(f"     ✓ Built skeleton from graph: {skeleton[:80]}...")

        return skeleton

    def _analyze_connector_grammar(self, connector_text: str) -> str:
        """Analyze the grammatical role of a connector using spaCy.

        Parses the connector within a dummy sentence context to determine its
        grammatical function. This helps distinguish between structural connectors
        (verbs, conjunctions) and punctuation.

        Args:
            connector_text: Text fragment to analyze (e.g., ", ", " is ", " but ")

        Returns:
            'VERB' (Action/Copula), 'CONJ' (Logic/Conjunction), 'PUNCT' (Punctuation), or 'OTHER'
        """
        if not connector_text or not connector_text.strip():
            return 'PUNCT'

        # Get spaCy model
        nlp = self._get_nlp()
        if nlp is None:
            # Fallback: Simple heuristic if spaCy unavailable
            if any(word in connector_text.lower() for word in ['is', 'are', 'was', 'were', 'be', 'defines', 'constitutes']):
                return 'VERB'
            elif any(word in connector_text.lower() for word in ['but', 'however', 'whereas', 'and', 'or', 'then']):
                return 'CONJ'
            elif ',' in connector_text or ';' in connector_text:
                return 'PUNCT'
            return 'OTHER'

        # Contextualize to help spaCy parse correctly
        # Create a dummy sentence: "Alpha <connector> Beta"
        dummy_sent = f"Alpha {connector_text} Beta"
        doc = nlp(dummy_sent)

        # Find tokens between "Alpha" (index 0) and "Beta" (last token)
        # The connector tokens are doc[1:-1]
        connector_tokens = [token for token in doc if token.i > 0 and token.i < len(doc) - 1]

        if not connector_tokens:
            return 'PUNCT'

        # Priority mapping: Higher number = more significant
        pos_priority = {
            'VERB': 4,
            'AUX': 4,  # Auxiliary verbs (is, are, was) are also structural
            'CCONJ': 3,  # Coordinating conjunctions (and, but, or)
            'SCONJ': 3,  # Subordinating conjunctions (because, if, when)
            'ADV': 2,  # Adverbs (however, therefore) - treat as conjunctions
            'PUNCT': 1
        }

        best_pos = 'OTHER'
        max_prio = 0

        for token in connector_tokens:
            prio = pos_priority.get(token.pos_, 0)
            if prio > max_prio:
                max_prio = prio
                # Map spaCy POS tags to our generic categories
                if token.pos_ in ['VERB', 'AUX']:
                    best_pos = 'VERB'
                elif token.pos_ in ['CCONJ', 'SCONJ']:
                    best_pos = 'CONJ'
                elif token.pos_ == 'ADV':
                    # Adverbs like "however", "therefore" function as conjunctions
                    best_pos = 'CONJ'
                elif token.pos_ == 'PUNCT':
                    best_pos = 'PUNCT'

        return best_pos

    def _synthesize_contrast_structure(
        self,
        openings: List[str],
        closings: List[str],
        pivot_connectors: List[str],
        all_connectors: List[str],
        num_propositions: int,
        verbose: bool = False
    ) -> str:
        """Synthesize a deterministic CONTRAST structure: (Misconceptions) [Pivot] (Reality).

        Logic: Group 1 (Misconceptions) [Pivot] Group 2 (Reality).
        Pivot: Find "but/however" in candidates; default to "; however, ".
        Connectors: Use "or" for Group 1, "and" for Group 2.

        Args:
            openings: List of opening phrases from candidates
            closings: List of closing phrases from candidates
            pivot_connectors: List of pivot connectors (but, however, etc.)
            all_connectors: All connectors found (for fallback)
            num_propositions: Number of propositions
            verbose: Enable verbose logging

        Returns:
            Synthesized contrast skeleton
        """
        parts = []

        # Select best opening
        best_opening = ""
        if openings:
            opening_counts = Counter(openings)
            best_opening = opening_counts.most_common(1)[0][0]

        if best_opening:
            parts.append(best_opening)

        if num_propositions >= 2:
            mid_point = num_propositions // 2
            misconception_slots = list(range(mid_point))
            correction_slots = list(range(mid_point, num_propositions))

            # Build misconception group (joined with "or"/"nor")
            for i, slot_idx in enumerate(misconception_slots):
                parts.append(f"[P{slot_idx}]")
                if i < len(misconception_slots) - 1:
                    parts.append(" or ")

            # CRITICAL: Force pivot connector between groups
            # Look for pivot connector in candidates
            pivot_conn = None
            if pivot_connectors:
                pivot_counts = Counter(pivot_connectors)
                pivot_conn = pivot_counts.most_common(1)[0][0]
            else:
                # Fallback: look for "but" or "however" in all connectors
                for conn in all_connectors:
                    conn_lower = conn.lower()
                    if 'but' in conn_lower or 'however' in conn_lower:
                        pivot_conn = conn
                        break

            # Default pivot if not found
            if not pivot_conn:
                pivot_conn = "; however, "  # Default pivot as specified
                if verbose:
                    print(f"     No pivot connector found, using default: '{pivot_conn}'")
            else:
                if verbose:
                    print(f"     Selected pivot connector: '{pivot_conn[:50]}...'")

            parts.append(f" {pivot_conn} ")

            # Build correction group (joined with "and")
            for i, slot_idx in enumerate(correction_slots):
                parts.append(f"[P{slot_idx}]")
                if i < len(correction_slots) - 1:
                    parts.append(" and ")
        else:
            # Single proposition
            parts.append("[P0]")

        # Select best closing
        best_closing = "."
        if closings:
            closing_counts = Counter(closings)
            best_closing = closing_counts.most_common(1)[0][0]

        if best_closing:
            parts.append(best_closing)

        skeleton = "".join(parts)

        # Ensure it ends with punctuation
        if not skeleton.endswith(('.', '!', '?')):
            skeleton += "."

        if verbose:
            print(f"     ✓ Synthesized CONTRAST structure: {skeleton[:80]}...")

        return skeleton

    def _synthesize_definition_structure(
        self,
        openings: List[str],
        closings: List[str],
        grammar_buckets: Dict[str, List[str]],
        all_connectors: List[str],
        num_propositions: int,
        verbose: bool = False
    ) -> str:
        """Synthesize a deterministic DEFINITION structure: [P0] [VERB] [P1], [P2]...

        Logic: [P0] [VERB] [P1], [P2]...
        Verb: Find "is/defines/constitutes" in candidates; default to " is ".

        Args:
            openings: List of opening phrases from candidates
            closings: List of closing phrases from candidates
            grammar_buckets: Grammar-classified connector buckets
            all_connectors: All connectors found (for fallback)
            num_propositions: Number of propositions
            verbose: Enable verbose logging

        Returns:
            Synthesized definition skeleton
        """
        parts = []

        # Select best opening
        best_opening = ""
        if openings:
            opening_counts = Counter(openings)
            best_opening = opening_counts.most_common(1)[0][0]

        if best_opening:
            parts.append(best_opening)

        # DEFINITION requires VERB connector (is, are, defines, constitutes)
        verb_connectors = grammar_buckets.get('VERB', [])
        best_def_verb = " is "  # Default copula as specified

        if verb_connectors:
            verb_counts = Counter(verb_connectors)
            best_def_verb = verb_counts.most_common(1)[0][0]
            if verbose:
                print(f"     Selected definition verb: '{best_def_verb[:50]}...'")
        else:
            # Fallback: look for definition verbs in all connectors
            definition_verbs = ['is', 'are', 'defines', 'constitutes', 'means', 'represents']
            for conn in all_connectors:
                conn_lower = conn.lower()
                for verb in definition_verbs:
                    if verb in conn_lower:
                        best_def_verb = conn
                        break
                if best_def_verb != " is ":
                    break

        if best_def_verb == " is " and verbose:
            print(f"     Using default copula 'is' for definition")

        # Build definition structure: [P0] is [P1], [P2]...
        if num_propositions >= 2:
            parts.append("[P0]")
            parts.append(best_def_verb)
            parts.append("[P1]")
            # Additional attributes with commas (not list enumeration, but definition attributes)
            for i in range(2, num_propositions):
                parts.append(", ")
                parts.append(f"[P{i}]")
        else:
            parts.append("[P0]")

        # Select best closing
        best_closing = "."
        if closings:
            closing_counts = Counter(closings)
            best_closing = closing_counts.most_common(1)[0][0]

        if best_closing:
            parts.append(best_closing)

        skeleton = "".join(parts)

        # Ensure it ends with punctuation
        if not skeleton.endswith(('.', '!', '?')):
            skeleton += "."

        if verbose:
            print(f"     ✓ Synthesized DEFINITION structure: {skeleton[:80]}...")

        return skeleton

    def _synthesize_template_from_candidates(
        self,
        candidates: List[Dict[str, Any]],
        input_signature: str,
        num_propositions: int,
        input_intent: str,
        verbose: bool = False
    ) -> str:
        """Synthesizes a new skeleton with exactly `num_propositions` slots using logic-aware structuring.

        Algorithm:
        1. Harvest Components: Extract openers, connectors (classified by type), closers from candidates
        2. Filter by Logic: Classify connectors into pivots, list separators, sequence markers
        3. Dynamic Assembly: Construct grammatically structured template based on signature

        Signature-Specific Structures:
        - CONTRAST: Split into misconceptions (first half) and corrections (second half)
        - LIST/DEFINITION: Standard enumeration with commas and finalizer
        - SEQUENCE: Sequential connectors (then, subsequently, etc.)
        - CAUSALITY: Cause -> Effect structure
        - Fallback: Clean list structure (always grammatically safe)

        Args:
            candidates: List of candidate dictionaries with 'skeleton' keys
            input_signature: Logical signature (CONTRAST, CAUSALITY, DEFINITION, etc.)
            num_propositions: Number of input propositions (determines slot count)
            input_intent: Rhetorical intent (DEFINITION, ARGUMENT, etc.)
            verbose: Enable verbose logging

        Returns:
            Synthesized skeleton string with exactly num_propositions slots, grammatically structured
        """

        if num_propositions == 0:
            return ""

        # Fallback if no candidates
        if not candidates or len(candidates) < 3:
            if verbose:
                print(f"     ⚠ Too few candidates ({len(candidates) if candidates else 0}), using generic template")
            return self._generate_generic_template(input_signature, num_propositions)

        # 1. Define Connector Classifications
        pivot_terms = ['but', 'however', 'whereas', 'yet', 'on the contrary', 'instead', 'rather', 'rather than', 'nevertheless']
        list_separator_terms = [',', ';', 'and', 'or', 'nor', 'as well as', 'along with']
        sequence_terms = ['then', 'subsequently', 'after', 'next', 'following', 'first', 'second', 'finally']
        finalizer_terms = ['and', 'finally', 'ultimately', 'in conclusion']

        # 2. Extract Components from Candidates
        openings = []
        all_connectors = []
        closings = []

        for c in candidates:
            skel = c.get('skeleton', '')
            if not skel:
                continue

            # Opener: Text before first [P#]
            op_match = re.match(r'^(.*?)(?=\[P\d+\])', skel, re.DOTALL)
            if op_match:
                opening = op_match.group(1).strip()
                # Filter out trivial openings (just whitespace/punctuation)
                if opening and len(opening) > 1:
                    openings.append(opening)

            # Closer: Text after last [P#]
            cl_match = re.search(r'\[P\d+\]([^\[]*)$', skel, re.DOTALL)
            if cl_match:
                closing = cl_match.group(1).strip()
                # Filter out trivial closings
                if closing and len(closing) > 0:
                    closings.append(closing)

            # Connectors: Text between ] and [
            conn_matches = re.findall(r'\](.*?)\[', skel, re.DOTALL)
            for cm in conn_matches:
                clean_conn = cm.strip()
                # Filter out trivial connectors (just whitespace)
                if len(clean_conn) > 1:
                    # Apply strict connector validation to prevent content phrases
                    if self._is_valid_connector(clean_conn):
                        all_connectors.append(clean_conn)

        # 3. Grammar-Driven Bucketization: Classify connectors by grammatical role
        grammar_buckets = {
            'VERB': [],  # Verbs/Auxiliaries (is, are, defines, constitutes)
            'CONJ': [],  # Conjunctions/Adverbs (but, however, and, then)
            'PUNCT': []  # Punctuation (commas, semicolons)
        }

        # Analyze each connector's grammatical role using spaCy
        for conn in all_connectors:
            grammar_role = self._analyze_connector_grammar(conn)
            if grammar_role in grammar_buckets:
                grammar_buckets[grammar_role].append(conn)

        # Legacy classification for backward compatibility (used for specific signature logic)
        pivot_connectors = []
        list_separators = []
        sequence_connectors = []
        finalizers = []

        for conn in all_connectors:
            conn_lower = conn.lower()
            if any(term in conn_lower for term in pivot_terms):
                pivot_connectors.append(conn)
            elif any(term in conn_lower for term in sequence_terms):
                sequence_connectors.append(conn)
            elif any(term in conn_lower for term in finalizer_terms):
                finalizers.append(conn)
            elif any(term in conn_lower for term in list_separator_terms) or ',' in conn or ';' in conn:
                list_separators.append(conn)

        # 4. Statistical Selection
        best_opening = ""
        if openings:
            opening_counts = Counter(openings)
            best_opening = opening_counts.most_common(1)[0][0]
            if verbose:
                print(f"     Selected opening: '{best_opening[:50]}...' (appeared {opening_counts[best_opening]} times)")

        best_closing = "."
        if closings:
            closing_counts = Counter(closings)
            best_closing = closing_counts.most_common(1)[0][0]
            if verbose:
                print(f"     Selected closing: '{best_closing[:50]}...' (appeared {closing_counts[best_closing]} times)")

        # 4. Grammar-Driven Selection: Map signature to grammatical requirement
        signature_grammar_map = {
            'DEFINITION': 'VERB',  # Definitions require copula/verb (is, defines, constitutes)
            'CAUSALITY': 'VERB',   # Causality often uses verbs (causes, leads to)
            'CONTRAST': 'CONJ',    # Contrast requires adversative conjunction (but, however)
            'CONDITIONAL': 'CONJ', # Conditionals use conjunctions (if, when)
            'SEQUENCE': 'CONJ',    # Sequences use conjunctions/adverbs (then, next)
            'LIST': 'PUNCT'        # Lists use punctuation (commas)
        }

        required_grammar = signature_grammar_map.get(input_signature, 'CONJ')

        # Select connector from required grammatical bucket
        best_connector = None
        if grammar_buckets[required_grammar]:
            connector_counts = Counter(grammar_buckets[required_grammar])
            best_connector = connector_counts.most_common(1)[0][0]
            if verbose:
                print(f"     Selected {required_grammar} connector: '{best_connector[:50]}...' (appeared {connector_counts[best_connector]} times)")
        else:
            # Fallback: Use safe defaults for each grammatical category
            fallback_map = {
                'VERB': " is ",
                'CONJ': " but " if input_signature == 'CONTRAST' else ", and ",
                'PUNCT': ", "
            }
            best_connector = fallback_map.get(required_grammar, ", ")
            if verbose:
                print(f"     No {required_grammar} connectors found, using fallback: '{best_connector}'")

        # Legacy connector selection (for backward compatibility with specific signature logic)
        best_pivot = None
        if pivot_connectors:
            pivot_counts = Counter(pivot_connectors)
            best_pivot = pivot_counts.most_common(1)[0][0]
            if verbose:
                print(f"     Selected pivot: '{best_pivot[:50]}...' (appeared {pivot_counts[best_pivot]} times)")

        best_list_sep = ", "
        if list_separators:
            list_counts = Counter(list_separators)
            # Prefer commas/semicolons for list separators
            for sep, count in list_counts.most_common():
                if ',' in sep or ';' in sep:
                    best_list_sep = sep
                    break
            else:
                best_list_sep = list_counts.most_common(1)[0][0]
            if verbose:
                print(f"     Selected list separator: '{best_list_sep[:50]}...'")

        best_sequence = None
        if sequence_connectors:
            seq_counts = Counter(sequence_connectors)
            best_sequence = seq_counts.most_common(1)[0][0]
            if verbose:
                print(f"     Selected sequence connector: '{best_sequence[:50]}...'")

        best_finalizer = ", and "
        if finalizers:
            final_counts = Counter(finalizers)
            # Prefer "and" for finalizers
            for fin, count in final_counts.most_common():
                if 'and' in fin.lower():
                    best_finalizer = f", {fin} "
                    break
            else:
                best_finalizer = f", {final_counts.most_common(1)[0][0]} "
            if verbose:
                print(f"     Selected finalizer: '{best_finalizer[:50]}...'")

        # 5. FORCE Deterministic Logic for Complex Signatures
        # Stop statistical guessing for CONTRAST and DEFINITION - use deterministic structures
        if input_signature == 'CONTRAST':
            return self._synthesize_contrast_structure(
                openings, closings, pivot_connectors, all_connectors, num_propositions, verbose
            )

        if input_signature == 'DEFINITION':
            return self._synthesize_definition_structure(
                openings, closings, grammar_buckets, all_connectors, num_propositions, verbose
            )

        # For other signatures (LIST, SEQUENCE, CAUSALITY), continue with existing logic
        # 6. Construct Template Based on Signature
        parts = []

        if best_opening:
            parts.append(best_opening)

        if input_signature == 'LIST':
            # LIST: Standard enumeration (grammar-driven: accepts PUNCT or CONJ)
            list_conn = best_connector if best_connector and required_grammar == 'PUNCT' else best_list_sep

            for i in range(num_propositions):
                parts.append(f"[P{i}]")
                if i < num_propositions - 2:
                    parts.append(list_conn)
                elif i == num_propositions - 2:
                    parts.append(best_finalizer)

        elif input_signature == 'SEQUENCE':
            # SEQUENCE: Use sequential connectors (grammar-driven: requires CONJ)
            seq_conn = best_connector if best_connector and required_grammar == 'CONJ' else (best_sequence if best_sequence else "then")

            for i in range(num_propositions):
                parts.append(f"[P{i}]")
                if i < num_propositions - 1:
                    parts.append(f", {seq_conn} ")

        elif input_signature == 'CAUSALITY':
            # CAUSALITY: Cause -> Effect structure (grammar-driven: requires VERB or CONJ)
            causality_conn = best_connector if best_connector and required_grammar in ['VERB', 'CONJ'] else (best_pivot if best_pivot else ", because ")

            if num_propositions >= 2:
                parts.append("[P0]")  # Cause
                parts.append(causality_conn)
                # Effects (joined with "and")
                for i in range(1, num_propositions):
                    parts.append(f"[P{i}]")
                    if i < num_propositions - 1:
                        parts.append(" and ")
            else:
                parts.append("[P0]")

        else:
            # Fallback: Use grammar-driven connector or clean list structure
            if verbose:
                print(f"     Using fallback structure for signature: {input_signature}")

            # Use grammar-selected connector if available, otherwise use list separator
            fallback_conn = best_connector if best_connector else best_list_sep

            for i in range(num_propositions):
                parts.append(f"[P{i}]")
                if i < num_propositions - 2:
                    parts.append(fallback_conn)
                elif i == num_propositions - 2:
                    parts.append(best_finalizer)

        if best_closing:
            parts.append(best_closing)

        synthesized = "".join(parts)

        # Clean up any double spaces
        synthesized = re.sub(r'\s+', ' ', synthesized).strip()
        # Fix spacing around punctuation
        synthesized = re.sub(r'\s+([,.;:])', r'\1', synthesized)
        synthesized = re.sub(r'([,.;:])\s*([A-Z\[])', r'\1 \2', synthesized)

        if verbose:
            print(f"     ✓ Synthesized template: {synthesized[:80]}...")

        return synthesized

    def _generate_generic_template(self, signature: str, num_propositions: int) -> Optional[str]:
        """Generate a generic template skeleton based on signature and number of propositions.

        This is a fallback when statistical synthesis is not possible (e.g., too few candidates).

        Args:
            signature: Logical signature (CONTRAST, CAUSALITY, DEFINITION, etc.).
            num_propositions: Number of input propositions.

        Returns:
            Generic skeleton string or None.
        """
        if num_propositions == 0:
            return None

        # Generate placeholders
        placeholders = ' '.join([f'[P{i}]' for i in range(num_propositions)])

        # Signature-based templates (content-blind, structure-focused)
        templates = {
            'CONTRAST': f"It is not [P0], but [P1]." if num_propositions >= 2 else f"[P0].",
            'CAUSALITY': f"Because [P0], [P1]." if num_propositions >= 2 else f"[P0].",
            'DEFINITION': f"[P0] is [P1]." if num_propositions >= 2 else f"[P0].",
            'SEQUENCE': f"First [P0], then [P1]." if num_propositions >= 2 else f"[P0].",
            'CONDITIONAL': f"If [P0], then [P1]." if num_propositions >= 2 else f"[P0].",
            'LIST': placeholders.replace(' ', ', ').replace('[P', '[').replace(']', ']') + '.'
        }

        # Default template for unknown signatures
        default_template = placeholders.replace(' ', ' and ') + '.'

        return templates.get(signature, default_template)

    def _build_where_clause(self, conditions: list) -> Optional[dict]:
        """Constructs a ChromaDB-compatible where clause from a list of conditions.

        Args:
            conditions: List of condition dicts, e.g., [{'role': 'INTRO'}, {'signature': 'CONTRAST'}]

        Returns:
            dict: ChromaDB where clause with $and operator if multiple conditions,
                  single dict if one condition, None if no conditions
        """
        # Filter out empty/None conditions
        valid_conditions = [c for c in conditions if c]

        if not valid_conditions:
            return None

        if len(valid_conditions) == 1:
            return valid_conditions[0]

        return {"$and": valid_conditions}

    def _generate_structural_summary(
        self,
        propositions: List[str],
        input_intent: str,
        input_signature: Optional[str] = None,
        verbose: bool = False
    ) -> Optional[str]:
        """Generate a structural summary (content-blind) for the input propositions.

        Args:
            propositions: List of input proposition strings.
            input_intent: Rhetorical intent (DEFINITION, ARGUMENT, etc.).
            input_signature: Logical signature (CONTRAST, CAUSALITY, etc.).
            verbose: Enable verbose logging.

        Returns:
            Structural summary string or None if generation fails.
        """
        try:
            system_prompt = "You are a Structural Analyst. Describe the logical/rhetorical structure of text, ignoring specific content."

            propositions_text = "\n".join([f"- {prop}" for prop in propositions[:5]])  # Limit to first 5 for efficiency

            user_prompt = f"""Propositions:
{propositions_text}

Intent: {input_intent}
Signature: {input_signature or 'UNKNOWN'}

Task: Describe the **Rhetorical Structure** of these propositions. Ignore specific names, topics, or entities. Use abstract terms like 'Contrast', 'Definition', 'Attribution', 'Conditional', 'Causality', 'Sequence', 'List'. Keep it under 15 words. Focus on the LOGICAL FORM, not the CONTENT.

Examples:
- "A definition of a concept followed by its historical origin."
- "A set of misconceptions followed by a clarification."
- "Historical attribution of a creation action to a named agent."
- "A contrast between a false view and a true statement."

Output: Return ONLY the structural summary, no JSON, no explanation."""

            response = self.llm_provider.call(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                model_type="editor",
                require_json=False,
                temperature=0.2,
                max_tokens=50
            )

            summary = response.strip()
            # Remove quotes if present
            if summary.startswith('"') and summary.endswith('"'):
                summary = summary[1:-1]

            return summary if summary else None

        except Exception as e:
            if verbose:
                print(f"     ⚠ Structural summary generation failed: {e}")
            return None

    def get_best_match(
        self,
        input_graph_data: Dict[str, Any],
        document_context: Optional[Dict[str, Any]] = None,
        verbose: bool = False
    ) -> Dict[str, Any]:
        """Get the best matching style graph for the input graph.

        Args:
            input_graph_data: Dict with 'description', 'mermaid', 'node_map', 'node_count'.
            document_context: Optional dict with 'current_index' and 'total_paragraphs' for role filtering.

        Returns:
            Dict with 'style_mermaid', 'node_mapping', 'style_metadata', and 'distance'.

        Raises:
            ValueError: If input_graph_data is missing required fields or no matches found.
        """
        # Validate input
        if 'description' not in input_graph_data:
            raise ValueError("input_graph_data must contain 'description' field")

        input_node_count = input_graph_data.get('node_count', len(input_graph_data.get('node_map', {})))
        input_intent = input_graph_data.get('intent')
        input_description = input_graph_data.get('description', '')

        if verbose:
            print(f"  🔍 Matching graph: {input_description[:80]}...")
            print(f"     Input intent: {input_intent}, Node count: {input_node_count}")

        # Step A: Context Determination & Semantic Search
        target_role = self._determine_role(document_context)

        if verbose and target_role:
            print(f"     Target role: {target_role}")

        # Query ChromaDB with larger top_k for diversity, then select top 5 diverse candidates
        query_text = input_graph_data['description']
        retrieval_k = 20  # Retrieve more candidates for diversity selection
        top_k = 5  # Final number of candidates to pass to synthesis

        # Step A.0: Check if metadata fields exist by sampling collection
        # This prevents errors when filtering on non-existent fields
        has_role_field = False
        has_intent_field = False
        try:
            sample = self.collection.get(limit=1, include=['metadatas'])
            if sample.get('metadatas') and len(sample['metadatas']) > 0:
                sample_meta = sample['metadatas'][0]
                has_role_field = 'paragraph_role' in sample_meta
                has_intent_field = 'intent' in sample_meta
        except Exception as e:
            # If we can't sample, assume fields don't exist (safer)
            print(f"Warning: Could not check metadata schema: {e}")

        # Step A.1: Try query with role filter if target_role is set AND field exists
        results = None
        query_kwargs = {
            'query_texts': [query_text],
            'n_results': retrieval_k  # Retrieve more for diversity
        }

        # Attempt role filtering only if target_role is set AND field exists in metadata
        if target_role and has_role_field:
            query_kwargs['where'] = {"paragraph_role": target_role}
            try:
                results = self.collection.query(**query_kwargs)
                # Validate results structure
                if (not results or
                    not results.get('ids') or
                    not results['ids'][0] or
                    len(results['ids'][0]) == 0):
                    results = None
            except Exception as e:
                # Metadata field might not exist or query syntax error
                print(f"Warning: Query with role filter '{target_role}' failed: {e}")
                results = None

        # Step A.2: Fallback to query without role filter (pure vector search)
        if not results:
            query_kwargs.pop('where', None)  # Remove where clause
            try:
                results = self.collection.query(**query_kwargs)
            except Exception as e:
                print(f"Warning: Query without filter also failed: {e}")
                # Last resort: try with minimal parameters
                try:
                    results = self.collection.query(
                        query_texts=[query_text],
                        n_results=min(retrieval_k, 10)  # Reduce retrieval_k for last attempt
                    )
                except Exception as e2:
                    print(f"Error: All query attempts failed. Last error: {e2}")
                    raise ValueError("No style graphs found in collection - ChromaDB query failed")

        if not results or not results.get('ids') or not results['ids'][0]:
            raise ValueError("No style graphs found in collection")

        # Step A.3: Extract candidates with distances
        ids = results['ids'][0]
        distances = results['distances'][0] if results.get('distances') else [0.0] * len(ids)
        metadatas = results['metadatas'][0] if results.get('metadatas') else [{}] * len(ids)
        documents = results['documents'][0] if results.get('documents') else [''] * len(ids)

        # Step A.4: Create candidate list and apply intent boosting
        candidates = []
        for i, graph_id in enumerate(ids):
            metadata = metadatas[i] if i < len(metadatas) else {}
            distance = distances[i] if i < len(distances) else float('inf')
            style_intent = metadata.get('intent')

            # Intent-based reranking: boost candidates with matching intent
            priority_score = distance
            intent_match = False
            if input_intent and style_intent:
                # Normalize intents for comparison (case-insensitive)
                if input_intent.upper() == style_intent.upper():
                    # Intent match: multiply distance by 0.5 (strong boost)
                    priority_score = distance * 0.5
                    intent_match = True

            candidates.append({
                'id': graph_id,
                'mermaid': metadata.get('mermaid', ''),
                'node_count': metadata.get('node_count', 0),
                'edge_types': metadata.get('edge_types', ''),
                'skeleton': metadata.get('skeleton', ''),
                'intent': style_intent,
                'paragraph_role': metadata.get('paragraph_role'),
                'original_text': metadata.get('original_text', ''),
                'distance': distance,
                'priority_score': priority_score,
                'intent_match': intent_match
            })

        # Step A.5: Sort by priority_score (lowest is best), then by distance as tiebreaker
        candidates.sort(key=lambda x: (x['priority_score'], x['distance']))

        if verbose:
            print(f"     Retrieved {len(candidates)} candidates from ChromaDB")
            if len(candidates) > 0:
                print(f"     Top 5 candidates:")
                for i, c in enumerate(candidates[:5]):
                    intent = c.get('intent', 'UNKNOWN')
                    distance = c.get('distance', float('inf'))
                    node_count = c.get('node_count', 0)
                    skeleton_preview = (c.get('skeleton', '')[:50] + '...') if len(c.get('skeleton', '')) > 50 else c.get('skeleton', '')
                    print(f"       {i+1}. Intent: {intent}, Distance: {distance:.3f}, Nodes: {node_count}, Skeleton: {skeleton_preview}")

        # Step B: Topological Filtering (Isomorphism Check) with Intent Prioritization
        # Filter candidates that meet node count constraint
        valid_candidates = [c for c in candidates if c['node_count'] >= input_node_count]

        if not valid_candidates:
            # Overflow handling: pick largest available style graph
            if candidates:
                print(f"Warning: No style graph meets node count constraint ({input_node_count}). "
                      f"Using largest available graph ({max(c['node_count'] for c in candidates)} nodes).")
                # Sort by node_count descending, then by priority_score
                candidates.sort(key=lambda x: (-x['node_count'], x['priority_score']))
                selected_candidates = candidates[:top_k]  # Take top_k for synthesis
            else:
                raise ValueError("No style graphs available in collection")
        else:
            # Select top_k diverse candidates (prioritize intent diversity)
            selected_candidates = self._select_diverse_candidates(valid_candidates, top_k, input_intent)

        if verbose:
            print(f"     Selected {len(selected_candidates)} diverse candidates for synthesis")
            intents = [c.get('intent', 'UNKNOWN') for c in selected_candidates]
            print(f"     Candidate intents: {', '.join(intents)}")

        # Step B.1: Synthesize best fit using LLM
        try:
            selected_candidate = self._synthesize_best_fit(input_graph_data, selected_candidates, verbose=verbose)
        except Exception as e:
            print(f"Warning: Synthesis failed: {e}, falling back to top candidate")
            # Safety fallback: use top candidate (lowest distance)
            if valid_candidates:
                selected_candidate = valid_candidates[0]
            elif candidates:
                selected_candidate = candidates[0]
            else:
                raise ValueError("No style graphs available in collection")

        # Step C: The Projection (Node Mapping)
        style_mermaid = selected_candidate['mermaid']
        style_node_count = selected_candidate['node_count']
        input_mermaid = input_graph_data.get('mermaid', '')
        input_node_map = input_graph_data.get('node_map', {})

        # Extract style node names
        style_nodes = self._parse_mermaid_nodes(style_mermaid)

        # Call LLM to map nodes
        if verbose:
            print(f"     📍 Mapping {len(input_node_map)} input nodes to {len(style_nodes)} style nodes")
            print(f"        Input nodes: {', '.join(input_node_map.keys())}")
            print(f"        Style nodes: {', '.join(style_nodes)}")

        system_prompt = (
            "You are a Graph Mapper. Your task is to fit user content into "
            "an author's structural template."
        )

        user_prompt = f"""Input Graph: {input_mermaid}
Style Graph: {style_mermaid}
Input Nodes: {json.dumps(input_node_map, indent=2)}
Input Node Count: {input_node_count}
Style Node Count: {style_node_count}

Task: Map the Input Nodes (P0, P1...) into the Style Graph slots.
- The Style Graph Structure is IMMUTABLE. You must fit content into it.
- If Style has more nodes than Input, mark excess Style nodes as 'UNUSED'.
- **CRITICAL: If the Input Graph has MORE nodes than the Style Graph, you must LOGICALLY GROUP multiple input nodes into a single Style slot.**
  Example: If Input has P0, P1, P2, P3 and Style has only ROOT and CLAIM, you might map:
  {{ 'ROOT': 'P0, P1', 'CLAIM': 'P2, P3' }}
  Do not drop content. All input nodes must be assigned.
- Preserve logical flow: if Input P0 causes P1, ensure Style mapping maintains causality.
- Group related propositions logically (e.g., if P0 and P1 are both conditions, they can share a slot).

Return JSON mapping: {{ 'StyleNodeA': 'P0', 'StyleNodeB': 'P1, P2', 'StyleNodeC': 'UNUSED' }}"""

        try:
            response = self.llm_provider.call(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                model_type="editor",
                require_json=True,
                temperature=0.3,
                max_tokens=500
            )

            # Strip markdown code blocks
            response = self._strip_markdown_code_blocks(response)

            # Parse JSON
            node_mapping = json.loads(response)

            if verbose:
                print(f"     ✓ Node mapping:")
                for style_node, input_ref in node_mapping.items():
                    if input_ref == 'UNUSED':
                        print(f"        {style_node} → UNUSED")
                    else:
                        print(f"        {style_node} → {input_ref}")

            # Validate that all style nodes are mapped
            if not isinstance(node_mapping, dict):
                raise ValueError("Node mapping must be a dictionary")

            # Ensure all style nodes are in the mapping
            for style_node in style_nodes:
                if style_node not in node_mapping:
                    print(f"Warning: Style node '{style_node}' not in mapping, adding as 'UNUSED'")
                    node_mapping[style_node] = 'UNUSED'

        except (json.JSONDecodeError, Exception) as e:
            print(f"Warning: LLM node mapping failed: {e}")
            # Fallback: naive mapping
            node_mapping = {}
            input_keys = sorted(input_node_map.keys())
            for i, style_node in enumerate(style_nodes):
                if i < len(input_keys):
                    node_mapping[style_node] = input_keys[i]
                else:
                    node_mapping[style_node] = 'UNUSED'

        # Prepare return value
        return {
            'style_mermaid': style_mermaid,
            'node_mapping': node_mapping,
            'style_metadata': {
                'node_count': style_node_count,
                'edge_types': selected_candidate['edge_types'].split(',') if isinstance(selected_candidate['edge_types'], str) else selected_candidate['edge_types'],
                'skeleton': selected_candidate.get('skeleton', ''),
                'original_text': selected_candidate['original_text'],
                'paragraph_role': selected_candidate.get('paragraph_role'),
                'intent': selected_candidate.get('intent')
            },
            'distance': selected_candidate['distance'],
            'intent_match': selected_candidate.get('intent_match', False)
        }
