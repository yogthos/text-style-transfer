#!/usr/bin/env python3
"""Build Style Graph Index.

This script extracts abstract logical topologies from author corpus text and stores
them as Mermaid graphs in ChromaDB for semantic search and style transfer.
"""

import argparse
import hashlib
import json
import os
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List, Optional, Tuple, Dict

import chromadb
from chromadb.utils import embedding_functions
from tqdm import tqdm

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Load spaCy with safety check
print("Loading spaCy model...")
try:
    import spacy
    nlp = spacy.load("en_core_web_sm")
except OSError:
    print("Downloading spaCy model...")
    from spacy.cli import download
    download("en_core_web_sm")
    import spacy
    nlp = spacy.load("en_core_web_sm")

from src.generator.llm_provider import LLMProvider


class StyleGraphIndexer:
    """Indexes author corpus by extracting abstract logical topologies."""

    def __init__(self, config_path: str = "config.json"):
        """Initialize the Style Graph Indexer.

        Args:
            config_path: Path to configuration file.
        """
        self.config_path = config_path

        # Load config
        with open(config_path, 'r') as f:
            self.config = json.load(f)

        # Load style_graph_indexer config with defaults
        indexer_config = self.config.get("style_graph_indexer", {})
        self.opener_percentage = indexer_config.get("opener_percentage", 0.15)
        self.closer_percentage = indexer_config.get("closer_percentage", 0.15)

        # Initialize LLM provider
        print("Initializing LLM provider...")
        self.llm_provider = LLMProvider(config_path=config_path)

        # Initialize Structure Analyzer for logical chain extraction
        from src.ingestion.structure_analyzer import StructureAnalyzer
        self.structure_analyzer = StructureAnalyzer(self.llm_provider)

        # Initialize ChromaDB client
        self.chroma_path = project_root / "atlas_cache" / "chroma"
        self.chroma_path.mkdir(parents=True, exist_ok=True)

        print(f"Initializing ChromaDB at {self.chroma_path}...")
        self.client = chromadb.PersistentClient(path=str(self.chroma_path))

        # Get or create collection
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

        # Store collection name for clearing
        self.collection_name = collection_name

    def clear_collection(self):
        """Clear the style_graphs collection."""
        collection_name = "style_graphs"
        try:
            self.client.delete_collection(name=collection_name)
            print(f"Deleted existing collection: {collection_name}")
        except Exception:
            pass  # Collection doesn't exist, that's fine

        # Recreate the collection
        embedding_fn = embedding_functions.DefaultEmbeddingFunction()
        self.collection = self.client.create_collection(
            name=collection_name,
            embedding_function=embedding_fn
        )
        print(f"Created new collection: {collection_name}")

    def _split_into_paragraphs(self, text: str) -> List[str]:
        """Split text into paragraphs using regex to handle various newline formats.

        Args:
            text: Input text to split.

        Returns:
            List of paragraph strings (non-empty, stripped).
        """
        # Use regex to handle \n\n, \r\n\r\n, and variations with whitespace
        paragraphs = [p.strip() for p in re.split(r'\n\s*\n', text) if p.strip()]
        return paragraphs

    def _assign_paragraph_role(self, para_idx: int, total_paragraphs: int) -> str:
        """Assign role to paragraph based on position in document.

        Args:
            para_idx: Paragraph index (0-based).
            total_paragraphs: Total number of paragraphs in document.

        Returns:
            Role string: "opener", "body", or "closer".
        """
        # Safety check for small documents
        if total_paragraphs < 3:
            if total_paragraphs == 1:
                return "body"
            elif total_paragraphs == 2:
                return "opener" if para_idx == 0 else "closer"

        # Normal percentage-based assignment
        opener_threshold = int(total_paragraphs * self.opener_percentage)
        closer_threshold = total_paragraphs - int(total_paragraphs * self.closer_percentage)

        if para_idx < opener_threshold:
            return "opener"
        elif para_idx >= closer_threshold:
            return "closer"
        else:
            return "body"

    def _split_into_utterances(self, text: str) -> List[str]:
        """Split text into sentences (utterances) with filtering.

        Args:
            text: Input text to split.

        Returns:
            List of valid utterances (sentences between 5 and 60 words).
        """
        # Process text with spaCy
        doc = nlp(text)
        sentences = [sent.text.strip() for sent in doc.sents if sent.text.strip()]

        valid_utterances = []
        for sentence in sentences:
            # Count words (split by whitespace)
            word_count = len(sentence.split())

            # Filter: discard if < 5 words (too simple) or > 60 words (too complex)
            if 5 <= word_count <= 60:
                valid_utterances.append(sentence)

        return valid_utterances

    def _parse_mermaid_nodes_from_string(self, mermaid: str) -> List[str]:
        """Parse Mermaid graph string to extract node names.

        Args:
            mermaid: Mermaid graph string (e.g., "graph LR; ROOT --> NODE1").

        Returns:
            List of unique node identifiers.
        """
        nodes = set()

        # Remove graph declaration if present
        mermaid = re.sub(r'^graph\s+\w+\s*;?\s*', '', mermaid, flags=re.IGNORECASE)

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

    def _extract_skeleton_placeholders(self, skeleton: str) -> List[str]:
        """Extract node ID placeholders from skeleton string.

        Args:
            skeleton: Skeleton string with placeholders like [ROOT], [CLAIM].

        Returns:
            List of node IDs found in placeholders (without brackets).
        """
        # Match [NODE_ID] patterns
        placeholder_pattern = r'\[([A-Z_][A-Z0-9_]*)\]'
        matches = re.findall(placeholder_pattern, skeleton)
        # Filter out invalid single-underscore placeholders (parsing artifacts)
        matches = [m for m in matches if m != '_']
        return matches

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

    def _fix_json_response(self, text: str) -> str:
        """Attempt to fix common JSON formatting issues.

        Args:
            text: JSON string that may have formatting issues.

        Returns:
            Potentially fixed JSON string.
        """
        # Remove leading/trailing whitespace
        text = text.strip()

        # If the response doesn't start with {, try to find the JSON object
        if not text.startswith('{'):
            # Look for first { and last }
            start_idx = text.find('{')
            end_idx = text.rfind('}')
            if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
                text = text[start_idx:end_idx + 1]

        return text

    def _aggressive_json_fix(self, text: str) -> str:
        """More aggressive JSON fixing for malformed responses.

        Args:
            text: JSON string with potential issues.

        Returns:
            Fixed JSON string.
        """
        # Extract JSON object boundaries
        start_idx = text.find('{')
        end_idx = text.rfind('}')
        if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
            json_text = text[start_idx:end_idx + 1]
        else:
            json_text = text

        # Use a state machine to properly escape newlines and tabs in string values
        # We're careful NOT to escape quotes since that's error-prone
        result = []
        in_string = False
        escape_next = False
        i = 0

        while i < len(json_text):
            char = json_text[i]

            if escape_next:
                result.append(char)
                escape_next = False
                i += 1
                continue

            if char == '\\':
                result.append(char)
                escape_next = True
                i += 1
                continue

            if char == '"':
                in_string = not in_string
                result.append(char)
                i += 1
                continue

            if in_string:
                # Inside a string value - escape newlines, tabs, but NOT quotes
                # (quotes are handled by the quote detection above)
                if char == '\n':
                    result.append('\\n')
                elif char == '\r':
                    result.append('\\r')
                elif char == '\t':
                    result.append('\\t')
                else:
                    result.append(char)
            else:
                # Outside string - keep as is
                result.append(char)

            i += 1

        return ''.join(result)

    def _extract_topology(self, text: str, paragraph_index: int = 0, total_paragraphs: int = 1) -> Optional[dict]:
        """Extract abstract logical topology from text using LLM.

        Args:
            text: Input sentence/utterance to analyze.
            paragraph_index: Index of the paragraph in the document (0-based).
            total_paragraphs: Total number of paragraphs in the document.

        Returns:
            Dictionary with 'mermaid', 'description', 'node_count', 'edge_types', 'skeleton', 'role',
            or None if extraction fails.
        """
        system_prompt = (
            "You are a Rhetorical Topologist. Your task is to strip away specific "
            "content and reveal the logical dependency graph of the sentence. Use abstract "
            "node names (e.g., ROOT, CLAIM, CONDITION, CONTRAST, RESULT) and rhetorical "
            "edge labels (e.g., leads_to, negates, defines, supports)."
        )

        user_prompt = f"""Input Text: "{text}"
Task:

1. Analyze the logical structure.
2. Generate a Mermaid.js graph string.
3. Write a natural language description of this structure (for search).
4. **Generate Structural Summary (CRITICAL):** Describe the **Rhetorical Structure** of this text. Ignore specific names, topics, or entities. Use abstract terms like 'Contrast', 'Definition', 'Attribution', 'Conditional', 'Causality', 'Sequence', 'List'. Keep it under 15 words. Focus on the LOGICAL FORM, not the CONTENT.
   - Example: "A definition of a concept followed by its historical origin."
   - Example: "A set of misconceptions followed by a clarification."
   - Example: "Historical attribution of a creation action to a named agent."
   - Example: "A contrast between a false view and a true statement."
5. Classify the **Rhetorical Intent**: `DEFINITION` (explaining what X is), `ARGUMENT` (persuading/contrasting), `NARRATIVE` (telling a sequence), `INTERROGATIVE` (asking), `IMPERATIVE` (giving commands/directives).
6. Classify the **Logical Signature** (Choose ONE based on the logical relationship between nodes):
   - CONTRAST (Conflict/Negation/Correction: "Not X, but Y"; "Many think X, however Y")
   - CAUSALITY (Reasoning/Result: "Because X, Y"; "X leads to Y")
   - DEFINITION (Description/Identity: "X is Y"; "X constitutes Y")
   - SEQUENCE (Time/Order: "First X, then Y"; "When X, Y")
   - CONDITIONAL (Hypothetical: "If X, then Y")
   - LIST (Grouping: "X, Y, and Z")
   - If classification is ambiguous, default to DEFINITION.
7. Create a 'Syntactic Skeleton':
   - Keep all original prepositions, conjunctions, negative markers ('no', 'not'), and punctuation.
   - Replace specific content words with the abstract NODE IDs from your Mermaid graph (e.g., [ROOT], [CLAIM]).
   - **CRITICAL: The placeholders in the skeleton MUST match the Node IDs used in your Mermaid graph exactly (e.g., if graph has A-->B, skeleton must use [A] and [B], not [Concept A] or [Node B]).**

Output JSON:
{{
  "mermaid": "graph LR; ROOT --negates--> CONCEPT_A; ROOT --defines--> CONCEPT_B",
  "description": "A definition that starts by negating an opposing view.",
  "structural_summary": "A definition that negates an opposing view then establishes a concept.",
  "node_count": 3,
  "edge_types": ["negates", "defines"],
  "intent": "DEFINITION",
  "signature": "CONTRAST",
  "skeleton": "[ROOT] is not [CONCEPT_A], but [CONCEPT_B]."
}}

Example: If input text is "Politics is war without bloodshed." and your Mermaid graph uses nodes ROOT, CLAIM, CONDITION, then skeleton should be "[ROOT] is [CLAIM] without [CONDITION]." (where ROOT, CLAIM, CONDITION are the exact node names from the Mermaid graph)."""

        try:
            # Call LLM with require_json=True
            response = self.llm_provider.call(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                model_type="editor",
                require_json=True,
                temperature=0.3,
                max_tokens=500
            )

            # Strip markdown code blocks before parsing
            response = self._strip_markdown_code_blocks(response)

            # Try to fix common JSON issues before parsing
            response = self._fix_json_response(response)

            # Parse JSON
            try:
                result = json.loads(response)
            except json.JSONDecodeError as e:
                # Try to extract JSON from the response if it's embedded in text
                json_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', response, re.DOTALL)
                if json_match:
                    try:
                        result = json.loads(json_match.group(0))
                    except json.JSONDecodeError:
                        raise e
                else:
                    raise e

            # Validate required fields
            required_fields = ['mermaid', 'description', 'node_count', 'edge_types', 'intent', 'skeleton']
            if not all(field in result for field in required_fields):
                print(f"Warning: Missing required fields in LLM response for: {text[:50]}...")
                return None

            # Generate structural_summary if not provided (fallback)
            if 'structural_summary' not in result or not result.get('structural_summary'):
                # Fallback: create from description, but make it more abstract
                result['structural_summary'] = result.get('description', 'A logical structure.')

            # Validate intent is one of the allowed values
            valid_intents = ['DEFINITION', 'ARGUMENT', 'NARRATIVE', 'INTERROGATIVE', 'IMPERATIVE']

            # Validate signature if present, or set default
            valid_signatures = ['CONTRAST', 'CAUSALITY', 'DEFINITION', 'SEQUENCE', 'CONDITIONAL', 'LIST']
            if 'signature' not in result:
                # Default to DEFINITION if not provided
                result['signature'] = 'DEFINITION'
            elif result.get('signature') not in valid_signatures:
                print(f"Warning: Invalid signature '{result.get('signature')}', defaulting to 'DEFINITION' for: {text[:50]}...")
                result['signature'] = 'DEFINITION'

            # Determine narrative role based on paragraph position
            if paragraph_index == 0:
                role = 'INTRO'
            elif paragraph_index == total_paragraphs - 1:
                role = 'CONCLUSION'
            else:
                role = 'BODY'
            result['role'] = role

            if result.get('intent') not in valid_intents:
                print(f"Warning: Invalid intent '{result.get('intent')}' for: {text[:50]}..., defaulting to 'ARGUMENT'")
                result['intent'] = 'ARGUMENT'  # Default fallback

            # Validate mermaid is a string
            if not isinstance(result['mermaid'], str):
                print(f"Warning: Mermaid field is not a string for: {text[:50]}...")
                return None

            # Validate description is a string
            if not isinstance(result['description'], str):
                print(f"Warning: Description field is not a string for: {text[:50]}...")
                return None

            # Validate node_count is an integer
            if not isinstance(result['node_count'], int):
                print(f"Warning: Node count is not an integer for: {text[:50]}...")
                return None

            # Validate edge_types is a list
            if not isinstance(result['edge_types'], list):
                print(f"Warning: Edge types is not a list for: {text[:50]}...")
                return None

            # Validate skeleton is a string and non-empty
            if not isinstance(result['skeleton'], str) or not result['skeleton'].strip():
                print(f"Warning: Skeleton field is not a non-empty string for: {text[:50]}...")
                return None

            # Validate skeleton placeholders match Mermaid node IDs
            mermaid_nodes = self._parse_mermaid_nodes_from_string(result['mermaid'])
            skeleton_placeholders = self._extract_skeleton_placeholders(result['skeleton'])

            # Check that all placeholders in skeleton match nodes in mermaid
            mismatched = []
            for placeholder in skeleton_placeholders:
                if placeholder not in mermaid_nodes:
                    mismatched.append(placeholder)

            if mismatched:
                print(f"Warning: Skeleton placeholders {mismatched} do not match Mermaid node IDs {mermaid_nodes} for: {text[:50]}...")
                return None

            return result

        except json.JSONDecodeError as e:
            # Try one more time with more aggressive JSON fixing
            try:
                fixed_response = self._aggressive_json_fix(response)
                result = json.loads(fixed_response)

                # Validate required fields
                required_fields = ['mermaid', 'description', 'node_count', 'edge_types', 'intent', 'skeleton']
                # Ensure signature is present
                valid_signatures = ['CONTRAST', 'CAUSALITY', 'DEFINITION', 'SEQUENCE', 'CONDITIONAL', 'LIST']
                if 'signature' not in result:
                    result['signature'] = 'DEFINITION'
                elif result.get('signature') not in valid_signatures:
                    result['signature'] = 'DEFINITION'
                if all(field in result for field in required_fields):
                    # Additional validation
                    if (isinstance(result.get('mermaid'), str) and
                        isinstance(result.get('description'), str) and
                        isinstance(result.get('node_count'), int) and
                        isinstance(result.get('edge_types'), list) and
                        isinstance(result.get('skeleton'), str) and
                        result.get('skeleton', '').strip()):
                        # Validate skeleton matches node IDs
                        mermaid_nodes = self._parse_mermaid_nodes_from_string(result['mermaid'])
                        skeleton_placeholders = self._extract_skeleton_placeholders(result['skeleton'])
                        mismatched = [p for p in skeleton_placeholders if p not in mermaid_nodes]
                        if not mismatched:
                            return result
            except Exception as fix_error:
                # Try one more approach: manually construct JSON from fields
                try:
                    # Extract fields using regex as last resort
                    mermaid_match = re.search(r'"mermaid"\s*:\s*"([^"]*(?:\\.[^"]*)*)"', response, re.DOTALL)
                    desc_match = re.search(r'"description"\s*:\s*"([^"]*(?:\\.[^"]*)*)"', response, re.DOTALL)
                    node_count_match = re.search(r'"node_count"\s*:\s*(\d+)', response)
                    edge_types_match = re.search(r'"edge_types"\s*:\s*\[(.*?)\]', response, re.DOTALL)
                    skeleton_match = re.search(r'"skeleton"\s*:\s*"([^"]*(?:\\.[^"]*)*)"', response, re.DOTALL)

                    if mermaid_match and desc_match and node_count_match and edge_types_match and skeleton_match:
                        # Manually construct valid JSON
                        mermaid = mermaid_match.group(1).replace('\\"', '"').replace('\n', '\\n').replace('\r', '\\r')
                        description = desc_match.group(1).replace('\\"', '"').replace('\n', '\\n').replace('\r', '\\r')
                        node_count = int(node_count_match.group(1))
                        skeleton = skeleton_match.group(1).replace('\\"', '"').replace('\n', '\\n').replace('\r', '\\r')

                        # Parse edge types
                        edge_types_str = edge_types_match.group(1)
                        edge_types = [e.strip().strip('"') for e in edge_types_str.split(',') if e.strip()]

                        result = {
                            'mermaid': mermaid,
                            'description': description,
                            'node_count': node_count,
                            'edge_types': edge_types,
                            'skeleton': skeleton
                        }
                        # Validate skeleton matches node IDs
                        mermaid_nodes = self._parse_mermaid_nodes_from_string(mermaid)
                        skeleton_placeholders = self._extract_skeleton_placeholders(skeleton)
                        mismatched = [p for p in skeleton_placeholders if p not in mermaid_nodes]
                        if not mismatched and skeleton.strip():
                            return result
                except Exception:
                    pass  # Fall through to error message

            print(f"Warning: Failed to parse JSON response for: {text[:50]}...")
            print(f"  Error: {e}")
            if len(response) < 500:
                print(f"  Response was: {response[:200]}...")
            return None
        except Exception as e:
            print(f"Warning: LLM call failed for: {text[:50]}...")
            print(f"  Error: {e}")
            return None

    def build_index(
        self,
        corpus_path: str,
        author: Optional[str] = None,
        clear_first: bool = False,
        max_workers: int = 5,
        checkpoint_interval: int = 100
    ):
        """Build the style graph index from corpus.

        This method is resumable: if interrupted, it can be rerun and will skip
        sentences that have already been processed. Uses parallel processing to speed up indexing.

        Args:
            corpus_path: Path to corpus text file.
            author: Optional author name for metadata.
            clear_first: If True, clear the collection before indexing.
            max_workers: Number of parallel workers for LLM calls (default: 5).
            checkpoint_interval: Save progress every N successful extractions (default: 100).
        """
        # Clear collection if requested
        if clear_first:
            print("\nClearing existing style_graphs collection...")
            self.clear_collection()

        # Read corpus
        print(f"\nReading corpus from: {corpus_path}")
        if not os.path.exists(corpus_path):
            raise FileNotFoundError(f"Corpus file not found: {corpus_path}")

        with open(corpus_path, 'r', encoding='utf-8') as f:
            corpus_text = f.read()

        # Create a unique identifier for this corpus (for resumability)
        corpus_hash = hashlib.md5(f"{corpus_path}:{author or 'unknown'}".encode()).hexdigest()[:8]

        # Split into paragraphs first
        print("Splitting into paragraphs...")
        paragraphs = self._split_into_paragraphs(corpus_text)
        total_paragraphs = len(paragraphs)
        print(f"Found {total_paragraphs} paragraphs")

        if total_paragraphs == 0:
            print("Error: No paragraphs found. Check corpus file format.")
            return

        # Process paragraphs and extract sentence graphs
        print("\nExtracting topologies from paragraphs...")

        # First, count total sentences for progress tracking
        total_sentences = 0
        paragraph_sentences = []
        for paragraph in paragraphs:
            sentences = self._split_into_utterances(paragraph)
            paragraph_sentences.append(sentences)
            total_sentences += len(sentences)

        print(f"Found {total_sentences} valid sentences to process")

        successful = 0
        skipped = 0
        duplicates = 0
        already_processed = 0

        # Track existing graph IDs (by Mermaid hash) for deduplication
        existing_graph_ids = set()
        # Track processed sentences (by sentence text hash) for resumability
        processed_sentences = set()

        try:
            existing = self.collection.get()
            existing_graph_ids = set(existing['ids'])
            print(f"Found {len(existing_graph_ids)} existing graphs in collection")

            # Build set of already-processed sentences by checking metadata
            # We use a combination of original_text + corpus_hash to identify processed sentences
            for metadata in existing.get('metadatas', []):
                original_text = metadata.get('original_text', '')
                stored_corpus_hash = metadata.get('corpus_hash', '')
                # If this sentence was from the same corpus, mark it as processed
                if original_text and stored_corpus_hash == corpus_hash:
                    # Create a unique ID for this sentence
                    sentence_id = hashlib.md5(original_text.encode()).hexdigest()
                    processed_sentences.add(sentence_id)

            if processed_sentences:
                print(f"Found {len(processed_sentences)} already-processed sentences from this corpus")
                print("  (Script is resumable - will skip these)")
        except Exception:
            pass  # Collection is empty, that's fine

        # Prepare all sentence tasks for parallel processing
        sentence_tasks = []
        for para_idx, paragraph in enumerate(paragraphs):
            paragraph_role = self._assign_paragraph_role(para_idx, total_paragraphs)
            sentences = paragraph_sentences[para_idx]

            if len(sentences) == 0:
                continue

            for sent_idx, sentence in enumerate(sentences):
                sentence_id = hashlib.md5(sentence.encode()).hexdigest()
                # Skip already processed sentences
                if sentence_id not in processed_sentences:
                    sentence_tasks.append({
                        'sentence': sentence,
                        'sentence_id': sentence_id,
                        'para_idx': para_idx,
                        'sent_idx': sent_idx,
                        'paragraph_role': paragraph_role,
                        'is_paragraph_start': (sent_idx == 0),
                        'is_paragraph_end': (sent_idx == len(sentences) - 1)
                    })
                else:
                    already_processed += 1

        print(f"Processing {len(sentence_tasks)} new sentences with {max_workers} parallel workers...")

        # Thread-safe batch processing
        batch_lock = threading.Lock()
        batch_documents = []
        batch_metadatas = []
        batch_ids = []
        batch_size = 50
        last_checkpoint = 0

        def process_sentence_task(task: Dict) -> Optional[Dict]:
            """Process a single sentence task and return result or None."""
            sentence = task['sentence']
            sentence_id = task['sentence_id']

            try:
                # Extract topology with timeout handling
                # Pass paragraph index and total paragraphs for role determination
                topology = self._extract_topology(
                    sentence,
                    paragraph_index=task['para_idx'],
                    total_paragraphs=total_paragraphs
                )

                if topology is None:
                    return {'type': 'skipped', 'sentence_id': sentence_id}

                # Generate unique ID from Mermaid string
                mermaid_string = topology['mermaid']
                graph_id = hashlib.md5(mermaid_string.encode()).hexdigest()

                # Check for duplicate graph structure (thread-safe check)
                with batch_lock:
                    if graph_id in existing_graph_ids:
                        return {'type': 'duplicate', 'sentence_id': sentence_id, 'graph_id': graph_id}
                    existing_graph_ids.add(graph_id)

                # Extract logical structure chain using StructureAnalyzer
                logic_signature = self.structure_analyzer.analyze_structure(sentence)

                # Prepare metadata
                metadata = {
                    'mermaid': mermaid_string,
                    'node_count': topology['node_count'],
                    'edge_types': ','.join(topology['edge_types']),
                    'skeleton': topology.get('skeleton', ''),
                    'intent': topology.get('intent', 'ARGUMENT'),
                    'signature': topology.get('signature', 'DEFINITION'),  # Store signature (single tag)
                    'logic_signature': logic_signature,  # Store logic chain (pipe-separated tags)
                    'role': topology.get('role', 'BODY'),  # Store narrative role (INTRO, BODY, CONCLUSION)
                    'structural_summary': topology.get('structural_summary', topology.get('description', '')),  # Store structural summary
                    'original_text': sentence,
                    'paragraph_role': task['paragraph_role'],
                    'paragraph_index': task['para_idx'],
                    'is_paragraph_start': task['is_paragraph_start'],
                    'is_paragraph_end': task['is_paragraph_end'],
                    'corpus_hash': corpus_hash,
                    'sentence_id': sentence_id,
                }
                if author:
                    metadata['author'] = author

                return {
                    'type': 'success',
                    'sentence_id': sentence_id,
                    'graph_id': graph_id,
                    'document': topology.get('structural_summary', topology['description']),  # Use structural_summary for embedding
                    'metadata': metadata
                }
            except Exception as e:
                print(f"Warning: Error processing sentence: {e}")
                return {'type': 'skipped', 'sentence_id': sentence_id}

        def flush_batch(force: bool = False):
            """Thread-safe batch flush to ChromaDB."""
            nonlocal batch_documents, batch_metadatas, batch_ids, successful, last_checkpoint

            with batch_lock:
                should_flush = force or len(batch_ids) >= batch_size
                if not should_flush or len(batch_ids) == 0:
                    return

                # Copy batch data locally
                local_docs = batch_documents.copy()
                local_metas = batch_metadatas.copy()
                local_ids = batch_ids.copy()
                batch_count = len(local_ids)

                # Clear batch
                batch_documents.clear()
                batch_metadatas.clear()
                batch_ids.clear()

            # Upsert outside lock to avoid blocking
            if batch_count > 0:
                try:
                    self.collection.upsert(
                        ids=local_ids,
                        documents=local_docs,
                        metadatas=local_metas
                    )
                    successful += batch_count

                    # Checkpoint if needed
                    if successful - last_checkpoint >= checkpoint_interval:
                        last_checkpoint = successful
                        print(f"\n  ✓ Checkpoint: {successful} graphs indexed so far...")
                except Exception as e:
                    print(f"Warning: Failed to upsert batch: {e}")

        # Process sentences in parallel
        with tqdm(total=len(sentence_tasks), desc="Processing sentences", unit="sent") as pbar:
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                # Submit all tasks
                future_to_task = {
                    executor.submit(process_sentence_task, task): task
                    for task in sentence_tasks
                }

                # Process completed futures
                for future in as_completed(future_to_task):
                    task = future_to_task[future]
                    try:
                        result = future.result(timeout=120)  # 2 minute timeout per sentence

                        if result is None:
                            skipped += 1
                            processed_sentences.add(task['sentence_id'])
                            pbar.update(1)
                            continue

                        if result['type'] == 'skipped':
                            skipped += 1
                            processed_sentences.add(result['sentence_id'])
                            pbar.update(1)
                            continue

                        if result['type'] == 'duplicate':
                            duplicates += 1
                            processed_sentences.add(result['sentence_id'])
                            pbar.update(1)
                            continue

                        if result['type'] == 'success':
                            # Add to batch (thread-safe)
                            with batch_lock:
                                batch_documents.append(result['document'])
                                batch_metadatas.append(result['metadata'])
                                batch_ids.append(result['graph_id'])
                                processed_sentences.add(result['sentence_id'])

                            # Flush batch if needed
                            flush_batch()
                            pbar.update(1)

                    except Exception as e:
                        # Handle timeout or other errors
                        skipped += 1
                        processed_sentences.add(task['sentence_id'])
                        pbar.update(1)
                        # Note: verbose logging removed to avoid thread conflicts

                    # Update progress bar stats
                    pbar.set_postfix({
                        'success': successful,
                        'skipped': skipped,
                        'duplicates': duplicates,
                        'resumed': already_processed
                    })

        # Final batch flush
        flush_batch(force=True)

        # Print summary
        print(f"\n✓ Indexing complete!")
        print(f"  Total paragraphs processed: {total_paragraphs}")
        print(f"  Total sentences found: {total_sentences}")
        print(f"  Successful extractions: {successful}")
        print(f"  Failed extractions: {skipped}")
        print(f"  Duplicate graphs skipped: {duplicates}")
        print(f"  Already processed (resumed): {already_processed}")
        print(f"  Unique graphs indexed: {successful}")
        print(f"  Collection: style_graphs")
        print(f"  Location: {self.chroma_path}")

        if already_processed > 0:
            print(f"\n  Note: {already_processed} sentences were already processed (script is resumable)")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Build Style Graph Index from author corpus",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument(
        "--corpus-file",
        type=str,
        default=None,
        help="Path to corpus file (default: data/corpus/mao.txt)"
    )

    parser.add_argument(
        "--author",
        type=str,
        default=None,
        help="Author name (optional, for metadata)"
    )

    parser.add_argument(
        "--config",
        type=str,
        default="config.json",
        help="Path to config.json (default: config.json)"
    )

    parser.add_argument(
        "--clear-style-graphs",
        action="store_true",
        help="Clear the style_graphs collection before indexing (use when adding new metadata fields)"
    )

    parser.add_argument(
        "--max-workers",
        type=int,
        default=5,
        help="Number of parallel workers for LLM calls (default: 5)"
    )

    parser.add_argument(
        "--checkpoint-interval",
        type=int,
        default=100,
        help="Save progress every N successful extractions (default: 100)"
    )

    args = parser.parse_args()

    # Determine corpus file
    if args.corpus_file:
        corpus_file = args.corpus_file
    else:
        # Default to data/corpus/mao.txt
        corpus_file = project_root / "data" / "corpus" / "mao.txt"

    if not os.path.exists(corpus_file):
        print(f"Error: Corpus file not found: {corpus_file}")
        print(f"  Please provide --corpus-file or ensure file exists")
        sys.exit(1)

    if not os.path.exists(args.config):
        print(f"Error: Config file not found: {args.config}")
        sys.exit(1)

    try:
        indexer = StyleGraphIndexer(config_path=args.config)
        indexer.build_index(
            corpus_path=str(corpus_file),
            author=args.author,
            clear_first=args.clear_style_graphs,
            max_workers=args.max_workers,
            checkpoint_interval=args.checkpoint_interval
        )
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

