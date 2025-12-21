"""Script to curate templates in the Style Atlas.

This script audits templates offline to ensure only high-quality templates
are retrieved at runtime. It scores templates on:
1. Style Fidelity (LLM-based audit)
2. Capacity (proposition count matching)
3. Anchor Health (structural validation)
"""

import sys
import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.atlas.builder import load_atlas, StyleAtlas
from src.generator.llm_provider import LLMProvider
from src.analyzer.structure_extractor import StructureExtractor

try:
    import chromadb
    from chromadb.config import Settings
    CHROMADB_AVAILABLE = True
except ImportError:
    CHROMADB_AVAILABLE = False
    chromadb = None
    Settings = None


def audit_style_fidelity(
    template_text: str,
    author_style_dna: str,
    llm_provider: LLMProvider,
    verbose: bool = False
) -> Tuple[int, str]:
    """Audit template for style fidelity using LLM.

    Args:
        template_text: Template string to audit
        author_style_dna: Author's style DNA description
        llm_provider: LLMProvider instance
        verbose: Enable verbose logging

    Returns:
        Tuple of (score: int 1-5, reason: str)
    """
    system_prompt = f"""You are an expert on {author_style_dna.split('.')[0] if author_style_dna else 'the author'}'s syntax and rhetorical style.

Your task is to evaluate sentence templates for stylistic distinctiveness."""

    user_prompt = f"""Analyze this sentence template: `{template_text}`

Does it reflect the rhetorical style of the author? Consider:
- Complexity and dialectical structure
- Signature phrases and constructions
- Register and tone
- Distinctiveness from generic English

Score (1-5):
- 5 = Perfect Signature Style (highly distinctive, complex, characteristic)
- 4 = Strong Style Match (distinctive, well-structured)
- 3 = Moderate Style (somewhat distinctive, acceptable)
- 2 = Weak Style (generic, bland)
- 1 = Generic/Bland (no distinctive features)

Output: JSON with 'score' (int 1-5) and 'reason' (string, brief explanation).

Example:
{{
  "score": 4,
  "reason": "Uses characteristic dialectical structure with nested clauses"
}}"""

    try:
        response = llm_provider.call(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            model_type="editor",  # Use editor model (can be lightweight)
            require_json=True,
            temperature=0.2,  # Low temperature for consistent scoring
            max_tokens=200,
            timeout=30
        )

        # Parse JSON response
        response = response.strip()
        json_match = re.search(r'\{.*?\}', response, re.DOTALL)
        if json_match:
            result = json.loads(json_match.group(0))
        else:
            result = json.loads(response)

        score = int(result.get("score", 3))
        reason = result.get("reason", "No reason provided")

        # Clamp score to 1-5
        score = max(1, min(5, score))

        if verbose:
            print(f"      Style score: {score}/5 - {reason}")

        return (score, reason)

    except Exception as e:
        if verbose:
            print(f"      ⚠ Style audit failed: {e}, defaulting to score 3")
        return (3, f"Audit failed: {str(e)}")


def calculate_template_capacity(template_text: str) -> int:
    """Calculate ideal proposition count for a template.

    Args:
        template_text: Template string with placeholders

    Returns:
        Ideal proposition count (integer)
    """
    # Count major slots: [NP], [VP], [CLAUSE], [ADJ], [ADV]
    # Pattern matches: [NP], [VP], [CLAUSE], [ADJ], [ADV] and nested structures
    slot_pattern = r'\[(?:NP|VP|CLAUSE|ADJ|ADV)(?:\s+[^\]]+)?\]'
    matches = re.findall(slot_pattern, template_text)

    count_major_slots = len(matches)

    # Heuristic: ideal_prop_count = max(1, round(count_major_slots / 1.5))
    ideal_prop_count = max(1, round(count_major_slots / 1.5))

    return ideal_prop_count


def check_anchor_health(template_text: str) -> Tuple[bool, int]:
    """Check anchor health of a template.

    Args:
        template_text: Template string to check

    Returns:
        Tuple of (is_valid: bool, anchor_count: int)
    """
    # Check length
    if len(template_text) < 20:
        return (False, 0)

    # Extract fixed anchors (everything outside placeholders)
    # Split by placeholders to get anchor segments
    placeholder_pattern = r'\[[^\]]+\]'
    anchor_segments = re.split(placeholder_pattern, template_text)

    # Count anchor words (non-empty, non-punctuation-only segments)
    anchor_words = []
    for segment in anchor_segments:
        segment = segment.strip()
        if segment:
            # Split into words and filter out punctuation-only
            words = [w for w in segment.split() if any(c.isalnum() for c in w)]
            anchor_words.extend(words)

    anchor_count = len(anchor_words)

    # Count slots
    slots = re.findall(placeholder_pattern, template_text)
    slot_count = len(slots)

    # Rules:
    # 1. Must have at least 2 anchor words
    if anchor_count < 2:
        return (False, anchor_count)

    # 2. Must not be 80%+ fixed words (too rigid)
    # BUT: If there are no slots at all, this is likely a plain text sentence
    # that should be treated as invalid (not a template)
    if slot_count == 0:
        # No placeholders found - this is not a valid template
        return (False, anchor_count)

    total_elements = slot_count + anchor_count
    if total_elements > 0:
        fixed_ratio = anchor_count / total_elements
        if fixed_ratio >= 0.8:
            return (False, anchor_count)

    return (True, anchor_count)


def extract_templates_from_atlas(
    atlas: StyleAtlas,
    author_id: Optional[str] = None,
    persist_directory: Optional[str] = None,
    verbose: bool = False
) -> List[Dict]:
    """Extract all templates from atlas.

    Args:
        atlas: StyleAtlas instance
        author_id: Optional author ID to filter by
        persist_directory: ChromaDB persist directory
        verbose: Enable verbose logging

    Returns:
        List of template records: [{'id': str, 'template': str, 'metadata': dict, 'entry_id': str, 'template_idx': int}]
    """
    if not CHROMADB_AVAILABLE:
        raise ImportError("ChromaDB is not available. Cannot extract templates.")

    # Use existing client/collection if available, otherwise initialize
    if not hasattr(atlas, '_client') or not hasattr(atlas, '_collection'):
        if not hasattr(atlas, '_client'):
            if persist_directory:
                atlas._client = chromadb.PersistentClient(path=persist_directory)
            else:
                atlas._client = chromadb.Client(Settings(anonymized_telemetry=False))

        if not hasattr(atlas, '_collection'):
            try:
                atlas._collection = atlas._client.get_collection(name=atlas.collection_name)
            except Exception as e:
                if verbose:
                    print(f"  ⚠ Could not get collection '{atlas.collection_name}': {e}")
                    try:
                        available = [c.name for c in atlas._client.list_collections()]
                        print(f"     Available collections: {available}")
                    except:
                        pass
                raise

    collection = atlas._collection

    # Query all entries (optionally filter by author_id)
    where_clause = {}
    if author_id:
        where_clause = {"author_id": author_id}

    try:
        results = collection.get(where=where_clause if where_clause else None)
    except Exception:
        # Fallback: get all without where clause
        results = collection.get()

    template_records = []

    for idx, entry_id in enumerate(results['ids']):
        metadata = results['metadatas'][idx] if results['metadatas'] else {}
        skeletons_json = metadata.get('skeletons', '[]')

        try:
            skeletons = json.loads(skeletons_json)
            if not isinstance(skeletons, list):
                if verbose:
                    print(f"  ⚠ Entry {entry_id}: skeletons is not a list, skipping")
                continue
        except json.JSONDecodeError as e:
            if verbose:
                print(f"  ⚠ Entry {entry_id}: Failed to parse skeletons JSON: {e}")
            continue

        # Extract individual templates
        for template_idx, skeleton in enumerate(skeletons):
            if isinstance(skeleton, dict):
                template = skeleton.get('template', '')
            elif isinstance(skeleton, str):
                template = skeleton
            else:
                if verbose:
                    print(f"  ⚠ Entry {entry_id}, template {template_idx}: Invalid skeleton format")
                continue

            if not template or not template.strip():
                continue

            template_records.append({
                'id': f"{entry_id}_template_{template_idx}",
                'template': template,
                'metadata': skeleton if isinstance(skeleton, dict) else {},
                'entry_id': entry_id,
                'template_idx': template_idx,
                'entry_metadata': metadata
            })

    if verbose:
        print(f"  Extracted {len(template_records)} templates from {len(results['ids'])} entries")

    return template_records


def update_template_metadata(
    collection,
    template_records: List[Dict],
    updates: Dict[str, Dict],
    verbose: bool = False
) -> Tuple[int, int]:
    """Update template metadata in ChromaDB.

    Args:
        collection: ChromaDB collection
        template_records: List of template records
        updates: Dict mapping template_id to update dict with 'style_score', 'ideal_prop_count', 'anchor_count', 'is_valid'
        verbose: Enable verbose logging

    Returns:
        Tuple of (updated_count, deleted_count)
    """
    # Group updates by entry_id
    entry_updates = {}

    for template_id, update_data in updates.items():
        # Find the template record
        template_record = next((t for t in template_records if t['id'] == template_id), None)
        if not template_record:
            continue

        entry_id = template_record['entry_id']
        template_idx = template_record['template_idx']

        if entry_id not in entry_updates:
            entry_updates[entry_id] = {
                'entry_metadata': template_record['entry_metadata'],
                'template_updates': {}
            }

        entry_updates[entry_id]['template_updates'][template_idx] = update_data

    updated_count = 0
    deleted_count = 0

    # Update each entry
    for entry_id, entry_data in entry_updates.items():
        try:
            # Get current skeletons
            entry_metadata = entry_data['entry_metadata']
            skeletons_json = entry_metadata.get('skeletons', '[]')
            skeletons = json.loads(skeletons_json)

            if not isinstance(skeletons, list):
                if verbose:
                    print(f"  ⚠ Entry {entry_id}: Invalid skeletons format, skipping")
                continue

            # Update templates
            updated_skeletons = []
            for template_idx, skeleton in enumerate(skeletons):
                if template_idx in entry_data['template_updates']:
                    update = entry_data['template_updates'][template_idx]

                    # If invalid, skip (delete)
                    if not update.get('is_valid', True):
                        deleted_count += 1
                        if verbose:
                            print(f"    Deleting template {template_idx} from entry {entry_id}")
                        continue

                    # Update metadata
                    if isinstance(skeleton, dict):
                        # Update existing dict - also update template if it was extracted
                        if 'extracted_template' in update:
                            skeleton['template'] = update['extracted_template']
                        skeleton['style_score'] = update.get('style_score', 3)
                        skeleton['ideal_prop_count'] = update.get('ideal_prop_count', 2)
                        skeleton['anchor_count'] = update.get('anchor_count', 0)
                        updated_skeletons.append(skeleton)
                    else:
                        # Convert string to dict - use extracted template if available
                        template_text = update.get('extracted_template', skeleton)
                        updated_skeletons.append({
                            'template': template_text,
                            'style_score': update.get('style_score', 3),
                            'ideal_prop_count': update.get('ideal_prop_count', 2),
                            'anchor_count': update.get('anchor_count', 0)
                        })
                    updated_count += 1
                else:
                    # Keep existing template
                    updated_skeletons.append(skeleton)

            # Update entry if skeletons changed
            if len(updated_skeletons) != len(skeletons) or any(
                isinstance(s, dict) and ('style_score' in s or 'ideal_prop_count' in s)
                for s in updated_skeletons
            ):
                # Update metadata
                new_metadata = entry_metadata.copy()
                new_metadata['skeletons'] = json.dumps(updated_skeletons)

                # Update in ChromaDB
                collection.update(
                    ids=[entry_id],
                    metadatas=[new_metadata]
                )

                if verbose:
                    print(f"  ✅ Updated entry {entry_id}: {len(updated_skeletons)} templates")

        except Exception as e:
            if verbose:
                print(f"  ⚠ Failed to update entry {entry_id}: {e}")
            continue

    return (updated_count, deleted_count)


def curate_atlas(
    atlas_path: str,
    author_id: Optional[str] = None,
    persist_directory: Optional[str] = None,
    config_path: str = "config.json",
    verbose: bool = False
) -> Dict[str, int]:
    """Curate templates in the Atlas.

    Args:
        atlas_path: Path to atlas.json file
        author_id: Optional author ID to curate (if None, curates all)
        persist_directory: ChromaDB persist directory
        config_path: Path to config file
        verbose: Enable verbose logging

    Returns:
        Dict with statistics: {'audited': int, 'deleted': int, 'updated': int}
    """
    if not CHROMADB_AVAILABLE:
        raise ImportError("ChromaDB is not available. Please install it.")

    if verbose:
        print(f"Loading atlas from {atlas_path}...")
        if persist_directory:
            print(f"Using ChromaDB persist directory: {persist_directory}")

    # Load atlas (this will set up client and collection if persist_directory is provided)
    atlas = load_atlas(atlas_path, persist_directory=persist_directory)

    # Ensure client and collection are set up
    if not hasattr(atlas, '_client') or not hasattr(atlas, '_collection'):
        if verbose:
            print("  Setting up ChromaDB client...")
        if persist_directory:
            atlas._client = chromadb.PersistentClient(path=persist_directory)
        else:
            atlas._client = chromadb.Client(Settings(anonymized_telemetry=False))

        try:
            atlas._collection = atlas._client.get_collection(name=atlas.collection_name)
            if verbose:
                print(f"  Connected to collection: {atlas.collection_name}")
        except Exception as e:
            if verbose:
                print(f"  ⚠ Could not connect to collection: {e}")
                print(f"     Available collections: {[c.name for c in atlas._client.list_collections()]}")
            raise

    # Get author style DNA
    author_style_dna = ""
    if author_id and atlas.author_style_dna:
        author_style_dna = atlas.author_style_dna.get(author_id, "")
    elif atlas.author_style_dna:
        # Use first author if no specific author_id
        first_author = next(iter(atlas.author_style_dna.keys()), None)
        if first_author:
            author_style_dna = atlas.author_style_dna[first_author]

    if not author_style_dna:
        author_style_dna = "the author"  # Fallback

    if verbose:
        print(f"Author style DNA: {author_style_dna[:100]}...")

    # Initialize LLM provider and structure extractor
    llm_provider = LLMProvider(config_path=config_path)
    structure_extractor = StructureExtractor(config_path=config_path)

    # Extract templates
    if verbose:
        print("Extracting templates from atlas...")
    template_records = extract_templates_from_atlas(
        atlas,
        author_id=author_id,
        persist_directory=persist_directory,
        verbose=verbose
    )

    if not template_records:
        if verbose:
            print("No templates found.")
        return {'audited': 0, 'deleted': 0, 'updated': 0}

    # Debug: Show sample templates
    if verbose and template_records:
        import re
        placeholder_pattern = r'\[[^\]]+\]'
        sample_templates = template_records[:3]
        print(f"\nSample templates (first {len(sample_templates)}):")
        for tr in sample_templates:
            slots = re.findall(placeholder_pattern, tr['template'])
            print(f"  {tr['id']}: {tr['template'][:100]}... (slots: {len(slots)})")

    # Get collection (should already be set by extract_templates_from_atlas)
    if not hasattr(atlas, '_collection'):
        if not hasattr(atlas, '_client'):
            if persist_directory:
                atlas._client = chromadb.PersistentClient(path=persist_directory)
            else:
                atlas._client = chromadb.Client(Settings(anonymized_telemetry=False))
        try:
            atlas._collection = atlas._client.get_collection(name=atlas.collection_name)
        except Exception as e:
            if verbose:
                print(f"  ⚠ Could not get collection '{atlas.collection_name}': {e}")
            raise

    collection = atlas._collection

    # Audit templates
    if verbose:
        print(f"Auditing {len(template_records)} templates...")

    updates = {}
    audited_count = 0

    for template_record in template_records:
        template_id = template_record['id']
        template_text = template_record['template']

        if verbose and audited_count % 100 == 0:
            print(f"  Progress: {audited_count}/{len(template_records)}")

        # Check if template has placeholders - if not, extract template from plain text
        import re
        placeholder_pattern = r'\[[^\]]+\]'
        has_placeholders = bool(re.search(placeholder_pattern, template_text))
        extracted_template_text = None  # Track if we extracted a template

        if not has_placeholders:
            # This is a plain text sentence, not a template - convert it
            if verbose and audited_count < 5:
                print(f"    Converting plain text to template: {template_id}")
            try:
                extracted_template = structure_extractor.extract_template(template_text)
                if extracted_template and extracted_template.strip():
                    extracted_template_text = extracted_template
                    template_text = extracted_template_text
                    # Update the template record so we use the extracted template
                    template_record['template'] = template_text
                else:
                    # Extraction failed, skip this template
                    if verbose:
                        print(f"    ⚠ Template extraction failed for {template_id}, skipping")
                    updates[template_id] = {
                        'style_score': 1,
                        'ideal_prop_count': 0,
                        'anchor_count': 0,
                        'is_valid': False,
                        'extracted_template': None
                    }
                    audited_count += 1
                    continue
            except Exception as e:
                if verbose:
                    print(f"    ⚠ Template extraction error for {template_id}: {e}, skipping")
                updates[template_id] = {
                    'style_score': 1,
                    'ideal_prop_count': 0,
                    'anchor_count': 0,
                    'is_valid': False,
                    'extracted_template': None
                }
                audited_count += 1
                continue

        # 1. Style Fidelity Audit
        style_score, style_reason = audit_style_fidelity(
            template_text, author_style_dna, llm_provider, verbose=False
        )

        # 2. Capacity Calculation
        ideal_prop_count = calculate_template_capacity(template_text)

        # 3. Anchor Health Check
        is_valid, anchor_count = check_anchor_health(template_text)

        # Determine if template should be kept
        # Delete if: style_score < 3 OR not valid
        should_keep = is_valid and style_score >= 3

        updates[template_id] = {
            'style_score': style_score,
            'ideal_prop_count': ideal_prop_count,
            'anchor_count': anchor_count,
            'is_valid': should_keep,
            'extracted_template': extracted_template_text  # Store extracted template if we converted it (None if already had placeholders)
        }

        audited_count += 1

        # Log all templates (success and failure) for first few, then only failures
        import re
        placeholder_pattern = r'\[[^\]]+\]'
        slots = re.findall(placeholder_pattern, template_text)
        total_elements = len(slots) + anchor_count
        fixed_ratio = anchor_count / total_elements if total_elements > 0 else 1.0

        if verbose:
            if should_keep:
                # Show first 20 successful templates in detail, then summary every 50
                if audited_count <= 20:
                    print(f"    ✅ Template {template_id}: style={style_score}, valid={is_valid}, anchors={anchor_count}, slots={len(slots)}, fixed_ratio={fixed_ratio:.2f}")
                    print(f"      Template: {template_text[:150]}")
                elif audited_count % 50 == 0:
                    # Show summary every 50 templates
                    print(f"    ✅ Template {template_id}: style={style_score}, valid={is_valid}, anchors={anchor_count}, slots={len(slots)}")
            else:
                # Always show failures
                print(f"    ❌ Template {template_id}: style={style_score}, valid={is_valid}, anchors={anchor_count}, slots={len(slots)}, fixed_ratio={fixed_ratio:.2f}")
                if audited_count <= 20 or not is_valid:  # Show detail for first 20 or invalid templates
                    print(f"      Template text: {template_text[:150]}")
                    print(f"      Length: {len(template_text)}, Slots: {len(slots)}, Anchor words: {anchor_count}, Total elements: {total_elements}")
                    if len(slots) == 0:
                        print(f"      ⚠ No placeholders found - template may be plain text, not a structural template")
                    elif anchor_count < 2:
                        print(f"      ⚠ Too few anchor words (< 2)")
                    elif fixed_ratio >= 0.8:
                        print(f"      ⚠ Too rigid (>= 80% fixed words)")
                    elif style_score < 3:
                        print(f"      ⚠ Style score too low (< 3)")

    # Update metadata
    if verbose:
        print("Updating template metadata...")
    updated_count, deleted_count = update_template_metadata(
        collection, template_records, updates, verbose=verbose
    )

    # Calculate success/failure stats
    successful_count = sum(1 for u in updates.values() if u.get('is_valid', False))
    failed_count = audited_count - successful_count

    if verbose:
        print(f"\n✅ Curation complete:")
        print(f"   Audited: {audited_count}")
        print(f"   Successful: {successful_count}")
        print(f"   Failed: {failed_count}")
        print(f"   Updated: {updated_count}")
        print(f"   Deleted: {deleted_count}")

    return {
        'audited': audited_count,
        'deleted': deleted_count,
        'updated': updated_count
    }


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Curate Template Atlas")
    parser.add_argument("--atlas", required=True, help="Path to atlas.json")
    parser.add_argument("--author", help="Author ID to curate (optional)")
    parser.add_argument("--persist-dir", help="ChromaDB persist directory (default: same directory as atlas.json)")
    parser.add_argument("--config", default="config.json", help="Config file path")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")
    args = parser.parse_args()

    # If persist-dir not provided, try to infer from atlas path
    persist_dir = args.persist_dir
    if not persist_dir:
        atlas_path = Path(args.atlas)
        # If atlas is in a directory, use that directory as persist_dir
        if atlas_path.parent.exists():
            persist_dir = str(atlas_path.parent)

    stats = curate_atlas(
        atlas_path=args.atlas,
        author_id=args.author,
        persist_directory=persist_dir,
        config_path=args.config,
        verbose=args.verbose
    )
    print(f"Audited: {stats['audited']}, Deleted: {stats['deleted']}, Updated: {stats['updated']}")

