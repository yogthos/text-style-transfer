"""Script to add text_length metadata to existing Style Atlas documents.

This script iterates over a ChromaDB collection and adds the text_length
(character count) metadata field to each document. This is required for
the new length window filtering in retrieval optimization.
"""

import sys
import json
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import chromadb
from chromadb.config import Settings


def add_length_metadata(
    collection_name: str = "style_atlas",
    persist_directory: str = None,
    batch_size: int = 100,
    dry_run: bool = False
):
    """Add text_length metadata to all documents in atlas.

    Args:
        collection_name: Name of ChromaDB collection.
        persist_directory: Directory where ChromaDB is persisted.
        batch_size: Number of documents to process in each batch.
        dry_run: If True, only show what would be updated without making changes.
    """
    # Initialize ChromaDB client
    if persist_directory:
        client = chromadb.PersistentClient(path=persist_directory)
    else:
        client = chromadb.Client(Settings(anonymized_telemetry=False))

    # Get collection
    try:
        collection = client.get_collection(name=collection_name)
    except Exception as e:
        print(f"Error: Could not access collection '{collection_name}': {e}")
        return

    # Get all documents
    try:
        all_results = collection.get(include=["metadatas", "documents"])
    except Exception as e:
        print(f"Error: Could not retrieve documents: {e}")
        return

    documents = all_results.get('documents', [])
    metadatas = all_results.get('metadatas', [])
    ids = all_results.get('ids', [])

    if not documents:
        print("No documents found in collection.")
        return

    print(f"Found {len(documents)} documents.")

    # Check how many already have text_length
    already_has_length = sum(1 for meta in metadatas if meta and 'text_length' in meta)
    needs_update = len(documents) - already_has_length

    if already_has_length > 0:
        print(f"  {already_has_length} documents already have text_length metadata")
    if needs_update > 0:
        print(f"  {needs_update} documents need text_length metadata")
    else:
        print("All documents already have text_length metadata. Nothing to do.")
        return

    if dry_run:
        print("\n[DRY RUN] Would update the following documents:")
    else:
        print("\nStarting metadata update...")

    # Process in batches
    updated_count = 0
    skipped_count = 0

    for i in range(0, len(documents), batch_size):
        batch_docs = documents[i:i+batch_size]
        batch_metas = metadatas[i:i+batch_size] if metadatas else [{}] * len(batch_docs)
        batch_ids = ids[i:i+batch_size]

        # Calculate text_length for each document
        updates = []
        update_texts = []  # Store texts for dry_run display
        for doc_id, doc_text, meta in zip(batch_ids, batch_docs, batch_metas):
            # Skip if already has text_length
            if meta and 'text_length' in meta:
                skipped_count += 1
                continue

            # Calculate text_length (character count)
            text_length = len(doc_text)

            # Update metadata
            new_meta = meta.copy() if meta else {}
            new_meta['text_length'] = text_length

            updates.append({
                'id': doc_id,
                'metadata': new_meta
            })
            update_texts.append(doc_text)

        # Update collection
        if updates:
            if dry_run:
                for update, doc_text in zip(updates[:5], update_texts[:5]):  # Show first 5 as examples
                    print(f"  ID: {update['id']}, text_length: {update['metadata']['text_length']}, "
                          f"text preview: {doc_text[:50]}...")
                if len(updates) > 5:
                    print(f"  ... and {len(updates) - 5} more")
            else:
                try:
                    collection.update(
                        ids=[u['id'] for u in updates],
                        metadatas=[u['metadata'] for u in updates]
                    )
                    updated_count += len(updates)
                    print(f"Updated {updated_count}/{needs_update} documents...")
                except Exception as e:
                    print(f"Warning: Failed to update batch: {e}")
                    import traceback
                    traceback.print_exc()

    if dry_run:
        print(f"\n[DRY RUN] Would update {len(updates)} documents in this batch.")
        print("Run without --dry-run to apply changes.")
    else:
        print(f"\n✓ Completed! Updated {updated_count} documents with text_length metadata.")
        if skipped_count > 0:
            print(f"  Skipped {skipped_count} documents that already had text_length.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Add text_length metadata to Style Atlas documents"
    )
    parser.add_argument(
        "--collection",
        default="style_atlas",
        help="ChromaDB collection name (default: style_atlas)"
    )
    parser.add_argument(
        "--persist-dir",
        help="ChromaDB persistence directory (default: from config.json)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=100,
        help="Batch size for processing (default: 100)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be updated without making changes"
    )

    args = parser.parse_args()

    # Load config if persist_dir not provided
    persist_dir = args.persist_dir
    if not persist_dir:
        try:
            with open("config.json", 'r') as f:
                config = json.load(f)
            atlas_config = config.get("atlas", {})
            persist_dir = atlas_config.get("persist_path", "atlas_cache")
        except Exception as e:
            print(f"Warning: Could not load config.json: {e}")
            print("Using default persist directory: atlas_cache")
            persist_dir = "atlas_cache"

    add_length_metadata(
        collection_name=args.collection,
        persist_directory=persist_dir,
        batch_size=args.batch_size,
        dry_run=args.dry_run
    )

