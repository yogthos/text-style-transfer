"""Check if ChromaDB has word_count and sentence_count metadata.

This script verifies whether the ChromaDB collection contains word_count
and sentence_count metadata. If present, we could use it for faster filtering.
If missing, we'll calculate on-the-fly (which is acceptable).
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import json

try:
    import chromadb
    from chromadb.config import Settings
    CHROMADB_AVAILABLE = True
except ImportError:
    CHROMADB_AVAILABLE = False
    print("ChromaDB not available. Install with: pip install chromadb")
    sys.exit(1)


def check_metadata(collection_name: str = "style_atlas", persist_directory: str = "atlas_cache/"):
    """Check if ChromaDB collection has complexity metadata.

    Args:
        collection_name: Name of ChromaDB collection.
        persist_directory: Directory where ChromaDB is persisted.
    """
    # Load config
    config_path = Path(__file__).parent.parent / "config.json"
    if config_path.exists():
        with open(config_path, 'r') as f:
            config = json.load(f)
        atlas_config = config.get("atlas", {})
        persist_directory = atlas_config.get("persist_path", persist_directory)
        collection_name = atlas_config.get("collection_name", collection_name)

    print(f"Checking ChromaDB collection: {collection_name}")
    print(f"Persist directory: {persist_directory}")
    print()

    try:
        # Initialize ChromaDB client
        client = chromadb.PersistentClient(path=persist_directory)
        collection = client.get_collection(name=collection_name)

        # Get a sample of documents to check metadata
        results = collection.get(limit=10)

        if not results.get('documents'):
            print("⚠ Collection is empty. No metadata to check.")
            return

        metadatas = results.get('metadatas', [])
        if not metadatas:
            print("⚠ No metadata found in collection.")
            return

        # Check first document for metadata fields
        first_meta = metadatas[0]

        has_word_count = 'word_count' in first_meta
        has_sentence_count = 'sentence_count' in first_meta
        has_text_length = 'text_length' in first_meta

        print("Metadata fields found:")
        print(f"  - word_count: {'✓' if has_word_count else '✗'}")
        print(f"  - sentence_count: {'✓' if has_sentence_count else '✗'}")
        print(f"  - text_length: {'✓' if has_text_length else '✗'}")
        print()

        # Check how many documents have these fields
        word_count_count = sum(1 for m in metadatas if 'word_count' in m)
        sentence_count_count = sum(1 for m in metadatas if 'sentence_count' in m)

        print(f"Documents with word_count: {word_count_count}/{len(metadatas)}")
        print(f"Documents with sentence_count: {sentence_count_count}/{len(metadatas)}")
        print()

        if has_word_count and has_sentence_count:
            print("✓ Collection has complexity metadata!")
            print("  Note: We can use this for faster filtering in the future.")
            print("  For now, we'll calculate on-the-fly which is acceptable.")
        else:
            print("⚠ Collection missing complexity metadata.")
            print("  We'll calculate word_count and sentence_count on-the-fly.")
            print("  This is acceptable - no database update needed.")

        # Show sample metadata
        if metadatas:
            print("\nSample metadata:")
            sample_keys = ['word_count', 'sentence_count', 'text_length', 'rhetorical_type']
            for key in sample_keys:
                if key in first_meta:
                    print(f"  {key}: {first_meta[key]}")

    except Exception as e:
        print(f"✗ Error checking collection: {e}")
        print("  Collection may not exist yet. This is okay - we'll calculate metadata on-the-fly.")


if __name__ == "__main__":
    check_metadata()

