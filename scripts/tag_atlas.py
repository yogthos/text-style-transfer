"""Script to tag existing Style Atlas with rhetorical types.

This script iterates over a ChromaDB collection and classifies each
document into a rhetorical type (DEFINITION, ARGUMENT, OBSERVATION, IMPERATIVE).
"""

import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.atlas.rhetoric import RhetoricalClassifier, RhetoricalType
from src.generator.llm_provider import LLMProvider
import chromadb
from chromadb.config import Settings


def tag_atlas(
    collection_name: str = "style_atlas",
    persist_directory: str = None,
    use_llm: bool = False,
    batch_size: int = 100
):
    """Tag all documents in atlas with rhetorical types.

    Args:
        collection_name: Name of ChromaDB collection.
        persist_directory: Directory where ChromaDB is persisted.
        use_llm: If True, use LLM classification (slower, more accurate).
                 If False, use heuristic classification (faster).
        batch_size: Number of documents to process in each batch.
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

    # Initialize classifier
    classifier = RhetoricalClassifier()
    llm_provider = None
    if use_llm:
        try:
            llm_provider = LLMProvider(config_path="config.json")
        except Exception as e:
            print(f"Warning: Could not initialize LLM provider: {e}")
            print("Falling back to heuristic classification.")
            use_llm = False

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

    print(f"Found {len(documents)} documents. Starting classification...")

    # Process in batches
    updated_count = 0
    for i in range(0, len(documents), batch_size):
        batch_docs = documents[i:i+batch_size]
        batch_metas = metadatas[i:i+batch_size] if metadatas else [{}] * len(batch_docs)
        batch_ids = ids[i:i+batch_size]

        # Classify each document
        updates = []
        for doc_id, doc_text, meta in zip(batch_ids, batch_docs, batch_metas):
            if use_llm and llm_provider:
                rtype = classifier.classify_llm(doc_text, llm_provider)
            else:
                rtype = classifier.classify_heuristic(doc_text)

            # Update metadata
            new_meta = meta.copy() if meta else {}
            new_meta['rhetorical_type'] = rtype.value

            updates.append({
                'id': doc_id,
                'metadata': new_meta
            })

        # Update collection
        try:
            collection.update(
                ids=[u['id'] for u in updates],
                metadatas=[u['metadata'] for u in updates]
            )
            updated_count += len(updates)
            print(f"Updated {updated_count}/{len(documents)} documents...")
        except Exception as e:
            print(f"Warning: Failed to update batch: {e}")

    print(f"\n✓ Completed! Tagged {updated_count} documents with rhetorical types.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Tag Style Atlas with rhetorical types")
    parser.add_argument("--collection", default="style_atlas", help="ChromaDB collection name")
    parser.add_argument("--persist-dir", help="ChromaDB persistence directory")
    parser.add_argument("--use-llm", action="store_true", help="Use LLM classification (slower, more accurate)")
    parser.add_argument("--batch-size", type=int, default=100, help="Batch size for processing")

    args = parser.parse_args()

    tag_atlas(
        collection_name=args.collection,
        persist_directory=args.persist_dir,
        use_llm=args.use_llm,
        batch_size=args.batch_size
    )

