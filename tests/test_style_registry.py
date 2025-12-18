"""Tests for Style Registry functionality."""

import json
import os
import sys
import tempfile
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.atlas.style_registry import StyleRegistry


def test_style_registry_initialization():
    """Test that StyleRegistry initializes correctly."""
    with tempfile.TemporaryDirectory() as tmpdir:
        registry = StyleRegistry(tmpdir)

        assert registry.cache_dir == tmpdir
        assert registry.path == os.path.join(tmpdir, "author_profiles.json")
        assert registry.profiles == {}


def test_style_registry_set_and_get_dna():
    """Test setting and getting Style DNA."""
    with tempfile.TemporaryDirectory() as tmpdir:
        registry = StyleRegistry(tmpdir)

        # Set DNA for an author
        author_name = "Hemingway"
        dna = "Stoic, minimalist, and journalistic. Uses short declarative sentences with concrete nouns and strong verbs."
        registry.set_dna(author_name, dna)

        # Get DNA back
        retrieved_dna = registry.get_dna(author_name)
        assert retrieved_dna == dna

        # Verify file was created
        assert os.path.exists(registry.path)


def test_style_registry_has_dna():
    """Test checking if DNA exists for an author."""
    with tempfile.TemporaryDirectory() as tmpdir:
        registry = StyleRegistry(tmpdir)

        # Initially no DNA
        assert not registry.has_dna("Hemingway")

        # Set DNA
        registry.set_dna("Hemingway", "Test DNA")
        assert registry.has_dna("Hemingway")

        # Check non-existent author
        assert not registry.has_dna("Shakespeare")


def test_style_registry_persistence():
    """Test that Style DNA persists across registry instances."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create first registry and set DNA
        registry1 = StyleRegistry(tmpdir)
        registry1.set_dna("Hemingway", "Test DNA 1")
        registry1.set_dna("Lovecraft", "Test DNA 2")

        # Create second registry (should load from file)
        registry2 = StyleRegistry(tmpdir)

        # Verify DNA was loaded
        assert registry2.get_dna("Hemingway") == "Test DNA 1"
        assert registry2.get_dna("Lovecraft") == "Test DNA 2"
        assert registry2.has_dna("Hemingway")
        assert registry2.has_dna("Lovecraft")


def test_style_registry_multiple_authors():
    """Test storing DNA for multiple authors."""
    with tempfile.TemporaryDirectory() as tmpdir:
        registry = StyleRegistry(tmpdir)

        # Set DNA for multiple authors
        authors = {
            "Hemingway": "Stoic, minimalist, and journalistic.",
            "Lovecraft": "Archaic, dense, and atmosphere-heavy.",
            "Dawkins": "Clear, precise, and scientific."
        }

        for author, dna in authors.items():
            registry.set_dna(author, dna)

        # Verify all are stored
        for author, expected_dna in authors.items():
            assert registry.get_dna(author) == expected_dna
            assert registry.has_dna(author)


def test_style_registry_update_existing_dna():
    """Test updating DNA for an existing author."""
    with tempfile.TemporaryDirectory() as tmpdir:
        registry = StyleRegistry(tmpdir)

        # Set initial DNA
        registry.set_dna("Hemingway", "Original DNA")
        assert registry.get_dna("Hemingway") == "Original DNA"

        # Update DNA
        registry.set_dna("Hemingway", "Updated DNA")
        assert registry.get_dna("Hemingway") == "Updated DNA"


def test_style_registry_get_all_profiles():
    """Test getting all profiles."""
    with tempfile.TemporaryDirectory() as tmpdir:
        registry = StyleRegistry(tmpdir)

        # Set DNA for multiple authors
        registry.set_dna("Hemingway", "DNA 1")
        registry.set_dna("Lovecraft", "DNA 2")

        # Get all profiles
        all_profiles = registry.get_all_profiles()

        assert len(all_profiles) == 2
        assert "Hemingway" in all_profiles
        assert "Lovecraft" in all_profiles
        assert all_profiles["Hemingway"]["style_dna"] == "DNA 1"
        assert all_profiles["Lovecraft"]["style_dna"] == "DNA 2"
        assert "last_updated" in all_profiles["Hemingway"]
        assert "last_updated" in all_profiles["Lovecraft"]


def test_style_registry_empty_dna():
    """Test handling of empty DNA string."""
    with tempfile.TemporaryDirectory() as tmpdir:
        registry = StyleRegistry(tmpdir)

        # Set empty DNA
        registry.set_dna("Author", "")

        # Should return empty string
        assert registry.get_dna("Author") == ""
        # has_dna returns False for empty strings (which is correct behavior)
        assert not registry.has_dna("Author")


def test_style_registry_nonexistent_author():
    """Test getting DNA for non-existent author."""
    with tempfile.TemporaryDirectory() as tmpdir:
        registry = StyleRegistry(tmpdir)

        # Get DNA for non-existent author
        dna = registry.get_dna("Nonexistent")
        assert dna == ""
        assert not registry.has_dna("Nonexistent")


def test_style_registry_json_format():
    """Test that the JSON file is properly formatted."""
    with tempfile.TemporaryDirectory() as tmpdir:
        registry = StyleRegistry(tmpdir)

        registry.set_dna("Hemingway", "Test DNA")

        # Read and parse JSON file
        with open(registry.path, 'r') as f:
            data = json.load(f)

        # Verify structure
        assert "Hemingway" in data
        assert "style_dna" in data["Hemingway"]
        assert "last_updated" in data["Hemingway"]
        assert data["Hemingway"]["style_dna"] == "Test DNA"


def test_style_registry_corrupted_json():
    """Test handling of corrupted JSON file."""
    with tempfile.TemporaryDirectory() as tmpdir:
        registry_path = os.path.join(tmpdir, "author_profiles.json")

        # Create corrupted JSON file
        with open(registry_path, 'w') as f:
            f.write("{ invalid json }")

        # Should handle gracefully and start with empty dict
        registry = StyleRegistry(tmpdir)
        assert registry.profiles == {}

        # Should still be able to set DNA
        registry.set_dna("Hemingway", "Test DNA")
        assert registry.get_dna("Hemingway") == "Test DNA"


def test_style_registry_directory_creation():
    """Test that StyleRegistry creates the cache directory if it doesn't exist."""
    with tempfile.TemporaryDirectory() as tmpdir:
        cache_dir = os.path.join(tmpdir, "new_cache_dir")

        # Directory doesn't exist yet
        assert not os.path.exists(cache_dir)

        # Create registry (should create directory)
        registry = StyleRegistry(cache_dir)

        # Directory should now exist
        assert os.path.exists(cache_dir)
        assert os.path.isdir(cache_dir)


def test_style_registry_long_dna():
    """Test storing long Style DNA strings."""
    with tempfile.TemporaryDirectory() as tmpdir:
        registry = StyleRegistry(tmpdir)

        # Create a long DNA string
        long_dna = " ".join(["Test"] * 100)  # 100 words

        registry.set_dna("Author", long_dna)
        retrieved = registry.get_dna("Author")

        assert retrieved == long_dna
        assert len(retrieved) == len(long_dna)


if __name__ == "__main__":
    print("Running Style Registry tests...\n")

    try:
        test_style_registry_initialization()
        print("✓ Initialization test passed")

        test_style_registry_set_and_get_dna()
        print("✓ Set and get DNA test passed")

        test_style_registry_has_dna()
        print("✓ Has DNA test passed")

        test_style_registry_persistence()
        print("✓ Persistence test passed")

        test_style_registry_multiple_authors()
        print("✓ Multiple authors test passed")

        test_style_registry_update_existing_dna()
        print("✓ Update existing DNA test passed")

        test_style_registry_get_all_profiles()
        print("✓ Get all profiles test passed")

        test_style_registry_empty_dna()
        print("✓ Empty DNA test passed")

        test_style_registry_nonexistent_author()
        print("✓ Nonexistent author test passed")

        test_style_registry_json_format()
        print("✓ JSON format test passed")

        test_style_registry_corrupted_json()
        print("✓ Corrupted JSON test passed")

        test_style_registry_directory_creation()
        print("✓ Directory creation test passed")

        test_style_registry_long_dna()
        print("✓ Long DNA test passed")

        print("\n✓ All Style Registry tests completed!")

    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

