"""Style Atlas module for ChromaDB-based style retrieval."""

from src.atlas.builder import StyleAtlas, build_style_atlas, save_atlas, load_atlas, clear_chromadb_collection
from src.atlas.navigator import (
    build_cluster_markov,
    predict_next_cluster,
    find_situation_match,
    find_structure_match,
    retrieve_window_match,
    StructureNavigator,
    sanitize_structural_template
)
from src.atlas.blender import StyleBlender

__all__ = [
    'StyleAtlas',
    'build_style_atlas',
    'save_atlas',
    'load_atlas',
    'clear_chromadb_collection',
    'build_cluster_markov',
    'predict_next_cluster',
    'find_situation_match',
    'find_structure_match',
    'retrieve_window_match',
    'StructureNavigator',
    'StyleBlender',
    'sanitize_structural_template',
]

