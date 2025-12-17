"""Style Atlas module for ChromaDB-based style retrieval."""

from src.atlas.builder import StyleAtlas, build_style_atlas, save_atlas, load_atlas
from src.atlas.navigator import build_cluster_markov, predict_next_cluster, retrieve_style_reference

__all__ = [
    'StyleAtlas',
    'build_style_atlas',
    'save_atlas',
    'load_atlas',
    'build_cluster_markov',
    'predict_next_cluster',
    'retrieve_style_reference',
]

