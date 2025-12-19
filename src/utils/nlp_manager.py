"""NLP Manager: Singleton service for spaCy model lifecycle management.

This module provides a centralized, thread-safe way to load and access
the spaCy language model, preventing memory leaks and redundant loading.
"""

import spacy
from typing import Optional


class NLPManager:
    """Singleton service for managing spaCy model lifecycle.

    Provides lazy loading, thread-safe access, and optional cache clearing
    for memory management.
    """
    _instance: Optional[spacy.Language] = None
    _model_name: str = "en_core_web_sm"

    @classmethod
    def get_nlp(cls) -> spacy.Language:
        """Lazy-load the spaCy model. Thread-safe singleton access.

        Returns:
            Loaded spaCy language model.

        Raises:
            RuntimeError: If model cannot be loaded or downloaded.
        """
        if cls._instance is None:
            print(f"Loading shared spaCy model: {cls._model_name}...")
            try:
                cls._instance = spacy.load(cls._model_name)
            except OSError:
                # Model not found, try to download it
                try:
                    from spacy.cli import download
                    download(cls._model_name)
                    cls._instance = spacy.load(cls._model_name)
                except Exception as e:
                    raise RuntimeError(
                        f"spaCy model '{cls._model_name}' not found and could not be downloaded. "
                        f"Please install it with: python -m spacy download {cls._model_name}. "
                        f"Error: {e}"
                    )
        return cls._instance

    @classmethod
    def clear_cache(cls) -> None:
        """Clear the cached model instance.

        Useful for memory management or when switching models.
        Note: This will force reload on next access.
        """
        cls._instance = None

    @classmethod
    def set_model(cls, model_name: str) -> None:
        """Set the model name to use (before first access).

        Args:
            model_name: Name of spaCy model to load (e.g., 'en_core_web_sm', 'en_core_web_lg').

        Raises:
            RuntimeError: If model is already loaded and different.
        """
        if cls._instance is not None and cls._model_name != model_name:
            raise RuntimeError(
                f"Model already loaded as '{cls._model_name}'. "
                "Clear cache first with clear_cache() before changing model."
            )
        cls._model_name = model_name

