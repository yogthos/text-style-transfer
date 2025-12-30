"""Triplet extraction using REBEL model.

Extracts (subject, relation, object) triplets from text using the
REBEL (Relation Extraction By End-to-end Language generation) model.

This provides more accurate entity-role extraction than hand-rolled
spaCy-based extraction.
"""

import re
from dataclasses import dataclass
from typing import List, Optional, Tuple
from functools import lru_cache

from ..utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class Triplet:
    """A subject-relation-object triplet."""
    subject: str
    relation: str
    object: str
    confidence: float = 1.0

    def __str__(self) -> str:
        return f"({self.subject} | {self.relation} | {self.object})"

    def to_dict(self) -> dict:
        return {
            "subject": self.subject,
            "relation": self.relation,
            "object": self.object,
            "confidence": self.confidence,
        }


class REBELExtractor:
    """Extract triplets using the REBEL model.

    REBEL is a seq2seq model that extracts relation triplets by
    generating them as a sequence of tokens.
    """

    _instance = None
    _model = None
    _tokenizer = None

    def __new__(cls):
        """Singleton pattern - only load model once."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        """Initialize the REBEL extractor."""
        if self._model is None:
            self._load_model()

    def _load_model(self):
        """Load the REBEL model and tokenizer."""
        try:
            from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
            import torch

            logger.info("Loading REBEL model (this may take a moment)...")

            model_name = "Babelscape/rebel-large"
            self._tokenizer = AutoTokenizer.from_pretrained(model_name)
            self._model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

            # Move to GPU if available
            if torch.cuda.is_available():
                self._model = self._model.cuda()
                logger.info("REBEL model loaded on GPU")
            else:
                logger.info("REBEL model loaded on CPU")

        except ImportError as e:
            logger.error(f"Failed to import transformers: {e}")
            raise
        except Exception as e:
            logger.error(f"Failed to load REBEL model: {e}")
            raise

    def extract(self, text: str, max_length: int = 256) -> List[Triplet]:
        """Extract triplets from text.

        Args:
            text: Input text to extract triplets from.
            max_length: Maximum token length for input/output.

        Returns:
            List of Triplet objects.
        """
        if not text or not text.strip():
            return []

        try:
            import torch

            # Tokenize
            inputs = self._tokenizer(
                text,
                return_tensors="pt",
                max_length=max_length,
                truncation=True,
            )

            # Move to same device as model
            if next(self._model.parameters()).is_cuda:
                inputs = {k: v.cuda() for k, v in inputs.items()}

            # Generate
            with torch.no_grad():
                outputs = self._model.generate(
                    **inputs,
                    max_length=max_length,
                    num_beams=3,
                    num_return_sequences=1,
                )

            # Decode
            decoded = self._tokenizer.decode(outputs[0], skip_special_tokens=False)

            # Parse triplets from REBEL output format
            triplets = self._parse_rebel_output(decoded)

            logger.debug(f"Extracted {len(triplets)} triplets from text")
            return triplets

        except Exception as e:
            logger.warning(f"Triplet extraction failed: {e}")
            return []

    def _parse_rebel_output(self, text: str) -> List[Triplet]:
        """Parse REBEL model output into triplets.

        REBEL outputs triplets in the format:
        <triplet> subject <subj> relation <obj> object

        Args:
            text: Raw REBEL output string.

        Returns:
            List of parsed Triplet objects.
        """
        triplets = []

        # REBEL format: <triplet> head <subj> relation <obj> tail
        # Split by <triplet> token
        parts = text.split("<triplet>")

        for part in parts[1:]:  # Skip first empty part
            try:
                # Extract components
                # Format: subject <subj> relation <obj> object </s>
                part = part.strip()

                # Find subject (before <subj>)
                if "<subj>" not in part:
                    continue

                subj_split = part.split("<subj>")
                subject = subj_split[0].strip()

                remainder = subj_split[1] if len(subj_split) > 1 else ""

                # Find relation and object (relation <obj> object)
                if "<obj>" not in remainder:
                    continue

                obj_split = remainder.split("<obj>")
                relation = obj_split[0].strip()
                obj_text = obj_split[1].strip() if len(obj_split) > 1 else ""

                # Clean up object (remove </s> and other tokens)
                obj_text = obj_text.replace("</s>", "").replace("<pad>", "").strip()

                if subject and relation and obj_text:
                    triplets.append(Triplet(
                        subject=subject,
                        relation=relation,
                        object=obj_text,
                    ))

            except Exception as e:
                logger.debug(f"Failed to parse triplet part: {part[:50]}... - {e}")
                continue

        return triplets


class FallbackExtractor:
    """Fallback triplet extractor using spaCy when REBEL isn't available."""

    def __init__(self):
        self._nlp = None

    @property
    def nlp(self):
        if self._nlp is None:
            from ..utils.nlp import get_nlp
            self._nlp = get_nlp()
        return self._nlp

    def extract(self, text: str) -> List[Triplet]:
        """Extract triplets using spaCy dependency parsing.

        This is a fallback when REBEL isn't available.
        Less accurate but works without GPU/large models.
        """
        if not text or not text.strip():
            return []

        triplets = []
        doc = self.nlp(text)

        for sent in doc.sents:
            # Find main verb
            verb = None
            for token in sent:
                if token.pos_ == "VERB" and token.dep_ in ("ROOT", "relcl", "advcl"):
                    verb = token
                    break

            if not verb:
                continue

            # Find subject
            subject = None
            for child in verb.children:
                if child.dep_ in ("nsubj", "nsubjpass"):
                    subject = self._get_span_text(child)
                    break

            # Find object
            obj = None
            for child in verb.children:
                if child.dep_ in ("dobj", "pobj", "attr", "oprd"):
                    obj = self._get_span_text(child)
                    break

            # Also check for prepositional objects
            if not obj:
                for child in verb.children:
                    if child.dep_ == "prep":
                        for grandchild in child.children:
                            if grandchild.dep_ == "pobj":
                                obj = self._get_span_text(grandchild)
                                break

            if subject and obj:
                triplets.append(Triplet(
                    subject=subject,
                    relation=verb.lemma_,
                    object=obj,
                    confidence=0.7,  # Lower confidence for fallback
                ))

        return triplets

    def _get_span_text(self, token) -> str:
        """Get the full noun phrase for a token."""
        # Get subtree for compound nouns and modifiers
        tokens = []
        for t in token.subtree:
            if t.dep_ not in ("punct", "cc", "conj", "prep"):
                tokens.append(t.text)
            if len(tokens) > 8:  # Limit length
                break
        return " ".join(tokens)


# Global extractor instance
_extractor: Optional[REBELExtractor] = None
_fallback_extractor: Optional[FallbackExtractor] = None


def get_triplet_extractor(use_rebel: bool = True) -> 'REBELExtractor | FallbackExtractor':
    """Get the triplet extractor (singleton).

    Args:
        use_rebel: Whether to try loading REBEL (requires transformers).

    Returns:
        Triplet extractor instance.
    """
    global _extractor, _fallback_extractor

    if use_rebel:
        if _extractor is None:
            try:
                _extractor = REBELExtractor()
            except Exception as e:
                logger.warning(f"REBEL not available, using fallback: {e}")
                if _fallback_extractor is None:
                    _fallback_extractor = FallbackExtractor()
                return _fallback_extractor
        return _extractor
    else:
        if _fallback_extractor is None:
            _fallback_extractor = FallbackExtractor()
        return _fallback_extractor


def extract_triplets(text: str, use_rebel: bool = True) -> List[Triplet]:
    """Convenience function to extract triplets from text.

    Args:
        text: Input text.
        use_rebel: Whether to use REBEL model.

    Returns:
        List of triplets.
    """
    extractor = get_triplet_extractor(use_rebel)
    return extractor.extract(text)
