"""Style Profiler: Forensic linguistic analysis for dynamic style extraction.

This module extracts a unique "voice fingerprint" from any text corpus by analyzing
POV, burstiness, vocabulary, sentence starters, and punctuation patterns.
"""

import re
from collections import Counter
from typing import Dict, List, Any
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from src.utils.nlp_manager import NLPManager


class StyleProfiler:
    """Extracts style characteristics from text corpus using forensic linguistic analysis."""

    def __init__(self):
        """Initialize the style profiler."""
        self.nlp = NLPManager.get_nlp()

    def analyze_style(self, text: str) -> Dict[str, Any]:
        """Analyze style characteristics from text corpus.

        Extracts 5 dimensions:
        1. POV & Address (Point of View)
        2. Burstiness (Jaggedness Index)
        3. Signature Vocabulary (Keywords)
        4. Connective Tissue (Sentence Starters)
        5. Punctuation Signature

        Args:
            text: Full corpus text to analyze

        Returns:
            Dictionary containing all style characteristics
        """
        if not text or not text.strip():
            return self._empty_profile()

        # Parse text with spaCy
        doc = self.nlp(text)
        sentences = list(doc.sents)

        if not sentences:
            return self._empty_profile()

        # Extract all dimensions
        pov_data = self._analyze_pov(doc)
        burstiness_data = self._analyze_burstiness(sentences)
        keywords_data = self._analyze_keywords(text)
        openers_data = self._analyze_sentence_starters(sentences)
        punctuation_data = self._analyze_punctuation(doc, len(sentences))

        return {
            **pov_data,
            **burstiness_data,
            **keywords_data,
            **openers_data,
            **punctuation_data
        }

    def _empty_profile(self) -> Dict[str, Any]:
        """Return an empty profile structure."""
        return {
            "pov": "Third Person",
            "pov_breakdown": {"first_singular": 0, "first_plural": 0, "third_person": 0},
            "burstiness": 0.0,
            "rhythm_desc": "Unknown",
            "keywords": [],
            "keyword_frequencies": {},
            "common_openers": [],
            "opener_pattern": "Unknown",
            "semicolons_per_100": 0.0,
            "dashes_per_100": 0.0,
            "exclamations_per_100": 0.0,
            "punctuation_preference": "Standard"
        }

    def _analyze_pov(self, doc) -> Dict[str, Any]:
        """Analyze Point of View by counting pronouns.

        Args:
            doc: spaCy document

        Returns:
            Dictionary with POV classification and breakdown
        """
        from src.utils.spacy_linguistics import get_pov_pronouns

        # Get POV pronouns using spaCy
        pov_dict = get_pov_pronouns(doc)
        first_singular = pov_dict["first_singular"]
        first_plural = pov_dict["first_plural"]
        third_person = pov_dict["third_person"]

        counts = {
            "first_singular": 0,
            "first_plural": 0,
            "third_person": 0
        }

        # Count pronouns (case-insensitive)
        for token in doc:
            if token.pos_ == "PRON" or token.pos_ == "DET":
                token_lower = token.text.lower()
                if token_lower in first_singular:
                    counts["first_singular"] += 1
                elif token_lower in first_plural:
                    counts["first_plural"] += 1
                elif token_lower in third_person:
                    counts["third_person"] += 1

        # Determine dominant POV
        max_count = max(counts.values())
        if max_count == 0:
            pov = "Third Person"  # Default
        elif counts["first_singular"] == max_count:
            pov = "First Person Singular"
        elif counts["first_plural"] == max_count:
            pov = "First Person Plural"
        else:
            pov = "Third Person"

        return {
            "pov": pov,
            "pov_breakdown": counts
        }

    def _analyze_burstiness(self, sentences: List) -> Dict[str, Any]:
        """Analyze sentence length variation (burstiness).

        Args:
            sentences: List of spaCy sentence spans

        Returns:
            Dictionary with burstiness score and description
        """
        if not sentences or len(sentences) < 2:
            return {
                "burstiness": 0.0,
                "rhythm_desc": "Unknown"
            }

        # Calculate sentence lengths (in words)
        sentence_lengths = [len([token for token in sent if not token.is_punct]) for sent in sentences]

        if not sentence_lengths:
            return {
                "burstiness": 0.0,
                "rhythm_desc": "Unknown"
            }

        mean_len = np.mean(sentence_lengths)
        if mean_len == 0:
            return {
                "burstiness": 0.0,
                "rhythm_desc": "Unknown"
            }

        std_dev = np.std(sentence_lengths)
        burstiness = std_dev / mean_len  # Coefficient of Variation

        # Classify
        if burstiness < 0.4:
            rhythm_desc = "Smooth/Monotonous"
        elif burstiness < 0.6:
            rhythm_desc = "Moderate Variation"
        else:
            rhythm_desc = "Jagged/Volatile"

        return {
            "burstiness": round(float(burstiness), 3),
            "rhythm_desc": rhythm_desc
        }

    def _analyze_keywords(self, text: str) -> Dict[str, Any]:
        """Extract signature vocabulary using TF-IDF.

        Args:
            text: Full corpus text

        Returns:
            Dictionary with keywords and frequencies
        """
        if not text or len(text.strip()) < 50:
            return {
                "keywords": [],
                "keyword_frequencies": {}
            }

        try:
            # Use TF-IDF to find distinctive words
            # Split text into sentences for document frequency calculation
            sentences = re.split(r'[.!?]+\s+', text)
            sentences = [s.strip() for s in sentences if s.strip() and len(s.strip()) > 10]

            if len(sentences) < 2:
                # Fallback: treat entire text as one document
                sentences = [text]

            # Initialize vectorizer with safety parameters
            vectorizer = TfidfVectorizer(
                stop_words='english',
                max_features=50,
                min_df=2,  # Filter rare words/typos
                max_df=0.8,  # Filter generic words (appear in >80% of sentences)
                token_pattern=r'\b[a-zA-Z]{3,}\b'  # Only words with 3+ letters
            )

            # Fit and transform
            tfidf_matrix = vectorizer.fit_transform(sentences)
            feature_names = vectorizer.get_feature_names_out()

            # Get mean TF-IDF scores across all documents
            mean_scores = np.mean(tfidf_matrix.toarray(), axis=0)

            # Get top 30 keywords
            top_indices = np.argsort(mean_scores)[-30:][::-1]
            keywords = [feature_names[i] for i in top_indices if mean_scores[i] > 0]
            keyword_frequencies = {feature_names[i]: float(mean_scores[i]) for i in top_indices if mean_scores[i] > 0}

            return {
                "keywords": keywords[:30],  # Top 30
                "keyword_frequencies": keyword_frequencies
            }
        except Exception as e:
            # Fallback: return empty if TF-IDF fails
            return {
                "keywords": [],
                "keyword_frequencies": {}
            }

    def _analyze_sentence_starters(self, sentences: List) -> Dict[str, Any]:
        """Extract common sentence openers (smart extraction skipping punctuation).

        Args:
            sentences: List of spaCy sentence spans

        Returns:
            Dictionary with common openers and pattern classification
        """
        if not sentences:
            return {
                "common_openers": [],
                "opener_pattern": "Unknown"
            }

        openers = []
        # Smart extraction: skip leading punctuation/quotes
        opener_pattern = re.compile(r'^\W*(\w+)', re.IGNORECASE)

        for sent in sentences:
            sent_text = sent.text.strip()
            match = opener_pattern.search(sent_text)
            if match:
                first_word = match.group(1).lower()
                # Skip very short words that are likely artifacts
                if len(first_word) >= 2:
                    openers.append(first_word)

        if not openers:
            return {
                "common_openers": [],
                "opener_pattern": "Unknown"
            }

        # Count and get top 10
        opener_counts = Counter(openers)
        top_openers = [word for word, count in opener_counts.most_common(10)]

        # Classify pattern
        from src.utils.spacy_linguistics import get_discourse_markers, get_conjunctions

        # Get doc from first sentence if available
        doc = sentences[0].doc if sentences else None

        if doc is not None:
            # Get discourse markers using spaCy
            discourse_markers_list = get_discourse_markers(doc)
            conjunctions_list = get_conjunctions(doc)

            # Convert to sets for membership testing
            logical_markers = set(discourse_markers_list)  # Discourse markers are typically logical
            narrative_markers = set(conjunctions_list)  # Conjunctions are typically narrative
        else:
            # Fallback to hardcoded lists if doc not available
            logical_markers = {"therefore", "however", "moreover", "furthermore", "consequently", "thus", "hence", "accordingly", "nevertheless", "nonetheless"}
            narrative_markers = {"and", "but", "then", "so", "or", "nor", "yet"}

        logical_count = sum(1 for opener in top_openers if opener in logical_markers)
        narrative_count = sum(1 for opener in top_openers if opener in narrative_markers)

        if logical_count > narrative_count:
            opener_pattern = "Logical"
        elif narrative_count > logical_count:
            opener_pattern = "Narrative"
        else:
            opener_pattern = "Mixed"

        return {
            "common_openers": top_openers,
            "opener_pattern": opener_pattern
        }

    def _analyze_punctuation(self, doc, num_sentences: int) -> Dict[str, Any]:
        """Analyze punctuation signature.

        Args:
            doc: spaCy document
            num_sentences: Number of sentences in the document

        Returns:
            Dictionary with punctuation statistics
        """
        if num_sentences == 0:
            return {
                "semicolons_per_100": 0.0,
                "dashes_per_100": 0.0,
                "exclamations_per_100": 0.0,
                "punctuation_preference": "Standard"
            }

        # Count punctuation marks
        semicolons = sum(1 for token in doc if token.text == ';')
        dashes = sum(1 for token in doc if token.text in ['—', '–', '-'] and token.pos_ == 'PUNCT')
        exclamations = sum(1 for token in doc if token.text == '!')

        # Calculate per 100 sentences
        multiplier = 100.0 / num_sentences
        semicolons_per_100 = semicolons * multiplier
        dashes_per_100 = dashes * multiplier
        exclamations_per_100 = exclamations * multiplier

        # Determine preference
        max_punct = max(semicolons_per_100, dashes_per_100, exclamations_per_100)
        if max_punct < 1.0:
            preference = "Standard"
        elif semicolons_per_100 == max_punct:
            preference = "Semicolons"
        elif dashes_per_100 == max_punct:
            preference = "Dashes"
        elif exclamations_per_100 == max_punct:
            preference = "Exclamations"
        else:
            preference = "Standard"

        return {
            "semicolons_per_100": round(semicolons_per_100, 2),
            "dashes_per_100": round(dashes_per_100, 2),
            "exclamations_per_100": round(exclamations_per_100, 2),
            "punctuation_preference": preference
        }

