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
        # Cache for word filtering to avoid redundant NLP calls
        self._word_filter_cache = {}

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

        # Build entity blocklist first (from original case-sensitive text)
        entity_blocklist = self._build_entity_blocklist(doc)

        # Extract all dimensions
        pov_data = self._analyze_pov(doc)
        burstiness_data = self._analyze_burstiness(sentences)
        keywords_data = self._analyze_keywords(doc, entity_blocklist)
        openers_data = self._analyze_sentence_starters(sentences, entity_blocklist)
        punctuation_data = self._analyze_punctuation(doc, len(sentences))
        vocabulary_palette = self._extract_vocabulary_palette(doc)

        return {
            **pov_data,
            **burstiness_data,
            **keywords_data,
            **openers_data,
            **punctuation_data,
            "vocabulary_palette": vocabulary_palette
        }

    def _build_entity_blocklist(self, doc) -> set:
        """
        Scans the document for Proper Nouns and Named Entities to create a
        strict blocklist for keyword extraction.

        Args:
            doc: spaCy Doc object (must be from original case-sensitive text)

        Returns:
            Set of lowercase words/lemmas to block
        """
        blocklist = set()

        for token in doc:
            # 1. Catch Standard Proper Nouns
            if token.pos_ == "PROPN":
                blocklist.add(token.text.lower())
                blocklist.add(token.lemma_.lower())

            # 2. Catch Named Entities (PERSON, ORG, GPE, etc.)
            # ent_type_ is often more accurate than POS for names
            if token.ent_type_ in {"PERSON", "ORG", "GPE", "LOC", "PRODUCT", "NORP"}:
                blocklist.add(token.text.lower())
                blocklist.add(token.lemma_.lower())

        return blocklist

    def _is_general_word(self, word: str, blocklist: set = None) -> bool:
        """Check if a word is a general word (not a proper noun or specific term).

        Args:
            word: Word to check (lowercase string) OR token object
            blocklist: Set of lowercase words/lemmas to block (from _build_entity_blocklist)

        Returns:
            True if word is general (should be kept), False if it's a proper noun/specific term (should be filtered)
        """
        # Handle both string and token inputs
        if hasattr(word, 'text'):
            # It's a token object
            token = word
            text_lower = token.text.lower()
            lemma_lower = token.lemma_.lower()
        else:
            # It's a string
            if not word or len(word) < 3:
                return False
            text_lower = word.lower()
            lemma_lower = text_lower  # For string input, use word as lemma
            token = None

        # 1. Check strict blocklist first (Names detected by NER from original text)
        if blocklist:
            if text_lower in blocklist or lemma_lower in blocklist:
                return False

        # If we have a token, use direct checks (more accurate)
        if token is not None:
            # 2. Standard Filters
            if token.is_stop or token.is_punct or token.is_digit or token.like_num:
                return False

            # 3. Fallback POS Check (in case NER missed it but it's tagged PROPN)
            if token.pos_ == "PROPN":
                return False

            # 4. Named Entity Check (double-check)
            if token.ent_type_ in {"PERSON", "ORG", "GPE", "LOC", "PRODUCT", "NORP"}:
                return False

            return True

        # For string input, use the old method (for backward compatibility)
        # Check cache first
        cache_key = f"{text_lower}:{bool(blocklist)}"
        if cache_key in self._word_filter_cache:
            return self._word_filter_cache[cache_key]

        # Default to True (conservative: include if unsure)
        result = True

        try:
            # Process word in minimal sentence context to get accurate POS tags
            test_sentence = f"The {text_lower}."
            doc = self.nlp(test_sentence)

            # Find the word token (skip "The" and punctuation)
            for token in doc:
                if token.text.lower() == text_lower and not token.is_punct:
                    # 1. Basic Filters
                    if token.is_stop or token.is_punct or token.is_digit or token.like_num:
                        result = False
                        break

                    # 2. Part of Speech Filter (Exclude Proper Nouns)
                    if token.pos_ == "PROPN":
                        result = False
                        break

                    # 3. Named Entity Filter
                    if token.ent_type_ in {"PERSON", "ORG", "GPE", "LOC", "PRODUCT", "NORP"}:
                        result = False
                        break

                    # Keep common parts of speech (general vocabulary)
                    if token.pos_ in ["NOUN", "VERB", "ADJ", "ADV", "PRON", "DET", "CONJ", "SCONJ", "ADP", "PART"]:
                        result = True
                        break
                    # For other POS tags, default to keeping (conservative)
                    result = True
                    break
        except Exception:
            # If processing fails, be conservative and include the word
            result = True

        # Cache the result
        self._word_filter_cache[cache_key] = result
        return result

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
            "punctuation_preference": "Standard",
            "vocabulary_palette": {
                "general": [],
                "sensory_verbs": [],
                "connectives": [],
                "intensifiers": []
            }
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

    def _analyze_keywords(self, doc, entity_blocklist: set) -> Dict[str, Any]:
        """Extract signature vocabulary using token-level frequency analysis.

        Args:
            doc: spaCy Doc object (from original case-sensitive text)
            entity_blocklist: Set of lowercase words/lemmas to block

        Returns:
            Dictionary with keywords and frequencies
        """
        if not doc or len(doc) < 10:
            return {
                "keywords": [],
                "keyword_frequencies": {}
            }

        try:
            # Count words using token-level filtering with blocklist
            word_counts = Counter()
            for token in doc:
                # Use token-level check with blocklist
                if self._is_general_word(token, entity_blocklist):
                    lemma = token.lemma_.lower()
                    word_counts[lemma] += 1

            # Calculate frequencies (Total clean words)
            total_words = sum(word_counts.values())
            if total_words == 0:
                return {
                    "keywords": [],
                    "keyword_frequencies": {}
                }

            # Get top 30 keywords with frequencies
            top_keywords = word_counts.most_common(30)
            keywords = [word for word, count in top_keywords]
            keyword_frequencies = {
                word: count / total_words
                for word, count in top_keywords
            }

            return {
                "keywords": keywords,
                "keyword_frequencies": keyword_frequencies
            }
        except Exception as e:
            # Fallback: return empty if processing fails
            return {
                "keywords": [],
                "keyword_frequencies": {}
            }

    def _extract_vocabulary_palette(self, doc) -> Dict[str, List[str]]:
        """
        Extracts the author's preferred words categorized by function.

        Args:
            doc: spaCy Doc object

        Returns:
            Dictionary with categorized vocabulary lists (lemmas)
        """
        palette = {
            "general": [],      # High-frequency content words (Nouns/Adjs/Verbs)
            "sensory_verbs": [], # Verbs related to perception
            "connectives": [],   # Transition words
            "intensifiers": []  # Adverbs modifying adjectives
        }

        # Specialized lists
        sensory_lemmas = {'see', 'hear', 'feel', 'grasp', 'watch', 'listen', 'touch', 'smell', 'sense', 'perceive'}
        connective_deps = {'cc', 'mark', 'advmod'}

        word_counts = Counter()

        for token in doc:
            # 1. Basic Filters
            if token.is_stop or token.is_punct or token.is_digit or token.like_num:
                continue

            # 2. Part of Speech Filter (Exclude Proper Nouns)
            # PROPN = Proper Noun (e.g., "Rainer", "London")
            if token.pos_ == "PROPN":
                continue

            # 3. Named Entity Filter (Double-check)
            # Sometimes 'Rainer' might be tagged as NOUN if lowercased,
            # so we check if it's part of a named entity (PERSON, ORG, GPE).
            if token.ent_type_ in {"PERSON", "ORG", "GPE", "LOC", "PRODUCT"}:
                continue

            lemma = token.lemma_.lower()

            # Note: We skip the _is_general_word check here because we've already
            # done all the necessary filtering directly on the token (PROPN, named entities, etc.).
            # The _is_general_word method is used for keyword extraction where we only have
            # the word string and need to test it in isolation.

            # 1. Sensory Verbs
            if token.pos_ == "VERB" and lemma in sensory_lemmas:
                if lemma not in palette["sensory_verbs"]:
                    palette["sensory_verbs"].append(lemma)

            # 2. Connectives (simplified heuristic)
            if token.dep_ in connective_deps and token.pos_ in {'ADV', 'CCONJ', 'SCONJ'}:
                if lemma not in palette["connectives"]:
                    palette["connectives"].append(lemma)

            # 3. Intensifiers (adverbs modifying adjectives)
            if token.dep_ == "advmod" and token.head.pos_ == "ADJ":
                if lemma not in palette["intensifiers"]:
                    palette["intensifiers"].append(lemma)

            # 4. General Palette (Nouns/Adjs/Verbs)
            if token.pos_ in {'NOUN', 'VERB', 'ADJ'}:
                word_counts[lemma] += 1

        # Keep top N for general palette
        palette["general"] = [w for w, c in word_counts.most_common(50)]

        # Deduplicate lists (already done above, but ensure)
        palette["sensory_verbs"] = list(set(palette["sensory_verbs"]))
        palette["connectives"] = list(set(palette["connectives"]))
        palette["intensifiers"] = list(set(palette["intensifiers"]))

        return palette

    def _analyze_sentence_starters(self, sentences: List, entity_blocklist: set) -> Dict[str, Any]:
        """Extract common sentence openers (smart extraction skipping punctuation).

        Args:
            sentences: List of spaCy sentence spans
            entity_blocklist: Set of lowercase words/lemmas to block

        Returns:
            Dictionary with common openers and pattern classification
        """
        if not sentences:
            return {
                "common_openers": [],
                "opener_pattern": "Unknown"
            }

        openers = []
        # Use token-level extraction to get accurate POS and entity info
        for sent in sentences:
            # Get first non-punctuation token
            for token in sent:
                if not token.is_punct:
                    # Check if it's a general word (not a name) using blocklist
                    if self._is_general_word(token, entity_blocklist):
                        openers.append(token.text.lower())
                    break  # Only take the first valid token

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

