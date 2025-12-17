"""
Semantic Word Mapper Module

Maps common, non-technical words to their closest semantic equivalents
in the sample text vocabulary using word embeddings. This ensures the
output text uses vocabulary from the sample text as much as possible.

Key features:
- Uses spaCy word vectors for semantic similarity
- Filters by part of speech for grammatical correctness
- Builds mapping upfront during initialization
- Applies mappings during text processing
"""

import re
import spacy
import numpy as np
import sqlite3
import hashlib
import json
from pathlib import Path
from typing import Dict, List, Set, Optional, Tuple
from collections import Counter, defaultdict

# NLTK imports
try:
    import nltk
    from nltk.corpus import brown, reuters, gutenberg
    from nltk import FreqDist
    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False
    print("  [SemanticWordMapper] WARNING: NLTK not available. Install with: pip install nltk")


class SemanticWordMapper:
    """
    Maps common words to sample text vocabulary using semantic similarity.

    Uses word embeddings to find the closest semantic match in the sample
    text for each common word, ensuring grammatical correctness through
    POS filtering.
    """

    # Common non-technical words to map (organized by POS)
    COMMON_WORDS = {
        'ADJ': {
            'important', 'significant', 'large', 'small', 'major', 'minor',
            'key', 'main', 'primary', 'secondary', 'essential', 'necessary',
            'vital', 'critical', 'crucial', 'fundamental', 'basic', 'central',
            'principal', 'chief', 'leading', 'prominent', 'notable', 'remarkable',
            'substantial', 'considerable', 'extensive', 'comprehensive', 'complete',
            'effective', 'efficient', 'successful', 'powerful', 'strong', 'weak',
            'clear', 'obvious', 'evident', 'apparent', 'visible', 'noticeable',
            'different', 'similar', 'same', 'various', 'diverse', 'multiple',
            'numerous', 'many', 'few', 'several', 'some', 'most', 'all',
            'new', 'old', 'recent', 'ancient', 'modern', 'contemporary',
            'good', 'bad', 'better', 'worse', 'best', 'worst', 'excellent',
            'poor', 'adequate', 'sufficient', 'insufficient', 'enough'
        },
        'VERB': {
            'show', 'indicate', 'demonstrate', 'reveal', 'suggest', 'imply',
            'appear', 'seem', 'look', 'become', 'remain', 'stay', 'keep',
            'make', 'do', 'get', 'take', 'give', 'put', 'set', 'go', 'come',
            'use', 'utilize', 'employ', 'apply', 'implement', 'execute',
            'create', 'produce', 'generate', 'develop', 'build', 'construct',
            'establish', 'found', 'form', 'shape', 'design', 'plan', 'prepare',
            'begin', 'start', 'continue', 'proceed', 'advance', 'progress',
            'end', 'finish', 'complete', 'conclude', 'terminate', 'stop',
            'change', 'modify', 'alter', 'transform', 'convert', 'adapt',
            'increase', 'decrease', 'grow', 'expand', 'reduce', 'shrink',
            'improve', 'enhance', 'strengthen', 'weaken', 'support', 'help',
            'require', 'need', 'demand', 'request', 'ask', 'seek', 'find',
            'obtain', 'acquire', 'gain', 'achieve', 'accomplish', 'reach',
            'understand', 'know', 'learn', 'recognize', 'realize', 'see',
            'think', 'believe', 'consider', 'regard', 'view', 'perceive'
        },
        'NOUN': {
            'way', 'method', 'approach', 'manner', 'means', 'process', 'system',
            'way', 'path', 'route', 'course', 'direction', 'strategy', 'tactic',
            'thing', 'item', 'object', 'element', 'component', 'part', 'piece',
            'aspect', 'feature', 'characteristic', 'attribute', 'property',
            'type', 'kind', 'sort', 'category', 'class', 'group', 'set',
            'number', 'amount', 'quantity', 'level', 'degree', 'extent',
            'time', 'period', 'duration', 'moment', 'instance', 'occasion',
            'place', 'location', 'position', 'site', 'spot', 'area', 'region',
            'person', 'people', 'individual', 'person', 'human', 'man', 'woman',
            'group', 'team', 'organization', 'institution', 'company', 'firm',
            'problem', 'issue', 'challenge', 'difficulty', 'obstacle', 'barrier',
            'solution', 'answer', 'response', 'reaction', 'result', 'outcome',
            'effect', 'impact', 'influence', 'consequence', 'implication',
            'reason', 'cause', 'factor', 'element', 'component', 'aspect',
            'purpose', 'goal', 'objective', 'aim', 'target', 'intention',
            'information', 'data', 'fact', 'detail', 'point', 'note', 'remark'
        }
    }

    # Flatten all common words for quick lookup (kept as fallback)
    ALL_COMMON_WORDS = set()
    for pos_words in COMMON_WORDS.values():
        ALL_COMMON_WORDS.update(pos_words)

    # Database path for storing mappings
    DB_PATH = Path(__file__).parent / "word_mappings.db"

    @staticmethod
    def _ensure_nltk_data():
        """Download required NLTK data if not already present."""
        if not NLTK_AVAILABLE:
            return False

        try:
            # Check if corpora are available
            try:
                brown.words()
            except LookupError:
                print("  [SemanticWordMapper] Downloading NLTK Brown corpus...")
                nltk.download('brown', quiet=True)

            try:
                reuters.words()
            except LookupError:
                print("  [SemanticWordMapper] Downloading NLTK Reuters corpus...")
                nltk.download('reuters', quiet=True)

            try:
                gutenberg.words()
            except LookupError:
                print("  [SemanticWordMapper] Downloading NLTK Gutenberg corpus...")
                nltk.download('gutenberg', quiet=True)

            # Download POS tagger if needed
            try:
                nltk.pos_tag(['test'])
            except LookupError:
                print("  [SemanticWordMapper] Downloading NLTK POS tagger...")
                nltk.download('averaged_perceptron_tagger', quiet=True)

            return True
        except Exception as e:
            print(f"  [SemanticWordMapper] Error downloading NLTK data: {e}")
            return False

    @staticmethod
    def _get_common_words_from_nltk(corpus_name: str = "brown", words_per_pos: int = 2000) -> Dict[str, Set[str]]:
        """
        Get common words from NLTK frequency lists.

        Args:
            corpus_name: Name of corpus to use ("brown", "reuters", "gutenberg", or "combined")
            words_per_pos: Number of top words to extract per POS category

        Returns:
            Dictionary mapping POS tags to sets of common words
        """
        if not NLTK_AVAILABLE:
            return {}

        # Ensure NLTK data is available
        SemanticWordMapper._ensure_nltk_data()

        # Map spaCy POS tags to NLTK POS tags
        spacy_to_nltk_pos = {
            'ADJ': ['JJ', 'JJR', 'JJS'],  # Adjective, comparative, superlative
            'VERB': ['VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ'],  # Verb forms
            'NOUN': ['NN', 'NNS', 'NNP', 'NNPS'],  # Noun forms
            'ADV': ['RB', 'RBR', 'RBS'],  # Adverb forms
        }

        # Select corpus
        corpora = []
        if corpus_name == "brown":
            corpora = [brown]
        elif corpus_name == "reuters":
            corpora = [reuters]
        elif corpus_name == "gutenberg":
            corpora = [gutenberg]
        elif corpus_name == "combined":
            corpora = [brown, reuters, gutenberg]
        else:
            corpora = [brown]  # Default to brown

        # Collect tagged words from all selected corpora
        all_tagged_words = []
        for corpus in corpora:
            try:
                tagged = corpus.tagged_words()
                all_tagged_words.extend(tagged)
            except Exception as e:
                print(f"  [SemanticWordMapper] Warning: Could not load {corpus_name} corpus: {e}")
                continue

        if not all_tagged_words:
            return {}

        # Count words by POS
        pos_word_counts = defaultdict(Counter)
        for word, tag in all_tagged_words:
            word_lower = word.lower()
            # Skip non-alphabetic, very short/long words, and stop words
            if not word_lower.isalpha() or len(word_lower) < 3 or len(word_lower) > 20:
                continue

            # Map NLTK tag to spaCy POS
            for spacy_pos, nltk_tags in spacy_to_nltk_pos.items():
                if tag in nltk_tags:
                    pos_word_counts[spacy_pos][word_lower] += 1
                    break

        # Get top N words per POS
        result = {
            'ADJ': set(),
            'VERB': set(),
            'NOUN': set(),
            'ADV': set(),
        }

        for pos, word_counts in pos_word_counts.items():
            if pos in result:
                top_words = [word for word, count in word_counts.most_common(words_per_pos)]
                result[pos].update(top_words)

        return result

    @staticmethod
    def _get_expanded_common_words(nlp_model, target_size: int = 2000):
        """
        Get expanded common words from spaCy vocabulary.

        Returns a dictionary of POS -> set of words, expanding beyond
        the hardcoded COMMON_WORDS to include more vocabulary.
        """
        expanded = {
            'ADJ': set(),
            'VERB': set(),
            'NOUN': set(),
            'ADV': set(),  # Add adverbs too
        }

        # Start with existing common words
        for pos, words in SemanticWordMapper.COMMON_WORDS.items():
            if pos in expanded:
                expanded[pos].update(words)

        # Add words from spaCy vocabulary
        words_added = 0
        for word_string in nlp_model.vocab.strings:
            if words_added >= target_size:
                break

            token = nlp_model.vocab[word_string]

            # Only add words with vectors, alphabetic, reasonable length
            if not token.has_vector or not word_string.isalpha():
                continue
            if len(word_string) < 3 or len(word_string) > 20:
                continue
            if token.is_stop:
                continue

            # Get POS by parsing
            try:
                doc = nlp_model(word_string)
                if doc:
                    pos = doc[0].pos_
                    if pos in expanded:
                        expanded[pos].add(word_string.lower())
                        words_added += 1
            except:
                continue

        return expanded

    def _init_mapping_db(self):
        """Initialize SQLite database for storing word mappings."""
        conn = sqlite3.connect(self.DB_PATH)
        cursor = conn.cursor()

        cursor.execute('''
            CREATE TABLE IF NOT EXISTS word_mappings (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                source_word TEXT NOT NULL,
                target_word TEXT NOT NULL,
                pos TEXT NOT NULL,
                similarity_score REAL NOT NULL,
                sample_hash TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(source_word, pos, sample_hash)
            )
        ''')

        # Create indexes for fast lookups
        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_source_pos
            ON word_mappings(source_word, pos)
        ''')

        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_sample_hash
            ON word_mappings(sample_hash)
        ''')

        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_similarity
            ON word_mappings(similarity_score)
        ''')

        conn.commit()
        conn.close()

    def _get_sample_hash(self, sample_text: str) -> str:
        """Generate hash of sample text for cache lookup."""
        return hashlib.md5(sample_text.encode()).hexdigest()

    def _save_mapping_to_db(self, source_word: str, target_word: str, pos: str, similarity_score: float, sample_hash: str):
        """Store a word mapping in the database."""
        conn = sqlite3.connect(self.DB_PATH)
        cursor = conn.cursor()

        try:
            cursor.execute('''
                INSERT OR REPLACE INTO word_mappings
                (source_word, target_word, pos, similarity_score, sample_hash)
                VALUES (?, ?, ?, ?, ?)
            ''', (source_word.lower(), target_word.lower(), pos, similarity_score, sample_hash))
            conn.commit()
        except sqlite3.Error as e:
            print(f"  [SemanticWordMapper] Error saving mapping to DB: {e}")
        finally:
            conn.close()

    def _load_mappings_from_db(self, sample_hash: str) -> Dict[str, Dict[str, str]]:
        """
        Load all cached mappings for a sample from database.

        Returns:
            Dictionary mapping POS -> {source_word -> target_word}
        """
        mappings = defaultdict(dict)
        conn = sqlite3.connect(self.DB_PATH)
        cursor = conn.cursor()

        try:
            cursor.execute('''
                SELECT source_word, target_word, pos, similarity_score
                FROM word_mappings
                WHERE sample_hash = ?
                ORDER BY similarity_score DESC
            ''', (sample_hash,))

            rows = cursor.fetchall()
            for source_word, target_word, pos, similarity_score in rows:
                # Only use mappings that meet similarity threshold
                # Ensure similarity_score is a valid number
                if similarity_score is not None and isinstance(similarity_score, (int, float)):
                    if float(similarity_score) >= self.similarity_threshold:
                        mappings[pos][source_word] = target_word
        except sqlite3.Error as e:
            print(f"  [SemanticWordMapper] Error loading mappings from DB: {e}")
        finally:
            conn.close()

        return mappings

    def _get_mapping_from_db(self, source_word: str, pos: str, sample_hash: str) -> Optional[str]:
        """Get a specific mapping from database if it exists."""
        conn = sqlite3.connect(self.DB_PATH)
        cursor = conn.cursor()

        try:
            cursor.execute('''
                SELECT target_word, similarity_score
                FROM word_mappings
                WHERE source_word = ? AND pos = ? AND sample_hash = ?
            ''', (source_word.lower(), pos, sample_hash))

            row = cursor.fetchone()
            if row:
                target_word, similarity_score = row
                # Only return if meets similarity threshold
                # Ensure similarity_score is a valid number
                if similarity_score is not None and isinstance(similarity_score, (int, float)):
                    if float(similarity_score) >= self.similarity_threshold:
                        return target_word
        except sqlite3.Error as e:
            print(f"  [SemanticWordMapper] Error getting mapping from DB: {e}")
        finally:
            conn.close()

        return None

    def __init__(self, sample_text: str, nlp_model=None, similarity_threshold: float = 0.3, min_sample_frequency: int = 1, max_spacy_words: int = 10000, use_nltk: bool = True, nltk_corpus: str = "brown", words_per_pos: int = 2000, cache_mappings: bool = True):
        """
        Initialize mapper with sample text vocabulary.

        Args:
            sample_text: The target style sample to derive mappings from
            nlp_model: spaCy model (if None, will load en_core_web_sm)
            similarity_threshold: Minimum cosine similarity for mapping (0.0-1.0)
            min_sample_frequency: Minimum occurrences in sample to consider a word
        """
        # Load spaCy model (prefer one with word vectors)
        # Check if passed model has vectors, if not, try to find one that does
        if nlp_model is not None:
            # Check if the passed model has vectors
            if nlp_model.vocab.vectors.size > 0:
                self.nlp = nlp_model
            else:
                # Passed model has no vectors, try to find one that does
                nlp_model = None  # Reset to None to trigger model search

        if nlp_model is None:
            # Try to load a model with vectors, fallback to sm if not available
            models_to_try = ["en_core_web_md", "en_core_web_lg", "en_core_web_sm"]
            self.nlp = None

            for model_name in models_to_try:
                try:
                    candidate_nlp = spacy.load(model_name)
                    # Check if model has vectors
                    if candidate_nlp.vocab.vectors.size > 0:
                        self.nlp = candidate_nlp
                        break
                    else:
                        # Model loaded but no vectors, try next
                        continue
                except OSError:
                    continue

            # If no model with vectors found, try to download md model
            if self.nlp is None or self.nlp.vocab.vectors.size == 0:
                try:
                    import subprocess
                    print(f"  [SemanticWordMapper] Downloading en_core_web_md (required for word vectors)...")
                    result = subprocess.run(["python", "-m", "spacy", "download", "en_core_web_md"],
                                           capture_output=True, text=True, check=False)
                    if result.returncode == 0:
                        self.nlp = spacy.load("en_core_web_md")
                    else:
                        raise OSError("Download failed")
                except (OSError, subprocess.CalledProcessError) as e:
                    # Fallback to sm model (will warn about no vectors)
                    try:
                        self.nlp = spacy.load("en_core_web_sm")
                    except OSError:
                        import subprocess
                        subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"], check=False)
                        self.nlp = spacy.load("en_core_web_sm")

        self.similarity_threshold = similarity_threshold
        self.min_sample_frequency = min_sample_frequency
        self.max_spacy_words = max_spacy_words
        self.use_nltk = use_nltk
        self.nltk_corpus = nltk_corpus
        self.words_per_pos = words_per_pos
        self.cache_mappings = cache_mappings
        self.sample_text = sample_text
        self.sample_hash = self._get_sample_hash(sample_text)

        # Initialize database for storing mappings
        self._init_mapping_db()

        # Extract sample vocabulary
        self.sample_vocabulary: Dict[str, Dict[str, Tuple[np.ndarray, int]]] = defaultdict(dict)
        self._extract_sample_vocabulary(sample_text)

        # Get common words from NLTK + spaCy hybrid approach
        self.expanded_common_words = self._get_common_words_from_sources()

        # Update ALL_COMMON_WORDS to include expanded set
        for pos_words in self.expanded_common_words.values():
            self.ALL_COMMON_WORDS.update(pos_words)

        # Build mappings from common words to sample equivalents
        self.mappings: Dict[str, Dict[str, str]] = defaultdict(dict)  # POS -> {common_word -> sample_word}

        # Only build mappings if model has vectors
        if self.nlp.vocab.vectors.size > 0:
            self._build_mappings()
            total_mappings = sum(len(m) for m in self.mappings.values())
            if total_mappings > 0:
                print(f"  [SemanticWordMapper] Built {total_mappings} word mappings")
            else:
                print(f"  [SemanticWordMapper] Built 0 word mappings (no matches found above similarity threshold)")
        else:
            # This should rarely happen now since we try to load a model with vectors
            print(f"  [SemanticWordMapper] WARNING: Model has no word vectors. Semantic mapping disabled.")
            print(f"  [SemanticWordMapper] Install en_core_web_md for word vector support: python -m spacy download en_core_web_md")

    def _get_common_words_from_sources(self) -> Dict[str, Set[str]]:
        """
        Get common words from NLTK + spaCy hybrid approach.

        Returns:
            Dictionary mapping POS tags to sets of common words
        """
        result = {
            'ADJ': set(),
            'VERB': set(),
            'NOUN': set(),
            'ADV': set(),
        }

        # Get words from NLTK if enabled
        if self.use_nltk and NLTK_AVAILABLE:
            nltk_words = self._get_common_words_from_nltk(
                corpus_name=self.nltk_corpus,
                words_per_pos=self.words_per_pos
            )
            for pos, words in nltk_words.items():
                if pos in result:
                    result[pos].update(words)
            print(f"  [SemanticWordMapper] Loaded {sum(len(w) for w in nltk_words.values())} words from NLTK {self.nltk_corpus} corpus")

        # Supplement with spaCy vocabulary
        spacy_words = self._get_expanded_common_words(self.nlp, target_size=self.words_per_pos)
        for pos, words in spacy_words.items():
            if pos in result:
                result[pos].update(words)

        # Fallback to manual COMMON_WORDS if NLTK not available and spaCy didn't provide enough
        if not self.use_nltk or not NLTK_AVAILABLE:
            for pos, words in self.COMMON_WORDS.items():
                if pos in result:
                    result[pos].update(words)

        return result

    def _extract_sample_vocabulary(self, text: str):
        """Extract vocabulary from sample text with word vectors and frequencies."""
        doc = self.nlp(text)

        # Count words by POS and store vectors
        pos_words: Dict[str, Counter] = defaultdict(Counter)
        word_vectors: Dict[str, Dict[str, np.ndarray]] = defaultdict(dict)

        for token in doc:
            if token.is_alpha and not token.is_stop and len(token.text) > 2:
                word = token.lemma_.lower()
                pos = token.pos_

                # Skip if word has no vector
                if not token.has_vector:
                    continue

                pos_words[pos][word] += 1

                # Store vector (use first occurrence's vector)
                if word not in word_vectors[pos]:
                    word_vectors[pos][word] = token.vector

        # Build vocabulary dictionary with vectors and frequencies
        # Include ALL words with vectors (min_sample_frequency is now 1 by default)
        for pos, words in pos_words.items():
            for word, freq in words.items():
                if freq >= self.min_sample_frequency and word in word_vectors[pos]:
                    self.sample_vocabulary[pos][word] = (word_vectors[pos][word], freq)

    def _cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Calculate cosine similarity between two vectors."""
        if np.all(vec1 == 0) or np.all(vec2 == 0):
            return 0.0

        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return dot_product / (norm1 * norm2)

    def _get_word_vector(self, word: str) -> Optional[np.ndarray]:
        """Get word vector from spaCy model."""
        token = self.nlp.vocab[word]
        if token.has_vector:
            return token.vector
        return None

    def _build_sample_to_sample_mappings(self):
        """Build mappings from sample words to other similar sample words (for style consistency)."""
        print("  [SemanticWordMapper] Building sample-to-sample mappings...")
        sample_to_sample_count = 0

        for pos, sample_words in self.sample_vocabulary.items():
            sample_word_list = list(sample_words.items())

            # For each sample word, find similar words in the same sample vocabulary
            for word1, (vec1, freq1) in sample_word_list:
                best_match = None
                best_similarity = -1.0

                for word2, (vec2, freq2) in sample_word_list:
                    # Skip if it's the same word
                    if word1 == word2:
                        continue

                    similarity = self._cosine_similarity(vec1, vec2)
                    # Use slightly higher threshold for sample-to-sample (0.35) to avoid too many mappings
                    if similarity > best_similarity and similarity >= max(0.35, self.similarity_threshold):
                        best_similarity = similarity
                        best_match = word2

                if best_match:
                    # Use a special key format to distinguish sample-to-sample mappings
                    # Store as: sample_word -> similar_sample_word
                    if word1 not in self.mappings[pos]:
                        self.mappings[pos][word1] = best_match
                        sample_to_sample_count += 1

        if sample_to_sample_count > 0:
            print(f"  [SemanticWordMapper] Built {sample_to_sample_count} sample-to-sample mappings")

    def _build_mappings(self):
        """Build comprehensive mappings from common words and spaCy vocabulary to sample vocabulary."""
        # Check cache first if enabled
        if self.cache_mappings:
            cached_mappings = self._load_mappings_from_db(self.sample_hash)
            if cached_mappings:
                self.mappings = cached_mappings
                total_cached = sum(len(m) for m in cached_mappings.values())
                print(f"  [SemanticWordMapper] Loaded {total_cached} mappings from cache")
                return

        # Cache miss - build mappings from scratch
        # Phase 1: Sample-to-sample mappings (for style consistency)
        self._build_sample_to_sample_mappings()

        # Phase 2: Expanded common words to sample vocabulary
        print("  [SemanticWordMapper] Building expanded common word mappings...")
        common_word_count = 0

        # Use expanded common words (from NLTK + spaCy)
        for pos, common_words in self.expanded_common_words.items():
            if pos not in self.sample_vocabulary:
                continue

            sample_words = self.sample_vocabulary[pos]

            for common_word in common_words:
                # Skip if already mapped (from sample-to-sample)
                if pos in self.mappings and common_word in self.mappings[pos]:
                    continue

                common_vec = self._get_word_vector(common_word)
                if common_vec is None:
                    continue

                # Find best match in sample vocabulary
                best_match = None
                best_similarity = -1.0

                for sample_word, (sample_vec, freq) in sample_words.items():
                    # Skip if it's the same word
                    if common_word == sample_word:
                        continue

                    similarity = self._cosine_similarity(common_vec, sample_vec)
                    if similarity > best_similarity and similarity >= self.similarity_threshold:
                        best_similarity = similarity
                        best_match = sample_word

                if best_match:
                    self.mappings[pos][common_word] = best_match
                    # Save to database if caching enabled
                    if self.cache_mappings:
                        self._save_mapping_to_db(common_word, best_match, pos, best_similarity, self.sample_hash)
                    common_word_count += 1

        if common_word_count > 0:
            print(f"  [SemanticWordMapper] Built {common_word_count} common word mappings")

        # Phase 3: Build mappings from spaCy vocabulary (all words with vectors)
        self._build_spacy_vocabulary_mappings()

        # Save sample-to-sample mappings to cache
        if self.cache_mappings:
            for pos, word_mappings in self.mappings.items():
                for source_word, target_word in word_mappings.items():
                    # Get similarity score (we'll use a default high score for sample-to-sample)
                    # In a more sophisticated version, we'd store the actual similarity
                    if source_word in self.sample_vocabulary.get(pos, {}):
                        self._save_mapping_to_db(source_word, target_word, pos, 0.8, self.sample_hash)

    def _build_spacy_vocabulary_mappings(self):
        """Build mappings from all words in spaCy vocabulary (with vectors) to sample vocabulary."""
        print("  [SemanticWordMapper] Building comprehensive spaCy vocabulary mappings...")
        spacy_count = 0

        # Get all words with vectors from spaCy
        # Limit to most common words to avoid excessive memory usage
        # We'll process words that are likely to appear in text
        max_words_to_process = self.max_spacy_words

        words_processed = 0
        for word_string in self.nlp.vocab.strings:
            if words_processed >= max_words_to_process:
                break

            token = self.nlp.vocab[word_string]

            # Only process words that have vectors and are alphabetic
            if not token.has_vector or not word_string.isalpha() or len(word_string) < 3:
                continue

            # Skip stop words and very common words (already in COMMON_WORDS)
            if token.is_stop or word_string.lower() in self.ALL_COMMON_WORDS:
                continue

            # Get POS tag
            # We need to parse the word to get its POS
            doc = self.nlp(word_string)
            if not doc:
                continue

            token_doc = doc[0]
            pos = token_doc.pos_

            # Skip if no sample vocabulary for this POS
            if pos not in self.sample_vocabulary:
                continue

            # Skip if already mapped
            word_lower = word_string.lower()
            if pos in self.mappings and word_lower in self.mappings[pos]:
                continue

            # Find best match in sample vocabulary
            word_vec = token.vector
            best_match = None
            best_similarity = -1.0

            for sample_word, (sample_vec, freq) in self.sample_vocabulary[pos].items():
                if word_lower == sample_word:
                    continue

                similarity = self._cosine_similarity(word_vec, sample_vec)
                if similarity > best_similarity and similarity >= self.similarity_threshold:
                    best_similarity = similarity
                    best_match = sample_word

            if best_match:
                self.mappings[pos][word_lower] = best_match
                # Save to database if caching enabled
                if self.cache_mappings:
                    self._save_mapping_to_db(word_lower, best_match, pos, best_similarity, self.sample_hash)
                spacy_count += 1

            words_processed += 1

        if spacy_count > 0:
            print(f"  [SemanticWordMapper] Built {spacy_count} spaCy vocabulary mappings")

    def map_word(self, word: str, pos: str) -> Optional[str]:
        """
        Get sample equivalent for a word.

        Args:
            word: Word to map (should be lemmatized, lowercase)
            pos: Part of speech tag

        Returns:
            Sample equivalent word or None if no mapping found
        """
        word_lower = word.lower()
        if pos in self.mappings and word_lower in self.mappings[pos]:
            return self.mappings[pos][word_lower]
        return None

    def apply_mapping(self, text: str) -> str:
        """
        Apply semantic mappings to text.

        Args:
            text: Text to process

        Returns:
            Text with common words replaced by sample equivalents
        """
        if not self.mappings:
            return text

        doc = self.nlp(text)
        result_tokens = []

        for token in doc:
            word = token.lemma_.lower()
            pos = token.pos_

            # Check if this word should be mapped
            mapped_word = self.map_word(word, pos)

            if mapped_word:
                # Preserve original case
                if token.text.isupper():
                    replacement = mapped_word.upper()
                elif token.text[0].isupper():
                    replacement = mapped_word.capitalize()
                else:
                    replacement = mapped_word.lower()

                # Preserve original token shape (punctuation, spacing)
                if token.whitespace_:
                    result_tokens.append(replacement + token.whitespace_)
                else:
                    result_tokens.append(replacement)
            else:
                # Keep original token
                result_tokens.append(token.text_with_ws)

        return ''.join(result_tokens)

    def get_mapping_stats(self) -> Dict[str, int]:
        """Get statistics about built mappings."""
        return {
            'total_mappings': sum(len(m) for m in self.mappings.values()),
            'adj_mappings': len(self.mappings.get('ADJ', {})),
            'verb_mappings': len(self.mappings.get('VERB', {})),
            'noun_mappings': len(self.mappings.get('NOUN', {})),
            'sample_vocab_size': sum(len(v) for v in self.sample_vocabulary.values())
        }


# Test function
if __name__ == '__main__':
    from pathlib import Path

    sample_path = Path(__file__).parent / "prompts" / "sample_mao.txt"
    if sample_path.exists():
        with open(sample_path, 'r', encoding='utf-8') as f:
            sample_text = f.read()

        print("=== Semantic Word Mapper Test ===\n")

        mapper = SemanticWordMapper(sample_text, similarity_threshold=0.5)

        print("\nMapping statistics:")
        stats = mapper.get_mapping_stats()
        for key, value in stats.items():
            print(f"  {key}: {value}")

        print("\nSample mappings (first 20):")
        count = 0
        for pos, mappings in mapper.mappings.items():
            for common, sample in list(mappings.items())[:10]:
                print(f"  {pos}: {common} -> {sample}")
                count += 1
                if count >= 20:
                    break
            if count >= 20:
                break

        # Test text
        test_text = """
        This is an important method that shows significant results.
        The system uses various approaches to demonstrate effective solutions.
        Many people find this process very useful for their work.
        """

        print("\n\nOriginal text:")
        print(test_text)

        print("\nMapped text:")
        mapped = mapper.apply_mapping(test_text)
        print(mapped)
    else:
        print("No sample file found.")

