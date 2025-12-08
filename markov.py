import spacy
import sqlite3
import numpy as np
import json
import random
import os
import re
import requests
import pickle
from pathlib import Path
from collections import Counter
from typing import List, Dict, Any, Optional
from glm import GLMProvider
from deepseek import DeepSeekProvider
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    print("Warning: sentence-transformers not available. Semantic matching will be disabled.")

# Configuration
DB_PATH = "style_brain.db"
SAMPLE_FILE = "prompts/sample.txt"

# Common stopwords to filter out
STOPWORDS = {
    'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by',
    'from', 'as', 'is', 'was', 'are', 'were', 'been', 'be', 'have', 'has', 'had', 'do', 'does',
    'did', 'will', 'would', 'could', 'should', 'may', 'might', 'must', 'can', 'this', 'that',
    'these', 'those', 'it', 'its', 'they', 'them', 'their', 'there', 'then', 'than', 'when',
    'where', 'what', 'which', 'who', 'whom', 'whose', 'why', 'how', 'all', 'each', 'every',
    'some', 'any', 'no', 'not', 'only', 'just', 'also', 'more', 'most', 'very', 'too', 'so',
    'such', 'both', 'either', 'neither', 'one', 'two', 'first', 'second', 'last', 'next', 'other',
    'another', 'many', 'much', 'few', 'little', 'own', 'same', 'different', 'new', 'old', 'good',
    'bad', 'great', 'small', 'large', 'long', 'short', 'high', 'low', 'big', 'little'
}

class EmbeddingSemanticMatcher:
    """Semantic matching using sentence embeddings for template selection."""

    def __init__(self, model_name='all-MiniLM-L6-v2'):
        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            self.model = None
            self.cache = {}
            return

        try:
            self.model = SentenceTransformer(model_name)
            self.cache = {}  # Cache embeddings for repeated texts
        except Exception as e:
            print(f"Warning: Could not load sentence transformer model: {e}")
            self.model = None
            self.cache = {}

    def embed_text(self, text):
        """Get embedding for text, using cache."""
        if self.model is None:
            return None

        if text in self.cache:
            return self.cache[text]

        try:
            embedding = self.model.encode(text, convert_to_tensor=False)
            self.cache[text] = embedding
            return embedding
        except Exception as e:
            print(f"Warning: Could not embed text: {e}")
            return None

    def similarity_score(self, text1, text2):
        """Calculate cosine similarity between embeddings."""
        if self.model is None:
            return 0.0

        emb1 = self.embed_text(text1)
        emb2 = self.embed_text(text2)

        if emb1 is None or emb2 is None:
            return 0.0

        # Cosine similarity
        norm1 = np.linalg.norm(emb1)
        norm2 = np.linalg.norm(emb2)
        if norm1 == 0 or norm2 == 0:
            return 0.0
        return np.dot(emb1, emb2) / (norm1 * norm2)

    def find_similar_paragraphs(self, input_text, sample_paragraphs, top_k=3):
        """
        Find top K most semantically similar paragraphs from sample.

        Args:
            input_text: Input paragraph text
            sample_paragraphs: List of dicts with 'text' and 'template' keys
            top_k: Number of top matches to return

        Returns:
            List of (paragraph_dict, similarity_score) tuples, sorted by score
        """
        if self.model is None or not sample_paragraphs:
            return []

        input_emb = self.embed_text(input_text)
        if input_emb is None:
            return []

        similarities = []
        for para_dict in sample_paragraphs:
            para_text = para_dict.get('text', '')
            if not para_text:
                continue

            para_emb = self.embed_text(para_text)
            if para_emb is None:
                continue

            score = self.similarity_score(input_text, para_text)
            similarities.append((para_dict, score))

        # Sort by similarity score (descending)
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_k]

class LearnedBurstinessAnalyzer:
    """Burstiness analyzer that learns distribution from sample text."""

    def __init__(self):
        self.length_bins = {'short': (1, 10), 'medium': (11, 25), 'long': (26, 1000)}
        self.sample_distribution = None

    def learn_from_sample(self, sample_text):
        """Learn sentence length distribution from sample text."""
        sentences = self._extract_sentences(sample_text)
        lengths = [len(s.split()) for s in sentences]

        total = len(lengths)
        if total == 0:
            self.sample_distribution = {'short': 0.2, 'medium': 0.5, 'long': 0.3}
            return

        # Count sentences in each bin
        distribution = {}
        for bin_name, (low, high) in self.length_bins.items():
            count = sum(1 for l in lengths if low <= l <= high)
            distribution[bin_name] = count / total

        self.sample_distribution = distribution
        print(f"Learned burstiness distribution: {distribution}")

    def _extract_sentences(self, text):
        """Extract sentences from text."""
        # Simple sentence splitting
        sentences = re.split(r'[.!?]+', text)
        return [s.strip() for s in sentences if s.strip()]

    def get_target_distribution(self):
        """Return learned distribution."""
        if self.sample_distribution is None:
            return {'short': 0.2, 'medium': 0.5, 'long': 0.3}  # Default
        return self.sample_distribution

    def recommend_adjustments(self, text):
        """Recommend adjustments to match learned distribution."""
        sentences = self._extract_sentences(text)
        if not sentences:
            return []

        lengths = [len(s.split()) for s in sentences]
        total = len(lengths)

        current_dist = {}
        for bin_name, (low, high) in self.length_bins.items():
            count = sum(1 for l in lengths if low <= l <= high)
            current_dist[bin_name] = count / total if total > 0 else 0

        target_dist = self.get_target_distribution()
        adjustments = []

        for bin_name in ['short', 'medium', 'long']:
            diff = target_dist[bin_name] - current_dist[bin_name]
            if abs(diff) > 0.1:  # Significant difference
                if diff > 0:
                    adjustments.append(f"Add more {bin_name} sentences")
                else:
                    adjustments.append(f"Reduce {bin_name} sentences")

        return adjustments

class PunctuationStyleLearner:
    """Learns punctuation patterns from sample text."""

    def __init__(self):
        self.punctuation_patterns = None

    def learn_from_sample(self, sample_text):
        """Learn punctuation frequency patterns from sample text."""
        paragraphs = [p.strip() for p in sample_text.split('\n\n') if p.strip()]

        total_sentences = 0
        punct_counts = {
            'em_dash': 0,
            'semicolon': 0,
            'colon': 0,
            'parentheses': 0
        }

        for para in paragraphs:
            # Count sentences
            sentences = re.split(r'[.!?]+', para)
            sentences = [s.strip() for s in sentences if s.strip()]
            total_sentences += len(sentences)

            # Count punctuation
            punct_counts['em_dash'] += len(re.findall(r'—|--', para))
            punct_counts['semicolon'] += para.count(';')
            punct_counts['colon'] += para.count(':')
            punct_counts['parentheses'] += para.count('(') + para.count(')')

        # Calculate frequency per sentence
        if total_sentences > 0:
            self.punctuation_patterns = {
                'em_dash_per_sentence': punct_counts['em_dash'] / total_sentences,
                'semicolon_per_sentence': punct_counts['semicolon'] / total_sentences,
                'colon_per_sentence': punct_counts['colon'] / total_sentences,
                'parentheses_per_sentence': punct_counts['parentheses'] / total_sentences
            }
        else:
            self.punctuation_patterns = {
                'em_dash_per_sentence': 0.1,
                'semicolon_per_sentence': 0.15,
                'colon_per_sentence': 0.1,
                'parentheses_per_sentence': 0.2
            }

        print(f"Learned punctuation patterns: {self.punctuation_patterns}")

    def get_target_patterns(self):
        """Return learned punctuation patterns."""
        if self.punctuation_patterns is None:
            return {
                'em_dash_per_sentence': 0.1,
                'semicolon_per_sentence': 0.15,
                'colon_per_sentence': 0.1,
                'parentheses_per_sentence': 0.2
            }
        return self.punctuation_patterns

    def apply_punctuation_style(self, text, target_patterns=None):
        """
        Apply punctuation style to text (guidance for LLM, not direct modification).
        Returns guidance string for LLM.
        """
        if target_patterns is None:
            target_patterns = self.get_target_patterns()

        guidance = []
        # DO NOT include em-dash guidance - em-dashes are forbidden in output
        if target_patterns.get('semicolon_per_sentence', 0) > 0.1:
            guidance.append(f"Use semicolons (;) approximately {target_patterns['semicolon_per_sentence']:.2f} times per sentence to connect related clauses")
        if target_patterns.get('colon_per_sentence', 0) > 0.1:
            guidance.append(f"Use colons (:) approximately {target_patterns['colon_per_sentence']:.2f} times per sentence for lists and explanations")

        return "; ".join(guidance) if guidance else ""

class ContextualVocabularyInjector:
    """Inject vocabulary using contextual embeddings for semantic matching."""

    def __init__(self, nlp):
        self.nlp = nlp

    def find_similar_words(self, target_word, preferred_words, threshold=0.7):
        """
        Find semantically similar words from preferred vocabulary.

        Args:
            target_word: Word to find replacement for
            preferred_words: List of preferred words to choose from
            threshold: Minimum cosine similarity threshold

        Returns:
            List of (word, similarity_score) tuples, sorted by similarity
        """
        if not preferred_words or not target_word:
            return []

        try:
            target_vector = self.nlp(target_word.lower()).vector
            if np.linalg.norm(target_vector) == 0:
                return []

            similarities = []
            for pref_word in preferred_words:
                try:
                    pref_vector = self.nlp(pref_word.lower()).vector
                    if np.linalg.norm(pref_vector) == 0:
                        continue

                    # Cosine similarity
                    similarity = np.dot(target_vector, pref_vector) / (np.linalg.norm(target_vector) * np.linalg.norm(pref_vector))
                    if similarity >= threshold:
                        similarities.append((pref_word, similarity))
                except:
                    continue

            # Sort by similarity (descending)
            similarities.sort(key=lambda x: x[1], reverse=True)
            return similarities
        except:
            return []

    def inject_vocabulary(self, text, preferred_words_dict, protected_terms=None):
        """
        Inject preferred vocabulary into text using contextual matching.
        This is a guidance method - returns suggestions for LLM, not direct substitution.

        Args:
            text: Input text
            preferred_words_dict: Dictionary with POS categories and word lists
            protected_terms: Set of terms that should not be replaced

        Returns:
            Dictionary with suggestions for each POS category
        """
        if protected_terms is None:
            protected_terms = set()

        suggestions = {}

        try:
            doc = self.nlp(text.lower())
            for token in doc:
                if token.text in protected_terms or token.is_punct or len(token.text) < 3:
                    continue

                word = token.lemma_.lower()
                pos_category = None

                if token.pos_ == "VERB":
                    pos_category = "verbs"
                elif token.pos_ == "NOUN":
                    pos_category = "nouns"
                elif token.pos_ == "ADJ":
                    pos_category = "adjectives"
                elif token.pos_ == "ADV":
                    pos_category = "adverbs"

                if pos_category and pos_category in preferred_words_dict:
                    preferred = preferred_words_dict[pos_category]
                    similar = self.find_similar_words(word, preferred, threshold=0.7)
                    if similar:
                        suggestions[word] = similar[0][0]  # Best match

        except Exception as e:
            print(f"Warning: Error in vocabulary injection: {e}")

        return suggestions

class VocabularyExtractor:
    """Extracts and filters vocabulary from sample text."""

    def __init__(self, nlp):
        self.nlp = nlp

    def extract_vocabulary(self, sample_text, min_frequency=2):
        """
        Extract non-technical words from sample text, grouped by POS category.

        Args:
            sample_text: The sample text to analyze
            min_frequency: Minimum word frequency to include (default: 2)

        Returns:
            Dictionary with POS categories as keys and word frequencies as values
        """
        doc = self.nlp(sample_text.lower())

        vocab = {
            "verbs": Counter(),
            "nouns": Counter(),
            "adjectives": Counter(),
            "adverbs": Counter()
        }

        for token in doc:
            # Skip stopwords, punctuation, and very short words
            if token.text in STOPWORDS or token.is_punct or len(token.text) < 3:
                continue

            # Skip technical terms (proper nouns, capitalized words in original)
            if token.pos_ == "PROPN":
                continue

            # Skip numbers
            if token.like_num:
                continue

            # Categorize by POS
            word = token.lemma_.lower()  # Use lemma (root form)

            if token.pos_ == "VERB":
                vocab["verbs"][word] += 1
            elif token.pos_ == "NOUN":
                vocab["nouns"][word] += 1
            elif token.pos_ == "ADJ":
                vocab["adjectives"][word] += 1
            elif token.pos_ == "ADV":
                vocab["adverbs"][word] += 1

        # Filter by minimum frequency and convert to dict
        result = {}
        for pos_category, counter in vocab.items():
            filtered = {word: count for word, count in counter.items() if count >= min_frequency}
            result[pos_category] = dict(sorted(filtered.items(), key=lambda x: x[1], reverse=True))

        return result

class TextAnalyzer:
    def __init__(self):
        print("Loading Spacy model...")
        self.nlp = spacy.load("en_core_web_sm")

    def analyze_sentence(self, sent) -> Optional[Dict[str, Any]]:
        """
        Level 1 Analysis: Tracks Voice, Complexity, and POS Ratios.
        """
        total_tokens = len(sent)
        if total_tokens == 0: return None

        # 1. Voice Detection (Passive vs Active)
        is_passive = False
        for token in sent:
            if token.dep_ == "auxpass":
                is_passive = True
                break

        # 2. Complexity Score (Clause Depth)
        # Count subordinate clauses
        clause_count = sum(1 for token in sent if token.dep_ in ["advcl", "ccomp", "xcomp", "relcl"])

        # 3. POS Ratios
        pos_counts = Counter(token.pos_ for token in sent)

        metrics = {
            "length": total_tokens,
            "is_passive": is_passive,
            "clause_count": clause_count,
            "verb_ratio": pos_counts.get("VERB", 0) / total_tokens,
            "noun_ratio": pos_counts.get("NOUN", 0) / total_tokens,
            "adj_ratio": pos_counts.get("ADJ", 0) / total_tokens,
        }

        # 4. Sentence Classification
        if total_tokens < 10: s_type = "Short"
        elif total_tokens < 25: s_type = "Medium"
        else: s_type = "Long"

        complexity = "Complex" if clause_count > 1 else "Simple"
        voice = "Pass" if is_passive else "Act"

        # Signature: "Medium_Complex_Act"
        metrics["type_signature"] = f"{s_type}_{complexity}_{voice}"

        return metrics

    def analyze_paragraph(self, text) -> Optional[Dict[str, Any]]:
        """
        Level 2 & 3 Analysis: Tracks paragraph flow and structure.
        """
        doc = self.nlp(text)
        sentences = list(doc.sents)
        if not sentences: return None

        sent_metrics = []
        for s in sentences:
            m = self.analyze_sentence(s)
            if m: sent_metrics.append(m)

        if not sent_metrics: return None

        # Paragraph Level Classification
        num_sents = len(sentences)
        if num_sents <= 2: p_len = "Brief"
        elif num_sents <= 5: p_len = "Standard"
        else: p_len = "Detailed"

        # Structure Flow Signature (First 4 sentences)
        # e.g., "Brief_Short_Simple_Act-Long_Complex_Act"
        flow_sig = "-".join([m['type_signature'] for m in sent_metrics[:4]])
        full_signature = f"{p_len}_{flow_sig}"

        return {
            "signature": full_signature,
            "sentences": sent_metrics,
            "raw_text": text
        }

    def analyze_sample_patterns(self, sample_text):
        """
        Extract structural patterns from sample text for validation.

        Returns:
            Dictionary with structural patterns:
            - sentence_lengths_by_position: Average sentence lengths by position in paragraph
            - punctuation_frequency: Frequency of em-dashes, semicolons, colons per paragraph
            - sentence_openers: Most common sentence starter words and their frequency
            - clause_structure: Patterns of where subordinate clauses appear
        """
        doc = self.nlp(sample_text)
        paragraphs = [p.strip() for p in sample_text.split('\n\n') if p.strip()]

        patterns = {
            "sentence_lengths_by_position": [],
            "punctuation_frequency": {"em_dash": 0, "semicolon": 0, "colon": 0, "paragraphs": 0},
            "sentence_openers": Counter(),
            "clause_structure": {"beginning": 0, "middle": 0, "end": 0, "total": 0}
        }

        for para in paragraphs:
            para_doc = self.nlp(para)
            sentences = list(para_doc.sents)

            if not sentences:
                continue

            patterns["punctuation_frequency"]["paragraphs"] += 1
            patterns["punctuation_frequency"]["em_dash"] += len(re.findall(r'—|--', para))
            patterns["punctuation_frequency"]["semicolon"] += para.count(';')
            patterns["punctuation_frequency"]["colon"] += para.count(':')

            # Analyze sentence lengths by position
            for i, sent in enumerate(sentences):
                length = len(sent)
                if i >= len(patterns["sentence_lengths_by_position"]):
                    patterns["sentence_lengths_by_position"].append([])
                patterns["sentence_lengths_by_position"][i].append(length)

                # Get sentence opener (first word, normalized)
                words = [t.text for t in sent if not t.is_punct]
                if words:
                    opener = words[0].lower().rstrip('.,;:')
                    if opener not in STOPWORDS and len(opener) > 2:
                        patterns["sentence_openers"][opener] += 1

                # Analyze clause structure
                tokens = list(sent)
                for j, token in enumerate(tokens):
                    if token.dep_ in ["advcl", "ccomp", "xcomp", "relcl"]:
                        patterns["clause_structure"]["total"] += 1
                        pos_ratio = j / len(tokens) if len(tokens) > 0 else 0
                        if pos_ratio < 0.33:
                            patterns["clause_structure"]["beginning"] += 1
                        elif pos_ratio < 0.67:
                            patterns["clause_structure"]["middle"] += 1
                        else:
                            patterns["clause_structure"]["end"] += 1

        # Calculate averages
        for i, lengths in enumerate(patterns["sentence_lengths_by_position"]):
            if lengths:
                patterns["sentence_lengths_by_position"][i] = sum(lengths) / len(lengths)

        # Normalize punctuation frequency per paragraph
        para_count = patterns["punctuation_frequency"]["paragraphs"]
        if para_count > 0:
            for key in ["em_dash", "semicolon", "colon"]:
                patterns["punctuation_frequency"][key] = patterns["punctuation_frequency"][key] / para_count

        return patterns

    def extract_structural_patterns(self, sample_text):
        """
        Extract high-level structural patterns from sample text.

        Returns:
            Dictionary with structural patterns:
            - section_headers: Format, style, frequency
            - citation_patterns: Footnote placement, frequency, formatting
            - quotation_usage: Frequency, formatting style
            - rhetorical_devices: Questions, examples, lists
            - paragraph_organization: Topic sentence position, argument flow
        """
        paragraphs = [p.strip() for p in sample_text.split('\n\n') if p.strip()]

        patterns = {
            'section_headers': {
                'numbered': 0,
                'unnumbered': 0,
                'format_styles': Counter(),
                'frequency': 0
            },
            'citation_patterns': {
                'footnote_placement': {'beginning': 0, 'middle': 0, 'end': 0},
                'frequency_per_paragraph': [],
                'formatting_style': 'footnote'  # or 'inline', 'parenthetical'
            },
            'quotation_usage': {
                'frequency': 0,
                'block_quotes': 0,
                'inline_quotes': 0
            },
            'rhetorical_devices': {
                'questions': 0,
                'examples': 0,
                'lists': {'semicolon': 0, 'bulleted': 0, 'numbered': 0}
            },
            'paragraph_organization': {
                'topic_sentence_position': {'first': 0, 'middle': 0, 'last': 0},
                'average_sentences_per_paragraph': 0
            }
        }

        total_paras = len(paragraphs)
        if total_paras == 0:
            return patterns

        for para in paragraphs:
            # Check for section headers (lines starting with # or numbers)
            if para.startswith('#') or re.match(r'^\d+[\.\)]\s', para):
                patterns['section_headers']['frequency'] += 1
                if re.match(r'^\d+', para):
                    patterns['section_headers']['numbered'] += 1
                else:
                    patterns['section_headers']['unnumbered'] += 1
                # Extract format style
                if para.startswith('#'):
                    level = len(para) - len(para.lstrip('#'))
                    patterns['section_headers']['format_styles'][f'h{level}'] += 1

            # Check for citations
            citations = re.findall(r'\[\^\d+\]', para)
            if citations:
                patterns['citation_patterns']['frequency_per_paragraph'].append(len(citations))
                # Check citation placement
                para_words = para.split()
                for i, word in enumerate(para_words):
                    if '[' in word and '^' in word:
                        pos_ratio = i / len(para_words) if para_words else 0
                        if pos_ratio < 0.33:
                            patterns['citation_patterns']['footnote_placement']['beginning'] += 1
                        elif pos_ratio < 0.67:
                            patterns['citation_patterns']['footnote_placement']['middle'] += 1
                        else:
                            patterns['citation_patterns']['footnote_placement']['end'] += 1

            # Check for quotations
            if '>' in para and para.strip().startswith('>'):
                patterns['quotation_usage']['block_quotes'] += 1
                patterns['quotation_usage']['frequency'] += 1
            elif '"' in para or "'" in para:
                patterns['quotation_usage']['inline_quotes'] += 1
                patterns['quotation_usage']['frequency'] += 1

            # Check for rhetorical questions
            if '?' in para:
                patterns['rhetorical_devices']['questions'] += 1

            # Check for lists
            if ';' in para and para.count(';') >= 2:
                patterns['rhetorical_devices']['lists']['semicolon'] += 1
            if re.search(r'^[\*\-\+]\s', para, re.MULTILINE):
                patterns['rhetorical_devices']['lists']['bulleted'] += 1
            if re.search(r'^\d+[\.\)]\s', para, re.MULTILINE):
                patterns['rhetorical_devices']['lists']['numbered'] += 1

            # Analyze paragraph organization (topic sentence position)
            sentences = [s.strip() for s in re.split(r'[.!?]+', para) if s.strip()]
            if sentences:
                patterns['paragraph_organization']['average_sentences_per_paragraph'] += len(sentences)
                # Simple heuristic: first sentence often topic sentence
                patterns['paragraph_organization']['topic_sentence_position']['first'] += 1

        # Normalize frequencies
        if total_paras > 0:
            patterns['section_headers']['frequency'] = patterns['section_headers']['frequency'] / total_paras
            patterns['quotation_usage']['frequency'] = patterns['quotation_usage']['frequency'] / total_paras
            patterns['rhetorical_devices']['questions'] = patterns['rhetorical_devices']['questions'] / total_paras
            patterns['paragraph_organization']['average_sentences_per_paragraph'] = patterns['paragraph_organization']['average_sentences_per_paragraph'] / total_paras

            if patterns['citation_patterns']['frequency_per_paragraph']:
                avg_citations = sum(patterns['citation_patterns']['frequency_per_paragraph']) / len(patterns['citation_patterns']['frequency_per_paragraph'])
                patterns['citation_patterns']['average_per_paragraph'] = avg_citations

        return patterns

    def validate_structure_match(self, output_text, sample_patterns):
        """
        Validate if output structure matches sample text patterns.

        Returns:
            (is_valid: bool, issues: list)
        """
        issues = []

        # Analyze output text
        output_patterns = self.analyze_sample_patterns(output_text)

        # Check punctuation frequency (within 50% tolerance)
        for punct_type in ["em_dash", "semicolon", "colon"]:
            sample_freq = sample_patterns["punctuation_frequency"].get(punct_type, 0)
            output_freq = output_patterns["punctuation_frequency"].get(punct_type, 0)

            if sample_freq > 0:
                ratio = output_freq / sample_freq if sample_freq > 0 else 0
                if ratio < 0.5 or ratio > 1.5:
                    issues.append(f"{punct_type} frequency mismatch: sample={sample_freq:.2f}, output={output_freq:.2f}")

        # Check sentence opener diversity
        sample_openers = sample_patterns["sentence_openers"]
        output_openers = output_patterns["sentence_openers"]

        if sample_openers:
            # Check if most common openers match
            sample_top = set([word for word, _ in sample_openers.most_common(10)])
            output_top = set([word for word, _ in output_openers.most_common(10)])
            overlap = len(sample_top & output_top) / len(sample_top) if sample_top else 0

            if overlap < 0.3:
                issues.append(f"Sentence opener diversity mismatch: only {overlap:.1%} overlap with sample")

        # Check sentence length distribution (first 3 sentences)
        sample_lengths = sample_patterns["sentence_lengths_by_position"][:3]
        output_lengths = output_patterns["sentence_lengths_by_position"][:3]

        for i, (sample_len, output_len) in enumerate(zip(sample_lengths, output_lengths)):
            if abs(sample_len - output_len) > sample_len * 0.4:  # 40% tolerance
                issues.append(f"Sentence {i+1} length mismatch: sample={sample_len:.1f}, output={output_len:.1f}")

        return len(issues) == 0, issues

    def calculate_style_match_score(self, output_text, sample_patterns):
        """
        Calculate a style match score (0-1) comparing output to sample patterns.
        Higher score = better match.
        """
        output_patterns = self.analyze_sample_patterns(output_text)
        score = 1.0
        deductions = []

        # Check sentence length distribution (weight: 30%)
        sample_lengths = sample_patterns.get("sentence_lengths_by_position", [])
        output_lengths = output_patterns.get("sentence_lengths_by_position", [])
        if sample_lengths and output_lengths:
            length_matches = 0
            for i in range(min(len(sample_lengths), len(output_lengths), 3)):
                sample_len = sample_lengths[i] if isinstance(sample_lengths[i], (int, float)) else 0
                output_len = output_lengths[i] if isinstance(output_lengths[i], (int, float)) else 0
                if sample_len > 0:
                    ratio = min(output_len, sample_len) / max(output_len, sample_len) if max(output_len, sample_len) > 0 else 0
                    length_matches += ratio
            length_score = length_matches / min(len(sample_lengths), len(output_lengths), 3) if min(len(sample_lengths), len(output_lengths), 3) > 0 else 0
            score *= 0.3 * length_score + 0.7  # 30% weight
        else:
            score *= 0.7  # Penalty if can't compare

        # Check punctuation frequency (weight: 20%)
        sample_punct = sample_patterns.get("punctuation_frequency", {})
        output_punct = output_patterns.get("punctuation_frequency", {})
        punct_matches = 0
        punct_total = 0
        for punct_type in ["em_dash", "semicolon", "colon"]:
            sample_freq = sample_punct.get(punct_type, 0)
            output_freq = output_punct.get(punct_type, 0)
            if sample_freq > 0:
                punct_total += 1
                ratio = min(output_freq, sample_freq) / max(output_freq, sample_freq) if max(output_freq, sample_freq) > 0 else 0
                punct_matches += ratio
        punct_score = punct_matches / punct_total if punct_total > 0 else 0.5
        score *= 0.2 * punct_score + 0.8  # 20% weight

        # Check sentence opener diversity (weight: 25%)
        sample_openers = sample_patterns.get("sentence_openers", Counter())
        output_openers = output_patterns.get("sentence_openers", Counter())
        if sample_openers:
            sample_top = set([word for word, _ in sample_openers.most_common(10)])
            output_top = set([word for word, _ in output_openers.most_common(10)])
            overlap = len(sample_top & output_top) / len(sample_top) if sample_top else 0
            score *= 0.25 * overlap + 0.75  # 25% weight
        else:
            score *= 0.75  # Penalty

        # Check clause structure (weight: 25%)
        sample_clause = sample_patterns.get("clause_structure", {})
        output_clause = output_patterns.get("clause_structure", {})
        if sample_clause.get("total", 0) > 0 and output_clause.get("total", 0) > 0:
            # Compare distribution of clause positions
            sample_total = sample_clause.get("total", 1)
            output_total = output_clause.get("total", 1)
            sample_begin = sample_clause.get("beginning", 0) / sample_total
            output_begin = output_clause.get("beginning", 0) / output_total
            sample_mid = sample_clause.get("middle", 0) / sample_total
            output_mid = output_clause.get("middle", 0) / output_total
            sample_end = sample_clause.get("end", 0) / sample_total
            output_end = output_clause.get("end", 0) / output_total

            clause_score = (
                min(sample_begin, output_begin) / max(sample_begin, output_begin) if max(sample_begin, output_begin) > 0 else 0.5 +
                min(sample_mid, output_mid) / max(sample_mid, output_mid) if max(sample_mid, output_mid) > 0 else 0.5 +
                min(sample_end, output_end) / max(sample_end, output_end) if max(sample_end, output_end) > 0 else 0.5
            ) / 3
            score *= 0.25 * clause_score + 0.75  # 25% weight
        else:
            score *= 0.75  # Penalty

        return score

class StyleDatabase:
    def __init__(self, db_path=DB_PATH):
        self.conn = sqlite3.connect(db_path)
        self.cursor = self.conn.cursor()
        self.setup_db()

    def setup_db(self):
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS states (
                id INTEGER PRIMARY KEY,
                state_signature TEXT UNIQUE,
                example_template TEXT
            )
        ''')
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS transitions (
                from_state_id INTEGER,
                to_state_id INTEGER,
                count INTEGER DEFAULT 1,
                PRIMARY KEY (from_state_id, to_state_id)
            )
        ''')
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS vocabulary (
                word TEXT PRIMARY KEY,
                pos_category TEXT,
                frequency INTEGER,
                is_technical BOOLEAN DEFAULT 0
            )
        ''')
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS sample_embeddings (
                paragraph_id INTEGER PRIMARY KEY,
                embedding BLOB,
                raw_text TEXT,
                FOREIGN KEY (paragraph_id) REFERENCES states(id)
            )
        ''')
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS learned_patterns (
                pattern_name TEXT PRIMARY KEY,
                pattern_data TEXT
            )
        ''')
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS structural_patterns (
                id INTEGER PRIMARY KEY,
                pattern_type TEXT,
                pattern_data TEXT,
                frequency REAL
            )
        ''')
        self.conn.commit()

    def store_vocabulary(self, vocab_dict):
        """Store extracted vocabulary in the database."""
        self.cursor.execute('DELETE FROM vocabulary')  # Clear existing vocabulary

        for pos_category, words in vocab_dict.items():
            for word, frequency in words.items():
                self.cursor.execute('''
                    INSERT OR REPLACE INTO vocabulary (word, pos_category, frequency, is_technical)
                    VALUES (?, ?, ?, 0)
                ''', (word, pos_category, frequency))

        self.conn.commit()

    def get_preferred_words(self, pos_category, limit=20):
        """Get top N words for a POS category, ordered by frequency."""
        self.cursor.execute('''
            SELECT word, frequency FROM vocabulary
            WHERE pos_category = ? AND is_technical = 0
            ORDER BY frequency DESC
            LIMIT ?
        ''', (pos_category, limit))

        results = self.cursor.fetchall()
        return [word for word, freq in results]

    def store_sample_embedding(self, paragraph_id, embedding, raw_text):
        """Store embedding for a sample paragraph."""
        if embedding is None:
            return
        embedding_blob = pickle.dumps(embedding)
        self.cursor.execute('''
            INSERT OR REPLACE INTO sample_embeddings (paragraph_id, embedding, raw_text)
            VALUES (?, ?, ?)
        ''', (paragraph_id, embedding_blob, raw_text))
        self.conn.commit()

    def get_sample_paragraphs_with_embeddings(self):
        """Get all sample paragraphs with their embeddings."""
        self.cursor.execute('''
            SELECT s.id, s.example_template, s.state_signature, se.embedding, se.raw_text
            FROM states s
            LEFT JOIN sample_embeddings se ON s.id = se.paragraph_id
        ''')
        results = []
        for row in self.cursor.fetchall():
            para_id, template_json, signature, embedding_blob, raw_text = row
            embedding = pickle.loads(embedding_blob) if embedding_blob else None
            template = json.loads(template_json) if template_json else None
            results.append({
                'id': para_id,
                'template': template,
                'signature': signature,
                'embedding': embedding,
                'text': raw_text
            })
        return results

    def store_learned_pattern(self, pattern_name, pattern_data):
        """Store a learned pattern (burstiness, punctuation, etc.)."""
        pattern_json = json.dumps(pattern_data)
        self.cursor.execute('''
            INSERT OR REPLACE INTO learned_patterns (pattern_name, pattern_data)
            VALUES (?, ?)
        ''', (pattern_name, pattern_json))
        self.conn.commit()

    def get_learned_pattern(self, pattern_name):
        """Retrieve a learned pattern."""
        self.cursor.execute('''
            SELECT pattern_data FROM learned_patterns WHERE pattern_name = ?
        ''', (pattern_name,))
        row = self.cursor.fetchone()
        if row:
            return json.loads(row[0])
        return None

    def train(self, paragraphs_metrics):
        print(f"Training Markov Chain on {len(paragraphs_metrics)} paragraphs...")
        prev_state_id = None

        for p_data in paragraphs_metrics:
            sig = p_data['signature']
            template_json = json.dumps(p_data['sentences'])

            # Insert State
            self.cursor.execute('INSERT OR IGNORE INTO states (state_signature, example_template) VALUES (?, ?)',
                                (sig, template_json))

            # Get ID
            self.cursor.execute('SELECT id FROM states WHERE state_signature = ?', (sig,))
            curr_state_id = self.cursor.fetchone()[0]

            # Record Transition
            if prev_state_id is not None:
                self.cursor.execute('''
                    INSERT INTO transitions (from_state_id, to_state_id, count)
                    VALUES (?, ?, 1)
                    ON CONFLICT(from_state_id, to_state_id)
                    DO UPDATE SET count = count + 1
                ''', (prev_state_id, curr_state_id))

            prev_state_id = curr_state_id

        self.conn.commit()

    def predict_next_template(self, current_state_signature=None, input_text=None, semantic_matcher=None):
        """
        Enhanced template prediction with semantic matching.

        Args:
            current_state_signature: Current Markov state signature
            input_text: Input paragraph text for semantic matching (optional)
            semantic_matcher: EmbeddingSemanticMatcher instance (optional)

        Returns:
            Dictionary with template, signature, and id
        """
        # Get candidate templates from Markov chain
        candidates = []

        if not current_state_signature:
            # Get random candidates
            self.cursor.execute('SELECT id, example_template, state_signature FROM states ORDER BY RANDOM() LIMIT 5')
            for row in self.cursor.fetchall():
                candidates.append({
                    "id": row[0],
                    "template": json.loads(row[1]),
                    "signature": row[2]
                })
        else:
            # Find current ID
            self.cursor.execute('SELECT id FROM states WHERE state_signature = ?', (current_state_signature,))
            res = self.cursor.fetchone()
            if not res:
                return self.predict_next_template(None, input_text, semantic_matcher)

            current_id = res[0]

            # Get transitions (weighted candidates)
            self.cursor.execute('SELECT to_state_id, count FROM transitions WHERE from_state_id = ?', (current_id,))
            transitions = self.cursor.fetchall()

            if not transitions:
                return self.predict_next_template(None, input_text, semantic_matcher)

            # Get top candidates by transition weight
            next_ids = [t[0] for t in transitions]
            weights = [t[1] for t in transitions]

            # Select top 5 candidates based on weights
            if len(next_ids) > 5:
                top_indices = sorted(range(len(next_ids)), key=lambda i: weights[i], reverse=True)[:5]
                next_ids = [next_ids[i] for i in top_indices]

            for next_id in next_ids:
                self.cursor.execute('SELECT example_template, state_signature FROM states WHERE id = ?', (next_id,))
                row = self.cursor.fetchone()
                if row:
                    candidates.append({
                        "id": next_id,
                        "template": json.loads(row[0]),
                        "signature": row[1]
                    })

        if not candidates:
            return None

        # If semantic matching is available and input text provided, use it
        if semantic_matcher and input_text and semantic_matcher.model is not None:
            # Get sample paragraphs with embeddings
            sample_paras = self.get_sample_paragraphs_with_embeddings()

            # Filter to only candidates
            candidate_ids = {c['id'] for c in candidates}
            candidate_samples = [p for p in sample_paras if p['id'] in candidate_ids and p['embedding'] is not None]

            if candidate_samples:
                # Find semantically similar
                similar = semantic_matcher.find_similar_paragraphs(input_text, candidate_samples, top_k=1)
                if similar and len(similar) > 0:
                    best_match = similar[0][0]
                    # Find corresponding candidate
                    for c in candidates:
                        if c['id'] == best_match['id']:
                            return c

        # Fallback to weighted random selection from candidates
        if len(candidates) == 1:
            return candidates[0]

        # Use Markov weights if available
        selected = random.choice(candidates)
        return selected

class StyleTransferAgent:
    def __init__(self, config_path: str = None):
        """
        Initialize the style transfer agent with model configuration.

        Args:
            config_path: Path to config.json file (default: config.json in project root)
        """
        self.analyzer = TextAnalyzer()
        self.db = StyleDatabase()
        self.current_markov_signature = None
        self.vocab_extractor = VocabularyExtractor(self.analyzer.nlp)
        self.semantic_matcher = EmbeddingSemanticMatcher()
        self.burstiness_analyzer = LearnedBurstinessAnalyzer()
        self.punctuation_learner = PunctuationStyleLearner()
        self.vocab_injector = ContextualVocabularyInjector(self.analyzer.nlp)

        # Load configuration
        if config_path is None:
            config_path = Path(__file__).parent / "config.json"
        else:
            config_path = Path(config_path)

        with open(config_path, "r", encoding="utf-8") as f:
            self.config = json.load(f)

        self.provider = self.config.get("provider", "ollama")

        # Initialize provider-specific settings
        if self.provider == "ollama":
            ollama_config = self.config.get("ollama", {})
            self.model_name = ollama_config.get("editor_model", "qwen3:32b")
            self.ollama_url = ollama_config.get("url", "http://localhost:11434/api/generate")
            self.glm_provider = None
            self.deepseek_provider = None
        elif self.provider == "glm":
            glm_config = self.config.get("glm", {})
            self.model_name = glm_config.get("editor_model", "glm-4.6")
            api_key = glm_config.get("api_key") or os.getenv("GLM_API_KEY")
            api_url = glm_config.get("api_url", "https://api.z.ai/api/paas/v4/chat/completions")
            self.glm_provider = GLMProvider(api_key, api_url)
            self.ollama_url = None
            self.deepseek_provider = None
        elif self.provider == "deepseek":
            deepseek_config = self.config.get("deepseek", {})
            self.model_name = deepseek_config.get("editor_model", "deepseek-chat")
            api_key = deepseek_config.get("api_key") or os.getenv("DEEPSEEK_API_KEY")
            api_url = deepseek_config.get("api_url", "https://api.deepseek.com/v1/chat/completions")
            self.deepseek_provider = DeepSeekProvider(api_key, api_url)
            self.ollama_url = None
            self.glm_provider = None
        else:
            raise ValueError(f"Unknown provider: {self.provider}. Must be 'ollama', 'glm', or 'deepseek'")

        print(f"Using provider: {self.provider}")
        print(f"Model: {self.model_name}")

    def call_model(self, prompt, system_prompt=""):
        """
        Call the configured model provider (Ollama, GLM, or DeepSeek).

        Args:
            prompt: User prompt
            system_prompt: System prompt (optional)

        Returns:
            Response text from the model
        """
        if self.provider == "ollama":
            return self._call_ollama(prompt, system_prompt)
        elif self.provider == "glm":
            return self.glm_provider.call(self.model_name, prompt, system_prompt, temperature=0.6, top_p=0.85)
        elif self.provider == "deepseek":
            return self.deepseek_provider.call(self.model_name, prompt, system_prompt, temperature=0.6, top_p=0.85)
        else:
            raise ValueError(f"Unknown provider: {self.provider}")

    def _call_ollama(self, prompt, system_prompt=""):
        """Internal method to call Ollama API."""
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "system": system_prompt,
            "stream": False,
            "options": {
                "temperature": 0.6,
                "top_p": 0.85,
                "num_ctx": 4096
            }
        }
        response = requests.post(self.ollama_url, json=payload, timeout=120)
        return response.json()['response']

    def learn_style(self, file_path=None):
        """
        Learn style from a sample file.

        Args:
            file_path: Path to sample file (default: prompts/sample.txt)
        """
        if file_path is None:
            file_path = Path(__file__).parent / SAMPLE_FILE
        else:
            file_path = Path(file_path)

        if not file_path.exists():
            print(f"Error: {file_path} not found.")
            return

        print(f"Reading {file_path}...")
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()

        # Extract vocabulary from sample text
        print("Extracting vocabulary from sample text...")
        vocab_dict = self.vocab_extractor.extract_vocabulary(text)
        self.db.store_vocabulary(vocab_dict)
        print(f"Extracted vocabulary: {sum(len(words) for words in vocab_dict.values())} words")

        # Learn burstiness distribution from sample
        print("Learning burstiness distribution from sample...")
        self.burstiness_analyzer.learn_from_sample(text)
        burstiness_dist = self.burstiness_analyzer.get_target_distribution()
        self.db.store_learned_pattern('burstiness', burstiness_dist)
        print(f"Learned burstiness: {burstiness_dist}")

        # Learn punctuation patterns from sample
        print("Learning punctuation patterns from sample...")
        self.punctuation_learner.learn_from_sample(text)
        punct_patterns = self.punctuation_learner.get_target_patterns()
        self.db.store_learned_pattern('punctuation', punct_patterns)
        print(f"Learned punctuation patterns: {punct_patterns}")

        raw_paras = [p.strip() for p in text.split('\n\n') if p.strip()]
        analyzed_chain = []

        for i, p in enumerate(raw_paras):
            if i % 10 == 0: print(f"Analyzing paragraph {i}/{len(raw_paras)}...")
            metrics = self.analyzer.analyze_paragraph(p)
            if metrics:
                analyzed_chain.append(metrics)

        self.db.train(analyzed_chain)

        # Store embeddings after training (when we have paragraph IDs)
        print("Storing sample paragraph embeddings...")
        for i, (p, metrics) in enumerate(zip(raw_paras, analyzed_chain)):
            if i % 10 == 0: print(f"Storing embeddings {i}/{len(analyzed_chain)}...")
            sig = metrics['signature']
            # Get paragraph ID from database
            self.db.cursor.execute('SELECT id FROM states WHERE state_signature = ?', (sig,))
            row = self.db.cursor.fetchone()
            if row:
                para_id = row[0]
                para_embedding = self.semantic_matcher.embed_text(p)
                if para_embedding is not None:
                    self.db.store_sample_embedding(para_id, para_embedding, p)

        # Extract and store structural patterns
        print("Extracting structural patterns from sample...")
        structural_patterns = self.analyzer.extract_structural_patterns(text)
        self.db.store_learned_pattern('structural_patterns', structural_patterns)
        print(f"Extracted structural patterns: headers, citations, quotations, rhetorical devices")

    def generate_human_readable_template(self, template_data, scaling_factor=1.0):
        """Converts raw metrics into LLM instructions, scaling lengths if needed."""
        instructions = []
        instructions.append(f"TARGET STRUCTURE: This paragraph must have exactly {len(template_data)} sentences.")

        # Get learned burstiness distribution
        burstiness_dist = self.burstiness_analyzer.get_target_distribution()
        instructions.append(f"BURSTINESS TARGET: {burstiness_dist['short']:.1%} short (≤10 words), {burstiness_dist['medium']:.1%} medium (11-25 words), {burstiness_dist['long']:.1%} long (>25 words)")

        # Get punctuation style guidance
        punct_guidance = self.punctuation_learner.apply_punctuation_style("", self.punctuation_learner.get_target_patterns())
        if punct_guidance:
            instructions.append(f"PUNCTUATION STYLE: {punct_guidance}")

        for i, sent in enumerate(template_data):
            # Scale length
            target_len = int(sent['length'] * scaling_factor)
            target_len = max(5, target_len) # Minimum 5 words

            # Calculate POS ratios as percentages
            verb_pct = int(sent['verb_ratio'] * 100)
            noun_pct = int(sent['noun_ratio'] * 100)
            adj_pct = int(sent['adj_ratio'] * 100)

            style_notes = []
            if sent['is_passive']: style_notes.append("PASSIVE voice")
            else: style_notes.append("ACTIVE voice")

            if sent['clause_count'] > 1: style_notes.append("COMPLEX syntax (subordinate clauses)")
            else: style_notes.append("SIMPLE syntax")

            # Get preferred words based on POS ratios
            preferred_words = []
            if verb_pct > 10:
                verbs = self.db.get_preferred_words("verbs", limit=8)
                if verbs:
                    preferred_words.append(f"Preferred verbs: {', '.join(verbs[:5])}")
            if noun_pct > 15:
                nouns = self.db.get_preferred_words("nouns", limit=8)
                if nouns:
                    preferred_words.append(f"Preferred nouns: {', '.join(nouns[:5])}")
            if adj_pct > 5:
                adjectives = self.db.get_preferred_words("adjectives", limit=8)
                if adjectives:
                    preferred_words.append(f"Preferred adjectives: {', '.join(adjectives[:5])}")

            word_prefs = ". ".join(preferred_words) + "." if preferred_words else ""

            # Build instruction with explicit POS ratios and word preferences
            pos_info = f"POS ratios: {verb_pct}% verbs, {noun_pct}% nouns, {adj_pct}% adjectives"
            note_str = ", ".join(style_notes)
            if word_prefs:
                instructions.append(f"  - Sentence {i+1}: ~{target_len} words. {pos_info}. {note_str}. {word_prefs}")
            else:
                instructions.append(f"  - Sentence {i+1}: ~{target_len} words. {pos_info}. {note_str}.")

        return "\n".join(instructions)

    def validate_and_retry(self, prompt, target_template_len, scaling_factor, max_retries=2):
        """The Loop: Generates text and checks if structure matches."""

        current_prompt = prompt
        system_prompt = "You are a Ghostwriter. Follow the structure instructions precisely."

        for attempt in range(max_retries + 1):
            print(f"  > Generation Attempt {attempt+1}...")

            try:
                generated_text = self.call_model(current_prompt, system_prompt)

                if not generated_text or not generated_text.strip():
                    print(f"    [x] Empty response from model")
                    continue

                # Validation Logic
                metrics = self.analyzer.analyze_paragraph(generated_text)
                if not metrics:
                    print(f"    [x] Could not analyze generated text")
                    continue

                out_sentences = len(metrics['sentences'])

                # Check 1: Sentence Count (Tolerance +/- 1)
                if abs(out_sentences - target_template_len) > 1:
                    print(f"    [x] Failed: Expected {target_template_len} sentences, got {out_sentences}.")
                    # Update Prompt with Error Info
                    current_prompt = prompt + (
                        f"\n\nSYSTEM ERROR: Your previous attempt had {out_sentences} sentences. "
                        f"You MUST write exactly {target_template_len} sentences."
                    )
                    continue

                # Check 2 (Optional): Length Check to ensure scaling worked
                avg_len = metrics['sentences'][0]['length'] # Crude check on first sentence
                # You could add more detailed checks here

                return generated_text # Pass

            except requests.exceptions.Timeout:
                print(f"    [x] Timeout error")
                return None
            except Exception as e:
                print(f"    [x] Connection Error: {e}")
                return None

        return None  # Return None if all attempts failed

    def restyle_paragraph(self, input_text):
        # 1. Gauge Input Content Mass
        input_metrics = self.analyzer.analyze_paragraph(input_text)
        if not input_metrics: return input_text

        input_word_count = sum(s['length'] for s in input_metrics['sentences'])

        # 2. Get Next Template
        prediction = self.db.predict_next_template(self.current_markov_signature)
        if not prediction:
            print("Database empty or prediction failed.")
            return input_text

        self.current_markov_signature = prediction['signature']
        template_data = prediction['template']

        # 3. Adaptive Scaling
        target_word_count = sum(s['length'] for s in template_data)
        scaling_factor = 1.0

        # If Target is > 1.5x Input, we scale down to prevent hallucination
        if target_word_count > (input_word_count * 1.5):
            scaling_factor = (input_word_count * 1.2) / target_word_count
            print(f"  [i] Scaling template by {scaling_factor:.2f}x to fit content.")

        # 4. Build Prompt
        blueprint = self.generate_human_readable_template(template_data, scaling_factor)

        prompt = (
            f"You are a Ghostwriter. Rewrite the INPUT CONTENT to match the TARGET STRUCTURE.\n\n"
            f"INPUT CONTENT:\n{input_text}\n\n"
            f"{blueprint}\n\n"
            f"CRITICAL RULES:\n"
            f"1. Do not add ANY new facts. If the template is too long, combine ideas, do NOT invent them.\n"
            f"2. Strictly follow the Passive/Active voice instructions for each sentence.\n"
            f"3. Maintain the exact sentence count."
        )

        # 5. Execute with Validation
        result = self.validate_and_retry(prompt, len(template_data), scaling_factor)
        return result

if __name__ == "__main__":
    # Load config from default location
    agent = StyleTransferAgent()

    # 1. Train
    sample_path = Path(__file__).parent / SAMPLE_FILE
    if sample_path.exists():
        agent.learn_style(sample_path)
    else:
        print(f"Please create '{sample_path}' with your training text first!")
        exit()

    # 2. Loop
    print("\n" + "="*40)
    print("      STYLE MIMIC ENGINE READY")
    print("="*40)
    print("Enter a paragraph to restyle (or 'q' to quit).\n")

    while True:
        try:
            user_input = input("Input > ")
            if user_input.lower() in ['q', 'quit']: break
            if not user_input.strip(): continue

            print("Thinking...")
            result = agent.restyle_paragraph(user_input)

            if result:
                print("\n--- Restyled Output ---")
                print(result)
            else:
                print("\n--- Error: Could not restyle paragraph ---")
            print("-" * 30 + "\n")
        except KeyboardInterrupt:
            break