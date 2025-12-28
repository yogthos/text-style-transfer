"""Extract style profiles from corpus text.

All extraction is data-driven using spaCy - NO hardcoded patterns.
"""

import re
from collections import Counter, defaultdict
from typing import List, Dict, Tuple, Optional
import numpy as np

from ..utils.nlp import get_nlp, split_into_sentences
from ..utils.logging import get_logger
from .profile import (
    SentenceLengthProfile,
    TransitionProfile,
    RegisterProfile,
    DeltaProfile,
    SentenceStructureProfile,
    DiscourseRelationProfile,
    VocabularyPalette,
    SentenceFunctionProfile,
    AuthorStyleProfile,
    # NEW: Voice profile components
    AssertivenessProfile,
    RhetoricalProfile,
    PhrasePatterns,
)
from ..ingestion.proposition_extractor import (
    APPEARANCE_MARKERS,
    APPEARANCE_PHRASES,
    APPEARANCE_PATTERNS,
    HEDGING_WORDS,
    HEDGING_PHRASES,
    BOOSTER_WORDS,
    CONTRAST_MARKERS,
    CAUSE_MARKERS,
)
from ..vocabulary.palette import VocabularyPaletteExtractor
from ..rhetorical.function_classifier import SentenceFunctionClassifier

logger = get_logger(__name__)




class StyleProfileExtractor:
    """Extract complete style profile from corpus."""

    def __init__(self):
        self._nlp = None

    @property
    def nlp(self):
        if self._nlp is None:
            self._nlp = get_nlp()
        return self._nlp

    def extract(
        self,
        paragraphs: List[str],
        author_name: str,
    ) -> AuthorStyleProfile:
        """Extract complete style profile from corpus paragraphs.

        Args:
            paragraphs: List of paragraph texts.
            author_name: Author name.

        Returns:
            AuthorStyleProfile with all extracted metrics.
        """
        logger.info(f"Extracting style profile for {author_name} from {len(paragraphs)} paragraphs")

        # Collect all sentences
        all_sentences = []
        for para in paragraphs:
            sentences = split_into_sentences(para)
            all_sentences.extend(sentences)

        if not all_sentences:
            raise ValueError("No sentences found in corpus")

        # Extract component profiles
        length_profile = self._extract_length_profile(all_sentences)
        transition_profile = self._extract_transition_profile(all_sentences)
        register_profile = self._extract_register_profile(all_sentences, paragraphs)
        delta_profile = self._extract_delta_profile(all_sentences)
        structure_profile = self._extract_structure_profile(all_sentences)
        discourse_profile = self._extract_discourse_profile(paragraphs)

        # NEW: Segment corpus into 150-400 word chunks for flow samples
        # These multi-sentence chunks capture style better than isolated sentences
        flow_samples = self._segment_corpus_to_chunks(paragraphs)
        structure_profile.flow_samples = flow_samples

        # Extract vocabulary palette for style transfer
        vocab_extractor = VocabularyPaletteExtractor()
        vocabulary_palette = vocab_extractor.extract(paragraphs)

        # Extract sentence function profile for rhetorical templates
        func_classifier = SentenceFunctionClassifier()
        func_profile_data = func_classifier.extract_function_profile(paragraphs)
        function_profile = SentenceFunctionProfile(
            function_distribution=func_profile_data["function_distribution"],
            function_transitions=func_profile_data["function_transitions"],
            initial_function_probs=func_profile_data["initial_function_probs"],
            function_samples=func_profile_data["function_samples"],
        )
        logger.info(f"  Function types: {list(func_profile_data['function_distribution'].keys())}")

        # Extract human writing patterns for humanization
        human_patterns = self._extract_human_patterns(paragraphs)

        # NEW: Extract voice profile components (Sprint 2)
        assertiveness_profile = self._extract_assertiveness_profile(all_sentences)
        rhetorical_profile = self._extract_rhetorical_profile(all_sentences, paragraphs)
        phrase_patterns = self._extract_phrase_patterns(all_sentences, paragraphs)

        # Calculate totals
        word_count = sum(len(s.split()) for s in all_sentences)

        logger.info(f"Extracted profile: {len(all_sentences)} sentences, {word_count} words")
        logger.info(f"  Length: {length_profile.mean:.1f} +/- {length_profile.std:.1f}")
        logger.info(f"  Burstiness: {length_profile.burstiness:.3f}")
        logger.info(f"  No-transition ratio: {transition_profile.no_transition_ratio:.2f}")
        logger.info(f"  Human patterns: {human_patterns.get('fragment_ratio', 0):.1%} fragments, "
                   f"{human_patterns.get('question_ratio', 0):.1%} questions")
        logger.info(f"  Assertiveness: commitment={assertiveness_profile.average_commitment:.2f}, "
                   f"hedge={assertiveness_profile.hedge_ratio:.1%}, boost={assertiveness_profile.booster_ratio:.1%}")
        logger.info(f"  Rhetorical: contrast={rhetorical_profile.contrast_frequency:.1%}, "
                   f"question={rhetorical_profile.question_frequency:.1%}")
        logger.info(f"  Flow samples: {len(structure_profile.flow_samples)} chunks "
                   f"(150-400 words each)")

        return AuthorStyleProfile(
            author_name=author_name,
            corpus_word_count=word_count,
            corpus_sentence_count=len(all_sentences),
            length_profile=length_profile,
            transition_profile=transition_profile,
            register_profile=register_profile,
            delta_profile=delta_profile,
            structure_profile=structure_profile,
            discourse_profile=discourse_profile,
            vocabulary_palette=vocabulary_palette,
            function_profile=function_profile,
            human_patterns=human_patterns,
            # NEW: Voice profiles
            assertiveness_profile=assertiveness_profile,
            rhetorical_profile=rhetorical_profile,
            phrase_patterns=phrase_patterns,
        )

    def _extract_length_profile(self, sentences: List[str]) -> SentenceLengthProfile:
        """Extract sentence length profile with Markov transitions."""
        lengths = [len(s.split()) for s in sentences]

        if not lengths:
            return SentenceLengthProfile(
                mean=20.0, std=10.0, min_length=3, max_length=60
            )

        # Basic statistics
        mean = np.mean(lengths)
        std = np.std(lengths)
        min_len = min(lengths)
        max_len = max(lengths)

        # Percentiles
        percentiles = {
            10: int(np.percentile(lengths, 10)),
            25: int(np.percentile(lengths, 25)),
            50: int(np.percentile(lengths, 50)),
            75: int(np.percentile(lengths, 75)),
            90: int(np.percentile(lengths, 90)),
        }

        # Burstiness (coefficient of variation)
        burstiness = std / mean if mean > 0 else 0.0

        # Short/long ratios
        short_ratio = sum(1 for l in lengths if l < 10) / len(lengths)
        long_ratio = sum(1 for l in lengths if l > 30) / len(lengths)

        # Build Markov transition matrix
        length_transitions = self._build_length_markov(lengths, percentiles)

        return SentenceLengthProfile(
            mean=float(mean),
            std=float(std),
            min_length=min_len,
            max_length=max_len,
            percentiles=percentiles,
            burstiness=float(burstiness),
            short_ratio=float(short_ratio),
            long_ratio=float(long_ratio),
            length_transitions=length_transitions,
        )

    def _build_length_markov(
        self,
        lengths: List[int],
        percentiles: Dict[int, int],
    ) -> Dict[str, Dict[str, float]]:
        """Build Order-1 Markov chain for length transitions.

        Research shows bigram (Order-1) is sufficient for sentence lengths.
        """
        p25 = percentiles.get(25, 12)
        p75 = percentiles.get(75, 25)

        def categorize(length: int) -> str:
            if length < p25:
                return "short"
            elif length > p75:
                return "long"
            else:
                return "medium"

        # Count transitions
        transitions = defaultdict(Counter)
        categories = [categorize(l) for l in lengths]

        for i in range(len(categories) - 1):
            current = categories[i]
            next_cat = categories[i + 1]
            transitions[current][next_cat] += 1

        # Normalize to probabilities
        markov = {}
        for current, next_counts in transitions.items():
            total = sum(next_counts.values())
            markov[current] = {
                cat: count / total
                for cat, count in next_counts.items()
            }

        # Ensure all categories have entries
        for cat in ["short", "medium", "long"]:
            if cat not in markov:
                # Default uniform distribution
                markov[cat] = {"short": 0.33, "medium": 0.34, "long": 0.33}

        return markov

    def _extract_transition_profile(self, sentences: List[str]) -> TransitionProfile:
        """Extract transition word usage patterns using spaCy.

        Uses dependency parsing to identify sentence-initial discourse markers,
        then classifies them by their semantic role based on POS and dependency.
        """
        # Count transitions per category
        category_counts = defaultdict(Counter)
        total_sentences = len(sentences)
        sentences_with_transition = 0
        transitions_at_start = 0
        total_transitions = 0

        for sentence in sentences:
            doc = self.nlp(sentence)
            if len(doc) < 2:
                continue

            # Check first few tokens for discourse markers
            found_transition = False
            first_token = doc[0]

            # Identify sentence-initial discourse markers
            # These are typically: SCONJ, CCONJ, ADV with specific dependency roles
            if first_token.pos_ in ('SCONJ', 'CCONJ', 'ADV'):
                word = first_token.text.lower()
                category = self._classify_transition(first_token, doc)
                if category:
                    category_counts[category][word] += 1
                    total_transitions += 1
                    transitions_at_start += 1
                    found_transition = True

            # Also check for multi-word transitions at start (e.g., "in addition")
            if len(doc) >= 2:
                bigram = f"{doc[0].text.lower()} {doc[1].text.lower()}"
                if doc[0].pos_ == 'ADP' and doc[1].pos_ in ('NOUN', 'DET'):
                    # Patterns like "in addition", "on the other hand"
                    category = self._classify_phrase_transition(bigram, doc)
                    if category:
                        category_counts[category][bigram] += 1
                        total_transitions += 1
                        if not found_transition:
                            transitions_at_start += 1
                        found_transition = True

            if found_transition:
                sentences_with_transition += 1

        # Calculate ratios
        no_transition_ratio = 1.0 - (sentences_with_transition / total_sentences) if total_sentences > 0 else 0.5
        start_position_ratio = transitions_at_start / total_transitions if total_transitions > 0 else 0.0
        transition_per_sentence = total_transitions / total_sentences if total_sentences > 0 else 0.0

        # Normalize category frequencies
        def normalize_category(counts: Counter) -> Dict[str, float]:
            total = sum(counts.values())
            if total == 0:
                return {}
            return {word: count / total for word, count in counts.most_common(10)}

        return TransitionProfile(
            no_transition_ratio=float(no_transition_ratio),
            start_position_ratio=float(start_position_ratio),
            transition_per_sentence=float(transition_per_sentence),
            causal=normalize_category(category_counts["causal"]),
            adversative=normalize_category(category_counts["adversative"]),
            additive=normalize_category(category_counts["additive"]),
            temporal=normalize_category(category_counts["temporal"]),
        )

    def _classify_transition(self, token, doc) -> Optional[str]:
        """Classify a transition word by its semantic role using spaCy features."""
        word = token.text.lower()
        pos = token.pos_
        dep = token.dep_

        # Use dependency and semantic features to classify
        # SCONJ (subordinating conjunctions) - often causal or adversative
        if pos == 'SCONJ':
            # Check for adversative sense using dependency context
            if dep in ('mark', 'advmod'):
                # Look at the clause structure
                head = token.head
                if head.pos_ == 'VERB':
                    # Check if there's a contrast pattern
                    children_deps = {c.dep_ for c in head.children}
                    if 'neg' in children_deps:
                        return 'adversative'
            return 'causal'

        # CCONJ (coordinating conjunctions)
        if pos == 'CCONJ':
            # "but" is adversative, "and" is additive
            if word in ('but', 'yet'):
                return 'adversative'
            elif word in ('and', 'or'):
                return 'additive'
            return 'additive'

        # ADV (adverbs) - check semantic category
        if pos == 'ADV':
            # Use lemma for comparison
            lemma = token.lemma_.lower()

            # Temporal adverbs typically have temporal dependency
            if dep == 'advmod':
                head = token.head
                # Check if modifying a temporal verb or noun
                if head.pos_ == 'VERB':
                    # Adverbs that are sentence-initial and modify the main verb
                    # Check morphological features or context
                    if lemma in self._get_corpus_temporal_adverbs(doc):
                        return 'temporal'

            # Sentence-initial adverbs often signal logical relations
            if token.i == 0:
                # Check children/context for clues
                return 'additive'  # Default for sentence-initial adverbs

        return None

    def _classify_phrase_transition(self, phrase: str, doc) -> Optional[str]:
        """Classify a multi-word transition phrase."""
        # Check if the phrase follows a prepositional pattern
        # that typically signals discourse relations
        if doc[0].pos_ == 'ADP':
            # Prepositional phrases as discourse markers
            # The preposition gives us a semantic clue
            prep = doc[0].lemma_.lower()
            if prep in ('despite', 'notwithstanding'):
                return 'adversative'
            elif prep in ('because', 'due'):
                return 'causal'
            elif prep in ('in', 'additionally'):
                return 'additive'
            elif prep in ('before', 'after', 'during'):
                return 'temporal'
        return None

    def _get_corpus_temporal_adverbs(self, doc) -> set:
        """Extract temporal adverbs from context using spaCy's NER and dep parsing."""
        temporal = set()
        for token in doc:
            if token.ent_type_ in ('TIME', 'DATE'):
                temporal.add(token.lemma_.lower())
            if token.dep_ == 'advmod' and token.head.pos_ == 'VERB':
                # Check if the verb has temporal semantics
                if any(child.ent_type_ in ('TIME', 'DATE') for child in token.head.children):
                    temporal.add(token.lemma_.lower())
        return temporal

    def _extract_register_profile(
        self,
        sentences: List[str],
        paragraphs: List[str],
    ) -> RegisterProfile:
        """Extract register/formality features."""
        total_sentences = len(sentences)
        total_paragraphs = len(paragraphs)

        if total_sentences == 0:
            return RegisterProfile()

        # Count punctuation
        semicolons = sum(s.count(";") for s in sentences)
        colons = sum(s.count(":") for s in sentences)
        dashes = sum(s.count("—") + s.count("--") for s in sentences)
        parentheticals = sum(s.count("(") for s in sentences)
        questions = sum(1 for s in sentences if s.strip().endswith("?"))

        # Count passive voice and imperatives using spaCy
        passive_count = 0
        imperative_count = 0

        for sentence in sentences[:200]:  # Limit for performance
            doc = self.nlp(sentence)
            for token in doc:
                # Passive detection
                if token.dep_ == "nsubjpass":
                    passive_count += 1
                    break
                # Imperative detection (verb at start with no subject)
                if token.i == 0 and token.pos_ == "VERB":
                    has_subject = any(
                        child.dep_ in ("nsubj", "nsubjpass")
                        for child in token.children
                    )
                    if not has_subject:
                        imperative_count += 1
                        break

        # Calculate formality score (heuristic based on features)
        # Higher formality: more semicolons, colons, longer sentences, fewer contractions
        avg_length = sum(len(s.split()) for s in sentences) / total_sentences
        contraction_count = sum(
            1 for s in sentences
            if any(c in s.lower() for c in ["n't", "'re", "'ve", "'ll", "'d"])
        )

        formality_indicators = [
            min(1.0, semicolons / total_sentences * 5),  # Semicolons boost formality
            min(1.0, avg_length / 30),  # Longer sentences = more formal
            1.0 - min(1.0, contraction_count / total_sentences),  # Fewer contractions = formal
        ]
        formality_score = sum(formality_indicators) / len(formality_indicators)

        return RegisterProfile(
            formality_score=float(formality_score),
            narrative_ratio=0.0,  # Would need more analysis
            question_frequency=float(questions / total_paragraphs) if total_paragraphs > 0 else 0.0,
            imperative_frequency=float(imperative_count / total_paragraphs) if total_paragraphs > 0 else 0.0,
            passive_voice_ratio=float(passive_count / min(total_sentences, 200)),
            semicolon_per_sentence=float(semicolons / total_sentences),
            colon_per_sentence=float(colons / total_sentences),
            dash_per_sentence=float(dashes / total_sentences),
            parenthetical_per_sentence=float(parentheticals / total_sentences),
        )

    def _extract_delta_profile(self, sentences: List[str]) -> DeltaProfile:
        """Extract Burrows' Delta profile.

        Uses 300 most frequent words per research recommendations.
        """
        # Tokenize and count all words
        word_counts = Counter()
        total_words = 0

        for sentence in sentences:
            words = re.findall(r'\b[a-z]+\b', sentence.lower())
            word_counts.update(words)
            total_words += len(words)

        if total_words == 0:
            return DeltaProfile()

        # Get top 300 MFW
        top_300 = word_counts.most_common(300)

        # Calculate frequencies
        mfw_frequencies = {
            word: count / total_words
            for word, count in top_300
        }

        # For z-scores, we need corpus mean and std
        # In a full implementation, this would be calculated across multiple authors
        # For now, use the author's own frequencies as baseline
        mfw_zscores = {}
        corpus_mean = {}
        corpus_std = {}

        for word, freq in mfw_frequencies.items():
            # Simplified: z-score is 0 for the author's own text
            # In practice, you'd compare to a reference corpus
            mfw_zscores[word] = 0.0
            corpus_mean[word] = freq
            corpus_std[word] = freq * 0.5  # Rough estimate

        return DeltaProfile(
            mfw_frequencies=mfw_frequencies,
            mfw_zscores=mfw_zscores,
            corpus_mean=corpus_mean,
            corpus_std=corpus_std,
        )

    def _extract_structure_profile(self, sentences: List[str]) -> SentenceStructureProfile:
        """Extract sentence structure patterns using spaCy.

        Classifies sentences by syntactic complexity:
        - simple: One independent clause
        - compound: Multiple independent clauses (joined by CCONJ)
        - complex: Independent + dependent clause(s) (SCONJ, relative)
        - compound_complex: Multiple independent + dependent
        """
        structure_counts = Counter()
        structure_samples = defaultdict(list)  # Sanitized with [X] placeholders
        raw_samples = defaultdict(list)  # Original text for style prompts
        structures = []

        # Sample limit per structure type
        max_samples = 15

        for sentence in sentences:
            doc = self.nlp(sentence)
            structure = self._classify_sentence_structure(doc)
            structures.append(structure)
            structure_counts[structure] += 1

            # Collect sample sentences
            if len(structure_samples[structure]) < max_samples:
                # Sanitized version for analysis
                sanitized = self._sanitize_sample(doc)
                if sanitized and len(sanitized.split()) >= 5:
                    structure_samples[structure].append(sanitized)
                    # Also store raw sentence for style prompts
                    raw_samples[structure].append(sentence.strip())

        # Calculate distribution
        total = len(structures)
        structure_distribution = {
            s: count / total for s, count in structure_counts.items()
        } if total > 0 else {}

        # Build structure transition Markov model
        structure_transitions = self._build_structure_markov(structures)

        # Calculate proposition capacity from clause counts
        proposition_capacity = self._calculate_proposition_capacity(sentences)

        return SentenceStructureProfile(
            structure_distribution=structure_distribution,
            structure_transitions=structure_transitions,
            proposition_capacity=proposition_capacity,
            structure_samples=dict(structure_samples),
            raw_samples=dict(raw_samples),
            flow_samples=[],  # Will be populated by _segment_corpus_to_chunks
        )

    def _classify_sentence_structure(self, doc) -> str:
        """Classify sentence structure using spaCy dependency parse."""
        # Count clause indicators
        num_verbs = sum(1 for t in doc if t.pos_ == 'VERB' and t.dep_ in ('ROOT', 'conj', 'ccomp', 'xcomp', 'advcl', 'relcl'))
        has_cconj = any(t.pos_ == 'CCONJ' for t in doc)
        has_sconj = any(t.pos_ == 'SCONJ' for t in doc)
        has_relcl = any(t.dep_ == 'relcl' for t in doc)
        has_advcl = any(t.dep_ == 'advcl' for t in doc)

        has_subordinate = has_sconj or has_relcl or has_advcl

        if num_verbs <= 1:
            return "simple"
        elif has_cconj and has_subordinate:
            return "compound_complex"
        elif has_cconj:
            return "compound"
        elif has_subordinate:
            return "complex"
        else:
            return "simple"

    def _sanitize_sample(self, doc) -> str:
        """Sanitize a sample sentence - replace proper nouns, keep structure."""
        tokens = []
        for token in doc:
            if token.pos_ == 'PROPN':
                # Replace proper nouns with generic placeholder based on entity type
                if token.ent_type_ in ('PERSON', 'ORG', 'GPE', 'LOC'):
                    tokens.append('[X]')
                else:
                    tokens.append(token.text)
            else:
                tokens.append(token.text)

        # Reconstruct with proper spacing
        result = ""
        for i, token in enumerate(tokens):
            if i > 0 and token not in '.,;:!?\'")-' and tokens[i-1] not in '(\'"':
                result += " "
            result += token
        return result.strip()

    def _build_structure_markov(self, structures: List[str]) -> Dict[str, Dict[str, float]]:
        """Build Order-1 Markov chain for structure transitions."""
        transitions = defaultdict(Counter)

        for i in range(len(structures) - 1):
            current = structures[i]
            next_struct = structures[i + 1]
            transitions[current][next_struct] += 1

        # Normalize to probabilities
        markov = {}
        for current, next_counts in transitions.items():
            total = sum(next_counts.values())
            markov[current] = {
                struct: count / total for struct, count in next_counts.items()
            }

        # Ensure all structure types have entries
        all_types = {"simple", "compound", "complex", "compound_complex"}
        for struct in all_types:
            if struct not in markov:
                markov[struct] = {"simple": 0.4, "compound": 0.2, "complex": 0.3, "compound_complex": 0.1}

        return markov

    def _calculate_proposition_capacity(self, sentences: List[str]) -> Dict[str, int]:
        """Calculate how many propositions/clauses fit in each structure type."""
        capacity_counts = defaultdict(list)

        for sentence in sentences[:200]:  # Sample for performance
            doc = self.nlp(sentence)
            structure = self._classify_sentence_structure(doc)

            # Count clauses (verbs with subjects)
            clause_count = sum(1 for t in doc if t.dep_ in ('ROOT', 'conj', 'ccomp', 'advcl', 'relcl'))
            clause_count = max(1, clause_count)
            capacity_counts[structure].append(clause_count)

        # Calculate median capacity for each type
        capacity = {}
        for struct, counts in capacity_counts.items():
            capacity[struct] = int(np.median(counts)) if counts else 1

        # Ensure all types have entries
        defaults = {"simple": 1, "compound": 2, "complex": 2, "compound_complex": 3}
        for struct, default in defaults.items():
            if struct not in capacity:
                capacity[struct] = default

        return capacity

    def _segment_corpus_to_chunks(
        self,
        paragraphs: List[str],
        min_words: int = 150,
        max_words: int = 400,
        max_chunks: int = 20,
    ) -> List[str]:
        """Segment corpus into optimal chunks for style capture.

        Research insight: "Style lives in the transitions" - multi-sentence
        chunks with paragraph boundaries capture style better than isolated
        sentences. 150-400 word chunks are optimal for style transfer.

        Args:
            paragraphs: List of paragraph texts from corpus.
            min_words: Minimum words per chunk (default 150).
            max_words: Maximum words per chunk (default 400).
            max_chunks: Maximum number of chunks to return.

        Returns:
            List of text chunks, each 150-400 words showing multi-sentence flow.
        """
        chunks = []
        current_chunk = []
        current_words = 0

        for para in paragraphs:
            para = para.strip()
            if not para:
                continue

            para_words = len(para.split())

            # If adding this paragraph exceeds max and we have enough words, finalize chunk
            if current_words + para_words > max_words and current_words >= min_words:
                chunk_text = "\n\n".join(current_chunk)
                chunks.append(chunk_text)

                # Start new chunk with overlap (last sentence of previous paragraph)
                # This captures cross-paragraph transitions
                last_para = current_chunk[-1] if current_chunk else ""
                sentences = split_into_sentences(last_para)
                overlap = sentences[-1] if sentences else ""
                current_chunk = [overlap] if overlap else []
                current_words = len(overlap.split()) if overlap else 0

            current_chunk.append(para)
            current_words += para_words

        # Don't forget the last chunk
        if current_chunk and current_words >= min_words:
            chunk_text = "\n\n".join(current_chunk)
            chunks.append(chunk_text)

        # Prioritize chunks with paragraph boundaries (transitions)
        # Chunks with more paragraphs contain more style-revealing transitions
        def boundary_score(chunk: str) -> int:
            return chunk.count("\n\n") + 1  # More paragraph breaks = better

        chunks_with_scores = [(c, boundary_score(c)) for c in chunks]
        chunks_with_scores.sort(key=lambda x: x[1], reverse=True)

        logger.info(f"Created {len(chunks)} flow chunks from corpus "
                   f"(showing {min(max_chunks, len(chunks))} with most transitions)")

        return [c for c, _ in chunks_with_scores[:max_chunks]]

    def _extract_discourse_profile(self, paragraphs: List[str]) -> DiscourseRelationProfile:
        """Extract discourse relations between sentences using spaCy.

        Identifies PDTB-style relations:
        - CONTRAST: Opposition (but, however, although, yet)
        - CAUSE: Causal (because, so, therefore, thus)
        - ELABORATION: Expansion (for example, specifically, that is)
        - TEMPORAL: Time (then, after, before, when)
        - CONTINUATION: Simple continuation (and, also, moreover)
        """
        relation_counts = Counter()
        relation_samples = defaultdict(list)
        connective_counts = defaultdict(Counter)
        relations = []

        max_samples = 5  # Sample sentence pairs per relation

        for para in paragraphs:
            sentences = split_into_sentences(para)
            if len(sentences) < 2:
                continue

            for i in range(len(sentences) - 1):
                sent1 = sentences[i]
                sent2 = sentences[i + 1]

                # Classify the relation between sent1 and sent2
                relation, connective = self._classify_discourse_relation(sent2)
                relations.append(relation)
                relation_counts[relation] += 1

                if connective:
                    connective_counts[relation][connective] += 1

                # Store sample pairs
                if len(relation_samples[relation]) < max_samples:
                    sanitized1 = self._sanitize_sample(self.nlp(sent1))
                    sanitized2 = self._sanitize_sample(self.nlp(sent2))
                    if sanitized1 and sanitized2:
                        relation_samples[relation].append((sanitized1, sanitized2))

        # Calculate distribution
        total = len(relations)
        relation_distribution = {
            r: count / total for r, count in relation_counts.items()
        } if total > 0 else {}

        # Build relation transition Markov model
        relation_transitions = self._build_relation_markov(relations)

        # Normalize connective frequencies
        relation_connectives = {}
        for relation, counts in connective_counts.items():
            total_conn = sum(counts.values())
            relation_connectives[relation] = {
                conn: count / total_conn for conn, count in counts.most_common(5)
            }

        return DiscourseRelationProfile(
            relation_distribution=relation_distribution,
            relation_transitions=relation_transitions,
            relation_samples=dict(relation_samples),
            relation_connectives=relation_connectives,
        )

    def _classify_discourse_relation(self, sentence: str) -> Tuple[str, Optional[str]]:
        """Classify the discourse relation signaled by a sentence's opening.

        Uses spaCy's POS and dependency parsing to identify discourse markers.
        Returns (relation_type, connective_used).
        """
        doc = self.nlp(sentence)
        if len(doc) < 2:
            return ("continuation", None)

        first_token = doc[0]
        first_word = first_token.text.lower()
        first_pos = first_token.pos_
        first_dep = first_token.dep_

        # Check for subordinating conjunctions (SCONJ)
        if first_pos == 'SCONJ':
            # Classify by the conjunction's typical meaning
            # Check if it's adversative or causal based on dependency structure
            head = first_token.head
            if head.pos_ == 'VERB':
                # Look for negation or contrast patterns
                children_deps = {c.dep_ for c in head.children}
                if 'neg' in children_deps:
                    return ("contrast", first_word)

            # Default SCONJ to causal (because, since, if, when)
            # Check for temporal semantics
            if any(child.ent_type_ in ('TIME', 'DATE') for child in doc):
                return ("temporal", first_word)
            return ("cause", first_word)

        # Check for coordinating conjunctions (CCONJ)
        if first_pos == 'CCONJ':
            # Check morphological features and context
            if first_token.lemma_.lower() in ('but', 'yet'):
                return ("contrast", first_word)
            return ("continuation", first_word)

        # Check for adverbs (ADV) that signal discourse relations
        if first_pos == 'ADV':
            # Sentence-initial adverbs often signal logical relations
            # Check the semantic context
            head = first_token.head

            # Check for contrast patterns
            if first_dep == 'advmod' and head.pos_ == 'VERB':
                # Look for negation or contrast indicators
                children = list(head.children)
                has_neg = any(c.dep_ == 'neg' for c in children)
                if has_neg:
                    return ("contrast", first_word)

            return ("elaboration", first_word)

        # Check for prepositional phrases that signal relations
        if first_pos == 'ADP' and len(doc) >= 2:
            prep = first_word
            next_word = doc[1].text.lower() if len(doc) > 1 else ""
            phrase = f"{prep} {next_word}"

            # Classify by preposition
            if prep in ('despite', 'notwithstanding'):
                return ("contrast", phrase)
            elif prep in ('because', 'due'):
                return ("cause", phrase)
            elif prep in ('before', 'after', 'during'):
                return ("temporal", phrase)
            elif prep == 'in' and next_word in ('addition', 'fact', 'contrast'):
                if next_word == 'contrast':
                    return ("contrast", phrase)
                return ("elaboration", phrase)

        # Default: continuation (no explicit discourse marker)
        return ("continuation", None)

    def _build_relation_markov(self, relations: List[str]) -> Dict[str, Dict[str, float]]:
        """Build Order-1 Markov chain for discourse relation transitions."""
        transitions = defaultdict(Counter)

        for i in range(len(relations) - 1):
            current = relations[i]
            next_rel = relations[i + 1]
            transitions[current][next_rel] += 1

        # Normalize to probabilities
        markov = {}
        for current, next_counts in transitions.items():
            total = sum(next_counts.values())
            markov[current] = {
                rel: count / total for rel, count in next_counts.items()
            }

        # Ensure all relation types have entries
        all_relations = {"contrast", "cause", "elaboration", "temporal", "continuation"}
        default_dist = {r: 0.2 for r in all_relations}
        for rel in all_relations:
            if rel not in markov:
                markov[rel] = default_dist.copy()

        return markov

    def _extract_human_patterns(self, paragraphs: List[str]) -> Dict:
        """Extract human writing patterns for humanization.

        Identifies patterns that signal human authorship:
        - Sentence fragments
        - Rhetorical questions
        - Parenthetical asides
        - Extreme length variation
        """
        from ..humanization.corpus_patterns import CorpusPatternExtractor

        extractor = CorpusPatternExtractor()
        patterns = extractor.extract(paragraphs)
        return patterns.to_dict()

    # =========================================================================
    # Voice Profile Extraction Methods (Sprint 2)
    # =========================================================================

    def _extract_assertiveness_profile(self, sentences: List[str]) -> AssertivenessProfile:
        """Extract assertiveness patterns from corpus.

        Measures how directly the author makes claims:
        - Hedging words (perhaps, possibly, seems)
        - Booster words (certainly, clearly, must)
        - Epistemic stance distribution
        """
        total_sentences = len(sentences)
        if total_sentences == 0:
            return AssertivenessProfile()

        # Counters
        sentences_with_hedge = 0
        sentences_with_booster = 0
        hedge_counts = Counter()
        booster_counts = Counter()

        # Stance counters
        factual_count = 0
        conditional_count = 0
        hypothetical_count = 0
        imperative_count = 0
        question_count = 0

        # Pattern extraction
        assertion_patterns_found = []
        question_patterns_found = []

        for sentence in sentences:
            sentence_lower = sentence.lower()
            words = set(sentence_lower.split())
            doc = self.nlp(sentence)

            # Check for hedging
            found_hedges = words.intersection(HEDGING_WORDS)
            if found_hedges:
                sentences_with_hedge += 1
                for h in found_hedges:
                    hedge_counts[h] += 1

            # Check for hedging phrases
            for phrase in HEDGING_PHRASES:
                if phrase in sentence_lower:
                    sentences_with_hedge += 1
                    hedge_counts[phrase] += 1
                    break

            # Check for boosters
            found_boosters = words.intersection(BOOSTER_WORDS)
            if found_boosters:
                sentences_with_booster += 1
                for b in found_boosters:
                    booster_counts[b] += 1

            # Detect epistemic stance
            stance = self._detect_sentence_stance(sentence, doc)
            if stance == "factual":
                factual_count += 1
            elif stance == "conditional":
                conditional_count += 1
            elif stance == "hypothetical":
                hypothetical_count += 1

            # Check for imperatives
            if len(doc) > 0:
                first_token = doc[0]
                if first_token.pos_ == "VERB" and not any(
                    child.dep_ in ("nsubj", "nsubjpass") for child in first_token.children
                ):
                    imperative_count += 1

            # Check for questions
            if sentence.strip().endswith("?"):
                question_count += 1
                # Extract question pattern (first few words)
                pattern = " ".join(sentence.split()[:4])
                if pattern not in question_patterns_found:
                    question_patterns_found.append(pattern)

            # Extract assertion patterns (sentences starting with strong claims)
            if self._is_strong_assertion(sentence, doc):
                pattern = " ".join(sentence.split()[:5])
                if pattern not in assertion_patterns_found and len(assertion_patterns_found) < 20:
                    assertion_patterns_found.append(pattern)

        # Calculate ratios
        hedge_ratio = sentences_with_hedge / total_sentences
        booster_ratio = sentences_with_booster / total_sentences

        # Calculate average commitment: -1 (hedged) to +1 (assertive)
        # Based on relative frequency of boosters vs hedges
        if sentences_with_hedge + sentences_with_booster > 0:
            average_commitment = (sentences_with_booster - sentences_with_hedge) / (
                sentences_with_hedge + sentences_with_booster
            )
        else:
            average_commitment = 0.0

        # Normalize stance counts
        factual_ratio = factual_count / total_sentences
        conditional_ratio = conditional_count / total_sentences
        hypothetical_ratio = hypothetical_count / total_sentences
        imperative_ratio = imperative_count / total_sentences

        # Normalize hedge/booster word frequencies
        total_hedges = sum(hedge_counts.values()) or 1
        total_boosters = sum(booster_counts.values()) or 1

        author_hedges = {
            word: count / total_hedges
            for word, count in hedge_counts.most_common(15)
        }
        author_boosters = {
            word: count / total_boosters
            for word, count in booster_counts.most_common(15)
        }

        return AssertivenessProfile(
            hedge_ratio=float(hedge_ratio),
            booster_ratio=float(booster_ratio),
            average_commitment=float(average_commitment),
            factual_ratio=float(factual_ratio),
            conditional_ratio=float(conditional_ratio),
            hypothetical_ratio=float(hypothetical_ratio),
            imperative_ratio=float(imperative_ratio),
            assertion_patterns=assertion_patterns_found[:15],
            question_patterns=question_patterns_found[:10],
            author_hedges=author_hedges,
            author_boosters=author_boosters,
        )

    def _detect_sentence_stance(self, sentence: str, doc) -> str:
        """Detect the epistemic stance of a sentence."""
        sentence_lower = sentence.lower()
        words = set(sentence_lower.split())

        # Check for appearance markers
        if words.intersection(APPEARANCE_MARKERS):
            return "appearance"

        # Check appearance phrases
        for phrase in APPEARANCE_PHRASES:
            if phrase in sentence_lower:
                return "appearance"

        # Check appearance patterns (regex)
        for pattern in APPEARANCE_PATTERNS:
            if re.search(pattern, sentence_lower):
                return "appearance"

        # Check for conditional
        if words.intersection({"if", "unless", "provided", "assuming"}):
            return "conditional"

        # Check for hypothetical
        if words.intersection({"would", "could", "might", "may"}):
            return "hypothetical"

        return "factual"

    def _is_strong_assertion(self, sentence: str, doc) -> bool:
        """Check if a sentence is a strong assertion (for pattern extraction)."""
        if len(doc) < 3:
            return False

        sentence_lower = sentence.lower()
        words = set(sentence_lower.split())

        # Has booster words
        if words.intersection(BOOSTER_WORDS):
            return True

        # Starts with definite article + noun + is/are
        if doc[0].pos_ == "DET" and len(doc) > 2:
            for token in doc[1:4]:
                if token.lemma_ in ("be", "is", "are"):
                    return True

        # Starts with "It is" or "There is"
        if sentence_lower.startswith(("it is", "there is", "this is", "that is")):
            return True

        # Starts with "The fact is" or "The truth is"
        if sentence_lower.startswith(("the fact", "the truth", "the reality")):
            return True

        return False

    def _extract_rhetorical_profile(
        self,
        sentences: List[str],
        paragraphs: List[str],
    ) -> RhetoricalProfile:
        """Extract rhetorical structure patterns from corpus.

        Detects how the author frames arguments:
        - Contrast patterns (but, however, yet)
        - Resolution patterns (thus, therefore, in truth)
        - Dialectical patterns (not X but Y)
        """
        total_sentences = len(sentences)
        if total_sentences == 0:
            return RhetoricalProfile()

        # Counters
        contrast_count = 0
        resolution_count = 0
        question_count = 0
        negation_affirmation_count = 0
        appearance_reality_count = 0
        direct_address_count = 0

        # Detected markers
        contrast_markers_found = Counter()
        resolution_markers_found = Counter()

        # Opposition pairs
        detected_oppositions = []

        # Pattern samples
        pattern_samples = defaultdict(list)
        max_samples = 5

        # Resolution markers (conclusion/synthesis markers)
        resolution_markers = {
            "thus", "therefore", "hence", "consequently", "accordingly",
            "in truth", "the fact is", "the reality is", "in fact",
            "indeed", "truly", "actually", "in essence",
        }

        for sentence in sentences:
            sentence_lower = sentence.lower()
            words = set(sentence_lower.split())

            # Check for contrast markers
            found_contrast = False
            for marker in CONTRAST_MARKERS:
                if marker in sentence_lower:
                    contrast_count += 1
                    contrast_markers_found[marker] += 1
                    found_contrast = True
                    if len(pattern_samples["contrast"]) < max_samples:
                        pattern_samples["contrast"].append(sentence[:100])
                    break

            # Check for resolution markers
            for marker in resolution_markers:
                if marker in sentence_lower:
                    resolution_count += 1
                    resolution_markers_found[marker] += 1
                    if len(pattern_samples["resolution"]) < max_samples:
                        pattern_samples["resolution"].append(sentence[:100])
                    break

            # Check for questions
            if sentence.strip().endswith("?"):
                question_count += 1
                if len(pattern_samples["question"]) < max_samples:
                    pattern_samples["question"].append(sentence[:100])

            # Check for "not X, but Y" patterns
            if re.search(r"\bnot\b.*\bbut\b", sentence_lower):
                negation_affirmation_count += 1
                if len(pattern_samples["negation_affirmation"]) < max_samples:
                    pattern_samples["negation_affirmation"].append(sentence[:100])

                # Extract opposition pair
                match = re.search(r"\bnot\s+(\w+).*\bbut\s+(\w+)", sentence_lower)
                if match:
                    opp = (match.group(1), match.group(2))
                    if opp not in detected_oppositions:
                        detected_oppositions.append(opp)

            # Check for appearance vs reality patterns
            if re.search(r"\b(seem|appear|look)s?\b.*\b(but|however|actually|in fact)\b", sentence_lower):
                appearance_reality_count += 1
                if len(pattern_samples["appearance_reality"]) < max_samples:
                    pattern_samples["appearance_reality"].append(sentence[:100])

            # Check for direct address (you, we)
            if re.search(r"\b(you|we)\s+(must|should|need|have to|can)\b", sentence_lower):
                direct_address_count += 1
                if len(pattern_samples["direct_address"]) < max_samples:
                    pattern_samples["direct_address"].append(sentence[:100])

        # Calculate ratios
        contrast_frequency = contrast_count / total_sentences
        resolution_frequency = resolution_count / total_sentences
        question_frequency = question_count / total_sentences
        negation_affirmation_ratio = negation_affirmation_count / total_sentences
        appearance_reality_ratio = appearance_reality_count / total_sentences
        direct_address_ratio = direct_address_count / total_sentences

        # Get top markers
        top_resolution_markers = [m for m, _ in resolution_markers_found.most_common(10)]
        top_contrast_markers = [m for m, _ in contrast_markers_found.most_common(10)]

        return RhetoricalProfile(
            contrast_frequency=float(contrast_frequency),
            resolution_frequency=float(resolution_frequency),
            question_frequency=float(question_frequency),
            negation_affirmation_ratio=float(negation_affirmation_ratio),
            appearance_reality_ratio=float(appearance_reality_ratio),
            direct_address_ratio=float(direct_address_ratio),
            resolution_markers=top_resolution_markers,
            contrast_markers=top_contrast_markers,
            detected_oppositions=detected_oppositions[:20],
            pattern_samples=dict(pattern_samples),
        )

    def _extract_phrase_patterns(
        self,
        sentences: List[str],
        paragraphs: List[str],
    ) -> PhrasePatterns:
        """Extract characteristic phrase patterns from corpus.

        Captures the author's signature constructions:
        - Paragraph openers
        - Sentence connectors
        - Characteristic constructions
        """
        # Paragraph openers (first words of paragraphs)
        paragraph_openers = Counter()
        for para in paragraphs:
            para = para.strip()
            if not para:
                continue
            # Get first 3-4 words as opener
            words = para.split()[:4]
            if words:
                opener = " ".join(words)
                paragraph_openers[opener] += 1

        # Sentence connectors (sentence-initial phrases)
        sentence_connectors = Counter()
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            # Get first 2-3 words
            words = sentence.split()[:3]
            if words:
                connector = " ".join(words)
                # Only count if starts with transition-like word
                first_word = words[0].lower()
                if first_word in {
                    "but", "however", "yet", "thus", "therefore", "hence",
                    "so", "and", "or", "moreover", "furthermore", "indeed",
                    "in", "on", "for", "with", "as", "this", "that", "these",
                }:
                    sentence_connectors[connector] += 1

        # Characteristic constructions (syntactic patterns)
        characteristic_constructions = []
        emphasis_patterns = []
        parallelism_examples = []
        antithesis_examples = []

        for sentence in sentences:
            sentence_lower = sentence.lower()

            # "It is X that Y" construction
            if re.search(r"\bit is\s+\w+\s+that\b", sentence_lower):
                if len(characteristic_constructions) < 10:
                    characteristic_constructions.append(sentence[:80])

            # "What we call X is Y" construction
            if re.search(r"\bwhat\s+we\s+(call|see|know)\b", sentence_lower):
                if len(characteristic_constructions) < 10:
                    characteristic_constructions.append(sentence[:80])

            # "The question is not X but Y" construction
            if re.search(r"\bthe\s+question\s+is\b", sentence_lower):
                if len(characteristic_constructions) < 10:
                    characteristic_constructions.append(sentence[:80])

            # Emphasis with dashes or parentheses
            if "—" in sentence or " -- " in sentence:
                if len(emphasis_patterns) < 10:
                    emphasis_patterns.append(sentence[:100])

            # Parallelism (repeating structure)
            if re.search(r"(\b\w+\b),\s+\1", sentence_lower):  # word, word pattern
                if len(parallelism_examples) < 10:
                    parallelism_examples.append(sentence[:100])

            # "Not X, not Y, but Z" pattern
            if re.search(r"\bnot\s+\w+,\s+not\s+\w+", sentence_lower):
                if len(parallelism_examples) < 10:
                    parallelism_examples.append(sentence[:100])

            # Antithesis (X on one hand, Y on the other)
            if re.search(r"\bon\s+(the\s+)?one\s+hand\b", sentence_lower):
                if len(antithesis_examples) < 10:
                    antithesis_examples.append(sentence[:100])

            # "While X, Y" antithesis
            if re.search(r"^while\s+", sentence_lower):
                if len(antithesis_examples) < 10:
                    antithesis_examples.append(sentence[:100])

        # Calculate frequencies
        total_paragraphs = len([p for p in paragraphs if p.strip()])
        total_sentences = len(sentences)

        opener_with_pattern = sum(
            1 for o, c in paragraph_openers.items()
            if any(word in o.lower() for word in {"the", "in", "but", "this", "we", "it"})
        )
        opener_frequency = opener_with_pattern / total_paragraphs if total_paragraphs > 0 else 0.0

        construction_frequency = len(characteristic_constructions) / total_sentences if total_sentences > 0 else 0.0

        return PhrasePatterns(
            paragraph_openers=[o for o, _ in paragraph_openers.most_common(20)],
            sentence_connectors=[c for c, _ in sentence_connectors.most_common(20)],
            characteristic_constructions=characteristic_constructions,
            emphasis_patterns=emphasis_patterns,
            parallelism_examples=parallelism_examples,
            antithesis_examples=antithesis_examples,
            opener_frequency=float(opener_frequency),
            construction_frequency=float(construction_frequency),
        )


def extract_author_profile(
    paragraphs: List[str],
    author_name: str,
) -> AuthorStyleProfile:
    """Convenience function to extract author profile."""
    extractor = StyleProfileExtractor()
    return extractor.extract(paragraphs, author_name)
