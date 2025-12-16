"""
Stage 2: Style Analysis Module

Extracts stylistic fingerprint from sample text to guide synthesis.
Analyzes:
- Vocabulary profile: word frequency, formality, domain-specific terms
- Sentence patterns: length distribution, structure templates, punctuation
- Tone markers: rhetorical devices, assertiveness, hedging patterns
"""

import spacy
import re
import json
import math
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, asdict, field
from collections import Counter, defaultdict


@dataclass
class VocabularyProfile:
    """Vocabulary characteristics of the sample text."""
    total_words: int
    unique_words: int
    vocabulary_richness: float  # unique/total ratio
    avg_word_length: float
    common_words: List[Tuple[str, int]]  # Most frequent content words
    rare_words: List[str]  # Words used only once
    formality_score: float  # 0=informal, 1=formal
    domain_terms: List[str]  # Subject-specific vocabulary
    transition_words: List[Tuple[str, int]]  # Connectives with frequency
    forbidden_words: List[str]  # Words to avoid (AI-typical)


@dataclass
class SentencePattern:
    """A sentence structure pattern from the sample."""
    template: str  # e.g., "SUBJ + VERB + OBJ + PREP_PHRASE"
    example: str
    word_count: int
    clause_count: int
    has_subordinate: bool
    opener_type: str  # How the sentence starts


@dataclass
class SentencePatterns:
    """Collection of sentence patterns from sample."""
    length_distribution: Dict[str, float]  # short/medium/long percentages
    avg_length: float
    length_variance: float
    patterns: List[SentencePattern]
    opener_distribution: Dict[str, float]  # Types of sentence openers
    punctuation_usage: Dict[str, float]  # Semicolons, em-dashes, etc per sentence


@dataclass
class ToneMarkers:
    """Tone and rhetorical characteristics."""
    assertiveness: float  # 0=hedged, 1=confident
    hedging_frequency: float  # How often hedging words appear
    rhetorical_questions: int
    imperatives: int
    passive_voice_ratio: float
    common_rhetorical_devices: List[str]
    paragraph_length_avg: float
    paragraph_length_variance: float


@dataclass
class PhrasalPattern:
    """A distinctive multi-word pattern from the sample."""
    phrase: str           # e.g., "contrary to metaphysics"
    position: str         # 'opener', 'mid', 'closer'
    frequency: int
    example_sentences: List[str]


@dataclass
class SyntacticConstruction:
    """A reusable sentence construction template."""
    pattern: str          # e.g., "Contrary to X, Y holds that Z"
    construction_type: str  # 'contrastive', 'definition', 'enumeration', 'causal'
    slots: Dict[str, str]  # {'X': 'noun_phrase', 'Y': 'subject', 'Z': 'claim'}
    examples: List[str]   # Actual sentences from sample


@dataclass
class DiscourseMarkerUsage:
    """How discourse markers are used in the sample."""
    marker: str
    typical_position: str  # 'sentence_start', 'clause_start', 'parenthetical'
    frequency_per_100_sentences: float
    example_contexts: List[str]


@dataclass
class DistinctivePatterns:
    """Collection of distinctive style patterns."""
    phrasal_patterns: List[PhrasalPattern]
    syntactic_constructions: List[SyntacticConstruction]
    discourse_markers: List[DiscourseMarkerUsage]


@dataclass
class StyleProfile:
    """Complete style profile of a text."""
    vocabulary: VocabularyProfile
    sentences: SentencePatterns
    tone: ToneMarkers
    example_sentences: Dict[str, List[str]]  # By category
    distinctive_patterns: Optional[DistinctivePatterns] = None  # New: pattern library

    def to_dict(self) -> Dict:
        result = {
            'vocabulary': asdict(self.vocabulary),
            'sentences': asdict(self.sentences),
            'tone': asdict(self.tone),
            'example_sentences': self.example_sentences
        }
        if self.distinctive_patterns:
            result['distinctive_patterns'] = asdict(self.distinctive_patterns)
        return result

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, default=str)

    def to_style_guide(self) -> str:
        """Generate a human-readable style guide for the LLM."""
        guide = []

        guide.append("# STYLE GUIDE")
        guide.append("")
        guide.append("## Vocabulary")
        guide.append(f"- Formality level: {'formal' if self.vocabulary.formality_score > 0.6 else 'moderate' if self.vocabulary.formality_score > 0.3 else 'informal'}")
        guide.append(f"- Average word length: {self.vocabulary.avg_word_length:.1f} characters")
        guide.append(f"- Vocabulary richness: {self.vocabulary.vocabulary_richness:.2f} (unique/total ratio)")

        if self.vocabulary.domain_terms:
            guide.append(f"- Domain terms to emulate: {', '.join(self.vocabulary.domain_terms[:10])}")

        if self.vocabulary.transition_words:
            top_transitions = [w for w, c in self.vocabulary.transition_words[:8]]
            guide.append(f"- Preferred transitions: {', '.join(top_transitions)}")

        if self.vocabulary.forbidden_words:
            guide.append(f"- AVOID these AI-typical words: {', '.join(self.vocabulary.forbidden_words)}")

        guide.append("")
        guide.append("## Sentence Structure")
        guide.append(f"- Average sentence length: {self.sentences.avg_length:.1f} words")

        dist = self.sentences.length_distribution
        guide.append(f"- Length mix: {dist.get('short', 0)*100:.0f}% short (5-10 words), {dist.get('medium', 0)*100:.0f}% medium (11-25 words), {dist.get('long', 0)*100:.0f}% long (26+ words)")

        if self.sentences.opener_distribution:
            openers = sorted(self.sentences.opener_distribution.items(), key=lambda x: -x[1])
            opener_desc = [f"{k}: {v*100:.0f}%" for k, v in openers[:4]]
            guide.append(f"- Sentence openers: {', '.join(opener_desc)}")

        punct = self.sentences.punctuation_usage
        if punct:
            guide.append(f"- Semicolons per sentence: {punct.get('semicolon', 0):.2f}")
            guide.append(f"- Commas per sentence: {punct.get('comma', 0):.2f}")

        guide.append("")
        guide.append("## Tone")
        guide.append(f"- Assertiveness: {'high' if self.tone.assertiveness > 0.7 else 'moderate' if self.tone.assertiveness > 0.4 else 'hedged'}")
        guide.append(f"- Passive voice: {self.tone.passive_voice_ratio*100:.0f}% of sentences")
        guide.append(f"- Average paragraph: {self.tone.paragraph_length_avg:.0f} sentences")

        # Add distinctive patterns section (NEW)
        if self.distinctive_patterns:
            guide.append("")
            guide.append("## DISTINCTIVE PATTERNS (USE THESE!)")
            guide.append("")
            guide.append("### Characteristic Phrases")
            guide.append("Use these multi-word patterns that define this style:")

            for pattern in self.distinctive_patterns.phrasal_patterns[:10]:
                guide.append(f'- "{pattern.phrase}" (typically at {pattern.position})')
                if pattern.example_sentences:
                    guide.append(f'  Example: "{pattern.example_sentences[0][:100]}..."')

            guide.append("")
            guide.append("### Syntactic Constructions")
            guide.append("IMPORTANT: Use these sentence patterns to structure your writing:")

            for construction in self.distinctive_patterns.syntactic_constructions[:8]:
                guide.append(f"\n**{construction.construction_type.upper()} pattern:**")
                guide.append(f"  Template: {construction.pattern}")
                if construction.examples:
                    guide.append(f"  Example: \"{construction.examples[0][:150]}...\"" if len(construction.examples[0]) > 150 else f"  Example: \"{construction.examples[0]}\"")

            guide.append("")
            guide.append("### Discourse Markers")
            guide.append("Use these connectors in these positions:")

            for marker in self.distinctive_patterns.discourse_markers[:10]:
                guide.append(f'- "{marker.marker}" at {marker.typical_position} ({marker.frequency_per_100_sentences:.1f} per 100 sentences)')

        # Add example sentences
        guide.append("")
        guide.append("## Example Sentences (for rhythm reference)")

        for category, examples in self.example_sentences.items():
            if examples:
                guide.append(f"\n### {category}")
                for ex in examples[:3]:
                    guide.append(f'- "{ex}"')

        return '\n'.join(guide)


class StyleAnalyzer:
    """
    Analyzes sample text to extract stylistic fingerprint.

    The goal is to capture HOW things are said, not WHAT is said.
    This allows the synthesizer to apply the same style to different content.
    """

    def __init__(self):
        """Initialize the style analyzer with spaCy model."""
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            print("Downloading spaCy model 'en_core_web_sm'...")
            import subprocess
            subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"])
            self.nlp = spacy.load("en_core_web_sm")

        # Common transition words to track
        self.transition_words = {
            'therefore', 'hence', 'thus', 'consequently', 'accordingly',
            'however', 'nevertheless', 'nonetheless', 'moreover', 'furthermore',
            'similarly', 'likewise', 'indeed', 'certainly', 'clearly',
            'first', 'second', 'finally', 'then', 'next', 'subsequently',
            'for example', 'for instance', 'specifically', 'namely',
            'in other words', 'that is', 'in fact', 'actually'
        }

        # Words that signal formal writing
        self.formal_markers = {
            'therefore', 'hence', 'thus', 'consequently', 'furthermore',
            'moreover', 'nevertheless', 'notwithstanding', 'whereas',
            'inasmuch', 'whereby', 'wherein', 'herein', 'thereof'
        }

        # Words that signal informal writing
        self.informal_markers = {
            "can't", "won't", "don't", "isn't", "aren't", "wasn't",
            'gonna', 'wanna', 'gotta', 'kinda', 'sorta',
            'yeah', 'okay', 'ok', 'hey', 'stuff', 'things'
        }

        # AI-typical words to avoid
        self.ai_typical_words = {
            'delve', 'leverage', 'seamless', 'tapestry', 'crucial', 'vibrant',
            'realm', 'landscape', 'symphony', 'orchestrate', 'myriad', 'plethora',
            'paradigm', 'synergy', 'holistic', 'robust', 'streamline', 'optimize',
            'nuanced', 'multifaceted', 'intricate', 'pivotal', 'transformative'
        }

        # Hedging words
        self.hedging_words = {
            'might', 'may', 'could', 'possibly', 'perhaps', 'likely',
            'probably', 'suggests', 'appears', 'seems', 'somewhat',
            'relatively', 'approximately', 'essentially', 'basically'
        }

    def analyze(self, text: str) -> StyleProfile:
        """
        Analyze text to extract its stylistic profile.

        Args:
            text: Sample text to analyze

        Returns:
            StyleProfile with vocabulary, sentence patterns, tone analysis,
            and distinctive patterns
        """
        # Pre-process text
        normalized_text = self._normalize_text(text)

        # Process with spaCy
        doc = self.nlp(normalized_text)

        # Extract base components
        vocabulary = self._analyze_vocabulary(doc, normalized_text)
        sentences = self._analyze_sentences(doc)
        tone = self._analyze_tone(doc, normalized_text)
        examples = self._extract_example_sentences(doc)

        # Extract distinctive patterns (new)
        phrasal_patterns = self._extract_phrasal_patterns(doc)
        syntactic_constructions = self._extract_syntactic_constructions(doc, normalized_text)
        discourse_markers = self._map_discourse_markers(doc, normalized_text)

        distinctive_patterns = DistinctivePatterns(
            phrasal_patterns=phrasal_patterns,
            syntactic_constructions=syntactic_constructions,
            discourse_markers=discourse_markers
        )

        return StyleProfile(
            vocabulary=vocabulary,
            sentences=sentences,
            tone=tone,
            example_sentences=examples,
            distinctive_patterns=distinctive_patterns
        )

    def _normalize_text(self, text: str) -> str:
        """Normalize text for analysis."""
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        # Remove tabs
        text = text.replace('\t', ' ')
        # Normalize quotes
        text = text.replace('"', '"').replace('"', '"')
        text = text.replace(''', "'").replace(''', "'")
        return text.strip()

    def _analyze_vocabulary(self, doc, text: str) -> VocabularyProfile:
        """Analyze vocabulary characteristics."""
        # Get all words (excluding punctuation and spaces)
        words = [token.text.lower() for token in doc
                if not token.is_punct and not token.is_space]

        total_words = len(words)
        unique_words = len(set(words))

        # Word frequency (content words only - nouns, verbs, adjectives, adverbs)
        content_words = [token.text.lower() for token in doc
                        if token.pos_ in ('NOUN', 'VERB', 'ADJ', 'ADV')
                        and not token.is_stop]
        word_freq = Counter(content_words)
        common_words = word_freq.most_common(20)

        # Rare words (hapax legomena)
        rare_words = [word for word, count in word_freq.items() if count == 1][:20]

        # Average word length
        avg_word_length = sum(len(w) for w in words) / max(1, len(words))

        # Formality score
        formality_score = self._calculate_formality(doc, text)

        # Domain terms (capitalized noun phrases, technical terms)
        domain_terms = self._extract_domain_terms(doc)

        # Transition words used
        text_lower = text.lower()
        transitions = []
        for tw in self.transition_words:
            count = text_lower.count(tw)
            if count > 0:
                transitions.append((tw, count))
        transitions.sort(key=lambda x: -x[1])

        # Check for AI-typical words (should be avoided)
        found_ai_words = [w for w in self.ai_typical_words if w in text_lower]

        return VocabularyProfile(
            total_words=total_words,
            unique_words=unique_words,
            vocabulary_richness=unique_words / max(1, total_words),
            avg_word_length=avg_word_length,
            common_words=common_words,
            rare_words=rare_words,
            formality_score=formality_score,
            domain_terms=domain_terms,
            transition_words=transitions,
            forbidden_words=list(self.ai_typical_words)  # Always avoid these
        )

    def _calculate_formality(self, doc, text: str) -> float:
        """Calculate formality score (0=informal, 1=formal)."""
        text_lower = text.lower()

        # Count formal and informal markers
        formal_count = sum(1 for word in self.formal_markers if word in text_lower)
        informal_count = sum(1 for word in self.informal_markers if word in text_lower)

        # Check for contractions (informal)
        contractions = len(re.findall(r"\w+n't|\w+'s|\w+'re|\w+'ve|\w+'ll|\w+'d", text_lower))
        informal_count += contractions

        # Check sentence length (longer = more formal typically)
        sentences = list(doc.sents)
        avg_sent_len = sum(len(list(s)) for s in sentences) / max(1, len(sentences))
        length_formality = min(1.0, avg_sent_len / 25)  # 25+ words = very formal

        # Calculate combined score
        if formal_count + informal_count == 0:
            marker_score = 0.5
        else:
            marker_score = formal_count / (formal_count + informal_count)

        # Weight: 60% markers, 40% length
        return 0.6 * marker_score + 0.4 * length_formality

    def _extract_domain_terms(self, doc) -> List[str]:
        """Extract domain-specific terminology."""
        domain_terms = []

        # Named entities
        for ent in doc.ents:
            if ent.label_ in ('ORG', 'PERSON', 'GPE', 'WORK_OF_ART', 'LAW'):
                domain_terms.append(ent.text)

        # Capitalized noun phrases (often technical terms)
        for chunk in doc.noun_chunks:
            if chunk.text[0].isupper() and chunk.root.pos_ == 'NOUN':
                domain_terms.append(chunk.text)

        # Deduplicate
        seen = set()
        unique_terms = []
        for term in domain_terms:
            if term.lower() not in seen:
                seen.add(term.lower())
                unique_terms.append(term)

        return unique_terms[:15]

    def _analyze_sentences(self, doc) -> SentencePatterns:
        """Analyze sentence structure patterns."""
        sentences = list(doc.sents)

        if not sentences:
            return SentencePatterns(
                length_distribution={'short': 0, 'medium': 0, 'long': 0},
                avg_length=0,
                length_variance=0,
                patterns=[],
                opener_distribution={},
                punctuation_usage={}
            )

        # Calculate length distribution
        lengths = []
        short_count = medium_count = long_count = 0

        for sent in sentences:
            word_count = len([t for t in sent if not t.is_punct and not t.is_space])
            lengths.append(word_count)

            if word_count <= 10:
                short_count += 1
            elif word_count <= 25:
                medium_count += 1
            else:
                long_count += 1

        total = len(sentences)
        length_distribution = {
            'short': short_count / total,
            'medium': medium_count / total,
            'long': long_count / total
        }

        avg_length = sum(lengths) / total
        length_variance = sum((l - avg_length) ** 2 for l in lengths) / total

        # Extract patterns
        patterns = [self._extract_sentence_pattern(sent) for sent in sentences[:50]]
        patterns = [p for p in patterns if p is not None]

        # Opener distribution
        opener_counts = Counter()
        for sent in sentences:
            opener = self._classify_opener(sent)
            opener_counts[opener] += 1

        opener_distribution = {k: v/total for k, v in opener_counts.items()}

        # Punctuation usage
        punctuation_usage = self._analyze_punctuation(sentences)

        return SentencePatterns(
            length_distribution=length_distribution,
            avg_length=avg_length,
            length_variance=length_variance,
            patterns=patterns,
            opener_distribution=opener_distribution,
            punctuation_usage=punctuation_usage
        )

    def _extract_sentence_pattern(self, sent) -> Optional[SentencePattern]:
        """Extract structural pattern from a sentence."""
        tokens = list(sent)
        word_count = len([t for t in tokens if not t.is_punct and not t.is_space])

        if word_count < 3:
            return None

        # Build template from dependency structure
        template_parts = []
        for token in tokens:
            if token.dep_ == 'ROOT':
                template_parts.append('VERB')
            elif token.dep_ in ('nsubj', 'nsubjpass'):
                template_parts.append('SUBJ')
            elif token.dep_ == 'dobj':
                template_parts.append('OBJ')
            elif token.dep_ == 'prep':
                template_parts.append('PREP')
            elif token.dep_ == 'advcl':
                template_parts.append('ADVCL')
            elif token.dep_ == 'relcl':
                template_parts.append('RELCL')

        # Deduplicate consecutive entries
        deduped = []
        for part in template_parts:
            if not deduped or deduped[-1] != part:
                deduped.append(part)

        template = ' + '.join(deduped) if deduped else 'SIMPLE'

        # Count clauses
        clause_count = 1 + sum(1 for t in tokens if t.dep_ in ('advcl', 'relcl', 'ccomp'))

        # Check for subordinate clause
        has_subordinate = any(t.dep_ in ('advcl', 'mark') for t in tokens)

        return SentencePattern(
            template=template,
            example=sent.text.strip(),
            word_count=word_count,
            clause_count=clause_count,
            has_subordinate=has_subordinate,
            opener_type=self._classify_opener(sent)
        )

    def _classify_opener(self, sent) -> str:
        """Classify how a sentence opens."""
        tokens = list(sent)
        if not tokens:
            return 'UNKNOWN'

        first_token = tokens[0]

        # Check for specific patterns
        if first_token.text.lower() in ('the', 'a', 'an'):
            return 'ARTICLE'
        elif first_token.pos_ == 'PRON':
            if first_token.text.lower() in ('this', 'that', 'these', 'those'):
                return 'DEMONSTRATIVE'
            elif first_token.text.lower() in ('it', 'there'):
                return 'EXPLETIVE'
            else:
                return 'PRONOUN'
        elif first_token.pos_ == 'ADV':
            return 'ADVERB'
        elif first_token.pos_ == 'SCONJ':
            return 'SUBORDINATOR'
        elif first_token.pos_ == 'CCONJ':
            return 'COORDINATOR'
        elif first_token.pos_ in ('NOUN', 'PROPN'):
            return 'NOUN'
        elif first_token.pos_ == 'VERB':
            return 'VERB'
        elif first_token.pos_ == 'ADP':
            return 'PREPOSITION'
        else:
            return 'OTHER'

    def _analyze_punctuation(self, sentences) -> Dict[str, float]:
        """Analyze punctuation usage per sentence."""
        total = len(sentences)
        if total == 0:
            return {}

        semicolons = sum(sent.text.count(';') for sent in sentences)
        commas = sum(sent.text.count(',') for sent in sentences)
        colons = sum(sent.text.count(':') for sent in sentences)
        em_dashes = sum(sent.text.count('—') + sent.text.count('--') for sent in sentences)
        parentheses = sum(sent.text.count('(') for sent in sentences)

        return {
            'semicolon': semicolons / total,
            'comma': commas / total,
            'colon': colons / total,
            'em_dash': em_dashes / total,
            'parenthesis': parentheses / total
        }

    def _analyze_tone(self, doc, text: str) -> ToneMarkers:
        """Analyze tone and rhetorical characteristics."""
        sentences = list(doc.sents)
        total_sentences = len(sentences)

        if total_sentences == 0:
            return ToneMarkers(
                assertiveness=0.5,
                hedging_frequency=0,
                rhetorical_questions=0,
                imperatives=0,
                passive_voice_ratio=0,
                common_rhetorical_devices=[],
                paragraph_length_avg=0,
                paragraph_length_variance=0
            )

        # Count hedging words
        text_lower = text.lower()
        hedging_count = sum(text_lower.count(word) for word in self.hedging_words)
        hedging_frequency = hedging_count / total_sentences

        # Assertiveness (inverse of hedging, plus presence of strong verbs)
        strong_verbs = {'must', 'shall', 'will', 'cannot', 'is', 'are', 'requires', 'demands'}
        strong_count = sum(text_lower.count(word) for word in strong_verbs)
        assertiveness = min(1.0, 0.5 + (strong_count / total_sentences * 0.3) - (hedging_frequency * 0.2))

        # Count rhetorical questions
        rhetorical_questions = sum(1 for sent in sentences if sent.text.strip().endswith('?'))

        # Count imperatives (sentences starting with verb)
        imperatives = sum(1 for sent in sentences
                        if list(sent)[0].pos_ == 'VERB' and not sent.text.strip().endswith('?'))

        # Passive voice ratio
        passive_count = sum(1 for sent in sentences if self._is_passive(sent))
        passive_voice_ratio = passive_count / total_sentences

        # Common rhetorical devices
        devices = self._identify_rhetorical_devices(doc, text)

        # Paragraph analysis
        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
        para_lengths = []
        for para in paragraphs:
            para_doc = self.nlp(para)
            para_lengths.append(len(list(para_doc.sents)))

        para_avg = sum(para_lengths) / max(1, len(para_lengths))
        para_variance = sum((l - para_avg) ** 2 for l in para_lengths) / max(1, len(para_lengths))

        return ToneMarkers(
            assertiveness=assertiveness,
            hedging_frequency=hedging_frequency,
            rhetorical_questions=rhetorical_questions,
            imperatives=imperatives,
            passive_voice_ratio=passive_voice_ratio,
            common_rhetorical_devices=devices,
            paragraph_length_avg=para_avg,
            paragraph_length_variance=para_variance
        )

    def _is_passive(self, sent) -> bool:
        """Check if a sentence uses passive voice."""
        for token in sent:
            if token.dep_ == 'nsubjpass':
                return True
            if token.dep_ == 'auxpass':
                return True
        return False

    def _identify_rhetorical_devices(self, doc, text: str) -> List[str]:
        """Identify common rhetorical devices used."""
        devices = []

        # Check for parallel structure
        sentences = list(doc.sents)
        for i in range(len(sentences) - 1):
            s1_opener = self._classify_opener(sentences[i])
            s2_opener = self._classify_opener(sentences[i + 1])
            if s1_opener == s2_opener and s1_opener not in ('ARTICLE', 'PRONOUN'):
                devices.append('PARALLELISM')
                break

        # Check for repetition (anaphora)
        words = [t.text.lower() for t in doc if not t.is_punct]
        word_freq = Counter(words)
        if any(count > 5 for word, count in word_freq.most_common(10)
               if word not in ('the', 'a', 'an', 'is', 'are', 'was', 'were', 'be')):
            devices.append('REPETITION')

        # Check for contrast markers (antithesis)
        if any(marker in text.lower() for marker in ('but', 'however', 'yet', 'contrary')):
            devices.append('ANTITHESIS')

        # Check for questions (rhetorical)
        if '?' in text:
            devices.append('RHETORICAL_QUESTION')

        # Check for lists/enumeration
        if any(marker in text.lower() for marker in ('first', 'second', 'third', 'finally')):
            devices.append('ENUMERATION')

        # Check for examples
        if any(marker in text.lower() for marker in ('for example', 'for instance', 'such as')):
            devices.append('EXEMPLIFICATION')

        return list(set(devices))

    def _extract_example_sentences(self, doc) -> Dict[str, List[str]]:
        """Extract example sentences for each category."""
        examples = {
            'short': [],      # 5-10 words
            'medium': [],     # 11-25 words
            'long': [],       # 26+ words
            'opening': [],    # Good paragraph openers
            'declarative': [],
            'with_subordinate': []
        }

        sentences = list(doc.sents)

        for sent in sentences:
            text = sent.text.strip()
            word_count = len([t for t in sent if not t.is_punct and not t.is_space])

            # Categorize by length
            if 5 <= word_count <= 10 and len(examples['short']) < 5:
                examples['short'].append(text)
            elif 11 <= word_count <= 25 and len(examples['medium']) < 5:
                examples['medium'].append(text)
            elif word_count >= 26 and len(examples['long']) < 5:
                examples['long'].append(text)

            # Check for subordinate clause
            has_sub = any(t.dep_ in ('advcl', 'mark') for t in sent)
            if has_sub and len(examples['with_subordinate']) < 5:
                examples['with_subordinate'].append(text)
            elif not has_sub and len(examples['declarative']) < 5:
                examples['declarative'].append(text)

        # Get opening sentences from paragraphs
        text = doc.text
        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
        for para in paragraphs[:5]:
            para_doc = self.nlp(para)
            first_sent = next(para_doc.sents, None)
            if first_sent and len(examples['opening']) < 5:
                examples['opening'].append(first_sent.text.strip())

        return examples

    def _extract_phrasal_patterns(self, doc) -> List[PhrasalPattern]:
        """
        Extract distinctive multi-word patterns (n-grams) from the text.

        Focuses on:
        - 2-4 word sequences that appear 2+ times
        - Filters out generic patterns, keeps distinctive ones
        - Records position in sentence and example usage
        """
        sentences = list(doc.sents)
        if not sentences:
            return []

        # Generic phrases to filter out (too common to be distinctive)
        generic_phrases = {
            'of the', 'in the', 'to the', 'and the', 'on the', 'at the',
            'for the', 'with the', 'from the', 'by the', 'is the', 'was the',
            'it is', 'there is', 'there are', 'this is', 'that is', 'which is',
            'as a', 'in a', 'to a', 'of a', 'is a', 'was a', 'be a',
            'and a', 'such as', 'as well', 'well as', 'as well as',
            'one of', 'part of', 'all of', 'some of', 'most of', 'many of'
        }

        # Extract n-grams (2, 3, and 4 words)
        ngram_counts = defaultdict(list)  # phrase -> list of (sentence_text, position)

        for sent in sentences:
            tokens = [t for t in sent if not t.is_space]
            sent_text = sent.text.strip()
            sent_len = len(tokens)

            for n in range(2, 5):  # 2-grams to 4-grams
                for i in range(len(tokens) - n + 1):
                    ngram_tokens = tokens[i:i+n]

                    # Skip if contains only punctuation
                    if all(t.is_punct for t in ngram_tokens):
                        continue

                    phrase = ' '.join(t.text.lower() for t in ngram_tokens)

                    # Skip generic phrases
                    if phrase in generic_phrases:
                        continue

                    # Determine position in sentence
                    if i == 0:
                        position = 'opener'
                    elif i + n >= sent_len - 1:
                        position = 'closer'
                    else:
                        position = 'mid'

                    ngram_counts[phrase].append((sent_text, position))

        # Filter to phrases appearing 2+ times
        patterns = []
        for phrase, occurrences in ngram_counts.items():
            if len(occurrences) >= 2:
                # Get most common position
                positions = [pos for _, pos in occurrences]
                position_counts = Counter(positions)
                typical_position = position_counts.most_common(1)[0][0]

                # Get unique example sentences (up to 3)
                example_sents = list(dict.fromkeys(sent for sent, _ in occurrences))[:3]

                patterns.append(PhrasalPattern(
                    phrase=phrase,
                    position=typical_position,
                    frequency=len(occurrences),
                    example_sentences=example_sents
                ))

        # Sort by frequency (highest first)
        patterns.sort(key=lambda p: -p.frequency)

        # Return top patterns (limit to avoid overwhelming the LLM)
        return patterns[:30]

    def _extract_syntactic_constructions(self, doc, text: str) -> List[SyntacticConstruction]:
        """
        Extract reusable syntactic construction templates.

        Identifies:
        - Contrastive patterns: "not X but Y", "contrary to X, Y"
        - Definition patterns: "X is called Y because"
        - Enumeration patterns: "first... second... finally"
        - Causal patterns: "therefore", "consequently", "hence"
        """
        constructions = []
        sentences = list(doc.sents)
        text_lower = text.lower()

        # Pattern definitions with regex and slot info
        construction_patterns = [
            # Contrastive patterns
            {
                'type': 'contrastive',
                'patterns': [
                    (r'contrary to ([^,]+), ([^,]+) (holds?|does not|regards?|maintains?)',
                     "Contrary to {X}, {Y} {verb} that {Z}",
                     {'X': 'contrasted_position', 'Y': 'subject', 'Z': 'claim'}),
                    (r'not ([^,]+), but ([^,.]+)',
                     "not {X}, but {Y}",
                     {'X': 'negated', 'Y': 'affirmed'}),
                    (r"doesn't mean that ([^,]+), but that ([^,.]+)",
                     "This doesn't mean that {X}, but that {Y}",
                     {'X': 'misconception', 'Y': 'truth'}),
                    (r'(this|that|it), however, does not mean',
                     "{subject}, however, does not mean that {X}",
                     {'subject': 'reference', 'X': 'misconception'}),
                ]
            },
            # Definition patterns
            {
                'type': 'definition',
                'patterns': [
                    (r'is called ([^\s]+) because',
                     "{X} is called {Y} because {reason}",
                     {'X': 'subject', 'Y': 'name', 'reason': 'explanation'}),
                    (r'([^\s]+) is the ([^,]+) of',
                     "{X} is the {Y} of {Z}",
                     {'X': 'term', 'Y': 'category', 'Z': 'domain'}),
                    (r'we (call|term|name) this ([^\s,]+)',
                     "We {verb} this {X}",
                     {'verb': 'naming_verb', 'X': 'name'}),
                ]
            },
            # Enumeration patterns
            {
                'type': 'enumeration',
                'patterns': [
                    (r'(firstly?|first of all)[,:]?.*?(secondly?|second)[,:]?',
                     "First, {X}; second, {Y}; ...",
                     {'X': 'first_point', 'Y': 'second_point'}),
                    (r'(a\)|1\)|\(a\)|\(1\)).*?(b\)|2\)|\(b\)|\(2\))',
                     "(a) {X}; (b) {Y}; ...",
                     {'X': 'first_item', 'Y': 'second_item'}),
                    (r'the (first|principal|main) features? (of|are)',
                     "The {ordinal} feature(s) of {X} are: {list}",
                     {'ordinal': 'order', 'X': 'subject', 'list': 'features'}),
                ]
            },
            # Causal/consequential patterns
            {
                'type': 'causal',
                'patterns': [
                    (r'the ([^\s]+) method therefore (holds?|requires?|regards?)',
                     "The {X} method therefore {verb} that {Y}",
                     {'X': 'method_name', 'verb': 'action', 'Y': 'conclusion'}),
                    (r'(hence|thus|therefore|consequently)[,]? ([^.]+)',
                     "{connector}, {conclusion}",
                     {'connector': 'causal_word', 'conclusion': 'result'}),
                    (r'it follows that ([^.]+)',
                     "It follows that {X}",
                     {'X': 'conclusion'}),
                    (r'from this it follows',
                     "From this it follows that {X}",
                     {'X': 'conclusion'}),
                ]
            },
            # Explanatory patterns
            {
                'type': 'explanatory',
                'patterns': [
                    (r'(this|it) means that ([^.]+)',
                     "{subject} means that {X}",
                     {'subject': 'reference', 'X': 'explanation'}),
                    (r'in other words[,]? ([^.]+)',
                     "In other words, {X}",
                     {'X': 'rephrased'}),
                    (r'that is to say[,]? ([^.]+)',
                     "That is to say, {X}",
                     {'X': 'clarification'}),
                ]
            },
            # Quotation/citation patterns
            {
                'type': 'quotation',
                'patterns': [
                    (r'"([^"]+)" says ([^\s,]+)',
                     '"{quote}" says {author}',
                     {'quote': 'quoted_text', 'author': 'speaker'}),
                    (r'as ([^\s]+) (says?|puts it|points out|notes)',
                     "As {author} {verb}, {quote}",
                     {'author': 'speaker', 'verb': 'speech_act', 'quote': 'content'}),
                    (r'(marx|engels|lenin|hegel) (says?|wrote|declared)',
                     "{author} {verb}: \"{quote}\"",
                     {'author': 'speaker', 'verb': 'speech_act', 'quote': 'content'}),
                ]
            }
        ]

        # Find matches for each pattern type
        for pattern_group in construction_patterns:
            construction_type = pattern_group['type']

            for regex, template, slots in pattern_group['patterns']:
                matches = list(re.finditer(regex, text_lower, re.IGNORECASE))

                if matches:
                    # Find example sentences containing these matches
                    examples = []
                    for match in matches[:5]:
                        match_start = match.start()
                        match_end = match.end()

                        # Find the sentence containing this match
                        for sent in sentences:
                            sent_start = sent.start_char
                            sent_end = sent.end_char

                            if sent_start <= match_start and match_end <= sent_end:
                                examples.append(sent.text.strip())
                                break

                    if examples:
                        constructions.append(SyntacticConstruction(
                            pattern=template,
                            construction_type=construction_type,
                            slots=slots,
                            examples=list(dict.fromkeys(examples))[:3]  # Unique, max 3
                        ))

        # Deduplicate by template pattern
        seen_patterns = set()
        unique_constructions = []
        for c in constructions:
            if c.pattern not in seen_patterns:
                seen_patterns.add(c.pattern)
                unique_constructions.append(c)

        return unique_constructions

    def _map_discourse_markers(self, doc, text: str) -> List[DiscourseMarkerUsage]:
        """
        Map discourse marker usage patterns.

        Catalogs:
        - Which markers are used: therefore, hence, thus, however, moreover, etc.
        - Their typical sentence position
        - Frequency per 100 sentences
        - Surrounding context examples
        """
        sentences = list(doc.sents)
        total_sentences = len(sentences)

        if total_sentences == 0:
            return []

        # Expanded discourse marker categories
        discourse_markers = {
            # Causal/Consequential
            'therefore': 'causal',
            'hence': 'causal',
            'thus': 'causal',
            'consequently': 'causal',
            'accordingly': 'causal',
            'as a result': 'causal',

            # Contrastive
            'however': 'contrastive',
            'nevertheless': 'contrastive',
            'nonetheless': 'contrastive',
            'on the other hand': 'contrastive',
            'in contrast': 'contrastive',
            'contrary to': 'contrastive',

            # Additive
            'moreover': 'additive',
            'furthermore': 'additive',
            'in addition': 'additive',
            'likewise': 'additive',
            'similarly': 'additive',

            # Elaborative
            'in other words': 'elaborative',
            'that is to say': 'elaborative',
            'that is': 'elaborative',
            'namely': 'elaborative',
            'specifically': 'elaborative',

            # Sequential
            'first': 'sequential',
            'firstly': 'sequential',
            'second': 'sequential',
            'secondly': 'sequential',
            'finally': 'sequential',
            'subsequently': 'sequential',
            'then': 'sequential',

            # Emphatic
            'indeed': 'emphatic',
            'in fact': 'emphatic',
            'certainly': 'emphatic',
            'clearly': 'emphatic',
            'undoubtedly': 'emphatic',

            # Exemplifying
            'for example': 'exemplifying',
            'for instance': 'exemplifying',
            'such as': 'exemplifying',

            # Concessive
            'although': 'concessive',
            'though': 'concessive',
            'even though': 'concessive',
            'while': 'concessive',
            'whereas': 'concessive',
        }

        marker_usages = []

        for marker, category in discourse_markers.items():
            # Find all occurrences in text
            occurrences = []
            text_lower = text.lower()

            # Use word boundary aware search
            pattern = r'\b' + re.escape(marker) + r'\b'
            matches = list(re.finditer(pattern, text_lower))

            if not matches:
                continue

            # For each match, find position and context
            positions = []
            example_contexts = []

            for match in matches:
                match_start = match.start()

                # Find containing sentence
                for sent in sentences:
                    sent_start = sent.start_char
                    sent_end = sent.end_char
                    sent_text = sent.text.strip()
                    sent_text_lower = sent_text.lower()

                    if sent_start <= match_start < sent_end:
                        # Determine position within sentence
                        marker_pos_in_sent = match_start - sent_start

                        if marker_pos_in_sent < 3:  # At or near start
                            positions.append('sentence_start')
                        elif sent_text_lower.find(marker) > 0:
                            # Check if after comma (parenthetical)
                            before_marker = sent_text_lower[:sent_text_lower.find(marker)]
                            if before_marker.rstrip().endswith(','):
                                positions.append('parenthetical')
                            else:
                                positions.append('clause_start')
                        else:
                            positions.append('mid_sentence')

                        if len(example_contexts) < 3:
                            example_contexts.append(sent_text)
                        break

            if positions:
                # Determine typical position
                position_counts = Counter(positions)
                typical_position = position_counts.most_common(1)[0][0]

                # Calculate frequency per 100 sentences
                freq_per_100 = (len(matches) / total_sentences) * 100

                marker_usages.append(DiscourseMarkerUsage(
                    marker=marker,
                    typical_position=typical_position,
                    frequency_per_100_sentences=round(freq_per_100, 2),
                    example_contexts=list(dict.fromkeys(example_contexts))[:3]
                ))

        # Sort by frequency (highest first)
        marker_usages.sort(key=lambda m: -m.frequency_per_100_sentences)

        return marker_usages


def analyze_style(text: str) -> StyleProfile:
    """Convenience function to analyze text style."""
    analyzer = StyleAnalyzer()
    return analyzer.analyze(text)


# Test function
if __name__ == '__main__':
    # Read sample text
    from pathlib import Path

    sample_path = Path(__file__).parent / "prompts" / "sample.txt"
    if sample_path.exists():
        with open(sample_path, 'r', encoding='utf-8') as f:
            sample_text = f.read()

        print("Analyzing sample text style...")
        analyzer = StyleAnalyzer()
        profile = analyzer.analyze(sample_text)

        print("\n=== Style Profile ===\n")
        print(profile.to_style_guide())
    else:
        print(f"Sample file not found at {sample_path}")

        # Use test text
        test_text = """Dialectical materialism is the world outlook of the Marxist-Leninist party. It is called dialectical materialism because its approach to the phenomena of nature, its method of studying and apprehending them, is dialectical, while its interpretation of the phenomena of nature, its conception of these phenomena, its theory, is materialistic.

Historical materialism is the extension of the principles of dialectical materialism to the study of social life. When describing their dialectical method, Marx and Engels usually refer to Hegel as the philosopher who formulated the main features of dialectics. This, however, does not mean that the dialectics of Marx and Engels is identical with the dialectics of Hegel."""

        print("Using test text...")
        analyzer = StyleAnalyzer()
        profile = analyzer.analyze(test_text)

        print("\n=== Style Profile ===\n")
        print(profile.to_style_guide())

