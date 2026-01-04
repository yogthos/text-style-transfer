"""Repetition reducer for post-processing generated text.

Lightweight post-processing that:
- Tracks word usage across the document
- Replaces overused words with synonyms
- Preserves entities and proper nouns

This is a simplified replacement for VocabularyTransformer that doesn't
require a style profile, making it compatible with the LoRA pipeline.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Set, Optional, Tuple
from collections import Counter

from ..utils.nlp import get_nlp
from ..utils.logging import get_logger

logger = get_logger(__name__)


# Common LLM-speak words to watch for
LLM_SPEAK = {
    # Corporate/tech jargon
    "utilize": "use",
    "utilization": "use",
    "implementation": "setup",
    "functionality": "feature",
    "leverage": "use",
    "facilitate": "help",
    "comprehensive": "full",
    "robust": "strong",
    "scalable": "flexible",
    "streamline": "simplify",
    "optimize": "improve",
    "enhance": "improve",
    "innovative": "new",
    "cutting-edge": "modern",
    "state-of-the-art": "latest",
    "paradigm": "model",
    "synergy": "cooperation",
    "holistic": "complete",
    "proactive": "active",
    "impactful": "effective",
    "actionable": "practical",
    "deliverables": "results",
    "stakeholders": "people involved",
    "bandwidth": "capacity",
    "circle back": "return to",
    "deep dive": "detailed look",
    "moving forward": "next",
    # Mechanical precision markers (AI tells)
    "specifically": "",  # Often unnecessary
    "particularly": "",
    "fundamentally": "",
    "essentially": "",
    "inherently": "",
    "intrinsically": "",
    "ultimately": "in the end",
    "subsequently": "then",
    "consequently": "so",
    "additionally": "also",
    "furthermore": "also",
    "moreover": "also",
    "nevertheless": "but",
    "nonetheless": "still",
    "notwithstanding": "despite",
    "accordingly": "so",
    "thereby": "so",
    "whereby": "where",
    "wherein": "where",
    "thereof": "of it",
    "therein": "in it",
    # Hedging/filler phrases
    "it is important to note that": "",
    "it should be noted that": "",
    "it is worth mentioning that": "",
    "it bears mentioning that": "",
    "it goes without saying that": "",
    "needless to say": "",
    "as previously mentioned": "",
    "as noted earlier": "",
    "as discussed above": "",
    "in this context": "",
    "in this regard": "",
    "in light of": "given",
    "with respect to": "about",
    "in terms of": "for",
    "in order to": "to",
    "due to the fact that": "because",
    "for the purpose of": "to",
    "in the event that": "if",
    "at this point in time": "now",
    "at the present time": "now",
    "prior to": "before",
    "subsequent to": "after",
    # Robotic formality
    "commence": "start",
    "terminate": "end",
    "endeavor": "try",
    "ascertain": "find out",
    "constitutes": "is",
    "demonstrates": "shows",
    "indicates": "shows",
    "necessitates": "needs",
    "encompasses": "includes",
    "pertains to": "relates to",
    "manifests": "shows",
    "exhibits": "shows",
    "exemplifies": "shows",
    "elucidates": "explains",
    "delineates": "describes",
    "underscores": "highlights",
    "illuminates": "shows",
    # Fix weird Qwen vocabulary substitutions
    "ticker": "watch",
    "vigil": "watch",
    "lookout": "watch",
    "sentinel": "watch",
    "sentry": "watch",
    "picket": "watch",
    "spotter": "watch",
    "scout": "watch",
    "timekeeper": "watch",
    "cogwheel": "gear",
    "gearing": "gear",
    "geartrain": "gears",
    "paraphernalia": "gear",
    "appurtenance": "part",
    "unmarried": "single",
    "undivided": "single",
    "exclusive": "single",
    "unharmed": "whole",
    "unhurt": "whole",
    "unscathed": "whole",
    "peregrine": "mobile",
    "roving": "mobile",
    "wandering": "mobile",
    "nomadic": "mobile",
    "earphone": "phone",
    "earpiece": "phone",
    "headphone": "phone",
    "telephony": "phone",
    "macrocosm": "world",
    "cosmos": "universe",
    "creation": "world",
    "existence": "world",
    "domain": "world",
    "corporeal": "physical",
    "at the end of the day": "ultimately",
    "think outside the box": "be creative",
}


@dataclass
class ReductionStats:
    """Statistics from repetition reduction."""
    words_checked: int = 0
    replacements_made: int = 0
    overused_words: List[str] = field(default_factory=list)
    replacements_detail: Dict[str, str] = field(default_factory=dict)
    sentence_length_variance: float = 0.0  # Higher = more varied = more human
    repeated_starters: List[str] = field(default_factory=list)  # Sentences starting same way


class RepetitionReducer:
    """Reduce word repetition in generated text.

    Tracks word usage across the document and replaces words that
    appear too frequently with synonyms from WordNet or a simple
    synonym list.

    Usage:
        reducer = RepetitionReducer(threshold=3)

        # Process each paragraph
        for para in paragraphs:
            para, stats = reducer.reduce(para)

        # Reset between documents
        reducer.reset()
    """

    def __init__(
        self,
        threshold: int = 3,
        use_wordnet: bool = True,
        synonym_replacement: bool = False,  # Disable by default - causes problems
    ):
        """Initialize the reducer.

        Args:
            threshold: Number of uses before a word is considered overused.
            use_wordnet: Whether to use WordNet for synonyms.
            synonym_replacement: Whether to replace overused words with synonyms.
                               Defaults to False as this often introduces weird vocabulary.
        """
        self.threshold = threshold
        self.use_wordnet = use_wordnet
        self.synonym_replacement = synonym_replacement
        self._nlp = None
        self._wordnet = None

        # Track word usage across document
        self.word_counts: Counter = Counter()
        self.used_replacements: Set[str] = set()

        # Simple synonym cache
        self._synonym_cache: Dict[str, List[str]] = {}

    @property
    def nlp(self):
        """Lazy load spaCy model."""
        if self._nlp is None:
            self._nlp = get_nlp()
        return self._nlp

    @property
    def wordnet(self):
        """Lazy load WordNet."""
        if self._wordnet is None and self.use_wordnet:
            try:
                from nltk.corpus import wordnet
                self._wordnet = wordnet
            except ImportError:
                logger.warning("WordNet not available, using simple synonyms")
                self.use_wordnet = False
        return self._wordnet

    def reduce(self, text: str) -> Tuple[str, ReductionStats]:
        """Reduce repetition in text.

        Args:
            text: Text to process.

        Returns:
            Tuple of (processed_text, stats).
        """
        stats = ReductionStats()

        # First: Replace multi-word AI phrases (case-insensitive)
        text = self._replace_ai_phrases(text, stats)

        doc = self.nlp(text)

        # First pass: count words
        for token in doc:
            if self._should_track(token):
                word_lower = token.lemma_.lower()
                self.word_counts[word_lower] += 1

        # Second pass: collect replacements
        replacements = []

        for token in doc:
            stats.words_checked += 1

            # Check for LLM-speak FIRST - bypass _should_track for these
            if token.text.lower() in LLM_SPEAK:
                replacement = LLM_SPEAK[token.text.lower()]
                # Always fix vocabulary issues
                replacements.append((token.idx, token.text, self._match_case(replacement, token.text)))
                stats.replacements_detail[token.text] = replacement
                continue

            if not self._should_track(token):
                continue

            # Only do synonym replacement if enabled (disabled by default)
            if not self.synonym_replacement:
                continue

            word_lower = token.lemma_.lower()

            # Check if overused
            if self.word_counts[word_lower] >= self.threshold:
                synonym = self._get_synonym(token)
                if synonym and synonym not in self.used_replacements:
                    replacements.append((token.idx, token.text, self._match_case(synonym, token.text)))
                    self.used_replacements.add(synonym)
                    stats.overused_words.append(token.text)
                    stats.replacements_detail[token.text] = synonym

        # Apply replacements in reverse order
        result = text
        for pos, original, replacement in sorted(replacements, key=lambda x: -x[0]):
            result = result[:pos] + replacement + result[pos + len(original):]
            stats.replacements_made += 1

        if stats.replacements_made > 0:
            logger.debug(f"Reduced {stats.replacements_made} repetitions: {stats.replacements_detail}")

        # Analyze sentence variety for diagnostics
        self.analyze_sentence_variety(result, stats)

        return result, stats

    def _should_track(self, token) -> bool:
        """Check if token should be tracked for repetition."""
        # Skip entities, proper nouns, punctuation, short words
        if token.ent_type_:
            return False
        if token.pos_ in ('PROPN', 'PUNCT', 'SPACE', 'NUM'):
            return False
        if not token.is_alpha or len(token.text) < 4:
            return False
        # Only track content words (nouns, verbs, adjectives, adverbs)
        if token.pos_ not in ('NOUN', 'VERB', 'ADJ', 'ADV'):
            return False
        return True

    def _get_synonym(self, token) -> Optional[str]:
        """Get a synonym for the token."""
        word = token.lemma_.lower()

        # Check cache first
        if word in self._synonym_cache:
            synonyms = self._synonym_cache[word]
            for syn in synonyms:
                # Skip LLM_SPEAK keys (problematic words we're trying to fix)
                if syn.lower() in LLM_SPEAK:
                    continue
                if syn not in self.used_replacements and syn != word:
                    return syn
            return None

        # Get synonyms from WordNet
        synonyms = []
        if self.wordnet:
            try:
                pos_map = {
                    'NOUN': 'n',
                    'VERB': 'v',
                    'ADJ': 'a',
                    'ADV': 'r',
                }
                wn_pos = pos_map.get(token.pos_)

                if wn_pos:
                    for synset in self.wordnet.synsets(word, pos=wn_pos):
                        for lemma in synset.lemmas():
                            name = lemma.name().replace('_', ' ')
                            if name != word and name.isalpha():
                                # Skip synonyms that are LLM_SPEAK keys (problematic words)
                                if name.lower() not in LLM_SPEAK:
                                    synonyms.append(name)
            except Exception as e:
                logger.debug(f"WordNet lookup failed for {word}: {e}")

        # Cache and return
        self._synonym_cache[word] = synonyms[:5]  # Keep top 5

        for syn in synonyms:
            # Skip LLM_SPEAK keys (problematic words we're trying to fix)
            if syn.lower() in LLM_SPEAK:
                continue
            if syn not in self.used_replacements:
                return syn
        return None

    def _match_case(self, replacement: str, original: str) -> str:
        """Match the case of the replacement to the original."""
        if not replacement or not original:
            return replacement
        if original.isupper():
            return replacement.upper()
        elif original[0].isupper():
            return replacement.capitalize()
        return replacement.lower()

    def _replace_ai_phrases(self, text: str, stats: ReductionStats) -> str:
        """Replace multi-word AI phrases with simpler alternatives.

        Handles phrases like "in order to" -> "to", case-insensitively.
        """
        import re

        # Multi-word phrases (sorted by length, longest first)
        phrases = sorted(
            [(k, v) for k, v in LLM_SPEAK.items() if ' ' in k],
            key=lambda x: -len(x[0])
        )

        for phrase, replacement in phrases:
            # Case-insensitive replacement
            pattern = re.compile(re.escape(phrase), re.IGNORECASE)
            matches = pattern.findall(text)
            if matches:
                for match in matches:
                    # Match case of first letter
                    if replacement:
                        repl = self._match_case(replacement, match)
                    else:
                        repl = ""
                    text = text.replace(match, repl, 1)
                    stats.replacements_made += 1
                    stats.replacements_detail[match] = repl if repl else "(removed)"

        # Sentence-level AI pattern fixes
        text = self._fix_sentence_patterns(text, stats)

        # Replace AI-favored punctuation
        text = self._simplify_punctuation(text, stats)

        # Clean up double spaces from removals
        text = re.sub(r'  +', ' ', text)
        # Clean up space before punctuation
        text = re.sub(r' ([.,;:!?])', r'\1', text)
        # Fix sentences starting with lowercase after removal
        text = re.sub(r'([.!?]\s+)([a-z])', lambda m: m.group(1) + m.group(2).upper(), text)

        return text

    def _simplify_punctuation(self, text: str, stats: ReductionStats) -> str:
        """Replace AI-favored punctuation with simpler alternatives.

        Em dashes, semicolons, and colons are overrepresented in AI text.
        Convert to periods and commas for more natural flow.
        """
        import re

        changes = 0

        # Em dash (—) handling
        # Parenthetical em dashes -> commas: "The system—which is complex—works"
        em_paren = re.findall(r'—([^—]{1,50})—', text)
        for match in em_paren:
            text = text.replace(f'—{match}—', f', {match}, ', 1)
            changes += 1

        # Em dash before clause -> period: "This is key—it matters"
        text, n = re.subn(r'—\s*(it|this|that|they|we|he|she|the|a|an)\b',
                         lambda m: '. ' + m.group(1).capitalize(), text, flags=re.IGNORECASE)
        changes += n

        # Remaining em dashes -> comma
        text, n = re.subn(r'—', ', ', text)
        changes += n

        # Semicolon handling - only between independent clauses, not in lists
        # List pattern: "X; Y; and Z" or "X; Y; Z" - keep as commas
        text, n = re.subn(r';\s+(and|or)\s+', r', \1 ', text, flags=re.IGNORECASE)
        changes += n

        # Semicolon followed by lowercase (likely list item) -> comma
        text, n = re.subn(r';\s+([a-z])', r', \1', text)
        changes += n

        # Semicolon followed by capital (independent clause) -> period
        text, n = re.subn(r';\s+([A-Z])', r'. \1', text)
        changes += n

        # Colon handling - only for explanation patterns, not lists
        # "This is key: it affects" -> "This is key. It affects"
        text, n = re.subn(r':\s+(it|this|that|they|we|he|she)\b',
                         lambda m: '. ' + m.group(1).capitalize(), text, flags=re.IGNORECASE)
        changes += n

        if changes > 0:
            stats.replacements_made += changes
            stats.replacements_detail["punctuation simplified"] = f"({changes} changes)"

        # Clean up spacing issues
        text = re.sub(r'\s*,\s*', ', ', text)  # Normalize comma spacing
        text = re.sub(r',\s*,', ',', text)  # Remove double commas
        text = re.sub(r'\s+', ' ', text)  # Normalize spaces
        text = re.sub(r'\s+\.', '.', text)  # No space before period

        return text

    def _fix_sentence_patterns(self, text: str, stats: ReductionStats) -> str:
        """Fix sentence-level AI patterns using linguistic analysis.

        Uses spaCy dependency parsing to detect structural patterns:
        - Impersonal "It [be] [ADJ] that..." constructions (expletive subjects)
        - Sentence-initial conjunctive adverbs (formulaic transitions)
        - Heavy nominalization (abstract -tion/-ment nouns as subjects)
        """
        doc = self.nlp(text)
        replacements = []  # (start, end, replacement_text)

        for sent in doc.sents:
            tokens = list(sent)
            if len(tokens) < 3:
                continue

            sent_start = sent.start_char
            sent_text = sent.text

            # Pattern 1: Expletive "it" constructions - "It is [ADJ] that..."
            # Linguistic pattern: pronoun "it" as subject + copula + that-clause
            # This is an impersonal hedge common in formal/AI writing
            first_token = tokens[0]
            if (first_token.text.lower() == "it" and
                first_token.dep_ in ("nsubj", "expl") and
                first_token.head.lemma_ == "be"):
                # Look for subordinating "that"
                for tok in tokens[2:8]:
                    if tok.dep_ == "mark" and tok.text.lower() == "that":
                        # Remove everything up to and including "that"
                        that_end = tok.idx + len(tok.text) + 1 - sent_start
                        original = sent_text[:that_end].strip()
                        replacements.append((sent.start_char, sent.start_char + that_end, ""))
                        stats.replacements_made += 1
                        stats.replacements_detail[original] = "(expletive removed)"
                        break

            # Pattern 2: Sentence-initial conjunctive adverbs
            # Detected by: first token is ADV with "advmod" to root, and it's a formal connector
            # These create mechanical flow: "Furthermore, X. Moreover, Y. Additionally, Z."
            if (first_token.pos_ == "ADV" and
                first_token.dep_ == "advmod" and
                len(first_token.text) > 6):  # Longer adverbs tend to be formal
                # Check if it's followed by comma (formulaic pattern)
                if len(tokens) > 1 and tokens[1].text == ",":
                    # Flag as formulaic transition
                    stats.replacements_detail[f"adv: {first_token.text}"] = "(formulaic transition)"

            # Pattern 3: Abstract nominalization as subject
            # Detected by: subject ends in -tion/-ment/-ness and is abstract
            for tok in tokens[:5]:  # Check first few tokens
                if tok.dep_ == "nsubj" and tok.pos_ == "NOUN":
                    word = tok.text.lower()
                    # Check for nominalization suffixes (created from verbs)
                    if (word.endswith(("tion", "ment", "ness", "ity", "ance", "ence")) and
                        len(word) > 8):  # Longer = more formal
                        stats.replacements_detail[f"nominalization: {tok.text}"] = "(flagged)"

            # Pattern 4: Passive voice density check
            # Too many passives in a row signals robotic writing
            passive_count = sum(1 for t in tokens if t.dep_ == "auxpass")
            if passive_count > 1:
                stats.replacements_detail["multiple passives"] = f"({passive_count} in sentence)"

        # Apply replacements in reverse order
        result = text
        for start, end, repl in sorted(replacements, key=lambda x: -x[0]):
            result = result[:start] + repl + result[end:]

        # Final cleanup: capitalize after removals
        import re
        result = re.sub(r'([.!?]\s*)([a-z])', lambda m: m.group(1) + m.group(2).upper(), result)
        result = re.sub(r'^\s*([a-z])', lambda m: m.group(1).upper(), result)

        return result

    def reset(self):
        """Reset word tracking between documents."""
        self.word_counts.clear()
        self.used_replacements.clear()

    def get_overused_words(self, limit: int = 10) -> List[Tuple[str, int]]:
        """Get the most overused words."""
        return [
            (word, count)
            for word, count in self.word_counts.most_common(limit)
            if count >= self.threshold
        ]

    def analyze_sentence_variety(self, text: str, stats: ReductionStats) -> None:
        """Analyze sentence variety and detect robotic patterns.

        Updates stats with:
        - sentence_length_variance: coefficient of variation (higher = more varied)
        - repeated_starters: sentences that start the same way
        """
        doc = self.nlp(text)
        sentences = list(doc.sents)

        if len(sentences) < 2:
            return

        # Calculate sentence length variance
        lengths = [len(list(sent)) for sent in sentences]
        mean_len = sum(lengths) / len(lengths)
        if mean_len > 0:
            variance = sum((l - mean_len) ** 2 for l in lengths) / len(lengths)
            std_dev = variance ** 0.5
            stats.sentence_length_variance = std_dev / mean_len  # CV
        else:
            stats.sentence_length_variance = 0.0

        # Detect repeated sentence starters (first 2-3 words)
        starters = Counter()
        for sent in sentences:
            tokens = [t.text.lower() for t in sent if not t.is_space][:3]
            if len(tokens) >= 2:
                starter = ' '.join(tokens[:2])
                starters[starter] += 1

        # Report starters used 2+ times
        stats.repeated_starters = [
            f"'{starter}' ({count}x)"
            for starter, count in starters.most_common()
            if count >= 2
        ]

        if stats.repeated_starters:
            logger.debug(f"Repeated sentence starters: {stats.repeated_starters}")
