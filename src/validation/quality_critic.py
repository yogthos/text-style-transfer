"""Quality critic for generated text.

Provides explicit, actionable feedback for fixing quality issues:
- Incomplete sentences
- Word clustering (overused adjectives, qualifiers)
- Missing content from source
- Hallucinated content not in source
"""

from dataclasses import dataclass, field
from typing import List, Dict, Set, Tuple, Optional
from collections import Counter

from ..utils.nlp import get_nlp, split_into_sentences
from ..utils.logging import get_logger
from ..utils.prompts import format_prompt

logger = get_logger(__name__)


# Words that are fine to repeat (function words, common verbs)
IGNORE_WORDS = {
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "must", "shall", "can", "need", "dare",
    "this", "that", "these", "those", "it", "its", "they", "their",
    "we", "our", "you", "your", "he", "his", "she", "her", "i", "my",
    "and", "or", "but", "if", "then", "because", "as", "of", "in", "on",
    "at", "to", "for", "with", "by", "from", "not", "no", "so", "such",
    "which", "who", "whom", "what", "when", "where", "how", "why",
    "all", "each", "every", "both", "few", "more", "most", "other",
    "some", "any", "one", "two", "first", "new", "also", "just", "only",
}


@dataclass
class QualityIssue:
    """A specific quality issue with fix instruction."""

    issue_type: str  # "incomplete_sentence", "word_cluster", "missing_content", "hallucination"
    severity: str  # "critical", "warning", "info"
    description: str  # Human-readable description
    fix_instruction: str  # Explicit instruction for fixing
    location: Optional[str] = None  # Sentence or word where issue occurs


@dataclass
class QualityCritique:
    """Complete quality critique with issues and fix instructions."""

    issues: List[QualityIssue] = field(default_factory=list)
    word_clusters: Dict[str, int] = field(default_factory=dict)  # word -> count
    missing_entities: List[str] = field(default_factory=list)
    incomplete_sentences: List[str] = field(default_factory=list)

    @property
    def has_critical_issues(self) -> bool:
        return any(i.severity == "critical" for i in self.issues)

    @property
    def fix_instructions(self) -> List[str]:
        """Get all fix instructions as a list."""
        return [i.fix_instruction for i in self.issues if i.fix_instruction]

    def to_repair_prompt(self) -> str:
        """Generate a repair prompt from the issues."""
        if not self.issues:
            return ""

        lines = ["Fix the following issues:"]
        for i, issue in enumerate(self.issues, 1):
            lines.append(f"{i}. {issue.fix_instruction}")

        return "\n".join(lines)


class QualityCritic:
    """Critic that analyzes generated text and provides explicit fix instructions.

    Uses spaCy for:
    - Detecting word clustering (overused adjectives, adverbs, qualifiers)
    - Extracting entities for coverage checking
    - Detecting incomplete sentences
    """

    def __init__(
        self,
        cluster_threshold: int = 3,  # Words used 3+ times trigger warning
        entity_coverage_threshold: float = 0.8,  # 80% entity coverage required
    ):
        self.cluster_threshold = cluster_threshold
        self.entity_coverage_threshold = entity_coverage_threshold
        self._nlp = None

    @property
    def nlp(self):
        """Lazy load spaCy model."""
        if self._nlp is None:
            self._nlp = get_nlp()
        return self._nlp

    def critique(
        self,
        source: str,
        generated: str,
    ) -> QualityCritique:
        """Analyze generated text and produce critique with fix instructions.

        Args:
            source: Original source text.
            generated: Generated/transformed text.

        Returns:
            QualityCritique with issues and fix instructions.
        """
        critique = QualityCritique()

        # Parse both texts
        source_doc = self.nlp(source)
        gen_doc = self.nlp(generated)

        # Check for incomplete sentences
        self._check_incomplete_sentences(generated, gen_doc, critique)

        # Check for word clustering
        self._check_word_clustering(gen_doc, critique)

        # Check entity coverage
        self._check_entity_coverage(source_doc, gen_doc, critique)

        # Check for hallucinated content
        self._check_hallucinations(source_doc, gen_doc, critique)

        # Check for truncation (generated much shorter than source)
        self._check_truncation(source, generated, critique)

        return critique

    def _check_incomplete_sentences(
        self,
        text: str,
        doc,
        critique: QualityCritique,
    ):
        """Detect sentences that appear incomplete."""
        sentences = list(doc.sents)

        for sent in sentences:
            sent_text = sent.text.strip()

            # Check for sentences ending without proper punctuation
            if sent_text and not sent_text[-1] in '.!?"\'':
                # Could be incomplete
                words = [t for t in sent if t.is_alpha]
                if len(words) >= 3:  # Only flag if it's a substantial fragment
                    critique.incomplete_sentences.append(sent_text)
                    critique.issues.append(QualityIssue(
                        issue_type="incomplete_sentence",
                        severity="critical",
                        description=f"Sentence appears incomplete: '{sent_text[-50:]}'",
                        fix_instruction=f"Complete the sentence ending with: '...{sent_text[-30:]}'",
                        location=sent_text,
                    ))

            # Check for sentences ending with prepositions or articles (likely truncated)
            tokens = [t for t in sent if not t.is_space]
            if tokens:
                last_token = tokens[-1]
                if last_token.pos_ in ('ADP', 'DET', 'CCONJ') and last_token.text not in '.!?':
                    critique.incomplete_sentences.append(sent_text)
                    critique.issues.append(QualityIssue(
                        issue_type="incomplete_sentence",
                        severity="critical",
                        description=f"Sentence ends with '{last_token.text}' (likely truncated)",
                        fix_instruction=f"Complete the sentence - it ends with '{last_token.text}' which suggests truncation",
                        location=sent_text,
                    ))

    def _check_word_clustering(
        self,
        doc,
        critique: QualityCritique,
    ):
        """Detect overused adjectives, adverbs, and qualifiers."""
        # Track content words by POS
        adj_counts = Counter()
        adv_counts = Counter()
        noun_counts = Counter()
        verb_counts = Counter()

        for token in doc:
            lemma = token.lemma_.lower()

            # Skip common/function words
            if lemma in IGNORE_WORDS:
                continue
            if len(lemma) < 4:
                continue
            if not token.is_alpha:
                continue

            if token.pos_ == 'ADJ':
                adj_counts[lemma] += 1
            elif token.pos_ == 'ADV':
                adv_counts[lemma] += 1
            elif token.pos_ == 'NOUN':
                noun_counts[lemma] += 1
            elif token.pos_ == 'VERB':
                verb_counts[lemma] += 1

        # Check for clustering
        all_clusters = []

        for lemma, count in adj_counts.items():
            if count >= self.cluster_threshold:
                all_clusters.append((lemma, count, 'adjective'))
                critique.word_clusters[lemma] = count

        for lemma, count in adv_counts.items():
            if count >= self.cluster_threshold:
                all_clusters.append((lemma, count, 'adverb'))
                critique.word_clusters[lemma] = count

        # Nouns and verbs have higher threshold (4+)
        for lemma, count in noun_counts.items():
            if count >= self.cluster_threshold + 1:
                all_clusters.append((lemma, count, 'noun'))
                critique.word_clusters[lemma] = count

        for lemma, count in verb_counts.items():
            if count >= self.cluster_threshold + 1:
                all_clusters.append((lemma, count, 'verb'))
                critique.word_clusters[lemma] = count

        # Generate fix instructions for clusters
        for lemma, count, pos in all_clusters:
            critique.issues.append(QualityIssue(
                issue_type="word_cluster",
                severity="warning",
                description=f"'{lemma}' ({pos}) used {count} times",
                fix_instruction=f"Vary vocabulary: replace some instances of '{lemma}' with synonyms or rephrase",
                location=lemma,
            ))

    def _check_entity_coverage(
        self,
        source_doc,
        gen_doc,
        critique: QualityCritique,
    ):
        """Check if key entities from source are preserved."""
        # Extract named entities
        source_entities = set()
        for ent in source_doc.ents:
            if ent.label_ in ('PERSON', 'ORG', 'GPE', 'LOC', 'PRODUCT', 'EVENT'):
                source_entities.add(ent.text.lower())

        gen_text_lower = gen_doc.text.lower()

        missing = []
        for entity in source_entities:
            if entity not in gen_text_lower:
                missing.append(entity)

        if missing:
            critique.missing_entities = missing
            coverage = 1 - (len(missing) / len(source_entities)) if source_entities else 1.0

            if coverage < self.entity_coverage_threshold:
                critique.issues.append(QualityIssue(
                    issue_type="missing_content",
                    severity="critical",
                    description=f"Missing entities: {', '.join(missing)}",
                    fix_instruction=f"Include these missing names/places: {', '.join(missing)}",
                    location=None,
                ))
            elif missing:
                critique.issues.append(QualityIssue(
                    issue_type="missing_content",
                    severity="warning",
                    description=f"Some entities not found: {', '.join(missing)}",
                    fix_instruction=f"Consider including: {', '.join(missing)}",
                    location=None,
                ))

    def _check_hallucinations(
        self,
        source_doc,
        gen_doc,
        critique: QualityCritique,
    ):
        """Detect potential hallucinated content not in source."""
        # Extract noun phrases and entities from both
        source_concepts = set()
        for chunk in source_doc.noun_chunks:
            # Get the head noun's lemma
            source_concepts.add(chunk.root.lemma_.lower())
        for ent in source_doc.ents:
            source_concepts.add(ent.text.lower())

        # Check for new entities in generated text not in source
        hallucinated = []
        source_text_lower = source_doc.text.lower()

        for ent in gen_doc.ents:
            ent_lower = ent.text.lower()
            # Check if this entity or its root appears anywhere in source
            # Use first 4 chars as stem for NORP (nationalities/groups)
            if ent_lower not in source_text_lower:
                # Check if stem matches (e.g., "Marxist" vs "Marxists")
                stem = ent_lower[:min(len(ent_lower), 5)]
                if stem not in source_text_lower:
                    # Could be hallucinated - check if it's a significant addition
                    if ent.label_ in ('PERSON', 'ORG', 'GPE', 'NORP', 'EVENT'):
                        hallucinated.append(f"{ent.text} ({ent.label_})")

        # Check for suspicious adjective phrases not in source
        source_text_lower = source_doc.text.lower()
        suspicious_phrases = []

        for token in gen_doc:
            if token.pos_ == 'ADJ' and token.head.pos_ == 'NOUN':
                phrase = f"{token.text} {token.head.text}".lower()
                # Check if this adjective-noun pair appears in source
                if phrase not in source_text_lower:
                    # Check if the adjective itself is distinctive/specific
                    if token.text.lower() in ('great', 'famous', 'renowned', 'legendary',
                                               'beloved', 'esteemed', 'distinguished'):
                        suspicious_phrases.append(f'"{token.text} {token.head.text}"')

        if hallucinated:
            critique.issues.append(QualityIssue(
                issue_type="hallucination",
                severity="critical",  # Hallucinations are critical - trigger repair
                description=f"Added content not in source: {', '.join(hallucinated[:3])}",
                fix_instruction=f"Do NOT add: {', '.join(hallucinated[:3])} - stick to source content only",
                location=None,
            ))

        if suspicious_phrases:
            critique.issues.append(QualityIssue(
                issue_type="hallucination",
                severity="critical",  # Added descriptors are critical
                description=f"Descriptors added not in source: {', '.join(suspicious_phrases[:3])}",
                fix_instruction=f"Remove these added descriptors: {', '.join(suspicious_phrases[:3])}",
                location=None,
            ))

    def _check_truncation(
        self,
        source: str,
        generated: str,
        critique: QualityCritique,
    ):
        """Check if generated text is significantly shorter than source."""
        source_words = len(source.split())
        gen_words = len(generated.split())

        ratio = gen_words / source_words if source_words > 0 else 1.0

        if ratio < 0.7:
            critique.issues.append(QualityIssue(
                issue_type="truncation",
                severity="critical",
                description=f"Generated text is {ratio:.0%} of source length ({gen_words} vs {source_words} words)",
                fix_instruction="Expand the text to cover all content from the source - significant content appears to be missing",
                location=None,
            ))
        elif ratio < 0.85:
            critique.issues.append(QualityIssue(
                issue_type="truncation",
                severity="warning",
                description=f"Generated text is shorter than source ({gen_words} vs {source_words} words)",
                fix_instruction="Check if any content from the source was omitted",
                location=None,
            ))

    def get_repair_system_prompt(
        self,
        author: str,
        critique: QualityCritique,
    ) -> str:
        """Generate a repair system prompt based on critique.

        Args:
            author: Author name for style.
            critique: The critique with issues to fix.

        Returns:
            System prompt for repair generation.
        """
        # Build concise fix instructions
        fixes = []

        for issue in critique.issues:
            if issue.severity == "critical":
                if issue.issue_type == "hallucination":
                    # For hallucinations, just remind to stay faithful
                    fixes.append("stay faithful to source content")
                elif issue.issue_type == "incomplete_sentence":
                    fixes.append("complete all sentences")
                elif issue.issue_type == "truncation":
                    fixes.append("include all content from source")

        # Deduplicate
        fixes = list(dict.fromkeys(fixes))

        if fixes:
            fix_text = ", ".join(fixes)
            return format_prompt(
                "quality_repair_with_issues",
                author=author,
                fix_text=fix_text.capitalize()
            )
        else:
            return format_prompt("quality_repair", author=author)
