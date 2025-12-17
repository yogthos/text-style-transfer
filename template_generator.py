"""
Template Generator Module

Generates syntactic skeleton templates that the LLM must fill.
Templates are like "Mad Libs" - they specify:
- Number of sentences
- POS tag sequence for each sentence (NOUN VERB DET NOUN...)
- Punctuation placement
- Expected word counts
- Sentence openers

Templates are statistically derived from sample text analysis
and selected based on position in document.
"""

import spacy
import random
import sqlite3
import hashlib
import json
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass, field, asdict
from collections import Counter, defaultdict
import numpy as np

# Optional import for word-level templates
try:
    from semantic_word_mapper import SemanticWordMapper
    SEMANTIC_WORD_MAPPER_AVAILABLE = True
except ImportError:
    SEMANTIC_WORD_MAPPER_AVAILABLE = False
    SemanticWordMapper = None

try:
    from semantic_extractor import SemanticContent
    SEMANTIC_CONTENT_AVAILABLE = True
except ImportError:
    SEMANTIC_CONTENT_AVAILABLE = False
    SemanticContent = None


@dataclass
class WordSlot:
    """A single word position in a sentence template."""
    position: int  # Position in sentence (0-indexed)
    pos_tag: str  # Required POS (NOUN, VERB, ADJ, DET, etc.)
    prepopulated_word: Optional[str] = None  # Pre-filled from mappings
    semantic_hint: Optional[str] = None  # Hint from semantic content (entity, concept)
    is_semantic_slot: bool = False  # Must be filled from semantic content
    is_required: bool = True  # Must be filled vs optional
    slot_type: str = 'content'  # 'content', 'function', 'connector'


@dataclass
class SentenceTemplate:
    """Template for a single sentence."""
    word_count: int
    pos_sequence: List[str]  # ['DET', 'NOUN', 'VERB', 'DET', 'ADJ', 'NOUN', 'PUNCT']
    punctuation: str  # Final punctuation: '.', '!', '?'
    opener_type: str  # 'noun', 'adverb', 'conjunction', 'prep_phrase', etc.
    has_subordinate_clause: bool
    clause_count: int  # Number of clauses (1 = simple, 2+ = complex)
    length_category: str  # 'short' (<=10), 'medium' (11-25), 'long' (26+)
    word_slots: List[WordSlot] = field(default_factory=list)  # NEW: word-level slots
    has_word_slots: bool = False  # Flag to indicate word-level mode

    def to_template_string(self) -> str:
        """Convert to a human-readable template string."""
        parts = []
        for i, pos in enumerate(self.pos_sequence):
            if pos == 'PUNCT':
                parts.append(f'[{pos}]')
            else:
                parts.append(f'[{pos}]')
        return ' '.join(parts)

    def to_constraint_string(self) -> str:
        """Convert to a constraint description for LLM."""
        opener_desc = {
            'noun': 'noun/subject first',
            'det_noun': '"The [noun]..."',
            'adverb': 'adverb (Hence/Therefore/Thus)',
            'conjunction': 'conjunction (But/And/Yet)',
            'prep_phrase': 'prepositional phrase (Contrary to.../In this...)',
            'pronoun': 'pronoun (It/This/They)',
            'verb_ing': 'gerund (-ing verb)',
        }

        opener = opener_desc.get(self.opener_type, self.opener_type)
        complexity = f'{self.clause_count} clause(s)' if self.clause_count > 1 else 'simple'
        subordinate = ' + subordinate clause' if self.has_subordinate_clause else ''

        # If word-level slots are available, show them
        if self.has_word_slots and self.word_slots:
            parts = []
            for slot in self.word_slots:
                if slot.prepopulated_word:
                    # Show pre-populated word
                    parts.append(f"[{slot.prepopulated_word}]")
                elif slot.is_semantic_slot:
                    # Show semantic slot with hint
                    hint = f":{slot.semantic_hint}" if slot.semantic_hint else ""
                    parts.append(f"[{slot.pos_tag}{hint}]")
                else:
                    # Show POS tag for empty slot
                    parts.append(f"[{slot.pos_tag}]")

            word_template = " ".join(parts)
            return (f"~{self.word_count} words | {complexity}{subordinate} | "
                    f"opener: {opener}\n"
                    f"Word-level template: {word_template}")
        else:
            # Fallback to POS pattern summary
            pos_summary = self._summarize_pos()
            return (f"~{self.word_count} words | {complexity}{subordinate} | "
                    f"opener: {opener} | pattern: {pos_summary}")

    def _summarize_pos(self) -> str:
        """Create a readable POS pattern summary."""
        if not self.pos_sequence:
            return "free form"

        # Take first few significant POS tags
        significant = []
        for pos in self.pos_sequence[:8]:
            if pos not in ('PUNCT', 'SPACE'):
                significant.append(pos)

        return ' → '.join(significant[:5]) + ('...' if len(significant) > 5 else '')


@dataclass
class ParagraphTemplate:
    """Template for a complete paragraph."""
    sentence_count: int
    sentences: List[SentenceTemplate]
    total_word_count: int
    structural_role: str  # 'section_opener', 'paragraph_opener', 'body', 'closer'
    position_ratio: float  # 0.0 = start, 1.0 = end of document
    length_distribution: Dict[str, float]  # {'short': 0.2, 'medium': 0.5, 'long': 0.3}
    short_threshold: int = 10  # Dynamic threshold for short sentences
    medium_threshold: int = 25  # Dynamic threshold for medium sentences

    def to_constraint_string(self, used_phrases: Optional[List[str]] = None) -> str:
        """Convert to a constraint description for LLM.

        Args:
            used_phrases: List of opener phrases recently used (to avoid repetition)
        """
        role_desc = {
            'section_opener': 'SECTION OPENER - introduces a major point',
            'paragraph_opener': 'PARAGRAPH OPENER - introduces topic',
            'body': 'BODY PARAGRAPH - develops argument',
            'closer': 'CLOSING PARAGRAPH - concludes/synthesizes',
        }

        lines = [
            f"## 📋 STRUCTURAL CONTRACT (you MUST follow this)",
            f"",
            f"**Role**: {role_desc.get(self.structural_role, self.structural_role)}",
            f"**Position**: {self.position_ratio:.0%} through document",
            f"**Structure**: {self.sentence_count} sentences, ~{self.total_word_count} words total",
            f"",
            f"### Sentence Structure (follow this pattern):",
        ]

        # Check if any sentence has word-level slots
        has_word_level = any(sent.has_word_slots for sent in self.sentences)

        if has_word_level:
            lines.append("")
            lines.append("### Word-Level Templates (CRITICAL - follow exactly):")
            lines.append("")
            lines.append("**Format:**")
            lines.append("- `[word]` = Pre-populated word (use exactly as shown)")
            lines.append("- `[POS]` = Fill with word matching this part of speech")
            lines.append("- `[POS:hint]` = Fill with semantic content matching the hint")
            lines.append("")
            lines.append("**Rules:**")
            lines.append("1. Use pre-populated words exactly as shown")
            lines.append("2. Fill semantic slots with content from the semantic content section")
            lines.append("3. Fill other slots with appropriate words matching the POS tag")
            lines.append("4. Maintain the exact word order and structure")
            lines.append("")

        for i, sent in enumerate(self.sentences, 1):
            role = self._sentence_role(i, self.sentence_count)
            lines.append(f"")
            lines.append(f"**S{i}** [{role}]: {sent.to_constraint_string()}")

            # Add semantic slot guidance if present
            if sent.has_word_slots and sent.word_slots:
                semantic_slots = [s for s in sent.word_slots if s.is_semantic_slot]
                if semantic_slots:
                    hints = [s.semantic_hint for s in semantic_slots if s.semantic_hint]
                    if hints:
                        lines.append(f"  → Fill semantic slots with: {', '.join(set(hints[:3]))}")

        # Add length distribution guidance (using dynamic thresholds)
        lines.append("")
        lines.append("### Sentence Length Mix:")
        short_pct = self.length_distribution.get('short', 0) * 100
        medium_pct = self.length_distribution.get('medium', 0) * 100
        long_pct = self.length_distribution.get('long', 0) * 100
        lines.append(f"- Short sentences (≤{self.short_threshold} words): {short_pct:.0f}%")
        lines.append(f"- Medium sentences ({self.short_threshold+1}-{self.medium_threshold} words): {medium_pct:.0f}%")
        lines.append(f"- Long sentences ({self.medium_threshold+1}+ words): {long_pct:.0f}%")
        lines.append("")
        lines.append("**IMPORTANT**: Vary sentence lengths within this paragraph to match the mix above.")

        # Add explicit phrase avoidance (CRITICAL for preventing repetition)
        if used_phrases:
            # Normalize and get unique phrases from last 5 used
            normalized_used = []
            seen = set()
            for phrase in used_phrases[-5:]:
                if phrase:
                    normalized = phrase.lower().strip()
                    if normalized and normalized not in seen:
                        normalized_used.append(normalized)
                        seen.add(normalized)

            if normalized_used:
                lines.append("")
                lines.append("### 🚫 CRITICAL: DO NOT USE THESE PHRASES")
                lines.append("**You MUST avoid starting the first sentence with any of these recently used phrases:**")
                for phrase in normalized_used:
                    # Capitalize first letter for display
                    display_phrase = phrase.capitalize() if phrase else phrase
                    lines.append(f"- \"{display_phrase}...\"")
                lines.append("")
                lines.append("**This is MANDATORY** - use a DIFFERENT opener phrase even if it matches the opener type above.")

        # Add general guidance based on role
        lines.append("")
        lines.append("### Constraints:")
        if self.structural_role == 'section_opener':
            lines.append("- First sentence MUST be declarative and set up the main argument")
            lines.append("- Use contrastive opener if appropriate ('Contrary to...')")
        elif self.structural_role == 'closer':
            lines.append("- Final sentence MUST synthesize or conclude")
            lines.append("- Use causal marker (Therefore/Thus/Hence)")
        else:
            lines.append("- Vary sentence structures within the pattern")
            lines.append("- Ensure logical flow between sentences")

        return '\n'.join(lines)

    def _sentence_role(self, index: int, total: int) -> str:
        if index == 1:
            return "OPENER"
        elif index == total:
            return "CLOSER"
        else:
            return "BODY"


class TemplateGenerator:
    """
    Generates syntactic templates from sample text analysis.

    The key insight: Sentences in specific positions have specific structures.
    We can extract these patterns statistically and use them as constraints.
    """

    DB_PATH = Path(__file__).parent / "template_cache.db"

    def __init__(self, sample_path: str = None, sample_length_distribution: Optional[Dict[str, float]] = None,
                 config: Optional[Dict[str, Any]] = None, word_mapper: Optional[Any] = None):
        """Initialize the template generator.

        Args:
            sample_path: Path to sample text file (overrides config if provided)
            sample_length_distribution: Target length distribution from style analyzer
                                      {'short': 0.2, 'medium': 0.5, 'long': 0.3}
            config: Configuration dictionary (optional, for reading sample path)
            word_mapper: Optional SemanticWordMapper instance for word-level templates
        """
        self.config = config or {}
        self.word_mapper = word_mapper
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            import subprocess
            subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"])
            self.nlp = spacy.load("en_core_web_sm")

        self._init_db()

        # Store target length distribution from sample
        self.sample_length_distribution = sample_length_distribution or {
            'short': 0.2,
            'medium': 0.5,
            'long': 0.3
        }

        # Dynamic length thresholds (calculated from sample)
        self.short_threshold = 10  # Default: ≤10 words = short
        self.medium_threshold = 25  # Default: 11-25 words = medium, 26+ = long

        # Load and analyze sample if provided
        self.sample_templates: Dict[str, List[ParagraphTemplate]] = {}
        if sample_path is None:
            # Try to get from config, with fallback to default
            if config and "sample" in config and "file" in config.get("sample", {}):
                sample_file = config["sample"]["file"]
                sample_path = Path(__file__).parent / sample_file
            else:
                sample_path = Path(__file__).parent / "prompts" / "sample.txt"

        if Path(sample_path).exists():
            sample_text = Path(sample_path).read_text()
            # Calculate dynamic thresholds from sample
            self._calculate_length_thresholds(sample_text)
            self._load_or_analyze_sample(sample_path)

    def _init_db(self):
        """Initialize SQLite cache for templates."""
        conn = sqlite3.connect(self.DB_PATH)
        cursor = conn.cursor()

        cursor.execute('''
            CREATE TABLE IF NOT EXISTS template_cache (
                sample_hash TEXT PRIMARY KEY,
                templates_json TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')

        conn.commit()
        conn.close()

    def _calculate_length_thresholds(self, sample_text: str):
        """
        Calculate dynamic sentence length thresholds from sample text statistics.

        Uses percentiles to find natural breakpoints:
        - Short: ≤ 33rd percentile
        - Medium: 34th-67th percentile
        - Long: > 67th percentile

        This ensures thresholds match the actual distribution of the sample.
        """
        # Parse sample text and extract all sentence lengths
        doc = self.nlp(sample_text)
        sentences = list(doc.sents)

        if not sentences:
            # Use defaults if no sentences found
            return

        # Calculate word count for each sentence
        sentence_lengths = []
        for sent in sentences:
            word_count = len([t for t in sent if not t.is_punct and not t.is_space])
            # Only include substantial sentences (skip fragments)
            if word_count >= 3:
                sentence_lengths.append(word_count)

        if len(sentence_lengths) < 10:
            # Not enough data, use defaults
            return

        # Calculate percentiles (33rd and 67th for roughly equal categories)
        sorted_lengths = sorted(sentence_lengths)
        n = len(sorted_lengths)

        # 33rd percentile (1/3 of sentences)
        p33_idx = int(n * 0.33)
        self.short_threshold = sorted_lengths[p33_idx] if p33_idx < n else sorted_lengths[-1]

        # 67th percentile (2/3 of sentences)
        p67_idx = int(n * 0.67)
        self.medium_threshold = sorted_lengths[p67_idx] if p67_idx < n else sorted_lengths[-1]

        # Ensure thresholds are reasonable (at least 5 words apart)
        if self.medium_threshold - self.short_threshold < 5:
            # If too close, use defaults or adjust
            avg_length = sum(sentence_lengths) / len(sentence_lengths)
            self.short_threshold = max(8, int(avg_length * 0.5))
            self.medium_threshold = max(self.short_threshold + 5, int(avg_length * 1.2))

        print(f"  [TemplateGen] Dynamic length thresholds: short≤{self.short_threshold}, "
              f"medium={self.short_threshold+1}-{self.medium_threshold}, long>{self.medium_threshold}")

    def _load_or_analyze_sample(self, sample_path: str):
        """Load templates from cache or analyze sample."""
        sample_text = Path(sample_path).read_text()
        sample_hash = hashlib.md5(sample_text.encode()).hexdigest()

        # Check cache
        conn = sqlite3.connect(self.DB_PATH)
        cursor = conn.cursor()
        cursor.execute(
            'SELECT templates_json FROM template_cache WHERE sample_hash = ?',
            (sample_hash,)
        )
        row = cursor.fetchone()
        conn.close()

        if row:
            print("  [TemplateGen] Loaded templates from cache")
            self._load_templates_from_json(row[0])
        else:
            print("  [TemplateGen] Analyzing sample text for templates...")
            self._analyze_sample(sample_text)
            self._save_templates_to_cache(sample_hash)

    def _analyze_sample(self, sample_text: str):
        """Analyze sample text to extract templates."""
        paragraphs = [p.strip() for p in sample_text.split('\n\n') if p.strip()]
        total_paragraphs = len(paragraphs)

        # Group templates by role
        self.sample_templates = {
            'section_opener': [],
            'paragraph_opener': [],
            'body': [],
            'closer': [],
        }

        # Filter to substantial paragraphs (not headers/short lines)
        substantial_paras = []
        for i, para in enumerate(paragraphs):
            word_count = len(para.split())
            # Skip very short paragraphs (likely headers)
            if word_count >= 20:
                position_ratio = i / max(total_paragraphs - 1, 1)
                substantial_paras.append((para, position_ratio, self._is_section_start(para)))

        for idx, (para, position_ratio, is_section_start) in enumerate(substantial_paras):
            total_substantial = len(substantial_paras)

            # Multi-factor role detection combining position and content patterns
            role = None

            # Section opener: explicit markers OR content-based detection OR first paragraph
            if idx == 0 or is_section_start or self._is_section_start_by_content(para, idx):
                role = 'section_opener'
            # Closer: last 20% OR content-based conclusion detection
            elif idx >= int(total_substantial * 0.8) or self._is_closer(para, idx, total_substantial):
                role = 'closer'
            # Paragraph opener: first 20% OR content-based detection
            elif idx < int(total_substantial * 0.2) or self._is_paragraph_opener(para, idx, total_substantial):
                role = 'paragraph_opener'
            else:
                role = 'body'

            template = self._extract_paragraph_template(para, role, position_ratio)
            # Only keep templates with reasonable structure
            if template and template.total_word_count >= 15 and template.sentence_count >= 1:
                self.sample_templates[role].append(template)

        # Ensure minimum template diversity per role (at least 5 templates)
        min_templates_per_role = 5
        for role in ['section_opener', 'paragraph_opener', 'closer']:
            current_count = len(self.sample_templates[role])
            if current_count < min_templates_per_role and self.sample_templates['body']:
                needed = min_templates_per_role - current_count

                # Prioritize variety when borrowing: different opener types, sentence counts, lengths
                body_templates = self.sample_templates['body'].copy()

                # Get existing opener types for this role to avoid duplicates
                existing_openers = set()
                if self.sample_templates[role]:
                    for t in self.sample_templates[role]:
                        if t.sentences and t.sentences[0].opener_type:
                            existing_openers.add(t.sentences[0].opener_type)

                # Score body templates by diversity (prefer different opener types)
                scored_body = []
                for t in body_templates:
                    opener_type = t.sentences[0].opener_type if t.sentences and t.sentences[0].opener_type else 'other'
                    # Boost score if opener type is different from existing
                    diversity_score = 2.0 if opener_type not in existing_openers else 1.0
                    # Also consider sentence count and length variety
                    length_score = 1.0 + (t.sentence_count / 10.0)  # Prefer multi-sentence
                    total_score = diversity_score * length_score
                    scored_body.append((total_score, t, opener_type))

                # Sort by score (highest diversity first)
                scored_body.sort(key=lambda x: -x[0])

                # Select diverse templates
                selected = []
                selected_openers = existing_openers.copy()
                for score, template, opener_type in scored_body:
                    if len(selected) >= needed:
                        break
                    # Prefer templates with different opener types
                    if opener_type not in selected_openers or len(selected) < needed // 2:
                        selected.append(template)
                        selected_openers.add(opener_type)

                # If we still need more, just take the highest scored ones
                if len(selected) < needed:
                    remaining = [t for _, t, _ in scored_body if t not in selected]
                    selected.extend(remaining[:needed - len(selected)])

                self.sample_templates[role].extend(selected[:needed])

        # Log stats
        for role, templates in self.sample_templates.items():
            if templates:
                avg_sents = np.mean([t.sentence_count for t in templates])
                avg_words = np.mean([t.total_word_count for t in templates])
                print(f"    {role}: {len(templates)} templates, avg {avg_sents:.1f} sentences, {avg_words:.0f} words")

    def _is_section_start(self, para: str) -> bool:
        """Check if paragraph starts a new section."""
        first_line = para.split('\n')[0].strip()
        # Check for section markers
        section_patterns = [
            r'^[0-9]+\)',
            r'^[a-z]\)',
            r'^\d+\.',
            r'^[IVX]+\.',
            r'^#+\s',
        ]
        import re
        return any(re.match(p, first_line) for p in section_patterns)

    def _is_section_start_by_content(self, para: str, idx: int) -> bool:
        """Detect section boundaries by content patterns."""
        import re
        para_lower = para.lower()
        first_sentence = para.split('.')[0].strip().lower() if '.' in para else para_lower[:100]

        # Check for transition markers that indicate new sections
        section_markers = [
            r'^before\b',
            r'^above all\b',
            r'^next\b',
            r'^similarly\b',
            r'^however\b',
            r'^but\b',
            r'^in the process\b',
            r'^contrary to\b',
            r'^as opposed to\b',
            r'^the principal features\b',
        ]

        # Check first sentence for section markers
        for pattern in section_markers:
            if re.match(pattern, first_sentence, re.IGNORECASE):
                return True

        # Check for enumeration markers
        enumeration_markers = [
            r'^first\b',
            r'^second\b',
            r'^third\b',
            r'^finally\b',
        ]
        for pattern in enumeration_markers:
            if re.match(pattern, first_sentence, re.IGNORECASE):
                return True

        # Check if paragraph is significantly longer (indicating major point)
        word_count = len(para.split())
        if word_count > 200:  # Very long paragraphs often indicate major sections
            return True

        return False

    def _is_paragraph_opener(self, para: str, idx: int, total: int) -> bool:
        """Detect paragraphs that open new topics."""
        import re
        para_lower = para.lower()
        first_sentence = para.split('.')[0].strip().lower() if '.' in para else para_lower[:100]

        # Check for transition markers at start
        opener_markers = [
            r'^hence\b',
            r'^thus\b',
            r'^therefore\b',
            r'^consequently\b',
            r'^further\b',
            r'^moreover\b',
            r'^in addition\b',
            r'^furthermore\b',
            r'^likewise\b',
            r'^additionally\b',
        ]

        for pattern in opener_markers:
            if re.match(pattern, first_sentence, re.IGNORECASE):
                return True

        # Check for topic introduction patterns (first sentence introduces new concept)
        # This is a heuristic: paragraphs that start with "In", "For", "To", "When", "If"
        topic_intro_patterns = [
            r'^in [a-z]+',
            r'^for [a-z]+',
            r'^to [a-z]+',
            r'^when [a-z]+',
            r'^if [a-z]+',
        ]

        # Only consider if not already in first 20% (to avoid double-counting)
        if idx >= int(total * 0.2):
            for pattern in topic_intro_patterns:
                if re.match(pattern, first_sentence, re.IGNORECASE):
                    return True

        return False

    def _is_closer(self, para: str, idx: int, total: int) -> bool:
        """Detect closing/synthesis paragraphs."""
        import re
        para_lower = para.lower()
        first_sentence = para.split('.')[0].strip().lower() if '.' in para else para_lower[:100]
        last_sentence = para.split('.')[-1].strip().lower() if '.' in para else para_lower[-100:]

        # Check for conclusion markers
        conclusion_markers = [
            r'^thus\b',
            r'^therefore\b',
            r'^hence\b',
            r'^in conclusion\b',
            r'^this means\b',
            r'^it follows\b',
            r'^such is\b',
            r'^this form\b',
            r'^our conclusion\b',
            r'^we conclude\b',
            r'^in summary\b',
        ]

        # Check first sentence
        for pattern in conclusion_markers:
            if re.match(pattern, first_sentence, re.IGNORECASE):
                return True

        # Check last sentence for synthesis patterns
        synthesis_patterns = [
            r'such is\b',
            r'this form\b',
            r'our conclusion\b',
            r'this means\b',
            r'it follows\b',
        ]

        for pattern in synthesis_patterns:
            if re.search(pattern, last_sentence, re.IGNORECASE):
                return True

        # Check if paragraph synthesizes previous content (contains summary words)
        summary_words = ['conclusion', 'summary', 'synthesis', 'therefore', 'thus', 'hence', 'consequently']
        summary_count = sum(1 for word in summary_words if word in para_lower)
        if summary_count >= 2:  # Multiple summary indicators
            return True

        return False

    def _extract_paragraph_template(self, para: str, role: str,
                                     position_ratio: float) -> Optional[ParagraphTemplate]:
        """Extract template from a single paragraph."""
        doc = self.nlp(para)
        sentences = list(doc.sents)

        if not sentences:
            return None

        sentence_templates = []
        for sent in sentences:
            template = self._extract_sentence_template(sent)
            if template:
                sentence_templates.append(template)

        if not sentence_templates:
            return None

        # Calculate length distribution for this paragraph
        length_counts = {'short': 0, 'medium': 0, 'long': 0}
        for sent_template in sentence_templates:
            length_counts[sent_template.length_category] += 1

        total_sents = len(sentence_templates)
        length_distribution = {
            'short': length_counts['short'] / total_sents,
            'medium': length_counts['medium'] / total_sents,
            'long': length_counts['long'] / total_sents
        }

        return ParagraphTemplate(
            sentence_count=len(sentence_templates),
            sentences=sentence_templates,
            total_word_count=sum(t.word_count for t in sentence_templates),
            structural_role=role,
            position_ratio=position_ratio,
            length_distribution=length_distribution,
            short_threshold=self.short_threshold,
            medium_threshold=self.medium_threshold
        )

    def _extract_sentence_template(self, sent) -> Optional[SentenceTemplate]:
        """Extract template from a single sentence."""
        # Filter to actual words (not punctuation, spaces, or special chars)
        tokens = [t for t in sent if not t.is_space and not t.is_punct]
        all_tokens = list(sent)

        # Skip very short sentences or fragments
        if len(tokens) < 5:
            return None

        # Skip sentences that start with special characters/numbers
        if tokens and (tokens[0].text.startswith('(') or
                       tokens[0].text.startswith('[') or
                       tokens[0].pos_ == 'NUM'):
            return None

        # Extract POS sequence (words only, simplified)
        pos_sequence = []
        for t in tokens[:20]:  # Limit to first 20 words
            # Simplify POS tags for template
            pos = t.pos_
            if pos in ('PROPN', 'NOUN'):
                pos_sequence.append('NOUN')
            elif pos in ('AUX', 'VERB'):
                pos_sequence.append('VERB')
            elif pos in ('ADJ', 'ADV'):
                pos_sequence.append('MOD')  # Modifier
            else:
                pos_sequence.append(pos)

        # Determine opener type
        opener_type = self._classify_opener(tokens)

        # Check for subordinate clauses
        has_subordinate = any(t.dep_ in ('mark', 'advcl', 'relcl', 'ccomp') for t in all_tokens)

        # Count clauses (based on finite verbs)
        finite_verbs = [t for t in all_tokens if t.pos_ == 'VERB' and t.dep_ in ('ROOT', 'conj', 'advcl', 'ccomp')]
        clause_count = max(1, len(finite_verbs))

        # Get final punctuation
        punctuation = '.'
        for t in reversed(all_tokens):
            if t.is_punct and t.text in '.!?':
                punctuation = t.text
                break

        word_count = len(tokens)

        # Categorize sentence length using dynamic thresholds
        if word_count <= self.short_threshold:
            length_category = 'short'
        elif word_count <= self.medium_threshold:
            length_category = 'medium'
        else:
            length_category = 'long'

        template = SentenceTemplate(
            word_count=word_count,
            pos_sequence=pos_sequence,
            punctuation=punctuation,
            opener_type=opener_type,
            has_subordinate_clause=has_subordinate,
            clause_count=clause_count,
            length_category=length_category
        )

        # Extract word-level slots if enabled
        use_word_level = self.config.get('template_generation', {}).get('use_word_level_templates', False)
        if use_word_level and self.word_mapper:
            word_slots = self._extract_word_level_template(sent, self.word_mapper)
            if word_slots:
                template.word_slots = word_slots
                template.has_word_slots = True
                # Pre-populate common words
                prepopulate = self.config.get('template_generation', {}).get('prepopulate_common_words', True)
                if prepopulate:
                    template = self.prepopulate_common_words(template, self.word_mapper, config=self.config)

        # Extract word-level slots if enabled
        use_word_level = self.config.get('template_generation', {}).get('use_word_level_templates', False)
        if use_word_level and self.word_mapper:
            word_slots = self._extract_word_level_template(sent, self.word_mapper)
            if word_slots:
                template.word_slots = word_slots
                template.has_word_slots = True
                # Pre-populate common words
                prepopulate = self.config.get('template_generation', {}).get('prepopulate_common_words', True)
                if prepopulate:
                    template = self.prepopulate_common_words(template, self.word_mapper, config=self.config)

        return template

    def _extract_word_level_template(self, sent, word_mapper: Optional[Any] = None) -> List[WordSlot]:
        """
        Extract word-level template from a sentence.

        Args:
            sent: spaCy sentence object
            word_mapper: Optional SemanticWordMapper instance for pre-population

        Returns:
            List of WordSlot objects representing each word position
        """
        if word_mapper is None:
            return []

        # Filter to actual words (not punctuation, spaces)
        tokens = [t for t in sent if not t.is_space and not t.is_punct]
        all_tokens = list(sent)

        if len(tokens) < 5:
            return []

        word_slots = []
        position = 0

        for token in all_tokens:
            if token.is_space:
                continue

            if token.is_punct:
                # Add punctuation as a slot
                pos_tag = 'PUNCT'
                slot_type = 'function'
            else:
                # Simplify POS tags
                pos = token.pos_
                if pos in ('PROPN', 'NOUN'):
                    pos_tag = 'NOUN'
                    slot_type = 'content'
                elif pos in ('AUX', 'VERB'):
                    pos_tag = 'VERB'
                    slot_type = 'content'
                elif pos in ('ADJ', 'ADV'):
                    pos_tag = 'MOD'
                    slot_type = 'content'
                elif pos in ('DET', 'PREP', 'CCONJ', 'SCONJ', 'PRON'):
                    pos_tag = pos
                    slot_type = 'function'
                else:
                    pos_tag = pos
                    slot_type = 'function'

            # Determine if this word can be pre-populated
            prepopulated_word = None
            if word_mapper and slot_type == 'function':
                # Try to map common function words
                word_lower = token.lemma_.lower()
                mapped = word_mapper.map_word(word_lower, token.pos_)
                if mapped:
                    prepopulated_word = mapped
            elif word_mapper and slot_type == 'content':
                # Check if it's a common word that can be mapped
                word_lower = token.lemma_.lower()
                mapped = word_mapper.map_word(word_lower, token.pos_)
                if mapped:
                    # Only pre-populate if it's a very common word
                    prepopulated_word = mapped

            # Determine if this should be a semantic slot (content words that aren't common)
            is_semantic_slot = False
            if slot_type == 'content' and not prepopulated_word:
                # Check if it's likely an entity or key concept
                if pos_tag == 'NOUN' and token.ent_type_:
                    is_semantic_slot = True
                elif pos_tag == 'NOUN' and token.text[0].isupper():
                    is_semantic_slot = True

            slot = WordSlot(
                position=position,
                pos_tag=pos_tag,
                prepopulated_word=prepopulated_word,
                is_semantic_slot=is_semantic_slot,
                is_required=True,
                slot_type=slot_type
            )

            word_slots.append(slot)
            if not token.is_punct:
                position += 1

        return word_slots

    def prepopulate_common_words(self, template: SentenceTemplate, word_mapper: Optional[Any] = None,
                                  context: Optional[str] = None, config: Optional[Dict[str, Any]] = None) -> SentenceTemplate:
        """
        Pre-populate common words in a sentence template using semantic word mappings.

        Args:
            template: SentenceTemplate to pre-populate
            word_mapper: SemanticWordMapper instance
            context: Optional preceding output for context-aware selection
            config: Configuration dict with pre-population settings

        Returns:
            Updated SentenceTemplate with pre-populated words
        """
        if word_mapper is None or not template.word_slots:
            return template

        # Get config settings
        if config:
            max_prepopulated_ratio = config.get('template_generation', {}).get('max_prepopulated_words', 0.4)
            min_prepopulated = config.get('template_generation', {}).get('min_prepopulated_words', 2)
        else:
            max_prepopulated_ratio = 0.4
            min_prepopulated = 2

        max_prepopulated = int(len(template.word_slots) * max_prepopulated_ratio)
        prepopulated_count = sum(1 for slot in template.word_slots if slot.prepopulated_word)

        # Pre-populate slots that don't have words yet
        updated_slots = []
        for slot in template.word_slots:
            if slot.prepopulated_word:
                # Already pre-populated
                updated_slots.append(slot)
                continue

            # Skip if we've reached max pre-populated ratio
            if prepopulated_count >= max_prepopulated:
                updated_slots.append(slot)
                continue

            # Try to pre-populate function words first (DET, PREP, CONJ, etc.)
            if slot.slot_type == 'function' and slot.pos_tag in ('DET', 'PREP', 'CCONJ', 'SCONJ', 'PRON'):
                # For function words, try to get a common mapping
                # Use a generic common word for this POS
                common_words_by_pos = {
                    'DET': ['the', 'a', 'an'],
                    'PREP': ['of', 'in', 'on', 'at', 'to', 'for'],
                    'CCONJ': ['and', 'or', 'but'],
                    'SCONJ': ['that', 'which', 'who'],
                    'PRON': ['it', 'this', 'that', 'they', 'we']
                }

                if slot.pos_tag in common_words_by_pos:
                    for common_word in common_words_by_pos[slot.pos_tag]:
                        mapped = word_mapper.map_word(common_word, slot.pos_tag)
                        if mapped:
                            slot.prepopulated_word = mapped
                            prepopulated_count += 1
                            break

            # For content words, only pre-populate if it's a very common word
            elif slot.slot_type == 'content' and prepopulated_count < min_prepopulated:
                # Try common content words
                common_content = {
                    'NOUN': ['way', 'method', 'approach', 'thing', 'aspect'],
                    'VERB': ['show', 'indicate', 'demonstrate', 'reveal'],
                    'MOD': ['important', 'significant', 'large', 'small']
                }

                if slot.pos_tag in common_content:
                    for common_word in common_content[slot.pos_tag]:
                        mapped = word_mapper.map_word(common_word, slot.pos_tag)
                        if mapped:
                            slot.prepopulated_word = mapped
                            prepopulated_count += 1
                            break

            updated_slots.append(slot)

        # Create updated template
        template.word_slots = updated_slots
        template.has_word_slots = True
        return template

    def _mark_semantic_slots(self, template: SentenceTemplate, semantic_content: Optional[Any] = None) -> SentenceTemplate:
        """
        Mark slots that should be filled from semantic content.

        Args:
            template: SentenceTemplate to mark
            semantic_content: SemanticContent with entities, claims, etc.

        Returns:
            Updated SentenceTemplate with semantic slots marked
        """
        if semantic_content is None or not template.word_slots:
            return template

        # Extract entities and key concepts from semantic content
        entities = []
        key_concepts = []

        if hasattr(semantic_content, 'entities'):
            entities = [e.text.lower() for e in semantic_content.entities]

        if hasattr(semantic_content, 'claims'):
            # Extract key nouns from claims
            for claim in semantic_content.claims:
                if claim.subject:
                    key_concepts.append(claim.subject.lower())
                for obj in claim.objects:
                    key_concepts.append(obj.lower())

        # Mark slots that should contain semantic content
        updated_slots = []
        for slot in template.word_slots:
            # Skip if already pre-populated (don't override)
            if slot.prepopulated_word:
                updated_slots.append(slot)
                continue

            # Mark NOUN slots as semantic if they're likely to contain entities/concepts
            if slot.pos_tag == 'NOUN' and not slot.is_semantic_slot:
                # Check if this position typically contains entities
                # First few NOUN slots are more likely to be entities
                if slot.position < 5:
                    slot.is_semantic_slot = True
                    slot.semantic_hint = "entity or key concept"

            # Mark VERB slots that might need semantic content
            elif slot.pos_tag == 'VERB' and slot.position < 3:
                # Early verbs might be predicates from claims
                slot.is_semantic_slot = True
                slot.semantic_hint = "predicate from semantic content"

            updated_slots.append(slot)

        template.word_slots = updated_slots
        return template

    def _classify_opener(self, tokens) -> str:
        """Classify how a sentence opens."""
        if not tokens:
            return 'unknown'

        first = tokens[0]
        first_two = tokens[:2] if len(tokens) >= 2 else tokens

        # Check common opener patterns
        if first.pos_ == 'SCONJ' or first.text.lower() in ('contrary', 'according'):
            return 'prep_phrase'
        elif first.pos_ == 'ADV':
            return 'adverb'
        elif first.pos_ == 'CCONJ':
            return 'conjunction'
        elif first.pos_ == 'PRON':
            return 'pronoun'
        elif first.pos_ == 'DET':
            return 'det_noun'
        elif first.pos_ == 'NOUN' or first.pos_ == 'PROPN':
            return 'noun'
        elif first.pos_ == 'VERB' and first.tag_ == 'VBG':
            return 'verb_ing'
        else:
            return 'other'

    def _extract_opener_phrase(self, text: str) -> str:
        """Extract the actual opener phrase (first 2-3 words) from text.

        Returns normalized phrase (lowercase, trimmed) for consistent matching.
        """
        if not text:
            return ""

        # Get first sentence
        doc = self.nlp(text)
        sentences = list(doc.sents)
        if not sentences:
            return ""

        first_sent = sentences[0]
        tokens = [t for t in first_sent if not t.is_space and not t.is_punct]

        # Get first 2-3 words (normalized to lowercase, trimmed)
        if len(tokens) >= 3:
            phrase = ' '.join([t.text.lower().strip() for t in tokens[:3]])
        elif len(tokens) >= 2:
            phrase = ' '.join([t.text.lower().strip() for t in tokens[:2]])
        elif tokens:
            phrase = tokens[0].text.lower().strip()
        else:
            return ""

        # Normalize whitespace
        phrase = ' '.join(phrase.split())
        return phrase

    def _get_distribution_based_templates(self,
                                         example_selector: Any,
                                         role: str,
                                         position_ratio: float,
                                         used_openers: List[str],
                                         used_phrases: List[str],
                                         target_count: int = 5) -> Tuple[List[ParagraphTemplate], List[str]]:
        """
        Get templates from paragraphs with underrepresented opener types.

        When all contextually similar paragraphs share the same opener,
        this method queries the sample for paragraphs with different opener types
        to ensure variety matches the sample distribution.

        Returns:
            Tuple of (templates, opener_phrases) for tracking
        """
        if not example_selector or not hasattr(example_selector, 'paragraphs'):
            return [], []

        # Get opener type distribution from sample
        opener_counts = Counter()
        for para in example_selector.paragraphs:
            if para.opener_type:
                opener_counts[para.opener_type] += 1

        total = sum(opener_counts.values())
        if total == 0:
            return [], []

        # Find underrepresented opener types (not recently used)
        underrepresented = []
        for opener_type, count in opener_counts.most_common():
            if opener_type not in used_openers[-3:]:
                underrepresented.append(opener_type)

        # If all openers are used, reset and use all
        if not underrepresented:
            underrepresented = list(opener_counts.keys())

        # Collect paragraphs with underrepresented opener types
        candidate_paragraphs = []
        for para in example_selector.paragraphs:
            if para.opener_type in underrepresented:
                # Check opener phrase to avoid exact repetition
                opener_phrase = self._extract_opener_phrase(para.text)
                if opener_phrase and opener_phrase not in used_phrases[-5:]:
                    candidate_paragraphs.append((para.text, opener_phrase))

        # Extract templates from these paragraphs
        templates = []
        template_phrases = []
        for para_text, opener_phrase in candidate_paragraphs[:target_count * 3]:  # Get more to filter
            template = self._extract_paragraph_template(para_text, role, position_ratio)
            if template and template.total_word_count >= 15:
                templates.append(template)
                template_phrases.append(opener_phrase)

        # Prefer multi-sentence templates (2+ sentences)
        if templates:
            multi_sentence = [(t, p) for t, p in zip(templates, template_phrases) if t.sentence_count >= 2]
            if multi_sentence:
                templates, template_phrases = zip(*multi_sentence)
                templates = list(templates)
                template_phrases = list(template_phrases)
            # If no multi-sentence, keep single-sentence as fallback

        return templates, template_phrases

    def _save_templates_to_cache(self, sample_hash: str):
        """Save templates to SQLite cache."""
        # Convert to JSON-serializable format
        templates_data = {}
        for role, templates in self.sample_templates.items():
            templates_data[role] = [
                {
                    'sentence_count': t.sentence_count,
                    'total_word_count': t.total_word_count,
                    'structural_role': t.structural_role,
                    'position_ratio': t.position_ratio,
                    'length_distribution': t.length_distribution,
                    'short_threshold': t.short_threshold,
                    'medium_threshold': t.medium_threshold,
                    'sentences': [
                        {
                    'word_count': s.word_count,
                    'pos_sequence': s.pos_sequence,
                    'punctuation': s.punctuation,
                    'opener_type': s.opener_type,
                    'has_subordinate_clause': s.has_subordinate_clause,
                    'clause_count': s.clause_count,
                    'length_category': s.length_category
                        }
                        for s in t.sentences
                    ]
                }
                for t in templates
            ]

        conn = sqlite3.connect(self.DB_PATH)
        cursor = conn.cursor()
        cursor.execute(
            'INSERT OR REPLACE INTO template_cache (sample_hash, templates_json) VALUES (?, ?)',
            (sample_hash, json.dumps(templates_data))
        )
        conn.commit()
        conn.close()

    def _load_templates_from_json(self, templates_json: str):
        """Load templates from JSON."""
        templates_data = json.loads(templates_json)

        self.sample_templates = {}
        for role, templates in templates_data.items():
            self.sample_templates[role] = [
                ParagraphTemplate(
                    sentence_count=t['sentence_count'],
                    total_word_count=t['total_word_count'],
                    structural_role=t['structural_role'],
                    position_ratio=t['position_ratio'],
                    length_distribution=t.get('length_distribution', {'short': 0.33, 'medium': 0.33, 'long': 0.34}),
                    short_threshold=t.get('short_threshold', self.short_threshold),
                    medium_threshold=t.get('medium_threshold', self.medium_threshold),
                    sentences=[
                        SentenceTemplate(
                            word_count=s['word_count'],
                            pos_sequence=s['pos_sequence'],
                            punctuation=s['punctuation'],
                            opener_type=s['opener_type'],
                            has_subordinate_clause=s['has_subordinate_clause'],
                            clause_count=s['clause_count'],
                            length_category=s.get('length_category', 'medium')
                        )
                        for s in t['sentences']
                    ]
                )
                for t in templates
            ]

    def generate_template(self, role: str, position_ratio: float,
                          semantic_weight: int = 0,
                          used_openers: Optional[List[str]] = None,
                          paragraph_index: int = 0) -> ParagraphTemplate:
        """
        Generate a template for a paragraph based on role and position.

        Args:
            role: 'section_opener', 'paragraph_opener', 'body', 'closer'
            position_ratio: 0.0 = start, 1.0 = end of document
            semantic_weight: Hint about content complexity (0=light, 3=heavy)
            used_openers: List of opener types already used in this document (for variety)
            paragraph_index: Current paragraph index (for deterministic variety)

        Returns:
            ParagraphTemplate with structure to follow
        """
        if used_openers is None:
            used_openers = []

        # Get templates for this role
        candidates = self.sample_templates.get(role, [])

        if not candidates:
            # Fallback: use body templates
            candidates = self.sample_templates.get('body', [])

        if not candidates:
            # Last resort: generate a default template
            return self._generate_default_template(role, position_ratio)

        # PRIORITY: Prefer multi-sentence templates (2+ sentences)
        multi_sentence_candidates = [t for t in candidates if t.sentence_count >= 2]
        if multi_sentence_candidates:
            candidates = multi_sentence_candidates

        # VARIETY MECHANISM: Filter out templates with recently used openers (CRITICAL: do this even if only 1 candidate)
        if used_openers and candidates:
            # Find templates with different openers
            varied_candidates = [
                t for t in candidates
                if t.sentences and t.sentences[0].opener_type not in used_openers[-3:]  # Avoid last 3 openers
            ]
            # If we filtered out all candidates, we need to be less strict or find alternatives
            if varied_candidates:
                candidates = varied_candidates
            # If no varied candidates but we have used_openers, at least try to avoid the most recent one
            elif len(candidates) == 1 and used_openers and candidates[0].sentences:
                # Check if this is the exact same opener as the last one
                if candidates[0].sentences[0].opener_type == used_openers[-1]:
                    # This is a problem - we're repeating the exact same opener
                    # But we have no alternatives, so we'll proceed (could log a warning)
                    pass

        # LENGTH DISTRIBUTION MATCHING: Score templates by how well they match sample distribution
        if len(candidates) > 1:
            # Score each template based on length distribution match
            scored_candidates = []
            for template in candidates:
                score = self._score_length_distribution_match(template.length_distribution)
                scored_candidates.append((score, template))

            # Sort by: 1) length distribution match (higher is better), 2) position distance
            scored_candidates.sort(
                key=lambda x: (-x[0], abs(x[1].position_ratio - position_ratio))
            )

            # Take top candidates (best length match + position)
            top_candidates = [t for _, t in scored_candidates[:min(5, len(scored_candidates))]]
            best_template = top_candidates[paragraph_index % len(top_candidates)]
        else:
            best_template = candidates[0]

        # Split long single sentences if we still got one
        if best_template.sentence_count == 1 and best_template.sentences[0].word_count > 60:
            best_template = self._split_long_sentence_template(best_template)

        # Adjust for semantic weight (more content = longer sentences)
        if semantic_weight > 0:
            adjusted_sentences = []
            for sent in best_template.sentences:
                adjusted_word_count = sent.word_count + (semantic_weight * 3)
                # Recalculate length category after adjustment using dynamic thresholds
                if adjusted_word_count <= self.short_threshold:
                    length_category = 'short'
                elif adjusted_word_count <= self.medium_threshold:
                    length_category = 'medium'
                else:
                    length_category = 'long'

                adjusted = SentenceTemplate(
                    word_count=adjusted_word_count,
                    pos_sequence=sent.pos_sequence,
                    punctuation=sent.punctuation,
                    opener_type=sent.opener_type,
                    has_subordinate_clause=sent.has_subordinate_clause or semantic_weight > 1,
                    clause_count=max(sent.clause_count, 1 + semantic_weight // 2),
                    length_category=length_category
                )
                adjusted_sentences.append(adjusted)

            # Recalculate length distribution after adjustment
            length_counts = {'short': 0, 'medium': 0, 'long': 0}
            for sent in adjusted_sentences:
                length_counts[sent.length_category] += 1

            total_sents = len(adjusted_sentences)
            adjusted_dist = {
                'short': length_counts['short'] / total_sents,
                'medium': length_counts['medium'] / total_sents,
                'long': length_counts['long'] / total_sents
            }

            return ParagraphTemplate(
                sentence_count=best_template.sentence_count,
                sentences=adjusted_sentences,
                total_word_count=sum(s.word_count for s in adjusted_sentences),
                structural_role=role,
                position_ratio=position_ratio,
                length_distribution=adjusted_dist,
                short_threshold=self.short_threshold,
                medium_threshold=self.medium_threshold
            )

        return best_template

    def _score_length_distribution_match(self, template_dist: Dict[str, float]) -> float:
        """
        Score how well a template's length distribution matches the sample distribution.

        Returns a score from 0.0 (poor match) to 1.0 (perfect match).
        Uses inverse of mean absolute error.
        """
        if not self.sample_length_distribution:
            return 0.5  # Neutral score if no target distribution

        # Calculate mean absolute error
        mae = sum(
            abs(template_dist.get(cat, 0) - self.sample_length_distribution.get(cat, 0))
            for cat in ['short', 'medium', 'long']
        ) / 3.0

        # Convert to score (lower MAE = higher score)
        # Perfect match (MAE=0) = 1.0, worst match (MAE=1.0) = 0.0
        score = max(0.0, 1.0 - mae)
        return score

    def _generate_default_template(self, role: str, position_ratio: float) -> ParagraphTemplate:
        """Generate a default template when no samples available."""
        # Default structures based on role
        if role == 'section_opener':
            sentences = [
                SentenceTemplate(35, ['DET', 'ADJ', 'NOUN', 'VERB'], '.', 'det_noun', True, 2, 'long'),
                SentenceTemplate(25, ['PRON', 'VERB', 'DET', 'NOUN'], '.', 'pronoun', False, 1, 'medium'),
                SentenceTemplate(30, ['ADV', 'DET', 'NOUN', 'VERB'], '.', 'adverb', True, 2, 'long'),
            ]
        elif role == 'closer':
            sentences = [
                SentenceTemplate(30, ['ADV', 'DET', 'NOUN', 'VERB'], '.', 'adverb', True, 2, 'long'),
                SentenceTemplate(25, ['DET', 'NOUN', 'VERB', 'ADJ'], '.', 'det_noun', False, 1, 'medium'),
            ]
        else:
            sentences = [
                SentenceTemplate(25, ['DET', 'NOUN', 'VERB', 'DET', 'NOUN'], '.', 'det_noun', False, 1, 'medium'),
                SentenceTemplate(30, ['PRON', 'VERB', 'SCONJ', 'DET', 'NOUN'], '.', 'pronoun', True, 2, 'long'),
            ]

        # Calculate length distribution
        length_counts = {'short': 0, 'medium': 0, 'long': 0}
        for sent in sentences:
            length_counts[sent.length_category] += 1

        total_sents = len(sentences)
        length_distribution = {
            'short': length_counts['short'] / total_sents,
            'medium': length_counts['medium'] / total_sents,
            'long': length_counts['long'] / total_sents
        }

        return ParagraphTemplate(
            sentence_count=len(sentences),
            sentences=sentences,
            total_word_count=sum(s.word_count for s in sentences),
            structural_role=role,
            position_ratio=position_ratio,
            length_distribution=length_distribution,
            short_threshold=self.short_threshold,
            medium_threshold=self.medium_threshold
        )

    def _scale_template_to_input_length(self, template: ParagraphTemplate, input_text: str) -> ParagraphTemplate:
        """
        Scale template to match input paragraph length.

        If input is longer than template, expand template proportionally.
        This ensures output matches input length.
        """
        input_word_count = len(input_text.split())
        input_sentence_count = len(list(self.nlp(input_text).sents))

        # If input is significantly longer, scale up the template
        if input_word_count > template.total_word_count * 1.2:
            scale_factor = input_word_count / max(template.total_word_count, 1)

            # Scale sentence word counts proportionally
            scaled_sentences = []
            for sent in template.sentences:
                scaled_word_count = int(sent.word_count * scale_factor)
                # Recalculate length category
                if scaled_word_count <= self.short_threshold:
                    length_category = 'short'
                elif scaled_word_count <= self.medium_threshold:
                    length_category = 'medium'
                else:
                    length_category = 'long'

                scaled_sent = SentenceTemplate(
                    word_count=scaled_word_count,
                    pos_sequence=sent.pos_sequence,
                    punctuation=sent.punctuation,
                    opener_type=sent.opener_type,
                    has_subordinate_clause=sent.has_subordinate_clause,
                    clause_count=sent.clause_count,
                    length_category=length_category
                )
                scaled_sentences.append(scaled_sent)

            # If input has more sentences, add more sentences to template
            if input_sentence_count > template.sentence_count:
                # Add sentences based on the last sentence pattern
                last_sent = template.sentences[-1] if template.sentences else scaled_sentences[-1]
                for _ in range(input_sentence_count - template.sentence_count):
                    # Use continuation pattern (pronoun or det_noun opener)
                    continuation_sent = SentenceTemplate(
                        word_count=int(last_sent.word_count * 0.8),  # Slightly shorter
                        pos_sequence=last_sent.pos_sequence[:10] if len(last_sent.pos_sequence) > 10 else last_sent.pos_sequence,
                        punctuation=last_sent.punctuation,
                        opener_type='pronoun' if len(scaled_sentences) % 2 == 0 else 'det_noun',
                        has_subordinate_clause=False,
                        clause_count=1,
                        length_category='medium'
                    )
                    scaled_sentences.append(continuation_sent)

            # Recalculate length distribution
            length_counts = {'short': 0, 'medium': 0, 'long': 0}
            for sent in scaled_sentences:
                length_counts[sent.length_category] += 1

            total_sents = len(scaled_sentences)
            length_distribution = {
                'short': length_counts['short'] / total_sents,
                'medium': length_counts['medium'] / total_sents,
                'long': length_counts['long'] / total_sents
            }

            return ParagraphTemplate(
                sentence_count=len(scaled_sentences),
                sentences=scaled_sentences,
                total_word_count=sum(s.word_count for s in scaled_sentences),
                structural_role=template.structural_role,
                position_ratio=template.position_ratio,
                length_distribution=length_distribution,
                short_threshold=template.short_threshold,
                medium_threshold=template.medium_threshold
            )

        return template

    def generate_template_from_context(self,
                                       input_text: str,
                                       preceding_output: Optional[str] = None,
                                       example_selector: Optional[Any] = None,
                                       role: str = 'body',
                                       position_ratio: float = 0.5,
                                       semantic_weight: int = 0,
                                       used_openers: Optional[List[str]] = None,
                                       used_phrases: Optional[List[str]] = None,
                                       paragraph_index: int = 0,
                                       phrase_analyzer: Optional[Any] = None,
                                       output_phrase_counts: Optional[Dict[str, int]] = None,
                                       output_total_paragraphs: int = 0) -> ParagraphTemplate:
        """
        Generate template by finding similar paragraphs from sample based on context.

        Uses:
        1. Input text + preceding output as context for similarity matching
        2. Example selector to find diverse similar paragraphs from sample
        3. Extracts templates from those similar paragraphs dynamically
        4. Enforces variety by checking opener types and phrases
        5. Falls back to distribution-based selection when all candidates share opener
        6. Scores templates by phrase distribution match (NEW)

        Args:
            input_text: Current paragraph being transformed
            preceding_output: Already-generated text (PRIMARY context)
            example_selector: ExampleSelector instance for semantic similarity
            role: Structural role of the paragraph
            position_ratio: Position in document (0-1)
            semantic_weight: Hint about content complexity (0=light, 3=heavy)
            used_openers: List of opener types already used (for variety)
            used_phrases: List of opener phrases already used (for exact repetition avoidance)
            paragraph_index: Current paragraph index
            phrase_analyzer: PhraseDistributionAnalyzer instance for distribution matching
            output_phrase_counts: Current phrase usage counts in output
            output_total_paragraphs: Total paragraphs in output so far

        Returns:
            ParagraphTemplate extracted from contextually similar paragraphs
        """
        if used_openers is None:
            used_openers = []
        if used_phrases is None:
            used_phrases = []

        # Build context: combine input with preceding output
        # Use preceding output as PRIMARY context (user's requirement)
        context_text = input_text
        if preceding_output:
            # Preceding output is the PRIMARY bias for template selection
            context_text = preceding_output + "\n\n" + input_text

        # Use example selector to find similar paragraphs from sample
        if example_selector:
            try:
                # IMPROVEMENT 1: Get larger initial candidate pool for better variety
                # Find diverse similar paragraphs from sample based on context
                # Uses MMR to balance relevance and diversity
                similar_paragraphs = example_selector.select_diverse_examples(
                    context_text,  # Use full context for matching
                    k=12  # Get top 12 diverse similar (increased from 5 for better variety)
                )

                if similar_paragraphs:
                    # Extract templates from similar paragraphs dynamically
                    templates = []
                    template_phrases = []  # Track actual opener phrases
                    for para_text in similar_paragraphs:
                        template = self._extract_paragraph_template(para_text, role, position_ratio)
                        if template and template.total_word_count >= 15:
                            templates.append(template)
                            # Extract actual opener phrase for variety tracking
                            opener_phrase = self._extract_opener_phrase(para_text)
                            template_phrases.append(opener_phrase)

                    if templates:
                        # IMPROVEMENT 2: Check variety early and apply filters before narrowing
                        # Check opener type variety in initial pool
                        opener_types = [t.sentences[0].opener_type for t in templates if t.sentences]
                        unique_openers = set(opener_types)

                        # IMPROVEMENT 3: Trigger distribution-based fallback more aggressively
                        # If low variety detected (few unique openers) OR all openers are recently used
                        low_variety = len(unique_openers) <= 2  # Only 1-2 unique opener types
                        all_recently_used = False
                        if len(unique_openers) == 1 and used_openers:
                            single_opener = list(unique_openers)[0]
                            all_recently_used = single_opener in used_openers[-3:]

                        if low_variety or all_recently_used:
                            # Get additional diverse templates from distribution-based selection
                            dist_templates, dist_phrases = self._get_distribution_based_templates(
                                example_selector, role, position_ratio,
                                used_openers, used_phrases, target_count=8
                            )
                            # Add to pool (avoiding duplicates by phrase)
                            existing_phrases = set(template_phrases)
                            for t, phrase in zip(dist_templates, dist_phrases):
                                if phrase and phrase not in existing_phrases:
                                    templates.append(t)
                                    template_phrases.append(phrase)
                                    existing_phrases.add(phrase)
                                    if len(templates) >= 15:  # Cap at 15 total
                                        break

                        # Re-check variety after adding distribution-based templates
                        opener_types = [t.sentences[0].opener_type for t in templates if t.sentences]
                        unique_openers = set(opener_types)

                        # Apply variety filters EARLY (before sentence count filtering)
                        if used_openers and templates:
                            varied = []
                            varied_phrases = []
                            for t, phrase in zip(templates, template_phrases):
                                if t.sentences and t.sentences[0].opener_type not in used_openers[-3:]:
                                    varied.append(t)
                                    varied_phrases.append(phrase)

                            # If we filtered out too many, get more alternatives
                            if len(varied) < 3 and used_openers:
                                dist_templates, dist_phrases = self._get_distribution_based_templates(
                                    example_selector, role, position_ratio,
                                    used_openers, used_phrases, target_count=10
                                )
                                for t, phrase in zip(dist_templates, dist_phrases):
                                    if t.sentences and t.sentences[0].opener_type not in used_openers[-3:]:
                                        if phrase not in [p for _, p in zip(varied, varied_phrases)]:
                                            varied.append(t)
                                            varied_phrases.append(phrase)
                                            if len(varied) >= 5:  # Ensure at least 5 varied options
                                                break

                            if varied:
                                templates = varied
                                template_phrases = varied_phrases

                        # Apply phrase variety filter EARLY
                        if used_phrases and templates:
                            normalized_used = [p.lower().strip() if p else "" for p in used_phrases[-5:]]
                            phrase_varied = []
                            phrase_varied_phrases = []
                            for t, phrase in zip(templates, template_phrases):
                                normalized_phrase = phrase.lower().strip() if phrase else ""
                                matches_used = False
                                if normalized_phrase:
                                    for used_phrase in normalized_used:
                                        if used_phrase and (normalized_phrase == used_phrase or
                                                          normalized_phrase.startswith(used_phrase) or
                                                          used_phrase.startswith(normalized_phrase)):
                                            matches_used = True
                                            break
                                if not matches_used:
                                    phrase_varied.append(t)
                                    phrase_varied_phrases.append(phrase if phrase else "")

                            # If filtered out too many, get alternatives
                            if len(phrase_varied) < 3 and used_phrases:
                                dist_templates, dist_phrases = self._get_distribution_based_templates(
                                    example_selector, role, position_ratio,
                                    used_openers, used_phrases, target_count=10
                                )
                                for t, phrase in zip(dist_templates, dist_phrases):
                                    normalized_phrase = phrase.lower().strip() if phrase else ""
                                    matches_used = False
                                    if normalized_phrase:
                                        for used_phrase in normalized_used:
                                            if used_phrase and (normalized_phrase == used_phrase or
                                                              normalized_phrase.startswith(used_phrase) or
                                                              used_phrase.startswith(normalized_phrase)):
                                                matches_used = True
                                                break
                                    if not matches_used:
                                        if phrase not in phrase_varied_phrases:
                                            phrase_varied.append(t)
                                            phrase_varied_phrases.append(phrase if phrase else "")
                                            if len(phrase_varied) >= 5:
                                                break

                            if phrase_varied:
                                templates = phrase_varied
                                template_phrases = phrase_varied_phrases

                        # PRIORITY 1: Filter by sentence count - prefer multi-sentence templates
                        input_word_count = len(input_text.split())

                        # Group templates by sentence count
                        by_sentence_count = {}
                        by_sentence_count_phrases = {}
                        for t, phrase in zip(templates, template_phrases):
                            count = t.sentence_count
                            if count not in by_sentence_count:
                                by_sentence_count[count] = []
                                by_sentence_count_phrases[count] = []
                            by_sentence_count[count].append(t)
                            by_sentence_count_phrases[count].append(phrase)

                        # Prefer multi-sentence templates (2+ sentences)
                        preferred_count = None
                        for count in sorted(by_sentence_count.keys(), reverse=True):
                            if count >= 2:  # Prefer 2+ sentences
                                preferred_count = count
                                break

                        # Select template group based on preference and input length
                        if preferred_count:
                            # Use multi-sentence templates
                            templates = by_sentence_count[preferred_count]
                            template_phrases = by_sentence_count_phrases[preferred_count]
                        elif input_word_count < 30:
                            # Allow single-sentence for very short inputs
                            if 1 in by_sentence_count:
                                templates = by_sentence_count[1]
                                template_phrases = by_sentence_count_phrases[1]
                        else:
                            # For longer inputs, reject single-sentence templates
                            multi_sentence_templates = [t for t in templates if t.sentence_count >= 2]
                            if multi_sentence_templates:
                                # Filter phrases to match
                                multi_indices = [i for i, t in enumerate(templates) if t.sentence_count >= 2]
                                templates = multi_sentence_templates
                                template_phrases = [template_phrases[i] for i in multi_indices]
                            else:
                                # Last resort: try distribution-based with multi-sentence requirement
                                dist_templates, dist_phrases = self._get_distribution_based_templates(
                                    example_selector, role, position_ratio,
                                    used_openers, used_phrases, target_count=5
                                )
                                multi_dist = [t for t in dist_templates if t.sentence_count >= 2]
                                if multi_dist:
                                    templates = multi_dist
                                    # Match phrases (approximate)
                                    template_phrases = dist_phrases[:len(multi_dist)]

                        # NOTE: Variety filtering already done earlier, but check again after sentence count filtering
                        # to ensure we still have variety after narrowing by sentence count
                        if templates:
                            opener_types = [t.sentences[0].opener_type for t in templates if t.sentences]
                            unique_openers = set(opener_types)

                            # If we lost variety after sentence count filtering, get more options
                            if len(unique_openers) <= 1 and used_openers:
                                # Try to get more diverse templates with same sentence count
                                preferred_count = templates[0].sentence_count if templates else None
                                dist_templates, dist_phrases = self._get_distribution_based_templates(
                                    example_selector, role, position_ratio,
                                    used_openers, used_phrases, target_count=8
                                )
                                # Filter to same sentence count and different openers
                                for t, phrase in zip(dist_templates, dist_phrases):
                                    if (t.sentence_count == preferred_count and
                                        t.sentences and
                                        t.sentences[0].opener_type not in used_openers[-3:] and
                                        phrase not in template_phrases):
                                        templates.append(t)
                                        template_phrases.append(phrase)
                                        if len(templates) >= 8:  # Ensure good variety
                                            break

                        # MULTI-FACTOR SCORING: Score by length distribution AND phrase distribution
                        if len(templates) > 1:
                            scored_templates = []
                            for template, phrase in zip(templates, template_phrases):
                                # Score 1: Length distribution match (40% weight)
                                length_score = self._score_length_distribution_match(template.length_distribution)

                                # Score 2: Phrase distribution match (30% weight if phrase analyzer available)
                                phrase_score = 1.0  # Default neutral
                                if phrase_analyzer and phrase:
                                    normalized_phrase = phrase.lower().strip()
                                    phrase_score = phrase_analyzer.score_phrase_distribution_match(
                                        normalized_phrase,
                                        output_phrase_counts or {},
                                        output_total_paragraphs,
                                        role=role
                                    )

                                # Score 3: Opener type variety (30% weight)
                                opener_score = 1.0  # Default neutral
                                if template.sentences and used_openers:
                                    opener_type = template.sentences[0].opener_type
                                    # Boost if opener type not recently used
                                    if opener_type not in used_openers[-3:]:
                                        opener_score = 1.5
                                    else:
                                        opener_score = 0.5  # Penalize recently used

                                # Combined score (weighted average)
                                combined_score = (
                                    length_score * 0.4 +
                                    phrase_score * 0.3 +
                                    opener_score * 0.3
                                )
                                scored_templates.append((combined_score, template, phrase))

                            # Sort by combined score (higher is better)
                            scored_templates.sort(key=lambda x: -x[0])

                            # Take top candidates, then select by paragraph index for variety
                            top_scored = [(t, p) for _, t, p in scored_templates[:min(5, len(scored_templates))]]
                            if top_scored:
                                selected, selected_phrase = top_scored[paragraph_index % len(top_scored)]
                            else:
                                selected = None
                        else:
                            selected = templates[0] if templates else None

                        # PRIORITY 2: Split long single sentences if we still have one
                        if selected and selected.sentence_count == 1:
                            input_word_count = len(input_text.split())
                            if selected.sentences[0].word_count > 60 or input_word_count > 50:
                                # Split long single sentence into multiple
                                selected = self._split_long_sentence_template(selected)

                        if selected:
                            # Scale template to match input length (CRITICAL for content preservation)
                            selected = self._scale_template_to_input_length(selected, input_text)

                            # Adjust for semantic weight if needed
                            if semantic_weight > 0:
                                adjusted_sentences = []
                                for sent in selected.sentences:
                                    adjusted_word_count = sent.word_count + (semantic_weight * 3)
                                    # Recalculate length category after adjustment using dynamic thresholds
                                    if adjusted_word_count <= self.short_threshold:
                                        length_category = 'short'
                                    elif adjusted_word_count <= self.medium_threshold:
                                        length_category = 'medium'
                                    else:
                                        length_category = 'long'

                                    adjusted = SentenceTemplate(
                                        word_count=adjusted_word_count,
                                        pos_sequence=sent.pos_sequence,
                                        punctuation=sent.punctuation,
                                        opener_type=sent.opener_type,
                                        has_subordinate_clause=sent.has_subordinate_clause or semantic_weight > 1,
                                        clause_count=max(sent.clause_count, 1 + semantic_weight // 2),
                                        length_category=length_category
                                    )
                                    adjusted_sentences.append(adjusted)

                                # Recalculate length distribution after adjustment
                                length_counts = {'short': 0, 'medium': 0, 'long': 0}
                                for sent in adjusted_sentences:
                                    length_counts[sent.length_category] += 1

                                total_sents = len(adjusted_sentences)
                                adjusted_dist = {
                                    'short': length_counts['short'] / total_sents,
                                    'medium': length_counts['medium'] / total_sents,
                                    'long': length_counts['long'] / total_sents
                                }

                                return ParagraphTemplate(
                                    sentence_count=selected.sentence_count,
                                    sentences=adjusted_sentences,
                                    total_word_count=sum(s.word_count for s in adjusted_sentences),
                                    structural_role=role,
                                    position_ratio=position_ratio,
                                    length_distribution=adjusted_dist,
                                    short_threshold=self.short_threshold,
                                    medium_threshold=self.medium_threshold
                                )

                            return selected
            except Exception as e:
                # Fallback to position-based if context-based fails
                print(f"  [TemplateGen] Context-based selection failed: {e}, using position-based")

        # Fallback to position-based selection
        template = self.generate_template(role, position_ratio, semantic_weight, used_openers, paragraph_index)
        # Scale to match input length
        return self._scale_template_to_input_length(template, input_text)

    def _split_long_sentence_template(self, template: ParagraphTemplate) -> ParagraphTemplate:
        """
        Split a single long sentence template into multiple sentences.

        Only called when template has 1 sentence with >60 words or input is >50 words.
        Creates a natural multi-sentence structure while preserving the overall structure.
        """
        if template.sentence_count != 1:
            return template

        long_sent = template.sentences[0]

        # Determine target sentence count based on word count
        if long_sent.word_count <= 60:
            return template  # Don't split if not long enough

        # Split into 2-3 sentences based on length
        if long_sent.word_count < 120:
            target_sentences = 2
        elif long_sent.word_count < 180:
            target_sentences = 3
        else:
            target_sentences = 4  # Very long sentences get 4

        words_per_sentence = long_sent.word_count // target_sentences

        new_sentences = []
        for i in range(target_sentences):
            # First sentence keeps original opener, others use continuation patterns
            if i == 0:
                opener = long_sent.opener_type
                has_sub = long_sent.has_subordinate_clause
            else:
                # Continuation sentences: use pronoun, det_noun, or adverb
                if i == 1:
                    opener = 'pronoun'  # "It", "This", "They"
                elif i == 2:
                    opener = 'det_noun'  # "The [noun]"
                else:
                    opener = 'adverb'  # "Hence", "Therefore"
                has_sub = False  # Continuation sentences are simpler

            # Distribute clause count across sentences
            clause_count = max(1, long_sent.clause_count // target_sentences)
            if i == 0 and long_sent.has_subordinate_clause:
                clause_count = max(clause_count, 2)  # First sentence can be more complex

            # Use truncated POS sequence for continuation sentences
            pos_seq = long_sent.pos_sequence[:10] if i == 0 else ['PRON', 'VERB', 'DET', 'NOUN', 'VERB']

            word_count = words_per_sentence + (3 if i == 0 else 0)  # First sentence slightly longer
            # Determine length category using dynamic thresholds
            if word_count <= self.short_threshold:
                length_category = 'short'
            elif word_count <= self.medium_threshold:
                length_category = 'medium'
            else:
                length_category = 'long'

            new_sent = SentenceTemplate(
                word_count=word_count,
                pos_sequence=pos_seq,
                punctuation='.' if i < target_sentences - 1 else long_sent.punctuation,
                opener_type=opener,
                has_subordinate_clause=has_sub,
                clause_count=clause_count,
                length_category=length_category
            )
            new_sentences.append(new_sent)

        # Calculate length distribution for split template
        length_counts = {'short': 0, 'medium': 0, 'long': 0}
        for sent in new_sentences:
            length_counts[sent.length_category] += 1

        total_sents = len(new_sentences)
        length_distribution = {
            'short': length_counts['short'] / total_sents,
            'medium': length_counts['medium'] / total_sents,
            'long': length_counts['long'] / total_sents
        }

        return ParagraphTemplate(
            sentence_count=target_sentences,
            sentences=new_sentences,
            total_word_count=template.total_word_count,  # Keep same total
            structural_role=template.structural_role,
            position_ratio=template.position_ratio,
            length_distribution=length_distribution,
            short_threshold=template.short_threshold,
            medium_threshold=template.medium_threshold
        )

    def get_template_prompt(self, role: str, position_ratio: float,
                            claim_count: int = 3,
                            used_openers: Optional[List[str]] = None,
                            used_phrases: Optional[List[str]] = None,
                            paragraph_index: int = 0) -> str:
        """
        Get a template as a prompt constraint for the LLM (position-based).

        Args:
            role: Structural role of the paragraph
            position_ratio: Position in document (0-1)
            claim_count: Number of semantic claims to express
            used_openers: List of opener types already used (for variety)
            used_phrases: List of opener phrases already used (for exact repetition avoidance)
            paragraph_index: Current paragraph index

        Returns:
            String to include in LLM prompt
        """
        # Weight by claim count (more claims = need longer sentences)
        semantic_weight = min(3, claim_count // 2)

        template = self.generate_template(
            role, position_ratio, semantic_weight,
            used_openers=used_openers,
            paragraph_index=paragraph_index
        )
        return template.to_constraint_string(used_phrases=used_phrases)

    def get_template_prompt_from_context(self,
                                         input_text: str,
                                         preceding_output: Optional[str] = None,
                                         example_selector: Optional[Any] = None,
                                         role: str = 'body',
                                         position_ratio: float = 0.5,
                                         claim_count: int = 3,
                                         used_openers: Optional[List[str]] = None,
                                         used_phrases: Optional[List[str]] = None,
                                         paragraph_index: int = 0) -> str:
        """
        Get template prompt using context-based selection (RECOMMENDED).

        Uses preceding output as PRIMARY context for finding similar paragraphs.

        Args:
            input_text: Current paragraph being transformed
            preceding_output: Already-generated text (PRIMARY context)
            example_selector: ExampleSelector instance for semantic similarity
            role: Structural role of the paragraph
            position_ratio: Position in document (0-1)
            claim_count: Number of semantic claims to express
            used_openers: List of opener types already used (for variety)
            used_phrases: List of opener phrases already used (for exact repetition avoidance)
            paragraph_index: Current paragraph index

        Returns:
            String to include in LLM prompt
        """
        semantic_weight = min(3, claim_count // 2)

        template = self.generate_template_from_context(
            input_text=input_text,
            preceding_output=preceding_output,
            example_selector=example_selector,
            role=role,
            position_ratio=position_ratio,
            semantic_weight=semantic_weight,
            used_openers=used_openers,
            used_phrases=used_phrases,
            paragraph_index=paragraph_index
        )

        return template.to_constraint_string(used_phrases=used_phrases)


# Test
if __name__ == '__main__':
    print("=== Template Generator Test ===\n")

    generator = TemplateGenerator()

    # Generate templates for different roles
    print("\n=== Section Opener Template ===")
    print(generator.get_template_prompt('section_opener', 0.0, claim_count=4))

    print("\n=== Body Paragraph Template ===")
    print(generator.get_template_prompt('body', 0.5, claim_count=2))

    print("\n=== Closing Paragraph Template ===")
    print(generator.get_template_prompt('closer', 0.95, claim_count=3))

