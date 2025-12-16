"""
Structural Analyzer Module

Analyzes document structure and maps patterns to structural roles.
Patterns are position-dependent:
- Section openers use different patterns than body sentences
- Paragraph openers differ from supporting sentences
- Transitions have specific markers

This module:
1. Analyzes sample text structure to extract role-based patterns
2. Caches analysis in SQLite for performance
3. Maps input text sentences to structural roles
4. Provides appropriate patterns for each role
"""

import sqlite3
import json
import re
import hashlib
import spacy
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, asdict, field
from collections import Counter, defaultdict


@dataclass
class StructuralRole:
    """The structural role of a text element."""
    level: str           # 'document', 'section', 'paragraph', 'sentence'
    role: str            # 'opener', 'body', 'transition', 'conclusion', 'support'
    position_in_parent: int  # 0 = first, -1 = last, else middle
    parent_role: Optional[str] = None  # Role of containing element


@dataclass
class StructuralPattern:
    """A pattern associated with a structural role."""
    role_level: str      # 'section_opener', 'paragraph_opener', etc.
    pattern_type: str    # 'phrase', 'construction', 'length', 'connector'
    pattern: str         # The actual pattern or template
    frequency: int       # How often this pattern appears in this role
    examples: List[str]  # Example sentences using this pattern


@dataclass
class SentenceAnalysis:
    """Analysis of a single sentence's structure."""
    text: str
    index: int           # Position in document
    paragraph_index: int # Which paragraph
    position_in_para: str  # 'first', 'middle', 'last', 'only'
    section_index: int   # Which section (if detectable)
    is_section_opener: bool
    word_count: int
    structural_role: StructuralRole


@dataclass
class DocumentStructure:
    """Complete structural analysis of a document."""
    sections: List[Dict[str, Any]]
    paragraphs: List[Dict[str, Any]]
    sentences: List[SentenceAnalysis]
    role_patterns: Dict[str, List[StructuralPattern]]  # role -> patterns


@dataclass
class TransformationHint:
    """Specific hint for transforming a sentence."""
    sentence_index: int
    current_text: str
    structural_role: str
    expected_patterns: List[str]
    issue: str           # What's wrong
    suggestion: str      # How to fix it
    priority: int        # 1=critical, 2=important, 3=minor


class StructuralAnalyzer:
    """
    Analyzes document structure and extracts role-based patterns.

    Key insight: Patterns are not random - they depend on WHERE
    in the document structure the text appears:
    - Section openers often use contrastive patterns
    - Paragraph openers introduce topics
    - Body sentences provide support/evidence
    - Conclusions summarize with causal markers
    """

    DB_PATH = Path(__file__).parent / "structural_cache.db"

    def __init__(self):
        """Initialize the structural analyzer."""
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            import subprocess
            subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"])
            self.nlp = spacy.load("en_core_web_sm")

        self._init_db()

        # Section header patterns (detect document structure)
        self.section_patterns = [
            r'^[0-9]+\)',           # 1) 2) 3)
            r'^[a-z]\)',            # a) b) c)
            r'^\d+\.\s+\w',         # 1. Title
            r'^[IVX]+\.',           # I. II. III.
            r'^Chapter\s+\d+',      # Chapter 1
            r'^Section\s+\d+',      # Section 1
            r'^#+\s+',              # Markdown headers
        ]

        # Transition markers that indicate structural boundaries
        self.transition_markers = {
            'section_opener': ['contrary to', 'the principal features', 'it is easy to understand'],
            'paragraph_opener': ['hence', 'further', 'thus', 'therefore', 'consequently'],
            'conclusion': ['such is', 'this means that', 'it follows that', 'in this connection'],
            'contrast': ['however', 'but', 'yet', 'on the other hand', 'contrary to'],
            'addition': ['moreover', 'furthermore', 'in addition', 'likewise'],
            'example': ['for example', 'for instance', 'such as'],
        }

    def _init_db(self):
        """Initialize SQLite database for caching."""
        conn = sqlite3.connect(self.DB_PATH)
        cursor = conn.cursor()

        cursor.execute('''
            CREATE TABLE IF NOT EXISTS structural_profiles (
                sample_hash TEXT PRIMARY KEY,
                profile_json TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')

        cursor.execute('''
            CREATE TABLE IF NOT EXISTS role_patterns (
                id INTEGER PRIMARY KEY,
                sample_hash TEXT,
                role_level TEXT,
                pattern_type TEXT,
                pattern TEXT,
                frequency INTEGER,
                examples_json TEXT,
                FOREIGN KEY (sample_hash) REFERENCES structural_profiles(sample_hash)
            )
        ''')

        conn.commit()
        conn.close()

    def _get_sample_hash(self, text: str) -> str:
        """Generate hash of sample text for cache lookup."""
        return hashlib.md5(text.encode()).hexdigest()

    def _load_cached_profile(self, sample_hash: str) -> Optional[Dict]:
        """Load cached structural profile if exists."""
        conn = sqlite3.connect(self.DB_PATH)
        cursor = conn.cursor()

        cursor.execute(
            'SELECT profile_json FROM structural_profiles WHERE sample_hash = ?',
            (sample_hash,)
        )
        result = cursor.fetchone()
        conn.close()

        if result:
            return json.loads(result[0])
        return None

    def _save_profile_to_cache(self, sample_hash: str, profile: Dict, patterns: List[StructuralPattern]):
        """Save structural profile to cache."""
        conn = sqlite3.connect(self.DB_PATH)
        cursor = conn.cursor()

        # Save main profile
        cursor.execute(
            'INSERT OR REPLACE INTO structural_profiles (sample_hash, profile_json) VALUES (?, ?)',
            (sample_hash, json.dumps(profile))
        )

        # Clear old patterns
        cursor.execute('DELETE FROM role_patterns WHERE sample_hash = ?', (sample_hash,))

        # Save patterns
        for pattern in patterns:
            cursor.execute('''
                INSERT INTO role_patterns
                (sample_hash, role_level, pattern_type, pattern, frequency, examples_json)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (
                sample_hash,
                pattern.role_level,
                pattern.pattern_type,
                pattern.pattern,
                pattern.frequency,
                json.dumps(pattern.examples)
            ))

        conn.commit()
        conn.close()

    def _load_patterns_from_cache(self, sample_hash: str) -> Dict[str, List[StructuralPattern]]:
        """Load role patterns from cache."""
        conn = sqlite3.connect(self.DB_PATH)
        cursor = conn.cursor()

        cursor.execute(
            'SELECT role_level, pattern_type, pattern, frequency, examples_json FROM role_patterns WHERE sample_hash = ?',
            (sample_hash,)
        )
        results = cursor.fetchall()
        conn.close()

        patterns = defaultdict(list)
        for row in results:
            role_level, pattern_type, pattern, frequency, examples_json = row
            patterns[role_level].append(StructuralPattern(
                role_level=role_level,
                pattern_type=pattern_type,
                pattern=pattern,
                frequency=frequency,
                examples=json.loads(examples_json)
            ))

        return dict(patterns)

    def analyze_sample(self, sample_text: str, force_refresh: bool = False) -> Dict[str, List[StructuralPattern]]:
        """
        Analyze sample text and extract patterns by structural role.
        Uses cache if available, otherwise generates and caches.

        Returns dict mapping role names to list of patterns.
        """
        sample_hash = self._get_sample_hash(sample_text)

        # Check cache unless force refresh
        if not force_refresh:
            cached_patterns = self._load_patterns_from_cache(sample_hash)
            if cached_patterns:
                print(f"  [Cache] Loaded structural patterns from cache")
                return cached_patterns

        print(f"  [Analyze] Extracting structural patterns from sample...")

        # Analyze document structure
        structure = self._analyze_document_structure(sample_text)

        # Extract patterns by role
        role_patterns = self._extract_patterns_by_role(structure, sample_text)

        # Cache results
        profile = {
            'num_sections': len(structure['sections']),
            'num_paragraphs': len(structure['paragraphs']),
            'num_sentences': len(structure['sentences']),
        }

        # Flatten patterns for caching
        all_patterns = []
        for patterns in role_patterns.values():
            all_patterns.extend(patterns)

        self._save_profile_to_cache(sample_hash, profile, all_patterns)
        print(f"  [Cache] Saved {len(all_patterns)} patterns to cache")

        return role_patterns

    def _analyze_document_structure(self, text: str) -> Dict:
        """Analyze the structural elements of a document."""
        lines = text.split('\n')

        # Detect sections
        sections = []
        current_section = {'title': 'Introduction', 'start_line': 0, 'paragraphs': []}

        for i, line in enumerate(lines):
            # Check for section headers
            is_header = False
            for pattern in self.section_patterns:
                if re.match(pattern, line.strip()):
                    is_header = True
                    break

            if is_header and line.strip():
                if current_section['paragraphs'] or current_section['title'] != 'Introduction':
                    sections.append(current_section)
                current_section = {'title': line.strip()[:100], 'start_line': i, 'paragraphs': []}

        sections.append(current_section)

        # Detect paragraphs
        paragraphs = []
        current_para = []
        current_para_start = 0

        for i, line in enumerate(lines):
            if line.strip():
                if not current_para:
                    current_para_start = i
                current_para.append(line.strip())
            elif current_para:
                para_text = ' '.join(current_para)
                if len(para_text) > 30:  # Minimum meaningful paragraph
                    paragraphs.append({
                        'text': para_text,
                        'start_line': current_para_start,
                        'end_line': i - 1
                    })
                current_para = []

        if current_para:
            para_text = ' '.join(current_para)
            if len(para_text) > 30:
                paragraphs.append({
                    'text': para_text,
                    'start_line': current_para_start,
                    'end_line': len(lines) - 1
                })

        # Assign paragraphs to sections
        for para in paragraphs:
            for section in sections:
                if para['start_line'] >= section['start_line']:
                    section['paragraphs'].append(para)

        # Analyze sentences within paragraphs
        sentences = []
        for para_idx, para in enumerate(paragraphs):
            doc = self.nlp(para['text'])
            para_sentences = list(doc.sents)

            for sent_idx, sent in enumerate(para_sentences):
                # Determine position in paragraph
                if len(para_sentences) == 1:
                    position = 'only'
                elif sent_idx == 0:
                    position = 'first'
                elif sent_idx == len(para_sentences) - 1:
                    position = 'last'
                else:
                    position = 'middle'

                # Check if this is a section opener
                is_section_opener = (para_idx == 0 or
                    any(para['start_line'] == s['start_line'] for s in sections if s['paragraphs']))

                # Determine section index
                section_idx = 0
                for idx, section in enumerate(sections):
                    if para in section.get('paragraphs', []):
                        section_idx = idx
                        break

                sentences.append({
                    'text': sent.text.strip(),
                    'index': len(sentences),
                    'paragraph_index': para_idx,
                    'position_in_para': position,
                    'section_index': section_idx,
                    'is_section_opener': is_section_opener and sent_idx == 0,
                    'word_count': len([t for t in sent if not t.is_punct])
                })

        return {
            'sections': sections,
            'paragraphs': paragraphs,
            'sentences': sentences
        }

    def _extract_patterns_by_role(self, structure: Dict, text: str) -> Dict[str, List[StructuralPattern]]:
        """Extract patterns grouped by structural role."""
        patterns = defaultdict(list)
        text_lower = text.lower()

        # Categorize sentences by role
        role_sentences = defaultdict(list)

        for sent in structure['sentences']:
            sent_text = sent['text']
            sent_lower = sent_text.lower()

            # Determine primary role
            if sent['is_section_opener']:
                role_sentences['section_opener'].append(sent_text)
            elif sent['position_in_para'] == 'first':
                role_sentences['paragraph_opener'].append(sent_text)
            elif sent['position_in_para'] == 'last':
                role_sentences['paragraph_closer'].append(sent_text)
            else:
                role_sentences['body'].append(sent_text)

            # Also categorize by detected function
            for marker_type, markers in self.transition_markers.items():
                for marker in markers:
                    if marker in sent_lower:
                        role_sentences[f'contains_{marker_type}'].append(sent_text)
                        break

        # Extract patterns for each role
        for role, sentences in role_sentences.items():
            if not sentences:
                continue

            # Phrase patterns (common n-grams)
            phrase_patterns = self._extract_phrase_patterns(sentences)
            for phrase, freq in phrase_patterns[:10]:
                patterns[role].append(StructuralPattern(
                    role_level=role,
                    pattern_type='phrase',
                    pattern=phrase,
                    frequency=freq,
                    examples=[s for s in sentences if phrase in s.lower()][:3]
                ))

            # Opening patterns
            opener_patterns = self._extract_opener_patterns(sentences)
            for opener, freq in opener_patterns[:5]:
                patterns[role].append(StructuralPattern(
                    role_level=role,
                    pattern_type='opener',
                    pattern=opener,
                    frequency=freq,
                    examples=[s for s in sentences if s.lower().startswith(opener.lower())][:3]
                ))

            # Length patterns
            lengths = [len(s.split()) for s in sentences]
            avg_length = sum(lengths) / len(lengths) if lengths else 0
            patterns[role].append(StructuralPattern(
                role_level=role,
                pattern_type='length',
                pattern=f"avg_{int(avg_length)}_words",
                frequency=len(sentences),
                examples=[]
            ))

        return dict(patterns)

    def _extract_phrase_patterns(self, sentences: List[str]) -> List[Tuple[str, int]]:
        """Extract common phrase patterns from sentences."""
        phrase_counts = Counter()

        for sent in sentences:
            words = sent.lower().split()
            # Extract 2-4 grams
            for n in range(2, 5):
                for i in range(len(words) - n + 1):
                    phrase = ' '.join(words[i:i+n])
                    # Filter out very generic phrases
                    if not all(w in ['the', 'a', 'an', 'is', 'are', 'of', 'to', 'in'] for w in words[i:i+n]):
                        phrase_counts[phrase] += 1

        # Return phrases appearing 2+ times
        return [(p, c) for p, c in phrase_counts.most_common(20) if c >= 2]

    def _extract_opener_patterns(self, sentences: List[str]) -> List[Tuple[str, int]]:
        """Extract common sentence opener patterns."""
        opener_counts = Counter()

        for sent in sentences:
            words = sent.split()
            if len(words) >= 2:
                # Get first 1-3 words as opener pattern
                for n in range(1, 4):
                    if len(words) >= n:
                        opener = ' '.join(words[:n])
                        # Normalize
                        opener = opener.rstrip('.,;:')
                        opener_counts[opener] += 1

        return [(o, c) for o, c in opener_counts.most_common(10) if c >= 2]

    def analyze_input_structure(self, input_text: str) -> List[SentenceAnalysis]:
        """
        Analyze input text and assign structural roles to each sentence.
        """
        structure = self._analyze_document_structure(input_text)

        analyses = []
        for sent_data in structure['sentences']:
            role = StructuralRole(
                level='sentence',
                role=self._determine_sentence_role(sent_data),
                position_in_parent=sent_data['index'],
                parent_role='paragraph'
            )

            analyses.append(SentenceAnalysis(
                text=sent_data['text'],
                index=sent_data['index'],
                paragraph_index=sent_data['paragraph_index'],
                position_in_para=sent_data['position_in_para'],
                section_index=sent_data['section_index'],
                is_section_opener=sent_data['is_section_opener'],
                word_count=sent_data['word_count'],
                structural_role=role
            ))

        return analyses

    def _determine_sentence_role(self, sent_data: Dict) -> str:
        """Determine the structural role of a sentence."""
        if sent_data['is_section_opener']:
            return 'section_opener'
        elif sent_data['position_in_para'] == 'first':
            return 'paragraph_opener'
        elif sent_data['position_in_para'] == 'last':
            return 'paragraph_closer'
        elif sent_data['position_in_para'] == 'only':
            return 'standalone'
        else:
            return 'body'

    def get_patterns_for_role(self, role: str, role_patterns: Dict[str, List[StructuralPattern]]) -> List[StructuralPattern]:
        """Get patterns appropriate for a given structural role."""
        patterns = role_patterns.get(role, [])

        # Also include related patterns
        if role == 'section_opener':
            patterns.extend(role_patterns.get('contains_section_opener', []))
            patterns.extend(role_patterns.get('contains_contrast', []))
        elif role == 'paragraph_opener':
            patterns.extend(role_patterns.get('contains_paragraph_opener', []))
        elif role == 'paragraph_closer':
            patterns.extend(role_patterns.get('contains_conclusion', []))

        return patterns

    def generate_transformation_hints(
        self,
        input_sentences: List[SentenceAnalysis],
        output_text: str,
        role_patterns: Dict[str, List[StructuralPattern]]
    ) -> List[TransformationHint]:
        """
        Compare output against expected patterns and generate specific hints.
        """
        hints = []
        output_lower = output_text.lower()

        # Split output into sentences for comparison
        doc = self.nlp(output_text)
        output_sentences = [s.text.strip() for s in doc.sents]

        for i, input_sent in enumerate(input_sentences):
            role = input_sent.structural_role.role
            expected_patterns = self.get_patterns_for_role(role, role_patterns)

            # Get corresponding output sentence (if exists)
            output_sent = output_sentences[i] if i < len(output_sentences) else ""
            output_sent_lower = output_sent.lower()

            # Check for missing patterns
            missing_patterns = []
            for pattern in expected_patterns:
                if pattern.pattern_type == 'phrase':
                    if pattern.pattern not in output_sent_lower and pattern.frequency >= 3:
                        missing_patterns.append(pattern)
                elif pattern.pattern_type == 'opener':
                    if role in ['section_opener', 'paragraph_opener']:
                        if not output_sent_lower.startswith(pattern.pattern.lower()):
                            missing_patterns.append(pattern)
                elif pattern.pattern_type == 'length':
                    expected_length = int(pattern.pattern.split('_')[1])
                    actual_length = len(output_sent.split())
                    if actual_length < expected_length * 0.6:  # More than 40% shorter
                        missing_patterns.append(pattern)

            # Generate hints for significant issues
            if missing_patterns:
                # Prioritize by pattern type
                phrase_missing = [p for p in missing_patterns if p.pattern_type == 'phrase']
                opener_missing = [p for p in missing_patterns if p.pattern_type == 'opener']
                length_issues = [p for p in missing_patterns if p.pattern_type == 'length']

                if opener_missing and role in ['section_opener', 'paragraph_opener']:
                    pattern = opener_missing[0]
                    hints.append(TransformationHint(
                        sentence_index=i,
                        current_text=output_sent[:100],
                        structural_role=role,
                        expected_patterns=[p.pattern for p in opener_missing[:3]],
                        issue=f"Sentence should open with pattern like '{pattern.pattern}'",
                        suggestion=f"Rewrite to start with: '{pattern.pattern}...' Example: \"{pattern.examples[0][:80]}...\"" if pattern.examples else f"Start with '{pattern.pattern}'",
                        priority=1
                    ))

                if phrase_missing:
                    pattern = phrase_missing[0]
                    hints.append(TransformationHint(
                        sentence_index=i,
                        current_text=output_sent[:100],
                        structural_role=role,
                        expected_patterns=[p.pattern for p in phrase_missing[:3]],
                        issue=f"Missing characteristic phrase '{pattern.pattern}'",
                        suggestion=f"Incorporate '{pattern.pattern}' into this sentence. Example usage: \"{pattern.examples[0][:80]}...\"" if pattern.examples else f"Use phrase '{pattern.pattern}'",
                        priority=2
                    ))

                if length_issues:
                    pattern = length_issues[0]
                    expected_length = int(pattern.pattern.split('_')[1])
                    hints.append(TransformationHint(
                        sentence_index=i,
                        current_text=output_sent[:100],
                        structural_role=role,
                        expected_patterns=[],
                        issue=f"Sentence too short ({len(output_sent.split())} words, expected ~{expected_length})",
                        suggestion=f"Expand sentence with additional clauses, subordinate phrases, or elaboration to reach ~{expected_length} words",
                        priority=2
                    ))

        # Sort by priority
        hints.sort(key=lambda h: h.priority)

        return hints


def analyze_structure(text: str) -> Dict[str, List[StructuralPattern]]:
    """Convenience function to analyze sample text structure."""
    analyzer = StructuralAnalyzer()
    return analyzer.analyze_sample(text)


# Test function
if __name__ == '__main__':
    from pathlib import Path

    sample_path = Path(__file__).parent / "prompts" / "sample.txt"
    if sample_path.exists():
        with open(sample_path, 'r', encoding='utf-8') as f:
            sample_text = f.read()

        print("=== Structural Analysis Test ===\n")

        analyzer = StructuralAnalyzer()

        # Analyze sample
        role_patterns = analyzer.analyze_sample(sample_text)

        print(f"\nExtracted patterns by role:")
        for role, patterns in role_patterns.items():
            print(f"\n  {role}:")
            for p in patterns[:3]:
                print(f"    - [{p.pattern_type}] {p.pattern} ({p.frequency}x)")

        # Test input analysis
        test_input = """Human experience confirms the rule of finitude. The biological cycle defines our reality.

A truly finite universe must exist within a larger context. We must consider this possibility."""

        print(f"\n\n=== Input Structure Analysis ===")
        input_analysis = analyzer.analyze_input_structure(test_input)
        for sent in input_analysis:
            print(f"  [{sent.structural_role.role}] {sent.text[:60]}...")
    else:
        print(f"Sample file not found at {sample_path}")

