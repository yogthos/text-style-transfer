"""Extract clause-level syntactic patterns from sentences.

These patterns give the LLM a structural skeleton to follow, improving
sentence variety and reducing the mechanical, repetitive structures
that flag AI-generated text.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from collections import Counter

from ..utils.nlp import get_nlp
from ..utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class ClausePattern:
    """A syntactic clause pattern extracted from corpus."""

    template: str  # e.g., "[SCONJ] [clause], [main] [relcl]"
    structure_type: str  # simple, compound, complex, compound_complex
    clause_depth: int  # How many levels of subordination
    example: str  # Original sentence this came from
    frequency: int = 1


@dataclass
class ClausePatternProfile:
    """Collection of clause patterns for an author."""

    # Patterns organized by structure type
    patterns_by_type: Dict[str, List[ClausePattern]] = field(default_factory=dict)

    # Top clause starters for each structure type
    clause_starters: Dict[str, List[str]] = field(default_factory=dict)

    # Average clause depth by structure type
    avg_clause_depth: Dict[str, float] = field(default_factory=dict)

    def get_template_for_type(self, structure_type: str) -> Optional[str]:
        """Get a random template for the given structure type."""
        import random
        patterns = self.patterns_by_type.get(structure_type, [])
        if patterns:
            return random.choice(patterns).template
        return None

    def get_starter_for_type(self, structure_type: str) -> Optional[str]:
        """Get a common clause starter for the structure type."""
        import random
        starters = self.clause_starters.get(structure_type, [])
        if starters:
            return random.choice(starters[:5])  # Top 5
        return None


class ClausePatternExtractor:
    """Extract clause patterns from sentences."""

    def __init__(self):
        self._nlp = None

    @property
    def nlp(self):
        if self._nlp is None:
            self._nlp = get_nlp()
        return self._nlp

    def extract_pattern(self, sentence: str) -> ClausePattern:
        """Extract clause pattern from a single sentence."""
        doc = self.nlp(sentence)

        # Determine structure type
        structure_type = self._classify_structure(doc)

        # Calculate clause depth
        clause_depth = self._calculate_clause_depth(doc)

        # Generate template
        template = self._generate_template(doc)

        return ClausePattern(
            template=template,
            structure_type=structure_type,
            clause_depth=clause_depth,
            example=sentence[:100] + "..." if len(sentence) > 100 else sentence,
        )

    def extract_profile(
        self,
        sentences_by_type: Dict[str, List[str]]
    ) -> ClausePatternProfile:
        """Extract clause pattern profile from categorized sentences."""
        profile = ClausePatternProfile()

        for struct_type, sentences in sentences_by_type.items():
            patterns = []
            starters = Counter()
            total_depth = 0

            for sent in sentences[:30]:  # Limit to 30 per type
                try:
                    pattern = self.extract_pattern(sent)
                    patterns.append(pattern)
                    total_depth += pattern.clause_depth

                    # Extract starter
                    doc = self.nlp(sent)
                    if len(doc) > 0:
                        first_token = doc[0]
                        if first_token.pos_ in ('SCONJ', 'CCONJ', 'ADV'):
                            starters[first_token.text.lower()] += 1
                        elif first_token.pos_ == 'VERB' and first_token.dep_ == 'ROOT':
                            starters['[imperative]'] += 1
                except Exception as e:
                    logger.debug(f"Error extracting pattern: {e}")
                    continue

            profile.patterns_by_type[struct_type] = patterns
            profile.clause_starters[struct_type] = [
                w for w, _ in starters.most_common(10)
            ]
            profile.avg_clause_depth[struct_type] = (
                total_depth / len(patterns) if patterns else 1.0
            )

        return profile

    def _classify_structure(self, doc) -> str:
        """Classify sentence structure type."""
        num_verbs = sum(1 for t in doc if t.pos_ == 'VERB' and
                       t.dep_ in ('ROOT', 'conj', 'ccomp', 'xcomp', 'advcl', 'relcl'))
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

    def _calculate_clause_depth(self, doc) -> int:
        """Calculate max depth of clause subordination."""
        max_depth = 0

        for token in doc:
            depth = 0
            current = token

            # Walk up the dependency tree counting clause relations
            while current.head != current:
                if current.dep_ in ('advcl', 'relcl', 'ccomp', 'xcomp', 'acl'):
                    depth += 1
                current = current.head

            max_depth = max(max_depth, depth)

        return max_depth + 1  # +1 for main clause

    def _generate_template(self, doc) -> str:
        """Generate a syntactic template from the sentence.

        Creates a pattern like:
        "[SCONJ] [clause], [NP] [VP] [relcl]"
        """
        parts = []
        seen_main = False

        for token in doc:
            # Skip punctuation
            if token.is_punct:
                if token.text == ',':
                    parts.append(',')
                continue

            # Subordinating conjunctions
            if token.pos_ == 'SCONJ':
                parts.append(f"[{token.text.lower()}]")

            # Coordinating conjunctions
            elif token.pos_ == 'CCONJ':
                parts.append(f"[{token.text.lower()}]")

            # Main clause verb
            elif token.dep_ == 'ROOT' and token.pos_ == 'VERB':
                if not seen_main:
                    parts.append('[MAIN-VP]')
                    seen_main = True

            # Relative clauses
            elif token.dep_ == 'relcl':
                parts.append('[relcl]')

            # Adverbial clauses
            elif token.dep_ == 'advcl':
                parts.append('[advcl]')

            # Subject
            elif token.dep_ in ('nsubj', 'nsubjpass') and not seen_main:
                parts.append('[SUBJ]')

            # Direct object
            elif token.dep_ == 'dobj':
                parts.append('[OBJ]')

        # Simplify consecutive duplicates
        simplified = []
        for part in parts:
            if not simplified or simplified[-1] != part:
                simplified.append(part)

        return ' '.join(simplified) if simplified else '[sentence]'


def get_clause_template_for_prompt(
    structure_type: str,
    profile: Optional[ClausePatternProfile] = None
) -> str:
    """Get a clause template string for injection into prompts."""
    if profile:
        template = profile.get_template_for_type(structure_type)
        starter = profile.get_starter_for_type(structure_type)

        hints = []
        if template:
            hints.append(f"Pattern: {template}")
        if starter:
            hints.append(f"Consider starting with: '{starter}'")
        if structure_type in profile.avg_clause_depth:
            depth = profile.avg_clause_depth[structure_type]
            if depth > 1.5:
                hints.append(f"Use nested clauses (depth ~{depth:.0f})")

        return "; ".join(hints) if hints else ""

    # Fallback: generic templates
    templates = {
        'simple': "[SUBJ] [MAIN-VP] [OBJ]",
        'compound': "[clause], [and/but] [clause]",
        'complex': "[SCONJ] [clause], [MAIN-VP]",
        'compound_complex': "[SCONJ] [clause], [main], [and] [clause] [relcl]",
    }
    return f"Pattern: {templates.get(structure_type, '')}"
