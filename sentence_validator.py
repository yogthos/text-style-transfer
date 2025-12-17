"""
Sentence Validator Module

Validates individual sentences against their assigned templates to ensure
they match the statistical distribution from the sample text.
"""

import spacy
from typing import List, Optional
from dataclasses import dataclass
from template_generator import SentenceTemplate


@dataclass
class ValidationIssue:
    """Represents a validation issue found in a sentence."""
    description: str
    fix_guidance: str


class SentenceValidator:
    """
    Validates sentences against their assigned templates.

    Checks multiple dimensions:
    - Word count (within tolerance)
    - Opener type (exact match)
    - Clause complexity (at least template requirement)
    - POS pattern (loose match on first few tags)
    - Length category (short/medium/long)
    """

    def __init__(self, word_count_tolerance: float = 0.35, require_exact_opener: bool = False):
        """
        Initialize sentence validator.

        Args:
            word_count_tolerance: Tolerance for word count (0.35 = ±35%)
            require_exact_opener: Whether opener type must match exactly (default: False for flexibility)
        """
        self.word_count_tolerance = word_count_tolerance
        self.require_exact_opener = require_exact_opener

        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            import subprocess
            subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"])
            self.nlp = spacy.load("en_core_web_sm")

    def validate_sentence(self,
                         sentence_text: str,
                         template: SentenceTemplate,
                         tolerance: Optional[float] = None) -> List[ValidationIssue]:
        """
        Validate a sentence against its template.

        Args:
            sentence_text: The sentence to validate
            template: The template it should match
            tolerance: Optional override for word count tolerance

        Returns:
            List of validation issues (empty if sentence matches)
        """
        issues = []
        tolerance = tolerance or self.word_count_tolerance

        if not sentence_text or not sentence_text.strip():
            return [ValidationIssue(
                "No sentence found",
                "Generate a valid sentence"
            )]

        doc = self.nlp(sentence_text)
        sentences = list(doc.sents)

        if not sentences:
            return [ValidationIssue(
                "No sentence found",
                "Generate a valid sentence"
            )]

        actual_sent = sentences[0]
        tokens = [t for t in actual_sent if not t.is_space and not t.is_punct]

        if not tokens:
            return [ValidationIssue(
                "Sentence has no words",
                "Generate a sentence with actual content"
            )]

        # Check 1: Word count (within tolerance)
        actual_word_count = len(tokens)
        expected_word_count = template.word_count
        tolerance_amount = expected_word_count * tolerance

        if abs(actual_word_count - expected_word_count) > tolerance_amount:
            issues.append(ValidationIssue(
                f"Word count mismatch: {actual_word_count} vs expected ~{expected_word_count} (±{int(tolerance_amount)})",
                f"Generate a sentence with approximately {expected_word_count} words (current: {actual_word_count})"
            ))

        # Check 2: Opener type (more flexible matching)
        actual_opener = self._classify_opener(tokens)
        if actual_opener != template.opener_type:
            # Check if openers are similar (both contrastive, both declarative, etc.)
            are_similar = self._openers_are_similar(actual_opener, template.opener_type)

            if self.require_exact_opener or not are_similar:
                # Only mark as critical if completely different types
                is_critical = not are_similar and not self._is_minor_opener_mismatch(actual_opener, template.opener_type)
                opener_example = self._get_opener_example(template.opener_type)
                priority_desc = "CRITICAL" if is_critical else "minor"
                issues.append(ValidationIssue(
                    f"Opener type mismatch ({priority_desc}): '{actual_opener}' vs expected '{template.opener_type}'",
                    f"Start sentence with {template.opener_type} opener. Example: {opener_example}"
                ))

        # Check 3: Clause complexity
        actual_clauses = self._count_clauses(actual_sent)
        if actual_clauses < template.clause_count:
            clause_guidance = "subordinate clause" if template.has_subordinate_clause else "coordination"
            issues.append(ValidationIssue(
                f"Clause complexity too low: {actual_clauses} clause(s) vs expected {template.clause_count}",
                f"Use {template.clause_count} clause(s) with {clause_guidance}"
            ))

        # Check 4: Subordinate clause requirement
        if template.has_subordinate_clause:
            has_subordinate = self._has_subordinate_clause(actual_sent)
            if not has_subordinate:
                issues.append(ValidationIssue(
                    "Missing subordinate clause",
                    "Add a subordinate clause (e.g., 'which...', 'that...', 'when...', 'if...')"
                ))

        # Check 5: POS pattern (loose match - check first few tags)
        actual_pos = [t.pos_ for t in tokens[:8]]
        expected_pos = template.pos_sequence[:8] if template.pos_sequence else []

        if expected_pos and not self._pos_patterns_match(actual_pos, expected_pos):
            pos_preview = ' → '.join(expected_pos[:5])
            issues.append(ValidationIssue(
                f"POS pattern mismatch in first 8 words",
                f"Follow this pattern: {pos_preview}..."
            ))

        # Check 6: Length category
        actual_category = self._get_length_category(actual_word_count, template)
        if actual_category != template.length_category:
            # This is less critical, so only warn if very off
            if abs(actual_word_count - expected_word_count) > expected_word_count * 0.4:
                issues.append(ValidationIssue(
                    f"Length category mismatch: {actual_category} vs expected {template.length_category}",
                    f"Generate a {template.length_category} sentence (~{expected_word_count} words)"
                ))

        # Check 7: Word-level template validation (if word slots are present)
        if template.has_word_slots and template.word_slots:
            word_level_issues = self._validate_word_level_template(actual_sent, template)
            issues.extend(word_level_issues)

        return issues

    def _classify_opener(self, tokens: List) -> str:
        """Classify the opener type of a sentence."""
        if not tokens:
            return 'other'

        first_token = tokens[0]
        pos = first_token.pos_
        text_lower = first_token.text.lower()

        # Check for determiner + noun
        if pos == 'DET' and len(tokens) > 1 and tokens[1].pos_ in ('NOUN', 'PROPN'):
            return 'det_noun'

        # Check for pronoun
        if pos == 'PRON' or text_lower in ('it', 'this', 'that', 'these', 'those', 'they', 'we', 'you'):
            return 'pronoun'

        # Check for adverb
        if pos == 'ADV' or text_lower in ('hence', 'therefore', 'thus', 'consequently', 'accordingly'):
            return 'adverb'

        # Check for conjunction
        if pos == 'CCONJ' or pos == 'SCONJ' or text_lower in ('but', 'and', 'yet', 'however', 'although', 'while'):
            return 'conjunction'

        # Check for prepositional phrase
        if pos == 'ADP' or text_lower in ('in', 'on', 'at', 'for', 'with', 'by', 'from', 'to', 'of', 'as', 'contrary'):
            return 'prep_phrase'

        # Check for gerund (-ing verb)
        if pos == 'VERB' and first_token.text.endswith('ing'):
            return 'verb_ing'

        # Check for noun
        if pos in ('NOUN', 'PROPN'):
            return 'noun'

        return 'other'

    def _get_opener_example(self, opener_type: str) -> str:
        """Get an example phrase for an opener type."""
        examples = {
            'noun': '"The principle..." or "Marxism..."',
            'det_noun': '"The theory..." or "A concept..."',
            'adverb': '"Therefore..." or "Hence..."',
            'conjunction': '"But..." or "And..." or "However..."',
            'prep_phrase': '"In this..." or "Contrary to..."',
            'pronoun': '"It..." or "This..." or "They..."',
            'verb_ing': '"Understanding..." or "Examining..."',
        }
        return examples.get(opener_type, f'"{opener_type} opener"')

    def _count_clauses(self, sent) -> int:
        """Count the number of clauses in a sentence."""
        # Count finite verbs (main indicator of clauses)
        finite_verbs = [t for t in sent if t.pos_ == 'VERB' and t.dep_ in ('ROOT', 'conj', 'advcl', 'ccomp')]

        # Also check for clause markers
        clause_markers = [t for t in sent if t.dep_ in ('mark', 'advcl', 'relcl', 'ccomp')]

        # At least 1 clause, plus additional for each finite verb beyond the root
        clause_count = max(1, len(finite_verbs))

        # If we have clause markers, likely have subordinate clauses
        if clause_markers:
            clause_count = max(clause_count, 2)

        return clause_count

    def _has_subordinate_clause(self, sent) -> bool:
        """Check if sentence has a subordinate clause."""
        # Check for subordinating conjunctions or relative pronouns
        for token in sent:
            if token.dep_ in ('mark', 'advcl', 'relcl', 'ccomp'):
                return True
            if token.text.lower() in ('which', 'that', 'who', 'when', 'where', 'if', 'although', 'while', 'because'):
                if token.dep_ in ('mark', 'relcl', 'advcl'):
                    return True
        return False

    def _pos_patterns_match(self, actual_pos: List[str], expected_pos: List[str]) -> bool:
        """
        Check if POS patterns match (loose match).

        Compares first 5 tags, allowing some variation.
        """
        if not expected_pos:
            return True  # No pattern to match

        # Take first 5 tags from each
        actual = actual_pos[:5]
        expected = expected_pos[:5]

        if len(actual) < 3 or len(expected) < 3:
            return True  # Too short to validate

        # Normalize POS tags (simplify)
        def normalize_pos(pos):
            if pos in ('PROPN', 'NOUN'):
                return 'NOUN'
            elif pos in ('AUX', 'VERB'):
                return 'VERB'
            elif pos in ('ADJ', 'ADV'):
                return 'MOD'
            return pos

        normalized_actual = [normalize_pos(p) for p in actual]
        normalized_expected = [normalize_pos(p) for p in expected]

        # Check if first 3 tags match (loose requirement)
        matches = 0
        for i in range(min(3, len(normalized_actual), len(normalized_expected))):
            if normalized_actual[i] == normalized_expected[i]:
                matches += 1

        # Require at least 2 out of first 3 to match
        return matches >= 2

    def _get_length_category(self, word_count: int, template: SentenceTemplate) -> str:
        """Get length category for a word count."""
        # Use template's thresholds if available
        # Otherwise use defaults
        if hasattr(template, 'short_threshold') and hasattr(template, 'medium_threshold'):
            short_threshold = template.short_threshold
            medium_threshold = template.medium_threshold
        else:
            short_threshold = 10
            medium_threshold = 25

        if word_count <= short_threshold:
            return 'short'
        elif word_count <= medium_threshold:
            return 'medium'
        else:
            return 'long'

    def _openers_are_similar(self, opener1: str, opener2: str) -> bool:
        """Check if two opener types are similar enough to be acceptable."""
        # Group similar openers
        contrastive = {'conjunction', 'adverb'}  # Both can be contrastive
        declarative = {'noun', 'det_noun', 'pronoun'}  # All declarative
        descriptive = {'prep_phrase', 'verb_ing'}  # Both descriptive

        if opener1 == opener2:
            return True

        # Check if both are in same group
        for group in [contrastive, declarative, descriptive]:
            if opener1 in group and opener2 in group:
                return True

        return False

    def _is_minor_opener_mismatch(self, actual: str, expected: str) -> bool:
        """Check if opener mismatch is minor (acceptable)."""
        # Minor mismatches: similar types that serve similar functions
        minor_pairs = [
            ('conjunction', 'adverb'),  # Both contrastive
            ('adverb', 'conjunction'),
            ('noun', 'det_noun'),  # Both declarative
            ('det_noun', 'noun'),
            ('pronoun', 'det_noun'),  # Both can start declarative statements
            ('det_noun', 'pronoun'),
        ]

        return (actual, expected) in minor_pairs or (expected, actual) in minor_pairs

    def _validate_word_level_template(self, actual_sent, template) -> List[ValidationIssue]:
        """
        Validate word-level template matching.

        Checks:
        1. Pre-populated words are used in correct positions
        2. Semantic slots are filled appropriately
        3. Word order matches template structure
        4. POS tags match expected slots

        Args:
            actual_sent: spaCy sentence object
            template: SentenceTemplate with word_slots

        Returns:
            List of ValidationIssue objects
        """
        issues = []

        if not template.word_slots:
            return issues

        # Get all tokens (including punctuation for position matching)
        all_tokens = list(actual_sent)
        word_tokens = [t for t in all_tokens if not t.is_space and not t.is_punct]

        # Check if we have enough words
        if len(word_tokens) < len([s for s in template.word_slots if s.pos_tag != 'PUNCT']):
            issues.append(ValidationIssue(
                "Word-level template: Not enough words in sentence",
                f"Template requires {len(template.word_slots)} word slots, but sentence has {len(word_tokens)} words"
            ))
            return issues

        # Match tokens to slots by position
        word_idx = 0
        for slot_idx, slot in enumerate(template.word_slots):
            if slot.pos_tag == 'PUNCT':
                # Check punctuation at end
                if slot_idx == len(template.word_slots) - 1:
                    # Last slot should be punctuation
                    if all_tokens and all_tokens[-1].is_punct:
                        expected_punct = slot.prepopulated_word if slot.prepopulated_word else '.'
                        actual_punct = all_tokens[-1].text
                        if actual_punct != expected_punct:
                            issues.append(ValidationIssue(
                                f"Word-level template: Punctuation mismatch at position {slot_idx}",
                                f"Expected '{expected_punct}' but found '{actual_punct}'"
                            ))
                continue

            if word_idx >= len(word_tokens):
                break

            token = word_tokens[word_idx]

            # Check 1: Pre-populated words must match exactly
            if slot.prepopulated_word:
                # Normalize for comparison (lowercase, handle case variations)
                expected_word = slot.prepopulated_word.lower()
                actual_word = token.text.lower()
                actual_lemma = token.lemma_.lower()

                # Allow lemma match for inflected forms (e.g., "the" vs "The", "demonstrates" vs "demonstrate")
                if actual_word != expected_word and actual_lemma != expected_word:
                    # Check if it's a case variation only
                    if actual_word.capitalize() == expected_word.capitalize():
                        pass  # Case variation is OK
                    else:
                        issues.append(ValidationIssue(
                            f"Word-level template: Pre-populated word mismatch at position {slot_idx}",
                            f"Expected '{slot.prepopulated_word}' but found '{token.text}'. Use the exact pre-populated word."
                        ))

            # Check 2: POS tag must match
            actual_pos = token.pos_
            expected_pos = slot.pos_tag

            # Normalize POS for comparison
            normalized_actual = self._normalize_pos_tag(actual_pos)
            normalized_expected = self._normalize_pos_tag(expected_pos)

            if normalized_actual != normalized_expected:
                # Some mismatches are acceptable (e.g., PROPN vs NOUN)
                if not self._pos_tags_compatible(normalized_actual, normalized_expected):
                    issues.append(ValidationIssue(
                        f"Word-level template: POS mismatch at position {slot_idx}",
                        f"Expected {expected_pos} but found {actual_pos} for word '{token.text}'"
                    ))

            # Check 3: Semantic slots should contain appropriate content
            if slot.is_semantic_slot and not slot.prepopulated_word:
                # Check if this looks like semantic content (not a generic word)
                if slot.semantic_hint:
                    # For entity slots, check if it's a proper noun or significant noun
                    if 'entity' in slot.semantic_hint.lower():
                        if actual_pos not in ('PROPN', 'NOUN'):
                            issues.append(ValidationIssue(
                                f"Word-level template: Semantic slot at position {slot_idx} should contain an entity",
                                f"Expected a noun/proper noun (entity) but found {actual_pos} '{token.text}'"
                            ))
                    elif 'concept' in slot.semantic_hint.lower():
                        if actual_pos not in ('NOUN', 'PROPN'):
                            issues.append(ValidationIssue(
                                f"Word-level template: Semantic slot at position {slot_idx} should contain a concept",
                                f"Expected a noun (concept) but found {actual_pos} '{token.text}'"
                            ))
                    elif 'predicate' in slot.semantic_hint.lower():
                        if actual_pos not in ('VERB', 'AUX'):
                            issues.append(ValidationIssue(
                                f"Word-level template: Semantic slot at position {slot_idx} should contain a predicate",
                                f"Expected a verb (predicate) but found {actual_pos} '{token.text}'"
                            ))

            word_idx += 1

        # Check 4: Word order - verify structure matches template
        if len(word_tokens) >= 3 and len(template.word_slots) >= 3:
            # Check first 3 words match template order
            for i in range(min(3, len(word_tokens), len([s for s in template.word_slots if s.pos_tag != 'PUNCT']))):
                slot = [s for s in template.word_slots if s.pos_tag != 'PUNCT'][i]
                token = word_tokens[i]

                # If slot has pre-populated word, it should match
                if slot.prepopulated_word:
                    expected = slot.prepopulated_word.lower()
                    actual = token.text.lower()
                    if actual != expected and token.lemma_.lower() != expected:
                        issues.append(ValidationIssue(
                            f"Word-level template: Word order mismatch - word {i+1} should be '{slot.prepopulated_word}'",
                            f"Found '{token.text}' instead. Follow the exact word order from the template."
                        ))

        return issues

    def _normalize_pos_tag(self, pos: str) -> str:
        """Normalize POS tag for comparison."""
        if pos in ('PROPN', 'NOUN'):
            return 'NOUN'
        elif pos in ('AUX', 'VERB'):
            return 'VERB'
        elif pos in ('ADJ', 'ADV'):
            return 'MOD'
        return pos

    def _pos_tags_compatible(self, actual: str, expected: str) -> bool:
        """Check if two POS tags are compatible (acceptable mismatch)."""
        # Exact match
        if actual == expected:
            return True

        # PROPN and NOUN are compatible
        if (actual == 'NOUN' and expected == 'NOUN') or (actual in ('PROPN', 'NOUN') and expected in ('PROPN', 'NOUN')):
            return True

        # VERB and AUX are compatible
        if (actual == 'VERB' and expected == 'VERB') or (actual in ('VERB', 'AUX') and expected in ('VERB', 'AUX')):
            return True

        # ADJ and ADV are both modifiers
        if (actual == 'MOD' and expected == 'MOD') or (actual in ('ADJ', 'ADV') and expected in ('ADJ', 'ADV')):
            return True

        return False


# Test function
if __name__ == '__main__':
    from template_generator import SentenceTemplate

    validator = SentenceValidator()

    # Test template
    template = SentenceTemplate(
        word_count=20,
        pos_sequence=['DET', 'NOUN', 'VERB', 'DET', 'ADJ', 'NOUN'],
        punctuation='.',
        opener_type='det_noun',
        has_subordinate_clause=False,
        clause_count=1,
        length_category='medium'
    )

    # Test cases
    test_cases = [
        ("The theory demonstrates the fundamental principle.", True),  # Should match
        ("But the theory demonstrates the fundamental principle.", False),  # Wrong opener
        ("The theory.", False),  # Too short
        ("The theory demonstrates the fundamental principle which explains the complex relationship between different concepts.", False),  # Too long
    ]

    print("=== Sentence Validator Test ===\n")
    for sentence, should_match in test_cases:
        issues = validator.validate_sentence(sentence, template)
        matches = len(issues) == 0
        status = "✓" if matches == should_match else "✗"
        print(f"{status} '{sentence[:50]}...'")
        if issues:
            print(f"  Issues: {len(issues)}")
            for issue in issues:
                print(f"    - {issue.description}")

