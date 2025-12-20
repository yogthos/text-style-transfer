"""Structural skeleton extractor for JIT structural templating.

This module extracts sentence skeletons from RAG samples by replacing
nouns, verbs, and adjectives with placeholders while preserving the
exact syntactic structure.

Also provides paragraph rhythm extraction for structural cloning.
"""

import re
from pathlib import Path
from typing import Optional, List, Dict
from src.generator.llm_provider import LLMProvider


def _load_prompt_template(template_name: str) -> str:
    """Load a prompt template from the prompts directory.

    Args:
        template_name: Name of the template file (e.g., 'structuralizer_skeleton_system.md')

    Returns:
        Template content as string.
    """
    prompts_dir = Path(__file__).parent.parent.parent / "prompts"
    template_path = prompts_dir / template_name
    if not template_path.exists():
        raise FileNotFoundError(f"Prompt template not found: {template_path}")
    return template_path.read_text().strip()

# Rhetorical connectors that define sentence flow (case-insensitive matching)
RHETORICAL_OPENERS = {
    'but', 'however', 'thus', 'therefore', 'yet', 'conversely', 'although',
    'in contrast', 'nevertheless', 'nonetheless', 'moreover', 'furthermore',
    'consequently', 'hence', 'accordingly', 'indeed', 'in fact', 'specifically',
    'notably', 'significantly', 'importantly', 'crucially', 'meanwhile',
    'alternatively', 'additionally', 'similarly', 'likewise', 'instead',
    'rather', 'still', 'though', 'despite', 'regardless', 'nonetheless'
}


class Structuralizer:
    """Extracts structural skeletons from example sentences."""

    def __init__(self, config_path: str = "config.json"):
        """Initialize the structuralizer.

        Args:
            config_path: Path to configuration file.
        """
        self.llm_provider = LLMProvider(config_path=config_path)
        self.config_path = config_path

    def extract_skeleton(self, text: str, input_text: Optional[str] = None) -> str:
        """Extract structural skeleton from a sentence.

        Replaces Nouns with [NP], Verbs with [VP], Adjectives with [ADJ].
        Keeps all prepositions, conjunctions, and punctuation.

        Args:
            text: Input sentence to extract skeleton from.
            input_text: Optional original input text for pruning (if single sentence, truncate skeleton to first sentence).

        Returns:
            Skeleton template with placeholders.
        """
        if not text or not text.strip():
            return ""

        system_prompt = _load_prompt_template("structuralizer_skeleton_system.md")
        user_template = _load_prompt_template("structuralizer_skeleton_user.md")
        user_prompt = user_template.format(text=text)

        try:
            response = self.llm_provider.call(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                temperature=0.2,  # Low temperature for consistent extraction
                max_tokens=200
            )

            # Clean response
            skeleton = response.strip()

            # Remove quotes if present
            skeleton = re.sub(r'^["\']|["\']$', '', skeleton)
            skeleton = skeleton.strip()

            # SKELETON PRUNING: If input is single-sentence, truncate skeleton to first sentence delimiter
            if input_text:
                input_sentences = re.split(r'[.!?]+', input_text.strip())
                input_sentences = [s.strip() for s in input_sentences if s.strip()]
                if len(input_sentences) == 1:
                    # Input is single sentence - truncate skeleton to first sentence delimiter
                    # Find first sentence-ending punctuation in skeleton
                    for delimiter in ['.', '!', '?']:
                        if delimiter in skeleton:
                            # Truncate at first sentence delimiter
                            delimiter_pos = skeleton.find(delimiter)
                            skeleton = skeleton[:delimiter_pos + 1].strip()
                            break

            # Validate: should contain at least one placeholder
            if '[NP]' not in skeleton and '[VP]' not in skeleton and '[ADJ]' not in skeleton:
                # Fallback: return empty to indicate failure
                return ""

            # Ghost Word Ban: Check if skeleton contains semantic words that should be slots
            # Common semantic words that should NOT appear in skeletons
            semantic_words = [
                'ideas', 'skies', 'mind', 'theory', 'knowledge', 'practice', 'standpoint',
                'correct', 'innate', 'drop', 'come', 'reinforce', 'affirm', 'serve',
                'primary', 'objective', 'dialectical', 'raw', 'clear', 'good', 'bad'
            ]
            skeleton_lower = skeleton.lower()
            found_semantic_words = [word for word in semantic_words if word in skeleton_lower]

            if found_semantic_words:
                # Reject skeleton with ghost words - return empty to trigger fallback
                return ""

            return skeleton

        except Exception as e:
            # On error, return empty skeleton (will trigger fallback)
            return ""

    def count_skeleton_slots(self, skeleton: str) -> int:
        """Count the number of placeholder slots in a skeleton.

        Args:
            skeleton: Skeleton template string.

        Returns:
            Number of placeholder slots ([NP], [VP], [ADJ]).
        """
        if not skeleton:
            return 0

        # Count all placeholders
        np_count = len(re.findall(r'\[NP\]', skeleton))
        vp_count = len(re.findall(r'\[VP\]', skeleton))
        adj_count = len(re.findall(r'\[ADJ\]', skeleton))

        return np_count + vp_count + adj_count

    def adapt_skeleton(self, skeleton: str, target_word_count: int) -> str:
        """Adapt skeleton to target word count by compressing or expanding.

        If skeleton has too many slots, simplify it. If too few, expand it.
        Preserves the author's voice and connectors.

        Args:
            skeleton: Original skeleton template.
            target_word_count: Target number of slots (word count proxy).

        Returns:
            Adapted skeleton template, or original if adaptation fails.
        """
        if not skeleton:
            return skeleton

        current_slots = self.count_skeleton_slots(skeleton)

        # If skeleton is within acceptable range, return as-is
        if 0.5 * target_word_count <= current_slots <= 2.0 * target_word_count:
            return skeleton

        system_prompt = _load_prompt_template("structuralizer_adapt_system.md")

        if current_slots > target_word_count * 2:
            # Too long: compress
            user_template = _load_prompt_template("structuralizer_adapt_compress_user.md")
            user_prompt = user_template.format(
                current_slots=current_slots,
                target_word_count=target_word_count,
                skeleton=skeleton
            )
        else:
            # Too short: expand
            user_template = _load_prompt_template("structuralizer_adapt_expand_user.md")
            user_prompt = user_template.format(
                current_slots=current_slots,
                target_word_count=target_word_count,
                skeleton=skeleton
            )

        try:
            response = self.llm_provider.call(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                temperature=0.3,  # Low temperature for consistent adaptation
                max_tokens=300
            )

            # Clean response
            adapted = response.strip()
            adapted = re.sub(r'^["\']|["\']$', '', adapted)
            adapted = adapted.strip()

            # Validate: should contain at least one placeholder
            if '[NP]' not in adapted and '[VP]' not in adapted and '[ADJ]' not in adapted:
                # Adaptation failed, return original
                return skeleton

            return adapted

        except Exception:
            # On error, return original skeleton
            return skeleton

def extract_paragraph_rhythm(text: str) -> List[Dict]:
        """Extract paragraph rhythm map from a text example.

        Analyzes sentence structure (length, type, opener) to create a rhythm
        template that can be used to force generated text to match human patterns.

        Args:
            text: Input paragraph text to analyze.

        Returns:
            List of dictionaries, each representing a sentence specification:
            [{'length': 'short', 'type': 'question', 'opener': None}, ...]
        """
        if not text or not text.strip():
            return []

        try:
            from nltk.tokenize import sent_tokenize
        except ImportError:
            # Fallback: simple sentence splitting
            sentences = re.split(r'[.!?]+\s+', text)
            sentences = [s.strip() for s in sentences if s.strip()]
        else:
            sentences = sent_tokenize(text)
            sentences = [s.strip() for s in sentences if s.strip()]

        rhythm_map = []

        for sentence in sentences:
            if not sentence:
                continue

            # Count words (simple split)
            words = sentence.split()
            word_count = len(words)

            # Determine length class
            if word_count < 10:
                length = 'short'
            elif word_count <= 25:
                length = 'medium'
            else:
                length = 'long'

            # Determine type
            sentence_lower = sentence.lower()
            if sentence.rstrip().endswith('?'):
                sent_type = 'question'
            elif 'if' in sentence_lower and ('then' in sentence_lower or ',' in sentence):
                # Check for conditional structure (if...then or if...,)
                sent_type = 'conditional'
            else:
                sent_type = 'standard'

            # Determine opener (only if it's a rhetorical connector)
            opener = None
            if words:
                first_word = words[0].rstrip('.,!?;:').lower()
                if first_word in RHETORICAL_OPENERS:
                    # Preserve original capitalization from sentence
                    opener = words[0].rstrip('.,!?;:')

            rhythm_map.append({
                'length': length,
                'type': sent_type,
                'opener': opener
            })

        return rhythm_map


def generate_structure_signature(rhythm_map: List[Dict]) -> str:
    """Generate normalized signature for a rhythm map.

    Creates a string representation that captures:
    - Sentence count
    - First sentence opener (if any)
    - Pattern type (conditional, declarative, question, mixed)
    - Length distribution pattern (short-heavy, long-heavy, balanced)

    Args:
        rhythm_map: List of sentence specifications from extract_paragraph_rhythm()

    Returns:
        Normalized signature string, e.g., "3_sent_if_conditional_long-heavy"
    """
    if not rhythm_map:
        return "0_sent_none_none_none"

    count = len(rhythm_map)
    opener = rhythm_map[0].get('opener', 'none') or 'none'
    opener = opener.lower() if opener else 'none'

    # Determine pattern type
    conditional_count = sum(1 for r in rhythm_map if r.get('type') == 'conditional')
    question_count = sum(1 for r in rhythm_map if r.get('type') == 'question')

    if conditional_count > len(rhythm_map) * 0.5:
        pattern_type = "conditional"
    elif question_count > 0:
        pattern_type = "question"
    elif all(r.get('type') == 'standard' for r in rhythm_map):
        pattern_type = "declarative"
    else:
        pattern_type = "mixed"

    # Determine length pattern
    lengths = [r.get('length', 'medium') for r in rhythm_map]
    short_count = lengths.count('short')
    long_count = lengths.count('long')

    if short_count > len(rhythm_map) * 0.5:
        length_pattern = "short-heavy"
    elif long_count > len(rhythm_map) * 0.5:
        length_pattern = "long-heavy"
    else:
        length_pattern = "balanced"

    return f"{count}_sent_{opener}_{pattern_type}_{length_pattern}"

