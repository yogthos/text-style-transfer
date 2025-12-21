"""Statistical Critic for validating archetype compliance.

This module validates that generated paragraphs match the statistical
archetype parameters (sentence length, burstiness, etc.) using spaCy.
"""

import json
import re
from collections import Counter
from pathlib import Path
from typing import Dict, Tuple, Optional, List


class StatisticalCritic:
    """Validates that output matches archetype statistics."""

    def __init__(self, config_path: str = "config.json"):
        """Initialize the statistical critic.

        Args:
            config_path: Path to configuration file.
        """
        # Load config
        with open(config_path, 'r') as f:
            self.config = json.load(f)

        self.critic_config = self.config.get("critic", {})
        self.stat_tolerance = self.critic_config.get("stat_tolerance", 0.3)

        # Initialize spaCy with safety check
        try:
            import spacy
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            try:
                from spacy.cli import download
                download("en_core_web_sm")
                import spacy
                self.nlp = spacy.load("en_core_web_sm")
            except Exception as e:
                raise RuntimeError(f"Could not load spaCy model: {e}")
        except ImportError:
            raise ImportError("spaCy is required for StatisticalCritic. Install with: pip install spacy && python -m spacy download en_core_web_sm")

    def measure_compliance(self, text: str, target_archetype: Dict) -> Tuple[float, str]:
        """Measure compliance of text against target archetype.

        Args:
            text: Generated paragraph text to validate
            target_archetype: Target archetype dictionary with stats

        Returns:
            Tuple of (compliance_score, feedback_message)
            compliance_score is between 0.0 and 1.0
        """
        if not text or not text.strip():
            return 0.0, "Empty text provided"

        # Parse text with spaCy
        doc = self.nlp(text)
        sentences = list(doc.sents)

        if not sentences:
            return 0.0, "No sentences found in text"

        # Calculate actual stats
        sentence_lengths = [len(list(sent)) for sent in sentences]  # Words per sentence
        actual_avg_len = sum(sentence_lengths) / len(sentence_lengths) if sentence_lengths else 0
        actual_avg_sents = len(sentences)

        # Calculate burstiness (variance in sentence lengths)
        if len(sentence_lengths) > 1:
            mean_len = actual_avg_len
            variance = sum((x - mean_len) ** 2 for x in sentence_lengths) / len(sentence_lengths)
            actual_burstiness = variance ** 0.5  # Standard deviation
        else:
            actual_burstiness = 0.0

        # Get target stats
        target_avg_len = target_archetype.get("avg_len", 0)
        target_avg_sents = target_archetype.get("avg_sents", 0)

        # Calculate compliance scores
        compliance_scores = []
        feedback_parts = []

        # Check average sentence length
        if target_avg_len > 0:
            len_diff = abs(actual_avg_len - target_avg_len) / target_avg_len
            if len_diff > self.stat_tolerance:
                compliance_scores.append(0.0)
                if actual_avg_len < target_avg_len:
                    feedback_parts.append(f"Your paragraph is too simple. Average sentence length is {actual_avg_len:.1f} words, but target is {target_avg_len:.1f} words. Combine sentences to increase complexity.")
                else:
                    feedback_parts.append(f"Your paragraph is too complex. Average sentence length is {actual_avg_len:.1f} words, but target is {target_avg_len:.1f} words. Break up long sentences.")
            else:
                compliance_scores.append(1.0 - (len_diff / self.stat_tolerance))
        else:
            compliance_scores.append(1.0)

        # Check average sentences per paragraph
        if target_avg_sents > 0:
            sents_diff = abs(actual_avg_sents - target_avg_sents) / target_avg_sents
            if sents_diff > self.stat_tolerance:
                compliance_scores.append(0.0)
                if actual_avg_sents < target_avg_sents:
                    feedback_parts.append(f"Your paragraph has too few sentences ({actual_avg_sents} vs target {target_avg_sents}). Add more sentences.")
                else:
                    feedback_parts.append(f"Your paragraph has too many sentences ({actual_avg_sents} vs target {target_avg_sents}). Combine or remove sentences.")
            else:
                compliance_scores.append(1.0 - (sents_diff / self.stat_tolerance))
        else:
            compliance_scores.append(1.0)

        # Overall compliance score (average of individual scores)
        overall_score = sum(compliance_scores) / len(compliance_scores) if compliance_scores else 0.0

        # Generate feedback
        if feedback_parts:
            feedback = " ".join(feedback_parts)
        else:
            feedback = "Paragraph matches archetype statistics."

        return overall_score, feedback

    def select_best_candidate(self, candidates: List[str], target_archetype: Dict) -> Tuple[str, float, str]:
        """Select the best candidate from a list based on archetype compliance.

        Args:
            candidates: List of candidate paragraph texts to evaluate
            target_archetype: Target archetype dictionary with stats

        Returns:
            Tuple of (best_candidate_text, best_score, qualitative_feedback)
        """
        if not candidates:
            raise ValueError("No candidates provided")

        # Evaluate all candidates
        candidate_scores = []
        for candidate in candidates:
            if not candidate or not candidate.strip():
                continue
            score, _ = self.measure_compliance(candidate, target_archetype)
            candidate_scores.append((candidate, score))

        if not candidate_scores:
            # All candidates were empty, return first one with low score
            return candidates[0] if candidates else "", 0.0, "All candidates were empty or invalid"

        # Sort by score (highest first)
        candidate_scores.sort(key=lambda x: x[1], reverse=True)

        # Get best candidate
        best_candidate, best_score = candidate_scores[0]

        # Generate qualitative feedback for the best candidate
        qualitative_feedback = self._generate_qualitative_feedback(best_candidate, target_archetype)

        return best_candidate, best_score, qualitative_feedback

    def _find_split_point(self, sentence_text: str, sentence_doc) -> Optional[Tuple[str, str, str]]:
        """Find the best split point in a sentence using robust hierarchy.

        Returns: (split_type, split_location, instruction) or None
        split_type: 'primary', 'secondary', or 'tertiary'
        split_location: The text to split at (e.g., "Union;", "Union, and")
        instruction: The specific instruction for the LLM
        """
        text_lower = sentence_text.lower()

        # Primary: Semicolons or colons
        import re
        # Find semicolons followed by space and word
        semicolon_match = re.search(r'([^;]+);\s+(\w+)', sentence_text)
        if semicolon_match:
            before = semicolon_match.group(1).strip()
            # Find the word before semicolon
            words_before = before.split()
            if words_before:
                split_word = words_before[-1]
                # Get the word after semicolon (already captured in group 2)
                next_word = semicolon_match.group(2)
                # Capitalize first letter
                next_word_cap = next_word[0].upper() + next_word[1:] if len(next_word) > 1 else next_word.upper()
                return ('primary', f"{split_word};",
                       f"Replace semicolon after '{split_word}' with period. Start new sentence with '{next_word_cap}'")

        # Find colons followed by space and word
        colon_match = re.search(r'([^:]+):\s+(\w+)', sentence_text)
        if colon_match:
            before = colon_match.group(1).strip()
            words_before = before.split()
            if words_before:
                split_word = words_before[-1]
                next_word = colon_match.group(2)
                next_word_cap = next_word[0].upper() + next_word[1:] if len(next_word) > 1 else next_word.upper()
                return ('primary', f"{split_word}:",
                       f"Replace colon after '{split_word}' with period. Start new sentence with '{next_word_cap}'")

        # Secondary: Coordinating conjunctions preceded by comma
        from src.utils.spacy_linguistics import get_conjunctions
        if sentence_doc is not None:
            conjunctions = get_conjunctions(sentence_doc)
        else:
            # Fallback to hardcoded list if doc not available
            conjunctions = ['and', 'but', 'or', 'nor', 'for', 'yet', 'so']
        for conj in conjunctions:
            pattern = rf',\s+{conj}\s+(\w+)'
            match = re.search(pattern, text_lower)
            if match:
                # Find the word before comma
                comma_pos = sentence_text.lower().find(f', {conj}')
                if comma_pos > 0:
                    before_comma = sentence_text[:comma_pos].strip()
                    words_before = before_comma.split()
                    if words_before:
                        split_word = words_before[-1]
                        # Get the word after conjunction (already captured)
                        next_word = match.group(1)
                        next_word_cap = next_word[0].upper() + next_word[1:] if len(next_word) > 1 else next_word.upper()
                        return ('secondary', f"{split_word}, {conj}",
                               f"Delete '{conj}', replace comma after '{split_word}' with period. Start new sentence with '{next_word_cap}'")

        # Tertiary: Relative pronouns
        from src.utils.spacy_linguistics import get_relative_pronouns
        if sentence_doc is not None:
            relative_pronouns = get_relative_pronouns(sentence_doc)
        else:
            # Fallback to hardcoded list if doc not available
            relative_pronouns = ['which', 'who', 'that', 'where', 'when', 'whom', 'whose']
        for pronoun in relative_pronouns:
            pattern = rf'\b{pronoun}\s+'
            match = re.search(pattern, text_lower)
            if match:
                # Find the word before relative pronoun
                pronoun_pos = match.start()
                before_pronoun = sentence_text[:pronoun_pos].strip()
                words_before = before_pronoun.split()
                if words_before:
                    # Use the subject of the sentence or "This"
                    subject = words_before[0] if words_before else "This"
                    return ('tertiary', pronoun,
                           f"Start new sentence with '{subject}'. Replace '{pronoun}' with appropriate subject reference")

        return None

    def _generate_qualitative_feedback(self, text: str, target_archetype: Dict) -> str:
        """Generate directive feedback with atomic instructions.

        Converts generic feedback into numbered, imperative editing instructions.

        Args:
            text: The text to analyze
            target_archetype: Target archetype dictionary with stats

        Returns:
            Directive feedback string with numbered atomic instructions
        """
        if not text or not text.strip():
            return "## Directive Fixes:\n1. The paragraph is empty. Generate a complete paragraph."

        # Parse text with spaCy
        doc = self.nlp(text)
        sentences = list(doc.sents)

        if not sentences:
            return "## Directive Fixes:\n1. No sentences found. Generate a paragraph with complete sentences."

        # Sentence-by-sentence audit
        sentence_texts = [sent.text.strip() for sent in sentences]
        sentence_lengths = [len(list(sent)) for sent in sentences]
        actual_avg_len = sum(sentence_lengths) / len(sentence_lengths) if sentence_lengths else 0
        actual_avg_sents = len(sentences)

        # Get target stats
        target_avg_len = target_archetype.get("avg_len", 0)
        target_avg_sents = target_archetype.get("avg_sents", 0)

        instructions = []
        instruction_num = 1

        # Sentence-by-sentence gap analysis
        if target_avg_len > 0:
            for i, (sent_text, sent_len) in enumerate(zip(sentence_texts, sentence_lengths), 1):
                gap = sent_len - target_avg_len

                # If sentence is too long (more than 10 words over target)
                if gap > 10:
                    # Find split point
                    sent_doc = self.nlp(sent_text)
                    split_info = self._find_split_point(sent_text, sent_doc)

                    if split_info:
                        split_type, split_location, instruction = split_info
                        instructions.append(
                            f"{instruction_num}. **Split Sentence {i}:**\n"
                            f"   - Current: {sent_len} words (target: ~{target_avg_len:.1f} words) - TOO LONG\n"
                            f"   - Action: Split Sentence {i} at '{split_location}'\n"
                            f"   - Instruction: {instruction}\n"
                            f"   - Result: Creates two sentences of approximately {sent_len // 2} words each"
                        )
                        instruction_num += 1
                    else:
                        # Fallback: generic split instruction
                        instructions.append(
                            f"{instruction_num}. **Split Sentence {i}:**\n"
                            f"   - Current: {sent_len} words (target: ~{target_avg_len:.1f} words) - TOO LONG\n"
                            f"   - Action: Break Sentence {i} into two sentences at a natural clause boundary\n"
                            f"   - Instruction: Find a comma, conjunction, or relative clause and split there. Replace with period and capitalize next word.\n"
                            f"   - Result: Creates two sentences of approximately {sent_len // 2} words each"
                        )
                        instruction_num += 1

                # If sentence is too short (more than 10 words under target)
                elif gap < -10 and i < len(sentence_texts):
                    # Combine with next sentence
                    next_sent = sentence_texts[i] if i < len(sentence_texts) else None
                    if next_sent:
                        instructions.append(
                            f"{instruction_num}. **Combine Sentences {i} and {i+1}:**\n"
                            f"   - Current: Sentence {i} has {sent_len} words (target: ~{target_avg_len:.1f} words) - TOO SHORT\n"
                            f"   - Action: Combine Sentence {i} with Sentence {i+1}\n"
                            f"   - Instruction: Join them using a comma and conjunction (e.g., 'and', 'but') or a relative clause\n"
                            f"   - Result: Creates one sentence of approximately {sent_len + sentence_lengths[i]} words"
                        )
                        instruction_num += 1

        # Structure audit: sentence count
        if target_avg_sents > 0:
            sents_diff = actual_avg_sents - target_avg_sents
            target_rounded = round(target_avg_sents)

            if abs(sents_diff) > 0.2 * target_avg_sents:  # More than 20% difference
                if actual_avg_sents < target_rounded:
                    # Need more sentences
                    needed = target_rounded - actual_avg_sents
                    if needed == 1:
                        # Try to split one sentence
                        if sentence_lengths:
                            # Find longest sentence to split
                            longest_idx = max(range(len(sentence_lengths)), key=lambda i: sentence_lengths[i])
                            longest_sent = sentence_texts[longest_idx]
                            longest_len = sentence_lengths[longest_idx]

                            split_info = self._find_split_point(longest_sent, self.nlp(longest_sent))
                            if split_info:
                                _, split_location, instruction = split_info
                                instructions.append(
                                    f"{instruction_num}. **Add Sentence (Split):**\n"
                                    f"   - Current: {actual_avg_sents} sentences (target: {target_rounded})\n"
                                    f"   - Action: Split Sentence {longest_idx + 1} at '{split_location}'\n"
                                    f"   - Instruction: {instruction}\n"
                                    f"   - Result: Paragraph will have {actual_avg_sents + 1} sentences total"
                                )
                            else:
                                instructions.append(
                                    f"{instruction_num}. **Add Sentence:**\n"
                                    f"   - Current: {actual_avg_sents} sentences (target: {target_rounded})\n"
                                    f"   - Action: Split one existing sentence OR add one new sentence\n"
                                    f"   - Instruction: Break the longest sentence at a natural boundary (comma, conjunction, or relative clause)\n"
                                    f"   - Result: Paragraph will have {target_rounded} sentences total"
                                )
                            instruction_num += 1
                else:
                    # Too many sentences - need to combine
                    excess = actual_avg_sents - target_rounded
                    instructions.append(
                        f"{instruction_num}. **Reduce Sentence Count:**\n"
                        f"   - Current: {actual_avg_sents} sentences (target: {target_rounded})\n"
                        f"   - Action: Combine {excess} pair(s) of sentences\n"
                        f"   - Instruction: Join related sentences using commas and conjunctions or relative clauses\n"
                        f"   - Result: Paragraph will have {target_rounded} sentences total"
                    )
                    instruction_num += 1

        # If no issues found, provide positive feedback
        if not instructions:
            return "## Directive Fixes:\n1. Paragraph matches the target archetype statistics well. No changes needed."

        # Format as numbered list
        feedback = "## Directive Fixes (Execute These Edits EXACTLY):\n\n" + "\n\n".join(instructions)
        return feedback

    def evaluate_sentence(self, sentence: str, target_length: int, tolerance: float = None) -> Tuple[float, str]:
        """Evaluate a single sentence against target length.

        Args:
            sentence: The sentence text to evaluate
            target_length: Target word count for the sentence
            tolerance: Acceptable deviation ratio (default: uses stat_tolerance from config)

        Returns:
            Tuple of (score, feedback) where:
            - score: 1.0 if within tolerance, else 0.0
            - feedback: Specific instruction for fixing length
        """
        if tolerance is None:
            tolerance = self.stat_tolerance

        if not sentence or not sentence.strip():
            return 0.0, "Sentence is empty. Generate a complete sentence."

        # Count words (simple split, excluding empty strings)
        words = [w for w in sentence.split() if w.strip()]
        word_count = len(words)

        if target_length <= 0:
            return 1.0, "Sentence length matches target."

        # Calculate difference ratio
        diff_ratio = abs(word_count - target_length) / target_length

        if diff_ratio <= tolerance:
            return 1.0, "Sentence length matches target."

        # Generate specific feedback
        if word_count > target_length:
            return 0.0, f"Too long ({word_count} words). Cut to {target_length} words."
        else:
            return 0.0, f"Too short ({word_count} words). Expand to {target_length} words."

    def check_action_echo(self, sentences: List[str]) -> List[str]:
        """Detect action verb repetition across consecutive sentences.

        Uses spaCy lemmatization to catch that "weaving" and "wove" are the same action.
        Ignores auxiliary verbs (was, had, did) to focus on meaningful action repetition.

        Args:
            sentences: List of sentence strings to check

        Returns:
            List of issue strings describing action echoes found
        """
        from src.utils.spacy_linguistics import get_main_verbs_excluding_auxiliaries

        if not sentences or len(sentences) < 2:
            return []

        issues = []

        # Process sentences in batch for efficiency
        docs = list(self.nlp.pipe(sentences, disable=["ner"]))  # Disable NER for speed

        for i in range(len(docs) - 1):
            # Extract main verbs (excluding auxiliaries and stopwords)
            verbs1 = get_main_verbs_excluding_auxiliaries(docs[i])
            verbs2 = get_main_verbs_excluding_auxiliaries(docs[i + 1])

            # Find overlapping verbs (same lemma = same action)
            overlap = verbs1.intersection(verbs2)

            if overlap:
                # Get the actual verb forms from the text for better error messages
                verb_forms1 = [t.text for t in docs[i]
                              if t.pos_ == "VERB" and not t.is_stop
                              and t.lemma_.lower() in overlap]
                verb_forms2 = [t.text for t in docs[i + 1]
                              if t.pos_ == "VERB" and not t.is_stop
                              and t.lemma_.lower() in overlap]

                verb_example = verb_forms1[0] if verb_forms1 else list(overlap)[0]
                issues.append(
                    f"Action Echo: The verb '{verb_example}' (lemma: '{list(overlap)[0]}') "
                    f"repeats across sentences {i+1} and {i+2}. MERGE or REPHRASE."
                )

        return issues

    def check_repetition(self, text: str) -> List[str]:
        """Detect repetitive phrasing in text.

        Scans for repeated n-grams (2-3 word phrases) and proximal word repetition.
        This is a code-based check to catch vocabulary "echoes" that the Assembly Line
        might miss when building sentences in isolation.

        Args:
            text: Text to check for repetition

        Returns:
            List of issue strings describing repetitions found
        """
        if not text or not text.strip():
            return []

        issues = []
        text_lower = text.lower()
        words = re.findall(r'\b\w+\b', text_lower)

        if len(words) < 4:  # Need at least 4 words for meaningful checks
            return []

        # Parse with spaCy for stopword detection
        doc = self.nlp(text)
        tokens = [token for token in doc if not token.is_punct and not token.is_space]

        if len(tokens) < 4:
            return []

        # 1. Check for Repeated N-grams (2-3 words) - WITH STOPWORD FILTER
        for n in [2, 3]:
            ngrams = []
            for i in range(len(tokens) - n + 1):
                window = tokens[i:i+n]
                # Skip if ANY token in the n-gram is a stopword
                if any(t.is_stop for t in window):
                    continue
                # Build phrase from non-stopword tokens
                phrase = ' '.join([t.text.lower() for t in window])
                ngrams.append(phrase)

            counts = Counter(ngrams)
            for phrase, count in counts.items():
                if count > 1:
                    issues.append(f"Repeated phrase: '{phrase}' (appears {count} times)")

        # 2. Check for Proximal Word Repetition (Same word twice within 10 words)
        # Filter out stopwords using spaCy's built-in detection (never hardcode stopwords)
        seen_words = set()
        for i in range(len(words)):
            word = words[i]
            # Use spaCy's stopword detection (already loaded in self.nlp)
            # This covers all stopwords including "as", "than", etc. that hardcoded sets miss
            if self.nlp.vocab[word].is_stop:
                continue
            if word in seen_words:
                continue  # Already reported this word

            # Check next 10 words for repetition
            window = words[i+1:min(i+11, len(words))]
            if word in window:
                issues.append(f"Repeated word: '{word}' appears twice within 10 words")
                seen_words.add(word)  # Only report once per word

        return issues

