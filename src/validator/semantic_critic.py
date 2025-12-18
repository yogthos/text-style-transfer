"""Semantic critic for Pipeline 2.0.

This module validates generated text using precision/recall semantic checks
based on semantic blueprints rather than text overlap.
"""

import json
import re
from typing import Dict, Tuple, Optional, Set
from src.ingestion.blueprint import SemanticBlueprint, BlueprintExtractor

try:
    from sentence_transformers import SentenceTransformer, util
    import torch
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    SentenceTransformer = None
    util = None
    torch = None

# Initialize spaCy for stop word filtering
try:
    import spacy
    # Load lightweight model (exclude parser/ner for speed)
    _spacy_nlp = spacy.load("en_core_web_sm", exclude=["parser", "ner"])

    # Domain-specific glue words
    DOMAIN_GLUE_WORDS = {
        "process", "act", "state", "nature", "way", "manner", "form", "condition",
        "method", "means", "approach", "system", "structure", "framework",
        "eventually", "typically", "fundamentally"
    }

    # Add to spaCy's stop words
    for word in DOMAIN_GLUE_WORDS:
        _spacy_nlp.Defaults.stop_words.add(word)
        _spacy_nlp.vocab[word].is_stop = True
except (ImportError, OSError):
    _spacy_nlp = None
    DOMAIN_GLUE_WORDS = set()


def _get_significant_tokens(text: str) -> Set[str]:
    """Extract significant tokens (non-stop, non-punct) using spaCy.

    Args:
        text: Text to extract tokens from.

    Returns:
        Set of lemmatized significant tokens (non-stop, non-punct).
    """
    if not _spacy_nlp:
        # Fallback to simple split if spaCy not available
        return set(text.lower().split())

    doc = _spacy_nlp(text.lower())
    significant_tokens = set()
    for token in doc:
        if not token.is_stop and not token.is_punct and not token.is_space:
            significant_tokens.add(token.lemma_)
    return significant_tokens


class SemanticCritic:
    """Validates generated text using precision/recall semantic checks."""

    def __init__(self, similarity_threshold: float = None, config_path: str = "config.json"):
        """Initialize the semantic critic.

        Args:
            similarity_threshold: Optional override for similarity threshold.
                If None, loads from config.
            config_path: Path to configuration file.
        """
        self.extractor = BlueprintExtractor()

        # Load config
        with open(config_path, 'r') as f:
            config = json.load(f)
        critic_config = config.get("semantic_critic", {})

        # Use config values with fallbacks
        self.similarity_threshold = similarity_threshold or critic_config.get("similarity_threshold", 0.7)
        self.recall_threshold = critic_config.get("recall_threshold", 0.80)
        self.precision_threshold = critic_config.get("precision_threshold", 0.75)
        self.fluency_threshold = critic_config.get("fluency_threshold", 0.8)
        self.accuracy_weight = critic_config.get("accuracy_weight", 0.7)
        self.fluency_weight = critic_config.get("fluency_weight", 0.3)

        if SENTENCE_TRANSFORMERS_AVAILABLE:
            try:
                self.semantic_model = SentenceTransformer('all-MiniLM-L6-v2')
            except Exception:
                self.semantic_model = None
        else:
            self.semantic_model = None

    def evaluate(
        self,
        generated_text: str,
        input_blueprint: SemanticBlueprint
    ) -> Dict[str, any]:
        """Evaluate generated text against input blueprint.

        Args:
            generated_text: Generated text to evaluate.
            input_blueprint: Original input blueprint to compare against.

            Returns:
            Dict with:
            - pass: bool
            - recall_score: float (0-1)
            - precision_score: float (0-1)
            - fluency_score: float (0-1)
            - feedback: str
            - score: float (0-1, weighted: accuracy * 0.7 + fluency * 0.3)
        """
        if not generated_text or not generated_text.strip():
            return {
                "pass": False,
                "recall_score": 0.0,
                "precision_score": 0.0,
                "fluency_score": 0.0,
                "score": 0.0,
                "feedback": "CRITICAL: Generated text is empty."
            }

        # First, validate citations and quotes (non-negotiable)
        citations_valid, citations_feedback = self._validate_citations_and_quotes(
            generated_text, input_blueprint
        )
        if not citations_valid:
            # Citations/quotes are missing - this is a critical failure
            return {
                "pass": False,
                "recall_score": 0.0,
                "precision_score": 0.0,
                "fluency_score": 0.0,
                "score": 0.0,
                "feedback": f"CRITICAL: {citations_feedback}"
            }

        # Extract blueprint from generated text
        try:
            generated_blueprint = self.extractor.extract(generated_text)
        except Exception:
            return {
                "pass": False,
                "recall_score": 0.0,
                "precision_score": 0.0,
                "fluency_score": 0.0,
                "score": 0.0,
                "feedback": "CRITICAL: Could not parse generated text."
            }

        # Recall check: Did we keep the meaning?
        recall_score, recall_feedback = self._check_recall(
            generated_blueprint,
            input_blueprint
        )

        # Precision check: Did we hallucinate?
        precision_score, precision_feedback = self._check_precision(
            generated_blueprint,
            input_blueprint
        )

        # Fluency check: Is it grammatically natural?
        fluency_score, fluency_feedback = self._check_fluency(generated_text)

        # Calculate weighted final score (accuracy + fluency)
        accuracy_score = (recall_score + precision_score) / 2.0
        final_score = accuracy_score * self.accuracy_weight + fluency_score * self.fluency_weight

        # Pass if recall, precision, and fluency are above thresholds
        passes = (recall_score >= self.recall_threshold and
                 precision_score >= self.precision_threshold and
                 fluency_score >= self.fluency_threshold)

        feedback_parts = []
        if recall_score < self.recall_threshold:
            feedback_parts.append(recall_feedback)
        if precision_score < self.precision_threshold:
            feedback_parts.append(precision_feedback)
        if fluency_score < self.fluency_threshold:
            feedback_parts.append(fluency_feedback)

        return {
            "pass": passes,
            "recall_score": recall_score,
            "precision_score": precision_score,
            "fluency_score": fluency_score,
            "score": final_score,
            "feedback": " ".join(feedback_parts) if feedback_parts else "Passed semantic validation."
        }

    def _check_recall(
        self,
        generated_blueprint: SemanticBlueprint,
        input_blueprint: SemanticBlueprint
    ) -> Tuple[float, str]:
        """Check if generated text preserved input meaning (90% of input keywords must match).

        Args:
            generated_blueprint: Blueprint extracted from generated text.
            input_blueprint: Original input blueprint.

        Returns:
            Tuple of (recall_score, feedback_string).
        """
        if not input_blueprint.core_keywords:
            return 1.0, ""

        if not self.semantic_model:
            # Fallback: simple set intersection
            input_keywords = input_blueprint.core_keywords
            generated_keywords = generated_blueprint.core_keywords
            overlap = len(input_keywords & generated_keywords)
            recall_ratio = overlap / len(input_keywords) if input_keywords else 1.0

            if recall_ratio < self.recall_threshold:
                missing = input_keywords - generated_keywords
                return recall_ratio, f"CRITICAL: Missing concepts: {', '.join(list(missing)[:5])}. Preserve all input meaning."
            return recall_ratio, ""

        # Encode keywords as vectors
        input_keywords = list(input_blueprint.core_keywords)
        generated_keywords = list(generated_blueprint.core_keywords)

        if not generated_keywords:
            return 0.0, "CRITICAL: Generated text has no keywords. Meaning lost."

        try:
            input_vecs = self.semantic_model.encode(input_keywords, convert_to_tensor=True)
            generated_vecs = self.semantic_model.encode(generated_keywords, convert_to_tensor=True)

            # Compute similarity matrix
            similarity_matrix = util.cos_sim(input_vecs, generated_vecs)

            # For each input keyword, find best match in generated
            max_scores, _ = torch.max(similarity_matrix, dim=1)

            # Count how many input keywords have a match above threshold
            covered_mask = max_scores > self.similarity_threshold
            covered_count = torch.sum(covered_mask).item()
            recall_ratio = covered_count / len(input_keywords) if input_keywords else 1.0

            if recall_ratio < self.recall_threshold:
                missing = [input_keywords[i] for i, score in enumerate(max_scores) if score.item() < self.similarity_threshold]
                return recall_ratio, f"CRITICAL: Missing concepts: {', '.join(missing[:5])}. Preserve all input meaning."

            return recall_ratio, ""
        except Exception:
            # Fallback on error
            return 0.5, "Warning: Could not compute recall score."

    def _check_precision(
        self,
        generated_blueprint: SemanticBlueprint,
        input_blueprint: SemanticBlueprint
    ) -> Tuple[float, str]:
        """Check if generated text hallucinated concepts (all heavy words must match input).

        Adjectives are treated as "style modifiers" and are more lenient (allowed if nouns/verbs match).

        Args:
            generated_blueprint: Blueprint extracted from generated text.
            input_blueprint: Original input blueprint.

        Returns:
            Tuple of (precision_score, feedback_string).
        """
        if not generated_blueprint.core_keywords:
            return 1.0, ""

        # We need to distinguish adjectives from nouns/verbs for lenient checking
        # Since blueprint only stores lemmas, we'll use a heuristic:
        # Re-extract from generated text to get POS tags
        try:
            doc = self.extractor.nlp(generated_blueprint.original_text)
            # Build map of lemma -> POS
            lemma_to_pos = {}
            for token in doc:
                if token.lemma_.lower() in generated_blueprint.core_keywords:
                    lemma_to_pos[token.lemma_.lower()] = token.pos_
        except Exception:
            # If extraction fails, treat all as nouns/verbs (stricter)
            lemma_to_pos = {}

        # Filter for "heavy words" using spaCy-based stop word filtering
        # This handles plurals automatically (e.g., "processes" -> "process")
        generated_significant = _get_significant_tokens(generated_blueprint.original_text)
        all_heavy_words = [w for w in generated_blueprint.core_keywords if w in generated_significant]

        # Separate nouns/verbs (strict) from adjectives (lenient style modifiers)
        heavy_nouns_verbs = [
            w for w in all_heavy_words
            if lemma_to_pos.get(w, "NOUN") != "ADJ"  # Treat as noun/verb if POS unknown
        ]
        heavy_adjectives = [
            w for w in all_heavy_words
            if lemma_to_pos.get(w, "") == "ADJ"
        ]

        # For precision, we check nouns/verbs strictly, adjectives are allowed as style modifiers
        heavy_words = heavy_nouns_verbs

        # Adjectives are allowed as "style modifiers" - they don't need to match input
        # Only check nouns/verbs for precision
        if not heavy_words:
            # If only adjectives, that's fine (style modifiers)
            return 1.0, ""

        # Check if each heavy word (noun/verb) matches input (via vector similarity)
        input_keywords = list(input_blueprint.core_keywords)
        if not input_keywords:
            # No input keywords to compare against - allow all
            return 1.0, ""

        if not self.semantic_model:
            # Fallback: simple set intersection
            heavy_set = set(heavy_words)
            input_set = set(input_keywords)
            overlap = len(heavy_set & input_set)
            precision_ratio = overlap / len(heavy_words) if heavy_words else 1.0

            if precision_ratio < self.precision_threshold:
                hallucinated = heavy_set - input_set
                return precision_ratio, f"CRITICAL: Hallucinated concepts: {', '.join(list(hallucinated)[:5])}. Remove concepts not in input."
            return precision_ratio, ""

        try:
            heavy_vecs = self.semantic_model.encode(heavy_words, convert_to_tensor=True)
            input_vecs = self.semantic_model.encode(input_keywords, convert_to_tensor=True)

            similarity_matrix = util.cos_sim(heavy_vecs, input_vecs)
            max_scores, _ = torch.max(similarity_matrix, dim=1)

            # Count how many heavy words have a match
            matched_mask = max_scores > self.similarity_threshold
            matched_count = torch.sum(matched_mask).item()
            precision_ratio = matched_count / len(heavy_words) if heavy_words else 1.0

            if precision_ratio < self.precision_threshold:
                hallucinated = [heavy_words[i] for i, score in enumerate(max_scores) if score.item() < self.similarity_threshold]
                return precision_ratio, f"CRITICAL: Hallucinated concepts: {', '.join(hallucinated[:5])}. Remove concepts not in input."

            return precision_ratio, ""
        except Exception:
            # Fallback on error
            return 0.5, "Warning: Could not compute precision score."

    def _check_fluency(self, generated_text: str) -> Tuple[float, str]:
        """Check grammatical fluency and naturalness of generated text.

        Uses heuristics to detect awkward phrasing:
        - Missing articles before nouns
        - Unnatural verb forms
        - Incomplete sentences
        - Awkward word combinations

        Args:
            generated_text: Text to evaluate for fluency.

        Returns:
            Tuple of (fluency_score, feedback_string).
            Score: 0.0 (completely awkward) to 1.0 (perfectly natural)
        """
        if not generated_text or not generated_text.strip():
            return 0.0, "CRITICAL: Empty text has no fluency."

        score = 1.0
        feedback_parts = []

        try:
            # Parse with spaCy
            doc = self.extractor.nlp(generated_text)

            # Check 1: Sentence completeness (must have subject and predicate)
            has_subject = False
            has_predicate = False

            for token in doc:
                if token.dep_ in ["nsubj", "nsubjpass", "csubj"]:
                    has_subject = True
                if token.pos_ == "VERB" and token.dep_ == "ROOT":
                    has_predicate = True

            if not has_subject or not has_predicate:
                score -= 0.3
                feedback_parts.append("Incomplete sentence structure")

            # Check 2: Missing articles before singular nouns
            # Pattern: verb + singular noun without article (e.g., "touch breaks")
            awkward_patterns = []
            for i, token in enumerate(doc):
                if token.pos_ == "NOUN" and token.tag_ in ["NN", "NNP"]:  # Singular noun
                    # Check if there's an article before it
                    has_article = False
                    # Look back for article
                    for j in range(max(0, i-3), i):
                        if doc[j].pos_ == "DET" and doc[j].text.lower() in ["the", "a", "an"]:
                            has_article = True
                            break
                    # Also check if it's a proper noun (doesn't need article)
                    if token.tag_ == "NNP":
                        has_article = True
                    # Check if it's part of a compound or has a possessive
                    if token.dep_ in ["compound", "nmod"]:
                        has_article = True

                    if not has_article and i > 0:
                        # Check if previous token is a verb (awkward pattern)
                        prev_token = doc[i-1]
                        if prev_token.pos_ == "VERB":
                            awkward_patterns.append(f"{prev_token.text} {token.text}")

            if awkward_patterns:
                score -= 0.2
                feedback_parts.append(f"Awkward phrasing: missing articles (e.g., '{awkward_patterns[0]}')")

            # Check 3: Awkward verb-object patterns
            # Detect patterns like "touch breaks" where verb + plural noun feels incomplete
            for token in doc:
                if token.pos_ == "VERB":
                    # Find direct objects
                    for child in token.children:
                        if child.dep_ == "dobj" and child.pos_ == "NOUN":
                            # Check if object has article or is plural
                            if child.tag_ == "NN":  # Singular noun without article
                                # This might be awkward
                                score -= 0.1
                                if not feedback_parts:
                                    feedback_parts.append("Unnatural verb-object construction")
                                break

            # Check 4: Basic grammar validation
            # Ensure sentence ends with punctuation
            if not generated_text.rstrip().endswith(('.', '!', '?')):
                score -= 0.1
                if not feedback_parts:
                    feedback_parts.append("Missing sentence-ending punctuation")

            # Ensure minimum length (very short sentences might be fragments)
            words = [t.text for t in doc if not t.is_punct and not t.is_space]
            if len(words) < 3:
                score -= 0.2
                feedback_parts.append("Sentence too short (likely fragment)")

            # Clamp score to [0.0, 1.0]
            score = max(0.0, min(1.0, score))

            feedback = " | ".join(feedback_parts) if feedback_parts else ""
            if score < 0.8 and not feedback:
                feedback = "Grammatical structure could be more natural"

            return score, feedback

        except Exception as e:
            # If parsing fails, return moderate score with warning
            return 0.6, f"Warning: Could not fully analyze fluency: {e}"

    def _validate_citations_and_quotes(
        self,
        generated_text: str,
        input_blueprint: SemanticBlueprint
    ) -> Tuple[bool, str]:
        """Validate that all citations and quotes from input are present in generated text.

        Citations must be present (position flexible).
        Quotes must be present AND exact word-for-word match.

        Args:
            generated_text: Generated text to validate.
            input_blueprint: Original input blueprint with citations and quotes.

        Returns:
            Tuple of (all_present, feedback_message).
            If all_present is False, feedback_message explains what's missing.
        """
        feedback_parts = []

        # Check citations
        if input_blueprint.citations:
            citation_pattern = r'\[\^\d+\]'
            generated_citations = set(re.findall(citation_pattern, generated_text))
            input_citations = set([cit[0] for cit in input_blueprint.citations])

            missing_citations = input_citations - generated_citations
            if missing_citations:
                feedback_parts.append(
                    f"Missing citations: {', '.join(sorted(missing_citations))}"
                )

        # Check quotes
        if input_blueprint.quotes:
            # Extract quotes from generated text
            quotation_pattern = r'["\'](?:[^"\']|(?<=\\)["\'])*["\']'
            generated_quotes = []
            for match in re.finditer(quotation_pattern, generated_text):
                quote_text = match.group(0)
                if len(quote_text.strip('"\'')) > 2:
                    generated_quotes.append(quote_text)

            # Check each input quote
            for input_quote_text, _ in input_blueprint.quotes:
                # Normalize quotes for comparison (handle both single and double)
                input_normalized = input_quote_text.strip('"\'').strip()
                found = False

                for gen_quote in generated_quotes:
                    gen_normalized = gen_quote.strip('"\'').strip()
                    # Exact match required (word-for-word)
                    if input_normalized == gen_normalized:
                        found = True
                        break

                if not found:
                    feedback_parts.append(
                        f"Missing or modified quote: {input_quote_text}"
                    )

        if feedback_parts:
            return False, " | ".join(feedback_parts)

        return True, ""

