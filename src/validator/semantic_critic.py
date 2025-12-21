"""Semantic critic for Pipeline 2.0.

This module validates generated text using precision/recall semantic checks
based on semantic blueprints rather than text overlap.
"""

import json
import re
import hashlib
from pathlib import Path
from functools import lru_cache
from typing import Dict, Tuple, Optional, Set, List
from src.ingestion.blueprint import SemanticBlueprint, BlueprintExtractor

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    np = None

try:
    import jsonrepair
    JSONREPAIR_AVAILABLE = True
except ImportError:
    JSONREPAIR_AVAILABLE = False
    jsonrepair = None

try:
    from sentence_transformers import SentenceTransformer, util
    import torch
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    SentenceTransformer = None
    util = None
    torch = None

# Domain-specific glue words
DOMAIN_GLUE_WORDS = {
    "process", "act", "state", "nature", "way", "manner", "form", "condition",
    "method", "means", "approach", "system", "structure", "framework",
    "eventually", "typically", "fundamentally"
}

# Use NLPManager for shared spaCy model (lazy loading)
_spacy_nlp = None


def _load_prompt_template(template_name: str) -> str:
    """Load a prompt template from the prompts directory.

    Args:
        template_name: Name of the template file (e.g., 'semantic_critic_coherence.md')

    Returns:
        Template content as string.
    """
    prompts_dir = Path(__file__).parent.parent.parent / "prompts"
    template_path = prompts_dir / template_name
    if not template_path.exists():
        raise FileNotFoundError(f"Prompt template not found: {template_path}")
    return template_path.read_text().strip()


def _get_significant_tokens(text: str) -> Set[str]:
    """Extract significant tokens (non-stop, non-punct) using spaCy.

    Args:
        text: Text to extract tokens from.

    Returns:
        Set of lemmatized significant tokens (non-stop, non-punct).
    """
    global _spacy_nlp

    # Lazy load spaCy model via NLPManager
    if _spacy_nlp is None:
        try:
            from src.utils.nlp_manager import NLPManager
            _spacy_nlp = NLPManager.get_nlp()
            # Add domain-specific glue words to stop words
            for word in DOMAIN_GLUE_WORDS:
                _spacy_nlp.Defaults.stop_words.add(word)
                _spacy_nlp.vocab[word].is_stop = True
        except (OSError, ImportError, RuntimeError):
            _spacy_nlp = False  # Mark as unavailable

    if not _spacy_nlp or _spacy_nlp is False:
        # Fallback to simple split if spaCy not available
        return set(text.lower().split())

    doc = _spacy_nlp(text.lower())
    significant_tokens = set()
    for token in doc:
        if not token.is_stop and not token.is_punct and not token.is_space:
            significant_tokens.add(token.lemma_)
    return significant_tokens


def _get_content_keywords(text: str) -> Set[str]:
    """Extract nouns and verbs (content keywords) from text using spaCy.

    Args:
        text: Text to extract keywords from.

    Returns:
        Set of lemmatized nouns and verbs (content keywords).
    """
    global _spacy_nlp

    # Lazy load spaCy model via NLPManager
    if _spacy_nlp is None:
        try:
            from src.utils.nlp_manager import NLPManager
            _spacy_nlp = NLPManager.get_nlp()
        except (OSError, ImportError, RuntimeError):
            _spacy_nlp = False  # Mark as unavailable

    if not _spacy_nlp or _spacy_nlp is False:
        # Fallback: extract all non-stop words
        words = text.lower().split()
        # Simple heuristic: assume most content words are nouns/verbs
        return set(w for w in words if len(w) > 2)

    doc = _spacy_nlp(text.lower())
    content_keywords = set()
    for token in doc:
        # Extract nouns (NOUN, PROPN) and verbs (VERB)
        if token.pos_ in ["NOUN", "PROPN", "VERB"] and not token.is_stop and not token.is_punct:
            content_keywords.add(token.lemma_)
    return content_keywords


class SemanticCritic:
    """Validates generated text using precision/recall semantic checks."""

    def __init__(self, similarity_threshold: float = None, config_path: str = "config.json"):
        """Initialize the semantic critic.

        Args:
            similarity_threshold: Optional override for similarity threshold.
                If None, loads from config.
            config_path: Path to configuration file.
        """
        self.config_path = config_path
        self.extractor = BlueprintExtractor()

        # Load config
        with open(config_path, 'r') as f:
            config = json.load(f)
        critic_config = config.get("semantic_critic", {})
        llm_provider_config = config.get("llm_provider", {})
        self.batch_timeout = llm_provider_config.get("batch_timeout", 180)  # Default 180 seconds for batch operations
        self.max_retries = llm_provider_config.get("max_retries", 3)
        self.retry_delay = llm_provider_config.get("retry_delay", 2)

        # Initialize evaluation cache for semantic caching
        self._evaluation_cache = {}

        # Use config values with fallbacks
        self.similarity_threshold = similarity_threshold or critic_config.get("similarity_threshold", 0.7)
        self.recall_threshold = critic_config.get("recall_threshold", 0.80)
        self.precision_threshold = critic_config.get("precision_threshold", 0.75)
        self.fluency_threshold = critic_config.get("fluency_threshold", 0.8)
        self.accuracy_weight = critic_config.get("accuracy_weight", 0.7)
        self.fluency_weight = critic_config.get("fluency_weight", 0.3)

        # Load new weights structure (with backward compatibility)
        weights_config = critic_config.get("weights")
        if weights_config:
            self.weights = weights_config
        else:
            # Backward compatibility: create weights dict from old fields
            self.weights = {
                "accuracy": self.accuracy_weight,
                "fluency": self.fluency_weight,
                "style": 0.0,
                "thesis_alignment": 0.0,
                "intent_compliance": 0.0,
                "keyword_coverage": 0.0
            }

        # Initialize thesis vector caching
        self._cached_thesis_vector = None
        self._cached_thesis_text = None

        if SENTENCE_TRANSFORMERS_AVAILABLE:
            try:
                self.semantic_model = SentenceTransformer('all-MiniLM-L6-v2')
            except Exception:
                self.semantic_model = None
        else:
            self.semantic_model = None

        # Initialize LLM provider for meaning verification (lazy initialization)
        self.use_llm_verification = critic_config.get("use_llm_verification", True)
        self.llm_provider = None
        if self.use_llm_verification:
            try:
                from src.generator.llm_provider import LLMProvider
                self.llm_provider = LLMProvider(config_path=config_path)
            except Exception:
                # LLM provider unavailable - verification will be skipped
                self.llm_provider = None

    def _safe_json_parse(self, json_string: str, verbose: bool = False) -> any:
        """Safely parse JSON string, attempting to repair if malformed.

        Uses multiple strategies to extract and repair JSON:
        1. Extract JSON from markdown code blocks
        2. Extract JSON object/array from text
        3. Try jsonrepair library if available
        4. Manual repair for common issues (trailing commas, unclosed brackets)

        Args:
            json_string: JSON string to parse.
            verbose: Whether to print debug information.

        Returns:
            Parsed JSON object.

        Raises:
            json.JSONDecodeError: If JSON cannot be parsed even after repair attempts.
        """
        if not json_string or not json_string.strip():
            raise json.JSONDecodeError("Empty JSON string", json_string, 0)

        original_string = json_string

        # Strategy 1: Extract JSON from markdown code blocks
        code_block_match = re.search(r'```(?:json)?\s*(\{.*?\}|\[.*?\])\s*```', json_string, re.DOTALL)
        if code_block_match:
            json_string = code_block_match.group(1)
            if verbose:
                print(f"  📝 Extracted JSON from markdown code block")

        # Strategy 2: Extract JSON object/array from text (more aggressive)
        if not json_string.startswith('{') and not json_string.startswith('['):
            # Try to find JSON object
            obj_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', json_string, re.DOTALL)
            if obj_match:
                json_string = obj_match.group(0)
                if verbose:
                    print(f"  📝 Extracted JSON object from text")
            else:
                # Try to find JSON array
                array_match = re.search(r'\[[^\[\]]*(?:\[[^\[\]]*\][^\[\]]*)*\]', json_string, re.DOTALL)
                if array_match:
                    json_string = array_match.group(0)
                    if verbose:
                        print(f"  📝 Extracted JSON array from text")

        # Strategy 3: Try normal parsing first
        try:
            return json.loads(json_string)
        except json.JSONDecodeError as e:
            if verbose:
                print(f"  ⚠ JSON parse error: {e}, attempting repair...")

            # Strategy 4: Try to repair using jsonrepair if available
            if JSONREPAIR_AVAILABLE and jsonrepair:
                try:
                    repaired = jsonrepair.repair_json(json_string)
                    parsed = json.loads(repaired)
                    if verbose:
                        print(f"  ✅ JSON repair successful using jsonrepair")
                    return parsed
                except Exception as repair_error:
                    if verbose:
                        print(f"  ⚠ JSON repair with jsonrepair failed: {repair_error}")

            # Strategy 5: Manual repair for common issues
            try:
                repaired = self._manual_json_repair(json_string, verbose=verbose)
                if repaired:
                    return json.loads(repaired)
            except Exception as manual_error:
                if verbose:
                    print(f"  ⚠ Manual JSON repair failed: {manual_error}")

            # If all repair strategies fail, raise original error
            raise json.JSONDecodeError(
                f"Failed to parse JSON after all repair attempts. Original error: {e.msg}",
                original_string,
                e.pos
            )

    def _manual_json_repair(self, json_string: str, verbose: bool = False) -> Optional[str]:
        """Manually repair common JSON issues.

        Fixes:
        - Trailing commas before closing brackets/braces
        - Unclosed brackets/braces
        - Missing quotes around keys
        - Single quotes instead of double quotes

        Args:
            json_string: JSON string to repair.
            verbose: Whether to print debug information.

        Returns:
            Repaired JSON string, or None if repair not possible.
        """
        repaired = json_string

        # Fix trailing commas before } or ]
        repaired = re.sub(r',\s*([}\]])', r'\1', repaired)

        # Fix single quotes to double quotes (but be careful with apostrophes in strings)
        # Only replace quotes that are clearly JSON syntax, not content
        repaired = re.sub(r"'(\w+)'\s*:", r'"\1":', repaired)  # Keys: 'key': -> "key":

        # Try to close unclosed brackets/braces (simple heuristic)
        open_braces = repaired.count('{') - repaired.count('}')
        open_brackets = repaired.count('[') - repaired.count(']')

        if open_braces > 0:
            repaired += '}' * open_braces
            if verbose:
                print(f"  🔧 Added {open_braces} closing braces")
        if open_brackets > 0:
            repaired += ']' * open_brackets
            if verbose:
                print(f"  🔧 Added {open_brackets} closing brackets")

        # Validate the repair worked
        try:
            json.loads(repaired)  # Test parse
            if verbose:
                print(f"  ✅ Manual JSON repair successful")
            return repaired
        except json.JSONDecodeError:
            if verbose:
                print(f"  ⚠ Manual repair did not produce valid JSON")
            return None

    def _calculate_thesis_alignment(self, text: str, thesis: str) -> float:
        """Calculate alignment between text and document thesis using vector similarity.

        Uses cached thesis vector to avoid re-encoding the same thesis multiple times
        during document processing.

        Args:
            text: Generated text to evaluate.
            thesis: Document thesis statement.

        Returns:
            Alignment score 0.0-1.0 (cosine similarity). Returns 1.0 if thesis is empty
            or semantic model unavailable (neutral score).
        """
        if not thesis or not thesis.strip():
            return 1.0  # Neutral score for empty thesis

        if not self.semantic_model:
            return 1.0  # Fallback if semantic model unavailable

        try:
            # Check cache: if thesis text matches, reuse cached vector
            if self._cached_thesis_text == thesis:
                thesis_vector = self._cached_thesis_vector
            else:
                # Encode thesis and cache it
                thesis_vector = self.semantic_model.encode(thesis, convert_to_tensor=True)
                self._cached_thesis_vector = thesis_vector
                self._cached_thesis_text = thesis

            # Encode text
            text_vector = self.semantic_model.encode(text, convert_to_tensor=True)

            # Calculate cosine similarity
            similarity = util.cos_sim(text_vector, thesis_vector).item()

            # Ensure result is in [0, 1] range (cosine similarity can be [-1, 1])
            return max(0.0, min(1.0, similarity))
        except Exception:
            # Fallback on any error
            return 1.0

    def _calculate_keyword_coverage(self, text: str, keywords: List[str]) -> float:
        """Calculate how many context keywords appear in the text.

        Uses hybrid matching: phrase matching for multi-word keywords,
        lemmatization for single-word keywords.

        Args:
            text: Generated text to evaluate.
            keywords: List of context keywords from global context.

        Returns:
            Coverage score 0.0-1.0 (fraction of keywords found). Returns 1.0
            if keywords list is empty (neutral score).
        """
        if not keywords or len(keywords) == 0:
            return 1.0  # Neutral score for empty keywords

        if not text or not text.strip():
            return 0.0  # No keywords found in empty text

        found_count = 0
        text_lower = text.lower()

        try:
            # Get spaCy model for lemmatization
            from src.utils.nlp_manager import NLPManager
            nlp = NLPManager.get_nlp()

            if nlp:
                # Process text once for lemmatization
                doc = nlp(text_lower)
                text_lemmas = set()
                for token in doc:
                    if not token.is_stop and not token.is_punct:
                        text_lemmas.add(token.lemma_.lower())
            else:
                # Fallback: simple word splitting
                text_lemmas = set(text_lower.split())
        except Exception:
            # Fallback: simple word splitting if NLP unavailable
            text_lemmas = set(text_lower.split())

        # Check each keyword
        for kw in keywords:
            if not kw or not kw.strip():
                continue

            kw_lower = kw.lower().strip()

            # Multi-word keyword: use phrase matching
            if " " in kw_lower:
                if kw_lower in text_lower:
                    found_count += 1
            else:
                # Single-word keyword: use lemmatization matching
                # Try to lemmatize the keyword
                try:
                    if nlp:
                        kw_doc = nlp(kw_lower)
                        kw_lemma = kw_doc[0].lemma_.lower() if len(kw_doc) > 0 else kw_lower
                    else:
                        kw_lemma = kw_lower
                except Exception:
                    kw_lemma = kw_lower

                # Check if keyword lemma exists in text lemmas
                if kw_lemma in text_lemmas or kw_lower in text_lower.split():
                    found_count += 1

        # Return fraction of keywords found (capped at 1.0)
        coverage = found_count / len(keywords) if keywords else 1.0
        return min(1.0, max(0.0, coverage))

    def _check_context_leak(
        self,
        generated_text: str,
        global_context: Optional[Dict],
        input_propositions: List[str]
    ) -> Tuple[bool, List[str], float, str]:
        """Check if global context keywords leaked into generated text.

        Detects keywords from global context that appear in output but NOT in input.
        This is a specific form of hallucination where context leaks into content.

        Args:
            generated_text: Generated text to check
            global_context: Optional global context dict with 'keywords' list
            input_propositions: List of input propositions (source of truth)

        Returns:
            Tuple of (has_leak: bool, leaked_keywords: List[str], score: float, reason: str)
            - has_leak: True if context leak detected
            - leaked_keywords: List of keywords that leaked
            - score: 0.0 if leak detected, 1.0 otherwise
            - reason: Explanation of leak or "No leak detected"
        """
        if not global_context or not global_context.get('keywords'):
            return False, [], 1.0, "No global context keywords to check"

        # Extract keywords from global context (handle both list and string formats)
        context_keywords = global_context.get('keywords', [])
        if isinstance(context_keywords, str):
            # If it's a comma-separated string, split it
            context_keywords = [k.strip() for k in context_keywords.split(',')]

        if not context_keywords:
            return False, [], 1.0, "No global context keywords to check"

        # Normalize keywords: lowercase, handle multi-word phrases
        context_keywords_lower = [k.lower().strip() for k in context_keywords if k.strip()]

        # Create a combined string of all input propositions for searching
        input_text_combined = " ".join(input_propositions).lower()

        # Check each context keyword
        leaked_keywords = []
        generated_lower = generated_text.lower()

        for keyword in context_keywords_lower:
            # Check if keyword appears in generated text
            if keyword in generated_lower:
                # Check if keyword also appears in input propositions
                if keyword not in input_text_combined:
                    # LEAK DETECTED: Keyword in output and global context, but NOT in input
                    leaked_keywords.append(keyword)

        if leaked_keywords:
            reason = f"Context leak detected: Keywords from global context ({', '.join(leaked_keywords[:3])}) appear in output but not in input propositions"
            return True, leaked_keywords, 0.0, reason

        return False, [], 1.0, "No context leak detected"

    def _verify_coherence(self, text: str, verbose: bool = False) -> Tuple[float, str]:
        """Verify text coherence using LLM-based evaluation.

        Checks for:
        1. Grammatical fluency
        2. Logical consistency (detects "word salad")
        3. Jargon hallucinations (technical terms that don't fit the narrative)
        4. Abstract academic jargon inserted into narrative

        Args:
            text: Text to evaluate for coherence.
            verbose: Whether to print debug information.

        Returns:
            Tuple of (coherence_score: float, reason: str).
            Returns (1.0, "Coherent") if LLM unavailable (neutral score).
        """
        if not self.llm_provider:
            # Fallback: return neutral score if LLM unavailable
            return 1.0, "LLM unavailable, assuming coherent"

        if not text or not text.strip():
            return 0.0, "Empty text"

        system_prompt = _load_prompt_template("semantic_critic_coherence.md")
        user_prompt = f"Analyze this text for coherence:\n\n{text}"

        try:
            response = self.llm_provider.call(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                model_type="critic",
                require_json=True,
                temperature=0.2,  # Low temperature for consistent evaluation
                max_tokens=200,
                timeout=30
            )

            # Parse JSON response
            try:
                result = json.loads(response)
            except json.JSONDecodeError:
                # Try to extract JSON from response
                json_match = re.search(r'\{[^}]+\}', response, re.DOTALL)
                if json_match:
                    result = json.loads(json_match.group())
                else:
                    # Fallback: assume coherent if parsing fails
                    return 0.5, "Could not parse LLM response, assuming partial coherence"

            # Extract score and reason
            is_coherent = result.get('is_coherent', True)
            score = float(result.get('score', 0.5))
            reason = result.get('reason', 'No reason provided')

            # Ensure score is in valid range
            score = max(0.0, min(1.0, score))

            # DEBUG: Log low coherence scores to help diagnose issues
            if verbose and score < 0.5:
                print(f"      DEBUG Coherence: score={score:.2f}, reason={reason}")
                print(f"      DEBUG Text evaluated: {text[:200]}...")
                print(f"      DEBUG LLM response: {response[:300]}...")

            return score, reason

        except Exception as e:
            # On any error, return neutral score
            return 0.5, f"Coherence check failed: {str(e)}"

    def _calculate_semantic_similarity(self, original: str, generated: str) -> float:
        """Calculate semantic similarity between original and generated text.

        Uses sentence-transformers to encode texts and compute cosine similarity.
        This automatically catches semantic issues like missing content or contradictions.

        Args:
            original: Original input text.
            generated: Generated text to compare.

        Returns:
            Similarity score between 0.0 and 1.0 (1.0 = identical meaning).
        """
        if not self.semantic_model:
            # Fallback: return 1.0 if model unavailable (let other checks handle it)
            return 1.0

        try:
            # Encode both texts
            original_embedding = self.semantic_model.encode(original, convert_to_tensor=True)
            generated_embedding = self.semantic_model.encode(generated, convert_to_tensor=True)

            # Calculate cosine similarity
            if util:
                similarity = util.cos_sim(original_embedding, generated_embedding).item()
            else:
                # Fallback if util not available
                import torch
                similarity = torch.nn.functional.cosine_similarity(
                    original_embedding, generated_embedding, dim=0
                ).item()

            return float(similarity)
        except Exception as e:
            # If encoding fails, return 1.0 to allow other checks to run
            return 1.0

    def check_ending_grounding(self, paragraph_text: str) -> Optional[str]:
        """Check if final sentence is too abstract/moralizing.

        Uses spaCy POS tagging and Matcher to detect if the ending is dominated by
        "Nouns of Thought" rather than "Nouns of Substance" (concrete objects, proper nouns).

        Args:
            paragraph_text: Full paragraph text to check

        Returns:
            Error message string if grounding issue found, None otherwise
        """
        from src.utils.spacy_linguistics import detect_moralizing_patterns
        from src.utils.nlp_manager import NLPManager

        if not paragraph_text or not paragraph_text.strip():
            return None

        try:
            nlp = NLPManager.get_nlp()
            doc = nlp(paragraph_text)

            # Get last sentence
            sentences = list(doc.sents)
            if not sentences:
                return None

            last_sent = sentences[-1]

            # Detect moralizing patterns in last sentence
            moralizing_info = detect_moralizing_patterns(last_sent)

            # Calculate concrete markers in last sentence
            concrete_count = sum(1 for t in last_sent
                                if t.pos_ in ["PROPN", "NUM"] or
                                (t.pos_ == "NOUN" and t.ent_type_))

            # Check if sentence is too abstract
            has_moralizing = moralizing_info["has_moralizing"]
            abstract_ratio = moralizing_info["abstract_ratio"]

            # Threshold: If moralizing patterns found AND concrete count is low
            if has_moralizing and concrete_count < 3:
                return (
                    "CRITICAL: Final sentence is too abstract/moralizing. "
                    "Rewrite to end with a concrete object, physical sensation, or specific detail. "
                    f"Found {concrete_count} concrete markers, abstract ratio: {abstract_ratio:.2f}"
                )

            # Also check if abstract ratio is very high even without explicit moralizing phrases
            if abstract_ratio > 0.5 and concrete_count < 2:
                return (
                    "CRITICAL: Final sentence is too abstract. "
                    "End with concrete sensory details or specific objects, not abstract concepts."
                )

            return None

        except Exception as e:
            # Fallback: return None on error (don't block generation)
            return None

    def evaluate(
        self,
        generated_text: str,
        input_blueprint: SemanticBlueprint,
        allowed_style_words: Optional[List[str]] = None,
        skeleton: Optional[str] = None,
        skeleton_type: Optional[str] = None,
        propositions: Optional[List[str]] = None,
        is_paragraph: bool = False,
        author_style_vector: Optional[np.ndarray] = None,
        style_lexicon: Optional[List[str]] = None,
        secondary_author_vector: Optional[np.ndarray] = None,
        blend_ratio: float = 0.5,
        global_context: Optional[Dict] = None,
        verbose: bool = False
    ) -> Dict[str, any]:
        """Evaluate generated text against input blueprint.

        Uses two-gate validation system:
        - Gate 1: Template Fit (skeleton adherence) - if skeleton provided
        - Gate 2: Meaning Preservation (recall)
        - Final Heuristic: LLM meaning verification

        Args:
            generated_text: Generated text to evaluate.
            input_blueprint: Original input blueprint to compare against.
            allowed_style_words: Optional list of style words to whitelist.
            skeleton: Optional skeleton template for adherence checking.

        Returns:
            Dict with:
            - pass: bool
            - recall_score: float (0-1)
            - precision_score: float (0-1)
            - adherence_score: float (0-1)
            - llm_meaning_score: float (0-1)
            - score: float (0-1, weighted: adherence * 0.5 + recall * 0.5)
            - feedback: str
        """
        # CRITICAL: Copy-Paste Check - reject exact copies of original text
        # This forces the system to transform the style, not just copy verbatim
        if generated_text.strip() == input_blueprint.original_text.strip():
            return {
                "pass": False,
                "score": 0.1,  # Not 0.0 to allow evolution to improve from this state
                "recall_score": 1.0,  # Technically perfect recall, but...
                "precision_score": 1.0,  # Technically perfect precision, but...
                "fluency_score": 1.0,  # Technically fluent, but...
                "feedback": "CRITICAL: Generated text is identical to input. You must transform the style, not copy it verbatim. The output must be different from the input while preserving meaning.",
                "primary_failure_type": "meaning"
            }

        if not generated_text or not generated_text.strip():
            return {
                "pass": False,
                "recall_score": 0.0,
                "precision_score": 0.0,
                "fluency_score": 0.0,
                "score": 0.0,
                "feedback": "CRITICAL: Generated text is empty."
            }

        # CRITICAL: Validate citations and quotes FIRST (non-negotiable)
        # This must happen before other checks to ensure citations are always validated
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

        original_text = input_blueprint.original_text

        # PARAGRAPH MODE: Use proposition-based evaluation
        if is_paragraph and propositions:
            return self._evaluate_paragraph_mode(
                generated_text, original_text, propositions, author_style_vector,
                style_lexicon=style_lexicon,
                secondary_author_vector=secondary_author_vector,
                blend_ratio=blend_ratio,
                verbose=verbose,
                global_context=global_context
            )

        # SENTENCE MODE: Continue with existing sentence-level logic
        # HARD GATE 1.5: Noun Preservation Check (ALWAYS RUN - critical for meaning preservation)
        # Extract concrete nouns from original and ensure they appear in output
        if original_text:
            try:
                # Use shared NLPManager for spaCy model
                from src.utils.nlp_manager import NLPManager
                nlp = NLPManager.get_nlp()
            except (OSError, ImportError, RuntimeError):
                nlp = None

            if nlp:
                try:
                    # Extract nouns from original text
                    original_doc = nlp(original_text)
                    original_nouns = set()
                    for token in original_doc:
                        if token.pos_ == "NOUN" and not token.is_stop:
                            # Use lemma for matching (handles plurals/singulars)
                            original_nouns.add(token.lemma_.lower())

                    # Extract nouns from generated text
                    generated_doc = nlp(generated_text)
                    generated_nouns = set()
                    for token in generated_doc:
                        if token.pos_ == "NOUN" and not token.is_stop:
                            generated_nouns.add(token.lemma_.lower())

                    # HARD GATE: If original had nouns but output has ZERO matching nouns, fail
                    # BUT: Use semantic similarity to catch synonyms (e.g., "nation-state" vs "country")
                    if original_nouns and not any(noun in generated_nouns for noun in original_nouns):
                        # Try semantic similarity check for synonyms
                        preserved_semantically = False
                        if self.semantic_model:
                            try:
                                # Check each original noun against all generated nouns using semantic similarity
                                for orig_noun in original_nouns:
                                    orig_emb = self.semantic_model.encode(orig_noun, convert_to_tensor=True)
                                    for gen_noun in generated_nouns:
                                        gen_emb = self.semantic_model.encode(gen_noun, convert_to_tensor=True)
                                        # Use cosine similarity
                                        try:
                                            from sentence_transformers import util
                                            similarity = util.cos_sim(orig_emb, gen_emb).item()
                                        except (ImportError, AttributeError):
                                            # Fallback: use torch directly
                                            import torch
                                            similarity = torch.nn.functional.cosine_similarity(
                                                orig_emb, gen_emb, dim=0
                                            ).item()

                                        if similarity > 0.6:  # Synonym threshold
                                            preserved_semantically = True
                                            break
                                    if preserved_semantically:
                                        break
                            except Exception as e:
                                # If semantic check fails, continue to entailment check
                                pass

                        # If semantic similarity didn't find a match, try entailment check
                        if not preserved_semantically and self.llm_provider and original_text:
                            try:
                                # Check if the generated text entails the original nouns using LLM
                                # Create a simple proposition: "The text mentions [noun]"
                                noun_propositions = [f"The text mentions {noun}." for noun in list(original_nouns)[:3]]
                                for prop in noun_propositions:
                                    entails, confidence = self._check_entailment(prop, generated_text, verbose=False)
                                    if entails and confidence >= 0.8:
                                        preserved_semantically = True
                                        break
                            except Exception:
                                # If entailment check fails, continue to fail case
                                pass

                        # Only fail if BOTH exact match AND semantic similarity AND entailment all fail
                        if not preserved_semantically:
                            missing_nouns = list(original_nouns)[:5]  # Show first 5 missing nouns
                            return {
                                "pass": False,
                                "recall_score": 0.0,
                                "precision_score": 0.0,
                                "fluency_score": 0.0,
                                "score": 0.0,
                                "feedback": f"CRITICAL: Key nouns from original text are missing in output. Missing nouns: {', '.join(missing_nouns)}. Restore all key concepts from the original text."
                            }
                except Exception:
                    # If noun check fails, continue (don't block on spaCy errors)
                    pass

        # HARD GATE: Semantic Similarity Check (Phase 3)
        # This replaces manual word-count and fragment checks with embedding-based validation
        similarity = None
        if self.semantic_model and original_text:
            similarity = self._calculate_semantic_similarity(original_text, generated_text)
            # Hard floor: 0.65 (below this is likely hallucination)
            # Lowered from 0.75 to allow style transfer with semantic variation
            if similarity < 0.65:
                return {
                    "pass": False,
                    "recall_score": 0.0,
                    "precision_score": 0.0,
                    "fluency_score": 0.0,
                    "score": 0.0,
                    "feedback": f"CRITICAL: Semantic similarity too low ({similarity:.2f} < 0.65). Generated text does not preserve meaning from original."
                }
            # Grace zone: 0.65-0.75 - will check recall/fluency later for override
            # 0.75-0.85 is acceptable for style transfer

        # Fallback checks (only if semantic model unavailable)
        # These are kept as backup but semantic similarity is preferred
        if not self.semantic_model and original_text:
            # HARD GATE 1: Compression Ratio Check (Fallback)
            # Check if output has lost too much content relative to input
            input_len = len(original_text.split())
            output_len = len(generated_text.split())
            ratio = output_len / max(1, input_len)

            # Stricter threshold based on input length
            # For short sentences (<10 words), require 60% retention
            # For longer sentences, require 50% retention
            if input_len < 10 and ratio <= 0.6:
                return {
                    "pass": False,
                    "recall_score": 0.0,
                    "precision_score": 0.0,
                    "fluency_score": 0.0,
                    "score": 0.0,
                    "feedback": f"CRITICAL: Semantic collapse. Output ({output_len} words) lost >40% of original content ({input_len} words). Restore missing concepts from the original text."
                }
            elif input_len >= 10 and ratio < 0.5:
                return {
                    "pass": False,
                    "recall_score": 0.0,
                    "precision_score": 0.0,
                    "fluency_score": 0.0,
                    "score": 0.0,
                    "feedback": f"CRITICAL: Semantic collapse. Output ({output_len} words) lost >50% of original content ({input_len} words). Restore missing concepts from the original text."
                }

        # HARD GATE 2: Fragment Check (Fallback - only if semantic model unavailable)
        # Semantic similarity check above should catch fragments, but keep this as backup
        if not self.semantic_model:
            try:
                # Use shared NLPManager for spaCy model
                from src.utils.nlp_manager import NLPManager
                nlp = NLPManager.get_nlp()

                if nlp:
                    doc = nlp(generated_text)
                    has_verb = any(token.pos_ == "VERB" for token in doc)
                    has_subj = any(token.dep_ == "nsubj" or token.dep_ == "nsubjpass" for token in doc)

                    # Also check for imperative sentences (no explicit subject but verb is root)
                    is_imperative = False
                    if has_verb and not has_subj:
                        # Check if root is a verb (imperative sentences)
                        root = [token for token in doc if token.dep_ == "ROOT"]
                        if root and root[0].pos_ == "VERB":
                            # Imperative sentences are valid (e.g., "Stop.")
                            is_imperative = True

                    if not (has_verb and (has_subj or is_imperative)):
                        return {
                            "pass": False,
                            "recall_score": 0.0,
                            "precision_score": 0.0,
                            "fluency_score": 0.0,
                            "score": 0.0,
                            "feedback": "CRITICAL: Sentence fragment detected (missing Subject or Verb). Complete the thought with a proper main clause."
                        }
            except (OSError, ImportError, RuntimeError):
                # If spaCy not available, skip fragment check
                pass
            except Exception:
                # If fragment check fails, continue (don't block on spaCy errors)
                pass

        # HARD GATE 3: Logic Contradiction Check (The Oxymoron Killer)
        # Use spaCy's dependency parsing and word vectors to detect semantic contradictions
        try:
            # Use shared NLPManager for spaCy model
            from src.utils.nlp_manager import NLPManager
            nlp = NLPManager.get_nlp()
            has_vectors = hasattr(nlp.vocab, 'vectors') and len(nlp.vocab.vectors) > 0

            if nlp:
                doc = nlp(generated_text)
                contradictions = []

                # Use dependency parsing to find modifier-noun pairs
                for token in doc:
                    if token.pos_ == "NOUN":
                        # Check for adjective modifiers (amod dependency)
                        for child in token.children:
                            if child.dep_ == "amod" and child.pos_ == "ADJ":
                                modifier = child
                                noun = token

                                # Use spaCy's semantic analysis to detect contradictions
                                if has_vectors and modifier.has_vector and noun.has_vector:
                                    # Check semantic compatibility using word vectors
                                    similarity = modifier.similarity(noun)

                                    # Very low similarity suggests potential contradiction
                                    # But we need to be smarter - check semantic properties
                                    # Words meaning "unlimited" should be similar to each other
                                    # Words meaning "limited" should be similar to each other
                                    # If modifier is "unlimited-type" and noun is "limited-type", it's a contradiction

                                    # Use semantic analysis to detect contradiction patterns
                                    # Dynamically determine if words are "unlimited" or "limited" using spaCy
                                    # Use seed concepts and semantic similarity
                                    modifier_unlimited_score = 0.0
                                    modifier_limited_score = 0.0

                                    # Use single seed words and find semantically similar words dynamically
                                    try:
                                        unlimited_seed = nlp.vocab["unlimited"]
                                        limited_seed = nlp.vocab["limited"]

                                        if unlimited_seed.has_vector and modifier.has_vector:
                                            modifier_unlimited_score = modifier.similarity(unlimited_seed)

                                        if limited_seed.has_vector and modifier.has_vector:
                                            modifier_limited_score = modifier.similarity(limited_seed)
                                    except (KeyError, AttributeError):
                                        pass

                                    # Check if noun is semantically similar to "limited" concepts
                                    noun_unlimited_score = 0.0
                                    noun_limited_score = 0.0

                                    try:
                                        unlimited_seed = nlp.vocab["unlimited"]
                                        limited_seed = nlp.vocab["limited"]

                                        if unlimited_seed.has_vector and noun.has_vector:
                                            noun_unlimited_score = noun.similarity(unlimited_seed)

                                        if limited_seed.has_vector and noun.has_vector:
                                            noun_limited_score = noun.similarity(limited_seed)
                                    except (KeyError, AttributeError):
                                        pass

                                    # Contradiction: modifier implies "unlimited" but noun implies "limited"
                                    # OR modifier implies "limited" but noun implies "unlimited"
                                    if (modifier_unlimited_score > 0.5 and noun_limited_score > 0.5) or \
                                       (modifier_limited_score > 0.5 and noun_unlimited_score > 0.5):
                                        contradictions.append((modifier.text, noun.text))

                                    # Also check direct similarity - very low similarity with modifier-noun
                                    # combined with semantic property mismatch is a strong signal
                                    elif similarity < 0.2 and (modifier_unlimited_score > 0.4 or modifier_limited_score > 0.4):
                                        contradictions.append((modifier.text, noun.text))

                # Also check for adverb-adjective pairs that might create contradictions
                for token in doc:
                    if token.pos_ == "ADJ":
                        for child in token.children:
                            if child.dep_ == "advmod" and child.pos_ == "ADV":
                                adverb = child
                                adjective = token

                                if has_vectors and adverb.has_vector and adjective.has_vector:
                                    # Similar semantic analysis for adverb-adjective pairs using dynamic detection
                                    adverb_unlimited_score = 0.0
                                    adj_limited_score = 0.0

                                    try:
                                        # Use seed words for unlimited/limited concepts
                                        unlimited_seed = nlp.vocab["unlimited"]
                                        limited_seed = nlp.vocab["limited"]

                                        # For adverbs, check similarity to unlimited concept
                                        if unlimited_seed.has_vector and adverb.has_vector:
                                            adverb_unlimited_score = adverb.similarity(unlimited_seed)

                                        # For adjectives, check similarity to limited concept
                                        if limited_seed.has_vector and adjective.has_vector:
                                            adj_limited_score = adjective.similarity(limited_seed)
                                    except (KeyError, AttributeError):
                                        pass

                                    if adverb_unlimited_score > 0.5 and adj_limited_score > 0.5:
                                        contradictions.append((adverb.text, adjective.text))

                if contradictions:
                    contradiction_pairs = [f"'{mod}' contradicts '{noun}'" for mod, noun in contradictions[:3]]
                    return {
                        "pass": False,
                        "recall_score": 0.0,
                        "precision_score": 0.0,
                        "fluency_score": 0.0,
                        "score": 0.0,
                        "feedback": f"CRITICAL: Logical contradiction detected: {', '.join(contradiction_pairs)}. Use modifiers that align with the noun's definition."
                    }
        except (OSError, ImportError, RuntimeError):
            # If spaCy not available, skip logic check
            pass
        except Exception:
            # If logic check fails, continue (don't block on spaCy errors)
            pass

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
            input_blueprint,
            allowed_style_words=allowed_style_words,
            skeleton=skeleton
        )

        # Gate 1: Template Fit (Skeleton Adherence)
        adherence_score = 1.0
        adherence_feedback = ""
        if skeleton:
            adherence_score, adherence_feedback = self._check_adherence(generated_text, skeleton)

        # Gate 2: Meaning Preservation (Recall)
        # Hard gate: If recall < 0.7, fail
        # Hard gate: If adherence < 0.8, fail
        passes = (recall_score >= 0.7 and adherence_score >= 0.8)

        # Final Heuristic: LLM Meaning Verification
        llm_meaning_preserved = True
        llm_confidence = 1.0
        llm_intent_score = 1.0
        llm_explanation = ""
        logic_fail = False
        if self.use_llm_verification and self.llm_provider:
            # Extract intent from global_context if available
            intent = None
            if global_context:
                intent = global_context.get('intent')

            llm_meaning_preserved, llm_confidence, llm_intent_score, llm_explanation = self._verify_meaning_with_llm(
                original_text, generated_text, global_context=global_context, intent=intent
            )
            # If LLM detects meaning loss with high confidence, override pass status
            if not llm_meaning_preserved and llm_confidence > 0.7:
                passes = False

            # LOGIC VETO: Use specialized logic verification with rhetorical context
            # Infer skeleton_type from skeleton if not provided
            inferred_skeleton_type = skeleton_type
            if not inferred_skeleton_type and skeleton:
                inferred_skeleton_type = self._infer_skeleton_type(skeleton, generated_text)

            logic_fail, logic_reason = self._verify_logic(
                original_text, generated_text, inferred_skeleton_type or "DECLARATIVE", global_context=global_context
            )
            if logic_fail:
                passes = False
                # Store logic reason for feedback
                llm_explanation = logic_reason  # Override with logic reason

        # Calculate context-aware metrics if global_context is available
        thesis_alignment = None
        keyword_coverage = None
        intent_score = None

        if global_context:
            # Thesis alignment
            thesis = global_context.get('thesis')
            if thesis:
                thesis_alignment = self._calculate_thesis_alignment(generated_text, thesis)

            # Keyword coverage
            keywords = global_context.get('keywords')
            if keywords and isinstance(keywords, list):
                keyword_coverage = self._calculate_keyword_coverage(generated_text, keywords)

            # Intent score (already calculated in LLM verification if intent provided)
            if global_context.get('intent'):
                intent_score = llm_intent_score

        # Calculate composite score with dynamic weight normalization
        final_score = self._calculate_composite_score({
            "accuracy": (adherence_score * 0.5) + (recall_score * 0.5),  # Combined accuracy metric
            "fluency": precision_score,  # Precision as fluency proxy
            "style": 1.0,  # Placeholder for style score (not calculated here)
            "thesis_alignment": thesis_alignment,
            "intent_compliance": intent_score,
            "keyword_coverage": keyword_coverage
        })

        # Apply logic veto cap after composite score calculation
        if logic_fail:
            # HARD CAP: Force score to 0.45 if logic is wrong, regardless of other scores
            final_score = min(final_score, 0.45)

        # Build metrics dictionary
        metrics = {
            "recall_score": recall_score,
            "precision_score": precision_score,
            "adherence_score": adherence_score,
            "llm_meaning_score": llm_confidence,
            "thesis_alignment": thesis_alignment,
            "intent_score": intent_score,
            "keyword_coverage": keyword_coverage
        }

        feedback_parts = []
        if recall_score < 0.7:
            feedback_parts.append(recall_feedback)
        if adherence_score < 0.8 and skeleton:
            feedback_parts.append(adherence_feedback)
        if not llm_meaning_preserved and llm_confidence > 0.7:
            feedback_parts.append(f"LLM Verification: {llm_explanation}")
        if logic_fail:
            # Use logic reason from _verify_logic (stored in llm_explanation if logic failed)
            logic_reason = llm_explanation if logic_fail and llm_explanation else "Logic verification failed"
            feedback_parts.append(f"LOGIC VETO: {logic_reason}")

        return {
            "pass": passes,
            "recall_score": recall_score,
            "precision_score": precision_score,
            "adherence_score": adherence_score,
            "llm_meaning_score": llm_confidence,
            "logic_fail": logic_fail,
            "score": final_score,
            "metrics": metrics,
            "thesis_alignment": thesis_alignment,
            "intent_score": intent_score,
            "keyword_coverage": keyword_coverage,
            "feedback": " ".join(feedback_parts) if feedback_parts else "Passed semantic validation."
        }

    def _verify_meaning_with_llm(self, original_text: str, generated_text: str, global_context: Optional[Dict] = None, intent: Optional[str] = None) -> Tuple[bool, float, float, str]:
        """Verify meaning preservation using LLM (Final Heuristic).

        Uses LLM to compare original and generated text, confirming core meaning is unchanged.
        Also assesses intent alignment if intent is provided.

        Args:
            original_text: Original input text.
            generated_text: Generated text to verify.
            global_context: Optional global context dictionary.
            intent: Optional intent string (e.g., "persuading", "narrating", "informing").

        Returns:
            Tuple of (meaning_preserved: bool, confidence: float, intent_score: float, explanation: str).
        """
        if not self.llm_provider:
            return True, 0.5, 1.0, "LLM verification unavailable"

        system_prompt = _load_prompt_template("semantic_critic_meaning_system.md")

        # Build user prompt from template
        user_template = _load_prompt_template("semantic_critic_meaning_user.md")

        # Build context section
        global_context_section = ""
        if global_context and global_context.get('thesis'):
            global_context_section = f"\n\nCONTEXT: The document is about {global_context['thesis']}. Verify that the generated text aligns with this context and doesn't introduce off-topic content."

        # Build intent task
        intent_task = ""
        intent_score_field = ""
        if intent:
            intent_task = f"""

Task 2: Assess intent alignment.
Does the generated text align with the intent '{intent}'? Consider:
- "persuading": Does it use persuasive language, arguments, or calls to action?
- "narrating": Does it tell a story or describe events in sequence?
- "informing": Does it present facts, explanations, or educational content?
- "analyzing": Does it break down concepts, compare, or examine relationships?"""
            intent_score_field = '"intent_score": 1.0/0.5/0.0,  // 1.0 = Yes, 0.5 = Partial, 0.0 = No\n    '

        user_prompt = user_template.format(
            original_text=original_text,
            generated_text=generated_text,
            global_context_section=global_context_section,
            intent_task=intent_task,
            intent_score_field=intent_score_field
        )

        try:
            response = self.llm_provider.call(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                model_type="critic",
                require_json=True,
                temperature=0.2,
                max_tokens=200
            )

            # Parse JSON response
            import json
            try:
                result = json.loads(response)
            except json.JSONDecodeError:
                # Try to extract JSON from response
                json_match = re.search(r'\{[^}]+\}', response, re.DOTALL)
                if json_match:
                    result = json.loads(json_match.group())
                else:
                    # Fallback: assume meaning preserved if parsing fails
                    return True, 0.5, "LLM response parsing failed"

            meaning_preserved = result.get("meaning_preserved", True)
            confidence = float(result.get("confidence", 0.5))
            intent_score = float(result.get("intent_score", 1.0)) if intent else 1.0
            explanation = result.get("explanation", "")

            return meaning_preserved, confidence, intent_score, explanation
        except Exception as e:
            # Handle errors gracefully - don't block evaluation
            return True, 0.5, 1.0, f"LLM verification unavailable: {str(e)}"

    def _calculate_composite_score(self, metrics: Dict[str, Optional[float]]) -> float:
        """Calculate composite score with dynamic weight normalization.

        Only includes metrics that are not None in the weighted average.
        This ensures scores are mathematically sound regardless of whether
        context is enabled or disabled (no artificial score inflation).

        Args:
            metrics: Dictionary of metric names to scores (None if not available).

        Returns:
            Composite score 0.0-1.0 (weighted average of available metrics).
        """
        # Identify active metrics (keys where value is not None)
        active_metrics = {k: v for k, v in metrics.items() if v is not None}

        if not active_metrics:
            # No active metrics - return neutral score
            return 0.5

        # Get weights for active metrics
        active_weights = {}
        for key in active_metrics.keys():
            if key in self.weights:
                active_weights[key] = self.weights[key]
            else:
                # Fallback: use default weight if not in config
                # This handles backward compatibility
                default_weights = {
                    "accuracy": self.accuracy_weight if hasattr(self, 'accuracy_weight') else 0.7,
                    "fluency": self.fluency_weight if hasattr(self, 'fluency_weight') else 0.3,
                    "style": 0.0,
                    "thesis_alignment": 0.0,
                    "intent_compliance": 0.0,
                    "keyword_coverage": 0.0
                }
                active_weights[key] = default_weights.get(key, 0.0)

        # Calculate total weight of active metrics
        total_weight = sum(active_weights.values())

        if total_weight == 0:
            # No weights configured - return simple average
            return sum(active_metrics.values()) / len(active_metrics)

        # Normalize weights and calculate weighted average
        composite = sum(
            metrics[k] * (active_weights[k] / total_weight)
            for k in active_metrics.keys()
        )

        return max(0.0, min(1.0, composite))

    def _infer_skeleton_type(self, skeleton: str, generated_text: str) -> str:
        """Infer skeleton type from skeleton pattern and generated text.

        Args:
            skeleton: Skeleton template string
            generated_text: Generated text to check

        Returns:
            Skeleton type: "RHETORICAL_QUESTION", "CONDITIONAL", or "DECLARATIVE"
        """
        skeleton_lower = skeleton.lower()
        text_lower = generated_text.lower()

        # Check for rhetorical question patterns
        if "?" in skeleton or "?" in generated_text:
            # Check if it's actually a question
            if generated_text.strip().endswith("?") or any(q in skeleton_lower for q in ["how", "why", "what", "when", "where", "who"]):
                return "RHETORICAL_QUESTION"

        # Check for conditional patterns
        if any(cond in skeleton_lower for cond in ["if", "when", "unless", "provided that", "in case"]):
            return "CONDITIONAL"

        # Default to declarative
        return "DECLARATIVE"

    def _verify_logic(
        self,
        original_text: str,
        generated_text: str,
        skeleton_type: str = "DECLARATIVE",
        global_context: Optional[Dict] = None
    ) -> Tuple[bool, str]:
        """Verify logic preservation with rhetorical context awareness.

        Uses specialized prompt that explicitly permits authorized rhetorical transformations.

        Args:
            original_text: Original input text
            generated_text: Generated text to verify
            skeleton_type: Type of skeleton structure (RHETORICAL_QUESTION, CONDITIONAL, DECLARATIVE)

        Returns:
            True if logic fails (should reject), False if logic is preserved
        """
        if not self.llm_provider:
            return False  # No LLM available, don't block on logic

        logic_template = _load_prompt_template("semantic_critic_logic.md")

        # Build global context section
        global_context_section = ""
        if global_context and global_context.get('thesis'):
            global_context_section = f"\n\nCONTEXT: The document is about {global_context['thesis']}. Verify that the generated text aligns with this context."

        try:
            prompt = logic_template.format(
                original_text=original_text,
                generated_text=generated_text,
                skeleton_type=skeleton_type,
                global_context_section=global_context_section
            )

            response = self.llm_provider.call(
                system_prompt="You are a precision logic validator. Output ONLY valid JSON.",
                user_prompt=prompt,
                model_type="critic",
                require_json=True,
                temperature=0.2,
                max_tokens=200
            )

            # Parse JSON response
            import json
            try:
                result = json.loads(response)
            except json.JSONDecodeError:
                # Try to extract JSON from response
                json_match = re.search(r'\{[^}]+\}', response, re.DOTALL)
                if json_match:
                    result = json.loads(json_match.group())
                else:
                    # Fallback: assume logic preserved if parsing fails
                    return False

            logic_fail = result.get("logic_fail", False)
            reason = result.get("reason", "")
            reason_lower = reason.lower()

            # WHITELIST LOGIC: Override false positives for authorized transformations
            if logic_fail:
                # Whitelist Rhetorical Questions
                if skeleton_type == "RHETORICAL_QUESTION" and "?" in generated_text:
                    if any(keyword in reason_lower for keyword in ["question", "asks", "interrogative", "query"]):
                        # OVERRIDE: This is an authorized transformation
                        return False, reason

                # Whitelist Conditional Formatting (e.g., "It is true that...", "If we consider...")
                if skeleton_type == "CONDITIONAL":
                    if "conditional relationship" in reason_lower and any(keyword in reason_lower for keyword in ["wrapper", "format", "structure", "framing"]):
                        # If the LLM flagged a "wrapper" clause, ignore it
                        return False, reason
                    if "adds condition" in reason_lower and "context" in reason_lower:
                        # If condition is contextual framing, allow it
                        return False, reason

            return logic_fail, reason

        except Exception as e:
            # If verification fails, don't block (assume logic preserved)
            return False, f"Logic verification unavailable: {str(e)}"

    def _check_adherence(self, generated_text: str, skeleton: str) -> Tuple[float, str]:
        """Check if generated text adheres to skeleton structure (Gate 1).

        Extracts anchor words from skeleton (all words outside [NP], [VP], [ADJ] brackets)
        and verifies they appear in generated text in roughly the same relative order.

        Args:
            generated_text: Generated text to check.
            skeleton: Skeleton template with placeholders.

        Returns:
            Tuple of (adherence_score, feedback_string).
        """
        if not skeleton or not skeleton.strip():
            return 1.0, ""

        if not generated_text or not generated_text.strip():
            return 0.0, "Failed to match template structure: generated text is empty"

        # Extract anchor words from skeleton (all words outside placeholders)
        skeleton_clean = re.sub(r'\[NP\]|\[VP\]|\[ADJ\]', '', skeleton, flags=re.IGNORECASE)
        skeleton_tokens = re.findall(r'\b\w+\b', skeleton_clean.lower())

        if not skeleton_tokens:
            # No anchor words in skeleton - cannot check adherence
            return 1.0, ""

        # Extract words from generated text
        generated_tokens = re.findall(r'\b\w+\b', generated_text.lower())

        # Check if anchor words appear in generated text in roughly the same order
        matched_count = 0
        generated_idx = 0

        for anchor_word in skeleton_tokens:
            # Search for anchor word in generated text starting from current position
            found = False
            for i in range(generated_idx, len(generated_tokens)):
                if generated_tokens[i] == anchor_word:
                    matched_count += 1
                    generated_idx = i + 1  # Move forward
                    found = True
                    break
            # If not found, continue (don't break - allow some flexibility)

        adherence_score = matched_count / len(skeleton_tokens) if skeleton_tokens else 0.0

        if adherence_score < 0.8:
            return adherence_score, "Failed to match template structure"

        return adherence_score, ""

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
        input_blueprint: SemanticBlueprint,
        allowed_style_words: Optional[List[str]] = None,
        skeleton: Optional[str] = None
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

        # Extract skeleton anchor words and add to whitelist
        skeleton_anchor_words = set()
        if skeleton:
            skeleton_clean = re.sub(r'\[NP\]|\[VP\]|\[ADJ\]', '', skeleton, flags=re.IGNORECASE)
            skeleton_tokens = re.findall(r'\b\w+\b', skeleton_clean.lower())
            skeleton_anchor_words = {token.lower() for token in skeleton_tokens}

        # Create whitelist: input keywords + skeleton anchor words
        allowed_words = set(input_keywords) | skeleton_anchor_words

        if not self.semantic_model:
            # Fallback: simple set intersection
            heavy_set = set(heavy_words)
            input_set = set(input_keywords)
            overlap = len(heavy_set & input_set)
            precision_ratio = overlap / len(heavy_words) if heavy_words else 1.0

            if precision_ratio < self.precision_threshold:
                hallucinated = heavy_set - allowed_words  # Use whitelist instead of just input_set
                # Filter out allowed style words (additional whitelist)
                if allowed_style_words:
                    style_words_set = {w.lower().strip() for w in allowed_style_words}
                    hallucinated = {w for w in hallucinated if w.lower() not in style_words_set}
                if hallucinated:
                    return precision_ratio, f"CRITICAL: Hallucinated concepts: {', '.join(list(hallucinated)[:5])}. Remove concepts not in input."
                # If all "hallucinated" words are actually style words, precision is acceptable
                return 1.0, ""
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
                # Filter out words in whitelist (input keywords + skeleton anchor words)
                hallucinated = [w for w in hallucinated if w.lower() not in allowed_words]
                # Filter out allowed style words (additional whitelist)
                if allowed_style_words:
                    style_words_set = {w.lower().strip() for w in allowed_style_words}
                    # Direct match filter
                    hallucinated = [w for w in hallucinated if w.lower() not in style_words_set]

                    # Embedding similarity check: if word is similar to any style word (> 0.8), also ignore it
                    # This allows 'existence' even if lexicon only has 'reality'
                    if self.semantic_model and hallucinated:
                        try:
                            # Encode hallucinated words and style words
                            hallucinated_vecs = self.semantic_model.encode(hallucinated, convert_to_tensor=True)
                            style_vecs = self.semantic_model.encode(list(style_words_set), convert_to_tensor=True)

                            # Calculate similarity matrix
                            similarity_matrix = util.cos_sim(hallucinated_vecs, style_vecs)
                            max_similarities, _ = torch.max(similarity_matrix, dim=1)

                            # Filter out words that are similar to style words (> 0.8 similarity)
                            filtered_hallucinated = [
                                hallucinated[i] for i, sim in enumerate(max_similarities)
                                if sim.item() <= 0.8  # Only keep if NOT similar to style words
                            ]
                            hallucinated = filtered_hallucinated
                        except Exception:
                            # If embedding check fails, continue with direct match filter only
                            pass

                if hallucinated:
                    return precision_ratio, f"CRITICAL: Hallucinated concepts: {', '.join(hallucinated[:5])}. Remove concepts not in input."
                # If all "hallucinated" words are actually style words or conceptually related, precision is acceptable
                return 1.0, ""

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
        - Interrupted future tense (stilted patterns)
        - Unnecessary passive voice

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
            # Check for interrupted future tense (stilted pattern)
            # Pattern: will/shall + [comma] + [any adverb phrase] + [comma] + verb
            # Catches: "will, in time, be", "shall, ultimately, succumb", "will, eventually, break"
            stilted_pattern = re.search(r'\b(will|shall)\s*,\s*[^,]+,\s*(be\s+)?\w+', generated_text, re.IGNORECASE)
            if stilted_pattern:
                score -= 0.5
                feedback_parts.append("CRITICAL: Stilted phrasing detected. Do not interrupt 'will/shall' with commas. Use 'eventually [verb]' instead.")

            # Check for future passive voice (weak construction)
            # Pattern: will be + past participle (ending in -en or -ed)
            # Matches: will be broken, will be taken, will be given
            future_passive = re.search(r'\bwill\s+be\s+\w+(?:en|ed)\b', generated_text, re.IGNORECASE)
            if future_passive:
                score -= 0.15
                feedback_parts.append("Prefer Active Voice for universal statements (e.g., 'breaks' instead of 'will be broken').")

            # Parse with spaCy
            nlp = self.extractor._get_nlp()
            if nlp is None:
                return 0.6, "Warning: Could not fully analyze fluency: spaCy model not available"
            doc = nlp(generated_text)

            # Check 1: Sentence completeness (must have subject and predicate)
            has_subject = False
            has_predicate = False

            for token in doc:
                if token.dep_ in ["nsubj", "nsubjpass", "csubj"]:
                    has_subject = True
                if token.pos_ == "VERB" and token.dep_ == "ROOT":
                    has_predicate = True

            # Relaxed check: Only flag incomplete structure if the current score is also low
            # Complex sentences with good overall fluency should not be flagged as incomplete
            # We check the score after other penalties to see if it's still high
            if not has_subject or not has_predicate:
                # Calculate what the score would be after this penalty
                potential_score = score - 0.3
                # If the score would still be > 0.8 after penalty, assume structure is fine
                # This prevents false positives on complex clauses that parse differently
                if potential_score <= 0.8:
                    score -= 0.3
                    feedback_parts.append("Incomplete sentence structure")
                # If score would remain > 0.8, trust the LLM's judgment - don't penalize

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

    def _check_proposition_recall(self, generated_text: str, propositions: List[str], similarity_threshold: float = 0.45, verbose: bool = False) -> Tuple[float, Dict]:
        """Check proposition recall by comparing each proposition against generated sentences.

        CRITICAL: Do NOT compare proposition vector against whole paragraph vector (too much noise).
        Instead, split paragraph into sentences and find the Maximum Similarity for each proposition.

        Uses three-tier matching:
        1. Vector similarity above threshold → preserved
        2. Hybrid keyword matching (0.25-0.30 similarity + 75% keyword overlap) → preserved
        3. Entailment check (0.20-0.30 similarity, LLM verifies no contradiction) → preserved

        Args:
            generated_text: Generated paragraph text.
            propositions: List of atomic proposition strings.
            similarity_threshold: Similarity threshold for matching (default: 0.45 for sentence mode,
                                0.30 recommended for paragraph mode to account for stylized text).
                                Hybrid keyword matching applies when similarity is 0.25-0.30.
            verbose: Whether to print debug information for entailment checks.

        Returns:
            Tuple of (recall_score: float, details_dict: Dict).
        """
        if not propositions:
            return 1.0, {"preserved": [], "missing": [], "scores": {}}

        if not self.semantic_model:
            # Fallback: simple keyword matching
            preserved = []
            missing = []
            for prop in propositions:
                # Simple check: if proposition keywords appear in generated text
                prop_words = set(prop.lower().split())
                gen_words = set(generated_text.lower().split())
                if prop_words.intersection(gen_words):
                    preserved.append(prop)
                else:
                    missing.append(prop)
            recall = len(preserved) / len(propositions) if propositions else 0.0
            return recall, {"preserved": preserved, "missing": missing, "scores": {}}

        # Split generated paragraph into sentences
        try:
            from nltk.tokenize import sent_tokenize
            generated_sentences = sent_tokenize(generated_text)
        except Exception:
            # Fallback: simple sentence splitting
            generated_sentences = re.split(r'[.!?]+\s+', generated_text)
            generated_sentences = [s.strip() for s in generated_sentences if s.strip()]

        if not generated_sentences:
            return 0.0, {"preserved": [], "missing": propositions, "scores": {}}

        # For each proposition, find max similarity across all sentences
        preserved = []
        missing = []
        scores = {}
        threshold = similarity_threshold  # Use parameter instead of hardcoded value
        # Default 0.45 for sentence mode, 0.30 for paragraph mode to account for stylized text
        # where short propositions are embedded in long, complex sentences (vector signal dilution)
        # Hybrid keyword matching (0.25-0.30 range) provides additional safety net

        for prop in propositions:
            max_similarity = 0.0
            best_sentence = None

            # Calculate similarity against each generated sentence
            prop_embedding = self.semantic_model.encode(prop, convert_to_tensor=True)
            for sent in generated_sentences:
                sent_embedding = self.semantic_model.encode(sent, convert_to_tensor=True)
                if util:
                    similarity = util.cos_sim(prop_embedding, sent_embedding).item()
                else:
                    import torch
                    similarity = torch.nn.functional.cosine_similarity(
                        prop_embedding, sent_embedding, dim=0
                    ).item()
                if similarity > max_similarity:
                    max_similarity = similarity
                    best_sentence = sent

            scores[prop] = max_similarity

            # Hybrid matching: if similarity is low but keywords match, still pass
            is_preserved = False
            if max_similarity > threshold:
                # Above threshold: definitely preserved
                is_preserved = True
            elif max_similarity > 0.25 and best_sentence:
                # Medium similarity (0.25-0.30): check keyword overlap (hybrid matching)
                prop_keywords = _get_content_keywords(prop)
                sent_keywords = _get_content_keywords(best_sentence)

                if prop_keywords:
                    # Calculate keyword overlap ratio
                    overlap = len(prop_keywords.intersection(sent_keywords))
                    keyword_overlap_ratio = overlap / len(prop_keywords) if prop_keywords else 0.0

                    # If 75%+ of keywords match, consider it preserved (saved by hybrid matching)
                    if keyword_overlap_ratio >= 0.75:
                        is_preserved = True
                        scores[f"{prop}_hybrid_match"] = keyword_overlap_ratio  # Store for debugging

            # ENTAILMENT CHECK: For borderline cases (0.20-0.30 similarity) that failed hybrid matching
            # This catches valid stylistic expansions that have low vector similarity but don't contradict
            if not is_preserved and max_similarity >= 0.20 and max_similarity < threshold and best_sentence:
                # Borderline case: similarity is meaningful (>0.20) but below threshold
                # Check if generated sentence entails (doesn't contradict) the proposition
                entails, confidence = self._check_entailment(prop, best_sentence, verbose=verbose)

                if entails and confidence >= 0.6:
                    # Generated sentence preserves meaning (stylistic expansion allowed)
                    is_preserved = True
                    scores[f"{prop}_entailment"] = confidence  # Store for debugging
                    scores[f"{prop}_entailment_match"] = best_sentence[:100]  # Store best match

            if is_preserved:
                preserved.append(prop)
            else:
                missing.append(prop)
                # Store best match info for debugging
                if best_sentence:
                    scores[f"{prop}_best_match"] = best_sentence[:100]  # First 100 chars

        recall = len(preserved) / len(propositions) if propositions else 0.0
        return recall, {"preserved": preserved, "missing": missing, "scores": scores}

    def _check_entailment(self, proposition: str, generated_sentence: str, verbose: bool = False) -> Tuple[bool, float]:
        """Check if generated sentence entails (preserves meaning of) proposition using LLM.

        This is used for borderline cases where similarity is low but we want to verify
        that the generated sentence doesn't contradict the proposition (stylistic expansion).

        Args:
            proposition: Original atomic proposition.
            generated_sentence: Generated sentence to check.
            verbose: Whether to print debug information.

        Returns:
            Tuple of (entails: bool, confidence: float).
            Returns (True, 0.5) if LLM unavailable (neutral, assume preserved).
        """
        if not self.llm_provider:
            # Fallback: assume preserved if LLM unavailable
            return True, 0.5

        if not proposition or not generated_sentence:
            return False, 0.0

        system_prompt = _load_prompt_template("semantic_critic_entailment.md")
        user_prompt = f"Proposition: {proposition}\n\nGenerated Sentence: {generated_sentence}\n\nDoes the Generated Sentence entail (preserve the meaning of) the Proposition?"

        try:
            response = self.llm_provider.call(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                model_type="critic",
                require_json=True,
                temperature=0.2,  # Low temperature for consistent evaluation
                max_tokens=150,
                timeout=20  # Shorter timeout for faster checks
            )

            # Parse JSON response
            try:
                result = json.loads(response)
            except json.JSONDecodeError:
                # Try to extract JSON from response
                json_match = re.search(r'\{[^}]+\}', response, re.DOTALL)
                if json_match:
                    result = json.loads(json_match.group())
                else:
                    # Fallback: assume preserved if parsing fails
                    if verbose:
                        print(f"      ⚠ Entailment check: Could not parse LLM response, assuming preserved")
                    return True, 0.5

            entails = result.get('entails', True)
            confidence = float(result.get('confidence', 0.5))
            confidence = max(0.0, min(1.0, confidence))  # Clamp to [0, 1]

            if verbose:
                reason = result.get('reason', '')
                print(f"      Entailment check: {entails} (confidence={confidence:.2f}, reason={reason[:50]}...)")

            return entails, confidence

        except Exception as e:
            # On any error, assume preserved (don't block evaluation)
            if verbose:
                print(f"      ⚠ Entailment check failed: {str(e)}, assuming preserved")
            return True, 0.5

    def _get_semantic_hash(self, template: str, propositions: List[str]) -> str:
        """Generate semantic hash for caching.

        Normalizes inputs to ensure cache hits for semantically identical inputs:
        - Sorts propositions alphabetically (order-independent)
        - Normalizes whitespace in template
        - Returns hash of normalized inputs

        Args:
            template: Template string to normalize
            propositions: List of proposition strings to normalize

        Returns:
            MD5 hash string of normalized inputs
        """
        # Sort propositions (order-independent) and normalize
        sorted_props = sorted([p.strip().lower() for p in propositions if p and p.strip()])

        # Normalize template whitespace
        normalized_template = ' '.join(template.split())

        # Create hash key
        key_string = normalized_template + '|' + '|'.join(sorted_props)
        return hashlib.md5(key_string.encode('utf-8')).hexdigest()

    def _extract_fixed_anchors(self, template: str) -> List[str]:
        """
        Extract fixed anchors from a template by keeping everything outside placeholders.

        Treats everything outside [Placeholder] brackets as an anchor, unless it's
        pure punctuation/whitespace. This preserves the author's rhetorical skeleton
        including words like "fact", "truth", "practice" even if they're nouns.

        Args:
            template: Template string with placeholders like [NP], [VP], [ADJ], [ADV]

        Returns:
            List of anchor strings (may include multi-word phrases)
        """
        if not template:
            return []

        # Regex pattern to match placeholders
        placeholder_pattern = r'\[(?:NP|VP|ADJ|ADV)\]'

        # Split template by placeholders
        segments = re.split(placeholder_pattern, template)

        anchors = []
        for segment in segments:
            # Clean whitespace but preserve structure
            segment = segment.strip()

            # Skip empty segments (pure whitespace between placeholders)
            if not segment:
                continue

            # Keep the segment as an anchor
            # This includes words like "fact", "truth", "practice" - they're part of the skeleton
            anchors.append(segment)

        return anchors

    def evaluate_sentence_fit(
        self,
        draft: str,
        assigned_propositions: List[str],
        template: str,
        verbose: bool = False
    ) -> Dict[str, any]:
        """
        Evaluate if a draft sentence fits both meaning and structure requirements.

        This is a structure-aware evaluation that checks:
        1. Anchor adherence: Does draft contain the fixed anchors from template? (strict)
        2. Semantic presence: Are the meaning atoms present? (flexible phrasing allowed)

        Args:
            draft: Generated sentence to evaluate
            assigned_propositions: List of propositions that must be expressed
            template: Template structure that must be matched
            verbose: Enable debug logging

        Returns:
            Dict with:
            - anchor_score: float (0.0-1.0) - How well fixed anchors are preserved
            - semantic_score: float (0.0-1.0) - How well meaning atoms are present
            - meaning_score: float (alias for semantic_score, backward compatibility)
            - structure_score: float (alias for anchor_score, backward compatibility)
            - pass: bool - True if both scores >= 0.9
            - anchor_feedback: str - Specific missing anchors
            - semantic_feedback: str - Missing meaning atoms (if any)
            - meaning_feedback: str (alias for semantic_feedback, backward compatibility)
            - structure_feedback: str (alias for anchor_feedback, backward compatibility)
            - overall_feedback: str - Combined feedback for repair
        """
        if not self.llm_provider:
            # Fallback: assume pass if LLM unavailable
            return {
                "anchor_score": 0.9,
                "semantic_score": 0.9,
                "meaning_score": 0.9,
                "structure_score": 0.9,
                "pass": True,
                "anchor_feedback": "",
                "semantic_feedback": "",
                "meaning_feedback": "",
                "structure_feedback": "",
                "overall_feedback": "LLM unavailable, assuming pass"
            }

        if not draft or not assigned_propositions or not template:
            return {
                "anchor_score": 0.0,
                "semantic_score": 0.0,
                "meaning_score": 0.0,
                "structure_score": 0.0,
                "pass": False,
                "anchor_feedback": "Missing required inputs",
                "semantic_feedback": "Missing required inputs",
                "meaning_feedback": "Missing required inputs",
                "structure_feedback": "Missing required inputs",
                "overall_feedback": "Missing required inputs"
            }

        # Generate semantic hash for caching
        semantic_hash = self._get_semantic_hash(template, assigned_propositions)

        # Check cache using semantic hash
        cache_key = (draft, semantic_hash)
        if hasattr(self, '_evaluation_cache') and cache_key in self._evaluation_cache:
            if verbose:
                print(f"      Cache hit for evaluation (hash: {semantic_hash[:8]}...)")
            return self._evaluation_cache[cache_key]

        # Extract fixed anchors from template
        anchors = self._extract_fixed_anchors(template)
        anchors_text = ", ".join([f"'{anchor}'" for anchor in anchors]) if anchors else "None"

        # Build propositions text
        props_text = "\n".join([f"- {prop}" for prop in assigned_propositions])

        system_prompt = """You are a Syntax Validator. Your job is to ensure the Draft creates a valid sentence that *strictly follows* the Template's rhetorical structure (anchors) while expressing the *Meaning*."""

        user_prompt = f"""Analyze the Draft against the Template.

**Template:** {template}
**Fixed Anchors (MUST be present exactly):** {anchors_text}
**Draft:** {draft}
**Required Meaning Atoms:**
{props_text}

**Check 1: Structural Anchors (The Author's Voice)**
- Does the Draft contain each fixed anchor from the template?
- Check for: {anchors_text}
- **IMPORTANT:** Allow small grammatical adjustments to fixed anchors to maintain subject-verb agreement. For example, 'is' ↔ 'are', 'has' ↔ 'have', 'was' ↔ 'were' are acceptable variations if they improve grammatical correctness.
- Score: 0.0-1.0 (1.0 = all anchors present in correct positions/order, with grammatical adjustments allowed)

**Check 2: Core Meaning (The Idea)**
- Does the draft convey the *core meaning* of the assigned propositions?
- **CRITICAL:** Do NOT check for specific words. Check if the *idea* is present.
- Allow flexible phrasing (synonyms, metaphors, rephrasing, word order changes)
- Score: 0.0-1.0 (1.0 = core meaning fully conveyed)

**Key Instruction:**
- DO enforce exact wording for Fixed Anchors (prepositions, conjunctions, "there was", "the fact is", etc.), BUT allow grammatical inflection (is/are, has/have, was/were) for subject-verb agreement.
- DO NOT enforce exact wording for the [Variables]. Allow rephrasing of meaning atoms.
- DO NOT check for specific nouns or individual words. Focus on whether the *concept* is present.

**Output Format:** JSON
{{
  "anchor_score": 0.85,
  "semantic_score": 0.95,
  "anchor_feedback": "Missing anchor 'In' at start.",
  "semantic_feedback": "Core meaning is present.",
  "overall_feedback": "Add 'In' at start while keeping current meaning.",
  "pass": false
}}
"""

        try:
            response = self.llm_provider.call(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                model_type="critic",
                require_json=True,
                temperature=0.2,  # Low temperature for consistent evaluation
                max_tokens=300,
                timeout=20
            )

            # Parse JSON response
            response = response.strip()
            json_match = re.search(r'\{.*?\}', response, re.DOTALL)
            if json_match:
                try:
                    result = json.loads(json_match.group(0))
                except json.JSONDecodeError:
                    # Try parsing entire response
                    result = json.loads(response)
            else:
                result = json.loads(response)

            # Extract scores and feedback
            anchor_score = float(result.get("anchor_score", 0.0))
            semantic_score = float(result.get("semantic_score", 0.0))
            anchor_feedback = result.get("anchor_feedback", "")
            semantic_feedback = result.get("semantic_feedback", "")
            overall_feedback = result.get("overall_feedback", "")

            # Clamp scores to [0, 1]
            anchor_score = max(0.0, min(1.0, anchor_score))
            semantic_score = max(0.0, min(1.0, semantic_score))

            # Determine pass (both scores >= 0.9)
            passed = (anchor_score >= 0.9) and (semantic_score >= 0.9)

            # Build overall feedback if not provided
            if not overall_feedback:
                feedback_parts = []
                if anchor_score < 0.9:
                    feedback_parts.append(anchor_feedback)
                if semantic_score < 0.9:
                    feedback_parts.append(semantic_feedback)
                overall_feedback = " ".join(feedback_parts) if feedback_parts else "No issues found"

            if verbose:
                print(f"      Evaluation: anchor={anchor_score:.2f}, semantic={semantic_score:.2f}, pass={passed}")

            # Build result dict
            result = {
                "anchor_score": anchor_score,
                "semantic_score": semantic_score,
                "meaning_score": semantic_score,  # Backward compatibility
                "structure_score": anchor_score,   # Backward compatibility
                "pass": passed,
                "anchor_feedback": anchor_feedback,
                "semantic_feedback": semantic_feedback,
                "meaning_feedback": semantic_feedback,  # Backward compatibility
                "structure_feedback": anchor_feedback,  # Backward compatibility
                "overall_feedback": overall_feedback
            }

            # Cache result
            if not hasattr(self, '_evaluation_cache'):
                self._evaluation_cache = {}
            # Limit cache size to 512 entries (LRU eviction)
            if len(self._evaluation_cache) >= 512:
                # Remove oldest entry (simple FIFO)
                oldest_key = next(iter(self._evaluation_cache))
                del self._evaluation_cache[oldest_key]
            self._evaluation_cache[cache_key] = result

            return result

        except Exception as e:
            if verbose:
                print(f"      ⚠ Evaluation failed: {e}, assuming pass")
            # Fallback: assume pass on error
            return {
                "anchor_score": 0.9,
                "semantic_score": 0.9,
                "meaning_score": 0.9,
                "structure_score": 0.9,
                "pass": True,
                "anchor_feedback": "",
                "semantic_feedback": "",
                "meaning_feedback": "",
                "structure_feedback": "",
                "overall_feedback": f"Evaluation error: {str(e)}, assuming pass"
            }

    def _calculate_repetition_penalty(
        self,
        candidate: str,
        previous_sentence: Optional[str],
        words_to_check: int = 5
    ) -> Tuple[float, str]:
        """Calculate penalty for repetitive sentence openers.

        Args:
            candidate: Current candidate sentence.
            previous_sentence: Previous sentence (or None).
            words_to_check: Number of words to compare.

        Returns:
            Tuple of (penalty_score: float, feedback: str).
            penalty_score is 0.0 (no penalty) to 0.5 (max penalty).
        """
        if not previous_sentence or not candidate:
            return 0.0, ""

        # Extract first N words from both sentences
        candidate_words = candidate.split()[:words_to_check]
        previous_words = previous_sentence.split()[:words_to_check]

        if len(candidate_words) < 3 or len(previous_words) < 3:
            return 0.0, ""

        # Calculate similarity for word sequence
        # Simple approach: count matching words in same position
        matches = sum(1 for c, p in zip(candidate_words, previous_words) if c.lower() == p.lower())
        similarity_ratio = matches / min(len(candidate_words), len(previous_words))

        if similarity_ratio >= 0.8:  # 80% of words match
            penalty = 0.5
            feedback = f"Repetitive Sentence Opener. You started with '{' '.join(candidate_words[:3])}' which is too similar to the previous sentence's opener '{' '.join(previous_words[:3])}'. Vary the phrasing."
            return penalty, feedback

        return 0.0, ""

    def _check_vocabulary_repetition(self, text: str) -> Tuple[bool, List[str]]:
        """Check for repetitive vocabulary using local NLP.

        Uses spaCy to extract content words and count frequencies.
        Only detects repetition if a word appears > 2 times.

        Args:
            text: Text to check for repetition

        Returns:
            Tuple of (has_repetition: bool, repeated_words: List[str])
        """
        try:
            from src.utils.nlp_manager import NLPManager
            from collections import Counter

            nlp = NLPManager.get_nlp()
            if nlp is None:
                return False, []

            doc = nlp(text)

            # Extract content words (nouns, verbs, adjectives)
            content_words = [token.lemma_.lower()
                           for token in doc
                           if token.pos_ in ['NOUN', 'VERB', 'ADJ']
                           and not token.is_stop
                           and len(token.lemma_) > 2]  # Filter very short words

            # Count frequencies
            word_counts = Counter(content_words)

            # Find repeated words (appears > 2 times)
            repeated = [word for word, count in word_counts.items() if count > 2]

            return len(repeated) > 0, repeated
        except Exception:
            # Fallback: no repetition detected
            return False, []

    def _evaluate_batch(
        self,
        candidate_populations: List[List[str]],
        templates: List[str],
        prop_map: List[List[str]],
        narrative_roles: List[str] = None,
        best_sentences: List[Optional[str]] = None,
        verbose: bool = False
    ) -> List[List[Dict]]:
        """Evaluate all candidates in a single batch LLM call.

        Args:
            candidate_populations: List[List[str]] where candidate_populations[i] is
                the population for slot i.
            templates: List[str] of templates (one per slot).
            prop_map: List[List[str]] where prop_map[i] contains propositions for slot i.
            verbose: Whether to print debug information.

        Returns:
            List[List[Dict]] where result[i][j] contains evaluation for candidate j of slot i.

        Raises:
            Exception: If batch evaluation fails (caller should fall back to individual).
        """
        import json
        import re

        # Pre-process all slots: extract anchors and build slot blocks
        slot_blocks = []
        total_candidates = 0
        slots_with_candidates = []
        slot_contexts = {}  # Store context for each slot (for repetition checking)

        for slot_idx in range(len(templates)):
            candidates = candidate_populations[slot_idx]
            template = templates[slot_idx]
            propositions = prop_map[slot_idx]

            # Skip empty populations (locked slots)
            if not candidates:
                continue

            # Get narrative role for this slot
            narrative_role = narrative_roles[slot_idx] if narrative_roles and slot_idx < len(narrative_roles) else "BODY"

            # Build previous context with soft fallback
            prev_sentence_context = "None (Start of Paragraph)"
            is_pending_draft = False
            prev_sentence_for_repetition = None  # For repetition penalty check
            if slot_idx > 0:
                if best_sentences and best_sentences[slot_idx - 1]:
                    prev_sentence_context = best_sentences[slot_idx - 1]
                    prev_sentence_for_repetition = best_sentences[slot_idx - 1]
                elif candidate_populations[slot_idx - 1]:
                    # FALLBACK: Use the first candidate of the previous slot as a proxy for context
                    # This prevents "Deadlock" where Slot 1 fails because Slot 0 isn't finished.
                    prev_sentence_context = candidate_populations[slot_idx - 1][0]
                    prev_sentence_for_repetition = candidate_populations[slot_idx - 1][0]
                    is_pending_draft = True

            # Store context for repetition checking
            slot_contexts[slot_idx] = {
                "prev_sentence": prev_sentence_for_repetition
            }

            # Extract fixed anchors
            anchors = self._extract_fixed_anchors(template)
            anchors_text = ", ".join([f'"{anchor}"' for anchor in anchors]) if anchors else "None"

            # Format propositions
            props_text = "\n".join([f"- {prop}" for prop in propositions])

            # Format candidates with indices
            candidates_text = "\n".join([f'{i}. "{candidate}"' for i, candidate in enumerate(candidates)])

            # Build context label with pending draft marker if applicable
            prev_context_label = prev_sentence_context
            if is_pending_draft:
                prev_context_label = f"[Pending Draft]: {prev_sentence_context}"

            # Build slot block
            slot_block = f"""---
### SLOT {slot_idx}
**Template:** "{template}"
**Narrative Role:** {narrative_role}
**Fixed Anchors (Must match exactly):** [{anchors_text}]
**Required Meaning Atoms:**
{props_text}
**Previous Sentence Context:** {prev_context_label}

**Candidates:**
{candidates_text}
---"""
            slot_blocks.append(slot_block)
            slots_with_candidates.append(slot_idx)
            total_candidates += len(candidates)

        if not slot_blocks:
            # No candidates to evaluate
            return [[] for _ in templates]

        # Check token limit (rough estimate: ~4 tokens per character)
        exam_sheet = "\n".join(slot_blocks)
        estimated_tokens = len(exam_sheet) // 4
        if estimated_tokens > 100000:  # Safety limit
            raise ValueError(f"Batch prompt too large ({estimated_tokens} estimated tokens), exceeds safety limit of 100000 tokens")

        # Build system prompt
        system_prompt = """You are a Batch Syntax & Semantic Validator. Evaluate multiple candidates across multiple structural slots simultaneously. Grade each candidate based on Structural Anchors (exact match required), Semantic Meaning (flexible phrasing allowed), and Narrative Flow (rhetorical function and coherence).

**Evaluation Criteria:**
1. **Anchor Adherence (Flexible POS) (0.0-1.0):**
   - **Primary Goal:** Check if the **Fixed Anchors** (words like 'It is', 'the', 'of', 'therefore', 'there was', 'the fact is') are present exactly.
   - **Secondary Goal:** Check the variable slots (`[NP]`, `[VP]`, `[ADJ]`, `[ADV]`).
   - **Flexibility Rule:** If the candidate uses a Noun Phrase where the template asked for an Adjective (or vice versa), **ACCEPT IT** as long as:
     * The sentence is grammatically correct
     * The rhythm and structure are preserved
     * The meaning is conveyed
   - **DO NOT fail a candidate solely for Part-of-Speech tag mismatches.**
   - **Grammar > Template Tags:** If changing a tag improved grammar, reward it, don't penalize it.
   - **IMPORTANT:** Allow small grammatical adjustments to fixed anchors to maintain subject-verb agreement. For example, 'is' ↔ 'are', 'has' ↔ 'have', 'was' ↔ 'were' are acceptable variations if they improve grammatical correctness.
2. **Semantic Presence (Register Aware) (0.0-1.0):** Does the candidate convey the core meaning of the required propositions?
   - **Register Translation:** The candidate should convey the *meaning* of the required propositions, not necessarily the exact words.
   - **Accept High-Register Synonyms:**
     * Input 'Hunger' → Candidate 'Material Deprivation' ✅
     * Input 'Ruins' → Candidate 'Collapsed Infrastructure' ✅
     * Input 'brought void' → Candidate 'precipitated a material crisis' ✅
   - **Accept Metaphorical Equivalents:**
     * Input 'scavenging' → Candidate 'systematic extraction of resources' ✅
   - **Fail Only If:** The *core concept* is missing entirely (not just rephrased).
   - **DO NOT penalize** for using elevated register or academic synonyms.
   - DO NOT check for specific words - check if the *idea* is present. Allow flexible phrasing, synonyms, metaphors, rephrasing.
3. **Narrative Flow (0.0-1.0):**
   - Role Check: Does this candidate fulfill the assigned Narrative Role: '{narrative_role}'?
   - Coherence Check: Does this candidate follow logically from the previous sentence?
   - (If Slot 0): Does it establish a strong opening?
   - (If Slot > 0): Does it connect to the context of the previous slot?
   - **IMPORTANT:** If 'Previous Sentence Context' is marked [Pending Draft], evaluate if the candidate *could* logically follow such a sentence, but be lenient on exact transitions. Grade based on internal coherence and role fulfillment rather than strict flow matching.
4. **Logical Entailment (0.0-1.0):** Does the sentence make internal sense?
   - **Connector Validation:** If the sentence uses strong logical connectors ('Therefore', 'Consequently', 'Paradoxically', 'Because', 'Thus', 'remains a'), analyze the content.
   - **Question:** Does the clause following the connector logically follow from the clause before it?
   - **Verdict:**
     * If the logic is non-existent (e.g., 'It is blue, therefore the economy collapsed'), mark as **FAIL** (logic_score = 0.0) even if the structure is perfect.
     * If the connector is justified by the content, mark as **PASS** (logic_score = 1.0).
   - **Examples:**
     * Bad: "...remains a paradox" (no paradox established) → logic_score = 0.0
     * Good: "The contradiction between X and Y remains a paradox" → logic_score = 1.0
     * Bad: "The sky is blue, therefore capitalism fails" → logic_score = 0.0 (non-sequitur)
     * Good: "Production declined, therefore the economy collapsed" → logic_score = 1.0 (causal link)
5. **Repetition Check:** If the candidate starts with the same words as the previous sentence, apply a -0.5 penalty to narrative_score. Vary sentence openers to maintain reader engagement.
6. **Noun Piling Check (0.0-1.0):** Reject sentences where nouns are jammed together without prepositions or connectors.
   - **Example failure:** "observation horror death" → noun_piling_score = 0.0 (nouns piled without connectors)
   - **Example pass:** "observation of horror and death" → noun_piling_score = 1.0 (proper grammatical connectors)
   - **Rule:** If a sentence contains consecutive nouns without prepositions (like "of", "and", "in", "from") or conjunctions, mark as FAIL (noun_piling_score = 0.0).
   - **Exception:** Allow noun compounds that are standard English (e.g., "coffee shop", "book store") but reject awkward noun chains.

**Key Instructions:**
- DO enforce exact wording for Fixed Anchors, BUT allow grammatical inflection (is/are, has/have, was/were) for subject-verb agreement
- DO NOT enforce exact wording for meaning atoms - allow rephrasing
- DO NOT check for specific nouns or individual words - focus on whether the *concept* is present
- A candidate passes if anchor_score >= 1.0 AND semantic_score >= 0.95 AND narrative_score >= 0.8 AND logic_score >= 0.8 AND noun_piling_score >= 0.8"""

        # Check for vocabulary repetition across all candidates (local check)
        all_candidates_text = " ".join([c for candidates in candidate_populations for c in candidates])
        has_vocab_repetition, repeated_words = self._check_vocabulary_repetition(all_candidates_text)

        # Build repetition instruction if needed
        repetition_instruction = ""
        if has_vocab_repetition:
            repeated_words_str = ", ".join(repeated_words[:10])  # Limit to first 10
            repetition_instruction = f"\n\n**IMPORTANT - Vocabulary Repetition Detected:**\nThe following words appear too frequently (>2 times): {repeated_words_str}\nApply a penalty to narrative_score if candidates overuse these words. Prefer variety in vocabulary."

        # Build user prompt (exam sheet)
        user_prompt = f"""Evaluate the following candidates across {len(slots_with_candidates)} structural slots.

{exam_sheet}
{repetition_instruction}

**Task:** For EACH candidate, evaluate:
1. **Anchor Adherence (0.0-1.0):** Are the fixed anchors present exactly?
2. **Semantic Presence (0.0-1.0):** Is the core meaning conveyed (flexible phrasing allowed)?
3. **Narrative Flow (0.0-1.0):** Does the candidate fulfill its Narrative Role and connect logically to the previous sentence?
4. **Logical Entailment (0.0-1.0):** Does the sentence make internal sense? Check if connectors like 'therefore', 'remains a', 'because', 'consequently' are justified by the preceding text.
5. **Noun Piling Check (0.0-1.0):** Are nouns properly connected with prepositions or conjunctions? Reject sentences where nouns are jammed together without connectors.

**Output Format:** Return a JSON object with keys "slot_0", "slot_1", etc., where each value is an array of evaluation objects (one per candidate, in order).

Each evaluation object must contain:
- "id": integer (candidate index, 0-based)
- "anchor_score": float (0.0-1.0)
- "semantic_score": float (0.0-1.0)
- "narrative_score": float (0.0-1.0)
- "logic_score": float (0.0-1.0)
- "noun_piling_score": float (0.0-1.0)
- "grammar_override_detected": boolean (true if candidate changed POS tags but grammar is valid)
- "pass": boolean (true if anchor_score >= 1.0 AND semantic_score >= 0.95 AND narrative_score >= 0.8 AND logic_score >= 0.8 AND noun_piling_score >= 0.8)
- "feedback": string (combined feedback explaining the scores)

**Example JSON structure:**
{{
  "slot_0": [
    {{
      "id": 0,
      "anchor_score": 1.0,
      "semantic_score": 0.95,
      "narrative_score": 0.8,
      "logic_score": 0.9,
      "noun_piling_score": 1.0,
      "grammar_override_detected": true,
      "pass": true,
      "feedback": "Candidate changed [ADJ] to [NP] for grammar, but structure preserved."
    }},
    {{
      "id": 1,
      "anchor_score": 1.0,
      "semantic_score": 0.5,
      "narrative_score": 0.8,
      "logic_score": 0.9,
      "noun_piling_score": 1.0,
      "grammar_override_detected": false,
      "pass": false,
      "feedback": "Missing semantic atom 'scavenged'."
    }},
    {{
      "id": 2,
      "anchor_score": 0.0,
      "semantic_score": 1.0,
      "narrative_score": 0.9,
      "logic_score": 0.8,
      "noun_piling_score": 1.0,
      "grammar_override_detected": false,
      "pass": false,
      "feedback": "Missing anchor 'In' at start."
    }}
  ],
  "slot_1": [...]
}}

Return ONLY valid JSON, no additional text."""

        if verbose:
            print(f"  Batch evaluating {total_candidates} total candidates across {len(slots_with_candidates)} slots")
            print(f"    Slots with candidates: {slots_with_candidates}")
            for slot_idx in slots_with_candidates:
                candidates = candidate_populations[slot_idx]
                template = templates[slot_idx]
                print(f"      Slot {slot_idx}: {len(candidates)} candidates, template=\"{template[:60]}{'...' if len(template) > 60 else ''}\"")

        # Call LLM
        if not self.llm_provider:
            raise ValueError("LLM provider not available for batch evaluation")

        # Calculate max_tokens (rough estimate: 300 per candidate with cap)
        max_tokens = min(300 * total_candidates, 16000)  # Cap at 16k for safety

        # Retry logic for batch evaluation with JSON parsing
        import time
        retry_delay = self.retry_delay
        batch_results = None

        for attempt in range(self.max_retries):
            try:
                if attempt > 0:
                    if verbose:
                        print(f"  🔄 Retry attempt {attempt + 1}/{self.max_retries} for batch evaluation (JSON parse failed)")
                    time.sleep(retry_delay)
                    retry_delay *= 2  # Exponential backoff

                response = self.llm_provider.call(
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    model_type="critic",
                    require_json=True,
                    temperature=0.1,  # Strict grading
                    max_tokens=max_tokens,
                    timeout=getattr(self, 'batch_timeout', 180)  # Configurable timeout for batch operations
                )

                # Parse JSON response with repair if needed
                response = response.strip()
                try:
                    batch_results = self._safe_json_parse(response, verbose=verbose)
                    # Successfully parsed - break out of retry loop
                    break
                except json.JSONDecodeError as e:
                    if verbose:
                        print(f"  ❌ Failed to parse JSON even after repair (attempt {attempt + 1}/{self.max_retries}): {e}")
                    if attempt < self.max_retries - 1:
                        # Will retry
                        continue
                    else:
                        # Last attempt failed - raise error
                        raise ValueError(f"Failed to parse batch evaluation JSON after {self.max_retries} attempts: {e}")
            except Exception as e:
                if attempt < self.max_retries - 1:
                    if verbose:
                        print(f"  ⚠ Batch evaluation error (attempt {attempt + 1}/{self.max_retries}): {e}, retrying...")
                    continue
                else:
                    # Last attempt - re-raise
                    raise

        if batch_results is None:
            raise ValueError(f"Failed to get batch evaluation results after {self.max_retries} attempts")

        if not isinstance(batch_results, dict):
            raise ValueError(f"Expected JSON object, got {type(batch_results)}")

        # Map batch results back to result structure
        results = [[] for _ in templates]

        for slot_idx in slots_with_candidates:
            slot_key = f"slot_{slot_idx}"
            if slot_key not in batch_results:
                if verbose:
                    print(f"    Warning: Slot {slot_idx} missing from batch response, using failure results")
                # Fill with failure results
                candidates = candidate_populations[slot_idx]
                for candidate_idx in range(len(candidates)):
                    results[slot_idx].append({
                        "pass": False,
                        "anchor_score": 0.0,
                        "semantic_score": 0.0,
                        "narrative_score": 0.0,
                        "logic_score": 0.0,
                        "noun_piling_score": 0.0,
                        "grammar_override_detected": False,
                        "combined_score": 0.0,
                        "feedback": "Missing from batch response",
                        "slot_index": slot_idx,
                        "candidate_index": candidate_idx
                    })
                continue

            slot_evaluations = batch_results[slot_key]
            if not isinstance(slot_evaluations, list):
                if verbose:
                    print(f"    Warning: Slot {slot_idx} has invalid format, using failure results")
                candidates = candidate_populations[slot_idx]
                for candidate_idx in range(len(candidates)):
                    results[slot_idx].append({
                        "pass": False,
                        "anchor_score": 0.0,
                        "semantic_score": 0.0,
                        "narrative_score": 0.0,
                        "logic_score": 0.0,
                        "noun_piling_score": 0.0,
                        "grammar_override_detected": False,
                        "combined_score": 0.0,
                        "feedback": "Invalid format in batch response",
                        "slot_index": slot_idx,
                        "candidate_index": candidate_idx
                    })
                continue

            candidates = candidate_populations[slot_idx]
            # Create a map of id -> evaluation for easy lookup
            eval_map = {}
            for eval_obj in slot_evaluations:
                if isinstance(eval_obj, dict) and "id" in eval_obj:
                    eval_map[eval_obj["id"]] = eval_obj

            # Map evaluations to candidates
            for candidate_idx in range(len(candidates)):
                if candidate_idx in eval_map:
                    eval_obj = eval_map[candidate_idx]
                    anchor_score = float(eval_obj.get("anchor_score", 0.0))
                    semantic_score = float(eval_obj.get("semantic_score", 0.0))
                    narrative_score = float(eval_obj.get("narrative_score", 0.0))
                    logic_score = float(eval_obj.get("logic_score", 1.0))  # Default to 1.0 if not provided (backward compatibility)
                    noun_piling_score = float(eval_obj.get("noun_piling_score", 1.0))  # Default to 1.0 if not provided (backward compatibility)
                    grammar_override_detected = bool(eval_obj.get("grammar_override_detected", False))

                    # Initialize feedback from evaluation object
                    feedback = str(eval_obj.get("feedback", ""))

                    # Apply repetition penalty
                    candidate = candidates[candidate_idx]
                    if slot_idx > 0:
                        prev_sentence = slot_contexts.get(slot_idx, {}).get("prev_sentence")
                        if prev_sentence:
                            penalty, feedback_addition = self._calculate_repetition_penalty(
                                candidate, prev_sentence
                            )
                            if penalty > 0:
                                narrative_score = max(0.0, narrative_score - penalty)
                                if feedback_addition:
                                    if feedback:
                                        feedback += f" {feedback_addition}"
                                    else:
                                        feedback = feedback_addition

                    combined_score = anchor_score + semantic_score
                    pass_flag = bool(eval_obj.get("pass", False))
                    # Re-check pass flag after applying repetition penalty and logic check
                    if narrative_score < 0.8:
                        pass_flag = False
                    if logic_score < 0.8:
                        pass_flag = False
                    if noun_piling_score < 0.8:
                        pass_flag = False

                    results[slot_idx].append({
                        "pass": pass_flag,
                        "anchor_score": anchor_score,
                        "semantic_score": semantic_score,
                        "narrative_score": narrative_score,
                        "logic_score": logic_score,
                        "noun_piling_score": noun_piling_score,
                        "grammar_override_detected": grammar_override_detected,
                        "combined_score": combined_score,
                        "feedback": feedback,
                        "slot_index": slot_idx,
                        "candidate_index": candidate_idx
                    })
                else:
                    # Missing evaluation for this candidate
                    if verbose:
                        print(f"    Warning: Slot {slot_idx}, Candidate {candidate_idx} missing from batch response")
                    results[slot_idx].append({
                        "pass": False,
                        "anchor_score": 0.0,
                        "semantic_score": 0.0,
                        "narrative_score": 0.0,
                        "logic_score": 0.0,
                        "noun_piling_score": 0.0,
                        "combined_score": 0.0,
                        "feedback": "Missing from batch response",
                        "slot_index": slot_idx,
                        "candidate_index": candidate_idx
                    })

        if verbose:
            total_evaluated = sum(len(r) for r in results)
            passing_candidates = sum(sum(1 for res in r if res.get("pass", False)) for r in results)
            print(f"  ✅ Batch evaluated {total_evaluated} candidates: {passing_candidates} passing")
            # Show summary per slot
            for slot_idx in slots_with_candidates:
                slot_results = results[slot_idx]
                if slot_results:
                    passing = sum(1 for r in slot_results if r.get("pass", False))
                    avg_combined = sum(r.get("combined_score", 0.0) for r in slot_results) / len(slot_results)
                    print(f"      Slot {slot_idx}: {passing}/{len(slot_results)} passing, avg_score={avg_combined:.2f}")

        return results

    def evaluate_candidate_populations(
        self,
        candidate_populations: List[List[str]],
        templates: List[str],
        prop_map: List[List[str]],
        narrative_roles: List[str] = None,
        best_sentences: List[Optional[str]] = None,
        verbose: bool = False
    ) -> List[List[Dict]]:
        """Evaluate all candidates for all slots and return per-candidate results.

        Uses batch evaluation with retry logic. If JSON parsing fails, retries the call
        up to max_retries times with improved JSON repair.

        Args:
            candidate_populations: List[List[str]] where candidate_populations[i] is
                the population for slot i.
            templates: List[str] of templates (one per slot).
            prop_map: List[List[str]] where prop_map[i] contains propositions for slot i.
            narrative_roles: Optional list of narrative role strings (one per slot).
            best_sentences: Optional list of best sentences for flow context (one per slot).
            verbose: Whether to print debug information.

        Returns:
            List[List[Dict]] where result[i][j] contains evaluation for candidate j of slot i.
            Each dict contains:
            - pass: bool
            - anchor_score: float
            - semantic_score: float
            - narrative_score: float (NEW)
            - combined_score: float (anchor_score + semantic_score)
            - feedback: str
            - slot_index: int
            - candidate_index: int
        """
        if len(candidate_populations) != len(templates) or len(templates) != len(prop_map):
            raise ValueError(
                f"Mismatched lengths: populations={len(candidate_populations)}, "
                f"templates={len(templates)}, prop_map={len(prop_map)}"
            )

        # Batch evaluation with retry logic (no fallback to individual)
        return self._evaluate_batch(
            candidate_populations=candidate_populations,
            templates=templates,
            prop_map=prop_map,
            narrative_roles=narrative_roles,
            best_sentences=best_sentences,
            verbose=verbose
        )

    def _check_style_alignment(
        self,
        generated_text: str,
        author_style_vector: Optional[np.ndarray] = None,
        style_lexicon: Optional[List[str]] = None,
        secondary_author_vector: Optional[np.ndarray] = None,
        blend_ratio: float = 0.5
    ) -> Tuple[float, Dict]:
        """Check style alignment between generated text and author style.

        REUSE EXISTING INFRASTRUCTURE: Uses get_style_vector from style_metrics.py
        or author_style_vector from Atlas. Also calculates lexicon density.

        Args:
            generated_text: Generated paragraph text.
            author_style_vector: Optional pre-computed author style vector.
            style_lexicon: Optional list of style words to check for lexicon density.

        Returns:
            Tuple of (style_score: float, details_dict: Dict).
        """
        try:
            from src.analyzer.style_metrics import get_style_vector
        except ImportError:
            # Fallback if style_metrics not available
            return 0.5, {"similarity": 0.5, "lexicon_density": 0.0, "avg_sentence_length": 0, "staccato_penalty": 0.0}

        # Get style vector for generated text
        generated_style_vector = get_style_vector(generated_text)

        # Calculate style similarity (vector-based)
        style_similarity = 0.5  # Default if no author vector
        if author_style_vector is not None and NUMPY_AVAILABLE and np is not None:
            # Interpolate vectors if secondary author provided
            if secondary_author_vector is not None:
                # Create "ghost vector": (1 - ratio) * primary + ratio * secondary
                # ratio=0.7 means 70% primary, 30% secondary
                target_vector = (author_style_vector * (1 - blend_ratio)) + (secondary_author_vector * blend_ratio)
            else:
                target_vector = author_style_vector

            # Calculate cosine similarity against target vector
            dot_product = np.dot(generated_style_vector, target_vector)
            norm_gen = np.linalg.norm(generated_style_vector)
            norm_target = np.linalg.norm(target_vector)
            if norm_gen > 0 and norm_target > 0:
                style_similarity = dot_product / (norm_gen * norm_target)
                style_similarity = max(0.0, min(1.0, style_similarity))  # Clamp to [0, 1]

        # Calculate lexicon density (percentage of words in generated_text that appear in style_lexicon)
        lexicon_density = 0.0
        if style_lexicon and generated_text:
            # Get lemmatized tokens from generated text
            generated_tokens = _get_significant_tokens(generated_text)

            # Normalize style_lexicon to lemmatized form for matching
            # Use spaCy if available, otherwise simple lowercasing
            # OPTIMIZATION: Batch process all words at once instead of one-by-one
            lexicon_lemmas = set()
            if _spacy_nlp:
                # Batch process: join all words and process once, then extract lemmas
                lexicon_text = " ".join([word.lower().strip() for word in style_lexicon if word.strip()])
                if lexicon_text:
                    doc = _spacy_nlp(lexicon_text)
                    for token in doc:
                        if not token.is_stop and not token.is_punct:
                            lexicon_lemmas.add(token.lemma_.lower())
            else:
                # Fallback: simple lowercasing
                lexicon_lemmas = {word.lower().strip() for word in style_lexicon if word.strip()}

            # Count matches (using lemmatized tokens)
            matches = len(generated_tokens & lexicon_lemmas)
            total_significant = len(generated_tokens)

            if total_significant > 0:
                lexicon_density = matches / total_significant
            else:
                lexicon_density = 0.0

        # Check sentence length distribution (punish "staccato" output)
        try:
            from nltk.tokenize import sent_tokenize
            sentences = sent_tokenize(generated_text)
        except Exception:
            sentences = re.split(r'[.!?]+\s+', generated_text)
            sentences = [s.strip() for s in sentences if s.strip()]

        avg_sentence_length = 0.0
        staccato_penalty = 0.0
        if sentences:
            word_counts = [len(s.split()) for s in sentences]
            avg_sentence_length = sum(word_counts) / len(word_counts) if word_counts else 0.0
            # Penalize if average < 15 words (too short/staccato)
            if avg_sentence_length < 15:
                staccato_penalty = (15 - avg_sentence_length) / 15.0  # Penalty 0-1
                style_similarity = max(0.0, style_similarity - staccato_penalty * 0.3)

        # Composite score: (Vector_Sim * 0.7) + (Lexicon_Density * 0.3)
        composite_score = (style_similarity * 0.7) + (lexicon_density * 0.3)

        # Final style score combines composite score and sentence length penalty
        style_score = composite_score * (1.0 - staccato_penalty * 0.2)

        return style_score, {
            "similarity": style_similarity,
            "lexicon_density": lexicon_density,
            "avg_sentence_length": avg_sentence_length,
            "staccato_penalty": staccato_penalty
        }

    def _evaluate_paragraph_mode(
        self,
        generated_text: str,
        original_text: str,
        propositions: List[str],
        author_style_vector: Optional[np.ndarray] = None,
        style_lexicon: Optional[List[str]] = None,
        secondary_author_vector: Optional[np.ndarray] = None,
        blend_ratio: float = 0.5,
        verbose: bool = False,
        global_context: Optional[Dict] = None
    ) -> Dict[str, any]:
        """Evaluate paragraph in paragraph mode using proposition recall and style alignment.

        Args:
            generated_text: Generated paragraph text.
            original_text: Original input text.
            propositions: List of atomic propositions extracted from original.
            author_style_vector: Optional author style vector.
            style_lexicon: Optional list of style words for lexicon density calculation.

        Returns:
            Dict with evaluation results (same format as sentence mode).
        """
        # Load paragraph fusion config
        with open(self.config_path, 'r') as f:
            config = json.load(f)
        paragraph_config = config.get("paragraph_fusion", {})
        proposition_recall_threshold = paragraph_config.get("proposition_recall_threshold", 0.8)
        meaning_weight = paragraph_config.get("meaning_weight", 0.9)  # Default 0.9 for meaning-first
        style_weight = paragraph_config.get("style_alignment_weight", 0.1)  # Default 0.1 (garnish only)

        # HARD GATE 0: Context Leak Detection (run first, before any other checks)
        if global_context:
            has_leak, leaked_keywords, leak_score, leak_reason = self._check_context_leak(
                generated_text, global_context, propositions
            )
            if has_leak:
                if verbose:
                    print(f"      ⚠ Context Leak Detected: {leak_reason}")
                return {
                    "pass": False,
                    "score": 0.0,
                    "proposition_recall": 0.0,  # Don't calculate if leak detected
                    "style_alignment": 0.0,
                    "coherence_score": 0.0,
                    "topic_similarity": 0.0,
                    "feedback": f"CRITICAL: {leak_reason}. Remove all keywords from global context that don't appear in input propositions.",
                    "context_leak_detected": True,
                    "leaked_keywords": leaked_keywords
                }

        # Metric 1: Proposition Recall (use relaxed threshold 0.30 for paragraph mode)
        proposition_recall, recall_details = self._check_proposition_recall(
            generated_text,
            propositions,
            similarity_threshold=0.30,  # More lenient for paragraph mode to avoid false negatives
            verbose=verbose
        )

        # MEANING GATE: Hard floor check for proposition recall BEFORE style calculation
        # If recall is too low, meaning is lost - fail immediately
        if proposition_recall < proposition_recall_threshold:
            if verbose:
                print(f"      ⚠ Proposition Recall FAILED: {proposition_recall:.2f} < {proposition_recall_threshold:.2f} → score=0.0")
            missing = recall_details.get("missing", [])
            feedback_msg = f"CRITICAL: Proposition recall too low ({proposition_recall:.2f} < {proposition_recall_threshold}). Lost meaning."
            if missing:
                feedback_msg += f" Missing propositions: {', '.join(missing[:3])}"
            return {
                "pass": False,
                "score": 0.0,
                "proposition_recall": proposition_recall,
                "style_alignment": 0.0,  # Don't calculate if meaning fails
                "coherence_score": 0.0,
                "topic_similarity": 0.0,
                "feedback": feedback_msg,
                "recall_details": recall_details
            }

        # Get thresholds from config
        coherence_threshold = paragraph_config.get("coherence_threshold", 0.8)
        topic_similarity_threshold = paragraph_config.get("topic_similarity_threshold", 0.6)

        # MEANING FIRST: Run coherence check BEFORE style calculation
        coherence_score, coherence_reason = self._verify_coherence(generated_text, verbose=verbose)
        # Instead of comparing to original_text, compare to propositions
        # Propositions are stripped of style, so they represent pure meaning
        # This prevents false negatives when doing radical style transfer (e.g., Memoir → Theory)
        propositions_text = " ".join(propositions) if propositions else original_text
        topic_similarity = self._calculate_semantic_similarity(propositions_text, generated_text)

        # HARD GATE: If incoherent, fail immediately (no style calculation needed)
        if coherence_score < coherence_threshold:
            if verbose:
                print(f"      ⚠ Coherence FAILED: {coherence_score:.2f} < {coherence_threshold:.2f} → score=0.0")
                print(f"      Generated text (first 200 chars): {generated_text[:200]}...")
                print(f"      Coherence reason: {coherence_reason}")
            return {
                "pass": False,
                "score": 0.0,
                "proposition_recall": proposition_recall,
                "style_alignment": 0.0,  # Don't calculate if incoherent
                "coherence_score": coherence_score,
                "topic_similarity": topic_similarity,
                "feedback": f"CRITICAL: Text is incoherent ({coherence_reason}). Meaning preservation requires coherent text.",
                "coherence_reason": coherence_reason
            }

        # Only calculate style if coherence passes
        style_alignment, style_details = self._check_style_alignment(
            generated_text,
            author_style_vector,
            style_lexicon=style_lexicon,
            secondary_author_vector=secondary_author_vector,
            blend_ratio=blend_ratio
        )

        # Log coherence and topic similarity (already calculated above)
        if verbose:
            print(f"      Coherence: {coherence_score:.2f} (threshold: {coherence_threshold:.2f}), Topic similarity: {topic_similarity:.2f} (threshold: {topic_similarity_threshold:.2f})")

        # Check topic similarity threshold (coherence already checked above)
        if topic_similarity < topic_similarity_threshold:
            if verbose:
                print(f"      ⚠ Topic similarity FAILED: {topic_similarity:.2f} < {topic_similarity_threshold:.2f} → score=0.0")
            final_score = 0.0
            passes = False
        else:
            # Final score with Meaning First weighting (0.9 meaning, 0.1 style)
            final_score = (proposition_recall * meaning_weight) + (style_alignment * style_weight)
            passes = proposition_recall >= proposition_recall_threshold

        # Check ending grounding (moralizing detection)
        ending_grounding_issue = self.check_ending_grounding(generated_text)
        if ending_grounding_issue:
            if verbose:
                print(f"      ⚠ Ending Grounding Issue: {ending_grounding_issue}")

        # Build feedback
        feedback_parts = []
        if ending_grounding_issue:
            feedback_parts.append(ending_grounding_issue)
        if proposition_recall < proposition_recall_threshold:
            missing = recall_details.get("missing", [])
            if missing:
                feedback_parts.append(
                    f"CRITICAL: Missing propositions ({len(missing)}/{len(propositions)}): {', '.join(missing[:3])}"
                )
            else:
                feedback_parts.append(
                    f"CRITICAL: Proposition recall too low ({proposition_recall:.2f} < {proposition_recall_threshold})"
                )

        if style_alignment < 0.7:
            feedback_parts.append(
                f"Style alignment low ({style_alignment:.2f}). Average sentence length: {style_details.get('avg_sentence_length', 0):.1f} words."
            )

        # Add coherence and topic similarity feedback
        if coherence_score < coherence_threshold:
            feedback_parts.append(
                f"Coherence too low ({coherence_score:.2f} < {coherence_threshold}): {coherence_reason}"
            )

        if topic_similarity < topic_similarity_threshold:
            feedback_parts.append(
                f"Topic similarity too low ({topic_similarity:.2f} < {topic_similarity_threshold}): Topic has drifted from original"
            )

        # Optional: Verify structural blueprint match (soft check)
        # Note: We're already in paragraph mode, so no need to check is_paragraph
        if hasattr(self, '_rhythm_map') and self._rhythm_map:
            try:
                from nltk.tokenize import sent_tokenize
                generated_sentences = sent_tokenize(generated_text)
                generated_sentence_count = len([s for s in generated_sentences if s.strip()])
                blueprint_sentence_count = len(self._rhythm_map)

                if abs(generated_sentence_count - blueprint_sentence_count) > 1:
                    # Add warning to feedback (not a failure)
                    feedback_parts.append(
                        f"[Warning: Generated paragraph has {generated_sentence_count} sentences, blueprint expected {blueprint_sentence_count}. Structural blueprint may not have been followed exactly.]"
                    )
            except Exception:
                # Ignore errors in verification (soft check)
                pass

        return {
            "pass": passes,
            "proposition_recall": proposition_recall,
            "style_alignment": style_alignment,
            "coherence_score": coherence_score,
            "topic_similarity": topic_similarity,
            "recall_score": proposition_recall,  # For compatibility
            "precision_score": 1.0,  # Not used in paragraph mode
            "adherence_score": 1.0,  # Not used in paragraph mode
            "score": final_score,
            "feedback": " ".join(feedback_parts) if feedback_parts else "Passed paragraph validation.",
            "recall_details": recall_details,
            "style_details": style_details,
            "coherence_reason": coherence_reason
        }

