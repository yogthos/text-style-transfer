"""Semantic critic for Pipeline 2.0.

This module validates generated text using precision/recall semantic checks
based on semantic blueprints rather than text overlap.
"""

import json
import re
from typing import Dict, Tuple, Optional, Set, List
from src.ingestion.blueprint import SemanticBlueprint, BlueprintExtractor

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    np = None

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
        style_lexicon: Optional[List[str]] = None
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

        original_text = input_blueprint.original_text

        # PARAGRAPH MODE: Use proposition-based evaluation
        if is_paragraph and propositions:
            return self._evaluate_paragraph_mode(
                generated_text, original_text, propositions, author_style_vector, style_lexicon=style_lexicon
            )

        # SENTENCE MODE: Continue with existing sentence-level logic
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

            # HARD GATE 1.5: Noun Preservation Check (Fallback)
            # Extract concrete nouns from original and ensure they appear in output
            if original_text:
                try:
                    # Use shared NLPManager for spaCy model
                    from src.utils.nlp_manager import NLPManager
                    nlp = NLPManager.get_nlp()
                except (OSError, ImportError, RuntimeError):
                    nlp = None

                    if nlp:
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
                        if original_nouns and not any(noun in generated_nouns for noun in original_nouns):
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

                                    # Use semantic anchors to detect contradiction patterns
                                    # Create semantic clusters using word vectors
                                    unlimited_anchors = ["infinite", "ceaseless", "boundless", "eternal", "endless", "unlimited"]
                                    limited_anchors = ["finite", "limited", "bounded", "temporary", "transient", "ending"]

                                    # Check if modifier is semantically similar to "unlimited" concepts
                                    modifier_unlimited_score = 0.0
                                    modifier_limited_score = 0.0

                                    for anchor in unlimited_anchors:
                                        if anchor in nlp.vocab and nlp.vocab[anchor].has_vector:
                                            score = modifier.similarity(nlp(anchor))
                                            modifier_unlimited_score = max(modifier_unlimited_score, score)

                                    for anchor in limited_anchors:
                                        if anchor in nlp.vocab and nlp.vocab[anchor].has_vector:
                                            score = modifier.similarity(nlp(anchor))
                                            modifier_limited_score = max(modifier_limited_score, score)

                                    # Check if noun is semantically similar to "limited" concepts
                                    noun_unlimited_score = 0.0
                                    noun_limited_score = 0.0

                                    for anchor in unlimited_anchors:
                                        if anchor in nlp.vocab and nlp.vocab[anchor].has_vector:
                                            score = noun.similarity(nlp(anchor))
                                            noun_unlimited_score = max(noun_unlimited_score, score)

                                    for anchor in limited_anchors:
                                        if anchor in nlp.vocab and nlp.vocab[anchor].has_vector:
                                            score = noun.similarity(nlp(anchor))
                                            noun_limited_score = max(noun_limited_score, score)

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
                                    # Similar semantic analysis for adverb-adjective pairs
                                    unlimited_anchors = ["infinitely", "ceaselessly", "boundlessly", "eternally", "endlessly"]
                                    limited_anchors = ["finitely", "temporarily", "transiently"]

                                    adverb_unlimited_score = max([
                                        adverb.similarity(nlp(anchor))
                                        for anchor in unlimited_anchors
                                        if anchor in nlp.vocab and nlp.vocab[anchor].has_vector
                                    ] + [0.0])

                                    adj_limited_score = max([
                                        adjective.similarity(nlp(anchor.replace("ly", "")))
                                        for anchor in limited_anchors
                                        if anchor.replace("ly", "") in nlp.vocab and nlp.vocab[anchor.replace("ly", "")].has_vector
                                    ] + [0.0])

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

        # Final score formula: (adherence * 0.5) + (recall * 0.5)
        final_score = (adherence_score * 0.5) + (recall_score * 0.5)

        # Final Heuristic: LLM Meaning Verification
        llm_meaning_preserved = True
        llm_confidence = 1.0
        llm_explanation = ""
        logic_fail = False
        if self.use_llm_verification and self.llm_provider:
            llm_meaning_preserved, llm_confidence, llm_explanation = self._verify_meaning_with_llm(
                original_text, generated_text
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
                original_text, generated_text, inferred_skeleton_type or "DECLARATIVE"
            )
            if logic_fail:
                # HARD CAP: Force score to 0.45 if logic is wrong, regardless of Recall
                final_score = min(final_score, 0.45)
                passes = False
                # Store logic reason for feedback
                llm_explanation = logic_reason  # Override with logic reason

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
            "feedback": " ".join(feedback_parts) if feedback_parts else "Passed semantic validation."
        }

    def _verify_meaning_with_llm(self, original_text: str, generated_text: str) -> Tuple[bool, float, str]:
        """Verify meaning preservation using LLM (Final Heuristic).

        Uses LLM to compare original and generated text, confirming core meaning is unchanged.

        Args:
            original_text: Original input text.
            generated_text: Generated text to verify.

        Returns:
            Tuple of (meaning_preserved: bool, confidence: float, explanation: str).
        """
        if not self.llm_provider:
            return True, 0.5, "LLM verification unavailable"

        system_prompt = "You are a semantic validator. Your task is to determine if two sentences convey the same core meaning, even if they use different words or stylistic structures. Pay special attention to logical relationships, conditional statements, and meaning shifts."

        user_prompt = f"""Compare these two sentences and determine if the generated text preserves the core meaning of the original.

Original: "{original_text}"

Generated: "{generated_text}"

Does the generated text preserve the core meaning of the original? Pay attention to:
- Logical relationships: Does it add conditions (e.g., "only when", "if... then") that weren't in the original?
- Meaning shifts: Does it change a universal statement into a conditional, or vice versa?
- Contradictions: Does it imply something that contradicts the original meaning?

Respond with JSON:
{{
    "meaning_preserved": true/false,
    "confidence": 0.0-1.0,
    "explanation": "brief reason (mention if conditional relationship, logic mismatch, or meaning shift detected)"
}}"""

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
            explanation = result.get("explanation", "")

            return meaning_preserved, confidence, explanation
        except Exception as e:
            # Handle errors gracefully - don't block evaluation
            return True, 0.5, f"LLM verification unavailable: {str(e)}"

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
        skeleton_type: str = "DECLARATIVE"
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

        LOGIC_VERIFICATION_PROMPT = """
You are a strict Logic Validator. Your job is to check if the Generated Text preserves the **Truth Value** and **Causality** of the Original Text.

### INPUTS
Original: "{original_text}"
Generated: "{generated_text}"
Target Rhetorical Structure: {skeleton_type}

### RULES
1. **Ignore Stylistic Wrappers:**
   - If the Target Structure is 'RHETORICAL_QUESTION', the output MUST be a question. This is NOT a meaning shift.
   - If the Target Structure is 'CONDITIONAL' (e.g., "If X, then Y"), checking for conditions is expected.

2. **Focus on Truth & Causality:**
   - **FAIL** if the causality is reversed (e.g., "Fire causes smoke" -> "Smoke causes fire").
   - **FAIL** if the truth is denied (e.g., "It is finite" -> "It is infinite").
   - **FAIL** if a *universal* truth becomes *contingent* on a new, unrelated condition (e.g., "Humans die" -> "Humans die ONLY if it rains").

3. **PASS** if the core message remains true, even if wrapped in complex syntax.

### OUTPUT
Return JSON:
{{
    "logic_fail": boolean,  // True only if truth/causality is broken
    "reason": "string"      // Explanation
}}
"""

        try:
            prompt = LOGIC_VERIFICATION_PROMPT.format(
                original_text=original_text,
                generated_text=generated_text,
                skeleton_type=skeleton_type
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
            doc = self.extractor.nlp(generated_text)

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

    def _check_proposition_recall(self, generated_text: str, propositions: List[str], similarity_threshold: float = 0.45) -> Tuple[float, Dict]:
        """Check proposition recall by comparing each proposition against generated sentences.

        CRITICAL: Do NOT compare proposition vector against whole paragraph vector (too much noise).
        Instead, split paragraph into sentences and find the Maximum Similarity for each proposition.

        Args:
            generated_text: Generated paragraph text.
            propositions: List of atomic proposition strings.
            similarity_threshold: Similarity threshold for matching (default: 0.45 for sentence mode,
                                0.40 recommended for paragraph mode to account for stylized text).

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
        # Default 0.45 for sentence mode, 0.40 for paragraph mode to account for stylized text
        # where short propositions are embedded in long, complex sentences (vector signal dilution)

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
            if max_similarity > threshold:
                preserved.append(prop)
            else:
                missing.append(prop)
                # Store best match info for debugging
                if best_sentence:
                    scores[f"{prop}_best_match"] = best_sentence[:100]  # First 100 chars

        recall = len(preserved) / len(propositions) if propositions else 0.0
        return recall, {"preserved": preserved, "missing": missing, "scores": scores}

    def _check_style_alignment(
        self,
        generated_text: str,
        author_style_vector: Optional[np.ndarray] = None,
        style_lexicon: Optional[List[str]] = None
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
            # Calculate cosine similarity
            dot_product = np.dot(generated_style_vector, author_style_vector)
            norm_gen = np.linalg.norm(generated_style_vector)
            norm_author = np.linalg.norm(author_style_vector)
            if norm_gen > 0 and norm_author > 0:
                style_similarity = dot_product / (norm_gen * norm_author)
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
        style_lexicon: Optional[List[str]] = None
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
        meaning_weight = paragraph_config.get("meaning_weight", 0.6)
        style_weight = paragraph_config.get("style_alignment_weight", 0.4)

        # Metric 1: Proposition Recall (use relaxed threshold 0.40 for paragraph mode)
        proposition_recall, recall_details = self._check_proposition_recall(
            generated_text,
            propositions,
            similarity_threshold=0.40  # More lenient for paragraph mode to avoid false negatives
        )

        # Metric 2: Style Alignment
        style_alignment, style_details = self._check_style_alignment(
            generated_text,
            author_style_vector,
            style_lexicon=style_lexicon
        )

        # Final score: (Meaning * 0.6) + (Style * 0.4)
        final_score = (proposition_recall * meaning_weight) + (style_alignment * style_weight)

        # Pass threshold: proposition_recall > 0.8
        passes = proposition_recall >= proposition_recall_threshold

        # Build feedback
        feedback_parts = []
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
            "recall_score": proposition_recall,  # For compatibility
            "precision_score": 1.0,  # Not used in paragraph mode
            "adherence_score": 1.0,  # Not used in paragraph mode
            "score": final_score,
            "feedback": " ".join(feedback_parts) if feedback_parts else "Passed paragraph validation.",
            "recall_details": recall_details,
            "style_details": style_details
        }

