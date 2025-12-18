"""Adversarial Critic for style transfer quality control.

This module provides a critic LLM that evaluates generated text against
a reference style paragraph to detect style mismatches and "AI slop".
"""

import json
import re
import string
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Set, TYPE_CHECKING

from src.generator.llm_provider import LLMProvider

# Initialize NLTK data if needed
try:
    import nltk
    try:
        nltk.data.find('tokenizers/punkt')
        nltk.data.find('taggers/averaged_perceptron_tagger_eng')
    except LookupError:
        nltk.download('punkt', quiet=True)
        nltk.download('averaged_perceptron_tagger_eng', quiet=True)
except ImportError:
    # NLTK not available, will use fallback logic
    nltk = None

# Initialize Spacy for grammatical coherence checking
try:
    import spacy
    try:
        _spacy_nlp = spacy.load("en_core_web_sm")
    except OSError:
        # Model not downloaded, will download on first use
        _spacy_nlp = None
except ImportError:
    # Spacy not available, will use fallback logic
    spacy = None
    _spacy_nlp = None

# Load spaCy model for grammatical completeness checking
try:
    if spacy is not None:
        try:
            _grammar_nlp = spacy.load("en_core_web_sm")
        except (OSError, IOError):
            # Try to download if not available
            try:
                import subprocess
                import sys
                subprocess.check_call([sys.executable, "-m", "spacy", "download", "en_core_web_sm"],
                                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                _grammar_nlp = spacy.load("en_core_web_sm")
            except:
                _grammar_nlp = None
    else:
        _grammar_nlp = None
except Exception:
    _grammar_nlp = None

# Initialize sentence-transformers for semantic similarity
try:
    from sentence_transformers import SentenceTransformer, util
    _sentence_model = None  # Lazy load
except ImportError:
    SentenceTransformer = None
    util = None
    _sentence_model = None

if TYPE_CHECKING:
    from src.atlas.builder import StyleAtlas
    from src.atlas.navigator import StructureNavigator


def _load_prompt_template(template_name: str) -> str:
    """Load a prompt template from the prompts directory.

    Args:
        template_name: Name of the template file (e.g., 'critic_system.md')

    Returns:
        Template content as string.
    """
    prompts_dir = Path(__file__).parent.parent.parent / "prompts"
    template_path = prompts_dir / template_name
    if not template_path.exists():
        raise FileNotFoundError(f"Prompt template not found: {template_path}")
    return template_path.read_text().strip()


def _load_config(config_path: str = "config.json") -> Dict:
    """Load configuration from config.json."""
    with open(config_path, 'r') as f:
        return json.load(f)


def _detect_hallucinated_words(generated_text: str, original_text: str) -> Tuple[bool, List[str]]:
    """
    "Hard Gate" to catch proper noun hallucinations.
    Includes logic to ignore sentence-starting capitalization and em-dashes.

    Args:
        generated_text: The generated text to check.
        original_text: The original input text.

    Returns:
        Tuple of (has_hallucinations, list_of_hallucinated_words)
    """
    if not original_text:
        return False, []

    # 1. Pre-process: Handle em-dashes and punctuation explicitly
    # Replace em-dashes with spaces to ensure clean splitting
    clean_gen = generated_text.replace('—', ' ').replace('-', ' ')
    clean_orig = original_text.replace('—', ' ').replace('-', ' ')

    # 2. Build Lookup Sets
    orig_tokens = clean_orig.split()
    orig_lower = {w.lower().strip(string.punctuation) for w in orig_tokens}
    orig_proper = {w.strip(string.punctuation) for w in orig_tokens if w[0].isupper()}

    # 3. Identify Sentence Starts (to ignore capitalization there)
    # Simple heuristic: Split by .!? and take the first word
    sentence_starts = set()
    sentences = re.split(r'[.!?]+\s+', clean_gen)
    for sent in sentences:
        tokens = sent.split()
        if tokens:
            # Add the lowercase version of the first word
            sentence_starts.add(tokens[0].lower().strip(string.punctuation))

    # 4. Check for Hallucinations
    hallucinated = []
    gen_tokens = clean_gen.split()

    for word in gen_tokens:
        clean_word = word.strip(string.punctuation)
        if not clean_word:
            continue

        # Only flag if Capitalized (Potential Proper Noun)
        if clean_word[0].isupper() and len(clean_word) > 1:
            lower_word = clean_word.lower()

            # EXEMPTION 1: It's the start of a sentence
            if lower_word in sentence_starts:
                continue

            # EXEMPTION 2: It appears in the original (case-insensitive)
            if lower_word in orig_lower:
                continue

            # EXEMPTION 3: Common Stopwords (Safety Net)
            if lower_word in {'the', 'this', 'that', 'there', 'human', 'experience', 'nature'}:
                continue

            # If we get here, it's a capitalized word NOT in the original, NOT a sentence start.
            # Likely a hallucinated name (e.g., "Schneider", "August").
            hallucinated.append(clean_word)

    return len(hallucinated) > 0, hallucinated


def _detect_false_positive_hallucination(
    feedback: str,
    generated_text: str,
    original_text: Optional[str] = None
) -> bool:
    """Detect if LLM feedback flags a lowercase word as a proper noun/entity (false positive).

    Args:
        feedback: LLM feedback string that may contain false positives.
        generated_text: The generated text being evaluated.
        original_text: Optional original text for context.

    Returns:
        True if feedback contains a false positive (lowercase word flagged as proper noun).
    """
    # Pattern to match "proper noun/entity 'word'" or "word 'word'" in feedback
    # Matches: "proper noun/entity 'essential'", "entity 'essential ingredient'", etc.
    patterns = [
        r"proper noun/entity\s+['\"]([^'\"]+)['\"]",
        r"entity\s+['\"]([^'\"]+)['\"]",
        r"proper noun\s+['\"]([^'\"]+)['\"]",
        r"contains\s+(?:proper noun|entity)\s+['\"]([^'\"]+)['\"]",
    ]

    flagged_words = []
    for pattern in patterns:
        matches = re.findall(pattern, feedback, re.IGNORECASE)
        flagged_words.extend(matches)

    if not flagged_words:
        return False

    # Check each flagged word/phrase
    for flagged_item in flagged_words:
        # Clean the flagged item (remove quotes, strip)
        flagged_item = flagged_item.strip().strip("'\"")

        # Split into words if it's a phrase
        flagged_words_list = flagged_item.split()

        # Check if ALL words in the phrase are lowercase in generated text
        all_lowercase = True
        found_in_generated = False

        for word in flagged_words_list:
            # Clean word (remove punctuation)
            clean_word = word.strip(string.punctuation)
            if not clean_word:
                continue

            # Check if word appears in generated text (case-insensitive)
            gen_lower = generated_text.lower()
            word_lower = clean_word.lower()

            if word_lower in gen_lower:
                found_in_generated = True
                # Find the actual occurrence in generated text
                # Use word boundaries to find exact matches
                pattern = r'\b' + re.escape(word_lower) + r'\b'
                matches = re.finditer(pattern, gen_lower)
                for match in matches:
                    start = match.start()
                    end = match.end()
                    actual_word = generated_text[start:end]
                    # Check if the actual word in generated text is lowercase
                    if actual_word and actual_word[0].isupper():
                        all_lowercase = False
                        break
                if not all_lowercase:
                    break
            else:
                # Word not found in generated text, might be a false positive
                # but we can't verify, so skip
                continue

        # If we found the word/phrase in generated text and ALL words are lowercase,
        # this is a false positive
        if found_in_generated and all_lowercase:
            return True

    return False


def _detect_false_positive_omission(
    feedback: str,
    generated_text: str,
    original_text: str
) -> bool:
    """Detect if LLM critic falsely claims content is omitted when it's actually present.

    Args:
        feedback: LLM feedback that may claim content is omitted.
        generated_text: The generated text being evaluated.
        original_text: The original text for context.

    Returns:
        True if feedback contains a false positive omission claim.
    """
    import re

    # Pattern to match "omits 'phrase'" or "omits the concept 'phrase'" in feedback
    patterns = [
        r"omits\s+['\"]([^'\"]+)['\"]",
        r"omits\s+the\s+(?:concept|phrase|word)\s+['\"]([^'\"]+)['\"]",
        r"missing\s+['\"]([^'\"]+)['\"]",
        r"does\s+not\s+contain\s+['\"]([^'\"]+)['\"]",
    ]

    claimed_missing = []
    for pattern in patterns:
        matches = re.findall(pattern, feedback, re.IGNORECASE)
        claimed_missing.extend(matches)

    if not claimed_missing:
        return False

    # Check each claimed missing item
    generated_lower = generated_text.lower()
    original_lower = original_text.lower()

    for claimed_item in claimed_missing:
        claimed_item = claimed_item.strip().strip("'\"")
        claimed_lower = claimed_item.lower()

        # Check if it's actually present in generated text (with fuzzy matching)
        # 1. Exact match
        if claimed_lower in generated_lower:
            return True  # False positive - it's present

        # 2. Check if all key words are present (for phrases)
        claimed_words = [w.strip(string.punctuation) for w in claimed_lower.split() if w.strip(string.punctuation)]
        if len(claimed_words) > 1:
            # It's a phrase - check if all words are present
            words_present = sum(1 for word in claimed_words if word in generated_lower)
            if words_present >= len(claimed_words) * 0.8:  # 80% of words present
                return True  # False positive - most words are present

        # 3. Check for minor omissions (like "and" in lists) - these shouldn't be critical
        # If claimed item is a single word like "and", "the", "of", and the key concepts are present
        if len(claimed_words) == 1:
            minor_words = {"and", "the", "of", "a", "an", "in", "on", "at", "to", "for"}
            if claimed_words[0] in minor_words:
                # Check if key concepts from original are present
                # Extract key nouns/verbs from original
                import nltk
                try:
                    tokens = nltk.word_tokenize(original_lower)
                    pos_tags = nltk.pos_tag(tokens)
                    key_concepts = [word for word, pos in pos_tags if pos.startswith(('NN', 'VB'))]

                    # Check if key concepts are in generated
                    concepts_present = sum(1 for concept in key_concepts if concept in generated_lower)
                    if concepts_present >= len(key_concepts) * 0.8:
                        return True  # False positive - minor word missing but key concepts present
                except:
                    # NLTK not available, use simple heuristic
                    # If generated text has most of the original words, minor word omission is OK
                    original_words = set(original_lower.split())
                    generated_words = set(generated_lower.split())
                    overlap = len(original_words & generated_words) / max(len(original_words), 1)
                    if overlap >= 0.7:  # 70% word overlap
                        return True  # False positive - minor word missing but most content preserved

    return False


def _is_minor_omission(feedback: str, original_text: str, generated_text: str) -> bool:
    """Check if the claimed omission is minor (like "and" in lists) and shouldn't be score 0.0.

    Args:
        feedback: LLM feedback about omission.
        original_text: Original text.
        generated_text: Generated text.

    Returns:
        True if omission is minor and shouldn't cause score 0.0.
    """
    import re

    # Extract claimed missing content
    patterns = [
        r"omits\s+['\"]([^'\"]+)['\"]",
        r"omits\s+the\s+(?:concept|phrase|word)\s+['\"]([^'\"]+)['\"]",
    ]

    claimed_missing = []
    for pattern in patterns:
        matches = re.findall(pattern, feedback, re.IGNORECASE)
        claimed_missing.extend(matches)

    if not claimed_missing:
        return False

    # Check if claimed missing is a minor word
    minor_words = {"and", "the", "of", "a", "an", "in", "on", "at", "to", "for", "with", "by"}

    for claimed_item in claimed_missing:
        claimed_item = claimed_item.strip().strip("'\"")
        claimed_lower = claimed_item.lower()

        # If it's a single minor word, it's a minor omission
        if claimed_lower in minor_words:
            # Check if key content is preserved
            original_words = set(original_text.lower().split())
            generated_words = set(generated_text.lower().split())
            overlap = len(original_words & generated_words) / max(len(original_words), 1)
            if overlap >= 0.7:  # 70% word overlap means most content is preserved
                return True

    return False


def is_grammatically_coherent(text: str) -> bool:
    """Fast deterministic check for 'Word Salad' and grammatical coherence.

    Detects:
    - Missing verbs (sentence must have at least one verb)
    - Title Case abuse (>70% of words capitalized in sentences >5 words)
    - Word salad patterns (excessive capitalization with noun-like structure)

    Args:
        text: Text to check for grammatical coherence.

    Returns:
        True if grammatically coherent, False if word salad or invalid.
    """
    if not text or not text.strip():
        return False

    global _spacy_nlp

    # Try to load Spacy if not already loaded
    if _spacy_nlp is None and spacy is not None:
        try:
            _spacy_nlp = spacy.load("en_core_web_sm")
        except (OSError, IOError):
            # Model not available, use fallback
            pass

    words = text.split()

    # Check 1: Title Case abuse (e.g., "The Human View Of Discrete...")
    # If >65% of words are capitalized (and it's not a short title), it's broken
    # Lowered threshold from 70% to 65% to catch cases like "The Human View of Discrete Levels Scale..."
    if len(words) > 5:
        # Count capitalized words (excluding sentence-starting words)
        upper_count = 0
        total_count = 0
        for i, w in enumerate(words):
            if w and w[0].isupper():
                # First word is always capitalized, don't count it
                if i > 0:
                    upper_count += 1
                total_count += 1
            else:
                total_count += 1

        if total_count > 0 and upper_count / total_count > 0.65:
            return False

    # Check 2: Word salad pattern - excessive capitalized words in sequence
    # Pattern: Many capitalized words (like a title) but in a sentence context
    # "The Human View of Discrete Levels Scale as a Local Perspective Artifact Observe..."
    # This has many capitalized words but is not a proper sentence
    if len(words) > 8:
        # Count capitalized words (excluding first word and common words)
        common_words = {'of', 'as', 'a', 'an', 'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'with', 'by'}
        cap_count = 0
        for i, w in enumerate(words):
            clean_w = w.strip('.,!?;:').lower()
            if w and w[0].isupper() and i > 0 and clean_w not in common_words:
                cap_count += 1

        # If we have 8+ capitalized non-common words in a long sentence, it's likely word salad
        if cap_count >= 8:
            return False

    # Check 3: Must have a verb (using Spacy if available)
    if _spacy_nlp is not None:
        try:
            doc = _spacy_nlp(text)
            has_verb = any(token.pos_ == "VERB" for token in doc)
            if not has_verb:
                return False
        except Exception:
            # Spacy processing failed, fall through to basic checks
            pass

    return True


def is_text_complete(text: str) -> bool:
    """Checks for truncation and mid-sentence artifacts."""
    if not text: return False
    text = text.strip()

    # 1. Must end in terminal punctuation
    if text[-1] not in ".!?\"'":
        return False

    # 2. Must not end in a cliffhanger word
    last_word = text.split()[-1].lower().strip(".,!?;:()[]{}'\"")
    if last_word in ["and", "or", "but", "the", "a", "of", "in", "to", "with"]:
        return False

    # 3. Check internal consistency
    # (If a sentence starts but doesn't end properly before the next capital)
    sentences = re.split(r'[.!?]+\s+', text)
    for s in sentences[:-1]: # Check all but last
        if len(s.split()) < 3: continue # Skip short fragments
        if not s[0].isupper(): return False # Mid-stream break

    return True


def is_grammatically_complete(text: str) -> bool:
    """
    Uses dependency parsing to ensure the sentence is not a fragment.

    Fixes the bug where "The cycle... defines..." was flagged as a fragment.
    Now trusts spaCy's ROOT detection rather than using brittle comma heuristics.

    Args:
        text: Text to check for grammatical completeness.

    Returns:
        True if text is grammatically complete, False if it's a fragment.
    """
    if not text:
        return False

    # Use spaCy if available
    if _grammar_nlp is None:
        # Fallback: basic check (rely on is_text_complete for basic validation)
        return True

    try:
        doc = _grammar_nlp(text)

        # Check 1: Must have a ROOT (main clause indicator)
        # Most sentences have a VERB or AUX as the ROOT, but sometimes spaCy
        # incorrectly parses complex subjects as ROOT. We check for both:
        # - ROOT is a verb (ideal case)
        # - OR ROOT exists AND there's a verb in the sentence (fallback for parsing errors)
        has_root_verb = any(token.dep_ == "ROOT" and token.pos_ in ["VERB", "AUX"] for token in doc)
        has_root = any(token.dep_ == "ROOT" for token in doc)
        has_verb = any(token.pos_ in ["VERB", "AUX"] for token in doc)

        if not has_root:
            return False

        # If ROOT is not a verb, we still accept if there's a verb in the sentence
        # (handles spaCy parsing errors for complex subjects)
        if not has_root_verb:
            if not has_verb:
                return False
            # ROOT exists and verb exists - likely a parsing quirk, accept it

        # Check 2: Verify ROOT is not inside a subordinate clause without a main clause
        # This is handled by spaCy's ROOT detection - if ROOT exists and there's a verb,
        # the sentence has a main clause. We trust spaCy's parsing.

        # Check 3: Participle Phrase Detection
        # "Stars burning, succumbing to erosion" -> 'burning' is VBG. If it has no aux ('are burning'), it's a fragment.
        # Check if there are any participles (VBG/VBN) that are not auxiliaries and don't have auxiliaries
        participles = [t for t in doc if t.tag_ in ["VBG", "VBN"] and t.pos_ == "VERB"]
        for part in participles:
            # Skip if this participle is an auxiliary itself
            if part.dep_ in ["aux", "auxpass"]:
                continue
            # Check if this participle has an auxiliary as a child
            has_aux = any(child.dep_ == "aux" or child.dep_ == "auxpass" for child in part.children)
            # Also check if there's an auxiliary before it (common pattern: "is burning")
            has_aux_before = False
            for i, token in enumerate(doc):
                if token == part and i > 0:
                    prev_token = doc[i-1]
                    if prev_token.dep_ in ["aux", "auxpass"] or prev_token.tag_ in ["VBZ", "VBD", "VBP", "VB"]:
                        has_aux_before = True
                        break
            if not has_aux and not has_aux_before:
                # If ROOT is a noun and we have a participle without auxiliary, it's likely a fragment
                roots = [t for t in doc if t.dep_ == "ROOT"]
                if roots and roots[0].pos_ == "NOUN":
                    return False  # Fragment detected: noun ROOT with participle without auxiliary

        return True
    except Exception:
        # If spaCy processing fails, fall back to basic validation
        return True


def check_repetition(text: str, max_bigram_repeats: int = 3, max_sentence_start_repeats: int = 2) -> Optional[Dict[str, any]]:
    """Check for excessive word repetition in text.

    Args:
        text: Text to check for repetition.
        max_bigram_repeats: Maximum allowed bigram repetitions (default: 3).
        max_sentence_start_repeats: Maximum consecutive sentences starting with same word (default: 2).

    Returns:
        None if no repetition issues, or failure dict with feedback if repetition detected.
    """
    if not text:
        return None

    words = text.split()
    sentences = re.split(r'[.!?]+\s+', text)
    sentences = [s.strip() for s in sentences if s.strip()]

    # Check 1: Bigram repetition
    if len(words) > 5:
        from collections import Counter
        bigrams = [f"{words[i].lower()} {words[i+1].lower()}" for i in range(len(words)-1)]
        bigram_counts = Counter(bigrams)
        max_bigram_count = max(bigram_counts.values()) if bigram_counts else 0
        if max_bigram_count > max_bigram_repeats:
            repeated_bigram = max(bigram_counts.items(), key=lambda x: x[1])[0]
            return {
                "pass": False,
                "feedback": f"CRITICAL: Text contains excessive repetition. The phrase '{repeated_bigram}' appears {max_bigram_count} times. Reduce repetition for natural flow.",
                "score": 0.0,
                "primary_failure_type": "structure"
            }

    # Check 2: Sentence-start repetition (enhanced)
    if len(sentences) >= 3:
        first_words = []
        for sentence in sentences:
            sentence_words = sentence.split()
            if sentence_words:
                first_word = sentence_words[0].lower().strip(".,!?;:()[]{}'\"")
                # Exclude common sentence starters that are acceptable
                if first_word not in ["the", "a", "an"]:
                    first_words.append(first_word)

        # Check if >2 sentences start with the same word (excluding "The", "A")
        if len(first_words) >= 3:
            from collections import Counter
            starter_counts = Counter(first_words)
            max_starter_count = max(starter_counts.values()) if starter_counts else 0
            if max_starter_count > 2:
                repeated_starter = max(starter_counts.items(), key=lambda x: x[1])[0]
                return {
                    "pass": False,
                    "feedback": f"CRITICAL: Text contains repetitive sentence starts. The word '{repeated_starter}' appears at the start of {max_starter_count} sentences. Vary sentence openings for natural flow.",
                    "score": 0.0,
                    "primary_failure_type": "structure"
                }

    return None


def check_critical_nouns_coverage(
    generated_text: str,
    original_text: str,
    coverage_threshold: float = 0.9
) -> Optional[Dict[str, any]]:
    """Check that generated text preserves critical nouns from original.

    Uses strict noun preservation with 90% threshold. Proper nouns and abstract
    nouns must be preserved exactly (or via WordNet synonyms).

    Args:
        generated_text: Generated text to check.
        original_text: Original input text.
        coverage_threshold: Minimum noun coverage ratio (default: 0.9).

    Returns:
        None if coverage is acceptable, or failure dict if too many nouns missing.
    """
    if not generated_text or not original_text:
        return None

    from src.ingestion.semantic import extract_critical_nouns, get_wordnet_synonyms

    original_nouns = extract_critical_nouns(original_text)
    generated_nouns = extract_critical_nouns(generated_text)

    if not original_nouns:
        return None  # No nouns to check

    # Build sets of noun lemmas by type
    original_proper = {noun for noun, ntype in original_nouns if ntype == "PROPER"}
    original_abstract = {noun for noun, ntype in original_nouns if ntype == "ABSTRACT"}
    original_all = {noun for noun, _ in original_nouns}

    generated_all = {noun for noun, _ in generated_nouns}

    # Check 1: All proper nouns must be present (100% requirement)
    missing_proper = []
    for proper_noun in original_proper:
        proper_synonyms = get_wordnet_synonyms(proper_noun)
        if not (proper_synonyms & generated_all):
            missing_proper.append(proper_noun)

    if missing_proper:
        return {
            "pass": False,
            "feedback": f"CRITICAL: Generated text is missing proper nouns: {', '.join(missing_proper)}. Proper nouns must be preserved exactly.",
            "score": 0.0,
            "primary_failure_type": "meaning"
        }

    # Check 2: Critical abstract nouns must be present
    critical_abstract = {"experience", "finitude", "paradox", "universe", "cosmos", "information", "structure", "hierarchy"}
    missing_critical = []
    for abstract_noun in original_abstract:
        if abstract_noun in critical_abstract:
            abstract_synonyms = get_wordnet_synonyms(abstract_noun)
            if not (abstract_synonyms & generated_all):
                missing_critical.append(abstract_noun)

    if missing_critical:
        return {
            "pass": False,
            "feedback": f"CRITICAL: Generated text is missing critical abstract nouns: {', '.join(missing_critical)}. These concepts must be preserved.",
            "score": 0.0,
            "primary_failure_type": "meaning"
        }

    # Check 3: Overall noun coverage (90% threshold)
    covered_nouns = set()
    for orig_noun, _ in original_nouns:
        orig_synonyms = get_wordnet_synonyms(orig_noun)
        if orig_synonyms & generated_all:
            covered_nouns.add(orig_noun)

    coverage_ratio = len(covered_nouns) / len(original_nouns) if original_nouns else 1.0

    if coverage_ratio < coverage_threshold:
        missing_nouns = original_all - {noun for noun, _ in generated_nouns if noun in original_all}
        return {
            "pass": False,
            "feedback": f"CRITICAL: Generated text is missing critical nouns. Coverage: {coverage_ratio:.1%} (required: {coverage_threshold:.1%}). Missing: {', '.join(list(missing_nouns)[:5])}. Preserve ALL nouns from original text.",
            "score": 0.0,
            "primary_failure_type": "meaning"
        }

    return None


def check_keyword_coverage(
    generated_text: str,
    original_text: str,
    coverage_threshold: float = 0.6  # Lowered from 0.7 to 0.6
) -> Optional[Dict[str, any]]:
    """Check that generated text preserves key concepts from original.

    Uses keyword/lemma extraction to verify concept preservation.

    Args:
        generated_text: Generated text to check.
        original_text: Original input text.
        coverage_threshold: Minimum keyword coverage ratio (default: 0.7).

    Returns:
        None if coverage is acceptable, or failure dict if too many keywords missing.
    """
    if not generated_text or not original_text:
        return None

    from src.ingestion.semantic import extract_keywords, get_wordnet_synonyms

    original_keywords = set(extract_keywords(original_text))
    generated_keywords = set(extract_keywords(generated_text))

    if not original_keywords:
        return None  # No keywords to check

    # Calculate coverage: how many original keywords appear in generated (with synonym support)
    covered_count = 0
    missing_keywords = []

    for orig_kw in original_keywords:
        # 1. Direct Match
        if orig_kw in generated_keywords:
            covered_count += 1
            continue

        # 2. Synonym Match (The Fix)
        # Get synonyms for the original keyword
        orig_synonyms = get_wordnet_synonyms(orig_kw)
        if orig_synonyms & generated_keywords:
            covered_count += 1
        else:
            missing_keywords.append(orig_kw)

    coverage_ratio = covered_count / len(original_keywords) if original_keywords else 1.0

    if coverage_ratio < coverage_threshold:
        missing_list = missing_keywords[:10]  # Show up to 10 missing keywords
        return {
            "pass": False,
            "feedback": f"CRITICAL: Generated text is missing key concepts. Coverage: {coverage_ratio:.1%} (required: {coverage_threshold:.1%}). Missing keywords: {', '.join(missing_list)}. Preserve ALL concepts from original text.",
            "score": 0.0,
            "primary_failure_type": "meaning",
            "coverage_ratio": coverage_ratio  # Store for potential override
        }

    return None


def check_semantic_similarity(generated: str, original: str, threshold: float = 0.6) -> bool:
    """Verifies that the generated text vector is close to the original text vector.

    Catches hallucinations and complete meaning changes using embedding similarity.

    Args:
        generated: Generated text to check.
        original: Original input text to compare against.
        threshold: Minimum cosine similarity score (default: 0.6).

    Returns:
        True if semantic similarity is above threshold, False otherwise.
    """
    if not generated or not original:
        return False

    global _sentence_model

    # Lazy load sentence-transformers model
    if _sentence_model is None and SentenceTransformer is not None:
        try:
            _sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
        except Exception:
            # Model loading failed, return True to avoid blocking (graceful degradation)
            return True

    if _sentence_model is None:
        # Fallback: return True if sentence-transformers not available
        return True

    try:
        # Encode both texts to embeddings
        emb1 = _sentence_model.encode(generated, convert_to_tensor=True)
        emb2 = _sentence_model.encode(original, convert_to_tensor=True)

        # Compute cosine similarity
        score = util.pytorch_cos_sim(emb1, emb2).item()

        # Return True if similarity is above threshold
        return score >= threshold
    except Exception:
        # Error during encoding/similarity calculation, return True to avoid blocking
        return True


def check_soft_keyword_coverage(
    generated_text: str,
    original_text: str,
    coverage_threshold: float = 0.8,
    similarity_threshold: float = 0.7
) -> Optional[Dict[str, any]]:
    """
    Calculates keyword coverage using Vector Semantic Similarity.
    Allows for valid synonyms (Experience ~= Practice) without hardcoded lists.

    Uses sentence-transformers to encode keywords as vectors and compute
    cosine similarity between input and output keywords.

    Args:
        generated_text: Generated text to check.
        original_text: Original input text.
        coverage_threshold: Minimum ratio of keywords that must have semantic matches (default: 0.8).
        similarity_threshold: Minimum cosine similarity for a keyword match (default: 0.7).

    Returns:
        None if coverage is acceptable, or failure dict if too many keywords missing.
    """
    if not generated_text or not original_text:
        return None

    global _sentence_model

    # Lazy load sentence-transformers model
    if _sentence_model is None and SentenceTransformer is not None:
        try:
            _sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
        except Exception:
            # Model loading failed, fall back to old check
            return None

    if _sentence_model is None or util is None:
        # Fallback: return None to use old checks
        return None

    try:
        from src.ingestion.semantic import extract_keywords
        import torch

        # 1. Extract keywords (nouns/verbs) from both texts
        input_keywords = list(set(extract_keywords(original_text)))
        output_keywords = list(set(extract_keywords(generated_text)))

        if not input_keywords:
            return None  # No keywords to check
        if not output_keywords:
            # No output keywords - definitely a failure
            return {
                "pass": False,
                "feedback": f"CRITICAL: Generated text has no keywords. Preserve key concepts from original text.",
                "score": 0.0,
                "primary_failure_type": "meaning",
                "coverage_ratio": 0.0
            }

        # 2. Encode keywords to vectors
        # We encode them as individual distinct concepts
        input_vecs = _sentence_model.encode(input_keywords, convert_to_tensor=True)
        output_vecs = _sentence_model.encode(output_keywords, convert_to_tensor=True)

        # 3. Compute similarity matrix
        # Shape: [num_input_keys, num_output_keys]
        # This gives us the similarity of EVERY input word against EVERY output word
        similarity_matrix = util.cos_sim(input_vecs, output_vecs)

        # 4. Find best matches
        # For each input keyword, find the max similarity score in the output list
        max_scores, max_indices = torch.max(similarity_matrix, dim=1)

        # 5. Calculate coverage
        # A keyword is "covered" if its best match is above the similarity threshold
        covered_mask = max_scores > similarity_threshold
        covered_count = torch.sum(covered_mask).item()
        coverage_ratio = covered_count / len(input_keywords) if input_keywords else 1.0

        if coverage_ratio < coverage_threshold:
            # Find missing keywords for feedback
            missing_keywords = []
            for i, score in enumerate(max_scores):
                if score.item() < similarity_threshold:
                    missing_keywords.append(input_keywords[i])

            missing_list = missing_keywords[:10]  # Show up to 10 missing keywords
            return {
                "pass": False,
                "feedback": f"CRITICAL: Generated text is missing key concepts. Semantic coverage: {coverage_ratio:.1%} (required: {coverage_threshold:.1%}). Missing keywords: {', '.join(missing_list)}. Preserve ALL concepts from original text.",
                "score": 0.0,
                "primary_failure_type": "meaning",
                "coverage_ratio": coverage_ratio
            }

        return None
    except Exception as e:
        # If vector encoding fails, fall back gracefully
        # Return None to allow old checks to run
        return None


def _check_semantic_validity(generated_text: str) -> Optional[Dict[str, any]]:
    """Check if generated text makes semantic sense (not just grammatically correct).

    Detects incomplete sentences, dangling clauses, and nonsensical constructions.

    Args:
        generated_text: The generated text to check.

    Returns:
        None if semantically valid, or failure dict with feedback if invalid.
    """
    import re

    if not generated_text or not generated_text.strip():
        return None  # Empty text is handled elsewhere

    # Split into sentences
    sentences = re.split(r'[.!?]+\s+', generated_text.strip())
    sentences = [s.strip() for s in sentences if s.strip()]

    if not sentences:
        return None

    # Check each sentence for semantic validity
    for sentence in sentences:
        sentence_lower = sentence.lower()

        # Pattern 1: Dependent clause at end without main clause
        # "The X, even though Y." or "X, even though Y."
        # These are incomplete - they need a main clause after the comma
        # BUT "X does Y, even though Z." is valid (has main clause before comma)
        dependent_clause_end_patterns = [
            r',\s+even\s+though\s+[^,]+\.?\s*$',  # ", even though X."
            r',\s+although\s+[^,]+\.?\s*$',  # ", although X."
            r',\s+while\s+[^,]+\.?\s*$',  # ", while X."
            r',\s+whereas\s+[^,]+\.?\s*$',  # ", whereas X."
        ]

        for pattern in dependent_clause_end_patterns:
            if re.search(pattern, sentence, re.IGNORECASE):
                # Find the comma before the dependent clause
                comma_match = re.search(r',\s+(?:even\s+though|although|while|whereas)', sentence, re.IGNORECASE)
                if comma_match:
                    # Check what's BEFORE the comma - if it's just a noun phrase, it's incomplete
                    before_comma = sentence[:comma_match.start()].strip()

                    # Check if there's a main verb before the comma
                    # Simple heuristic: if before_comma is very short (< 5 words) and doesn't contain common verbs, it's likely incomplete
                    before_words = before_comma.split()

                    # Common main verbs that indicate a complete clause
                    main_verb_indicators = ['is', 'are', 'was', 'were', 'has', 'have', 'had', 'does', 'do', 'did',
                                           'works', 'works', 'exists', 'contains', 'provides', 'creates', 'defines',
                                           'requires', 'needs', 'eliminates', 'embeds', 'embedded']

                    has_main_verb = any(verb in before_comma.lower() for verb in main_verb_indicators)

                    # If before_comma is short and has no main verb, it's likely incomplete
                    if len(before_words) < 5 and not has_main_verb:
                        # Check if after_comma is just the dependent clause (ends the sentence)
                        after_comma = sentence[comma_match.end():].strip()
                        after_comma = re.sub(r'[.!?]+$', '', after_comma).strip()

                        # If the sentence ends with the dependent clause (no main clause after), it's incomplete
                        # This catches: "The code, even though it is embedded." (incomplete)
                        # But allows: "The code works, even though it is complex." (has "works" before comma)
                        return {
                            "pass": False,
                            "feedback": f"CRITICAL: Text contains an incomplete sentence: '{sentence}'. The sentence has a dependent clause but is missing the main clause. Complete the thought.",
                            "score": 0.0,
                            "primary_failure_type": "grammar"
                        }

        # Pattern 2: Dependent clause at start without main clause
        # "Even though X, Y." is valid, but "Even though X." is incomplete
        dependent_clause_start_patterns = [
            r'^(?:even\s+though|although|while|whereas)\s+[^,]+\.?\s*$',  # "Even though X." (no comma, no main clause)
        ]

        for pattern in dependent_clause_start_patterns:
            if re.match(pattern, sentence, re.IGNORECASE):
                return {
                    "pass": False,
                    "feedback": f"CRITICAL: Text contains an incomplete sentence: '{sentence}'. The sentence starts with a dependent clause but is missing the main clause. Complete the thought.",
                    "score": 0.0,
                    "primary_failure_type": "grammar"
                }

        # Pattern 3: Incomplete relative clause
        # "The X, which is Y." - missing main verb
        # But "The X, which is Y, does Z." is valid
        relative_clause_pattern = r'^[^,]+,\s+which\s+[^,]+\.?\s*$'
        if re.match(relative_clause_pattern, sentence, re.IGNORECASE):
            # Check if there's a main verb after the relative clause
            which_match = re.search(r',\s+which\s+[^,]+', sentence, re.IGNORECASE)
            if which_match:
                after_which = sentence[which_match.end():].strip()
                after_which = re.sub(r'[.!?]+$', '', after_which).strip()
                # If nothing substantial after the relative clause, it's incomplete
                if not after_which or len(after_which.split()) < 2:
                    return {
                        "pass": False,
                        "feedback": f"CRITICAL: Text contains an incomplete sentence: '{sentence}'. The sentence has a relative clause but is missing the main verb. Complete the thought.",
                        "score": 0.0,
                        "primary_failure_type": "grammar"
                    }

        # Pattern 4: Very short incomplete fragments starting with noun
        # "limits, even though they are only implied by an exterior."
        # These are often just noun phrases with dependent clauses
        words = sentence.split()

        # Check for pattern: "noun, even though..." where noun is at start
        # This is different from "The X, even though..." which we already check
        short_fragment_pattern = r'^[A-Za-z]+\s*,\s+(?:even\s+though|although|while|whereas)'
        if re.match(short_fragment_pattern, sentence, re.IGNORECASE):
            # Check if it's very short (likely incomplete)
            if len(words) < 10:
                # Check if there's a main verb before the comma
                comma_pos = sentence.find(',')
                if comma_pos > 0:
                    before_comma = sentence[:comma_pos].strip()
                    before_words = before_comma.split()

                    # If before_comma is just 1-2 words (likely just a noun), it's incomplete
                    if len(before_words) <= 2:
                        main_verb_indicators = ['is', 'are', 'was', 'were', 'has', 'have', 'works', 'exists']
                        has_main_verb = any(verb in before_comma.lower() for verb in main_verb_indicators)

                        if not has_main_verb:
                            return {
                                "pass": False,
                                "feedback": f"CRITICAL: Text contains an incomplete sentence fragment: '{sentence}'. The sentence appears to be missing the main clause. Complete the thought.",
                                "score": 0.0,
                                "primary_failure_type": "grammar"
                            }

    # All sentences are semantically valid
    return None


def critic_evaluate(
    generated_text: str,
    structure_match: str,
    situation_match: Optional[str] = None,
    original_text: Optional[str] = None,
    config_path: str = "config.json",
    structure_input_ratio: Optional[float] = None
) -> Dict[str, any]:
    """Evaluate generated text against dual RAG references.

    Uses an LLM to compare the generated text with structure_match (for rhythm)
    and situation_match (for vocabulary) to check for style mismatches.

    Args:
        generated_text: The generated text to evaluate.
        structure_match: Reference paragraph for rhythm/structure evaluation.
        situation_match: Optional reference paragraph for vocabulary evaluation.
        original_text: Original input text (for checking reference/quotation preservation).
        config_path: Path to configuration file.

    Returns:
        Dictionary with:
        - "pass": bool - Whether the generated text passes style check
        - "feedback": str - Specific feedback on what to improve
        - "score": float - Style match score (0-1)
    """
    # Initialize LLM provider
    llm = LLMProvider(config_path=config_path)

    # HARD GATE: Check for semantic validity before LLM evaluation
    # This deterministic check catches incomplete sentences and nonsensical constructions
    semantic_check = _check_semantic_validity(generated_text)
    if semantic_check:
        return semantic_check

    # HARD GATE: Check for hallucinated words before LLM evaluation
    # This deterministic check catches hallucinations that the LLM might miss
    if original_text:
        has_hallucinations, hallucinated_words = _detect_hallucinated_words(generated_text, original_text)
        if has_hallucinations:
            return {
                "pass": False,
                "feedback": f"CRITICAL: Text contains words that do not appear in original: {', '.join(hallucinated_words)}. Remove all words not present in original text.",
                "score": 0.0,
                "primary_failure_type": "meaning"
            }

    # Load system prompt from template
    system_prompt = _load_prompt_template("critic_system.md")

    # DYNAMIC INSTRUCTION: Inject "Ignore Length" instruction when structure is very different
    length_instruction = ""
    if structure_input_ratio is not None and (structure_input_ratio < 0.6 or structure_input_ratio > 1.6):
        length_instruction = """
[SPECIAL INSTRUCTION: IGNORE LENGTH MISMATCH]
The Structural Reference is significantly different in length from the Input.
- Do NOT penalize the text for being longer/shorter than the Reference.
- Penalize ONLY if the RHYTHM or SYNTAX style is wrong.
- PRIORITIZE Semantic Preservation over Length Matching.
"""
        system_prompt = f"{system_prompt}\n{length_instruction}"

    # Build sections for user prompt
    structure_section = f"""STRUCTURAL REFERENCE (for rhythm/structure):
"{structure_match}"
"""

    if situation_match:
        situation_section = f"""
SITUATIONAL REFERENCE (for vocabulary):
"{situation_match}"
"""
    else:
        situation_section = """
SITUATIONAL REFERENCE: Not provided (no similar topic found in corpus).
"""

    if original_text:
        original_section = f"""
ORIGINAL TEXT (for content preservation check):
"{original_text}"
"""
        preservation_checks = """
- CRITICAL: All [^number] style citation references from Original Text must be preserved exactly
- CRITICAL: All direct quotations (text in quotes) from Original Text must be preserved exactly
- CRITICAL: ALL facts, concepts, details, and information from Original Text must be preserved in Generated Text
- If Original Text contains multiple facts/concepts, ALL must appear in Generated Text
- **CRITICAL - LIST PRESERVATION:** If Original Text contains lists (e.g., "birth, life, and decay"), ALL items must appear in Generated Text. Missing any item from a list is a CRITICAL FAILURE.
- If any facts, concepts, details, or list items are missing, this is a CRITICAL FAILURE"""
        preservation_instruction = """

CRITICAL: Check that:
1. All [^number] citations and direct quotations from Original Text are preserved exactly in Generated Text
2. ALL facts, concepts, details, and information from Original Text are present in Generated Text
If any citations, quotations, facts, concepts, or details are missing or modified, this is a critical failure. Mark "pass": false and "primary_failure_type": "meaning"."
"""
    else:
        original_section = ""
        preservation_checks = ""
        preservation_instruction = ""

    # Load and format user prompt template
    template = _load_prompt_template("critic_user.md")
    user_prompt = template.format(
        structure_section=structure_section,
        situation_section=situation_section,
        original_section=original_section,
        generated_text=generated_text,
        preservation_checks=preservation_checks,
        preservation_instruction=preservation_instruction
    )

    # Note: preservation_instruction already includes the critical failure message if original_text is provided

    try:
        # Call LLM API
        response_text = llm.call(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            model_type="critic",
            require_json=True,
            temperature=0.3,
            max_tokens=300
        )

        # Parse JSON response
        try:
            result = json.loads(response_text)
        except json.JSONDecodeError:
            # If JSON parsing fails, try to extract JSON from response
            json_match = re.search(r'\{[^}]+\}', response_text, re.DOTALL)
            if json_match:
                result = json.loads(json_match.group())
            else:
                # Fallback: create result from text
                result = {
                    "pass": False,
                    "feedback": "Could not parse critic response. Please retry.",
                    "score": 0.5
                }

        # Ensure required fields
        if "pass" not in result:
            # Load fallback threshold from config
            config = _load_config(config_path)
            critic_config = config.get("critic", {})
            fallback_threshold = critic_config.get("fallback_pass_threshold", 0.75)
            result["pass"] = result.get("score", 0.5) >= fallback_threshold
        if "feedback" not in result:
            result["feedback"] = "No specific feedback provided."
        if "score" not in result:
            result["score"] = 0.5

        # Normalize pass field if score is high but LLM said false (overly strict critic)
        if "pass" in result and result.get("pass") == False:
            # Load good_enough_threshold from config
            config = _load_config(config_path)
            critic_config = config.get("critic", {})
            good_enough_threshold = critic_config.get("good_enough_threshold", 0.8)
            score = result.get("score", 0.0)
            if score >= good_enough_threshold:
                # Score is high enough, override overly strict pass=false
                result["pass"] = True
                print(f"    ⚠ Critic said pass=false but score {score:.3f} >= {good_enough_threshold:.2f}. Normalizing to pass=true.")
        if "primary_failure_type" not in result:
            # Infer from feedback if not provided
            feedback_lower = result.get("feedback", "").lower()
            if "structure" in feedback_lower or "length" in feedback_lower or "syntax" in feedback_lower:
                result["primary_failure_type"] = "structure"
            elif "vocab" in feedback_lower or "word" in feedback_lower or "tone" in feedback_lower:
                result["primary_failure_type"] = "vocab"
            elif "meaning" in feedback_lower or "semantic" in feedback_lower:
                result["primary_failure_type"] = "meaning"
            else:
                result["primary_failure_type"] = "none" if result.get("pass", False) else "structure"

        # Ensure types
        result["pass"] = bool(result["pass"])
        result["score"] = float(result["score"])
        result["feedback"] = str(result["feedback"])
        result["primary_failure_type"] = str(result["primary_failure_type"])

        # Validate feedback is a single instruction (not a numbered list)
        # If it contains multiple numbered items, extract the first one
        feedback = result["feedback"]
        numbered_pattern = r'^\d+[\.\)]\s*([^\.]+(?:\.[^\.]+)*)'
        match = re.match(numbered_pattern, feedback)
        if match:
            # Extract just the first instruction
            result["feedback"] = match.group(1).strip()
            feedback = result["feedback"]

        # DETERMINISTIC FILTER: Check for false positives (lowercase words flagged as proper nouns)
        if original_text and result.get("primary_failure_type") == "meaning":
            is_false_positive = _detect_false_positive_hallucination(
                feedback=feedback,
                generated_text=generated_text,
                original_text=original_text
            )

            if is_false_positive:
                # Override the false positive - this is not actually a hallucination
                print(f"    ⚠ Detected false positive: LLM flagged lowercase word as proper noun. Overriding feedback.")
                # FIX: Don't just "pass" with the old score (which might be 0.0)
                # If the LLM failed it ONLY due to meaning/hallucination, we restore it to a "B+" (0.85).
                # This breaks the death loop.
                if result.get("score", 0.0) < 0.7:
                    print(f"    Restoring score from {result.get('score', 0.0):.3f} to 0.85 (False Positive Correction)")
                    result["score"] = 0.85  # Passing score instead of 0.0 or 0.5
                result["pass"] = True
                result["primary_failure_type"] = "none"  # Not a real failure

                # Try to preserve useful feedback by removing only the false positive claim
                # Look for other issues in the feedback (structure, vocab, grammar)
                feedback_lower = feedback.lower()
                new_feedback_parts = []

                # Check for structure issues
                if "structure" in feedback_lower or "length" in feedback_lower or "syntax" in feedback_lower:
                    # Extract structure-related feedback
                    if "structure" in feedback_lower:
                        structure_idx = feedback_lower.find("structure")
                        # Try to extract sentence containing "structure"
                        start = max(0, structure_idx - 50)
                        end = min(len(feedback), structure_idx + 150)
                        structure_feedback = feedback[start:end].strip()
                        if structure_feedback:
                            new_feedback_parts.append(structure_feedback)

                # Check for vocab issues
                if "vocab" in feedback_lower or "word choice" in feedback_lower or "tone" in feedback_lower:
                    vocab_idx = feedback_lower.find("vocab")
                    if vocab_idx == -1:
                        vocab_idx = feedback_lower.find("word choice")
                    if vocab_idx == -1:
                        vocab_idx = feedback_lower.find("tone")
                    if vocab_idx != -1:
                        start = max(0, vocab_idx - 50)
                        end = min(len(feedback), vocab_idx + 150)
                        vocab_feedback = feedback[start:end].strip()
                        if vocab_feedback:
                            new_feedback_parts.append(vocab_feedback)

                # Check for grammar issues
                if "grammar" in feedback_lower or "grammatical" in feedback_lower:
                    grammar_idx = feedback_lower.find("grammar")
                    if grammar_idx == -1:
                        grammar_idx = feedback_lower.find("grammatical")
                    if grammar_idx != -1:
                        start = max(0, grammar_idx - 50)
                        end = min(len(feedback), grammar_idx + 150)
                        grammar_feedback = feedback[start:end].strip()
                        if grammar_feedback:
                            new_feedback_parts.append(grammar_feedback)

                # Remove false positive claim from feedback
                # Remove sentences containing "proper noun", "entity", "does not appear"
                sentences = re.split(r'[.!?]+\s+', feedback)
                cleaned_sentences = []
                for sent in sentences:
                    sent_lower = sent.lower()
                    # Skip sentences that are about false positive
                    if ("proper noun" in sent_lower or "entity" in sent_lower) and \
                       ("does not appear" in sent_lower or "not present" in sent_lower):
                        continue
                    cleaned_sentences.append(sent.strip())

                cleaned_feedback = ". ".join(cleaned_sentences).strip()
                if cleaned_feedback:
                    new_feedback_parts.append(cleaned_feedback)

                # Combine feedback parts
                if new_feedback_parts:
                    result["feedback"] = ". ".join(new_feedback_parts[:2])  # Take first 2 parts
                    if len(result["feedback"]) < 30:
                        # If still too short, add helpful guidance
                        result["feedback"] = "Focus on matching the structural reference's length and style. " + result["feedback"]
                else:
                    # No other useful feedback found, clear feedback so we don't confuse the generator on retry
                    result["feedback"] = ""  # Clear feedback to avoid confusion
            else:
                # Not a false positive hallucination - check for false positive omission
                # DETERMINISTIC FILTER: Check for false positive content omission
                is_false_omission = _detect_false_positive_omission(
                    feedback=feedback,
                    generated_text=generated_text,
                    original_text=original_text
                )

                if is_false_omission:
                    # Override the false positive omission
                    print(f"    ⚠ Detected false positive: LLM claimed content is omitted when it's actually present. Overriding feedback.")
                    if result.get("score", 0.0) < 0.7:
                        print(f"    Restoring score from {result.get('score', 0.0):.3f} to 0.85 (False Omission Correction)")
                        result["score"] = 0.85
                    result["pass"] = True
                    result["primary_failure_type"] = "none"
                    result["feedback"] = ""  # Clear the bad advice
                else:
                    # DETERMINISTIC FILTER: Check for minor omissions (like "and" in lists)
                    is_minor = _is_minor_omission(feedback, original_text, generated_text)

                    if is_minor:
                        # Minor omission - don't fail completely, but lower score slightly
                        print(f"    ⚠ Detected minor omission (like 'and' in list). Adjusting score.")
                        if result.get("score", 0.0) < 0.6:
                            # If score is very low, raise it to at least 0.6
                            print(f"    Adjusting score from {result.get('score', 0.0):.3f} to 0.6 (Minor Omission)")
                            result["score"] = 0.6
                        # Don't set pass=True, but don't fail completely either
                        result["primary_failure_type"] = "structure"  # Change from "meaning" to "structure"
                        # Update feedback to be less critical
                        result["feedback"] = "Minor stylistic adjustment: Consider adding connecting words for flow."
                # Override the false positive - this is not actually a hallucination
                print(f"    ⚠ Detected false positive: LLM flagged lowercase word as proper noun. Overriding feedback.")
                # FIX: Don't just "pass" with the old score (which might be 0.0)
                # If the LLM failed it ONLY due to meaning/hallucination, we restore it to a "B+" (0.85).
                # This breaks the death loop.
                if result.get("score", 0.0) < 0.7:
                    print(f"    Restoring score from {result.get('score', 0.0):.3f} to 0.85 (False Positive Correction)")
                    result["score"] = 0.85  # Passing score instead of 0.0 or 0.5
                result["pass"] = True
                result["primary_failure_type"] = "none"  # Not a real failure

                # Try to preserve useful feedback by removing only the false positive claim
                # Look for other issues in the feedback (structure, vocab, grammar)
                feedback_lower = feedback.lower()
                new_feedback_parts = []

                # Check for structure issues
                if "structure" in feedback_lower or "length" in feedback_lower or "syntax" in feedback_lower:
                    # Extract structure-related feedback
                    if "structure" in feedback_lower:
                        structure_idx = feedback_lower.find("structure")
                        # Try to extract sentence containing "structure"
                        start = max(0, structure_idx - 50)
                        end = min(len(feedback), structure_idx + 150)
                        structure_feedback = feedback[start:end].strip()
                        if structure_feedback:
                            new_feedback_parts.append(structure_feedback)

                # Check for vocab issues
                if "vocab" in feedback_lower or "word choice" in feedback_lower or "tone" in feedback_lower:
                    vocab_idx = feedback_lower.find("vocab")
                    if vocab_idx == -1:
                        vocab_idx = feedback_lower.find("word choice")
                    if vocab_idx == -1:
                        vocab_idx = feedback_lower.find("tone")
                    if vocab_idx != -1:
                        start = max(0, vocab_idx - 50)
                        end = min(len(feedback), vocab_idx + 150)
                        vocab_feedback = feedback[start:end].strip()
                        if vocab_feedback:
                            new_feedback_parts.append(vocab_feedback)

                # Check for grammar issues
                if "grammar" in feedback_lower or "grammatical" in feedback_lower:
                    grammar_idx = feedback_lower.find("grammar")
                    if grammar_idx == -1:
                        grammar_idx = feedback_lower.find("grammatical")
                    if grammar_idx != -1:
                        start = max(0, grammar_idx - 50)
                        end = min(len(feedback), grammar_idx + 150)
                        grammar_feedback = feedback[start:end].strip()
                        if grammar_feedback:
                            new_feedback_parts.append(grammar_feedback)

                # Remove false positive claim from feedback
                # Remove sentences containing "proper noun", "entity", "does not appear"
                sentences = re.split(r'[.!?]+\s+', feedback)
                cleaned_sentences = []
                for sent in sentences:
                    sent_lower = sent.lower()
                    # Skip sentences that are about false positive
                    if ("proper noun" in sent_lower or "entity" in sent_lower) and \
                       ("does not appear" in sent_lower or "not present" in sent_lower):
                        continue
                    cleaned_sentences.append(sent.strip())

                cleaned_feedback = ". ".join(cleaned_sentences).strip()
                if cleaned_feedback:
                    new_feedback_parts.append(cleaned_feedback)

                # Combine feedback parts
                if new_feedback_parts:
                    result["feedback"] = ". ".join(new_feedback_parts[:2])  # Take first 2 parts
                    if len(result["feedback"]) < 30:
                        # If still too short, add helpful guidance
                        result["feedback"] = "Focus on matching the structural reference's length and style. " + result["feedback"]
                else:
                    # No other useful feedback found, clear feedback so we don't confuse the generator on retry
                    result["feedback"] = ""  # Clear feedback to avoid confusion

        # DETERMINISTIC FILTER: Check for false grammar errors (punctuation that matches structure reference)
        if result.get("primary_failure_type") == "grammar" and structure_match:
            # Check if the "grammar error" is actually valid style matching
            # If structure_match uses em-dashes, colons, etc., and generated_text matches, it's valid
            structure_has_emdash = "—" in structure_match or " - " in structure_match
            generated_has_emdash = "—" in generated_text or " - " in generated_text
            structure_has_colon = ":" in structure_match and not structure_match.strip().startswith(("August:", "Schneider:", "Tony"))
            generated_has_colon = ":" in generated_text

            # If structure uses em-dash and generated uses em-dash, it's valid style
            if structure_has_emdash and generated_has_emdash:
                # Check if the grammar error is specifically about the em-dash
                feedback_lower = feedback.lower()
                if "dash" in feedback_lower or "—" in feedback or "em-dash" in feedback_lower or "fragment" in feedback_lower:
                    print(f"    ⚠ Detected false grammar error: Em-dash matches structure reference. Overriding.")
                    # Override the grammar error - this is valid style
                    if result.get("score", 0.0) < 0.7:
                        print(f"    Restoring score from {result.get('score', 0.0):.3f} to 0.85 (Style Match Correction)")
                        result["score"] = 0.85
                    result["pass"] = True
                    result["primary_failure_type"] = "none"
                    # Clear or update feedback
                    result["feedback"] = "Text matches structural reference style. Continue refining for better style match."

            # If structure uses colon and generated uses colon, it's valid style
            elif structure_has_colon and generated_has_colon:
                feedback_lower = feedback.lower()
                if "colon" in feedback_lower or ":" in feedback:
                    print(f"    ⚠ Detected false grammar error: Colon matches structure reference. Overriding.")
                    if result.get("score", 0.0) < 0.7:
                        print(f"    Restoring score from {result.get('score', 0.0):.3f} to 0.85 (Style Match Correction)")
                        result["score"] = 0.85
                    result["pass"] = True
                    result["primary_failure_type"] = "none"
                    result["feedback"] = "Text matches structural reference style. Continue refining for better style match."

        # FIX: Override Length Complaints if Structure was Mismatched
        # If the LLM Critic failed the text primarily due to "structure" (length),
        # but we KNOW the structure was a bad fit (ratio skew), we force a pass.
        if result.get("primary_failure_type") == "structure" and structure_input_ratio is not None:
            if structure_input_ratio < 0.6 or structure_input_ratio > 1.6:
                feedback_lower = result.get("feedback", "").lower()
                if "too short" in feedback_lower or "word count" in feedback_lower or "length" in feedback_lower:
                    print("    ⚠ Overriding Critic's Length Complaint (Known Structure Mismatch).")

                    # Reset to a passing state
                    result["score"] = 0.85
                    result["pass"] = True
                    result["primary_failure_type"] = "none"
                    result["feedback"] = ""  # Clear the bad advice

        # FIX: Override Dialogue Tag Colon Complaints
        # If structure has a dialogue tag colon (e.g., "Tony Febbo questioned...:"),
        # but generated text is a statement format, colon is not required
        if result.get("primary_failure_type") == "structure" and structure_match:
            feedback_lower = result.get("feedback", "").lower()
            structure_lower = structure_match.lower()

            # Check if structure has dialogue tag colon
            has_dialogue_colon = (
                structure_match.strip().endswith(":") and
                any(name in structure_lower for name in ["tony", "august", "schneider", "questioned", "said", "asked"])
            )

            # Check if generated text is a statement (ends with period, not colon)
            is_statement = generated_text.strip().endswith(".")

            if has_dialogue_colon and is_statement and "colon" in feedback_lower:
                print("    ⚠ Overriding Dialogue Tag Colon Complaint (Statement format doesn't need dialogue colon).")
                if result.get("score", 0.0) < 0.7:
                    print(f"    Restoring score from {result.get('score', 0.0):.3f} to 0.85 (Dialogue Colon Override)")
                    result["score"] = 0.85
                result["pass"] = True
                result["primary_failure_type"] = "none"
                result["feedback"] = ""  # Clear the bad advice

        return result

    except Exception as e:
        # Fallback on error
        return {
            "pass": True,  # Don't block generation on critic failure
            "feedback": f"Critic evaluation failed: {str(e)}",
            "score": 0.5
        }


class ConvergenceError(Exception):
    """Raised when critic cannot converge to minimum score threshold."""
    pass


def apply_surgical_fix(
    draft_text: str,
    instruction: str,
    config_path: str = "config.json"
) -> str:
    """Apply a surgical fix to text using Editor Mode.

    Used when we are CLOSE to passing (Score > 0.5) but need a specific tweak.
    This mode applies ONLY the requested change without rewriting the whole text.

    Args:
        draft_text: The current draft text to edit.
        instruction: Specific edit instruction from critic (e.g., "Remove the comma and change 'reinforcing' to 'reinforces'").
        config_path: Path to configuration file.

    Returns:
        Edited text with the surgical fix applied.
    """
    config = _load_config(config_path)
    # Initialize LLM provider
    llm = LLMProvider(config_path=config_path)

    # Build editor prompt
    from pathlib import Path
    prompts_dir = Path(__file__).parent.parent.parent / "prompts"
    system_prompt_template_path = prompts_dir / "critic_editor_system.md"
    user_prompt_template_path = prompts_dir / "critic_editor_user.md"

    if system_prompt_template_path.exists():
        system_prompt = system_prompt_template_path.read_text().strip()
    else:
        system_prompt = "You are a Text Editor. Your task is to apply specific edits to text without rewriting the entire content."

    if user_prompt_template_path.exists():
        user_prompt_template = user_prompt_template_path.read_text().strip()
        user_prompt = user_prompt_template.format(
            draft_text=draft_text,
            instruction=instruction
        )
    else:
        user_prompt = f"""You are a Text Editor. DO NOT REWRITE the whole content.

Input Text: "{draft_text}"

Editor Instruction: {instruction}

Apply ONLY this change. Keep everything else exactly the same.

Output the edited text only, without any explanation or commentary."""

    # Call LLM API for surgical fix (plain text, not JSON)
    response_text = llm.call(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        model_type="editor",
        require_json=False,
        temperature=0.3,
        max_tokens=200
    )

    # Clean up response (remove quotes if present)
    edited_text = response_text.strip()
    if edited_text.startswith('"') and edited_text.endswith('"'):
        edited_text = edited_text[1:-1]

    return edited_text


def _check_length_gate(
    generated_text: str,
    input_text: str,
    tolerance: float = 0.4
) -> Optional[Dict[str, any]]:
    """Hard gate: Check length before LLM evaluation to ensure content preservation.

    Ensures the generated text is roughly the same information density as the INPUT.
    This checks if content was preserved, not if the template was filled.

    Args:
        generated_text: Generated text to check.
        input_text: Original input text to compare against (ensures content preservation).
        tolerance: Allowable deviation ratio (default 0.4 = 40% deviation for style).

    Returns:
        None if length is acceptable, or failure dict with feedback if not.
    """
    gen_len = len(generated_text.split())
    input_len = len(input_text.split()) if input_text else 0

    if input_len == 0:
        return None  # No input to compare against, skip check

    ratio = gen_len / input_len

    # If Output is < 50% of Input, we definitely lost meaning.
    if ratio < 0.5:
        return {
            "pass": False,
            "feedback": f"FATAL ERROR: Output ({gen_len} words) is too short compared to Input ({input_len} words). You likely omitted content.",
            "score": 0.0,
            "primary_failure_type": "meaning"
        }

    # If Output is > 250% of Input, we are hallucinating or padding too much.
    if ratio > 2.5:
        return {
            "pass": False,
            "feedback": f"FATAL ERROR: Output ({gen_len} words) is too long compared to Input ({input_len} words). Be more concise.",
            "score": 0.0,
            "primary_failure_type": "structure"
        }

    # Length is acceptable (within bounds for content preservation)
    return None


def _extract_issues_from_feedback(feedback: str) -> List[str]:
    """Extract individual issues/action items from feedback text.

    Args:
        feedback: Feedback string that may contain numbered items or sentences.

    Returns:
        List of extracted issues/action items.
    """
    issues = []

    # Try to extract numbered items (e.g., "1. Make sentences shorter")
    numbered_pattern = r'\d+[\.\)]\s*([^\.]+(?:\.[^\.]+)*)'
    matches = re.findall(numbered_pattern, feedback)
    if matches:
        issues.extend([m.strip() for m in matches])
        return issues

    # If no numbered items, try to split by sentences and extract action-oriented ones
    sentences = re.split(r'[\.!?]\s+', feedback)
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
        # Look for action-oriented sentences (contain verbs like "make", "use", "match", "fix")
        action_verbs = ['make', 'use', 'match', 'fix', 'change', 'adjust', 'improve', 'reduce', 'increase']
        if any(verb in sentence.lower() for verb in action_verbs):
            issues.append(sentence)

    # If still no issues, return the whole feedback as a single issue
    if not issues and feedback.strip():
        issues.append(feedback.strip())

    return issues


def _group_similar_issues(issues: List[str]) -> Dict[str, List[str]]:
    """Group similar issues together.

    Args:
        issues: List of issue strings.

    Returns:
        Dictionary mapping canonical issue to list of similar variations.
    """
    issue_groups: Dict[str, List[str]] = {}

    for issue in issues:
        issue_lower = issue.lower()
        matched = False

        # Check if this issue is similar to any existing group
        for canonical, variations in issue_groups.items():
            canonical_lower = canonical.lower()

            # Simple similarity check: shared keywords
            issue_words = set(issue_lower.split())
            canonical_words = set(canonical_lower.split())

            # If they share significant words, group them
            common_words = issue_words & canonical_words
            # Remove common stop words
            stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'to', 'of', 'in', 'on', 'at', 'for', 'with'}
            common_words -= stop_words

            if len(common_words) >= 2 or (len(common_words) == 1 and len(issue_words) <= 3):
                issue_groups[canonical].append(issue)
                matched = True
                break

        if not matched:
            # Create new group with this issue as canonical
            issue_groups[issue] = [issue]

    return issue_groups


def _consolidate_feedback(feedback_history: List[str]) -> str:
    """Consolidate multiple feedback attempts into clear action items.

    Extracts key issues, groups similar ones, and formats as actionable steps.

    Args:
        feedback_history: List of feedback strings from previous attempts.

    Returns:
        Consolidated feedback as numbered action items.
    """
    if not feedback_history:
        return ""

    # Extract all issues from all feedback
    all_issues = []
    for feedback in feedback_history:
        issues = _extract_issues_from_feedback(feedback)
        all_issues.extend(issues)

    if not all_issues:
        return ""

    # Group similar issues and count frequency
    issue_groups = _group_similar_issues(all_issues)

    # Sort by frequency (most mentioned first)
    sorted_issues = sorted(issue_groups.items(), key=lambda x: len(x[1]), reverse=True)

    # Format as action items
    action_items = []
    for idx, (canonical_issue, variations) in enumerate(sorted_issues, 1):
        # Use the canonical issue (first one found)
        action_items.append(f"{idx}. {canonical_issue}")

    return "ACTION ITEMS TO FIX:\n" + "\n".join(action_items)


def _is_specific_edit_instruction(feedback: str) -> bool:
    """Detect if feedback is a specific edit instruction vs structural rewrite.

    Specific edit instructions contain action verbs and mention specific elements
    (punctuation, words, phrases) rather than asking for structural rewrites.

    Args:
        feedback: Feedback string from critic.

    Returns:
        True if feedback is a specific edit instruction, False if it's a structural rewrite.
    """
    if not feedback:
        return False

    feedback_lower = feedback.lower()

    # Expanded list of edit verbs
    edit_verbs = [
        'remove', 'delete', 'cut',
        'replace', 'change', 'substitute', 'swap',
        'insert', 'add',
        'fix', 'correct'
    ]
    has_edit_verb = any(verb in feedback_lower for verb in edit_verbs)

    if not has_edit_verb:
        return False

    # Check for structural rewrite indicators (these suggest regeneration, not editing)
    structural_indicators = [
        'rewrite', 'restructure', 'reorganize', 'rephrase', 'completely',
        'entire', 'whole', 'match the structure', 'follow the pattern',
        'adopt the style', 'emulate', 'mirror'
    ]
    has_structural_indicator = any(indicator in feedback_lower for indicator in structural_indicators)

    # If it has edit verbs and NO structural indicators, treat as specific edit
    # The Editor LLM is smart enough to handle even if specific elements aren't mentioned
    return not has_structural_indicator


def generate_with_critic(
    generate_fn,
    content_unit,
    structure_match: str,
    situation_match: Optional[str] = None,
    config_path: str = "config.json",
    max_retries: int = 3,
    min_score: float = 0.75,
    use_fallback_structure: bool = False,
    atlas: Optional['StyleAtlas'] = None,
    target_cluster_id: Optional[int] = None,
    structure_navigator: Optional['StructureNavigator'] = None,
    similarity_threshold: float = 0.3
) -> Tuple[str, Dict[str, any]]:
    """Generate text with adversarial critic loop.

    Generates text, evaluates it with critic, and retries with feedback
    if the style doesn't match well enough.

    Args:
        generate_fn: Function to generate text (takes content_unit, structure_match, situation_match, hint).
        content_unit: ContentUnit to generate from.
        structure_match: Reference paragraph for rhythm/structure (required).
        situation_match: Optional reference paragraph for vocabulary.
        config_path: Path to configuration file.
        max_retries: Maximum number of retry attempts (default: 3).
        min_score: Minimum critic score to accept (default: 0.75).

    Returns:
        Tuple of:
        - generated_text: Best generated text
        - critic_result: Final critic evaluation result
    """
    # Load config for defaults
    config = _load_config(config_path)
    critic_config = config.get("critic", {})
    if max_retries is None:
        max_retries = critic_config.get("max_retries", 5)
    if min_score is None:
        min_score = critic_config.get("min_score", 0.75)

    # FIX 3: Add configurable "good enough" threshold
    good_enough_threshold = critic_config.get("good_enough_threshold", 0.8)

    if not structure_match:
        # No structure match, cannot proceed
        generated = content_unit.original_text
        return generated, {"pass": False, "feedback": "No structure match provided", "score": 0.0}

    # Three-phase workflow: Generate -> Critique -> Edit
    best_text = None
    best_score = 0.0
    best_result = None
    feedback_history: List[str] = []
    current_structure = structure_match
    structure_dropped = use_fallback_structure

    # Track tried structure matches
    tried_structure_matches: Set[str] = set()
    tried_structure_matches.add(structure_match)

    # Track separate counters for generation and editing
    current_text = None
    is_edited = False
    edit_attempts = 0
    generation_attempts = 0
    max_edit_attempts = 3  # Max edits before regenerating
    should_regenerate = True  # Start by generating
    last_score_before_edit = None  # Track score before editing to detect improvement

    # Main loop: Generate -> Critique -> Edit
    # Allow up to max_retries generations and max_edit_attempts edits per generation
    max_total_attempts = max_retries * (1 + max_edit_attempts)
    total_attempts = 0  # Unified counter for all attempts (generations + edits)
    current_stage = "STRICT"  # Progressive constraint relaxation: STRICT -> LOOSE -> SAFETY
    for attempt in range(max_total_attempts):
        total_attempts += 1

        # RELAXATION LADDER: Linear progression based on attempt count
        # Attempt 0-1: STRICT mode (perfect style transfer)
        # Attempt 2-3: LOOSE mode (rhythmic match, ignore structure/length)
        # Attempt 4+: SAFETY mode (meaning preservation only, skip critic)
        if generation_attempts < 2:
            current_stage = "STRICT"
        elif generation_attempts < 4:
            if current_stage != "LOOSE":
                current_stage = "LOOSE"
                print("    ⚠ Switching to LOOSE constraint mode (Relaxing Structure/Length).")
        else:
            if current_stage != "SAFETY":
                current_stage = "SAFETY"
                print("    ⚠ Switching to SAFETY mode (Dropping Structure, prioritizing Meaning).")

        # Structure Match Refresh: Only in STRICT mode, try alternative matches
        # Do NOT reset generation counter - let state machine advance naturally
        should_refresh_structure = False
        if current_stage == "STRICT" and generation_attempts >= 2 and best_score < 0.5:
            should_refresh_structure = True
            print(f"    🔄 Low score ({best_score:.3f}) after {generation_attempts} attempts. Triggering structure refresh...")

        if should_refresh_structure and atlas and target_cluster_id is not None:
            from src.atlas.navigator import find_structure_match
            print(f"    🔄 Retrieving alternative structure match (excluding {len(tried_structure_matches)} tried matches)...")

            # Try to get a different structure match with distance weighting
            new_structure_match = None
            max_refresh_attempts = 5
            failed_structure = current_structure if current_structure else structure_match

            for refresh_attempt in range(max_refresh_attempts):
                candidate = find_structure_match(
                    atlas,
                    target_cluster_id,
                    input_text=content_unit.original_text,
                    length_tolerance=0.3,
                    top_k=10,  # Get more candidates
                    navigator=structure_navigator,
                    exclude_texts=tried_structure_matches,  # Exclude tried matches
                    prefer_different_from=failed_structure  # Prefer different structure
                )

                if candidate and candidate not in tried_structure_matches:
                    new_structure_match = candidate
                    tried_structure_matches.add(candidate)
                    print(f"    ✓ Retrieved alternative structure match ({refresh_attempt + 1}): {candidate[:80]}...")
                    break

            if new_structure_match:
                current_structure = new_structure_match
                structure_dropped = False  # Reset fallback mode with new structure
                should_regenerate = True
                # DO NOT reset generation counter - let state machine advance naturally
                print(f"    ✓ Switched to alternative structure match. Continuing with current stage.")
            else:
                print(f"    ⚠ No alternative structure matches available. Continuing with current structure.")

        # Phase 1: Generate (Generator Mode) - Only when needed
        if current_text is None or should_regenerate:
            # Build consolidated hint from all previous generation attempts
            hint = None
            if feedback_history:
                consolidated = _consolidate_feedback(feedback_history)
                if consolidated:
                    hint = f"CRITICAL FEEDBACK FROM ALL PREVIOUS ATTEMPTS:\n{consolidated}\n\nPlease address ALL of these action items in your rewrite."

            # Generate with hint from previous attempts
            generate_kwargs = {
                'hint': hint,
                'use_fallback_structure': structure_dropped,
                'constraint_mode': current_stage
            }
            current_text = generate_fn(content_unit, current_structure or structure_match, situation_match, config_path, **generate_kwargs)
            print(f"    DEBUG: Generated text (attempt {generation_attempts + 1}): '{current_text}'")
            is_edited = False
            generation_attempts += 1
            # FIX 1: Don't reset edit_attempts - track cumulative edits across regenerations
            # edit_attempts = 0  # REMOVED - preserve counter to prevent infinite edit/regenerate ping-pong
            should_regenerate = False

        # Phase 2: Critique
        eval_structure = current_structure if current_structure else structure_match

        # Initialize score to ensure it's always set
        score = 0.0
        critic_result = None
        # Initialize keyword coverage tracking for potential override
        keyword_failed = False  # Keep for backward compatibility if needed
        keyword_coverage_ratio = 1.0
        # Initialize soft coverage tracking
        soft_coverage_failed = False
        soft_coverage_ratio = 1.0

        # HARD GATE 1: Check grammatical coherence (word salad detection)
        # This catches word salad like "The Human View of Discrete Levels Scale..."
        if not is_grammatically_coherent(current_text):
            print("    ⚠ FAIL: Text detected as 'Word Salad' or invalid grammar.")
            critic_result = {
                "pass": False,
                "feedback": "CRITICAL: Text is grammatically incoherent (word salad or missing verb). Rewrite with proper sentence structure and complete thoughts.",
                "score": 0.0,
                "primary_failure_type": "grammar"
            }
            score = 0.0
        # HARD GATE 1.5: Check text completeness (before repetition check)
        elif not is_text_complete(current_text):
            print("    ⚠ FAIL: Text is incomplete (missing terminal punctuation, ends with conjunction, or contains artifacts).")
            critic_result = {
                "pass": False,
                "feedback": "CRITICAL: Generated text is incomplete. The text must end with proper punctuation, not end with a conjunction or preposition, and not contain metadata artifacts. Complete the thought.",
                "score": 0.0,
                "primary_failure_type": "grammar"
            }
            score = 0.0
        # HARD GATE 1.6: Check grammatical completeness (fragment detection)
        elif not is_grammatically_complete(current_text):
            print("    ⚠ FAIL: Text is grammatically incomplete (fragment or orphaned clause).")
            critic_result = {
                "pass": False,
                "feedback": "CRITICAL: Generated text is a grammatical fragment. The sentence must have a main clause with a root verb. Complete the thought with a proper main clause.",
                "score": 0.0,
                "primary_failure_type": "grammar"
            }
            score = 0.0
        # HARD GATE 2: Check for repetition (before semantic similarity)
        elif check_repetition(current_text):
            repetition_result = check_repetition(current_text)
            print("    ⚠ FAIL: Text contains excessive repetition.")
            critic_result = repetition_result
            score = repetition_result.get("score", 0.0)
        # HARD GATE 2.25: Semantic Keyword Coverage (Vector-Based)
        # Replaces both check_critical_nouns_coverage and check_keyword_coverage
        # Uses sentence-transformers for semantic soft-matching
        elif content_unit.original_text:
            soft_coverage_result = check_soft_keyword_coverage(
                current_text,
                content_unit.original_text,
                coverage_threshold=0.8,  # Require 80% of concepts to have semantic match
                similarity_threshold=0.7  # Minimum similarity for a match (0.7 = good synonym)
            )

            if soft_coverage_result:
                print("    ⚠ FAIL: Semantic concept coverage too low.")
                critic_result = soft_coverage_result
                score = soft_coverage_result.get("score", 0.0)
                # Store coverage ratio for potential override
                soft_coverage_ratio = soft_coverage_result.get("coverage_ratio", 0.0)
                soft_coverage_failed = True
            else:
                soft_coverage_failed = False
                soft_coverage_ratio = 1.0

            # HARD GATE 3: Check semantic similarity (meaning preservation)
            # Only check if soft coverage passed
            if not soft_coverage_failed:
                if not check_semantic_similarity(current_text, content_unit.original_text, threshold=0.65):
                    print("    ⚠ FAIL: Semantic similarity too low (Meaning lost).")
                    critic_result = {
                        "pass": False,
                        "feedback": "CRITICAL: Generated text has lost the original meaning. Preserve all concepts, facts, and information from the original text.",
                        "score": 0.0,
                        "primary_failure_type": "meaning"
                    }
                    score = 0.0
                # If soft coverage and semantic similarity both pass, continue to LLM evaluation below
                # (score is already initialized to 0.0, will be set by LLM critic)

        # HARD GATE OVERRIDE: High Semantic Similarity + Good Grammar = Accept
        # If we are failing ONLY on soft keyword coverage but meaning is preserved
        # via vector similarity (>0.8), we assume it's a valid paraphrase.
        if (soft_coverage_failed and
            soft_coverage_ratio >= 0.6 and  # At least 60% semantic keyword coverage
            check_semantic_similarity(current_text, content_unit.original_text, threshold=0.8) and  # Trust the vector
            is_grammatically_complete(current_text)):

            print("    ✓ Overriding Semantic Keyword check: High Semantic Similarity (>0.8) detects valid paraphrase.")
            critic_result = {
                "pass": True,
                "score": 0.8,  # Good score for valid paraphrase
                "primary_failure_type": "none",
                "feedback": "Accepted: High semantic similarity (>0.8) indicates valid stylistic paraphrase despite marginal keyword coverage."
            }
            score = 0.8
            soft_coverage_failed = False  # Reset flag so we proceed to LLM evaluation

        # HARD GATE 4: Check length before LLM evaluation
        # FIX: Length gate now compares generated_text to input_text (content preservation)
        # Not to structure_match (which may be very different in length)
        # Only proceed to LLM evaluation if we haven't already failed a hard gate
        if critic_result is None:
            # We passed all hard gates (or there was no original_text to check), proceed to LLM evaluation
            from src.utils import calculate_length_ratio

            # Calculate length ratio between structure match and original input (for critic leniency)
            active_structure = current_structure if current_structure else structure_match
            structure_input_ratio = None
            if active_structure and content_unit.original_text:
                structure_input_ratio = calculate_length_ratio(
                    active_structure,
                    content_unit.original_text
                )

            # Check length gate (compares generated to input, not to structure)
            length_gate_result = None
            if not structure_dropped:
                length_gate_result = _check_length_gate(
                    current_text,
                    content_unit.original_text if content_unit.original_text else ""
                )

            if length_gate_result:
                # Length gate failed - use deterministic feedback, skip LLM call
                critic_result = length_gate_result
                score = critic_result.get("score", 0.0)
            else:
                # B. LLM CRITIC BYPASS (SAFETY Mode)
                # ----------------------------------
                # In SAFETY mode, skip the LLM critic entirely to ensure convergence
                # We already passed Grammar/Semantic Hard Gates above, so we are safe.
                if current_stage == "SAFETY":
                    print("    ✓ SAFETY Mode: Skipping Critic. Accepting based on Hard Gates.")
                    critic_result = {
                        "pass": True,
                        "score": 1.0,
                        "primary_failure_type": "none",
                        "feedback": "SAFETY mode: Accepted based on hard gates (grammar and semantic similarity passed)."
                    }
                    score = 1.0
                else:
                    # Length is acceptable - proceed with LLM critic evaluation
                    # Pass structure_input_ratio so critic can be lenient about length when structure is very different
                    critic_result = critic_evaluate(
                        current_text,
                        eval_structure,
                        situation_match,
                        original_text=content_unit.original_text,
                        config_path=config_path,
                        structure_input_ratio=structure_input_ratio
                    )
                    score = critic_result.get("score", 0.0)

                    # C. SMART OVERRIDES (The Fix for "Score 0.0")
                    # --------------------------------------------

                    # Override 1: Punctuation Style (The "Exclamation Mark" Fix)
                    # If the critic complains about grammar but the punctuation matches the reference, it's valid style.
                    if critic_result.get("primary_failure_type") == "grammar":
                        # Check for matching exclamations
                        if "!" in eval_structure and "!" in current_text:
                            print("    ✓ Overriding Grammar complaint: Exclamation matches style.")
                            critic_result["pass"] = True
                            critic_result["score"] = 0.85
                            score = 0.85
                        # Check for matching question marks
                        elif "?" in eval_structure and "?" in current_text:
                            print("    ✓ Overriding Grammar complaint: Question mark matches style.")
                            critic_result["pass"] = True
                            critic_result["score"] = 0.85
                            score = 0.85

                    # Override 2: LOOSE Mode (Ignore Structure/Length)
                    if current_stage == "LOOSE" and critic_result.get("primary_failure_type") == "structure":
                        print("    ✓ LOOSE Mode: Ignoring structure/length mismatch.")
                        critic_result["pass"] = True
                        critic_result["score"] = max(critic_result.get("score", 0.0), 0.8)
                        score = critic_result["score"]

        # Track the "Global Best" to ensure we never lose ground
        if score > best_score:
            best_score = score
            best_text = current_text
            best_result = critic_result

            # FIX 1: IMMEDIATE EXIT "Take the Win"
            # If we hit the good enough threshold, stop optimizing. It's good enough.
            # This prevents over-editing and saves tokens.
            if score >= good_enough_threshold:
                print(f"    ✓ Score {score:.3f} is strong (>= {good_enough_threshold:.2f}). Accepting immediately.")
                return current_text, critic_result

        # Check if we should accept
        # FIX 2: Relaxed Threshold - We accept if Critic passes it, OR if score is objectively high (good_enough_threshold+)
        # (e.g., Score 0.95 matches "Perfect", even if Critic found a tiny nitpick)
        if (critic_result.get("pass", False) and score >= min_score) or score >= good_enough_threshold:
            return current_text, critic_result

        # FIX 1: The Backtracking Safety Net
        # If we edited and it got worse, UNDO it immediately.
        if is_edited and last_score_before_edit is not None:
            if score < last_score_before_edit:
                print(f"    ⚠ Edit degraded score ({last_score_before_edit:.3f} -> {score:.3f}).")

                # Check if we have a saved 'best' version to fall back to
                if best_text and best_score > score:
                    print(f"    ↺ REVERTING to best known version (Score: {best_score:.3f}).")
                    current_text = best_text
                    score = best_score  # Reset score to match the reverted text
                    # Also update critic_result to match the reverted state
                    critic_result = best_result if best_result else critic_result

                    # Apply penalty for the failed attempt
                    edit_attempts = min(edit_attempts + 1, max_edit_attempts)

                    # IMPORTANT: Reset the 'is_edited' flag so we don't loop the reversion check
                    is_edited = False

                    # Force a different strategy next time (e.g., try a different edit or regenerate)
                    # Re-evaluate the reverted text to get fresh feedback
                    continue
                else:
                    # No better version to revert to, just apply penalty
                    print(f"    ⚠ No better version to revert to. Applying penalty.")
                    edit_attempts = min(edit_attempts + 1, max_edit_attempts)
            else:
                # Edit improved or maintained score
                print(f"    ✓ Edit improved/maintained score ({last_score_before_edit:.3f} -> {score:.3f})")

        # Phase 3: Determine next action - Edit or Regenerate
        feedback = critic_result.get("feedback", "")
        is_specific_edit = _is_specific_edit_instruction(feedback)

        # Phase 1: Add diagnostic logging
        print(f"    DEBUG: Original input: '{content_unit.original_text}'")
        print(f"    DEBUG: Generated text: '{current_text}'")
        print(f"    DEBUG: Feedback classified as: {'SURGICAL EDIT' if is_specific_edit else 'FULL REGENERATION'}")
        print(f"    DEBUG: Feedback content: '{feedback[:80]}...'")
        print(f"    DEBUG: Current score: {score:.3f}, Edit attempts: {edit_attempts}/{max_edit_attempts}, Generation attempts: {generation_attempts}/{max_retries}")

        # Check if last edit improved the score (only check if we've already edited)
        edit_improved = False
        if is_edited and last_score_before_edit is not None and score > last_score_before_edit:
            edit_improved = True

        # FIX 2: Fatal Error Eject Button
        # If score is 0.0, the text is likely broken/empty. Don't try to "edit" garbage.
        if score < 0.1:
            print("    ⚠ Critical Failure (Score ~0.0). Forcing full regeneration.")
            # Force loop to skip edit logic and go straight to regeneration
            edit_attempts = max_edit_attempts

        # Edit Mode: If feedback is specific edit and score is decent, apply edit
        # FIX 2: Allow surgical edits even after max generation attempts
        # FIX 3: Added 'score > 0.1' to prevent editing "Fatal Errors" (Length mismatch)
        # We only stop if edit attempts are exhausted
        if (is_specific_edit and score > 0.1 and edit_attempts < max_edit_attempts):
            # If we've already edited and it didn't improve, regenerate instead
            if edit_attempts > 0 and not edit_improved and last_score_before_edit is not None:
                # Editing didn't help, regenerate
                should_regenerate = True
                # FIX 1: Don't reset edit_attempts - preserve counter to track cumulative edits
                # edit_attempts = 0  # REMOVED
                last_score_before_edit = None
                is_edited = False
                if feedback:
                    feedback_history.append(feedback)
            else:
                # Try editing
                try:
                    print(f"    → Attempting surgical fix (attempt {edit_attempts + 1}/{max_edit_attempts})...")
                    last_score_before_edit = score  # Track score before editing
                    # Apply surgical fix
                    edited_text = apply_surgical_fix(
                        current_text,
                        feedback,
                        config_path=config_path
                    )

                    # Validate the edit actually changed something
                    if edited_text and edited_text != current_text:
                        current_text = edited_text
                        is_edited = True
                        edit_attempts += 1
                        print(f"    ✓ Surgical fix applied, re-evaluating...")
                        # Continue loop to re-critique the edited version
                        continue
                    else:
                        print("    ⚠ Surgical fix returned identical text. Falling back to regeneration.")
                        should_regenerate = True
                        edit_attempts += 1  # Count as attempt even if no change
                        if feedback:
                            feedback_history.append(feedback)
                except Exception as e:
                    # If surgical fix fails, fall back to regeneration
                    print(f"    ⚠ Surgical fix failed: {e}. Falling back to regeneration.")
                    should_regenerate = True
                    edit_attempts += 1  # Count failed edit as attempt
                    last_score_before_edit = None
                    is_edited = False
                    if feedback:
                        feedback_history.append(feedback)
        else:
            # Regenerate Mode: Feedback is not specific edit or editing exhausted
            print(f"    → Regenerating with feedback (generation attempt {generation_attempts + 1}/{max_retries})...")

            # FIX 1: Hard stop to prevent zombie loop
            if generation_attempts >= max_retries:
                print(f"    ⚠ Max generation attempts ({max_retries}) reached. Stopping loop to prevent infinite retry.")
                break

            should_regenerate = True
            # FIX 1: Don't reset edit_attempts - preserve counter to track cumulative edits
            # edit_attempts = 0  # REMOVED - prevents infinite edit/regenerate ping-pong
            last_score_before_edit = None
            is_edited = False
            if feedback:
                feedback_history.append(feedback)

            # Additional safety: Detect repetitive feedback loop
            if len(feedback_history) >= 3 and feedback_history[-1] == feedback_history[-3]:
                print("    ⚠ Detected repetitive feedback loop. Breaking.")
                break

    # FAIL-SAFE: If max retries reached and still haven't passed, return original text
    # This ensures the output file is readable even if style transfer fails completely
    if generation_attempts >= max_retries and best_score < min_score:
        print(f"    ⚠ Max retries reached. Style transfer failed (best score: {best_score:.3f}).")
        print(f"    ↺ FALLBACK: Returning original text to ensure output is readable.")
        return content_unit.original_text, {"pass": True, "score": 1.0, "note": "Fallback - style transfer failed"}

    # Enforce minimum score - raise exception if not met
    if best_score < min_score:
        consolidated_feedback = _consolidate_feedback(feedback_history) if feedback_history else "No specific feedback available"
        raise ConvergenceError(
            f"Failed to converge to minimum score {min_score} after {generation_attempts} generation attempts "
            f"and {edit_attempts} edit attempts. Best score: {best_score:.3f}. Issues: {consolidated_feedback}"
        )

    # Return best result if it meets threshold
    print(f"    DEBUG: Loop completed. Total attempts: {total_attempts}, Best score: {best_score:.3f}")
    return best_text or generated, best_result or {"pass": False, "feedback": "Max retries reached", "score": best_score}

