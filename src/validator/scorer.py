"""Scoring engine for validating generated text.

This module provides scoring functions to evaluate generated text on:
1. Meaning preservation (BERTScore)
2. Style matching (POS distribution KL-Divergence)
3. Structure adherence (template matching)
4. Hallucination detection (new entities/facts)
5. LLM-based style evaluation (using LLM to evaluate style match)
"""

import numpy as np
import json
import requests
from typing import Dict, Tuple, List, Set, Optional
from scipy.stats import entropy
from sentence_transformers import SentenceTransformer

from src.models import StyleProfile
from src.ingestion.semantic import _extract_entities, _extract_content_words


def _calculate_pos_distribution(text: str) -> Dict[str, float]:
    """Calculate POS tag distribution in text.

    Args:
        text: The text to analyze.

    Returns:
        Dictionary mapping POS tags to their relative frequencies.
    """
    from nltk.tokenize import word_tokenize
    from nltk.tag import pos_tag

    tokens = word_tokenize(text)
    if not tokens:
        return {}

    pos_tags = pos_tag(tokens)
    pos_sequence = [tag for _, tag in pos_tags]

    # Count POS tags
    pos_counts = {}
    for tag in pos_sequence:
        pos_counts[tag] = pos_counts.get(tag, 0) + 1

    # Normalize to probabilities
    total = sum(pos_counts.values())
    if total == 0:
        return {}

    return {tag: count / total for tag, count in pos_counts.items()}


def _kl_divergence(p: Dict[str, float], q: Dict[str, float]) -> float:
    """Calculate KL divergence between two probability distributions.

    Args:
        p: First distribution (generated text).
        q: Second distribution (target style).

    Returns:
        KL divergence value (lower is better).
    """
    # Get all unique keys
    all_keys = set(p.keys()) | set(q.keys())

    if not all_keys:
        return 0.0

    # Create aligned probability vectors
    p_vec = np.array([p.get(k, 0.0) for k in all_keys])
    q_vec = np.array([q.get(k, 0.0) for k in all_keys])

    # Add small epsilon to avoid log(0)
    epsilon = 1e-10
    p_vec = p_vec + epsilon
    q_vec = q_vec + epsilon

    # Normalize
    p_vec = p_vec / p_vec.sum()
    q_vec = q_vec / q_vec.sum()

    # Calculate KL divergence
    kl = entropy(p_vec, q_vec)
    return kl


def _check_structure_match(generated_text: str, target_template: list) -> bool:
    """Check if generated text matches the target POS template structure.

    Args:
        generated_text: The generated sentence.
        target_template: List of POS tags representing target structure.

    Returns:
        True if structure matches approximately, False otherwise.
    """
    from nltk.tokenize import word_tokenize
    from nltk.tag import pos_tag

    try:
        tokens = word_tokenize(generated_text)
        if not tokens:
            return False

        pos_tags = pos_tag(tokens)
        actual_pos = [tag for _, tag in pos_tags]

        # Remove punctuation from both for comparison
        target_clean = [tag for tag in target_template if tag not in ['.', ',', '!', '?', ';', ':']]
        actual_clean = [tag for tag in actual_pos if tag not in ['.', ',', '!', '?', ';', ':']]

        # Check if lengths are similar (within 2 tokens)
        if abs(len(target_clean) - len(actual_clean)) > 2:
            return False

        # Check if major POS categories match (at least 60%)
        matches = sum(1 for t, a in zip(target_clean, actual_clean)
                     if t == a or (t.startswith('NN') and a.startswith('NN')) or
                     (t.startswith('VB') and a.startswith('VB')) or
                     (t.startswith('JJ') and a.startswith('JJ')) or
                     (t.startswith('DT') and a.startswith('DT')) or
                     (t.startswith('IN') and a.startswith('IN')))

        match_ratio = matches / max(len(target_clean), len(actual_clean), 1)
        return match_ratio >= 0.6
    except Exception:
        return False


def _detect_hallucinations(
    generated_text: str,
    original_text: str,
    original_entities: List[str] = None,
    original_content_words: List[str] = None
) -> Tuple[float, List[str], List[str]]:
    """Detect hallucinations (new entities/facts) in generated text.

    Args:
        generated_text: The generated text to check.
        original_text: The original input text.
        original_entities: List of entities from original (optional, will extract if None).
        original_content_words: List of content words from original (optional, will extract if None).

    Returns:
        A tuple of:
        - Hallucination score (0.0 = no hallucinations, 1.0 = many hallucinations)
        - List of new entities found in generated text
        - List of new important content words not in original
    """
    # Extract entities from both texts
    if original_entities is None:
        original_entities = _extract_entities(original_text)
    generated_entities = _extract_entities(generated_text)

    # Extract content words from both texts
    if original_content_words is None:
        original_content_words = _extract_content_words(original_text)
    generated_content_words = _extract_content_words(generated_text)

    # Normalize to lowercase sets for comparison
    original_entities_set = {e.lower() for e in original_entities}
    generated_entities_set = {e.lower() for e in generated_entities}
    original_words_set = set(original_content_words)
    generated_words_set = set(generated_content_words)

    # Find new entities (entities in generated but not in original)
    new_entities = []
    for entity in generated_entities:
        entity_lower = entity.lower()
        # Check if this entity or any part of it is in original
        entity_parts = entity_lower.split()
        found = False
        for orig_entity in original_entities_set:
            orig_parts = orig_entity.split()
            # Check for overlap
            if any(part in orig_parts for part in entity_parts) or any(orig_part in entity_parts for orig_part in orig_parts):
                found = True
                break
        if not found:
            new_entities.append(entity)

    # Find new important content words (proper nouns, capitalized words that might be entities)
    # Focus on capitalized words that aren't sentence starts
    from nltk.tokenize import word_tokenize
    from nltk.tag import pos_tag

    gen_tokens = word_tokenize(generated_text)
    gen_pos = pos_tag(gen_tokens)

    new_important_words = []
    for word, tag in gen_pos:
        word_lower = word.lower()
        # Check for capitalized proper nouns not in original
        if tag == 'NNP' and word[0].isupper() and word_lower not in original_words_set:
            # Check if it's a known entity
            if word not in original_entities:
                new_important_words.append(word)

    # Calculate hallucination score
    # Penalize based on number of new entities and important words
    entity_penalty = min(1.0, len(new_entities) * 0.5)  # 0.5 per new entity, max 1.0
    word_penalty = min(1.0, len(new_important_words) * 0.2)  # 0.2 per new important word, max 1.0
    hallucination_score = max(entity_penalty, word_penalty)

    return hallucination_score, new_entities, new_important_words


def _verify_references_and_quotations(
    generated_text: str,
    original_text: str
) -> Tuple[float, List[str], List[str]]:
    """Verify that all citation references and direct quotations are preserved.

    Extracts all [^number] style references and direct quotations from the original text
    and checks that they are present in the generated text.

    Args:
        generated_text: The generated text to verify.
        original_text: The original input text.

    Returns:
        A tuple of:
        - Preservation score (0.0 = missing items, 1.0 = all preserved)
        - List of missing references (e.g., ["[^155]", "[^25]"])
        - List of missing quotations (exact quoted text that's missing)
    """
    import re

    # Extract all [^number] style references from original
    reference_pattern = r'\[\^\d+\]'
    original_references = set(re.findall(reference_pattern, original_text))

    # Extract all direct quotations from original (text between quotes)
    # Match both single and double quotes, handling nested quotes
    quotation_pattern = r'["\'](?:[^"\']|(?<=\\)["\'])*["\']'
    original_quotations = []
    for match in re.finditer(quotation_pattern, original_text):
        quote_text = match.group(0)
        # Only consider substantial quotations (more than just punctuation)
        if len(quote_text.strip('"\'')) > 2:
            original_quotations.append(quote_text)

    # Check which references are in generated text
    generated_references = set(re.findall(reference_pattern, generated_text))
    missing_references = list(original_references - generated_references)

    # Check which quotations are in generated text
    generated_quotations = []
    for match in re.finditer(quotation_pattern, generated_text):
        quote_text = match.group(0)
        if len(quote_text.strip('"\'')) > 2:
            generated_quotations.append(quote_text)

    # Check for missing quotations (exact match required)
    missing_quotations = []
    for orig_quote in original_quotations:
        # Normalize quotes for comparison (handle both single and double)
        orig_normalized = orig_quote.strip('"\'').strip()
        found = False
        for gen_quote in generated_quotations:
            gen_normalized = gen_quote.strip('"\'').strip()
            if orig_normalized == gen_normalized:
                found = True
                break
        if not found:
            missing_quotations.append(orig_quote)

    # Calculate preservation score
    # If there are no references or quotations, score is 1.0 (nothing to preserve)
    total_items = len(original_references) + len(original_quotations)
    if total_items == 0:
        return 1.0, [], []

    missing_items = len(missing_references) + len(missing_quotations)
    preservation_score = 1.0 - (missing_items / total_items) if total_items > 0 else 1.0

    return preservation_score, missing_references, missing_quotations


def _calculate_bertscore(
    generated_text: str,
    original_text: str,
    model: SentenceTransformer = None
) -> float:
    """Calculate semantic similarity using sentence embeddings.

    Uses cosine similarity of sentence embeddings as a proxy for BERTScore.

    Args:
        generated_text: The generated text.
        original_text: The original input text.
        model: SentenceTransformer model (will load if None).

    Returns:
        Similarity score between 0 and 1 (higher is better).
    """
    if model is None:
        model = SentenceTransformer('all-MiniLM-L6-v2')

    # Get embeddings
    embeddings = model.encode([generated_text, original_text])

    # Calculate cosine similarity
    gen_emb = embeddings[0]
    orig_emb = embeddings[1]

    similarity = np.dot(gen_emb, orig_emb) / (
        np.linalg.norm(gen_emb) * np.linalg.norm(orig_emb) + 1e-8
    )

    # Normalize to [0, 1] range (cosine similarity is already in [-1, 1])
    return (similarity + 1) / 2


def score_output(
    generated_text: str,
    original_input: str,
    target_style_profile: StyleProfile,
    target_template: list,
    meaning_threshold: Optional[float] = None,
    style_threshold: Optional[float] = None,
    structure_required: bool = True,
    original_entities: List[str] = None,
    original_content_words: List[str] = None,
    hallucination_threshold: Optional[float] = None,
    sample_text: Optional[str] = None,  # Sample text for LLM style evaluation
    config_path: str = "config.json",
    llm_style_threshold: Optional[float] = None
) -> Tuple[float, Dict[str, float], bool, Dict[str, any]]:
    """Score generated output on multiple metrics.

    Evaluates the generated text on:
    1. Meaning preservation (BERTScore-like similarity)
    2. Style matching (POS distribution KL-Divergence)
    3. Structure adherence (template matching)
    4. Hallucination detection (new entities/facts)
    5. LLM-based style evaluation (if sample_text provided)

    Args:
        generated_text: The generated sentence.
        original_input: The original input text.
        target_style_profile: StyleProfile containing target style characteristics.
        target_template: List of POS tags representing target structure.
        meaning_threshold: Minimum meaning score to pass (default: 0.85).
        style_threshold: Maximum KL divergence to pass (default: 1.0).
        structure_required: Whether structure matching is required (default: True).
        original_entities: List of entities from original (optional).
        original_content_words: List of content words from original (optional).
        hallucination_threshold: Maximum hallucination score to pass (default: 0.1).
        sample_text: Sample text for LLM-based style evaluation (optional).
        config_path: Path to configuration file.
        llm_style_threshold: Minimum LLM style match score to pass (default: 0.75).

    Returns:
        A tuple of:
        - Overall score (0.0 to 1.0)
        - Dictionary with individual metric scores
        - Boolean indicating if all thresholds are met
        - Dictionary with diagnostic information (hallucinations, LLM feedback, etc.)
    """
    # Load config for threshold defaults
    with open(config_path, 'r') as f:
        config = json.load(f)
    scorer_config = config.get("scorer", {})
    if meaning_threshold is None:
        meaning_threshold = scorer_config.get("meaning_threshold", 0.85)
    if style_threshold is None:
        style_threshold = scorer_config.get("style_threshold", 1.0)
    if hallucination_threshold is None:
        hallucination_threshold = scorer_config.get("hallucination_threshold", 0.1)
    if llm_style_threshold is None:
        llm_style_threshold = scorer_config.get("llm_style_threshold", 0.75)

    # Metric 1: Meaning preservation
    meaning_score = _calculate_bertscore(generated_text, original_input)
    meaning_pass = meaning_score >= meaning_threshold

    # Metric 2: Style matching (POS distribution)
    generated_pos_dist = _calculate_pos_distribution(generated_text)

    # We need to compare against target style, but StyleProfile doesn't have POS distribution
    # Instead, we'll use a simplified check: compare against a reference distribution
    # For now, we'll use a heuristic based on the target template
    target_pos_dist = _calculate_pos_distribution(" ".join(target_template))

    # Calculate KL divergence (lower is better)
    kl_div = _kl_divergence(generated_pos_dist, target_pos_dist)
    style_score = max(0.0, 1.0 - (kl_div / style_threshold))  # Convert to score [0, 1]
    style_pass = kl_div <= style_threshold

    # Metric 3: Structure adherence
    structure_match = _check_structure_match(generated_text, target_template)
    structure_score = 1.0 if structure_match else 0.0
    structure_pass = structure_match if structure_required else True

    # Metric 4: Hallucination detection
    hallucination_score, new_entities, new_words = _detect_hallucinations(
        generated_text, original_input, original_entities, original_content_words
    )
    hallucination_pass = hallucination_score <= hallucination_threshold
    # Convert to a "no-hallucination" score (higher is better)
    no_hallucination_score = 1.0 - hallucination_score

    # Metric 5: Reference and quotation preservation (CRITICAL)
    preservation_score, missing_references, missing_quotations = _verify_references_and_quotations(
        generated_text, original_input
    )
    # This is a hard requirement - all references and quotations must be preserved
    preservation_pass = preservation_score >= 1.0  # Must be perfect (1.0)

    # Metric 6: LLM-based style evaluation (if sample text provided)
    llm_meaning_score = None
    llm_style_score = None
    llm_feedback = ""
    llm_style_pass = True  # Default to pass if not evaluated

    if sample_text:
        try:
            llm_meaning_score, llm_style_score, llm_feedback = evaluate_style_match(
                generated_text=generated_text,
                original_text=original_input,
                sample_text=sample_text,
                config_path=config_path
            )
            llm_style_pass = llm_style_score >= llm_style_threshold
            # Use LLM meaning score if available (more accurate than BERTScore)
            if llm_meaning_score is not None:
                meaning_score = (meaning_score + llm_meaning_score) / 2  # Average both
        except Exception as e:
            # If LLM evaluation fails, continue with other metrics
            llm_feedback = f"LLM evaluation failed: {e}"
            llm_style_score = 0.5  # Neutral score

    # Calculate overall score (weighted average)
    # Note: Preservation is a hard requirement, so if it fails, overall score is penalized heavily
    preservation_weight = 0.15  # Significant weight for preservation

    if llm_style_score is not None:
        # Include LLM style evaluation
        weights = {
            'meaning': 0.25,
            'style': 0.12,  # POS-based style
            'llm_style': 0.20,  # LLM-based style (higher weight)
            'structure': 0.08,
            'hallucination': 0.15,
            'preservation': preservation_weight
        }
        overall_score = (
            weights['meaning'] * meaning_score +
            weights['style'] * style_score +
            weights['llm_style'] * llm_style_score +
            weights['structure'] * structure_score +
            weights['hallucination'] * no_hallucination_score +
            weights['preservation'] * preservation_score
        )
    else:
        # Fallback to original weights if no LLM evaluation
        weights = {
            'meaning': 0.35,
            'style': 0.18,
            'structure': 0.12,
            'hallucination': 0.20,
            'preservation': preservation_weight
        }
        overall_score = (
            weights['meaning'] * meaning_score +
            weights['style'] * style_score +
            weights['structure'] * structure_score +
            weights['hallucination'] * no_hallucination_score +
            weights['preservation'] * preservation_score
        )

    # Heavy penalty if preservation fails (hard requirement)
    if not preservation_pass:
        overall_score = overall_score * 0.3  # Reduce score by 70% if preservation fails

    # All metrics must pass (including hallucination, preservation, and LLM style check)
    all_pass = bool(meaning_pass and style_pass and structure_pass and
                   hallucination_pass and preservation_pass and llm_style_pass)

    metrics = {
        'meaning': float(meaning_score),
        'style': float(style_score),
        'structure': float(structure_score),
        'hallucination': float(no_hallucination_score),
        'hallucination_score': float(hallucination_score),  # Raw score (lower is better)
        'preservation': float(preservation_score),  # Reference/quotation preservation
        'kl_divergence': float(kl_div),
        'overall': float(overall_score)
    }

    # Add LLM evaluation metrics if available
    if llm_style_score is not None:
        metrics['llm_style'] = float(llm_style_score)
        metrics['llm_meaning'] = float(llm_meaning_score) if llm_meaning_score is not None else float(meaning_score)

    diagnostics = {
        'new_entities': new_entities,
        'new_words': new_words,
        'has_hallucinations': len(new_entities) > 0 or len(new_words) > 0,
        'missing_references': missing_references,
        'missing_quotations': missing_quotations,
        'preservation_failed': not preservation_pass,
        'llm_feedback': llm_feedback,
        'llm_style_score': float(llm_style_score) if llm_style_score is not None else None
    }

    return float(overall_score), metrics, all_pass, diagnostics




def evaluate_style_match(
    generated_text: str,
    original_text: str,
    sample_text: str,
    config_path: str = "config.json"
) -> Tuple[float, float, str]:
    """Use LLM to evaluate style match and meaning preservation.

    Args:
        generated_text: The generated text to evaluate.
        original_text: The original input text.
        sample_text: The sample text defining target style.
        config_path: Path to configuration file.

    Returns:
        A tuple of:
        - Meaning preservation score (0.0 to 1.0)
        - Style match score (0.0 to 1.0)
        - Feedback string with specific suggestions
    """
    # Initialize LLM provider
    from src.generator.llm_provider import LLMProvider
    llm = LLMProvider(config_path=config_path)

    # Extract a few style examples
    from nltk.tokenize import sent_tokenize
    sample_sentences = sent_tokenize(sample_text)
    style_examples = [s.strip() for s in sample_sentences[:3] if 20 <= len(s.strip()) <= 200]
    if not style_examples:
        style_examples = [s.strip() for s in sample_sentences[:3] if len(s.strip()) > 10]

    style_examples_str = "\n".join([f"  - {ex}" for ex in style_examples]) if style_examples else "N/A"

    # Build evaluation prompt
    from pathlib import Path
    prompts_dir = Path(__file__).parent.parent.parent / "prompts"
    evaluation_template_path = prompts_dir / "scorer_evaluation_user.md"
    system_template_path = prompts_dir / "scorer_evaluation_system.md"

    if evaluation_template_path.exists():
        evaluation_template = evaluation_template_path.read_text().strip()
        evaluation_prompt = evaluation_template.format(
            original_text=original_text,
            generated_text=generated_text,
            style_examples_str=style_examples_str
        )
    else:
        # Fallback
        evaluation_prompt = f"""Evaluate the following text transformation:

ORIGINAL TEXT:
{original_text}

GENERATED TEXT:
{generated_text}

TARGET STYLE EXAMPLES:
{style_examples_str}

Evaluate on two dimensions:

1. MEANING PRESERVATION (0.0 to 1.0): Does the generated text preserve the exact meaning of the original?
   - 1.0 = Perfect meaning preservation
   - 0.5 = Some meaning lost or changed
   - 0.0 = Completely different meaning

2. STYLE MATCH (0.0 to 1.0): Does the generated text match the writing style of the examples?
   - 1.0 = Perfect style match (reads like same author)
   - 0.5 = Partial style match
   - 0.0 = No style match (completely different style)

Provide your response in this exact format:
MEANING_SCORE: [number between 0.0 and 1.0]
STYLE_SCORE: [number between 0.0 and 1.0]
FEEDBACK: [specific feedback on what's good/bad and how to improve]

Be specific in your feedback. If style doesn't match, explain what's wrong (e.g., "too verbose", "not direct enough", "sentence structure too complex")."""

    # Call LLM evaluator
    if system_template_path.exists():
        system_prompt = system_template_path.read_text().strip()
    else:
        system_prompt = "You are an expert text evaluator. Provide clear, specific feedback and scores."
    response = llm.call(
        system_prompt=system_prompt,
        user_prompt=evaluation_prompt,
        model_type="critic",
        require_json=False,
        temperature=0.2,  # Low temperature for consistent evaluation
        max_tokens=300
    )

    # Parse response
    meaning_score = 0.5  # Default
    style_score = 0.5   # Default
    feedback = response

    # Try to extract scores from response
    lines = response.split('\n')
    for line in lines:
        line_lower = line.lower().strip()
        if line_lower.startswith('meaning_score:'):
            try:
                meaning_score = float(line.split(':')[1].strip())
                meaning_score = max(0.0, min(1.0, meaning_score))
            except (ValueError, IndexError):
                pass
        elif line_lower.startswith('style_score:'):
            try:
                style_score = float(line.split(':')[1].strip())
                style_score = max(0.0, min(1.0, style_score))
            except (ValueError, IndexError):
                pass
        elif line_lower.startswith('feedback:'):
            feedback = line.split(':', 1)[1].strip() if ':' in line else response

    # If we couldn't parse scores, try to extract from feedback text
    if meaning_score == 0.5 and style_score == 0.5:
        # Look for numeric scores in the text
        import re
        scores = re.findall(r'\b(0\.\d+|1\.0)\b', response)
        if len(scores) >= 2:
            try:
                meaning_score = float(scores[0])
                style_score = float(scores[1])
            except ValueError:
                pass

    return meaning_score, style_score, feedback

