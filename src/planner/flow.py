"""Flow planning module for style transfer.

This module selects sentence templates based on context, previous structure,
and semantic sentiment to guide the generation process.
"""

import numpy as np
from typing import List, Optional, Tuple, Dict
from collections import defaultdict

from src.analyzer.structure import classify_sentence_type, build_flow_markov
from src.models import ContentUnit


def _extract_pos_template(sentence: str) -> List[str]:
    """Extract POS tag sequence (template) from a sentence.

    Args:
        sentence: The sentence to analyze.

    Returns:
        A list of POS tags representing the sentence structure.
    """
    from nltk.tokenize import word_tokenize
    from nltk.tag import pos_tag

    tokens = word_tokenize(sentence)
    if not tokens:
        return []

    pos_tags = pos_tag(tokens)
    return [tag for _, tag in pos_tags]


def _detect_sentiment(sentence: str) -> str:
    """Detect sentiment polarity of a sentence.

    Args:
        sentence: The sentence to analyze.

    Returns:
        One of: 'Positive', 'Negative', 'Neutral'
    """
    from nltk.sentiment import SentimentIntensityAnalyzer

    try:
        sia = SentimentIntensityAnalyzer()
        scores = sia.polarity_scores(sentence)
        compound = scores['compound']

        if compound >= 0.05:
            return 'Positive'
        elif compound <= -0.05:
            return 'Negative'
        else:
            return 'Neutral'
    except LookupError:
        # Fallback: use spaCy semantic similarity for sentiment
        sentence_lower = sentence.lower()

        # Use spaCy to determine sentiment dynamically
        try:
            from src.utils.nlp_manager import NLPManager
            nlp = NLPManager.get_nlp()
            doc = nlp(sentence_lower)

            positive_score = 0.0
            negative_score = 0.0

            positive_seed = nlp.vocab.get("good")
            negative_seed = nlp.vocab.get("bad")

            for token in doc:
                if token.has_vector:
                    if positive_seed and positive_seed.has_vector:
                        positive_score += token.similarity(positive_seed)
                    if negative_seed and negative_seed.has_vector:
                        negative_score += token.similarity(negative_seed)

            # Normalize by token count
            token_count = len([t for t in doc if t.has_vector])
            if token_count > 0:
                positive_score /= token_count
                negative_score /= token_count

            if positive_score > negative_score + 0.1:
                return 'Positive'
            elif negative_score > positive_score + 0.1:
                return 'Negative'
            else:
                return 'Neutral'
        except (ImportError, OSError, KeyError, AttributeError):
            # If spaCy unavailable, return neutral
            return 'Neutral'


def _get_likely_next_type(
    previous_structure_type: str,
    flow_markov: np.ndarray,
    type_to_index: dict
) -> Optional[str]:
    """Determine the most likely next sentence type based on Markov chain.

    Args:
        previous_structure_type: The type of the previous sentence.
        flow_markov: The sentence flow transition matrix.
        type_to_index: Mapping from sentence types to matrix indices.

    Returns:
        The most likely next sentence type, or None if not found.
    """
    if previous_structure_type not in type_to_index:
        return None

    prev_idx = type_to_index[previous_structure_type]

    if flow_markov.size == 0 or prev_idx >= flow_markov.shape[0]:
        return None

    # Get transition probabilities from previous type
    transition_probs = flow_markov[prev_idx, :]

    # Find the most likely next type
    next_idx = np.argmax(transition_probs)

    # Map back to sentence type
    index_to_type = {idx: stype for stype, idx in type_to_index.items()}
    return index_to_type.get(next_idx)


def select_template(
    previous_structure_type: str,
    input_sentiment: str,
    sample_corpus: str,
    flow_markov: Optional[np.ndarray] = None,
    type_to_index: Optional[dict] = None
) -> List[str]:
    """Select a sentence template based on context and style.

    This function:
    1. Consults sentence_flow_markov to determine likely next sentence type
    2. Filters sample_corpus for sentences matching that type AND similar sentiment
    3. Extracts the POS structure (skeleton) of the selected sentence

    Args:
        previous_structure_type: The structure type of the previously generated sentence
            (e.g., 'Simple', 'Compound', 'Complex', 'Fragment').
        input_sentiment: The sentiment polarity of the input chunk ('Positive', 'Negative', 'Neutral').
        sample_corpus: The sample text corpus to draw templates from.
        flow_markov: Optional pre-computed flow Markov matrix. If None, will compute from sample_corpus.
        type_to_index: Optional pre-computed type to index mapping. If None, will compute from sample_corpus.

    Returns:
        A list of POS tags representing the target template structure.
        Example: ['DT', 'JJ', 'NN', 'VBZ', 'IN', 'DT', 'NN']
    """
    from nltk.tokenize import sent_tokenize

    # Build flow markov if not provided
    if flow_markov is None or type_to_index is None:
        _, type_to_index, _ = build_flow_markov(sample_corpus)
        flow_markov, _, _ = build_flow_markov(sample_corpus)

    # Determine likely next sentence type
    likely_type = _get_likely_next_type(previous_structure_type, flow_markov, type_to_index)

    # If we can't determine from Markov chain, use the previous type
    if likely_type is None:
        likely_type = previous_structure_type

    # Tokenize sample corpus into sentences
    sentences = sent_tokenize(sample_corpus)

    # Filter sentences by type and sentiment
    candidate_sentences = []
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue

        # Check sentence type
        sent_type = classify_sentence_type(sentence)
        if sent_type != likely_type:
            continue

        # Check sentiment
        sent_sentiment = _detect_sentiment(sentence)
        if sent_sentiment != input_sentiment:
            continue

        candidate_sentences.append(sentence)

    # If no exact matches, relax constraints
    if not candidate_sentences:
        # Try matching just by type
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            sent_type = classify_sentence_type(sentence)
            if sent_type == likely_type:
                candidate_sentences.append(sentence)

    # If still no matches, use any sentence of the likely type
    if not candidate_sentences:
        # Fallback: use first sentence of any type
        for sentence in sentences:
            sentence = sentence.strip()
            if sentence:
                candidate_sentences.append(sentence)
                break

    # Select the first candidate and extract its POS template
    if candidate_sentences:
        selected_sentence = candidate_sentences[0]
        template = _extract_pos_template(selected_sentence)
        return template

    # Ultimate fallback: return a simple default template
    return ['DT', 'JJ', 'NN', 'VBZ', 'IN', 'DT', 'NN']


def _build_context_embedding(
    paragraph_position: str,
    sentence_position: str,
    previous_type: str,
    sentiment: str,
    input_length_category: str
) -> np.ndarray:
    """Build a context embedding for similarity matching.

    Creates a text representation of the context and converts it to an embedding
    using sentence-transformers.

    Args:
        paragraph_position: One of 'first', 'middle', 'last'
        sentence_position: One of 'first', 'middle', 'last'
        previous_type: Previous sentence type (e.g., 'Simple', 'Compound')
        sentiment: Sentiment polarity ('Positive', 'Negative', 'Neutral')
        input_length_category: Input length category ('short', 'medium', 'long')

    Returns:
        Normalized embedding vector for cosine similarity.
    """
    try:
        from sentence_transformers import SentenceTransformer
        import os

        # Use a lightweight model for embeddings
        model_name = 'all-MiniLM-L6-v2'

        # Cache model in a module-level variable to avoid reloading
        if not hasattr(_build_context_embedding, '_model'):
            _build_context_embedding._model = SentenceTransformer(model_name)

        model = _build_context_embedding._model
    except ImportError:
        # Fallback: return a simple hash-based embedding if sentence-transformers not available
        context_str = f"{paragraph_position}_{sentence_position}_{previous_type}_{sentiment}_{input_length_category}"
        # Create a simple deterministic embedding from hash
        import hashlib
        hash_val = int(hashlib.md5(context_str.encode()).hexdigest(), 16)
        # Create a 384-dimensional vector (same as MiniLM) from hash
        np.random.seed(hash_val % (2**32))
        embedding = np.random.randn(384).astype(np.float32)
        embedding = embedding / np.linalg.norm(embedding)  # Normalize
        return embedding

    # Build context description
    context_parts = [
        f"paragraph position: {paragraph_position}",
        f"sentence position: {sentence_position}",
        f"previous sentence type: {previous_type}",
        f"sentiment: {sentiment}",
        f"input length: {input_length_category}"
    ]
    context_text = ". ".join(context_parts)

    # Generate embedding
    embedding = model.encode(context_text, normalize_embeddings=True)
    return embedding


def _categorize_length(sentence: str) -> str:
    """Categorize sentence length.

    Args:
        sentence: The sentence text.

    Returns:
        One of 'short', 'medium', 'long'
    """
    word_count = len(sentence.split())
    if word_count < 10:
        return 'short'
    elif word_count < 20:
        return 'medium'
    else:
        return 'long'


def _get_position_label(idx: int, total: int) -> str:
    """Get position label (first, middle, last).

    Args:
        idx: Current index (0-based).
        total: Total count.

    Returns:
        One of 'first', 'middle', 'last'
    """
    if total == 1:
        return 'first'
    elif idx == 0:
        return 'first'
    elif idx == total - 1:
        return 'last'
    else:
        return 'middle'


def _build_sample_context_index(sample_text: str) -> List[Dict]:
    """Build a context index from sample text for similarity matching.

    Analyzes sample text paragraph by paragraph, extracting context information
    and POS templates for each sentence.

    Args:
        sample_text: The sample text to analyze.

    Returns:
        List of context records, each containing:
        - 'paragraph_position': str (first/middle/last)
        - 'sentence_position': str (first/middle/last)
        - 'sentence_type': str (Simple/Compound/Complex/Fragment)
        - 'sentiment': str (Positive/Negative/Neutral)
        - 'pos_template': List[str] (POS tag sequence)
        - 'embedding': np.ndarray (context embedding)
        - 'sentence': str (original sentence text)
    """
    from nltk.tokenize import sent_tokenize

    # Split into paragraphs
    paragraphs = [p.strip() for p in sample_text.split('\n\n') if p.strip()]
    if len(paragraphs) == 1 and '\n' in sample_text:
        paragraphs = [p.strip() for p in sample_text.split('\n') if p.strip()]
    if not paragraphs:
        paragraphs = [sample_text.strip()] if sample_text.strip() else []

    total_paragraphs = len(paragraphs)
    context_index = []

    for para_idx, paragraph in enumerate(paragraphs):
        sentences = sent_tokenize(paragraph)
        paragraph_length = len([s for s in sentences if s.strip()])

        paragraph_position = _get_position_label(para_idx, total_paragraphs)

        for sent_idx, sentence in enumerate(sentences):
            sentence = sentence.strip()
            if not sentence:
                continue

            # Extract context information
            sentence_type = classify_sentence_type(sentence)
            sentiment = _detect_sentiment(sentence)
            pos_template = _extract_pos_template(sentence)
            sentence_position = _get_position_label(sent_idx, paragraph_length)
            length_category = _categorize_length(sentence)

            # Build context embedding
            # For previous_type, use 'Simple' as default for first sentence, otherwise use previous
            previous_type = 'Simple' if sent_idx == 0 else classify_sentence_type(sentences[sent_idx - 1].strip())
            embedding = _build_context_embedding(
                paragraph_position=paragraph_position,
                sentence_position=sentence_position,
                previous_type=previous_type,
                sentiment=sentiment,
                input_length_category=length_category
            )

            context_record = {
                'paragraph_position': paragraph_position,
                'sentence_position': sentence_position,
                'sentence_type': sentence_type,
                'sentiment': sentiment,
                'pos_template': pos_template,
                'embedding': embedding,
                'sentence': sentence
            }
            context_index.append(context_record)

    return context_index


def _find_similar_contexts(
    target_embedding: np.ndarray,
    context_index: List[Dict],
    top_k: int = 5
) -> List[Dict]:
    """Find similar contexts using cosine similarity.

    Args:
        target_embedding: The target context embedding to match.
        context_index: List of context records from sample text.
        top_k: Number of top matches to return.

    Returns:
        List of top-k most similar context records, sorted by similarity (highest first).
    """
    if not context_index:
        return []

    similarities = []
    for record in context_index:
        embedding = record['embedding']
        # Cosine similarity (embeddings are already normalized)
        similarity = np.dot(target_embedding, embedding)
        similarities.append((similarity, record))

    # Sort by similarity (descending)
    similarities.sort(key=lambda x: x[0], reverse=True)

    # Return top-k
    return [record for _, record in similarities[:top_k]]


def _build_statistical_template(
    similar_contexts: List[Dict],
    target_length: int
) -> List[str]:
    """Build a statistical template from similar contexts.

    Analyzes POS patterns from similar contexts and builds a composite template
    that matches the statistical profile.

    Args:
        similar_contexts: List of similar context records with POS templates.
        target_length: Target template length (based on input sentence length).

    Returns:
        A list of POS tags representing the composite template.
    """
    if not similar_contexts:
        # Fallback to default template
        return ['DT', 'JJ', 'NN', 'VBZ', 'IN', 'DT', 'NN']

    # Collect all POS templates
    all_templates = [ctx['pos_template'] for ctx in similar_contexts if ctx['pos_template']]

    if not all_templates:
        return ['DT', 'JJ', 'NN', 'VBZ', 'IN', 'DT', 'NN']

    # Find the most common length (or use target_length if close)
    lengths = [len(t) for t in all_templates]
    avg_length = int(np.mean(lengths))

    # Use target_length if it's within reasonable range of average
    if abs(target_length - avg_length) <= 5:
        template_length = target_length
    else:
        template_length = avg_length

    # Build template by position: for each position, find most common POS tag
    template = []
    max_len = max(len(t) for t in all_templates)

    for pos_idx in range(template_length):
        # Collect POS tags at this position from all templates
        pos_tags_at_position = []
        for t in all_templates:
            if pos_idx < len(t):
                pos_tags_at_position.append(t[pos_idx])

        if pos_tags_at_position:
            # Find most common POS tag at this position
            from collections import Counter
            most_common = Counter(pos_tags_at_position).most_common(1)[0][0]
            template.append(most_common)
        else:
            # If no data at this position, use a common default pattern
            if pos_idx == 0:
                template.append('DT')  # Determiner
            elif pos_idx == 1:
                template.append('JJ')  # Adjective
            elif pos_idx == 2:
                template.append('NN')  # Noun
            else:
                template.append('NN')  # Default to noun

    return template


def _normalize_template_length(template: List[str], target_length: int) -> List[str]:
    """Adjust template length to match target.

    Args:
        template: Original template.
        target_length: Desired length.

    Returns:
        Template adjusted to target length (truncated or padded).
    """
    if len(template) == target_length:
        return template
    elif len(template) > target_length:
        return template[:target_length]
    else:
        # Pad with common pattern
        padding = []
        common_pattern = ['DT', 'JJ', 'NN', 'VBZ', 'IN']
        while len(template) + len(padding) < target_length:
            padding.append(common_pattern[len(padding) % len(common_pattern)])
        return template + padding


def select_template_statistical(
    content_unit: ContentUnit,
    generated_text_so_far: List[str],
    sample_corpus: str,
    context_index: Optional[List[Dict]] = None
) -> List[str]:
    """Select a sentence template using statistical analysis and context similarity.

    This function:
    1. Builds a context embedding for the current position
    2. Finds similar contexts in sample text using embedding similarity
    3. Extracts POS templates from matching contexts
    4. Builds a composite template using statistical patterns

    Args:
        content_unit: ContentUnit with position metadata.
        generated_text_so_far: List of previously generated sentences.
        sample_corpus: The sample text corpus to draw templates from.
        context_index: Optional pre-computed context index. If None, will build it.

    Returns:
        A list of POS tags representing the target template structure.
    """
    # Build context index if not provided
    if context_index is None:
        context_index = _build_sample_context_index(sample_corpus)

    # Determine paragraph and sentence positions
    para_pos = _get_position_label(content_unit.paragraph_idx, content_unit.total_paragraphs)
    sent_pos = _get_position_label(content_unit.sentence_idx, content_unit.paragraph_length)

    # Determine previous sentence type
    if generated_text_so_far:
        previous_type = classify_sentence_type(generated_text_so_far[-1])
    else:
        previous_type = 'Simple'

    # Detect sentiment
    sentiment = _detect_sentiment(content_unit.original_text)

    # Categorize input length
    length_category = _categorize_length(content_unit.original_text)
    target_length = len(content_unit.original_text.split())

    # Build context embedding
    target_embedding = _build_context_embedding(
        paragraph_position=para_pos,
        sentence_position=sent_pos,
        previous_type=previous_type,
        sentiment=sentiment,
        input_length_category=length_category
    )

    # Find similar contexts
    similar_contexts = _find_similar_contexts(target_embedding, context_index, top_k=5)

    # Build statistical template
    if similar_contexts:
        template = _build_statistical_template(similar_contexts, target_length)
    else:
        # Fallback: use position-based heuristics
        if sent_pos == 'first':
            template = ['DT', 'JJ', 'NN', 'VBZ']  # Simple opening
        elif sent_pos == 'last':
            template = ['DT', 'NN', 'VBZ', 'IN', 'DT', 'NN']  # Slightly more complex closing
        else:
            template = ['DT', 'JJ', 'NN', 'VBZ', 'IN', 'DT', 'NN']  # Default

    # Normalize length
    template = _normalize_template_length(template, target_length)

    return template

