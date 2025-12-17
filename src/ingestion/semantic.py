"""Semantic parsing module for style transfer.

This module extracts semantic meaning from input text while stripping away
its original style, including Subject-Verb-Object triples, named entities,
and sentiment polarity.
"""

import nltk
from typing import List, Optional, Tuple
from src.models import ContentUnit

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt_tab', quiet=True)

try:
    nltk.data.find('taggers/averaged_perceptron_tagger_eng')
except LookupError:
    nltk.download('averaged_perceptron_tagger_eng', quiet=True)

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

try:
    nltk.data.find('sentiment/vader_lexicon')
except LookupError:
    nltk.download('vader_lexicon', quiet=True)


def _extract_svo(sentence: str) -> Optional[Tuple[str, str, str]]:
    """Extract Subject-Verb-Object triple from a sentence with full phrases.

    Enhanced to capture:
    - Full noun phrases (with modifiers)
    - Verb phrases
    - Prepositional phrases for objects

    Args:
        sentence: The sentence to parse.

    Returns:
        A tuple of (subject_phrase, verb_phrase, object_phrase) or None if extraction fails.
        Phrases may be empty strings if not found.
    """
    from nltk.tokenize import word_tokenize
    from nltk.tag import pos_tag

    tokens = word_tokenize(sentence)
    if not tokens:
        return None

    pos_tags = pos_tag(tokens)

    # Find the main verb (first verb that's not auxiliary)
    verb_idx = None
    verb = ""
    for idx, (word, tag) in enumerate(pos_tags):
        if tag.startswith('VB') and tag != 'VBD':  # Verb but not past tense (could be main verb)
            verb = word.lower()
            verb_idx = idx
            break

    if verb_idx is None:
        # Try to find any verb
        for idx, (word, tag) in enumerate(pos_tags):
            if tag.startswith('VB'):
                verb = word.lower()
                verb_idx = idx
                break

    if verb_idx is None:
        return None

    # Extract subject phrase (noun phrase before the verb)
    subject_words = []
    for idx in range(verb_idx - 1, -1, -1):
        word, tag = pos_tags[idx]
        # Include nouns, adjectives, determiners, possessive pronouns
        if tag.startswith('NN') or tag.startswith('JJ') or tag in ['DT', 'PRP$', 'CD']:
            subject_words.insert(0, word.lower())
        elif tag.startswith('PRP'):  # Pronoun
            subject_words.insert(0, word.lower())
            break
        elif tag in [',', '.', '!', '?', ';', ':']:  # Stop at punctuation
            break
        elif tag in ['IN', 'CC']:  # Stop at prepositions/conjunctions (start of new phrase)
            break

    subject_phrase = ' '.join(subject_words) if subject_words else ""

    # Extract verb phrase (verb + auxiliaries/adverbs)
    verb_words = [verb]
    # Check for adverbs before verb
    for idx in range(verb_idx - 1, max(0, verb_idx - 3), -1):
        word, tag = pos_tags[idx]
        if tag.startswith('RB'):  # Adverb
            verb_words.insert(0, word.lower())
        else:
            break

    verb_phrase = ' '.join(verb_words)

    # Extract object phrase (noun phrase after the verb, including prepositional phrases)
    obj_words = []
    skip_prep = False
    for idx in range(verb_idx + 1, len(pos_tags)):
        word, tag = pos_tags[idx]
        if tag.startswith('NN') or tag.startswith('JJ') or tag in ['DT', 'PRP$', 'CD']:
            obj_words.append(word.lower())
            skip_prep = False
        elif tag.startswith('PRP'):  # Pronoun
            obj_words.append(word.lower())
            break
        elif tag == 'IN':  # Preposition - include it and continue
            if not skip_prep:
                obj_words.append(word.lower())
            skip_prep = True
        elif tag in [',', '.', '!', '?', ';', ':']:  # Stop at punctuation
            break
        elif tag in ['CC'] and obj_words:  # Stop at conjunctions if we have object
            break
        elif skip_prep and tag not in ['DT', 'JJ', 'NN']:  # After prep, only continue for noun phrase
            if obj_words:
                break

    object_phrase = ' '.join(obj_words) if obj_words else ""

    return (subject_phrase, verb_phrase, object_phrase)


def _extract_content_words(sentence: str) -> List[str]:
    """Extract all important content words (nouns, verbs, adjectives) from a sentence.

    This helps preserve semantic meaning by capturing all key concepts,
    not just SVO structure.

    Args:
        sentence: The sentence to analyze.

    Returns:
        A list of content words (lowercased).
    """
    from nltk.tokenize import word_tokenize
    from nltk.tag import pos_tag
    from nltk.corpus import stopwords

    try:
        stop_words = set(stopwords.words('english'))
    except LookupError:
        stop_words = set()

    tokens = word_tokenize(sentence)
    if not tokens:
        return []

    pos_tags = pos_tag(tokens)
    content_words = []

    for word, tag in pos_tags:
        word_lower = word.lower()
        # Include nouns, verbs, adjectives, and important adverbs
        if (tag.startswith('NN') or tag.startswith('VB') or
            tag.startswith('JJ') or tag.startswith('RB')):
            if word_lower not in stop_words and len(word_lower) > 1:
                content_words.append(word_lower)

    return content_words


def _extract_entities(sentence: str) -> List[str]:
    """Extract named entities from a sentence.

    Args:
        sentence: The sentence to analyze.

    Returns:
        A list of named entity strings.
    """
    from nltk.tokenize import word_tokenize
    from nltk.tag import pos_tag
    from nltk.chunk import ne_chunk

    tokens = word_tokenize(sentence)
    if not tokens:
        return []

    pos_tags = pos_tag(tokens)

    # Use NLTK's named entity chunker
    try:
        tree = ne_chunk(pos_tags, binary=False)
        entities = []
        for subtree in tree:
            if isinstance(subtree, nltk.Tree):
                entity_text = ' '.join([token for token, _ in subtree.leaves()])
                entities.append(entity_text)
        return entities
    except LookupError:
        # If ne_chunk data is not available, use a simple heuristic
        # Look for capitalized words that are proper nouns
        entities = []
        for word, tag in pos_tags:
            if tag == 'NNP' and word[0].isupper() and len(word) > 1:
                entities.append(word)
        return entities


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
        # Fallback: simple keyword-based sentiment
        positive_words = ['good', 'great', 'excellent', 'wonderful', 'happy', 'love', 'like']
        negative_words = ['bad', 'terrible', 'awful', 'hate', 'sad', 'angry', 'dislike']

        sentence_lower = sentence.lower()
        pos_count = sum(1 for word in positive_words if word in sentence_lower)
        neg_count = sum(1 for word in negative_words if word in sentence_lower)

        if pos_count > neg_count:
            return 'Positive'
        elif neg_count > pos_count:
            return 'Negative'
        else:
            return 'Neutral'


def extract_meaning(input_text: str) -> List[ContentUnit]:
    """Extract semantic meaning from input text.

    Parses the input text paragraph by paragraph, sentence by sentence, extracting:
    - Subject-Verb-Object triples
    - Named entities
    - Sentiment polarity
    - Position metadata (paragraph and sentence positions)

    Args:
        input_text: The input text to analyze.

    Returns:
        A list of ContentUnit objects, one per sentence, with position metadata.
    """
    from nltk.tokenize import sent_tokenize

    # Split into paragraphs (try double newlines first, then single newlines)
    paragraphs = [p.strip() for p in input_text.split('\n\n') if p.strip()]

    # If no double newlines, try single newlines
    if len(paragraphs) == 1 and '\n' in input_text:
        paragraphs = [p.strip() for p in input_text.split('\n') if p.strip()]

    # If still only one, treat entire text as one paragraph
    if not paragraphs:
        paragraphs = [input_text.strip()] if input_text.strip() else []

    total_paragraphs = len(paragraphs)
    content_units = []

    for para_idx, paragraph in enumerate(paragraphs):
        # Tokenize paragraph into sentences
        sentences = sent_tokenize(paragraph)
        paragraph_length = len([s for s in sentences if s.strip()])

        for sent_idx, sentence in enumerate(sentences):
            sentence = sentence.strip()
            if not sentence:
                continue

            # Extract SVO triple
            svo_triple = _extract_svo(sentence)
            svo_triples = [svo_triple] if svo_triple else []

            # Extract named entities
            entities = _extract_entities(sentence)

            # Extract content words
            content_words = _extract_content_words(sentence)

            # Determine position metadata
            is_first_paragraph = (para_idx == 0)
            is_last_paragraph = (para_idx == total_paragraphs - 1)
            is_first_sentence = (sent_idx == 0)
            is_last_sentence = (sent_idx == paragraph_length - 1)

            # Create ContentUnit with position metadata
            content_unit = ContentUnit(
                svo_triples=svo_triples,
                entities=entities,
                original_text=sentence,
                content_words=content_words,
                paragraph_idx=para_idx,
                sentence_idx=sent_idx,
                is_first_paragraph=is_first_paragraph,
                is_last_paragraph=is_last_paragraph,
                is_first_sentence=is_first_sentence,
                is_last_sentence=is_last_sentence,
                total_paragraphs=total_paragraphs,
                paragraph_length=paragraph_length
            )
            content_units.append(content_unit)

    return content_units

