"""Semantic parsing module for style transfer.

This module extracts semantic meaning from input text while stripping away
its original style, including Subject-Verb-Object triples, named entities,
and sentiment polarity.
"""

import nltk
from typing import List, Optional, Tuple, Set
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
        pos_count = sum(1 for word in positive_words if word in sentence_lower)
        neg_count = sum(1 for word in negative_words if word in sentence_lower)

        if pos_count > neg_count:
            return 'Positive'
        elif neg_count > pos_count:
            return 'Negative'
        else:
            return 'Neutral'


def extract_keywords(text: str) -> List[str]:
    """Extract keywords (noun/verb lemmas) from text using spaCy or NLTK.

    Args:
        text: Text to extract keywords from.

    Returns:
        List of keyword lemmas (lowercase, no duplicates).
    """
    if not text:
        return []

    # Try to use spaCy for better lemmatization
    try:
        from src.utils.nlp_manager import NLPManager
        nlp = NLPManager.get_nlp()
        doc = nlp(text)
        keywords = []
        for token in doc:
            # Extract nouns and verbs as keywords
            if token.pos_ in ['NOUN', 'VERB'] and not token.is_stop:
                lemma = token.lemma_.lower()
                if lemma and len(lemma) > 2:  # Filter very short words
                    keywords.append(lemma)
        return list(set(keywords))  # Remove duplicates
    except (ImportError, OSError):
        # Fallback: use NLTK
        try:
            from nltk.tokenize import word_tokenize
            from nltk.tag import pos_tag
            from nltk.stem import WordNetLemmatizer
            from nltk.corpus import stopwords

            try:
                stop_words = set(stopwords.words('english'))
            except LookupError:
                nltk.download('stopwords', quiet=True)
                stop_words = set(stopwords.words('english'))

            lemmatizer = WordNetLemmatizer()
            tokens = word_tokenize(text.lower())
            pos_tags = pos_tag(tokens)
            keywords = []
            for word, pos in pos_tags:
                if pos.startswith(('NN', 'VB')):  # Nouns and verbs
                    if word.lower() not in stop_words and len(word) > 2:
                        # Lemmatize based on POS
                        if pos.startswith('NN'):
                            lemma = lemmatizer.lemmatize(word, pos='n')
                        else:
                            lemma = lemmatizer.lemmatize(word, pos='v')
                        if lemma and len(lemma) > 2:
                            keywords.append(lemma)
            return list(set(keywords))  # Remove duplicates
        except (ImportError, LookupError):
            # If NLTK is not available, return empty list
            return []


def get_wordnet_synonyms(word: str) -> Set[str]:
    """Get WordNet synonyms for a word (for flexible matching).

    Args:
        word: Word to get synonyms for.

    Returns:
        Set of synonym lemmas (including the word itself).
    """
    try:
        from nltk.corpus import wordnet as wn
        synonyms = set([word.lower()])
        for syn in wn.synsets(word.lower()):
            for lemma in syn.lemmas():
                synonyms.add(lemma.name().lower().replace('_', ' '))
        return synonyms
    except (ImportError, LookupError):
        # Try to download wordnet if not available
        try:
            nltk.download('wordnet', quiet=True)
            from nltk.corpus import wordnet as wn
            synonyms = set([word.lower()])
            for syn in wn.synsets(word.lower()):
                for lemma in syn.lemmas():
                    synonyms.add(lemma.name().lower().replace('_', ' '))
            return synonyms
        except (ImportError, LookupError):
            return {word.lower()}


def extract_critical_nouns(text: str) -> List[Tuple[str, str]]:
    """Extract critical nouns from text with their types.

    Returns list of (noun_lemma, noun_type) tuples where noun_type is:
    - "PROPER": Proper noun (capitalized, mid-sentence)
    - "ABSTRACT": Abstract/concept noun (e.g., "finitude", "experience", "paradox")
    - "CONCRETE": Concrete noun (e.g., "bottle", "room", "star")

    Args:
        text: Text to extract nouns from.

    Returns:
        List of (noun_lemma, noun_type) tuples.
    """
    if not text:
        return []

    # Try to use spaCy for better noun extraction
    try:
        from src.utils.nlp_manager import NLPManager
        nlp = NLPManager.get_nlp()
        doc = nlp(text)
        critical_nouns = []

        for token in doc:
            if token.pos_ == 'NOUN' and not token.is_stop:
                lemma = token.lemma_.lower()
                if lemma and len(lemma) > 2:
                    # Determine noun type using spaCy's linguistic features
                    noun_type = "CONCRETE"
                    # Check if proper noun (capitalized mid-sentence)
                    if token.is_upper or (token.text[0].isupper() and token.i > 0):
                        # Capitalized mid-sentence = proper noun
                        noun_type = "PROPER"
                    elif token.has_vector:
                        # Use semantic properties to identify abstract nouns
                        # Abstract nouns tend to have specific semantic properties
                        # Check similarity to known abstract concept vectors
                        abstract_score = 0.0
                        try:
                            # Abstract nouns are often less concrete, have higher semantic centrality
                            # Use word frequency and vector properties
                            if hasattr(token, 'prob'):
                                # Very common words are often abstract concepts
                                freq_score = -token.prob if token.prob < 0 else 0.0
                                abstract_score += freq_score * 0.5

                            # Check vector magnitude (abstract concepts often have different vector properties)
                            # Calculate magnitude without numpy dependency
                            vector_magnitude = sum(x * x for x in token.vector) ** 0.5
                            abstract_score += min(vector_magnitude / 10.0, 1.0) * 0.5

                            # If score suggests abstractness, classify as ABSTRACT
                            if abstract_score > 0.3:
                                noun_type = "ABSTRACT"
                        except (AttributeError, RuntimeError):
                            pass

                    critical_nouns.append((lemma, noun_type))

        return list(set(critical_nouns))  # Remove duplicates
    except (ImportError, OSError):
        # Fallback: use NLTK
        try:
            from nltk.tokenize import word_tokenize
            from nltk.tag import pos_tag
            from nltk.stem import WordNetLemmatizer
            from nltk.corpus import stopwords

            try:
                stop_words = set(stopwords.words('english'))
            except LookupError:
                nltk.download('stopwords', quiet=True)
                stop_words = set(stopwords.words('english'))

            lemmatizer = WordNetLemmatizer()
            tokens = word_tokenize(text)
            pos_tags = pos_tag(tokens)
            critical_nouns = []

            for idx, (word, pos) in enumerate(pos_tags):
                if pos.startswith('NN') and word.lower() not in stop_words:
                    if len(word) > 2:
                        # Lemmatize
                        lemma = lemmatizer.lemmatize(word.lower(), pos='n')
                        if lemma and len(lemma) > 2:
                            # Determine noun type
                            noun_type = "CONCRETE"
                            # Check if proper noun (NNP or capitalized mid-sentence)
                            if pos == 'NNP' or (word[0].isupper() and idx > 0):
                                noun_type = "PROPER"
                            # For NLTK fallback, we can't easily determine abstractness without spaCy
                            # So we'll mark as CONCRETE and let spaCy path handle abstract detection

                            critical_nouns.append((lemma, noun_type))

            return list(set(critical_nouns))  # Remove duplicates
        except (ImportError, LookupError):
            # If NLTK is not available, return empty list
            return []


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

