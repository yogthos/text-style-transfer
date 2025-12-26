"""Style profiler for generating author style profiles using LLM."""

from typing import Optional, List

from ..models import AuthorProfile, StyleProfile
from ..llm.provider import LLMProvider
from ..llm.session import LLMSession
from ..utils.logging import get_logger
from .loader import Corpus, CorpusDocument
from .analyzer import StatisticalAnalyzer, FeatureVector
from .style_extractor import StyleFeatures

logger = get_logger(__name__)

# Prompts for style DNA extraction
STYLE_DNA_SYSTEM_PROMPT = """You are a literary analyst specializing in authorial style analysis. Your task is to analyze writing samples and produce a precise, actionable description of an author's unique writing style.

Focus on:
1. Sentence structure and rhythm patterns
2. Vocabulary preferences and word choices
3. Use of punctuation and formatting
4. Rhetorical devices and argumentation style
5. Tone and voice characteristics
6. How the author builds and connects ideas

Be specific and avoid generic descriptions. Focus on distinctive patterns that could be used to replicate the style."""

STYLE_DNA_USER_PROMPT = """Analyze the following writing samples from {author_name} and produce a style DNA description.

The description should be 2-3 paragraphs that capture the essence of this author's writing style in a way that could guide rewriting text to match this style.

Writing samples:
---
{samples}
---

Statistical observations about this author's writing:
- Average sentence length: {avg_sentence_length:.1f} words
- Sentence length variation (burstiness): {burstiness:.2f}
- Perspective: {perspective}
- Notable punctuation patterns: {punctuation_patterns}

Provide your style DNA description:"""


class StyleProfiler:
    """Generates author style profiles using LLM and statistical analysis.

    Combines:
    - LLM-generated style DNA (qualitative description)
    - Statistical features (sentence length, burstiness, etc.)
    - Top vocabulary extraction
    """

    def __init__(
        self,
        llm_provider: LLMProvider,
        analyzer: Optional[StatisticalAnalyzer] = None,
        max_sample_chars: int = 8000,
        num_samples: int = 5
    ):
        """Initialize style profiler.

        Args:
            llm_provider: LLM provider for style DNA generation.
            analyzer: Statistical analyzer instance.
            max_sample_chars: Maximum characters to include in LLM prompt.
            num_samples: Number of sample paragraphs to include.
        """
        self.llm_provider = llm_provider
        self.analyzer = analyzer or StatisticalAnalyzer()
        self.max_sample_chars = max_sample_chars
        self.num_samples = num_samples

    def profile_corpus(self, corpus: Corpus) -> AuthorProfile:
        """Generate an author profile from a corpus.

        Args:
            corpus: Corpus of author documents.

        Returns:
            AuthorProfile with style DNA and statistical features.
        """
        if not corpus.documents:
            raise ValueError("Corpus is empty")

        author_name = corpus.author or corpus.documents[0].author

        logger.info(f"Generating style profile for {author_name}")

        # Step 1: Collect all text and compute aggregate features
        all_text = self._collect_text(corpus)
        features = self.analyzer.analyze_text(all_text)

        # Step 2: Extract enhanced style features (transitions, openers, etc.)
        style_features = StyleFeatures.extract_from_text(all_text)

        # Step 3: Select representative samples for LLM
        samples = self._select_samples(corpus)

        # Step 4: Generate style DNA using LLM
        style_dna = self._generate_style_dna(author_name, samples, features)

        # Step 5: Build author profile with all features
        profile = AuthorProfile(
            name=author_name,
            style_dna=style_dna,
            top_vocab=features.top_words,
            avg_sentence_length=features.avg_sentence_length,
            burstiness=features.burstiness,
            punctuation_freq=features.punctuation_freq,
            perspective=features.perspective,
            # Enhanced style features from corpus analysis
            transitions=style_features.transitions,
            voice_ratio=style_features.voice_ratio,
            sentence_openers=style_features.openers,
            signature_phrases=style_features.phrases.get("signature_phrases", []),
            punctuation_patterns=style_features.punctuation
        )

        logger.info(
            f"Generated profile for {author_name}: "
            f"avg_len={features.avg_sentence_length:.1f}, "
            f"burstiness={features.burstiness:.2f}, "
            f"voice_ratio={style_features.voice_ratio:.2f}"
        )

        return profile

    def profile_document(self, doc: CorpusDocument) -> AuthorProfile:
        """Generate an author profile from a single document.

        Args:
            doc: Single corpus document.

        Returns:
            AuthorProfile with style DNA and statistical features.
        """
        logger.info(f"Generating style profile from document: {doc.filename}")

        # Compute features
        features = self.analyzer.analyze_document(doc.processed)

        # Select samples from this document
        samples = self._select_samples_from_doc(doc)

        # Generate style DNA
        style_dna = self._generate_style_dna(doc.author, samples, features)

        return AuthorProfile(
            name=doc.author,
            style_dna=style_dna,
            top_vocab=features.top_words,
            avg_sentence_length=features.avg_sentence_length,
            burstiness=features.burstiness,
            punctuation_freq=features.punctuation_freq,
            perspective=features.perspective
        )

    def profile_text(self, text: str, author_name: str = "unknown") -> AuthorProfile:
        """Generate an author profile from raw text.

        Args:
            text: Text to analyze.
            author_name: Name for the author.

        Returns:
            AuthorProfile with style DNA and statistical features.
        """
        logger.info(f"Generating style profile from text for {author_name}")

        # Compute features
        features = self.analyzer.analyze_text(text)

        # Use text directly as sample (truncated)
        samples = text[:self.max_sample_chars]

        # Generate style DNA
        style_dna = self._generate_style_dna(author_name, samples, features)

        return AuthorProfile(
            name=author_name,
            style_dna=style_dna,
            top_vocab=features.top_words,
            avg_sentence_length=features.avg_sentence_length,
            burstiness=features.burstiness,
            punctuation_freq=features.punctuation_freq,
            perspective=features.perspective
        )

    def _collect_text(self, corpus: Corpus) -> str:
        """Collect all text from corpus documents.

        Args:
            corpus: Corpus to collect from.

        Returns:
            Concatenated text from all documents.
        """
        texts = []
        for doc in corpus.documents:
            texts.append(doc.processed.cleaned_text)
        return "\n\n".join(texts)

    def _select_samples(self, corpus: Corpus) -> str:
        """Select representative samples from corpus.

        Tries to get diverse samples from different positions in documents.

        Args:
            corpus: Corpus to select from.

        Returns:
            Concatenated sample text.
        """
        samples = []
        total_chars = 0

        # Get paragraphs from all documents
        all_paragraphs = []
        for doc in corpus.documents:
            for para in doc.processed.paragraphs:
                all_paragraphs.append(para.text)

        if not all_paragraphs:
            return ""

        # Select evenly spaced paragraphs
        step = max(1, len(all_paragraphs) // self.num_samples)
        for i in range(0, len(all_paragraphs), step):
            if len(samples) >= self.num_samples:
                break
            para = all_paragraphs[i]
            if total_chars + len(para) <= self.max_sample_chars:
                samples.append(para)
                total_chars += len(para)

        return "\n\n".join(samples)

    def _select_samples_from_doc(self, doc: CorpusDocument) -> str:
        """Select samples from a single document.

        Args:
            doc: Document to select from.

        Returns:
            Concatenated sample text.
        """
        samples = []
        total_chars = 0

        paragraphs = [p.text for p in doc.processed.paragraphs]

        if not paragraphs:
            return ""

        # Select evenly spaced paragraphs
        step = max(1, len(paragraphs) // self.num_samples)
        for i in range(0, len(paragraphs), step):
            if len(samples) >= self.num_samples:
                break
            para = paragraphs[i]
            if total_chars + len(para) <= self.max_sample_chars:
                samples.append(para)
                total_chars += len(para)

        return "\n\n".join(samples)

    def _generate_style_dna(
        self,
        author_name: str,
        samples: str,
        features: FeatureVector
    ) -> str:
        """Generate style DNA using LLM.

        Args:
            author_name: Name of the author.
            samples: Sample text to analyze.
            features: Statistical features.

        Returns:
            Style DNA description.
        """
        # Format punctuation patterns for prompt
        punct_patterns = self._format_punctuation_patterns(features.punctuation_freq)

        # Build user prompt
        user_prompt = STYLE_DNA_USER_PROMPT.format(
            author_name=author_name,
            samples=samples,
            avg_sentence_length=features.avg_sentence_length,
            burstiness=features.burstiness,
            perspective=features.perspective,
            punctuation_patterns=punct_patterns
        )

        # Call LLM
        try:
            style_dna = self.llm_provider.call(
                system_prompt=STYLE_DNA_SYSTEM_PROMPT,
                user_prompt=user_prompt,
                temperature=0.3  # Lower temperature for consistent analysis
            )
            return style_dna.strip()

        except Exception as e:
            logger.error(f"Failed to generate style DNA: {e}")
            # Return a basic fallback
            return self._generate_fallback_dna(author_name, features)

    def _format_punctuation_patterns(self, punct_freq: dict) -> str:
        """Format punctuation frequencies for display.

        Args:
            punct_freq: Dictionary of punctuation frequencies.

        Returns:
            Human-readable string.
        """
        if not punct_freq:
            return "no notable patterns"

        # Sort by frequency
        sorted_punct = sorted(
            punct_freq.items(),
            key=lambda x: x[1],
            reverse=True
        )

        # Take top 3 non-zero
        notable = [(name, freq) for name, freq in sorted_punct if freq > 0.1][:3]

        if not notable:
            return "minimal distinctive punctuation"

        parts = [f"{name} ({freq:.1f}/1000 chars)" for name, freq in notable]
        return ", ".join(parts)

    def _generate_fallback_dna(
        self,
        author_name: str,
        features: FeatureVector
    ) -> str:
        """Generate fallback style DNA when LLM fails.

        Args:
            author_name: Author name.
            features: Statistical features.

        Returns:
            Basic style description.
        """
        length_desc = "short" if features.avg_sentence_length < 12 else \
                      "medium" if features.avg_sentence_length < 20 else "long"

        rhythm_desc = "uniform" if features.burstiness < 0.2 else \
                      "varied" if features.burstiness < 0.5 else "highly variable"

        return (
            f"{author_name} writes with {length_desc} sentences "
            f"(avg {features.avg_sentence_length:.1f} words) and {rhythm_desc} rhythm "
            f"(burstiness: {features.burstiness:.2f}). "
            f"Primary perspective: {features.perspective}."
        )


def create_style_profile(
    author_profile: AuthorProfile,
    secondary_author: Optional[AuthorProfile] = None,
    blend_ratio: float = 1.0
) -> StyleProfile:
    """Create a StyleProfile from an AuthorProfile.

    Args:
        author_profile: Primary author profile.
        secondary_author: Optional secondary author for blending.
        blend_ratio: Blend ratio (1.0 = 100% primary).

    Returns:
        StyleProfile instance.
    """
    return StyleProfile(
        primary_author=author_profile,
        secondary_author=secondary_author,
        blend_ratio=blend_ratio
    )
