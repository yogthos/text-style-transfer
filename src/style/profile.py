"""Data-driven author style profiles.

All values are extracted from corpus - NO hardcoding.
Based on stylometry research (Burrows' Delta, Markov chains).
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import json


@dataclass
class SentenceLengthProfile:
    """Sentence length distribution extracted from corpus.

    Uses Order-1 Markov chain for length transitions (research shows
    bigram is sufficient, higher orders become deterministic).
    """

    mean: float
    std: float
    min_length: int
    max_length: int

    # Percentiles for sampling
    percentiles: Dict[int, int] = field(default_factory=dict)
    # e.g., {10: 8, 25: 12, 50: 18, 75: 25, 90: 35}

    burstiness: float = 0.0  # coefficient of variation (std/mean)
    short_ratio: float = 0.0  # % sentences < 10 words
    long_ratio: float = 0.0  # % sentences > 30 words

    # Order-1 Markov transition matrix
    # P(next_length_category | current_length_category)
    length_transitions: Dict[str, Dict[str, float]] = field(default_factory=dict)
    # e.g., {"short": {"short": 0.2, "medium": 0.5, "long": 0.3}, ...}

    def get_length_category(self, length: int) -> str:
        """Categorize a sentence length."""
        p25 = self.percentiles.get(25, 12)
        p75 = self.percentiles.get(75, 25)

        if length < p25:
            return "short"
        elif length > p75:
            return "long"
        else:
            return "medium"

    def to_dict(self) -> Dict:
        """Serialize to dictionary."""
        return {
            "mean": self.mean,
            "std": self.std,
            "min_length": self.min_length,
            "max_length": self.max_length,
            "percentiles": self.percentiles,
            "burstiness": self.burstiness,
            "short_ratio": self.short_ratio,
            "long_ratio": self.long_ratio,
            "length_transitions": self.length_transitions,
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "SentenceLengthProfile":
        """Deserialize from dictionary."""
        return cls(**data)


@dataclass
class TransitionProfile:
    """Transition word usage patterns extracted from corpus.

    Key insight: authors differ in HOW OFTEN they use transitions,
    not just which ones. The no_transition_ratio is critical.
    """

    # Core metrics
    no_transition_ratio: float = 0.0  # % sentences with NO connector
    start_position_ratio: float = 0.0  # % transitions at sentence start
    transition_per_sentence: float = 0.0  # avg transitions per sentence

    # Per-category word frequencies (from actual corpus counts)
    causal: Dict[str, float] = field(default_factory=dict)
    # e.g., {"thus": 0.4, "therefore": 0.3, "hence": 0.2}

    adversative: Dict[str, float] = field(default_factory=dict)
    # e.g., {"however": 0.5, "but": 0.3, "yet": 0.2}

    additive: Dict[str, float] = field(default_factory=dict)
    # e.g., {"also": 0.4, "moreover": 0.3, "furthermore": 0.2}

    temporal: Dict[str, float] = field(default_factory=dict)
    # e.g., {"then": 0.5, "finally": 0.3}

    def get_all_transitions(self) -> Dict[str, float]:
        """Get all transitions with their frequencies."""
        all_trans = {}
        for category in [self.causal, self.adversative, self.additive, self.temporal]:
            all_trans.update(category)
        return all_trans

    def to_dict(self) -> Dict:
        """Serialize to dictionary."""
        return {
            "no_transition_ratio": self.no_transition_ratio,
            "start_position_ratio": self.start_position_ratio,
            "transition_per_sentence": self.transition_per_sentence,
            "causal": self.causal,
            "adversative": self.adversative,
            "additive": self.additive,
            "temporal": self.temporal,
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "TransitionProfile":
        """Deserialize from dictionary."""
        return cls(**data)


@dataclass
class RegisterProfile:
    """Register/formality features extracted from corpus.

    These are transferable style features (not topic-specific).
    """

    formality_score: float = 0.5  # 0.0 = informal, 1.0 = formal
    narrative_ratio: float = 0.0  # % narrative vs argumentative
    question_frequency: float = 0.0  # questions per paragraph
    imperative_frequency: float = 0.0  # commands per paragraph
    passive_voice_ratio: float = 0.0  # % passive constructions

    # Punctuation patterns
    semicolon_per_sentence: float = 0.0
    colon_per_sentence: float = 0.0
    dash_per_sentence: float = 0.0
    parenthetical_per_sentence: float = 0.0

    def to_dict(self) -> Dict:
        """Serialize to dictionary."""
        return {
            "formality_score": self.formality_score,
            "narrative_ratio": self.narrative_ratio,
            "question_frequency": self.question_frequency,
            "imperative_frequency": self.imperative_frequency,
            "passive_voice_ratio": self.passive_voice_ratio,
            "semicolon_per_sentence": self.semicolon_per_sentence,
            "colon_per_sentence": self.colon_per_sentence,
            "dash_per_sentence": self.dash_per_sentence,
            "parenthetical_per_sentence": self.parenthetical_per_sentence,
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "RegisterProfile":
        """Deserialize from dictionary."""
        return cls(**data)


@dataclass
class DiscourseRelationProfile:
    """Discourse relation patterns extracted from corpus using spaCy.

    Tracks PDTB-style discourse relations:
    - CONTRAST: Opposition/concession (but, however, although)
    - CAUSE: Causal relations (because, so, therefore)
    - ELABORATION: Expansion/detail (for example, specifically)
    - TEMPORAL: Time sequences (then, after, before)
    - CONTINUATION: Simple continuation (and, also)
    """

    # Relation distribution
    relation_distribution: Dict[str, float] = field(default_factory=dict)

    # Discourse relation transition Markov model
    # P(next_relation | current_relation)
    relation_transitions: Dict[str, Dict[str, float]] = field(default_factory=dict)

    # Sample sentence pairs for each relation type (for few-shot prompting)
    relation_samples: Dict[str, List[Tuple[str, str]]] = field(default_factory=dict)

    # Connectives used for each relation type (extracted from corpus)
    relation_connectives: Dict[str, Dict[str, float]] = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return {
            "relation_distribution": self.relation_distribution,
            "relation_transitions": self.relation_transitions,
            "relation_samples": {k: list(v) for k, v in self.relation_samples.items()},
            "relation_connectives": self.relation_connectives,
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "DiscourseRelationProfile":
        samples = data.get("relation_samples", {})
        # Convert lists back to tuples
        samples = {k: [tuple(v) if isinstance(v, list) else v for v in vals]
                   for k, vals in samples.items()}
        return cls(
            relation_distribution=data.get("relation_distribution", {}),
            relation_transitions=data.get("relation_transitions", {}),
            relation_samples=samples,
            relation_connectives=data.get("relation_connectives", {}),
        )


@dataclass
class SentenceStructureProfile:
    """Sentence structure patterns extracted from corpus using spaCy.

    Classifies sentences by syntactic complexity and builds a Markov model
    of structure transitions. Also stores sample sentences for each type.
    """

    # Structure type distribution (simple, compound, complex, compound-complex)
    structure_distribution: Dict[str, float] = field(default_factory=dict)
    # e.g., {"simple": 0.3, "compound": 0.25, "complex": 0.35, "compound_complex": 0.1}

    # Structure transition Markov model
    structure_transitions: Dict[str, Dict[str, float]] = field(default_factory=dict)
    # e.g., {"simple": {"simple": 0.2, "compound": 0.3, ...}, ...}

    # Proposition capacity per structure type (how many clauses/ideas fit)
    proposition_capacity: Dict[str, int] = field(default_factory=lambda: {
        "simple": 1, "compound": 2, "complex": 2, "compound_complex": 3
    })

    # Sample sentences for each structure type (sanitized with [X] placeholders)
    structure_samples: Dict[str, List[str]] = field(default_factory=dict)
    # e.g., {"simple": ["The [X] sat.", "Birds fly."], ...}

    # Raw sample sentences without sanitization (for style prompts)
    raw_samples: Dict[str, List[str]] = field(default_factory=dict)
    # e.g., {"simple": ["The cat sat on the mat.", "Birds fly south."], ...}

    def to_dict(self) -> Dict:
        return {
            "structure_distribution": self.structure_distribution,
            "structure_transitions": self.structure_transitions,
            "proposition_capacity": self.proposition_capacity,
            "structure_samples": self.structure_samples,
            "raw_samples": self.raw_samples,
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "SentenceStructureProfile":
        # Handle backward compatibility for profiles without raw_samples
        if "raw_samples" not in data:
            data["raw_samples"] = data.get("structure_samples", {})
        return cls(**data)


@dataclass
class DeltaProfile:
    """Burrows' Delta configuration.

    Research shows 200-300 most frequent words give best attribution.
    Features above rank 100 have minimal discriminative power.
    """

    # Top 300 most frequent words with z-score normalized frequencies
    mfw_frequencies: Dict[str, float] = field(default_factory=dict)
    mfw_zscores: Dict[str, float] = field(default_factory=dict)

    # Corpus-level statistics for z-score calculation
    corpus_mean: Dict[str, float] = field(default_factory=dict)
    corpus_std: Dict[str, float] = field(default_factory=dict)

    def calculate_delta(self, text_frequencies: Dict[str, float]) -> float:
        """Calculate Burrows' Delta distance to this profile.

        Args:
            text_frequencies: Word frequencies from text to compare.

        Returns:
            Delta score (lower = more similar).
        """
        if not self.mfw_zscores:
            return float("inf")

        delta_sum = 0.0
        count = 0

        for word, zscore in self.mfw_zscores.items():
            text_freq = text_frequencies.get(word, 0.0)

            # Z-score normalize the text frequency
            mean = self.corpus_mean.get(word, 0.0)
            std = self.corpus_std.get(word, 1.0)
            if std > 0:
                text_zscore = (text_freq - mean) / std
            else:
                text_zscore = 0.0

            delta_sum += abs(zscore - text_zscore)
            count += 1

        return delta_sum / count if count > 0 else float("inf")

    def to_dict(self) -> Dict:
        """Serialize to dictionary."""
        return {
            "mfw_frequencies": self.mfw_frequencies,
            "mfw_zscores": self.mfw_zscores,
            "corpus_mean": self.corpus_mean,
            "corpus_std": self.corpus_std,
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "DeltaProfile":
        """Deserialize from dictionary."""
        return cls(**data)


@dataclass
class AuthorStyleProfile:
    """Complete author style profile.

    All values extracted from corpus analysis - nothing hardcoded.
    """

    author_name: str
    corpus_word_count: int
    corpus_sentence_count: int

    # Component profiles
    length_profile: SentenceLengthProfile = field(
        default_factory=lambda: SentenceLengthProfile(
            mean=20.0, std=10.0, min_length=3, max_length=60
        )
    )
    transition_profile: TransitionProfile = field(default_factory=TransitionProfile)
    register_profile: RegisterProfile = field(default_factory=RegisterProfile)
    delta_profile: DeltaProfile = field(default_factory=DeltaProfile)
    structure_profile: SentenceStructureProfile = field(default_factory=SentenceStructureProfile)
    discourse_profile: DiscourseRelationProfile = field(default_factory=DiscourseRelationProfile)

    # Human writing patterns (for humanization)
    human_patterns: Dict = field(default_factory=dict)
    # Contains: fragments, questions, asides, dash_patterns, short_sentences, etc.

    # Style DNA (generated description)
    style_dna: str = ""

    def to_dict(self) -> Dict:
        """Serialize to dictionary."""
        return {
            "author_name": self.author_name,
            "corpus_word_count": self.corpus_word_count,
            "corpus_sentence_count": self.corpus_sentence_count,
            "length_profile": self.length_profile.to_dict(),
            "transition_profile": self.transition_profile.to_dict(),
            "register_profile": self.register_profile.to_dict(),
            "delta_profile": self.delta_profile.to_dict(),
            "structure_profile": self.structure_profile.to_dict(),
            "discourse_profile": self.discourse_profile.to_dict(),
            "human_patterns": self.human_patterns,
            "style_dna": self.style_dna,
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "AuthorStyleProfile":
        """Deserialize from dictionary."""
        return cls(
            author_name=data["author_name"],
            corpus_word_count=data["corpus_word_count"],
            corpus_sentence_count=data["corpus_sentence_count"],
            length_profile=SentenceLengthProfile.from_dict(data["length_profile"]),
            transition_profile=TransitionProfile.from_dict(data["transition_profile"]),
            register_profile=RegisterProfile.from_dict(data["register_profile"]),
            delta_profile=DeltaProfile.from_dict(data["delta_profile"]),
            structure_profile=SentenceStructureProfile.from_dict(data.get("structure_profile", {})),
            discourse_profile=DiscourseRelationProfile.from_dict(data.get("discourse_profile", {})),
            human_patterns=data.get("human_patterns", {}),
            style_dna=data.get("style_dna", ""),
        )

    def save(self, path: str) -> None:
        """Save profile to JSON file."""
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: str) -> "AuthorStyleProfile":
        """Load profile from JSON file."""
        with open(path, "r") as f:
            data = json.load(f)
        return cls.from_dict(data)
