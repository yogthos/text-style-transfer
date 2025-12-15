import requests
import json
import torch
import math
import sys
import warnings
import re
import os
from collections import Counter
from pathlib import Path
from transformers import GPT2LMHeadModel, GPT2TokenizerFast
from glm import GLMProvider
from deepseek import DeepSeekProvider
from markov import StyleTransferAgent, MetaphorDetector

# Suppress the loss_type warning
warnings.filterwarnings("ignore", message=".*loss_type.*")

class AlgorithmicSentenceManipulator:
    """
    Algorithmically manipulate sentences to evade GPTZero detection.
    Key insight: AI detection works on HOW text is generated, not just WHAT it says.
    By doing manipulation algorithmically (not via AI), we avoid introducing AI patterns.

    IMPORTANT: Avoid adding formal transitions like "In fact," "Indeed," "Moreover,"
    as these are flagged as AI patterns by detectors.

    This class uses LEARNED PATTERNS from the sample text (prompts/sample.txt) to guide
    transformations. The sample text passes GPTZero, so we mirror its patterns.
    """

    def __init__(self, learned_patterns=None):
        # Conjunctions for merging sentences - simple, natural ones only
        self.merge_conjunctions = ['; ', ', and ', ', but ']
        # NO TRANSITIONS - GPTZero flags "In fact," "Indeed," "Moreover," etc. as AI patterns
        # Instead, we rely on sentence merging and clause reordering only
        self.transitions = []  # Empty - don't add any transitions
        # Clause reordering markers
        self.subordinators = ['when', 'while', 'as', 'since', 'because', 'although', 'though', 'if']

        # Learned patterns from sample text
        self.learned_patterns = learned_patterns or {}

        # Extract specific patterns if available
        self.target_burstiness = self.learned_patterns.get('burstiness', {'short': 0.2, 'medium': 0.5, 'long': 0.3})
        self.target_openers = self.learned_patterns.get('sentence_openers', {})
        self.target_lengths_by_pos = self.learned_patterns.get('sentence_lengths_by_position', [])
        self.punctuation_patterns = self.learned_patterns.get('punctuation', {})

    def merge_short_sentences(self, paragraph):
        """
        Algorithmically merge consecutive short sentences (<12 words) using semicolons and conjunctions.
        This creates natural burstiness without AI intervention.
        """
        sentences = self._split_sentences(paragraph)
        if len(sentences) < 2:
            return paragraph

        result = []
        i = 0
        merge_idx = 0  # Cycle through different merge styles

        while i < len(sentences):
            sent = sentences[i].strip()
            sent_words = len(sent.split())

            # Check if this and next sentence are both short
            if i + 1 < len(sentences):
                next_sent = sentences[i + 1].strip()
                next_words = len(next_sent.split())

                # Merge if both are short (< 12 words) or if creating better variation
                if sent_words < 12 and next_words < 15:
                    # Lowercase next sentence start, but preserve "I" (first person pronoun)
                    if next_sent and next_sent[0] == 'I' and (len(next_sent) == 1 or not next_sent[1].isalpha()):
                        # Keep "I" uppercase
                        next_start = next_sent
                    else:
                        next_start = next_sent[0].lower() + next_sent[1:] if next_sent else ""

                    # AVOID semicolons - sample has only 0.70 per 100 words
                    # Use commas and "and" more often (matching sample style)
                    if merge_idx % 4 == 0:
                        # Use comma + and for compound structure
                        merged = f"{sent}, and {next_start}"
                    else:
                        # Keep as separate sentences (most natural for sample style)
                        cap_next = next_start[0].upper() + next_start[1:] if next_start else ''
                        merged = f"{sent}. {cap_next}"

                    result.append(merged)
                    merge_idx += 1
                    i += 2
                    continue

            result.append(sent)
            i += 1

        return '. '.join(result) + '.'

    def vary_sentence_starters(self, paragraph):
        """
        Algorithmically vary sentence starters using LEARNED PATTERNS from sample text.

        1. Uses common sentence openers from the sample (self.target_openers)
        2. Moves prepositional phrases to the front
        3. Inverts clause order

        NOTE: We do NOT add formal transition words like "In fact," "Indeed," etc.
        as these are flagged as AI patterns by detectors.
        """
        sentences = self._split_sentences(paragraph)
        if len(sentences) < 3:
            return paragraph

        # Get top sentence openers from the sample text (these pass GPTZero)
        sample_openers = list(self.target_openers.keys()) if self.target_openers else []

        # Track starters to avoid repetition
        starters_used = []
        result = []

        for i, sent in enumerate(sentences):
            sent = sent.strip()
            if not sent:
                continue

            words = sent.split()
            starter = words[0].lower() if words else ''

            # If starter was used in last 2 sentences, try to change it
            if starter in starters_used[-2:] and len(words) > 5:
                # Try to move a prepositional phrase to front
                modified = self._move_phrase_to_front(sent)
                if modified != sent:
                    result.append(modified)
                    starters_used.append(modified.split()[0].lower())
                    continue

                # Try to invert clause order if there's a comma
                # BUT: Don't invert if second clause starts with a pronoun (would lose referent)
                if ',' in sent:
                    parts = sent.split(',', 1)
                    second_part = parts[1].strip() if len(parts) > 1 else ''
                    second_first_word = second_part.split()[0].lower() if second_part.split() else ''

                    # Pronouns that would lose their referent if inverted
                    pronouns = {'it', 'he', 'she', 'they', 'this', 'that', 'these', 'those', 'its', 'their'}

                    if len(parts) == 2 and len(second_part.split()) > 3 and second_first_word not in pronouns:
                        # Move second part to front (safe - no pronoun referent issues)
                        inverted = second_part.capitalize() + ', ' + parts[0].lower()
                        result.append(inverted)
                        starters_used.append(inverted.split()[0].lower())
                        continue

            result.append(sent)
            starters_used.append(starter)

        return '. '.join(result) + '.'

    def _move_phrase_to_front(self, sentence):
        """Move a prepositional phrase from the end to the beginning."""
        # Look for phrases like "in the X", "from the Y", "through the Z"
        prep_patterns = [
            r'^(.+?)\s+(in the \w+)\.?$',
            r'^(.+?)\s+(from the \w+)\.?$',
            r'^(.+?)\s+(through the \w+)\.?$',
            r'^(.+?)\s+(during the \w+)\.?$',
            r'^(.+?)\s+(within the \w+)\.?$',
        ]

        for pattern in prep_patterns:
            match = re.match(pattern, sentence, re.IGNORECASE)
            if match:
                main_clause = match.group(1)
                prep_phrase = match.group(2)
                # Move prep phrase to front
                return f"{prep_phrase.capitalize()}, {main_clause[0].lower() + main_clause[1:]}"

        return sentence

    def add_parenthetical_asides(self, paragraph):
        """
        Add parenthetical asides to break up the rhythm.
        Human writers naturally include asides; AI tends not to.

        NOTE: Keep asides simple and rare. Avoid formal ones like "to be sure"
        which can be flagged as AI patterns.
        """
        sentences = self._split_sentences(paragraph)
        result = []

        # Simple, casual asides only - avoid formal ones
        asides = [', though,', ', really,', ', I think,', ', at least,']

        aside_idx = 0
        for i, sent in enumerate(sentences):
            sent = sent.strip()
            if not sent:
                continue

            words = sent.split()
            # Add aside to long sentences (>18 words) very occasionally
            if len(words) > 18 and i % 5 == 3 and aside_idx < len(asides):
                # Insert aside after the subject (roughly after 4-6 words)
                insert_pos = min(5, len(words) // 3)
                words.insert(insert_pos, asides[aside_idx])
                aside_idx += 1
                result.append(' '.join(words).replace('  ', ' '))
            else:
                result.append(sent)

        return '. '.join(result) + '.'

    def create_burstiness(self, paragraph):
        """
        Create burstiness by matching the LEARNED sentence length distribution from sample text.

        Uses self.target_burstiness which is learned from the sample that passes GPTZero.
        """
        sentences = self._split_sentences(paragraph)
        if len(sentences) < 4:
            return paragraph

        lengths = [len(s.split()) for s in sentences]
        total = len(lengths)

        # Calculate current distribution
        current_short = sum(1 for l in lengths if l <= 10) / total if total else 0
        current_medium = sum(1 for l in lengths if 11 <= l <= 25) / total if total else 0
        current_long = sum(1 for l in lengths if l > 25) / total if total else 0

        # Get target distribution from learned patterns (defaults if not learned)
        target_short = self.target_burstiness.get('short', 0.2)
        target_medium = self.target_burstiness.get('medium', 0.5)
        target_long = self.target_burstiness.get('long', 0.3)

        # Determine what adjustments are needed
        need_more_short = current_short < target_short - 0.1
        need_more_long = current_long < target_long - 0.1
        need_less_medium = current_medium > target_medium + 0.15

        result = []
        merged_one = False

        for i, sent in enumerate(sentences):
            sent = sent.strip()
            if not sent:
                continue

            words = sent.split()

            # If we need more long sentences and have medium ones, try merging
            if need_more_long and not merged_one and i + 1 < len(sentences):
                next_sent = sentences[i + 1].strip()
                if 11 <= len(words) <= 25 and 11 <= len(next_sent.split()) <= 25:
                    # Merge two medium sentences into one long one
                    merged = f"{sent}; {next_sent[0].lower() + next_sent[1:]}"
                    result.append(merged)
                    merged_one = True
                    sentences[i + 1] = ''  # Mark as used
                    continue

            result.append(sent)

        return '. '.join([s for s in result if s]) + '.'

    def process_paragraph(self, paragraph, genre='mixed'):
        """
        Apply all algorithmic transformations to a paragraph.

        Uses LEARNED PATTERNS from sample text to guide transformations.
        The goal is to match the sample's sentence length distribution and
        punctuation patterns, which pass GPTZero.
        """
        # Step 1: Merge short sentences (aggressive merging to match sample's long sentences)
        # Sample has 45% long sentences - we need to create more long sentences
        result = self.merge_short_sentences(paragraph)

        # Step 2: Create burstiness using learned distribution
        result = self.create_burstiness(result)

        # Step 3: Vary sentence starters (but avoid AI-flagged transitions)
        result = self.vary_sentence_starters(result)

        # Step 4: Add semicolons to match sample's punctuation pattern
        # Sample has 0.19 semicolons per sentence - add more semicolons
        result = self._add_semicolons_like_sample(result)

        # Step 5: Clean up
        result = self._cleanup_punctuation(result)

        return result.strip()

    def _add_semicolons_like_sample(self, paragraph):
        """
        Add semicolons to match the sample text's punctuation pattern.

        DISABLED: Sample has only 0.70 semicolons per 100 words.
        For a 100-word paragraph, that's less than 1 semicolon.
        It's better to NOT add semicolons algorithmically.
        """
        # DO NOT add semicolons - the sample text uses very few
        # 0.70 per 100 words means most paragraphs should have ZERO
        return paragraph

    def _cleanup_punctuation(self, text):
        """Clean up punctuation artifacts from algorithmic manipulation."""
        # Fix period followed by punctuation (e.g., ".,")
        text = re.sub(r'\.,', ',', text)
        text = re.sub(r'\.;', ';', text)
        text = re.sub(r'\.\s+,', ',', text)
        text = re.sub(r'\.\s+;', ';', text)

        # Remove double periods
        text = re.sub(r'\.+', '.', text)

        # Fix spacing around punctuation
        text = re.sub(r'\s+\.', '.', text)  # Remove space before period
        text = re.sub(r'\s+,', ',', text)  # Remove space before comma
        text = re.sub(r'\s+;', ';', text)  # Remove space before semicolon

        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text)

        # Ensure proper spacing after punctuation
        text = re.sub(r'\.([A-Z])', r'. \1', text)
        text = re.sub(r',([a-zA-Z])', r', \1', text)
        text = re.sub(r';([a-zA-Z])', r'; \1', text)

        # Fix lowercase 'i' that should be 'I' (only standalone)
        text = re.sub(r'\bi\b', 'I', text)

        # Fix double punctuation
        text = re.sub(r'[,;]\s*[,;]', ';', text)
        text = re.sub(r'[,;]\s*\.', '.', text)

        # Fix cases like ". ;" or " .;"
        text = re.sub(r'\.\s*;', ';', text)
        text = re.sub(r'\.\s*,', ',', text)

        # Clean up any remaining oddities
        text = re.sub(r';\s*\.', '.', text)
        text = re.sub(r',\s*\.', '.', text)

        # Ensure sentence starts with capital
        sentences = re.split(r'(?<=[.!?])\s+', text)
        result = []
        for sent in sentences:
            if sent:
                sent = sent[0].upper() + sent[1:] if len(sent) > 1 else sent.upper()
                result.append(sent)

        return ' '.join(result)

    def _split_sentences(self, text):
        """Split text into sentences, preserving structure."""
        # Split on period, exclamation, question mark followed by space and capital
        sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', text)
        return [s.strip() for s in sentences if s.strip()]


class GPTZeroMetricOptimizer:
    """Optimize text to evade GPTZero detection by targeting specific metrics."""

    def __init__(self, analyzer):
        self.analyzer = analyzer
        self.sentence_manipulator = AlgorithmicSentenceManipulator()

    def calculate_metrics(self, text):
        """Calculate GPTZero detection metrics."""
        sentences = self._split_into_sentences(text)
        if not sentences:
            return {}

        lengths = [len(s.split()) for s in sentences]
        avg_length = sum(lengths) / len(lengths) if lengths else 0
        variance = sum((l - avg_length) ** 2 for l in lengths) / len(lengths) if lengths else 0

        # Get sentence starters
        starters = []
        for sent in sentences:
            words = sent.split()
            if words:
                starter = words[0].lower().rstrip('.,;:')
                starters.append(starter)

        unique_starters = len(set(starters)) / len(starters) if starters else 0

        # Count sentence types (simple, compound, complex)
        # Simple heuristic: count commas and conjunctions
        simple = sum(1 for s in sentences if ',' not in s and 'and' not in s.lower() and 'but' not in s.lower())
        compound = sum(1 for s in sentences if ',' in s and ('and' in s.lower() or 'but' in s.lower()))
        complex_count = len(sentences) - simple - compound

        return {
            'sentence_length_variance': variance,
            'unique_starter_ratio': unique_starters,
            'sentence_type_mix': {
                'simple': simple / len(sentences) if sentences else 0,
                'compound': compound / len(sentences) if sentences else 0,
                'complex': complex_count / len(sentences) if sentences else 0
            },
            'avg_sentence_length': avg_length
        }

    def _split_into_sentences(self, text):
        """Split text into sentences."""
        sentences = re.split(r'[.!?]+', text)
        return [s.strip() for s in sentences if s.strip()]

    def apply_algorithmic_fixes(self, text):
        """Apply algorithmic sentence manipulation (no AI involved)."""
        return self.sentence_manipulator.process_paragraph(text)

    def validate_metrics(self, text, target_metrics=None):
        """
        Validate if text passes GPTZero detection thresholds.

        Returns:
            (is_valid: bool, issues: list)
        """
        metrics = self.calculate_metrics(text)
        issues = []

        # Check sentence-length variance (should be > 30)
        if metrics['sentence_length_variance'] < 30:
            issues.append(f"Low sentence-length variance: {metrics['sentence_length_variance']:.1f} (target: >30)")

        # Check unique-starter ratio (should be > 40%)
        if metrics['unique_starter_ratio'] < 0.4:
            issues.append(f"Low unique-starter ratio: {metrics['unique_starter_ratio']:.1%} (target: >40%)")

        # Check sentence type mix (should have variety)
        type_mix = metrics['sentence_type_mix']
        if type_mix['simple'] > 0.7 or type_mix['compound'] > 0.7 or type_mix['complex'] > 0.7:
            issues.append(f"Limited sentence type variety: {type_mix}")

        return len(issues) == 0, issues

    def optimize_metrics(self, text, target_metrics=None):
        """
        Apply targeted fixes to optimize GPTZero metrics.
        Returns guidance string for LLM (not direct modification).
        """
        metrics = self.calculate_metrics(text)
        guidance = []

        if metrics['sentence_length_variance'] < 30:
            guidance.append("Vary sentence lengths more: mix short (5-10 words), medium (15-25 words), and long (30+ words) sentences")

        if metrics['unique_starter_ratio'] < 0.4:
            guidance.append("Vary sentence beginnings: avoid starting multiple sentences with the same word")

        type_mix = metrics['sentence_type_mix']
        if type_mix['simple'] > 0.7:
            guidance.append("Add more compound and complex sentences (use commas, conjunctions, subordinate clauses)")
        elif type_mix['compound'] > 0.7:
            guidance.append("Add more simple and complex sentences for variety")
        elif type_mix['complex'] > 0.7:
            guidance.append("Add more simple and compound sentences for variety")

        return "; ".join(guidance) if guidance else ""

class AgenticHumanizer:
    def __init__(self, config_path: str = None):
        """
        Initialize the humanizer with configuration.

        Args:
            config_path: Path to config.json file (default: config.json in project root)
        """
        # Load configuration
        if config_path is None:
            config_path = Path(__file__).parent / "config.json"
        else:
            config_path = Path(config_path)

        with open(config_path, "r", encoding="utf-8") as f:
            self.config = json.load(f)

        self.provider = self.config.get("provider", "ollama")

        # Initialize provider-specific settings
        if self.provider == "ollama":
            ollama_config = self.config.get("ollama", {})
            self.editor_model = ollama_config.get("editor_model", "qwen3:32b")
            self.critic_model = ollama_config.get("critic_model", "deepseek-r1:8b")
            self.ollama_url = ollama_config.get("url", "http://localhost:11434/api/generate")
            self.glm_provider = None
            self.deepseek_provider = None
        elif self.provider == "glm":
            glm_config = self.config.get("glm", {})
            self.editor_model = glm_config.get("editor_model", "glm-4.6")
            self.critic_model = glm_config.get("critic_model", "glm-4.6")
            # Get API key from config or environment variable
            api_key = glm_config.get("api_key") or os.getenv("GLM_API_KEY")
            api_url = glm_config.get("api_url", "https://api.z.ai/api/paas/v4/chat/completions")
            self.glm_provider = GLMProvider(api_key, api_url)
            self.ollama_url = None
            self.deepseek_provider = None
        elif self.provider == "deepseek":
            deepseek_config = self.config.get("deepseek", {})
            self.editor_model = deepseek_config.get("editor_model", "deepseek-chat")
            self.critic_model = deepseek_config.get("critic_model", "deepseek-chat")
            # Get API key from config or environment variable
            api_key = deepseek_config.get("api_key") or os.getenv("DEEPSEEK_API_KEY")
            api_url = deepseek_config.get("api_url", "https://api.deepseek.com/v1/chat/completions")
            self.deepseek_provider = DeepSeekProvider(api_key, api_url)
            self.ollama_url = None
            self.glm_provider = None
        else:
            raise ValueError(f"Unknown provider: {self.provider}. Must be 'ollama', 'glm', or 'deepseek'")

        print(f"Using provider: {self.provider}")
        print(f"Editor model: {self.editor_model}")
        print(f"Critic model: {self.critic_model}")

        # Load prompts from files
        prompts_dir = Path(__file__).parent / "prompts"
        with open(prompts_dir / "editor.md", "r", encoding="utf-8") as f:
            self.editor_system = f.read().strip()
        with open(prompts_dir / "system.md", "r", encoding="utf-8") as f:
            self.critic_system = f.read().strip()
        with open(prompts_dir / "structural_analyst.md", "r", encoding="utf-8") as f:
            self.structural_analyst_prompt = f.read().strip()
        with open(prompts_dir / "structural_editor.md", "r", encoding="utf-8") as f:
            self.structural_editor_prompt = f.read().strip()
        with open(prompts_dir / "paragraph_rewrite.md", "r", encoding="utf-8") as f:
            self.paragraph_rewrite_template = f.read().strip()
        # Load style sample for Few-Shot Style Transfer
        with open(prompts_dir / "sample.txt", "r", encoding="utf-8") as f:
            self.style_sample = f.read().strip()

        # Cache for style guide (extracted once, reused)
        self._style_guide = None

        # Cache for sample patterns (extracted once, reused)
        self._sample_patterns = None

        # Load Perplexity Scorer (Math Check)
        print("Loading Scoring Model...")
        self.scorer_id = "gpt2-large"
        self.tokenizer = GPT2TokenizerFast.from_pretrained(self.scorer_id)
        self.model = GPT2LMHeadModel.from_pretrained(self.scorer_id)
        # Set max sequence length for the model
        self.max_length = self.model.config.max_position_embeddings
        # Use a safe chunk size (slightly less than max to avoid edge cases)
        self.chunk_size = min(1024, self.max_length - 10)

        # Device selection: prioritize MPS (Metal) on macOS, then CUDA, then CPU
        if torch.backends.mps.is_available():
            self.device = "mps"
            print("Using Metal (MPS) acceleration on macOS")
        elif torch.cuda.is_available():
            self.device = "cuda"
            print("Using CUDA acceleration")
        else:
            self.device = "cpu"
            print("Using CPU (no GPU acceleration available)")

        self.model.to(self.device)

        # Initialize Markov chain style transfer agent
        self.markov_agent = StyleTransferAgent(config_path)

        # Initialize metaphor detector for AI-typical flourish detection
        self.metaphor_detector = MetaphorDetector(nlp=self.markov_agent.analyzer.nlp)

        # Load metaphor simplification prompt
        simplify_prompt_path = Path(__file__).parent / "prompts" / "simplify_metaphor.md"
        if simplify_prompt_path.exists():
            with open(simplify_prompt_path, 'r') as f:
                self.simplify_metaphor_prompt = f.read()
        else:
            self.simplify_metaphor_prompt = None

        # Check if database exists, train if needed
        db_path = Path("style_brain.db")
        # Check if database exists and has been trained
        needs_training = False
        if not db_path.exists():
            needs_training = True
        else:
            # Check if database has states (has been trained)
            try:
                import sqlite3
                conn = sqlite3.connect(str(db_path))
                cursor = conn.cursor()
                cursor.execute("SELECT COUNT(*) FROM states")
                count = cursor.fetchone()[0]
                conn.close()
                if count == 0:
                    needs_training = True
            except:
                needs_training = True

        if needs_training:
            print("Training Markov chain database...")
            sample_path = Path(__file__).parent / "prompts" / "sample.txt"
            self.markov_agent.learn_style(sample_path)

        # Initialize state tracking for Markov chain
        self.current_markov_signature = None
        self.gptzero_optimizer = GPTZeroMetricOptimizer(self.markov_agent.analyzer)

    def _chunk_text(self, text, chunk_size):
        """Split text into chunks that fit within the model's context window."""
        # Tokenize the entire text to get accurate token positions
        tokens = self.tokenizer.encode(text, add_special_tokens=False)

        if len(tokens) <= chunk_size:
            return [text]

        chunks = []
        # Split into non-overlapping chunks
        for i in range(0, len(tokens), chunk_size):
            chunk_tokens = tokens[i:i + chunk_size]
            if not chunk_tokens:
                break
            # Decode back to text
            chunk_text = self.tokenizer.decode(chunk_tokens, skip_special_tokens=True)
            chunks.append(chunk_text)

        return chunks if chunks else [text]

    def calculate_perplexity(self, text):
        """Calculate perplexity for text, handling long texts by chunking."""
        # Handle empty text
        if not text or not text.strip():
            return 1.0  # Return low perplexity for empty text

        # Check token count without creating tensors first
        tokens = self.tokenizer.encode(text, add_special_tokens=False)
        input_length = len(tokens)

        # Handle empty token sequence
        if input_length == 0:
            return 1.0  # Return low perplexity for text with no tokens

        if input_length <= self.chunk_size:
            # Text fits in one chunk
            input_ids = torch.tensor([tokens]).to(self.device)
            with torch.no_grad():
                outputs = self.model(input_ids, labels=input_ids)
                loss = outputs.loss
            return math.exp(loss.item())
        else:
            # Text is too long, split into chunks
            chunks = self._chunk_text(text, self.chunk_size)
            print(f"  Text too long ({input_length} tokens), splitting into {len(chunks)} chunks...")
            perplexities = []

            for i, chunk in enumerate(chunks):
                if not chunk.strip():
                    continue
                chunk_tokens = self.tokenizer.encode(chunk, add_special_tokens=False)
                # Skip empty token sequences
                if len(chunk_tokens) == 0:
                    continue
                # Ensure chunk doesn't exceed max length
                if len(chunk_tokens) > self.chunk_size:
                    chunk_tokens = chunk_tokens[:self.chunk_size]
                input_ids = torch.tensor([chunk_tokens]).to(self.device)
                with torch.no_grad():
                    outputs = self.model(input_ids, labels=input_ids)
                    loss = outputs.loss
                ppl = math.exp(loss.item())
                perplexities.append(ppl)

            # Return average perplexity across chunks
            avg_ppl = sum(perplexities) / len(perplexities) if perplexities else 0.0
            return avg_ppl

    def call_model(self, model, prompt, system_prompt=""):
        """
        Call the configured model provider (Ollama or GLM).

        Args:
            model: Model name
            prompt: User prompt
            system_prompt: System prompt (optional)

        Returns:
            Response text from the model
        """
        if self.provider == "ollama":
            return self._call_ollama(model, prompt, system_prompt)
        elif self.provider == "glm":
            # Lower temperature for strict meaning preservation (variance comes from Style Guide)
            return self.glm_provider.call(model, prompt, system_prompt, temperature=0.5, top_p=0.85)
        elif self.provider == "deepseek":
            # Lower temperature for strict meaning preservation (variance comes from Style Guide)
            return self.deepseek_provider.call(model, prompt, system_prompt, temperature=0.5, top_p=0.85)
        else:
            raise ValueError(f"Unknown provider: {self.provider}")

    def _call_ollama(self, model, prompt, system_prompt=""):
        """Internal method to call Ollama API."""
        payload = {
            "model": model,
            "prompt": prompt,
            "system": system_prompt,
            "stream": False,
            # Lower temperature for strict meaning preservation (variance comes from Style Guide)
            "options": {"temperature": 0.5, "top_p": 0.85, "num_ctx": 8192}
        }
        # Add timeout to prevent hanging (120 seconds for local Ollama)
        response = requests.post(self.ollama_url, json=payload, timeout=120)
        return response.json()['response']

    def _get_sample_paragraphs(self):
        """Extract paragraphs from the style sample for few-shot examples."""
        paragraphs = [p.strip() for p in self.style_sample.split('\n\n') if p.strip()]
        # Filter to substantial paragraphs (at least 50 words)
        substantial = [p for p in paragraphs if len(p.split()) >= 50]
        return substantial

    def _select_example_paragraphs(self, input_para, num_examples=2):
        """Select example paragraphs from sample that are similar in length to input."""
        sample_paras = self._get_sample_paragraphs()
        if not sample_paras:
            return []

        input_length = len(input_para.split())

        # Sort by similarity in length to input paragraph
        sorted_paras = sorted(sample_paras, key=lambda p: abs(len(p.split()) - input_length))

        # Return the closest matches
        return sorted_paras[:num_examples]

    def _is_special_content(self, para):
        """Check if paragraph is special content that should be preserved exactly."""
        para_stripped = para.strip()

        # HTML tags (div, span, etc.)
        if re.search(r'<[^>]+>', para):
            return True

        # Code blocks
        if para_stripped.startswith('```') or para_stripped.startswith('~~~'):
            return True

        # Bullet points or numbered lists (poem-like structures)
        if re.match(r'^[\*\-\+]\s', para_stripped) or re.match(r'^\d+\.\s', para_stripped):
            return True

        # Multiple lines starting with * (verse/poem)
        lines = para.split('\n')
        if len(lines) > 1 and all(line.strip().startswith('*') for line in lines if line.strip()):
            return True

        # Block quotes
        if para_stripped.startswith('>'):
            return True

        return False

    def _calculate_word_overlap(self, original, rewritten):
        """Calculate the percentage of original words preserved in rewritten text."""
        # Normalize: lowercase, remove punctuation for comparison
        def normalize_words(text):
            # Remove punctuation except apostrophes in contractions
            text = re.sub(r'[^\w\s\']', ' ', text.lower())
            words = text.split()
            # Filter out very short words (punctuation artifacts)
            return set(w for w in words if len(w) > 1)

        orig_words = normalize_words(original)
        new_words = normalize_words(rewritten)

        if not orig_words:
            return 1.0

        # Calculate overlap
        preserved = orig_words & new_words
        overlap = len(preserved) / len(orig_words)

        return overlap

    def _check_vocabulary_repetition(self, text, max_per_paragraph=4, max_percentage=0.05):
        """
        Check for excessive vocabulary repetition that triggers GPTZero detection.

        Only flags truly problematic repetition - words that appear multiple times
        AND indicate vocabulary injection from the sample text.

        Args:
            text: Text to check
            max_per_paragraph: Max occurrences of a non-stopword per paragraph (default: 4)
            max_percentage: Max percentage of total words for any single word (default: 5%)

        Returns:
            (is_valid: bool, issues: list, repeated_words: dict)
        """
        # Common stopwords that are OK to repeat
        stopwords = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
            'of', 'with', 'by', 'from', 'as', 'is', 'was', 'are', 'were', 'been',
            'be', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would',
            'could', 'should', 'may', 'might', 'must', 'shall', 'can', 'need',
            'that', 'which', 'who', 'whom', 'whose', 'this', 'these', 'those',
            'it', 'its', 'itself', 'they', 'them', 'their', 'we', 'us', 'our',
            'i', 'me', 'my', 'you', 'your', 'he', 'him', 'his', 'she', 'her',
            'not', 'no', 'nor', 'so', 'if', 'when', 'then', 'than', 'also',
            'just', 'only', 'very', 'too', 'more', 'most', 'such', 'even',
            'all', 'any', 'each', 'every', 'both', 'few', 'many', 'much',
            'some', 'other', 'another', 'same', 'into', 'through', 'during',
            'before', 'after', 'above', 'below', 'between', 'under', 'over',
            'again', 'further', 'once', 'here', 'there', 'where', 'why', 'how',
            'what', 'own', 'about', 'out', 'up', 'down', 'off', 'being'
        }

        issues = []
        repeated_words = {}

        # Normalize text
        text_lower = re.sub(r'[^\w\s\']', ' ', text.lower())
        words = [w for w in text_lower.split() if len(w) > 2]  # Min 3 chars
        total_words = len(words)

        if total_words == 0:
            return True, [], {}

        # Count word frequencies
        word_counts = Counter(words)

        # Known problematic words from sample text that signal vocabulary injection
        # These are philosophical/academic terms from the Mao sample that shouldn't appear in narrative
        # Also include words that trigger AI detection due to overuse
        injected_vocab_signals = {
            'process', 'development', 'contradiction', 'principal', 'aspect', 'concrete',
            'revolutionary', 'condition', 'mechanism', 'framework', 'paradigm', 'dialectic',
            'materialist', 'synthesis', 'thesis', 'antithesis', 'bourgeoisie', 'proletariat',
            'unity', 'fundamental', 'phenomenon', 'phenomena', 'dynamics', 'dynamic'
        }

        # Check for injected vocabulary first - these are always bad
        for word in injected_vocab_signals:
            if word in word_counts and word_counts[word] >= 2:
                count = word_counts[word]
                issues.append(f"Injected vocabulary '{word}' appears {count}x - this word likely comes from sample text")
                repeated_words[word] = count

        # Check for excessive repetition of non-stopwords (only very high thresholds)
        for word, count in word_counts.most_common(20):  # Only check top 20
            if word in stopwords:
                continue
            if word in injected_vocab_signals:
                continue  # Already checked above

            # Only flag if VERY repetitive (4+ times, 5%+ of text)
            percentage = count / total_words
            if count >= 4 and percentage > max_percentage:
                issues.append(f"Word '{word}' appears {count}x ({percentage:.1%} of text) - excessive repetition")
                repeated_words[word] = count

        is_valid = len(issues) == 0
        return is_valid, issues, repeated_words

    def _validate_perplexity(self, original, output, tolerance=0.5):
        """
        Validate that output perplexity is in acceptable range relative to original.

        GPTZero uses perplexity as a key metric - AI text tends to have LOWER perplexity
        (more predictable). Human text has HIGHER perplexity (more surprising).

        Args:
            original: Original input text
            output: Generated output text
            tolerance: How much lower perplexity is acceptable (0.5 = 50% lower is OK)

        Returns:
            (is_valid: bool, original_ppl: float, output_ppl: float, feedback: str)
        """
        try:
            orig_ppl = self.calculate_perplexity(original)
            out_ppl = self.calculate_perplexity(output)

            # If output perplexity is significantly lower than original, it's too predictable
            # This indicates the model is making the text MORE AI-like
            min_acceptable = orig_ppl * tolerance

            # Also check against sample text perplexity baseline (human text typically 50-150)
            # If our sample is available, use its perplexity as reference
            if not hasattr(self, '_sample_perplexity'):
                self._sample_perplexity = self.calculate_perplexity(self.style_sample[:5000])

            is_valid = out_ppl >= min_acceptable

            if not is_valid:
                feedback = f"Output too predictable: perplexity {out_ppl:.1f} vs original {orig_ppl:.1f} (min: {min_acceptable:.1f})"
            elif out_ppl < 30:
                # Very low perplexity is always suspicious
                is_valid = False
                feedback = f"Output perplexity extremely low ({out_ppl:.1f}) - highly detectable as AI"
            else:
                feedback = f"Perplexity OK: output {out_ppl:.1f}, original {orig_ppl:.1f}, sample {self._sample_perplexity:.1f}"

            return is_valid, orig_ppl, out_ppl, feedback

        except Exception as e:
            # If perplexity calculation fails, don't block the pipeline
            return True, 0.0, 0.0, f"Perplexity validation skipped: {e}"

    def _identify_ai_flagged_sentences(self, text):
        """
        Identify specific sentences that have AI detection patterns.
        Returns sentences that need modification vs those that are fine.

        AI patterns detected:
        1. Uniform sentence length (all sentences similar word count)
        2. Repetitive sentence starters
        3. Empty filler words ("Crucially", "Significantly", "Importantly")
        4. Passive hedging ("It is important to note...")

        Returns:
            dict with 'flagged' (list of sentences needing change) and 'ok' (list of fine sentences)
        """
        sentences = self._split_into_sentences(text)
        if len(sentences) < 2:
            return {'flagged': [], 'ok': sentences, 'issues': {}}

        flagged = []
        ok = []
        issues = {}  # sentence -> list of issues

        # AI filler words/phrases to flag
        ai_fillers = [
            'crucially', 'significantly', 'importantly', 'notably', 'interestingly',
            'it is important to note', 'it should be noted', 'it is worth noting',
            'it goes without saying', 'needless to say', 'in conclusion',
            'to summarize', 'in essence', 'fundamentally', 'essentially'
        ]

        # Forbidden vocabulary that signals AI
        forbidden_words = {
            'delve', 'leverage', 'seamless', 'tapestry', 'crucial', 'vibrant',
            'realm', 'landscape', 'symphony', 'orchestrate', 'myriad', 'plethora',
            'paradigm', 'synergy', 'holistic', 'robust', 'streamline', 'optimize',
            'cutting-edge', 'game-changing', 'groundbreaking', 'revolutionary'
        }

        # Get sentence lengths for uniformity check
        lengths = [len(s.split()) for s in sentences]
        avg_length = sum(lengths) / len(lengths)

        # Get starters for repetition check
        starters = [s.split()[0].lower() if s.split() else '' for s in sentences]
        starter_counts = Counter(starters)

        for i, sent in enumerate(sentences):
            sent_issues = []
            sent_lower = sent.lower()

            # Check for AI filler words
            for filler in ai_fillers:
                if filler in sent_lower:
                    sent_issues.append(f"AI filler: '{filler}'")

            # Check for forbidden vocabulary
            for word in forbidden_words:
                if word in sent_lower:
                    sent_issues.append(f"Forbidden word: '{word}'")

            # Check for passive hedging
            if re.search(r'\bit (is|was|would be) (important|worth|notable|necessary)', sent_lower):
                sent_issues.append("Passive hedging detected")

            # Check for uniform length - only flag if VERY uniform (5+ similar sentences in a row)
            # This is a strong signal of AI, but 3 similar is normal for human text
            sent_length = lengths[i]
            # Count consecutive similar-length sentences
            consecutive_similar = 0
            for j in range(max(0, i-2), min(len(lengths), i+3)):
                if abs(lengths[j] - sent_length) <= 3:
                    consecutive_similar += 1
            if consecutive_similar >= 5:  # Need 5+ similar sentences - stronger threshold
                sent_issues.append(f"Uniform length ({sent_length} words, {consecutive_similar} consecutive similar)")

            # Check for repetitive starter - only flag if 4+ (3 is common in human text)
            if starters[i] and starter_counts[starters[i]] >= 4:
                sent_issues.append(f"Repetitive starter: '{starters[i]}' ({starter_counts[starters[i]]}x)")

            if sent_issues:
                flagged.append(sent)
                issues[sent] = sent_issues
            else:
                ok.append(sent)

        return {'flagged': flagged, 'ok': ok, 'issues': issues}

    def _detect_genre(self, text):
        """
        Detect the genre of text to apply appropriate processing.

        Returns:
            str: 'narrative', 'academic', 'technical', or 'mixed'
        """
        text_lower = text.lower()

        # Narrative indicators
        narrative_indicators = [
            r'\bi\s+(was|am|had|have|went|saw|felt|thought|remember)\b',  # First person
            r'\bmy\s+(childhood|life|father|mother|family|story)\b',
            r'\bmemory|memories\b',
            r'\bi\s+\w+ed\b',  # Past tense first person
            r'\b(one day|years ago|when i was|growing up)\b'
        ]
        narrative_score = sum(1 for pattern in narrative_indicators if re.search(pattern, text_lower))

        # Academic indicators
        academic_indicators = [
            r'\[\^?\d+\]',  # Citations
            r'\b(study|research|analysis|hypothesis|theory|evidence)\b',
            r'\b(according to|findings suggest|data shows|literature)\b',
            r'\b(furthermore|moreover|consequently|therefore)\b',
            r'\b(abstract|conclusion|methodology|references)\b'
        ]
        academic_score = sum(1 for pattern in academic_indicators if re.search(pattern, text_lower))

        # Technical indicators
        technical_indicators = [
            r'```',  # Code blocks
            r'\b(function|class|method|variable|parameter|return)\b',
            r'\b(api|http|json|xml|sql|html|css)\b',
            r'\b(install|configure|deploy|execute|compile)\b',
            r'[A-Z][a-zA-Z]+[A-Z]',  # CamelCase
        ]
        technical_score = sum(1 for pattern in technical_indicators if re.search(pattern, text_lower))

        # Determine genre
        scores = {
            'narrative': narrative_score,
            'academic': academic_score,
            'technical': technical_score
        }

        max_score = max(scores.values())
        if max_score == 0:
            return 'mixed'

        # Get genre with highest score
        genre = max(scores.keys(), key=lambda k: scores[k])

        # If scores are close, it's mixed
        sorted_scores = sorted(scores.values(), reverse=True)
        if len(sorted_scores) > 1 and sorted_scores[0] - sorted_scores[1] <= 1:
            return 'mixed'

        return genre

    def _get_sample_patterns(self):
        """Get cached sample patterns from the Markov agent."""
        try:
            # Try to get from analyzer's cached patterns
            if hasattr(self.markov_agent.analyzer, 'cached_sample_patterns'):
                return self.markov_agent.analyzer.cached_sample_patterns

            # Otherwise, analyze sample file
            sample_path = Path(__file__).parent / "prompts" / "sample.txt"
            if sample_path.exists():
                with open(sample_path, 'r') as f:
                    sample_text = f.read()
                return self.markov_agent.analyzer.analyze_sample_patterns(sample_text)
        except Exception:
            pass
        return {}

    def _get_learned_patterns_for_manipulator(self):
        """
        Get learned patterns from the sample text (prompts/sample.txt) for the algorithmic manipulator.

        These patterns are from human text that passes GPTZero, so we mirror them.
        """
        patterns = {}

        try:
            # Get burstiness distribution from database
            burstiness = self.markov_agent.db.get_learned_pattern('burstiness')
            if burstiness:
                patterns['burstiness'] = burstiness

            # Get punctuation patterns
            punctuation = self.markov_agent.db.get_learned_pattern('punctuation')
            if punctuation:
                patterns['punctuation'] = punctuation

            # Get sample patterns from analyzer
            sample_patterns = self._get_sample_patterns()
            if sample_patterns:
                # Extract sentence openers
                if 'sentence_openers' in sample_patterns:
                    patterns['sentence_openers'] = dict(sample_patterns['sentence_openers'])

                # Extract sentence lengths by position
                if 'sentence_lengths_by_position' in sample_patterns:
                    patterns['sentence_lengths_by_position'] = sample_patterns['sentence_lengths_by_position']

        except Exception as e:
            print(f">> Warning: Could not load learned patterns: {e}")

        return patterns

    def _get_sample_sentence_patterns(self):
        """
        Extract sentence patterns from sample text for voice matching.

        Returns dict with:
        - examples: Sample sentences showing the voice/style
        - avg_sentence_length: Average words per sentence
        - sentence_structures: Common sentence structure patterns
        - voice_characteristics: Direct vs complex, active vs passive, etc.
        """
        try:
            sample_path = Path(__file__).parent / "prompts" / "sample.txt"
            if not sample_path.exists():
                return self._default_sentence_patterns()

            with open(sample_path, 'r') as f:
                sample_text = f.read()

            doc = self.markov_agent.analyzer.nlp(sample_text[:5000])  # First 5000 chars
            sentences = [str(s).strip() for s in doc.sents if len(str(s).split()) > 5]

            if not sentences:
                return self._default_sentence_patterns()

            # Calculate average sentence length
            lengths = [len(s.split()) for s in sentences]
            avg_length = sum(lengths) / len(lengths) if lengths else 15

            # Get example sentences that represent the voice
            good_examples = [s for s in sentences if 10 <= len(s.split()) <= 30][:10]

            # Analyze sentence structures
            structures = self._analyze_sentence_structures(sentences[:50])

            return {
                'examples': '\n'.join(good_examples),
                'avg_sentence_length': avg_length,
                'sentence_structures': structures,
                'voice_characteristics': {
                    'is_direct': avg_length < 20,
                    'uses_complex_clauses': any('which' in s or 'that is' in s for s in sentences[:20]),
                    'uses_references': any('said' in s or 'according' in s.lower() for s in sentences[:20]),
                    'declarative_style': True  # Sample is declarative/analytical
                }
            }
        except Exception as e:
            print(f">> Warning: Could not extract sample patterns: {e}")
            return self._default_sentence_patterns()

    def _default_sentence_patterns(self):
        """Default patterns if sample cannot be analyzed."""
        return {
            'examples': 'The law of contradiction in things is the most basic law in materialist dialectics.',
            'avg_sentence_length': 18,
            'sentence_structures': {'declarative': 0.7, 'complex': 0.2, 'compound': 0.1},
            'voice_characteristics': {
                'is_direct': True,
                'uses_complex_clauses': True,
                'uses_references': True,
                'declarative_style': True
            }
        }

    def _analyze_sentence_structures(self, sentences):
        """Analyze the structural patterns of sample sentences."""
        structures = {'declarative': 0, 'complex': 0, 'compound': 0, 'question': 0}

        for sent in sentences:
            sent_lower = sent.lower()
            if '?' in sent:
                structures['question'] += 1
            elif ', and ' in sent or ', but ' in sent:
                structures['compound'] += 1
            elif 'which' in sent_lower or 'that' in sent_lower or 'because' in sent_lower:
                structures['complex'] += 1
            else:
                structures['declarative'] += 1

        total = sum(structures.values()) or 1
        return {k: v / total for k, v in structures.items()}

    def _matches_sample_voice(self, text, sample_patterns):
        """
        Check if text roughly matches the sample's voice patterns.

        Returns True if the text already sounds like the sample.
        """
        if not text or not sample_patterns:
            return True

        doc = self.markov_agent.analyzer.nlp(text)
        sentences = list(doc.sents)

        if not sentences:
            return True

        # Check average sentence length (should be within 30% of sample)
        avg_length = sum(len(str(s).split()) for s in sentences) / len(sentences)
        sample_avg = sample_patterns.get('avg_sentence_length', 18)

        if abs(avg_length - sample_avg) > sample_avg * 0.5:
            return False  # Too different in length

        # Check for AI-typical patterns that don't match sample
        text_lower = text.lower()
        ai_patterns = ['it is important to note', 'significantly', 'furthermore', 'moreover',
                       'plays a crucial role', 'in conclusion', 'the implications']

        if any(p in text_lower for p in ai_patterns):
            return False  # Has AI patterns

        return True

    def _restructure_sentence_groups(self, paragraph):
        """
        Restructure groups of 2-3 sentences to match sample voice.

        This allows content to move BETWEEN sentences while preserving meaning.
        The goal is to match the sentence patterns and voice of the sample text.
        """
        doc = self.markov_agent.analyzer.nlp(paragraph)
        sentences = [str(s).strip() for s in doc.sents]

        if len(sentences) < 2:
            return paragraph

        # Get sample sentence patterns for comparison
        sample_patterns = self._get_sample_sentence_patterns()

        result_groups = []
        i = 0
        groups_restructured = 0

        while i < len(sentences):
            # Take groups of 2-3 sentences
            group_size = min(3, len(sentences) - i)
            group = sentences[i:i + group_size]
            group_text = ' '.join(group)

            # Check if group needs restructuring
            if not self._matches_sample_voice(group_text, sample_patterns) and groups_restructured < 3:
                # Restructure to match sample voice
                restructured = self._restructure_to_sample_voice(group_text, sample_patterns)
                if restructured and restructured != group_text:
                    result_groups.append(restructured)
                    groups_restructured += 1
                    print(f"      >> Restructured sentence group to match sample voice")
                else:
                    result_groups.append(group_text)
            else:
                result_groups.append(group_text)

            i += group_size

        return ' '.join(result_groups)

    def _restructure_to_sample_voice(self, sentence_group, sample_patterns):
        """
        Use LLM to restructure sentence group to match sample voice.

        Key constraints:
        1. Preserve ALL factual content
        2. Match sample sentence structure patterns
        3. Can combine/split sentences
        4. Can move clauses between sentences
        """
        examples = sample_patterns.get('examples', '')[:600]
        voice_char = sample_patterns.get('voice_characteristics', {})
        avg_len = sample_patterns.get('avg_sentence_length', 18)

        voice_desc = []
        if voice_char.get('is_direct'):
            voice_desc.append("direct and clear")
        if voice_char.get('uses_complex_clauses'):
            voice_desc.append("uses subordinate clauses (which, that, because)")
        if voice_char.get('declarative_style'):
            voice_desc.append("declarative and analytical")

        voice_str = ", ".join(voice_desc) if voice_desc else "clear and direct"

        prompt = f"""Restructure these sentences to match this writing style:

SAMPLE STYLE (match this voice):
{examples}

STYLE CHARACTERISTICS:
- Voice: {voice_str}
- Average sentence length: ~{int(avg_len)} words
- Can combine short sentences or split long ones
- Can move clauses between sentences

SENTENCES TO RESTRUCTURE:
{sentence_group}

CRITICAL RULES:
1. Preserve ALL factual content and meaning - do NOT remove any information
2. Match the sentence structure patterns of the sample
3. You MAY combine sentences or split them
4. You MAY move clauses between sentences to improve flow
5. AVOID these AI-typical phrases: "It is important to note", "significantly impacts", "plays a crucial role", "furthermore", "moreover"
6. Use simple, direct language like the sample
7. AVOID semicolons unless absolutely necessary
8. Output ONLY the restructured text, nothing else"""

        try:
            result = self.call_model(self.editor_model, prompt, system_prompt=self.structural_editor_prompt)
            if result and result.strip():
                # Validate the result preserves key content
                original_words = set(sentence_group.lower().split())
                result_words = set(result.lower().split())

                # Check that we haven't lost too many content words
                content_words = {w for w in original_words if len(w) > 4 and w.isalpha()}
                preserved = len(content_words & result_words) / len(content_words) if content_words else 1

                if preserved >= 0.7:  # At least 70% of content words preserved
                    return result.strip()
                else:
                    print(f"      >> Restructure rejected: only {preserved:.0%} content preserved")
                    return sentence_group
            return sentence_group
        except Exception as e:
            print(f"      >> Warning: Could not restructure sentences: {e}")
            return sentence_group

    def _simplify_metaphors(self, text):
        """
        Simplify AI-typical metaphors and flowery language.

        This addresses the ROOT CAUSE of ZeroGPT detection:
        AI text has dense poetic flourishes that human text doesn't have.

        Example transformations:
        - "shrouded in fog" → "uncertain"
        - "trembling victories" → "fragile wins"
        - "map the leviathan" → "understand the system"
        """
        if not self.simplify_metaphor_prompt:
            return text

        # Detect sentences with metaphors
        flagged_sentences = self.metaphor_detector.detect_metaphor_sentences(text)

        if not flagged_sentences:
            return text

        print(f"    >> Found {len(flagged_sentences)} sentences with AI-typical metaphors")

        result = text
        for item in flagged_sentences:
            original_sent = item['sentence']
            metaphor_words = item['metaphor_words']

            # Skip if sentence is too short
            if len(original_sent.split()) < 5:
                continue

            # Call LLM to simplify
            prompt = f"""{self.simplify_metaphor_prompt}

INPUT: "{original_sent}"
OUTPUT:"""

            try:
                simplified = self.call_model(self.editor_model, prompt, "")
                simplified = simplified.strip().strip('"').strip("'")

                # Validate: simplified should be similar length (not truncated or expanded)
                orig_len = len(original_sent.split())
                simp_len = len(simplified.split())

                if simp_len < orig_len * 0.5 or simp_len > orig_len * 1.5:
                    print(f"      ! Skipping bad simplification (length mismatch)")
                    continue

                # Check that it doesn't reintroduce metaphors
                new_metaphors = self.metaphor_detector.find_metaphor_words(simplified)
                if len(new_metaphors) >= len(metaphor_words):
                    print(f"      ! Skipping - still has metaphors: {new_metaphors}")
                    continue

                # Replace in text
                result = result.replace(original_sent, simplified)
                print(f"      ✓ Simplified: {metaphor_words} removed")

            except Exception as e:
                print(f"      ! Error simplifying: {e}")
                continue

        return result

    def _apply_structural_templates(self, paragraph):
        """
        Apply structural templates from sample text to a paragraph.

        This transfers STYLE (punctuation placement) without changing WORDS.
        The key insight: GPTZero detects AI by structure, not vocabulary.
        """
        try:
            nlp = self.markov_agent.analyzer.nlp
            extractor = self.markov_agent.structural_extractor

            doc = nlp(paragraph)
            sentences = list(doc.sents)

            result_sentences = []
            for sent in sentences:
                # Get word count
                word_count = sum(1 for t in sent if not t.is_punct and not t.is_space)

                if word_count < 5:
                    result_sentences.append(str(sent))
                    continue

                # Get matching template from sample
                template = extractor.get_matching_template(word_count)

                if template is None:
                    result_sentences.append(str(sent))
                    continue

                # Apply template (changes punctuation only, not words)
                modified = extractor.apply_template(sent, template)
                result_sentences.append(modified)

            return ' '.join(result_sentences)

        except Exception as e:
            print(f">> Warning: Could not apply structural templates: {e}")
            return paragraph

    def _restructure_paragraphs(self, paragraphs):
        """
        Adjust paragraph structure to match sample patterns including RHYTHM.

        Actions:
        1. Get target rhythm from sample (long/short/medium sequence)
        2. Split very long paragraphs to match target
        3. Merge short paragraphs to match target

        Returns:
            List of restructured paragraphs
        """
        # First pass: basic restructuring
        result = []
        i = 0

        while i < len(paragraphs):
            para = paragraphs[i]

            # Skip empty, headers, or special content
            if not para.strip() or para.strip().startswith('#') or self._is_special_content(para):
                result.append(para)
                i += 1
                continue

            word_count = len(para.split())

            # Check if should split (very long > 250 words)
            if word_count > 250:
                split_paras = self._split_long_paragraph(para)
                result.extend(split_paras)
                print(f"      >> Split long paragraph ({word_count} words) into {len(split_paras)} parts")
                i += 1
                continue

            # Check if should merge with next (very short < 40 words)
            if word_count < 40 and i + 1 < len(paragraphs):
                next_para = paragraphs[i + 1]
                if not next_para.strip().startswith('#') and not self._is_special_content(next_para):
                    next_word_count = len(next_para.split())
                    combined_count = word_count + next_word_count

                    # Only merge if combined isn't too long
                    if combined_count < 200:
                        merged = para.strip() + ' ' + next_para.strip()
                        result.append(merged)
                        print(f"      >> Merged short paragraphs ({word_count} + {next_word_count} words)")
                        i += 2
                        continue

            result.append(para)
            i += 1

        return result

    def _split_long_paragraph(self, para):
        """Split a long paragraph at natural boundaries."""
        doc = self.markov_agent.analyzer.nlp(para)
        sentences = list(doc.sents)

        if len(sentences) <= 2:
            return [para]  # Can't meaningfully split

        # Target: split roughly in half at sentence boundary
        target_split = len(sentences) // 2

        first_part = ' '.join(str(s) for s in sentences[:target_split])
        second_part = ' '.join(str(s) for s in sentences[target_split:])

        return [first_part.strip(), second_part.strip()]

    def _enhance_transitions(self, paragraphs):
        """
        Improve how paragraphs connect to each other.

        CONSERVATIVE: Only add transitions sparingly to avoid repetition.

        Returns list of paragraphs with improved transitions.
        """
        if len(paragraphs) < 2:
            return paragraphs

        result = [paragraphs[0]]  # First paragraph stays as-is

        # Track how many transitions we've added to avoid over-transitioning
        transitions_added = 0
        max_transitions = len(paragraphs) // 5  # Add at most 20% of paragraphs

        for i in range(1, len(paragraphs)):
            para = paragraphs[i]
            prev_para = paragraphs[i - 1]

            # Skip headers, empty, or special content
            if not para.strip() or para.strip().startswith('#') or self._is_special_content(para):
                result.append(para)
                continue

            if not prev_para.strip() or prev_para.strip().startswith('#'):
                result.append(para)
                continue

            # Don't add transitions if paragraph already starts with common transition words
            first_word = para.split()[0].lower() if para.split() else ""
            if first_word in ['the', 'this', 'these', 'such', 'however', 'but', 'yet', 'thus', 'also', 'i', 'we']:
                result.append(para)
                continue

            # Only add reference transitions when there are clearly shared nouns
            if transitions_added < max_transitions:
                prev_doc = self.markov_agent.analyzer.nlp(prev_para[-150:])
                curr_doc = self.markov_agent.analyzer.nlp(para[:100])

                prev_nouns = {t.lemma_.lower() for t in prev_doc if t.pos_ == 'NOUN' and len(t.text) > 4}
                curr_nouns = {t.lemma_.lower() for t in curr_doc if t.pos_ == 'NOUN' and len(t.text) > 4}

                shared_nouns = prev_nouns & curr_nouns

                # Only add reference transition if there are at least 2 shared nouns
                if len(shared_nouns) >= 2:
                    # Change "The X..." to "This X..."
                    if para.lower().startswith('the '):
                        para = 'This ' + para[4:]
                        transitions_added += 1
                        print(f"      >> Added 'This' transition")

            result.append(para)

        return result

    def _comprehensive_gptzero_validation(self, original, output, sample_patterns=None):
        """
        Comprehensive validation against all GPTZero detection metrics.

        Checks:
        1. Vocabulary repetition (no word >2% of text)
        2. Perplexity range (within tolerance of original)
        3. Burstiness (sentence length variation)
        4. Vocabulary diversity (unique words / total words)
        5. Sentence complexity variance
        6. AI-flagged sentences

        Returns:
            (is_valid: bool, score: float 0-1, issues: list, details: dict)
        """
        issues = []
        details = {}
        score_components = []

        # 1. Vocabulary repetition check
        vocab_valid, vocab_issues, repeated_words = self._check_vocabulary_repetition(output)
        details['vocabulary_repetition'] = {
            'valid': vocab_valid,
            'repeated_words': repeated_words,
            'issues': vocab_issues
        }
        if not vocab_valid:
            issues.extend(vocab_issues[:3])  # Limit to top 3 issues
            score_components.append(0.0)
        else:
            score_components.append(1.0)

        # 2. Perplexity validation
        ppl_valid, orig_ppl, out_ppl, ppl_feedback = self._validate_perplexity(original, output)
        details['perplexity'] = {
            'valid': ppl_valid,
            'original': orig_ppl,
            'output': out_ppl,
            'feedback': ppl_feedback
        }
        if not ppl_valid:
            issues.append(ppl_feedback)
            score_components.append(0.3)  # Partial score
        else:
            score_components.append(1.0)

        # 3. Burstiness check (sentence length variation)
        burstiness = self._calculate_burstiness(output)
        if burstiness['lengths']:
            variance = sum((l - sum(burstiness['lengths'])/len(burstiness['lengths']))**2
                          for l in burstiness['lengths']) / len(burstiness['lengths'])
            burstiness_valid = variance > 20  # Need reasonable variance
            details['burstiness'] = {
                'valid': burstiness_valid,
                'variance': variance,
                'lengths': burstiness['lengths']
            }
            if not burstiness_valid:
                issues.append(f"Low sentence length variance: {variance:.1f} (target: >20)")
                score_components.append(0.5)
            else:
                score_components.append(1.0)

        # 4. Vocabulary diversity (unique words / total words)
        text_lower = re.sub(r'[^\w\s]', ' ', output.lower())
        words = [w for w in text_lower.split() if len(w) > 2]
        if words:
            unique_ratio = len(set(words)) / len(words)
            diversity_valid = unique_ratio > 0.35  # Human text typically >35% unique
            details['vocabulary_diversity'] = {
                'valid': diversity_valid,
                'unique_ratio': unique_ratio,
                'total_words': len(words),
                'unique_words': len(set(words))
            }
            if not diversity_valid:
                issues.append(f"Low vocabulary diversity: {unique_ratio:.1%} unique (target: >35%)")
                score_components.append(0.5)
            else:
                score_components.append(1.0)

        # 5. AI-flagged sentences check
        ai_check = self._identify_ai_flagged_sentences(output)
        total_sentences = len(ai_check['flagged']) + len(ai_check['ok'])
        if total_sentences > 0:
            flagged_ratio = len(ai_check['flagged']) / total_sentences
            ai_valid = flagged_ratio < 0.3  # Less than 30% flagged
            details['ai_patterns'] = {
                'valid': ai_valid,
                'flagged_count': len(ai_check['flagged']),
                'total_sentences': total_sentences,
                'flagged_ratio': flagged_ratio,
                'sample_issues': dict(list(ai_check['issues'].items())[:3])
            }
            if not ai_valid:
                issues.append(f"Too many AI-pattern sentences: {flagged_ratio:.1%} ({len(ai_check['flagged'])}/{total_sentences})")
                score_components.append(0.3)
            else:
                score_components.append(1.0)

        # Calculate overall score
        overall_score = sum(score_components) / len(score_components) if score_components else 0.0
        is_valid = overall_score >= 0.7 and len([s for s in score_components if s < 0.5]) == 0

        return is_valid, overall_score, issues, details

    def _validate_word_usage(self, output_text, preferred_words_dict):
        """
        Validate if output uses preferred words from sample text.

        Args:
            output_text: The generated output text
            preferred_words_dict: Dictionary with POS categories as keys and word lists as values

        Returns:
            (is_valid: bool, usage_percentage: float, feedback: str)
        """
        # Normalize output text and get lemmas
        def normalize_words(text):
            text = re.sub(r'[^\w\s\']', ' ', text.lower())
            words = text.split()
            return [w for w in words if len(w) > 1]

        # Also check with lemmatization for better matching
        try:
            doc = self.markov_agent.analyzer.nlp(output_text.lower())
            output_lemmas = [token.lemma_.lower() for token in doc if not token.is_punct and len(token.text) > 1]
        except:
            output_lemmas = normalize_words(output_text)

        output_words = normalize_words(output_text)
        if not output_words:
            return True, 0.0, "No words to validate"

        # Count how many output words/lemmas are in preferred vocabulary
        total_preferred = set()
        for words in preferred_words_dict.values():
            total_preferred.update(words)

        # Check both original words and lemmas
        used_preferred = [w for w in output_words if w in total_preferred]
        used_preferred_lemmas = [w for w in output_lemmas if w in total_preferred]

        # Use the higher count (either direct match or lemma match)
        total_matches = max(len(used_preferred), len(used_preferred_lemmas))
        usage_percentage = total_matches / len(output_words) if output_words else 0.0

        # Threshold: at least 15% of words should be from preferred vocabulary
        is_valid = usage_percentage >= 0.15

        if not is_valid:
            feedback = f"Low preferred word usage ({usage_percentage:.1%}). Try using more words from the sample text vocabulary."
        else:
            feedback = f"Preferred word usage: {usage_percentage:.1%}"

        return is_valid, usage_percentage, feedback

    def _split_into_sentences(self, text):
        """Split text into sentences, handling common punctuation."""
        # Split on sentence-ending punctuation
        sentences = re.split(r'[.!?]+', text)
        # Filter out empty strings and clean up
        sentences = [s.strip() for s in sentences if s.strip()]
        return sentences

    def _calculate_burstiness(self, text):
        """Calculate sentence length distribution (burstiness metrics)."""
        sentences = self._split_into_sentences(text)
        if not sentences:
            return {'short': 0, 'medium': 0, 'long': 0, 'total': 0}

        lengths = [len(s.split()) for s in sentences]
        total = len(lengths)

        short = sum(1 for l in lengths if 5 <= l <= 10)
        medium = sum(1 for l in lengths if 15 <= l <= 25)
        long_sent = sum(1 for l in lengths if l >= 30)

        return {
            'short': short / total if total > 0 else 0,
            'medium': medium / total if total > 0 else 0,
            'long': long_sent / total if total > 0 else 0,
            'total': total,
            'lengths': lengths
        }

    def _validate_gptzero_patterns(self, text):
        """
        Validate text against GPTZero detection patterns.
        Returns (pass: bool, issues: list)
        """
        issues = []
        sentences = self._split_into_sentences(text)

        if len(sentences) < 2:
            return True, []  # Single sentence paragraphs are fine

        # Check 1: Sentence length variation
        burstiness = self._calculate_burstiness(text)
        if burstiness['total'] > 0:
            # Need at least some variation
            has_short = burstiness['short'] > 0
            has_medium = burstiness['medium'] > 0 or any(11 <= l <= 29 for l in burstiness['lengths'])
            has_long = burstiness['long'] > 0

            variation_count = sum([has_short, has_medium, has_long])
            if variation_count < 2:
                issues.append(f"Low sentence length variation - need mix of short/medium/long sentences")

        # Check 2: Sentence starter diversity
        starters = []
        for sent in sentences:
            words = sent.split()
            if words:
                # Get first word, normalized
                first = words[0].lower().rstrip('.,;:')
                starters.append(first)

        if starters:
            starter_counts = Counter(starters)
            most_common_count = starter_counts.most_common(1)[0][1] if starter_counts else 0
            max_percentage = most_common_count / len(starters)

            if max_percentage > 0.3:  # More than 30% start with same word
                most_common = starter_counts.most_common(1)[0][0]
                issues.append(f"Repetitive sentence starters - {max_percentage:.1%} start with '{most_common}'")

        # Check 3: Punctuation balance
        em_dashes = len(re.findall(r'—|--', text))
        semicolons = len(re.findall(r';', text))
        periods = len(re.findall(r'\.', text))

        total_punct = em_dashes + semicolons + periods
        if total_punct > 0:
            em_dash_ratio = em_dashes / total_punct
            semicolon_ratio = semicolons / total_punct

            if em_dash_ratio > 0.4:  # More than 40% em-dashes
                issues.append(f"Excessive em-dash usage ({em_dash_ratio:.1%})")
            if semicolon_ratio > 0.4:  # More than 40% semicolons
                issues.append(f"Excessive semicolon usage ({semicolon_ratio:.1%})")

        # Check 4: Repetitive structure (3+ sentences with similar structure)
        if len(sentences) >= 3:
            # Simple check: sentences with same length (within 2 words)
            length_groups = {}
            for i, sent in enumerate(sentences):
                length = len(sent.split())
                # Group by similar lengths
                key = (length // 3) * 3  # Group into buckets
                if key not in length_groups:
                    length_groups[key] = []
                length_groups[key].append(i)

            # Check if any group has 3+ sentences
            for key, indices in length_groups.items():
                if len(indices) >= 3:
                    issues.append(f"Repetitive structure - {len(indices)} sentences with similar length ({key}-{key+2} words)")
                    break

        return len(issues) == 0, issues

    def _fix_sentence_variation(self, text, issues):
        """
        Apply targeted fixes to address GPTZero pattern issues.
        Returns fixed text.
        """
        # For now, we'll use the model to fix issues
        # This is a placeholder - in practice, we might want more algorithmic fixes
        fix_prompt = f"""The following text has been flagged for AI detection patterns:

ISSUES TO FIX:
{chr(10).join(f"- {issue}" for issue in issues)}

TEXT TO FIX:
{text}

TASK: Fix the issues by:
1. Varying sentence lengths (add some short 5-8 word sentences, some medium 15-25 word, some long 30+ word)
2. Varying sentence starters (don't start multiple sentences with the same word)
3. Balancing punctuation (mix periods, semicolons, and em-dashes - don't overuse one type)
4. Breaking up repetitive patterns

Keep the SAME WORDS and meaning. Only change structure and punctuation.
Output ONLY the fixed text - no explanations."""

        try:
            fixed = self.call_model(self.editor_model, fix_prompt, system_prompt=self.structural_editor_prompt)
            return fixed.strip()
        except Exception as e:
            print(f"    ! Error fixing variation: {e}")
            return text  # Return original if fix fails

    def _remove_ai_transitions(self, text):
        """
        Remove transition phrases that are flagged as AI patterns by detectors.
        GPTZero specifically flags: "In fact,", "Indeed,", "Moreover,", "Furthermore,", etc.
        """
        # List of transitions to remove (case-insensitive at sentence start)
        ai_transitions = [
            'In fact, ', 'In fact,', 'Indeed, ', 'Indeed,',
            'Moreover, ', 'Moreover,', 'Furthermore, ', 'Furthermore,',
            'Additionally, ', 'Additionally,', 'Consequently, ', 'Consequently,',
            'Subsequently, ', 'Subsequently,', 'Therefore, ', 'Therefore,',
            'Hence, ', 'Hence,', 'Thus, ', 'Thus,',
            'As a result, ', 'As a result,', 'To be sure, ', 'To be sure,',
            'Needless to say, ', 'Needless to say,', 'It is worth noting that ',
        ]

        for transition in ai_transitions:
            # Remove at start of sentence (after period, semicolon, or start of text)
            text = re.sub(r'(?<=[.;!?]\s)' + re.escape(transition), '', text, flags=re.IGNORECASE)
            text = re.sub(r'^' + re.escape(transition), '', text, flags=re.IGNORECASE)

        # Fix capitalization: lowercase after semicolons (not sentence endings)
        # But preserve "I" and proper nouns - only lowercase common words
        def fix_semicolon_cap(match):
            word_start = match.group(1)
            # Don't lowercase "I" standing alone
            if word_start == 'I':
                return '; I'
            # Lowercase common words that shouldn't be capitalized after semicolon
            return '; ' + word_start.lower()

        text = re.sub(r'; ([A-Z])(?=[a-z])', fix_semicolon_cap, text)
        # But capitalize after periods, question marks, exclamation marks
        text = re.sub(r'([.!?]\s)([a-z])', lambda m: m.group(1) + m.group(2).upper(), text)

        return text.strip()

    def _remove_em_dashes(self, text):
        """
        Remove em dashes from text, replacing them with appropriate punctuation.
        Replaces em-dashes (—) and double hyphens (--) with commas or semicolons.
        """
        # Replace em-dash (—) with comma or semicolon based on context
        # Pattern: word—word or word— word -> word, word or word; word
        text = re.sub(r'([a-zA-Z0-9])—([a-zA-Z0-9])', r'\1, \2', text)
        text = re.sub(r'([a-zA-Z0-9])—\s+([a-zA-Z0-9])', r'\1, \2', text)
        text = re.sub(r'([a-zA-Z0-9])\s+—([a-zA-Z0-9])', r'\1, \2', text)
        text = re.sub(r'([a-zA-Z0-9])\s+—\s+([a-zA-Z0-9])', r'\1, \2', text)

        # Replace double hyphens (--) with comma
        text = re.sub(r'([a-zA-Z0-9])--([a-zA-Z0-9])', r'\1, \2', text)
        text = re.sub(r'([a-zA-Z0-9])--\s+([a-zA-Z0-9])', r'\1, \2', text)
        text = re.sub(r'([a-zA-Z0-9])\s+--([a-zA-Z0-9])', r'\1, \2', text)
        text = re.sub(r'([a-zA-Z0-9])\s+--\s+([a-zA-Z0-9])', r'\1, \2', text)

        # Handle standalone em-dashes (spaces around them)
        text = re.sub(r'\s+—\s+', ', ', text)
        text = re.sub(r'\s+--\s+', ', ', text)

        # Clean up any double commas that might result
        text = re.sub(r',\s*,', ',', text)
        text = re.sub(r',\s+,\s+', ', ', text)

        return text

    def _check_em_dashes(self, text):
        """Check if text contains em dashes. Returns (has_em_dash: bool, count: int)."""
        em_dash_count = len(re.findall(r'—|--', text))
        return em_dash_count > 0, em_dash_count

    def _add_natural_variation(self, text):
        """
        Add natural variation to text while maintaining correct grammar.
        Ensures mix of sentence lengths and structures.
        """
        sentences = self._split_into_sentences(text)
        if len(sentences) < 3:
            return text  # Not enough sentences to vary

        burstiness = self._calculate_burstiness(text)

        # Check if we need more variation
        needs_short = burstiness['short'] == 0
        needs_long = burstiness['long'] == 0

        # If variation is already good, return as-is
        if not needs_short and not needs_long:
            return text

        # Use model to add variation
        variation_prompt = f"""The following text needs more natural variation in sentence length:

CURRENT TEXT:
{text}

TASK: Add natural variation by:
- If missing short sentences: Add 1-2 very short sentences (5-8 words) for emphasis
- If missing long sentences: Combine some medium sentences into longer complex sentences (35+ words)
- Maintain correct grammar throughout
- Keep the SAME WORDS - only restructure

Output ONLY the varied text - no explanations."""

        try:
            varied = self.call_model(self.editor_model, variation_prompt, system_prompt=self.structural_editor_prompt)
            return varied.strip()
        except Exception as e:
            print(f"    ! Error adding variation: {e}")
            return text  # Return original if variation fails

    def _final_humanization_pass(self, text):
        """
        Final review and polish pass for entire output.
        Ensures overall burstiness and rhythm match sample.
        """
        # Check overall patterns
        pass_check, issues = self._validate_gptzero_patterns(text)

        if pass_check:
            return text  # Already good

        # If there are issues, try a final polish
        if len(issues) <= 2:  # Only minor issues
            polish_prompt = f"""Final polish pass. Fix these minor issues:

ISSUES:
{chr(10).join(f"- {issue}" for issue in issues)}

TEXT:
{text}

Make minimal changes to fix these issues. Keep all words and meaning.
Output ONLY the polished text."""

            try:
                polished = self.call_model(self.editor_model, polish_prompt, system_prompt=self.structural_editor_prompt)
                return polished.strip()
            except Exception as e:
                print(f"    ! Error in final polish: {e}")

        return text  # Return as-is if polish fails

    def extract_style_dna(self):
        """Extracts the structural skeleton (Cadence, Syntax, POS) from the sample."""
        # Return cached style guide if already extracted
        if self._style_guide is not None:
            return self._style_guide

        print(">> Extracting Syntactic Skeleton from sample...")
        # Use the structural analyst prompt from file
        system_prompt = self.structural_analyst_prompt
        # Analyze first 2500 chars for structural patterns
        prompt = f"{self.style_sample[:2500]}"
        try:
            self._style_guide = self.call_model(self.critic_model, prompt, system_prompt)
            print(f"\nCaptured Syntactic Blueprint:\n{self._style_guide[:200]}...\n")
        except requests.exceptions.Timeout:
            raise RuntimeError("Timeout while extracting syntactic skeleton. The API may be slow or unresponsive.")
        except Exception as e:
            raise RuntimeError(f"Error extracting syntactic skeleton: {e}")
        return self._style_guide

    def get_sample_patterns(self):
        """Extract and cache structural patterns from sample text."""
        if self._sample_patterns is not None:
            return self._sample_patterns

        print(">> Analyzing sample text structural patterns...")
        self._sample_patterns = self.markov_agent.analyzer.analyze_sample_patterns(self.style_sample)
        return self._sample_patterns

    def _extract_key_elements(self, text):
        """Extract key elements for meaning validation: citations, proper nouns, numbers, key phrases, and entities."""
        # Use spaCy NER for better entity extraction
        try:
            doc = self.markov_agent.analyzer.nlp(text)
            # Extract named entities (PERSON, ORG, GPE, etc.)
            entities = set()
            for ent in doc.ents:
                if ent.label_ in ['PERSON', 'ORG', 'GPE', 'EVENT', 'WORK_OF_ART']:
                    entities.add(ent.text)
        except:
            entities = set()

        # Extract proper nouns more carefully - exclude common sentence starters
        # Only count words that are actually proper nouns (not just capitalized at start of sentence)
        proper_nouns = set()
        try:
            doc = self.markov_agent.analyzer.nlp(text)
            for token in doc:
                # Only count as proper noun if:
                # 1. It's tagged as PROPN by spaCy
                # 2. It's not at the start of a sentence (unless it's a multi-word entity)
                # 3. It's not a common word like "The", "It", "This", etc.
                if token.pos_ == "PROPN":
                    # Check if it's part of a named entity (more reliable)
                    is_entity = any(ent.start <= token.i < ent.end for ent in doc.ents)
                    # Exclude common words that might be capitalized
                    common_words = {'the', 'it', 'this', 'that', 'these', 'those', 'a', 'an'}
                    if token.text.lower() not in common_words or is_entity:
                        proper_nouns.add(token.text)
        except:
            # Fallback to regex but filter common words
            all_caps = set(re.findall(r'\b[A-Z][a-z]+\b', text))
            common_words = {'The', 'It', 'This', 'That', 'These', 'Those', 'A', 'An'}
            proper_nouns = all_caps - common_words

        elements = {
            'citations': set(re.findall(r'\[\^\d+\]', text)),
            'proper_nouns': proper_nouns,
            'entities': entities,  # spaCy NER entities
            'numbers': set(re.findall(r'\b\d+\b', text)),
            'length': len(text),
            'paragraph_count': len([p for p in text.split('\n\n') if p.strip()]),
        }
        # Extract key phrases (3-word sequences with important conceptual words)
        words = text.lower().split()
        key_phrases = set()
        # Focus on substantive conceptual terms, not common verbs
        important_words = {
            'system', 'collapse', 'soviet', 'union', 'energy', 'entropy', 'material', 'physics',
            'biology', 'technology', 'digital', 'network', 'structure', 'pattern', 'rule', 'law',
            'force', 'process', 'evolution', 'complexity', 'emergent', 'framework', 'mechanism',
            'principle', 'concept', 'theory', 'evidence', 'data', 'research', 'study', 'analysis',
            'result', 'conclusion', 'finding', 'discovery', 'observation', 'experiment', 'method',
            'approach', 'technique', 'strategy', 'solution', 'problem', 'challenge', 'issue',
            'question', 'answer', 'explanation', 'understanding', 'knowledge', 'information', 'fact',
            'truth', 'reality', 'existence', 'nature', 'world', 'universe', 'society', 'culture',
            'history', 'future', 'past', 'present', 'matter', 'life', 'death', 'growth',
            'development', 'change', 'transformation', 'revolution', 'progress', 'innovation'
        }
        for i in range(len(words) - 2):
            phrase = ' '.join(words[i:i+3])
            if any(word in phrase for word in important_words):
                key_phrases.add(phrase)
        elements['key_phrases'] = key_phrases
        return elements

    def _verify_entities_preserved(self, original, rewritten):
        """
        Verify that all entities (proper nouns, named entities, citations) are preserved.

        Returns:
            (is_valid: bool, missing_entities: list, feedback: str)
        """
        orig_elements = self._extract_key_elements(original)
        rewrite_elements = self._extract_key_elements(rewritten)

        missing = []

        # Check citations (critical - must be 100% preserved)
        missing_citations = orig_elements['citations'] - rewrite_elements['citations']
        if missing_citations:
            missing.append(f"citations: {missing_citations}")

        # Check named entities (critical - must be 100% preserved)
        missing_entities = orig_elements['entities'] - rewrite_elements['entities']
        if missing_entities:
            missing.append(f"entities: {missing_entities}")

        # Check proper nouns (should be mostly preserved, but allow some flexibility)
        if orig_elements['proper_nouns']:
            missing_nouns = orig_elements['proper_nouns'] - rewrite_elements['proper_nouns']
            # Only flag if significant proper nouns are missing (more than 50% of non-common words)
            # Filter out very short words that might be false positives
            significant_nouns = {n for n in orig_elements['proper_nouns'] if len(n) > 3}
            if significant_nouns:
                missing_significant = significant_nouns - rewrite_elements['proper_nouns']
                if len(missing_significant) > len(significant_nouns) * 0.5:  # More than 50% missing
                    missing.append(f"proper_nouns: {missing_significant}")

        is_valid = len(missing) == 0
        feedback = "All entities preserved" if is_valid else f"Missing: {', '.join(missing)}"

        return is_valid, missing, feedback

    def validate_meaning_preservation(self, original, rewritten):
        """Validate that key meaning elements are preserved in the rewritten text."""
        orig_elements = self._extract_key_elements(original)
        rewrite_elements = self._extract_key_elements(rewritten)

        issues = []

        # Check citations
        missing_citations = orig_elements['citations'] - rewrite_elements['citations']
        if missing_citations:
            issues.append(f"Missing citations: {missing_citations}")

        # Check entities (named entities from NER)
        missing_entities = orig_elements['entities'] - rewrite_elements['entities']
        if missing_entities:
            issues.append(f"Missing entities: {missing_entities}")

        # Check length (should be within 20% of original)
        length_ratio = rewrite_elements['length'] / orig_elements['length'] if orig_elements['length'] > 0 else 1.0
        if length_ratio < 0.8:
            issues.append(f"Text shortened significantly ({length_ratio:.1%} of original length)")
        elif length_ratio > 1.2:
            issues.append(f"Text lengthened significantly ({length_ratio:.1%} of original length)")

        # Check paragraph count (should be similar)
        para_diff = abs(orig_elements['paragraph_count'] - rewrite_elements['paragraph_count'])
        if para_diff > 2:
            issues.append(f"Paragraph structure changed significantly ({orig_elements['paragraph_count']} -> {rewrite_elements['paragraph_count']} paragraphs)")

        # Check key phrases overlap (at least 60% should be preserved)
        if orig_elements['key_phrases']:
            preserved_phrases = orig_elements['key_phrases'] & rewrite_elements['key_phrases']
            phrase_ratio = len(preserved_phrases) / len(orig_elements['key_phrases'])
            if phrase_ratio < 0.6:
                issues.append(f"Key concepts may be lost (only {phrase_ratio:.1%} of key phrases preserved)")

        # Check proper nouns (important ones should be preserved)
        if orig_elements['proper_nouns']:
            preserved_nouns = orig_elements['proper_nouns'] & rewrite_elements['proper_nouns']
            noun_ratio = len(preserved_nouns) / len(orig_elements['proper_nouns'])
            if noun_ratio < 0.7:
                issues.append(f"Many proper nouns missing ({noun_ratio:.1%} preserved)")

        return len(issues) == 0, issues

    def critique_text(self, text):
        """Asks the critic model to find AI patterns."""
        return self.call_model(self.critic_model, f"Analyze this text:\n\n{text}", system_prompt=self.critic_system)

    def humanize(self, text, max_retries=3):
        """
        Humanize text using a two-phase approach:

        Phase 1: ALGORITHMIC MANIPULATION (no AI)
        - Merge short sentences
        - Vary sentence starters
        - Create burstiness (length variation)
        - Add parenthetical asides

        Phase 2: AI-ASSISTED POLISH (only if needed)
        - Only for paragraphs that still fail metrics
        - Minimal, targeted changes

        This approach avoids introducing AI patterns by doing most work algorithmically.
        """
        # Detect genre to adjust processing
        genre = self._detect_genre(text)
        print(f"\n>> Detected genre: {genre}")

        # Get learned patterns from sample text
        learned_patterns = self._get_learned_patterns_for_manipulator()

        # Initialize algorithmic manipulator WITH learned patterns from sample
        manipulator = AlgorithmicSentenceManipulator(learned_patterns=learned_patterns)
        print(f">> Loaded learned patterns: burstiness={learned_patterns.get('burstiness', 'default')}")

        # Split by Paragraphs (preserve newlines)
        paragraphs = text.split('\n\n')

        # ========== PRE-PROCESSING: PARAGRAPH RESTRUCTURE ==========
        print(f"\n========== PARAGRAPH RESTRUCTURE ==========")
        paragraphs = self._restructure_paragraphs(paragraphs)

        # ========== PRE-PROCESSING: TRANSITION ENHANCEMENT ==========
        print(f"\n========== TRANSITION ENHANCEMENT ==========")
        paragraphs = self._enhance_transitions(paragraphs)

        final_output = []

        print(f"\n========== PHASE 1: ALGORITHMIC MANIPULATION ==========")
        print(f"Processing {len(paragraphs)} paragraphs...")

        for index, para in enumerate(paragraphs):
            if not para.strip():  # Handle empty lines
                final_output.append(para)
                continue

            # Skip headers or very short lines (they don't need humanizing)
            if para.strip().startswith('#') or len(para.split()) < 10:
                final_output.append(para)
                continue

            # Skip special content that should be preserved exactly
            if self._is_special_content(para):
                print(f"  > Skipping special content (paragraph {index + 1})")
                final_output.append(para)
                continue

            total_paras = len([p for p in paragraphs if p.strip() and not (p.strip().startswith('#') or len(p.split()) < 10)])
            print(f"  > Paragraph {index + 1}/{total_paras}...")

            # ========== PHASE 0: METAPHOR SIMPLIFICATION ==========
            # FIRST: Simplify AI-typical metaphors and flowery language
            # This addresses the ROOT CAUSE of ZeroGPT detection
            metaphor_density = self.metaphor_detector.get_metaphor_density(para)
            if metaphor_density > 0.3:  # More than 0.3 metaphors per 100 words
                print(f"    [META] Metaphor density: {metaphor_density:.2f}% - simplifying...")
                simplified_para = self._simplify_metaphors(para)
            else:
                simplified_para = para

            # ========== PHASE 0.5: SENTENCE GROUP RESTRUCTURING ==========
            # Restructure groups of 2-3 sentences to match sample voice
            # This can move content BETWEEN sentences to match the sample's tone
            print(f"    [VOICE] Checking voice match...")
            sample_patterns = self._get_sample_sentence_patterns()
            if not self._matches_sample_voice(simplified_para, sample_patterns):
                print(f"    [VOICE] Restructuring sentences to match sample voice...")
                voice_matched_para = self._restructure_sentence_groups(simplified_para)
            else:
                voice_matched_para = simplified_para

            # ========== PHASE 1: STRUCTURAL TEMPLATE TRANSFER ==========
            # Apply structural templates from sample text
            # This transfers punctuation patterns WITHOUT changing words
            structured_para = self._apply_structural_templates(voice_matched_para)

            # ========== PHASE 2: ALGORITHMIC MANIPULATION ==========
            # Apply algorithmic sentence manipulation (NO AI INVOLVED)
            # This is the key to avoiding AI patterns
            algo_result = manipulator.process_paragraph(structured_para)

            # Check metrics after algorithmic manipulation
            algo_metrics = self.gptzero_optimizer.calculate_metrics(algo_result)
            algo_variance = algo_metrics.get('sentence_length_variance', 0)
            algo_starters = algo_metrics.get('unique_starter_ratio', 0)

            print(f"    [ALGO] Variance: {algo_variance:.1f}, Unique starters: {algo_starters:.1%}")

            # Check if algorithmic result is good enough
            algo_pass, algo_issues = self.gptzero_optimizer.validate_metrics(algo_result)

            if algo_pass:
                print(f"    ✓ Algorithmic manipulation passed! Using result.")
                new_para = algo_result
            else:
                # Try enhanced algorithmic manipulation
                print(f"    [ALGO] Issues: {', '.join(algo_issues[:2])}")
                print(f"    [ALGO] Trying enhanced manipulation...")

                # Apply additional algorithmic fixes
                enhanced = manipulator.add_parenthetical_asides(algo_result)
                enhanced_metrics = self.gptzero_optimizer.calculate_metrics(enhanced)
                enhanced_pass, _ = self.gptzero_optimizer.validate_metrics(enhanced)

                if enhanced_pass:
                    print(f"    ✓ Enhanced algorithmic manipulation passed!")
                    new_para = enhanced
                else:
                    # ========== PHASE 2: AI-ASSISTED (only if needed) ==========
                    print(f"    [AI] Algorithmic not enough, using minimal AI assistance...")

                    # Very constrained AI prompt - just fix specific issues
                    fix_prompt = f"""Fix ONLY these specific issues in the text below:
{chr(10).join(f"- {issue}" for issue in algo_issues[:2])}

TEXT TO FIX:
{algo_result}

RULES:
1. Keep ALL original words - only change punctuation and sentence boundaries
2. To vary sentence length: merge short sentences with ", and" or split long ones with periods
3. AVOID semicolons (;) - use periods or ", and" instead
4. To vary starters: move prepositional phrases to the beginning (e.g., "In the morning," "From the ruins,")
5. NEVER reorder clauses where "it", "he", "she", "they", "this", "that" would lose their referent
6. DO NOT add new words, phrases, or vocabulary
7. DO NOT paraphrase - preserve exact meaning and clause order
8. Output ONLY the fixed text, nothing else"""

                    try:
                        ai_result = self.call_model(self.editor_model, fix_prompt, system_prompt=self.structural_editor_prompt)
                        if ai_result and ai_result.strip():
                            # Validate AI didn't mess things up
                            word_overlap = self._calculate_word_overlap(para, ai_result)
                            if word_overlap >= 0.90:  # 90% word preservation
                                new_para = ai_result.strip()
                                print(f"    ✓ AI fix applied (word preservation: {word_overlap:.1%})")
                            else:
                                print(f"    ! AI changed too many words ({word_overlap:.1%}). Using algorithmic result.")
                                new_para = algo_result
                        else:
                            new_para = algo_result
                    except Exception as e:
                        print(f"    ! AI error: {e}. Using algorithmic result.")
                        new_para = algo_result

            # Final cleanup - remove em dashes and AI transitions
            new_para = self._remove_em_dashes(new_para)
            new_para = self._remove_ai_transitions(new_para)

            # Validate word preservation
            word_overlap = self._calculate_word_overlap(para, new_para)
            if word_overlap < 0.85:
                print(f"    ! Word preservation too low ({word_overlap:.1%}). Using original.")
                final_output.append(para)
                continue

            print(f"    ✓ Word preservation: {word_overlap:.1%}")
            final_output.append(new_para)

        result = "\n\n".join(final_output)

        # ========== FINAL VALIDATION ==========
        print(f"\n========== FINAL VALIDATION ==========")

        # Remove any remaining em dashes
        has_em_dash, em_dash_count = self._check_em_dashes(result)
        if has_em_dash:
            print(f">> Removing {em_dash_count} em-dash(es)...")
            result = self._remove_em_dashes(result)

        # Remove AI transition phrases
        result = self._remove_ai_transitions(result)

        # Calculate metaphor density (before vs after)
        original_metaphor_density = self.metaphor_detector.get_metaphor_density(text)
        final_metaphor_density = self.metaphor_detector.get_metaphor_density(result)
        original_metaphor_words = self.metaphor_detector.find_metaphor_words(text)
        final_metaphor_words = self.metaphor_detector.find_metaphor_words(result)
        print(f"   Metaphor density: {original_metaphor_density:.2f}% → {final_metaphor_density:.2f}%")
        print(f"   Metaphor words: {len(original_metaphor_words)} → {len(final_metaphor_words)}")
        if final_metaphor_words:
            print(f"   Remaining metaphors: {final_metaphor_words}")

        # Calculate final metrics
        final_metrics = self.gptzero_optimizer.calculate_metrics(result)
        print(f"   Sentence length variance: {final_metrics.get('sentence_length_variance', 0):.1f} (target: >30)")
        print(f"   Unique starter ratio: {final_metrics.get('unique_starter_ratio', 0):.1%} (target: >40%)")
        print(f"   Sentence type mix: {final_metrics.get('sentence_type_mix', {})}")

        final_pass, final_issues = self.gptzero_optimizer.validate_metrics(result)
        if final_pass:
            print("✓ Final output passes GPTZero validation")
        else:
            print(f"⚠ Final output has issues: {', '.join(final_issues)}")

        return result

    def _old_humanize_paragraph(self, para, index, total_paras):
        """Legacy paragraph processing - AI-based approach (deprecated)."""
        # This is the old approach that introduced AI patterns
        # Kept for reference but not used

        # Analyze input paragraph to get metrics and word count for scaling
        input_metrics = self.markov_agent.analyzer.analyze_paragraph(para)
        if not input_metrics:
            return para
        input_word_count = sum(s['length'] for s in input_metrics['sentences'])

        # Get Markov template based on current state
        prediction = self.markov_agent.db.predict_next_template(
            self.current_markov_signature,
            input_text=para,
            semantic_matcher=self.markov_agent.semantic_matcher
        )
        if not prediction:
            return para

        template_data = prediction['template']

        # Calculate scaling factor
        target_word_count = sum(s['length'] for s in template_data)
        scaling_factor = 1.0
        if target_word_count > (input_word_count * 1.5):
            scaling_factor = (input_word_count * 1.2) / target_word_count

        # Generate template
        markov_template = self.markov_agent.generate_human_readable_template(template_data, scaling_factor)

        # Select examples
        example_paras = self._select_example_paragraphs(para, num_examples=2)
        examples_text = "\n\n---\n\n".join(example_paras) if example_paras else "No examples available."

        # Use the paragraph rewrite template
        chunk_prompt = self.paragraph_rewrite_template.format(
            examples=examples_text,
            markov_template=markov_template,
            input_content=para
        )

        combined_system_prompt = self.structural_editor_prompt

        try:
            new_para = self.call_model(self.editor_model, chunk_prompt, system_prompt=combined_system_prompt)
            return new_para.strip() if new_para else para
        except:
            return para


# Legacy code - keeping the old template validation for reference
def _legacy_template_validation():
    """This was the old template validation that's no longer used."""
    pass
    # Template Validation - Check if output matches the template structure
    # output_metrics = self.markov_agent.analyzer.analyze_paragraph(new_para)
    # if output_metrics:
    #     output_sentences = output_metrics['sentences']
    #     template_sent_count = len(template_data)
    #     output_sent_count = len(output_sentences)
    #
    #     # Check sentence count (tolerance ±1)
    #     if abs(output_sent_count - template_sent_count) > 1:
    #         print(f"    ! Template validation failed: Expected {template_sent_count} sentences, got {output_sent_count}.")
    #         # Try to fix with feedback
    #         feedback_prompt = f"""Your previous attempt had {output_sent_count} sentences, but the template requires exactly {template_sent_count} sentences.
    #
    # TEMPLATE:
    # {markov_template}
    #
    # CURRENT OUTPUT:
    # {new_para}
    #
if __name__ == "__main__":
    # Require input file argument
    if len(sys.argv) < 2:
        print("Usage: python humanizer.py <input_file> [output_file]")
        print("  input_file:  Path to the input markdown file")
        print("  output_file: Optional path to output file (default: output/<input_filename>)")
        sys.exit(1)

    input_file = Path(sys.argv[1])

    if len(sys.argv) >= 3:
        output_file = Path(sys.argv[2])
    else:
        # Default: same filename in output folder
        output_file = Path("output") / input_file.name

    # Read input file
    print(f"Reading input from: {input_file}")
    with open(input_file, "r", encoding="utf-8") as f:
        text = f.read()

    # Process text
    pipeline = AgenticHumanizer()
    result = pipeline.humanize(text)

    # Write output file
    output_file.parent.mkdir(parents=True, exist_ok=True)
    print(f"\nWriting output to: {output_file}")
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(result)

    print("Done!")