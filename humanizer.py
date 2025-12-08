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
from markov import StyleTransferAgent

# Suppress the loss_type warning
warnings.filterwarnings("ignore", message=".*loss_type.*")

class GPTZeroMetricOptimizer:
    """Optimize text to evade GPTZero detection by targeting specific metrics."""

    def __init__(self, analyzer):
        self.analyzer = analyzer

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
        Humanize text by processing paragraph-by-paragraph for maximum accuracy.
        This approach preserves facts and citations better than whole-text processing.
        Uses minimal intervention - only changes structure, not vocabulary.
        """
        # Detect genre to adjust processing
        genre = self._detect_genre(text)
        print(f"\n>> Detected genre: {genre}")

        # For narrative text, we use minimal intervention (no vocabulary injection)
        minimal_mode = genre in ('narrative', 'mixed')
        if minimal_mode:
            print(">> Using MINIMAL INTERVENTION mode (preserving author's voice)")

        # Split by Paragraphs (preserve newlines)
        paragraphs = text.split('\n\n')
        final_output = []

        print(f"Processing {len(paragraphs)} paragraphs individually for maximum accuracy...")

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
            print(f"  > Restructuring Paragraph {index + 1}/{total_paras}...")

            # Analyze input paragraph to get metrics and word count for scaling
            input_metrics = self.markov_agent.analyzer.analyze_paragraph(para)
            if not input_metrics:
                print(f"    ! Could not analyze input paragraph. Using original.")
                final_output.append(para)
                continue
            input_word_count = sum(s['length'] for s in input_metrics['sentences'])

            # Get Markov template based on current state (or random if first paragraph)
            # Use semantic matching if available
            prediction = self.markov_agent.db.predict_next_template(
                self.current_markov_signature,
                input_text=para,
                semantic_matcher=self.markov_agent.semantic_matcher
            )
            if not prediction:
                print(f"    ! Database empty or prediction failed. Using original paragraph.")
                final_output.append(para)
                continue

            template_data = prediction['template']
            template_signature = prediction['signature']

            # Calculate scaling factor to adapt template to input content
            target_word_count = sum(s['length'] for s in template_data)
            scaling_factor = 1.0
            if target_word_count > (input_word_count * 1.5):
                scaling_factor = (input_word_count * 1.2) / target_word_count
                print(f"    [i] Scaling template by {scaling_factor:.2f}x to fit content.")

            # Generate detailed template with POS ratios (for sentence length guidance only)
            markov_template = self.markov_agent.generate_human_readable_template(template_data, scaling_factor)

            # Select example paragraphs similar in length to current paragraph (for rhythm reference)
            example_paras = self._select_example_paragraphs(para, num_examples=2)
            examples_text = "\n\n---\n\n".join(example_paras) if example_paras else "No examples available."

            # Use the paragraph rewrite template - NO vocabulary injection
            chunk_prompt = self.paragraph_rewrite_template.format(
                examples=examples_text,
                markov_template=markov_template,
                input_content=para
            )

            # Use the structural editor prompt from file
            combined_system_prompt = self.structural_editor_prompt

            # Run the model with timeout handling
            try:
                print(f"    Calling {self.editor_model}...", end="", flush=True)
                new_para = self.call_model(self.editor_model, chunk_prompt, system_prompt=combined_system_prompt)
                print(f" ✓ Response received ({len(new_para)} chars)")
            except requests.exceptions.Timeout:
                print(f" ✗ TIMEOUT")
                print(f"    ! Timeout for Para {index+1}. Using original.")
                final_output.append(para)
                continue
            except requests.exceptions.RequestException as e:
                print(f" ✗ REQUEST ERROR: {type(e).__name__}")
                print(f"    ! Request error for Para {index+1}: {e}. Using original.")
                final_output.append(para)
                continue
            except Exception as e:
                print(f" ✗ ERROR: {type(e).__name__}")
                print(f"    ! Unexpected error for Para {index+1}: {e}. Using original.")
                final_output.append(para)
                continue

            # Validate that we got a non-empty response
            if not new_para or not new_para.strip():
                print(f"    ! Empty response for Para {index+1}. Using original.")
                final_output.append(para)
                continue

            # Template Validation - Check if output matches the template structure
            output_metrics = self.markov_agent.analyzer.analyze_paragraph(new_para)
            if output_metrics:
                output_sentences = output_metrics['sentences']
                template_sent_count = len(template_data)
                output_sent_count = len(output_sentences)

                # Check sentence count (tolerance ±1)
                if abs(output_sent_count - template_sent_count) > 1:
                    print(f"    ! Template validation failed: Expected {template_sent_count} sentences, got {output_sent_count}.")
                    # Try to fix with feedback
                    feedback_prompt = f"""Your previous attempt had {output_sent_count} sentences, but the template requires exactly {template_sent_count} sentences.

TEMPLATE:
{markov_template}

CURRENT OUTPUT:
{new_para}

Fix the sentence count to match the template exactly. Keep all words and meaning.
Output ONLY the corrected paragraph."""
                    try:
                        corrected_para = self.call_model(self.editor_model, feedback_prompt, system_prompt=combined_system_prompt)
                        if corrected_para and corrected_para.strip():
                            # Re-validate
                            corrected_metrics = self.markov_agent.analyzer.analyze_paragraph(corrected_para)
                            if corrected_metrics:
                                corrected_count = len(corrected_metrics['sentences'])
                                if abs(corrected_count - template_sent_count) <= 1:
                                    print(f"    ✓ Template validation passed after correction")
                                    new_para = corrected_para
                                else:
                                    print(f"    ! Correction still failed. Using original paragraph.")
                                    final_output.append(para)
                                    continue
                            else:
                                print(f"    ! Could not analyze corrected paragraph. Using original.")
                                final_output.append(para)
                                continue
                        else:
                            print(f"    ! Empty correction response. Using original paragraph.")
                            final_output.append(para)
                            continue
                    except Exception as e:
                        print(f"    ! Error during template correction: {e}. Using original paragraph.")
                        final_output.append(para)
                        continue
                else:
                    # Check sentence lengths (tolerance ±2 words per sentence)
                    length_mismatches = []
                    for i, (template_sent, output_sent) in enumerate(zip(template_data, output_sentences[:len(template_data)])):
                        target_len = int(template_sent['length'] * scaling_factor)
                        target_len = max(5, target_len)
                        actual_len = output_sent['length']
                        if abs(actual_len - target_len) > 2:
                            length_mismatches.append(f"Sentence {i+1}: expected ~{target_len} words, got {actual_len}")

                    if length_mismatches:
                        print(f"    ! Sentence length mismatches: {', '.join(length_mismatches[:3])}")
                        # Don't fail, just warn - lengths are approximate
                    else:
                        print(f"    ✓ Template validation passed (sentence count and lengths)")
            else:
                print(f"    ! Could not analyze output for template validation")

            # Entity Verification - Hard Fail if critical entities drop
            entities_valid, missing_entities, entity_feedback = self._verify_entities_preserved(para, new_para)
            if not entities_valid:
                print(f"    ! Entity preservation failed in Para {index+1}: {entity_feedback}")
                # Try to regenerate with entity preservation feedback
                entity_fix_prompt = f"""Your previous output is missing critical entities that must be preserved:

MISSING ENTITIES:
{entity_feedback}

ORIGINAL PARAGRAPH:
{para}

CURRENT OUTPUT:
{new_para}

TASK: Regenerate the output ensuring ALL entities (proper nouns, citations, named entities) from the original are preserved. Output ONLY the corrected paragraph."""
                try:
                    fixed_para = self.call_model(self.editor_model, entity_fix_prompt, system_prompt=combined_system_prompt)
                    if fixed_para and fixed_para.strip():
                        fixed_valid, _, _ = self._verify_entities_preserved(para, fixed_para)
                        if fixed_valid:
                            print(f"    ✓ Entities preserved after correction")
                            new_para = fixed_para
                        else:
                            print(f"    ! Entity correction failed. Falling back to original paragraph (HARD FAIL).")
                            final_output.append(para)
                            continue
                    else:
                        print(f"    ! Empty correction response. Falling back to original paragraph (HARD FAIL).")
                        final_output.append(para)
                        continue
                except Exception as e:
                    print(f"    ! Error during entity correction: {e}. Falling back to original paragraph (HARD FAIL).")
                    final_output.append(para)
                    continue
            else:
                print(f"    ✓ {entity_feedback}")

            # WORD PRESERVATION CHECK - Ensure minimal word changes
            word_overlap = self._calculate_word_overlap(para, new_para)
            if word_overlap < 0.85:  # Require 85% word preservation
                print(f"    ! Word preservation too low ({word_overlap:.1%}). Model paraphrased instead of restructured.")
                print(f"    ! Falling back to original paragraph.")
                final_output.append(para)
                continue

            print(f"    ✓ Word preservation: {word_overlap:.1%}")

            # VOCABULARY REPETITION CHECK - Reject outputs with excessive word repetition
            vocab_valid, vocab_issues, repeated_words = self._check_vocabulary_repetition(new_para)
            if not vocab_valid:
                print(f"    ! Vocabulary repetition detected: {', '.join(vocab_issues[:2])}")
                # This is a critical issue - fall back to original paragraph
                # The model is likely injecting vocabulary from the sample text
                print(f"    ! Falling back to original paragraph to avoid repetition.")
                final_output.append(para)
                continue
            else:
                print(f"    ✓ Vocabulary diversity OK (no excessive repetition)")

            # GPTZero Metric Optimization
            gptzero_valid, gptzero_issues = self.gptzero_optimizer.validate_metrics(new_para)
            if not gptzero_valid:
                print(f"    ! GPTZero metrics below threshold: {', '.join(gptzero_issues)}")
                print(f"    ! Attempting to optimize...")

                # Get optimization guidance
                optimization_guidance = self.gptzero_optimizer.optimize_metrics(new_para)
                if optimization_guidance:
                    gptzero_prompt = f"""Your output needs optimization to pass GPTZero detection:

OPTIMIZATION NEEDED:
{optimization_guidance}

CURRENT OUTPUT:
{new_para}

TASK: Revise the output to address the optimization needs while preserving all meaning and structure. Output ONLY the optimized paragraph."""
                    try:
                        optimized_para = self.call_model(self.editor_model, gptzero_prompt, system_prompt=combined_system_prompt)
                        if optimized_para and optimized_para.strip():
                            optimized_valid, _ = self.gptzero_optimizer.validate_metrics(optimized_para)
                            if optimized_valid:
                                print(f"    ✓ GPTZero metrics optimized")
                                new_para = optimized_para
                            else:
                                print(f"    ! GPTZero optimization incomplete, but continuing")
                        else:
                            print(f"    ! Empty optimization response, but continuing")
                    except Exception as e:
                        print(f"    ! Error during GPTZero optimization: {e}, but continuing")
                else:
                    # Fallback to old method if no guidance
                    fixed_para = self._fix_sentence_variation(new_para, gptzero_issues)
                    fixed_pass, _ = self.gptzero_optimizer.validate_metrics(fixed_para)
                    if fixed_pass:
                        print(f"    ✓ GPTZero patterns fixed (fallback method)")
                        new_para = fixed_para
            else:
                print(f"    ✓ GPTZero metrics passed")

            # Structure Pattern Validation (Extended) - Compare to sample text patterns
            sample_patterns = self.get_sample_patterns()
            structure_valid, structure_issues = self.markov_agent.analyzer.validate_structure_match(new_para, sample_patterns)

            # Calculate style match score
            style_score = self.markov_agent.analyzer.calculate_style_match_score(new_para, sample_patterns)
            print(f"    Style match score: {style_score:.2%}")

            if not structure_valid or style_score < 0.6:
                print(f"    ! Structure pattern issues (score: {style_score:.2%}): {', '.join(structure_issues[:2])}")
                # Try to fix structure with feedback
                structure_fix_prompt = f"""Your output structure doesn't match the sample text patterns.

ISSUES:
{chr(10).join(f"- {issue}" for issue in structure_issues[:3])}
Style match score: {style_score:.2%} (target: >60%)

SAMPLE PATTERNS:
- Em-dash frequency: {sample_patterns['punctuation_frequency'].get('em_dash', 0):.2f} per paragraph
- Semicolon frequency: {sample_patterns['punctuation_frequency'].get('semicolon', 0):.2f} per paragraph
- Common sentence openers: {', '.join([w for w, _ in sample_patterns['sentence_openers'].most_common(5)])}

CURRENT OUTPUT:
{new_para}

TASK: Revise the output to better match the sample text structure patterns. Keep the same meaning and words. Output ONLY the revised paragraph."""
                try:
                    fixed_structure = self.call_model(self.editor_model, structure_fix_prompt, system_prompt=combined_system_prompt)
                    if fixed_structure and fixed_structure.strip():
                        fixed_valid, _ = self.markov_agent.analyzer.validate_structure_match(fixed_structure, sample_patterns)
                        fixed_score = self.markov_agent.analyzer.calculate_style_match_score(fixed_structure, sample_patterns)
                        if fixed_valid and fixed_score > style_score:
                            print(f"    ✓ Structure patterns fixed (score: {fixed_score:.2%})")
                            new_para = fixed_structure
                        else:
                            print(f"    ! Structure fix attempt failed (score: {fixed_score:.2%}), keeping current version")
                except Exception as e:
                    print(f"    ! Error fixing structure: {e}")
            else:
                print(f"    ✓ Structure patterns match sample (score: {style_score:.2%})")

            # Check for and remove em dashes
            has_em_dash, em_dash_count = self._check_em_dashes(new_para)
            if has_em_dash:
                print(f"    ! Found {em_dash_count} em-dash(es), removing...")
                new_para = self._remove_em_dashes(new_para)
                # Verify removal
                has_em_dash_after, _ = self._check_em_dashes(new_para)
                if has_em_dash_after:
                    print(f"    ! Warning: Some em-dashes may remain after removal")
                else:
                    print(f"    ✓ All em-dashes removed")

            # Analyze styled output and update Markov signature for next paragraph prediction
            styled_metrics = self.markov_agent.analyzer.analyze_paragraph(new_para)
            if styled_metrics:
                self.current_markov_signature = styled_metrics['signature']
                print(f"    ✓ Updated Markov state: {styled_metrics['signature'][:50]}...")

            final_output.append(new_para)

        result = "\n\n".join(final_output)

        # Final validation
        original_cites = set(re.findall(r'\[\^\d+\]', text))
        result_cites = set(re.findall(r'\[\^\d+\]', result))
        if original_cites != result_cites:
            print(f"\nWARNING: Final output missing citations: {original_cites - result_cites}")

        # Remove any remaining em dashes before final pass
        has_em_dash, em_dash_count = self._check_em_dashes(result)
        if has_em_dash:
            print(f"\n>> Removing {em_dash_count} em-dash(es) from final output...")
            result = self._remove_em_dashes(result)

        # Phase 5: Final Humanization Pass
        print("\n>> Running final humanization pass...")
        result = self._final_humanization_pass(result)

        # Final em-dash check and removal
        has_em_dash, em_dash_count = self._check_em_dashes(result)
        if has_em_dash:
            print(f">> Removing {em_dash_count} remaining em-dash(es)...")
            result = self._remove_em_dashes(result)

        # COMPREHENSIVE GPTZero validation on complete output
        print("\n>> Running comprehensive GPTZero validation...")
        comp_valid, comp_score, comp_issues, comp_details = self._comprehensive_gptzero_validation(text, result)

        print(f"   Overall score: {comp_score:.1%}")
        if 'vocabulary_repetition' in comp_details:
            vocab_info = comp_details['vocabulary_repetition']
            if vocab_info['repeated_words']:
                print(f"   ! Repetition issues: {list(vocab_info['repeated_words'].keys())[:5]}")
        if 'perplexity' in comp_details:
            ppl_info = comp_details['perplexity']
            print(f"   Perplexity: output={ppl_info['output']:.1f}, original={ppl_info['original']:.1f}")
        if 'vocabulary_diversity' in comp_details:
            div_info = comp_details['vocabulary_diversity']
            print(f"   Vocabulary diversity: {div_info['unique_ratio']:.1%}")
        if 'ai_patterns' in comp_details:
            ai_info = comp_details['ai_patterns']
            print(f"   AI-flagged sentences: {ai_info['flagged_count']}/{ai_info['total_sentences']}")

        if comp_valid:
            print("✓ Final output passes comprehensive GPTZero validation")
        else:
            print(f"⚠ Final output has issues (score: {comp_score:.1%}): {', '.join(comp_issues[:3])}")

        return result

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