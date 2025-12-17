"""
Stage 3: Synthesis Module

Generates new text by combining extracted semantics with target style.
Uses LLM to re-express semantic content in the style of the sample text.

Now includes:
- Structural role awareness for each sentence
- Hint-based refinement for iterative improvement
"""

import json
import os
import requests
from pathlib import Path
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field

from semantic_extractor import SemanticContent, SemanticExtractor
from style_analyzer import StyleProfile, StyleAnalyzer
from structural_analyzer import (
    StructuralAnalyzer,
    TransformationHint,
    SentenceAnalysis,
    StructuralPattern
)
from example_selector import ExampleSelector
from ai_word_replacer import AIWordReplacer
from semantic_word_mapper import SemanticWordMapper
from template_generator import TemplateGenerator, ParagraphTemplate


@dataclass
class SynthesisResult:
    """Result of synthesis operation."""
    output_text: str
    semantic_input: SemanticContent
    style_profile: StyleProfile
    provider_used: str
    model_used: str
    iteration: int = 0
    hints_applied: List[str] = field(default_factory=list)
    template_opener_type: Optional[str] = None  # For tracking variety across paragraphs
    template_opener_phrase: Optional[str] = None  # For exact repetition avoidance
    paragraph_template: Optional[ParagraphTemplate] = None  # Template used for generation


class LLMProvider:
    """Unified interface for LLM providers."""

    def __init__(self, config_path: str = None):
        """Initialize provider from config."""
        if config_path is None:
            config_path = Path(__file__).parent / "config.json"
        else:
            config_path = Path(config_path)

        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = json.load(f)

        self.provider = self.config.get('provider', 'ollama')

        if self.provider == 'deepseek':
            ds_config = self.config.get('deepseek', {})
            self.api_key = ds_config.get('api_key') or os.getenv('DEEPSEEK_API_KEY')
            self.api_url = ds_config.get('api_url', 'https://api.deepseek.com/v1/chat/completions')
            self.model = ds_config.get('editor_model', 'deepseek-chat')
        elif self.provider == 'glm':
            glm_config = self.config.get('glm', {})
            self.api_key = glm_config.get('api_key') or os.getenv('GLM_API_KEY')
            self.api_url = glm_config.get('api_url', 'https://api.z.ai/api/paas/v4/chat/completions')
            self.model = glm_config.get('editor_model', 'glm-4.6')
        elif self.provider == 'ollama':
            ollama_config = self.config.get('ollama', {})
            self.api_url = ollama_config.get('url', 'http://localhost:11434/api/generate')
            self.model = ollama_config.get('editor_model', 'qwen3:32b')
            self.api_key = None
        else:
            raise ValueError(f"Unknown provider: {self.provider}")

    def generate(self, prompt: str, system_prompt: str = "",
                 temperature: float = 0.7, max_tokens: int = 4096) -> str:
        """Generate text using the configured provider."""
        if self.provider == 'ollama':
            return self._call_ollama(prompt, system_prompt, temperature, max_tokens)
        else:
            return self._call_api(prompt, system_prompt, temperature, max_tokens)

    def _call_ollama(self, prompt: str, system_prompt: str,
                     temperature: float, max_tokens: int) -> str:
        """Call Ollama local API."""
        full_prompt = f"{system_prompt}\n\n{prompt}" if system_prompt else prompt

        payload = {
            "model": self.model,
            "prompt": full_prompt,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens
            }
        }

        response = requests.post(self.api_url, json=payload, timeout=120)
        response.raise_for_status()

        result = response.json()
        return result.get('response', '')

    def _call_api(self, prompt: str, system_prompt: str,
                  temperature: float, max_tokens: int) -> str:
        """Call OpenAI-compatible API (DeepSeek, GLM)."""
        messages = []

        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        messages.append({"role": "user", "content": prompt})

        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": False
        }

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }

        response = requests.post(self.api_url, json=payload, headers=headers, timeout=120)
        response.raise_for_status()

        result = response.json()

        if "choices" not in result or len(result.get("choices", [])) == 0:
            return ""

        return result["choices"][0]["message"].get("content", "")


class Synthesizer:
    """
    Synthesizes new text by combining semantic content with target style.

    This is the core of the style transfer: we give the LLM:
    1. The MEANING to express (from semantic extraction)
    2. The STYLE to use (from style analysis)
    3. STRUCTURAL ROLE information for each sentence
    4. Examples of the target style
    5. TRANSFORMATION HINTS from previous iterations

    The LLM then generates new text that expresses the meaning in the target style.
    """

    def __init__(self, config_path: str = None):
        """Initialize synthesizer with LLM provider."""
        self.llm = LLMProvider(config_path)
        self.semantic_extractor = SemanticExtractor()
        self.style_analyzer = StyleAnalyzer()
        self.structural_analyzer = StructuralAnalyzer()

        # Load config for example selector
        if config_path is None:
            config_path = Path(__file__).parent / "config.json"
        else:
            config_path = Path(config_path)

        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = json.load(f)

        # Load sample text for few-shot examples
        # Get sample path from config, with fallback to default
        sample_file = self.config.get("sample", {}).get("file", "prompts/sample.txt")
        sample_path = Path(__file__).parent / sample_file
        if sample_path.exists():
            with open(sample_path, 'r', encoding='utf-8') as f:
                self.sample_text = f.read()
        else:
            print(f"  [Synthesizer] Warning: Sample file not found at {sample_path}")
            self.sample_text = ""

        # Initialize contextual example selector
        self.example_selector = None
        if self.sample_text:
            print("  [Synthesizer] Initializing contextual example selector...")
            self.example_selector = ExampleSelector(self.sample_text, self.config)

        # Initialize AI word replacer (uses sample vocabulary for replacements)
        self.ai_word_replacer = None
        if self.sample_text:
            self.ai_word_replacer = AIWordReplacer(self.sample_text)

        # Initialize semantic word mapper (maps common words to sample vocabulary)
        self.semantic_word_mapper = None
        if self.sample_text:
            # Get config settings with defaults (updated for comprehensive mapping)
            semantic_config = self.config.get("semantic_mapping", {})
            enabled = semantic_config.get("enabled", True)
            similarity_threshold = semantic_config.get("similarity_threshold", 0.3)
            min_sample_frequency = semantic_config.get("min_sample_frequency", 1)
            max_spacy_words = semantic_config.get("max_spacy_words", 10000)
            use_nltk = semantic_config.get("use_nltk", True)
            nltk_corpus = semantic_config.get("nltk_corpus", "brown")
            words_per_pos = semantic_config.get("words_per_pos", 2000)
            cache_mappings = semantic_config.get("cache_mappings", True)

            if enabled:
                print("  [Synthesizer] Initializing semantic word mapper...")
                # Reuse spaCy model from AIWordReplacer if available
                nlp_model = self.ai_word_replacer.nlp if self.ai_word_replacer else None
                self.semantic_word_mapper = SemanticWordMapper(
                    self.sample_text,
                    nlp_model=nlp_model,
                    similarity_threshold=similarity_threshold,
                    min_sample_frequency=min_sample_frequency,
                    max_spacy_words=max_spacy_words,
                    use_nltk=use_nltk,
                    nltk_corpus=nltk_corpus,
                    words_per_pos=words_per_pos,
                    cache_mappings=cache_mappings
                )

        # Cache style profile (needed for template generator)
        self._cached_style_profile = None
        self._cached_role_patterns = None

        # Initialize template generator for structural contracts
        self.template_generator = None
        if self.sample_text:
            print("  [Synthesizer] Initializing template generator...")
            # Get style profile to extract length distribution
            style_profile = self.get_style_profile()
            length_distribution = style_profile.sentences.length_distribution if style_profile else None
            self.template_generator = TemplateGenerator(
                sample_length_distribution=length_distribution,
                config=self.config,
                word_mapper=self.semantic_word_mapper  # Pass word mapper for word-level templates
            )

    def get_style_profile(self) -> StyleProfile:
        """Get or compute style profile for sample text."""
        if self._cached_style_profile is None:
            self._cached_style_profile = self.style_analyzer.analyze(self.sample_text)
        return self._cached_style_profile

    def get_role_patterns(self) -> Dict[str, List[StructuralPattern]]:
        """Get or compute structural role patterns from sample text."""
        if self._cached_role_patterns is None:
            self._cached_role_patterns = self.structural_analyzer.analyze_sample(self.sample_text)
        return self._cached_role_patterns

    def synthesize(self, input_text: str,
                   semantic_content: Optional[SemanticContent] = None,
                   style_profile: Optional[StyleProfile] = None,
                   document_context: Optional[str] = None,
                   preceding_output: Optional[str] = None,
                   transformation_hints: Optional[List[TransformationHint]] = None,
                   iteration: int = 0,
                   position_in_document: Optional[tuple] = None,
                   used_openers: Optional[List[str]] = None,
                   used_phrases: Optional[List[str]] = None,
                   verbose: bool = False) -> SynthesisResult:
        """
        Synthesize new text expressing input meaning in sample style.

        Args:
            input_text: Original text to transform
            semantic_content: Pre-extracted semantics (or None to extract)
            style_profile: Pre-analyzed style (or None to analyze sample)
            document_context: Full document for context (when processing chunks)
            preceding_output: Already-transformed preceding paragraphs (for consistency)
            transformation_hints: Hints from previous iteration to guide refinement
            iteration: Current iteration number
            position_in_document: Tuple of (paragraph_index, total_paragraphs) for template selection
            used_openers: List of opener types already used in this document (for variety)
            used_phrases: List of opener phrases already used (for exact repetition avoidance)
            verbose: Print template information

        Returns:
            SynthesisResult with output text and metadata
        """
        # Extract semantics if not provided
        if semantic_content is None:
            semantic_content = self.semantic_extractor.extract(input_text)

        # Get style profile if not provided
        if style_profile is None:
            style_profile = self.get_style_profile()

        # Get structural role patterns
        role_patterns = self.get_role_patterns()

        # Analyze input structure for role-aware synthesis
        input_structure = self.structural_analyzer.analyze_input_structure(input_text)

        # Generate structural template based on CONTEXT (input + preceding output)
        structural_template = None
        template_opener_type = None
        paragraph_template = None
        if self.template_generator and position_in_document:
            para_idx, total_paras = position_in_document
            position_ratio = para_idx / max(total_paras - 1, 1) if total_paras > 1 else 0.5

            # Determine role based on position
            if para_idx == 0:
                role = 'section_opener'
            elif position_ratio > 0.9:
                role = 'closer'
            elif position_ratio < 0.1:
                role = 'paragraph_opener'
            else:
                role = 'body'

            # Build context: use preceding output as PRIMARY bias
            context_for_template = input_text
            if preceding_output:
                # Use preceding output as primary context (user's requirement)
                context_for_template = preceding_output + "\n\n" + input_text

            # Get template using CONTEXT-BASED selection (not just position)
            # This uses semantic similarity to find similar paragraphs from sample
            claim_count = len(semantic_content.claims)

            # Use context-based method with example selector for semantic similarity
            structural_template = self.template_generator.get_template_prompt_from_context(
                input_text=input_text,
                preceding_output=preceding_output,  # PRIMARY bias
                example_selector=self.example_selector,  # For semantic similarity
                role=role,
                position_ratio=position_ratio,
                claim_count=claim_count,
                used_openers=used_openers,
                used_phrases=used_phrases,
                paragraph_index=para_idx
            )

            # Track opener type from this template for variety tracking
            paragraph_template = self.template_generator.generate_template_from_context(
                input_text=input_text,
                preceding_output=preceding_output,
                example_selector=self.example_selector,
                role=role,
                position_ratio=position_ratio,
                semantic_weight=claim_count // 2,
                used_openers=used_openers,
                used_phrases=used_phrases,
                paragraph_index=para_idx
            )

            # Mark semantic slots if enabled
            if paragraph_template and semantic_content:
                mark_semantic = self.config.get('template_generation', {}).get('mark_semantic_slots', True)
                if mark_semantic:
                    for sent_template in paragraph_template.sentences:
                        if sent_template.has_word_slots:
                            self.template_generator._mark_semantic_slots(sent_template, semantic_content)
            template_opener_phrase = None
            if paragraph_template and paragraph_template.sentences:
                template_opener_type = paragraph_template.sentences[0].opener_type
                # Note: opener phrase will be extracted from output text after synthesis

        # Build the synthesis prompt with structural awareness
        system_prompt = self._build_system_prompt(style_profile, iteration > 0)
        user_prompt = self._build_user_prompt(
            input_text, semantic_content, style_profile,
            document_context=document_context,
            preceding_output=preceding_output,
            input_structure=input_structure,
            role_patterns=role_patterns,
            transformation_hints=transformation_hints,
            iteration=iteration,
            structural_template=structural_template,
            paragraph_template=paragraph_template
        )

        # Adjust temperature based on iteration (more aggressive decrease for convergence)
        # Start at 0.5, decrease faster: 0.5 -> 0.4 -> 0.3 -> 0.2 -> 0.1 (min)
        temperature = max(0.1, 0.5 - (iteration * 0.1))

        # Calculate max_tokens based on input length
        # Style transfer should produce similar or slightly longer text
        input_token_estimate = len(input_text) // 4  # ~4 chars per token
        # Allow 100% buffer to ensure we can generate full-length output (input length + style expansion)
        # For long paragraphs, ensure we have enough tokens
        max_tokens = min(16384, max(4096, int(input_token_estimate * 2.0)))

        # Generate using LLM
        output_text = self.llm.generate(
            prompt=user_prompt,
            system_prompt=system_prompt,
            temperature=temperature,
            max_tokens=max_tokens
        )

        # Clean output (pass input_text to detect fabricated citations)
        output_text = self._clean_output(output_text, input_text)

        # Extract opener phrase from output text for variety tracking
        if output_text and not template_opener_phrase:
            template_opener_phrase = self.template_generator._extract_opener_phrase(output_text)

        # Track which hints were applied
        hints_applied = []
        if transformation_hints:
            for hint in transformation_hints:
                if hint.priority == 1:
                    hints_applied.append(hint.issue[:50])

        return SynthesisResult(
            output_text=output_text,
            semantic_input=semantic_content,
            style_profile=style_profile,
            provider_used=self.llm.provider,
            model_used=self.llm.model,
            iteration=iteration,
            hints_applied=hints_applied,
            template_opener_type=template_opener_type,
            template_opener_phrase=template_opener_phrase,
            paragraph_template=paragraph_template
        )

    def synthesize_from_chunk(self,
                              semantic_chunk: SemanticContent,
                              style_profile: Optional[StyleProfile] = None,
                              document_context: Optional[str] = None,
                              preceding_output: Optional[str] = None,
                              transformation_hints: Optional[List[TransformationHint]] = None,
                              iteration: int = 0,
                              position_in_document: Optional[tuple] = None,
                              used_openers: Optional[List[str]] = None,
                              used_phrases: Optional[List[str]] = None,
                              target_length: int = 0,
                              target_sentence_count: int = 0,
                              verbose: bool = False) -> SynthesisResult:
        """
        Synthesize text from a semantic chunk (regrouped semantic content).

        Similar to synthesize() but takes SemanticContent directly and uses
        target_length/target_sentence_count constraints.

        Args:
            semantic_chunk: Pre-extracted semantic content to synthesize
            style_profile: Pre-analyzed style (or None to analyze sample)
            document_context: Full document for context
            preceding_output: Already-transformed preceding paragraphs
            transformation_hints: Hints from previous iteration
            iteration: Current iteration number
            position_in_document: Tuple of (chunk_index, total_chunks)
            used_openers: List of opener types already used
            used_phrases: List of opener phrases already used
            target_length: Target word count for output
            target_sentence_count: Target sentence count for output
            verbose: Print template information

        Returns:
            SynthesisResult with output text and metadata
        """
        # Get style profile if not provided
        if style_profile is None:
            style_profile = self.get_style_profile()

        # Determine position
        para_idx, total_paras = position_in_document if position_in_document else (0, 1)
        position_ratio = para_idx / max(total_paras - 1, 1)

        # Determine role (will be set by caller, default to body)
        role = 'body'

        # Build system prompt
        system_prompt = self._build_system_prompt(style_profile, is_refinement=(iteration > 0))

        # Build user prompt with semantic content
        # Create a dummy input text for formatting (not used for extraction)
        dummy_input = " ".join([claim.text for claim in semantic_chunk.claims[:3]])

        # Get template if available
        structural_template = ""
        paragraph_template = None
        if self.template_generator:
            paragraph_template = self.template_generator.generate_template_from_context(
                input_text=dummy_input,
                preceding_output=preceding_output,
                example_selector=self.example_selector,
                role=role,
                position_ratio=position_ratio,
                semantic_weight=len(semantic_chunk.claims),
                used_openers=used_openers,
                used_phrases=used_phrases,
            )

            # Mark semantic slots if enabled
            if paragraph_template and semantic_chunk:
                mark_semantic = self.config.get('template_generation', {}).get('mark_semantic_slots', True)
                if mark_semantic:
                    for sent_template in paragraph_template.sentences:
                        if sent_template.has_word_slots:
                            self.template_generator._mark_semantic_slots(sent_template, semantic_chunk)

            if paragraph_template:
                structural_template = paragraph_template.to_constraint_string(used_phrases=used_phrases)

        # Add target length/sentence constraints if provided
        if target_length > 0 or target_sentence_count > 0:
            if structural_template:
                structural_template += "\n\n"
            structural_template += "## TARGET STRUCTURE\n"
            if target_length > 0:
                structural_template += f"Target word count: approximately {target_length} words\n"
            if target_sentence_count > 0:
                structural_template += f"Target sentence count: approximately {target_sentence_count} sentences\n"

        # Get input structure and role patterns (needed for _build_user_prompt)
        input_structure = None
        role_patterns = None
        if self.structural_analyzer:
            # Extract structure from dummy input for formatting
            try:
                input_structure = self.structural_analyzer.analyze_input_structure(dummy_input)
                role_patterns = self.get_role_patterns()
            except:
                pass

        user_prompt = self._build_user_prompt(
            input_text=dummy_input,
            semantic_content=semantic_chunk,
            style_profile=style_profile,
            document_context=document_context,
            preceding_output=preceding_output,
            input_structure=input_structure,
            role_patterns=role_patterns,
            transformation_hints=transformation_hints,
            iteration=iteration,
            structural_template=structural_template,
            paragraph_template=paragraph_template
        )

        # Adjust temperature based on iteration (more aggressive decrease for convergence)
        # Start at 0.5, decrease faster: 0.5 -> 0.4 -> 0.3 -> 0.2 -> 0.1 (min)
        temperature = max(0.1, 0.5 - (iteration * 0.1))

        # Calculate max_tokens based on target length
        max_tokens = min(8192, max(4096, int(target_length * 1.5) if target_length > 0 else 4096))

        # Generate using LLM
        raw_output = self.llm.generate(
            prompt=user_prompt,
            system_prompt=system_prompt,
            temperature=temperature,
            max_tokens=max_tokens
        )

        # Clean output
        cleaned_output = self._clean_output(raw_output, input_text=dummy_input)

        return SynthesisResult(
            output_text=cleaned_output,
            semantic_input=semantic_chunk,
            style_profile=style_profile,
            provider_used=self.llm.provider,
            model_used=self.llm.model,
            iteration=iteration,
            hints_applied=[hint.suggestion for hint in transformation_hints] if transformation_hints else [],
            paragraph_template=paragraph_template
        )

    def _build_system_prompt(self, style_profile: StyleProfile, is_refinement: bool = False) -> str:
        """Build system prompt that defines the synthesis task."""

        if is_refinement:
            prompt = """You are a Style Transfer Refinement Engine. You are refining a previous attempt that did not fully capture the target style.

CRITICAL: Your previous output did NOT sufficiently adopt the target style. You MUST:
1. Apply the TRANSFORMATION HINTS provided - they tell you EXACTLY what to fix
2. Use MORE distinctive patterns from the target style
3. Make sentences LONGER with multiple clauses (target ~30 words average)
4. COMPLETELY REWRITE sentences, don't just adjust wording

"""
        else:
            prompt = """You are a Style Transfer Engine. Your task is to COMPLETELY REWRITE text in a DIFFERENT STYLE.

CRITICAL: You must FULLY TRANSFORM the text, not just make minor edits. The output should:
1. Use COMPLETELY DIFFERENT sentence structures than the input
2. Incorporate the DISTINCTIVE PATTERNS from the target style
3. Be RECOGNIZABLE as the target style, not the input style

"""

        prompt += """## CRITICAL PRIORITY ORDER

**MOST IMPORTANT**: Your output must be COHERENT and MAKE SENSE. Write clear, grammatical sentences that express the claims.

**Priority Order**:
1. **COHERENCE FIRST** - Write sentences that make grammatical and semantic sense. Never sacrifice coherence for anything.
2. **EXPRESS ALL CLAIMS** - Every claim must be clearly expressed with subject + predicate/verb. Use simple, clear language.
3. **Natural flow** - Sentences should flow naturally and be easy to understand.
4. **Style matching** - Only match style characteristics if they don't interfere with coherence. Do NOT force specific phrases or patterns.

**How to Express a Claim**:
- Include the SUBJECT (what the claim is about)
- Include the PREDICATE/VERB (the action or relationship - can be paraphrased)
- Include OBJECTS if present (what is acted upon)

**Important**: You can paraphrase predicates (e.g., "defines" can become "establishes", "determines", "characterizes"). The meaning must be preserved, but exact wording can vary.

## CORE TASK
- Take the semantic content (facts, claims, relationships)
- Express it in COHERENT, GRAMMATICAL sentences that make sense
- Use vocabulary and patterns from the target style, but only when they fit naturally
- Vary your sentence structures - do NOT overuse any single pattern

## 🔒 STRUCTURAL CONTRACT (GUIDELINES, NOT RIGID RULES)
If a STRUCTURAL CONTRACT is provided in the prompt, use it as a GUIDE:
1. Aim for the specified number of sentences
2. Aim for the target word counts for each sentence (±10 words is acceptable)
3. Try to use the specified opener type for each sentence, but only if it makes sense
4. Follow the general sentence structure, but prioritize making coherent sentences

**CRITICAL**: If following the template exactly would create nonsense or ungrammatical text, prioritize coherence. It's better to have a coherent sentence that slightly deviates from the template than a nonsensical sentence that matches it exactly.

## WHAT YOU MUST PRESERVE
1. All facts and claims (meaning) - EXPRESS ALL CLAIMS CLEARLY
2. ONLY citations that exist in the original - DO NOT ADD NEW CITATIONS
3. Technical terms and proper nouns
4. Logical flow and coherence

## WHAT YOU MUST CHANGE
1. Sentence structure - aim for target style patterns, but prioritize coherence
2. Sentence openers - try to match the opener type, but only if it makes sense
3. Sentence length - aim for target word counts, but coherence is more important
4. Discourse markers - Use sparingly: "therefore", "hence", "consequently" should NOT appear in every paragraph
5. Word choice - Use target vocabulary when it fits naturally, don't force it

## CRITICAL: AVOID REPETITION
- Do NOT use the same discourse marker (therefore/hence/consequently) more than twice per page
- Do NOT start every paragraph with "Contrary to..."
- Do NOT add any citations, footnotes, or references that are not in the original text
- Vary your sentence structures throughout the document

## FORBIDDEN (AI-TYPICAL PATTERNS) - NEVER USE THESE
### Words to NEVER use:
delve, leverage, seamless, tapestry, crucial, vibrant, realm, landscape, symphony,
orchestrate, myriad, plethora, paradigm, synergy, holistic, robust, pivotal,
innovative, transformative, streamline, optimize, scalable, nuanced, comprehensive

### Phrases to NEVER use:
- "conventional notions", "local perspective", "internal complexity"
- "grander whole", "intrinsic scaffolding", "broader context"
- "teeming with", "sits within", "operates under"
- "key insight", "importantly", "it is worth noting"
- "we must consider", "we must acknowledge", "consider the fact"
- "this suggests", "this implies", "this indicates"
- "we can see", "we observe", "we find", "we note"
- "it is important to", "it is essential to", "it is clear that"
- "plays a crucial role", "at the end of the day", "first and foremost"
- "a wide range of", "a variety of", "in order to"
- "due to the fact", "for the purpose of", "with regard to"
- "could potentially", "might possibly", "seems to suggest"
- "in essence", "essentially", "basically", "fundamentally"
- "in today's world", "in the realm of", "at its core"
- "to some extent", "to a certain degree", "in some ways"
- "unique perspective", "rich tapestry", "serves as a"

### INSTEAD use direct, assertive language like the target style:
- "Contrary to X, Y holds that..." (use sparingly - not every paragraph)
- "The X therefore holds that..."
- "It follows from this that..."
- Simple declarative statements with strong verbs

## CRITICAL RULES
1. DO NOT ADD CITATIONS - only preserve citations from the original text
2. DO NOT overuse any word or phrase - variety is essential
3. If input has no citations, output should have no citations

## OUTPUT
Return ONLY the rewritten text. No commentary, no explanations, no added citations.
"""

        # Add style guide
        prompt += "\n\n## TARGET STYLE GUIDE\n"
        prompt += style_profile.to_style_guide()

        return prompt

    def _build_user_prompt(self, input_text: str,
                           semantic_content: SemanticContent,
                           style_profile: StyleProfile,
                           document_context: Optional[str] = None,
                           preceding_output: Optional[str] = None,
                           input_structure: Optional[List[SentenceAnalysis]] = None,
                           role_patterns: Optional[Dict[str, List[StructuralPattern]]] = None,
                           transformation_hints: Optional[List[TransformationHint]] = None,
                           iteration: int = 0,
                           structural_template: Optional[str] = None,
                           paragraph_template: Optional[ParagraphTemplate] = None) -> str:
        """Build user prompt with semantic content, structural roles, examples, hints, and template."""
        parts = []

        # Add structural template FIRST - this is the contract they must follow
        if structural_template:
            parts.append(structural_template)
            parts.append("")

        # Add transformation hints if this is a refinement iteration (PROGRESSIVE FEEDING)
        if transformation_hints and iteration > 0:
            parts.append("## ⚠️ FOCUS REQUIREMENTS (PROGRESSIVE APPROACH)")
            parts.append("You are working on a PROGRESSIVE refinement process. Focus ONLY on these requirements:")
            parts.append("Once these are met, we will address additional requirements in subsequent iterations.")
            parts.append("")

            # Show only the active requirements (already filtered by RequirementTracker)
            # These are the 1-2 most critical requirements for this iteration
            for i, hint in enumerate(transformation_hints, 1):
                priority_marker = "🔴 CRITICAL" if hint.priority == 1 else "🟡 IMPORTANT" if hint.priority == 2 else "🟢 MINOR"
                parts.append(f"### Requirement {i}: {priority_marker}")
                parts.append(f"**Issue**: {hint.issue}")
                parts.append(f"**Fix**: {hint.suggestion}")

                # Add before/after example if available
                if hint.current_text and hint.expected_patterns:
                    parts.append(f"**Current**: \"{hint.current_text[:80]}...\"")
                    if hint.expected_patterns:
                        parts.append(f"**Expected pattern**: {hint.expected_patterns[0]}")
                parts.append("")

            parts.append("**IMPORTANT**: Focus ONLY on fixing these requirements. Do not worry about other issues yet.")
            parts.append("")

        # TEMPORARILY DISABLED: Style examples are causing gibberish by forcing pattern mimicry
        # The LLM is creating nonsensical variations of sample phrases
        # Focus on coherence first, style will follow naturally from the simplified guidance below

        # Add minimal style guidance (statistical only, no examples)
        parts.append("## STYLE GUIDANCE (GENERAL CHARACTERISTICS ONLY)")
        parts.append("**IMPORTANT**: These are general characteristics, NOT exact phrases to copy.")
        parts.append("Do NOT force specific phrases from the sample. Write naturally and coherently.")
        parts.append("")
        if style_profile:
            formality = style_profile.vocabulary.formality_score if style_profile.vocabulary else 0.5
            avg_length = style_profile.sentences.avg_length if style_profile.sentences else 20
            parts.append(f"- Formality level: {formality:.2f}")
            parts.append(f"- Average sentence length: ~{avg_length:.0f} words")
            parts.append("- Use varied sentence lengths (mix of short, medium, long)")
            parts.append("- Avoid repetitive sentence structures")
        parts.append("")
        parts.append("**CRITICAL**: If trying to match a style pattern would create nonsense, DON'T use it.")
        parts.append("Coherent, clear sentences are more important than style matching.")

        # TEMPORARILY DISABLED: Structural role guidance is too prescriptive and causing gibberish
        # The LLM is forcing patterns that don't fit the semantic content
        # Focus on expressing claims clearly first

        # Add document context if processing a chunk (SIMPLIFIED - only if needed for references)
        # Removed full document context to reduce prompt complexity and improve convergence
        # Preceding paragraph provides sufficient context for consistency

        # Add preceding transformed output for consistency (LIMITED to last paragraph only)
        if preceding_output:
            # Only include the last paragraph to reduce context size
            preceding_paras = preceding_output.split('\n\n')
            last_para = preceding_paras[-1] if preceding_paras else preceding_output
            parts.append("\n\n## PRECEDING PARAGRAPH (for consistency)")
            parts.append("The immediately preceding paragraph (match the voice and tone):")
            parts.append(f"\n{last_para}")
            parts.append("\n(Match the vocabulary choices and tone established above.)")

        # Add semantic content summary (FOCUSED - only chunk claims)
        parts.append("\n\n## 🔒 SEMANTIC CONTENT TO EXPRESS")
        parts.append(self._format_semantic_content(semantic_content))

        # Add preserved elements
        parts.append("\n\n## MUST PRESERVE EXACTLY")
        preserved = semantic_content.preserved_elements
        if preserved.get('citations'):
            parts.append(f"- Citations: {', '.join(preserved['citations'])}")
        if preserved.get('technical_terms'):
            parts.append(f"- Technical terms: {', '.join(preserved['technical_terms'][:10])}")
        if preserved.get('quoted_text'):
            parts.append(f"- Quotes: {', '.join(preserved['quoted_text'][:5])}")

        # Add the original text to rewrite
        if document_context:
            parts.append("\n\n## PARAGRAPH TO REWRITE (from the document above)")
        else:
            parts.append("\n\n## ORIGINAL TEXT TO REWRITE")
        parts.append(input_text)

        # Add explicit length requirement
        input_word_count = len(input_text.split())
        # Use semantic extractor's nlp for sentence counting
        doc = self.semantic_extractor.nlp(input_text)
        input_sentence_count = len(list(doc.sents))

        # Add simplified self-verification step
        parts.append("\n\n## ✅ CHECK BEFORE SUBMITTING")
        total_claims = len(semantic_content.claims)
        parts.append(f"Before submitting, verify:")
        parts.append(f"1. All {total_claims} claim(s) are expressed (subject + verb + objects)")
        if paragraph_template:
            parts.append(f"2. You generated {paragraph_template.sentence_count} sentence(s)")
        parts.append("3. Output matches the target style")
        parts.append("")

        parts.append("\n\n## LENGTH REQUIREMENT (CRITICAL)")
        parts.append(f"**Your output MUST be approximately the same length as the input:**")
        parts.append(f"- Input: ~{input_word_count} words, ~{input_sentence_count} sentences")
        parts.append(f"- Output target: ~{input_word_count} words (±10%), ~{input_sentence_count} sentences (±1)")
        parts.append(f"**DO NOT shorten or condense the content. Express all claims with full detail.**")
        parts.append(f"If the input is long and detailed, your output must also be long and detailed.")

        # Final instruction - simplified
        parts.append("\n\n## YOUR OUTPUT")
        if preceding_output:
            parts.append(f"Rewrite the paragraph above in the target style. Express ALL claims clearly. Match the style of the example. Output approximately {input_word_count} words. Output only the rewritten paragraph.")
        else:
            parts.append(f"Rewrite the original text in the target style. Express ALL claims clearly. Match the style of the example. Output approximately {input_word_count} words. Output only the rewritten text.")

        return '\n'.join(parts)

    def _format_pattern_guidance(self, style_profile: StyleProfile) -> str:
        """Format distinctive patterns as explicit guidance for the LLM."""
        parts = []

        if not style_profile.distinctive_patterns:
            return ""

        dp = style_profile.distinctive_patterns

        parts.append("\n\n## SYNTACTIC PATTERNS TO USE")
        parts.append("Apply these constructions where appropriate to match the target style:")

        # Add syntactic constructions with clear templates
        if dp.syntactic_constructions:
            parts.append("\n### Sentence Templates")
            for i, construction in enumerate(dp.syntactic_constructions[:6], 1):
                parts.append(f"\n{i}. **{construction.construction_type.upper()}**: {construction.pattern}")
                if construction.examples:
                    # Show a truncated example
                    ex = construction.examples[0]
                    ex_display = ex[:120] + "..." if len(ex) > 120 else ex
                    parts.append(f"   Use like: \"{ex_display}\"")

        # Add key phrasal patterns
        if dp.phrasal_patterns:
            parts.append("\n### Characteristic Phrases to Incorporate")
            top_phrases = [p for p in dp.phrasal_patterns[:8]]
            for pattern in top_phrases:
                parts.append(f"- \"{pattern.phrase}\" (at {pattern.position} of sentence)")

        # Add discourse markers with positions
        if dp.discourse_markers:
            parts.append("\n### Discourse Markers to Use")
            for marker in dp.discourse_markers[:8]:
                parts.append(f"- \"{marker.marker}\" → place at {marker.typical_position}")

        return '\n'.join(parts)

    def _format_structural_guidance(self,
                                     input_structure: List[SentenceAnalysis],
                                     role_patterns: Dict[str, List[StructuralPattern]]) -> str:
        """Format structural role guidance for each sentence."""
        parts = []
        parts.append("\n\n## STRUCTURAL ROLE GUIDANCE")
        parts.append("Each sentence has a structural role. Apply patterns appropriate for that role:")

        for sent in input_structure:
            role = sent.structural_role.role
            patterns = self.structural_analyzer.get_patterns_for_role(role, role_patterns)

            # Get key patterns for this role
            phrase_patterns = [p for p in patterns if p.pattern_type == 'phrase'][:2]
            opener_patterns = [p for p in patterns if p.pattern_type == 'opener'][:2]

            parts.append(f"\n**Sentence {sent.index + 1}** (role: {role}):")
            parts.append(f"   Original: \"{sent.text[:80]}...\"")

            if role == 'section_opener':
                parts.append("   → MUST start with 'Contrary to...' or 'The principal features...'")
                parts.append("   → Should be 30-40 words with subordinate clauses")
            elif role == 'paragraph_opener':
                parts.append("   → Should use opener pattern like 'Hence,...' or 'The X method therefore...'")
                parts.append("   → Sets up the paragraph's main point")
            elif role == 'paragraph_closer':
                parts.append("   → Should use concluding pattern with 'therefore', 'hence', or 'consequently'")
            elif role == 'body':
                parts.append("   → Support sentence - elaborate, provide evidence, or transition")

            if phrase_patterns:
                phrases = [p.pattern for p in phrase_patterns]
                parts.append(f"   Suggested phrases: {', '.join(phrases)}")

        return '\n'.join(parts)

    def _format_semantic_content(self, content: SemanticContent) -> str:
        """Format semantic content for the prompt."""
        parts = []

        # Key claims - list ALL claims explicitly with clear numbering
        total_claims = len(content.claims)
        parts.append(f"## CLAIMS TO EXPRESS (EXACT COUNT: {total_claims})")
        parts.append(f"**CRITICAL**: You must express EXACTLY {total_claims} claim(s). Each claim must be clearly and separately expressed.")
        parts.append("")
        parts.append("**HOW TO EXPRESS A CLAIM:**")
        parts.append("- Include the SUBJECT (what the claim is about)")
        parts.append("- Include the PREDICATE/VERB (the action or relationship)")
        parts.append("- Include OBJECTS if present (what is acted upon)")
        parts.append("- You can paraphrase, but the core meaning must be present")
        parts.append("")

        # Add explicit examples of good and bad claim expression
        parts.append("**EXAMPLES OF CLAIM EXPRESSION:**")
        parts.append("")
        parts.append("✅ GOOD EXAMPLE:")
        parts.append("  Input Claim: Subject='U.S. strategy', Predicate='signals', Objects=['transition']")
        parts.append("  Good Output: 'The U.S. strategy signals a transition toward...'")
        parts.append("  ✓ Subject present: 'U.S. strategy'")
        parts.append("  ✓ Predicate present: 'signals' (or semantically similar: 'indicates', 'shows')")
        parts.append("  ✓ Object present: 'transition'")
        parts.append("")
        parts.append("❌ BAD EXAMPLE (Missing Predicate):")
        parts.append("  Input Claim: Subject='U.S. strategy', Predicate='signals', Objects=['transition']")
        parts.append("  Bad Output: 'The U.S. strategy involves a transition...'")
        parts.append("  ✗ Subject present: 'U.S. strategy' ✓")
        parts.append("  ✗ Predicate MISSING: 'signals' not expressed (only 'involves' which is different)")
        parts.append("  ✗ Object present: 'transition' ✓")
        parts.append("  Result: CLAIM NOT FULLY EXPRESSED")
        parts.append("")
        parts.append("✅ GOOD EXAMPLE (Paraphrased):")
        parts.append("  Input Claim: Subject='U.S. strategy', Predicate='signals', Objects=['transition']")
        parts.append("  Good Output: 'The American approach indicates a transition toward...'")
        parts.append("  ✓ Subject paraphrased but present: 'American approach' ≈ 'U.S. strategy'")
        parts.append("  ✓ Predicate semantically similar: 'indicates' ≈ 'signals'")
        parts.append("  ✓ Object present: 'transition'")
        parts.append("")
        parts.append("")

        for i, claim in enumerate(content.claims, 1):
            confidence_str = "certain" if claim.confidence > 0.8 else "likely" if claim.confidence > 0.5 else "possible"
            citations_str = f" {' '.join(claim.citations)}" if claim.citations else ""

            # Break down claim components for clarity
            claim_parts = []
            if claim.subject:
                claim_parts.append(f"Subject: {claim.subject}")
            if claim.predicate:
                claim_parts.append(f"Predicate: {claim.predicate}")
            if claim.objects:
                claim_parts.append(f"Objects: {', '.join(claim.objects)}")

            parts.append(f"**Claim {i}/{total_claims}** [{confidence_str}]:")
            parts.append(f"  Full text: {claim.text}{citations_str}")
            if claim_parts:
                parts.append(f"  Components: {' | '.join(claim_parts)}")
            parts.append("")  # Blank line between claims for clarity

        if total_claims > 0:
            parts.append(f"**SELF-VERIFICATION CHECKLIST (VERIFY BEFORE SUBMITTING):**")
            parts.append("")
            parts.append("Before you submit your output, verify each of the following:")
            parts.append("")
            for i, claim in enumerate(content.claims, 1):
                parts.append(f"Claim {i}: '{claim.text[:60]}{'...' if len(claim.text) > 60 else ''}'")
                parts.append(f"  □ Subject '{claim.subject}' is present (or clearly paraphrased)")
                if claim.predicate:
                    parts.append(f"  □ Predicate '{claim.predicate}' is present (or semantically similar verb)")
                if claim.objects:
                    parts.append(f"  □ At least one object from {claim.objects} is present")
                parts.append("")
            parts.append(f"**FINAL CHECK:**")
            parts.append(f"  □ All {total_claims} claim(s) are present and fully expressed")
            parts.append(f"  □ Each claim has subject AND predicate/verb (or subject AND object)")
            parts.append("")
            parts.append(f"**CRITICAL REMINDER**:")
            parts.append(f"- ALL {total_claims} claim(s) must appear in your output")
            parts.append(f"- Each claim must have its subject and predicate/verb clearly expressed")
            parts.append(f"- You can paraphrase predicates (e.g., 'signals' → 'indicates', 'shows', 'demonstrates')")
            parts.append(f"- Do NOT combine claims into one sentence")
            parts.append(f"- Do NOT omit any claims")
            parts.append(f"- Do NOT condense claims - express each one fully")

        # Key relationships
        if content.relationships:
            parts.append("\n### Logical Relationships (preserve these connections):")
            for rel in content.relationships[:10]:
                parts.append(f"- {rel.relation_type}: {rel.source[:50]}... → {rel.target[:50]}...")

        # Paragraph structure
        parts.append("\n### Paragraph Structure (maintain similar organization):")
        for para in content.paragraph_structure:
            parts.append(f"- Para {para['index'] + 1}: {para['function']} ({para['sentence_count']} sentences)")
            if para.get('key_concepts'):
                parts.append(f"  Concepts: {', '.join(para['key_concepts'][:5])}")

        return '\n'.join(parts)

    def _get_contextual_examples(self, input_text: str) -> List[str]:
        """
        Get contextually relevant examples for the input text.

        Uses the ExampleSelector to find sample paragraphs that are most
        structurally and semantically similar to the input being transformed.
        Falls back to static excerpts if selector is unavailable.

        Args:
            input_text: The text being transformed

        Returns:
            List of relevant example paragraphs from sample text
        """
        # Use contextual selector if available
        if self.example_selector:
            # Use diverse selection to get relevant but varied examples
            examples = self.example_selector.select_diverse_examples(input_text)
            if examples:
                return examples

        # Fallback to static excerpts
        return self._get_sample_excerpts_static()

    def _get_sample_excerpts_static(self) -> List[str]:
        """Fallback: Get static excerpts from sample text."""
        if not self.sample_text:
            return []

        # Split into paragraphs
        paragraphs = [p.strip() for p in self.sample_text.split('\n\n') if p.strip()]

        # Filter to reasonable length paragraphs
        excerpts = []
        for para in paragraphs:
            word_count = len(para.split())
            if 50 <= word_count <= 200:
                excerpts.append(para)
                if len(excerpts) >= 5:
                    break

        return excerpts[:3]

    def _clean_output(self, text: str, input_text: str = "") -> str:
        """
        Clean LLM output and remove AI fingerprints.

        Uses AIWordReplacer to find contextually appropriate replacements
        from the sample text vocabulary instead of hardcoded alternatives.

        Args:
            text: The LLM output to clean
            input_text: The original input (to check for citations)
        """
        import re

        # Remove common prefixes/suffixes
        prefixes_to_remove = [
            "Here is the rewritten text:",
            "Here's the rewritten text:",
            "Rewritten text:",
            "Output:",
            "Here is",
            "Here's"
        ]

        text = text.strip()

        for prefix in prefixes_to_remove:
            if text.lower().startswith(prefix.lower()):
                text = text[len(prefix):].strip()

        # Remove markdown code blocks if present
        if text.startswith('```'):
            lines = text.split('\n')
            # Remove first and last ``` lines
            if lines[0].startswith('```'):
                lines = lines[1:]
            if lines and lines[-1].strip() == '```':
                lines = lines[:-1]
            text = '\n'.join(lines)

        # Remove fabricated citations if input had none
        if input_text:
            input_has_citations = bool(re.search(r'\[\^?\d+\]', input_text))
            if not input_has_citations:
                # First remove citation definitions at the end [^1]: text or [1]: text
                # This must come first to avoid partial matches
                text = re.sub(r'^\[\^?\d+\]:.*$', '', text, flags=re.MULTILINE)
                # Then remove inline footnote citations [^1] or [1]
                text = re.sub(r'\s*\[\^?\d+\]', '', text)
                # Clean up multiple blank lines
                text = re.sub(r'\n{3,}', '\n\n', text)
                # Clean up any leftover blank lines at the end
                text = text.strip()

        # Use AIWordReplacer for contextual replacements from sample vocabulary
        if self.ai_word_replacer:
            text = self.ai_word_replacer.replace_ai_words(text)

        # Apply semantic word mapping (maps common words to sample vocabulary)
        if self.semantic_word_mapper:
            text = self.semantic_word_mapper.apply_mapping(text)

        # Additional phrase-level cleanup (stylistic hedges not in AI word list)
        stylistic_fixes = {
            r'\bwe must consider\b': 'it is necessary to examine',
            r'\bwe must acknowledge\b': 'one must recognize',
            r'\bwe must recognize\b': 'one must recognize',
            r'\bconsider the fact\b': 'observe',
            r'\bconsider how\b': 'examine how',
            r'\bconsider that\b': 'recognize that',
            r'\bit is important to\b': 'it is necessary to',
            r'\bit is essential to\b': 'it is necessary to',
            r'\bit is clear that\b': 'clearly',
            r'\bit is evident that\b': 'evidently',
            r'\bit is obvious that\b': 'plainly',
            r'\bthis suggests\b': 'this demonstrates',
            r'\bthis implies\b': 'this indicates',
            r'\bthis indicates\b': 'this shows',
            r'\bwe can see\b': 'it is apparent',
            r'\bwe observe\b': 'one observes',
            r'\bwe find\b': 'one finds',
            r'\bwe note\b': 'one notes',
            r'\bcould potentially\b': 'may',
            r'\bmight possibly\b': 'may',
            r'\bseems to suggest\b': 'indicates',
            r'\bappears to be\b': 'is',
            r'\btends to be\b': 'is generally',
            r'\bin essence\b': '',
            r'\bessentially\b': '',
            r'\bbasically\b': '',
            r'\bthe fact that\b': 'that',
        }

        for pattern, replacement in stylistic_fixes.items():
            text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)

        # Reduce overused discourse markers
        text = self._reduce_repetition(text)

        return text.strip()

    def _reduce_repetition(self, text: str) -> str:
        """
        Reduce overused words by replacing excess occurrences with alternatives.

        The sample text uses discourse markers sparingly:
        - "consequently": ~5 times in 400+ lines
        - "therefore": ~11 times
        - "hence": ~0 times

        We should match this frequency, not overuse them.
        """
        import re

        # Words that get overused and their alternatives
        overused_words = {
            'consequently': ['as a result', 'thus', 'accordingly', ''],
            'therefore': ['thus', 'accordingly', 'so', ''],
            'hence': ['thus', 'accordingly', 'so', ''],
            'however': ['yet', 'but', 'still', ''],
        }

        # Maximum allowed occurrences per ~500 words (roughly one page)
        word_count = len(text.split())
        max_per_page = 2
        max_occurrences = max(2, (word_count // 500) * max_per_page)

        for word, alternatives in overused_words.items():
            pattern = re.compile(r'\b' + word + r'\b', re.IGNORECASE)

            # Count occurrences
            matches = list(pattern.finditer(text))
            excess = len(matches) - max_occurrences

            if excess > 0:
                # Replace excess occurrences from the end backwards
                # This avoids index shifting issues
                alt_idx = 0
                for match in reversed(matches[max_occurrences:]):
                    replacement = alternatives[alt_idx % len(alternatives)]
                    alt_idx += 1

                    # Match case
                    if match.group(0)[0].isupper():
                        replacement = replacement.capitalize() if replacement else ''

                    # Replace this specific occurrence
                    start, end = match.start(), match.end()
                    text = text[:start] + replacement + text[end:]

        # Clean up any double spaces or punctuation issues from empty replacements
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'\s+([,.])', r'\1', text)
        text = re.sub(r',\s*,', ',', text)
        text = re.sub(r'\.\s*,', '.', text)

        return text

    def synthesize_paragraph(self, paragraph: str,
                             style_profile: Optional[StyleProfile] = None) -> str:
        """
        Synthesize a single paragraph (for chunked processing).

        Args:
            paragraph: Single paragraph to transform
            style_profile: Pre-analyzed style profile

        Returns:
            Transformed paragraph
        """
        result = self.synthesize(paragraph, style_profile=style_profile)
        return result.output_text

    def synthesize_chunked(self, text: str,
                           chunk_by: str = 'paragraph') -> str:
        """
        Synthesize text by processing it in chunks.

        This is useful for long documents where processing the entire
        text at once might exceed context limits.

        Args:
            text: Full text to transform
            chunk_by: How to chunk ('paragraph' or 'section')

        Returns:
            Full transformed text
        """
        style_profile = self.get_style_profile()

        if chunk_by == 'paragraph':
            paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
            transformed = []

            for i, para in enumerate(paragraphs):
                print(f"Processing paragraph {i + 1}/{len(paragraphs)}...")
                result = self.synthesize(para, style_profile=style_profile)
                transformed.append(result.output_text)

            return '\n\n'.join(transformed)

        else:  # chunk_by == 'section'
            # Split by headers or multiple newlines
            import re
            sections = re.split(r'\n{3,}|(?=^#+\s)', text, flags=re.MULTILINE)
            sections = [s.strip() for s in sections if s.strip()]

            transformed = []
            for i, section in enumerate(sections):
                print(f"Processing section {i + 1}/{len(sections)}...")
                result = self.synthesize(section, style_profile=style_profile)
                transformed.append(result.output_text)

            return '\n\n\n'.join(transformed)


def synthesize_text(input_text: str, config_path: str = None) -> str:
    """Convenience function to synthesize text."""
    synthesizer = Synthesizer(config_path)
    result = synthesizer.synthesize(input_text)
    return result.output_text


# Test function
if __name__ == '__main__':
    test_text = """Human experience reinforces the rule of finitude. The biological cycle of birth, life, and decay defines our reality. Every object we touch eventually breaks. Every star burning in the night sky eventually succumbs to erosion. But we encounter a logical trap when we apply that same finiteness to the universe itself. A cosmos with a definitive beginning and a hard boundary implies a container. Logic demands we ask a difficult question. If the universe has edges, what exists outside them?

A truly finite universe must exist within a larger context. Anything with limits implies the existence of an exterior. A bottle possesses a finite volume because the bottle sits within a room. The room exists within a house. The observable universe sits within the expanse of the greater cosmos. We must consider the possibility that our reality is a component of a grander whole."""

    print("=== Synthesizer Test ===\n")
    print("Input text:")
    print(test_text[:200] + "...\n")

    try:
        synthesizer = Synthesizer()
        print(f"Using provider: {synthesizer.llm.provider}")
        print(f"Using model: {synthesizer.llm.model}")
        print("\nGenerating synthesis...")

        result = synthesizer.synthesize(test_text)

        print("\n=== Output ===\n")
        print(result.output_text)

    except Exception as e:
        print(f"Error: {e}")
        print("(This is expected if no API key is configured)")

