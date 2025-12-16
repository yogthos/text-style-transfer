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

        # Load sample text for few-shot examples
        sample_path = Path(__file__).parent / "prompts" / "sample.txt"
        if sample_path.exists():
            with open(sample_path, 'r', encoding='utf-8') as f:
                self.sample_text = f.read()
        else:
            self.sample_text = ""

        # Cache style profile
        self._cached_style_profile = None
        self._cached_role_patterns = None

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
                   iteration: int = 0) -> SynthesisResult:
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

        # Build the synthesis prompt with structural awareness
        system_prompt = self._build_system_prompt(style_profile, iteration > 0)
        user_prompt = self._build_user_prompt(
            input_text, semantic_content, style_profile,
            document_context=document_context,
            preceding_output=preceding_output,
            input_structure=input_structure,
            role_patterns=role_patterns,
            transformation_hints=transformation_hints,
            iteration=iteration
        )

        # Adjust temperature based on iteration (more deterministic as we refine)
        temperature = max(0.3, 0.7 - (iteration * 0.1))

        # Generate using LLM
        output_text = self.llm.generate(
            prompt=user_prompt,
            system_prompt=system_prompt,
            temperature=temperature,
            max_tokens=4096
        )

        # Clean output
        output_text = self._clean_output(output_text)

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
            hints_applied=hints_applied
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

        prompt += """## CORE TASK
- Take the semantic content (facts, claims, relationships)
- Express it using the EXACT patterns, phrases, and constructions of the target style
- EVERY sentence should reflect the target style's characteristics

## WHAT YOU MUST PRESERVE
1. All facts and claims (meaning)
2. All citations [^N]
3. Technical terms and proper nouns
4. Logical flow

## WHAT YOU MUST CHANGE (MANDATORY)
1. Sentence structure - MUST use target style patterns
2. Sentence openers - Use "Contrary to...", "Hence,...", "The X method therefore..."
3. Sentence length - Target 25-35 words with multiple clauses
4. Discourse markers - Use "therefore", "hence", "consequently", "however"
5. Word choice - Match target vocabulary

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
- "Contrary to X, Y holds that..."
- "The X method therefore holds that..."
- "Hence, it is clear that..."
- "It follows from this that..."

## OUTPUT
Return ONLY the rewritten text. No commentary, no explanations.
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
                           iteration: int = 0) -> str:
        """Build user prompt with semantic content, structural roles, examples, and hints."""
        parts = []

        # Add transformation hints FIRST if this is a refinement iteration
        if transformation_hints and iteration > 0:
            parts.append("## ⚠️ TRANSFORMATION HINTS (MUST ADDRESS)")
            parts.append("Your previous output had these issues. FIX THEM:")
            for hint in transformation_hints[:10]:  # Top 10 hints
                priority_marker = "🔴 CRITICAL" if hint.priority == 1 else "🟡 IMPORTANT" if hint.priority == 2 else "🟢 MINOR"
                parts.append(f"\n{priority_marker}: {hint.issue}")
                parts.append(f"   → FIX: {hint.suggestion}")
            parts.append("")

        # Add example passages from sample text
        parts.append("## STYLE EXAMPLES (your output MUST look like these)")
        examples = self._get_sample_excerpts()
        for i, excerpt in enumerate(examples[:3], 1):
            parts.append(f"\n### Example {i}:")
            parts.append(excerpt)

        # Add pattern templates to apply
        parts.append(self._format_pattern_guidance(style_profile))

        # Add structural role guidance
        if input_structure and role_patterns:
            parts.append(self._format_structural_guidance(input_structure, role_patterns))

        # Add document context if processing a chunk
        if document_context:
            parts.append("\n\n## FULL DOCUMENT CONTEXT")
            parts.append("This paragraph is part of a larger document. Here is the full document for context:")
            parts.append(f"\n{document_context}")
            parts.append("\n(Use this context to understand references, maintain terminology consistency, and preserve cross-paragraph connections.)")

        # Add preceding transformed output for consistency
        if preceding_output:
            parts.append("\n\n## ALREADY TRANSFORMED PARAGRAPHS")
            parts.append("These paragraphs have already been transformed. Continue in the same voice and maintain consistency:")
            parts.append(f"\n{preceding_output}")
            parts.append("\n(Match the vocabulary choices and tone established above.)")

        # Add semantic content summary
        parts.append("\n\n## SEMANTIC CONTENT TO EXPRESS")
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

        # Final instruction
        parts.append("\n\n## YOUR OUTPUT")
        if preceding_output:
            parts.append("Rewrite ONLY the paragraph above in the target style. Use the syntactic constructions and phrases provided. Maintain consistency with the already-transformed paragraphs. Express all the same meaning using the vocabulary, sentence patterns, and tone established. Output only the rewritten paragraph, nothing else.")
        else:
            parts.append("Rewrite the original text in the target style. ACTIVELY USE the syntactic constructions and characteristic phrases provided. Express all the same meaning using the vocabulary, sentence patterns, and tone of the style examples. Output only the rewritten text.")

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

        # Key claims
        parts.append("### Key Claims (must all be expressed):")
        for i, claim in enumerate(content.claims[:15], 1):
            confidence_str = "certain" if claim.confidence > 0.8 else "likely" if claim.confidence > 0.5 else "possible"
            citations_str = f" {' '.join(claim.citations)}" if claim.citations else ""
            parts.append(f"{i}. [{confidence_str}] {claim.text}{citations_str}")

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

    def _get_sample_excerpts(self) -> List[str]:
        """Get representative excerpts from sample text."""
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

        return excerpts

    def _clean_output(self, text: str) -> str:
        """Clean LLM output and remove AI fingerprints."""
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

        # POST-PROCESSING: Replace stubborn AI patterns with better alternatives
        ai_replacements = {
            # Phrases with direct replacements
            r'\bconventional notions\b': 'prevailing views',
            r'\blocal perspective\b': 'our particular standpoint',
            r'\binternal complexity\b': 'inner structure',
            r'\bgrander whole\b': 'greater totality',
            r'\bintrinsic scaffolding\b': 'inherent structure',
            r'\bbroader context\b': 'wider conditions',
            r'\bkey insight\b': 'central point',
            r'\bteeming with\b': 'containing',
            r'\bsits within\b': 'exists within',
            r'\boperates under\b': 'functions according to',
            r'\bunique perspective\b': 'distinct viewpoint',
            r'\brich tapestry\b': 'complex system',
            r'\bserves as a\b': 'constitutes a',
            r'\bstands as a\b': 'represents a',
            r'\bacts as a reminder\b': 'demonstrates',
            r'\bin the realm of\b': 'in the domain of',
            r'\bat its core\b': 'fundamentally',
            r'\bwhen it comes to\b': 'regarding',
            r'\bin today\'s world\b': 'under present conditions',
            r'\bmoving forward\b': 'henceforth',
            r'\bat the end of the day\b': 'ultimately',
            r'\bthe fact that\b': 'that',
            r'\bin order to\b': 'to',
            r'\bdue to the fact\b': 'because',
            r'\bfor the purpose of\b': 'for',
            r'\bwith regard to\b': 'concerning',
            r'\ba wide range of\b': 'various',
            r'\ba variety of\b': 'various',
            r'\bplays a crucial role\b': 'is essential',
            r'\bis of utmost importance\b': 'is essential',
            r'\bit goes without saying\b': 'clearly',
            r'\bneedless to say\b': 'clearly',
            r'\blast but not least\b': 'finally',
            r'\bfirst and foremost\b': 'primarily',
            # Weak phrases - remove or simplify
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
            r'\bfundamentally\b': 'in principle',
            r'\bto some extent\b': 'partially',
            r'\bto a certain degree\b': 'partially',
            r'\bin some ways\b': 'partially',
            r'\bin many ways\b': 'in several respects',
        }

        for pattern, replacement in ai_replacements.items():
            text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)

        return text.strip()

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

