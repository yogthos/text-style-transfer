"""Prompt builder for RAG-based style transfer.

This module constructs highly constrained prompts using RAG data to prevent
'LLM Slop' by explicitly separating vocabulary guidance (situation match)
from structure guidance (structure match).
"""

import random
import re
import textwrap
from pathlib import Path
from typing import Optional, Dict, List, Tuple

from src.analyzer.style_metrics import get_style_vector


def _load_prompt_template(template_name: str) -> str:
    """Load a prompt template from the prompts directory.

    Args:
        template_name: Name of the template file (e.g., 'generator_system.md')

    Returns:
        Template content as string.
    """
    prompts_dir = Path(__file__).parent.parent.parent / "prompts"
    template_path = prompts_dir / template_name
    if not template_path.exists():
        raise FileNotFoundError(f"Prompt template not found: {template_path}")
    return template_path.read_text().strip()


def sanitize_structural_reference(text: str) -> str:
    """
    Strips dialogue tags and metadata artifacts from structural templates.

    Removes patterns like:
    - "August: No, it's an experience!" → "No, it's an experience!"
    - "Schneider: And that's a slogan?" → "And that's a slogan?"
    - "15. Some text" → "Some text"
    - "[1] Some text" → "Some text"

    Args:
        text: Structural reference text that may contain dialogue tags or citations.

    Returns:
        Cleaned text with dialogue tags and citations removed.
    """
    if not text:
        return text

    # Regex to remove "Name:" or "Name [ACTION]:" at the start
    # Matches "Word:", "Word Word:", "WORD:" followed by space
    # Examples: "August:", "Tony Febbo:", "SCHNEIDER:"
    clean_text = re.sub(r'^[A-Z][a-z]+(?:\s+[A-Za-z]+)?:\s*', '', text)

    # Also strip leading citation numbers like "15." or "[1]"
    clean_text = re.sub(r'^(?:\[?\d+\]?\.?)\s*', '', clean_text)

    return clean_text.strip()


def _analyze_structure(text: str) -> Dict[str, any]:
    """Analyze structural features of a text snippet.

    Args:
        text: Text to analyze.

    Returns:
        Dictionary with structural features:
        - word_count: Total word count
        - sentence_count: Number of sentences
        - avg_sentence_len: Average words per sentence
        - punctuation_pattern: List of punctuation marks used
        - clause_count: Estimated number of clauses
        - voice: "active" or "passive" (heuristic)
        - has_dashes: Whether text contains dashes
        - has_semicolons: Whether text contains semicolons
        - has_parentheses: Whether text contains parentheses
        - has_asterisks: Whether text contains asterisks
        - complexity: "simple", "compound", "complex", or "compound-complex"
    """
    words = text.split()
    word_count = len(words)

    # Count sentences
    sentence_endings = text.count('.') + text.count('!') + text.count('?')
    sentence_count = max(1, sentence_endings)
    avg_sentence_len = word_count / sentence_count if sentence_count > 0 else word_count

    # Analyze punctuation
    has_dashes = ('—' in text or '-' in text)
    has_semicolons = ';' in text
    has_parentheses = '(' in text and ')' in text
    has_asterisks = '*' in text
    has_commas = ',' in text

    punctuation_pattern = []
    if has_commas:
        punctuation_pattern.append("commas")
    if has_dashes:
        punctuation_pattern.append("dashes")
    if has_semicolons:
        punctuation_pattern.append("semicolons")
    if has_parentheses:
        punctuation_pattern.append("parentheses")
    if has_asterisks:
        punctuation_pattern.append("asterisks")

    # Estimate clause count (rough heuristic: count of conjunctions + relative pronouns)
    clause_indicators = len(re.findall(r'\b(and|or|but|because|since|while|when|if|that|which|who)\b', text, re.IGNORECASE))
    clause_count = max(1, clause_indicators + 1)  # At least 1 clause

    # Heuristic for voice (look for passive indicators)
    passive_indicators = len(re.findall(r'\b(is|are|was|were|been|being)\s+\w+ed\b', text, re.IGNORECASE))
    active_indicators = len(re.findall(r'\b\w+ed\s+(the|a|an|this|that|these|those)\b', text, re.IGNORECASE))
    voice = "passive" if passive_indicators > active_indicators else "active"

    # Determine complexity
    if clause_count == 1:
        complexity = "simple"
    elif clause_count == 2:
        complexity = "compound"
    elif clause_count >= 3:
        complexity = "complex"
    else:
        complexity = "simple"

    return {
        'word_count': word_count,
        'sentence_count': sentence_count,
        'avg_sentence_len': avg_sentence_len,
        'punctuation_pattern': punctuation_pattern,
        'clause_count': clause_count,
        'voice': voice,
        'has_dashes': has_dashes,
        'has_semicolons': has_semicolons,
        'has_parentheses': has_parentheses,
        'has_asterisks': has_asterisks,
        'has_commas': has_commas,
        'complexity': complexity
    }


def generate_author_style_dna(author_name: str, sample_text: str, config_path: str = "config.json") -> str:
    """Generates a dense 'Style DNA' string to prime the main generator.

    Uses an LLM to analyze the author's writing style and generate a concise
    description that activates the model's latent knowledge of the author.

    Args:
        author_name: Name of the author (e.g., "Hemingway", "Lovecraft").
        sample_text: Representative sample text from the author (should be from centroid).
        config_path: Path to configuration file.

    Returns:
        Style DNA string (max 40 words) describing the author's style characteristics.
        Returns fallback string if generation fails or sample is too short.
    """
    # Validate sample text
    if not sample_text or len(sample_text) < 50:
        fallback = f"Distinctive voice of {author_name}."
        print(f"    ⚠ Sample text too short for {author_name}, using fallback DNA.")
        return fallback

    # Truncate sample to first 1000 chars for context
    sample_snippet = sample_text[:1000]
    if len(sample_text) > 1000:
        sample_snippet += "..."

    # Build prompt for Style DNA generation
    system_prompt_template = _load_prompt_template("prompt_builder_style_dna_system.md")
    system_prompt = system_prompt_template

    user_prompt_template = _load_prompt_template("prompt_builder_style_dna_user.md")
    user_prompt = user_prompt_template.format(
        author_name=author_name,
        sample_snippet=sample_snippet
    )

    try:
        # Use LLMProvider to generate Style DNA
        from src.generator.llm_provider import LLMProvider
        llm = LLMProvider(config_path=config_path)

        dna_string = llm.call(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            model_type="editor",  # Use editor model for generation
            require_json=False,
            temperature=0.3,  # Lower temperature for more consistent analysis
            max_tokens=100
        ).strip()

        # Validate and clean the response
        if not dna_string or len(dna_string) < 10:
            fallback = f"Distinctive voice of {author_name}."
            print(f"    ⚠ Generated DNA too short for {author_name}, using fallback.")
            return fallback

        # Print for verification (as specified in plan)
        print(f"    🧬 Generated Style DNA for {author_name}: {dna_string}")

        return dna_string

    except Exception as e:
        # Graceful fallback on error
        fallback = f"Distinctive voice of {author_name}."
        print(f"    ⚠ Style DNA generation failed for {author_name}: {e}. Using fallback.")
        return fallback


class PromptAssembler:
    """Constructs highly constrained prompts using RAG data to prevent 'LLM Slop'."""

    def __init__(self, target_author_name: str = "Target Author", banned_words: Optional[List[str]] = None, style_dna: Optional[str] = None, style_dna_dict: Optional[Dict[str, str]] = None):
        """Initialize the prompt assembler.

        Args:
            target_author_name: Name of the target author (for persona definition).
            banned_words: List of words to ban (prevents generic AI language).
            style_dna: Style DNA string for single-author mode.
            style_dna_dict: Dictionary mapping author names to Style DNA strings (for blend mode).
        """
        self.author_name = target_author_name

        # Negative constraints to kill "Assistant Voice"
        self.banned_words = banned_words or [
            "delve", "testament", "underscore", "landscape", "tapestry",
            "bustling", "crucial", "meticulous", "comprehensive", "fostering"
        ]

        # Style DNA for persona priming
        self.style_dna = style_dna
        self.style_dna_dict = style_dna_dict

    def build_system_message(self) -> str:
        """Define the rigid persona. The LLM is not a writer; it is a style engine.

        Returns:
            System prompt string.
        """
        template = _load_prompt_template("generator_system.md")
        base_prompt = template.format(author_name=self.author_name)

        # Inject Style DNA if available
        style_dna_section = self._build_style_dna_section()
        if style_dna_section:
            # Insert Style DNA section after the role definition but before operational rules
            # Find the first "YOUR DIRECTIVES" or "OPERATIONAL" section and insert before it
            if "YOUR DIRECTIVES" in base_prompt:
                base_prompt = base_prompt.replace("YOUR DIRECTIVES:", f"{style_dna_section}\n\nYOUR DIRECTIVES:")
            elif "OPERATIONAL RULES" in base_prompt:
                base_prompt = base_prompt.replace("OPERATIONAL RULES", f"{style_dna_section}\n\nOPERATIONAL RULES")
            else:
                # Fallback: append before the end
                base_prompt = f"{base_prompt}\n\n{style_dna_section}"

        return base_prompt

    def _build_style_dna_section(self) -> str:
        """Build the Style DNA section for the system prompt.

        Returns:
            Style DNA section string, or empty string if no DNA available.
        """
        # Handle blend mode (multiple authors)
        if self.style_dna_dict and len(self.style_dna_dict) >= 2:
            # Get author names from dict keys
            authors = list(self.style_dna_dict.keys())
            if len(authors) >= 2:
                dna_a = self.style_dna_dict.get(authors[0], "")
                dna_b = self.style_dna_dict.get(authors[1], "")

                if dna_a and dna_b:
                    return f"""### YOUR STYLE DNA (THE INSTRUCTIONS)
To authentically write as {authors[0]} and {authors[1]}, you must adhere to these rules:

**PRIMARY STYLE ({authors[0]}):**
> {dna_a}

**SECONDARY INFLUENCE ({authors[1]}):**
> {dna_b}

Blend these styles naturally, with the primary style dominant."""
                elif dna_a:
                    return f"""### YOUR STYLE DNA (THE INSTRUCTIONS)
To authentically write as {authors[0]}, you must adhere to these rules:
> **{dna_a}**"""
                elif dna_b:
                    return f"""### YOUR STYLE DNA (THE INSTRUCTIONS)
To authentically write as {authors[1]}, you must adhere to these rules:
> **{dna_b}**"""

        # Handle single-author mode
        if self.style_dna:
            return f"""### YOUR STYLE DNA (THE INSTRUCTIONS)
To authentically write as {self.author_name}, you must adhere to these rules:
> **{self.style_dna}**"""

        # No Style DNA available
        return ""

    def build_generation_prompt(
        self,
        input_text: str,
        situation_match: Optional[str],
        structure_match: str,
        style_metrics: Optional[Dict[str, float]] = None,
        global_vocab_list: Optional[List[str]] = None,
        use_fallback_structure: bool = False,
        constraint_mode: str = "STRICT"
    ) -> str:
        """Assemble the Few-Shot prompt using Dual-RAG data.

        Args:
            input_text: The original input text to rewrite.
            situation_match: Retrieved paragraph for vocabulary grounding (or None).
            structure_match: Retrieved paragraph for rhythm/structure (required).
            style_metrics: Optional style metrics dict (will be extracted from structure_match if not provided).
            global_vocab_list: Optional list of global vocabulary words to inject for variety.

        Returns:
            Complete user prompt string.
        """
        # CRITICAL: Sanitize structure_match to remove dialogue tags like "August:" before using it
        # This prevents the generator from copying proper nouns from structure matches
        structure_match = sanitize_structural_reference(structure_match)

        # Extract style metrics from structure_match if not provided
        if style_metrics is None:
            style_vec = get_style_vector(structure_match)
            # Estimate average sentence length from structure_match
            words = structure_match.split()
            sentences = structure_match.count('.') + structure_match.count('!') + structure_match.count('?')
            if sentences > 0:
                avg_sentence_len = len(words) / sentences
            else:
                avg_sentence_len = len(words)
            style_metrics = {'avg_sentence_len': avg_sentence_len}

        # Handle constraint modes and fallback structure mode
        # SAFETY mode overrides use_fallback_structure
        if constraint_mode == "SAFETY":
            use_fallback_structure = True

        # Build constraint instruction based on mode
        if constraint_mode == "STRICT":
            constraint_instruction = """### STRUCTURAL INSTRUCTIONS
1. **RHYTHM:** Use the Structural Reference as a guide for tone and pacing.
2. **LENGTH:** You are NOT bound by the reference's word count.
   - **EXPAND** the structure if your input has more information.
   - **COMPLETE** your sentences. Do not write fragments just to be short.
   - **GRAMMAR:** Grammatical correctness is more important than matching the template structure.
3. **STRUCTURE:** Match the sentence structure pattern (simple, compound, complex) and punctuation style.
4. **DO NOT COPY WORDS**: Do not use phrases like "If you could see", "Then came", "Concerning" from the reference.
5. **DO NOT REPEAT**: Avoid using transition words like "therefore", "concerning", "thus" more than once.
6. **ADAPT**: Adapt the structure to fit ALL content from Input Text.
7. **PRESERVE FLOW**: Preserve natural English flow - do not create stilted or unnatural phrasing."""
        elif constraint_mode == "LOOSE":
            constraint_instruction = """### STRUCTURAL INSTRUCTIONS
1. **RHYTHM:** Use the Structural Reference only for rhythm and pacing.
2. **LENGTH:** You are NOT bound by the reference's word count.
   - **EXPAND** if needed to include all content.
   - **COMPLETE** your sentences. Grammar > Length.
3. **NATURAL:** Write natural, grammatical English.
4. **PUNCTUATION:** Do not copy strange punctuation (like ' . ') from the reference."""
        else:  # SAFETY
            constraint_instruction = "INSTRUCTION: Ignore the Structural Reference. Rewrite the input content clearly using the target author's vocabulary and general style. Preserve all meaning."

        # Handle fallback structure mode
        if use_fallback_structure:
            # Use simplified structure instructions for fallback mode
            structure_instructions_text = """- Use simple, clear Subject-Verb-Object structure
- Match the approximate length of the input text
- Maintain natural flow and readability"""
            # Use basic structure analysis for fallback
            structure_analysis = {
                'word_count': len(input_text.split()),
                'avg_sentence_len': len(input_text.split()) / max(1, input_text.count('.') + input_text.count('!') + input_text.count('?'))
            }
        else:
            # Analyze structure match in detail
            structure_analysis = _analyze_structure(structure_match)

            # Build explicit structure instructions
            structure_instructions = []
            structure_instructions.append(f"- Word count: ~{structure_analysis['word_count']} words")
            structure_instructions.append(f"- Sentence structure: {structure_analysis['complexity']} ({structure_analysis['clause_count']} clauses)")
            structure_instructions.append(f"- Voice: {structure_analysis['voice']}")

            if structure_analysis.get('punctuation_pattern'):
                structure_instructions.append(f"- Punctuation: Use {', '.join(structure_analysis['punctuation_pattern'])}")

            if structure_analysis.get('has_dashes'):
                structure_instructions.append("- Include dashes (— or -) for parenthetical or explanatory elements")
            if structure_analysis.get('has_semicolons'):
                structure_instructions.append("- Use semicolons to connect related independent clauses")
            if structure_analysis.get('has_parentheses'):
                structure_instructions.append("- Include parenthetical asides using parentheses")
            if structure_analysis.get('has_asterisks'):
                structure_instructions.append("- Use asterisks (*) for emphasis or special notation")

            structure_instructions_text = "\n".join(structure_instructions)

        # Add Expansion & Logic Rules
        expansion_rules = """
### EXPANSION & LOGIC RULES (NEW)
1. **EXPANSION IS ALLOWED:** You may expand the text if necessary to improve flow, clarity, or rhetorical weight.
   - You can add transitional phrases to connect ideas smoothly.
   - You can add adjectives or short clauses to strengthen the existing argument.
   - **LIMIT:** Do not invent new facts. Only expand on *existing* ideas.

2. **LOGIC FIRST:** The logic of the Input Text is supreme.
   - If the Structural Reference starts with "Therefore" but the Input is a standalone statement, DELETE the "Therefore."
   - Never write "Therefore, because" or "Thus, however."

3. **GRAMMAR OVER STRUCTURE:** If adhering to the Structural Reference would create a fragment or awkward sentence, BREAK THE STRUCTURE. Your final output must be grammatically perfect English.
"""

        # Prepend expansion rules and constraint instruction to structure instructions
        structure_instructions_text = f"{expansion_rules}\n\n{constraint_instruction}\n\n{structure_instructions_text}"

        # Build situation match content
        if situation_match:
            situation_match_label = "(VOCABULARY PALETTE)"
            situation_match_content = f"""The author has written about this topic before.
Observe their specific word choices and tone in this snippet:
"{situation_match}"

*Instruction: Borrow specific adjectives and verbs from this snippet if they fit.*"""
        else:
            situation_match_label = ""
            situation_match_content = """No direct topic match found in corpus. Rely strictly on the Structural Reference for tone."""

        # Build vocabulary block
        vocab_block = ""
        if global_vocab_list and len(global_vocab_list) > 0:
            sample_size = min(20, len(global_vocab_list))  # Increased from 10 to 20
            flavor_words = ", ".join(random.sample(global_vocab_list, sample_size))
            vocab_block = f"""### VOCABULARY INSPIRATION
1. PRIMARY SOURCE: Use words from the 'Situational Reference' above.
2. SECONDARY SOURCE: Incorporate some of these characteristic author words if they fit:
   [{flavor_words}]

"""

        # Add author format instruction
        author_format_instruction = ""
        if self.author_name:
            # Determine format based on author (Mao = speech/essay, Hemingway = narrative, etc.)
            author_lower = self.author_name.lower()
            if "mao" in author_lower:
                author_format_instruction = "\n\n### AUTHOR FORMAT\nRewrite this as if it were a speech or essay by the target author. The style should be didactic and authoritative, suitable for political or philosophical discourse."
            # Add more author-specific format instructions as needed

        # Append format instruction to vocab_block if present
        if author_format_instruction:
            vocab_block += author_format_instruction

        # Calculate word counts
        input_word_count = len(input_text.split())
        target_word_count = int(input_word_count * 1.2)

        # Load and format the generation prompt template
        template = _load_prompt_template("generation_prompt.md")
        return template.format(
            structure_match=structure_match,
            structure_instructions=structure_instructions_text,
            avg_sentence_len=int(structure_analysis['avg_sentence_len']),
            situation_match_label=situation_match_label,
            situation_match_content=situation_match_content,
            vocab_block=vocab_block,
            input_word_count=input_word_count,
            target_word_count=target_word_count,
            input_text=input_text,
            banned_words=", ".join(self.banned_words)
        )

    def build_blended_prompt(
        self,
        input_text: str,
        bridge_template: str,
        hybrid_vocab: List[str],
        author_a: str,
        author_b: str,
        blend_ratio: float = 0.5,
        constraint_mode: str = "STRICT"
    ) -> str:
        """Build a blended style prompt for mixing two author styles.

        Args:
            input_text: The original input text to rewrite.
            bridge_template: Bridge text that naturally connects the two styles.
            hybrid_vocab: List of words sampled from both authors.
            author_a: First author name.
            author_b: Second author name.
            blend_ratio: Blend ratio (0.0 = All Author A, 1.0 = All Author B).
            constraint_mode: Constraint mode ("STRICT", "LOOSE", or "SAFETY").

        Returns:
            Complete user prompt string for blended style generation.
        """
        # Build constraint instruction based on mode
        if constraint_mode == "STRICT":
            constraint_instruction = """CRITICAL: Use the bridge template as a SYNTAX BLUEPRINT, not a word source.
    - Match the sentence structure pattern (simple, compound, complex)
    - Match the punctuation style (commas, dashes, semicolons)
    - Match the rhythm and pacing
    - **DO NOT COPY WORDS**: Do not use phrases from the bridge template verbatim
    - Adapt the structure to fit ALL content from Input Text
    - Prioritize grammatical flow and natural phrasing.
    - Match the rhythm and punctuation pattern of the template
    - If the template forces an awkward sentence, smooth it out while maintaining the structural style
    - Preserve natural English flow - do not create stilted or unnatural phrasing"""
        elif constraint_mode == "LOOSE":
            constraint_instruction = "GUIDANCE: Use the bridge template as a rhythm guide, but prioritize Meaning. You may expand or contract the structure to fit the content. You may change punctuation if necessary for grammar. Prioritize natural, flowing prose over exact structural mimicry."
        else:  # SAFETY
            constraint_instruction = "INSTRUCTION: Ignore the bridge template's syntax. Rewrite the input content clearly using the blended author vocabulary and general style. Preserve all meaning."

        # Load blended prompt template
        template = _load_prompt_template("generation_blended.md")

        # Format blend description
        if blend_ratio < 0.3:
            blend_desc = f"primarily {author_a} with subtle {author_b} influences"
        elif blend_ratio > 0.7:
            blend_desc = f"primarily {author_b} with subtle {author_a} influences"
        else:
            blend_desc = f"a balanced blend of {author_a} and {author_b}"

        # Format vocabulary list
        vocab_text = ", ".join(hybrid_vocab) if hybrid_vocab else "N/A"

        # Prepend constraint instruction to bridge template description
        bridge_template_with_constraint = f"{constraint_instruction}\n\nBridge Template: {bridge_template}"

        return template.format(
            bridge_template=bridge_template_with_constraint,
            hybrid_vocab=vocab_text,
            author_a=author_a,
            author_b=author_b,
            blend_desc=blend_desc,
            input_text=input_text
        )

