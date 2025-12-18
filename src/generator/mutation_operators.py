"""Directed mutation operators for gradient-based evolution.

These operators perform specific, narrow tasks based on the type of defect
detected in the current draft, rather than asking the LLM to "fix everything."
"""

from pathlib import Path
from typing import List, Optional
from src.ingestion.blueprint import SemanticBlueprint
from src.atlas.rhetoric import RhetoricalType


def _load_prompt_template(template_name: str) -> str:
    """Load a prompt template from the prompts directory."""
    prompts_dir = Path(__file__).parent.parent.parent / "prompts"
    template_path = prompts_dir / template_name
    if not template_path.exists():
        raise FileNotFoundError(f"Prompt template not found: {template_path}")
    return template_path.read_text().strip()


# Load prompts from markdown files
BATCH_GENERATION_PROMPT = _load_prompt_template("batch_generation.md")
PARAGRAPH_FUSION_PROMPT = _load_prompt_template("paragraph_fusion.md")


class MutationOperator:
    """Base class for mutation operators."""

    def generate(
        self,
        current_draft: str,
        blueprint: SemanticBlueprint,
        author_name: str,
        style_dna: str,
        rhetorical_type: RhetoricalType,
        llm_provider,
        temperature: float = 0.6,
        max_tokens: int = 300
    ) -> str:
        """Generate a mutated version of the draft.

        Args:
            current_draft: Current draft to mutate.
            blueprint: Original semantic blueprint.
            author_name: Target author name.
            style_dna: Style DNA description.
            rhetorical_type: Rhetorical mode.
            llm_provider: LLM provider instance.
            temperature: Temperature for generation.
            max_tokens: Maximum tokens to generate.

        Returns:
            Mutated text.
        """
        raise NotImplementedError


class SemanticInjectionOperator(MutationOperator):
    """Operator for inserting missing keywords with style (trigger: low recall).

    This operator focuses on adding missing concepts while mimicking the
    author's tone and vocabulary from RAG examples.
    """

    def generate(
        self,
        current_draft: str,
        blueprint: SemanticBlueprint,
        author_name: str,
        style_dna: str,
        rhetorical_type: RhetoricalType,
        llm_provider,
        temperature: float = 0.6,
        max_tokens: int = 300,
        missing_keywords: Optional[List[str]] = None,
        style_lexicon: Optional[List[str]] = None,
        rag_example: Optional[str] = None
    ) -> str:
        """Insert missing keywords into the draft using stylistic repair.

        Args:
            missing_keywords: List of keywords that are missing from the draft.
            style_lexicon: List of style words to use when inserting keywords.
            rag_example: Example text from RAG to mimic tone and vocabulary.
        """
        if missing_keywords is None:
            # Extract missing keywords by comparing blueprints
            try:
                from src.ingestion.blueprint import BlueprintExtractor
                extractor = BlueprintExtractor()
                draft_blueprint = extractor.extract(current_draft)
                input_keywords = blueprint.core_keywords
                draft_keywords = draft_blueprint.core_keywords
                missing_keywords = list(input_keywords - draft_keywords)[:5]  # Top 5 missing
            except Exception:
                missing_keywords = []

        if not missing_keywords:
            return current_draft  # Nothing to inject

        keywords_text = ", ".join(missing_keywords)

        # Get all required keywords from blueprint (for anchoring)
        all_keywords = sorted(blueprint.core_keywords) if blueprint.core_keywords else []
        required_keywords_text = ", ".join(all_keywords) if all_keywords else "None"

        # Build style guidance section - ALWAYS style-infuse, even if no RAG example
        style_guidance = ""
        examples_text = ""

        if rag_example:
            examples_text = f'"{rag_example}"'
        elif style_lexicon and len(style_lexicon) > 0:
            # Use first few style words as example context
            examples_text = f'Examples using words like: {", ".join(style_lexicon[:5])}'

        if style_lexicon:
            lexicon_text = ", ".join(style_lexicon[:20])  # Show more words
            style_guidance = f"""
**STYLE REQUIREMENT (CRITICAL - DO NOT WRITE GENERIC ENGLISH):**
- You are a specific author editor, NOT a generic corporate writer
- Insert the missing concepts: {keywords_text}
- TRANSFORM the sentence to sound like the author found in these examples: {examples_text}
- USE this vocabulary if possible: {lexicon_text}
- Do NOT write generic corporate English - be distinct and match the author's voice
- The author has a distinctive style - mimic it precisely"""
        elif rag_example:
            style_guidance = f"""
**STYLE REQUIREMENT (CRITICAL - DO NOT WRITE GENERIC ENGLISH):**
- You are a specific author editor, NOT a generic corporate writer
- Insert the missing concepts: {keywords_text}
- TRANSFORM the sentence to sound like the author found in this example: {examples_text}
- Do NOT write generic corporate English - be distinct and match the author's voice"""

        system_prompt = f"""You are a specific author editor working on stylistic text transformation.
Your task is to INSERT missing keywords into a sentence while transforming it to match the target author's distinctive voice.
You must preserve the existing structure and add the missing concepts using the author's unique style - NOT generic English."""

        user_prompt = f"""### TASK: Author-Specific Semantic Repair
**Goal:** Insert missing keywords while transforming the sentence to match the target author's distinctive voice.

**Original text:** "{blueprint.original_text}"
**Current draft:** "{current_draft}"
**Missing keywords that MUST be included:** {keywords_text}
**ALL required keywords (do not delete any of these):** {required_keywords_text}{style_guidance}

### INSTRUCTIONS
1. **Step 1 (Reasoning):** Analyze the current draft and identify where the missing words logically belong. Study the style examples to understand the author's voice, vocabulary, and sentence structure.

2. **Step 2 (Rough Draft):** Write a version that has ALL required keywords. Ensure every keyword from the required list is present. Use the author's vocabulary and style to connect concepts naturally.

3. **Step 3 (Polish):** Transform Step 2 to match the author's distinctive voice. Use the style vocabulary and examples as your guide. Do NOT write generic corporate English - be distinct.

**CRITICAL:**
- Ensure the final output contains **ALL** of these words: {required_keywords_text}
- Do NOT delete existing valid keywords to make room for new ones
- You MUST transform the sentence to match the author's voice - do NOT use generic English
- If style vocabulary is provided, USE IT to connect concepts naturally
- The output should sound like the specific author, not generic corporate writing

**Output:** Return ONLY the final polished sentence from Step 3."""

        try:
            mutated = llm_provider.call(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                temperature=temperature,
                max_tokens=max_tokens
            )
            from src.generator.llm_interface import clean_generated_text
            mutated = clean_generated_text(mutated)
            mutated = mutated.strip()
            return mutated if mutated else current_draft
        except Exception:
            return current_draft


class GrammarRepairOperator(MutationOperator):
    """Operator for fixing sentence structure (trigger: low fluency).

    This operator focuses on grammar and flow issues like stilted phrasing,
    missing articles, and awkward constructions.
    """

    def generate(
        self,
        current_draft: str,
        blueprint: SemanticBlueprint,
        author_name: str,
        style_dna: str,
        rhetorical_type: RhetoricalType,
        llm_provider,
        temperature: float = 0.6,
        max_tokens: int = 300
    ) -> str:
        """Fix sentence structure and grammar."""
        system_prompt = f"""You are a professional copyeditor specializing in sentence structure.
Your task is to fix grammatical issues and improve flow WITHOUT changing the meaning or style.
Focus on:
- Converting stilted patterns like 'will, in time, be' to 'eventually is'
- Ensuring Subject-Verb-Object agreement
- Fixing missing articles
- Improving natural word order"""

        # Get all required keywords from blueprint (for anchoring)
        all_keywords = sorted(blueprint.core_keywords) if blueprint.core_keywords else []
        required_keywords_text = ", ".join(all_keywords) if all_keywords else "None"

        user_prompt = f"""### TASK: Grammar Repair
**Goal:** Fix sentence structure while preserving ALL meaning and keywords.
**Original text:** "{blueprint.original_text}"
**Current draft:** "{current_draft}"
**ALL required keywords (must keep all of these):** {required_keywords_text}

### INSTRUCTIONS
1. **Step 1 (Reasoning):** Identify the grammatical issues in the current draft. What makes it awkward or incorrect?

2. **Step 2 (Rough Draft):** Write a grammatically correct version that includes ALL required keywords. It's okay if it's not perfectly polished yet.

3. **Step 3 (Polish):** Refine Step 2 into natural, flowing English while ensuring ALL keywords remain present.

**CRITICAL:**
- Keep ALL keywords: {required_keywords_text}
- Do NOT remove any keywords while fixing grammar
- Preserve the exact meaning from the original text

**Output:** Return ONLY the final polished sentence from Step 3."""

        try:
            mutated = llm_provider.call(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                temperature=temperature,
                max_tokens=max_tokens
            )
            from src.generator.llm_interface import clean_generated_text
            mutated = clean_generated_text(mutated)
            mutated = mutated.strip()
            return mutated if mutated else current_draft
        except Exception:
            return current_draft


class StylePolishOperator(MutationOperator):
    """Operator for enhancing style (trigger: high recall+fluency, low style).

    This operator focuses on matching the target author's voice while
    preserving all nouns and verbs exactly.
    """

    def generate(
        self,
        current_draft: str,
        blueprint: SemanticBlueprint,
        author_name: str,
        style_dna: str,
        rhetorical_type: RhetoricalType,
        llm_provider,
        temperature: float = 0.6,
        max_tokens: int = 300,
        style_lexicon: Optional[List[str]] = None,
        style_structure: Optional[str] = None
    ) -> str:
        """Enhance style while preserving meaning.

        Args:
            style_lexicon: Optional list of style words from extracted DNA.
            style_structure: Optional structural rule from extracted DNA.
        """
        system_prompt = f"""You are a ghostwriter specializing in the style of {author_name}.

Style characteristics: {style_dna}

Your task is to rewrite the sentence in this style while preserving ALL nouns and verbs exactly.
You may change:
- Adjectives and adverbs
- Sentence structure
- Word order
- Phrasing

You must NOT change:
- Core nouns
- Core verbs
- Meaning"""

        # Get all required keywords from blueprint (for anchoring)
        all_keywords = sorted(blueprint.core_keywords) if blueprint.core_keywords else []
        required_keywords_text = ", ".join(all_keywords) if all_keywords else "None"

        # Use extracted style DNA if provided
        style_instruction = ""
        if style_lexicon:
            lexicon_text = ", ".join(style_lexicon[:10])  # Limit to 10
            style_instruction = f"\n**Style Lexicon (integrate these words):** {lexicon_text}"
            if style_structure:
                style_instruction += f"\n**Style Structure:** {style_structure}"

        user_prompt = f"""### TASK: Style Polish
**Goal:** Enhance style while preserving ALL meaning and keywords.
**Original text:** "{blueprint.original_text}"
**Current draft:** "{current_draft}"
**Rhetorical type:** {rhetorical_type.value}
**ALL required keywords (must keep all of these):** {required_keywords_text}{style_instruction}

### INSTRUCTIONS
1. **Step 1 (Reasoning):** Analyze how the current draft differs from the target style. What stylistic elements need enhancement?

2. **Step 2 (Rough Draft):** Write a version in the target style that includes ALL required keywords. Focus on style transformation first, even if it's not perfectly polished.

3. **Step 3 (Polish):** Refine Step 2 into natural, flowing prose in the target style while ensuring ALL keywords remain present.

**CRITICAL:**
- Keep ALL keywords: {required_keywords_text}
- Preserve ALL nouns and verbs exactly
- Do NOT remove any keywords while enhancing style
- Maintain the exact meaning from the original text
{f"- **Vocabulary:** You MUST attempt to integrate words from this list: {', '.join(style_lexicon[:10])}" if style_lexicon else ""}
{f"- **Structure:** Follow this structural rule: {style_structure}" if style_structure else ""}

**Output:** Return ONLY the final polished sentence from Step 3."""

        try:
            mutated = llm_provider.call(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                temperature=temperature,
                max_tokens=max_tokens
            )
            from src.generator.llm_interface import clean_generated_text
            mutated = clean_generated_text(mutated)
            mutated = mutated.strip()
            return mutated if mutated else current_draft
        except Exception:
            return current_draft


class DynamicStyleOperator(MutationOperator):
    """Operator for dynamic style enhancement using RAG-extracted style DNA.

    This operator uses style lexicon and structure extracted from ChromaDB examples
    to perform style transfer without hardcoding.
    """

    def generate(
        self,
        current_draft: str,
        blueprint: SemanticBlueprint,
        author_name: str,
        style_dna: str,
        rhetorical_type: RhetoricalType,
        llm_provider,
        temperature: float = 0.6,
        max_tokens: int = 300,
        style_lexicon: Optional[List[str]] = None,
        style_structure: Optional[str] = None,
        style_tone: Optional[str] = None
    ) -> str:
        """Enhance style using dynamically extracted style DNA.

        Args:
            style_lexicon: List of style words extracted from examples.
            style_structure: Structural rule extracted from examples.
            style_tone: Tone adjective extracted from examples.
        """
        if not style_lexicon:
            # Fallback to regular style polish if no lexicon
            operator = StylePolishOperator()
            return operator.generate(
                current_draft, blueprint, author_name, style_dna,
                rhetorical_type, llm_provider, temperature, max_tokens
            )

        # Get all required keywords from blueprint
        all_keywords = sorted(blueprint.core_keywords) if blueprint.core_keywords else []
        required_keywords_text = ", ".join(all_keywords) if all_keywords else "None"

        lexicon_text = ", ".join(style_lexicon[:10])  # Limit to 10

        system_prompt = f"""You are a ghostwriter specializing in the style of {author_name}.

Style Tone: {style_tone or 'Authoritative'}
Style Structure: {style_structure or 'Standard sentence structure'}

Your task is to rewrite the sentence to match this author's distinctive voice using their signature vocabulary and structural patterns."""

        user_prompt = f"""### TASK: Dynamic Style Enhancement
**Goal:** Rewrite text to match target voice using extracted style characteristics.
**Original text:** "{blueprint.original_text}"
**Current draft:** "{current_draft}"
**Rhetorical type:** {rhetorical_type.value}
**ALL required keywords (must keep all of these):** {required_keywords_text}

**Extracted Style DNA:**
- **Lexicon (integrate these words):** {lexicon_text}
- **Tone:** {style_tone or 'Authoritative'}
- **Structure:** {style_structure or 'Standard sentence structure'}

### INSTRUCTIONS
1. **Step 1 (Reasoning):** Analyze the current draft. How can you integrate the style lexicon words naturally? What structural changes are needed?

2. **Step 2 (Rough Draft):** Write a version that:
   - Includes ALL required keywords
   - Attempts to integrate words from the lexicon: {lexicon_text}
   - Follows the structural rule: {style_structure or 'Standard sentence structure'}
   It's okay if it's clunky at this stage.

3. **Step 3 (Polish):** Refine Step 2 into natural, flowing prose that:
   - Maintains ALL required keywords
   - Naturally incorporates style lexicon words
   - Follows the structural pattern
   - Preserves the exact meaning

**CRITICAL:**
- Keep ALL keywords: {required_keywords_text}
- Do NOT lose original meaning
- You MAY add 'connective tissue' words to fit the style
- Integrate lexicon words naturally, don't force them

**Output:** Return ONLY the final polished sentence from Step 3."""

        try:
            mutated = llm_provider.call(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                temperature=temperature,
                max_tokens=max_tokens
            )
            from src.generator.llm_interface import clean_generated_text
            mutated = clean_generated_text(mutated)
            mutated = mutated.strip()
            return mutated if mutated else current_draft
        except Exception:
            return current_draft


class StructuralCloneOperator(MutationOperator):
    """Operator for structural cloning using skeleton templates.

    This operator forces generation to match the exact syntactic structure
    of a RAG sample by injecting meaning into a pre-extracted skeleton.
    """

    def generate(
        self,
        current_draft: str,
        blueprint: SemanticBlueprint,
        author_name: str,
        style_dna: str,
        rhetorical_type: RhetoricalType,
        llm_provider,
        temperature: float = 0.6,
        max_tokens: int = 300,
        skeleton: Optional[str] = None,
        style_lexicon: Optional[List[str]] = None
    ) -> str:
        """Generate text by injecting meaning into a structural skeleton.

        Args:
            skeleton: Structural skeleton template with [NP], [VP], [ADJ] placeholders.
            style_lexicon: Optional list of style words to prioritize during generation.
        """
        if not skeleton:
            # No skeleton provided, return current draft
            return current_draft

        system_prompt = f"""You are a precision structural clone generator. Your task is to inject the meaning of the original text into a specific sentence skeleton, matching the exact syntactic structure."""

        # Build style instruction if lexicon is provided
        style_instruction = ""
        if style_lexicon:
            # Limit to first 20 words for prompt size
            lexicon_preview = ', '.join(style_lexicon[:20])
            if len(style_lexicon) > 20:
                lexicon_preview += f" (and {len(style_lexicon) - 20} more)"
            style_instruction = f"""
**STYLE VOCABULARY (Prioritize these words):**
{lexicon_preview}

When filling the [SLOTS], you MUST prioritize using words from this list where possible.
This ensures the output matches the target author's voice.
"""

        # Extract keywords from blueprint for explicit mapping
        subjects = blueprint.get_subjects()
        verbs = blueprint.get_verbs()
        objects = blueprint.get_objects()
        core_keywords = list(blueprint.core_keywords)[:10]  # Limit to top 10 for prompt size

        # Build explicit concept mapping section
        mapping_section = "### STEP 1: PLANNING - Explicit Concept Mapping\n"
        mapping_section += "**You MUST map the Input Keywords into the Skeleton Slots BEFORE generating.**\n\n"

        if subjects:
            subjects_text = ", ".join([f'"{s}"' for s in subjects[:3]])
            mapping_section += f"* **Subjects** ({subjects_text}) → Maps to [NP] slots (noun phrases)\n"
        if verbs:
            verbs_text = ", ".join([f'"{v}"' for v in verbs[:3]])
            mapping_section += f"* **Verbs** ({verbs_text}) → Maps to [VP] slots (verb phrases)\n"
        if objects:
            objects_text = ", ".join([f'"{o}"' for o in objects[:3]])
            mapping_section += f"* **Objects** ({objects_text}) → Maps to [NP] slots (noun phrases)\n"
        if core_keywords:
            keywords_text = ", ".join([f'"{k}"' for k in core_keywords[:5]])
            mapping_section += f"* **Core Keywords** ({keywords_text}) → Must appear in output\n"

        mapping_section += "\n**CRITICAL:** You MUST include ALL mapped keywords in the final output. Do not replace them with synonyms or related concepts.\n"
        mapping_section += "For example: Do NOT replace 'Rule' with 'Process' or 'reinforces' with 'affirms'.\n\n"

        user_prompt = f"""### TASK: Structural Cloning
**Goal:** Inject the meaning of the Original Text into the Target Skeleton structure.

**Original Text (meaning to convey):** "{blueprint.original_text}"

**Target Skeleton (structure to match):** "{skeleton}"
{style_instruction}

{mapping_section}

### STEP 2: GENERATION
1. **Write the sentence** using the Skeleton structure.
2. **Fill the slots** using the mapped keywords from Step 1.
3. **CRITICAL:** You MUST include the mapped keywords. Do not replace "Rule" with "Process" or "reinforces" with "affirms".
4. **Preserve:** Keep ALL connecting words, prepositions, conjunctions, and punctuation from the skeleton exactly as they are.

**CRITICAL OUTPUT REQUIREMENT:**
- The output must be a standard English sentence. **Do NOT output brackets** like `[NP]` or `[VP]`.
- Replace every placeholder with real words representing the Input Meaning.
- If you output brackets, the generation has failed.

**CRITICAL CONSTRAINTS:**
- **Structure:** You MUST use the exact structure of the skeleton
- **Meaning:** You MUST use the concepts from the Original Text - do NOT hallucinate irrelevant words
- **Mapping:** Map input concepts to skeleton slots (e.g., "Human experience" → Subject [NP], "reinforces" → Verb [VP])
- **No Hallucination:** Do NOT fill slots with random style words that don't relate to the input meaning
- You MUST preserve all structural words (prepositions, conjunctions, articles)
- You may expand concepts to fill slots, but preserve the original meaning
- Do NOT simplify the skeleton structure
- Do NOT change the core message of the original text

**STOP EARLY IF NEEDED:**
- If the Skeleton is significantly longer than your Input Content, STOP writing after you have conveyed the meaning
- Do NOT invent new philosophy or facts just to fill the remaining slots
- It is better to leave the tail of the skeleton empty than to generate gibberish
- Only fill slots that you can meaningfully map from the Original Text

**LOGIC MISMATCH ESCAPE HATCH:**
- If the input logic (e.g., Universal Truth like "Stars erode") explicitly contradicts the skeleton logic (e.g., Conditional Hypothesis like "If... then..."), you MUST output: **SKIPPING: LOGIC_MISMATCH**
- Do NOT force a bad sentence that changes the meaning (e.g., turning "Stars erode" into "Stars erode only when...").
- Examples of logic mismatches:
  * Universal statement → Conditional template: "Stars erode" cannot become "If stars erode, then..."
  * Factual statement → Hypothetical template: "The rule exists" cannot become "If the rule existed..."
- If you detect a logic mismatch, output **SKIPPING: LOGIC_MISMATCH** instead of generating a contradictory sentence.

**Output:** Return ONLY the final sentence with meaning from the Original Text injected into the skeleton structure, OR **SKIPPING: LOGIC_MISMATCH** if the template forces a logical contradiction."""

        try:
            mutated = llm_provider.call(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                temperature=temperature,
                max_tokens=max_tokens
            )
            from src.generator.llm_interface import clean_generated_text
            mutated = clean_generated_text(mutated)
            mutated = mutated.strip()
            return mutated if mutated else current_draft
        except Exception:
            return current_draft


# Operator constants for easy reference
OP_SEMANTIC_INJECTION = "semantic_injection"
OP_GRAMMAR_REPAIR = "grammar_repair"
OP_STYLE_POLISH = "style_polish"
OP_DYNAMIC_STYLE = "dynamic_style"
OP_STRUCTURAL_CLONE = "structural_clone"


def get_operator(operator_type: str) -> MutationOperator:
    """Get a mutation operator by type.

    Args:
        operator_type: One of OP_SEMANTIC_INJECTION, OP_GRAMMAR_REPAIR, OP_STYLE_POLISH, OP_DYNAMIC_STYLE, OP_STRUCTURAL_CLONE.

    Returns:
        MutationOperator instance.
    """
    operators = {
        OP_SEMANTIC_INJECTION: SemanticInjectionOperator(),
        OP_GRAMMAR_REPAIR: GrammarRepairOperator(),
        OP_STYLE_POLISH: StylePolishOperator(),
        OP_DYNAMIC_STYLE: DynamicStyleOperator(),
        OP_STRUCTURAL_CLONE: StructuralCloneOperator()
    }
    return operators.get(operator_type, GrammarRepairOperator())  # Default fallback

