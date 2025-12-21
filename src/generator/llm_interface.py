"""LLM interface utilities for text cleaning.

This module provides text cleaning functions to remove LLM artifacts
and preserve citations and direct quotations.
"""

import re


def clean_generated_text(text: str) -> str:
    """Deterministic cleanup of common LLM artifacts.

    Protects citations [^number] and quotes from being modified by cleanup operations.
    Also removes prompt leakage, instruction lines, and hallucinated citation formats.
    """
    if not text: return ""

    # NEW: Extract text after "REWRITE:" marker (if present)
    if "REWRITE:" in text.upper() or "REWRITE" in text.upper():
        # Try to find text after REWRITE: marker
        parts = re.split(r'REWRITE:?\s*', text, flags=re.IGNORECASE)
        if len(parts) > 1:
            # Take the last part (after the last REWRITE marker)
            text = parts[-1].strip()
        # Also handle "Rewrite:" with lowercase
        parts = re.split(r'Rewrite:?\s*', text, flags=re.IGNORECASE)
        if len(parts) > 1:
            text = parts[-1].strip()

    # NEW: Remove prompt instruction lines
    lines = text.split('\n')
    filtered_lines = []
    skip_keywords = [
        "CRITICAL:", "REWRITE:", "INPUT TEXT:", "TASK:", "INSTRUCTIONS:",
        "Subjects:", "Actions:", "Objects:", "Entities:", "Keywords:",
        "Citations:", "Quotes:", "Citations to include:", "Quotes to preserve:",
        "Citations to preserve:", "Direct quotes to preserve exactly:"
    ]
    for line in lines:
        line_stripped = line.strip()
        # Skip lines that start with any of the instruction keywords
        if any(line_stripped.startswith(kw) for kw in skip_keywords):
            continue
        # Skip lines that are just separators or empty
        if line_stripped in ["===", "---", ""]:
            continue
        filtered_lines.append(line)
    text = '\n'.join(filtered_lines)

    # NEW: Remove academic citation patterns BEFORE protecting valid citations
    # Remove (Author, Year, p. #) format
    text = re.sub(r'\([A-Z][a-z]+,?\s+\d{4}(?:,\s*p\.\s*#?)?\)', '', text)
    # Remove (Author, Year, p. #) template pattern
    text = re.sub(r'\(Author,?\s+Year,?\s+p\.\s*#\)', '', text, flags=re.IGNORECASE)
    # Remove (Smith 42) format
    text = re.sub(r'\([A-Z][a-z]+\s+\d+\)', '', text)

    # Protect citations and quotes by temporarily replacing them with placeholders
    citation_pattern = r'\[\^\d+\]'
    quote_pattern = r'["\'](?:[^"\']|(?<=\\)["\'])*["\']'

    # Store protected content
    protected_items = []
    placeholder_map = {}

    # Replace citations with placeholders
    def replace_citation(match):
        placeholder = f"__CITATION_{len(protected_items)}__"
        protected_items.append(match.group(0))
        placeholder_map[placeholder] = match.group(0)
        return placeholder

    text = re.sub(citation_pattern, replace_citation, text)

    # Replace quotes with placeholders
    def replace_quote(match):
        quote_text = match.group(0)
        # Only protect substantial quotes
        if len(quote_text.strip('"\'')) > 2:
            placeholder = f"__QUOTE_{len(protected_items)}__"
            protected_items.append(quote_text)
            placeholder_map[placeholder] = quote_text
            return placeholder
        return quote_text

    text = re.sub(quote_pattern, replace_quote, text)

    # 1. Fix Punctuation Spacing (" . " -> ". ")
    # But avoid affecting protected placeholders
    text = re.sub(r'\s+([.,!?;:])', r'\1', text)

    # 2. Fix Multiple Periods (".." -> ".") excluding ellipses
    text = re.sub(r'\.{2,}', '...', text)
    text = re.sub(r'\.\.\.(?!\.)', '...', text) # Ensure consistent ellipses

    # 3. Capitalization (Start of sentence)
    # But avoid capitalizing inside protected placeholders
    def capitalize_match(m):
        return f"{m.group(1)}{m.group(2).upper()}"
    text = re.sub(r'(^|[.!?]\s+)([a-z])', capitalize_match, text)

    # 4. Strip Metadata Artifacts (e.g., "Chapter 1")
    text = re.sub(r'(?i)\b(chapter|section|part)\s+([0-9]+|[ivx]+)\b', '', text)

    # 5. Remove "Output:" prefixes
    text = re.sub(r'^(output|response|rewritten):\s*', '', text, flags=re.IGNORECASE)

    # Restore protected citations and quotes
    for placeholder, original in placeholder_map.items():
        text = text.replace(placeholder, original)

    return text.strip()

