#!/usr/bin/env python3
"""Clean sample text files for ChromaDB ingestion.

Removes:
- Page numbers (standalone numbers)
- Roman numerals (IX, X, xi, etc.)
- Repeated section headers
- Excessive blank lines
- Lines with only whitespace and headers
"""

import re
import sys
from pathlib import Path
from typing import List


def is_roman_numeral(text: str) -> bool:
    """Check if text is a Roman numeral (case-insensitive)."""
    text = text.strip().upper()
    # Match common Roman numerals (I through XXX, including XIV, XVI, XVII, etc.)
    # This pattern matches standalone Roman numerals used for chapters/sections
    roman_pattern = r'^(I{1,3}|IV|VI{0,3}|IX|X{1,3}|XI{0,3}|XIV|XV|XVI|XVII|XVIII|XIX|XX|XXX)$'
    return bool(re.match(roman_pattern, text))


def is_page_number(text: str) -> bool:
    """Check if line is just a page number."""
    text = text.strip()
    # Match standalone numbers (1-9999, typical page range)
    return bool(re.match(r'^\d{1,4}$', text))


def is_repeated_header(text: str, seen_headers: set) -> bool:
    """Check if line is a repeated section header."""
    text = text.strip()
    # Common section headers that might repeat
    common_headers = {
        'Introduction', 'Preface', 'Chapter', 'Contents',
        'The Blind Watchmaker', 'Doomed rivals', 'Acknowledgements'
    }

    if text in common_headers:
        if text in seen_headers:
            return True
        seen_headers.add(text)
    return False


def join_hyphenated_words(lines: List[str]) -> List[str]:
    """Join words that are split across lines with hyphens.

    Example:
        "The physi-\n" + "cist's problem" -> "The physicist's problem"
        "dif-\n" + "ferent" -> "different"

    Handles multiple consecutive hyphenated words.

    Args:
        lines: List of text lines.

    Returns:
        List of lines with hyphenated words joined.
    """
    if not lines:
        return []

    joined_lines = []
    i = 0

    while i < len(lines):
        line = lines[i].rstrip()

        # Skip empty lines (preserve them)
        if not line:
            if joined_lines and joined_lines[-1].strip():
                joined_lines.append('\n')
            i += 1
            continue

        # Check if line ends with a hyphenated word
        stripped = line.rstrip()
        if stripped.endswith('-') and i + 1 < len(lines):
            # Try to join with next line(s) - handle multiple consecutive splits
            current_line = stripped[:-1].rstrip()  # Remove hyphen
            j = i + 1
            joined_anything = False

            # Keep joining while we find more hyphenated continuations
            while j < len(lines):
                next_line_raw = lines[j].strip()
                if not next_line_raw:
                    break

                # Get first word of next line
                next_words = next_line_raw.split()
                if not next_words:
                    break

                first_word = next_words[0]

                # Check if it's a continuation
                # More strict: the first word should be a short fragment that completes the word
                # to avoid joining "physi-" with "biologist" (different word)
                line_before_hyphen = current_line
                word_before_hyphen = line_before_hyphen.split()[-1] if line_before_hyphen.split() else ""

                # Get first word without punctuation for length check
                first_word_clean = first_word.rstrip('.,!?;:')

                # Check if first word of next line could be a continuation
                # Conditions:
                # 1. Starts with lowercase (not a new sentence)
                # 2. First word is short (likely a fragment, not a complete word like "biologist")
                # 3. Word before hyphen is short/incomplete (indicating it was split)
                # 4. Not after sentence-ending punctuation
                is_continuation = (
                    first_word[0].islower() and
                    len(first_word_clean) <= 8 and  # Short fragment (e.g., "cist's", "ferent", not "biologist")
                    len(word_before_hyphen) <= 6 and  # Short incomplete word (e.g., "physi", "dif", not "continuation")
                    not (line_before_hyphen.endswith('.') or
                         line_before_hyphen.endswith('!') or
                         line_before_hyphen.endswith('?') or
                         line_before_hyphen.endswith(':'))
                )

                if is_continuation:
                    # Join this line - preserve ALL content from next_line_raw
                    current_line = current_line + next_line_raw
                    j += 1
                    joined_anything = True

                    # Check if the newly joined line also ends with hyphen
                    current_line_stripped = current_line.rstrip()
                    if current_line_stripped.endswith('-'):
                        # Continue joining - remove hyphen but keep rest of line
                        current_line = current_line_stripped[:-1].rstrip()
                        # Make sure we preserve any content after the hyphen on the same line
                        # (though typically there won't be any if it's a line break)
                        continue
                    else:
                        # Done joining, add the complete line with all content
                        joined_lines.append(current_line + '\n')
                        i = j
                        break
                else:
                    # Not a continuation, stop joining
                    # If we already joined something, add what we have
                    if joined_anything:
                        joined_lines.append(current_line + '\n')
                        i = j
                    else:
                        # Never joined anything, keep original line
                        joined_lines.append(line + '\n')
                        i += 1
                    break

            # If we tried to join but couldn't find any continuation, add original line
            if not joined_anything:
                joined_lines.append(line + '\n')
                i += 1
                continue

            # If we reached end of file while joining, add what we have
            if j >= len(lines) and joined_anything:
                joined_lines.append(current_line + '\n')
                i = j
                continue

        # Not a split word, keep as is
        joined_lines.append(line + '\n')
        i += 1

    return joined_lines


def clean_sample_text(input_path: str, output_path: str = None) -> str:
    """Clean sample text file for ChromaDB ingestion.

    Args:
        input_path: Path to input text file.
        output_path: Optional path to output file. If None, overwrites input.

    Returns:
        Cleaned text as string.
    """
    input_file = Path(input_path)
    if not input_file.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    # Read input file
    with open(input_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    # First pass: Join hyphenated words split across lines
    lines = join_hyphenated_words(lines)

    cleaned_lines = []
    seen_headers = set()
    prev_line_blank = False

    for line in lines:
        stripped = line.strip()

        # Skip empty lines (we'll add them back selectively)
        if not stripped:
            # Only add blank line if previous line wasn't blank (normalize spacing)
            if not prev_line_blank:
                cleaned_lines.append('\n')
            prev_line_blank = True
            continue

        prev_line_blank = False

        # Skip page numbers
        if is_page_number(stripped):
            continue

        # Skip Roman numerals
        if is_roman_numeral(stripped):
            continue

        # Skip repeated headers
        if is_repeated_header(stripped, seen_headers):
            continue

        # Skip lines that are just headers with lots of whitespace
        # (e.g., "Introduction" centered on a line)
        if len(stripped) < 20 and stripped in seen_headers:
            # Check if the original line had mostly whitespace
            if len(line) - len(stripped) > 20:
                continue

        # Keep the line
        cleaned_lines.append(line.rstrip() + '\n')

    # Join cleaned lines
    cleaned_text = ''.join(cleaned_lines)

    # Final pass: normalize multiple blank lines to max 2 consecutive
    cleaned_text = re.sub(r'\n{3,}', '\n\n', cleaned_text)

    # Normalize whitespace: remove leading spaces/tabs from each line
    # This handles cases where entire file or individual lines have leading whitespace
    normalized_lines = []
    for line in cleaned_text.split('\n'):
        # Remove leading spaces/tabs but preserve the line structure
        # Keep empty lines as-is
        if line.strip():
            normalized_lines.append(line.lstrip())
        else:
            normalized_lines.append('')

    cleaned_text = '\n'.join(normalized_lines)

    # Remove leading/trailing whitespace from entire document
    cleaned_text = cleaned_text.strip() + '\n'

    # Write output
    if output_path is None:
        output_path = input_path

    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(cleaned_text)

    return cleaned_text


def main():
    """CLI entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description='Clean sample text files for ChromaDB ingestion',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s prompts/sample_dawkins.txt
  %(prog)s prompts/sample_dawkins.txt -o prompts/sample_dawkins_clean.txt
  %(prog)s prompts/sample.txt --in-place
        """
    )

    parser.add_argument(
        'input',
        type=str,
        help='Input text file to clean'
    )

    parser.add_argument(
        '-o', '--output',
        type=str,
        default=None,
        help='Output file path (default: overwrites input)'
    )

    parser.add_argument(
        '--in-place',
        action='store_true',
        help='Modify input file in place (same as not specifying -o)'
    )

    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='Enable verbose output'
    )

    args = parser.parse_args()

    # Determine output path
    if args.in_place or args.output is None:
        output_path = args.input
    else:
        output_path = args.output

    try:
        if args.verbose:
            print(f"Cleaning: {args.input}")
            print(f"Output: {output_path}")

        cleaned = clean_sample_text(args.input, output_path)

        # Count statistics
        original_lines = len(Path(args.input).read_text(encoding='utf-8').splitlines())
        cleaned_lines = len(cleaned.splitlines())
        removed = original_lines - cleaned_lines

        if args.verbose:
            print(f"\n✓ Cleaned successfully")
            print(f"  Original lines: {original_lines}")
            print(f"  Cleaned lines: {cleaned_lines}")
            print(f"  Removed: {removed} lines")
        else:
            print(f"✓ Cleaned {args.input} -> {output_path} (removed {removed} lines)")

        return 0

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())

