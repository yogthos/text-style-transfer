Analyze these {num_examples} text samples. Extract the following style characteristics:

1. **Lexicon:** List 5-10 distinct words or phrases this author uses frequently (e.g., 'dialectical', 'manifestation', 'objective', 'struggle', 'contradiction'). Include both single words and short phrases (2-3 words max). Focus on distinctive vocabulary, not common words.

2. **Tone:** One adjective describing the voice (e.g., 'Didactic', 'Minimalist', 'Authoritative', 'Poetic', 'Technical').

3. **Structure:** One rule about sentence structure (e.g., 'Uses passive voice frequently', 'Starts sentences with conjunctions', 'Prefers short, declarative sentences', 'Uses complex subordinate clauses').

Text samples:
{examples_text}

Output your analysis as a JSON object with exactly these keys:
{{
  "lexicon": ["word1", "word2", "phrase1", ...],
  "tone": "Adjective",
  "structure": "One structural rule"
}}

Return ONLY the JSON object, no additional text.

