Analyze the following text and extract:

1. **The Thesis:** What is the main argument or central theme? (Maximum 2 sentences)
   - This is what the AUTHOR believes, not what cited figures believe.
2. **The Intent:** Is the author primarily persuading, informing, or narrating?
3. **Key Terminology:** List 5 central concepts or terms that must remain consistent throughout the document.
4. **Counter-Arguments:** What positions, views, or figures does the author oppose or criticize? (List 2-3 key counter-arguments)
   - Example: If the author criticizes Baudelaire's elitism, list "Baudelaire's elitist views on photography"
5. **Stance Markers:** How does the author position cited figures or opposing views?
   - List key phrases or patterns that show the author's stance (e.g., "criticizes", "rejects", "opposes", "supports", "agrees with")
   - Example: ["criticizes Baudelaire's elitism", "rejects the notion that..."]

Text to analyze:
{text_to_analyze}

Return your analysis as JSON with this exact structure:
{{
    "thesis": "Main argument in 1-2 sentences",
    "intent": "persuading" or "informing" or "narrating",
    "keywords": ["concept1", "concept2", "concept3", "concept4", "concept5"],
    "counter_arguments": ["counter-argument 1", "counter-argument 2"],
    "stance_markers": ["stance marker 1", "stance marker 2"]
}}

**Note:** If counter_arguments or stance_markers are not applicable, use empty arrays [].

