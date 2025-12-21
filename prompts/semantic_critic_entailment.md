You are checking for **Entailment** (logical preservation), not just similarity.

**Task:** Determine if the Generated Sentence **entails** (preserves the meaning of) the Original Proposition, even if it uses different words or adds stylistic framing.

**Rules:**
1. If the Generated Sentence *contradicts* the Proposition → FAIL (e.g., "finite" vs "infinite", "exists" vs "does not exist")
2. If the Generated Sentence adds stylistic framing that doesn't contradict → PASS
3. If the Generated Sentence preserves core facts but rephrases → PASS
4. Style expansion is allowed if it doesn't change core meaning

**Examples:**
- Proposition: "Social structures are vapor."
- Generated: "Social structures are vapor before the material forces of history."
- Evaluation: Does Generated contradict Proposition? NO. Does it add framing? YES.
- Verdict: **PASS** (Style expansion is allowed).

- Proposition: "The economy collapsed."
- Generated: "The economy flourished."
- Evaluation: Does Generated contradict Proposition? YES (collapsed ≠ flourished).
- Verdict: **FAIL** (Contradiction detected).

- Proposition: "I spent my childhood scavenging."
- Generated: "My childhood was characterized by scavenging activities."
- Evaluation: Does Generated contradict Proposition? NO. Does it preserve meaning? YES.
- Verdict: **PASS** (Rephrasing preserves meaning).

**Output JSON:** {'entails': bool, 'confidence': float (0.0-1.0), 'reason': '...'}

