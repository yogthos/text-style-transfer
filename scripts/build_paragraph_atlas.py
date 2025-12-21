import json
import os
import argparse
import spacy
import numpy as np
import pandas as pd
import re
import sys
from pathlib import Path
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from collections import defaultdict, Counter
import chromadb
from chromadb.config import Settings
from typing import List, Dict, Any

# Add project root to path to import src modules
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.generator.llm_provider import LLMProvider

# Configuration
MIN_SENTENCES = 2   # Ignore single-sentence fragments
MIN_STYLE_SCORE = 4 # Only keep "Strong Style Match" or better (1-5 scale)

def calculate_optimal_clusters(num_paragraphs: int, min_clusters: int = 8, max_clusters: int = 50) -> int:
    """
    Calculate optimal number of clusters based on corpus size.
    Heuristic: aim for sqrt(n) clusters to maximize diversity without over-segmenting.
    """
    if num_paragraphs < 10:
        return min(num_paragraphs, min_clusters)

    optimal = int(np.sqrt(num_paragraphs))
    optimal = max(min_clusters, min(optimal, max_clusters))
    return optimal

def audit_style_fidelity(text: str, llm: LLMProvider, author: str) -> int:
    """
    Uses LLM to rate the distinctiveness of the paragraph.
    Returns score 1-5.
    """
    # Quick filter: Skip very short fragments to save API calls
    if len(text) < 50:
        return 1

    system_prompt = f"You are an expert on the rhetorical style of {author}."
    user_prompt = f"""Analyze this paragraph:
    "{text[:1000]}"

    Does it reflect the distinct rhetorical style of {author}?
    Score (1-5):
    5 = Perfect Signature Style (Complex, distinctive, dialectical)
    4 = Strong Style Match (Clear rhetorical voice)
    3 = Moderate (Acceptable but standard)
    2 = Weak/Generic (Could be any author)
    1 = Irrelevant/Bad Data

    Output JSON ONLY: {{"score": int}}"""

    try:
        response = llm.call(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            model_type="editor", # Use a fast/cheap model here
            require_json=True,
            max_tokens=50
        )
        # Robust JSON parsing
        match = re.search(r'\{.*?\}', response, re.DOTALL)
        if match:
            data = json.loads(match.group(0))
            return int(data.get("score", 3))
        return 3
    except Exception as e:
        # Default to 3 (Moderate) on error to avoid dropping valid data due to API glitches
        # But print warning so user knows
        print(f"    ⚠ Audit warning: {e}")
        return 3

# Load Spacy
print("Loading spaCy model...")
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    print("Downloading spaCy model...")
    from spacy.cli import download
    download("en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")

class ParagraphAnalyzer:
    def __init__(self, config_path="config.json", author_name="the author"):
        self.scaler = StandardScaler()
        # KMeans will be initialized later once we know K
        self.kmeans = None
        # Initialize LLM for auditing
        print(f"Initializing LLM Provider for style auditing (Config: {config_path})...")
        self.llm = LLMProvider(config_path=config_path)
        self.author_name = author_name

    def get_paragraph_features(self, text: str) -> Dict[str, float]:
        """
        Extracts a statistical fingerprint from a paragraph.
        """
        doc = nlp(text)
        sents = list(doc.sents)

        if len(sents) < MIN_SENTENCES:
            return None

        # 1. Rhythm & Burstiness
        sent_lens = [len(sent) for sent in sents]
        avg_len = np.mean(sent_lens)
        # Coefficient of Variation measures relative burstiness
        burstiness = np.std(sent_lens) / avg_len if avg_len > 0 else 0

        # 2. Syntax & Complexity
        # Average depth of dependency tree
        depths = [token.head.i - token.i for token in doc]
        complexity_score = np.mean([abs(d) for d in depths])

        # Clause density
        verb_count = len([t for t in doc if t.pos_ == "VERB"])
        clause_count = len([t for t in doc if t.dep_ in ["mark", "advcl"]])
        clause_density = clause_count / len(sents)

        # 3. Flavor (POS Ratios)
        total_tokens = len(doc)
        noun_ratio = len([t for t in doc if t.pos_ in ["NOUN", "PROPN"]]) / total_tokens
        verb_ratio = verb_count / total_tokens
        adj_ratio = len([t for t in doc if t.pos_ == "ADJ"]) / total_tokens

        # 4. Structure
        is_question = 1.0 if text.strip().endswith("?") else 0.0

        return {
            "sentence_count": float(len(sents)),
            "avg_len": avg_len,
            "burstiness": burstiness,
            "complexity": complexity_score,
            "clause_density": clause_density,
            "noun_ratio": noun_ratio,
            "verb_ratio": verb_ratio,
            "adj_ratio": adj_ratio,
            "ends_question": is_question
        }

    def analyze_corpus(self, file_paths: List[str]):
        """
        Reads files, Audits for Style, extracts features.
        """
        raw_data = []
        feature_matrix = []

        total_paras = 0
        kept_paras = 0
        skipped_short = 0
        skipped_style = 0

        print(f"Analyzing {len(file_paths)} files...")

        for fp in file_paths:
            with open(fp, 'r', encoding='utf-8') as f:
                content = f.read()

            paras = [p.strip() for p in content.split('\n\n') if p.strip()]

            doc_paras = []

            # Process paragraphs
            print(f"  File {os.path.basename(fp)}: {len(paras)} paragraphs found.")

            for i, p_text in enumerate(paras):
                total_paras += 1

                # 1. Feature Check (Cheap)
                feats = self.get_paragraph_features(p_text)
                if not feats:
                    skipped_short += 1
                    continue

                # 2. LLM Style Audit (Expensive - Gatekeeper)
                # Print progress dot every 5 paras
                if i % 5 == 0:
                    print(".", end="", flush=True)

                style_score = audit_style_fidelity(p_text, self.llm, self.author_name)

                if style_score < MIN_STYLE_SCORE:
                    skipped_style += 1
                    continue # Drop generic paragraphs

                # If passed:
                entry = {"text": p_text, "style_score": style_score, **feats}
                doc_paras.append(entry)
                raw_data.append(entry)
                feature_matrix.append(list(feats.values()))
                kept_paras += 1

            print() # Newline after dots

            # Store sequential relationship
            for i, entry in enumerate(doc_paras):
                entry['doc_id'] = fp
                entry['seq_id'] = i

        print(f"\nAnalysis Complete:")
        print(f"  Total Paragraphs: {total_paras}")
        print(f"  Skipped (Too Short): {skipped_short}")
        print(f"  Skipped (Weak Style < {MIN_STYLE_SCORE}): {skipped_style}")
        print(f"  Kept (High Quality): {kept_paras}")

        return raw_data, np.array(feature_matrix)

    def build_archetypes(self, raw_data, feature_matrix):
        print("Clustering paragraphs into archetypes...")
        scaled_features = self.scaler.fit_transform(feature_matrix)
        labels = self.kmeans.fit_predict(scaled_features)

        for i, entry in enumerate(raw_data):
            entry['archetype_id'] = int(labels[i])

        return raw_data

    def describe_archetypes(self, raw_data, num_clusters):
        df = pd.DataFrame(raw_data)
        descriptions = {}

        # Calculate global averages across all paragraphs
        global_avg_noun_ratio = df['noun_ratio'].mean()
        global_avg_verb_ratio = df['verb_ratio'].mean()
        global_avg_adj_ratio = df['adj_ratio'].mean()
        global_avg_clause_density = df['clause_density'].mean()

        for i in range(num_clusters):
            cluster = df[df['archetype_id'] == i]
            if len(cluster) == 0:
                continue

            desc = {
                "id": i,
                "count": int(len(cluster)),
                "avg_sents": round(cluster['sentence_count'].mean(), 1),
                "avg_len": round(cluster['avg_len'].mean(), 1),
                "burstiness": "High" if cluster['burstiness'].mean() > 0.6 else "Low",
                "style": "Noun-Heavy/Academic" if cluster['noun_ratio'].mean() > 0.25 else "Balanced",
                "avg_style_score": round(cluster['style_score'].mean(), 1),
                # Pick the highest style-scored text as the example
                "example": cluster.sort_values('style_score', ascending=False).iloc[0]['text'],
                # Add cluster-specific averages for style directives
                "noun_ratio": round(cluster['noun_ratio'].mean(), 3),
                "verb_ratio": round(cluster['verb_ratio'].mean(), 3),
                "adj_ratio": round(cluster['adj_ratio'].mean(), 3),
                "clause_density": round(cluster['clause_density'].mean(), 3)
            }
            descriptions[i] = desc
            print(f"Archetype {i} (n={desc['count']}, Score={desc['avg_style_score']}): {desc['style']}, {desc['burstiness']} burstiness")

        # Store global averages in return value for metadata
        descriptions["_global_averages"] = {
            "noun_ratio": round(global_avg_noun_ratio, 3),
            "verb_ratio": round(global_avg_verb_ratio, 3),
            "adj_ratio": round(global_avg_adj_ratio, 3),
            "clause_density": round(global_avg_clause_density, 3)
        }

        return descriptions

    def build_transition_matrix(self, raw_data):
        transitions = defaultdict(Counter)

        # Group by doc
        by_doc = defaultdict(list)
        for entry in raw_data:
            if entry.get('doc_id'):
                by_doc[entry['doc_id']].append(entry)

        # Sort by sequence
        for doc_id in by_doc:
            by_doc[doc_id].sort(key=lambda x: x['seq_id'])

        # Count transitions
        for doc_id, doc_paras in by_doc.items():
            for i in range(len(doc_paras) - 1):
                curr = doc_paras[i]
                next_p = doc_paras[i + 1]

                # Verify consecutive
                if curr['seq_id'] + 1 == next_p['seq_id']:
                    transitions[curr['archetype_id']][next_p['archetype_id']] += 1

        # Calculate probabilities
        matrix = {}
        for src, counts in transitions.items():
            total = sum(counts.values())
            if total > 0:
                matrix[src] = {tgt: count/total for tgt, count in counts.items()}

        return matrix

def main():
    parser = argparse.ArgumentParser(description="Build a paragraph atlas from corpus")
    parser.add_argument("corpus_file", type=str, help="Path to corpus text file")
    parser.add_argument("--author", type=str, required=True, help="Author name")
    parser.add_argument("--config", type=str, default="config.json", help="Path to config.json")
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--clusters", type=int, default=None)
    parser.add_argument("--min-clusters", type=int, default=8)
    parser.add_argument("--max-clusters", type=int, default=60)

    args = parser.parse_args()

    # Validate inputs
    if not os.path.exists(args.corpus_file):
        raise FileNotFoundError(f"Corpus file not found: {args.corpus_file}")

    safe_author = "".join(c for c in args.author if c.isalnum() or c in (' ', '-', '_')).strip().replace(' ', '_').lower()

    if args.output_dir:
        output_dir = args.output_dir
    else:
        output_dir = os.path.join("atlas_cache", "paragraph_atlas", safe_author)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print(f"Corpus: {args.corpus_file}")
    print(f"Author: {args.author} (Safe: {safe_author})")
    print(f"Output: {output_dir}")

    # Initialize Analyzer
    analyzer = ParagraphAnalyzer(config_path=args.config, author_name=args.author)

    # 1. Analyze & Audit
    raw_data, feature_matrix = analyzer.analyze_corpus([args.corpus_file])

    if len(raw_data) == 0:
        raise ValueError("No valid paragraphs found (Check style scores or file format).")

    # 1.5. Extract Style Profile from full corpus
    print("\nExtracting style profile...")
    with open(args.corpus_file, 'r', encoding='utf-8') as f:
        full_corpus_text = f.read()

    from src.analysis.style_profiler import StyleProfiler
    profiler = StyleProfiler()
    style_profile = profiler.analyze_style(full_corpus_text)

    # Save style profile
    style_profile_path = os.path.join(output_dir, "style_profile.json")
    with open(style_profile_path, 'w') as f:
        json.dump(style_profile, f, indent=2)
    print(f"Style profile saved to {style_profile_path}")
    print(f"  POV: {style_profile.get('pov', 'Unknown')}")
    print(f"  Rhythm: {style_profile.get('rhythm_desc', 'Unknown')} (Burstiness: {style_profile.get('burstiness', 0.0)})")
    print(f"  Top Keywords: {', '.join(style_profile.get('keywords', [])[:10])}")
    print(f"  Common Openers: {', '.join(style_profile.get('common_openers', [])[:5])}")

    # 2. Determine Clusters
    if args.clusters is None:
        num_clusters = calculate_optimal_clusters(len(raw_data), args.min_clusters, args.max_clusters)
        print(f"Auto-calculated clusters: {num_clusters}")
    else:
        num_clusters = args.clusters

    analyzer.kmeans = KMeans(n_clusters=num_clusters, random_state=42)

    # 3. Cluster
    labeled_data = analyzer.build_archetypes(raw_data, feature_matrix)

    # 4. Describe
    print("\nArchetype Descriptions:")
    archetype_descriptions = analyzer.describe_archetypes(labeled_data, num_clusters)

    # 5. Transitions
    print("\nBuilding Transition Matrix...")
    transition_matrix = analyzer.build_transition_matrix(labeled_data)
    print(f"Found transitions for {len(transition_matrix)} archetypes")

    # 6. Save Artifacts
    print("\nSaving artifacts...")

    # Extract global averages before adding metadata
    global_averages = archetype_descriptions.pop("_global_averages", {})

    # Add metadata
    archetype_descriptions["_metadata"] = {
        "author": args.author,
        "corpus_file": args.corpus_file,
        "num_clusters": num_clusters,
        "total_paragraphs": len(labeled_data),
        "min_style_score": MIN_STYLE_SCORE,
        "global_averages": global_averages
    }

    # Save JSONs
    with open(os.path.join(output_dir, "archetypes.json"), 'w') as f:
        json.dump(archetype_descriptions, f, indent=2)

    transition_data = {
        "matrix": transition_matrix,
        "_metadata": {
            "author": args.author,
            "num_transitions": sum(len(probs) for probs in transition_matrix.values())
        }
    }

    with open(os.path.join(output_dir, "transition_matrix.json"), 'w') as f:
        json.dump(transition_data, f, indent=2)

    # Save to ChromaDB
    client = chromadb.PersistentClient(path=os.path.join(output_dir, "chroma"))
    collection_name = f"paragraph_archetypes_{safe_author}"

    # Reset collection
    try:
        client.delete_collection(collection_name)
    except:
        pass

    collection = client.create_collection(collection_name)

    # Add to Chroma
    ids = [str(i) for i in range(len(labeled_data))]
    docs = [d['text'] for d in labeled_data]
    metadatas = [
        {"archetype_id": d['archetype_id'], "style_score": d['style_score']}
        for d in labeled_data
    ]

    # Batch upload
    batch_size = 100
    for i in range(0, len(ids), batch_size):
        collection.add(
            ids=ids[i:i+batch_size],
            documents=docs[i:i+batch_size],
            metadatas=metadatas[i:i+batch_size]
        )

    print(f"Done. Atlas built at {output_dir}")

if __name__ == "__main__":
    main()