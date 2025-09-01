---

# 3) File: plagiarism_checker.py  
Create `plagiarism_checker.py` at repo root. This is ready-to-run.

```python
#!/usr/bin/env python3
"""
PlagiarismChecker (simple)
- Put .txt files inside sample_data/
- Run: python plagiarism_checker.py
"""
import os
import glob
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

DATA_DIR = "sample_data"
SENT_SIM_THRESHOLD = 0.75  # sentence-level similarity threshold (0-1)

def load_documents(data_dir):
    paths = sorted(glob.glob(os.path.join(data_dir, "*.txt")))
    if not paths:
        print(f"No .txt files found in {data_dir}. Please add sample_data/*.txt")
        return [], []
    docs = []
    filenames = []
    for p in paths:
        with open(p, "r", encoding="utf-8") as f:
            text = f.read().strip()
        docs.append(text)
        filenames.append(os.path.basename(p))
    return filenames, docs

def doc_level_similarity(docs, filenames):
    vec = TfidfVectorizer(stop_words="english")
    X = vec.fit_transform(docs)
    sim = cosine_similarity(X)
    n = len(filenames)
    print("\n--- Document-level similarity (%) ---")
    header = ["{:>20}".format("")] + ["{:>20}".format(name) for name in filenames]
    print("".join(header))
    for i in range(n):
        row = ["{:>20}".format(filenames[i])]
        for j in range(n):
            pct = sim[i,j]*100
            row.append("{:20.2f}".format(pct))
        print("".join(row))
    return sim

def split_sentences(text):
    # simple sentence splitter (no NLTK)
    sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', text) if s.strip()]
    return sentences

def sentence_level_matches(docs, filenames, top_n=5):
    n = len(docs)
    print("\n--- Sentence-level possible matches (threshold {:.2f}) ---".format(SENT_SIM_THRESHOLD))
    for i in range(n):
        for j in range(i+1, n):
            sents_i = split_sentences(docs[i])
            sents_j = split_sentences(docs[j])
            if not sents_i or not sents_j:
                continue
            all_sents = sents_i + sents_j
            vec = TfidfVectorizer(stop_words="english").fit_transform(all_sents)
            Ai = vec[:len(sents_i)]
            Aj = vec[len(sents_i):]
            sim_matrix = cosine_similarity(Ai, Aj)
            # find pairs exceeding threshold
            pairs = []
            for a in range(sim_matrix.shape[0]):
                for b in range(sim_matrix.shape[1]):
                    score = sim_matrix[a,b]
                    if score >= SENT_SIM_THRESHOLD:
                        pairs.append((score, sents_i[a], sents_j[b]))
            if pairs:
                pairs.sort(key=lambda x: x[0], reverse=True)
                print(f"\nMatches between '{filenames[i]}' and '{filenames[j]}':")
                for k, (score, sent_i, sent_j) in enumerate(pairs[:top_n], 1):
                    print(f"{k}. [{score:.2f}]")
                    print(f"   - {filenames[i]}: {sent_i}")
                    print(f"   - {filenames[j]}: {sent_j}")
    print("\nDone.")

def main():
    filenames, docs = load_documents(DATA_DIR)
    if not docs:
        return
    doc_level_similarity(docs, filenames)
    sentence_level_matches(docs, filenames)

if __name__ == "__main__":
    main()
