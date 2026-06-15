# Part 3 — Measuring Similarity

> Turning "close in embedding space" into a single number you can rank by — Euclidean distance, the dot product, and the cosine similarity that powers RAG, all by hand.

[📖 Read the essay](https://www.mefby.com/essays/measuring-similarity) · [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mftnakrsu/rag-by-hand/blob/main/part-03-measuring-similarity/similarity.ipynb)

## What it covers
- **Similarity vs. distance** — same idea, opposite direction: similarity goes *up* as vectors get alike, distance goes *down*.
- **Euclidean distance** — the straight-line gap, intuitive but fooled by length.
- **Dot product** — fast and the one that runs at scale, but unnormalized: longer vectors win regardless of direction.
- **Cosine similarity** — angle only, length ignored; the default metric in RAG, and why.
- **The 'aha' + top-k** — cosine *is* the dot product of normalized vectors, and top-k is the ranking function at the heart of every RAG system.

## Files
- **`similarity.py`** — the single runnable script: all three metrics, the normalized-dot 'aha', and top-k retrieval, with the essay's worked numbers checked as asserts.
- **`similarity.ipynb`** — step-by-step notebook: a markdown *why* before each small code step, built cell by cell.

## Run it
```bash
# from the repo root
pip install -r requirements.txt
python part-03-measuring-similarity/similarity.py        # runs offline — no API key
```
Prefer it step by step? Open `similarity.ipynb` in Jupyter, or click **Open in Colab** above.

## Offline by design
This part is arithmetic a calculator can do — pure NumPy and the standard library (Euclidean, dot, cosine, top-k). NumPy powers the one-liner, but a transparent pure-Python fallback covers the no-NumPy case, so every line runs the same way with no model, key, or network.

---
← [Part 2 — Embeddings](../part-02-embeddings/) · [Series index](../) · [Part 4 — Vector Databases & Indexing](../part-04-vector-databases/) →
