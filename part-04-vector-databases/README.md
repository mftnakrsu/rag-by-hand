# Part 4 — Vector Databases & Indexing

> From exact, brute-force k-NN to an approximate index, and the one trade-off the whole field turns on: speed versus recall.

[📖 Read the essay](https://www.mefby.com/essays/vector-databases) · [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mftnakrsu/rag-by-hand/blob/main/part-04-vector-databases/vector_db.ipynb)

## What it covers
- **Brute-force k-NN**: exact but O(n) — one dot product per stored vector, fine for thousands, a wall at millions.
- **recall@k**: the word for the sliver of accuracy approximate search gives up (find 8 of the true 10 → 0.8).
- **IVF**: cluster vectors onto k-means "shelves," then search only the `nprobe` nearest — the dial that slides you along the speed-recall curve.
- An **nprobe sweep** that prints recall and speedup side by side, so you watch the trade-off in numbers.
- **Product Quantization**: a sketch of compressing the vectors themselves into tiny codes at a small accuracy cost.

## Files
- **`vector_db.py`** — the single runnable script: builds the point cloud, runs brute force, builds IVF, sweeps `nprobe`, and sketches PQ, top to bottom.
- **`vector_db.ipynb`** — step-by-step notebook: a markdown *why* before each small code step, built cell by cell.

## Run it
```bash
# from the repo root
pip install -r requirements.txt
python part-04-vector-databases/vector_db.py        # runs offline — no API key
```
Prefer it step by step? Open `vector_db.ipynb` in Jupyter, or click **Open in Colab** above.

## Offline by design
Pure NumPy and the standard library: a seeded point cloud stands in for embeddings, and the brute-force k-NN, k-means/IVF, and PQ sketch are all written by hand so nothing is hidden. The real production path (FAISS, the reference ANN toolkit) is shown as the headline inside a `try/except` that falls back to the transparent NumPy index if it isn't installed. No network, no model, same numbers every run.

---
← [Part 3 — Measuring Similarity](../part-03-measuring-similarity/) · [Series index](../) · [Part 5 — Documents & Chunking](../part-05-chunking/) →
