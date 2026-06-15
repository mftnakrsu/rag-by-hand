# Part 2 — Embeddings

> Turning meaning into numbers so a computer can search by meaning, not by spelling.

[📖 Read the essay](https://www.mefby.com/essays/embeddings) · [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mftnakrsu/rag-by-hand/blob/main/part-02-embeddings/embeddings.ipynb)

## What it covers
- Why a computer needs text as numbers: it compares numbers fast, but can't compare meanings directly.
- Two naive encodings that fail instructively — **one-hot** (captures identity) and **bag-of-words** (captures word overlap, ignores order) — both huge, sparse, and meaning-blind.
- **Dense embeddings**, where meaning becomes *position in space*: similar texts land close, different ones land far.
- The chunk-vs-query comparison that finally works — and the classic `king − man + woman ≈ queen` arithmetic, where a *direction* in the space carries a concept.

## Files
- **`embeddings.py`** — the single runnable script: tokenize → one-hot → bag-of-words → dense embedding → analogy, top to bottom.
- **`embeddings.ipynb`** — step-by-step notebook: a markdown *why* before each small code step, built cell by cell.

## Run it
```bash
# from the repo root
pip install -r requirements.txt
python part-02-embeddings/embeddings.py        # runs offline — no API key
```
Prefer it step by step? Open `embeddings.ipynb` in Jupyter, or click **Open in Colab** above.

## Offline by design
The sparse baselines are pure numpy/stdlib, so they're transparent and always runnable. The dense embedder loads the real `all-MiniLM-L6-v2` inside a `try/except`; if the model or network isn't there, it falls back to a deterministic hashing stand-in that mimics the interface (text → fixed-length unit vector) so every line still runs with no download. The fallback can't capture synonyms — it's a hash — but it shows the *shape* and geometry honestly.

---
← [Part 1 — Why RAG Exists](../part-01-why-rag/) · [Series index](../) · [Part 3 — Measuring Similarity](../part-03-measuring-similarity/) →
