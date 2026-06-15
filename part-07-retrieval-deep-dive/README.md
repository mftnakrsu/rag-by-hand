# Part 7 — Retrieval Deep Dive

> Dense and sparse retrieval fail in opposite directions — so we build both by hand and fuse them.

[📖 Read the essay](https://www.mefby.com/essays/retrieval-deep-dive) · [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mftnakrsu/rag-by-hand/blob/main/part-07-retrieval-deep-dive/rag_hybrid.ipynb)

## What it covers
- Why **dense** (semantic) search blurs an exact rare token — a code, SKU, or name — and buries the chunk that literally answers the query.
- **Sparse** keyword retrieval: TF-IDF, then **BM25** with its two corrections — term-frequency saturation and document-length normalization.
- The complementarity insight — the two fail in *opposite* directions, so combining them wins.
- **Hybrid search** two ways: a normalized **weighted sum** (with the alpha knob) and **Reciprocal Rank Fusion (RRF)**, which merges by rank and skips normalization.
- **Top-k** as a real dial: too small starves recall, too large invites noise and *lost-in-the-middle*.

## Files
- **`rag_hybrid.py`** — the single runnable script: BM25, both fusion functions, and the E-4042 rescue printed top to bottom.
- **`rag_hybrid.ipynb`** — step-by-step notebook: a markdown *why* before each small code step, built cell by cell.

## Run it
```bash
# from the repo root
pip install -r requirements.txt
python part-07-retrieval-deep-dive/rag_hybrid.py        # runs offline — no API key
```
Prefer it step by step? Open `rag_hybrid.ipynb` in Jupyter, or click **Open in Colab** above.

## Offline by design
`rag_hybrid.py` is pure standard library: a hand-rolled BM25, RRF, and weighted fusion, with a deterministic dense stand-in in place of the Part 6 embedder so the demo prints the same numbers every run. The E-4042 exact-code example is preserved, and nothing touches the network.

---
← [Part 6 — Build Your First RAG](../part-06-build-your-first-rag/) · [Series index](../) · [Part 8 — Making Retrieval Smarter](../part-08-making-retrieval-smarter/) →
