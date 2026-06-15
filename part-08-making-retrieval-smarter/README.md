# Part 8 — Making Retrieval Smarter

> First-pass retrieval is fast but only roughly right; here we sharpen it with three levers around retrieval — and the star is reranking.

[📖 Read the essay](https://www.mefby.com/essays/making-retrieval-smarter) · [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mftnakrsu/rag-by-hand/blob/main/part-08-making-retrieval-smarter/rag_rerank.ipynb)

## What it covers
- Why a bi-encoder first pass is structurally imprecise: query and chunk are embedded separately, so the best chunk can sit at rank six.
- Three levers, in pipeline order: **before** (transform the query), **during** (filter by metadata), **after** (rerank).
- The DURING lever: metadata pre-filtering shrinks the candidate set before scoring — for precision, freshness, and security.
- The AFTER lever: a cross-encoder reads `(query, chunk)` together and judges relevance far more accurately, but is too slow to run over a whole corpus.
- The headline pattern: **two-stage retrieve-then-rerank** — cast a wide net fast, then rerank it down to the best few.

## Files
- **`rag_rerank.py`** — the single runnable script: a corpus with metadata, a metadata filter, wide first-pass retrieval, and a cross-encoder rerank, top to bottom.
- **`rag_rerank.ipynb`** — step-by-step notebook: a markdown *why* before each small code step, built cell by cell.

## Run it
```bash
# from the repo root
pip install -r requirements.txt
python part-08-making-retrieval-smarter/rag_rerank.py        # runs offline — no API key
```
Prefer it step by step? Open `rag_rerank.ipynb` in Jupyter, or click **Open in Colab** above.

## Offline by design
The `.py` uses real models — a SentenceTransformer bi-encoder for the wide net and a `ms-marco` cross-encoder for the rerank — so it needs only a one-time model download, never an API key. The notebook goes further: it guards both stages with transparent, deterministic stand-ins (a hashing embedder and a lexical-overlap reranker), so the full two-stage retrieve-then-rerank pipeline runs even with no network and no weights cached.

---
← [Part 7 — Retrieval Deep Dive](../part-07-retrieval-deep-dive/) · [Series index](../) · [Part 9 — Advanced Retrieval Patterns](../part-09-advanced-retrieval-patterns/) →
