# Part 5 — Documents & Chunking

> The cut itself: how a raw document becomes the chunks we embed — and why that one decision quietly sets the ceiling for everything downstream.

[📖 Read the essay](https://www.mefby.com/essays/documents-and-chunking) · [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mftnakrsu/rag-by-hand/blob/main/part-05-chunking/chunking.ipynb)

## What it covers
- Why chunking is the highest-leverage, most underrated step in RAG: garbage chunks in, garbage retrieval out.
- The central tension — too small turns ambiguous, too large blurs the embedding — and aiming for semantically coherent units.
- The ladder of strategies: fixed-size, recursive character (the sensible default), structure-aware, and a semantic-ish cut.
- The two dials that decide retrieval quality — chunk size and overlap (a sliding window that protects ideas on a seam).
- Metadata enrichment and a context prefix, so each chunk carries its origin and stands on its own.

## Files
- **`chunking.py`** — the single runnable script: all six lenses applied to one sample document, top to bottom.
- **`chunking.ipynb`** — step-by-step notebook: a markdown *why* before each small code step, built cell by cell.

## Run it
```bash
# from the repo root
pip install -r requirements.txt
python part-05-chunking/chunking.py        # runs offline — no API key
```
Prefer it step by step? Open `chunking.ipynb` in Jupyter, or click **Open in Colab** above.

## Offline by design
Every strategy here is pure Python standard library — no models, no network, no API key (`numpy` is optional, used in one tiny helper). Where production would reach for a real tool — LangChain's `RecursiveCharacterTextSplitter`, or an embedding model for the semantic cut — that code is shown as labelled reference only, and a transparent lexical stand-in (Jaccard word overlap) keeps the semantic split executable.

---
← [Part 4 — Vector Databases & Indexing](../part-04-vector-databases/) · [Series index](../) · [Part 6 — Build Your First RAG](../part-06-build-your-first-rag/) →
