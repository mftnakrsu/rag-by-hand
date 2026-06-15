# Part 10 — Advanced RAG Architectures

> From a fixed retrieve-then-generate pipeline to a decision-making loop that grades what it got and corrects course before it answers.

[📖 Read the essay](https://www.mefby.com/essays/advanced-rag-architectures) · [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mftnakrsu/rag-by-hand/blob/main/part-10-advanced-architectures/corrective_rag.ipynb)

## What it covers
- The leap from a fixed pipeline (retrieve, then generate, every time) to a loop with **control flow**: decide, judge, retry.
- **Corrective RAG (CRAG)** in full: a lightweight **retrieval evaluator** grades chunks *relevant / ambiguous / irrelevant* and acts on the grade *before* generating.
- The three corrective branches — answer now, reformulate and retry our own index, or rewrite and fall back to a different source (web search).
- Honest refusal when neither source can answer, rather than a confident guess (Part 6's grounding).
- A reference-only peek at the **Agentic RAG** ReAct loop that CRAG is one disciplined slice of.

## Files
- **`corrective_rag.py`** — the single runnable script: two tiny corpora, an embedder, a retriever per source, the evaluator, both corrective actions, and the `corrective_rag()` loop over four worked cases, top to bottom.
- **`corrective_rag.ipynb`** — step-by-step notebook: a markdown *why* before each small code step, built cell by cell.

## Run it
```bash
# from the repo root
pip install -r requirements.txt
python part-10-advanced-architectures/corrective_rag.py        # runs offline — no API key
```
Prefer it step by step? Open `corrective_rag.ipynb` in Jupyter, or click **Open in Colab** above.

## Offline by design
CRAG is fully guarded so the whole demo runs with no network and no API key: a deterministic hashing-bag-of-words embedder stands in for sentence-transformers, a similarity-plus-overlap threshold grader stands in for an LLM evaluator, an extractive generator quotes the best chunk, and the "web search" fallback is a SIMULATED offline mini-corpus. The real model paths sit behind `try/except`, so they light up automatically when a model or key is present — but no real network call ever runs.

---
← [Part 9 — Advanced Retrieval Patterns](../part-09-advanced-retrieval-patterns/) · [Series index](../) · [Part 11 — Evaluating RAG](../part-11-evaluating-rag/) →
