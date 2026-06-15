# Part 15 — Adaptive RAG

> A small complexity classifier routes each query to the pipeline it actually needs — no retrieval, one lookup, or a multi-step decomposition. The close of the Frontier Track.

[📖 Read the essay](https://www.mefby.com/essays/adaptive-rag) · [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mftnakrsu/rag-by-hand/blob/main/part-15-adaptive-rag/adaptive_rag.ipynb)

## What it covers
- Why one fixed pipeline is wrong at both ends: it **over-serves** easy queries (paying for retrieval a greeting doesn't need) and **under-serves** hard ones (a single lookup can't answer a comparison).
- `classify_complexity(query)` — a tiny, fully legible rule/keyword classifier returning `none` / `single` / `multi`. In production it's a small trained model; here it's transparent rules so every routing decision is inspectable.
- The three routes, each a pipeline you already built: **none** answers directly (no retrieval), **single** runs the Part 6 retrieve → generate, **multi** runs the Part 10 decompose → retrieve → synthesize.
- `route()` as the **conductor** over Parts 6–10 — and why this is route-by-**complexity**, distinct from Part 8 (transform the query) and Part 10 (route by knowledge *source*).

## Files
- **`adaptive_rag.py`** — the single runnable script: the support KB, the classifier, an embedder + one-store retriever, the three route handlers, and the `route()` conductor over six worked queries (two per class), top to bottom.
- **`adaptive_rag.ipynb`** — step-by-step notebook: a markdown *why* before each small code step, built cell by cell.

## Run it
```bash
# from the repo root; numpy is the only dependency
pip install numpy
python3 part-15-adaptive-rag/adaptive_rag.py       # runs offline — no API key
# optional, for the REAL embedder path: pip install sentence-transformers
```
Prefer it step by step? Open `adaptive_rag.ipynb` in Jupyter, or click **Open in Colab** above.

## Key idea
Read the query *before* you reach for retrieval. A greeting needs none, a fact needs one lookup, a comparison needs several. A complexity classifier reads each query and dispatches it to the right one of the three pipelines we already know how to build — keeping quality on the hard queries while sparing the easy ones the cost. The classifier is itself a failure surface (a misroute means under-retrieval), so grade it like any other component (Part 11). The latency/cost wins often quoted for Adaptive RAG (roughly ~35% latency, ~28% cost) come from a 2026 vendor production report, **not** the original paper — treat them as indicative direction, not measured fact.

## Offline by design
The whole demo runs with no network and no API key: a deterministic rule/keyword classifier stands in for a trained router, a deterministic hashing embedder stands in for sentence-transformers, and an extractive generator quotes the best retrieved chunk. The real model paths sit behind `try/except`, so they light up automatically when a model or key is present. numpy is the only required dependency; if `sentence-transformers` is installed and a model is cached, the real embedder runs locally with no network, and only the cosine scores change — the route labels and the none/single/multi tally are identical either way.

---
← [Part 14 — Context-Aware Chunking](../part-14-context-aware-chunking/) · [Series index](../) · *Frontier Track complete — the capstone is back in [Part 12](../part-12-rag-in-production/).*
