# Part 12 — RAG in Production

> The finale: the gap between RAG that works in a notebook and RAG that survives real users, made runnable with two small touches on the app — refuse when retrieval is weak, reuse when the question is one you have already answered.

[📖 Read the essay](https://www.mefby.com/essays/rag-in-production) · [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mftnakrsu/rag-by-hand/blob/main/part-12-rag-in-production/rag_production.ipynb)

## What it covers
- The demo-to-production gap: a working pipeline is ~20% of the job; latency, cost, reliability, security, and observability are the other 80%.
- The language model dominates both latency and cost, so the big levers (stream, parallelize, right-size, cache) all aim at how — or whether — you call it.
- A graceful no-context guard: when the top retrieval score falls below a relevance floor, say "I don't know" instead of hallucinating.
- A semantic cache: serve repeat *and paraphrased* questions by meaning, skipping retrieve + generate entirely.
- The capstone: the whole series mapped onto one production request, plus security (prompt injection, access control) and index freshness.

## Files
- **`rag_production.py`** — the single runnable script: a lexical embedder, the refund-policy corpus, retrieve/generate, then the no-context guard and semantic cache, ending in a three-query demo.
- **`rag_production.ipynb`** — step-by-step notebook: a markdown *why* before each small code step, built cell by cell.

## Run it
```bash
# from the repo root
pip install -r requirements.txt
python part-12-rag-in-production/rag_production.py        # runs offline — no API key
```
Prefer it step by step? Open `rag_production.ipynb` in Jupyter, or click **Open in Colab** above.

## Offline by design
To stay runnable with zero dependencies, the script uses a transparent lexical bag-of-words cosine as a stand-in for Part 2's real embedder, so you can watch a `SemanticCache` hit and a relevance-floor refusal with no network and no API key. Swap a real embedder back in and every function behaves the same, just with sharper matches.

---
← [Part 11 — Evaluating RAG](../part-11-evaluating-rag/) · [Series index](../)
