# Part 11 — Evaluating RAG

> Replace vibes with numbers: measure a RAG system, then locate which half — retrieval or generation — actually broke.

[📖 Read the essay](https://www.mefby.com/essays/evaluating-rag) · [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mftnakrsu/rag-by-hand/blob/main/part-11-evaluating-rag/rag_eval.ipynb)

## What it covers
- A wrong answer comes from exactly one of two places — **retrieval** or **generation** — so we measure the two halves separately instead of with one blended score.
- **Context recall** (retrieval): did we fetch the chunk that actually holds the answer?
- **Faithfulness** (generation): is every claim in the answer supported by the retrieved context? — the LLM-as-a-judge shape, with a transparent offline stand-in.
- A **diagnostic rule** that reads the two metrics in pipeline order and points at the failing stage.
- Why an eval set must include out-of-scope questions that *should* be refused — the case vibes always miss.

## Files
- **`rag_eval.py`** — the single runnable script: corpus, lexical retriever, both metrics, the eval set, and the score table, top to bottom.
- **`rag_eval.ipynb`** — step-by-step notebook: a markdown *why* before each small code step, built cell by cell.

## Run it
```bash
# from the repo root
pip install -r requirements.txt
python part-11-evaluating-rag/rag_eval.py        # runs offline — no API key
```
Prefer it step by step? Open `rag_eval.ipynb` in Jupyter, or click **Open in Colab** above.

## Offline by design
Pure standard library — no network, no model download, no API key. Context recall uses an id-based golden-chunk stand-in, and faithfulness uses a transparent deterministic fallback for the LLM judge (it shows the real call shape if `OPENAI_API_KEY` is set, but always falls through to the same output). The payoff is the two-refusals row: two identical "I don't know" answers that one metric pulls apart into *fix the retriever* versus *leave it alone*.

---
← [Part 10 — Advanced RAG Architectures](../part-10-advanced-architectures/) · [Series index](../) · [Part 12 — RAG in Production](../part-12-rag-in-production/) →
