# Part 14 — Context-Aware Chunking

> A chunk that reads fine on its own can be useless once it leaves its document: "She" loses its antecedent "Alice", "the policy" loses its subject. Two training-free fixes put the context back.

[📖 Read the essay](https://www.mefby.com/essays/context-aware-chunking) · [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mftnakrsu/rag-by-hand/blob/main/part-14-context-aware-chunking/context_aware_chunking.ipynb)

## What it covers
- The **context-loss problem** (back to Part 5): split a refund-policy note into sentences and one chunk becomes "She set the refund window at 30 days." Embedded alone, that vector never sees "Alice" — so a query naming Alice can't find the chunk that answers it.
- **Late chunking** (introduced by the Jina AI team): embed *all* the document's tokens in one pass through a long-context encoder, **then** mean-pool each chunk's token span. Because pooling happens *after* the transformer, every chunk vector is contextualized by the whole document. Different from Part 5's static title-prepend — nothing is glued onto the text.
- **Contextual Retrieval** (Anthropic): an LLM writes a one-sentence situating note per chunk and you **prepend it before embedding**. Model-agnostic and drop-in; it costs one extra LLM call per chunk at index time (prompt caching mitigates that).
- The same coreference query run through naive / late-chunked / contextual embeddings, watching the buried answer chunk surface — plus a small comparison of the two fixes.

## Files
- **`context_aware_chunking.py`** — the single runnable script: a contextualizing token embedder, `late_chunk()` span pooling, `contextualize()` prepend-then-embed, and a three-way ranking demo on the refund / `E-4042` corpus.
- **`context_aware_chunking.ipynb`** — step-by-step notebook: a markdown *why* before each small code step, built cell by cell.

## Run it
```bash
# from the repo root
pip install -r requirements.txt
python part-14-context-aware-chunking/context_aware_chunking.py   # runs offline — no API key
```
Prefer it step by step? Open `context_aware_chunking.ipynb` in Jupyter, or click **Open in Colab** above.

## Key idea
`late_chunk(token_vecs, spans)` pools each chunk's token span **after** a whole-document encoder pass, so the chunk vector inherits its neighbours' context. `contextualize(chunk, doc)` prepends an LLM-written situating sentence before embedding. Both lift the orphaned "She set the refund window" chunk from buried to top-ranked on a query that names Alice — same chunk text, context restored.

## Evidence (the one number to quote)
Contextual embeddings **alone** cut the top-20 retrieval-failure rate by **35%** (5.7% → 3.7%, [Anthropic](https://www.anthropic.com/news/contextual-retrieval)). Larger cumulative figures bundle in BM25 and reranking and overstate the chunking-only effect, so we don't quote them.

## Offline by design
The intended path is a real long-context encoder (per-token vectors from `all-MiniLM-L6-v2`) plus an LLM situating-note writer, both loaded inside a `try/except`. If a model or network isn't there, the script falls back to a deterministic stand-in: a hashing token embedder that **contextualizes** (each token's vector mixes in a distance-decayed blend of its neighbours, a toy attention window) and a grounded extractive `generate()`. The fallback can't capture real synonymy — it's a hash — but it reproduces the *mechanism* honestly, so every line runs with no download and the buried chunk still surfaces.

---
← [Part 13 — Late-Interaction Retrieval](../part-13-late-interaction/) · [Series index](../) · [Part 15 — Adaptive RAG](../part-15-adaptive-rag/) →
