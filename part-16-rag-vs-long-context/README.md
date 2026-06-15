# Part 16 — RAG vs Long-Context vs CAG

> Part 1 asked why RAG exists. Part 16 asks the harder follow-up: when do you even need retrieval? Context windows reach about a million tokens in 2026, so a small, stable corpus can be stuffed (long-context) or cached once and reused (CAG). The prompt-caching economics decide between them.

[📖 Read the essay](https://www.mefby.com/essays/rag-vs-long-context) · [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mftnakrsu/rag-by-hand/blob/main/part-16-rag-vs-long-context/rag_vs_long_context.ipynb)

## What it covers
- The question Part 1 assumed away: in 2026 a **~1M-token** window can hold a small corpus, so you can skip the index. The catch is the single fact this part turns on: **every prompt token is billed on every request**, so "just stuff it" trades a fits-in-the-window win for a paid-on-every-query cost.
- **Three strategies**, priced side by side: **RAG** (send only the top-*k* slice; cost tracks *k*, not corpus size), **long-context** (send the whole corpus fresh every query; linear in corpus size forever), and **CAG** (preload the corpus once, cache the KV state, reuse it: long-context made affordable by caching).
- **Prompt-caching economics** as the bridge: a cached input token costs about **0.1x** the fresh rate, against a one-time **~1.25x** write premium. That ten-fold discount on reuse is the entire economic engine behind CAG.
- The crossover behaviour the decision matrix is built on: at 1 query CAG is the **worst** option (you paid the write, got one read); by 100 queries the write has amortized and CAG drops below both. Sweep the corpus size and the winner flips from **CAG** (small) to **RAG** (massive), because long-context pays for the whole corpus fresh every query while RAG only ever sends *k* chunks.

## Files
- **`rag_vs_long_context.py`**: the single runnable cost model: the `Pricing` and `Workload` dataclasses, `cost_rag` / `cost_long_context` / `cost_cag`, the 1-query and 100-query tables, the crossover finder, a corpus-size sweep, two "try it yourself" experiments (kill the cache discount; break the prefix with volatile tokens), and the decision matrix, top to bottom.
- **`rag_vs_long_context.ipynb`**: step-by-step notebook: a markdown *why* before each small code step, built cell by cell.

## Run it
```bash
# from the repo root; pure standard library (no numpy, no API key, no network)
python3 part-16-rag-vs-long-context/rag_vs_long_context.py
```
Prefer it step by step? Open `rag_vs_long_context.ipynb` in Jupyter, or click **Open in Colab** above.

## Key idea
The three strategies differ only in *what you pay per query*, so the decision is, at heart, an economics problem. RAG sends a small slice each time; long-context sends everything fresh each time; CAG sends everything once and reuses the cached state. With a cached read at 0.1x of fresh, a small, stable corpus answered many times is cheapest under CAG (the write amortizes), while a massive corpus points back at RAG (whose cost never grows with the corpus). The crossover *is* the decision matrix, derived from the pricing rather than asserted: **massive, fast-moving, or private to RAG; small and stable to CAG (when you reuse it enough) or plain long-context (when it is tiny or rarely queried); mid-size to long-context.**

## A caution on the numbers
The model is deliberately a toy in **relative cost units, not a vendor quote**. It uses the public *shape* of 2026 prompt caching (fresh input = 1 unit, cache write = 1.25x, cache read = 0.1x), ignores output-token differences beyond a flat charge, and assumes a perfectly stable prefix. Plug in your provider's real per-token rates and the exact crossover point will move. What does not move is the shape: long-context grows linearly with corpus size on every query, CAG grows the same way but ~10x slower, and RAG stays flat in corpus size. Trust the shape, verify the digits against your own bill. CAG itself is from Chan et al., [arXiv:2412.15605](https://arxiv.org/abs/2412.15605).

## Offline by design
There is no model to load here: the whole part is a pricing argument, so it runs with **no network, no API key, and not even numpy**: pure standard library. The two experiments make the essay's pitfalls numeric rather than asserted: Experiment 1 sets the cache read to the fresh rate and watches CAG collapse into long-context (the 0.1x discount is the load-bearing wall); Experiment 2 adds a `volatile_tokens` block billed fresh on every call and watches CAG's edge evaporate (the "volatile tool results break the cache" failure). Change the `Pricing` or `Workload` numbers, re-run, and the crossover moves under your hands.

---
← [Part 15: Adaptive RAG](../part-15-adaptive-rag/) · [Series index](../) · *Frontier Track. The capstone checklist still lives in [Part 12](../part-12-rag-in-production/).*
