# Part 13 — Late-Interaction Retrieval

> Keep a vector per token, not one per passage, and score with MaxSim: cross-encoder-quality term matching at bi-encoder serving cost — extended to document page images by ColPali. The first part of the **Frontier Track** (the core series ends at Part 12).

[📖 Read the essay](https://www.mefby.com/essays/late-interaction-retrieval) · [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mftnakrsu/rag-by-hand/blob/main/part-13-late-interaction/late_interaction.ipynb)

## What it covers
- The **pooling problem**: a single dense vector (Part 7) averages a whole passage into one point, so a query term that matches one word in a long passage gets washed out.
- **Token-level multi-vectors**: keep a vector per token, for both the query and the document.
- **MaxSim** in ~3 lines of numpy: for each query token, take the max similarity over all doc tokens, then sum. A worked head-to-head where MaxSim and pooled cosine **disagree** at rank 1 on the support knowledge base (the buried `E-4042` code).
- **Why it's still cheap to serve**: doc multi-vectors are precomputed offline; only MaxSim runs at query time — unlike Part 8's cross-encoder, which runs the model on every query-doc pair at query time.
- **The storage tradeoff**: per-token vectors cost more; ColBERTv2 residual compression (1–2 bits/dim) cuts MS MARCO's ~154 GiB index to ~16–25 GiB.
- **ColPali / ColQwen as a mechanism**: a VLM embeds page *images* as patch vectors, scored by the same MaxSim — no OCR, no chunking. Shown with a toy patch-vector stand-in.

## Files
- **`late_interaction.py`** — the single runnable script: a per-token embedder, `maxsim`, the MaxSim-vs-pooled ranking, the storage arithmetic, and the toy ColPali patch demo, top to bottom.
- **`late_interaction.ipynb`** — step-by-step notebook: a markdown *why* before each small code step, built cell by cell.

## Run it
```bash
# from the repo root
pip install -r requirements.txt
python3 part-13-late-interaction/late_interaction.py        # runs offline — no API key
```
Prefer it step by step? Open `late_interaction.ipynb` in Jupyter, or click **Open in Colab** above.

## Key idea
`maxsim(query_vecs, doc_vecs)` = sum over query tokens of the **max** cosine each finds among the doc tokens. That `max` rewards a single strong, exact hit (the rare `E-4042` token), so the right document surfaces even when the term is buried in a long passage — exactly what a single pooled vector averages away. The same MaxSim, run over a page image's patch vectors instead of text tokens, is how ColPali retrieves documents without OCR or chunking.

## Offline by design
Fully guarded so the whole demo runs with no network and no API key: the headline path pulls per-token embeddings from `sentence-transformers` (all-MiniLM-L6-v2); if no model or network is present, a deterministic hashing stand-in keeps every line runnable and prints a clear label. The teaching point (MaxSim surfacing the exact-term doc that pooled cosine buries) holds on **both** paths. **ColPali is taught strictly as a mechanism with a toy stand-in — no real vision-language model runs offline here.**

---
← [Part 12 — RAG in Production](../part-12-rag-in-production/) · [Series index](../) · [Part 14 — Context-Aware Chunking](../part-14-context-aware-chunking/) →
