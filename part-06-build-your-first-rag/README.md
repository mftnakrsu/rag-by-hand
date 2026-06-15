# Part 6 — Build Your First RAG

> Five parts of theory become one running program: a chat-with-your-documents app, built by hand so every line maps to a concept you already learned.

[📖 Read the essay](https://www.mefby.com/essays/build-your-first-rag) · [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mftnakrsu/rag-by-hand/blob/main/part-06-build-your-first-rag/rag_app.ipynb)

## What it covers
- The whole RAG loop in one file: **retrieve** relevant chunks, **augment** the prompt with them, **generate** a grounded answer.
- A transparent vector store: a list of chunks beside a NumPy matrix; cosine top-k retrieval is one `argsort` over a dot product.
- The same-model rule: embed the query with the *same* model as the chunks, or the scores are silently meaningless.
- A grounded prompt template ("answer only from the context; otherwise say you don't know") as the highest-leverage line against hallucination.
- How the LLM hides behind one swappable `generate(prompt)` function — and how a real vector database (Chroma) drops in without touching the rest.

## Files
- **`rag_app.py`** — the single runnable script: corpus → chunk → embed → store → retrieve → augment → generate → a tiny REPL, top to bottom.
- **`rag_app.ipynb`** — step-by-step notebook: a markdown *why* before each small code step, built cell by cell.

## Run it
```bash
# from the repo root
pip install -r requirements.txt
python part-06-build-your-first-rag/rag_app.py        # runs offline — no API key
```
Prefer it step by step? Open `rag_app.ipynb` in Jupyter, or click **Open in Colab** above.

## Offline by design
The script ships a real local embedder (`sentence-transformers`) and `generate()` variants for OpenAI, Ollama, and Anthropic. The notebook needs none of them: it guards both ends — a hashing-embedder fallback and an offline grounded-extractive `generate()` — so the full retrieve-augment-generate loop runs with no key and no network, using only NumPy.

---
← [Part 5 — Documents & Chunking](../part-05-chunking/) · [Series index](../) · [Part 7 — Retrieval Deep Dive](../part-07-retrieval-deep-dive/) →
