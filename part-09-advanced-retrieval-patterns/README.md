# Part 9 — Advanced Retrieval Patterns

> Search on something small and sharp; hand the model something large and rich. They do not have to be the same text.

[📖 Read the essay](https://www.mefby.com/essays/advanced-retrieval-patterns) · [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mftnakrsu/rag-by-hand/blob/main/part-09-advanced-retrieval-patterns/rag_parent_document.ipynb)

## What it covers
- **The core move: decoupling.** The best unit to *search* on (small, sharp) is rarely the best unit to *generate* from (large, rich) — so stop forcing one chunk to do both.
- **Parent-document retrieval (small to big).** Embed and search tiny child chunks; on a hit, serve the larger parent they came from. Index children, serve parents.
- **Sentence-window**, its close cousin: return the matched sentence plus a window of neighbours instead of a fixed parent.
- **Self-querying and contextual compression**, the other two patterns: turn natural language into a search-plus-filter, and trim noise out of chunks before generation.
- These patterns **stack and are not free** — add one when failure analysis points at the problem it solves, never by default.

## Files
- **`rag_parent_document.py`** — the single runnable script: builds parents, splits them into children, scores children against a query, and serves the parent on a hit, top to bottom.
- **`rag_parent_document.ipynb`** — step-by-step notebook: a markdown *why* before each small code step, built cell by cell.

## Run it
```bash
# from the repo root
pip install -r requirements.txt
python part-09-advanced-retrieval-patterns/rag_parent_document.py        # runs offline — no API key
```
Prefer it step by step? Open `rag_parent_document.ipynb` in Jupyter, or click **Open in Colab** above.

## Offline by design
The script demonstrates parent-document / sentence-window retrieval: it scores child chunks with the real sentence-transformers embedding model when it is installed, and falls back to a transparent keyword-overlap scorer otherwise. The parent/child bookkeeping — the actual lesson — is identical either way, and either path runs with no network and prints the same numbers every time.

---
← [Part 8 — Making Retrieval Smarter](../part-08-making-retrieval-smarter/) · [Series index](../) · [Part 10 — Advanced RAG Architectures](../part-10-advanced-architectures/) →
