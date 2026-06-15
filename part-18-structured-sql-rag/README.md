# Part 18 — Structured and SQL RAG

> Most enterprise knowledge lives in databases and tables, not documents, and dense retrieval cannot answer a question whose answer has to be computed. Retrieve the schema, generate SQL, execute, answer. The close of the series.

[📖 Read the essay](https://www.mefby.com/essays/structured-sql-rag) · [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mftnakrsu/rag-by-hand/blob/main/part-18-structured-sql-rag/sql_rag.ipynb)

## What it covers
- Why dense passage retrieval **misses** here: ask "what was our total revenue from shipped orders?" and no passage holds the answer. The number does not exist as text until a query computes it (join, filter, multiply, sum). Retrieval finds *similar text*; it cannot do arithmetic over rows.
- **Text-to-SQL with RAG** swaps two words of the series spine. The document loop was *embed, retrieve, ground, generate*; the structured loop is **retrieve schema, generate SQL, execute, answer**. The database does the arithmetic exactly, so the model never guesses it.
- `retrieve_schema(question, k)` — retrieve the relevant *schema* subset (the tables and columns this question needs), never the whole catalog. This is **schema linking**, the make-or-break step for text-to-SQL at scale: real schemas are far too large for the prompt budget, and tabular reasoning degrades as the table set grows, well inside the context window.
- What a card carries beyond table names: column descriptions, sample rows, and a **domain glossary** that maps business words ("revenue") to the columns and computation that encode them (`price * quantity`, summed) so the model does not invent columns that do not exist.
- The **execution** step that has no analog in the document pipeline: the generated SQL runs against a real SQLite database and returns real rows. That is what makes the answer exact instead of guessed.
- `route()` as the Part 15 callback: aggregational / numeric questions ("how many", "total", "revenue") go to the SQL path; everything else falls back to the document path. Part 15 routed by *complexity*; this adds a second axis, *structure*, in the same classifier.

## Files
- **`sql_rag.py`** — the single runnable script: a tiny in-memory SQLite DB (three tables an e-commerce support bot might sit in front of), the schema cards + glossary, the schema retriever, the rule-based SQL generator, the execute-and-answer loop, and the structured router over four worked questions, top to bottom.
- **`sql_rag.ipynb`** — step-by-step notebook: a markdown *why* before each small code step, built cell by cell.

## Run it
```bash
# from the repo root; the demo is pure standard library (stdlib sqlite3)
python3 part-18-structured-sql-rag/sql_rag.py       # runs offline — no API key
# optional, for the REAL schema-embedder path: pip install sentence-transformers
```
Prefer it step by step? Open `sql_rag.ipynb` in Jupyter, or click **Open in Colab** above.

## Key idea
When the answer must be *computed* over records, it is not a retrieval problem, it is a *query* problem. So change what "retrieval" produces: instead of passages to read, retrieve enough of the database **schema** for a model to write correct SQL, generate that query, **execute** it against the database, and answer from the result rows. The arithmetic is done by the database, never hallucinated by the model. You retrieve a schema *subset* (schema linking) rather than pasting the catalog for two reasons: enterprise schemas dwarf any prompt budget, and tabular reasoning degrades as the table set grows even when it fits. Most wrong answers on structured data are schema-linking failures wearing the costume of a model that "can't write SQL." It usually can; it just got handed the wrong tables.

## Offline by design
The whole demo runs with no network and no API key. The **execution step is genuinely real** (the generated SQL runs against a real `sqlite3` database and returns real rows); two pieces are mocked so the file stays dependency-light. Schema retrieval is keyword overlap rather than embeddings, and SQL generation is a transparent rule-based stub rather than a model call. The *control flow* is the production one exactly: retrieve schema → generate SQL → execute → answer. The real schema-embedder path sits behind a `try/except`, so it lights up automatically when `sentence-transformers` is installed and a model is cached (still no network); only the cosine scores change, and the answers (`1` enterprise customer, `436.0` shipped revenue, `1` refunded order) are identical either way. Swap the keyword scorer for an embedder and the stub for a model and you have a working text-to-SQL RAG system with the same spine.

---
← [Part 15 — Adaptive RAG](../part-15-adaptive-rag/) · [Series index](../) · *Series complete — eighteen parts from "why RAG exists" to structured knowledge in databases. The production capstone is [Part 12](../part-12-rag-in-production/).*
