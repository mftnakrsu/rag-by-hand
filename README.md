# rag-by-hand

Build a Retrieval-Augmented Generation system from first principles — one runnable
Python file per concept, no frameworks hiding the moving parts. Companion code for
the 12-part **RAG from First Principles** series on
[mefby.com](https://www.mefby.com/essays).

> "Build it by hand, understand every line."

Each folder maps 1:1 to an essay. The early parts (2–5) are pure NumPy / standard
library and run offline with no API key. Part 6 assembles them into a working
"chat with your documents" app; Parts 7–12 layer on hybrid retrieval, reranking,
advanced patterns, evaluation, and production hardening.

Every part ships **two ways to learn the same concept**: a single runnable `.py`
(the whole idea, top to bottom) and a step-by-step **Jupyter notebook** (`.ipynb`)
that rebuilds it cell by cell, with the *why* spelled out before each small step.
Both run offline — a real embedding model / LLM is used automatically when one is
available, and a transparent, deterministic fallback keeps every cell runnable
otherwise. Part 1 is a concept-only notebook (the code starts in Part 2).

## The series

| Part | Topic | Code | Notebook | Essay |
|---|---|---|---|---|
| 1 | Why RAG Exists | [part-01-why-rag](part-01-why-rag/) (concept) | [why-rag.ipynb](part-01-why-rag/why-rag.ipynb) | [read](https://www.mefby.com/essays/why-rag-exists) |
| 2 | Embeddings | [embeddings.py](part-02-embeddings/embeddings.py) | [embeddings.ipynb](part-02-embeddings/embeddings.ipynb) | [read](https://www.mefby.com/essays/embeddings) |
| 3 | Measuring Similarity | [similarity.py](part-03-measuring-similarity/similarity.py) | [similarity.ipynb](part-03-measuring-similarity/similarity.ipynb) | [read](https://www.mefby.com/essays/measuring-similarity) |
| 4 | Vector Databases & Indexing | [vector_db.py](part-04-vector-databases/vector_db.py) | [vector_db.ipynb](part-04-vector-databases/vector_db.ipynb) | [read](https://www.mefby.com/essays/vector-databases) |
| 5 | Documents & Chunking | [chunking.py](part-05-chunking/chunking.py) | [chunking.ipynb](part-05-chunking/chunking.ipynb) | [read](https://www.mefby.com/essays/documents-and-chunking) |
| 6 | Build Your First RAG | [rag_app.py](part-06-build-your-first-rag/rag_app.py) | [rag_app.ipynb](part-06-build-your-first-rag/rag_app.ipynb) | [read](https://www.mefby.com/essays/build-your-first-rag) |
| 7 | Retrieval Deep Dive | [rag_hybrid.py](part-07-retrieval-deep-dive/rag_hybrid.py) | [rag_hybrid.ipynb](part-07-retrieval-deep-dive/rag_hybrid.ipynb) | [read](https://www.mefby.com/essays/retrieval-deep-dive) |
| 8 | Making Retrieval Smarter | [rag_rerank.py](part-08-making-retrieval-smarter/rag_rerank.py) | [rag_rerank.ipynb](part-08-making-retrieval-smarter/rag_rerank.ipynb) | [read](https://www.mefby.com/essays/making-retrieval-smarter) |
| 9 | Advanced Retrieval Patterns | [rag_parent_document.py](part-09-advanced-retrieval-patterns/rag_parent_document.py) | [rag_parent_document.ipynb](part-09-advanced-retrieval-patterns/rag_parent_document.ipynb) | [read](https://www.mefby.com/essays/advanced-retrieval-patterns) |
| 10 | Advanced RAG Architectures | [corrective_rag.py](part-10-advanced-architectures/corrective_rag.py) | [corrective_rag.ipynb](part-10-advanced-architectures/corrective_rag.ipynb) | [read](https://www.mefby.com/essays/advanced-rag-architectures) |
| 11 | Evaluating RAG | [rag_eval.py](part-11-evaluating-rag/rag_eval.py) | [rag_eval.ipynb](part-11-evaluating-rag/rag_eval.ipynb) | [read](https://www.mefby.com/essays/evaluating-rag) |
| 12 | RAG in Production | [rag_production.py](part-12-rag-in-production/rag_production.py) | [rag_production.ipynb](part-12-rag-in-production/rag_production.ipynb) | [read](https://www.mefby.com/essays/rag-in-production) |

## Quick start

```bash
git clone https://github.com/mftnakrsu/rag-by-hand
cd rag-by-hand
pip install -r requirements.txt

# Parts 2–5 run offline, no API key:
python part-03-measuring-similarity/similarity.py

# Part 6 — the full app. Set one provider (below), then:
python part-06-build-your-first-rag/rag_app.py
```

Prefer to learn step by step? Open the matching notebook instead — every cell runs
offline, no API key required:

```bash
pip install jupyter           # one-time, to run the notebooks
jupyter notebook part-03-measuring-similarity/similarity.ipynb
```

## LLM providers

The generation step (Part 6 onward) is isolated behind a single `generate(prompt)`
function so the provider is a one-line swap. Three backends are shown:
**OpenAI** (the default), **Ollama** (local, free, no key), and
**Anthropic / Claude**. Set the matching API key — or run Ollama locally — and
swap the function body. The retrieval, chunking, and similarity code is provider-
agnostic and needs no key at all.

## License

MIT — see [LICENSE](LICENSE).
