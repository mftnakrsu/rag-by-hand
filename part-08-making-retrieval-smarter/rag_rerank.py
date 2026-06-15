"""
Smarter retrieval: metadata filtering + cross-encoder reranking.
RAG from First Principles, Part 8. This EXTENDS the Part 6 app (rag_app.py):
the embed / store / first-pass retrieve steps are the same idea, with two
additions around retrieval:

  - a DURING lever: filter candidates by metadata before scoring.
  - an AFTER  lever: rerank a wide candidate set with a cross-encoder,
                     then keep the best few (the two-stage pattern).

Run:
  pip install sentence-transformers numpy
  python rag_rerank.py

NOTE: reranking models, library APIs, and model names move fast and I have a
knowledge cutoff. Treat the cross-encoder model name below as a snapshot and
check the current docs; the printed scores are illustrative and will vary by
model and version. The ms-marco cross-encoder below in particular returns an
unbounded relevance logit (not a 0 to 1 score), so your real output will not
match the tidy numbers in the article. Only the resulting order is the point.
"""

import numpy as np
from sentence_transformers import SentenceTransformer, CrossEncoder

# ---------------------------------------------------------------------------
# Corpus, now with METADATA on every chunk (Parts 4 and 5). The `section` and
# `year` fields are what the DURING lever filters on.
# ---------------------------------------------------------------------------
CORPUS = [
    {"text": "Refunds are accepted within 30 days of purchase, provided the item is unused.", "section": "returns", "year": 2024},
    {"text": "Items marked final sale cannot be returned or exchanged.",                       "section": "returns", "year": 2024},
    {"text": "Standard shipping takes 3 to 5 business days; express is next-day.",             "section": "shipping", "year": 2024},
    {"text": "To start a return, email support@example.com with your order number.",          "section": "returns", "year": 2023},
    {"text": "Worn or washed clothing counts as used and is not eligible for a refund.",      "section": "returns", "year": 2024},
    {"text": "Our winter jacket collection is on sale through the end of the month.",         "section": "promotions", "year": 2024},
    {"text": "Gift cards never expire and are non-refundable.",                               "section": "giftcards", "year": 2022},
    {"text": "All electronics include a one-year limited warranty.",                          "section": "warranty", "year": 2024},
    {"text": "Shipping fees are non-refundable; late returns get store credit.",              "section": "returns", "year": 2024},
    {"text": "Exchanges for a different size are free within the return window.",             "section": "returns", "year": 2024},
]

# ---------------------------------------------------------------------------
# Embed + store (Part 6, unchanged). Each chunk keeps its metadata alongside.
# ---------------------------------------------------------------------------
embedder = SentenceTransformer("all-MiniLM-L6-v2")          # 384 dims, local


def embed(texts):
    return embedder.encode(texts, normalize_embeddings=True)  # unit vectors -> dot == cosine


chunks = [dict(c) for c in CORPUS]                          # the "store": text + metadata
vectors = embed([c["text"] for c in chunks])                # the parallel vector matrix


# ---------------------------------------------------------------------------
# DURING lever: metadata filtering. Keep only chunks whose metadata matches
# hard criteria. This is PRE-filtering: we shrink the candidate set BEFORE
# scoring, which is cheaper and enforces rules like access control or freshness.
# Returns the row indices that survive, so we can slice the vector matrix.
# ---------------------------------------------------------------------------
def matching_indices(where):
    # `where` is a dict of metadata field -> required value, e.g. {"section": "returns"}.
    return [i for i, c in enumerate(chunks)
            if all(c.get(field) == value for field, value in where.items())]


# ---------------------------------------------------------------------------
# Stage 1, the WIDE net: fast first-pass retrieval (bi-encoder cosine), exactly
# Part 6's retrieve, but we fetch a large `n` (e.g. 10) instead of the final k,
# optionally restricted to chunks that pass the metadata filter.
# ---------------------------------------------------------------------------
def first_pass(query, n=10, where=None):
    pool = matching_indices(where) if where else list(range(len(chunks)))
    q = embed([query])[0]
    scores = vectors[pool] @ q                              # cosine against the filtered pool
    order = np.argsort(-scores)[:n]                         # best n within the pool
    return [{"text": chunks[pool[j]]["text"],
             "section": chunks[pool[j]]["section"],
             "score": float(scores[j])} for j in order]


# ---------------------------------------------------------------------------
# Stage 2, the RERANK: a cross-encoder reads (query, chunk) together and scores
# true relevance. We reorder the wide candidate set by that score and keep top_k.
# ---------------------------------------------------------------------------
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")  # check current model names


def rerank(query, candidates, top_k=3):
    pairs = [(query, c["text"]) for c in candidates]        # one pair per candidate
    rel = reranker.predict(pairs)                           # a relevance score per pair
    ranked = sorted(zip(candidates, rel), key=lambda cr: cr[1], reverse=True)
    return [{**c, "rerank": float(s)} for c, s in ranked[:top_k]]


# ---------------------------------------------------------------------------
# Put the two stages together: filter -> retrieve wide -> rerank -> keep top_k.
# (Augment + generate from Part 6 are unchanged, so they are omitted here.)
# ---------------------------------------------------------------------------
def smart_retrieve(query, n=10, top_k=3, where=None):
    candidates = first_pass(query, n=n, where=where)
    return rerank(query, candidates, top_k=top_k)


if __name__ == "__main__":
    query = "Can I get a refund on a jacket I've already worn?"

    print(f"QUERY: {query}\n")

    wide = first_pass(query, n=10)
    print("STAGE 1, first-pass order (bi-encoder cosine, top 10):")
    for rank, c in enumerate(wide, 1):
        print(f"  {rank:>2}. {c['score']:.2f}  {c['text']}")

    final = rerank(query, wide, top_k=3)
    print("\nSTAGE 2, after cross-encoder rerank (top 3 kept):")
    for rank, c in enumerate(final, 1):
        print(f"  {rank:>2}. {c['rerank']:.2f}  {c['text']}")

    # The DURING lever on its own: restrict retrieval to the returns section only.
    filtered = first_pass(query, n=3, where={"section": "returns"})
    print("\nWith a metadata filter (section == 'returns'), the sale ad never even competes:")
    for rank, c in enumerate(filtered, 1):
        print(f"  {rank:>2}. {c['score']:.2f}  {c['text']}")
