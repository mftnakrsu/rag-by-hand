"""
rag_parent_document.py  -  RAG from First Principles, Part 9 ("Advanced Retrieval Patterns")

Parent-document retrieval (a.k.a. "small-to-big") on the running app.
The whole idea in one sentence: INDEX CHILDREN, SERVE PARENTS. We embed and
search tiny child chunks so matching is sharp, but when a child hits we hand the
LLM its larger PARENT chunk so generation has room to breathe. The unit we
search is decoupled from the unit we return.

This is an *extension* of the Part 6 app, not a rebuild. In your real app the
embeddings come from Part 6's sentence-transformers model and the child vectors
live in your vector store. To keep this file standalone and its output
deterministic, it falls back to a transparent keyword-overlap scorer when
sentence-transformers is not installed; the parent/child bookkeeping, which is
the actual lesson, is identical either way.

    python rag_parent_document.py
"""

import re
from collections import Counter


# ---------------------------------------------------------------------------
# Step 0. One real document, not a list of pre-split chunks. The unit of
#         retrieval is the question of THIS chapter, so we start from prose and
#         decide how to split it ourselves. Each top-level string is a PARENT
#         (a coherent section); generation wants this much context.
# ---------------------------------------------------------------------------
PARENTS = [
    # parent 0
    "Refunds. We accept refunds within 30 days of purchase, as long as the item "
    "is unused and in its original packaging. To start a return, email "
    "support@example.com with your order number. Once we receive the item, your "
    "refund is processed back to the original payment method within five business "
    "days. Shipping fees are not refundable.",
    # parent 1
    "Exchanges. If you want a different size or color, request an exchange instead "
    "of a refund. Exchanges ship free of charge. Items marked final sale cannot be "
    "returned or exchanged. Gift cards are non-refundable and cannot be exchanged.",
    # parent 2
    "Warranty. All electronics include a one-year limited warranty that covers "
    "manufacturing defects. The warranty does not cover accidental damage or normal "
    "wear. To make a warranty claim, contact support with a photo of the defect and "
    "your order number.",
]


# ---------------------------------------------------------------------------
# Step 1. Split each parent into CHILD chunks. Here a child is one sentence:
#         small, focused, and therefore sharp to embed and match (Part 5's
#         "small chunks retrieve precisely"). We keep a child -> parent map so a
#         hit on any child can be traded up for the whole parent it came from.
# ---------------------------------------------------------------------------
def split_sentences(text):
    # naive sentence split; good enough for the demo. Use a real splitter for prose.
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]


children = []          # the small units we actually embed and search
child_to_parent = []   # child_to_parent[i] = index of the parent child i belongs to
for p_idx, parent_text in enumerate(PARENTS):
    for sentence in split_sentences(parent_text):
        children.append(sentence)
        child_to_parent.append(p_idx)


# ---------------------------------------------------------------------------
# Step 2. Score children against the query. In the real app this is Part 6's
#         cosine-over-embeddings search; the fallback is a transparent
#         keyword-overlap score so this file runs anywhere and prints the same
#         numbers every time. Either way we score CHILDREN, never parents.
# ---------------------------------------------------------------------------
def tokenize(text):
    return re.findall(r"[a-z0-9]+", text.lower())


try:
    from sentence_transformers import SentenceTransformer
    import numpy as np

    _model = SentenceTransformer("all-MiniLM-L6-v2")
    _child_vecs = _model.encode(children, normalize_embeddings=True)

    def child_scores(query):
        q = _model.encode([query], normalize_embeddings=True)[0]
        return list(_child_vecs @ q)            # cosine sim, all unit length (Part 3)

except Exception:                                # transparent standalone fallback
    def child_scores(query):
        q = Counter(tokenize(query))
        out = []
        for child in children:
            terms = Counter(tokenize(child))
            overlap = sum(min(q[t], terms[t]) for t in q)
            out.append(overlap / (len(terms) + 1))   # length-normalized overlap
        return out


# ---------------------------------------------------------------------------
# Step 3. Parent-document retrieve. Find the best CHILD, then return its PARENT.
#         If several of the top children point at the same parent, we still
#         return that parent once (a tiny taste of "auto-merging").
# ---------------------------------------------------------------------------
def retrieve_small_return_big(query, k_children=3):
    scores = child_scores(query)
    top = sorted(range(len(children)), key=lambda i: scores[i], reverse=True)[:k_children]

    best_child = top[0]
    # de-duplicate parents while preserving the order the children ranked in
    parent_ids = list(dict.fromkeys(child_to_parent[i] for i in top))

    return {
        "matched_child": children[best_child],          # what we SEARCHED and hit
        "child_score": float(scores[best_child]),
        "returned_parents": [PARENTS[p] for p in parent_ids],  # what the LLM GETS
    }


# ---------------------------------------------------------------------------
# Step 4. See the decoupling. Compare what a naive "return the chunk you matched"
#         search hands the model versus what parent-document hands it.
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    query = "can I get my money back if the box is already open?"
    # k_children=1 keeps the before/after crisp: one sharp child, one rich parent.
    # Raise it and retrieve_small_return_big de-duplicates parents for you, which
    # is the seed of "auto-merging" retrieval discussed in the prose.
    result = retrieve_small_return_big(query, k_children=1)

    print(f"Query: {query!r}\n")

    print("NAIVE  (search small, return small)  ->  the LLM receives:")
    print(f"  [{result['child_score']:.3f}] {result['matched_child']}\n")

    print("PARENT-DOCUMENT  (search small, return big)  ->  the LLM receives:")
    for parent in result["returned_parents"]:
        print(f"  {parent}\n")

    print("Same sharp match. Far more context to answer from.")


# ---------------------------------------------------------------------------
# Expected output (keyword-overlap fallback; deterministic):
#
# Query: 'can I get my money back if the box is already open?'
#
# NAIVE  (search small, return small)  ->  the LLM receives:
#   [0.158] Once we receive the item, your refund is processed back to the
#           original payment method within five business days.
#
# PARENT-DOCUMENT  (search small, return big)  ->  the LLM receives:
#   Refunds. We accept refunds within 30 days of purchase, as long as the item is
#   unused and in its original packaging. To start a return, email
#   support@example.com with your order number. Once we receive the item, your
#   refund is processed back to the original payment method within five business
#   days. Shipping fees are not refundable.
#
# Same sharp match. Far more context to answer from.
#
# The naive result matched a single true sentence but starves the model: it never
# learns about the 30-day window or the "unused / original packaging" condition
# that actually decides this question. The parent hands over the whole refund
# section. We searched the child; we served the parent. That is the entire
# pattern. Swap in the real sentence-transformers model and a *different* child
# may win the match (semantics over keyword overlap), but it still lives inside
# parent 0, so the parent that comes back is identical. Which child you search is
# an implementation detail; which parent you serve is the answer.
# ---------------------------------------------------------------------------
