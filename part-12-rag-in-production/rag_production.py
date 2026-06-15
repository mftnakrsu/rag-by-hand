"""
rag_production.py  -  RAG from First Principles, Part 12 ("RAG in Production", the finale)

Two small production touches on the running app, no rebuild required.
This is an *extension* of the Part 6 app (rag_app.py): the embed / store /
retrieve / generate steps are the same shape. We add the two things that
separate a demo from something you would actually ship:

  (A) a graceful "no relevant context" guard. If retrieval comes back empty
      or weak, we say "I don't know" instead of letting the model invent an
      answer. A confident wrong answer is worse than an honest refusal.
  (B) a small SEMANTIC CACHE. Repeat questions (and their paraphrases) are
      served from a cache keyed by MEANING, not by exact string match, so a
      reworded question can still hit.

Run:
  python3 rag_production.py

NOTE: to stay runnable with ZERO dependencies (no numpy, no
sentence-transformers, no network), this file uses a TRANSPARENT lexical
(bag-of-words cosine) stand-in wherever Part 2's real embedding model would
go. A real embedder would catch deeper paraphrases that share no words; the
lexical stand-in only catches paraphrases that reuse vocabulary. The point
here is the MECHANISM (the relevance floor, the semantic cache), not the
quality of the similarity score. Swap the real embedder back in and every
function below behaves the same, just with sharper matches. No em dashes
anywhere in this series.
"""

import re
from collections import Counter
from math import sqrt


# ---------------------------------------------------------------------------
# A deterministic lexical embedding. We tokenize lowercased words into a
# bag-of-words vector (a plain dict term -> count), and compare two such
# vectors with cosine similarity. This is the zero-dependency stand-in for a
# real embedding model (Part 2): same interface (text in, vector out; cosine
# to compare), just a transparent lexical signal instead of learned semantics.
#
# We drop a few high-frequency stopwords first. A real embedder learns to
# down-weight these; our bag-of-words has no such notion, so without this an
# off-topic question ("what is the boiling point of water?") would score high
# just by sharing "is / the / of" with a chunk. Removing them keeps the lexical
# signal on content words, which is the honest thing a keyword scorer can do.
# ---------------------------------------------------------------------------
STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "were", "be", "to", "of", "in",
    "on", "for", "and", "or", "i", "you", "it", "that", "this", "with",
    "do", "does", "can", "could", "would", "will", "my", "your", "what",
    "get", "have", "has",
}


def tokenize(text):
    return [t for t in re.findall(r"[a-z0-9]+", text.lower()) if t not in STOPWORDS]


def embed(text):
    # bag-of-words vector: {term: count}. Deterministic, order-independent.
    return Counter(tokenize(text))


def cosine(a, b):
    # cosine over two dict vectors. Only shared terms contribute to the dot.
    dot = sum(a[t] * b[t] for t in a if t in b)
    na = sqrt(sum(v * v for v in a.values()))
    nb = sqrt(sum(v * v for v in b.values()))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


# ---------------------------------------------------------------------------
# The corpus: the same refund-policy chunks the series has used throughout.
# We embed each chunk once up front; the (vector, text) pairs ARE the store.
# (In the real app these vectors live in Part 6's vector database.)
# ---------------------------------------------------------------------------
CORPUS = [
    "Refunds are accepted within 30 days of purchase, provided the item is unused.",
    "Worn or washed clothing counts as used and is not eligible for a refund.",
    "To start a return, email support@example.com with your order number.",
    "Standard shipping takes 3 to 5 business days; express is next-day.",
    "Items marked final sale cannot be returned or exchanged.",
    "Exchanges for a different size are free within the return window.",
    "Gift cards never expire and are non-refundable.",
]

STORE = [(embed(text), text) for text in CORPUS]   # embed once, keep alongside text


# ---------------------------------------------------------------------------
# retrieve(query, k): cosine of the query vector against every stored chunk,
# sorted best-first, top k. Exactly Part 6's retrieve with the lexical embed.
# ---------------------------------------------------------------------------
def retrieve(query, k=3):
    q = embed(query)
    scored = [(cosine(q, vec), text) for vec, text in STORE]
    scored.sort(key=lambda st: st[0], reverse=True)
    return scored[:k]


# ---------------------------------------------------------------------------
# generate(query, context): the STAND-IN for Part 6's real LLM call. It returns
# a deterministic templated answer built from the retrieved context, so the
# whole file stays reproducible. Part 6's real generate (prompt + model) slots
# in right here, taking the same query and context.
# ---------------------------------------------------------------------------
def generate(query, context):
    top_chunk = context[0][1]                      # context is [(score, text), ...]
    return f"Based on the policy: {top_chunk}"


# ---------------------------------------------------------------------------
# (A) The graceful guard. RELEVANCE_FLOOR is the minimum top score we will
# trust. If retrieval returns nothing, or the best match scores below the
# floor, we refuse honestly instead of generating from junk context. Tune the
# floor to your embedder and corpus: too low and you answer off-topic
# questions, too high and you refuse good ones.
# ---------------------------------------------------------------------------
RELEVANCE_FLOOR = 0.15
REFUSAL = "I don't have information about that in the knowledge base."


def answer(query):
    hits = retrieve(query, k=3)
    if not hits or hits[0][0] < RELEVANCE_FLOOR:
        return REFUSAL                             # the "I don't know" path
    return generate(query, hits)                   # enough signal: answer for real


# ---------------------------------------------------------------------------
# (B) The semantic cache. Each entry is (query_vector, answer). On .get we
# embed the incoming query and return the answer of the FIRST stored entry
# whose cosine is >= threshold (a HIT). Otherwise None (a MISS). The threshold
# trades recall for precision and MUST be tuned per embedder: too low and you
# serve a stale answer to a different question, too high and paraphrases miss.
# ---------------------------------------------------------------------------
class SemanticCache:
    def __init__(self, threshold=0.6):
        self.threshold = threshold
        self.entries = []                          # list of (query_vector, answer)

    def get(self, query):
        q = embed(query)
        for i, (vec, ans) in enumerate(self.entries):
            sim = cosine(q, vec)
            if sim >= self.threshold:
                return ans, i, sim                 # HIT: answer, which entry, cosine
        return None                                # MISS

    def put(self, query, answer):
        self.entries.append((embed(query), answer))


# ---------------------------------------------------------------------------
# cached_answer: check the cache first. On a HIT, serve the cached answer (and
# report the matched entry and cosine). On a MISS, compute via answer(), store
# the result, and return it. This is the production read path: cheap repeat /
# paraphrase questions skip the retrieve + generate work entirely.
# ---------------------------------------------------------------------------
def cached_answer(query, cache):
    hit = cache.get(query)
    if hit is not None:
        ans, idx, sim = hit
        print(f"  cache: HIT (entry {idx}, cosine {sim:.3f}) -> served from cache")
        return ans
    print("  cache: MISS -> computing, then storing")
    ans = answer(query)
    cache.put(query, ans)
    return ans


# ---------------------------------------------------------------------------
# Demo. Fully deterministic. Q1 misses the cache and is computed + stored;
# Q2 is a paraphrase of Q1 that reuses "refund / item / unused", so it HITS
# the cache; Q3 is off-topic, so its top retrieval score falls below the floor
# and we refuse gracefully.
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    cache = SemanticCache(threshold=0.6)

    queries = [
        "Can I get a refund on an unused item?",
        "Could I receive a refund for an item that is unused?",
        "What is the boiling point of water?",
    ]

    for n, query in enumerate(queries, 1):
        print(f"Q{n}: {query}")
        result = cached_answer(query, cache)
        # For the off-topic query, show the top retrieval score so the floor
        # decision is transparent (why we refused rather than answered).
        if result == REFUSAL:
            hits = retrieve(query, k=3)
            top = hits[0][0] if hits else 0.0
            print(f"  retrieval: top score {top:.3f} < floor {RELEVANCE_FLOOR} -> refuse")
        print(f"  answer: {result}\n")

    print("Two touches, one app: refuse when retrieval is weak, reuse when")
    print("the question is (semantically) one we have already answered.")
