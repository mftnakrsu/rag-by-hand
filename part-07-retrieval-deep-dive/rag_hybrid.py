"""
rag_hybrid.py  -  RAG from First Principles, Part 7 ("Retrieval Deep Dive")

Extends the Part 6 app with SPARSE retrieval (BM25) and HYBRID fusion
(Reciprocal Rank Fusion + a weighted alternative). This is an *extension*,
not a rebuild: in your real app the dense scores come straight from Part 6's
cosine-over-embeddings search. Everything here is pure standard library, so the
file runs on its own and prints a deterministic demo.

Library note: BM25 and dense retrieval have great off-the-shelf libraries
(e.g. `rank_bm25`, `sentence-transformers`) whose APIs move fast. The hand-
rolled BM25 below is for understanding; verify current library usage before
shipping. The fusion functions, though, are exactly what you would use.

    python rag_hybrid.py
"""

import math
import re
from collections import Counter


# ---------------------------------------------------------------------------
# The same little support knowledge base our Part 6 app indexed.
# Chunk 0 is the one that literally answers the query, but it is terse and
# code-heavy, so a dense model "rounds off" the exact code E-4042.
# ---------------------------------------------------------------------------
CORPUS = [
    "Error E-4042: the authentication token has expired. Refresh the token and retry the request.",
    "Troubleshooting checkout and payment failures: common causes and first steps.",
    "Resolving login and authentication issues for returning customers.",
    "The checkout page shows a generic error after the customer clicks Pay.",
    "Contact our support team about an existing order or delivery.",
    "Refund policy: refunds are accepted within 30 days of purchase.",
]

QUERY = "how do I fix error E-4042 at checkout?"


# ---------------------------------------------------------------------------
# Sparse retrieval: a compact, readable BM25.
# ---------------------------------------------------------------------------
def tokenize(text):
    # keep hyphenated codes like "e-4042" intact as one token
    return re.findall(r"[a-z0-9]+(?:-[a-z0-9]+)*", text.lower())


class BM25:
    """Okapi BM25. k1 controls term-frequency saturation; b controls how much
    we normalize for document length."""

    def __init__(self, corpus, k1=1.5, b=0.75):
        self.k1, self.b = k1, b
        self.docs = [tokenize(d) for d in corpus]
        self.N = len(self.docs)
        self.doc_len = [len(d) for d in self.docs]
        self.avgdl = sum(self.doc_len) / self.N
        # document frequency: how many docs contain each term
        df = Counter()
        for d in self.docs:
            for term in set(d):
                df[term] += 1
        # inverse document frequency: rare terms weigh more
        self.idf = {
            term: math.log(1 + (self.N - n + 0.5) / (n + 0.5))
            for term, n in df.items()
        }

    def scores(self, query):
        q = tokenize(query)
        out = [0.0] * self.N
        for i, doc in enumerate(self.docs):
            tf = Counter(doc)
            for term in q:
                if term not in tf:
                    continue
                freq = tf[term]
                # saturation: more occurrences help less and less
                numer = freq * (self.k1 + 1)
                denom = freq + self.k1 * (1 - self.b + self.b * self.doc_len[i] / self.avgdl)
                out[i] += self.idf.get(term, 0.0) * numer / denom
        return out


# ---------------------------------------------------------------------------
# Dense retrieval: in your real app, import it from Part 6.
# The fallback below lets this file run standalone; the numbers mimic what a
# dense model returns for QUERY (on-topic chunks score high, the exact-code
# chunk gets blurred down). See the Part 7 prose for the intuition.
# ---------------------------------------------------------------------------
try:
    from rag import dense_search          # your Part 6 function
except Exception:                          # standalone demo fallback
    _DEMO_DENSE = [0.55, 0.88, 0.80, 0.70, 0.62, 0.45]

    def dense_search(query, corpus):
        return list(_DEMO_DENSE)


# ---------------------------------------------------------------------------
# Fusion.
# ---------------------------------------------------------------------------
def min_max(scores):
    """Squash a score list into 0..1 so dense and sparse become comparable.
    Mandatory before any weighted sum: the two live on different scales."""
    lo, hi = min(scores), max(scores)
    if hi == lo:
        return [0.0 for _ in scores]
    return [(s - lo) / (hi - lo) for s in scores]


def weighted_fusion(dense, sparse, alpha=0.5):
    """Convex combination after normalization. alpha=1 -> pure dense,
    alpha=0 -> pure sparse."""
    d, s = min_max(dense), min_max(sparse)
    return [alpha * d[i] + (1 - alpha) * s[i] for i in range(len(d))]


def rrf(*rankings, k=60):
    """Reciprocal Rank Fusion. Each ranking is a list of doc indices, best
    first. A doc earns 1/(k+rank) from each list; we sum across lists. Merges
    by *rank*, so no normalization and no alpha are needed."""
    fused = Counter()
    for ranking in rankings:
        for rank, doc_id in enumerate(ranking, start=1):
            fused[doc_id] += 1.0 / (k + rank)
    return fused


def order(scores):
    """Doc indices sorted by score, best first."""
    return sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)


def show(title, ranked, scores):
    print(f"\n{title}")
    for rank, i in enumerate(ranked[:3], start=1):
        print(f"  {rank}. [{scores[i]:.3f}] {CORPUS[i][:54]}")


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print(f"Query: {QUERY!r}")

    dense = dense_search(QUERY, CORPUS)
    sparse = BM25(CORPUS).scores(QUERY)

    dense_rank = order(dense)
    sparse_rank = order(sparse)

    show("DENSE only (meaning):", dense_rank, dense)
    show("SPARSE only (BM25 keywords):", sparse_rank, sparse)

    # Hybrid via RRF, the robust default.
    fused = rrf(dense_rank, sparse_rank)
    rrf_rank = [i for i, _ in fused.most_common()]
    show("HYBRID via RRF:", rrf_rank, {i: fused[i] for i in range(len(CORPUS))})

    # Hybrid via weighted sum, for comparison.
    w = weighted_fusion(dense, sparse, alpha=0.5)
    show("HYBRID via weighted sum (alpha=0.5):", order(w), w)


# ---------------------------------------------------------------------------
# Expected output (deterministic):
#
# Query: 'how do I fix error E-4042 at checkout?'
#
# DENSE only (meaning):
#   1. [0.880] Troubleshooting checkout and payment failures: common
#   2. [0.800] Resolving login and authentication issues for returnin
#   3. [0.700] The checkout page shows a generic error after the cust
#
# SPARSE only (BM25 keywords):
#   1. [2.253] Error E-4042: the authentication token has expired. Re
#   2. [1.950] The checkout page shows a generic error after the cust
#   3. [1.059] Troubleshooting checkout and payment failures: common
#
# HYBRID via RRF:
#   1. [0.032] Troubleshooting checkout and payment failures: common
#   2. [0.032] The checkout page shows a generic error after the cust
#   3. [0.032] Error E-4042: the authentication token has expired. Re
#
# HYBRID via weighted sum (alpha=0.5):
#   1. [0.735] Troubleshooting checkout and payment failures: common
#   2. [0.723] The checkout page shows a generic error after the cust
#   3. [0.616] Error E-4042: the authentication token has expired. Re
#
# The exact-code chunk is INVISIBLE in dense's top 3 (dense ranks it 5th: the
# code "E-4042" gets rounded off). Sparse ranks it 1st. Either fusion rescues
# it into the top 3. Push alpha toward 0 (more sparse) and it climbs higher
# still. That rescue is the whole point of hybrid search.
# ---------------------------------------------------------------------------
