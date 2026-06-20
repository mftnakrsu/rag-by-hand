"""
From fixed pipeline to decision-making loop.
RAG from First Principles, Part 10. This is the smallest honest sketch of the
two ideas the essay turns on:

  - CORRECTIVE RAG (CRAG): retrieve, GRADE the chunks, and only fall back to an
    outside source when the grade is bad. The grader is the whole point: it is
    what stops the system answering confidently from irrelevant context.
  - AGENTIC RAG: a reason-act-observe loop that picks a tool each step. The loop
    is powerful and open-ended, which is exactly why it needs a BUDGET CAP so it
    cannot spin forever.

Everything here is mocked deliberately. There is no real LLM call. The "grader"
is a cosine threshold over local embeddings, and the "tools" are dictionaries.
The shapes of the control flow (the grade, the fallback, the step cap, the call
counter) are the load-bearing part, not the toy scores.

Run:
  python agentic_loop.py

If sentence-transformers is installed (as in the run that produced the Expected
output below) it uses real embeddings, so the grader can tell a refund question
from a battery-spec one and routes correctly. Without it, the file falls back to
a tiny lexical hashing embedder so it still RUNS on numpy alone, but those scores
are too crude to route semantically (a refund query shares no content words with
a "Refunds are..." chunk), so the printed routes in the fallback are not
meaningful. The control flow, the grade, the fallback branch, the step cap, and
the call counter, is the point either way.
"""

import numpy as np

# ---------------------------------------------------------------------------
# Embedding: a real model if available, else a tiny deterministic hashing
# embedder so the file runs on numpy alone (same fallback pattern as the other
# companions in this series). Only the control flow below is the point.
# ---------------------------------------------------------------------------
try:
    from sentence_transformers import SentenceTransformer

    _model = SentenceTransformer("all-MiniLM-L6-v2")

    def embed(texts):
        return _model.encode(texts, normalize_embeddings=True)

except Exception:  # no model / no network: deterministic bag-of-hashed-words
    import hashlib
    import re

    _DIM = 1024  # wide enough that distinct words rarely collide

    def _bucket(token):
        # a STABLE hash (process-independent), unlike the builtin hash()
        return int(hashlib.md5(token.encode()).hexdigest(), 16) % _DIM

    def embed(texts):
        out = np.zeros((len(texts), _DIM), dtype=float)
        for i, t in enumerate(texts):
            for tok in re.findall(r"[a-z0-9]+", t.lower()):  # drop punctuation
                out[i, _bucket(tok)] += 1.0
        norms = np.linalg.norm(out, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return out / norms


# ---------------------------------------------------------------------------
# Two "sources" the system can search. The local index holds ONLY store
# policies (the running example in the series). The web is the outside fallback.
# Each call increments a global counter so we can SEE the cost of each strategy.
# ---------------------------------------------------------------------------
LLM_CALLS = {"count": 0}


def _charge(n=1):
    LLM_CALLS["count"] += n


LOCAL_INDEX = [
    "Refunds are accepted within 30 days of purchase, provided the item is unused.",
    "Standard shipping takes 3 to 5 business days; express is next-day.",
    "All electronics include a one-year limited warranty.",
    "Items marked final sale cannot be returned or exchanged.",
]

WEB = [
    "The X1 wireless earbuds deliver up to 8 hours of battery life on a single charge.",
    "The X1 charging case extends total playback to about 30 hours.",
]


def _search(index, query, k=2):
    qv = embed([query])[0]
    mv = embed(index)
    scores = mv @ qv
    order = np.argsort(-scores)[:k]
    return [(index[i], float(scores[i])) for i in order]


def retrieve_local(query, k=2):
    return _search(LOCAL_INDEX, query, k)


def web_search(query, k=2):
    return _search(WEB, query, k)


# ---------------------------------------------------------------------------
# The retrieval evaluator (CRAG's one new component). It grades the best chunk:
# relevant / ambiguous / irrelevant. Here it is a cosine threshold; in a real
# CRAG it is a small classifier. The thresholds are the dial you tune, and the
# pitfall is setting `irrelevant_below` too HIGH so the web fallback triggers on
# perfectly good local answers (over-triggering: slower, costlier, and it leaks
# every query to an outside source).
# ---------------------------------------------------------------------------
def grade(query, chunks, relevant_above=0.45, irrelevant_below=0.30):
    if not chunks:
        return "irrelevant"
    best = max(s for _, s in chunks)
    if best >= relevant_above:
        return "relevant"
    if best < irrelevant_below:
        return "irrelevant"
    return "ambiguous"


def generate(query, chunks):
    _charge()  # the one generation call
    top = chunks[0][0] if chunks else "(no context)"
    return f"[answer grounded in: {top!r}]"


# ---------------------------------------------------------------------------
# CORRECTIVE RAG: retrieve -> grade -> correct-or-generate. The grader gates the
# fallback so we do not pay for web search on every query, only on bad ones.
# ---------------------------------------------------------------------------
def corrective_rag(query, max_tries=2):
    for _ in range(max_tries):
        local = retrieve_local(query)
        g = grade(query, local)
        if g == "relevant":
            return generate(query, local), "local"
        if g == "irrelevant":
            web = web_search(query)        # fall back to a different source
            return generate(query, web), "web-fallback"
        # ambiguous: loop and try the local index once more (a real system would
        # reformulate the query here). Bounded by max_tries either way.
    return "I don't know based on the available sources.", "refused"


# ---------------------------------------------------------------------------
# AGENTIC RAG: a reason-act-observe loop with a HARD BUDGET CAP. Without the cap
# an agent that keeps grading its own retrieval as "not good enough" can loop
# forever, burning a model call every step. `max_steps` is the seatbelt: when it
# runs out, the agent stops and answers (or refuses) instead of spinning.
# ---------------------------------------------------------------------------
def agentic_rag(query, max_steps=4):
    tried_web = False
    for step in range(1, max_steps + 1):
        _charge()  # every reason-act-observe cycle is a model call
        chunks = web_search(query) if tried_web else retrieve_local(query)
        g = grade(query, chunks)
        if g == "relevant":
            return generate(query, chunks), step
        if not tried_web:
            tried_web = True   # the corrective action: switch tools, then retry
            continue
        # already tried both sources and still not satisfied: keep looping until
        # the budget runs out. THIS is where an uncapped agent would never stop.
    # budget exhausted: stop and answer from whatever we last had, do not spin.
    return generate(query, chunks), max_steps


if __name__ == "__main__":
    # A question the LOCAL policy index CAN answer.
    in_corpus = "What is the refund window?"
    # A product-spec question the policy index CANNOT answer (needs the web).
    out_of_corpus = "What is the battery life of the X1 wireless earbuds?"

    print("CORRECTIVE RAG")
    for q in (in_corpus, out_of_corpus):
        LLM_CALLS["count"] = 0
        ans, route = corrective_rag(q)
        print(f"  Q: {q}")
        print(f"     route={route}  llm_calls={LLM_CALLS['count']}")
        print(f"     {ans}")

    print("\nAGENTIC RAG (budget cap = 4 steps)")
    for q in (in_corpus, out_of_corpus):
        LLM_CALLS["count"] = 0
        ans, steps = agentic_rag(q)
        print(f"  Q: {q}")
        print(f"     steps={steps}  llm_calls={LLM_CALLS['count']}")
        print(f"     {ans}")

    print("\nThe cost gap is the whole lesson: the agent spends more LLM calls")
    print("for the same answers. Reach for it only when the task needs the loop.")

# Expected output (with sentence-transformers installed):
# CORRECTIVE RAG
#   Q: What is the refund window?
#      route=local  llm_calls=1
#      [answer grounded in: 'Refunds are accepted within 30 days of purchase, provided the item is unused.']
#   Q: What is the battery life of the X1 wireless earbuds?
#      route=web-fallback  llm_calls=1
#      [answer grounded in: 'The X1 wireless earbuds deliver up to 8 hours of battery life on a single charge.']
#
# AGENTIC RAG (budget cap = 4 steps)
#   Q: What is the refund window?
#      steps=1  llm_calls=2
#      [answer grounded in: 'Refunds are accepted within 30 days of purchase, provided the item is unused.']
#   Q: What is the battery life of the X1 wireless earbuds?
#      steps=2  llm_calls=3
#      [answer grounded in: 'The X1 wireless earbuds deliver up to 8 hours of battery life on a single charge.']
#
# The cost gap is the whole lesson: the agent spends more LLM calls
# for the same answers. Reach for it only when the task needs the loop.
#
# Notice the call counts: CRAG spends 1 generation call per query (the grader is
# cheap), while the agent spends 2 to 3 (one per reason-act-observe step). On a
# real multi-hop task that gap widens to the 3 to 10x the essay warns about.
