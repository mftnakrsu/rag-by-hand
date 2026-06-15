"""
Adaptive RAG: one classifier that routes each query to the right pipeline.
RAG from First Principles, Part 15 (Frontier Track close): don't pay for
retrieval an easy query doesn't need, don't under-retrieve a hard one.

For nine parts we built pipelines that run the SAME path every time. Part 10
gave that pipeline a brain (control flow); this part gives it a CONDUCTOR. A
tiny complexity classifier reads each query and routes it to one of three
pipelines we already know how to build:

  - none   -> answer directly, no retrieval at all   (greeting / model already knows)
  - single -> one retrieve -> generate               (the Part 6 pipeline)
  - multi  -> decompose -> retrieve per part -> synthesize (the Part 10 shape)

This is route-by-COMPLEXITY. It is distinct from Part 8 (which TRANSFORMS the
query to retrieve better) and from Part 10's source routing (which routes by
WHICH knowledge source, not by how hard the question is).

Stack:
  - Classifier : a deterministic rule/keyword classifier (the offline default).
                 A production system would use a small TRAINED classifier; that
                 path sits behind a try/except and prints a banner when it loads.
  - Embeddings : sentence-transformers (the headline) with a transparent
                 deterministic hashing fallback (the Part 2 pattern), so the
                 single- and multi-step routes run with NO model and NO network.
  - Generation : a grounded extractive stand-in (quotes the best chunk), so the
                 whole demo produces sensible, source-backed output with no LLM.

Run:
  pip install numpy                 # numpy is the only dependency
  python3 adaptive_rag.py           # runs offline; no API key, no network
  # Optional, for the REAL embedder path: pip install sentence-transformers

NOTE: the classifier here is intentionally a transparent set of rules, not a
learned model. That is the teaching point: you can see exactly WHY each query
routes the way it does. The cost/latency wins Adaptive RAG is sold on (~35%
latency, ~28% cost in one 2026 vendor report) are INDICATIVE, not measured
facts from the paper; treat them as direction, not numbers to quote.

Expected output (deterministic fallback path, e.g. when sentence-transformers
is absent; the cosine scores below are from the hashing stand-in). If the model
is instead cached locally, the REAL embedder runs with no network and only the
cosine scores change -- the route labels and the none/single/multi tally are
identical either way. (HF_HUB_OFFLINE=1 does NOT by itself force the fallback:
with a cached model it still loads the real one.)

======================================================================
ADAPTIVE RAG  -  route each query to the pipeline it actually needs
======================================================================
[embed] sentence-transformers unavailable (OSError); using deterministic hashing fallback
[classify] trained classifier unavailable (OSError); using deterministic rule/keyword classifier (offline default)

Knowledge base: 5 support chunks (refund policy, the E-4042 error code, shipping, warranty).

----------------------------------------------------------------------
ROUTING 6 EXAMPLE QUERIES
----------------------------------------------------------------------

QUERY: "hi there"
  classify_complexity -> none
  route -> NO RETRIEVAL (answer directly)
  ANSWER: Hi! I'm the support assistant. Ask me about refunds, shipping, the E-4042 error, or your warranty and I'll look it up.

QUERY: "thanks"
  classify_complexity -> none
  route -> NO RETRIEVAL (answer directly)
  ANSWER: You're welcome! Anything else about your order or our policies?

QUERY: "What is our refund window?"
  classify_complexity -> single
  route -> SINGLE-STEP retrieve -> generate (Part 6)
    retrieved (top-1, score=0.20): Refunds are accepted within 30 days of pur...
  ANSWER: Based on the retrieved policy: Refunds are accepted within 30 days of purchase, provided the item is unused and in its original packaging.

QUERY: "How do I fix the E-4042 error?"
  classify_complexity -> single
  route -> SINGLE-STEP retrieve -> generate (Part 6)
    retrieved (top-1, score=0.34): Error E-4042 means the payment was decline...
  ANSWER: Based on the retrieved policy: Error E-4042 means the payment was declined by the bank; ask the customer to retry with a different card or contact their bank.

QUERY: "Compare the refund window and the warranty period, and explain the difference."
  classify_complexity -> multi
  route -> MULTI-STEP decompose -> retrieve per part -> synthesize (Part 10)
    sub-query 1: "refund window"
      retrieved (top-1, score=0.20): Refunds are accepted within 30 days of pur...
    sub-query 2: "the warranty period"
      retrieved (top-1, score=0.22): All electronics include a one-year limited...
  ANSWER: Putting the pieces together:
            - refund window: Refunds are accepted within 30 days of purchase, provided the item is unused and in its original packaging.
            - the warranty period: All electronics include a one-year limited warranty covering manufacturing defects.

QUERY: "What is the difference between the refund window and the warranty period?"
  classify_complexity -> multi
  route -> MULTI-STEP decompose -> retrieve per part -> synthesize (Part 10)
    sub-query 1: "the refund window"
      retrieved (top-1, score=0.20): Refunds are accepted within 30 days of pur...
    sub-query 2: "the warranty period"
      retrieved (top-1, score=0.22): All electronics include a one-year limited...
  ANSWER: Putting the pieces together:
            - the refund window: Refunds are accepted within 30 days of purchase, provided the item is unused and in its original packaging.
            - the warranty period: All electronics include a one-year limited warranty covering manufacturing defects.

----------------------------------------------------------------------
ROUTE TALLY: none=2  single=2  multi=2
----------------------------------------------------------------------
Takeaway: one fixed pipeline over-serves the greetings (paying for retrieval
they don't need) and under-serves the comparison (one lookup can't answer it).
The classifier is the conductor over Parts 6 to 10 -- and is itself a failure
surface: a misroute means under-retrieval, so grade it like any other component.
"""

import os
import re

import numpy as np

# ---------------------------------------------------------------------------
# Step 0. The support knowledge base, carried over from Parts 6 to 12 so this
#         part feels continuous: refunds, the E-4042 error code, shipping,
#         warranty. Short docs, so each doc is its own chunk (Part 5).
# ---------------------------------------------------------------------------
KNOWLEDGE_BASE = [
    "Refunds are accepted within 30 days of purchase, provided the item is unused and in its original packaging.",
    "To start a return, email support@example.com with your order number. Refunds are processed within five business days of us receiving the item.",
    "Error E-4042 means the payment was declined by the bank; ask the customer to retry with a different card or contact their bank.",
    "Standard shipping takes 3 to 5 business days. Express shipping arrives the next business day.",
    "All electronics include a one-year limited warranty covering manufacturing defects.",
]


# ===========================================================================
# Step 1. The complexity classifier -- the heart of Adaptive RAG.
#
#   This is the correctness-critical function. It reads ONE query and returns
#   the route it deserves, by inspecting cheap, transparent signals:
#     - very short / greeting / chit-chat            -> 'none'  (no retrieval)
#     - a comparison / conjunction / multi-question  -> 'multi' (decompose)
#     - everything else                              -> 'single'(one lookup)
#
#   A production system swaps this for a small TRAINED classifier (the real
#   path below). The rules are deliberately legible so you can see exactly WHY
#   a query routes the way it does -- the whole teaching point of the part.
# ===========================================================================
def classify_complexity(query: str) -> str:
    """Route a query by complexity: 'none' (no retrieval needed),
    'single' (one retrieve->generate), or 'multi' (decompose, multi-step).
    Deterministic rule/keyword classifier; a real system would use a small
    trained classifier, shown here as the fallback path."""
    q = query.lower().strip()
    if re.search(r"\b(hi|hello|thanks|who are you)\b", q) or len(q.split()) <= 2:
        return "none"
    multi_signals = ("compare", "versus", " vs ", "difference between",
                     " and then", "across", "each of", "both", "trade-off")
    if any(s in q for s in multi_signals) or q.count("?") > 1:
        return "multi"
    return "single"


# ---------------------------------------------------------------------------
# Step 1b. The REAL classifier path (headline; sits behind try/except).
#
#   In production the router is usually a small trained text classifier
#   (e.g. a fine-tuned encoder or a tiny LLM). We show the SHAPE and load it
#   behind a guard so this file still runs with no model and no network: if the
#   load fails, we keep the deterministic rules above and print a clear banner.
# ---------------------------------------------------------------------------
def load_real_classifier():
    """Try to load a small trained complexity classifier; None if unavailable."""
    try:
        # Intended path: a tiny text-classification model returning one of the
        # three route labels. Names/APIs move fast; check current docs.
        from transformers import pipeline  # noqa: F401

        clf = pipeline("text-classification", model="adaptive-rag-router")

        def classify(query: str) -> str:
            label = clf(query)[0]["label"].lower()
            return label if label in {"none", "single", "multi"} else classify_complexity(query)

        return classify
    except Exception as exc:  # not installed / no weights / offline
        print(f"[classify] trained classifier unavailable ({type(exc).__name__}); "
              "using deterministic rule/keyword classifier (offline default)")
        return None


# ===========================================================================
# Step 2. Embeddings, with a transparent deterministic fallback (Part 2).
#
#   The single- and multi-step routes need real retrieval, so they need an
#   embedder. The headline is sentence-transformers; if it can't load offline
#   we drop to a deterministic hashing embedder so the routes still run.
# ===========================================================================
_TOKEN_RE = re.compile(r"[a-z0-9]+")

# Grammatical words the offline embedder ignores so a query's CONTENT words
# (refund, window, warranty) drive the cosine instead of "what / is / our".
# A real model handles this for free; the hashing stand-in needs the hint.
_STOPWORDS = {
    "what", "is", "are", "the", "of", "a", "an", "for", "on", "in", "to", "how",
    "do", "does", "and", "my", "i", "there", "with", "your", "our", "between",
    "explain", "compare", "difference",
}


def _stem(tok):
    """Crudest possible stemmer: drop a trailing plural 's' so 'refunds'
    and 'refund' hash to the same bucket. A real model needs no such hint;
    the lexical stand-in does, or 'refund window' never meets 'refunds'."""
    return tok[:-1] if len(tok) > 3 and tok.endswith("s") else tok


def _tokens(text):
    """Lowercase word/number tokens -- the level the essay stays at."""
    return _TOKEN_RE.findall(text.lower())


class _HashingEmbedder:
    """Deterministic, model-free stand-in for a sentence embedder (Part 2).

    Hash each CONTENT token into a fixed-width vector and accumulate, then
    L2-normalize. Stop-words are dropped so shared content words dominate the
    cosine -- crude, but enough to put 'refund window' next to the refund chunk
    without a 90 MB download. A real model captures meaning; this captures
    content-word overlap projected into a short dense vector."""

    def __init__(self, dim=256):
        self.dim = dim

    def encode(self, texts, normalize_embeddings=True):
        vecs = np.zeros((len(texts), self.dim), dtype=np.float64)
        for r, text in enumerate(texts):
            toks = [_stem(t) for t in _tokens(text) if t not in _STOPWORDS]
            for tok in toks:
                # Stable hash -> bucket; sign spreads collisions. Deterministic
                # across runs (no dependence on PYTHONHASHSEED).
                h = 0
                for ch in tok:
                    h = (h * 131 + ord(ch)) & 0xFFFFFFFF
                vecs[r, h % self.dim] += 1.0 if (h >> 1) & 1 else -1.0
        if normalize_embeddings:
            norms = np.linalg.norm(vecs, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            vecs = vecs / norms
        return vecs


def load_real_embedder():
    """Try the real model first; fall back transparently if unavailable."""
    try:
        from sentence_transformers import SentenceTransformer  # REAL path

        model = SentenceTransformer("all-MiniLM-L6-v2")        # 384 dims
        print("[embed] using sentence-transformers (all-MiniLM-L6-v2)")
        return model, True
    except Exception as exc:  # not installed / no weights / offline
        print(f"[embed] sentence-transformers unavailable ({type(exc).__name__}); "
              "using deterministic hashing fallback")
        return _HashingEmbedder(), False


# ---------------------------------------------------------------------------
# Step 3. A tiny vector store + retriever (Part 4 + Part 6). The "store" is the
#         chunks side by side with a matrix of their vectors; retrieve embeds
#         the query with the SAME model and keeps the top-k by cosine.
# ---------------------------------------------------------------------------
class VectorStore:
    def __init__(self, corpus, embed):
        self.chunks = list(corpus)
        self.vectors = embed(self.chunks)  # (n_chunks, dim)
        self._embed = embed

    def retrieve(self, query, k=1):
        q = self._embed([query])[0]               # same model as the chunks
        scores = self.vectors @ q                 # cosine sim (unit vectors)
        top = np.argsort(-scores)[:k]             # indices of k highest scores
        return [(self.chunks[i], float(scores[i])) for i in top]


# ---------------------------------------------------------------------------
# Step 4. Generation -- a grounded extractive stand-in (Part 6's grounding).
#         The real path is one swappable hosted-LLM call; offline we quote the
#         best retrieved chunk so the answer is visibly tied to a source.
# ---------------------------------------------------------------------------
def generate(retrieved):
    """Grounded extractive generate(): quote the single best-retrieved chunk."""
    if not retrieved:
        return "I don't know based on the available sources."
    best_text, _score = retrieved[0]
    return f"Based on the retrieved policy: {best_text}"


# ===========================================================================
# Step 5. The router -- dispatch each query to the pipeline it deserves.
#
#   route() is the conductor. It classifies the query, then runs ONE of three
#   pipelines we already built across Parts 6 to 10:
#     none   -> a direct templated answer, no retrieval, no embedder call;
#     single -> the Part 6 retrieve -> generate pipeline;
#     multi  -> decompose into sub-queries, retrieve PER sub-query, synthesize.
#   `trace` makes the chosen path visible (the demo turns it on).
# ===========================================================================
# Canned no-retrieval replies for the 'none' route. A real system would let the
# LLM answer directly here; offline we template so it stays deterministic.
_DIRECT_REPLIES = {
    "greeting": ("Hi! I'm the support assistant. Ask me about refunds, shipping, "
                 "the E-4042 error, or your warranty and I'll look it up."),
    "thanks": "You're welcome! Anything else about your order or our policies?",
    "identity": ("I'm the support assistant for this store. I answer from our "
                 "policy docs -- refunds, shipping, errors, and warranty."),
}


def _direct_answer(query: str) -> str:
    """The 'none' route: answer without touching the index."""
    q = query.lower()
    if "thank" in q:
        return _DIRECT_REPLIES["thanks"]
    if "who are you" in q or "what are you" in q:
        return _DIRECT_REPLIES["identity"]
    return _DIRECT_REPLIES["greeting"]


# Split a 'multi' query into sub-queries on the same surface signals the
# classifier keyed on: comparisons, conjunctions, and multiple questions.
_DECOMPOSE_RE = re.compile(
    r"\s+and then\s+|\s+versus\s+|\s+vs\.?\s+|\s+difference between\s+"
    r"|,?\s+and\s+|\s*\?\s*",
    flags=re.IGNORECASE,
)
# Lead-in phrases to strip from each sub-query so retrieval sees the content.
_LEADINS = ("compare the", "compare", "explain the", "explain",
            "the difference between", "difference between", "what is")


def decompose(query: str) -> list:
    """Break a complex query into smaller, independently-retrievable parts."""
    raw = [p.strip(" .") for p in _DECOMPOSE_RE.split(query) if p.strip(" .")]
    subs = []
    for part in raw:
        cleaned = part
        low = cleaned.lower()
        for lead in _LEADINS:
            if low.startswith(lead):
                cleaned = cleaned[len(lead):].strip()
                break
        # Drop fragments that are nothing but connective/stop words (e.g. a
        # leftover "the difference") -- they have no content to retrieve on.
        if cleaned and any(t not in _STOPWORDS for t in _tokens(cleaned)):
            subs.append(cleaned)
    # Always keep at least the original so 'multi' never retrieves nothing.
    return subs or [query]


def route(query: str, store: VectorStore, classify, trace=True) -> str:
    """Classify the query, then run the matching pipeline (none/single/multi)."""
    def log(msg):
        if trace:
            print(msg)

    complexity = classify(query)
    log(f'\nQUERY: "{query}"')
    log(f"  classify_complexity -> {complexity}")

    if complexity == "none":
        log("  route -> NO RETRIEVAL (answer directly)")
        return _direct_answer(query)

    if complexity == "single":
        log("  route -> SINGLE-STEP retrieve -> generate (Part 6)")
        retrieved = store.retrieve(query, k=1)
        text, score = retrieved[0]
        log(f"    retrieved (top-1, score={score:.2f}): {text[:42]}...")
        return generate(retrieved)

    # complexity == "multi": decompose -> retrieve per part -> synthesize.
    log("  route -> MULTI-STEP decompose -> retrieve per part -> synthesize (Part 10)")
    subs = decompose(query)
    pieces = []
    for i, sub in enumerate(subs, start=1):
        retrieved = store.retrieve(sub, k=1)
        text, score = retrieved[0]
        log(f'    sub-query {i}: "{sub}"')
        log(f"      retrieved (top-1, score={score:.2f}): {text[:42]}...")
        pieces.append((sub, text))
    lines = "\n".join(f"          - {sub}: {text}" for sub, text in pieces)
    return "Putting the pieces together:\n" + lines


# ===========================================================================
# Demo. Everything below RUNS OFFLINE. It uses the real classifier/embedder if
#       available and the deterministic stand-ins otherwise, with clear labels.
# ===========================================================================
if __name__ == "__main__":
    line = "=" * 70

    print(line)
    print("ADAPTIVE RAG  -  route each query to the pipeline it actually needs")
    print(line)

    embedder, real_embed = load_real_embedder()

    def embed(texts):
        return np.asarray(embedder.encode(texts, normalize_embeddings=True))

    real_classify = load_real_classifier()
    use_real_classify = real_classify is not None
    classify = real_classify if use_real_classify else classify_complexity
    if use_real_classify:
        print("[classify] using trained complexity classifier")

    store = VectorStore(KNOWLEDGE_BASE, embed)
    print(f"\nKnowledge base: {len(KNOWLEDGE_BASE)} support chunks "
          "(refund policy, the E-4042 error code, shipping, warranty).")

    # Two queries per class: the conductor should send each to a different path.
    examples = [
        "hi there",                                                              # none
        "thanks",                                                                # none
        "What is our refund window?",                                            # single
        "How do I fix the E-4042 error?",                                        # single
        "Compare the refund window and the warranty period, and explain the difference.",  # multi
        "What is the difference between the refund window and the warranty period?",        # multi
    ]

    print("\n" + "-" * 70)
    print(f"ROUTING {len(examples)} EXAMPLE QUERIES")
    print("-" * 70)

    tally = {"none": 0, "single": 0, "multi": 0}
    for q in examples:
        tally[classify(q)] += 1
        answer = route(q, store, classify, trace=True)
        # Wrap the answer onto continuation lines for readable output.
        print(f"  ANSWER: {answer}".replace("\n", "\n  "))

    print("\n" + "-" * 70)
    print(f"ROUTE TALLY: none={tally['none']}  single={tally['single']}  multi={tally['multi']}")
    print("-" * 70)
    print("Takeaway: one fixed pipeline over-serves the greetings (paying for retrieval")
    print("they don't need) and under-serves the comparison (one lookup can't answer it).")
    print("The classifier is the conductor over Parts 6 to 10 -- and is itself a failure")
    print("surface: a misroute means under-retrieval, so grade it like any other component.")
