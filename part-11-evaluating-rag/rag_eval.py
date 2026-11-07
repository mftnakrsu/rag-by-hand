"""
rag_eval.py  -  RAG from First Principles, Part 11 ("Evaluating RAG")

A tiny OFFLINE evaluation loop over the Part 6 app. Not a framework: just the
*shape* of one. It measures two of the four core metrics and uses them to point
at the failing half of the pipeline (retrieval vs generation):

  - Context Recall  : did retrieval fetch the chunk that holds the answer?   (RETRIEVAL)
  - Faithfulness    : is every claim in the answer supported by that context? (GENERATION)

The diagnostic rule, the whole point of measuring component-by-component:
  low recall            -> a RETRIEVAL problem (the answer never reached the model)
  high recall, low faith-> a GENERATION problem (had the context, still made things up)

Run (pure standard library, no API key or model needed):

    python rag_eval.py

NOTE: real eval frameworks (RAGAS, TruLens, DeepEval, Arize Phoenix, LangSmith,
and the cloud eval services) compute richer versions of these with an
LLM-as-a-judge, and they move FAST: metric names and APIs get renamed across
releases. Treat this file as the underlying idea, pin a version before you rely
on a library, and check the current docs.
"""

import os
import re

# ---------------------------------------------------------------------------
# The same little store-policy corpus the Part 6 app indexed. doc_i is its id.
# ---------------------------------------------------------------------------
CORPUS = [
    "Refunds are accepted within 30 days of purchase, provided the item is unused and in its original packaging.",
    "To start a return, email support@example.com with your order number. Refunds are processed within five business days of us receiving the item.",
    "Standard shipping takes 3 to 5 business days. Express shipping arrives the next business day.",
    "Shipping fees are non-refundable, and items marked final sale cannot be returned or exchanged.",
    "All electronics include a one-year limited warranty covering manufacturing defects.",
]
DOCS = [{"id": f"doc_{i}", "text": t} for i, t in enumerate(CORPUS)]

STOP = {
    "a", "an", "the", "to", "of", "for", "and", "or", "is", "are", "was", "be",
    "in", "on", "at", "it", "its", "i", "you", "your", "we", "our", "my", "me",
    "do", "does", "how", "can", "will", "with", "within", "from", "have", "has",
    "if", "that", "this", "they", "them", "their", "us", "what", "about",
}


def tokenize(text):
    # keep hyphenated codes/ids like "e-4042" or "1-800-returns" intact
    return re.findall(r"[a-z0-9]+(?:-[a-z0-9]+)*", text.lower())


def content_tokens(text):
    return [t for t in tokenize(text) if t not in STOP]


# ---------------------------------------------------------------------------
# Retrieval. In your real app this is Part 6's cosine-over-embeddings search;
# here it is a transparent lexical (token-overlap) retriever so the file runs
# standalone. It returns only chunks that actually overlap the query, which is
# exactly why a *semantic* query with no shared words gets nothing back, the
# dense-vs-sparse gap from Part 7.
# ---------------------------------------------------------------------------
def retrieve(query, k=3):
    q = set(content_tokens(query))
    scored = []
    for d in DOCS:
        overlap = len(q & set(content_tokens(d["text"])))
        if overlap > 0:                       # no overlap -> not retrieved at all
            scored.append((overlap, d["id"], d))
    # best overlap first; ties broken by id so the demo is deterministic
    scored.sort(key=lambda s: (-s[0], s[1]))
    return [d for _score, _id, d in scored[:k]]


# ---------------------------------------------------------------------------
# Metric 1 (RETRIEVAL): Context Recall.
# Did the retrieved set contain the "golden" chunk(s) that actually hold the
# answer? Production tools decompose the *reference answer* into claims and check
# coverage; this id-based version is the transparent stand-in. Returns None when
# the question is out of scope (no golden chunk exists to retrieve).
# ---------------------------------------------------------------------------
def context_recall(golden_ids, retrieved):
    if not golden_ids:
        return None
    got = {d["id"] for d in retrieved}
    hit = sum(1 for g in golden_ids if g in got)
    return hit / len(golden_ids)


# ---------------------------------------------------------------------------
# Metric 2 (GENERATION): Faithfulness via LLM-as-a-judge.
# The real method: an LLM breaks the answer into atomic claims and checks each
# one for support against the retrieved context (an "NLI" / entailment check:
# does the context entail this claim?). Score = supported claims / total claims.
# ---------------------------------------------------------------------------
JUDGE_PROMPT = """You are grading a RAG answer for FAITHFULNESS.
Given the CONTEXT and the ANSWER, list each atomic claim in the answer, then for
each claim decide whether the CONTEXT supports it (true) or not (false).
Return faithfulness = (supported claims) / (total claims), a number from 0 to 1.

CONTEXT:
{context}

ANSWER:
{answer}
"""

REFUSALS = ("don't know", "do not know", "cannot find", "couldn't find")


def judge_faithfulness(question, context, answer):
    """Score how grounded `answer` is in `context`. Set RAG_EVAL_USE_LLM_JUDGE=1
    to call a hosted judge (shown for shape); by default it uses the transparent,
    cost-free fallback, so simply running this file never spends tokens even if an
    OPENAI_API_KEY happens to be exported."""
    if os.getenv("RAG_EVAL_USE_LLM_JUDGE"):
        try:
            from openai import OpenAI
            client = OpenAI()
            client.chat.completions.create(            # the real call shape
                model="gpt-6-luna",                    # successor to gpt-4o-mini; check current names
                messages=[{"role": "user",
                           "content": JUDGE_PROMPT.format(context=context, answer=answer)}],
                temperature=0,
            )
            # Parse the judge's number here. We fall through to the deterministic
            # version below so this file always prints the same demo output.
        except Exception:
            pass
    return _fallback_faithfulness(answer, context)


def _fallback_faithfulness(answer, context):
    """A deterministic stand-in for the LLM judge: split the answer into atomic
    claims and call a claim 'supported' when most of its content words appear in
    the context. Crude, but it shows the supported/total shape, and it catches
    the invented '60 days' / 'instant refund' style hallucination."""
    low = answer.lower()
    if any(p in low for p in REFUSALS):       # an honest refusal invents nothing
        return 1.0, ["(refusal: no claims to verify)"]
    ctx = set(content_tokens(context))
    claims = _split_claims(answer)
    verdicts, supported = [], 0
    for c in claims:
        toks = content_tokens(c)
        if not toks:
            continue
        frac = sum(1 for t in toks if t in ctx) / len(toks)
        ok = frac >= 0.6                      # most of the claim is grounded
        supported += ok
        verdicts.append(f"[{'ok ' if ok else 'NO '}{frac:.2f}] {c.strip()}")
    score = supported / len(verdicts) if verdicts else 1.0
    return score, verdicts


def _split_claims(answer):
    parts = re.split(r"[.!?]\s+", answer)     # sentences
    claims = []
    for p in parts:
        claims.extend(s for s in p.split(" and ") if s.strip())  # then conjuncts
    return claims


# ---------------------------------------------------------------------------
# The eval set. Each case: a question, the golden chunk id(s) that answer it
# (empty = out of scope, the system SHOULD refuse), and a reference answer.
# `answer` here is a canned stand-in for what `ask(q)` returns in Part 6, so the
# demo is deterministic; in your loop you would call the real app instead.
# ---------------------------------------------------------------------------
EVAL_SET = [
    {
        "q": "How many days do I have to return an item for a refund?",
        "golden": ["doc_0"],
        "reference": "30 days from purchase, if the item is unused and in its original packaging.",
        "answer": "You have 30 days from purchase to return an item for a refund, "
                  "if it is unused and in its original packaging.",
    },
    {
        "q": "How do I start a return?",
        "golden": ["doc_1"],
        "reference": "Email support@example.com with your order number.",
        # right chunk retrieved, but the model invents a hotline and an instant refund
        "answer": "Call our hotline at 1-800-RETURNS and we will issue an instant refund.",
    },
    {
        "q": "Will you repair my earbuds if they stop working on their own?",
        "golden": ["doc_4"],   # answerable from the warranty chunk...
        "reference": "Electronics carry a one-year limited warranty for manufacturing defects.",
        # ...but a lexical query with no shared words retrieves nothing, so the app refuses
        "answer": "I don't know based on the provided documents.",
    },
    {
        "q": "What is the battery life of the X1 wireless earbuds?",
        "golden": [],          # genuinely out of scope: no spec in the corpus
        "reference": "Not in the documents; the system should refuse.",
        "answer": "I don't know based on the provided documents.",
    },
]


def diagnose(recall, faith, refused, out_of_scope):
    if out_of_scope:
        return "correct refusal" if refused else "FIX generation (should refuse)"
    if recall is not None and recall < 0.99:
        return "FIX retrieval"
    if faith < 0.6:
        return "FIX generation (faithfulness)"
    return "pass"


# ---------------------------------------------------------------------------
# Validating the judge: Cohen's kappa.
# A scaled metric is only trustworthy if the LLM judge agrees with humans on
# the hard calls, not just the easy ones. Hand-label a sample, have the judge
# label the same sample, and compute chance-corrected agreement. Raw agreement
# flatters: if most cases are "faithful", a judge that always says "faithful"
# scores high while having learned nothing. Kappa subtracts the agreement you
# would expect by chance, so it only rewards tracking the genuine disagreements.
#   kappa = (p_observed - p_chance) / (1 - p_chance)
# Rough reading: > ~0.6 is decent agreement; below that, do not trust the judge
# on this task yet (re-prompt it, or fall back to more human labels).
# ---------------------------------------------------------------------------
def cohen_kappa(human, judge):
    """Cohen's kappa for two binary label lists (1 = faithful, 0 = not).
    Pure stdlib, no dependencies."""
    n = len(human)
    if n == 0 or n != len(judge):
        raise ValueError("label lists must be non-empty and equal length")
    observed = sum(1 for h, j in zip(human, judge) if h == j) / n
    # chance agreement from each rater's marginal rate of saying "faithful"
    ph, pj = sum(human) / n, sum(judge) / n
    chance = ph * pj + (1 - ph) * (1 - pj)
    if chance == 1.0:                         # both raters constant and identical
        return 1.0
    return (observed - chance) / (1 - chance)


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("Offline eval over the store-policy app  (k=3 retrieval)\n")
    header = f"{'question':46}  {'recall':>6}  {'faith':>6}   verdict"
    print(header)
    print("-" * len(header))

    for case in EVAL_SET:
        retrieved = retrieve(case["q"], k=3)
        context = "\n".join(d["text"] for d in retrieved)

        recall = context_recall(case["golden"], retrieved)
        faith, _verdicts = judge_faithfulness(case["q"], context, case["answer"])
        refused = any(p in case["answer"].lower() for p in REFUSALS)
        out_of_scope = not case["golden"]

        verdict = diagnose(recall, faith, refused, out_of_scope)
        r = " n/a " if recall is None else f"{recall:5.2f}"
        print(f"{case['q'][:46]:46}  {r:>6}  {faith:6.2f}   {verdict}")

    print(
        "\nRead the table in pipeline order. Row 3 and row 4 give the SAME answer\n"
        "('I don't know'), but Context Recall tells them apart: row 4 is a correct\n"
        "refusal (nothing to retrieve), while row 3 is a retrieval MISS (the answer\n"
        "was in the corpus, lexical search just couldn't find it). That is the\n"
        "diagnosis you cannot make by eyeballing answers, and the reason to measure."
    )

    # -----------------------------------------------------------------------
    # Validating the judge against human labels on a 10-case sample.
    # 1 = faithful, 0 = not. The judge agrees on every case EXCEPT one (index
    # 6), so raw agreement is high (9/10) but kappa is lower, because most cases
    # are "faithful" and chance alone would already get a lot of them right.
    # -----------------------------------------------------------------------
    human_labels = [1, 1, 0, 1, 1, 1, 0, 1, 1, 1]
    judge_labels = [1, 1, 0, 1, 1, 1, 1, 1, 1, 1]   # disagrees on case 6
    raw = sum(1 for h, j in zip(human_labels, judge_labels) if h == j) / len(human_labels)
    kappa = cohen_kappa(human_labels, judge_labels)
    print(
        f"\nJudge validation on 10 hand-labelled cases:"
        f"\n  raw agreement = {raw:.2f}   Cohen's kappa = {kappa:.2f}"
        f"\n  (raw agreement flatters; kappa is the number to trust."
        f" Below ~0.6, re-prompt the judge or add human labels.)"
    )

# ---------------------------------------------------------------------------
# Expected output (deterministic):
#
# Offline eval over the store-policy app  (k=3 retrieval)
#
# question                                        recall   faith   verdict
# ------------------------------------------------------------------------
# How many days do I have to return an item for     1.00    1.00   pass
# How do I start a return?                          1.00    0.00   FIX generation (faithfulness)
# Will you repair my earbuds if they stop workin    0.00    1.00   FIX retrieval
# What is the battery life of the X1 wireless ea    n/a     1.00   correct refusal
#
# Read the table in pipeline order. Row 3 and row 4 give the SAME answer
# ('I don't know'), but Context Recall tells them apart: row 4 is a correct
# refusal (nothing to retrieve), while row 3 is a retrieval MISS (the answer
# was in the corpus, lexical search just couldn't find it). That is the
# diagnosis you cannot make by eyeballing answers, and the reason to measure.
#
# Judge validation on 10 hand-labelled cases:
#   raw agreement = 0.90   Cohen's kappa = 0.62
#   (raw agreement flatters; kappa is the number to trust. Below ~0.6, re-prompt the judge or add human labels.)
# ---------------------------------------------------------------------------
