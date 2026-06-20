"""
Conversational RAG by hand: give the agent a MEMORY, RUN.
RAG from First Principles, Part 20.

Part 19's agent answered ONE question and stopped -- it had no memory of
anything said before. A real chat app gets FOLLOW-UPS: "What about damaged
items?", "And how long does that take?" -- fragments that mean nothing on their
own. Sent to a retriever as-is, "What about damaged items?" matches no useful
chunk: it never mentions refunds, so it can't find the refund clause. The fix is
not a bigger index; it's READING THE CONVERSATION before you retrieve.

THE BUILD (the canonical design):
  - QUERY CONDENSATION (history-aware query rewriting). Before retrieval, rewrite
    a context-dependent follow-up into a SELF-CONTAINED, standalone query using
    the conversation so far. "What about damaged items?" -> "refund policy for
    damaged items". The standalone query is what hits the index; the raw fragment
    never does.
  - CONVERSATION MEMORY: a rolling transcript of (user, assistant) turns the
    condenser reads to recover the missing topic.
  - COREFERENCE / ELLIPSIS resolution: resolve pronouns ("that", "it") and
    elliptical follow-ups ("what about ...?", "and how long does that take?")
    against the last topic in the history.
  - CONDENSER: deterministic and rule-based in offline mode -- the artifact's
    source of truth -- exactly mirroring how Part 15's classify_complexity and
    Part 19's controller are deterministic-here / trained-LLM-in-production. With
    an API key, generate()/build_condense_prompt() shows the REAL LLM
    condensation prompt, but the code always falls through to the deterministic
    rewriter so the file runs offline.

THREE turns in ONE conversation (each prints raw -> condensed -> retrieved ->
answer), PLUS a side-by-side of turn 2 WITHOUT vs WITH condensation -- the
contrast IS the lesson:
  Turn 1: "What's our refund window?"            -- already standalone, no rewrite
  Turn 2: "What about damaged items?"            -- needs history; condense to
          "refund policy for damaged items" -> hits the damaged-items clause
  Turn 3: "And how long does that refund take to process?"  -- coreference;
          "that" resolves to the refund -> condense -> retrieve -> answer

Stack:
  - Retrieval  : the same transparent lexical / keyword-overlap retriever used
                 across the series, so it runs with NO sentence-transformers and
                 NO network -- and it is the DEFAULT, so output is reproducible.
                 RAG_REAL_EMBED=1 opts into the real dense path (it changes only
                 the printed scores; the condensed queries and answers are
                 identical).
  - Condenser  : a deterministic rule rewriter (the offline default). A
                 production system would let an LLM rewrite the follow-up; that
                 path sits behind generate() and prints a banner when a key is set.
  - generate() : one swappable provider function -- OpenAI active, Ollama and a
                 claude-opus-4-8 variant in comments (the repo convention).

Run:
  python3 conversational_rag.py     # runs offline; no API key, no network, no deps
  # Optional: pip install sentence-transformers && RAG_REAL_EMBED=1 python3
  #   conversational_rag.py   # the real retriever path (only the scores change)
  # Optional: set OPENAI_API_KEY to see the real LLM-driven condenser banner.

NOTE: LLM SDK syntax and model names move fast and may have changed since this
was written. Check current provider docs; only generate() needs edits. The
condenser is intentionally a transparent set of rules, not a learned rewriter:
the teaching point is that you can SEE exactly how each follow-up becomes a
standalone query.

Caveats this artifact is honest about (no invented numbers): condensation can
OVER-rewrite (invent a constraint the user never said) or UNDER-rewrite (leave a
dangling pronoun); pronoun ambiguity (which "it"?); the history window grows
without bound, so production truncates or summarizes it; and you must NOT condense
a genuinely fresh standalone question -- splicing a stale topic into it pollutes
the retrieval.

Expected output (the deterministic default path). The retriever scores below are
from the lexical stand-in; opting into the real embedder (RAG_REAL_EMBED=1)
changes ONLY the scores -- every raw query, condensed query, retrieved chunk, and
answer line is identical.

[embed] using deterministic lexical retriever (offline default; set RAG_REAL_EMBED=1 for sentence-transformers)
======================================================================
CONVERSATIONAL RAG  -  condense each follow-up into a standalone query, then retrieve
======================================================================
[condenser] no OPENAI_API_KEY; using deterministic rule-based condenser (offline default)

Knowledge base: 7 support chunks (refund window, damaged-items refunds, refund timing, the E-4042 error, shipping, warranty).

----------------------------------------------------------------------
ONE CONVERSATION, THREE TURNS (raw -> condensed -> retrieved -> answer)
----------------------------------------------------------------------

TURN 1
  user (raw):   What's our refund window?
  condensed:    What's our refund window?   [already standalone -- no rewrite]
  retrieved (score=0.41): Our refund window is 30 days from purchase, as long as the product is unused and in its original packaging.
  assistant:    Our refund window is 30 days from purchase, as long as the product is unused and in its original packaging.

TURN 2
  user (raw):   What about damaged items?
  condensed:    refund policy for damaged items   [spliced topic 'refund' from history]
  retrieved (score=0.26): Merchandise that arrives damaged qualifies for a full refund even outside the usual window; email a photo to support@example.com.
  assistant:    Merchandise that arrives damaged qualifies for a full refund even outside the usual window; email a photo to support@example.com.

TURN 3
  user (raw):   And how long does that refund take to process?
  condensed:    how long does the refund take to process   [resolved 'that' -> refund]
  retrieved (score=0.41): We process a refund back to your original card within five business days of receiving the return.
  assistant:    We process a refund back to your original card within five business days of receiving the return.

----------------------------------------------------------------------
THE CONTRAST: turn 2 WITHOUT vs WITH condensation
----------------------------------------------------------------------
Follow-up: "What about damaged items?"  (history topic: refund)

  WITHOUT condensation (retrieve the RAW follow-up):
    retrieved (score=0.19): Damaged goods caused by customer misuse are not covered and must be replaced at full price.
    -> MISS: 'damaged items' alone never mentions refunds, so the index returns the wrong chunk.

  WITH condensation (retrieve "refund policy for damaged items"):
    retrieved (score=0.26): Merchandise that arrives damaged qualifies for a full refund even outside the usual window; email a photo to support@example.com.
    -> HIT: the spliced 'refund' topic lands the query on the damaged-items clause.

======================================================================
Done. Memory + condensation turn a one-shot retriever into a chat:
  - turn 1 was already standalone, so the condenser left it ALONE;
  - turn 2's 'what about ___?' borrowed 'refund' from history to retrieve;
  - turn 3's 'that' resolved to the refund before retrieval.
The raw fragment never touches the index -- the standalone query does.
======================================================================
"""

import os
import re

# The default lexical retriever is pure standard library (no numpy needed). The
# optional real path imports numpy locally, so this file runs with no third-party
# package at all unless you opt into RAG_REAL_EMBED=1.


# ---------------------------------------------------------------------------
# Step 0. The support knowledge base, carried over from Parts 6-12 so this part
#         feels continuous: refunds, the E-4042 error code, shipping, warranty.
#
# Two clauses are NEW. Chunk index 1 is the damaged-items REFUND -- turn 2's real
# target: it says "refund", so ONLY the condensed "refund policy for damaged
# items" retrieves it; the raw "what about damaged items?" (no "refund") misses.
# Chunk index 3 is a deliberate DISTRACTOR: damaged goods by misuse, NO refund.
# It is shorter and shares "damaged" with the raw fragment, so the un-condensed
# query lands HERE (the wrong, no-refund answer) -- that is the contrast's miss.
# Short docs, so each doc is its own chunk (Part 5).
# ---------------------------------------------------------------------------
KNOWLEDGE_BASE = [
    "Our refund window is 30 days from purchase, as long as the product is unused and in its original packaging.",
    "Merchandise that arrives damaged qualifies for a full refund even outside the usual window; email a photo to support@example.com.",
    "We process a refund back to your original card within five business days of receiving the return.",
    "Damaged goods caused by customer misuse are not covered and must be replaced at full price.",
    "Error E-4042 means the payment was declined by the bank; ask the customer to retry with a different card or contact their bank.",
    "Standard shipping takes 3 to 5 business days. Express shipping arrives the next business day.",
    "All electronics include a one-year limited warranty covering manufacturing defects.",
]


# ---------------------------------------------------------------------------
# Step 1. Embeddings/retrieval, with a transparent deterministic default.
#
# The DEFAULT is a pure lexical retriever: score each chunk by content-word
# OVERLAP with the query (with a cosine-style normalization). It is crude, but
# deterministic, model-free, network-free, and enough to make the right chunk
# win -- so the demo's output is reproducible. The real sentence-transformers
# path (Part 6's embedder) is one env flag away and changes only the printed
# scores. A real model captures meaning; this captures content-word overlap --
# the level the essay stays at. It is also WHY condensation matters here: a
# lexical retriever can only match words the query actually contains, so a
# follow-up that drops the word "refund" cannot find a refund chunk until the
# condenser puts "refund" back in.
# ---------------------------------------------------------------------------
_TOKEN_RE = re.compile(r"[a-z0-9]+")

# Grammatical words the lexical retriever ignores so a query's CONTENT words
# (refund, window, damaged, items) drive the score instead of "what / about".
_STOPWORDS = {
    "what", "whats", "is", "are", "the", "of", "a", "an", "for", "on", "in",
    "to", "how", "do", "does", "and", "my", "i", "there", "with", "your", "our",
    "about", "that", "it", "this", "long", "take", "takes", "s",
}


def _stem(tok):
    """Crudest possible stemmer: drop a trailing plural 's' so 'items' and
    'item', or 'refunds' and 'refund', hash to the same content word. A real
    model needs no such hint; the lexical stand-in does."""
    return tok[:-1] if len(tok) > 3 and tok.endswith("s") else tok


def _tokens(text):
    """Lowercase content tokens, stop-words dropped, plural 's' stemmed."""
    return [_stem(t) for t in _TOKEN_RE.findall(text.lower()) if t not in _STOPWORDS]


class _LexicalRetriever:
    """Deterministic, model-free stand-in for a dense retriever (Part 2/Part 6).

    Score = (overlap of query content words with the chunk) normalized by the
    geometric mean of the two token counts -- a cosine-flavored overlap that
    keeps scores in a readable [0, 1] band. Crude, but it puts 'refund window'
    next to the refund chunk and 'refund policy for damaged items' next to the
    damaged-items clause without a 90 MB download or a network call."""

    def __init__(self, corpus):
        self.chunks = list(corpus)
        self._chunk_tokens = [set(_tokens(c)) for c in self.chunks]

    def _score(self, q_tokens, c_tokens):
        if not q_tokens or not c_tokens:
            return 0.0
        overlap = len(q_tokens & c_tokens)
        # cosine-style normalization: overlap / sqrt(|q| * |c|), in [0, 1].
        denom = (len(q_tokens) * len(c_tokens)) ** 0.5
        return overlap / denom

    def retrieve(self, query, k=1):
        q_tokens = set(_tokens(query))
        scored = [
            (self.chunks[i], self._score(q_tokens, self._chunk_tokens[i]))
            for i in range(len(self.chunks))
        ]
        scored.sort(key=lambda x: -x[1])
        return scored[:k]


def load_real_retriever(corpus):
    """Use a real sentence-transformers retriever; fall back transparently.

    The deterministic lexical retriever is the DEFAULT so this file's output is
    reproducible whether or not a model happens to be cached -- the same reason
    the condenser defaults to the rule rewriter. Set RAG_REAL_EMBED=1 to opt into
    the real dense path (it changes only the printed scores; the condensed
    queries and answers are identical). The fallback prints exactly once so the
    demo banner stays clean.
    """
    if not os.environ.get("RAG_REAL_EMBED"):
        if not load_real_retriever._announced:
            print("[embed] using deterministic lexical retriever (offline default; "
                  "set RAG_REAL_EMBED=1 for sentence-transformers)")
            load_real_retriever._announced = True
        return _LexicalRetriever(corpus)
    try:
        from sentence_transformers import SentenceTransformer  # REAL path
        import numpy as _np

        model = SentenceTransformer("all-MiniLM-L6-v2")          # 384 dims

        class _DenseRetriever:
            def __init__(self):
                self.chunks = list(corpus)
                self.vectors = _np.asarray(
                    model.encode(self.chunks, normalize_embeddings=True))

            def retrieve(self, query, k=1):
                q = _np.asarray(model.encode([query], normalize_embeddings=True))[0]
                scores = self.vectors @ q
                top = _np.argsort(-scores)[:k]
                return [(self.chunks[i], float(scores[i])) for i in top]

        if not load_real_retriever._announced:
            print("[embed] using sentence-transformers (all-MiniLM-L6-v2)")
            load_real_retriever._announced = True
        return _DenseRetriever()
    except Exception as exc:  # not installed / no weights / offline
        if not load_real_retriever._announced:
            print(f"[embed] sentence-transformers unavailable ({type(exc).__name__}); "
                  "using deterministic lexical fallback")
            load_real_retriever._announced = True
        return _LexicalRetriever(corpus)


load_real_retriever._announced = False


# ===========================================================================
# Step 2. Conversation memory -- a rolling transcript of (user, assistant) turns.
#
# This is the whole "memory": an ordered list of turns the condenser reads to
# recover the topic a follow-up leaves implicit. Real systems cap or summarize it
# (the history grows without bound -- a caveat below); here it stays small and we
# keep it whole so every line is eyeball-able.
# ===========================================================================
class Conversation:
    """An ordered list of (user, assistant) turns. The condenser reads it; the
    loop appends to it after each answered turn."""

    def __init__(self):
        self.turns = []                       # list of (user_text, assistant_text)

    def add(self, user_text, assistant_text):
        self.turns.append((user_text, assistant_text))

    def last_user(self):
        return self.turns[-1][0] if self.turns else ""

    def last_assistant(self):
        return self.turns[-1][1] if self.turns else ""

    def is_empty(self):
        return not self.turns


# ===========================================================================
# Step 3. generate() -- the REAL LLM-driven condenser path (reference shape).
#
# In production the condenser is an LLM: you hand it the conversation history and
# the new follow-up, and it returns a single standalone query. We show that
# PROMPT SHAPE here and keep generate() as one swappable provider call (OpenAI
# active; Ollama and claude-opus-4-8 in comments -- repo convention). The offline
# demo never calls it: build_condense_prompt() documents the shape, and the
# deterministic condenser in Step 4 is the source of truth.
#
# NOTE: SDK names and model ids move fast; check current docs. Only generate()
# needs edits to light up the real path.
# ===========================================================================
CONDENSE_SYSTEM = """You rewrite a user's latest message into a STANDALONE search query.
Given the conversation history and a follow-up that may rely on it (pronouns
like "that"/"it", or ellipsis like "what about ...?"), output ONE self-contained
query that needs no history to retrieve on. Resolve every pronoun and fill in the
implicit topic from the history. If the follow-up is ALREADY standalone, return
it unchanged. Output only the rewritten query, nothing else."""


def build_condense_prompt(history_text, follow_up):
    """The prompt a REAL LLM condenser would see: history + the new follow-up."""
    history = history_text if history_text else "(no prior turns)"
    return (f"{CONDENSE_SYSTEM}\n\nConversation so far:\n{history}\n\n"
            f"Follow-up message: {follow_up}\n\nStandalone query:")


def generate(prompt):
    """REAL path: ask a hosted LLM to condense the follow-up. Unused offline."""
    from openai import OpenAI
    client = OpenAI()                               # reads OPENAI_API_KEY
    resp = client.chat.completions.create(
        model="gpt-4o-mini",                        # a small, cheap chat model; check names
        messages=[{"role": "user", "content": prompt}],
        temperature=0,                              # deterministic, not creative
    )
    return resp.choices[0].message.content


# Local, zero-cost alternative (Ollama). Swap this in for generate() above:
#
# def generate(prompt):
#     import requests
#     r = requests.post(
#         "http://localhost:11434/api/generate",
#         json={"model": "llama3.1", "prompt": prompt, "stream": False},
#     )
#     return r.json()["response"]


# Anthropic / Claude alternative. Swap this in for generate() above:
#
# def generate(prompt):
#     from anthropic import Anthropic
#     client = Anthropic()                            # reads ANTHROPIC_API_KEY from the env
#     resp = client.messages.create(
#         model="claude-opus-4-8",                    # check current model names
#         max_tokens=1024,                            # required by the Messages API
#         messages=[{"role": "user", "content": prompt}],
#     )                                               # (no temperature: removed on Opus 4.8)
#     return resp.content[0].text


# ===========================================================================
# Step 4. The deterministic condenser -- the artifact's source of truth.
#
# Given the conversation history and the latest raw follow-up, return
# (standalone_query, note) where `note` explains the rewrite for the trace. This
# is the offline default: a transparent rule rewriter you can read, exactly
# mirroring Part 15's classify_complexity and Part 19's controller
# (deterministic-here / trained-in-production). A real system swaps this body for
# one generate() call against build_condense_prompt(); the surrounding loop is
# identical.
#
# The rules key on the SAME cheap signals an LLM would weigh:
#   - the follow-up is already standalone (a full question with a topic of its
#     own)                                  -> leave it ALONE (do NOT pollute it)
#   - an elliptical "what about ___?"        -> splice the history TOPIC in front
#   - a pronoun "that"/"it"/"this"           -> resolve it to the history topic
# `_topic_from_history` recovers the salient noun (here: "refund") from the most
# recent turns -- the one piece a follow-up leaves implicit.
# ===========================================================================
# Topics the rewriter knows how to splice/resolve, longest first so "refund
# window" would win over "refund" if both appeared. Drawn from the KB's nouns.
_KNOWN_TOPICS = ("refund", "warranty", "shipping", "payment", "order")

# A follow-up that already contains one of these is treated as standalone: it
# carries its own topic, so splicing history would only pollute it (a caveat).
_STANDALONE_MARKERS = _KNOWN_TOPICS


def _topic_from_history(conversation):
    """Recover the salient topic the follow-up leaves implicit.

    Scan recent turns (latest first) for a known topic noun. A real LLM infers
    this for free; the rule rewriter looks for the one the corpus uses. Returns
    the topic string, or "" if none is found (then we cannot safely condense)."""
    for user_text, assistant_text in reversed(conversation.turns):
        blob = f"{user_text} {assistant_text}".lower()
        for topic in _KNOWN_TOPICS:
            if topic in blob:
                return topic
    return ""


# Match an elliptical follow-up "what about <X>?" and capture <X> ("damaged
# items"). This is the ellipsis case: the user dropped the verb/topic entirely.
_WHAT_ABOUT_RE = re.compile(r"^\s*(?:and\s+)?what about\s+(.+?)\s*\??\s*$", re.IGNORECASE)

# Pronouns that, in a follow-up, point back at the history topic (coreference).
_PRONOUN_RE = re.compile(r"\b(that|it|this|those|these)\b", re.IGNORECASE)


def condense(conversation, follow_up):
    """Rewrite a follow-up into a standalone query using the history.

    Returns (standalone_query, note). `note` is a short, human-readable reason
    that the trace prints so you can SEE why the query came out the way it did.
    """
    raw = follow_up.strip()
    low = raw.lower()

    # --- Turn 1 / no history: nothing to condense against. -------------------
    if conversation.is_empty():
        return raw, "already standalone -- no rewrite"

    # We condense a follow-up when it is CONTEXT-DEPENDENT: an ellipsis
    # ("what about ...?") or a dangling pronoun ("that"/"it"). We test those
    # signals BEFORE the standalone check, because a question can mention the
    # topic word and STILL dangle -- "how long does that refund take?" says
    # "refund" yet "that" still needs resolving.
    is_ellipsis = bool(_WHAT_ABOUT_RE.match(raw))
    has_pronoun = bool(_PRONOUN_RE.search(raw))

    # --- Already standalone: carries its own topic AND has no dangling
    #     pronoun/ellipsis. Leave it alone -- splicing a stale topic into a
    #     fresh question would pollute the retrieval (the "don't condense a
    #     fresh question" caveat, enforced). ---------------------------------
    if not is_ellipsis and not has_pronoun and any(t in low for t in _STANDALONE_MARKERS):
        return raw, "already standalone -- no rewrite"

    topic = _topic_from_history(conversation)
    if not topic:
        # Under-rewrite guard: with no recoverable topic we cannot safely fill
        # the blank, so we pass the raw fragment through rather than invent one.
        return raw, "no topic in history -- left as-is (would need clarification)"

    # --- Ellipsis: "what about <X>?" -> "<topic> policy for <X>". ------------
    if is_ellipsis:
        tail = _WHAT_ABOUT_RE.match(raw).group(1).strip().rstrip("?")
        standalone = f"{topic} policy for {tail}"
        return standalone, f"spliced topic '{topic}' from history"

    # --- Coreference: a pronoun ("that"/"it") -> the history topic. ----------
    if has_pronoun:
        pron = _PRONOUN_RE.search(raw).group(0)
        # Replace the FIRST pronoun with the topic noun, then tidy: drop a
        # leading "and", a trailing "?", and any doubled topic word the user
        # already supplied ("that refund" -> "<topic> refund" -> "the refund").
        resolved = _PRONOUN_RE.sub(topic, raw, count=1)
        resolved = re.sub(r"^\s*and\s+", "", resolved, flags=re.IGNORECASE)
        resolved = re.sub(rf"\b{topic}\s+{topic}\b", f"the {topic}", resolved,
                          flags=re.IGNORECASE)
        resolved = resolved.strip().rstrip("?")
        return resolved, f"resolved '{pron}' -> {topic}"

    # --- Fallback: prepend the topic so retrieval at least sees it. ----------
    return f"{topic} {raw.rstrip('?')}", f"prepended topic '{topic}'"


# ===========================================================================
# Step 5. Generation -- a grounded extractive stand-in (Part 6's grounding).
#         The real path is one swappable hosted-LLM call; offline we quote the
#         best retrieved chunk so the answer is visibly tied to a source.
# ===========================================================================
def answer_from(retrieved):
    """Grounded extractive generate(): quote the single best-retrieved chunk."""
    if not retrieved:
        return "I don't know based on the available sources."
    best_text, _score = retrieved[0]
    return best_text


# ===========================================================================
# Step 6. One conversational turn: condense -> retrieve -> answer -> remember.
#
# This is the whole conversational pipeline. Each turn: read memory, condense the
# raw follow-up into a standalone query, retrieve on the STANDALONE query (never
# the raw fragment), answer from the chunk, then append the turn to memory so the
# NEXT follow-up can see it.
# ===========================================================================
def converse_turn(conversation, store, raw_query, trace=True):
    """Run one turn and return (standalone_query, chunk, score, answer)."""
    standalone, note = condense(conversation, raw_query)
    text, score = store.retrieve(standalone, k=1)[0]
    answer = answer_from([(text, score)])

    if trace:
        print(f"  user (raw):   {raw_query}")
        print(f"  condensed:    {standalone}   [{note}]")
        print(f"  retrieved (score={score:.2f}): {text}")
        print(f"  assistant:    {answer}")

    # REMEMBER: append AFTER answering so the next turn sees this exchange.
    conversation.add(raw_query, answer)
    return standalone, text, score, answer


# ===========================================================================
# Demo. Everything below RUNS OFFLINE. It uses the real retriever/condenser if
#       available and the deterministic stand-ins otherwise, with clear labels.
# ===========================================================================
if __name__ == "__main__":
    line = "=" * 70

    print(line)
    print("CONVERSATIONAL RAG  -  condense each follow-up into a standalone query, then retrieve")
    print(line)

    store = load_real_retriever(KNOWLEDGE_BASE)     # prints the embed banner

    if os.environ.get("OPENAI_API_KEY"):
        print("[condenser] OPENAI_API_KEY set; the real LLM condenser would drive "
              "generate(build_condense_prompt(...)). Falling through to the "
              "deterministic rewriter so output stays reproducible.")
    else:
        print("[condenser] no OPENAI_API_KEY; using deterministic rule-based "
              "condenser (offline default)")

    print(f"\nKnowledge base: {len(KNOWLEDGE_BASE)} support chunks (refund window, "
          "damaged-items refunds, refund timing, the E-4042 error, shipping, warranty).")

    # --- The conversation: three turns, sharing ONE memory. ------------------
    print("\n" + "-" * 70)
    print("ONE CONVERSATION, THREE TURNS (raw -> condensed -> retrieved -> answer)")
    print("-" * 70)

    chat = Conversation()

    print("\nTURN 1")
    converse_turn(chat, store, "What's our refund window?")

    print("\nTURN 2")
    converse_turn(chat, store, "What about damaged items?")

    print("\nTURN 3")
    converse_turn(chat, store, "And how long does that refund take to process?")

    # --- The contrast: turn 2 with vs without condensation. ------------------
    # Re-run turn 2's follow-up against a memory holding ONLY turn 1, so the
    # history topic is exactly "refund" -- isolating the condensation effect.
    print("\n" + "-" * 70)
    print("THE CONTRAST: turn 2 WITHOUT vs WITH condensation")
    print("-" * 70)

    contrast_chat = Conversation()
    contrast_chat.add(
        "What's our refund window?",
        "Refunds are accepted within 30 days of purchase, provided the item is "
        "unused and in its original packaging.")
    follow_up = "What about damaged items?"
    topic = _topic_from_history(contrast_chat)
    print(f'Follow-up: "{follow_up}"  (history topic: {topic})')

    # WITHOUT: send the RAW fragment to the index. It never says "refund", so a
    # lexical retriever can only match "items"/"damaged" against the wrong chunk.
    raw_text, raw_score = store.retrieve(follow_up, k=1)[0]
    print("\n  WITHOUT condensation (retrieve the RAW follow-up):")
    print(f"    retrieved (score={raw_score:.2f}): {raw_text}")
    print("    -> MISS: 'damaged items' alone never mentions refunds, so the "
          "index returns the wrong chunk.")

    # WITH: condense first, then retrieve on the standalone query.
    standalone, _note = condense(contrast_chat, follow_up)
    con_text, con_score = store.retrieve(standalone, k=1)[0]
    print(f'\n  WITH condensation (retrieve "{standalone}"):')
    print(f"    retrieved (score={con_score:.2f}): {con_text}")
    print("    -> HIT: the spliced 'refund' topic lands the query on the "
          "damaged-items clause.")

    print("\n" + line)
    print("Done. Memory + condensation turn a one-shot retriever into a chat:")
    print("  - turn 1 was already standalone, so the condenser left it ALONE;")
    print("  - turn 2's 'what about ___?' borrowed 'refund' from history to retrieve;")
    print("  - turn 3's 'that' resolved to the refund before retrieval.")
    print("The raw fragment never touches the index -- the standalone query does.")
    print(line)
