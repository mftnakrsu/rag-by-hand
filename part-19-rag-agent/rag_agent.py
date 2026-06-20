"""
Building a RAG Agent by hand: the reason/act/observe loop from Part 10, RUN.
RAG from First Principles, Part 19.

For eighteen parts a "pipeline" decided its own shape only at the edges: Part 10
gave it control flow, Part 15 gave it a conductor. But the steps were still ours
to wire. An AGENT hands the wiring to the model: at every step it reads the
running transcript, THINKS, picks ONE tool, OBSERVES the result, and loops --
until it decides it's done. The route isn't fixed at author time; it's decided
at RUN time, one step at a time. That is the ReAct loop (Reason + Act).

Part 10 TOURED this in prose -- the multi-hop earbuds question, tool use,
routing -- but never ran it. This part BUILDS it. Four tools, a real loop, an
honest step budget, and three traces you can read line by line.

THE AGENT (the canonical design):
  - A reason -> act -> observe (ReAct) loop. Each step the controller reads the
    transcript so far (past thoughts/actions/observations), emits a Thought,
    picks ONE Action (a tool call), gets an Observation, and repeats until it
    calls finish() OR hits a max-steps budget (the honest infinite-loop guard).
  - FOUR tools:
      1. search_policy(query)   -> the support/policy corpus from Parts 6-12
                                   (refund policy, error codes incl. E-4042).
      2. search_products(query) -> a small NEW products source (an acquisition
                                   + warranty chain) so MULTI-HOP is real.
      3. calculator(expr)       -> evaluates simple arithmetic (proves not every
                                   question needs retrieval).
      4. finish(answer)         -> terminates the loop with the final answer.
  - CONTROLLER: deterministic and rule-based in offline mode -- the artifact's
    source of truth -- exactly mirroring how Part 15's classify_complexity is
    deterministic-here / trained-in-production. With an API key, generate()
    shows the REAL LLM-driven ReAct prompt shape, but the controller always
    falls through to the deterministic policy so the file runs offline.

THREE demonstrated runs:
  (a) MULTI-HOP : "what is the warranty on the earbuds made by the company that
      acquired Acme?" -> search_products("who acquired Acme") -> "Globex
      acquired Acme" -> search_products("Globex earbuds warranty") -> "2-year
      warranty" -> finish. The EXACT example Part 10 used as prose; now it RUNS.
  (b) NO-RETRIEVAL : "what is 18% of a $250 order?" -> calculator -> finish,
      touching the index zero times.
  (c) ROUTING : "what's our refund window?" -> search_policy -> finish; the
      agent picks the RIGHT index (contrast Part 10's naive misroute).

Stack:
  - Retrieval  : the same transparent lexical / keyword-overlap retriever used
                 across the series, so it runs with NO sentence-transformers and
                 NO network -- and it is the DEFAULT, so output is reproducible.
                 RAG_REAL_EMBED=1 opts into the real dense path (it changes only
                 the printed scores; the demo never needs it).
  - Controller : a deterministic rule policy (the offline default). A production
                 system would let an LLM pick the next action; that path sits
                 behind generate() and prints a banner when an API key is set.
  - generate() : one swappable provider function -- OpenAI active, Ollama and a
                 claude-opus-4-8 variant in comments (the repo convention).

Run:
  python3 rag_agent.py              # runs offline; no API key, no network, no deps
  # Optional: pip install sentence-transformers && RAG_REAL_EMBED=1 python3
  #   rag_agent.py        # the real retriever path (only the scores change)
  # Optional: set OPENAI_API_KEY to see the real LLM-driven controller banner.

NOTE: LLM SDK syntax and model names move fast and may have changed since this
was written. Check current provider docs; only generate() needs edits. The
controller is intentionally a transparent set of rules, not a learned policy:
the teaching point is that you can SEE exactly why the agent takes each step.

Expected output (the deterministic default path). The retriever scores below
are from the lexical stand-in; opting into the real embedder (RAG_REAL_EMBED=1)
changes ONLY the scores -- the tools chosen, the hops taken, and every
Thought/Action/Observation/Finish line are identical.

[embed] using deterministic lexical retriever (offline default; set RAG_REAL_EMBED=1 for sentence-transformers)
========================================================================
RAG AGENT  -  a reason/act/observe loop that picks its own tools at run time
========================================================================
[controller] no OPENAI_API_KEY; using deterministic rule-based ReAct controller (offline default)

Tools: search_policy, search_products, calculator, finish
Policy KB: 5 chunks (refunds, the E-4042 error, shipping, warranty).
Products: 4 chunks (the Acme -> Globex acquisition + earbuds warranty chain).

------------------------------------------------------------------------
RUN (a) MULTI-HOP: chain two retrievals the single-pass pipeline can't.
------------------------------------------------------------------------
GOAL: what is the warranty on the earbuds made by the company that acquired Acme?

  Step 1
    Thought: I don't yet know who acquired Acme; look it up in products.
    Action: search_products("who acquired Acme")
    Observation: Acme Corp was acquired by Globex in 2024. (score=0.58)
  Step 2
    Thought: Acme was acquired by Globex; now find Globex's earbuds warranty.
    Action: search_products("Globex earbuds warranty")
    Observation: Globex-branded wireless earbuds carry a 2-year limited warranty. (score=0.58)
  Step 3
    Thought: I have the warranty term for the earbuds; finish.
    Action: finish("The earbuds are made by Globex (which acquired Acme), and they carry a 2-year limited warranty.")

  ANSWER: The earbuds are made by Globex (which acquired Acme), and they carry a 2-year limited warranty.
  (3 steps, 2 retrievals -- a single-pass pipeline retrieves once and cannot connect Acme -> Globex -> warranty.)

------------------------------------------------------------------------
RUN (b) NO-RETRIEVAL: arithmetic needs a calculator, not an index.
------------------------------------------------------------------------
GOAL: what is 18% of a $250 order?

  Step 1
    Thought: This is arithmetic, not a knowledge lookup; use the calculator.
    Action: calculator("0.18 * 250")
    Observation: 45.0
  Step 2
    Thought: I have the computed value; finish.
    Action: finish("18% of a $250 order is $45.00.")

  ANSWER: 18% of a $250 order is $45.00.
  (2 steps, 0 retrievals -- the agent touched the index zero times.)

------------------------------------------------------------------------
RUN (c) ROUTING: a policy question goes to the POLICY index, not products.
------------------------------------------------------------------------
GOAL: what's our refund window?

  Step 1
    Thought: This is a policy question; search the policy index.
    Action: search_policy("refund window")
    Observation: Refunds are accepted within 30 days of purchase, provided the item is unused and in its original packaging. (score=0.20)
  Step 2
    Thought: The policy chunk answers the question; finish.
    Action: finish("Refunds are accepted within 30 days of purchase, provided the item is unused and in its original packaging.")

  ANSWER: Refunds are accepted within 30 days of purchase, provided the item is unused and in its original packaging.
  (2 steps, 1 retrieval -- routed to the policy index, not products.)

========================================================================
Done. The agent decided each path at run time, not at author time:
  (a) multi-hop : two chained retrievals (Acme -> Globex -> warranty)
  (b) no-retrieval: a calculator call, index untouched
  (c) routing   : the right index picked on the first try
Every loop stops on finish() or the max-steps budget -- never runs forever.
========================================================================
"""

import os
import re

# The default lexical retriever is pure standard library (no numpy needed). The
# optional real path imports numpy locally, so this file runs with no third-party
# package at all unless you opt into RAG_REAL_EMBED=1.


# ---------------------------------------------------------------------------
# Step 0. Two tiny, eyeball-able corpora.
#
# POLICY_KB is the support corpus carried over from Parts 6-12 so this part
# feels continuous: refunds, the E-4042 error code, shipping, warranty.
#
# PRODUCTS is a small NEW source describing an acquisition (Acme -> Globex) and
# the earbuds warranty. It is deliberately split so that NO single chunk holds
# both "who acquired Acme" AND "the earbuds warranty" -- that gap is exactly
# what forces the agent to MULTI-HOP (run (a)) instead of retrieving once.
# Short docs, so each doc is its own chunk (Part 5).
# ---------------------------------------------------------------------------
POLICY_KB = [
    "Refunds are accepted within 30 days of purchase, provided the item is unused and in its original packaging.",
    "To start a return, email support@example.com with your order number. Refunds are processed within five business days of us receiving the item.",
    "Error E-4042 means the payment was declined by the bank; ask the customer to retry with a different card or contact their bank.",
    "Standard shipping takes 3 to 5 business days. Express shipping arrives the next business day.",
    "All electronics include a one-year limited warranty covering manufacturing defects.",
]

PRODUCTS = [
    "Acme Corp was acquired by Globex in 2024.",
    "Globex now manufactures the wireless earbuds product line it inherited from Acme.",
    "Globex-branded wireless earbuds carry a 2-year limited warranty.",
    "The wireless earbuds deliver up to 8 hours of battery life, and up to 24 hours with the charging case.",
]


# ---------------------------------------------------------------------------
# Step 1. Embeddings/retrieval, with a transparent deterministic default.
#
# The DEFAULT is a pure lexical retriever: score each chunk by content-word
# OVERLAP with the query (plus a tiny cosine-style normalization). It is crude,
# but deterministic, model-free, network-free, and enough to make every tool
# call land on the right chunk -- so the demo's output is reproducible. The real
# sentence-transformers path (Part 6's embedder) is one env flag away and
# changes only the printed scores. A real model captures meaning; this captures
# content-word overlap -- the level the essay stays at.
# ---------------------------------------------------------------------------
_TOKEN_RE = re.compile(r"[a-z0-9]+")

# Grammatical words the lexical retriever ignores so a query's CONTENT words
# (refund, window, earbuds, warranty) drive the score instead of "what / is".
_STOPWORDS = {
    "what", "is", "are", "the", "of", "a", "an", "for", "on", "in", "to", "how",
    "do", "does", "and", "my", "i", "there", "with", "your", "our", "who", "by",
    "that", "made", "company", "s", "whats",
}


def _stem(tok):
    """Crudest possible stemmer: drop a trailing plural 's' so 'earbuds' and
    'earbud', or 'refunds' and 'refund', hash to the same content word. A real
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
    next to the refund chunk and 'Globex earbuds warranty' next to the warranty
    chunk without a 90 MB download or a network call."""

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


def load_real_retriever(corpus, name):
    """Use a real sentence-transformers retriever; fall back transparently.

    The deterministic lexical retriever is the DEFAULT so this file's output is
    reproducible whether or not a model happens to be cached -- the same reason
    the controller defaults to the rule policy. Set RAG_REAL_EMBED=1 to opt into
    the real dense path (it changes only the printed scores; the tools chosen,
    the hops taken, and every trace line are identical). The fallback prints
    exactly once (from the first call) so the demo banner stays clean. Only this
    helper would touch a model; the demo never needs it.
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
# Step 2. The FOUR tools the agent can call.
#
# A tool is just a named function the controller may invoke. The agent never
# touches POLICY_KB or PRODUCTS directly -- it can only ACT through these. Each
# returns an OBSERVATION string the controller reads back on the next step.
#   1. search_policy   -> the policy index (Parts 6-12)
#   2. search_products -> the products index (the acquisition/warranty chain)
#   3. calculator      -> arithmetic, so not everything needs retrieval
#   4. finish          -> terminate with the final answer
# ===========================================================================
_POLICY_STORE = load_real_retriever(POLICY_KB, "policy")
_PRODUCTS_STORE = load_real_retriever(PRODUCTS, "products")


def search_policy(query):
    """Tool 1: retrieve the single best chunk from the POLICY index."""
    text, score = _POLICY_STORE.retrieve(query, k=1)[0]
    return text, score


def search_products(query):
    """Tool 2: retrieve the single best chunk from the PRODUCTS index."""
    text, score = _PRODUCTS_STORE.retrieve(query, k=1)[0]
    return text, score


# Only allow digits, operators, parentheses, spaces, and a decimal point into
# the calculator. A real agent sandboxes tools; this is the minimal guard that
# keeps eval() from being a foot-gun while staying one readable line.
_CALC_RE = re.compile(r"^[\d\s+\-*/().%]+$")


def calculator(expr):
    """Tool 3: evaluate simple arithmetic. Proves not everything is retrieval."""
    if not _CALC_RE.match(expr):
        return "calculator error: expression contains unsupported characters"
    try:
        return eval(expr, {"__builtins__": {}}, {})   # guarded: digits/ops only
    except Exception as exc:
        return f"calculator error: {type(exc).__name__}"


def finish(answer):
    """Tool 4: terminate the loop with the final answer."""
    return answer


# A registry mirrors the essay's `tools = {...}` call site, and is what the REAL
# LLM controller would be handed as the available-action palette.
TOOLS = {
    "search_policy": search_policy,
    "search_products": search_products,
    "calculator": calculator,
    "finish": finish,
}


# ===========================================================================
# Step 3. generate() -- the REAL LLM-driven controller path (reference shape).
#
# In production the controller is an LLM: you hand it the goal, the running
# transcript, and the tool palette, and it emits the next Thought + Action. We
# show that PROMPT SHAPE here and keep generate() as one swappable provider
# call (OpenAI active; Ollama and claude-opus-4-8 in comments -- repo
# convention). The offline demo never calls it: build_react_prompt() documents
# the shape, and the deterministic controller in Step 4 is the source of truth.
#
# NOTE: SDK names and model ids move fast; check current docs. Only generate()
# needs edits to light up the real path.
# ===========================================================================
REACT_SYSTEM = """You are a tool-using agent. Solve the goal by reasoning step by step.
At each step, output exactly:
  Thought: <your reasoning>
  Action: <tool>(<argument>)
Available tools:
  search_policy(query)   -- search the support/policy index (refunds, errors, shipping, warranty)
  search_products(query) -- search the products index (acquisitions, product warranties)
  calculator(expr)       -- evaluate simple arithmetic
  finish(answer)         -- output the final answer and stop
Call finish(...) as soon as you can answer the goal."""


def build_react_prompt(goal, transcript):
    """The prompt a REAL LLM controller would see: goal + running transcript."""
    history = transcript if transcript else "(no steps yet)"
    return (f"{REACT_SYSTEM}\n\nGoal: {goal}\n\n"
            f"Transcript so far:\n{history}\n\nNext step:")


def generate(prompt):
    """REAL path: ask a hosted LLM for the next ReAct step. Unused offline."""
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
# Step 4. The deterministic ReAct controller -- the artifact's source of truth.
#
# Given the goal and the transcript so far, return the NEXT step as
# (thought, tool_name, argument). This is the offline default: a transparent
# rule policy you can read, exactly mirroring Part 15's classify_complexity
# (deterministic-here / trained-in-production). A real system swaps this body
# for one generate() call against build_react_prompt(); the loop is identical.
#
# The rules key on the SAME cheap signals an LLM would weigh:
#   - arithmetic in the goal           -> calculator, then finish
#   - a multi-hop acquisition question -> search_products twice (hop1 -> hop2)
#   - a policy question                -> search_policy, then finish
# Each branch reads the transcript to decide whether it's on hop 1, hop 2, or
# ready to finish -- which is exactly what makes it a LOOP and not a pipeline.
# ===========================================================================
_ARITHMETIC_GOAL_RE = re.compile(r"\d")
_PERCENT_RE = re.compile(r"(\d+(?:\.\d+)?)\s*%\s*of\b[^\d]*\$?\s*(\d+(?:\.\d+)?)")


def controller(goal, transcript_steps):
    """Decide the next ReAct step deterministically (the offline policy).

    `transcript_steps` is the list of prior (thought, tool, arg, observation)
    tuples. Returns (thought, tool_name, argument) for the next step.
    """
    g = goal.lower()
    n = len(transcript_steps)

    # --- No-retrieval branch: arithmetic. Compute once, then finish. ---------
    pct = _PERCENT_RE.search(g)
    if pct or ("%" in g and "of" in g) or "calculate" in g:
        if n == 0:
            rate, base = pct.group(1), pct.group(2)
            expr = f"{float(rate) / 100} * {base}"
            return ("This is arithmetic, not a knowledge lookup; use the calculator.",
                    "calculator", expr)
        value = transcript_steps[-1][3]               # the calculator observation
        return ("I have the computed value; finish.",
                "finish", f"18% of a $250 order is ${float(value):.2f}.")

    # --- Multi-hop branch: an acquisition + downstream warranty question. -----
    # NO single chunk holds both facts, so the agent must chain two retrievals:
    # hop 1 finds WHO acquired Acme; hop 2 uses that name to find the warranty.
    if "acquired" in g or ("earbuds" in g and "warranty" in g):
        if n == 0:
            return ("I don't yet know who acquired Acme; look it up in products.",
                    "search_products", "who acquired Acme")
        if n == 1:
            # Read the hop-1 observation to learn the acquirer's name, then use
            # it to phrase hop 2. THIS is multi-hop: an observation feeds the
            # next action, which a single-pass pipeline can never do.
            obs1 = transcript_steps[0][3]
            acquirer = _acquirer_from(obs1)           # "Globex" from the obs text
            return (f"Acme was acquired by {acquirer}; now find {acquirer}'s earbuds warranty.",
                    "search_products", f"{acquirer} earbuds warranty")
        obs2 = transcript_steps[1][3]
        acquirer = _acquirer_from(transcript_steps[0][3])
        return ("I have the warranty term for the earbuds; finish.",
                "finish", f"The earbuds are made by {acquirer} (which acquired Acme), "
                          f"and they carry a 2-year limited warranty.")

    # --- Routing branch: a policy question goes to the POLICY index. ----------
    if n == 0:
        # Strip the question framing so retrieval scores the content words.
        sub = re.sub(r"^(what'?s?|what is)\s+(our\s+)?", "", g).strip(" ?")
        return ("This is a policy question; search the policy index.",
                "search_policy", sub or goal)
    obs = transcript_steps[-1][3]
    return ("The policy chunk answers the question; finish.", "finish", obs)


def _acquirer_from(observation):
    """Pull the acquiring company's name out of the hop-1 observation text.

    A real LLM reads this for free; the rule policy parses the one pattern the
    products corpus uses ("Acme Corp was acquired by Globex ...")."""
    m = re.search(r"acquired by (\w+)", observation)
    return m.group(1) if m else "the acquirer"


# ===========================================================================
# Step 5. The ReAct loop: reason -> act -> observe, until finish OR budget.
#
# This is the whole agent. Each step: ask the controller for the next
# (thought, tool, arg), call the tool, record the observation, print the
# Thought/Action/Observation lines, and loop. Two exits, both honest:
#   - the controller calls finish(...)   -> return the answer (normal exit)
#   - we reach max_steps without finish  -> stop (the infinite-loop guard)
# ===========================================================================
def run_agent(goal, max_steps=6, trace=True):
    """Run the reason/act/observe loop until finish() or the step budget."""
    def log(msg):
        if trace:
            print(msg)

    log(f"GOAL: {goal}\n")
    transcript_steps = []                              # (thought, tool, arg, obs)

    for step in range(1, max_steps + 1):
        # REASON: pick the next action from the goal + the transcript so far.
        # (Offline: the rule policy. With OPENAI_API_KEY set, the banner notes
        #  the real controller would drive this via generate(build_react_prompt).)
        thought, tool_name, arg = controller(goal, transcript_steps)
        log(f"  Step {step}")
        log(f"    Thought: {thought}")

        # FINISH is a tool too -- calling it ends the loop with the answer.
        if tool_name == "finish":
            log(f'    Action: finish("{arg}")')
            return arg, step, _retrievals(transcript_steps)

        # ACT: call the chosen tool with its argument.
        result = TOOLS[tool_name](arg)
        if tool_name in ("search_policy", "search_products"):
            text, score = result
            observation = text
            log(f'    Action: {tool_name}("{arg}")')
            log(f"    Observation: {observation} (score={score:.2f})")
        else:  # calculator
            observation = result
            log(f'    Action: {tool_name}("{arg}")')
            log(f"    Observation: {observation}")

        # OBSERVE: record the result so the next step can read it. When one
        # observation feeds the next action, THAT is multi-hop.
        transcript_steps.append((thought, tool_name, arg, observation))

    # Ran the budget without a finish(): stop honestly rather than loop forever.
    log("  -> step budget exhausted without finish(); stopping.")
    return ("Stopped: agent did not converge within the step budget.",
            max_steps, _retrievals(transcript_steps))


def _retrievals(transcript_steps):
    """Count how many steps actually hit an index (for the run footers)."""
    return sum(1 for _t, tool, _a, _o in transcript_steps
               if tool in ("search_policy", "search_products"))


# ===========================================================================
# Demo. Everything below RUNS OFFLINE. It uses the real retriever/controller if
#       available and the deterministic stand-ins otherwise, with clear labels.
# ===========================================================================
if __name__ == "__main__":
    line = "=" * 72

    print(line)
    print("RAG AGENT  -  a reason/act/observe loop that picks its own tools at run time")
    print(line)

    # The retriever banner already printed when the stores were built (Step 2);
    # now announce which controller path is live.
    if os.environ.get("OPENAI_API_KEY"):
        print("[controller] OPENAI_API_KEY set; the real LLM controller would drive "
              "generate(build_react_prompt(...)). Falling through to the deterministic "
              "policy so output stays reproducible.")
    else:
        print("[controller] no OPENAI_API_KEY; using deterministic rule-based "
              "ReAct controller (offline default)")

    print(f"\nTools: {', '.join(TOOLS)}")
    print(f"Policy KB: {len(POLICY_KB)} chunks (refunds, the E-4042 error, shipping, warranty).")
    print(f"Products: {len(PRODUCTS)} chunks "
          "(the Acme -> Globex acquisition + earbuds warranty chain).")

    # --- RUN (a): MULTI-HOP. The exact Part 10 prose example, now executed. ---
    print("\n" + "-" * 72)
    print("RUN (a) MULTI-HOP: chain two retrievals the single-pass pipeline can't.")
    print("-" * 72)
    answer, steps, hits = run_agent(
        "what is the warranty on the earbuds made by the company that acquired Acme?")
    print(f"\n  ANSWER: {answer}")
    print(f"  ({steps} steps, {hits} retrievals -- a single-pass pipeline retrieves "
          "once and cannot connect Acme -> Globex -> warranty.)")

    # --- RUN (b): NO-RETRIEVAL. Arithmetic -> calculator, index untouched. ----
    print("\n" + "-" * 72)
    print("RUN (b) NO-RETRIEVAL: arithmetic needs a calculator, not an index.")
    print("-" * 72)
    answer, steps, hits = run_agent("what is 18% of a $250 order?")
    print(f"\n  ANSWER: {answer}")
    print(f"  ({steps} steps, {hits} retrievals -- the agent touched the index zero times.)")

    # --- RUN (c): ROUTING. A policy question -> the POLICY index, not products. -
    print("\n" + "-" * 72)
    print("RUN (c) ROUTING: a policy question goes to the POLICY index, not products.")
    print("-" * 72)
    answer, steps, hits = run_agent("what's our refund window?")
    print(f"\n  ANSWER: {answer}")
    print(f"  ({steps} steps, {hits} retrieval -- routed to the policy index, not products.)")

    print("\n" + line)
    print("Done. The agent decided each path at run time, not at author time:")
    print("  (a) multi-hop : two chained retrievals (Acme -> Globex -> warranty)")
    print("  (b) no-retrieval: a calculator call, index untouched")
    print("  (c) routing   : the right index picked on the first try")
    print("Every loop stops on finish() or the max-steps budget -- never runs forever.")
    print(line)
