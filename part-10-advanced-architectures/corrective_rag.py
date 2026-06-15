"""
Corrective RAG (CRAG) by hand: the corrective branch from Part 10.
RAG from First Principles, Part 10: Advanced RAG Architectures.

This is the one pattern Part 10 makes most concrete. A naive pipeline
(Part 6) retrieves once and generates from whatever it got, even garbage.
CRAG inserts a single check between retrieve and generate: a lightweight
*retrieval evaluator* that grades the chunks as relevant / ambiguous /
irrelevant, and the system acts on that grade BEFORE generating:

  - relevant   -> generate a grounded answer now.
  - ambiguous  -> reformulate the query and loop to try our index again.
  - irrelevant -> do NOT generate on bad context; rewrite for an outside
                  source and fall back to web search (a different source).
  - out of tries -> refuse honestly ("I don't know..."), Part 6's grounding.

That bad-retrieval -> correct -> retry branch is the whole story of Part 10
in miniature. CRAG is one disciplined slice of the open-ended Agentic-RAG
loop, and often all you actually need: one extra evaluation step, one
well-defined branch, no open-ended wandering.

Stack:
  - Embeddings : sentence-transformers if installed, else a transparent
                 deterministic hashing-bag-of-words stand-in (numpy only).
  - Evaluator  : a hosted LLM grader if available, else a deterministic
                 similarity-threshold grader (the "transparent fallback that
                 runs without a model" pattern the later essays use).
  - Generation : a hosted LLM if available, else an extractive stand-in that
                 stitches the retrieved chunks into a grounded reply.
  - Web search : a real web call in production; here a stubbed/simulated
                 offline mini-corpus so the whole demo runs with no network.

Run:
  python3 corrective_rag.py        # runs fully offline, no API key, no network
  # optionally: pip install sentence-transformers, set OPENAI_API_KEY, and the
  # REAL paths below light up automatically. The demo always works offline.

NOTE: LLM SDK syntax and model names move fast and may have changed since this
was written. Check current provider docs; only the *_real() helpers need edits.
"""

import os
import re
import numpy as np

# ---------------------------------------------------------------------------
# Step 0. Two tiny, eyeball-able corpora.
#
# OUR INDEX is the same store-policy corpus from Part 6: refunds, shipping,
# warranty. It deliberately holds NO product specs, exactly the routing
# failure the flagship example in Part 10 is built around.
#
# THE WEB is the stubbed/simulated outside source CRAG falls back to when our
# own index cannot possibly hold the answer. In production this is a real web
# search; offline we hard-code a couple of "pages" so the demo is honest about
# what a fallback would return without ever touching the network.
# ---------------------------------------------------------------------------
OUR_INDEX = [
    "Refunds are accepted within 30 days of purchase, provided the item is unused and in its original packaging.",
    "To start a return, email support@example.com with your order number. Refunds are processed within five business days of us receiving the item.",
    "Standard shipping takes 3 to 5 business days. Express shipping arrives the next business day.",
    "Shipping fees are non-refundable, and items marked final sale cannot be returned or exchanged.",
    "All electronics include a one-year limited warranty covering manufacturing defects.",
]

# Simulated web "pages" (the corrective fallback source). Offline + deterministic.
WEB_PAGES = [
    "The X1 wireless earbuds deliver up to 8 hours of battery life on a single charge, and up to 24 hours total with the charging case.",
    "The X1 wireless earbuds support Bluetooth 5.3 and are rated IPX4 for sweat and water resistance.",
    "Acme Corp was acquired by Globex in 2024; Globex now manufactures the X1 wireless earbuds.",
]


# ---------------------------------------------------------------------------
# Step 1. Embeddings, with a transparent deterministic fallback.
#
# The PRIMARY/intended path is sentence-transformers, exactly as in Part 6.
# If it is not installed, we drop to a pure-numpy hashing bag-of-words: hash
# each token into a fixed-width vector and L2-normalize. It is crude, but it
# is deterministic, needs no model or network, and gives a real cosine score,
# enough to make the CRAG control flow concrete. normalize=True means a dot
# product equals cosine similarity later (the trick from Part 3).
# ---------------------------------------------------------------------------
_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tokens(text):
    return _TOKEN_RE.findall(text.lower())


class _HashingEmbedder:
    """Deterministic, model-free stand-in for a sentence embedder (Part 2)."""

    def __init__(self, dim=256):
        self.dim = dim

    def encode(self, texts, normalize_embeddings=True):
        vecs = np.zeros((len(texts), self.dim), dtype=np.float64)
        for r, text in enumerate(texts):
            for tok in _tokens(text):
                # Stable hash -> bucket. Sign spreads collisions, like the
                # hashing trick. Deterministic across runs (no PYTHONHASHSEED).
                h = 0
                for ch in tok:
                    h = (h * 131 + ord(ch)) & 0xFFFFFFFF
                vecs[r, h % self.dim] += 1.0 if (h >> 1) & 1 else -1.0
        if normalize_embeddings:
            norms = np.linalg.norm(vecs, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            vecs = vecs / norms
        return vecs


def _load_embedder():
    """Try the real model first; fall back transparently if unavailable."""
    try:
        from sentence_transformers import SentenceTransformer  # REAL path
        model = SentenceTransformer("all-MiniLM-L6-v2")         # 384 dims
        print("[embed] using sentence-transformers (all-MiniLM-L6-v2)")
        return model, True
    except Exception as exc:  # not installed / no weights / offline
        print(f"[embed] sentence-transformers unavailable ({type(exc).__name__}); "
              "falling back to deterministic hashing embedder")
        return _HashingEmbedder(), False


_EMBEDDER, _REAL_EMBEDDER = _load_embedder()


def embed(texts):
    return np.asarray(_EMBEDDER.encode(texts, normalize_embeddings=True))


# ---------------------------------------------------------------------------
# Step 2. A tiny vector store + retriever, one per source.
#
# The "vector store" is just chunks plus a matrix of their vectors kept side
# by side (Part 4). retrieve() embeds the query with the SAME model, scores by
# cosine similarity, and keeps the top-k (Part 3 + Part 4). We build one store
# for OUR_INDEX and one for the simulated web, so CRAG can route between them.
# ---------------------------------------------------------------------------
class VectorStore:
    def __init__(self, name, corpus):
        self.name = name
        self.chunks = [{"text": doc, "source": f"{name}_{i}"} for i, doc in enumerate(corpus)]
        self.vectors = embed([c["text"] for c in self.chunks])  # (n_chunks, dim)

    def retrieve(self, query, k=3):
        q = embed([query])[0]                       # same model as the chunks
        scores = self.vectors @ q                   # cosine sim (unit vectors)
        top = np.argsort(-scores)[:k]               # indices of k highest scores
        return [(self.chunks[i]["text"], float(scores[i])) for i in top]


OUR_STORE = VectorStore("policy", OUR_INDEX)
WEB_STORE = VectorStore("web", WEB_PAGES)


def retrieve(query, k=3):
    """CRAG searches our OWN index first (the essay's `retrieve(query)`)."""
    return OUR_STORE.retrieve(query, k=k)


# ---------------------------------------------------------------------------
# Step 3. The retrieval evaluator: the one component CRAG adds (Part 10).
#
# Its only job is to grade the chunks retrieval returned. The grade is coarse,
# relevant / ambiguous / irrelevant, and the system acts on it BEFORE
# generating. The PRIMARY path is a small LLM/classifier grader; offline we use
# a transparent deterministic grader that scores how well the BEST chunk
# actually addresses the query, then bands the score:
#
#     relevant   if the best chunk clearly answers it
#     ambiguous  if there is some signal but not enough (reformulate, retry)
#     irrelevant if no chunk addresses the question (our index can't help)
#
# Rather than threshold a raw cosine alone (fragile: it shifts with whichever
# embedder is installed), we combine cosine similarity with CONTENT-WORD
# OVERLAP, the fraction of the query's meaningful words the best chunk actually
# contains. Overlap is what makes the flagship example honest: a product-spec
# question shares almost no content words with a policy-only index, so it grades
# irrelevant regardless of the embedder, exactly the routing failure Part 10
# describes. Thresholds are chosen so the worked examples land on the right
# branch with EITHER embedder (real MiniLM or the hashing fallback).
# ---------------------------------------------------------------------------
RELEVANT_THRESHOLD = 0.50    # combined score at/above this -> trust the chunks
AMBIGUOUS_THRESHOLD = 0.20   # between the two bands -> some signal, reformulate
#                            below AMBIGUOUS_THRESHOLD -> irrelevant -> fall back


def grade_real(query, retrieved):
    """REAL path: ask a small LLM to grade the chunks (intended production code).

    A retrieval evaluator is typically a small model or classifier. Here is the
    shape with a hosted model; swap for any classifier you like. Only this
    helper touches the network, and the demo never calls it offline.
    """
    from openai import OpenAI                       # SDK names move; check docs
    client = OpenAI()                               # reads OPENAI_API_KEY
    context = "\n".join(f"- {t}" for t, _ in retrieved)
    prompt = (
        "You grade whether retrieved context can answer a question.\n"
        "Reply with exactly one word: relevant, ambiguous, or irrelevant.\n\n"
        f"Question: {query}\n\nContext:\n{context}\n\nGrade:"
    )
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )
    word = resp.choices[0].message.content.strip().lower()
    return word if word in {"relevant", "ambiguous", "irrelevant"} else "ambiguous"


def _content_overlap(query, chunk_text):
    """Fraction of the query's content words that appear in the chunk."""
    q_words = {w for w in _tokens(query) if w not in _STOPWORDS}
    if not q_words:
        return 0.0
    c_words = set(_tokens(chunk_text))
    return len(q_words & c_words) / len(q_words)


def _combined(query, text, cos):
    """Blend cosine similarity with content-word overlap into one [0,1] score."""
    return 0.5 * max(cos, 0.0) + 0.5 * _content_overlap(query, text)


def _best_chunk(query, retrieved):
    """The chunk that best ANSWERS the query, ranked by the combined score.

    Ranking by combined score (not raw cosine) keeps the evaluator and the
    generator consistent and makes selection robust to which embedder is loaded:
    the chunk with the most content-word overlap wins ties the cosine alone
    would get wrong.
    """
    return max(retrieved, key=lambda r: _combined(query, r[0], r[1]))


def grade_score(query, retrieved):
    """Combined relevance signal for the best-answering chunk."""
    if not retrieved:
        return 0.0
    best_text, best_cos = _best_chunk(query, retrieved)
    return _combined(query, best_text, best_cos)


def grade_fallback(query, retrieved):
    """Deterministic, model-free grader: cosine similarity + content overlap.

    The combined score blends the embedder's cosine with the lexical fraction of
    the query's content words the best chunk contains. The overlap term is what
    makes the grade robust to which embedder is installed and faithful to the
    essay: a product-spec question against a policy index has near-zero overlap,
    so it grades irrelevant, the routing failure CRAG is built to catch.
    """
    score = grade_score(query, retrieved)
    if score >= RELEVANT_THRESHOLD:
        return "relevant"
    if score >= AMBIGUOUS_THRESHOLD:
        return "ambiguous"
    return "irrelevant"


class Evaluator:
    """Mirrors the essay's `evaluator.grade(query, chunks)` call site."""

    def __init__(self, use_real):
        self.use_real = use_real

    def grade(self, query, retrieved):
        if self.use_real:
            try:
                return grade_real(query, retrieved)
            except Exception:
                pass  # fall through to the transparent stand-in
        return grade_fallback(query, retrieved)


# Use the real grader only if we have BOTH a real embedder and an API key.
evaluator = Evaluator(use_real=_REAL_EMBEDDER and bool(os.environ.get("OPENAI_API_KEY")))


# ---------------------------------------------------------------------------
# Step 4. The two corrective actions: query reformulation and web fallback.
#
# reformulate() lightly rewrites the query to try OUR index again (the
# `ambiguous` branch). rewrite_for_web() reshapes it for an outside source,
# and web_search() is the stubbed/simulated fallback to a DIFFERENT source
# (the `irrelevant` branch). In production web_search() is a real API call;
# offline it just queries the simulated WEB_STORE. No network, deterministic.
# ---------------------------------------------------------------------------
# _STOPWORDS: the small set of grammatical words the overlap metric ignores when
# it decides whether a chunk addresses the query's content words.
_STOPWORDS = {
    "what", "is", "are", "the", "of", "a", "an", "for", "on", "in", "to", "how",
    "do", "does", "and", "my", "i", "there", "with", "your",
}

# _FILLER: a LARGER set of conversational chatter that reformulate() also strips.
# Keeping it separate from _STOPWORDS is deliberate: a chatty query can grade
# ambiguous on the first pass (the filler dilutes the cosine), and reformulating
# it down to its content words is what lets the retry succeed, exactly the
# ambiguous -> reformulate -> retry branch Part 10 describes.
_FILLER = _STOPWORDS | {"can", "you", "tell", "me", "about", "please", "could", "would"}


def reformulate(query):
    """Ambiguous branch: tweak the query to retry our own index."""
    # A real system would call an LLM to paraphrase. Here we deterministically
    # drop low-signal filler so the next retrieval scores the content words.
    kept = [w for w in _tokens(query) if w not in _FILLER]
    reformed = " ".join(kept) if kept else query
    return reformed


def rewrite_for_web(query):
    """Irrelevant branch: reshape the query for an outside source."""
    return f"{reformulate(query)} specifications"


def web_search_real(query, k=3):
    """REAL path: a live web-search API. The only networked helper; unused offline."""
    import requests                                 # pragma: no cover
    resp = requests.get(
        "https://api.example-search.com/v1/search",  # placeholder endpoint
        params={"q": query, "k": k},
        timeout=10,
    )
    hits = resp.json()["results"]
    return [(h["snippet"], float(h.get("score", 0.0))) for h in hits]


def web_search(query, k=3):
    """Offline stand-in: query the simulated WEB_STORE instead of the network."""
    return WEB_STORE.retrieve(query, k=k)


# ---------------------------------------------------------------------------
# Step 5. Generation, with a grounded extractive fallback.
#
# The PROMPT_TEMPLATE and grounding instruction are lifted straight from Part 6
# so the model answers ONLY from context and refuses otherwise. The PRIMARY
# path is a hosted LLM; offline we use an extractive stand-in that stitches the
# retrieved chunks into a clearly-grounded reply, so the demo always produces
# sensible, source-backed output with no model.
# ---------------------------------------------------------------------------
PROMPT_TEMPLATE = """Answer the question using ONLY the context below.
If the answer is not in the context, say "I don't know based on the available sources."

Context:
{context}

Question: {question}
Answer:"""


def build_prompt(query, retrieved):
    context = "\n".join(f"- {text}" for text, _score in retrieved)
    return PROMPT_TEMPLATE.format(context=context, question=query)


def generate_real(prompt):
    """REAL path: one swappable hosted-LLM call (Part 6's generate())."""
    from openai import OpenAI
    client = OpenAI()                               # reads OPENAI_API_KEY
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0,                              # grounded, not creative
    )
    return resp.choices[0].message.content


def generate_fallback(query, retrieved):
    """Deterministic extractive generator: ground the answer in the chunks."""
    if not retrieved:
        return "I don't know based on the available sources."
    # Pick the best-ANSWERING chunk (same ranking as the grader) and quote it.
    best_text, best_cos = _best_chunk(query, retrieved)
    return (f"Based on the retrieved sources: {best_text} "
            f"(grounding score {_combined(query, best_text, best_cos):.2f})")


def generate(query, retrieved):
    """Build the prompt (real path) or extract offline. Always grounded."""
    if _REAL_EMBEDDER and os.environ.get("OPENAI_API_KEY"):
        try:
            return generate_real(build_prompt(query, retrieved))
        except Exception:
            pass
    return generate_fallback(query, retrieved)


# ===========================================================================
# Step 6. The Corrective RAG control-flow function.
#
# This is a faithful, runnable expansion of the essay's `corrective_rag`
# sketch (Part 10). Same shape: retrieve, grade, then either correct and retry
# or generate. The added `verbose` tracing makes the branch it takes visible.
# ===========================================================================
def corrective_rag(query, max_tries=2, verbose=True):
    def log(msg):
        if verbose:
            print(msg)

    log(f'\nQUERY: "{query}"')
    for attempt in range(max_tries):
        log(f"  attempt {attempt + 1}/{max_tries}")
        chunks = retrieve(query)                       # search our own index first
        grade = evaluator.grade(query, chunks)         # relevant/ambiguous/irrelevant
        log(f'    retrieved query="{query}"  '
            f'grade_score={grade_score(query, chunks):.2f}  grade={grade}')

        if grade == "relevant":
            log("    -> RELEVANT: good context, answer now.")
            return generate(query, chunks)             # good context: answer now

        # bad context: do NOT generate on it. take a corrective action.
        if grade == "irrelevant":
            log("    -> IRRELEVANT: our index can't answer; fall back to web search.")
            query = rewrite_for_web(query)             # reformulate for outside source
            web_chunks = web_search(query)             # fall back to a DIFFERENT source
            web_grade = evaluator.grade(query, web_chunks)  # a real CRAG grades these too
            log(f'       web search query="{query}"  '
                f'grade_score={grade_score(query, web_chunks):.2f}  grade={web_grade}')
            if web_grade == "irrelevant":
                log("       -> web fallback also weak; refuse honestly.")
                return "I don't know based on the available sources."
            log("       -> web context usable, answer from it.")
            return generate(query, web_chunks)

        # ambiguous: tweak the query and loop to try our index again.
        log("    -> AMBIGUOUS: reformulate the query and retry our index.")
        query = reformulate(query)

    # ran out of tries without good context: refuse honestly (Part 6's grounding).
    log("  -> out of tries without good context; refuse honestly.")
    return "I don't know based on the available sources."


# ===========================================================================
# Step 7. REFERENCE ONLY (not executed): an Agentic-RAG ReAct loop skeleton.
#
# CRAG above is one disciplined slice of the open-ended Agentic-RAG loop. For
# contrast, here is the fuller shape Part 10 describes: an agent that runs a
# reason -> act -> observe loop, choosing among a couple of TOOLS (retrieval is
# now just one tool, plus a calculator) until it judges the task done. This is
# left as a commented/stubbed reference; the executable demo is the CRAG path.
#
#   TOOLS = {
#       "search_policy": lambda q: OUR_STORE.retrieve(q),   # retrieval is a tool
#       "search_web":    lambda q: WEB_STORE.retrieve(q),   # routing target
#       "calculator":    lambda expr: eval(expr),           # so it stops fumbling math
#   }
#
#   def react_agent(goal, llm, tools=TOOLS, max_steps=6):
#       scratchpad = ""                       # the running reason/act/observe trace
#       for _ in range(max_steps):
#           # REASON: the model decides the next action from the goal + history.
#           thought = llm(f"Goal: {goal}\n{scratchpad}\n"
#                         f"Available tools: {list(tools)}\n"
#                         "Think, then emit: ACTION <tool> <input>  OR  FINISH <answer>")
#           if thought.startswith("FINISH"):
#               return thought[len('FINISH'):].strip()       # task judged done
#           # ACT: parse and call the chosen tool (number/order chosen at run time).
#           _, tool, arg = thought.split(maxsplit=2)
#           observation = tools[tool](arg)
#           # OBSERVE: feed the result back in and loop. This is multi-hop when one
#           # observation (e.g. "Globex acquired Acme") feeds the next retrieval.
#           scratchpad += f"\nACTION {tool} {arg}\nOBSERVATION {observation}"
#       return "Stopped: agent did not converge within step budget."
#
# The catch from Part 10: agents are powerful but slower (many model calls),
# costlier, less predictable, and harder to debug. CRAG buys most of the win,
# catching a bad retrieval, with one evaluation step and one branch.
# ===========================================================================


if __name__ == "__main__":
    print("=" * 72)
    print("Corrective RAG (CRAG) demo, Part 10. Runs offline; no API key, no network.")
    print("=" * 72)
    print(f"OUR index: {len(OUR_INDEX)} store-policy chunks (no product specs).")
    print(f"Simulated web fallback: {len(WEB_PAGES)} pages.")

    # --- Case 1: RELEVANT. A policy question our index can answer directly. ---
    print("\n" + "-" * 72)
    print("CASE 1 (expect RELEVANT branch): a question our policy index holds.")
    ans = corrective_rag("Are refunds accepted on unused items in original packaging?")
    print(f"ANSWER: {ans}")

    # --- Case 2: AMBIGUOUS -> reformulate -> retry. A chatty, filler-heavy ---
    # question whose first retrieval is only middling. CRAG does NOT generate on
    # it; it reformulates down to the content words and tries our index again,
    # which this time grades relevant.
    print("\n" + "-" * 72)
    print("CASE 2 (expect AMBIGUOUS -> reformulate -> retry then RELEVANT).")
    ans = corrective_rag("Can you tell me about the warranty for electronics please?")
    print(f"ANSWER: {ans}")

    # --- Case 3: IRRELEVANT -> web fallback. The flagship Part 10 example: a ---
    # product-spec question against a policy-only corpus. Naive RAG would guess;
    # CRAG grades the policy chunks irrelevant and routes to the web source.
    print("\n" + "-" * 72)
    print("CASE 3 (expect IRRELEVANT -> WEB fallback): the flagship example.")
    ans = corrective_rag("What is the battery life of the X1 wireless earbuds?")
    print(f"ANSWER: {ans}")

    # --- Case 4: refusal. A question NEITHER source can answer. After the ---
    # corrective web fallback also comes back weak, CRAG refuses honestly
    # rather than confidently guessing (Part 6's grounding).
    print("\n" + "-" * 72)
    print("CASE 4 (expect honest REFUSAL): nothing in either source.")
    ans = corrective_rag("How do I reset my smart thermostat to factory settings?")
    print(f"ANSWER: {ans}")

    print("\n" + "=" * 72)
    print("Done. The branch each query took is traced above:")
    print("  relevant   -> answer now")
    print("  ambiguous  -> reformulate & retry our index")
    print("  irrelevant -> rewrite & fall back to web search (a different source)")
    print("  exhausted  -> refuse honestly")
    print("(The Agentic-RAG ReAct loop in Step 7 is reference-only, not run.)")
    print("=" * 72)
