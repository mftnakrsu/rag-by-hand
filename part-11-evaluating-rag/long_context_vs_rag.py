"""
long_context_vs_rag.py  -  RAG from First Principles, Part 11 ("Evaluating RAG")
A companion experiment: long-context LLM vs RAG, measured head-to-head.

"Context windows are huge now, so RAG is dead." You have heard it. This file is
the honest, runnable answer. It is NOT "RAG is dead" and it is NOT "RAG always
wins". The long-context-vs-RAG debate is real, and the answer is the least
satisfying one in engineering: IT DEPENDS -- on corpus size, query type, and
budget. So instead of arguing, we measure.

The trap with this measurement is data leakage. If you ask a real model "what
year was the French Revolution", it answers from PRETRAINING, not from your
documents -- and then "long context" looks magically accurate for the wrong
reason. The fix (used by the U-NIAH benchmark, arXiv 2503.00353) is a SYNTHETIC,
FICTIONAL corpus with inserted "needles": facts stated in unique invented tokens
that appear in no training set anywhere. If the answer is right, it came from the
text in front of the model -- never from memorized knowledge.

We build a tiny fictional world ("Starlight Academy") and two strategies:

  - llm_alone(question, corpus): stuff the WHOLE corpus into the context window
    and scan it for the answer. Cost = every token, every time.
  - rag(question, corpus, k):    retrieve the top-k chunks by a deterministic
    embedder's cosine, then answer from just those. Cost = retrieved tokens only.

Both "answer" by the same offline, transparent rule -- a substring search for the
gold needle -- and we score accuracy by needle match. We tabulate accuracy, cost
(token count), and a latency proxy (tokens scanned) across a couple of corpus
sizes, so you can watch the tradeoff move as the haystack grows.

The two pragmatic syntheses worth knowing (both cited in the Part 11 essay):
  - Self-Route (Li et al., EMNLP 2024 industry track, arXiv 2407.16833): let a
    cheap router send easy queries to long-context and hard ones to RAG.
  - LaRA (Li et al., ICML 2025, arXiv 2502.09977): a benchmark whose own title
    says it plainly -- "No Silver Bullet for LC or RAG Routing".

Run (pure standard library + numpy, no API key, no model, no network):

    python long_context_vs_rag.py

This is deliberately a TOY. Real long-context models suffer "lost in the middle"
effects this substring oracle does not model; real retrieval can miss the needle
when paraphrased. The point is the SHAPE of the tradeoff (flat-ish RAG cost vs
linearly-growing long-context cost at equal accuracy on clean needles), not a
benchmark number to quote.

================================ Expected output ===============================
(pasted verbatim from a real offline run: `python long_context_vs_rag.py`)

==============================================================================
LONG-CONTEXT LLM vs RAG  -  a fictional-corpus, no-leakage head-to-head
==============================================================================
Fictional world: Starlight Academy. Needles use invented tokens, so a
correct answer can ONLY come from the text in front of the model --
never from anything it memorized in pretraining.

The needles we hide and then ask about:
  Q1: What is the name of the headmaster of Starlight Academy?
       gold needle -> 'Zephyrine Quorvax'
  Q2: Which potion does the Glimwort Ceremony require?
       gold needle -> 'Moonsilver Draught'
  Q3: What is the secret password to the Astral Library?
       gold needle -> 'quorvex-lumens-7'

==============================================================================
WORKED EXAMPLE  (small corpus, question 3: the Astral Library password)
==============================================================================
  LLM-alone: scanned all 12 chunks (125 tokens) -> answer = 'quorvex-lumens-7'
  RAG (k=3): retrieved ['c8', 'c4', 'c7'] (31 tokens) -> answer = 'quorvex-lumens-7'
  Same answer; RAG read far fewer tokens to get there.

HEAD-TO-HEAD across two corpus sizes:

==============================================================================
CORPUS: 12 chunks  (~125 tokens)   [3 needles, 3 filler chunks per needle]
==============================================================================
  strategy      accuracy   avg cost (tok)   avg latency (tok scanned)
  -------------------------------------------------------------------
  LLM-alone        100%            125.0                       125.0
  RAG (k=3)        100%             31.7                        31.7

==============================================================================
CORPUS: 39 chunks  (~418 tokens)   [3 needles, 12 filler chunks per needle]
==============================================================================
  strategy      accuracy   avg cost (tok)   avg latency (tok scanned)
  -------------------------------------------------------------------
  LLM-alone        100%            418.0                       418.0
  RAG (k=3)        100%             33.3                        33.3

==============================================================================
Reading the table:
  - Accuracy is equal here because the needle is clean and retrieval
    finds it. On a clean needle, BOTH strategies are correct.
  - Cost/latency tell the real story: LLM-alone pays for EVERY token of
    the corpus on every query, so its bill climbs as the corpus grows.
    RAG reads only the k retrieved chunks, so its cost stays roughly flat.
  - This is a TOY. Real long-context models hit 'lost in the middle'
    accuracy dips; real retrieval can miss a paraphrased needle. The
    point is the SHAPE of the tradeoff, not the exact numbers.

So: not 'RAG is dead' and not 'RAG always wins'. IT DEPENDS -- on corpus
size, query type, and budget. Self-Route (arXiv 2407.16833) and LaRA
(ICML 2025, arXiv 2502.09977) both land on routing between the two.
==============================================================================
================================================================================
"""

import re

import numpy as np

# ---------------------------------------------------------------------------
# A tiny FICTIONAL world. Every "needle" fact uses invented tokens (Zephyrine,
# Quorvex, glimwort...) that no pretrained model has ever seen, so a correct
# answer can only have come from the text we put in front of it -- never from
# memorized knowledge. This is the U-NIAH trick at miniature scale.
# ---------------------------------------------------------------------------
NEEDLES = [
    {
        "question": "What is the name of the headmaster of Starlight Academy?",
        "fact": "The headmaster of Starlight Academy is Professor Zephyrine Quorvax.",
        "needle": "Zephyrine Quorvax",
    },
    {
        "question": "Which potion does the Glimwort Ceremony require?",
        "fact": "The Glimwort Ceremony at Starlight Academy requires the Moonsilver Draught.",
        "needle": "Moonsilver Draught",
    },
    {
        "question": "What is the secret password to the Astral Library?",
        "fact": "The secret password to the Astral Library is 'quorvex-lumens-7'.",
        "needle": "quorvex-lumens-7",
    },
]

# Bland, plausible "filler" sentences that share the world's vocabulary but hold
# no needle. These are the haystack; we grow how many of them surround each
# needle to simulate a larger corpus. None of them answers any question.
FILLER = [
    "Starlight Academy sits on a floating island above the Cinderhaven Sea.",
    "Students at the academy wear robes dyed with crushed glimwort petals.",
    "The east tower houses the observatory and three brass orreries.",
    "Every autumn the academy hosts a lantern regatta on the inner lake.",
    "First-year students are sorted into one of the four star-houses.",
    "The dining hall serves spiced cloudberry tart on festival evenings.",
    "A statue of the academy's founder stands in the central courtyard.",
    "The greenhouse cultivates rare moonsilver vines along its north wall.",
    "Owls deliver the academy post twice daily, at dawn and at dusk.",
    "The bell in the west tower has rung on the hour for four centuries.",
    "Apprentice alchemists practice in the vaulted basement laboratories.",
    "The academy crest shows a comet crossing a crescent moon.",
]


def build_corpus(n_filler_per_needle: int) -> list[dict]:
    """Build a fictional corpus: each needle fact, padded with filler chunks so
    the needle is buried in a haystack of size we control. Returns a list of
    chunks (id + text); exactly len(NEEDLES) of them carry an answer."""
    corpus, cid = [], 0
    for ni, needle in enumerate(NEEDLES):
        # drop the needle fact in, then surround it with rotating filler
        corpus.append({"id": f"c{cid}", "text": needle["fact"], "needle_for": ni})
        cid += 1
        for j in range(n_filler_per_needle):
            corpus.append({"id": f"c{cid}", "text": FILLER[(ni + j) % len(FILLER)],
                           "needle_for": None})
            cid += 1
    return corpus


# ---------------------------------------------------------------------------
# Tokenization + a deterministic, offline "embedder". The same hashing stand-in
# the rest of the series uses (Part 2): any text -> a fixed-length unit vector,
# reproducible with no model and no network. Texts that share words land closer,
# which is enough to retrieve a needle whose question reuses its rare tokens.
# ---------------------------------------------------------------------------
def tokenize(text: str) -> list[str]:
    # keep hyphenated codes like 'quorvex-lumens-7' intact
    return re.findall(r"[a-z0-9]+(?:-[a-z0-9]+)*", text.lower())


def count_tokens(text: str) -> int:
    return len(tokenize(text))


DIM = 256


def embed(text: str) -> np.ndarray:
    """Deterministic hashing embedder: sum a stable pseudo-random direction per
    token, then L2-normalize. No model, no network -- same shape as Part 2."""
    toks = tokenize(text)
    vec = np.zeros(DIM)
    for tok in toks:
        # a fixed seed per token -> the same direction every run
        seed = int.from_bytes(tok.encode("utf-8"), "little") % (2**32)
        rng = np.random.default_rng(seed)
        vec += rng.standard_normal(DIM)
    norm = np.linalg.norm(vec)
    return vec if norm == 0 else vec / norm


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b))   # inputs are unit vectors -> dot == cosine


# ---------------------------------------------------------------------------
# The "answer" oracle. We do NOT call an LLM (offline, deterministic, no leakage
# risk). Both strategies answer the same honest way: scan the text they are
# allowed to see and return the gold needle if it is present, else a refusal.
# This isolates the variable we care about -- WHAT TEXT each strategy got to see
# -- from the messiness of a real generator.
# ---------------------------------------------------------------------------
def answer_from_text(question_idx: int, text: str) -> str:
    needle = NEEDLES[question_idx]["needle"]
    if needle.lower() in text.lower():
        return needle
    return "(not found in the provided text)"


def is_correct(question_idx: int, answer: str) -> bool:
    # accuracy = substring match on the gold needle
    return NEEDLES[question_idx]["needle"].lower() in answer.lower()


# ---------------------------------------------------------------------------
# Strategy A: the long-context LLM. Stuff EVERYTHING into the window and scan it.
# On a clean needle it is perfectly accurate -- but it pays for every token in
# the corpus on every single query. As the haystack grows, so does the bill.
# ---------------------------------------------------------------------------
def llm_alone(question_idx: int, corpus: list[dict]) -> dict:
    full_context = "\n".join(c["text"] for c in corpus)
    answer = answer_from_text(question_idx, full_context)
    tokens = sum(count_tokens(c["text"]) for c in corpus)
    return {
        "answer": answer,
        "correct": is_correct(question_idx, answer),
        "cost_tokens": tokens,          # billed: the whole corpus, every query
        "latency_proxy": tokens,        # proxy: tokens the model must scan
    }


# ---------------------------------------------------------------------------
# Strategy B: RAG. Retrieve the top-k chunks by embedder cosine, then answer from
# JUST those. Cost is the retrieved tokens, which stays roughly flat as the
# corpus grows -- you only ever read k chunks. The risk is the inverse of
# long-context's: if retrieval misses the needle, no amount of context helps.
# ---------------------------------------------------------------------------
def rag(question_idx: int, corpus: list[dict], k: int = 3) -> dict:
    q_vec = embed(NEEDLES[question_idx]["question"])
    scored = sorted(
        ((cosine(q_vec, embed(c["text"])), c) for c in corpus),
        key=lambda s: -s[0],
    )
    retrieved = [c for _score, c in scored[:k]]
    context = "\n".join(c["text"] for c in retrieved)
    answer = answer_from_text(question_idx, context)
    tokens = sum(count_tokens(c["text"]) for c in retrieved)
    return {
        "answer": answer,
        "correct": is_correct(question_idx, answer),
        "cost_tokens": tokens,          # billed: only the k retrieved chunks
        "latency_proxy": tokens,        # proxy: tokens scanned to answer
        "retrieved_ids": [c["id"] for c in retrieved],
    }


# ---------------------------------------------------------------------------
# The head-to-head: run every question under both strategies, across a couple of
# corpus sizes, and tabulate accuracy / cost / latency. The numbers are toy; the
# SHAPE is the lesson.
# ---------------------------------------------------------------------------
def run_head_to_head(filler_sizes: list[int], k: int = 3) -> None:
    line = "=" * 78
    for n_filler in filler_sizes:
        corpus = build_corpus(n_filler)
        corpus_tokens = sum(count_tokens(c["text"]) for c in corpus)
        print(line)
        print(f"CORPUS: {len(corpus)} chunks  (~{corpus_tokens} tokens)   "
              f"[{len(NEEDLES)} needles, {n_filler} filler chunks per needle]")
        print(line)

        results = {"llm_alone": [], "rag": []}
        for qi in range(len(NEEDLES)):
            results["llm_alone"].append(llm_alone(qi, corpus))
            results["rag"].append(rag(qi, corpus, k=k))

        header = f"  {'strategy':<12}  {'accuracy':>8}  {'avg cost (tok)':>15}  {'avg latency (tok scanned)':>26}"
        print(header)
        print("  " + "-" * (len(header) - 2))
        for name in ("llm_alone", "rag"):
            rs = results[name]
            acc = sum(r["correct"] for r in rs) / len(rs)
            avg_cost = sum(r["cost_tokens"] for r in rs) / len(rs)
            avg_lat = sum(r["latency_proxy"] for r in rs) / len(rs)
            label = "LLM-alone" if name == "llm_alone" else f"RAG (k={k})"
            print(f"  {label:<12}  {acc:>7.0%}  {avg_cost:>15.1f}  {avg_lat:>26.1f}")
        print()


if __name__ == "__main__":
    line = "=" * 78

    print(line)
    print("LONG-CONTEXT LLM vs RAG  -  a fictional-corpus, no-leakage head-to-head")
    print(line)
    print("Fictional world: Starlight Academy. Needles use invented tokens, so a")
    print("correct answer can ONLY come from the text in front of the model --")
    print("never from anything it memorized in pretraining.\n")

    # Show the needles so the reader can see what 'correct' means.
    print("The needles we hide and then ask about:")
    for ni, nd in enumerate(NEEDLES):
        print(f"  Q{ni + 1}: {nd['question']}")
        print(f"       gold needle -> {nd['needle']!r}")
    print()

    # One worked example at the small size, so the mechanism is visible.
    demo_corpus = build_corpus(n_filler_per_needle=3)
    print(line)
    print("WORKED EXAMPLE  (small corpus, question 3: the Astral Library password)")
    print(line)
    qi = 2
    la = llm_alone(qi, demo_corpus)
    rg = rag(qi, demo_corpus, k=3)
    print(f"  LLM-alone: scanned all {len(demo_corpus)} chunks "
          f"({la['cost_tokens']} tokens) -> answer = {la['answer']!r}")
    print(f"  RAG (k=3): retrieved {rg['retrieved_ids']} "
          f"({rg['cost_tokens']} tokens) -> answer = {rg['answer']!r}")
    print("  Same answer; RAG read far fewer tokens to get there.\n")

    # The head-to-head across two corpus sizes.
    print("HEAD-TO-HEAD across two corpus sizes:\n")
    run_head_to_head(filler_sizes=[3, 12], k=3)

    print(line)
    print("Reading the table:")
    print("  - Accuracy is equal here because the needle is clean and retrieval")
    print("    finds it. On a clean needle, BOTH strategies are correct.")
    print("  - Cost/latency tell the real story: LLM-alone pays for EVERY token of")
    print("    the corpus on every query, so its bill climbs as the corpus grows.")
    print("    RAG reads only the k retrieved chunks, so its cost stays roughly flat.")
    print("  - This is a TOY. Real long-context models hit 'lost in the middle'")
    print("    accuracy dips; real retrieval can miss a paraphrased needle. The")
    print("    point is the SHAPE of the tradeoff, not the exact numbers.")
    print()
    print("So: not 'RAG is dead' and not 'RAG always wins'. IT DEPENDS -- on corpus")
    print("size, query type, and budget. Self-Route (arXiv 2407.16833) and LaRA")
    print("(ICML 2025, arXiv 2502.09977) both land on routing between the two.")
    print(line)
