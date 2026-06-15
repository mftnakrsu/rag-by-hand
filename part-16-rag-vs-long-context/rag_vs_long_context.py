"""
rag_vs_long_context.py  -  RAG from First Principles, Part 16
                           ("RAG vs Long-Context vs CAG")

Part 1 asked WHY RAG exists. Part 16 asks the harder follow-up: when do you even
NEED retrieval? In 2026 context windows reach about a million tokens, so for a
small, stable corpus you can skip the index entirely and just put everything in
the prompt. And there is a sharper version of that idea -- Cache-Augmented
Generation (CAG, arXiv 2412.15605) -- which preloads a small, stable corpus once,
caches the model's internal KV state, and reuses it on every query instead of
retrieving.

The catch everyone quotes is "context is expensive": every retrieved or stuffed
token is billed on EVERY request. That is true at the FRESH input rate. It is
much less true once PROMPT CACHING enters the picture -- a cached input token
costs roughly an order of magnitude less than a fresh one. That ten-fold discount
on reuse is the entire economic engine behind CAG.

This file is the accountant for that trade. It is NOT "RAG is dead" and it is NOT
"just stuff everything". It is a small, honest cost model you can run and poke at,
so the decision matrix in the essay has numbers under it instead of vibes.

The model prices three strategies for answering N queries against the same corpus:

  - RAG:          embed the query, retrieve top-k chunks, and send ONLY those
                  k chunks as context. Small per-query input, but you re-send the
                  retrieved chunks fresh on every request (no stable prefix to
                  cache, because which chunks you send changes per query). Its
                  cost tracks k, NOT the corpus size -- the property that never
                  goes away.
  - long_context: stuff the WHOLE corpus into the prompt on every request, at the
                  fresh input rate, every time. The naive "just use the big
                  window" baseline. Linear in corpus size, forever.
  - cag:          stuff the WHOLE corpus into the prompt ONCE as a stable,
                  cacheable PREFIX (a write at ~1.25x the fresh rate), then on
                  every subsequent request the corpus tokens are served from
                  cache at ~0.1x. This is the prompt-caching shadow of CAG:
                  preload a small stable corpus once and reuse the cached state.

The pricing constants below mirror the PUBLIC shape of 2025-2026 prompt caching
on a frontier model (fresh input = 1 unit, cache write = 1.25x, cache read =
0.1x, output billed separately). They are deliberately in RELATIVE "cost units"
so the file makes a point about SHAPE, not a quote of any one vendor's price
sheet. Plug in your own provider's numbers and the digits move; the crossover
behaviour does not.

Beyond the website companion, this script also makes two of the essay's
"try it yourself" experiments runnable:

  1. KILL THE CACHE DISCOUNT (set cache_read = fresh): watch CAG collapse into
     naive long-context. The 0.1x read is the load-bearing wall -- without it,
     CAG was never viable.
  2. BREAK THE PREFIX with volatile tokens (per-request tool results that sneak
     into the cached prefix): watch CAG's advantage shrink as those tokens get
     billed fresh on every call. This is the "volatile tool results break the
     cache" pitfall, made numeric.

Run (pure standard library, no numpy, no API key, no network):

    python3 rag_vs_long_context.py

================================ Expected output ===============================
(pasted verbatim from a real run: `python3 rag_vs_long_context.py`)

==============================================================================
PROMPT-CACHING ECONOMICS  -  RAG vs long-context vs CAG
==============================================================================
Pricing (relative cost units, per token):
  fresh input         1.000   (a token the model has never seen this call)
  cache write         1.250   (first time a stable prefix is laid down)
  cache read          0.100   (a token served from a cached prefix)
  output              5.000   (billed the same for every strategy)

Workload:
  corpus              4000 tokens (small + stable: a candidate for CAG)
  query                 40 tokens
  retrieved (top-k)    600 tokens (the slice RAG sends per query)
  answer               250 tokens
  fixed instructions   200 tokens (system prompt, stable prefix)
  volatile tokens        0 tokens (per-request content inside the prefix)

------------------------------------------------------------------------------
COST TO ANSWER 1 QUERY
------------------------------------------------------------------------------
  strategy        input cost  output cost     total   vs RAG
  ----------------------------------------------------------
  rag                  890.0       1250.0    2140.0    1.00x
  long_context        4240.0       1250.0    5490.0    2.57x
  cag                 5290.0       1250.0    6540.0    3.06x

  At 1 query, CAG is WORST: you paid the cache-write premium and got one read.
  The whole point of CAG is amortization, so 1 query is exactly the wrong test.

------------------------------------------------------------------------------
COST TO ANSWER 100 QUERIES (same stable corpus)
------------------------------------------------------------------------------
  strategy        input cost  output cost     total   vs RAG
  ----------------------------------------------------------
  rag                66230.0     125000.0  191230.0    1.00x
  long_context      424000.0     125000.0  549000.0    2.87x
  cag                50830.0     125000.0  175830.0    0.92x

  Now CAG has amortized the write across 100 cache reads and overtaken RAG,
  while long_context (no caching) stays ~2.9x the bill. The corpus is small
  and stable, so CAG buys you long-context's simplicity below RAG's cost.

------------------------------------------------------------------------------
THE CROSSOVER  (where CAG total drops below long-context, and below RAG)
------------------------------------------------------------------------------
  CAG beats naive long-context after  2 query.
  CAG beats RAG after 24 queries.

------------------------------------------------------------------------------
WHAT MOVES THE ANSWER: corpus size sweep (100 queries, cache read = 0.10x)
------------------------------------------------------------------------------
  corpus tokens   rag total     long_context     cag total     winner
  -------------------------------------------------------------------
           1000    191230.0         249000.0      142380.0     cag
           4000    191230.0         549000.0      175830.0     cag
          16000    191230.0        1749000.0      309630.0     rag
          64000    191230.0        6549000.0      844830.0     rag
         256000    191230.0       25749000.0     2985630.0     rag

  Reading the sweep:
    - long_context cost grows LINEARLY with corpus size on every query: it
      re-reads the whole corpus fresh each time. This is the 'just stuff it'
      tax, and it is brutal as the corpus grows.
    - CAG grows too (the cached corpus is still read, at 0.1x, every query),
      but ~10x slower than long_context. For a SMALL stable corpus it is a
      great deal; for a large one the 0.1x reads still add up.
    - RAG stays cheapest here because it sends only the top-k slice. Its cost
      is driven by k, not by corpus size -- which is exactly why a massive
      corpus points at RAG.

------------------------------------------------------------------------------
EXPERIMENT 1 -- KILL THE CACHE DISCOUNT (cache read 0.10x -> 1.00x, 100 queries)
------------------------------------------------------------------------------
  strategy           with cache (0.10x)     no discount (1.00x)
  ------------------------------------------------------------
  rag                          191230.0                209050.0
  long_context                 549000.0                549000.0
  cag                          175830.0                550050.0

  Without the discount CAG (550050.0) collapses toward naive long-context
  (549000.0): preloading once buys you nothing if every read is full price.
  The 0.1x cached read is not a detail -- it is the load-bearing wall that
  makes CAG viable. Before cheap cached reads existed, CAG was just expensive
  long-context.

------------------------------------------------------------------------------
EXPERIMENT 2 -- BREAK THE PREFIX (volatile tokens inside the cached prefix)
------------------------------------------------------------------------------
  Per-request tool results / timestamps that slip INTO the cached prefix get
  billed FRESH every call (they change, so they never cache). Watch CAG's
  100-query advantage over RAG erode as the volatile block grows:

  volatile tokens   cag total     vs RAG     CAG still wins?
  ----------------------------------------------------------
                0    175830.0      0.92x     yes
              100    185830.0      0.97x     yes
              300    205830.0      1.08x     no
             1000    275830.0      1.44x     no

  A clean, stable prefix is the whole assumption. The moment something volatile
  (a timestamp, a per-request id, a changing tool result) lands in the prefix,
  you pay fresh-input rates for the corpus you thought was nearly free -- and
  CAG's edge evaporates. Keep volatile content LAST, after the final cache point.

==============================================================================
THE DECISION MATRIX  (two axes -- size and volatility -- plus cost as tiebreaker)
==============================================================================
  corpus shape                         ->  pick        because
  ----------------------------------------------------------------------------
  massive (bigger than the window)     ->  RAG         only RAG's cost tracks k,
                                                       not corpus size
  fast-moving (changes often)          ->  RAG         a moving corpus recomputes
                                                       the cache constantly
  private (per-user access filtering)  ->  RAG         only retrieval can filter
                                                       before the prompt
  small + stable + reused a lot        ->  CAG         the write amortizes over
                                                       cheap 0.1x reads
  tiny, or queried rarely              ->  long_context caching machinery is not
                                                       worth the bother
  mid-size + stable                    ->  long_context fits the window, too big
                                                       to want cached per call

  The meta-point (the spine of the whole series): the question is no longer
  "RAG or not?" answered once and globally. It is "what does THIS corpus want?"
  answered from its size, its volatility, and your traffic. Sometimes the
  corpus demands a full retrieval pipeline; sometimes it just wants to be
  cached. Knowing the difference is the last skill this series had to teach.

  The pricing here is RELATIVE units; swap in your provider's real rates and
  re-run before you decide. Trust the SHAPE, verify the digits against your bill.
==============================================================================
================================================================================
"""

from dataclasses import dataclass, replace
from typing import Optional


# ---------------------------------------------------------------------------
# Pricing, in RELATIVE cost units per token. The public shape of 2025-2026
# prompt caching on a frontier model: a fresh input token is the unit; laying
# down a cacheable prefix costs a ~1.25x write premium once; reading from that
# prefix later costs ~0.1x (an order of magnitude cheaper); output is billed the
# same no matter how the input got there. These are units, not dollars -- plug
# in your provider's real per-token rates to get a real bill.
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Pricing:
    fresh_input: float = 1.0
    cache_write: float = 1.25   # one-time premium to write a stable prefix
    cache_read: float = 0.10    # every subsequent read of that prefix
    output: float = 5.0         # same for all strategies; carried for honesty


# ---------------------------------------------------------------------------
# The workload: one small, stable corpus answered many times. Token counts are
# illustrative but in a realistic ratio (a few-thousand-token corpus, a short
# query, a top-k slice an order of magnitude smaller than the corpus, a couple-
# hundred-token answer, a fixed instruction block).
#
# `volatile_tokens` models the cache-breaking pitfall: per-request content (a
# timestamp, a tool result) that slips INTO the cached prefix. It changes every
# call, so it can never cache -- it is billed fresh every time, defeating the
# point. Default 0 (a clean prefix); Experiment 2 raises it.
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Workload:
    corpus_tokens: int = 4000        # the whole knowledge base (small + stable)
    query_tokens: int = 40           # the user's question (volatile, never cached)
    retrieved_tokens: int = 600      # the top-k slice RAG actually sends
    answer_tokens: int = 250         # generated output, billed per strategy
    instruction_tokens: int = 200    # system prompt / fixed prefix
    volatile_tokens: int = 0         # per-request content that breaks the prefix


# ---------------------------------------------------------------------------
# Per-strategy cost to answer `n` queries against the SAME corpus.
#
# The key modelling choice is WHICH tokens are cacheable. A stable prefix can be
# cached; anything that changes per request cannot. The query always changes, so
# it is always fresh. The RETRIEVED slice changes per query too (different
# queries pull different chunks), so RAG cannot cache its context -- that is the
# subtlety the essay leans on. CAG's whole trick is that its context (the corpus)
# is the SAME every call, so it lives in the cacheable prefix.
# ---------------------------------------------------------------------------
def cost_rag(n: int, w: Workload, p: Pricing) -> dict:
    """Retrieve top-k per query; send only those chunks, fresh every time.
    Instructions are a stable prefix (cache once, read thereafter). The
    retrieved slice and the query are volatile, so they are fresh each call."""
    # instructions: one write, then (n-1) reads
    instr = w.instruction_tokens * p.cache_write + \
        w.instruction_tokens * p.cache_read * (n - 1)
    # retrieved slice + query: fresh on every one of the n calls
    per_query_fresh = (w.retrieved_tokens + w.query_tokens) * p.fresh_input
    input_cost = instr + per_query_fresh * n
    output_cost = w.answer_tokens * p.output * n
    return {"input": input_cost, "output": output_cost,
            "total": input_cost + output_cost}


def cost_long_context(n: int, w: Workload, p: Pricing) -> dict:
    """Stuff the whole corpus in on every request at the FRESH rate (the naive
    'just use the big window' baseline -- no caching at all). Instructions and
    corpus are re-read fresh every single call. Volatile tokens (if any) are
    fresh here too, but long-context never caches anyway, so they cost the same
    as any other input token."""
    per_query_fresh = (w.instruction_tokens + w.corpus_tokens + w.query_tokens +
                       w.volatile_tokens) * p.fresh_input
    input_cost = per_query_fresh * n
    output_cost = w.answer_tokens * p.output * n
    return {"input": input_cost, "output": output_cost,
            "total": input_cost + output_cost}


def cost_cag(n: int, w: Workload, p: Pricing) -> dict:
    """Stuff the whole corpus in ONCE as a cacheable prefix (a write), then read
    it from cache on every subsequent call. This is the prompt-caching shadow of
    Cache-Augmented Generation: the corpus is the stable prefix; only the query
    is fresh. The instruction block sits in the same cacheable prefix.

    Volatile tokens model the pitfall: anything that changes per request cannot
    cache, so it is billed FRESH on every call even though it sits 'in' the
    prefix. With volatile_tokens=0 (the clean case) this is the pure CAG win."""
    prefix = w.instruction_tokens + w.corpus_tokens
    # the stable prefix: one write, then (n-1) cache reads
    prefix_cost = prefix * p.cache_write + prefix * p.cache_read * (n - 1)
    # the query is volatile -> fresh on every call
    query_cost = w.query_tokens * p.fresh_input * n
    # volatile tokens that slipped into the prefix: never cache -> fresh each call
    volatile_cost = w.volatile_tokens * p.fresh_input * n
    input_cost = prefix_cost + query_cost + volatile_cost
    output_cost = w.answer_tokens * p.output * n
    return {"input": input_cost, "output": output_cost,
            "total": input_cost + output_cost}


STRATEGIES = {
    "rag": cost_rag,
    "long_context": cost_long_context,
    "cag": cost_cag,
}


def table(n: int, w: Workload, p: Pricing) -> None:
    rows = {name: fn(n, w, p) for name, fn in STRATEGIES.items()}
    base = rows["rag"]["total"]
    header = (f"  {'strategy':<14}  {'input cost':>10}  {'output cost':>11}  "
              f"{'total':>8}   {'vs RAG':>6}")
    print(header)
    print("  " + "-" * (len(header) - 2))
    for name, c in rows.items():
        ratio = c["total"] / base if base else float("inf")
        print(f"  {name:<14}  {c['input']:>10.1f}  {c['output']:>11.1f}  "
              f"{c['total']:>8.1f}   {ratio:>5.2f}x")


def first_crossover(beats: str, against: str, w: Workload, p: Pricing,
                    cap: int = 100000) -> Optional[int]:
    """Smallest n at which strategy `beats` has a lower total than `against`."""
    for n in range(1, cap + 1):
        if STRATEGIES[beats](n, w, p)["total"] < \
                STRATEGIES[against](n, w, p)["total"]:
            return n
    return None


if __name__ == "__main__":
    line = "=" * 78
    p = Pricing()
    w = Workload()

    print(line)
    print("PROMPT-CACHING ECONOMICS  -  RAG vs long-context vs CAG")
    print(line)
    print("Pricing (relative cost units, per token):")
    print(f"  fresh input        {p.fresh_input:>6.3f}   "
          "(a token the model has never seen this call)")
    print(f"  cache write        {p.cache_write:>6.3f}   "
          "(first time a stable prefix is laid down)")
    print(f"  cache read         {p.cache_read:>6.3f}   "
          "(a token served from a cached prefix)")
    print(f"  output             {p.output:>6.3f}   "
          "(billed the same for every strategy)")
    print()
    print("Workload:")
    print(f"  corpus            {w.corpus_tokens:>6} tokens "
          "(small + stable: a candidate for CAG)")
    print(f"  query             {w.query_tokens:>6} tokens")
    print(f"  retrieved (top-k) {w.retrieved_tokens:>6} tokens "
          "(the slice RAG sends per query)")
    print(f"  answer            {w.answer_tokens:>6} tokens")
    print(f"  fixed instructions{w.instruction_tokens:>6} tokens "
          "(system prompt, stable prefix)")
    print(f"  volatile tokens   {w.volatile_tokens:>6} tokens "
          "(per-request content inside the prefix)")
    print()

    print("-" * 78)
    print("COST TO ANSWER 1 QUERY")
    print("-" * 78)
    table(1, w, p)
    print()
    print("  At 1 query, CAG is WORST: you paid the cache-write premium and "
          "got one read.")
    print("  The whole point of CAG is amortization, so 1 query is exactly "
          "the wrong test.")
    print()

    print("-" * 78)
    print("COST TO ANSWER 100 QUERIES (same stable corpus)")
    print("-" * 78)
    table(100, w, p)
    print()
    print("  Now CAG has amortized the write across 100 cache reads and "
          "overtaken RAG,")
    print("  while long_context (no caching) stays ~2.9x the bill. The corpus "
          "is small")
    print("  and stable, so CAG buys you long-context's simplicity below "
          "RAG's cost.")
    print()

    print("-" * 78)
    print("THE CROSSOVER  (where CAG total drops below long-context, and "
          "below RAG)")
    print("-" * 78)
    cag_vs_lc = first_crossover("cag", "long_context", w, p)
    cag_vs_rag = first_crossover("cag", "rag", w, p)
    if cag_vs_lc is not None:
        print(f"  CAG beats naive long-context after  {cag_vs_lc} query.")
    else:
        print("  CAG never beats naive long-context on this workload.")
    if cag_vs_rag is not None:
        print(f"  CAG beats RAG after {cag_vs_rag} queries.")
    else:
        r1 = cost_cag(1, w, p)["total"] / cost_rag(1, w, p)["total"]
        r100 = cost_cag(100, w, p)["total"] / cost_rag(100, w, p)["total"]
        print("  CAG never beats RAG on this workload (RAG sends far fewer "
              "input tokens),")
        print(f"  but it closes from {r1:.2f}x at 1 query to {r100:.2f}x "
              "at 100.")
    print()

    print("-" * 78)
    print("WHAT MOVES THE ANSWER: corpus size sweep (100 queries, "
          "cache read = 0.10x)")
    print("-" * 78)
    header = (f"  {'corpus tokens':>13}   {'rag total':>9}     "
              f"{'long_context':>12}     {'cag total':>9}     winner")
    print(header)
    print("  " + "-" * (len(header) - 2))
    for corpus in (1000, 4000, 16000, 64000, 256000):
        ws = replace(w, corpus_tokens=corpus)
        totals = {name: fn(100, ws, p)["total"]
                  for name, fn in STRATEGIES.items()}
        winner = min(totals, key=totals.get)
        print(f"  {corpus:>13}   {totals['rag']:>9.1f}     "
              f"{totals['long_context']:>12.1f}     "
              f"{totals['cag']:>9.1f}     {winner}")
    print()
    print("  Reading the sweep:")
    print("    - long_context cost grows LINEARLY with corpus size on every "
          "query: it")
    print("      re-reads the whole corpus fresh each time. This is the "
          "'just stuff it'")
    print("      tax, and it is brutal as the corpus grows.")
    print("    - CAG grows too (the cached corpus is still read, at 0.1x, "
          "every query),")
    print("      but ~10x slower than long_context. For a SMALL stable corpus "
          "it is a")
    print("      great deal; for a large one the 0.1x reads still add up.")
    print("    - RAG stays cheapest here because it sends only the top-k "
          "slice. Its cost")
    print("      is driven by k, not by corpus size -- which is exactly why a "
          "massive")
    print("      corpus points at RAG.")
    print()

    # ----- Experiment 1: kill the cache discount -----------------------------
    print("-" * 78)
    print("EXPERIMENT 1 -- KILL THE CACHE DISCOUNT (cache read 0.10x -> 1.00x, "
          "100 queries)")
    print("-" * 78)
    p_nodisc = replace(p, cache_read=1.0)   # caching now saves nothing
    print(f"  {'strategy':<14}     {'with cache (0.10x)':>18}     "
          f"{'no discount (1.00x)':>19}")
    print("  " + "-" * 60)
    for name, fn in STRATEGIES.items():
        with_cache = fn(100, w, p)["total"]
        no_disc = fn(100, w, p_nodisc)["total"]
        print(f"  {name:<14}     {with_cache:>18.1f}     {no_disc:>19.1f}")
    cag_nodisc = cost_cag(100, w, p_nodisc)["total"]
    lc_total = cost_long_context(100, w, p)["total"]
    print()
    print(f"  Without the discount CAG ({cag_nodisc:.1f}) collapses toward "
          f"naive long-context")
    print(f"  ({lc_total:.1f}): preloading once buys you nothing if every read "
          "is full price.")
    print("  The 0.1x cached read is not a detail -- it is the load-bearing "
          "wall that")
    print("  makes CAG viable. Before cheap cached reads existed, CAG was just "
          "expensive")
    print("  long-context.")
    print()

    # ----- Experiment 2: break the prefix with volatile tokens ---------------
    print("-" * 78)
    print("EXPERIMENT 2 -- BREAK THE PREFIX (volatile tokens inside the cached "
          "prefix)")
    print("-" * 78)
    print("  Per-request tool results / timestamps that slip INTO the cached "
          "prefix get")
    print("  billed FRESH every call (they change, so they never cache). Watch "
          "CAG's")
    print("  100-query advantage over RAG erode as the volatile block grows:")
    print()
    print(f"  {'volatile tokens':>15}   {'cag total':>9}     {'vs RAG':>6}     "
          "CAG still wins?")
    print("  " + "-" * 58)
    rag_total = cost_rag(100, w, p)["total"]
    for vol in (0, 100, 300, 1000):
        wv = replace(w, volatile_tokens=vol)
        cag_total = cost_cag(100, wv, p)["total"]
        ratio = cag_total / rag_total
        wins = "yes" if cag_total < rag_total else "no"
        print(f"  {vol:>15}   {cag_total:>9.1f}     {ratio:>5.2f}x     {wins}")
    print()
    print("  A clean, stable prefix is the whole assumption. The moment "
          "something volatile")
    print("  (a timestamp, a per-request id, a changing tool result) lands in "
          "the prefix,")
    print("  you pay fresh-input rates for the corpus you thought was nearly "
          "free -- and")
    print("  CAG's edge evaporates. Keep volatile content LAST, after the "
          "final cache point.")
    print()

    # ----- The decision matrix -----------------------------------------------
    print(line)
    print("THE DECISION MATRIX  (two axes -- size and volatility -- plus cost "
          "as tiebreaker)")
    print(line)
    print(f"  {'corpus shape':<36} ->  {'pick':<11} because")
    print("  " + "-" * 76)
    rows = [
        ("massive (bigger than the window)", "RAG",
         "only RAG's cost tracks k,", "not corpus size"),
        ("fast-moving (changes often)", "RAG",
         "a moving corpus recomputes", "the cache constantly"),
        ("private (per-user access filtering)", "RAG",
         "only retrieval can filter", "before the prompt"),
        ("small + stable + reused a lot", "CAG",
         "the write amortizes over", "cheap 0.1x reads"),
        ("tiny, or queried rarely", "long_context",
         "caching machinery is not", "worth the bother"),
        ("mid-size + stable", "long_context",
         "fits the window, too big", "to want cached per call"),
    ]
    for shape, pick, why1, why2 in rows:
        print(f"  {shape:<36} ->  {pick:<11} {why1}")
        print(f"  {'':<36}     {'':<11} {why2}")
    print()
    print("  The meta-point (the spine of the whole series): the question is "
          "no longer")
    print("  \"RAG or not?\" answered once and globally. It is \"what does THIS "
          "corpus want?\"")
    print("  answered from its size, its volatility, and your traffic. "
          "Sometimes the")
    print("  corpus demands a full retrieval pipeline; sometimes it just wants "
          "to be")
    print("  cached. Knowing the difference is the last skill this series had "
          "to teach.")
    print()
    print("  The pricing here is RELATIVE units; swap in your provider's real "
          "rates and")
    print("  re-run before you decide. Trust the SHAPE, verify the digits "
          "against your bill.")
    print(line)
