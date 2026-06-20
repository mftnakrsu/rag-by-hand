# Part 16 — Building a RAG Agent

> A reason/act/observe (ReAct) loop that reads its own transcript, picks one of four tools each step, and decides its route at run time — not author time. The Applied Track opener.

[📖 Read the essay](https://www.mefby.com/essays/rag-agent) · [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mftnakrsu/rag-by-hand/blob/main/part-16-rag-agent/rag_agent.ipynb)

## What it covers
- The **ReAct loop** (Reason + Act): each step the controller reads the running transcript, emits a **Thought**, picks **one** Action (a tool call), gets an **Observation**, and repeats until it calls `finish()` — the route is decided one step at a time, not wired in advance.
- **Four tools** the agent acts through: `search_policy(query)` (the support corpus from Parts 6–12 — refunds, the E-4042 error), `search_products(query)` (a small **new** source with an Acme → Globex acquisition + warranty chain, so multi-hop is real), `calculator(expr)` (arithmetic, proving not everything needs retrieval), and `finish(answer)` (terminate).
- **Termination** with two honest exits: stop on `finish()` **or** hit a max-steps budget — the guard against an agent that repeats a failing action, loops between states, or never decides it's done.
- Three runnable traces: **(a) multi-hop** chains two retrievals (Acme → Globex → warranty) — the *exact* example Part 10 only toured in prose, now executed; **(b) no-retrieval** routes arithmetic to the calculator, touching the index zero times; **(c) routing** sends a policy question to the policy index, not products.
- A deterministic, rule-based `controller()` as the offline source of truth — exactly mirroring Part 15's `classify_complexity` (deterministic-here / trained-in-production); with an API key, `generate()` shows the real LLM-driven ReAct prompt shape but always falls through to the rule policy so the file runs offline.

## Files
- **`rag_agent.py`** — the single runnable script: two tiny corpora, the four tools, the real-LLM controller shape behind `generate()` / `build_react_prompt()`, the deterministic `controller()`, the `run_agent()` reason/act/observe loop, and the three worked runs printed Thought/Action/Observation line by line.
- **`rag_agent.ipynb`** — step-by-step notebook: a markdown *why* before each small code step, built cell by cell.

## Run it
```bash
# from the repo root; no dependencies, no API key, no network
python3 part-16-rag-agent/rag_agent.py             # runs offline
# optional, for the REAL embedder path: pip install sentence-transformers && RAG_REAL_EMBED=1 …
# optional: set OPENAI_API_KEY to see the real LLM-driven controller banner
```
Prefer it step by step? Open `rag_agent.ipynb` in Jupyter, or click **Open in Colab** above.

## Key idea
For fifteen parts the *pipeline* decided its own shape only at the edges — Part 10 gave it control flow, Part 15 gave it a conductor — but the steps were still ours to wire. An **agent** hands the wiring to the model: it reads the transcript, thinks, picks one tool, observes, and loops until it's done. That's what lets the multi-hop earbuds question actually run — one observation (*Globex acquired Acme*) feeds the next action (*Globex earbuds warranty*), chaining no single-pass pipeline can do. The trade is real and worth naming: because each step issues another model call, a multi-step agent costs more tokens and latency than one retrieval-and-answer pass, and that cost isn't known in advance — it depends on how many cycles the model decides to take, which is exactly why production systems impose a hard step cap. The loop traces back to ReAct (Yao et al., ICLR 2023; arXiv:2210.03629), with tool-augmented LMs anchored by related early work like Toolformer (Schick et al., arXiv:2302.04761).

## Offline by design
The whole demo runs with no network and no API key. A deterministic rule-based `controller()` stands in for a trained LLM router — every Thought/Action it picks is a rule you can read — and a deterministic lexical (keyword-overlap) retriever stands in for sentence-transformers, so output is reproducible. The real paths sit behind flags: set `OPENAI_API_KEY` and `generate()` prints a banner noting the real controller would drive the loop (it still falls through to the rule policy); set `RAG_REAL_EMBED=1` with `sentence-transformers` installed for the dense retriever — and only the printed scores change. The tools chosen, the hops taken, and every Thought/Action/Observation/Finish line are identical either way.

---
← [Part 15 — Adaptive RAG](../part-15-adaptive-rag/) · [Series index](../) · [Part 17 — Conversational RAG →](../part-17-conversational-rag/)
