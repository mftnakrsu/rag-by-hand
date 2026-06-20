# Part 20 — Conversational RAG

> The agent answered one question and stopped — it remembered nothing. A real chat app gets follow-ups: "What about damaged items?", "And how long does that take?" — fragments that mean nothing on their own. Give the retriever a memory, and condense each follow-up into a standalone query before it ever touches the index.

[📖 Read the essay](https://www.mefby.com/essays/conversational-rag) · [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mftnakrsu/rag-by-hand/blob/main/part-20-conversational-rag/conversational_rag.ipynb)

## What it covers
- **Query condensation** (history-aware query rewriting): before retrieving, rewrite a context-dependent follow-up into a **self-contained, standalone query** using the conversation so far. `"What about damaged items?"` → `"refund policy for damaged items"`. The standalone query is what hits the index; the raw fragment never does.
- **Conversation memory**: a rolling transcript of `(user, assistant)` turns the condenser reads to recover the missing topic.
- **Coreference / ellipsis resolution**: resolve pronouns (`"that"`, `"it"`) and elliptical follow-ups (`"what about …?"`, `"and how long does that take?"`) against the last topic in the history.
- **Three turns in one conversation**, each printed raw → condensed → retrieved → answer: **turn 1** `"What's our refund window?"` is already standalone (no rewrite → "30 days"); **turn 2** `"What about damaged items?"` borrows `refund` from history → hits the damaged-items clause; **turn 3** `"And how long does that refund take to process?"` resolves `that` → refund → retrieves the timing chunk.
- **The contrast that is the lesson**: turn 2 run **without** condensation retrieves the raw fragment and lands on the wrong, no-refund distractor (a miss); **with** condensation it lands on the damaged-items refund clause (a hit).
- A deterministic, rule-based `condense()` as the offline source of truth — exactly mirroring Part 15's `classify_complexity` and Part 19's `controller` (deterministic-here / trained-LLM-in-production); with an API key, `generate()` / `build_condense_prompt()` shows the real LLM-driven condensation prompt shape but always falls through to the rule rewriter so the file runs offline.

## Files
- **`conversational_rag.py`** — the single runnable script: the support KB carried from Parts 6–12 (with a new damaged-items refund clause and a deliberate no-refund distractor), the lexical retriever, the real-LLM condenser shape behind `generate()` / `build_condense_prompt()`, the deterministic `condense()` rewriter, conversation memory, and the three-turn conversation plus the with/without contrast printed line by line.
- **`conversational_rag.ipynb`** — step-by-step notebook: a markdown *why* before each small code step, built cell by cell.

## Run it
```bash
# from the repo root; no dependencies, no API key, no network
python3 part-20-conversational-rag/conversational_rag.py     # runs offline
# optional, for the REAL embedder path: pip install sentence-transformers && RAG_REAL_EMBED=1 …
# optional: set OPENAI_API_KEY to see the real LLM-driven condenser banner
```
Prefer it step by step? Open `conversational_rag.ipynb` in Jupyter, or click **Open in Colab** above.

## Key idea
A one-shot retriever sees only the current query, so a follow-up like `"What about damaged items?"` — which never says the word *refund* — can match no refund chunk. The fix isn't a bigger index; it's **reading the conversation before you retrieve**. Condensation splices the salient topic from recent turns into the follow-up, producing a standalone query the index can actually answer. This pattern — variously called *query condensation*, *condense question*, *history-aware retrieval*, or *standalone-question rewriting* — was popularized in production by conversational retrieval chains (notably LangChain's `ConversationalRetrievalChain` condense-question step, since refactored into the `create_history_aware_retriever` helper). It rests on the academic line of **Conversational Query Rewriting** — rewriting a context-dependent turn into a self-contained question before retrieval, as in CANARD (Elgohary et al., EMNLP 2019) and the rewrite-then-retrieve-then-read formalization in QReCC (Anantha et al., NAACL 2021). The catchy names are practitioner vocabulary, not a single paper's coinage. The trade is worth naming: the rewrite is an **extra model call on the critical path** — it adds latency and a new failure point, and it can **over-rewrite** (invent a constraint the user never said), **under-rewrite** (leave a dangling pronoun), or pick the wrong antecedent and silently send retrieval after the wrong entity. And a genuinely fresh, standalone question must **not** be condensed — splicing a stale topic into it only pollutes the retrieval.

## Offline by design
The whole demo runs with no network and no API key. A deterministic, rule-based `condense()` stands in for a trained LLM rewriter — every splice and pronoun resolution is a rule you can read — and a deterministic lexical (keyword-overlap) retriever stands in for sentence-transformers, so output is reproducible. The real paths sit behind flags: set `OPENAI_API_KEY` and `generate()` / `build_condense_prompt()` print the real condensation prompt banner (the code still falls through to the rule rewriter); set `RAG_REAL_EMBED=1` with `sentence-transformers` installed for the dense retriever — and only the printed scores change. Every raw query, condensed query, retrieved chunk, and answer line is identical either way.

---
← [Part 19 — Building a RAG Agent](../part-19-rag-agent/) · [Series index](../)
