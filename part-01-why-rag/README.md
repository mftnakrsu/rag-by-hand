# Part 1 — Why RAG Exists

> **Conceptual part, no code.** This chapter builds the intuition the rest of the
> series rests on. The first line of code arrives in Part 2.

📖 **Read the full essay:** https://www.mefby.com/essays/why-rag-exists

📓 **Prefer a notebook?** This chapter is also a (concept-only) notebook: [`why-rag.ipynb`](why-rag.ipynb) — or [open it in Colab](https://colab.research.google.com/github/mftnakrsu/rag-by-hand/blob/main/part-01-why-rag/why-rag.ipynb).

A large language model is a brilliant reasoner with **no access to your information
and no clock**. It answers confidently and wrong about your own documents and about
anything recent. RAG (Retrieval-Augmented Generation) fixes that by handing the
model the right information at the moment it answers.

## The four LLM limitations

An LLM is trained to do one thing: given some text, predict the next chunk of text.
Four limitations fall directly out of that single design choice.

1. **Hallucination** — it states false things with the same confidence as true
   ones, because it optimizes for *plausible text*, not *verified fact*. It has no
   internal notion of ground truth to check against.
2. **Knowledge cutoff** — its training data ends on a fixed date; everything after
   that is a blind spot. It cannot know about yesterday's incident.
3. **No private / proprietary knowledge** — it only ever saw public text. Your wiki,
   Slack, design docs, and policy PDFs were never in the training pile, so they are
   not in the model. A bigger model does not help; the information was never in the room.
4. **Context-window limits** — you can't just paste everything in. The data doesn't
   fit, sending it is wasteful (cost + latency per request), and burying the relevant
   line among irrelevant ones *degrades* the answer.

## The fix: Retrieve → Augment → Generate

> Fetch the most relevant pieces of your own data, place them into the model's
> prompt, and let it answer from *evidence you supplied* instead of from memory alone.

- **R — Retrieval:** search your knowledge for the pieces most relevant to the
  question and pull them out (supplies private + fresh data).
- **A — Augmentation:** insert those pieces into the prompt next to the question,
  with an instruction to answer from them (sidesteps the context-window wall).
- **G — Generation:** the model writes a fluent answer, now grounded in the supplied
  evidence rather than spun from probability (hallucination drops sharply).

We never retrain the model or touch its weights — we only change *what it sees at the
moment it answers*.

## Mental model: the open-book exam

Same student, same brain, same skill — the only difference is access to the source.

- **Closed-book exam = vanilla LLM:** guesses from memory in confident handwriting.
- **Open-book exam = RAG:** flips to the right page, lays it open beside the question,
  writes the answer in her own words.

Mapping: flip to the page **= retrieval**, lay it open beside the question
**= augmentation**, write in her own words **= generation**. The analogy predicts
RAG's failure modes too: if the textbook lacks the answer, or she flips to the wrong
page, she's confidently wrong — which is why **retrieval quality is the whole ballgame**.

## The two-phase pipeline

```
document → chunk → embed → store  │  retrieve → augment → generate
        INDEXING (once, up front) │  QUERYING (per question)
```

**Indexing** (done ahead of time, whenever documents change):

1. **Document** — start with raw source (PDF, wiki page, transcript, DB row).
2. **Chunk** — split into bite-sized passages (specific, yet self-contained).
3. **Embed** — convert each chunk into a vector that captures its *meaning*.
4. **Store** — save the vectors in a vector store built for fast similarity search.

**Querying** (done per question):

5. **Retrieve** — embed the question, ask the store for the closest-in-meaning chunks.
6. **Augment** — drop those chunks into the prompt with the question.
7. **Generate** — send the assembled prompt to the LLM for a grounded answer.

## When to reach for RAG

- **Bigger prompt** — tiny, known-in-advance knowledge (one short doc). Just copy-paste.
- **RAG** — large, changing, private, or you-don't-know-which-piece knowledge.
- **Fine-tuning** — to change *how the model talks* (behavior, format, style), not
  *what it knows*. One-liner: fine-tuning changes how it talks; RAG changes what it
  knows right now.

## Next up

**The code starts in Part 2 — Embeddings.** We slow down on the word "embed" and
answer the question the whole pipeline rests on: how do you turn the meaning of a
sentence into numbers, and why does that make search by meaning possible?
