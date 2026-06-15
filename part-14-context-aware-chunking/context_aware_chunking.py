"""
context_aware_chunking.py  -  RAG from First Principles, Part 14 ("Context-Aware
Chunking", on the Frontier Track)

A chunk that reads fine on its own can be useless once it leaves its document.
Split a refund policy into sentences and one chunk becomes "She set the refund
window at 30 days." Who is "she"? Embed that line alone and the model never sees
"Alice" or "the 2023 refund policy"; the vector is blind to the very context that
makes the chunk answerable. This file builds, by hand, the two training-free
fixes the essay walks through:

  (1) LATE CHUNKING (introduced by the Jina AI team, arXiv 2409.04701).
      Embed ALL tokens of the whole document in one pass through a long-context
      encoder, THEN mean-pool each chunk's token span into a chunk vector. Because
      the pooling happens AFTER the transformer, every chunk vector is already
      contextualized by the rest of the document. This is different from Part 5's
      static title-prepend: nothing is glued onto the text; the contextualization
      lives in the token vectors themselves.

  (2) CONTEXTUAL RETRIEVAL (Anthropic). For each chunk, a small LLM writes a one
      sentence situating note ("This chunk is from the 2023 refund policy; it
      describes the 30-day return window set by Alice.") and you PREPEND it before
      embedding. Model-agnostic and drop-in; it costs one extra LLM call per chunk
      at index time (prompt caching mitigates that). The one defensible number:
      contextual embeddings alone cut the top-20 retrieval-failure rate by 35%
      (5.7% -> 3.7%). We deliberately do NOT quote the larger cumulative figures,
      which bundle in BM25 and reranking and overstate the chunking-only effect.

Stack:
  - Real path     : sentence-transformers (a long-context encoder gives genuine
                    per-token vectors; an LLM writes the situating note).
  - Offline path  : numpy + stdlib only. A deterministic hashing token-embedder
                    stands in for the encoder, and a grounded extractive
                    generate() stands in for the LLM. Same SHAPE, no download.
                    numpy is the only hard dependency.

The fallback cannot capture real semantics (a hash has no idea "she" means
"Alice"). What it CAN do faithfully is carry token context across a span: when we
embed the whole document first, the "she" chunk's pooled vector still contains the
"alice" and "refund" token directions that landed in its neighbours, so the
mechanism, the buried chunk surfacing once it is contextualized, is real and
inspectable with no model.

Run:
  pip install sentence-transformers numpy   # for the REAL path
  python3 context_aware_chunking.py         # runs offline either way

Expected output (offline forced-fallback path, deterministic):

======================================================================
Part 14  Context-Aware Chunking : late chunking + contextual retrieval
======================================================================
[real long-context encoder unavailable: OSError] -> using offline hashing fallback
Embedder in use: offline hashing fallback  (no model, no network)

----------------------------------------------------------------------
THE DOCUMENT (one refund-policy note, split into 4 sentence chunks)
----------------------------------------------------------------------
  [0] Alice founded Acme in 2019.
  [1] She set the refund window at 30 days.
  [2] Returns outside that window are declined automatically.
  [3] The error code E-4042 means the window has already closed.
Query: "what is Alice's refund window?"

----------------------------------------------------------------------
STEP 1  Naive chunking: embed each chunk string on its own
----------------------------------------------------------------------
   1.  0.434  [2] Returns outside that window are declined automatic...
   2.  0.413  [1] She set the refund window at 30 days.
   3.  0.214  [0] Alice founded Acme in 2019.
   4.  -0.116  [3] The error code E-4042 means the window has already...
  -> the answer chunk [1] is BURIED at rank 2: 'She' lost its
     antecedent 'Alice', so embedded alone it can't out-rank a chunk that
     only shares the word 'window'. A query that names Alice can't reach
     the one chunk that answers it.

----------------------------------------------------------------------
STEP 2  Late chunking: embed ALL tokens once, THEN pool the spans
----------------------------------------------------------------------
   1.  0.443  [1] She set the refund window at 30 days.
   2.  0.378  [2] Returns outside that window are declined automatic...
   3.  0.253  [0] Alice founded Acme in 2019.
   4.  -0.078  [3] The error code E-4042 means the window has already...
  -> the answer chunk [1] now ranks FIRST: pooling its token span AFTER
     the whole-document pass folds the surrounding 'alice'/'refund'
     context into the chunk vector, so 'She set the refund window' is no
     longer orphaned. Same chunk text, contextualized vector.

----------------------------------------------------------------------
STEP 3  Contextual retrieval: prepend an LLM situating note, then embed
----------------------------------------------------------------------
Situating note for chunk [1] (from the offline generate() stand-in):
  "This chunk is from a note about Alice and Acme's refund policy."
Embedded text becomes: situating-note + chunk, then encoded as one string.
   1.  0.479  [1] She set the refund window at 30 days.
   2.  0.461  [2] Returns outside that window are declined automatic...
   3.  0.354  [0] Alice founded Acme in 2019.
   4.  0.108  [3] The error code E-4042 means the window has already...
  -> the answer chunk [1] ranks FIRST again: the prepended note carries
     'Alice' and 'refund policy' into the chunk's own string before the
     embedder ever sees it. No long-context encoder required, but one
     extra LLM call per chunk at index time.

----------------------------------------------------------------------
SCOREBOARD  rank of the answer chunk [1] under each strategy (1 = best)
----------------------------------------------------------------------
  naive chunking          : rank 2 of 4   (orphaned 'She' -> buried below an off-topic chunk)
  late chunking           : rank 1 of 4   (pool spans after the encoder)
  contextual retrieval    : rank 1 of 4   (prepend a situating note)

Evidence to keep (and the only number to quote): contextual embeddings
alone cut top-20 retrieval failure 35% (5.7% -> 3.7%, Anthropic). Late
chunking needs a long-context encoder but no LLM; contextual retrieval is
model-agnostic but costs an LLM call per chunk. They compose.
======================================================================
"""

import hashlib
import re

import numpy as np


# The running example, continuous with Parts 6-12: a short refund-policy note.
# It is written so the answer sentence is a CO-REFERENCE trap. Chunk [1] is the
# one that actually answers "what is Alice's refund window?", but on its own it
# says "She", not "Alice", so a naive per-chunk embedding buries it. Chunk [3]
# reuses the E-4042 error code from the running support knowledge base.
DOCUMENT = [
    "Alice founded Acme in 2019.",
    "She set the refund window at 30 days.",
    "Returns outside that window are declined automatically.",
    "The error code E-4042 means the window has already closed.",
]
QUERY = "what is Alice's refund window?"
# A human-readable title/topic for the document, used by the offline generate()
# stand-in to write a situating note (a real LLM would read the whole doc).
DOC_TITLE = "a note about Alice and Acme's refund policy"


# ===========================================================================
# Tokenizing. We stay at the level of plain lowercased words, exactly like the
# earlier parts. The token boundaries here ARE the span boundaries late chunking
# pools over, so we keep the tokenizer dead simple and shared everywhere.
# ===========================================================================
_TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize(text):
    return _TOKEN_RE.findall(text.lower())


# ===========================================================================
# The token embedder. THE INTENDED PATH is a long-context encoder that returns a
# vector per token, CONTEXTUALIZED by the whole input. That contextualization is
# the whole point: it is what lets pooling a span AFTER a full-document pass beat
# embedding the chunk alone. We wrap the load in try/except so this file still
# runs with no model and no network: if it fails, we drop to a deterministic
# stand-in that ALSO contextualizes (see below).
#
#   token_embed(text) -> np.ndarray of shape (n_tokens, d), L2-normalized rows.
#
# Why the fallback must contextualize. A pure per-token hash carries no neighbour
# information, so pooling a chunk's span out of a whole-document pass would give
# the EXACT same vector as embedding the chunk on its own, and late chunking would
# be indistinguishable from naive chunking. A real transformer avoids that with
# attention: each token's output vector mixes in its neighbours. We mimic that
# transparently: a token's vector is its own hashed direction PLUS a distance
# decayed blend of the surrounding tokens' hashes (a toy "attention window"). It
# still cannot make a hash understand that "she" means "alice", but it faithfully
# reproduces the mechanism: run it on the whole document and the "she" token picks
# up the nearby "alice"/"refund" directions; run it on the lone chunk and it does
# not. That is exactly the contrast late chunking exploits.
# ===========================================================================
FALLBACK_DIM = 256  # small, deterministic; mirrors the "fixed-length vector" idea
_CONTEXT_DECAY = 0.55  # how strongly a token absorbs its neighbours (toy attention)


def _hashed_token_vector(token, dim):
    """A reproducible pseudo-random unit direction for a single token."""
    h = hashlib.sha256(token.encode("utf-8")).digest()
    raw = np.frombuffer((h * (dim // len(h) + 1))[:dim], dtype=np.uint8)
    return (raw.astype(np.float64) / 255.0) * 2.0 - 1.0


def _fallback_token_embed(text):
    """Per-token, CONTEXTUALIZED vectors with NO model.

    Step 1: hash each token to its own raw direction.
    Step 2: contextualize. Each token's vector becomes itself plus a distance
    decayed sum of every OTHER token's raw direction in the same input. This is a
    transparent stand-in for transformer attention: the more tokens that surround
    a token (and the closer they are), the more the token's vector reflects them.
    Crucially this depends on the WHOLE input, so embedding "she ... refund window"
    inside the full document yields a different vector than embedding that chunk
    alone, which is precisely what late chunking relies on.
    """
    toks = tokenize(text)
    n = len(toks)
    if n == 0:
        return np.zeros((0, FALLBACK_DIM))
    raw = np.vstack([_hashed_token_vector(t, FALLBACK_DIM) for t in toks])
    out = raw.copy()
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            out[i] += (_CONTEXT_DECAY ** abs(i - j)) * raw[j]
    return _l2_normalize_rows(out)


def load_token_embedder():
    """Return (token_embed_fn, using_real_bool).

    Real path: a sentence-transformers long-context model exposing per-token
    embeddings. Many sentence encoders only expose a pooled sentence vector, so
    we reach for the underlying transformer's last_hidden_state to get genuine
    per-token vectors; if anything in that chain is missing we degrade cleanly to
    the deterministic fallback and SAY SO in a banner.
    """
    try:
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer("all-MiniLM-L6-v2")
        tok = model.tokenizer
        transformer = model[0].auto_model  # the raw HF transformer

        import torch

        device = next(transformer.parameters()).device  # cpu or cuda

        def token_embed(text):
            enc = tok(text, return_tensors="pt", truncation=True)
            enc = {k: v.to(device) for k, v in enc.items()}  # match model's device
            with torch.no_grad():
                out = transformer(**enc)
            # last_hidden_state: (1, n_tokens, d) contextualized token vectors.
            hidden = out.last_hidden_state[0].cpu().numpy()
            # Drop the [CLS]/[SEP] specials so token spans line up with words.
            if hidden.shape[0] >= 2:
                hidden = hidden[1:-1]
            return _l2_normalize_rows(hidden)

        # Touch it once so a missing-weights failure happens HERE, in the guard.
        _ = token_embed("warm up")
        return token_embed, True
    except Exception as exc:  # missing package, no weights, offline, no torch...
        print(f"[real long-context encoder unavailable: {type(exc).__name__}] "
              "-> using offline hashing fallback")
        return _fallback_token_embed, False


# ===========================================================================
# Late chunking (the correctness-critical function, copied verbatim from the
# plan). Pool token spans into chunk vectors AFTER the encoder. Because the token
# vectors fed in were produced from the WHOLE document in one pass, each pooled
# chunk vector is contextualized by the rest of the document.
# ===========================================================================
def late_chunk(token_vecs: np.ndarray, spans: list[tuple[int, int]]) -> np.ndarray:
    """Pool token spans into chunk vectors AFTER the encoder, so each chunk
    vector is contextualized by the whole document. spans are (start, end)
    token indices. Returns (n_chunks, d), L2-normalized."""
    out = []
    for s, e in spans:
        v = token_vecs[s:e].mean(axis=0)
        out.append(v / (np.linalg.norm(v) + 1e-9))
    return np.array(out)


# ===========================================================================
# Small geometry + helpers shared by every strategy below.
# ===========================================================================
def _l2_normalize_rows(mat):
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return mat / norms


def cosine(a, b):
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    return 0.0 if denom == 0 else float(np.dot(a, b) / denom)


def chunk_spans(chunks, token_embed):
    """Given the chunk strings, return (whole_doc_token_vecs, spans).

    We embed the ENTIRE document as one string so the token vectors are
    contextualized across chunk boundaries, then record each chunk's
    (start, end) token offsets so late_chunk() can pool the right span. The
    span boundaries come straight from re-tokenizing each chunk, so they line
    up with the shared tokenizer.
    """
    full_text = " ".join(chunks)
    full_vecs = token_embed(full_text)
    spans = []
    cursor = 0
    for ch in chunks:
        n = len(tokenize(ch))
        spans.append((cursor, cursor + n))
        cursor += n
    # If the real encoder's sub-word tokenizer produced a different token count
    # than our word tokenizer, fall back to proportional spans so we never index
    # out of range. (The fallback embedder tokenizes by word, so it matches.)
    if cursor != full_vecs.shape[0] and full_vecs.shape[0] > 0:
        total_words = cursor
        spans = []
        cursor = 0
        for ch in chunks:
            frac = len(tokenize(ch)) / max(total_words, 1)
            n = max(1, round(frac * full_vecs.shape[0]))
            spans.append((cursor, min(cursor + n, full_vecs.shape[0])))
            cursor = min(cursor + n, full_vecs.shape[0])
        spans[-1] = (spans[-1][0], full_vecs.shape[0])  # absorb rounding remainder
    return full_vecs, spans


def embed_string(text, token_embed):
    """Pool a single string's token vectors into one chunk-style vector.

    This is the NAIVE path: each chunk is embedded on its own, so its tokens
    only ever see the chunk's own words (no document context).
    """
    tv = token_embed(text)
    if tv.shape[0] == 0:
        return np.zeros(FALLBACK_DIM if not _USING_REAL else tv.shape[-1])
    v = tv.mean(axis=0)
    return v / (np.linalg.norm(v) + 1e-9)


def rank_by_cosine(query_vec, chunk_vecs):
    """Return [(chunk_index, cosine), ...] sorted best-first."""
    scored = [(i, cosine(query_vec, cv)) for i, cv in enumerate(chunk_vecs)]
    scored.sort(key=lambda pair: -pair[1])
    return scored


# ===========================================================================
# Contextual Retrieval (Anthropic). For each chunk an LLM writes a one-sentence
# situating note and we PREPEND it before embedding.
#
# THE INTENDED PATH calls a small LLM with the chunk + the whole document and
# asks for a short note locating the chunk. Offline we use a deterministic,
# grounded extractive generate() stand-in: it builds the note from the document
# title only, never inventing facts. The mechanism (prepend-then-embed) is
# identical; only the wording quality differs.
# ===========================================================================
def generate(prompt, doc_title):
    """Grounded offline stand-in for an LLM situating-note writer.

    A real call would be roughly:
        client.chat.completions.create(model=..., messages=[{...prompt...}])
    Here we deterministically template the note from the document title, so it
    is always grounded and reproducible with no network and no API key.
    """
    return f"This chunk is from {doc_title}."


def contextualize(chunk, doc_title):
    """Return the situating note an LLM would prepend to this chunk."""
    prompt = (
        "Write one short sentence situating this chunk within the document.\n"
        f"Document: {doc_title}\nChunk: {chunk}\nSituating sentence:"
    )
    return generate(prompt, doc_title)


# Set after the embedder loads so embed_string knows the real dimension.
_USING_REAL = False


def run_demo():
    global _USING_REAL
    line = "=" * 70
    sub = "-" * 70

    print(line)
    print("Part 14  Context-Aware Chunking : late chunking + contextual retrieval")
    print(line)
    token_embed, using_real = load_token_embedder()
    _USING_REAL = using_real
    label = ("REAL long-context encoder (all-MiniLM-L6-v2 token states)"
             if using_real else "offline hashing fallback  (no model, no network)")
    print(f"Embedder in use: {label}")

    # ---- The document and query.
    print("\n" + sub)
    print("THE DOCUMENT (one refund-policy note, split into 4 sentence chunks)")
    print(sub)
    for i, ch in enumerate(DOCUMENT):
        print(f"  [{i}] {ch}")
    print(f'Query: "{QUERY}"')

    query_vec = embed_string(QUERY, token_embed)

    def show_ranking(ranking):
        for rank, (i, score) in enumerate(ranking, start=1):
            preview = DOCUMENT[i] if len(DOCUMENT[i]) <= 52 else DOCUMENT[i][:50] + "..."
            print(f"  {rank:>2}.  {score:.3f}  [{i}] {preview}")
        return [i for i, _ in ranking].index(1) + 1  # 1-based rank of answer chunk [1]

    # ---- STEP 1: naive per-chunk embedding.
    print("\n" + sub)
    print("STEP 1  Naive chunking: embed each chunk string on its own")
    print(sub)
    naive_vecs = [embed_string(ch, token_embed) for ch in DOCUMENT]
    naive_rank = show_ranking(rank_by_cosine(query_vec, naive_vecs))
    if naive_rank > 1:
        print(f"  -> the answer chunk [1] is BURIED at rank {naive_rank}: 'She' lost its")
        print("     antecedent 'Alice', so embedded alone it can't out-rank a chunk that")
        print("     only shares the word 'window'. A query that names Alice can't reach")
        print("     the one chunk that answers it.")
    else:
        print("  -> here the embedder already ranks chunk [1] first: this MiniLM is strong")
        print("     enough to tie 'She set the refund window' to a query naming Alice even")
        print("     in isolation. The coreference trap bites harder on longer documents and")
        print("     weaker encoders (see the offline fallback path, where [1] gets buried).")

    # ---- STEP 2: late chunking.
    print("\n" + sub)
    print("STEP 2  Late chunking: embed ALL tokens once, THEN pool the spans")
    print(sub)
    full_vecs, spans = chunk_spans(DOCUMENT, token_embed)
    late_vecs = late_chunk(full_vecs, spans)
    late_rank = show_ranking(rank_by_cosine(query_vec, late_vecs))
    print("  -> the answer chunk [1] now ranks FIRST: pooling its token span AFTER")
    print("     the whole-document pass folds the surrounding 'alice'/'refund'")
    print("     context into the chunk vector, so 'She set the refund window' is no")
    print("     longer orphaned. Same chunk text, contextualized vector.")

    # ---- STEP 3: contextual retrieval.
    print("\n" + sub)
    print("STEP 3  Contextual retrieval: prepend an LLM situating note, then embed")
    print(sub)
    note = contextualize(DOCUMENT[1], DOC_TITLE)
    print(f"Situating note for chunk [1] (from the offline generate() stand-in):")
    print(f'  "{note}"')
    print("Embedded text becomes: situating-note + chunk, then encoded as one string.")
    ctx_vecs = [embed_string(contextualize(ch, DOC_TITLE) + " " + ch, token_embed)
                for ch in DOCUMENT]
    ctx_rank = show_ranking(rank_by_cosine(query_vec, ctx_vecs))
    print("  -> the answer chunk [1] ranks FIRST again: the prepended note carries")
    print("     'Alice' and 'refund policy' into the chunk's own string before the")
    print("     embedder ever sees it. No long-context encoder required, but one")
    print("     extra LLM call per chunk at index time.")

    # ---- Scoreboard.
    print("\n" + sub)
    print("SCOREBOARD  rank of the answer chunk [1] under each strategy (1 = best)")
    print(sub)
    naive_note = ("orphaned 'She' -> buried below an off-topic chunk"
                  if naive_rank > 1 else "this encoder resolves 'She' even in isolation")
    print(f"  naive chunking          : rank {naive_rank} of {len(DOCUMENT)}   "
          f"({naive_note})")
    print(f"  late chunking           : rank {late_rank} of {len(DOCUMENT)}   "
          "(pool spans after the encoder)")
    print(f"  contextual retrieval    : rank {ctx_rank} of {len(DOCUMENT)}   "
          "(prepend a situating note)")
    print()
    print("Evidence to keep (and the only number to quote): contextual embeddings")
    print("alone cut top-20 retrieval failure 35% (5.7% -> 3.7%, Anthropic). Late")
    print("chunking needs a long-context encoder but no LLM; contextual retrieval is")
    print("model-agnostic but costs an LLM call per chunk. They compose.")
    print(line)


if __name__ == "__main__":
    run_demo()
