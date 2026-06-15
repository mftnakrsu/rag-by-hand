"""
Late-interaction retrieval by hand: token-level multi-vectors scored by MaxSim.
RAG from First Principles, Part 13 (Frontier Track): the quality paradigm that
sits between single-vector dense retrieval (Part 7) and cross-encoder reranking
(Part 8), extended to document page IMAGES by ColPali.

The core series ended at Part 12. This is optional, 2026-frontier material you
reach for once the core is solid.

The idea, in one breath: a single pooled vector per passage (Part 7) averages a
whole passage into one point, so a query term that matches ONE word in a long
passage gets washed out. Late interaction instead keeps a vector PER TOKEN, for
both the query and the doc, and scores a (query, doc) pair with MaxSim:

    for each query token, take its MAX similarity over all doc tokens, then SUM.

You get cross-encoder-style fine-grained term matching, but the doc vectors are
PRECOMPUTED offline (unlike a cross-encoder, which must run the model on every
query-doc pair at query time). "Cross-encoder quality, bi-encoder serving cost."

This file walks the exact ladder the essay climbs:
  1. token_embed(text)            -> a vector PER TOKEN (not one pooled vector)
  2. maxsim(query_vecs, doc_vecs) -> the late-interaction score, ~3 lines numpy
  3. rank by MaxSim vs by pooled cosine on a term-match query (they disagree)
  4. storage_report(...)          -> the per-token storage cost + compression
  5. colpali_patch_demo()         -> ColPali as a MECHANISM: a toy "page" of
                                     patch vectors scored by the SAME MaxSim

Stack:
  - Real path     : sentence-transformers all-MiniLM-L6-v2 token embeddings
                    (the model's per-token hidden states, before pooling).
  - Offline path  : a transparent deterministic hashing embedder -- any token
                    -> a reproducible unit vector, numpy only, no network, no
                    weights. Same pattern as Part 2. It is NOT a real model: a
                    hash has no idea "refund" and "reimbursement" are cousins.
                    It exists so the SHAPE of late interaction is runnable and
                    inspectable. A printed banner always names the active path.

ColPali is taught strictly as a MECHANISM with a toy multi-vector stand-in. No
real vision-language model runs offline here; the patch grid is hashed numbers
standing in for a page image's patch embeddings, run through the same MaxSim.

Citations (re-verified): ColBERT (Khattab & Zaharia, SIGIR 2020, arXiv 2004.12832);
ColBERTv2 (Santhanam et al., NAACL 2022, arXiv 2112.01488); ColPali (Faysse et
al., ICLR 2025, arXiv 2407.01449).

Run:
  pip install sentence-transformers numpy   # for the REAL token embeddings
  python3 late_interaction.py               # runs fully offline either way

================================ Expected output ================================
(Forced-offline run: HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HOME=<empty>.)
(If sentence-transformers is installed but its weights are absent, the library
may print a warning to stderr, e.g. "No sentence-transformers model found ...
Creating a new one with mean pooling.", just before the banner below. It does
not affect the exit code or correctness; the fallback banner still appears.)

======================================================================
  [real model unavailable: OSError] -> using offline fallback
Embedder in use: offline hashing fallback (deterministic, numpy only)
======================================================================

======================================================================
STEP 1  One pooled vector vs a vector PER TOKEN
======================================================================
Doc: "Refund requests are processed within five business days."
  pooled vector : 1 vector  x 384 dims      (the whole passage -> one point)
  multi-vector  : 8 vectors x 384 dims      (one vector per token, nothing averaged away)

======================================================================
STEP 2 & 3  MaxSim vs pooled cosine on a term-match query
======================================================================
Query: "E-4042 error"

Ranking by POOLED cosine (one vector per doc, Part 7):
  rank 1  cos=+0.503  A general error occurred while loading the page.... [DISTRACTOR]
  rank 2  cos=+0.151  Standard shipping takes three to five business days to ...
  rank 3  cos=+0.123  Our internal billing log emits the diagnostic identifie... [exact term]
  rank 4  cos=-0.104  Refunds are accepted within 30 days of purchase if the ...
  -> the exact-term doc is NOT at the top: pooling a long passage averaged the
     rare E-4042 token away, and a short generic "error" doc pooled higher.

Ranking by MaxSim (token-level late interaction):
  rank 1  maxsim=2.269  Our internal billing log emits the diagnostic identifie... [exact term]
  rank 2  maxsim=1.766  A general error occurred while loading the page.... [DISTRACTOR]
  rank 3  maxsim=0.947  Standard shipping takes three to five business days to ...
  rank 4  maxsim=0.660  Refunds are accepted within 30 days of purchase if the ...
  -> late interaction surfaces the exact-term doc: the query token "e4042" finds
     its exact match among the doc tokens, and MaxSim rewards that one strong hit
     even though it is buried in a long, otherwise-unrelated passage.

======================================================================
STEP 4  The storage tradeoff (toy numbers, then MS MARCO scale)
======================================================================
Toy corpus: 4 docs, ~12 tokens/doc, 384 dims, float32 (4 bytes/dim).
  pooled (1 vec/doc)      :        6,144 bytes  (   0.01 MB)
  multi-vector (1 vec/tok):       73,728 bytes  (   0.07 MB)  -> 12.0x more
  multi-vector, 2-bit     :        4,608 bytes  (   0.00 MB)  -> ColBERTv2 residual compression
At MS MARCO scale the ColBERTv2 paper reports the vanilla multi-vector index at
~154 GiB, shrunk to ~16 GiB (1-bit) / ~25 GiB (2-bit) by residual compression
(a 6-10x reduction). Per-token vectors cost more; compression makes it practical.

======================================================================
STEP 5  ColPali as a MECHANISM: MaxSim over page-image patches (toy stand-in)
======================================================================
NOTE: no real vision-language model runs offline. This is a TOY: a 4x4 grid of
hashed "patch" vectors stands in for a document page image's patch embeddings.
A real ColPali/ColQwen embeds the page image into ~1024 patch vectors and skips
OCR and chunking entirely; the scoring is the SAME MaxSim you built above.
  toy page: 16 patch vectors x 384 dims (standing in for an image's patches)
  query "refund window policy" (3 tokens) vs the page patches:
    maxsim(query tokens, page patches) = 1.062
  -> mechanically identical to text late interaction: each query token takes its
     max over the patch vectors, then we sum. ColPali just makes the doc tokens
     be image patches instead of text tokens.

======================================================================
Takeaway: keep a vector per token, score with MaxSim, precompute the doc
vectors offline. You buy fine-grained term matching at bi-encoder serving cost,
paying in storage (which compression brings back down). ColPali pushes the same
trick onto page images, retrieving documents without OCR or chunking.
Next, Part 14: Context-Aware Chunking.
======================================================================
"""

import hashlib
import re

import numpy as np

# ---------------------------------------------------------------------------
# The running example. We reuse the support knowledge base from Parts 6-12: the
# refund-policy line, a shipping line, and the E-4042 error code -- but here the
# error-code line is LONG and buries E-4042 deep among unrelated words, the way
# a real passage would. The query is a TERM-MATCH query ("E-4042 error"), and
# the DISTRACTOR is a short, generic "error" sentence that shares the topical
# word but NOT the rare code. This is exactly the case where a single pooled
# vector struggles (it averages the rare token away, and the short generic doc
# pools higher) while late interaction's per-token MaxSim catches the exact hit.
# ---------------------------------------------------------------------------
CORPUS = [
    # The exact-term doc: E-4042 is buried inside a long, mostly-unrelated passage.
    "Our internal billing log emits the diagnostic identifier E-4042 deep inside a long "
    "batch reconciliation report alongside dozens of unrelated routine status lines.",
    # The distractor: short and generic, shares "error" but not the rare code.
    "A general error occurred while loading the page.",
    "Standard shipping takes three to five business days to arrive at the address on the order.",
    "Refunds are accepted within 30 days of purchase if the item is unused and in its packaging.",
]
QUERY = "E-4042 error"

# A short doc used only to SHOW the pooled-vs-multi-vector shape in Step 1.
SHAPE_DEMO_DOC = "Refund requests are processed within five business days."


def tokenize(text):
    """Lowercase word/number tokens -- the level the essay stays at.
    'E-4042' splits into 'e' and '4042', which is exactly why a rare-term query
    still finds an exact token match downstream."""
    return re.findall(r"[a-z0-9]+", text.lower())


# ===========================================================================
# Step 1. token_embed: a vector PER TOKEN.
#
#   THE INTENDED PATH. A late-interaction model (ColBERT) keeps the encoder's
#   per-token hidden states instead of pooling them into one vector. With
#   sentence-transformers we can pull the token embeddings out of the underlying
#   transformer. We wrap the load in try/except so this file still runs with no
#   model and no network: on failure we drop to a transparent deterministic
#   hashing embedder (Step 1b) and a banner says which path is live.
# ===========================================================================
def load_real_token_embedder():
    """Return token_embed(text)->(n_tokens, d) from a real model, or None offline."""
    try:
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer("all-MiniLM-L6-v2")  # 384 dims

        def token_embed(text):
            # Pull the transformer's per-token hidden states (before pooling).
            features = model.tokenize([text])
            features = {k: v.to(model.device) for k, v in features.items()}
            out = model.forward(features)
            # token_embeddings: (1, seq_len, d); mask out padding tokens.
            tok = out["token_embeddings"][0].detach().cpu().numpy()
            mask = features["attention_mask"][0].detach().cpu().numpy().astype(bool)
            tok = tok[mask]
            # L2-normalize each token vector so a dot product reads as cosine.
            norms = np.linalg.norm(tok, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            return tok / norms

        return token_embed
    except Exception as exc:  # missing package, no network, no cached weights...
        print(f"  [real model unavailable: {type(exc).__name__}] -> using offline fallback")
        return None


# ---------------------------------------------------------------------------
# Step 1b. The transparent offline fallback: hash each token to a unit vector.
#
#   A real late-interaction model learned, from billions of sentences, to place
#   token meanings sensibly in space. We cannot reproduce that without weights.
#   What we CAN do, with zero dependencies, is mimic the INTERFACE: any token ->
#   a fixed-length dense unit vector, deterministic and comparable in one space.
#   Identical tokens (the query's "4042" and the doc's "4042") map to the SAME
#   vector, so an exact term match scores ~1.0 -- enough to make MaxSim's
#   fine-grained matching visible, while being honest that it does NOT
#   understand synonyms.
# ---------------------------------------------------------------------------
FALLBACK_DIM = 384  # mirror all-MiniLM-L6-v2's 384 dims on purpose.


def _hashed_token_vector(token, dim):
    # Deterministic hash -> a reproducible pseudo-random direction for this token.
    h = hashlib.sha256(token.encode("utf-8")).digest()
    raw = np.frombuffer((h * (dim // len(h) + 1))[:dim], dtype=np.uint8)
    vec = (raw.astype(np.float64) / 255.0) * 2.0 - 1.0
    norm = np.linalg.norm(vec)
    return vec if norm == 0 else vec / norm


def fallback_token_embed(text, dim=FALLBACK_DIM):
    """Any text -> (n_tokens, dim) of L2-normalized per-token vectors."""
    tokens = tokenize(text)
    if not tokens:
        return np.zeros((1, dim))
    return np.vstack([_hashed_token_vector(t, dim) for t in tokens])


# ===========================================================================
# Step 2. MaxSim -- the late-interaction score. The correctness-critical
#         function, ~3 lines of numpy. (Given verbatim in the plan.)
# ===========================================================================
def maxsim(query_vecs: np.ndarray, doc_vecs: np.ndarray) -> float:
    """Late-interaction score: for each query token take the max cosine
    similarity over all doc tokens, then sum. Inputs are L2-normalized
    (n_q, d) and (n_d, d), so the dot product IS cosine similarity."""
    sims = query_vecs @ doc_vecs.T          # (n_q, n_d) pairwise cosine
    return float(sims.max(axis=1).sum())    # max over doc tokens, sum over query tokens


# ---------------------------------------------------------------------------
# Pooled-cosine baseline (Part 7): mean the token vectors into ONE vector per
# text, then compare with a single cosine. This is what late interaction is
# competing against; the whole point of Step 3 is that they DISAGREE.
# ---------------------------------------------------------------------------
def pool(token_vecs):
    """Mean-pool per-token vectors into one passage vector, L2-normalized."""
    v = token_vecs.mean(axis=0)
    norm = np.linalg.norm(v)
    return v if norm == 0 else v / norm


def pooled_cosine(query_vecs, doc_vecs):
    """One pooled query vector vs one pooled doc vector (Part 7 single-vector)."""
    return float(np.dot(pool(query_vecs), pool(doc_vecs)))


# ===========================================================================
# Step 4. The storage tradeoff. Per-token vectors cost (avg_tokens x) more than
#         one pooled vector; ColBERTv2 residual compression (1-2 bits/dim)
#         brings it back down. We compute the toy numbers and quote the paper's
#         MS MARCO scale (~154 GiB -> ~16-25 GiB) for context.
# ===========================================================================
def storage_report(n_docs, avg_tokens, dim, bytes_per_dim=4, compress_bits=2):
    pooled = n_docs * dim * bytes_per_dim
    multi = n_docs * avg_tokens * dim * bytes_per_dim
    multi_compressed = int(n_docs * avg_tokens * dim * (compress_bits / 8.0))
    print(f"Toy corpus: {n_docs} docs, ~{avg_tokens} tokens/doc, {dim} dims, "
          f"float32 ({bytes_per_dim} bytes/dim).")
    print(f"  pooled (1 vec/doc)      : {pooled:>12,} bytes  ({pooled/1e6:>7.2f} MB)")
    print(f"  multi-vector (1 vec/tok): {multi:>12,} bytes  ({multi/1e6:>7.2f} MB)"
          f"  -> {multi/pooled:.1f}x more")
    print(f"  multi-vector, {compress_bits}-bit     : {multi_compressed:>12,} bytes  "
          f"({multi_compressed/1e6:>7.2f} MB)  -> ColBERTv2 residual compression")
    print("At MS MARCO scale the ColBERTv2 paper reports the vanilla multi-vector index at")
    print("~154 GiB, shrunk to ~16 GiB (1-bit) / ~25 GiB (2-bit) by residual compression")
    print("(a 6-10x reduction). Per-token vectors cost more; compression makes it practical.")


# ===========================================================================
# Step 5. ColPali as a MECHANISM. A real ColPali/ColQwen embeds a document PAGE
#         IMAGE into ~1024 patch vectors and scores them against the query
#         tokens with the SAME MaxSim -- no OCR, no chunking. We CANNOT run a
#         real VLM offline, so we stand in a toy 4x4 grid of hashed "patch"
#         vectors to make the mechanism concrete: identical MaxSim call.
# ===========================================================================
def colpali_patch_demo(token_embed, query="refund window policy", grid=4, dim=FALLBACK_DIM):
    print("NOTE: no real vision-language model runs offline. This is a TOY: a "
          f"{grid}x{grid} grid of")
    print('hashed "patch" vectors stands in for a document page image\'s patch embeddings.')
    print("A real ColPali/ColQwen embeds the page image into ~1024 patch vectors and skips")
    print("OCR and chunking entirely; the scoring is the SAME MaxSim you built above.")
    # Build a toy "page": grid*grid patch vectors, deterministically hashed from
    # their (row, col) position so the demo is reproducible. Stand-in for an
    # image encoder's patch embeddings -- numbers, not pixels.
    patches = np.vstack(
        [_hashed_token_vector(f"patch_{r}_{c}", dim) for r in range(grid) for c in range(grid)]
    )
    q_vecs = token_embed(query)
    score = maxsim(q_vecs, patches)
    print(f"  toy page: {patches.shape[0]} patch vectors x {patches.shape[1]} dims "
          "(standing in for an image's patches)")
    print(f'  query "{query}" ({q_vecs.shape[0]} tokens) vs the page patches:')
    print(f"    maxsim(query tokens, page patches) = {score:.3f}")
    print("  -> mechanically identical to text late interaction: each query token takes its")
    print("     max over the patch vectors, then we sum. ColPali just makes the doc tokens")
    print("     be image patches instead of text tokens.")


# ===========================================================================
# Demo. Everything below RUNS OFFLINE. Real model if available, deterministic
#       fallback otherwise, with clearly labelled output throughout.
# ===========================================================================
if __name__ == "__main__":
    line = "=" * 70

    print(line)
    real = load_real_token_embedder()
    using_real = real is not None
    token_embed = real if using_real else fallback_token_embed
    label = "REAL all-MiniLM-L6-v2 token vectors" if using_real else \
        "offline hashing fallback (deterministic, numpy only)"
    print(f"Embedder in use: {label}")
    print(line)

    # ---- Step 1: one pooled vector vs a vector per token. ----
    print("\n" + line)
    print("STEP 1  One pooled vector vs a vector PER TOKEN")
    print(line)
    demo_vecs = token_embed(SHAPE_DEMO_DOC)
    n_tok, d = demo_vecs.shape
    print(f'Doc: "{SHAPE_DEMO_DOC}"')
    print(f"  pooled vector : 1 vector  x {d} dims      (the whole passage -> one point)")
    print(f"  multi-vector  : {n_tok} vectors x {d} dims      "
          "(one vector per token, nothing averaged away)")

    # ---- Step 2 & 3: MaxSim vs pooled cosine on the term-match query. ----
    print("\n" + line)
    print("STEP 2 & 3  MaxSim vs pooled cosine on a term-match query")
    print(line)
    print(f'Query: "{QUERY}"')
    q_vecs = token_embed(QUERY)
    doc_vecs = [token_embed(doc) for doc in CORPUS]

    # Tag the two docs that carry the lesson so the rankings read clearly.
    def tag(doc):
        if "E-4042" in doc:
            return "[exact term]"
        if doc.startswith("A general error"):
            return "[DISTRACTOR]"
        return ""

    pooled_ranked = sorted(
        ((pooled_cosine(q_vecs, dv), doc) for dv, doc in zip(doc_vecs, CORPUS)),
        key=lambda pair: -pair[0],
    )
    print("\nRanking by POOLED cosine (one vector per doc, Part 7):")
    for rank, (score, doc) in enumerate(pooled_ranked, 1):
        print(f"  rank {rank}  cos={score:+.3f}  {doc[:55]}... {tag(doc)}".rstrip())
    print("  -> the exact-term doc is NOT at the top: pooling a long passage averaged the")
    print('     rare E-4042 token away, and a short generic "error" doc pooled higher.')

    maxsim_ranked = sorted(
        ((maxsim(q_vecs, dv), doc) for dv, doc in zip(doc_vecs, CORPUS)),
        key=lambda pair: -pair[0],
    )
    print("\nRanking by MaxSim (token-level late interaction):")
    for rank, (score, doc) in enumerate(maxsim_ranked, 1):
        print(f"  rank {rank}  maxsim={score:.3f}  {doc[:55]}... {tag(doc)}".rstrip())
    print('  -> late interaction surfaces the exact-term doc: the query token "e4042" finds')
    print("     its exact match among the doc tokens, and MaxSim rewards that one strong hit")
    print("     even though it is buried in a long, otherwise-unrelated passage.")

    # ---- Step 4: the storage tradeoff. ----
    print("\n" + line)
    print("STEP 4  The storage tradeoff (toy numbers, then MS MARCO scale)")
    print(line)
    storage_report(n_docs=len(CORPUS), avg_tokens=12, dim=d)

    # ---- Step 5: ColPali as a mechanism. ----
    print("\n" + line)
    print("STEP 5  ColPali as a MECHANISM: MaxSim over page-image patches (toy stand-in)")
    print(line)
    colpali_patch_demo(token_embed)

    print("\n" + line)
    print("Takeaway: keep a vector per token, score with MaxSim, precompute the doc")
    print("vectors offline. You buy fine-grained term matching at bi-encoder serving cost,")
    print("paying in storage (which compression brings back down). ColPali pushes the same")
    print("trick onto page images, retrieving documents without OCR or chunking.")
    print("Next, Part 14: Context-Aware Chunking.")
    print(line)
