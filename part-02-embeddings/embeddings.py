"""
Embeddings, truly understood: from sparse failures to a dense meaning-space.
RAG from First Principles, Part 2: how text becomes numbers that carry meaning.

This file walks the exact ladder the essay climbs:
  1. one-hot encoding      -> captures identity, not meaning (sparse, fails)
  2. bag-of-words          -> captures word overlap, not meaning (sparse, fails)
  3. dense embeddings      -> meaning becomes POSITION IN SPACE (the real thing)
and ends with the classic "king - man + woman ~= queen" arithmetic.

Stack:
  - Sparse baselines : pure Python / NumPy (one-hot, bag-of-words) -- transparent
  - Real embeddings  : sentence-transformers SentenceTransformer("all-MiniLM-L6-v2")
                       384-dimensional dense vectors (the INTENDED path, shown below)
  - Offline fallback : a tiny deterministic hashing embedding so this file RUNS with
                       NO model, NO network, NO API key. Same pattern the later parts
                       use: real library is the headline, a stand-in keeps it runnable.

Run:
  pip install sentence-transformers numpy   # for the REAL embeddings
  python embeddings.py                      # runs offline either way

NOTE: The fallback is NOT a real embedding model -- it cannot actually capture
meaning (a hash has no idea "refund" and "reimbursement" are cousins). It exists
so the *shape* of the pipeline (text in, fixed-length dense vector out, compared
in one shared space) is runnable and inspectable without a 90 MB download. When
sentence-transformers is installed, the demo uses the genuine model automatically.
"""

import hashlib
import re

import numpy as np

# The running example from the essay: one chunk of a refund policy, and a user
# question phrased in completely different words. Keyword search struggles here;
# a good embedding should still place them close together in meaning-space.
CHUNK = "Refunds are accepted within 30 days of purchase."
QUERY = "What is our refund window?"


# ===========================================================================
# Step 1. Naive idea #1 -- one-hot encoding.
#         List every distinct word (the VOCABULARY); represent one word as a
#         vector that is all zeros except a single 1 in that word's slot.
#         It is a valid way to turn a word into numbers -- and useless for
#         meaning, because every word sits in its own private slot, touching
#         nothing else. "king" is exactly as far from "queen" as from "banana".
# ===========================================================================
def tokenize(text):
    # Lowercase words only; the essay stays at the level of plain words.
    return re.findall(r"[a-z0-9]+", text.lower())


def build_vocabulary(corpus):
    # Stable, sorted vocabulary so slot positions are reproducible.
    vocab = sorted({word for doc in corpus for word in tokenize(doc)})
    return {word: i for i, word in enumerate(vocab)}


def one_hot(word, vocab):
    # All zeros except a single 1 at this word's position -> a SPARSE vector.
    vec = np.zeros(len(vocab), dtype=int)
    if word in vocab:
        vec[vocab[word]] = 1
    return vec


# ===========================================================================
# Step 2. Naive idea #2 -- bag-of-words.
#         Encode a whole passage by COUNTING how often each vocabulary word
#         appears. Most slots stay 0; a few hold counts (1, 2, 3...). We throw
#         the words in a sack and count them, keeping NO record of order:
#         "the dog bit the man" and "the man bit the dog" get the same vector.
#         Two passages look similar only if they reuse the SAME words, so our
#         "refund window" / "refunds within 30 days" pair scores near zero.
# ===========================================================================
def bag_of_words(text, vocab):
    vec = np.zeros(len(vocab), dtype=int)
    for word in tokenize(text):
        if word in vocab:
            vec[vocab[word]] += 1
    return vec


# ===========================================================================
# Step 3. The real thing -- dense embeddings.
#
#   THE INTENDED PATH (what you'd actually ship). This is the exact, illustrative
#   snippet from the essay -- text in, a fixed-length dense vector out:
#
#       from sentence_transformers import SentenceTransformer
#       model = SentenceTransformer("all-MiniLM-L6-v2")   # 384 dimensions
#       vector = model.encode("Refunds are accepted within 30 days of purchase.")
#       print(len(vector))   # 384 -> this model returns 384 numbers, every time
#       print(vector[:4])    # [0.021, -0.34, 0.088, 0.12]  (illustrative values)
#
#   Embed the user's question the same way and you get ANOTHER 384 numbers; the
#   two lists are directly comparable because they live in the same 384-dim space.
#   We normalize to unit length so a dot product reads straight off as cosine
#   similarity -- the distance trick that is the whole subject of Part 3.
#
#   We wrap that load in try/except so this file still runs with no model and no
#   network: if the import or download fails, we drop to a transparent, fully
#   deterministic stand-in (Step 3b) and the rest of the demo is unchanged.
# ===========================================================================
def load_real_model():
    """Return a real SentenceTransformer, or None if it can't be loaded offline."""
    try:
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer("all-MiniLM-L6-v2")  # 384 dimensions

        def encode(texts):
            # normalize_embeddings=True -> every vector is unit length, so a dot
            # product equals cosine similarity (the trick from Part 3).
            return np.asarray(model.encode(texts, normalize_embeddings=True))

        return encode
    except Exception as exc:  # missing package, no network, no cached weights...
        print(f"[real model unavailable: {type(exc).__name__}] -> using offline fallback")
        return None


# ---------------------------------------------------------------------------
# Step 3b. The transparent offline fallback.
#
#   A REAL embedding model is a neural network that learned, from billions of
#   sentences, to place similar meanings near each other ("you shall know a word
#   by the company it keeps"). We cannot reproduce that without the model. What
#   we CAN do, with zero dependencies, is mimic the *interface*: any text -> a
#   fixed-length dense unit vector, deterministic and comparable in one space.
#
#   How: hash each token into the vector's dimensions and accumulate. Same text
#   -> same vector, always. Texts that SHARE WORDS land closer (shared tokens
#   push the same dimensions the same way), so it behaves a little like
#   bag-of-words projected into a short dense vector -- enough to show the shape
#   and the geometry, while being honest that it does NOT understand synonyms.
# ---------------------------------------------------------------------------
FALLBACK_DIM = 384  # mirror all-MiniLM-L6-v2's 384 dimensions on purpose.


def _hashed_token_vector(token, dim):
    # Deterministic hash -> a reproducible pseudo-random direction for this token.
    h = hashlib.sha256(token.encode("utf-8")).digest()
    # Stretch the 32-byte digest into `dim` floats in [-1, 1], deterministically.
    raw = np.frombuffer((h * (dim // len(h) + 1))[:dim], dtype=np.uint8)
    return (raw.astype(np.float64) / 255.0) * 2.0 - 1.0


def _fallback_encode_one(text, dim=FALLBACK_DIM):
    tokens = tokenize(text)
    if not tokens:
        return np.zeros(dim)
    vec = np.zeros(dim)
    for token in tokens:
        vec += _hashed_token_vector(token, dim)
    vec /= len(tokens)  # average so length doesn't depend on token count
    return unit(vec)    # unit length -> dot product == cosine (Part 3)


def fallback_encode(texts):
    return np.vstack([_fallback_encode_one(t) for t in texts])


# ---------------------------------------------------------------------------
# Small geometry helpers (Part 3 territory, used lightly here).
# ---------------------------------------------------------------------------
def unit(vec):
    norm = np.linalg.norm(vec)
    return vec if norm == 0 else vec / norm


def cosine(a, b):
    # Cosine similarity: 1.0 == same direction, 0.0 == unrelated, -1.0 == opposite.
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    return 0.0 if denom == 0 else float(np.dot(a, b) / denom)


# ===========================================================================
# Step 4. king - man + woman ~= queen.
#         The classic word-embedding party trick: meaning becomes GEOMETRY, so
#         "subtract man, add woman" is a little journey meaning "change the
#         gender, keep the royalty". Start at king, take that step, and you land
#         near queen. We compute the analogy vector and rank the candidates by
#         closeness to it. (The essay is honest that this magic comes from older
#         per-word embeddings like word2vec/GloVe; modern sentence embeddings are
#         richer and won't do crisp subtraction. The fallback can't do it either
#         -- it's a hash -- so we expose a real-model-only helper here too.)
# ===========================================================================
def analogy(encode, a, b, c, candidates):
    """Return candidates ranked by closeness to (a - b + c). E.g. king-man+woman."""
    va, vb, vc = encode([a]), encode([b]), encode([c])
    target = unit((va - vb + vc)[0])
    ranked = []
    for word in candidates:
        ranked.append((word, cosine(target, encode([word])[0])))
    ranked.sort(key=lambda pair: -pair[1])
    return ranked


# ===========================================================================
# Demo. Everything below RUNS OFFLINE. It uses the real model if available and
#       the deterministic fallback otherwise, and prints clearly labelled output.
# ===========================================================================
if __name__ == "__main__":
    line = "=" * 70

    # ---- Step 1 + 2: show the sparse baselines failing on the running example.
    print(line)
    print("STEP 1 & 2  Sparse baselines: one-hot and bag-of-words")
    print(line)
    corpus = [CHUNK, QUERY]
    vocab = build_vocabulary(corpus)
    print(f"Vocabulary size for our two tiny sentences: {len(vocab)} words")
    print(f"A one-hot vector is {len(vocab)} numbers long with a single 1 in it.")
    oh_refunds = one_hot("refunds", vocab)
    print(f"  one_hot('refunds') has {int(oh_refunds.sum())} hot slot "
          f"out of {len(oh_refunds)}  -> sparse, all identity, no meaning")

    bow_chunk = bag_of_words(CHUNK, vocab)
    bow_query = bag_of_words(QUERY, vocab)
    print(f"\nbag-of-words(chunk) nonzero slots: {int((bow_chunk > 0).sum())}")
    print(f"bag-of-words(query) nonzero slots: {int((bow_query > 0).sum())}")
    print(f"  cosine(chunk, query) under bag-of-words = {cosine(bow_chunk, bow_query):.3f}")
    print("  -> near zero: almost no shared WORDS, so it rates them barely related,")
    print("     even though they ask and answer the very same thing.")

    # Order-blindness: a bag keeps no record of word order.
    v1 = bag_of_words("the dog bit the man", build_vocabulary(["the dog bit the man"]))
    v2 = bag_of_words("the man bit the dog", build_vocabulary(["the dog bit the man"]))
    print(f"\n  'the dog bit the man' vs 'the man bit the dog' -> identical bag? "
          f"{bool(np.array_equal(v1, v2))}  (order is thrown away)")

    # ---- Step 3: real dense embeddings (or transparent fallback).
    print("\n" + line)
    print("STEP 3  Dense embeddings: text in, a fixed-length dense vector out")
    print(line)
    encode = load_real_model()
    using_real = encode is not None
    if not using_real:
        encode = fallback_encode
    print(f"Embedder in use: {'REAL all-MiniLM-L6-v2' if using_real else 'offline hashing fallback'}")

    # Embed the chunk and the query INTO THE SAME SPACE.
    chunk_vec = encode([CHUNK])[0]
    query_vec = encode([QUERY])[0]
    print(f"\nencode(chunk): {len(chunk_vec)} numbers   (the model returns this many, every time)")
    print(f"  first 4 values: {np.round(chunk_vec[:4], 3).tolist()}  (meaningless to a human; shape is the point)")
    print(f"encode(query): {len(query_vec)} numbers   (same space -> directly comparable)")
    print(f"\ncosine(chunk, query) with dense embeddings = {cosine(chunk_vec, query_vec):.3f}")
    if using_real:
        print("  -> high: a real model places 'refund window?' next to 'refunds within 30 days'")
        print("     because they keep the same company. THAT is search by meaning.")
    else:
        print("  -> the fallback is only a hash of shared words, so don't read meaning into it;")
        print("     install sentence-transformers to see the genuine 'close in meaning' effect.")

    # Contrast: the chunk vs an unrelated sentence should score lower.
    unrelated = "Here is a banana bread recipe with ripe bananas."
    unrelated_vec = encode([unrelated])[0]
    print(f"\ncosine(chunk, '{unrelated}')")
    print(f"  = {cosine(chunk_vec, unrelated_vec):.3f}   (different topic -> farther apart)")

    # ---- Step 4: king - man + woman ~= queen.
    print("\n" + line)
    print("STEP 4  Vector arithmetic: king - man + woman ~= queen")
    print(line)
    candidates = ["queen", "king", "prince", "princess", "banana", "throne", "woman", "man"]
    if using_real:
        ranked = analogy(encode, "king", "man", "woman", candidates)
        print("Nearest words to (king - man + woman), best first:")
        for word, score in ranked[:5]:
            print(f"  {word:<10} cosine = {score:+.3f}")
        print("  -> 'queen' should rank at or near the top: the DIRECTION captured a concept.")
    else:
        print("This party trick needs genuine learned word geometry (word2vec/GloVe-style).")
        print("The offline hashing fallback has no such geometry -- a hash of 'king' is")
        print("unrelated to a hash of 'queen' -- so we skip the numbers rather than fake them.")
        print("Intuition to keep: 'subtract man, add woman' = change gender, keep royalty,")
        print("so the journey from king lands you in queen's neighborhood.")

    print("\n" + line)
    print("Takeaway: sparse encodings capture spelling; dense embeddings capture")
    print("meaning, so 'relevant' becomes 'closest point in the shared space'.")
    print("Next, Part 3: how to actually MEASURE that closeness (cosine similarity).")
    print(line)
