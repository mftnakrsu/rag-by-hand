"""
Measuring similarity between vectors, by hand.
RAG from First Principles, Part 3: how a computer turns "close" into a number.

Part 2 left a debt: it kept saying two embeddings are "close" or "near" without
ever defining how a machine measures closeness. This file pays that debt with the
three metrics from the essay -- Euclidean distance, the dot product, and cosine
similarity -- plus the 'aha' that ties them together: cosine similarity is just the
dot product of normalized (unit-length) vectors. We finish with top-k retrieval,
the ranking function at the heart of every RAG system.

Stack:
  - Math       : pure Python first (so you can see every multiply and add),
                 then the NumPy one-liner the essay shows as the "in code" version.
  - No model   : this part is arithmetic a calculator can do. No embedding model,
                 no API key, no network. It runs fully offline as-is.

Run:
  python3 similarity.py        # NumPy is used but optional; a fallback covers it.

NOTE: real embeddings have hundreds of dimensions, but the procedure here is
*identical*, just with more terms to add. We use tiny 2-D vectors so the numbers
stay on paper and you can check them yourself.
"""

from __future__ import annotations

import math

# ---------------------------------------------------------------------------
# Step 0. NumPy is nice for the one-liner cosine, but this whole part is just
#         multiplies, adds, and one square root. If NumPy is missing we fall
#         back to a transparent pure-Python stand-in so the demo always runs.
#         (Later parts use this same "works without the heavy library" pattern.)
# ---------------------------------------------------------------------------
try:
    import numpy as np

    HAVE_NUMPY = True
except ImportError:  # pragma: no cover - exercised only on a NumPy-less box
    np = None
    HAVE_NUMPY = False


Vector = list  # a vector is just an ordered list of numbers (Part 2)


# ---------------------------------------------------------------------------
# Step 1. Euclidean distance: the straight-line gap between two arrowheads.
#         Pythagoras in n dimensions: difference per component, square each,
#         sum, square-root. It is a DISTANCE -> smaller means more similar,
#         and identical vectors score 0. Its weakness for text: it is fooled
#         by magnitude (length), so a short note and a long passage about the
#         same topic can look far apart.
#
#         euclidean(A, B) = sqrt((A1 - B1)^2 + ... + (An - Bn)^2)
# ---------------------------------------------------------------------------
def euclidean(a: Vector, b: Vector) -> float:
    return math.sqrt(sum((ai - bi) ** 2 for ai, bi in zip(a, b)))


# ---------------------------------------------------------------------------
# Step 2. Dot product: multiply matching components, add them up.
#         It is a SIMILARITY -> bigger means more aligned. It quietly blends
#         two things at once: how much the vectors point the same way (angle)
#         AND how long they are (magnitude). Same direction -> big positive;
#         right angles -> 0; opposite -> negative. Fast (just multiplies and
#         adds), which is why it runs under the hood at scale (Part 4).
#
#         A . B = A1*B1 + A2*B2 + ... + An*Bn
# ---------------------------------------------------------------------------
def dot(a: Vector, b: Vector) -> float:
    return sum(ai * bi for ai, bi in zip(a, b))


# ---------------------------------------------------------------------------
# Step 3. Magnitude (a.k.a. L2 norm, written ||A||): the length of the vector,
#         the distance from the origin to its tip. Square the components, add,
#         square-root -- exactly Euclidean distance measured from the origin.
#
#         ||A|| = sqrt(A1^2 + A2^2 + ... + An^2)
# ---------------------------------------------------------------------------
def magnitude(a: Vector) -> float:
    return math.sqrt(sum(ai * ai for ai in a))


# ---------------------------------------------------------------------------
# Step 4. Cosine similarity: throw away length, measure only the ANGLE.
#         It is the dot product divided by BOTH magnitudes -- and that division
#         is exactly the step that cancels magnitude out. What survives is pure
#         direction. Range -1 (opposite) .. 0 (right angles) .. 1 (same way);
#         for typical text embeddings it lands between 0 and 1. This is the
#         DEFAULT metric in RAG, because embedding models put meaning in the
#         direction, not the length.
#
#         cosine(A, B) = (A . B) / (||A|| * ||B||)
# ---------------------------------------------------------------------------
def cosine(a: Vector, b: Vector) -> float:
    return dot(a, b) / (magnitude(a) * magnitude(b))


# ---------------------------------------------------------------------------
# Step 5. Normalization: rescale a vector to length 1, keeping its direction.
#         The result is a UNIT vector, written  hat(A) = A / ||A||.
# ---------------------------------------------------------------------------
def normalize(a: Vector) -> Vector:
    mag = magnitude(a)
    return [ai / mag for ai in a]


# ---------------------------------------------------------------------------
# Step 6. The 'aha' from the essay: cosine similarity IS the dot product of
#         normalized vectors. Once both vectors are unit length their magnitudes
#         are both 1, so dividing by them does nothing and the cosine formula
#         collapses to a plain dot product. Many vector DBs normalize once at
#         storage time, then run the cheap dot product at query time -- cosine's
#         meaning-focused behavior at the dot product's speed.
#
#         cosine(A, B) = hat(A) . hat(B)
# ---------------------------------------------------------------------------
def cosine_via_dot(a: Vector, b: Vector) -> float:
    return dot(normalize(a), normalize(b))


# ---------------------------------------------------------------------------
# Step 7. The NumPy one-liner the essay shows ("if you'd rather see it in code").
#         Real systems lean on this: @ is the dot product, np.linalg.norm is the
#         magnitude. If NumPy is not installed we transparently route back to the
#         pure-Python version above so the demo still runs and prints the same
#         number -- the "fallback that works without the library" pattern.
# ---------------------------------------------------------------------------
def cosine_numpy(a: Vector, b: Vector) -> float:
    if not HAVE_NUMPY:
        # Transparent stand-in: identical math, no dependency.
        return cosine(a, b)
    a_arr, b_arr = np.array(a, dtype=float), np.array(b, dtype=float)
    return float(a_arr @ b_arr / (np.linalg.norm(a_arr) * np.linalg.norm(b_arr)))


# ---------------------------------------------------------------------------
# Step 8. Top-k retrieval: the ranking function at the heart of RAG.
#         Score the query against EVERY stored vector with cosine similarity,
#         then keep the k highest. This is the brute-force / exact search the
#         essay describes -- fine for hundreds or thousands of chunks; Part 4
#         is about making it fast for millions. Each stored item is a (label,
#         vector) pair so we can show WHICH chunk won, the way a real store
#         keeps the text beside the vector.
# ---------------------------------------------------------------------------
def top_k(query: Vector, store: list[tuple[str, Vector]], k: int = 3):
    scored = [(label, cosine(query, vec)) for label, vec in store]
    # A SIMILARITY -> higher is better, so sort descending and take the first k.
    scored.sort(key=lambda pair: pair[1], reverse=True)
    return scored[:k]


# ---------------------------------------------------------------------------
# Step 9. Carry the essay's worked-by-hand examples over as assert statements.
#         If these pass, the implementations match the numbers you can check on
#         paper -- a metric you can't compute is a metric you don't trust.
# ---------------------------------------------------------------------------
def _check_worked_examples() -> None:
    A = [3, 4]
    B = [4, 3]
    C = [6, 8]  # exactly A doubled: same direction, twice as long.

    # --- A vs B ---------------------------------------------------------
    # Dot product: (3*4) + (4*3) = 12 + 12 = 24
    assert dot(A, B) == 24, dot(A, B)
    # Magnitudes: sqrt(9 + 16) = sqrt(25) = 5  (both A and B)
    assert magnitude(A) == 5, magnitude(A)
    assert magnitude(B) == 5, magnitude(B)
    # Cosine: 24 / (5 * 5) = 24 / 25 = 0.96  -> nearly the same direction
    assert math.isclose(cosine(A, B), 0.96), cosine(A, B)
    assert math.isclose(cosine_numpy(A, B), 0.96), cosine_numpy(A, B)
    # Euclidean: sqrt((3-4)^2 + (4-3)^2) = sqrt(2) ~= 1.41
    assert math.isclose(euclidean(A, B), math.sqrt(2)), euclidean(A, B)

    # --- A vs C: the example that makes cosine earn its keep --------------
    # Cosine: (3*6 + 4*8) / (5 * 10) = 50 / 50 = 1.00  -> identical direction
    assert math.isclose(cosine(A, C), 1.00), cosine(A, C)
    # Euclidean: sqrt((3-6)^2 + (4-8)^2) = sqrt(9 + 16) = sqrt(25) = 5  -> far apart
    assert euclidean(A, C) == 5, euclidean(A, C)
    # Dot product jumped to 50, rewarding C purely for being long: 18 + 32 = 50
    assert dot(A, C) == 50, dot(A, C)

    # --- The 'aha' made concrete ----------------------------------------
    # Normalize A and C: both become the SAME unit vector [0.6, 0.8].
    assert all(math.isclose(x, y) for x, y in zip(normalize(A), [0.6, 0.8]))
    assert all(math.isclose(x, y) for x, y in zip(normalize(C), [0.6, 0.8]))
    # Their dot product is (0.6*0.6) + (0.8*0.8) = 0.36 + 0.64 = 1.00 = the cosine.
    assert math.isclose(cosine_via_dot(A, C), 1.00), cosine_via_dot(A, C)
    # cosine == dot of normalized vectors, for A vs B too (same calculation):
    assert math.isclose(cosine(A, B), cosine_via_dot(A, B))


# ---------------------------------------------------------------------------
# Step 10. Demonstrate everything with clear, labelled output.
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    _check_worked_examples()
    print(f"NumPy available: {HAVE_NUMPY}  (falls back to pure Python if not)\n")

    A = [3, 4]
    B = [4, 3]
    C = [6, 8]
    print("Three tiny 2-D vectors (real embeddings just have more dimensions):")
    print(f"  A = {A}")
    print(f"  B = {B}")
    print(f"  C = {C}   (A doubled: same direction, twice as long)\n")

    print("Pairwise scores -- three metrics, three different verdicts:")
    print(f"{'pair':>6} | {'euclidean':>9} | {'dot':>5} | {'cosine':>7}")
    print("-" * 38)
    for name, x, y in [("A,B", A, B), ("A,C", A, C), ("B,C", B, C)]:
        print(
            f"{name:>6} | {euclidean(x, y):>9.2f} | "
            f"{dot(x, y):>5.0f} | {cosine(x, y):>7.2f}"
        )

    print("\nThe 'aha': cosine == dot product of normalized (unit-length) vectors")
    print(f"  normalize(A) = {[round(v, 2) for v in normalize(A)]}")
    print(f"  normalize(C) = {[round(v, 2) for v in normalize(C)]}   (same unit vector!)")
    print(f"  cosine(A, C)              = {cosine(A, C):.2f}")
    print(f"  cosine_via_dot(A, C)      = {cosine_via_dot(A, C):.2f}")
    print(f"  cosine_numpy(A, C)        = {cosine_numpy(A, C):.2f}   (the @ one-liner)")

    # --- Top-k retrieval over a small set of stored vectors --------------
    # A query and a handful of "chunk" vectors. We score the query against
    # every one with cosine and keep the best (Part 3 + Part 4). The winner
    # is whichever vector points most nearly the SAME WAY as the query.
    print("\nTop-k retrieval (the RAG ranking function):")
    query = [1.0, 0.0]  # points straight along the x-axis
    store = [
        ("chunk_0  same direction, far away", [9.0, 0.0]),   # cos 1.00 (length ignored!)
        ("chunk_1  45 degrees off",           [1.0, 1.0]),   # cos ~0.71
        ("chunk_2  near-aligned",             [4.0, 1.0]),   # cos ~0.97
        ("chunk_3  right angle, unrelated",   [0.0, 5.0]),   # cos 0.00
        ("chunk_4  pointing opposite",        [-2.0, 0.0]),  # cos -1.00
    ]
    print(f"  query = {query}")
    results = top_k(query, store, k=3)
    for rank, (label, score) in enumerate(results, start=1):
        print(f"  {rank}. score={score:>6.2f}  {label}")
    print("\n  Note chunk_0 ties for first despite being 9x longer than the query:")
    print("  cosine ignores length and rewards pure direction -- the whole point.")

    print("\nAll worked examples from the essay checked out. Done.")
