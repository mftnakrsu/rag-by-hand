"""
Vector databases and indexing, by hand.
RAG from First Principles, Part 4: from brute-force k-NN to an approximate index,
and the one trade-off the whole field turns on -- speed versus recall.

Part 3 left us with a working ranking function and a warning: scoring a query
against EVERY stored vector is exact but O(n), fine for thousands of chunks and
brutal for millions. This file makes that gap concrete. We build the brute-force
baseline, then a tiny IVF-style index (k-means clusters + nprobe), measure the
recall it gives up against the speed it buys, and sketch Product Quantization.

Stack:
  - Vectors      : a seeded NumPy point cloud (no model, no network needed)
  - Exact search : brute_force_topk() -- the O(n) baseline from the essay
  - ANN index    : IVFIndex -- k-means centroids + search the nprobe nearest
  - Compression  : product_quantize() -- the PQ idea in a few lines
  - Real path    : FAISS (the reference ANN toolkit) shown but optional

Run:
  pip install numpy            # numpy is the only hard dependency
  pip install faiss-cpu        # OPTIONAL -- enables the "real library" path
  python3 vector_db.py

Everything here runs offline with nothing but NumPy. FAISS is shown as the
intended production path (the "library / raw index" row in the essay's table),
but the executable demo always falls back to the transparent NumPy code so it
runs with no model and no API key installed.
"""

import time

import numpy as np


# ---------------------------------------------------------------------------
# Step 0. A seeded, eyeball-able point cloud standing in for chunk embeddings.
#         In a real RAG app these vectors come from an embedding model (Part 2),
#         already unit-normalized so a dot product equals cosine similarity --
#         the trick from Part 3. Here we synthesize them so the file runs with
#         no model and no network: fixed seed in, identical numbers out.
#
#         We deliberately draw them in CLUMPS, not uniform noise. Real embeddings
#         cluster by meaning (refund chunks near refund chunks), and clustering
#         is exactly what IVF will exploit later.
# ---------------------------------------------------------------------------
def make_point_cloud(n=2000, d=32, n_blobs=12, seed=42):
    """Return an (n, d) matrix of unit vectors arranged in n_blobs clusters."""
    rng = np.random.default_rng(seed)            # seeded: NOT Math.random, reproducible
    centers = rng.normal(size=(n_blobs, d))      # one random center per blob
    labels = rng.integers(0, n_blobs, size=n)    # assign each point to a blob
    # Each point = its blob center + a little Gaussian jitter.
    vectors = centers[labels] + 0.35 * rng.normal(size=(n, d))
    return unit_normalize(vectors), rng


def unit_normalize(x):
    """Scale each row to length 1 so dot product == cosine similarity (Part 3)."""
    norms = np.linalg.norm(x, axis=-1, keepdims=True)
    return x / np.maximum(norms, 1e-12)          # guard against a zero vector


# ---------------------------------------------------------------------------
# Step 1. The baseline we just outgrew: brute-force k-NN.
#         This is the essay's snippet, almost verbatim. It is EXACT -- by
#         inspecting every candidate it returns the genuine k nearest -- and it
#         is O(n): one dot product per stored vector. Wonderful and fatal.
# ---------------------------------------------------------------------------
def brute_force_topk(query, vectors, k=4):
    """Return the indices of the k vectors most similar to `query` (exact)."""
    scores = vectors @ query                     # one dot product per vector -> O(N)
    return np.argsort(-scores)[:k]               # indices of the k highest scores


# ---------------------------------------------------------------------------
# Step 2. Recall@k -- the word for the sliver of accuracy ANN gives up.
#         In the nearest-neighbor sense, recall is "of the genuine top-k, how
#         many did we actually find." The essay's example: the exact top-10 has
#         ten specific chunks; an ANN index returns eight of them -> 0.8 recall.
#         Note this is a SET overlap; the order within the k does not matter.
# ---------------------------------------------------------------------------
def recall_at_k(approx_idx, true_idx):
    """Fraction of the true top-k that the approximate result recovered."""
    true_set = set(int(i) for i in true_idx)
    if not true_set:
        return 1.0
    found = sum(1 for i in approx_idx if int(i) in true_set)
    return found / len(true_set)


# ---------------------------------------------------------------------------
# Step 3. k-means -- the clustering step behind IVF.
#         IVF groups stored vectors into clusters "by proximity" (the essay's
#         library shelves). k-means does that: pick k centroids, assign every
#         point to its nearest centroid, move each centroid to the mean of its
#         members, repeat. We write it in plain NumPy so nothing is hidden.
# ---------------------------------------------------------------------------
def kmeans(vectors, n_clusters, n_iter=25, seed=0):
    """Lloyd's algorithm. Returns (centroids, assignment-per-vector)."""
    rng = np.random.default_rng(seed)
    n = vectors.shape[0]
    # Initialize centroids at n_clusters distinct randomly chosen points.
    init = rng.choice(n, size=n_clusters, replace=False)
    centroids = vectors[init].copy()
    assign = np.zeros(n, dtype=int)
    for _ in range(n_iter):
        # Assign: each point to the nearest centroid by squared Euclidean dist.
        # ||a - b||^2 expanded; argmin over centroids gives the cluster.
        dists = (
            np.sum(vectors ** 2, axis=1, keepdims=True)
            - 2.0 * vectors @ centroids.T
            + np.sum(centroids ** 2, axis=1)[None, :]
        )
        new_assign = np.argmin(dists, axis=1)
        if np.array_equal(new_assign, assign):
            break                                # converged: nothing moved
        assign = new_assign
        # Update: move each centroid to the mean of its members.
        for c in range(n_clusters):
            members = vectors[assign == c]
            if len(members) > 0:
                centroids[c] = members.mean(axis=0)
            else:
                centroids[c] = vectors[rng.integers(0, n)]  # re-seed an empty cluster
    return centroids, assign


# ---------------------------------------------------------------------------
# Step 4. IVF -- the Inverted File Index, "sorting vectors onto shelves."
#         Build time: cluster the vectors with k-means; each cluster's centroid
#         is its shelf label, and an inverted list records which vectors live on
#         which shelf. Query time: compare the query against the (few) centroids,
#         pick the nprobe nearest shelves, and search ONLY inside those.
#
#         The essay's worked example: a million vectors split into a thousand
#         clusters of a thousand each, looking in one cluster, replaces a million
#         comparisons with ~1000 (centroids) + ~1000 (one cluster) ~= 2000.
#         nprobe is the dial that slides us along the speed-recall curve.
# ---------------------------------------------------------------------------
class IVFIndex:
    def __init__(self, n_clusters=40, seed=0):
        self.n_clusters = n_clusters
        self.seed = seed
        self.centroids = None
        self.inverted_lists = None               # cluster id -> array of vector ids
        self.vectors = None
        self.last_comparisons = 0                # comparisons made by the last search

    def build(self, vectors):
        """Train the centroids and bucket every vector onto its shelf."""
        self.vectors = vectors
        self.centroids, assign = kmeans(vectors, self.n_clusters, seed=self.seed)
        # Inverted lists: the "file" that maps each shelf to its members.
        self.inverted_lists = [
            np.where(assign == c)[0] for c in range(self.n_clusters)
        ]
        return self

    def search(self, query, k=4, nprobe=1):
        """Approximate top-k: search only the nprobe nearest clusters."""
        # 1) Compare the query against the centroids only (cheap: few of them).
        centroid_scores = self.centroids @ query           # nprobe stage
        comparisons = self.n_clusters                       # one per centroid
        nearest_clusters = np.argsort(-centroid_scores)[:nprobe]

        # 2) Gather the members of those clusters -- the only vectors we score.
        candidate_ids = np.concatenate(
            [self.inverted_lists[c] for c in nearest_clusters]
        ) if nprobe > 0 else np.array([], dtype=int)

        if candidate_ids.size == 0:
            self.last_comparisons = comparisons
            return np.array([], dtype=int)

        # 3) Brute force, but only WITHIN the opened shelves (the whole point).
        scores = self.vectors[candidate_ids] @ query
        comparisons += candidate_ids.size                   # one per candidate
        self.last_comparisons = comparisons

        top_local = np.argsort(-scores)[:k]
        return candidate_ids[top_local]


# Real-library path (the intended production route). FAISS is the "library /
# raw index" from the essay's table: not a database, but THE reference toolkit
# for the indexes themselves. The IVFIndex above is a teaching re-implementation
# of exactly this. Kept in a function so its import never blocks the demo.
def build_faiss_ivf(vectors, n_clusters=40, nprobe=1):
    """Build a real FAISS IndexIVFFlat. Returns None if faiss isn't installed."""
    try:
        import faiss
    except ImportError:
        return None                              # transparent fallback: caller uses IVFIndex
    d = vectors.shape[1]
    quantizer = faiss.IndexFlatIP(d)             # inner product == cosine (unit vectors)
    index = faiss.IndexIVFFlat(quantizer, d, n_clusters, faiss.METRIC_INNER_PRODUCT)
    vectors = np.ascontiguousarray(vectors, dtype="float32")
    index.train(vectors)                         # k-means under the hood
    index.add(vectors)
    index.nprobe = nprobe                        # the same dial, same meaning
    return index


# ---------------------------------------------------------------------------
# Step 5. nprobe sweep -- watching the speed-versus-recall curve in numbers.
#         For each nprobe we run many queries, average the recall against the
#         exact top-k, and count the comparisons (our proxy for "work" / speed).
#         Low nprobe: tiny comparison count, lower recall (we miss neighbors
#         sitting just across a cluster boundary we never opened). High nprobe:
#         recall climbs toward 1.0 as comparisons climb toward brute force.
# ---------------------------------------------------------------------------
def nprobe_sweep(index, queries, vectors, k=10, nprobe_values=(1, 2, 4, 8, 16)):
    """Return a list of dict rows, one per nprobe, with mean recall + comparisons."""
    # Exact answers once, reused as ground truth for every nprobe setting.
    exact = [brute_force_topk(q, vectors, k=k) for q in queries]
    brute_comparisons = vectors.shape[0]         # brute force compares against all N

    rows = []
    for nprobe in nprobe_values:
        recalls, comps = [], []
        for q, true_idx in zip(queries, exact):
            approx = index.search(q, k=k, nprobe=nprobe)
            recalls.append(recall_at_k(approx, true_idx))
            comps.append(index.last_comparisons)
        rows.append({
            "nprobe": nprobe,
            "recall": float(np.mean(recalls)),
            "comparisons": float(np.mean(comps)),
            # Speedup vs brute force: how much less work, on average.
            "speedup": brute_comparisons / float(np.mean(comps)),
        })
    return rows, brute_comparisons


# ---------------------------------------------------------------------------
# Step 6. Product Quantization (PQ) -- a sketch of compressing the vectors.
#         Orthogonal to the index: a million 768-d float32 vectors are several
#         GB before any index. PQ chops each vector into m short SUB-VECTORS and
#         replaces each piece with the nearest entry from a small learned
#         CODEBOOK (one codebook per sub-space, learned by k-means). A long list
#         of precise floats collapses into m tiny integer codes. Distances can be
#         computed on the codes; the price is a little lost precision (recall).
#         Often paired with clustering as IVF+PQ. This is a sketch, not FAISS PQ.
# ---------------------------------------------------------------------------
def product_quantize(vectors, m=4, ksub=16, seed=0):
    """Train m codebooks of ksub entries each; encode vectors into m codes apiece.

    Returns (codes, codebooks):
      codes     : (n, m) int array -- each vector is now just m small indices
      codebooks : list of m arrays, each (ksub, d/m) -- the learned sub-vectors
    """
    n, d = vectors.shape
    assert d % m == 0, "sub-vector count m must divide the dimension d"
    dsub = d // m                                 # length of each sub-vector
    codebooks, codes = [], np.zeros((n, m), dtype=int)
    for j in range(m):
        sub = vectors[:, j * dsub:(j + 1) * dsub]  # the j-th slice of every vector
        # Learn this sub-space's codebook by k-means; centroids ARE the codebook.
        book, assign = kmeans(sub, ksub, seed=seed + j)
        codebooks.append(book)
        codes[:, j] = assign                       # store the nearest code, not the floats
    return codes, codebooks


def pq_reconstruct(codes, codebooks):
    """Rebuild approximate vectors by stitching each sub-vector's codebook entry."""
    pieces = [codebooks[j][codes[:, j]] for j in range(len(codebooks))]
    return np.concatenate(pieces, axis=1)


def pq_memory_bytes(n, d, m, ksub):
    """Compare raw float32 storage against PQ codes (codebooks are tiny, shared)."""
    raw = n * d * 4                                # 4 bytes per float32
    code_bytes = 1 if ksub <= 256 else 2          # one byte per code if <=256 entries
    compressed = n * m * code_bytes               # the codebooks are negligible at scale
    return raw, compressed


# ---------------------------------------------------------------------------
# Step 7. The demo. Build the cloud, run the baseline, build IVF, sweep nprobe,
#         sketch PQ -- printing clear, labelled output at each stage.
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    N, D, K = 2000, 32, 10
    vectors, rng = make_point_cloud(n=N, d=D, n_blobs=12, seed=42)
    print("=" * 70)
    print(f"Part 4: Vector Databases and Indexing")
    print(f"Point cloud: N={N} vectors, d={D} dims, seeded (reproducible)")
    print("=" * 70)

    # --- Brute force: exact, O(n), the baseline we outgrew -------------------
    print("\n[1] Brute-force k-NN (exact, O(n))")
    query = unit_normalize(vectors[0] + 0.1 * rng.normal(size=D))  # near vector 0
    t0 = time.perf_counter()
    exact_top = brute_force_topk(query, vectors, k=K)
    dt = (time.perf_counter() - t0) * 1e3
    print(f"    exact top-{K} indices : {[int(i) for i in exact_top]}")
    print(f"    comparisons          : {N} (one dot product per stored vector)")
    print(f"    time                 : {dt:.3f} ms")

    # --- recall@k: the metric for the sliver of accuracy ANN gives up -------
    print("\n[2] recall@k sanity check")
    perfect = recall_at_k(exact_top, exact_top)
    eight_of_ten = recall_at_k(exact_top[:8], exact_top)  # essay's 8-of-10 example
    print(f"    recall of the exact set against itself : {perfect:.2f}  (must be 1.00)")
    print(f"    recall when only 8 of the true 10 found: {eight_of_ten:.2f}  "
          f"(the essay's 0.8 / 80% example)")

    # --- IVF index: cluster onto shelves, search the nprobe nearest ---------
    print("\n[3] IVF index (k-means shelves + nprobe)")
    n_clusters = 40
    index = IVFIndex(n_clusters=n_clusters, seed=0).build(vectors)
    sizes = [len(lst) for lst in index.inverted_lists]
    print(f"    clusters             : {n_clusters}")
    print(f"    cluster sizes        : min={min(sizes)}, "
          f"max={max(sizes)}, mean={np.mean(sizes):.1f}")
    faiss_index = build_faiss_ivf(vectors, n_clusters=n_clusters, nprobe=1)
    print(f"    FAISS available      : {'yes (real IndexIVFFlat built)' if faiss_index else 'no -> using transparent NumPy IVFIndex'}")

    # --- nprobe sweep: the speed-versus-recall curve, in numbers ------------
    print("\n[4] nprobe sweep: the speed-vs-recall trade-off")
    n_queries = 200
    # Queries = stored vectors nudged a little, so each has a real nearby answer.
    q_ids = rng.choice(N, size=n_queries, replace=False)
    queries = unit_normalize(vectors[q_ids] + 0.1 * rng.normal(size=(n_queries, D)))
    rows, brute_comps = nprobe_sweep(index, queries, vectors, k=K,
                                     nprobe_values=(1, 2, 4, 8, 16))
    print(f"    brute force compares against all {int(brute_comps)} vectors per query.\n")
    print(f"    {'nprobe':>6} | {'recall@%d' % K:>9} | {'comparisons':>12} | {'speedup':>8}")
    print(f"    {'-'*6}-+-{'-'*9}-+-{'-'*12}-+-{'-'*8}")
    for r in rows:
        print(f"    {r['nprobe']:>6} | {r['recall']:>9.3f} | "
              f"{r['comparisons']:>12.0f} | {r['speedup']:>7.1f}x")
    print("\n    Read it as the essay does: nprobe up -> recall up, but more")
    print("    comparisons (less speed). nprobe is the dial on the curve.")

    # --- Product Quantization: a sketch of compressing the vectors ----------
    print("\n[5] Product Quantization (PQ) sketch")
    m, ksub = 4, 16                               # 4 sub-vectors, 16-entry codebooks
    codes, codebooks = product_quantize(vectors, m=m, ksub=ksub, seed=0)
    approx = pq_reconstruct(codes, codebooks)
    # How faithful is the compressed form? Cosine between original and rebuilt.
    fidelity = float(np.mean(np.sum(unit_normalize(approx) * vectors, axis=1)))
    raw_bytes, pq_bytes = pq_memory_bytes(N, D, m, ksub)
    print(f"    each {D}-d vector -> {m} codes from {ksub}-entry codebooks")
    print(f"    one vector now    : {codes[0].tolist()}  (was {D} floats)")
    print(f"    memory raw float32: {raw_bytes:,} bytes")
    print(f"    memory PQ codes   : {pq_bytes:,} bytes  "
          f"({raw_bytes / pq_bytes:.0f}x smaller)")
    print(f"    rebuild fidelity  : {fidelity:.3f} mean cosine to the original")
    print("    (smaller memory, a small precision/recall cost -- the PQ bargain;")
    print("     pair with IVF as IVF+PQ to cluster AND compress.)")

    print("\n" + "=" * 70)
    print("Takeaway: brute force is exact but O(n); IVF gives up a sliver of")
    print("recall to skip most comparisons; nprobe slides you along that curve;")
    print("PQ shrinks the vectors themselves. Speed vs recall, all the way down.")
    print("=" * 70)
