"""
Documents and Chunking, by hand.
RAG from First Principles, Part 5: how a raw document becomes the chunks we embed.

This is the cut itself. Parts 2-4 turned chunks into embeddings, scored them by
cosine similarity, and retrieved the top-k fast. The whole time we leaned on one
word we never earned: "chunks". Real sources are not chunks. They are PDFs, web
pages, message threads, slide decks. Someone has to decide where to slice. That
decision is chunking, and its mistakes propagate up through everything: garbage
chunks in, garbage retrieval out.

This file walks the ladder of strategies from the essay, fastest/blindest first:
  - fixed-size      : blind cuts every N characters
  - recursive       : try paragraph -> sentence -> word boundaries under a cap
  - sliding window  : chunk overlap, so an idea on a seam survives whole
  - structure-aware : split on the document's own structure (Markdown headers)
  - semantic-ish    : cut where adjacent-sentence similarity drops
  - metadata        : attach source/section + a context prefix to each chunk

Stack:
  - Pure standard library. numpy is optional and only used (if present) to make
    the semantic-similarity stand-in a touch tidier; the demo runs without it.
  - The "real" recursive splitter in production is
    langchain_text_splitters.RecursiveCharacterTextSplitter; the "real" semantic
    cut uses an embedding model (Part 2). Both are shown as labelled reference
    code, but the executable demo uses a transparent pure-Python stand-in so it
    runs fully offline, with NO network and NO API key.

Run:
  python3 chunking.py
"""

from __future__ import annotations

import re

# numpy is optional. We only reach for it in one tiny helper; if it is missing
# we fall back to plain Python so the demo always runs. (The later parts use
# this same "transparent fallback that runs without the heavy dep" pattern.)
try:
    import numpy as np  # noqa: F401
    HAVE_NUMPY = True
except Exception:  # pragma: no cover - numpy is installed here, but be safe.
    HAVE_NUMPY = False


# ---------------------------------------------------------------------------
# Step 0. A small, eyeball-able sample document.
#         Short on purpose, with real Markdown headers so the structure-aware
#         splitter has something to follow, and a few topics so the semantic
#         splitter has a topic shift to find. This is the same store-policy
#         world as the other parts (refunds, shipping, warranty).
# ---------------------------------------------------------------------------
SAMPLE_DOC = """# Refund Policy

## Returns

Refunds are accepted within 30 days of purchase, provided the item is unused \
and in its original packaging. To start a return, email support@example.com \
with your order number. Refunds are processed within five business days of us \
receiving the item.

## Shipping

Standard shipping takes 3 to 5 business days. Express shipping arrives the next \
business day. Shipping fees are non-refundable. Items marked final sale cannot \
be returned or exchanged.

## Warranty

All electronics include a one-year limited warranty covering manufacturing \
defects. The warranty does not cover accidental damage or normal wear and tear.
"""

# The document's title, used later as a context prefix (the "contextual" seed).
SAMPLE_TITLE = "Refund Policy"


# ===========================================================================
# Strategy 1. FIXED-SIZE CHUNKING
#   Split the text every N characters, full stop. Simplest, fastest, most
#   predictable, and blind: it will happily cut mid-sentence, even mid-word,
#   because it counts characters and nothing else. Good for a quick baseline.
#   This is exactly the one-liner from the essay, generalized to a stride.
# ===========================================================================
def fixed_size_chunks(text: str, size: int = 200, stride: int | None = None) -> list[str]:
    """Blind cuts. With stride < size you get a sliding window (overlap).

    The essay's worked one-liner is `[text[i:i+200] for i in range(0, len(text), 200)]`.
    Here `stride` controls how far we advance each step; stride == size means no
    overlap, stride < size means consecutive chunks share text at their seams.
    """
    if stride is None:
        stride = size  # advance by a full chunk => adjacent, non-overlapping
    return [text[i:i + size] for i in range(0, len(text), stride)]


# ===========================================================================
# Strategy 2. RECURSIVE CHARACTER CHUNKING
#   Try a hierarchy of separators in order, coarsest first:
#       ["\n\n", "\n", ". ", " ", ""]
#   Paragraphs, then lines, then sentences, then words, then raw characters.
#   Split on the COARSEST boundary that keeps pieces under the size budget,
#   only resorting to finer cuts when a piece is still too big. This respects
#   natural boundaries and is the sensible default for prose.
#
#   The library equivalent (production):
#       from langchain_text_splitters import RecursiveCharacterTextSplitter
#       splitter = RecursiveCharacterTextSplitter(
#           chunk_size=200, chunk_overlap=20,
#           separators=["\n\n", "\n", ". ", " ", ""],
#       )
#       recursive = splitter.split_text(text)
#   We re-implement the core idea in pure Python so it runs with no deps.
# ===========================================================================
DEFAULT_SEPARATORS = ["\n\n", "\n", ". ", " ", ""]


def _split_keep_separator(text: str, sep: str) -> list[str]:
    """Split on `sep` but keep `sep` attached to the LEFT piece.

    Keeping the separator means re-joining pieces never silently drops the
    paragraph break / period / space we split on, so chunk lengths stay honest.
    The empty separator "" means "split into individual characters".
    """
    if sep == "":
        return list(text)
    parts = text.split(sep)
    out = []
    for i, p in enumerate(parts):
        if i < len(parts) - 1:
            out.append(p + sep)  # reattach the separator we just split on
        elif p:                  # last piece: no trailing separator
            out.append(p)
    return out


def recursive_split(text: str, chunk_size: int = 200,
                    separators: list[str] | None = None) -> list[str]:
    """Recursively split `text` so every chunk is <= chunk_size where possible.

    For the current separator: cut the text into pieces, then greedily merge
    adjacent pieces back together while they still fit under chunk_size. Any
    single piece that is STILL too big is handed down to the next, finer
    separator and split again. This is the heart of recursive splitting.
    """
    if separators is None:
        separators = DEFAULT_SEPARATORS

    # If it already fits, we are done. No need to cut a coherent thought.
    if len(text) <= chunk_size:
        return [text] if text else []

    sep = separators[0]
    finer = separators[1:]
    pieces = _split_keep_separator(text, sep)

    chunks: list[str] = []
    current = ""
    for piece in pieces:
        # A single piece bigger than the budget at this level can't be fixed by
        # merging; flush what we have and recurse on it with a finer separator.
        if len(piece) > chunk_size:
            if current:
                chunks.append(current)
                current = ""
            if finer:
                chunks.extend(recursive_split(piece, chunk_size, finer))
            else:
                # No finer separator left ("" already splits to characters):
                # hard-cut as a last resort, like fixed-size.
                chunks.extend(piece[i:i + chunk_size]
                              for i in range(0, len(piece), chunk_size))
            continue

        # Otherwise, try to grow the current chunk with this piece.
        if len(current) + len(piece) <= chunk_size:
            current += piece
        else:
            if current:
                chunks.append(current)
            current = piece

    if current:
        chunks.append(current)
    return chunks


# ===========================================================================
# Strategy 3. SLIDING-WINDOW OVERLAP
#   Chunk overlap is the fix for the worst failure of chunk size: an idea that
#   straddles a boundary gets sliced in half. A sliding window makes consecutive
#   chunks share text at their seams, so the straddling sentence survives whole
#   in at least one chunk. We add overlap on top of any pre-made chunk list by
#   prepending the tail of the previous chunk to the next one.
# ===========================================================================
def add_overlap(chunks: list[str], overlap: int = 20) -> list[str]:
    """Prepend the last `overlap` characters of each chunk to the next one.

    Rule of thumb from the essay: overlap of roughly 10-20% of the chunk size.
    With overlap == 0 this is a no-op (adjacent, non-overlapping chunks).
    """
    if overlap <= 0 or len(chunks) < 2:
        return list(chunks)
    out = [chunks[0]]
    for prev, cur in zip(chunks, chunks[1:]):
        tail = prev[-overlap:]      # the seam we carry forward
        out.append(tail + cur)
    return out


# ===========================================================================
# Strategy 4. DOCUMENT-STRUCTURE-AWARE CHUNKING
#   Split along the document's OWN structure rather than generic punctuation.
#   Here: Markdown headers (lines starting with #). Each header begins a new
#   section, and we keep the heading text so the chunk knows what it is about.
#   The same idea maps to HTML sections, functions in source, slides in a deck.
# ===========================================================================
HEADER_RE = re.compile(r"^(#{1,6})\s+(.*)$")


def structure_aware_split(text: str) -> list[dict]:
    """Split on Markdown headers, returning {heading, level, text} sections.

    Returns dicts (not bare strings) because the heading is structure worth
    keeping: it feeds both metadata (Strategy 6) and the context prefix.
    """
    sections: list[dict] = []
    current = {"heading": None, "level": 0, "lines": []}

    def flush():
        body = "\n".join(current["lines"]).strip()
        if current["heading"] is not None or body:
            sections.append({
                "heading": current["heading"],
                "level": current["level"],
                "text": body,
            })

    for line in text.splitlines():
        m = HEADER_RE.match(line)
        if m:
            flush()                              # close the previous section
            level = len(m.group(1))              # number of '#' => heading depth
            current = {"heading": m.group(2).strip(), "level": level, "lines": []}
        else:
            current["lines"].append(line)
    flush()                                      # close the final section
    return sections


# ===========================================================================
# Strategy 5. SEMANTIC-ISH CHUNKING
#   The smartest and slowest: walk the text keeping adjacent sentences together
#   while they stay similar in MEANING, and cut precisely where the similarity
#   between neighbors drops. Boundaries land on genuine changes of subject.
#
#   The real version (production) embeds each sentence with a model (Part 2) and
#   measures cosine similarity between neighbors (Part 3):
#
#       from sentence_transformers import SentenceTransformer
#       model = SentenceTransformer("all-MiniLM-L6-v2")
#       vecs = model.encode(sentences, normalize_embeddings=True)  # unit length
#       sims = [float(vecs[i] @ vecs[i + 1]) for i in range(len(vecs) - 1)]
#       # ...then cut where sims[i] falls below a threshold.
#
#   OFFLINE STAND-IN: we have no model, so we approximate "do these neighbors
#   talk about the same thing?" with a cheap, deterministic LEXICAL OVERLAP
#   (Jaccard similarity of their word sets). Same shape of signal -- a number in
#   [0, 1] that is high for on-topic neighbors and low at a topic shift -- so the
#   cutting logic is identical to the embedding version; only the scorer differs.
# ===========================================================================
def split_sentences(text: str) -> list[str]:
    """Naive sentence split on ., !, ? followed by whitespace. Good enough here."""
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []
    parts = re.split(r"(?<=[.!?])\s+", text)
    return [p.strip() for p in parts if p.strip()]


_WORD_RE = re.compile(r"[a-z0-9]+")


def _words(s: str) -> set[str]:
    return set(_WORD_RE.findall(s.lower()))


def lexical_similarity(a: str, b: str) -> float:
    """Jaccard overlap of word sets: |A ∩ B| / |A ∪ B|, in [0, 1].

    Pure-Python stand-in for cosine similarity between two sentence embeddings.
    High when two sentences share vocabulary (likely same topic), low when they
    do not (likely a topic shift). Deterministic and offline.
    """
    wa, wb = _words(a), _words(b)
    if not wa and not wb:
        return 1.0
    inter = len(wa & wb)
    union = len(wa | wb)
    return inter / union if union else 0.0


def semantic_split(text: str, threshold: float = 0.08) -> list[str]:
    """Group adjacent sentences; cut where neighbor similarity drops below
    `threshold`. Chunk sizes vary -- that unpredictability is inherent to
    semantic chunking, and the price you pay for boundaries on real meaning.
    """
    sentences = split_sentences(text)
    if not sentences:
        return []

    chunks: list[str] = []
    current = [sentences[0]]
    for prev, sent in zip(sentences, sentences[1:]):
        sim = lexical_similarity(prev, sent)
        if sim < threshold:
            # Similarity dropped: the subject changed here. Cut.
            chunks.append(" ".join(current))
            current = [sent]
        else:
            current.append(sent)
    chunks.append(" ".join(current))
    return chunks


# ===========================================================================
# Step 6. METADATA ENRICHMENT
#   A chunk is more than its text. At split time, attach metadata (source,
#   section, ...) -- it powers the metadata FILTERING from Part 4 and the clean
#   CITATIONS later. And prepend a little context (title + heading) so an
#   isolated chunk stays self-explanatory: "Refund Policy > Returns: <text>".
#   This is the seed of the "contextual" techniques later parts explore.
# ===========================================================================
def enrich(section: dict, source: str, title: str) -> dict:
    """Turn a {heading, text} section into a stored chunk with metadata and a
    context-prefixed `text` field ready to embed."""
    heading = section.get("heading")
    body = section.get("text", "")
    # Context prefix: title (> heading) so the chunk carries its own origin.
    prefix = title if not heading else f"{title} > {heading}"
    return {
        "text": f"{prefix}: {body}" if body else prefix,
        "raw_text": body,
        "metadata": {
            "source": source,
            "section": heading,
            "title": title,
        },
    }


# ---------------------------------------------------------------------------
# A tiny helper for the demo: print a list of chunks with their lengths.
# ---------------------------------------------------------------------------
def show(label: str, chunks: list[str], limit: int = 70) -> None:
    print(f"{label}  ->  {len(chunks)} chunk(s)")
    for i, c in enumerate(chunks):
        preview = c.replace("\n", "\\n")
        if len(preview) > limit:
            preview = preview[:limit] + "..."
        print(f"  [{i}] len={len(c):>3}  {preview!r}")
    print()


# ===========================================================================
# Step 7. The demo. Run each strategy on the sample document and show how the
#         boundaries move. Same document, same size budget, very different cuts.
# ===========================================================================
if __name__ == "__main__":
    print("=" * 72)
    print("Documents and Chunking, by hand (Part 5)")
    print("numpy available:", HAVE_NUMPY, "(optional; demo runs either way)")
    print("=" * 72)
    print()

    text = SAMPLE_DOC
    print(f"Sample document: {len(text)} characters, "
          f"title = {SAMPLE_TITLE!r}\n")

    # -- Strategy 1: fixed-size, blind cuts every 200 chars --------------------
    print("-" * 72)
    print("1) FIXED-SIZE (size=200): blind, cuts mid-sentence / mid-word")
    print("-" * 72)
    fixed = fixed_size_chunks(text, size=200)
    show("fixed-size", fixed)

    # -- Strategy 2: recursive, prefers natural boundaries --------------------
    print("-" * 72)
    print("2) RECURSIVE (chunk_size=200): paragraph -> line -> sentence -> word")
    print("-" * 72)
    recursive = recursive_split(text, chunk_size=200)
    show("recursive", recursive)
    print("Note: same 200-char budget as fixed-size, but these break at natural")
    print("boundaries, so each chunk reads like a coherent thought.\n")

    # -- Strategy 3: sliding-window overlap -----------------------------------
    print("-" * 72)
    print("3) OVERLAP (sliding window, overlap=20 ~= 10% of 200)")
    print("-" * 72)
    overlapped = add_overlap(recursive, overlap=20)
    show("recursive + overlap", overlapped)
    if len(recursive) >= 2:
        seam = recursive[0][-20:]
        print(f"The 20-char seam carried forward from chunk [0] -> [1]:")
        print(f"  {seam!r}")
        print("So an idea on the boundary survives whole in chunk [1].\n")

    # -- Strategy 4: structure-aware (Markdown headers) -----------------------
    print("-" * 72)
    print("4) STRUCTURE-AWARE: split on Markdown headers (the author's own units)")
    print("-" * 72)
    sections = structure_aware_split(text)
    for s in sections:
        head = s["heading"] if s["heading"] is not None else "(preamble)"
        body = s["text"].replace("\n", " ")
        if len(body) > 60:
            body = body[:60] + "..."
        print(f"  h{s['level']} {head!r:<24} -> {body!r}")
    print()

    # -- Strategy 5: semantic-ish (cut where similarity drops) ----------------
    print("-" * 72)
    print("5) SEMANTIC-ISH: cut where adjacent-sentence similarity drops")
    print("   (offline stand-in: lexical Jaccard overlap in place of embeddings)")
    print("-" * 72)
    # Run on the Returns + Shipping prose so there is a real topic shift to find.
    prose = " ".join(s["text"] for s in sections if s["text"])
    sents = split_sentences(prose)
    print("Adjacent-sentence similarities (low value = topic shift = a cut):")
    for prev, nxt in zip(sents, sents[1:]):
        sim = lexical_similarity(prev, nxt)
        marker = "  <-- CUT" if sim < 0.08 else ""
        print(f"  sim={sim:.3f}  {prev[:34]!r:<38} | {nxt[:30]!r}{marker}")
    print()
    semantic = semantic_split(prose, threshold=0.08)
    show("semantic-ish", semantic)

    # -- Step 6: metadata enrichment + context prefix -------------------------
    print("-" * 72)
    print("6) METADATA ENRICHMENT: attach source/section + a context prefix")
    print("-" * 72)
    enriched = [enrich(s, source="refund-policy.md", title=SAMPLE_TITLE)
                for s in sections if s["text"]]
    for c in enriched:
        meta = c["metadata"]
        print(f"  metadata: source={meta['source']!r} section={meta['section']!r}")
        prefixed = c["text"].replace("\n", " ")
        if len(prefixed) > 64:
            prefixed = prefixed[:64] + "..."
        print(f"  text    : {prefixed!r}")
        print()
    print("Each chunk now carries its origin (for citations + Part 4 filtering)")
    print("and a heading prefix so it stays self-explanatory once retrieved.")
    print()

    print("=" * 72)
    print("Same document, six lenses. The cut is the foundation everything else")
    print("(embeddings, similarity, retrieval) stands on. Next: Part 6 assembles")
    print("clean chunks + embeddings + similarity + retrieval into a full RAG app.")
    print("=" * 72)
