"""
A minimal, from-scratch Retrieval-Augmented Generation app.
RAG from First Principles, Part 6: build it by hand, understand every line.

Stack:
  - Embeddings : sentence-transformers (a small local model, free and offline)
  - Vector store: a Python list of chunks + a NumPy array of vectors (transparent)
  - Generation : one swappable generate(prompt) function (hosted API or local Ollama)

Run:
  pip install -r requirements.txt
  # For generation, either set OPENAI_API_KEY in your environment,
  # or use the Ollama version of generate() further down (free, local, no key).
  python rag_app.py

NOTE: LLM SDK syntax and model names move fast and may have changed since this
was written. Check the current provider docs; only generate() needs editing.
"""

import numpy as np
from sentence_transformers import SentenceTransformer

# ---------------------------------------------------------------------------
# Step 0. A tiny, eyeball-able corpus: a few short docs on one topic.
# ---------------------------------------------------------------------------
CORPUS = [
    "Refunds are accepted within 30 days of purchase, provided the item is unused and in its original packaging.",
    "To start a return, email support@example.com with your order number. Refunds are processed within five business days of us receiving the item.",
    "Standard shipping takes 3 to 5 business days. Express shipping arrives the next business day.",
    "Shipping fees are non-refundable, and items marked final sale cannot be returned or exchanged.",
    "All electronics include a one-year limited warranty covering manufacturing defects.",
]


# ---------------------------------------------------------------------------
# Step 1. Load and chunk. Our docs are short, so each doc is one chunk.
#         Attach metadata here (Part 5); we keep a source field for citations.
# ---------------------------------------------------------------------------
def chunk(corpus):
    return [{"text": doc, "source": f"doc_{i}"} for i, doc in enumerate(corpus)]


# ---------------------------------------------------------------------------
# Step 2. Embed. Encode each chunk with a small local model (Part 2).
#         normalize_embeddings=True makes every vector unit length, so a dot
#         product equals cosine similarity later (the trick from Part 3).
# ---------------------------------------------------------------------------
model = SentenceTransformer("all-MiniLM-L6-v2")  # 384 dimensions


def embed(texts):
    return model.encode(texts, normalize_embeddings=True)


# ---------------------------------------------------------------------------
# Step 3. Store. The "vector store" is just the chunks plus a matrix of their
#         vectors, kept side by side. The vector finds a chunk; the text is
#         what we feed the model, so we keep them together (Part 4).
# ---------------------------------------------------------------------------
chunks = chunk(CORPUS)
vectors = embed([c["text"] for c in chunks])  # shape: (n_chunks, 384)


# ---------------------------------------------------------------------------
# Step 4. Retrieve. Embed the query with the SAME model, score by cosine
#         similarity against every chunk, and keep the top-k (Part 3 + Part 4).
# ---------------------------------------------------------------------------
def retrieve(query, k=3):
    q = embed([query])[0]                       # (384,), same model as the chunks
    scores = vectors @ q                        # cosine sim, since all are unit length
    top = np.argsort(-scores)[:k]               # indices of the k highest scores
    return [(chunks[i]["text"], float(scores[i])) for i in top]


# ---------------------------------------------------------------------------
# Step 5. Augment. Inject the retrieved context into a prompt template with an
#         explicit grounding instruction that curbs hallucination (Part 1).
# ---------------------------------------------------------------------------
PROMPT_TEMPLATE = """Answer the question using ONLY the context below.
If the answer is not in the context, say "I don't know based on the provided documents."

Context:
{context}

Question: {question}
Answer:"""


def build_prompt(query, retrieved):
    context = "\n".join(f"- {text}" for text, _score in retrieved)
    return PROMPT_TEMPLATE.format(context=context, question=query)


# ---------------------------------------------------------------------------
# Step 6. Generate. One swappable function isolates the LLM provider.
#         Hosted example below; the local Ollama version is in the comment.
# ---------------------------------------------------------------------------
def generate(prompt):
    from openai import OpenAI
    client = OpenAI()                           # reads OPENAI_API_KEY from the env
    resp = client.chat.completions.create(
        model="gpt-4o-mini",                    # a small, cheap chat model; check names
        messages=[{"role": "user", "content": prompt}],
        temperature=0,                          # grounded, not creative
    )
    return resp.choices[0].message.content


# Local, zero-cost alternative (Ollama). Swap this in for generate() above:
#
# def generate(prompt):
#     import requests
#     r = requests.post(
#         "http://localhost:11434/api/generate",
#         json={"model": "llama3.1", "prompt": prompt, "stream": False},
#     )
#     return r.json()["response"]


# Anthropic / Claude alternative. Swap this in for generate() above:
#
# def generate(prompt):
#     from anthropic import Anthropic
#     client = Anthropic()                        # reads ANTHROPIC_API_KEY from the env
#     resp = client.messages.create(
#         model="claude-opus-4-8",                # check current model names
#         max_tokens=1024,                        # required by the Messages API
#         messages=[{"role": "user", "content": prompt}],
#     )                                           # (no temperature: removed on Opus 4.8)
#     return resp.content[0].text


# ---------------------------------------------------------------------------
# Step 7. The app. Retrieve, augment, generate, then a tiny REPL.
# ---------------------------------------------------------------------------
def ask(question, k=3):
    retrieved = retrieve(question, k=k)
    prompt = build_prompt(question, retrieved)
    return generate(prompt)


if __name__ == "__main__":
    print("Ask about the store policy (Ctrl-C to quit).\n")
    while True:
        try:
            q = input("> ").strip()
            if q:
                print(ask(q), "\n")
        except (EOFError, KeyboardInterrupt):
            print("\nBye.")
            break
