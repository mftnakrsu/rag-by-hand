"""
Structured and SQL RAG: retrieve the schema, generate SQL, execute, answer.
RAG from First Principles, Part 18 (the close of the series).

For seventeen parts we treated knowledge as text: documents chunked into
passages, embedded, retrieved by similarity, read by a model. That works when
the answer is SITTING in a passage, waiting to be found. This final part
confronts the knowledge that does not work that way: the rows, columns, and
tables that hold most of what an enterprise actually knows.

Ask "what was our total revenue from shipped orders?" and there is no passage
anywhere that holds the answer. The number does not exist as text until someone
COMPUTES it: join orders to products, filter to shipped, multiply price by
quantity, sum. Dense passage retrieval (the engine of every part before this
one) finds text similar in meaning to the query. It is very good at that and
completely unable to do arithmetic over a set of rows. So the pipeline changes
shape. The document spine was: embed, retrieve, ground, generate. The
structured spine swaps two words:

    question
      -> retrieve the relevant SCHEMA subset (not passages)
      -> generate a SQL query grounded in that schema
      -> EXECUTE the SQL against the database
      -> answer from the result rows

This file builds a tiny SQLite database in memory (stdlib sqlite3 only), then
runs that loop end to end. Two pieces are deliberately mocked so the demo needs
no LLM and no network:

  - RETRIEVAL is keyword overlap between the question and each table's schema
    card (name + column names + a one-line description), not embeddings. The
    SHAPE is the point: in production you embed the schema cards and retrieve the
    nearest few; here the lexical scorer keeps it dependency-free, and the chosen
    tables are identical for these demo questions either way. If
    sentence-transformers is installed and a model is cached, the REAL embedder
    path below scores the cards by cosine instead, with no network; only the
    scores change, not the retrieved tables.

  - SQL GENERATION is a small rule-based stub, not a model. A real system sends
    the retrieved schema + a domain glossary + few-shot SQL exemplars to an LLM
    and parses the SQL it returns. The stub matches a handful of intents so the
    demo prints a real answer; the LOAD-BEARING part is the control flow around
    it (retrieve schema -> generate SQL -> execute -> answer), not the stub.

The EXECUTION step is genuinely real: the generated SQL runs against a real
SQLite database and returns real rows. That is the part that transfers. Swap the
keyword scorer for an embedder and the stub for a model, and you have a working
text-to-SQL RAG system with the same spine.

The router at the bottom ties back to Part 15: aggregational / numeric questions
("how many", "total", "average", "top", "revenue") route to the SQL path;
everything else falls back to the document path (the Parts 6-17 loop, mocked
here as a stub since you already built the real one). Part 15 routed by
COMPLEXITY (none / single / multi); this adds a second axis, STRUCTURE, in the
same classifier.

Run:
  python3 sql_rag.py                # runs offline; no API key, no network
  # Optional, for the REAL schema embedder path: pip install sentence-transformers

NOTE: the schema retriever and the SQL "generator" here are intentionally
transparent (lexical overlap + a handful of rules), not learned. That is the
teaching point: you can see exactly WHY each table is retrieved and WHY each
query is emitted. The execution and the control flow are the production ones.
"""

import re
import sqlite3


# ---------------------------------------------------------------------------
# 1. A tiny structured database. Three tables an e-commerce support bot might
#    sit in front of. This is the kind of knowledge that lives in a database,
#    not in documents: it is queried and aggregated, never "read".
# ---------------------------------------------------------------------------
def build_db():
    db = sqlite3.connect(":memory:")
    db.executescript(
        """
        CREATE TABLE customers (
            id      INTEGER PRIMARY KEY,
            name    TEXT,
            country TEXT,
            tier    TEXT          -- 'free' | 'pro' | 'enterprise'
        );
        CREATE TABLE products (
            id       INTEGER PRIMARY KEY,
            name     TEXT,
            category TEXT,
            price    REAL
        );
        CREATE TABLE orders (
            id          INTEGER PRIMARY KEY,
            customer_id INTEGER,
            product_id  INTEGER,
            quantity    INTEGER,
            status      TEXT,     -- 'shipped' | 'refunded' | 'pending'
            order_date  TEXT      -- ISO date
        );
        """
    )
    db.executemany(
        "INSERT INTO customers VALUES (?,?,?,?)",
        [
            (1, "Ada", "DE", "pro"),
            (2, "Bashir", "TR", "enterprise"),
            (3, "Chen", "SG", "free"),
            (4, "Dilan", "TR", "pro"),
        ],
    )
    db.executemany(
        "INSERT INTO products VALUES (?,?,?,?)",
        [
            (10, "Aurora Lamp", "lighting", 49.0),
            (11, "Nimbus Speaker", "audio", 129.0),
            (12, "Coil Cable", "audio", 9.0),
        ],
    )
    db.executemany(
        "INSERT INTO orders VALUES (?,?,?,?,?,?)",
        [
            (100, 1, 11, 2, "shipped", "2026-03-04"),
            (101, 2, 10, 1, "shipped", "2026-03-09"),
            (102, 2, 12, 5, "refunded", "2026-03-12"),
            (103, 4, 11, 1, "shipped", "2026-03-20"),
            (104, 3, 10, 3, "pending", "2026-03-28"),
        ],
    )
    db.commit()
    return db


# ---------------------------------------------------------------------------
# 2. Schema cards: one short, retrievable description per table. In production
#    these carry column descriptions and a few sample rows too, and you embed
#    them. Here each card is a name, its columns, and a one-line gloss; we
#    retrieve by keyword overlap so the file stays stdlib-only.
# ---------------------------------------------------------------------------
SCHEMA_CARDS = {
    "customers": {
        "columns": ["id", "name", "country", "tier"],
        "doc": "one row per customer: their name, country code, and plan tier "
               "(free, pro, enterprise). Use for how many customers questions.",
    },
    "products": {
        "columns": ["id", "name", "category", "price"],
        "doc": "one row per product for sale: its name, category, and unit price. "
               "Revenue is computed from price here times order quantity.",
    },
    "orders": {
        "columns": ["id", "customer_id", "product_id", "quantity", "status",
                    "order_date"],
        "doc": "one row per order placed: which customer bought which product, "
               "the quantity, the status (shipped, refunded, pending), and the date. "
               "Revenue and refunded order counts come from here.",
    },
}

# A domain glossary maps business words the user says to the columns/values that
# encode them. This is what stops the model guessing that "revenue" lives in some
# column that does not exist. A real system retrieves the relevant glossary lines
# alongside the schema; here it just documents the mapping the stub relies on.
GLOSSARY = {
    "revenue": "price * quantity, summed over orders (no revenue COLUMN exists)",
    "enterprise customer": "customers.tier = 'enterprise'",
    "refund / refunded": "orders.status = 'refunded'",
}


def _tokens(text):
    return set(re.findall(r"[a-z0-9]+", text.lower()))


def _card_text(table, card):
    """The text we score / embed for one card: name + columns + the gloss."""
    return table + " " + " ".join(card["columns"]) + " " + card["doc"]


# --- The real schema embedder path (headline; falls back transparently) -----
# In production you EMBED each schema card and retrieve the nearest few by
# cosine, exactly as Part 6 embeds document chunks. We try that first; if
# sentence-transformers is missing or no model is cached, we drop to the lexical
# overlap scorer below. Either way the retrieved tables are identical for these
# demo questions, so the lesson (retrieve a SUBSET of the schema) is the same.
def _load_real_embedder():
    try:
        from sentence_transformers import SentenceTransformer  # REAL path
        model = SentenceTransformer("all-MiniLM-L6-v2")
        print("[schema] using sentence-transformers to embed schema cards")
        return model
    except Exception as exc:  # not installed / no weights / offline
        print(f"[schema] sentence-transformers unavailable ({type(exc).__name__}); "
              "scoring schema cards by keyword overlap (offline default)")
        return None


_EMBEDDER = _load_real_embedder()


def _overlap_score(question, table, card):
    """Lexical fallback: count shared tokens between question and card text."""
    return len(_tokens(question) & _tokens(_card_text(table, card)))


def _embed_score(question, table, card):
    """Real path: cosine between the question and the card embedding."""
    import numpy as np
    vecs = _EMBEDDER.encode([question, _card_text(table, card)],
                            normalize_embeddings=True)
    return float(np.dot(vecs[0], vecs[1]))


def retrieve_schema(question, k=2):
    """Score each table's card against the question, keep the top-k.
    Returns [(table, score, card)]. This is the SHAPE of real schema retrieval
    (embed the cards, take the nearest few). The point is that we retrieve a
    SUBSET of the schema, never the whole catalog; for a large database this
    schema-linking step is what makes text-to-SQL work at all."""
    scored = []
    for table, card in SCHEMA_CARDS.items():
        if _EMBEDDER is not None:
            score = _embed_score(question, table, card)
        else:
            score = _overlap_score(question, table, card)
        scored.append((table, score, card))
    scored.sort(key=lambda x: -x[1])
    return scored[:k]


def render_schema(retrieved):
    """The retrieved schema subset, formatted the way you would paste it into a
    prompt: CREATE-style column lists plus the one-line description per table."""
    lines = []
    for table, _score, card in retrieved:
        cols = ", ".join(card["columns"])
        lines.append(f"TABLE {table}({cols})  -- {card['doc']}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 3. SQL generation: a rule-based STUB standing in for an LLM. A real system
#    sends (question + retrieved schema + glossary + few-shot SQL exemplars) to
#    a model and parses the SQL it returns. This stub recognizes the handful of
#    intents the demo asks about, so it can emit real, executable SQL. The
#    control flow around it is identical to the production one.
# ---------------------------------------------------------------------------
def generate_sql(question, retrieved):
    q = question.lower()
    tables = {t for t, _, _ in retrieved}

    # Intent is read from the QUESTION first; the retrieved schema only has to
    # contain the tables the chosen SQL touches. Order matters: the most specific
    # intent (refunds, revenue) is matched before the generic customer count, so
    # "how many orders were refunded?" is not captured by the "how many" branch.

    # "total revenue" -> revenue is price*quantity (glossary), so JOIN + SUM.
    # Note the GUARD: this needs BOTH orders and products in the retrieved
    # schema. If retrieval handed us too few tables, we decline rather than
    # guess (try k=1 to see this fire).
    if "revenue" in q and {"orders", "products"} <= tables:
        return (
            "SELECT SUM(p.price * o.quantity) "
            "FROM orders o JOIN products p ON o.product_id = p.id "
            "WHERE o.status = 'shipped';"
        )

    # "how many refunded orders" -> COUNT over orders filtered by status.
    if "refund" in q and "orders" in tables:
        return "SELECT COUNT(*) FROM orders WHERE status = 'refunded';"

    # "how many <tier> customers" -> COUNT over customers, filtered by tier.
    if "how many" in q and "customers" in tables:
        for tier in ("enterprise", "pro", "free"):
            if tier in q:
                return f"SELECT COUNT(*) FROM customers WHERE tier = '{tier}';"
        return "SELECT COUNT(*) FROM customers;"

    # Fallback: no intent matched, or the retrieved schema is missing a table the
    # query needs. A real system would ask to clarify or decline; we surface the
    # miss instead of guessing wrong SQL. On structured data, declining loudly
    # beats fabricating a confident, ungrounded number.
    return None


def answer_from_rows(question, sql, rows):
    """Turn the result rows into a one-line answer. A real system passes the rows
    back to the model to phrase; here we format the single scalar each demo
    query returns, which is enough to show the loop closed correctly."""
    if not rows:
        return "No rows matched."
    value = rows[0][0]
    return f"{value}"


# ---------------------------------------------------------------------------
# 4. The SQL-RAG loop, end to end: retrieve schema -> generate SQL -> execute
#    -> answer. Returns everything so the demo can show the work at each stage.
# ---------------------------------------------------------------------------
def sql_rag(db, question, k=2):
    retrieved = retrieve_schema(question, k=k)
    sql = generate_sql(question, retrieved)
    if sql is None:
        return {
            "tables": [t for t, _, _ in retrieved],
            "sql": None,
            "answer": "I could not turn that into a query over the known schema.",
        }
    rows = db.execute(sql).fetchall()        # <-- the real execution step
    return {
        "tables": [t for t, _, _ in retrieved],
        "sql": sql,
        "answer": answer_from_rows(question, sql, rows),
    }


# ---------------------------------------------------------------------------
# 5. The router (Part 15 callback): decide whether a question is STRUCTURED
#    (aggregational / numeric -> the SQL path) or unstructured (-> documents).
#    The document path is mocked as a stub here; the point is the FORK. This is
#    the same machinery as Part 15's classify_complexity, with one more axis:
#    Part 15 asked HOW MUCH (none/single/multi); here we ask WHAT KIND
#    (structured vs unstructured). In production both decisions compose.
# ---------------------------------------------------------------------------
SQL_SIGNALS = ("how many", "count", "total", "sum", "average", "avg",
               "most", "top", "revenue", "per ", "number of")


def route(question):
    q = question.lower()
    if any(s in q for s in SQL_SIGNALS):
        return "sql"
    return "documents"


if __name__ == "__main__":
    db = build_db()

    print("Glossary (business word -> what it means in the schema):")
    for term, meaning in GLOSSARY.items():
        print(f"  {term:<22} = {meaning}")
    print()

    DEMO = [
        "How many enterprise customers do we have?",
        "What was our total revenue from shipped orders?",
        "How many orders were refunded?",
        # Routes to the document path: not aggregational, no SQL signal.
        "What is your refund policy?",
    ]

    for q in DEMO:
        r = route(q)
        print(f"Q: {q}")
        print(f"   route={r}")
        if r != "sql":
            print("   -> document path (Parts 6-17): retrieve passages, ground, generate.\n")
            continue
        out = sql_rag(db, q)
        print(f"   schema retrieved: {out['tables']}")
        print(f"   SQL: {out['sql']}")
        print(f"   answer: {out['answer']}\n")

    # Confirm the loop produced the values we expect from the seeded data.
    checks = {
        "How many enterprise customers do we have?": "1",       # Bashir
        "What was our total revenue from shipped orders?": "436.0",  # 129*2 + 49*1 + 129*1
        "How many orders were refunded?": "1",                  # order 102
    }
    ok = all(sql_rag(db, q)["answer"] == want for q, want in checks.items())
    print("all SQL answers match the seeded data:", ok)


# Expected output (lexical fallback path, i.e. sentence-transformers not installed):
#
# [schema] sentence-transformers unavailable (ModuleNotFoundError); scoring schema cards by keyword overlap (offline default)
# Glossary (business word -> what it means in the schema):
#   revenue                = price * quantity, summed over orders (no revenue COLUMN exists)
#   enterprise customer    = customers.tier = 'enterprise'
#   refund / refunded      = orders.status = 'refunded'
#
# Q: How many enterprise customers do we have?
#    route=sql
#    schema retrieved: ['customers', 'products']
#    SQL: SELECT COUNT(*) FROM customers WHERE tier = 'enterprise';
#    answer: 1
#
# Q: What was our total revenue from shipped orders?
#    route=sql
#    schema retrieved: ['orders', 'products']
#    SQL: SELECT SUM(p.price * o.quantity) FROM orders o JOIN products p ON o.product_id = p.id WHERE o.status = 'shipped';
#    answer: 436.0
#
# Q: How many orders were refunded?
#    route=sql
#    schema retrieved: ['customers', 'orders']
#    SQL: SELECT COUNT(*) FROM orders WHERE status = 'refunded';
#    answer: 1
#
# Q: What is your refund policy?
#    route=documents
#    -> document path (Parts 6-17): retrieve passages, ground, generate.
#
# all SQL answers match the seeded data: True
#
# Notes on what is mocked vs load-bearing:
#  - Schema retrieval is keyword overlap (or real cosine if a model is cached),
#    not the whole catalog; the SHAPE (retrieve a SUBSET of the schema) is the
#    lesson. The top-2 tables for "revenue" are orders+products, exactly what the
#    JOIN needs. Set k=1 and the revenue query DECLINES rather than guessing.
#  - SQL generation is a rule-based stub, not an LLM; the control flow around it
#    (retrieve schema -> generate SQL -> EXECUTE -> answer from rows) is the part
#    that transfers. The execution step is real: the SQL runs against real SQLite.
#  - Revenue = 129*2 (order 100) + 49*1 (order 101) + 129*1 (order 103) = 436.0,
#    counting only shipped orders, exactly as the WHERE clause says.
