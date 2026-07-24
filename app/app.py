"""
Opposing-Argument Simulator — Phase 3: Streamlit UI
-----------------------------------------------------
Self-represented litigants describe their Illinois landlord-tenant
situation. The app retrieves relevant case law from RDS pgvector
and uses Claude (Amazon Bedrock) to simulate the opposing side's
arguments, grounded strictly in retrieved citations.

Run:
    streamlit run app.py
"""

import json
import os

import boto3
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

# ── Page config ────────────────────────────────────────────────
st.set_page_config(
    page_title="Opposing-Argument Simulator",
    page_icon="⚖️",
    layout="wide",
)

# ── Lazy-load retrieval components ─────────────────────────────
@st.cache_resource(show_spinner="Loading retrieval system...")
def load_retrieval():
    """Load embedding model + build BM25 index once per session."""
    import psycopg2
    import psycopg2.extras
    from rank_bm25 import BM25Okapi
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer("all-MiniLM-L6-v2")

    conn = psycopg2.connect(
        host=os.getenv("DB_HOST"),
        port=os.getenv("DB_PORT"),
        dbname=os.getenv("DB_NAME"),
        user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASSWORD"),
        connect_timeout=30,
    )

    cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
    cur.execute(
        """
        SELECT chunk_id, citation, date, chunk_index, total_chunks, chunk_text
        FROM case_chunks
        WHERE jurisdiction = 'Illinois' AND case_type = 'tenancy'
        ORDER BY id
        """
    )
    rows = [dict(r) for r in cur.fetchall()]
    conn.close()

    tokenized = [r["chunk_text"].lower().split() for r in rows]
    bm25 = BM25Okapi(tokenized)
    return model, bm25, rows


def retrieve(query: str, top_k: int = 5) -> list[dict]:
    import psycopg2
    import psycopg2.extras

    model, bm25, bm25_meta = load_retrieval()

    # Dense search
    vec = model.encode(query, normalize_embeddings=True).tolist()
    vec_str = "[" + ",".join(str(x) for x in vec) + "]"

    conn = psycopg2.connect(
        host=os.getenv("DB_HOST"), port=os.getenv("DB_PORT"),
        dbname=os.getenv("DB_NAME"), user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASSWORD"), connect_timeout=30,
    )
    cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
    cur.execute(
        """
        SELECT chunk_id, citation, date, chunk_index, total_chunks, chunk_text,
               1 - (embedding <=> %s::vector) AS score
        FROM case_chunks
        WHERE jurisdiction = 'Illinois' AND case_type = 'tenancy'
        ORDER BY embedding <=> %s::vector
        LIMIT 20
        """,
        (vec_str, vec_str),
    )
    dense = [dict(r) for r in cur.fetchall()]
    conn.close()

    # Sparse search
    tokens = query.lower().split()
    scores = bm25.get_scores(tokens)
    top_idx = sorted(range(len(scores)), key=lambda i: -scores[i])[:20]
    sparse = [
        {**bm25_meta[i], "score": float(scores[i])}
        for i in top_idx if scores[i] > 0
    ]

    # RRF fusion
    rrf_scores, sources, chunks = {}, {}, {}
    K = 60
    for rank, item in enumerate(dense, 1):
        cid = item["chunk_id"]
        rrf_scores[cid] = rrf_scores.get(cid, 0) + 1.0 / (K + rank)
        sources[cid] = "dense"
        chunks[cid] = item
    for rank, item in enumerate(sparse, 1):
        cid = item["chunk_id"]
        rrf_scores[cid] = rrf_scores.get(cid, 0) + 1.0 / (K + rank)
        sources[cid] = "both" if sources.get(cid) == "dense" else "sparse"
        chunks.setdefault(cid, item)

    ranked = sorted(rrf_scores.items(), key=lambda x: -x[1])[:top_k]
    return [
        {**chunks[cid], "rrf_score": score, "match_type": sources[cid]}
        for cid, score in ranked
    ]


def generate_arguments(situation: str, retrieved_chunks: list[dict]) -> str:
    """Call Claude via Bedrock to simulate opposing arguments."""
    context_blocks = []
    for i, chunk in enumerate(retrieved_chunks, 1):
        context_blocks.append(
            f"[{i}] {chunk['citation']} ({chunk['date'][:4] if chunk['date'] else 'n/a'}):\n"
            f"{chunk['chunk_text'][:600]}..."
        )
    context = "\n\n".join(context_blocks)

    prompt = f"""You are a legal argument simulator for Illinois landlord-tenant disputes.
A self-represented litigant has described their situation. Your job is to simulate the
OPPOSING SIDE'S strongest arguments — the arguments the other party's lawyer is likely
to raise at the hearing.

IMPORTANT RULES:
- Every argument you make MUST be grounded in one of the provided case citations below.
- Cite the case using its citation string (e.g. "355 Ill. App. 3d 885").
- Do NOT invent cases or statutes not present in the context.
- Be direct and adversarial — this is what the opposing side will actually argue.
- Structure your response as 3-5 numbered arguments.

LITIGANT'S SITUATION:
{situation}

RELEVANT ILLINOIS CASE LAW (retrieved from CAP corpus):
{context}

Now simulate the opposing side's 3-5 strongest arguments, each citing a specific case above:"""

    bedrock = boto3.client(
        "bedrock-runtime",
        region_name=os.getenv("AWS_REGION", "us-east-1"),
        aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
    )

    body = json.dumps({
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 1024,
        "messages": [{"role": "user", "content": prompt}],
    })

    model_id = os.getenv("BEDROCK_MODEL", "us.anthropic.claude-haiku-4-5-20251001-v1:0")
    response = bedrock.invoke_model(
        modelId=model_id,
        body=body,
        contentType="application/json",
        accept="application/json",
    )
    result = json.loads(response["body"].read())
    return result["content"][0]["text"]


# ── UI ─────────────────────────────────────────────────────────

st.title("⚖️ Opposing-Argument Simulator")
st.caption("Illinois Landlord-Tenant Case Law · Powered by CAP + Amazon Bedrock")

st.markdown("""
Describe your situation as a **self-represented litigant** in an Illinois
landlord-tenant dispute. The simulator will retrieve relevant case law and
show you the **strongest arguments the opposing side is likely to raise** at
your hearing — so you can prepare your responses.
""")

st.divider()

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("📋 Your Situation")
    situation = st.text_area(
        "Describe what happened:",
        placeholder=(
            "e.g. My landlord has not repaired the heating system for 6 weeks "
            "despite multiple written requests. The temperature in my apartment "
            "has been below 55°F. I withheld last month's rent until repairs are made."
        ),
        height=180,
    )

    top_k = st.slider("Number of results", min_value=3, max_value=10, value=5)
    run = st.button("⚡ Simulate Opposing Arguments", type="primary", use_container_width=True)

with col2:
    st.subheader("📚 How it works")
    st.markdown("""
    1. **Retrieval** — hybrid BM25 + vector search over 9,408 Illinois
       landlord-tenant cases (Harvard CAP corpus, 1825–2011)
    2. **Grounding** — only arguments backed by retrieved citations
       are generated — no hallucinated case law
    3. **Generation** — Claude (Amazon Bedrock) simulates the
       opposing side's lawyer
    4. **Citations** — every argument links back to a real Illinois case
    """)

st.divider()

if run:
    if not situation.strip():
        st.warning("Please describe your situation first.")
        st.stop()

    with st.spinner("Retrieving relevant case law..."):
        try:
            chunks = retrieve(situation.strip(), top_k=top_k)
        except Exception as e:
            st.error(f"Retrieval error: {e}")
            st.stop()

    with st.spinner("Generating opposing arguments via Claude..."):
        try:
            arguments = generate_arguments(situation.strip(), chunks)
        except Exception as e:
            st.error(f"Claude error: {e}")
            st.stop()

    # ── Results ──────────────────────────────────────────────
    st.subheader("⚔️ Opposing Side's Likely Arguments")
    st.markdown(arguments)

    st.divider()
    st.subheader(f"📖 Retrieved Case Law ({len(chunks)} chunks)")

    for i, chunk in enumerate(chunks, 1):
        badge = {"both": "🟢 Dense+Sparse", "dense": "🔵 Dense", "sparse": "🟡 Sparse"}
        with st.expander(
            f"#{i} — {chunk['citation']}  ({chunk['date'][:7] if chunk['date'] else 'n/a'})  "
            f"· {badge.get(chunk['match_type'], chunk['match_type'])}"
        ):
            st.caption(
                f"Chunk {chunk['chunk_index']} of {chunk['total_chunks']-1}  "
                f"· RRF score: {chunk['rrf_score']:.5f}"
            )
            st.text(chunk["chunk_text"][:800] + ("..." if len(chunk["chunk_text"]) > 800 else ""))
