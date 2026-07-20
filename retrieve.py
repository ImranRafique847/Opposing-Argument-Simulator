"""
Opposing-Argument Simulator — Phase 2, Step 4: Hybrid Retrieval
----------------------------------------------------------------
Implements hybrid search over the case_chunks table in RDS pgvector:

  1. DENSE search  — embedding similarity (cosine) via pgvector HNSW index
  2. SPARSE search — BM25 keyword matching via rank_bm25 (in-memory index)
  3. FUSION        — Reciprocal Rank Fusion (RRF) to merge both ranked lists
  4. FILTER        — jurisdiction + case_type pre-filter (never retrieves
                     out-of-scope case law)
  5. RERANK        — returns top-K after fusion (default K=5)

BM25 index is built once on first call and cached in memory.
The dense index lives in RDS (HNSW) — no in-memory cost.

Usage:
    python retrieve.py                    # runs 3 built-in test queries
    python retrieve.py --query "your query here"
"""

import argparse
import json
import os
import time
from typing import Optional

import psycopg2
import psycopg2.extras
from dotenv import load_dotenv
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

load_dotenv()

# =========================================================
# CONFIG
# =========================================================

JURISDICTION = "Illinois"
CASE_TYPE    = "tenancy"
DENSE_TOP_N  = 20    # candidates from dense search
SPARSE_TOP_N = 20    # candidates from sparse search
FINAL_TOP_K  = 5     # final results after RRF fusion
RRF_K        = 60    # RRF constant (standard value)

EMBED_MODEL  = "all-MiniLM-L6-v2"

# =========================================================
# DB CONNECTION
# =========================================================

def get_conn():
    return psycopg2.connect(
        host=os.getenv("DB_HOST"),
        port=os.getenv("DB_PORT"),
        dbname=os.getenv("DB_NAME"),
        user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASSWORD"),
        connect_timeout=30,
    )


# =========================================================
# LAZY-LOADED GLOBALS (built once, reused across queries)
# =========================================================

_model     = None
_bm25      = None
_bm25_meta = None   # list of dicts with chunk_id, citation, etc.


def get_model():
    global _model
    if _model is None:
        print("Loading embedding model...")
        _model = SentenceTransformer(EMBED_MODEL)
    return _model


def get_bm25(conn):
    """
    Build BM25 index from all chunks in DB filtered by jurisdiction/case_type.
    Cached after first build.
    """
    global _bm25, _bm25_meta
    if _bm25 is not None:
        return _bm25, _bm25_meta

    print("Building BM25 index from DB (first call only)...")
    t = time.time()

    cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
    cur.execute(
        """
        SELECT chunk_id, citation, date, chunk_index, total_chunks,
               token_count, chunk_text
        FROM case_chunks
        WHERE jurisdiction = %s AND case_type = %s
        ORDER BY id
        """,
        (JURISDICTION, CASE_TYPE),
    )
    rows = cur.fetchall()

    _bm25_meta = [dict(r) for r in rows]
    tokenized  = [r["chunk_text"].lower().split() for r in _bm25_meta]
    _bm25      = BM25Okapi(tokenized)

    print(f"BM25 index built: {len(_bm25_meta):,} docs in {time.time()-t:.1f}s")
    return _bm25, _bm25_meta


# =========================================================
# DENSE SEARCH
# =========================================================

def dense_search(conn, query_embedding: list, top_n: int) -> list[dict]:
    """
    Cosine similarity search via pgvector HNSW index.
    Pre-filtered by jurisdiction + case_type.
    Returns list of {chunk_id, citation, date, chunk_index,
                     total_chunks, chunk_text, score}.
    """
    vec_str = "[" + ",".join(str(x) for x in query_embedding) + "]"

    cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
    cur.execute(
        """
        SELECT chunk_id, citation, date, chunk_index, total_chunks, chunk_text,
               1 - (embedding <=> %s::vector) AS score
        FROM case_chunks
        WHERE jurisdiction = %s AND case_type = %s
        ORDER BY embedding <=> %s::vector
        LIMIT %s
        """,
        (vec_str, JURISDICTION, CASE_TYPE, vec_str, top_n),
    )
    return [dict(r) for r in cur.fetchall()]


# =========================================================
# SPARSE (BM25) SEARCH
# =========================================================

def sparse_search(bm25, bm25_meta: list, query: str, top_n: int) -> list[dict]:
    """
    BM25 keyword search over in-memory index.
    Returns list of {chunk_id, citation, date, chunk_index,
                     total_chunks, chunk_text, score}.
    """
    tokens = query.lower().split()
    scores = bm25.get_scores(tokens)

    # Get top_n indices by score
    top_indices = sorted(range(len(scores)), key=lambda i: -scores[i])[:top_n]

    results = []
    for idx in top_indices:
        if scores[idx] <= 0:
            break
        meta = bm25_meta[idx]
        results.append({
            "chunk_id":    meta["chunk_id"],
            "citation":    meta["citation"],
            "date":        meta["date"],
            "chunk_index": meta["chunk_index"],
            "total_chunks":meta["total_chunks"],
            "chunk_text":  meta["chunk_text"],
            "score":       float(scores[idx]),
        })
    return results


# =========================================================
# RECIPROCAL RANK FUSION
# =========================================================

def reciprocal_rank_fusion(
    dense_results: list[dict],
    sparse_results: list[dict],
    k: int = RRF_K,
    final_k: int = FINAL_TOP_K,
) -> list[dict]:
    """
    Merge two ranked lists using RRF.
    RRF score = 1/(k + rank_dense) + 1/(k + rank_sparse)
    """
    scores  = {}  # chunk_id -> rrf_score
    sources = {}  # chunk_id -> "dense" | "sparse" | "both"
    chunks  = {}  # chunk_id -> chunk dict

    for rank, item in enumerate(dense_results, start=1):
        cid = item["chunk_id"]
        scores[cid]  = scores.get(cid, 0) + 1.0 / (k + rank)
        sources[cid] = "dense"
        chunks[cid]  = item

    for rank, item in enumerate(sparse_results, start=1):
        cid = item["chunk_id"]
        scores[cid]  = scores.get(cid, 0) + 1.0 / (k + rank)
        sources[cid] = "both" if sources.get(cid) == "dense" else "sparse"
        if cid not in chunks:
            chunks[cid] = item

    ranked = sorted(scores.items(), key=lambda x: -x[1])[:final_k]

    results = []
    for cid, rrf_score in ranked:
        item = dict(chunks[cid])
        item["rrf_score"] = rrf_score
        item["source"]    = sources[cid]
        results.append(item)

    return results


# =========================================================
# MAIN RETRIEVE FUNCTION
# =========================================================

def retrieve(query: str, verbose: bool = True) -> list[dict]:
    """
    Run hybrid retrieval for a query.
    Returns top-K results with full metadata.
    """
    conn  = get_conn()
    model = get_model()
    bm25, bm25_meta = get_bm25(conn)

    # Embed query
    t0 = time.time()
    query_vec = model.encode(query, normalize_embeddings=True).tolist()

    # Dense + sparse search
    dense_res  = dense_search(conn, query_vec, DENSE_TOP_N)
    sparse_res = sparse_search(bm25, bm25_meta, query, SPARSE_TOP_N)

    # Fuse
    results = reciprocal_rank_fusion(dense_res, sparse_res)

    elapsed = time.time() - t0
    conn.close()

    if verbose:
        print(f"\n{'='*65}")
        print(f"Query: \"{query}\"")
        print(f"Retrieval time: {elapsed:.2f}s  |  "
              f"Dense: {len(dense_res)} candidates  |  "
              f"Sparse: {len(sparse_res)} candidates")
        print(f"{'='*65}")
        for i, r in enumerate(results, 1):
            print(f"\n  #{i}  [{r['source'].upper()}]  RRF score: {r['rrf_score']:.5f}")
            print(f"  Citation:    {r['citation']}")
            print(f"  Date:        {r['date']}")
            print(f"  Chunk:       {r['chunk_index']} of {r['total_chunks']-1}")
            print(f"  Text:        {r['chunk_text'][:250]}...")

    return results


# =========================================================
# CLI / TEST QUERIES
# =========================================================

TEST_QUERIES = [
    "landlord failed to repair heating",
    "eviction notice improper",
    "security deposit not returned",
]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", type=str, default=None,
                        help="Custom query to test")
    args = parser.parse_args()

    if args.query:
        retrieve(args.query)
    else:
        print(f"Running {len(TEST_QUERIES)} test queries...\n")
        for q in TEST_QUERIES:
            retrieve(q)
            print()
