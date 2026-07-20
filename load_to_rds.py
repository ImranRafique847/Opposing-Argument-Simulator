"""
Opposing-Argument Simulator — Phase 2, Step 3: Load embeddings into RDS pgvector
----------------------------------------------------------------------------------
Reads embedded_chunks.jsonl and bulk-inserts all chunks into the
case_chunks table on Amazon RDS PostgreSQL (pgvector).

Uses COPY via psycopg2 execute_values for fast batch inserts.
Tracks progress and supports resume (skips already-inserted chunk_ids).

Run:
    python load_to_rds.py
"""

import json
import os
import time

import psycopg2
import psycopg2.extras
from dotenv import load_dotenv

load_dotenv()

INPUT_FILE = "embedded_chunks.jsonl"
BATCH_SIZE = 500  # rows per INSERT batch


def get_conn():
    return psycopg2.connect(
        host=os.getenv("DB_HOST"),
        port=os.getenv("DB_PORT"),
        dbname=os.getenv("DB_NAME"),
        user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASSWORD"),
        connect_timeout=30,
    )


def main():
    if not os.path.exists(INPUT_FILE):
        raise SystemExit(f"ERROR: {INPUT_FILE} not found.")

    conn = get_conn()
    cur = conn.cursor()

    # Check how many rows already loaded (resume support)
    cur.execute("SELECT COUNT(*) FROM case_chunks;")
    already_loaded = cur.fetchone()[0]
    print(f"Rows already in DB: {already_loaded:,}")

    # Count total lines
    with open(INPUT_FILE, encoding="utf-8") as f:
        total = sum(1 for _ in f)
    print(f"Total chunks to load: {total:,}")

    if already_loaded >= total:
        print("All chunks already loaded. Nothing to do.")
        conn.close()
        return

    start = time.time()
    loaded = 0
    skipped = 0
    batch = []

    with open(INPUT_FILE, encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue

            # Skip rows already inserted (resume)
            if i < already_loaded:
                skipped += 1
                continue

            rec = json.loads(line)
            embedding = rec["embedding"]  # list of 384 floats

            batch.append((
                rec["chunk_id"],
                str(rec.get("source_case_id", "")),
                rec.get("citation", ""),
                rec.get("jurisdiction", "Illinois"),
                rec.get("case_type", "tenancy"),
                rec.get("date", ""),
                rec.get("source", "CAP"),
                rec.get("chunk_index", 0),
                rec.get("total_chunks", 1),
                rec.get("token_count", 0),
                rec.get("text", ""),
                embedding,
            ))

            if len(batch) >= BATCH_SIZE:
                psycopg2.extras.execute_values(
                    cur,
                    """
                    INSERT INTO case_chunks
                      (chunk_id, source_case_id, citation, jurisdiction, case_type,
                       date, source, chunk_index, total_chunks, token_count,
                       chunk_text, embedding)
                    VALUES %s
                    ON CONFLICT (chunk_id) DO NOTHING
                    """,
                    batch,
                    template="(%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s::vector)",
                )
                conn.commit()
                loaded += len(batch)
                batch = []

                elapsed = time.time() - start
                rate = loaded / elapsed
                remaining = (total - already_loaded - loaded)
                eta = remaining / rate / 60 if rate > 0 else 0
                print(f"  Loaded {loaded + already_loaded:>7,}/{total:,}  "
                      f"({rate:.0f} rows/s)  ETA: {eta:.1f} min", end="\r")

    # Flush remainder
    if batch:
        psycopg2.extras.execute_values(
            cur,
            """
            INSERT INTO case_chunks
              (chunk_id, source_case_id, citation, jurisdiction, case_type,
               date, source, chunk_index, total_chunks, token_count,
               chunk_text, embedding)
            VALUES %s
            ON CONFLICT (chunk_id) DO NOTHING
            """,
            batch,
            template="(%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s::vector)",
        )
        conn.commit()
        loaded += len(batch)

    elapsed = time.time() - start
    print(f"\n\nLoad complete in {elapsed:.1f}s")
    print(f"  Newly inserted: {loaded:,}")
    print(f"  Skipped (already existed): {skipped:,}")

    # Verify final count
    cur.execute("SELECT COUNT(*) FROM case_chunks;")
    final = cur.fetchone()[0]
    print(f"  Total rows in DB: {final:,}")

    conn.close()


if __name__ == "__main__":
    main()
