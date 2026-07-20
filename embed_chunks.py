"""
Opposing-Argument Simulator — Phase 2, Step 2: Local Embedding
---------------------------------------------------------------
Embeds chunked_corpus.jsonl using sentence-transformers all-MiniLM-L6-v2.
Runs on GPU (CUDA) if available, falls back to CPU automatically.

Model: all-MiniLM-L6-v2
  - 384-dim embeddings
  - ~80MB model size
  - Fast on GPU: ~1,000-5,000 chunks/min depending on batch size
  - Well-suited for semantic similarity / RAG retrieval

Output: embedded_chunks.jsonl
  All original chunk metadata preserved + "embedding" field (384-dim list)

Resume support: tracks last completed index in embed_progress.json
  so interrupted runs can continue without re-embedding.

Usage:
    # Full run (all 110,163 chunks):
    python embed_chunks.py

    # Test on first 100 chunks only:
    python embed_chunks.py --test
"""

import argparse
import json
import os
import time

from dotenv import load_dotenv

load_dotenv()

INPUT_FILE    = "chunked_corpus.jsonl"
OUTPUT_FILE   = "embedded_chunks.jsonl"
PROGRESS_FILE = "embed_progress.json"

BATCH_SIZE    = 64     # chunks per forward pass — fits comfortably in 4GB VRAM
TEST_LIMIT    = 100    # chunks to process in --test mode


# =========================================================
# PROGRESS TRACKER
# =========================================================

def load_progress():
    if os.path.exists(PROGRESS_FILE):
        with open(PROGRESS_FILE) as f:
            return json.load(f).get("last_completed", -1)
    return -1

def save_progress(idx):
    with open(PROGRESS_FILE, "w") as f:
        json.dump({"last_completed": idx}, f)


# =========================================================
# MAIN
# =========================================================

def main(test_mode: bool):
    # Import here so we get a clear error if torch isn't installed yet
    import torch
    from sentence_transformers import SentenceTransformer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        gpu_name = torch.cuda.get_device_name(0)
        vram_gb  = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"Device: {gpu_name} ({vram_gb:.1f}GB VRAM)")
    else:
        print("Device: CPU (no CUDA available)")

    print(f"Loading all-MiniLM-L6-v2 ...")
    model = SentenceTransformer("all-MiniLM-L6-v2", device=device)
    print(f"Model loaded. Embedding dim: {model.get_sentence_embedding_dimension()}")

    if not os.path.exists(INPUT_FILE):
        raise SystemExit(f"ERROR: {INPUT_FILE} not found.")

    # Count total chunks
    with open(INPUT_FILE, encoding="utf-8") as f:
        total = sum(1 for _ in f)

    limit        = TEST_LIMIT if test_mode else total
    resume_from  = load_progress() + 1 if not test_mode else 0
    mode_label   = f"TEST ({TEST_LIMIT} chunks)" if test_mode else f"FULL ({total:,} chunks)"

    print(f"\nMode:     {mode_label}")
    print(f"Input:    {INPUT_FILE}  ({total:,} chunks)")
    print(f"Output:   {OUTPUT_FILE}")
    print(f"Resume:   from index {resume_from}" if resume_from > 0 else "Resume:   fresh start")
    print(f"Batch:    {BATCH_SIZE} chunks/pass\n")

    # Clear output for test mode; append for full/resume
    out_mode = "w" if (test_mode or resume_from == 0) else "a"

    start         = time.time()
    processed     = 0
    batch_chunks  = []   # metadata dicts
    batch_texts   = []   # texts for encoding
    batch_indices = []   # global line indices

    def flush_batch():
        nonlocal processed
        if not batch_texts:
            return

        embeddings = model.encode(
            batch_texts,
            batch_size=BATCH_SIZE,
            show_progress_bar=False,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )

        with open(OUTPUT_FILE, out_mode if processed == 0 else "a", encoding="utf-8") as f:
            for chunk, emb in zip(batch_chunks, embeddings):
                record = {
                    "chunk_id":       chunk["chunk_id"],
                    "source_case_id": chunk["source_case_id"],
                    "citation":       chunk["citation"],
                    "jurisdiction":   chunk["jurisdiction"],
                    "case_type":      chunk["case_type"],
                    "date":           chunk["date"],
                    "source":         chunk["source"],
                    "chunk_index":    chunk["chunk_index"],
                    "total_chunks":   chunk["total_chunks"],
                    "token_count":    chunk["token_count"],
                    "text":           chunk["text"],
                    "embedding":      emb.tolist(),
                }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

        processed += len(batch_texts)
        if not test_mode:
            save_progress(batch_indices[-1])

        elapsed = time.time() - start
        rate    = processed / elapsed
        eta_min = (limit - processed) / rate / 60 if rate > 0 else 0
        print(f"  [{processed:>7,}/{limit:,}]  "
              f"{rate:.0f} chunks/s  |  ETA: {eta_min:.1f} min", end="\r")

        batch_chunks.clear()
        batch_texts.clear()
        batch_indices.clear()

    with open(INPUT_FILE, encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= limit:
                break
            if i < resume_from:
                continue

            line = line.strip()
            if not line:
                continue

            chunk = json.loads(line)
            batch_chunks.append(chunk)
            batch_texts.append(chunk["text"])
            batch_indices.append(i)

            if len(batch_texts) >= BATCH_SIZE:
                flush_batch()

    flush_batch()  # remainder

    elapsed = time.time() - start
    rate    = processed / elapsed if elapsed > 0 else 0
    print(f"\n\n{'='*60}")
    print(f"{'TEST COMPLETE' if test_mode else 'EMBEDDING COMPLETE'}")
    print(f"  Chunks embedded:  {processed:,}")
    print(f"  Elapsed:          {elapsed:.1f}s")
    print(f"  Rate:             {rate:.0f} chunks/s")
    print(f"  Output:           {OUTPUT_FILE}")

    if test_mode:
        full_est_min = total / rate / 60 if rate > 0 else 0
        print(f"\n  Full run estimate: {full_est_min:.0f} min ({full_est_min/60:.1f} hrs)")
        print(f"  Run 'python embed_chunks.py' (no --test) to embed all {total:,} chunks.")

    # Show 3 sample records
    print(f"\n{'='*60}")
    print("SAMPLE EMBEDDED RECORDS (metadata only)")
    print("="*60)
    with open(OUTPUT_FILE, encoding="utf-8") as f:
        for j, line in enumerate(f):
            if j >= 3:
                break
            rec = json.loads(line)
            emb = rec["embedding"]
            print(f"\nRecord {j+1}:")
            print(f"  chunk_id:       {rec['chunk_id']}")
            print(f"  source_case_id: {rec['source_case_id']}")
            print(f"  citation:       {rec['citation']}")
            print(f"  date:           {rec['date']}")
            print(f"  chunk_index:    {rec['chunk_index']} of {rec['total_chunks']-1}")
            print(f"  token_count:    {rec['token_count']}")
            print(f"  embedding:      [{emb[0]:.6f}, {emb[1]:.6f}, ... ({len(emb)}d)]")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", action="store_true",
                        help=f"Embed only first {TEST_LIMIT} chunks to validate setup")
    args = parser.parse_args()
    main(test_mode=args.test)
