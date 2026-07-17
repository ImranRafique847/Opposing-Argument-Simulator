"""
Opposing-Argument Simulator — Phase 2, Step 1: Chunking
---------------------------------------------------------
Reads legal_corpus_final.jsonl and splits each case opinion into
overlapping chunks suitable for embedding and RAG retrieval.

Chunking strategy:
  1. Structure-aware: split on paragraph boundaries (\n\n) first,
     then sentence/line boundaries (\n), then spaces — before ever
     splitting mid-sentence. This keeps legal reasoning intact.
  2. Target: 400 tokens per chunk (fits well within Titan V2's 8192
     token limit while keeping enough context per chunk for retrieval).
  3. Overlap: 60 tokens between consecutive chunks so a retrieval hit
     near a chunk boundary doesn't lose surrounding context.
  4. Each chunk carries full parent metadata: case id, citation, date,
     jurisdiction — so every retrieved chunk is traceable to a real case.

Output: legal_corpus_chunked.jsonl
  Each line: {chunk_id, case_id, source, citation, date, jurisdiction,
              case_type, chunk_index, total_chunks, token_count, text}

Run:
    python chunk_corpus.py
"""

import json
import os
import re
import time
import tiktoken

# =========================================================
# CONFIG
# =========================================================

INPUT_FILE  = "legal_corpus_final.jsonl"
OUTPUT_FILE = "legal_corpus_chunked.jsonl"

TARGET_TOKENS  = 400   # target chunk size in tokens
OVERLAP_TOKENS = 60    # overlap between consecutive chunks
MIN_TOKENS     = 50    # discard chunks shorter than this (boilerplate/headers)

# Tokeniser — cl100k_base is the GPT-4/Titan-compatible BPE tokeniser.
# We use it purely for counting; the actual embedding tokenisation is
# handled server-side by Bedrock.
TOKENISER = tiktoken.get_encoding("cl100k_base")

# Paragraph separators in priority order (try wider splits first)
SEPARATORS = ["\n\n", "\n", ". ", " ", ""]


# =========================================================
# TOKENISER HELPERS
# =========================================================

def count_tokens(text):
    return len(TOKENISER.encode(text))

def tokens_to_text(tokens):
    return TOKENISER.decode(tokens)

def text_to_tokens(text):
    return TOKENISER.encode(text)


# =========================================================
# RECURSIVE CHARACTER SPLITTER
# =========================================================

def recursive_split(text, separators, target, overlap):
    """
    Split text recursively on separator hierarchy.
    Returns a list of strings, each ~target tokens with ~overlap token overlap.
    """
    # Try each separator in order
    for sep in separators:
        if sep == "":
            # Last resort: split on token boundary directly
            parts = [text]
        else:
            parts = text.split(sep)

        if len(parts) > 1:
            break

    if len(parts) == 1:
        # Can't split further — return as-is even if oversized
        return [text] if count_tokens(text) >= MIN_TOKENS else []

    # Merge small parts into target-sized chunks with overlap
    chunks = []
    current_tokens = []
    current_text_parts = []

    for part in parts:
        part = part.strip()
        if not part:
            continue

        part_tokens = text_to_tokens(part)

        # If this single part is larger than target, recurse into it
        if len(part_tokens) > target:
            # Flush current buffer first
            if current_tokens:
                chunk_text = (sep if sep else " ").join(current_text_parts)
                if count_tokens(chunk_text) >= MIN_TOKENS:
                    chunks.append(chunk_text)
                current_tokens = current_tokens[-overlap:] if overlap else []
                current_text_parts = []

            sub_chunks = recursive_split(
                part,
                separators[separators.index(sep) + 1:] if sep in separators[1:] else separators[1:],
                target,
                overlap,
            )
            chunks.extend(sub_chunks)
            continue

        # Would adding this part exceed the target?
        if len(current_tokens) + len(part_tokens) > target and current_tokens:
            # Emit current chunk
            chunk_text = (sep if sep else " ").join(current_text_parts)
            if count_tokens(chunk_text) >= MIN_TOKENS:
                chunks.append(chunk_text)

            # Seed the next chunk with the overlap window
            # (keep last N tokens worth of parts)
            overlap_tokens = current_tokens[-overlap:] if overlap else []
            overlap_text = tokens_to_text(overlap_tokens).strip()
            current_tokens = list(overlap_tokens)
            current_text_parts = [overlap_text] if overlap_text else []

        current_tokens.extend(part_tokens)
        current_text_parts.append(part)

    # Flush remainder
    if current_text_parts:
        chunk_text = (sep if sep else " ").join(current_text_parts)
        if count_tokens(chunk_text) >= MIN_TOKENS:
            chunks.append(chunk_text)

    return chunks


# =========================================================
# CHUNK A SINGLE CASE
# =========================================================

def chunk_case(record):
    """Split one case record into overlapping chunks with metadata."""
    text = record.get("text", "").strip()
    if not text:
        return []

    raw_chunks = recursive_split(text, SEPARATORS, TARGET_TOKENS, OVERLAP_TOKENS)

    chunks = []
    for i, chunk_text in enumerate(raw_chunks):
        chunks.append({
            "chunk_id":     f"{record['id']}_chunk_{i}",
            "case_id":      record["id"],
            "source":       record.get("source", "CAP"),
            "citation":     record.get("citation", ""),
            "date":         record.get("date", ""),
            "jurisdiction": record.get("jurisdiction", "Illinois"),
            "case_type":    record.get("case_type", "tenancy"),
            "chunk_index":  i,
            "total_chunks": len(raw_chunks),   # updated after loop below
            "token_count":  count_tokens(chunk_text),
            "text":         chunk_text,
        })

    # Back-fill total_chunks now we know the real count
    for c in chunks:
        c["total_chunks"] = len(chunks)

    return chunks


# =========================================================
# MAIN
# =========================================================

def main():
    if not os.path.exists(INPUT_FILE):
        raise SystemExit(f"ERROR: {INPUT_FILE} not found.")

    print(f"Input:  {INPUT_FILE}")
    print(f"Output: {OUTPUT_FILE}")
    print(f"Target: {TARGET_TOKENS} tokens/chunk, {OVERLAP_TOKENS} token overlap\n")

    start = time.time()

    total_cases   = 0
    total_chunks  = 0
    token_counts  = []

    if os.path.exists(OUTPUT_FILE):
        os.remove(OUTPUT_FILE)

    with open(INPUT_FILE, "r", encoding="utf-8") as infile, \
         open(OUTPUT_FILE, "w", encoding="utf-8") as outfile:

        for line in infile:
            line = line.strip()
            if not line:
                continue

            record = json.loads(line)
            chunks = chunk_case(record)

            for chunk in chunks:
                outfile.write(json.dumps(chunk, ensure_ascii=False) + "\n")
                token_counts.append(chunk["token_count"])

            total_cases  += 1
            total_chunks += len(chunks)

            if total_cases % 1000 == 0:
                print(f"  Processed {total_cases:,} cases | {total_chunks:,} chunks so far...")

    elapsed = time.time() - start

    avg_tokens = sum(token_counts) / len(token_counts) if token_counts else 0
    min_tokens = min(token_counts) if token_counts else 0
    max_tokens = max(token_counts) if token_counts else 0

    # Token distribution buckets
    buckets = {"<100": 0, "100-200": 0, "200-300": 0, "300-400": 0,
               "400-500": 0, "500+": 0}
    for t in token_counts:
        if t < 100:        buckets["<100"]     += 1
        elif t < 200:      buckets["100-200"]  += 1
        elif t < 300:      buckets["200-300"]  += 1
        elif t < 400:      buckets["300-400"]  += 1
        elif t < 500:      buckets["400-500"]  += 1
        else:              buckets["500+"]     += 1

    print(f"\nDone in {elapsed:.1f}s.")
    print(f"Cases processed:  {total_cases:,}")
    print(f"Total chunks:     {total_chunks:,}")
    print(f"Avg chunks/case:  {total_chunks/total_cases:.1f}")
    print(f"\nToken distribution per chunk:")
    print(f"  Min: {min_tokens}  |  Avg: {avg_tokens:.0f}  |  Max: {max_tokens}")
    for bucket, count in buckets.items():
        bar = "█" * (count // 100)
        print(f"  {bucket:>8} tokens: {count:6,}  {bar}")

    # Show 2 sample chunks for sanity check
    print(f"\nSample chunks (first 2 from output):")
    with open(OUTPUT_FILE, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= 2:
                break
            c = json.loads(line)
            print(f"\n  chunk_id:    {c['chunk_id']}")
            print(f"  citation:    {c['citation']}")
            print(f"  date:        {c['date']}")
            print(f"  chunk_index: {c['chunk_index']} / {c['total_chunks']-1}")
            print(f"  token_count: {c['token_count']}")
            print(f"  text[:200]:  {c['text'][:200]}...")

    print(f"\nOutput written: {OUTPUT_FILE}")
    print(f"Next step: embed chunks with Amazon Titan Text Embeddings V2 via Bedrock.")


if __name__ == "__main__":
    main()
