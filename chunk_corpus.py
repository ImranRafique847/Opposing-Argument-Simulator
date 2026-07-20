"""
Opposing-Argument Simulator — Phase 2, Step 1: Chunking
---------------------------------------------------------
Reads legal_corpus_final.jsonl and splits each case opinion into
overlapping chunks for embedding and RAG retrieval.

CITATION TRACEABILITY is the primary design constraint: every chunk
carries enough metadata that any argument the LLM generates can be
traced back to a specific, verifiable Illinois case citation.

Chunking strategy (in order of preference):
  1. Split on paragraph boundaries (\n\n) — CAP OCR preserves paragraph
     structure from the original opinion text.
  2. For paragraphs exceeding the target size, split on SENTENCE BOUNDARIES
     using NLTK's Punkt tokenizer — handles legal citations containing
     periods (e.g. "2 Ill. App. 3d 538") correctly, unlike naive "." splits.
  3. For sentences still exceeding the target, split on spaces (last resort).
  4. 50-token overlap between consecutive chunks from the same case.
  5. Discard chunks under 50 tokens (headers, lone citations, fragments).

Output: chunked_corpus.jsonl
  Fields per chunk:
    chunk_id       — unique ID: "{case_id}_chunk_{index}"
    source_case_id — original record's "id" field (for full traceability)
    citation       — exact case citation string (e.g. "365 Ill. App. 3d 621")
    jurisdiction   — always "Illinois"
    case_type      — always "tenancy"
    date           — decision date from source record
    source         — always "CAP"
    chunk_index    — 0-based position within the case
    total_chunks   — total number of chunks for this case
    token_count    — BPE token count of this chunk's text
    text           — the chunk text itself

Run:
    python chunk_corpus.py
"""

import json
import os
import time

import nltk
import tiktoken

# Ensure NLTK sentence tokenizer data is available
try:
    nltk.data.find("tokenizers/punkt_tab")
except LookupError:
    nltk.download("punkt_tab", quiet=True)

# =========================================================
# CONFIG
# =========================================================

INPUT_FILE  = "legal_corpus_final.jsonl"
OUTPUT_FILE = "chunked_corpus.jsonl"

TARGET_TOKENS  = 400   # target chunk size (within 300-500 spec range)
OVERLAP_TOKENS = 60    # overlap between consecutive chunks (within 50-75 spec)
MIN_TOKENS     = 50    # discard chunks shorter than this

# cl100k_base matches Titan V2 / GPT-4 tokenisation closely enough for
# size estimation. Actual tokenisation is done server-side by Bedrock.
TOKENISER = tiktoken.get_encoding("cl100k_base")


# =========================================================
# TOKEN HELPERS
# =========================================================

def count_tokens(text: str) -> int:
    return len(TOKENISER.encode(text))

def encode(text: str) -> list:
    return TOKENISER.encode(text)

def decode(tokens: list) -> str:
    return TOKENISER.decode(tokens)


# =========================================================
# SENTENCE SPLITTER (NLTK-based, citation-safe)
# =========================================================

def split_sentences(text: str) -> list[str]:
    """
    Split text into sentences using NLTK Punkt tokenizer.
    Correctly handles legal citations like "2 Ill. App. 3d 538"
    without treating abbreviation periods as sentence boundaries.
    """
    sentences = nltk.sent_tokenize(text)
    return [s.strip() for s in sentences if s.strip()]


# =========================================================
# CORE CHUNKER
# =========================================================

def make_chunks_with_overlap(units: list[str], target: int, overlap: int) -> list[str]:
    """
    Given a list of text units (paragraphs or sentences), pack them into
    chunks of ~target tokens with ~overlap token overlap between chunks.
    """
    chunks = []
    current_units = []
    current_tokens = []

    for unit in units:
        unit_toks = encode(unit)

        # Single unit too large for target — split it on spaces as last resort
        if len(unit_toks) > target:
            # Flush current buffer first
            if current_units:
                chunk_text = " ".join(current_units)
                if count_tokens(chunk_text) >= MIN_TOKENS:
                    chunks.append(chunk_text)
                # Seed overlap from tail of current buffer
                overlap_text = decode(current_tokens[-overlap:]).strip() if overlap else ""
                current_units = [overlap_text] if overlap_text else []
                current_tokens = current_tokens[-overlap:] if overlap else []

            # Split oversized unit on spaces
            words = unit.split(" ")
            word_buffer = []
            word_toks = []
            for word in words:
                w_toks = encode(word + " ")
                if len(word_toks) + len(w_toks) > target and word_buffer:
                    chunk_text = " ".join(word_buffer)
                    if count_tokens(chunk_text) >= MIN_TOKENS:
                        chunks.append(chunk_text)
                    overlap_text = decode(word_toks[-overlap:]).strip() if overlap else ""
                    word_buffer = [overlap_text] if overlap_text else []
                    word_toks = word_toks[-overlap:] if overlap else []
                word_buffer.append(word)
                word_toks.extend(w_toks)
            if word_buffer:
                chunk_text = " ".join(word_buffer)
                if count_tokens(chunk_text) >= MIN_TOKENS:
                    current_units = [chunk_text]
                    current_tokens = encode(chunk_text)
            continue

        # Adding this unit would exceed target — emit current chunk first
        if len(current_tokens) + len(unit_toks) > target and current_units:
            chunk_text = " ".join(current_units)
            if count_tokens(chunk_text) >= MIN_TOKENS:
                chunks.append(chunk_text)
            # Overlap: seed next chunk with tail tokens of what we just emitted
            overlap_toks = current_tokens[-overlap:] if overlap else []
            overlap_text = decode(overlap_toks).strip() if overlap_toks else ""
            current_units = [overlap_text] if overlap_text else []
            current_tokens = list(overlap_toks)

        current_units.append(unit)
        current_tokens.extend(unit_toks)

    # Flush remainder
    if current_units:
        chunk_text = " ".join(current_units)
        if count_tokens(chunk_text) >= MIN_TOKENS:
            chunks.append(chunk_text)

    return chunks


def chunk_case(record: dict) -> list[dict]:
    """
    Split one case record into overlapping, metadata-tagged chunks.
    Strategy: paragraph split → sentence split within large paragraphs →
    space split as last resort.
    """
    text = record.get("text", "").strip()
    if not text:
        return []

    # ── Step 1: paragraph split ──────────────────────────────────────────
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    if not paragraphs:
        paragraphs = [text]

    # ── Step 2: sentence-split any paragraph exceeding the target ────────
    units = []
    for para in paragraphs:
        if count_tokens(para) <= TARGET_TOKENS:
            units.append(para)
        else:
            # Use NLTK sentence tokenizer (citation-safe)
            sentences = split_sentences(para)
            units.extend(sentences if sentences else [para])

    # ── Step 3: pack units into overlapping chunks ────────────────────────
    raw_chunks = make_chunks_with_overlap(units, TARGET_TOKENS, OVERLAP_TOKENS)

    if not raw_chunks:
        return []

    # ── Step 4: attach metadata to every chunk ────────────────────────────
    chunks = []
    n = len(raw_chunks)
    for i, chunk_text in enumerate(raw_chunks):
        chunks.append({
            "chunk_id":       f"{record['id']}_chunk_{i}",
            "source_case_id": record["id"],          # traceability anchor
            "citation":       record.get("citation", ""),
            "jurisdiction":   record.get("jurisdiction", "Illinois"),
            "case_type":      record.get("case_type", "tenancy"),
            "date":           record.get("date", ""),
            "source":         record.get("source", "CAP"),
            "chunk_index":    i,
            "total_chunks":   n,
            "token_count":    count_tokens(chunk_text),
            "text":           chunk_text,
        })

    return chunks


# =========================================================
# MAIN
# =========================================================

def main():
    if not os.path.exists(INPUT_FILE):
        raise SystemExit(f"ERROR: {INPUT_FILE} not found.")

    print(f"Input:   {INPUT_FILE}")
    print(f"Output:  {OUTPUT_FILE}")
    print(f"Target:  {TARGET_TOKENS} tokens/chunk  |  Overlap: {OVERLAP_TOKENS}  |  Min: {MIN_TOKENS}\n")

    start = time.time()
    total_cases  = 0
    total_chunks = 0
    token_counts = []

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
                print(f"  {total_cases:,} cases | {total_chunks:,} chunks...")

    elapsed = time.time() - start
    avg  = sum(token_counts) / len(token_counts) if token_counts else 0
    buckets = {"<100": 0, "100-200": 0, "200-300": 0,
               "300-400": 0, "400-500": 0, "500+": 0}
    for t in token_counts:
        if   t < 100: buckets["<100"]     += 1
        elif t < 200: buckets["100-200"]  += 1
        elif t < 300: buckets["200-300"]  += 1
        elif t < 400: buckets["300-400"]  += 1
        elif t < 500: buckets["400-500"]  += 1
        else:         buckets["500+"]     += 1

    print(f"\nDone in {elapsed:.1f}s")
    print(f"Cases:          {total_cases:,}")
    print(f"Total chunks:   {total_chunks:,}")
    print(f"Avg/case:       {total_chunks/total_cases:.1f}")
    print(f"Token stats:    min={min(token_counts)}  avg={avg:.0f}  max={max(token_counts)}")
    print(f"\nToken distribution:")
    for bucket, count in buckets.items():
        bar = "█" * (count // 200)
        print(f"  {bucket:>9}: {count:6,}  {bar}")

    # ── Show 3 full sample chunks for citation traceability review ────────
    print(f"\n{'='*65}")
    print("SAMPLE CHUNKS — verify citation traceability")
    print("='*65")
    with open(OUTPUT_FILE, "r", encoding="utf-8") as f:
        samples = []
        # Grab chunks from 3 different cases for diversity
        seen_cases = set()
        for line in f:
            c = json.loads(line)
            if c["source_case_id"] not in seen_cases:
                samples.append(c)
                seen_cases.add(c["source_case_id"])
            if len(samples) == 3:
                break

    for s in samples:
        print(f"\nchunk_id:       {s['chunk_id']}")
        print(f"source_case_id: {s['source_case_id']}")
        print(f"citation:       {s['citation']}")
        print(f"jurisdiction:   {s['jurisdiction']}")
        print(f"case_type:      {s['case_type']}")
        print(f"date:           {s['date']}")
        print(f"chunk_index:    {s['chunk_index']} of {s['total_chunks']-1}")
        print(f"token_count:    {s['token_count']}")
        print(f"text:\n  {s['text'][:500]}{'...' if len(s['text'])>500 else ''}")
        print()

    print(f"Output: {OUTPUT_FILE}  ({total_chunks:,} chunks)")
    print("Awaiting review before proceeding to embedding.")


if __name__ == "__main__":
    main()
