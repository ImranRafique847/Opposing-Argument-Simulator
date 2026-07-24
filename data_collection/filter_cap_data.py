"""
Opposing-Argument Simulator — Phase 1: CAP Data Filter
--------------------------------------------------------
Reads the Illinois CAP dataset downloaded from Kaggle
(harvardlil/caselaw-dataset-illinois) and filters for
landlord-tenant cases using whole-word keyword matching.

Source: Harvard Caselaw Access Project (CAP) — Illinois bulk file
        Downloaded via: kaggle datasets download -d harvardlil/caselaw-dataset-illinois -p data --unzip
        Input files: data/text.data.jsonl.xz

Uses whole-word regex matching (\bkeyword\b) to eliminate substring
false positives (e.g. "released" != "lease", "lieutenant" != "tenant").

Filter rules (both must pass):
  1. At least one PRIMARY keyword matches as a whole word.
  2. Either:
     a. Primary keyword appears 3+ times (whole-word), OR
     b. At least 2 SUPPORT terms also appear as whole words.

Output: legal_corpus_final.jsonl
  Fields: id, source, type, jurisdiction, case_type,
          citation, date, text

Result: 9,408 clean Illinois landlord-tenant cases from
        183,149 total Illinois cases (5.1% match rate)

Run:
    python data_collection/filter_cap_data.py
"""

import json
import os
import re
import time
from collections import Counter

import pandas as pd

# =========================================================
# CONFIG — Illinois (Kaggle CAP bulk download)
# =========================================================

# Illinois CAP data downloaded from Kaggle:
# kaggle datasets download -d harvardlil/caselaw-dataset-illinois -p data --unzip
# The downloaded file is: data/text.data.jsonl.xz
import lzma

INPUT_XZ_FILE = os.path.join(os.path.dirname(__file__), "..", "data", "text.data.jsonl.xz")
OUTPUT_FILE   = "legal_corpus_final.jsonl"
TARGET_STATE  = "Illinois"

# Primary keywords — whole-word regex
PRIMARY_KEYWORDS = ["landlord", "tenant", "eviction", "lease", "habitability"]
SUPPORT_TERMS    = [
    "rent", "premises", "possession", "dispossess", "lessee",
    "lessor", "sublease", "subletting", "security deposit",
    "notice to quit", "unlawful detainer", "holdover",
    "month-to-month", "tenancy",
]

PRIMARY_PATTERNS = {
    kw: re.compile(r"\b" + re.escape(kw) + r"\b", re.IGNORECASE)
    for kw in PRIMARY_KEYWORDS
}
SUPPORT_PATTERNS = {
    st: re.compile(r"\b" + re.escape(st) + r"\b", re.IGNORECASE)
    for st in SUPPORT_TERMS
}


# =========================================================
# FILTER LOGIC
# =========================================================

def primary_hit_count(text):
    return sum(len(p.findall(text)) for p in PRIMARY_PATTERNS.values())

def support_hit_count(text):
    return sum(1 for p in SUPPORT_PATTERNS.values() if p.search(text))

def is_match(text):
    hits = primary_hit_count(text)
    if hits == 0:
        return False
    if hits >= 3:
        return True
    return support_hit_count(text) >= 2


# =========================================================
# EXTRACTION HELPERS (handles both parquet and JSONL schemas)
# =========================================================

def extract_text(row):
    """Pull opinion text from a CAP parquet row."""
    casebody = row.get("casebody") if isinstance(row, dict) else getattr(row, "casebody", None)
    if casebody is not None:
        # Parquet stores casebody as dict with data.opinions[].text
        if isinstance(casebody, dict):
            data = casebody.get("data", {})
            if isinstance(data, dict):
                opinions = data.get("opinions", [])
                if opinions:
                    return "\n\n".join(
                        op.get("text", "") for op in opinions if op.get("text")
                    )
        # Some versions store it as a string
        if isinstance(casebody, str):
            return casebody
    return ""

def extract_citation(row):
    citations = row.get("citations") if isinstance(row, dict) else getattr(row, "citations", None)
    if citations and isinstance(citations, list) and len(citations) > 0:
        first = citations[0]
        if isinstance(first, dict):
            return first.get("cite", "")
        return str(first)
    return ""


# =========================================================
# PROCESS ONE STATE FILE
# =========================================================

def process_state(parquet_path, state_name, outfile):
    if not os.path.exists(parquet_path):
        print(f"  MISSING: {parquet_path} — skipping {state_name}")
        print(f"  Download: huggingface_hub.hf_hub_download(")
        print(f"      repo_id='free-law/Caselaw_Access_Project',")
        print(f"      filename='{'cal' if 'cal' in parquet_path else 'nm'}/{'cal' if 'cal' in parquet_path else 'nm'}.parquet',")
        print(f"      repo_type='dataset')")
        return 0

    print(f"\n{'='*60}")
    print(f"Processing: {state_name}  ({parquet_path})")
    start = time.time()

    df = pd.read_parquet(parquet_path)
    print(f"  Loaded {len(df):,} rows")

    matched = 0
    for _, row in df.iterrows():
        text = extract_text(row)
        if not text:
            continue
        if not is_match(text):
            continue

        citation = extract_citation(row)
        record = {
            "id":           row.get("id") if isinstance(row, dict) else getattr(row, "id", ""),
            "source":       "CAP",
            "type":         "case_law",
            "jurisdiction": state_name,
            "case_type":    CASE_TYPE_LABEL,
            "citation":     citation,
            "date":         row.get("decision_date") if isinstance(row, dict)
                            else getattr(row, "decision_date", ""),
            "text":         text,
        }
        outfile.write(json.dumps(record, ensure_ascii=False) + "\n")
        matched += 1

        if matched % 500 == 0:
            print(f"  {matched:,} matched so far...")

    elapsed = time.time() - start
    rate = len(df) / elapsed if elapsed > 0 else 0
    print(f"  Done in {elapsed:.0f}s — {matched:,} / {len(df):,} records matched "
          f"({matched/len(df)*100:.1f}%)")
    return matched


# =========================================================
# VALIDATION
# =========================================================

def validate(output_path):
    print(f"\n{'='*60}")
    print("VALIDATION")
    print("="*60)

    if not os.path.exists(output_path):
        print("Output file not found.")
        return

    records_by_state = {}
    with open(output_path, "r", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            state = rec.get("jurisdiction", "unknown")
            records_by_state.setdefault(state, []).append(rec)

    print(f"\nRecord counts by state:")
    total = 0
    for state, recs in records_by_state.items():
        print(f"  {state}: {len(recs):,}")
        total += len(recs)
    print(f"  TOTAL:  {total:,}")

    # Date sanity check per state
    print(f"\nDate range per state:")
    for state, recs in records_by_state.items():
        years = []
        bad = []
        for r in recs:
            d = str(r.get("date", ""))
            if d and len(d) >= 4:
                try:
                    y = int(d[:4])
                    years.append(y)
                    if y < 1850:
                        bad.append((r["id"], r["citation"], d))
                except ValueError:
                    pass
        if years:
            print(f"  {state}: {min(years)}–{max(years)}  "
                  f"(suspicious pre-1850: {len(bad)})")

    # 3 sample records per state with keyword context
    def find_context(text, window=200):
        for kw, pat in PRIMARY_PATTERNS.items():
            m = pat.search(text)
            if m:
                s = max(0, m.start() - window)
                e = min(len(text), m.end() + window)
                snippet = text[s:e].replace("\n", " ")
                rel = m.start() - s
                rel_e = rel + len(m.group())
                snippet = snippet[:rel] + "[" + snippet[rel:rel_e] + "]" + snippet[rel_e:]
                return kw, "..." + snippet + "..."
        return None, ""

    print(f"\n3 sample records per state (citation traceability check):")
    for state, recs in records_by_state.items():
        print(f"\n--- {state} ---")
        for i, rec in enumerate(recs[:3]):
            kw, ctx = find_context(rec["text"])
            hits = primary_hit_count(rec["text"])
            print(f"\n  Sample {i+1}:")
            print(f"    id:         {rec['id']}")
            print(f"    citation:   {rec['citation']}")
            print(f"    date:       {rec['date']}")
            print(f"    kw_hits:    {hits}")
            print(f"    context:    {ctx[:300]}")


# =========================================================
# MAIN
# =========================================================

def main():
    print(f"Output: {OUTPUT_FILE}")
    print(f"States: {[s for _, s in STATE_FILES]}")

    # Check which files are available
    available = [(p, s) for p, s in STATE_FILES if os.path.exists(p)]
    missing   = [(p, s) for p, s in STATE_FILES if not os.path.exists(p)]

    if missing:
        print(f"\nMISSING DATA FILES:")
        for p, s in missing:
            print(f"  {s}: {p}")
        print(f"\nTo download (requires free-law/CAP gated access approval):")
        print(f"  Run: python download_cap_states.py")
        if not available:
            print("\nNo state files available — nothing to process.")
            return

    if os.path.exists(OUTPUT_FILE):
        os.remove(OUTPUT_FILE)

    total_matched = 0
    with open(OUTPUT_FILE, "w", encoding="utf-8") as outfile:
        for parquet_path, state_name in STATE_FILES:
            n = process_state(parquet_path, state_name, outfile)
            total_matched += n

    print(f"\nTotal records written: {total_matched:,} -> {OUTPUT_FILE}")

    if total_matched > 0:
        validate(OUTPUT_FILE)
    else:
        print("No records written — check data files and re-run once downloaded.")


if __name__ == "__main__":
    main()
