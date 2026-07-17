"""
Opposing-Argument Simulator — Corpus Merge Script
---------------------------------------------------
Combines the CAP corpus (legal_corpus_raw.jsonl) with the CourtListener
supplement (legal_corpus_courtlistener_supplement.jsonl) into a single
deduplicated output file (legal_corpus_final.jsonl).

Deduplication logic:
  - Primary: exact citation string match (normalised to lowercase, stripped)
  - Fallback: if citation is empty on either side, skip dedup and include both
    (better to have a near-duplicate than silently drop a real case)

CourtListener cases are newer (post-2018/2020) so they genuinely extend the
CAP historical snapshot rather than overlap with it. Expect very few actual
duplicates unless a case was indexed in both sources.

Run:
    python merge_corpus.py
"""

import json
import os

CAP_FILE        = "legal_corpus_raw.jsonl"
SUPPLEMENT_FILE = "legal_corpus_courtlistener_supplement.jsonl"
OUTPUT_FILE     = "legal_corpus_final.jsonl"


def normalise_citation(citation):
    """Normalise a citation string for comparison."""
    return citation.strip().lower() if citation else ""


def load_jsonl(path):
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def main():
    # ── Load CAP corpus ──────────────────────────────────────────────────
    if not os.path.exists(CAP_FILE):
        raise SystemExit(f"ERROR: CAP corpus not found: {CAP_FILE}")

    cap_records = load_jsonl(CAP_FILE)
    print(f"CAP corpus loaded:        {len(cap_records):,} records")

    # Build citation index from CAP (for dedup lookup)
    cap_citations = set()
    for rec in cap_records:
        c = normalise_citation(rec.get("citation", ""))
        if c:
            cap_citations.add(c)

    print(f"CAP citations indexed:    {len(cap_citations):,} unique citations")

    # ── Load CourtListener supplement ────────────────────────────────────
    if not os.path.exists(SUPPLEMENT_FILE):
        print(f"\nNo supplement file found ({SUPPLEMENT_FILE}).")
        print("Writing CAP corpus directly to output as legal_corpus_final.jsonl.")
        with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
            for rec in cap_records:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        print(f"Output: {len(cap_records):,} records -> {OUTPUT_FILE}")
        return

    supplement_records = load_jsonl(SUPPLEMENT_FILE)
    print(f"CourtListener supplement: {len(supplement_records):,} records")

    # ── Dedup CourtListener records against CAP ───────────────────────────
    new_records   = []
    skipped_dupes = 0
    skipped_empty = 0

    for rec in supplement_records:
        citation = normalise_citation(rec.get("citation", ""))

        if not citation:
            # No citation to check — include but flag
            new_records.append(rec)
            skipped_empty += 1
            continue

        if citation in cap_citations:
            skipped_dupes += 1
            continue

        # New case — add to set so we also dedup within the supplement itself
        cap_citations.add(citation)
        new_records.append(rec)

    print(f"\nDedup results:")
    print(f"  Skipped (citation exists in CAP): {skipped_dupes:,}")
    print(f"  Included (no citation to check):  {skipped_empty:,}")
    print(f"  New unique records added:          {len(new_records) - skipped_empty:,}")
    print(f"  Total new CourtListener records:   {len(new_records):,}")

    # ── Write merged output ───────────────────────────────────────────────
    total = len(cap_records) + len(new_records)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for rec in cap_records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        for rec in new_records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"\nMerge complete:")
    print(f"  CAP records:              {len(cap_records):,}")
    print(f"  New CourtListener records:{len(new_records):,}")
    print(f"  Total in final corpus:    {total:,}")
    print(f"  Output: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
