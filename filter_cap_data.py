"""
Opposing-Argument Simulator — Phase 1: CAP Data Filter v2
-----------------------------------------------------------
Reads the Illinois CAP dataset (text.data.jsonl.xz) directly.
Uses whole-word regex matching (\bkeyword\b) to eliminate substring
false positives like "released" matching "lease" or "lieutenant"
matching "tenant".

Tighter filter rules (both must pass):
  1. At least one PRIMARY keyword matches as a whole word.
  2. Either:
     a. Primary keyword appears 3+ times (whole-word), OR
     b. At least 2 SUPPORT terms also appear as whole words.

Output: legal_corpus_raw.jsonl with fields:
  id, source, type, jurisdiction, case_type, citation, date, text
"""

import json
import lzma
import os
import re
import time
from collections import Counter

# =========================================================
# CONFIG
# =========================================================

INPUT_FILE  = r"d:\Opposing-Argument Simulator\data\text.data.jsonl.xz"
OUTPUT_FILE = r"d:\Opposing-Argument Simulator\legal_corpus_raw.jsonl"

TARGET_STATE    = "Illinois"
CASE_TYPE_LABEL = "tenancy"

# Compile whole-word regex patterns for all keywords
PRIMARY_KEYWORDS = ["landlord", "tenant", "eviction", "lease", "habitability"]
SUPPORT_TERMS    = [
    "rent", "premises", "possession", "dispossess", "lessee",
    "lessor", "sublease", "subletting", "security deposit",
    "notice to quit", "unlawful detainer", "holdover",
    "month-to-month", "tenancy",
]

PRIMARY_PATTERNS = {kw: re.compile(r"\b" + re.escape(kw) + r"\b", re.IGNORECASE)
                    for kw in PRIMARY_KEYWORDS}
SUPPORT_PATTERNS = {st: re.compile(r"\b" + re.escape(st) + r"\b", re.IGNORECASE)
                    for st in SUPPORT_TERMS}


# =========================================================
# FILTER LOGIC
# =========================================================

def primary_hit_count(text):
    """Total whole-word hits across all primary keywords."""
    return sum(len(p.findall(text)) for p in PRIMARY_PATTERNS.values())

def support_hit_count(text):
    """Number of distinct support terms that appear as whole words."""
    return sum(1 for p in SUPPORT_PATTERNS.values() if p.search(text))

def is_match(text):
    """Return True only if the case genuinely discusses tenancy topics."""
    # Must have at least one primary keyword as a whole word
    primary_hits = primary_hit_count(text)
    if primary_hits == 0:
        return False
    # Strong signal: 3+ primary keyword hits
    if primary_hits >= 3:
        return True
    # Moderate signal: any primary keyword + 2+ support terms
    if support_hit_count(text) >= 2:
        return True
    return False


# =========================================================
# EXTRACTION HELPERS
# =========================================================

def extract_text(case):
    casebody = case.get("casebody", {})
    if isinstance(casebody, dict):
        data = casebody.get("data", {})
        if isinstance(data, dict):
            opinions = data.get("opinions", [])
            if opinions:
                return "\n\n".join(op.get("text", "") for op in opinions if op.get("text"))
    return ""

def extract_citation(case):
    citations = case.get("citations", [])
    if citations and isinstance(citations, list):
        first = citations[0]
        return first.get("cite", "") if isinstance(first, dict) else str(first)
    return ""


# =========================================================
# MAIN FILTER PASS
# =========================================================

def run_filter():
    if not os.path.exists(INPUT_FILE):
        raise SystemExit(f"ERROR: Input file not found: {INPUT_FILE}")

    print(f"Source: {INPUT_FILE}")
    print(f"Filter: whole-word regex, primary keywords: {PRIMARY_KEYWORDS}")
    print(f"Rule:   (>=3 primary hits) OR (>=1 primary + >=2 support terms)\n")

    total_scanned = 0
    total_matched = 0
    start = time.time()

    if os.path.exists(OUTPUT_FILE):
        os.remove(OUTPUT_FILE)

    with lzma.open(INPUT_FILE, mode="rt", encoding="utf-8") as infile, \
         open(OUTPUT_FILE, "w", encoding="utf-8") as outfile:

        for line in infile:
            line = line.strip()
            if not line:
                continue
            total_scanned += 1

            try:
                case = json.loads(line)
            except json.JSONDecodeError:
                continue

            text = extract_text(case)
            if not text:
                continue

            if not is_match(text):
                continue

            record = {
                "id":           case.get("id", ""),
                "source":       "CAP",
                "type":         "case_law",
                "jurisdiction": TARGET_STATE,
                "case_type":    CASE_TYPE_LABEL,
                "citation":     extract_citation(case),
                "date":         case.get("decision_date", ""),
                "text":         text,
            }
            outfile.write(json.dumps(record, ensure_ascii=False) + "\n")
            total_matched += 1

            if total_scanned % 20000 == 0:
                print(f"  Scanned {total_scanned:,} | Matched {total_matched:,} ...")

    elapsed = time.time() - start
    print(f"\nFilter done in {elapsed:.1f}s.")
    print(f"Total scanned: {total_scanned:,}")
    print(f"Total matched: {total_matched:,}")
    print(f"Match rate:    {total_matched/total_scanned*100:.1f}%")
    return total_matched


# =========================================================
# VALIDATION
# =========================================================

def find_context(text, patterns, window=250):
    """Find first whole-word match and return surrounding context."""
    for kw, pat in patterns.items():
        m = pat.search(text)
        if m:
            start = max(0, m.start() - window)
            end   = min(len(text), m.end() + window)
            snippet = text[start:end].replace("\n", " ")
            rel = m.start() - start
            rel_end = rel + len(m.group())
            snippet = snippet[:rel] + "[" + snippet[rel:rel_end] + "]" + snippet[rel_end:]
            return kw, "..." + snippet + "..."
    return None, ""

def validate():
    print("\n" + "=" * 70)
    print("VALIDATION — 5 SAMPLE RECORDS WITH KEYWORD CONTEXT")
    print("=" * 70)

    with open(OUTPUT_FILE, "r", encoding="utf-8") as f:
        lines = f.readlines()

    # Show records 1-5
    for i, line in enumerate(lines[:5]):
        rec = json.loads(line)
        kw, ctx = find_context(rec["text"], PRIMARY_PATTERNS)
        primary_hits = primary_hit_count(rec["text"])
        support_hits = support_hit_count(rec["text"])
        print(f"\nRecord {i+1}:")
        print(f"  id:            {rec['id']}")
        print(f"  citation:      {rec['citation']}")
        print(f"  date:          {rec['date']}")
        print(f"  primary hits:  {primary_hits}  |  support terms: {support_hits}")
        print(f"  matched kw:    [{kw}]")
        print(f"  context:       {ctx[:400]}")

    # Check for burglary case specifically
    print("\n" + "=" * 70)
    print("BURGLARY CASE CHECK (id=2747110 should NOT be present)")
    print("=" * 70)
    found_burglary = any(json.loads(l).get("id") == 2747110 for l in lines)
    print(f"  id=2747110 present: {found_burglary}  {'<-- PROBLEM' if found_burglary else '✓ REMOVED'}")

    # Jurisdiction check
    print("\n" + "=" * 70)
    print("JURISDICTION + DATE DISTRIBUTION")
    print("=" * 70)
    dates = []
    bad_dates = []
    for line in lines:
        rec = json.loads(line)
        d = rec.get("date", "")
        if d:
            try:
                year = int(str(d)[:4])
                dates.append(year)
                if year < 1820:  # Illinois became a state in 1818
                    bad_dates.append((rec["id"], rec["citation"], d))
            except ValueError:
                pass

    print(f"  Total records: {len(lines):,}")
    print(f"  Date range:    {min(dates)} — {max(dates)}")
    print(f"  Pre-1820 dates (suspicious): {len(bad_dates)}")
    for bid, bcite, bdate in bad_dates[:5]:
        print(f"    id={bid}, citation={bcite}, date={bdate}")

    decade_counts = Counter((y // 10) * 10 for y in dates)
    print("\n  Decade distribution (top 12):")
    for decade, count in sorted(decade_counts.items(), key=lambda x: -x[1])[:12]:
        bar = "█" * (count // 50)
        print(f"    {decade}s: {count:5,}  {bar}")

    # Spot-check: show 3 records that scored highest on primary hits
    print("\n" + "=" * 70)
    print("TOP 3 RECORDS BY PRIMARY KEYWORD DENSITY (strongest signal)")
    print("=" * 70)
    scored = []
    for line in lines:
        rec = json.loads(line)
        scored.append((primary_hit_count(rec["text"]), rec))
    scored.sort(key=lambda x: -x[0])
    for score, rec in scored[:3]:
        kw, ctx = find_context(rec["text"], PRIMARY_PATTERNS)
        print(f"\n  id={rec['id']} | citation={rec['citation']} | date={rec['date']}")
        print(f"  primary keyword hits: {score}")
        print(f"  context: {ctx[:350]}")

    print(f"\n\nPhase 1 validation complete. Final corpus: {len(lines):,} records.")
    print(f"Output: {OUTPUT_FILE}")


# =========================================================
# RUN
# =========================================================

if __name__ == "__main__":
    count = run_filter()
    if count > 0:
        validate()
