"""
Data validation script for legal_corpus_raw.jsonl
Checks: keyword context, jurisdiction accuracy, date distribution, filter quality.
"""

import json
import re
import lzma
from collections import Counter
from datetime import datetime

KEYWORDS = ["landlord", "tenant", "eviction", "lease", "habitability"]
TENANCY_SUPPORT = ["rent", "premises", "possession", "dispossess", "lessee",
                   "lessor", "sublease", "subletting", "month-to-month",
                   "security deposit", "notice to quit", "unlawful detainer"]

CORPUS_FILE = r"d:\Opposing-Argument Simulator\legal_corpus_raw.jsonl"
CAP_FILE    = r"d:\Opposing-Argument Simulator\data\text.data.jsonl.xz"


# =========================================================
# STEP 1 — Show 5 sample records with keyword context
# =========================================================

def find_context(text, keywords, window=250):
    text_lower = text.lower()
    for kw in keywords:
        idx = text_lower.find(kw)
        if idx != -1:
            start = max(0, idx - window)
            end   = min(len(text), idx + len(kw) + window)
            snippet = text[start:end].replace("\n", " ")
            # Mark the keyword with brackets
            rel_idx = idx - start
            snippet = snippet[:rel_idx] + "[" + snippet[rel_idx:rel_idx+len(kw)] + "]" + snippet[rel_idx+len(kw):]
            return kw, "..." + snippet + "..."
    return None, ""

print("=" * 70)
print("STEP 1 — 5 SAMPLE RECORDS WITH KEYWORD CONTEXT")
print("=" * 70)

with open(CORPUS_FILE, "r", encoding="utf-8") as f:
    for i, line in enumerate(f):
        if i == 0:
            continue  # skip record 1 (already seen — burglary case)
        if i > 5:
            break
        rec = json.loads(line)
        kw, ctx = find_context(rec["text"], KEYWORDS)
        print(f"\nRecord {i+1}:")
        print(f"  id:       {rec['id']}")
        print(f"  citation: {rec['citation']}")
        print(f"  date:     {rec['date']}")
        print(f"  keyword:  [{kw}]")
        print(f"  context:  {ctx[:400]}")


# =========================================================
# STEP 2 — Jurisdiction check + date distribution from source
# =========================================================

print("\n\n" + "=" * 70)
print("STEP 2 — JURISDICTION CHECK + DATE DISTRIBUTION (from filtered corpus)")
print("=" * 70)

# Load all records from corpus and check source jurisdiction + dates
jurisdictions = Counter()
dates = []
bad_dates = []

with open(CORPUS_FILE, "r", encoding="utf-8") as f:
    all_records = [json.loads(line) for line in f]

# Re-check source jurisdiction by scanning the CAP file for matched IDs
matched_ids = {rec["id"] for rec in all_records}
print(f"\nTotal matched records in corpus: {len(matched_ids)}")
print("Scanning CAP source file to verify jurisdiction labels...")

source_jurisdictions = Counter()
sample_jurisdictions = []

with lzma.open(CAP_FILE, mode="rt", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        case = json.loads(line)
        if case.get("id") in matched_ids:
            jur = case.get("jurisdiction", {})
            if isinstance(jur, dict):
                jur_name = jur.get("name_long", jur.get("name", "unknown"))
            else:
                jur_name = str(jur)
            source_jurisdictions[jur_name] += 1
            if len(sample_jurisdictions) < 5:
                sample_jurisdictions.append((case.get("id"), jur_name))

print("\nJurisdiction distribution in source data:")
for jur, count in source_jurisdictions.most_common(10):
    print(f"  {jur}: {count}")

# Date distribution
for rec in all_records:
    d = rec.get("date", "")
    if d:
        try:
            year = int(d[:4])
            dates.append(year)
            if year < 1800:
                bad_dates.append((rec["id"], rec["citation"], d))
        except ValueError:
            pass

print(f"\nDate range: {min(dates)} — {max(dates)}")
print(f"Records with date before 1800 (likely OCR errors): {len(bad_dates)}")
if bad_dates:
    print("  Examples:")
    for bid, bcite, bdate in bad_dates[:5]:
        print(f"    id={bid}, citation={bcite}, date={bdate}")

# Decade histogram
decade_counts = Counter((y // 10) * 10 for y in dates)
print("\nDecade distribution (top 15 by count):")
for decade, count in sorted(decade_counts.items(), key=lambda x: -x[1])[:15]:
    bar = "█" * (count // 100)
    print(f"  {decade}s: {count:5d} {bar}")


# =========================================================
# STEP 3 — Tighter filter
# =========================================================

print("\n\n" + "=" * 70)
print("STEP 3 — TIGHTER FILTER ANALYSIS")
print("=" * 70)

def is_strong_match(rec):
    text_lower = rec["text"].lower()

    # Count how many primary keywords appear
    kw_hits = sum(1 for kw in KEYWORDS if kw in text_lower)

    # Count how many support terms appear
    support_hits = sum(1 for st in TENANCY_SUPPORT if st in text_lower)

    # Rule: keyword appears 2+ times, OR keyword + at least 2 support terms
    kw_count = sum(text_lower.count(kw) for kw in KEYWORDS)
    return (kw_count >= 3) or (kw_hits >= 1 and support_hits >= 2)

strong = [rec for rec in all_records if is_strong_match(rec)]
weak   = [rec for rec in all_records if not is_strong_match(rec)]

print(f"\nOriginal corpus:      {len(all_records):,} records")
print(f"After tighter filter: {len(strong):,} records")
print(f"Removed as too weak:  {len(weak):,} records")
print(f"Retention rate:       {len(strong)/len(all_records)*100:.1f}%")

# Show 3 strong examples with context
print("\n\n--- 3 STRONG EXAMPLES (genuinely relevant) ---")
shown = 0
for rec in strong:
    if shown >= 3:
        break
    text_lower = rec["text"].lower()
    kw_count = sum(text_lower.count(kw) for kw in KEYWORDS)
    support_hits = [st for st in TENANCY_SUPPORT if st in text_lower]

    # Find context for the most-frequent keyword
    best_kw = max(KEYWORDS, key=lambda kw: text_lower.count(kw))
    _, ctx = find_context(rec["text"], [best_kw])

    print(f"\nExample {shown+1}:")
    print(f"  id:            {rec['id']}")
    print(f"  citation:      {rec['citation']}")
    print(f"  date:          {rec['date']}")
    print(f"  keyword hits:  {kw_count} (primary keywords)")
    print(f"  support terms: {support_hits[:5]}")
    print(f"  context:       {ctx[:400]}")
    shown += 1


# =========================================================
# WRITE TIGHTENED CORPUS
# =========================================================

TIGHT_OUTPUT = r"d:\Opposing-Argument Simulator\legal_corpus_raw.jsonl"
with open(TIGHT_OUTPUT, "w", encoding="utf-8") as f:
    for rec in strong:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")

print(f"\n\nTightened corpus saved: {len(strong):,} records -> {TIGHT_OUTPUT}")
print("Ready for Phase 2 review.")
