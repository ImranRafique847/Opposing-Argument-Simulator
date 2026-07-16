"""
Opposing-Argument Simulator — Phase 1: Data Collection
--------------------------------------------------------
Run this locally on your laptop (CPU only, no GPU needed for this step).

What it does:
1. Streams the Case Access Project (CAP) dataset from Hugging Face and
   filters cases to your chosen state + case-type keywords.
2. Pulls supplementary case metadata from CourtListener (optional, needs a free API token).
3. Normalizes everything into a single clean legal_corpus_raw.jsonl file,
   ready for chunking + embedding in Phase 2.

Install dependencies first:
    pip install datasets huggingface_hub pandas requests tqdm python-dotenv hf_transfer

.env file (place in the same folder as this script):
    HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxx
    COURTLISTENER_API_TOKEN=your_courtlistener_token_here   # optional
"""

import os

# Enable Hugging Face's accelerated Rust-based downloader BEFORE importing datasets/huggingface_hub
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

import json
import time
import pandas as pd
import requests
from dotenv import load_dotenv
from huggingface_hub import login
from datasets import load_dataset

# =========================================================
# LOAD SECRETS FROM .env
# =========================================================

load_dotenv()  # reads .env in the same folder as this script

HF_TOKEN = os.getenv("HF_TOKEN", "")
COURTLISTENER_API_TOKEN = os.getenv("COURTLISTENER_API_TOKEN", "")

if HF_TOKEN:
    login(token=HF_TOKEN)
    print("Logged into Hugging Face using token from .env")
else:
    print("No HF_TOKEN found in .env — proceeding without login (may hit rate limits).")

# =========================================================
# CONFIG — edit these for your chosen jurisdiction/scope
# =========================================================

TARGET_STATE = "Illinois"                 # change to your chosen state
CASE_TYPE_LABEL = "tenancy"               # label saved into your metadata
KEYWORDS = ["landlord", "tenant", "eviction", "lease", "habitability"]
MAX_CASES = 2000                          # cap size for a manageable local prototype
SCAN_LIMIT = None                         # set an int to cap how many CAP records to scan (for quick testing), None = no cap

COURTLISTENER_QUERY = "landlord tenant"
COURTLISTENER_COURT_CODE = "ill"          # adjust to your target court(s)

OUTPUT_CASES_PARQUET = "cases_raw.parquet"
OUTPUT_CORPUS_JSONL = "legal_corpus_raw.jsonl"


# =========================================================
# STEP 1 — Stream + filter CAP case law
# =========================================================

def collect_cap_cases():
    print(f"\n[1/3] Streaming CAP dataset, filtering for {TARGET_STATE} / {CASE_TYPE_LABEL}...")

    dataset = load_dataset(
        "free-law/Caselaw_Access_Project",
        split="train",
        streaming=True,  # avoids downloading the full multi-GB dataset to disk
    )

    collected = []
    scanned = 0

    for case in dataset:
        scanned += 1

        text = (case.get("text") or "")
        jurisdiction = (case.get("jurisdiction") or "")

        if TARGET_STATE.lower() in jurisdiction.lower() and any(k in text.lower() for k in KEYWORDS):
            collected.append(case)

        if scanned % 5000 == 0:
            print(f"  scanned {scanned} cases, collected {len(collected)} so far...")

        if len(collected) >= MAX_CASES:
            break
        if SCAN_LIMIT and scanned >= SCAN_LIMIT:
            break

    df = pd.DataFrame(collected)
    df.to_parquet(OUTPUT_CASES_PARQUET)
    print(f"  Done. Saved {len(df)} cases -> {OUTPUT_CASES_PARQUET}")
    return df


# =========================================================
# STEP 2 — Pull supplementary metadata from CourtListener (optional)
# =========================================================

def collect_courtlistener_metadata():
    if not COURTLISTENER_API_TOKEN:
        print("\n[2/3] Skipping CourtListener pull (no API token set). "
              "Register free at https://www.courtlistener.com/help/api/ to enable this.")
        return []

    print(f"\n[2/3] Querying CourtListener for '{COURTLISTENER_QUERY}' in court '{COURTLISTENER_COURT_CODE}'...")

    headers = {"Authorization": f"Token {COURTLISTENER_API_TOKEN}"}
    params = {"q": COURTLISTENER_QUERY, "court": COURTLISTENER_COURT_CODE}

    try:
        response = requests.get(
            "https://www.courtlistener.com/api/rest/v4/search/",
            params=params,
            headers=headers,
            timeout=30,
        )
        response.raise_for_status()
        data = response.json()
        results = data.get("results", [])
        print(f"  Retrieved {len(results)} metadata records.")
        return results
    except requests.RequestException as e:
        print(f"  CourtListener request failed: {e}")
        return []


# =========================================================
# STEP 3 — Normalize everything into one clean JSONL corpus
# =========================================================

def build_corpus(cases_df, courtlistener_records):
    print(f"\n[3/3] Normalizing into {OUTPUT_CORPUS_JSONL}...")

    records = []

    # CAP case law records
    for _, row in cases_df.iterrows():
        records.append({
            "id": row.get("id"),
            "source": "CAP",
            "type": "case_law",
            "jurisdiction": TARGET_STATE,
            "case_type": CASE_TYPE_LABEL,
            "citation": row.get("citation", ""),
            "date": row.get("decision_date", ""),
            "text": row.get("text", ""),
        })

    # CourtListener supplementary metadata records
    for item in courtlistener_records:
        records.append({
            "id": item.get("id"),
            "source": "CourtListener",
            "type": "case_metadata",
            "jurisdiction": TARGET_STATE,
            "case_type": CASE_TYPE_LABEL,
            "citation": item.get("citation", ""),
            "date": item.get("dateFiled", ""),
            "text": item.get("snippet", "") or item.get("caseName", ""),
        })

    with open(OUTPUT_CORPUS_JSONL, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"  Done. Wrote {len(records)} total records -> {OUTPUT_CORPUS_JSONL}")


# =========================================================
# MAIN
# =========================================================

if __name__ == "__main__":
    start = time.time()

    cases_df = collect_cap_cases()
    courtlistener_records = collect_courtlistener_metadata()
    build_corpus(cases_df, courtlistener_records)

    elapsed = time.time() - start
    print(f"\nAll done in {elapsed:.1f}s.")
    print(f"Next step: chunk + embed {OUTPUT_CORPUS_JSONL} for the RAG pipeline.")
